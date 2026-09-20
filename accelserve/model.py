from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from accelserve.config import ModelConfig


@dataclass
class SequenceState:
    past_key_values: list[tuple[torch.Tensor, torch.Tensor]]
    next_logits: torch.Tensor
    sequence_length: int


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return self.weight * x * torch.rsqrt(variance + self.eps)


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_size
        self.attn_norm = RMSNorm(h)
        self.q_proj = nn.Linear(h, h, bias=False)
        self.k_proj = nn.Linear(h, h, bias=False)
        self.v_proj = nn.Linear(h, h, bias=False)
        self.o_proj = nn.Linear(h, h, bias=False)
        self.ffn_norm = RMSNorm(h)
        self.gate_proj = nn.Linear(h, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(h, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, h, bias=False)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.config.num_heads, self.config.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, _, t, _ = x.shape
        return x.transpose(1, 2).contiguous().view(b, t, self.config.hidden_size)

    def prefill(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        residual = x
        x_norm = self.attn_norm(x)
        q = self._split_heads(self.q_proj(x_norm))
        k = self._split_heads(self.k_proj(x_norm))
        v = self._split_heads(self.v_proj(x_norm))

        t = x.shape[1]
        causal = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.config.head_dim**0.5)
        scores = scores.masked_fill(causal.view(1, 1, t, t), float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        attn = torch.matmul(probs, v)
        x = residual + self.o_proj(self._merge_heads(attn))

        residual = x
        x_norm = self.ffn_norm(x)
        gated = F.silu(self.gate_proj(x_norm)) * self.up_proj(x_norm)
        x = residual + self.down_proj(gated)
        return x, (k.detach(), v.detach())

    def decode_batch(
        self,
        x: torch.Tensor,
        past: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        residual = x
        x_norm = self.attn_norm(x)
        q = self._split_heads(self.q_proj(x_norm))
        new_k = self._split_heads(self.k_proj(x_norm))
        new_v = self._split_heads(self.v_proj(x_norm))

        lengths = [item[0].shape[2] + 1 for item in past]
        max_len = max(lengths)
        b = len(past)
        h = self.config.num_heads
        d = self.config.head_dim

        k_batch = x.new_zeros((b, h, max_len, d))
        v_batch = x.new_zeros((b, h, max_len, d))
        valid = torch.zeros((b, max_len), dtype=torch.bool, device=x.device)
        updated: list[tuple[torch.Tensor, torch.Tensor]] = []

        for i, (old_k, old_v) in enumerate(past):
            cat_k = torch.cat([old_k, new_k[i : i + 1]], dim=2)
            cat_v = torch.cat([old_v, new_v[i : i + 1]], dim=2)
            seq_len = cat_k.shape[2]
            k_batch[i, :, :seq_len, :] = cat_k[0]
            v_batch[i, :, :seq_len, :] = cat_v[0]
            valid[i, :seq_len] = True
            updated.append((cat_k.detach(), cat_v.detach()))

        scores = torch.matmul(q, k_batch.transpose(-2, -1)) / (self.config.head_dim**0.5)
        scores = scores.masked_fill(~valid.view(b, 1, 1, max_len), float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        attn = torch.matmul(probs, v_batch)
        x = residual + self.o_proj(self._merge_heads(attn))

        residual = x
        x_norm = self.ffn_norm(x)
        gated = F.silu(self.gate_proj(x_norm)) * self.up_proj(x_norm)
        x = residual + self.down_proj(gated)
        return x, updated


class TinyDecoderLM(nn.Module):
    """Reference decoder-only transformer with explicit KV-cache operations."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_sequence_length, config.hidden_size)
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.num_layers)])
        self.final_norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    @classmethod
    def deterministic(cls, config: ModelConfig, device: torch.device) -> "TinyDecoderLM":
        cpu_state = torch.random.get_rng_state()
        torch.manual_seed(config.seed)
        model = cls(config)
        torch.random.set_rng_state(cpu_state)
        return model.eval().to(device)

    @torch.inference_mode()
    def prefill(self, input_ids: list[int]) -> SequenceState:
        if not input_ids:
            raise ValueError("input_ids cannot be empty")
        if len(input_ids) > self.config.max_sequence_length:
            raise ValueError("prompt exceeds maximum sequence length")

        device = self.token_embedding.weight.device
        ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
        positions = torch.arange(ids.shape[1], device=device).unsqueeze(0)
        x = self.token_embedding(ids) + self.position_embedding(positions)

        cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        for block in self.blocks:
            x, layer_cache = block.prefill(x)
            cache.append(layer_cache)

        logits = self.lm_head(self.final_norm(x[:, -1:, :]))[0, 0]
        return SequenceState(cache, logits, len(input_ids))

    @torch.inference_mode()
    def decode_batch(self, states: list[SequenceState], current_tokens: list[int]) -> None:
        if len(states) != len(current_tokens):
            raise ValueError("states and current_tokens must have the same length")
        if not states:
            return

        device = self.token_embedding.weight.device
        positions = torch.tensor(
            [state.sequence_length for state in states], dtype=torch.long, device=device
        ).unsqueeze(1)
        if int(positions.max().item()) >= self.config.max_sequence_length:
            raise ValueError("sequence length exceeds model capacity")

        ids = torch.tensor(current_tokens, dtype=torch.long, device=device).unsqueeze(1)
        x = self.token_embedding(ids) + self.position_embedding(positions)

        per_layer_updated: list[list[tuple[torch.Tensor, torch.Tensor]]] = []
        for layer_index, block in enumerate(self.blocks):
            past = [state.past_key_values[layer_index] for state in states]
            x, updated = block.decode_batch(x, past)
            per_layer_updated.append(updated)

        logits = self.lm_head(self.final_norm(x))[:, 0, :]
        for request_index, state in enumerate(states):
            state.past_key_values = [
                per_layer_updated[layer_index][request_index]
                for layer_index in range(len(self.blocks))
            ]
            state.next_logits = logits[request_index]
            state.sequence_length += 1
