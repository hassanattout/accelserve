from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from accelserve.config import ModelConfig
from accelserve.paged_kv import PagedKVCache


@dataclass
class SequenceState:
    request_id: str
    next_logits: torch.Tensor
    sequence_length: int


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        values = x.float()
        variance = values.pow(2).mean(dim=-1, keepdim=True)
        normalized = values * torch.rsqrt(variance + self.eps)
        return (self.weight.float() * normalized).to(input_dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    head_dim = q.shape[-1]
    if head_dim % 2:
        raise ValueError("RoPE requires an even head dimension")
    inv_freq = 1.0 / (
        theta
        ** (
            torch.arange(
                0, head_dim, 2, device=q.device, dtype=torch.float32
            )
            / head_dim
        )
    )
    pos = positions.float().unsqueeze(-1)
    freqs = pos * inv_freq
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(q.dtype).unsqueeze(1)
    sin = emb.sin().to(q.dtype).unsqueeze(1)
    return (
        q * cos + rotate_half(q) * sin,
        k * cos + rotate_half(k) * sin,
    )


def repeat_kv(x: torch.Tensor, groups: int) -> torch.Tensor:
    if groups == 1:
        return x
    return x.repeat_interleave(groups, dim=1)


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_size
        kv_width = config.kv_heads * config.head_dim
        self.attn_norm = RMSNorm(h, config.rms_norm_eps)
        self.q_proj = nn.Linear(h, h, bias=config.qkv_bias)
        self.k_proj = nn.Linear(h, kv_width, bias=config.qkv_bias)
        self.v_proj = nn.Linear(h, kv_width, bias=config.qkv_bias)
        self.o_proj = nn.Linear(h, h, bias=False)
        self.ffn_norm = RMSNorm(h, config.rms_norm_eps)
        self.gate_proj = nn.Linear(h, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(h, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, h, bias=False)

    def _split_q(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(
            b, t, self.config.num_heads, self.config.head_dim
        ).transpose(1, 2)

    def _split_kv(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(
            b, t, self.config.kv_heads, self.config.head_dim
        ).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, _, t, _ = x.shape
        return x.transpose(1, 2).contiguous().view(
            b, t, self.config.hidden_size
        )

    def prefill(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = x
        x_norm = self.attn_norm(x)
        q = self._split_q(self.q_proj(x_norm))
        k = self._split_kv(self.k_proj(x_norm))
        v = self._split_kv(self.v_proj(x_norm))
        if self.config.use_rope:
            q, k = apply_rope(
                q, k, positions, self.config.rope_theta
            )

        t = x.shape[1]
        causal = torch.triu(
            torch.ones(
                t, t, device=x.device, dtype=torch.bool
            ),
            diagonal=1,
        )
        k_attn = repeat_kv(k, self.config.kv_groups)
        v_attn = repeat_kv(v, self.config.kv_groups)
        scores = torch.matmul(
            q, k_attn.transpose(-2, -1)
        ) / (self.config.head_dim**0.5)
        scores = scores.masked_fill(
            causal.view(1, 1, t, t), float("-inf")
        )
        probs = torch.softmax(
            scores.float(), dim=-1
        ).to(q.dtype)
        attn = torch.matmul(probs, v_attn)
        x = residual + self.o_proj(
            self._merge_heads(attn)
        )

        residual = x
        x_norm = self.ffn_norm(x)
        gated = (
            F.silu(self.gate_proj(x_norm))
            * self.up_proj(x_norm)
        )
        return (
            residual + self.down_proj(gated),
            k.detach(),
            v.detach(),
        )

    def decode_batch(
        self,
        x: torch.Tensor,
        states: list[SequenceState],
        layer_index: int,
        cache: PagedKVCache,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        residual = x
        x_norm = self.attn_norm(x)
        q = self._split_q(self.q_proj(x_norm))
        new_k = self._split_kv(self.k_proj(x_norm))
        new_v = self._split_kv(self.v_proj(x_norm))
        if self.config.use_rope:
            q, new_k = apply_rope(
                q, new_k, positions, self.config.rope_theta
            )

        lengths = [
            state.sequence_length + 1
            for state in states
        ]
        max_len = max(lengths)
        b = len(states)
        h = self.config.num_heads
        d = self.config.head_dim

        k_batch = x.new_zeros(
            (b, h, max_len, d)
        )
        v_batch = x.new_zeros(
            (b, h, max_len, d)
        )
        valid = torch.zeros(
            (b, max_len),
            dtype=torch.bool,
            device=x.device,
        )

        for i, state in enumerate(states):
            cache.append(
                state.request_id,
                layer_index,
                state.sequence_length,
                new_k[i : i + 1],
                new_v[i : i + 1],
            )
            k, v = cache.read(
                state.request_id,
                layer_index,
                state.sequence_length + 1,
            )
            k = repeat_kv(
                k, self.config.kv_groups
            )
            v = repeat_kv(
                v, self.config.kv_groups
            )
            seq_len = k.shape[2]
            k_batch[i, :, :seq_len, :] = k[0]
            v_batch[i, :, :seq_len, :] = v[0]
            valid[i, :seq_len] = True

        scores = torch.matmul(
            q, k_batch.transpose(-2, -1)
        ) / (self.config.head_dim**0.5)
        scores = scores.masked_fill(
            ~valid.view(b, 1, 1, max_len),
            float("-inf"),
        )
        probs = torch.softmax(
            scores.float(), dim=-1
        ).to(q.dtype)
        attn = torch.matmul(probs, v_batch)
        x = residual + self.o_proj(
            self._merge_heads(attn)
        )

        residual = x
        x_norm = self.ffn_norm(x)
        gated = (
            F.silu(self.gate_proj(x_norm))
            * self.up_proj(x_norm)
        )
        return residual + self.down_proj(gated)


class PagedDecoderLM(nn.Module):
    """Decoder LM with explicit paged K/V writes and scheduler-owned decode."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(
            config.vocab_size, config.hidden_size
        )
        self.position_embedding = (
            nn.Embedding(
                config.max_sequence_length,
                config.hidden_size,
            )
            if config.use_learned_positions
            else None
        )
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(config)
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )
        if config.tie_word_embeddings:
            self.lm_head.weight = (
                self.token_embedding.weight
            )

    @classmethod
    def deterministic(
        cls,
        config: ModelConfig,
        device: torch.device,
    ) -> "PagedDecoderLM":
        cpu_state = torch.random.get_rng_state()
        torch.manual_seed(config.seed)
        model = cls(config)
        torch.random.set_rng_state(cpu_state)
        return model.eval().to(device)

    def _embed(
        self,
        ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        x = self.token_embedding(ids)
        if self.position_embedding is not None:
            x = x + self.position_embedding(
                positions
            )
        return x

    @torch.inference_mode()
    def prefill(
        self,
        request_id: str,
        input_ids: list[int],
        cache: PagedKVCache,
    ) -> SequenceState:
        if not input_ids:
            raise ValueError("input_ids cannot be empty")
        if (
            len(input_ids)
            > self.config.max_sequence_length
        ):
            raise ValueError(
                "prompt exceeds maximum sequence length"
            )

        cache.reserve(request_id, len(input_ids))
        device = self.token_embedding.weight.device
        ids = torch.tensor(
            input_ids,
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)
        positions = torch.arange(
            ids.shape[1], device=device
        ).unsqueeze(0)
        x = self._embed(ids, positions)

        for layer_index, block in enumerate(
            self.blocks
        ):
            x, k, v = block.prefill(
                x, positions
            )
            cache.write_prefill(
                request_id,
                layer_index,
                k,
                v,
            )

        logits = self.lm_head(
            self.final_norm(x[:, -1:, :])
        )[0, 0]
        return SequenceState(
            request_id=request_id,
            next_logits=logits,
            sequence_length=len(input_ids),
        )

    @torch.inference_mode()
    def decode_batch(
        self,
        states: list[SequenceState],
        current_tokens: list[int],
        cache: PagedKVCache,
    ) -> None:
        if len(states) != len(current_tokens):
            raise ValueError(
                "states and current_tokens must "
                "have the same length"
            )
        if not states:
            return

        device = self.token_embedding.weight.device
        positions = torch.tensor(
            [
                state.sequence_length
                for state in states
            ],
            dtype=torch.long,
            device=device,
        ).unsqueeze(1)
        if (
            int(positions.max().item())
            >= self.config.max_sequence_length
        ):
            raise ValueError(
                "sequence length exceeds model capacity"
            )

        ids = torch.tensor(
            current_tokens,
            dtype=torch.long,
            device=device,
        ).unsqueeze(1)
        x = self._embed(ids, positions)

        for layer_index, block in enumerate(
            self.blocks
        ):
            x = block.decode_batch(
                x,
                states,
                layer_index,
                cache,
                positions,
            )

        logits = self.lm_head(
            self.final_norm(x)
        )[:, 0, :]
        for request_index, state in enumerate(
            states
        ):
            state.next_logits = logits[
                request_index
            ]
            state.sequence_length += 1


TinyDecoderLM = PagedDecoderLM
