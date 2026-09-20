from __future__ import annotations

from dataclasses import dataclass

import torch

from accelserve.config import ModelConfig
from accelserve.model import PagedDecoderLM


class HFTokenizerAdapter:
    def __init__(self, tokenizer) -> None:
        self._tokenizer = tokenizer
        self.EOS_ID = tokenizer.eos_token_id
        if self.EOS_ID is None:
            raise ValueError("tokenizer must define eos_token_id")

    def encode(self, text: str, *, add_bos: bool = True) -> list[int]:
        return list(
            self._tokenizer.encode(
                text, add_special_tokens=add_bos
            )
        )

    def decode(self, ids: list[int]) -> str:
        return self._tokenizer.decode(
            ids, skip_special_tokens=True
        )


@dataclass
class LoadedOpenWeightModel:
    model: PagedDecoderLM
    tokenizer: HFTokenizerAdapter
    config: ModelConfig
    source_model_id: str


def _copy(dst: torch.nn.Parameter, src: torch.Tensor) -> None:
    if dst.shape != src.shape:
        raise ValueError(
            f"weight shape mismatch: destination {tuple(dst.shape)} "
            f"source {tuple(src.shape)}"
        )
    dst.data.copy_(
        src.to(device=dst.device, dtype=dst.dtype)
    )


def load_qwen2_pretrained(
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> LoadedOpenWeightModel:
    """Load Qwen2/Qwen2.5 weights into AccelServe's own decoder runtime."""
    try:
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Qwen loading requires the optional 'open-weight' dependencies. "
            "Install with: pip install -e '.[open-weight]'"
        ) from exc

    target_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if dtype is None:
        dtype = (
            torch.float16
            if target_device.type == "cuda"
            else torch.float32
        )

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    hf_model.eval()
    hf_config = hf_model.config

    arch_names = set(
        getattr(hf_config, "architectures", []) or []
    )
    if (
        not any("Qwen2" in name for name in arch_names)
        and getattr(hf_config, "model_type", None)
        not in {"qwen2", "qwen2_moe"}
    ):
        raise ValueError(
            f"{model_id} is not a supported Qwen2-family checkpoint"
        )

    config = ModelConfig(
        vocab_size=hf_config.vocab_size,
        hidden_size=hf_config.hidden_size,
        num_layers=hf_config.num_hidden_layers,
        num_heads=hf_config.num_attention_heads,
        num_kv_heads=getattr(
            hf_config,
            "num_key_value_heads",
            hf_config.num_attention_heads,
        ),
        intermediate_size=hf_config.intermediate_size,
        max_sequence_length=hf_config.max_position_embeddings,
        rms_norm_eps=getattr(
            hf_config, "rms_norm_eps", 1e-6
        ),
        rope_theta=float(
            getattr(hf_config, "rope_theta", 10000.0)
        ),
        use_rope=True,
        use_learned_positions=False,
        qkv_bias=True,
        tie_word_embeddings=bool(
            getattr(
                hf_config,
                "tie_word_embeddings",
                False,
            )
        ),
    )

    model = PagedDecoderLM(config).to(
        device=target_device, dtype=dtype
    )

    base = hf_model.model
    with torch.no_grad():
        _copy(
            model.token_embedding.weight,
            base.embed_tokens.weight,
        )
        for dst, src in zip(
            model.blocks, base.layers, strict=True
        ):
            _copy(
                dst.attn_norm.weight,
                src.input_layernorm.weight,
            )
            _copy(
                dst.ffn_norm.weight,
                src.post_attention_layernorm.weight,
            )
            _copy(
                dst.q_proj.weight,
                src.self_attn.q_proj.weight,
            )
            _copy(
                dst.k_proj.weight,
                src.self_attn.k_proj.weight,
            )
            _copy(
                dst.v_proj.weight,
                src.self_attn.v_proj.weight,
            )
            if dst.q_proj.bias is not None:
                _copy(
                    dst.q_proj.bias,
                    src.self_attn.q_proj.bias,
                )
                _copy(
                    dst.k_proj.bias,
                    src.self_attn.k_proj.bias,
                )
                _copy(
                    dst.v_proj.bias,
                    src.self_attn.v_proj.bias,
                )
            _copy(
                dst.o_proj.weight,
                src.self_attn.o_proj.weight,
            )
            _copy(
                dst.gate_proj.weight,
                src.mlp.gate_proj.weight,
            )
            _copy(
                dst.up_proj.weight,
                src.mlp.up_proj.weight,
            )
            _copy(
                dst.down_proj.weight,
                src.mlp.down_proj.weight,
            )

        _copy(model.final_norm.weight, base.norm.weight)
        if not config.tie_word_embeddings:
            _copy(
                model.lm_head.weight,
                hf_model.lm_head.weight,
            )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    del hf_model
    if target_device.type == "cuda":
        torch.cuda.empty_cache()

    return LoadedOpenWeightModel(
        model=model.eval(),
        tokenizer=HFTokenizerAdapter(tokenizer),
        config=config,
        source_model_id=model_id,
    )
