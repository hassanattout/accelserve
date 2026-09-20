from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 259
    hidden_size: int = 128
    num_layers: int = 4
    num_heads: int = 4
    num_kv_heads: int | None = None
    intermediate_size: int = 512
    max_sequence_length: int = 1024
    seed: int = 42
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    use_rope: bool = False
    use_learned_positions: bool = True
    qkv_bias: bool = False
    tie_word_embeddings: bool = True

    @property
    def head_dim(self) -> int:
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        return self.hidden_size // self.num_heads

    @property
    def kv_heads(self) -> int:
        return self.num_kv_heads or self.num_heads

    @property
    def kv_groups(self) -> int:
        if self.num_heads % self.kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        return self.num_heads // self.kv_heads


@dataclass(frozen=True)
class RuntimeConfig:
    max_batch_size: int = 16
    max_pending_requests: int = 256
    kv_block_size: int = 16
    kv_num_blocks: int = 512
    scheduler_tick_ms: float = 1.0
    default_max_new_tokens: int = 32
    hard_max_new_tokens: int = 256
    temperature: float = 0.8
    top_k: int = 40
