from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 259
    hidden_size: int = 128
    num_layers: int = 4
    num_heads: int = 4
    intermediate_size: int = 512
    max_sequence_length: int = 1024
    seed: int = 42

    @property
    def head_dim(self) -> int:
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        return self.hidden_size // self.num_heads


@dataclass(frozen=True)
class RuntimeConfig:
    max_batch_size: int = 16
    max_pending_requests: int = 256
    kv_block_size: int = 16
    kv_num_blocks: int = 4096
    scheduler_tick_ms: float = 1.0
    default_max_new_tokens: int = 32
    hard_max_new_tokens: int = 256
    temperature: float = 0.8
    top_k: int = 40
