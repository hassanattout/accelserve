from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil

import torch


@dataclass
class CacheAllocation:
    request_id: str
    blocks: list[int] = field(default_factory=list)
    tokens_reserved: int = 0


class KVBlockAllocator:
    """Owns the physical block table for paged KV memory."""

    def __init__(self, *, block_size: int, num_blocks: int) -> None:
        if block_size <= 0 or num_blocks <= 0:
            raise ValueError("block_size and num_blocks must be positive")
        self.block_size = block_size
        self.num_blocks = num_blocks
        self._free_blocks = list(range(num_blocks - 1, -1, -1))
        self._allocations: dict[str, CacheAllocation] = {}

    @property
    def free_blocks(self) -> int:
        return len(self._free_blocks)

    @property
    def used_blocks(self) -> int:
        return self.num_blocks - self.free_blocks

    @property
    def utilization(self) -> float:
        return self.used_blocks / self.num_blocks

    def reserve(self, request_id: str, tokens: int) -> CacheAllocation:
        if tokens <= 0:
            raise ValueError("tokens must be positive")
        allocation = self._allocations.setdefault(
            request_id, CacheAllocation(request_id=request_id)
        )
        required_blocks = ceil(tokens / self.block_size)
        additional_blocks = required_blocks - len(allocation.blocks)
        if additional_blocks > self.free_blocks:
            raise MemoryError(
                f"KV cache exhausted: need {additional_blocks} additional blocks, "
                f"only {self.free_blocks} available"
            )
        for _ in range(additional_blocks):
            allocation.blocks.append(self._free_blocks.pop())
        allocation.tokens_reserved = max(allocation.tokens_reserved, tokens)
        return allocation

    def release(self, request_id: str) -> None:
        allocation = self._allocations.pop(request_id, None)
        if allocation is None:
            return
        self._free_blocks.extend(allocation.blocks)

    def allocation_for(self, request_id: str) -> CacheAllocation | None:
        return self._allocations.get(request_id)


class PagedKVCache:
    """Physically preallocated K/V slabs addressed through per-request block tables.

    Layout: [layer, physical_block, kv_head, token_in_block, head_dim]

    Reads currently gather pages into contiguous tensors for PyTorch attention.
    A direct paged-attention kernel is the next optimization milestone.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.device = device
        self.dtype = dtype
        self.allocator = KVBlockAllocator(block_size=block_size, num_blocks=num_blocks)
        shape = (num_layers, num_blocks, num_heads, block_size, head_dim)
        self.key = torch.empty(shape, dtype=dtype, device=device)
        self.value = torch.empty(shape, dtype=dtype, device=device)

    @property
    def bytes_reserved(self) -> int:
        return (self.key.numel() + self.value.numel()) * self.key.element_size()

    def reserve(self, request_id: str, tokens: int) -> CacheAllocation:
        return self.allocator.reserve(request_id, tokens)

    def release(self, request_id: str) -> None:
        self.allocator.release(request_id)

    def allocation_for(self, request_id: str) -> CacheAllocation | None:
        return self.allocator.allocation_for(request_id)

    def write_prefill(
        self,
        request_id: str,
        layer: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        if k.shape != v.shape or k.ndim != 4 or k.shape[0] != 1:
            raise ValueError("k/v must be [1, heads, tokens, head_dim]")
        if k.shape[1] != self.num_heads or k.shape[3] != self.head_dim:
            raise ValueError("k/v shape does not match cache geometry")
        tokens = k.shape[2]
        allocation = self.reserve(request_id, tokens)
        start = 0
        for logical_block, physical_block in enumerate(allocation.blocks):
            block_start = logical_block * self.block_size
            if block_start >= tokens:
                break
            take = min(self.block_size, tokens - block_start)
            self.key[layer, physical_block, :, :take, :].copy_(
                k[0, :, start : start + take, :]
            )
            self.value[layer, physical_block, :, :take, :].copy_(
                v[0, :, start : start + take, :]
            )
            start += take

    def append(
        self,
        request_id: str,
        layer: int,
        position: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        if position < 0:
            raise ValueError("position must be non-negative")
        if k.shape != v.shape or k.shape != (1, self.num_heads, 1, self.head_dim):
            raise ValueError("decode k/v must be [1, heads, 1, head_dim]")
        allocation = self.reserve(request_id, position + 1)
        logical_block = position // self.block_size
        offset = position % self.block_size
        physical_block = allocation.blocks[logical_block]
        self.key[layer, physical_block, :, offset, :].copy_(k[0, :, 0, :])
        self.value[layer, physical_block, :, offset, :].copy_(v[0, :, 0, :])

    def read(
        self,
        request_id: str,
        layer: int,
        length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if length <= 0:
            raise ValueError("length must be positive")
        allocation = self.allocator.allocation_for(request_id)
        if allocation is None:
            raise KeyError(f"no cache allocation for request {request_id}")
        if length > allocation.tokens_reserved:
            raise ValueError("requested length exceeds reserved tokens")

        key_parts = []
        value_parts = []
        remaining = length
        for physical_block in allocation.blocks:
            if remaining <= 0:
                break
            take = min(self.block_size, remaining)
            key_parts.append(self.key[layer, physical_block, :, :take, :])
            value_parts.append(self.value[layer, physical_block, :, :take, :])
            remaining -= take
        if remaining != 0:
            raise RuntimeError("block table is inconsistent with requested length")

        k = torch.cat(key_parts, dim=1).unsqueeze(0)
        v = torch.cat(value_parts, dim=1).unsqueeze(0)
        return k, v
