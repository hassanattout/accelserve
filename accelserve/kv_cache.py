from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil


@dataclass
class CacheAllocation:
    request_id: str
    blocks: list[int] = field(default_factory=list)
    tokens_reserved: int = 0


class KVBlockAllocator:
    """Logical block allocator for KV-cache capacity accounting.

    The reference PyTorch backend stores tensors per request. This allocator
    models the page/block lifecycle used by production paged-KV systems so the
    scheduler can enforce a finite cache budget and expose cache utilization.
    """

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
