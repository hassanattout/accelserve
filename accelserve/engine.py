from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field

import torch

from accelserve.config import ModelConfig, RuntimeConfig
from accelserve.model import SequenceState, TinyDecoderLM
from accelserve.paged_kv import PagedKVCache
from accelserve.sampling import sample_next_token
from accelserve.tokenizer import ByteTokenizer


@dataclass
class GenerationRequest:
    prompt: str
    max_new_tokens: int
    temperature: float
    top_k: int
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: float = field(default_factory=time.perf_counter)


@dataclass
class GenerationResult:
    request_id: str
    text: str
    token_ids: list[int]
    prompt_tokens: int
    generated_tokens: int
    queue_ms: float
    ttft_ms: float
    tpot_ms: float
    generation_ms: float
    total_ms: float
    finish_reason: str


@dataclass
class _ActiveRequest:
    request: GenerationRequest
    prompt_ids: list[int]
    state: SequenceState
    generated_ids: list[int]
    current_token: int
    first_token_at: float
    future: asyncio.Future[GenerationResult]
    generator: torch.Generator


class ContinuousBatchingEngine:
    """Iteration-level scheduler using a physically paged K/V cache."""

    def __init__(
        self,
        *,
        model_config: ModelConfig | None = None,
        runtime_config: RuntimeConfig | None = None,
        device: str | None = None,
        model=None,
        tokenizer=None,
    ) -> None:
        self.runtime_config = runtime_config or RuntimeConfig()
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        if model is None:
            self.model_config = model_config or ModelConfig()
            self.model = TinyDecoderLM.deterministic(
                self.model_config, self.device
            )
            self.tokenizer = tokenizer or ByteTokenizer()
        else:
            self.model = model
            self.model_config = getattr(model, "config", model_config)
            if self.model_config is None:
                raise ValueError("a model config is required for custom models")
            self.tokenizer = tokenizer or ByteTokenizer()
            self.device = next(self.model.parameters()).device

        dtype = next(self.model.parameters()).dtype
        self.kv_cache = PagedKVCache(
            num_layers=self.model_config.num_layers,
            num_blocks=self.runtime_config.kv_num_blocks,
            block_size=self.runtime_config.kv_block_size,
            num_heads=self.model_config.kv_heads,
            head_dim=self.model_config.head_dim,
            device=self.device,
            dtype=dtype,
        )
        self.kv_allocator = self.kv_cache.allocator

        self._pending: asyncio.Queue[
            tuple[GenerationRequest, asyncio.Future[GenerationResult]]
        ] = asyncio.Queue(maxsize=self.runtime_config.max_pending_requests)
        self._active: dict[str, _ActiveRequest] = {}
        self._task: asyncio.Task[None] | None = None
        self._stopping = False

        self.completed_requests = 0
        self.failed_requests = 0
        self.total_generated_tokens = 0

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    @classmethod
    def from_qwen2_pretrained(
        cls,
        model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
        *,
        runtime_config: RuntimeConfig | None = None,
        device: str | None = None,
    ) -> "ContinuousBatchingEngine":
        from accelserve.open_weight import load_qwen2_pretrained

        loaded = load_qwen2_pretrained(model_id, device=device)
        runtime_config = runtime_config or RuntimeConfig(
            max_batch_size=8,
            kv_num_blocks=512,
        )
        return cls(
            model_config=loaded.config,
            runtime_config=runtime_config,
            device=str(next(loaded.model.parameters()).device),
            model=loaded.model,
            tokenizer=loaded.tokenizer,
        )

    async def start(self) -> None:
        if self.running:
            return
        self._stopping = False
        self._task = asyncio.create_task(
            self._run_loop(), name="accelserve-scheduler"
        )

    async def stop(self) -> None:
        self._stopping = True
        if self._task is not None:
            await self._task
        self._task = None

    async def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
    ) -> GenerationResult:
        if not self.running:
            await self.start()

        max_tokens = max_new_tokens or self.runtime_config.default_max_new_tokens
        if not 1 <= max_tokens <= self.runtime_config.hard_max_new_tokens:
            raise ValueError(
                f"max_new_tokens must be in [1, {self.runtime_config.hard_max_new_tokens}]"
            )
        if not prompt:
            raise ValueError("prompt cannot be empty")

        request = GenerationRequest(
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=self.runtime_config.temperature
            if temperature is None
            else temperature,
            top_k=self.runtime_config.top_k if top_k is None else top_k,
        )
        future = asyncio.get_running_loop().create_future()
        try:
            self._pending.put_nowait((request, future))
        except asyncio.QueueFull as exc:
            raise RuntimeError("request queue is full") from exc

        try:
            return await future
        except asyncio.CancelledError:
            active = self._active.pop(request.request_id, None)
            if active is not None:
                self.kv_cache.release(request.request_id)
            raise

    def stats(self) -> dict[str, int | float | str | bool]:
        return {
            "device": str(self.device),
            "running": self.running,
            "pending_requests": self._pending.qsize(),
            "active_requests": len(self._active),
            "completed_requests": self.completed_requests,
            "failed_requests": self.failed_requests,
            "total_generated_tokens": self.total_generated_tokens,
            "kv_used_blocks": self.kv_allocator.used_blocks,
            "kv_free_blocks": self.kv_allocator.free_blocks,
            "kv_utilization": self.kv_allocator.utilization,
            "kv_bytes_reserved": self.kv_cache.bytes_reserved,
            "kv_cache_mode": "physical_paged",
        }

    async def _admit_pending(self) -> None:
        while len(self._active) < self.runtime_config.max_batch_size:
            try:
                request, future = self._pending.get_nowait()
            except asyncio.QueueEmpty:
                return

            if future.cancelled():
                self._pending.task_done()
                continue

            try:
                prompt_ids = self.tokenizer.encode(request.prompt)
                max_sequence = len(prompt_ids) + request.max_new_tokens
                if max_sequence > self.model_config.max_sequence_length:
                    raise ValueError(
                        f"request requires {max_sequence} tokens but model capacity is "
                        f"{self.model_config.max_sequence_length}"
                    )

                state = self.model.prefill(
                    request.request_id, prompt_ids, self.kv_cache
                )
                generator = torch.Generator(device=self.device.type)
                generator.manual_seed(
                    self.model_config.seed ^ int(request.request_id[:8], 16)
                )
                first_token = sample_next_token(
                    state.next_logits,
                    temperature=request.temperature,
                    top_k=request.top_k,
                    generator=generator,
                )
                self.kv_cache.reserve(
                    request.request_id, len(prompt_ids) + 1
                )
                first_token_at = time.perf_counter()
                active = _ActiveRequest(
                    request=request,
                    prompt_ids=prompt_ids,
                    state=state,
                    generated_ids=[first_token],
                    current_token=first_token,
                    first_token_at=first_token_at,
                    future=future,
                    generator=generator,
                )
                self._active[request.request_id] = active
                if self._should_finish(active):
                    self._finish(active)
            except Exception as exc:
                self.kv_cache.release(request.request_id)
                self.failed_requests += 1
                if not future.done():
                    future.set_exception(exc)
            finally:
                self._pending.task_done()

    def _should_finish(self, active: _ActiveRequest) -> bool:
        return (
            active.current_token == self.tokenizer.EOS_ID
            or len(active.generated_ids) >= active.request.max_new_tokens
        )

    def _finish(self, active: _ActiveRequest) -> None:
        now = time.perf_counter()
        generated = len(active.generated_ids)
        ttft_ms = (
            active.first_token_at - active.request.created_at
        ) * 1000.0
        post_first_ms = max(
            0.0, (now - active.first_token_at) * 1000.0
        )
        result = GenerationResult(
            request_id=active.request.request_id,
            text=self.tokenizer.decode(active.generated_ids),
            token_ids=list(active.generated_ids),
            prompt_tokens=len(active.prompt_ids),
            generated_tokens=generated,
            queue_ms=ttft_ms,
            ttft_ms=ttft_ms,
            tpot_ms=post_first_ms / (generated - 1)
            if generated > 1
            else 0.0,
            generation_ms=post_first_ms,
            total_ms=(now - active.request.created_at) * 1000.0,
            finish_reason="stop"
            if active.current_token == self.tokenizer.EOS_ID
            else "length",
        )
        self.completed_requests += 1
        self.total_generated_tokens += generated
        self.kv_cache.release(active.request.request_id)
        self._active.pop(active.request.request_id, None)
        if not active.future.done():
            active.future.set_result(result)

    async def _decode_iteration(self) -> None:
        batch = list(self._active.values())[
            : self.runtime_config.max_batch_size
        ]
        if not batch:
            return

        self.model.decode_batch(
            [item.state for item in batch],
            [item.current_token for item in batch],
            self.kv_cache,
        )

        for active in batch:
            try:
                token = sample_next_token(
                    active.state.next_logits,
                    temperature=active.request.temperature,
                    top_k=active.request.top_k,
                    generator=active.generator,
                )
                active.current_token = token
                active.generated_ids.append(token)
                self.kv_cache.reserve(
                    active.request.request_id,
                    len(active.prompt_ids) + len(active.generated_ids),
                )
                if self._should_finish(active):
                    self._finish(active)
            except Exception as exc:
                request_id = active.request.request_id
                self.kv_cache.release(request_id)
                self._active.pop(request_id, None)
                self.failed_requests += 1
                if not active.future.done():
                    active.future.set_exception(exc)

    async def _run_loop(self) -> None:
        while not self._stopping or not self._pending.empty() or self._active:
            await self._admit_pending()
            await self._decode_iteration()
            await asyncio.sleep(
                self.runtime_config.scheduler_tick_ms / 1000.0
            )
