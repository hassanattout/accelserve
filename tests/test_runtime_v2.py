import asyncio

import torch

from accelserve.config import ModelConfig, RuntimeConfig
from accelserve.engine import ContinuousBatchingEngine
from accelserve.kv_cache import KVBlockAllocator
from accelserve.model import TinyDecoderLM
from accelserve.tokenizer import ByteTokenizer


def test_byte_tokenizer_roundtrip():
    tokenizer = ByteTokenizer()
    text = "AccelServe ⚡"
    ids = tokenizer.encode(text)
    assert ids[0] == tokenizer.BOS_ID
    assert tokenizer.decode(ids) == text


def test_kv_allocator_reserve_grow_release():
    allocator = KVBlockAllocator(block_size=4, num_blocks=8)
    allocation = allocator.reserve("req-1", 5)
    assert len(allocation.blocks) == 2
    assert allocator.used_blocks == 2

    allocation = allocator.reserve("req-1", 9)
    assert len(allocation.blocks) == 3
    assert allocator.used_blocks == 3

    allocator.release("req-1")
    assert allocator.used_blocks == 0
    assert allocator.free_blocks == 8


def test_kv_allocator_enforces_capacity():
    allocator = KVBlockAllocator(block_size=4, num_blocks=1)
    allocator.reserve("a", 4)
    try:
        allocator.reserve("b", 1)
    except MemoryError:
        pass
    else:
        raise AssertionError("expected KV-cache exhaustion")


def test_prefill_then_batched_decode_updates_lengths_and_cache():
    config = ModelConfig(
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        intermediate_size=64,
        max_sequence_length=64,
    )
    model = TinyDecoderLM.deterministic(config, torch.device("cpu"))
    tokenizer = ByteTokenizer()

    state_a = model.prefill(tokenizer.encode("A"))
    state_b = model.prefill(tokenizer.encode("longer"))
    len_a = state_a.sequence_length
    len_b = state_b.sequence_length

    model.decode_batch([state_a, state_b], [68, 69])

    assert state_a.sequence_length == len_a + 1
    assert state_b.sequence_length == len_b + 1
    assert state_a.next_logits.shape == (config.vocab_size,)
    assert state_b.next_logits.shape == (config.vocab_size,)
    for key, value in state_a.past_key_values:
        assert key.shape[2] == state_a.sequence_length
        assert value.shape[2] == state_a.sequence_length


def test_engine_generates_and_releases_cache():
    async def run():
        engine = ContinuousBatchingEngine(
            model_config=ModelConfig(
                hidden_size=32,
                num_layers=2,
                num_heads=4,
                intermediate_size=64,
                max_sequence_length=128,
            ),
            runtime_config=RuntimeConfig(
                max_batch_size=4,
                kv_block_size=4,
                kv_num_blocks=64,
                scheduler_tick_ms=0,
                default_max_new_tokens=3,
                hard_max_new_tokens=8,
            ),
            device="cpu",
        )
        await engine.start()
        result = await engine.generate("hello", max_new_tokens=3, temperature=0)
        assert result.generated_tokens == 3
        assert result.prompt_tokens > 0
        assert result.finish_reason == "length"
        assert engine.completed_requests == 1
        assert engine.kv_allocator.used_blocks == 0
        await engine.stop()

    asyncio.run(run())


def test_engine_continuous_batching_handles_concurrent_requests():
    async def run():
        engine = ContinuousBatchingEngine(
            model_config=ModelConfig(
                hidden_size=32,
                num_layers=2,
                num_heads=4,
                intermediate_size=64,
                max_sequence_length=128,
            ),
            runtime_config=RuntimeConfig(
                max_batch_size=4,
                kv_block_size=4,
                kv_num_blocks=128,
                scheduler_tick_ms=0,
                default_max_new_tokens=4,
                hard_max_new_tokens=8,
            ),
            device="cpu",
        )
        await engine.start()
        results = await asyncio.gather(
            engine.generate("one", max_new_tokens=4, temperature=0),
            engine.generate("two", max_new_tokens=4, temperature=0),
            engine.generate("three", max_new_tokens=4, temperature=0),
        )
        assert len(results) == 3
        assert all(result.generated_tokens == 4 for result in results)
        assert engine.completed_requests == 3
        assert engine.total_generated_tokens == 12
        assert engine.kv_allocator.used_blocks == 0
        await engine.stop()

    asyncio.run(run())


def test_engine_rejects_sequence_over_capacity():
    async def run():
        engine = ContinuousBatchingEngine(
            model_config=ModelConfig(
                hidden_size=32,
                num_layers=1,
                num_heads=4,
                intermediate_size=64,
                max_sequence_length=8,
            ),
            runtime_config=RuntimeConfig(
                max_batch_size=2,
                kv_block_size=4,
                kv_num_blocks=32,
                scheduler_tick_ms=0,
                hard_max_new_tokens=8,
            ),
            device="cpu",
        )
        await engine.start()
        try:
            await engine.generate("abcdef", max_new_tokens=4, temperature=0)
        except ValueError as exc:
            assert "model capacity" in str(exc)
        else:
            raise AssertionError("expected capacity error")
        await engine.stop()

    asyncio.run(run())


def test_openai_style_completions_api():
    from fastapi.testclient import TestClient
    from accelserve.api import app

    with TestClient(app) as client:
        response = client.post(
            "/v1/completions",
            json={
                "model": "accelserve-tiny",
                "prompt": "hello",
                "max_tokens": 2,
                "temperature": 0,
                "top_k": 0,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "text_completion"
        assert data["usage"]["completion_tokens"] == 2
        assert data["runtime"]["device"] == "cpu"

        stats = client.get("/v1/runtime/stats")
        assert stats.status_code == 200
        assert stats.json()["completed_requests"] >= 1


def test_metrics_endpoint_exposes_v2_metrics():
    from fastapi.testclient import TestClient
    from accelserve.api import app

    with TestClient(app) as client:
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "accelserve_v2_requests_total" in response.text
        assert "accelserve_v2_kv_cache_utilization_ratio" in response.text


def test_batched_decode_matches_independent_decode():
    config = ModelConfig(
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        intermediate_size=64,
        max_sequence_length=64,
    )
    model_batch = TinyDecoderLM.deterministic(config, torch.device("cpu"))
    model_single = TinyDecoderLM.deterministic(config, torch.device("cpu"))
    tokenizer = ByteTokenizer()

    prompts = [tokenizer.encode("ab"), tokenizer.encode("abcdef")]
    batch_states = [model_batch.prefill(prompt) for prompt in prompts]
    single_states = [model_single.prefill(prompt) for prompt in prompts]
    tokens = [
        tokenizer.encode("x", add_bos=False)[0],
        tokenizer.encode("y", add_bos=False)[0],
    ]

    model_batch.decode_batch(batch_states, tokens)
    model_single.decode_batch([single_states[0]], [tokens[0]])
    model_single.decode_batch([single_states[1]], [tokens[1]])

    assert torch.allclose(
        batch_states[0].next_logits, single_states[0].next_logits, atol=1e-5
    )
    assert torch.allclose(
        batch_states[1].next_logits, single_states[1].next_logits, atol=1e-5
    )
