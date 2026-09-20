import asyncio

import torch
from fastapi.testclient import TestClient

from accelserve.config import ModelConfig, RuntimeConfig
from accelserve.engine import ContinuousBatchingEngine
from accelserve.model import TinyDecoderLM
from accelserve.paged_kv import KVBlockAllocator, PagedKVCache
from accelserve.tokenizer import ByteTokenizer


def tiny_config(**overrides):
    values = dict(
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        intermediate_size=64,
        max_sequence_length=128,
    )
    values.update(overrides)
    return ModelConfig(**values)


def make_cache(config, num_blocks=64, block_size=4):
    return PagedKVCache(
        num_layers=config.num_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        num_heads=config.kv_heads,
        head_dim=config.head_dim,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def test_byte_tokenizer_roundtrip():
    tokenizer = ByteTokenizer()
    text = "AccelServe ⚡"
    ids = tokenizer.encode(text)
    assert ids[0] == tokenizer.BOS_ID
    assert tokenizer.decode(ids) == text


def test_kv_allocator_reserve_grow_release():
    allocator = KVBlockAllocator(
        block_size=4, num_blocks=8
    )
    allocation = allocator.reserve("req-1", 5)
    assert len(allocation.blocks) == 2
    assert allocator.used_blocks == 2
    allocation = allocator.reserve("req-1", 9)
    assert len(allocation.blocks) == 3
    allocator.release("req-1")
    assert allocator.used_blocks == 0
    assert allocator.free_blocks == 8


def test_kv_allocator_enforces_capacity():
    allocator = KVBlockAllocator(
        block_size=4, num_blocks=1
    )
    allocator.reserve("a", 4)
    try:
        allocator.reserve("b", 1)
    except MemoryError:
        pass
    else:
        raise AssertionError(
            "expected KV-cache exhaustion"
        )


def test_physical_cache_prefill_read_append_release():
    config = tiny_config()
    cache = make_cache(config)
    request_id = "r1"
    k = torch.arange(
        1
        * config.kv_heads
        * 6
        * config.head_dim,
        dtype=torch.float32,
    ).view(
        1,
        config.kv_heads,
        6,
        config.head_dim,
    )
    v = k + 1000
    cache.write_prefill(request_id, 0, k, v)
    read_k, read_v = cache.read(
        request_id, 0, 6
    )
    assert torch.equal(read_k, k)
    assert torch.equal(read_v, v)

    new_k = torch.ones(
        (
            1,
            config.kv_heads,
            1,
            config.head_dim,
        )
    ) * 77
    new_v = torch.ones_like(new_k) * 88
    cache.append(
        request_id, 0, 6, new_k, new_v
    )
    read_k, read_v = cache.read(
        request_id, 0, 7
    )
    assert torch.equal(
        read_k[:, :, 6:7, :], new_k
    )
    assert torch.equal(
        read_v[:, :, 6:7, :], new_v
    )
    assert cache.allocator.used_blocks == 2

    cache.release(request_id)
    assert cache.allocator.used_blocks == 0


def test_physical_cache_is_preallocated_and_reused():
    config = tiny_config()
    cache = make_cache(
        config, num_blocks=8, block_size=4
    )
    key_ptr = cache.key.data_ptr()
    value_ptr = cache.value.data_ptr()
    bytes_reserved = cache.bytes_reserved
    cache.reserve("a", 7)
    cache.release("a")
    cache.reserve("b", 7)
    assert cache.key.data_ptr() == key_ptr
    assert cache.value.data_ptr() == value_ptr
    assert cache.bytes_reserved == bytes_reserved


def test_gqa_rope_model_prefill_and_decode():
    config = tiny_config(
        num_heads=4,
        num_kv_heads=2,
        use_rope=True,
        use_learned_positions=False,
        qkv_bias=True,
    )
    model = TinyDecoderLM.deterministic(
        config, torch.device("cpu")
    )
    cache = make_cache(config, num_blocks=64)
    tokenizer = ByteTokenizer()
    state = model.prefill(
        "gqa", tokenizer.encode("hello"), cache
    )
    initial_length = state.sequence_length
    model.decode_batch(
        [state],
        [
            tokenizer.encode(
                "x", add_bos=False
            )[0]
        ],
        cache,
    )
    assert (
        state.sequence_length
        == initial_length + 1
    )
    assert state.next_logits.shape == (
        config.vocab_size,
    )
    k, v = cache.read(
        "gqa", 0, state.sequence_length
    )
    assert k.shape[1] == config.kv_heads
    assert v.shape == k.shape


def test_batched_decode_with_paged_cache_matches_independent():
    config = tiny_config()
    tokenizer = ByteTokenizer()
    batch_model = TinyDecoderLM.deterministic(
        config, torch.device("cpu")
    )
    single_model = TinyDecoderLM.deterministic(
        config, torch.device("cpu")
    )
    batch_cache = make_cache(config, 128)
    single_cache = make_cache(config, 128)

    state_a = batch_model.prefill(
        "a", tokenizer.encode("ab"), batch_cache
    )
    state_b = batch_model.prefill(
        "b",
        tokenizer.encode("abcdef"),
        batch_cache,
    )
    one_a = single_model.prefill(
        "a",
        tokenizer.encode("ab"),
        single_cache,
    )
    one_b = single_model.prefill(
        "b",
        tokenizer.encode("abcdef"),
        single_cache,
    )
    tokens = [
        tokenizer.encode(
            "x", add_bos=False
        )[0],
        tokenizer.encode(
            "y", add_bos=False
        )[0],
    ]

    batch_model.decode_batch(
        [state_a, state_b],
        tokens,
        batch_cache,
    )
    single_model.decode_batch(
        [one_a], tokens[:1], single_cache
    )
    single_model.decode_batch(
        [one_b], tokens[1:], single_cache
    )

    assert torch.allclose(
        state_a.next_logits,
        one_a.next_logits,
        atol=1e-5,
    )
    assert torch.allclose(
        state_b.next_logits,
        one_b.next_logits,
        atol=1e-5,
    )


def test_engine_generates_concurrently_and_releases_cache():
    async def run():
        engine = ContinuousBatchingEngine(
            model_config=tiny_config(),
            runtime_config=RuntimeConfig(
                max_batch_size=4,
                kv_block_size=4,
                kv_num_blocks=128,
                scheduler_tick_ms=0,
                hard_max_new_tokens=8,
            ),
            device="cpu",
        )
        await engine.start()
        results = await asyncio.gather(
            engine.generate(
                "one",
                max_new_tokens=4,
                temperature=0,
            ),
            engine.generate(
                "two",
                max_new_tokens=4,
                temperature=0,
            ),
            engine.generate(
                "three",
                max_new_tokens=4,
                temperature=0,
            ),
        )
        assert all(
            result.generated_tokens == 4
            for result in results
        )
        assert all(
            result.ttft_ms >= 0
            for result in results
        )
        assert all(
            result.tpot_ms >= 0
            for result in results
        )
        stats = engine.stats()
        assert (
            stats["kv_cache_mode"]
            == "physical_paged"
        )
        assert stats["kv_bytes_reserved"] > 0
        assert stats["kv_used_blocks"] == 0
        assert engine.completed_requests == 3
        await engine.stop()

    asyncio.run(run())


def test_engine_rejects_sequence_over_capacity():
    async def run():
        engine = ContinuousBatchingEngine(
            model_config=tiny_config(
                max_sequence_length=8,
                num_layers=1,
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
            await engine.generate(
                "abcdef",
                max_new_tokens=4,
                temperature=0,
            )
        except ValueError as exc:
            assert "model capacity" in str(exc)
        else:
            raise AssertionError(
                "expected capacity error"
            )
        await engine.stop()

    asyncio.run(run())


def test_openai_style_completions_api_reports_v06_timings():
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
        assert (
            data["object"]
            == "text_completion"
        )
        assert (
            data["usage"][
                "completion_tokens"
            ]
            == 2
        )
        assert (
            data["runtime"]["device"]
            == "cpu"
        )
        assert (
            data["runtime"][
                "kv_cache_mode"
            ]
            == "physical_paged"
        )
        assert (
            data["runtime"]["ttft_ms"] >= 0
        )
        assert (
            data["runtime"]["tpot_ms"] >= 0
        )


def test_metrics_endpoint_exposes_v06_metrics():
    from accelserve.api import app

    with TestClient(app) as client:
        response = client.get("/metrics")
        assert response.status_code == 200
        assert (
            "accelserve_v06_ttft_seconds"
            in response.text
        )
        assert (
            "accelserve_v06_tpot_seconds"
            in response.text
        )
        assert (
            "accelserve_v06_kv_cache_bytes_reserved"
            in response.text
        )
