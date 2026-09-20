"""Benchmark AccelServe request latency, TTFT, TPOT and token throughput."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time

from accelserve.config import ModelConfig, RuntimeConfig
from accelserve.engine import ContinuousBatchingEngine


def percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    return values[
        min(len(values) - 1, round((len(values) - 1) * p))
    ]


async def benchmark(
    concurrency: int,
    max_new_tokens: int,
    device: str | None,
) -> dict[str, float | int | str]:
    engine = ContinuousBatchingEngine(
        model_config=ModelConfig(),
        runtime_config=RuntimeConfig(
            max_batch_size=max(concurrency, 1),
            kv_num_blocks=512,
        ),
        device=device,
    )
    await engine.start()

    started = time.perf_counter()
    results = await asyncio.gather(
        *[
            engine.generate(
                f"benchmark request {i}",
                max_new_tokens=max_new_tokens,
                temperature=0,
            )
            for i in range(concurrency)
        ]
    )
    wall = time.perf_counter() - started
    stats = engine.stats()
    await engine.stop()

    latencies = [x.total_ms for x in results]
    ttfts = [x.ttft_ms for x in results]
    tpots = [x.tpot_ms for x in results]
    total_tokens = sum(x.generated_tokens for x in results)

    return {
        "backend": "accelserve-reference",
        "device": str(engine.device),
        "concurrency": concurrency,
        "max_new_tokens": max_new_tokens,
        "wall_time_s": wall,
        "tokens_generated": total_tokens,
        "tokens_per_second": total_tokens / wall,
        "latency_mean_ms": statistics.mean(latencies),
        "latency_p50_ms": percentile(latencies, 0.50),
        "latency_p95_ms": percentile(latencies, 0.95),
        "latency_p99_ms": percentile(latencies, 0.99),
        "ttft_mean_ms": statistics.mean(ttfts),
        "ttft_p50_ms": percentile(ttfts, 0.50),
        "ttft_p95_ms": percentile(ttfts, 0.95),
        "tpot_mean_ms": statistics.mean(tpots),
        "tpot_p50_ms": percentile(tpots, 0.50),
        "kv_bytes_reserved": int(stats["kv_bytes_reserved"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], default=None
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = asyncio.run(
        benchmark(
            args.concurrency,
            args.max_new_tokens,
            args.device,
        )
    )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for key, value in result.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
