"""CPU/GPU benchmark for AccelServe v2 scheduling and decode throughput."""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time

from accelserve.config import ModelConfig, RuntimeConfig
from accelserve.engine import ContinuousBatchingEngine


async def benchmark(concurrency: int, max_new_tokens: int, device: str | None) -> None:
    engine = ContinuousBatchingEngine(
        model_config=ModelConfig(),
        runtime_config=RuntimeConfig(max_batch_size=max(concurrency, 1)),
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
    await engine.stop()

    latencies = sorted(result.total_ms for result in results)
    total_tokens = sum(result.generated_tokens for result in results)

    def percentile(values: list[float], p: float) -> float:
        if not values:
            return 0.0
        index = min(len(values) - 1, round((len(values) - 1) * p))
        return values[index]

    print(f"device: {engine.device}")
    print(f"concurrency: {concurrency}")
    print(f"max_new_tokens: {max_new_tokens}")
    print(f"wall_time_s: {wall:.4f}")
    print(f"tokens_generated: {total_tokens}")
    print(f"tokens_per_second: {total_tokens / wall:.2f}")
    print(f"latency_mean_ms: {statistics.mean(latencies):.2f}")
    print(f"latency_p50_ms: {percentile(latencies, 0.50):.2f}")
    print(f"latency_p95_ms: {percentile(latencies, 0.95):.2f}")
    print(f"latency_p99_ms: {percentile(latencies, 0.99):.2f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    args = parser.parse_args()
    asyncio.run(benchmark(args.concurrency, args.max_new_tokens, args.device))


if __name__ == "__main__":
    main()
