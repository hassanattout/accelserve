"""NVTX workload entry point for Nsight profiling."""

from __future__ import annotations

import argparse
import asyncio

import torch

from accelserve.config import RuntimeConfig
from accelserve.engine import ContinuousBatchingEngine


async def run(
    model_id: str,
    requests: int,
    max_tokens: int,
) -> None:
    engine = ContinuousBatchingEngine.from_qwen2_pretrained(
        model_id,
        runtime_config=RuntimeConfig(
            max_batch_size=requests,
            kv_num_blocks=512,
        ),
        device="cuda",
    )
    await engine.start()
    torch.cuda.nvtx.range_push(
        "accelserve_qwen_batch"
    )
    try:
        await asyncio.gather(
            *[
                engine.generate(
                    f"Profile inference request {i}:",
                    max_new_tokens=max_tokens,
                    temperature=0,
                    top_k=0,
                )
                for i in range(requests)
            ]
        )
    finally:
        torch.cuda.nvtx.range_pop()
        await engine.stop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
    )
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument(
        "--max-tokens", type=int, default=32
    )
    args = parser.parse_args()
    asyncio.run(
        run(
            args.model,
            args.requests,
            args.max_tokens,
        )
    )


if __name__ == "__main__":
    main()
