"""Benchmark Triton RMSNorm/RoPE against PyTorch references on one GPU."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from kernels.triton_rmsnorm import (
    rmsnorm_reference,
    rmsnorm_triton,
)
from kernels.triton_rope import (
    rope_reference,
    rope_triton,
)


def bench(fn, *args, warmup: int = 20, runs: int = 100):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        fn(*args)
        torch.cuda.synchronize()
        samples.append(
            (time.perf_counter() - start) * 1000.0
        )
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.mean(samples),
        "min_ms": min(samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=896)
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=14)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--head-dim", type=int, default=64)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")

    x = torch.randn(
        args.rows,
        args.hidden,
        device="cuda",
        dtype=torch.float16,
    )
    weight = torch.ones(
        args.hidden,
        device="cuda",
        dtype=torch.float16,
    )
    q = torch.randn(
        1,
        args.heads,
        args.tokens,
        args.head_dim,
        device="cuda",
        dtype=torch.float16,
    )
    positions = torch.arange(
        args.tokens, device="cuda"
    ).unsqueeze(0)

    rms_ref = rmsnorm_reference(x, weight)
    rms_tri = rmsnorm_triton(x, weight)
    rope_ref = rope_reference(q, positions)
    rope_tri = rope_triton(q, positions)

    result = {
        "gpu": torch.cuda.get_device_name(0),
        "rmsnorm_max_abs_error": float(
            (rms_ref - rms_tri).abs().max().item()
        ),
        "rope_max_abs_error": float(
            (rope_ref - rope_tri).abs().max().item()
        ),
        "rmsnorm_pytorch": bench(
            rmsnorm_reference, x, weight
        ),
        "rmsnorm_triton": bench(
            rmsnorm_triton, x, weight
        ),
        "rope_pytorch": bench(
            rope_reference, q, positions
        ),
        "rope_triton": bench(
            rope_triton, q, positions
        ),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
