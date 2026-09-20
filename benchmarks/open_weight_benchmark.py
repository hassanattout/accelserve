"""Same-host throughput comparison for AccelServe, Transformers and vLLM.

This is a workload harness, not a universal leaderboard. Transformers uses
batched generation, AccelServe uses concurrent request scheduling, and vLLM
uses its normal request engine.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import time

import torch

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    return values[
        min(len(values) - 1, round((len(values) - 1) * p))
    ]


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


async def run_accelserve(
    model_id: str,
    prompts: list[str],
    max_tokens: int,
    device: str | None,
) -> dict:
    from accelserve.config import RuntimeConfig
    from accelserve.engine import ContinuousBatchingEngine

    engine = ContinuousBatchingEngine.from_qwen2_pretrained(
        model_id,
        runtime_config=RuntimeConfig(
            max_batch_size=len(prompts),
            kv_num_blocks=512,
        ),
        device=device,
    )
    await engine.start()
    started = time.perf_counter()
    results = await asyncio.gather(
        *[
            engine.generate(
                prompt,
                max_new_tokens=max_tokens,
                temperature=0,
                top_k=0,
            )
            for prompt in prompts
        ]
    )
    wall = time.perf_counter() - started
    await engine.stop()

    tokens = sum(x.generated_tokens for x in results)
    result = {
        "backend": "accelserve",
        "mode": "concurrent_requests",
        "requests": len(results),
        "wall_s": wall,
        "tokens": tokens,
        "tokens_per_second": tokens / wall,
        "latency_p50_ms": percentile(
            [x.total_ms for x in results], 0.5
        ),
        "latency_p95_ms": percentile(
            [x.total_ms for x in results], 0.95
        ),
        "ttft_p50_ms": percentile(
            [x.ttft_ms for x in results], 0.5
        ),
        "ttft_p95_ms": percentile(
            [x.ttft_ms for x in results], 0.95
        ),
        "tpot_mean_ms": sum(
            x.tpot_ms for x in results
        )
        / len(results),
    }
    del engine
    cleanup_cuda()
    return result


def run_transformers(
    model_id: str,
    prompts: list[str],
    max_tokens: int,
    device: str | None,
) -> dict:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    target = torch.device(
        device
        or (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    )
    dtype = (
        torch.bfloat16
        if target.type == "cuda"
        and torch.cuda.is_bf16_supported()
        else (
            torch.float16
            if target.type == "cuda"
            else torch.float32
        )
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).eval().to(target)

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
    ).to(target)

    if target.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            min_new_tokens=max_tokens,
        )
    if target.type == "cuda":
        torch.cuda.synchronize()
    wall = time.perf_counter() - started

    input_length = inputs["input_ids"].shape[1]
    tokens = int(
        sum(
            max(0, row.shape[0] - input_length)
            for row in output
        )
    )
    result = {
        "backend": "transformers-generate",
        "mode": "static_batch",
        "requests": len(prompts),
        "wall_s": wall,
        "tokens": tokens,
        "tokens_per_second": tokens / wall,
        "batch_latency_ms": wall * 1000.0,
    }

    del output, inputs, model, tokenizer
    cleanup_cuda()
    return result


def run_vllm(
    model_id: str,
    prompts: list[str],
    max_tokens: int,
) -> dict:
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError(
            "Install vLLM separately on a supported GPU host"
        ) from exc

    llm = LLM(model=model_id)
    params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        min_tokens=max_tokens,
    )
    started = time.perf_counter()
    outputs = llm.generate(
        prompts, params, use_tqdm=False
    )
    wall = time.perf_counter() - started
    tokens = sum(
        len(out.outputs[0].token_ids)
        for out in outputs
    )
    result = {
        "backend": "vllm",
        "mode": "request_engine",
        "requests": len(outputs),
        "wall_s": wall,
        "tokens": tokens,
        "tokens_per_second": tokens / wall,
    }
    del outputs, llm
    cleanup_cuda()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default=DEFAULT_MODEL
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=[
            "accelserve",
            "transformers",
            "vllm",
        ],
        default=["accelserve", "transformers"],
    )
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument(
        "--max-tokens", type=int, default=32
    )
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], default=None
    )
    args = parser.parse_args()

    prompts = [
        (
            "Explain one inference systems bottleneck. "
            f"Request {i}:"
        )
        for i in range(args.requests)
    ]

    results = []
    for backend in args.backends:
        if backend == "accelserve":
            results.append(
                asyncio.run(
                    run_accelserve(
                        args.model,
                        prompts,
                        args.max_tokens,
                        args.device,
                    )
                )
            )
        elif backend == "transformers":
            results.append(
                run_transformers(
                    args.model,
                    prompts,
                    args.max_tokens,
                    args.device,
                )
            )
        else:
            results.append(
                run_vllm(
                    args.model,
                    prompts,
                    args.max_tokens,
                )
            )

    print(
        json.dumps(
            {
                "model": args.model,
                "results": results,
                "note": (
                    "Backends use their natural serving modes; "
                    "compare throughput on the same hardware/software stack."
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
