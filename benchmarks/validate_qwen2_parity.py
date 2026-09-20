"""Validate AccelServe Qwen2 prefill logits against Transformers."""

from __future__ import annotations

import argparse
import json
import sys

import torch

from accelserve.config import RuntimeConfig
from accelserve.engine import ContinuousBatchingEngine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
    )
    parser.add_argument(
        "--prompt",
        default="The fastest way to reduce inference latency is",
    )
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], default=None
    )
    parser.add_argument(
        "--max-error",
        type=float,
        default=None,
        help="Fail if maximum absolute logit error exceeds this value.",
    )
    parser.add_argument(
        "--require-argmax",
        action="store_true",
        help="Fail unless AccelServe and Transformers select the same next token.",
    )
    args = parser.parse_args()

    try:
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
        )
    except ImportError as exc:
        raise SystemExit(
            "Install: pip install -e '.[open-weight]'"
        ) from exc

    engine = ContinuousBatchingEngine.from_qwen2_pretrained(
        args.model,
        runtime_config=RuntimeConfig(
            kv_num_blocks=128,
            max_batch_size=1,
        ),
        device=args.device,
    )
    target = engine.device
    dtype = next(engine.model.parameters()).dtype

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    hf = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).eval().to(target)

    ids = tokenizer.encode(
        args.prompt, add_special_tokens=True
    )
    request_id = "parity"
    state = engine.model.prefill(
        request_id, ids, engine.kv_cache
    )

    with torch.inference_mode():
        tensor = torch.tensor([ids], device=target)
        hf_logits = hf(
            input_ids=tensor, use_cache=True
        ).logits[0, -1].float()

    accel_logits = state.next_logits.float()
    diff = (accel_logits - hf_logits).abs()
    max_error = float(diff.max().item())
    mean_error = float(diff.mean().item())
    argmax_match = bool(
        torch.argmax(accel_logits).item()
        == torch.argmax(hf_logits).item()
    )
    result = {
        "model": args.model,
        "prompt_tokens": len(ids),
        "max_abs_logit_error": max_error,
        "mean_abs_logit_error": mean_error,
        "argmax_match": argmax_match,
        "accel_argmax": int(
            torch.argmax(accel_logits).item()
        ),
        "hf_argmax": int(
            torch.argmax(hf_logits).item()
        ),
    }
    engine.kv_cache.release(request_id)
    print(json.dumps(result, indent=2))

    failures = []
    if (
        args.max_error is not None
        and max_error > args.max_error
    ):
        failures.append(
            f"max_abs_logit_error={max_error:.6g} "
            f"> threshold={args.max_error:.6g}"
        )
    if args.require_argmax and not argmax_match:
        failures.append("next-token argmax differs")

    if failures:
        print(
            "Parity validation failed: "
            + "; ".join(failures),
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
