from __future__ import annotations

from contextlib import asynccontextmanager
import os
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from accelserve.config import RuntimeConfig
from accelserve.engine import ContinuousBatchingEngine
from accelserve.metrics import (
    GENERATED_TOKENS,
    QUEUE_LATENCY,
    REQUEST_LATENCY,
    REQUESTS,
    TPOT,
    TTFT,
    sync_runtime_gauges,
)


def build_engine() -> ContinuousBatchingEngine:
    model_id = os.getenv("ACCELSERVE_MODEL_ID")
    if not model_id:
        return ContinuousBatchingEngine()

    runtime = RuntimeConfig(
        kv_num_blocks=int(
            os.getenv("ACCELSERVE_KV_BLOCKS", "512")
        ),
        max_batch_size=int(
            os.getenv("ACCELSERVE_MAX_BATCH", "8")
        ),
    )
    return ContinuousBatchingEngine.from_qwen2_pretrained(
        model_id,
        runtime_config=runtime,
        device=os.getenv("ACCELSERVE_DEVICE"),
    )


engine = build_engine()


@asynccontextmanager
async def lifespan(_: FastAPI):
    await engine.start()
    try:
        yield
    finally:
        await engine.stop()


app = FastAPI(
    title="AccelServe Runtime",
    description=(
        "Continuous-batched LLM inference runtime with physically paged K/V "
        "memory, explicit prefill/decode and OpenAI-style completions."
    ),
    version="0.6.0",
    lifespan=lifespan,
)


class CompletionRequest(BaseModel):
    model: str = "accelserve"
    prompt: str = Field(min_length=1, max_length=32768)
    max_tokens: int = Field(default=32, ge=1, le=256)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_k: int = Field(default=40, ge=0)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "runtime": "v0.6",
        **engine.stats(),
    }


@app.get("/v1/runtime/stats")
async def runtime_stats():
    stats = engine.stats()
    sync_runtime_gauges(stats)
    return stats


@app.get("/metrics")
async def metrics():
    sync_runtime_gauges(engine.stats())
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    started = time.perf_counter()
    try:
        result = await engine.generate(
            request.prompt,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
        )
    except ValueError as exc:
        REQUESTS.labels(status="invalid").inc()
        raise HTTPException(
            status_code=422, detail=str(exc)
        ) from exc
    except RuntimeError as exc:
        REQUESTS.labels(status="unavailable").inc()
        raise HTTPException(
            status_code=503, detail=str(exc)
        ) from exc
    except Exception as exc:
        REQUESTS.labels(status="error").inc()
        raise HTTPException(
            status_code=500, detail="generation failed"
        ) from exc

    REQUESTS.labels(status="success").inc()
    GENERATED_TOKENS.inc(result.generated_tokens)
    REQUEST_LATENCY.observe(result.total_ms / 1000.0)
    QUEUE_LATENCY.observe(result.queue_ms / 1000.0)
    TTFT.observe(result.ttft_ms / 1000.0)
    TPOT.observe(result.tpot_ms / 1000.0)
    sync_runtime_gauges(engine.stats())

    return {
        "id": f"cmpl-{result.request_id}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "text": result.text,
                "finish_reason": result.finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.generated_tokens,
            "total_tokens": (
                result.prompt_tokens
                + result.generated_tokens
            ),
        },
        "runtime": {
            "queue_ms": result.queue_ms,
            "ttft_ms": result.ttft_ms,
            "tpot_ms": result.tpot_ms,
            "generation_ms": result.generation_ms,
            "total_ms": result.total_ms,
            "wall_ms": (
                time.perf_counter() - started
            )
            * 1000.0,
            "device": str(engine.device),
            "kv_cache_mode": engine.stats()[
                "kv_cache_mode"
            ],
        },
    }
