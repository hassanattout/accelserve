import logging
import math
import time
from typing import Annotated

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

from api.backends import INPUT_DIM, create_backend

logger = logging.getLogger("accelserve")
backend = create_backend()
MAX_BATCH_SIZE = min(256, backend.maximum_batch_size)
MODEL_VERSION = f"demo-mlp-{backend.model_fingerprint[:12]}"


class InferenceRequest(BaseModel):
    inputs: Annotated[
        list[list[float]],
        Field(min_length=1, max_length=MAX_BATCH_SIZE),
    ]


class InferenceResponse(BaseModel):
    outputs: list[list[float]]
    batch_size: int
    latency_ms: float
    backend: str
    device: str
    model_version: str


REQUEST_COUNT = Counter(
    "accelserve_requests_total",
    "Total inference requests",
    ["endpoint", "backend", "device", "status"],
)

INFERENCE_LATENCY = Histogram(
    "accelserve_inference_latency_seconds",
    "Inference latency in seconds",
    ["backend", "device"],
)

app = FastAPI(
    title="AccelServe",
    description="GPU-accelerated AI inference service",
    version="0.3.0",
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "backend": backend.name,
        "device": backend.device,
        "cuda_available": torch.cuda.is_available(),
        "model_version": MODEL_VERSION,
        "model_fingerprint": backend.model_fingerprint,
        "input_dimension": INPUT_DIM,
        "maximum_batch_size": MAX_BATCH_SIZE,
    }


@app.get("/metrics")
def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


def validate_inputs(inputs: list[list[float]]) -> None:
    for row_index, row in enumerate(inputs):
        if len(row) != INPUT_DIM:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Input row {row_index} must contain "
                    f"{INPUT_DIM} values."
                ),
            )
        if not all(math.isfinite(value) for value in row):
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Input row {row_index} contains a non-finite value."
                ),
            )


@app.post("/infer", response_model=InferenceResponse)
def infer(request: InferenceRequest):
    validate_inputs(request.inputs)

    try:
        start = time.perf_counter()
        outputs = backend.infer(request.inputs)
        latency_ms = (time.perf_counter() - start) * 1000.0
    except Exception as exc:
        logger.exception("Inference backend failed")
        REQUEST_COUNT.labels(
            endpoint="/infer",
            backend=backend.name,
            device=backend.device,
            status="error",
        ).inc()
        raise HTTPException(
            status_code=500,
            detail="Inference backend failed.",
        ) from exc

    REQUEST_COUNT.labels(
        endpoint="/infer",
        backend=backend.name,
        device=backend.device,
        status="success",
    ).inc()
    INFERENCE_LATENCY.labels(
        backend=backend.name,
        device=backend.device,
    ).observe(latency_ms / 1000.0)

    return InferenceResponse(
        outputs=outputs,
        batch_size=len(request.inputs),
        latency_ms=latency_ms,
        backend=backend.name,
        device=backend.device,
        model_version=MODEL_VERSION,
    )
