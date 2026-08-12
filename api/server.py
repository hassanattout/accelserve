import time

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)
from typing import List

from api.backends import (
    INPUT_DIM,
    create_backend,
)


backend = create_backend()


class InferenceRequest(BaseModel):
    inputs: List[List[float]]


class InferenceResponse(BaseModel):
    outputs: List[List[float]]
    batch_size: int
    latency_ms: float
    backend: str
    device: str


REQUEST_COUNT = Counter(
    "accelserve_requests_total",
    "Total inference requests",
    ["endpoint", "backend", "device"],
)

INFERENCE_LATENCY = Histogram(
    "accelserve_inference_latency_seconds",
    "Inference latency in seconds",
    ["backend", "device"],
)


app = FastAPI(
    title="AccelServe",
    description="GPU-accelerated AI inference service",
    version="0.2.0",
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "backend": backend.name,
        "device": backend.device,
        "cuda_available": torch.cuda.is_available(),
    }


@app.get("/metrics")
def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post(
    "/infer",
    response_model=InferenceResponse,
)
def infer(
    request: InferenceRequest,
):
    if len(request.inputs) == 0:
        raise HTTPException(
            status_code=400,
            detail="Input batch cannot be empty",
        )

    for row in request.inputs:
        if len(row) != INPUT_DIM:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Each input must contain "
                    f"{INPUT_DIM} values"
                ),
            )

    try:
        start = time.perf_counter()

        outputs = backend.infer(
            request.inputs
        )

        end = time.perf_counter()

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    latency_ms = (
        end - start
    ) * 1000.0

    REQUEST_COUNT.labels(
        endpoint="/infer",
        backend=backend.name,
        device=backend.device,
    ).inc()

    INFERENCE_LATENCY.labels(
        backend=backend.name,
        device=backend.device,
    ).observe(
        latency_ms / 1000.0
    )

    return InferenceResponse(
        outputs=outputs,
        batch_size=len(request.inputs),
        latency_ms=latency_ms,
        backend=backend.name,
        device=backend.device,
    )