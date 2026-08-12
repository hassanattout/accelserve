import time
import torch

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from api.backends import (
    INPUT_DIM,
    create_backend
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


app = FastAPI(
    title="AccelServe",
    description="GPU-accelerated AI inference service",
    version="0.2.0"
)


@app.get("/health")
def health():

    return {
        "status": "ok",
        "backend": backend.name,
        "device": backend.device,
        "cuda_available":
            torch.cuda.is_available()
    }


@app.post(
    "/infer",
    response_model=InferenceResponse
)
def infer(
    request: InferenceRequest
):

    if len(request.inputs) == 0:
        raise HTTPException(
            status_code=400,
            detail="Input batch cannot be empty"
        )

    for row in request.inputs:

        if len(row) != INPUT_DIM:

            raise HTTPException(
                status_code=400,
                detail=(
                    f"Each input must contain "
                    f"{INPUT_DIM} values"
                )
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
            detail=str(exc)
        ) from exc

    return InferenceResponse(
        outputs=outputs,
        batch_size=len(request.inputs),
        latency_ms=(
            end - start
        ) * 1000.0,
        backend=backend.name,
        device=backend.device
    )