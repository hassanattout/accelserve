from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import time
import torch
import torch.nn as nn


# ============================================================
# Configuration
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INPUT_DIM = 1024
HIDDEN_DIM = 4096
OUTPUT_DIM = 1000


# ============================================================
# Model
# ============================================================

class InferenceMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, OUTPUT_DIM)
        )

    def forward(self, x):
        return self.net(x)


model = InferenceMLP().eval().to(DEVICE)

if DEVICE == "cuda":
    model = model.half()


# ============================================================
# API schema
# ============================================================

class InferenceRequest(BaseModel):
    inputs: List[List[float]]


class InferenceResponse(BaseModel):
    outputs: List[List[float]]
    batch_size: int
    latency_ms: float
    device: str


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="AccelServe",
    description="GPU-accelerated AI inference service",
    version="0.1.0"
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available()
    }


@app.post(
    "/infer",
    response_model=InferenceResponse
)
def infer(request: InferenceRequest):

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

    dtype = (
        torch.float16
        if DEVICE == "cuda"
        else torch.float32
    )

    x = torch.tensor(
        request.inputs,
        dtype=dtype,
        device=DEVICE
    )

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.inference_mode():
        output = model(x)

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    end = time.perf_counter()

    latency_ms = (
        end - start
    ) * 1000.0

    output_cpu = (
        output
        .float()
        .cpu()
        .tolist()
    )

    return InferenceResponse(
        outputs=output_cpu,
        batch_size=len(request.inputs),
        latency_ms=latency_ms,
        device=DEVICE
    )