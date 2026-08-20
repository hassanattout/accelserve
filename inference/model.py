import hashlib
import os
from pathlib import Path

import torch
import torch.nn as nn


INPUT_DIM = 1024
HIDDEN_DIM = 4096
OUTPUT_DIM = 1000

MODEL_SEED = 42


class InferenceMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, OUTPUT_DIM),
        )

    def forward(self, x):
        return self.net(x)


def create_model(checkpoint_path=None):
    torch.manual_seed(MODEL_SEED)
    model = InferenceMLP()

    checkpoint_path = checkpoint_path or os.getenv(
        "ACCELSERVE_CHECKPOINT_PATH"
    )
    if checkpoint_path:
        state_dict = torch.load(
            Path(checkpoint_path),
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state_dict)

    return model


def model_fingerprint(model):
    """Return a stable SHA-256 identity for a model state dictionary."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()
