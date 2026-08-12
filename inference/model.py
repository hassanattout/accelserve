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


def create_model():
    torch.manual_seed(MODEL_SEED)

    model = InferenceMLP()

    return model
