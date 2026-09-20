from __future__ import annotations

import torch


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    generator: torch.Generator | None = None,
) -> int:
    """Sample one token from a 1-D logits tensor."""
    if logits.ndim != 1:
        raise ValueError("logits must be one-dimensional")

    if temperature <= 0:
        return int(torch.argmax(logits).item())

    scaled = logits / temperature

    if top_k > 0 and top_k < scaled.numel():
        values, indices = torch.topk(scaled, top_k)
        probs = torch.softmax(values, dim=-1)
        choice = torch.multinomial(probs, 1, generator=generator)
        return int(indices[choice].item())

    probs = torch.softmax(scaled, dim=-1)
    return int(torch.multinomial(probs, 1, generator=generator).item())
