"""Optional Triton RoPE kernel for GPU performance experiments."""

from __future__ import annotations

import torch


def triton_available() -> bool:
    try:
        import triton  # noqa: F401
        import triton.language  # noqa: F401
    except ImportError:
        return False
    return torch.cuda.is_available()


def rope_reference(
    x: torch.Tensor,
    positions: torch.Tensor,
    theta: float = 10000.0,
) -> torch.Tensor:
    """Apply Qwen-style rotate-half RoPE to [B, H, T, D]."""
    if x.ndim != 4:
        raise ValueError("x must be [batch, heads, tokens, head_dim]")
    if x.shape[-1] % 2:
        raise ValueError("head_dim must be even")
    b, _, t, d = x.shape
    if positions.shape != (b, t):
        raise ValueError("positions must be [batch, tokens]")

    inv_freq = 1.0 / (
        theta
        ** (
            torch.arange(
                0, d, 2, device=x.device, dtype=torch.float32
            )
            / d
        )
    )
    freqs = positions.float().unsqueeze(-1) * inv_freq
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(x.dtype).unsqueeze(1)
    sin = emb.sin().to(x.dtype).unsqueeze(1)
    half = d // 2
    rotated = torch.cat(
        (-x[..., half:], x[..., :half]), dim=-1
    )
    return x * cos + rotated * sin


def rope_triton(
    x: torch.Tensor,
    positions: torch.Tensor,
    theta: float = 10000.0,
) -> torch.Tensor:
    if not triton_available():
        raise RuntimeError(
            "Triton RoPE requires Triton and an NVIDIA GPU"
        )
    if x.ndim != 4 or not x.is_cuda:
        raise ValueError(
            "x must be a CUDA tensor [batch, heads, tokens, head_dim]"
        )
    if positions.ndim != 2 or not positions.is_cuda:
        raise ValueError(
            "positions must be a CUDA tensor [batch, tokens]"
        )

    import triton
    import triton.language as tl

    @triton.jit
    def _kernel(
        x_ptr,
        pos_ptr,
        out_ptr,
        heads: tl.constexpr,
        tokens: tl.constexpr,
        dim: tl.constexpr,
        theta: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < dim
        half = dim // 2

        token = row % tokens
        batch_head = row // tokens
        batch = batch_head // heads

        x_base = row * dim
        values = tl.load(
            x_ptr + x_base + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        partner_offsets = tl.where(
            offsets < half,
            offsets + half,
            offsets - half,
        )
        partner = tl.load(
            x_ptr + x_base + partner_offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        rotated = tl.where(
            offsets < half,
            -partner,
            partner,
        )

        freq_index = offsets % half
        exponent = (2.0 * freq_index) / dim
        inv_freq = tl.exp(-tl.log(theta) * exponent)
        position = tl.load(
            pos_ptr + batch * tokens + token
        ).to(tl.float32)
        angle = position * inv_freq
        output = values * tl.cos(angle) + rotated * tl.sin(angle)
        tl.store(
            out_ptr + x_base + offsets,
            output,
            mask=mask,
        )

    b, h, t, d = x.shape
    if d % 2:
        raise ValueError("head_dim must be even")
    if positions.shape != (b, t):
        raise ValueError("positions must be [batch, tokens]")

    output = torch.empty_like(x)
    block = triton.next_power_of_2(d)
    _kernel[(b * h * t,)](
        x,
        positions,
        output,
        heads=h,
        tokens=t,
        dim=d,
        theta=theta,
        BLOCK=block,
    )
    return output
