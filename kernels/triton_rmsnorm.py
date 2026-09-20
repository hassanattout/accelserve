"""Optional Triton RMSNorm kernel for GPU performance experiments."""

from __future__ import annotations

import torch


def triton_available() -> bool:
    try:
        import triton  # noqa: F401
        import triton.language  # noqa: F401
    except ImportError:
        return False
    return torch.cuda.is_available()


def rmsnorm_reference(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    normalized = x.float() * torch.rsqrt(variance + eps)
    return (normalized * weight.float()).to(x.dtype)


def rmsnorm_triton(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    if not triton_available():
        raise RuntimeError("Triton RMSNorm requires Triton and an NVIDIA GPU")

    import triton
    import triton.language as tl

    @triton.jit
    def _kernel(
        x_ptr,
        w_ptr,
        out_ptr,
        stride,
        n_cols: tl.constexpr,
        eps: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < n_cols
        x_row = x_ptr + row * stride
        values = tl.load(x_row + offsets, mask=mask, other=0.0).to(tl.float32)
        variance = tl.sum(values * values, axis=0) / n_cols
        inv_rms = tl.rsqrt(variance + eps)
        weights = tl.load(w_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + row * stride + offsets, values * inv_rms * weights, mask=mask)

    if x.ndim != 2:
        raise ValueError("x must be [rows, hidden_size]")
    if weight.ndim != 1 or weight.shape[0] != x.shape[1]:
        raise ValueError("weight must match x.shape[-1]")
    if not x.is_cuda or not weight.is_cuda:
        raise ValueError("x and weight must be CUDA tensors")

    rows, hidden = x.shape
    output = torch.empty_like(x)
    block = triton.next_power_of_2(hidden)
    _kernel[(rows,)](
        x,
        weight,
        output,
        x.stride(0),
        n_cols=hidden,
        eps=eps,
        BLOCK=block,
    )
    return output
