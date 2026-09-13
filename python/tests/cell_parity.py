"""Numerical comparison for the small, two-layer FP32 CPU cell fixtures."""
from __future__ import annotations

import torch


def assert_fp32_cell_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Bound every error relative to its token's activation scale.

    Dense GEMM and sharded GEMM plus all-reduce have different FP32 reduction
    orders. The unscaled random weights in these fixtures amplify roundoff:
    a small coordinate after cancellation is not a useful relative-error
    denominator. Linux measurements against a separate FP64 reference found
    roundoff in both paths, including larger errors in the dense reference.

    Keep a fixed 32*eps budget for these two 16-wide layers. Compare the worst
    coordinate of each token, not a batch-wide norm that can hide corruption.
    This is a test-fixture budget, not a GPU or physical-inference guarantee.
    Cache sizes, alias freedom, fork/promotion and wire bytes remain separate
    exact assertions in the calling tests.
    """
    if actual.shape != expected.shape:
        raise AssertionError(f"cell shape mismatch: {actual.shape} != {expected.shape}")
    if actual.dtype != torch.float32 or expected.dtype != torch.float32:
        raise AssertionError("cell parity requires FP32 tensors")
    if actual.device != expected.device or actual.device.type != "cpu":
        raise AssertionError("cell parity requires tensors on the same CPU device")
    if actual.ndim != 3 or actual.numel() == 0:
        raise AssertionError("cell parity requires non-empty [batch, tokens, hidden] tensors")
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise AssertionError("cell parity requires finite tensors")
    reference = expected.double()
    scale = reference.abs().amax(dim=-1, keepdim=True).clamp_min(1.0)
    torch.testing.assert_close(
        actual.double() / scale,
        reference / scale,
        rtol=0,
        atol=32 * torch.finfo(torch.float32).eps,
    )
