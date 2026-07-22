from __future__ import annotations

from dataclasses import dataclass
import platform
from typing import Any

import torch


SUPPORTED_TORCH_DEVICE_REQUESTS = ("auto", "cpu", "cuda")


@dataclass(frozen=True)
class TorchExecutionDevice:
    """One truthful, already-validated Torch execution target.

    PyTorch exposes ROCm accelerators through the ``cuda`` device namespace.
    ``backend`` therefore records the effective implementation separately from
    the Torch device string.  DirectML/Vulkan are deliberately absent: this
    runtime does not currently execute Hugging Face stages through either API.
    """

    requested: str
    device: torch.device
    kind: str
    backend: str
    name: str
    accelerated: bool
    total_memory_bytes: int | None

    def snapshot(
        self, *, weight_bytes: int, precision: str
    ) -> dict[str, Any]:
        if not isinstance(weight_bytes, int) or isinstance(weight_bytes, bool):
            raise TypeError("weight_bytes must be an integer")
        if weight_bytes < 0:
            raise ValueError("weight_bytes cannot be negative")
        if not isinstance(precision, str) or not precision.strip():
            raise ValueError("precision cannot be empty")

        allocated: int | None = None
        reserved: int | None = None
        peak_allocated: int | None = None
        if self.device.type == "cuda":
            try:
                allocated = int(torch.cuda.memory_allocated(self.device))
                reserved = int(torch.cuda.memory_reserved(self.device))
                peak_allocated = int(torch.cuda.max_memory_allocated(self.device))
            except (AssertionError, RuntimeError):
                # A lost accelerator is reflected by the surrounding runtime
                # health state. Metrics remain available without inventing
                # allocator values after the CUDA/ROCm context is gone.
                allocated = reserved = peak_allocated = None

        return {
            "requested_device": self.requested,
            "device": str(self.device),
            "device_kind": self.kind,
            "backend": self.backend,
            "device_name": self.name,
            "accelerated": self.accelerated,
            "precision": precision,
            "weight_bytes": weight_bytes,
            "total_memory_bytes": self.total_memory_bytes,
            "allocated_bytes": allocated,
            "reserved_bytes": reserved,
            "peak_allocated_bytes": peak_allocated,
        }


def resolve_torch_execution_device(requested: str = "auto") -> TorchExecutionDevice:
    """Resolve CPU or a real CUDA/ROCm device without optimistic fallbacks.

    ``auto`` prefers an accelerator only when the installed Torch build reports
    it usable. An explicit ``cuda`` request fails closed rather than silently
    running on CPU, which keeps product telemetry honest.
    """

    normalized = normalize_torch_device_request(requested)
    if normalized == "auto":
        normalized = "cuda:0" if _cuda_is_usable() else "cpu"
    if normalized == "cpu":
        return TorchExecutionDevice(
            requested=requested.strip().lower(),
            device=torch.device("cpu"),
            kind="cpu",
            backend="cpu",
            name=platform.processor().strip() or platform.machine() or "CPU",
            accelerated=False,
            total_memory_bytes=None,
        )

    if not _cuda_is_usable():
        raise RuntimeError(
            f"Torch device {normalized!r} was requested, but the installed "
            "PyTorch build cannot use a CUDA/ROCm accelerator"
        )
    device = torch.device(normalized)
    index = 0 if device.index is None else device.index
    count = int(torch.cuda.device_count())
    if not 0 <= index < count:
        raise RuntimeError(
            f"Torch device {normalized!r} does not exist; {count} CUDA/ROCm "
            "device(s) are available"
        )
    properties = torch.cuda.get_device_properties(index)
    hip_version = getattr(torch.version, "hip", None)
    backend = "rocm" if isinstance(hip_version, str) and hip_version else "cuda"
    return TorchExecutionDevice(
        requested=requested.strip().lower(),
        device=torch.device(f"cuda:{index}"),
        kind="gpu",
        backend=backend,
        name=str(properties.name),
        accelerated=True,
        total_memory_bytes=int(properties.total_memory),
    )


def describe_torch_execution_device(device: torch.device | str) -> TorchExecutionDevice:
    """Describe an execution device already selected by a specialised runner."""

    parsed = torch.device(device)
    if parsed.type == "cpu":
        return resolve_torch_execution_device("cpu")
    if parsed.type != "cuda":
        raise RuntimeError(
            f"Torch device type {parsed.type!r} has no certified dense-stage backend"
        )
    requested = str(parsed)
    return resolve_torch_execution_device(requested)


def normalize_torch_device_request(requested: str) -> str:
    if not isinstance(requested, str) or not requested.strip():
        raise ValueError("device request cannot be empty")
    normalized = requested.strip().lower()
    if normalized in SUPPORTED_TORCH_DEVICE_REQUESTS:
        return normalized
    if normalized.startswith("cuda:"):
        index_text = normalized.removeprefix("cuda:")
        if index_text.isdigit():
            return f"cuda:{int(index_text)}"
    raise ValueError("device must be auto, cpu, cuda or cuda:<index>")


def _cuda_is_usable() -> bool:
    try:
        return bool(torch.cuda.is_available()) and int(torch.cuda.device_count()) > 0
    except (AssertionError, RuntimeError):
        return False
