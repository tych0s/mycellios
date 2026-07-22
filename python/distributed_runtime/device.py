from __future__ import annotations

from dataclasses import dataclass
import platform
from typing import Any

import torch


SUPPORTED_TORCH_DEVICE_REQUESTS = ("auto", "cpu", "cuda", "mps", "xpu")


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
        elif self.device.type == "xpu":
            try:
                allocated = int(torch.xpu.memory_allocated(self.device))
                reserved = int(torch.xpu.memory_reserved(self.device))
                peak_allocated = int(torch.xpu.max_memory_allocated(self.device))
            except (AttributeError, AssertionError, RuntimeError):
                allocated = reserved = peak_allocated = None
        elif self.device.type == "mps":
            try:
                allocated = int(torch.mps.current_allocated_memory())
                reserved = int(torch.mps.driver_allocated_memory())
            except (AttributeError, AssertionError, RuntimeError):
                allocated = reserved = None

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
    """Resolve CPU or a physically usable CUDA/ROCm, XPU or MPS device.

    ``auto`` prefers an accelerator only when the installed Torch build reports
    it usable. An explicit ``cuda`` request fails closed rather than silently
    running on CPU, which keeps product telemetry honest.
    """

    normalized = normalize_torch_device_request(requested)
    if normalized == "auto":
        if _cuda_is_usable():
            normalized = "cuda:0"
        elif _xpu_is_usable():
            normalized = "xpu:0"
        elif _mps_is_usable():
            normalized = "mps"
        else:
            normalized = "cpu"
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

    device = torch.device(normalized)
    if device.type == "cuda":
        if not _cuda_is_usable():
            raise RuntimeError(
                f"Torch device {normalized!r} was requested, but the installed "
                "PyTorch build cannot use a CUDA/ROCm accelerator"
            )
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

    if device.type == "xpu":
        if not _xpu_is_usable():
            raise RuntimeError(
                f"Torch device {normalized!r} was requested, but the installed "
                "PyTorch build cannot use an Intel XPU accelerator"
            )
        index = 0 if device.index is None else device.index
        count = int(torch.xpu.device_count())
        if not 0 <= index < count:
            raise RuntimeError(
                f"Torch device {normalized!r} does not exist; {count} XPU device(s) are available"
            )
        properties = torch.xpu.get_device_properties(index)
        return TorchExecutionDevice(
            requested=requested.strip().lower(),
            device=torch.device(f"xpu:{index}"),
            kind="gpu",
            backend="xpu",
            name=str(properties.name),
            accelerated=True,
            total_memory_bytes=int(properties.total_memory),
        )

    if device.type == "mps":
        if not _mps_is_usable():
            raise RuntimeError(
                "Torch device 'mps' was requested, but this PyTorch/macOS combination "
                "cannot use Metal Performance Shaders"
            )
        return TorchExecutionDevice(
            requested=requested.strip().lower(),
            device=torch.device("mps"),
            kind="gpu",
            backend="mps",
            name=_mps_device_name(),
            accelerated=True,
            total_memory_bytes=_mps_total_memory(),
        )

    raise RuntimeError(f"Torch device type {device.type!r} has no certified dense-stage backend")


def describe_torch_execution_device(device: torch.device | str) -> TorchExecutionDevice:
    """Describe an execution device already selected by a specialised runner."""

    parsed = torch.device(device)
    if parsed.type == "cpu":
        return resolve_torch_execution_device("cpu")
    if parsed.type not in {"cuda", "mps", "xpu"}:
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
    if normalized.startswith("xpu:"):
        index_text = normalized.removeprefix("xpu:")
        if index_text.isdigit():
            return f"xpu:{int(index_text)}"
    raise ValueError("device must be auto, cpu, cuda[:index], mps or xpu[:index]")


def _cuda_is_usable() -> bool:
    try:
        return bool(torch.cuda.is_available()) and int(torch.cuda.device_count()) > 0
    except (AssertionError, RuntimeError):
        return False


def _xpu_is_usable() -> bool:
    try:
        xpu = getattr(torch, "xpu", None)
        return xpu is not None and bool(xpu.is_available()) and int(xpu.device_count()) > 0
    except (AttributeError, AssertionError, RuntimeError):
        return False


def _mps_is_usable() -> bool:
    try:
        backend = getattr(torch.backends, "mps", None)
        return backend is not None and bool(backend.is_built()) and bool(backend.is_available())
    except (AttributeError, AssertionError, RuntimeError):
        return False


def _mps_device_name() -> str:
    backend = getattr(torch.backends, "mps", None)
    getter = getattr(backend, "get_name", None)
    if callable(getter):
        name = str(getter()).strip()
        if name:
            return name
    return "Apple GPU"


def _mps_total_memory() -> int | None:
    try:
        value = int(torch.mps.recommended_max_memory())
        return value if value > 0 else None
    except (AttributeError, AssertionError, RuntimeError):
        return None
