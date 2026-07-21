from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence

from safetensors.torch import load_file
import torch
import torch.distributed as distributed


_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
_CUDA_DEVICE = re.compile(r"cuda:(0|[1-9][0-9]*)\Z")
MAX_CELL_ACTIVATION_BATCH_SIZE = 8


@dataclass(frozen=True)
class CellExecutorBackend:
    """Physical execution ABI shared by local and external TP-cell ranks.

    PyTorch deliberately exposes AMD ROCm collectives through the same
    ``cuda`` device and ``nccl`` backend names used by CUDA.  Consequently a
    single contract covers NVIDIA/CUDA and AMD/ROCm builds while the manifest
    still advertises the concrete member capabilities used for placement.

    Validation is split in two: construction validates the portable contract;
    ``activate`` probes the current host and selects its concrete accelerator.
    This lets a coordinator compile a GPU launch plan on a CPU-only machine,
    while every rank still fails closed before loading weights or announcing
    READY if its runtime cannot execute the plan.
    """

    collective_backend: str = "gloo"
    device: str = "cpu"
    compute_dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.compute_dtype not in _DTYPES:
            raise ValueError("cell compute dtype must be float32, float16 or bfloat16")
        if self.collective_backend == "gloo":
            if self.device != "cpu" or self.compute_dtype != "float32":
                raise ValueError("the reference Gloo backend requires cpu/float32")
            return
        if self.collective_backend == "nccl":
            if _CUDA_DEVICE.fullmatch(self.device) is None:
                raise ValueError("NCCL cell devices must use an explicit cuda:<index>")
            return
        raise ValueError("cell collective backend must be gloo or nccl")

    @property
    def torch_dtype(self) -> torch.dtype:
        return _DTYPES[self.compute_dtype]

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.device)

    def activate(self) -> torch.device:
        """Validate host support and select the rank-local accelerator."""

        device = self.torch_device
        if self.collective_backend == "gloo":
            if not distributed.is_gloo_available():
                raise RuntimeError("this PyTorch build has no Gloo backend")
            return device
        if not torch.cuda.is_available():
            raise RuntimeError("the planned NCCL cell rank has no CUDA/ROCm device")
        if not distributed.is_nccl_available():
            raise RuntimeError("this PyTorch build has no NCCL/RCCL backend")
        index = device.index
        if index is None or index >= torch.cuda.device_count():
            raise RuntimeError(f"planned cell device {self.device} is not available")
        torch.cuda.set_device(device)
        return device

    def validate_fixture_dtype(self, manifest_dtype: object) -> None:
        if manifest_dtype != self.compute_dtype:
            raise ValueError(
                f"fixture dtype {manifest_dtype!r} does not match planned "
                f"cell compute dtype {self.compute_dtype!r}"
            )

    def load_weights(self, shard_file: str) -> dict[str, torch.Tensor]:
        """Load only one rank shard, then move it to its planned device.

        SafeTensors is opened on CPU for portability and integrity checks.  No
        other rank or full checkpoint is materialized.  GPU placement happens
        one rank file at a time and preserves the fixture dtype exactly.
        """

        self.activate()
        source = load_file(shard_file, device="cpu")
        result = {
            name: tensor.to(
                device=self.torch_device,
                dtype=self.torch_dtype,
                non_blocking=False,
            ).contiguous()
            for name, tensor in source.items()
        }
        if any(
            tensor.device != self.torch_device or tensor.dtype != self.torch_dtype
            for tensor in result.values()
        ):
            raise RuntimeError("cell rank weights did not reach the planned device/dtype")
        return result

    def broadcast_activation(
        self,
        tensor: torch.Tensor | None,
        shape_value: object,
        *,
        source_rank: int = 0,
    ) -> torch.Tensor:
        """Move one bounded ingress batch to rank zero and broadcast collectively.

        Only rank zero receives the activation batch from the GDLP/control plane.
        Other members allocate the sealed shape locally.  This removes the
        previous N-way TCP replication and also gives NCCL/RCCL a device-local
        tensor for the collective.  Batch size one remains the exact legacy
        path; larger first dimensions represent independent requests with an
        equal token count and cache length, validated by the stage runner.
        """

        shape = _activation_shape(shape_value)
        if tensor is None:
            value = torch.empty(shape, device=self.torch_device, dtype=self.torch_dtype)
        else:
            if tuple(tensor.shape) != shape or not tensor.is_floating_point():
                raise ValueError("rank-zero activation does not match the collective shape")
            value = tensor.detach().to(
                device=self.torch_device,
                dtype=self.torch_dtype,
                non_blocking=False,
            ).contiguous()
        distributed.broadcast(value, src=source_rank)
        return value

    def memory_report(self) -> dict[str, int | str]:
        if self.torch_device.type != "cuda":
            return {
                "device": self.device,
                "computeDtype": self.compute_dtype,
                "collectiveBackend": self.collective_backend,
                "allocatedBytes": 0,
                "reservedBytes": 0,
                "peakAllocatedBytes": 0,
            }
        torch.cuda.synchronize(self.torch_device)
        return {
            "device": self.device,
            "computeDtype": self.compute_dtype,
            "collectiveBackend": self.collective_backend,
            "allocatedBytes": int(torch.cuda.memory_allocated(self.torch_device)),
            "reservedBytes": int(torch.cuda.memory_reserved(self.torch_device)),
            "peakAllocatedBytes": int(torch.cuda.max_memory_allocated(self.torch_device)),
        }


def rank_backend(
    collective_backend: str,
    rank_devices: Sequence[str],
    compute_dtype: str,
    rank: int,
) -> CellExecutorBackend:
    if isinstance(rank_devices, (str, bytes, bytearray)):
        raise TypeError("cell rank_devices must be an ordered sequence")
    if not 0 <= rank < len(rank_devices):
        raise ValueError("cell rank has no declared device")
    return CellExecutorBackend(
        collective_backend=collective_backend,
        device=rank_devices[rank],
        compute_dtype=compute_dtype,
    )


def _activation_shape(value: object) -> tuple[int, int, int]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or len(value) != 3
        or any(
            not isinstance(item, int) or isinstance(item, bool) or item < 1
            for item in value
        )
    ):
        raise ValueError("cell activation shape must contain three positive integers")
    shape = tuple(int(item) for item in value)
    if shape[0] > MAX_CELL_ACTIVATION_BATCH_SIZE:
        raise ValueError(
            "cell activation batch exceeds the bounded physical batch size"
        )
    return shape


__all__ = [
    "CellExecutorBackend",
    "MAX_CELL_ACTIVATION_BATCH_SIZE",
    "rank_backend",
]
