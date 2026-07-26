"""Native bounded RAM/accelerator residency for dense Mycellios stages.

The checkpoint loader remains the authoritative storage -> RAM tier.  This
module owns the second boundary: it either proves that the complete local stage
fits the currently available accelerator budget, or keeps the authoritative
weights in host RAM and moves a deterministic, bounded layer working set.

No allocator estimate is presented as an observation.  Weight bytes are exact
from live tensors, transfer durations are measured around the actual ``to``
operation, and available device/host bytes come from the operating system or
the active Torch backend.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass
import ctypes
import os
import threading
import time
from typing import Any, Callable, Iterable

import torch
from torch import nn

from .device import TorchExecutionDevice


_AUTO_ACCELERATOR_RESERVE_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class DenseTieringConfig:
    """Sealed memory limits for one dense stage.

    Zero host/device budgets select a value measured at startup.  An omitted
    activation reserve keeps a conservative automatic reserve; an explicit
    zero is valid when a planner has already reserved activation/KV memory
    separately.
    """

    host_ram_budget_bytes: int = 0
    vram_budget_bytes: int = 0
    activation_reserve_bytes: int | None = None
    admit_next_layer: bool = False
    allow_bounded_layer_cache: bool = True

    def __post_init__(self) -> None:
        for name, value in (
            ("host_ram_budget_bytes", self.host_ram_budget_bytes),
            ("vram_budget_bytes", self.vram_budget_bytes),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.activation_reserve_bytes is not None and (
            not isinstance(self.activation_reserve_bytes, int)
            or isinstance(self.activation_reserve_bytes, bool)
            or self.activation_reserve_bytes < 0
        ):
            raise ValueError(
                "activation_reserve_bytes must be null or a non-negative integer"
            )
        if not isinstance(self.admit_next_layer, bool):
            raise TypeError("admit_next_layer must be boolean")
        if not isinstance(self.allow_bounded_layer_cache, bool):
            raise TypeError("allow_bounded_layer_cache must be boolean")


@dataclass(frozen=True, slots=True)
class DenseMemoryUnit:
    key: str
    module: nn.Module
    bytes: int

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("dense memory unit key cannot be empty")
        if self.bytes < 1:
            raise ValueError("dense memory unit bytes must be positive")


@dataclass(frozen=True, slots=True)
class _HostModuleState:
    parameters: tuple[tuple[nn.Module, str, nn.Parameter], ...]
    buffers: tuple[tuple[nn.Module, str, torch.Tensor], ...]


class DenseTieringError(RuntimeError):
    """A dense stage cannot honor its measured memory contract."""


def add_dense_tiering_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dense-host-ram-budget-bytes",
        type=int,
        default=0,
        help=(
            "Sealed host-RAM ceiling for dense weights; zero uses OS-reported "
            "available memory at startup."
        ),
    )
    parser.add_argument(
        "--dense-vram-budget-bytes",
        type=int,
        default=0,
        help=(
            "Sealed accelerator-memory ceiling for dense weights and activation "
            "reserve; zero uses backend-reported free memory at startup."
        ),
    )
    parser.add_argument(
        "--dense-activation-reserve-bytes",
        type=int,
        help=(
            "Bytes kept outside the dense weight cache for activations and KV; "
            "omission selects the native conservative reserve."
        ),
    )
    parser.add_argument(
        "--dense-enable-next-layer-admission",
        action="store_true",
        help=(
            "Enable synchronous admission of the statically known next layer. "
            "Disabled until a physical A/B proves a gain; it does not overlap "
            "copy and compute."
        ),
    )
    parser.add_argument(
        "--dense-require-full-residency",
        action="store_true",
        help=(
            "Fail before allocation unless all local dense weights fit the "
            "sealed accelerator budget."
        ),
    )


def dense_tiering_config_from_args(args: argparse.Namespace) -> DenseTieringConfig:
    return DenseTieringConfig(
        host_ram_budget_bytes=args.dense_host_ram_budget_bytes,
        vram_budget_bytes=args.dense_vram_budget_bytes,
        activation_reserve_bytes=args.dense_activation_reserve_bytes,
        admit_next_layer=args.dense_enable_next_layer_admission,
        allow_bounded_layer_cache=not args.dense_require_full_residency,
    )


class BoundedLayerResidency:
    """Deterministic byte-bounded LRU for a sequential dense layer range.

    ``move`` performs the real RAM <-> accelerator operation.  Supplying it as
    a seam keeps the accounting policy independently testable; production
    always uses ``nn.Module.to`` through :func:`configure_dense_tiering`.
    """

    def __init__(
        self,
        units: Iterable[DenseMemoryUnit],
        *,
        capacity_bytes: int,
        move: Callable[[DenseMemoryUnit, bool], int | None],
        clock_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        ordered = tuple(units)
        if not ordered:
            raise ValueError("bounded dense residency requires at least one layer")
        if (
            not isinstance(capacity_bytes, int)
            or isinstance(capacity_bytes, bool)
            or capacity_bytes < 1
        ):
            raise ValueError("dense residency capacity must be positive")
        keys = [unit.key for unit in ordered]
        if len(set(keys)) != len(keys):
            raise ValueError("dense memory unit keys must be unique")
        largest = max(unit.bytes for unit in ordered)
        if largest > capacity_bytes:
            raise DenseTieringError(
                "dense_tiering_largest_layer_exceeds_vram_cache:"
                f"required={largest}:capacity={capacity_bytes}"
            )
        self.units = ordered
        self.capacity_bytes = capacity_bytes
        self._by_key = {unit.key: unit for unit in ordered}
        self._move = move
        self._clock_ns = clock_ns
        self._resident: OrderedDict[str, None] = OrderedDict()
        self._next_admitted: set[str] = set()
        self._resident_bytes = 0
        self._peak_resident_bytes = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._next_layer_admissions = 0
        self._next_layer_admission_hits = 0
        self._startup_warmups = 0
        self._evictions = 0
        self._ram_to_vram_bytes = 0
        self._vram_to_ram_bytes = 0
        self._vram_freed_bytes = 0
        self._ram_to_vram_nanoseconds = 0
        self._vram_to_ram_nanoseconds = 0
        self._eviction_release_nanoseconds = 0
        self._lock = threading.RLock()

    def acquire(self, key: str) -> None:
        """Make ``key`` resident, charging a hit only to demanded execution."""

        with self._lock:
            if key in self._resident:
                self._cache_hits += 1
                if key in self._next_admitted:
                    self._next_layer_admission_hits += 1
                    self._next_admitted.discard(key)
                self._resident.move_to_end(key)
                return
            self._cache_misses += 1
            self._ensure_resident(key, next_admitted=False)

    def admit_next(self, key: str) -> None:
        """Synchronously admit the statically known next layer.

        This deliberately makes no overlap claim: the copy completes after the
        current layer and before the next pre-hook.
        """

        with self._lock:
            self._next_layer_admissions += 1
            if key in self._resident:
                self._resident.move_to_end(key)
                self._next_admitted.add(key)
                return
            self._ensure_resident(key, next_admitted=True)

    def warmup(self, key: str) -> None:
        """Load one first-demand unit during startup, outside user latency."""

        with self._lock:
            self._startup_warmups += 1
            if key in self._resident:
                self._resident.move_to_end(key)
                return
            self._ensure_resident(key, next_admitted=False)

    def close(self) -> None:
        """Return all cached weights to their authoritative host tier."""

        with self._lock:
            for key in list(self._resident):
                self._evict(key)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "cacheCapacityBytes": self.capacity_bytes,
                "cacheResidentBytes": self._resident_bytes,
                "cachePeakResidentBytes": self._peak_resident_bytes,
                "cacheResidentUnits": list(self._resident),
                "cacheHits": self._cache_hits,
                "cacheMisses": self._cache_misses,
                "nextLayerAdmissions": self._next_layer_admissions,
                "nextLayerAdmissionHits": self._next_layer_admission_hits,
                "startupWarmups": self._startup_warmups,
                "evictions": self._evictions,
                "ramToVramBytes": self._ram_to_vram_bytes,
                "vramToRamBytes": self._vram_to_ram_bytes,
                # Tensor references released from the live working set. Torch
                # may retain the blocks in its allocator reserve.
                "vramTensorBytesReleased": self._vram_freed_bytes,
                "ramToVramNanoseconds": self._ram_to_vram_nanoseconds,
                "vramToRamNanoseconds": self._vram_to_ram_nanoseconds,
                "evictionReleaseNanoseconds": self._eviction_release_nanoseconds,
            }

    def _ensure_resident(self, key: str, *, next_admitted: bool) -> None:
        unit = self._by_key.get(key)
        if unit is None:
            raise KeyError(f"unknown dense memory unit {key!r}")
        while self._resident_bytes + unit.bytes > self.capacity_bytes:
            if not self._resident:
                raise DenseTieringError("dense_tiering_cache_accounting_underflow")
            self._evict(next(iter(self._resident)))
        started = self._clock_ns()
        copied = self._move(unit, True)
        elapsed = max(0, self._clock_ns() - started)
        copied_bytes = unit.bytes if copied is None else copied
        if (
            not isinstance(copied_bytes, int)
            or isinstance(copied_bytes, bool)
            or not 0 <= copied_bytes <= unit.bytes
        ):
            raise ValueError("dense residency mover returned invalid copied bytes")
        self._ram_to_vram_bytes += copied_bytes
        self._ram_to_vram_nanoseconds += elapsed
        self._resident[key] = None
        self._resident_bytes += unit.bytes
        self._peak_resident_bytes = max(
            self._peak_resident_bytes, self._resident_bytes
        )
        if next_admitted:
            self._next_admitted.add(key)

    def _evict(self, key: str) -> None:
        unit = self._by_key[key]
        started = self._clock_ns()
        copied = self._move(unit, False)
        elapsed = max(0, self._clock_ns() - started)
        copied_bytes = unit.bytes if copied is None else copied
        if (
            not isinstance(copied_bytes, int)
            or isinstance(copied_bytes, bool)
            or not 0 <= copied_bytes <= unit.bytes
        ):
            raise ValueError("dense residency mover returned invalid copied bytes")
        self._vram_to_ram_bytes += copied_bytes
        if copied_bytes > 0:
            self._vram_to_ram_nanoseconds += elapsed
        self._vram_freed_bytes += unit.bytes
        self._eviction_release_nanoseconds += elapsed
        self._resident.pop(key)
        self._next_admitted.discard(key)
        self._resident_bytes -= unit.bytes
        self._evictions += 1


class DenseTieringRuntime:
    """Live residency state and forward hooks for one StageRunner."""

    def __init__(
        self,
        *,
        mode: str,
        execution_device: TorchExecutionDevice,
        compute_dtype: torch.dtype,
        total_weight_bytes: int,
        host_budget_bytes: int | None,
        host_budget_source: str,
        vram_budget_bytes: int | None,
        vram_budget_source: str,
        activation_reserve_bytes: int,
        static_vram_bytes: int,
        storage_evidence: dict[str, Any],
        startup_ram_to_vram_bytes: int = 0,
        startup_ram_to_vram_nanoseconds: int = 0,
        cache: BoundedLayerResidency | None = None,
        hooks: tuple[Any, ...] = (),
    ) -> None:
        self.mode = mode
        self.execution_device = execution_device
        self.compute_dtype = compute_dtype
        self.total_weight_bytes = total_weight_bytes
        self.host_budget_bytes = host_budget_bytes
        self.host_budget_source = host_budget_source
        self.vram_budget_bytes = vram_budget_bytes
        self.vram_budget_source = vram_budget_source
        self.activation_reserve_bytes = activation_reserve_bytes
        self.static_vram_bytes = static_vram_bytes
        self.startup_ram_to_vram_bytes = startup_ram_to_vram_bytes
        self.startup_ram_to_vram_nanoseconds = startup_ram_to_vram_nanoseconds
        self.cache = cache
        self._hooks = hooks
        self.storage_evidence = dict(storage_evidence)

    def close(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks = ()
        if self.cache is not None:
            self.cache.close()

    def snapshot(self) -> dict[str, Any]:
        cache = {} if self.cache is None else self.cache.snapshot()
        cache_resident = int(cache.get("cacheResidentBytes", 0))
        layer_transfer_bytes = int(cache.get("ramToVramBytes", 0))
        layer_transfer_nanoseconds = int(
            cache.get("ramToVramNanoseconds", 0)
        )
        if self.mode == "cpu-resident":
            host_resident = self.total_weight_bytes
        elif self.mode == "bounded-layer-cache":
            host_resident = max(
                0,
                self.total_weight_bytes - self.static_vram_bytes,
            )
        else:
            host_resident = 0
        return {
            "schema": "mycellios-dense-tiering/1",
            "mode": self.mode,
            "hostBudgetBytes": self.host_budget_bytes,
            "hostBudgetSource": self.host_budget_source,
            "vramBudgetBytes": self.vram_budget_bytes,
            "vramBudgetSource": self.vram_budget_source,
            "activationReserveBytes": self.activation_reserve_bytes,
            "totalWeightBytes": self.total_weight_bytes,
            "hostResidentWeightBytes": host_resident,
            "staticVramWeightBytes": self.static_vram_bytes,
            "currentVramWeightBytes": self.static_vram_bytes + cache_resident,
            "peakVramWeightBytes": self.static_vram_bytes
            + int(cache.get("cachePeakResidentBytes", 0)),
            "storageToRam": dict(self.storage_evidence),
            **cache,
            "startupRamToVramBytes": self.startup_ram_to_vram_bytes,
            "startupRamToVramNanoseconds":
                self.startup_ram_to_vram_nanoseconds,
            "layerRamToVramBytes": layer_transfer_bytes,
            "layerRamToVramNanoseconds": layer_transfer_nanoseconds,
            "ramToVramBytes":
                self.startup_ram_to_vram_bytes + layer_transfer_bytes,
            "ramToVramNanoseconds":
                self.startup_ram_to_vram_nanoseconds
                + layer_transfer_nanoseconds,
        }


def configure_dense_tiering(
    model: nn.Module,
    *,
    execution_device: TorchExecutionDevice,
    compute_dtype: torch.dtype,
    config: DenseTieringConfig,
    storage_evidence: dict[str, Any] | None = None,
    permit_bounded_tiering: bool = True,
) -> DenseTieringRuntime:
    """Place a dense stage without ever attempting an unbudgeted full GPU copy."""

    evidence = dict(storage_evidence or {})
    source_weight_bytes = unique_tensor_bytes(model)
    host_available = available_host_memory_bytes()
    host_budget, host_source = _effective_budget(
        requested=config.host_ram_budget_bytes,
        measured=(
            None
            if host_available is None
            else host_available + source_weight_bytes
        ),
        requested_source="sealed",
        measured_source="os-available-plus-loaded-stage",
    )
    if host_budget is not None and source_weight_bytes > host_budget:
        raise DenseTieringError(
            "dense_tiering_stage_exceeds_host_ram_budget:"
            f"required={source_weight_bytes}:budget={host_budget}"
        )

    if not execution_device.accelerated or execution_device.device.type == "cpu":
        if any(
            tensor.is_floating_point() and tensor.dtype != compute_dtype
            for tensor in _module_tensors(model)
        ):
            model.to(device=torch.device("cpu"), dtype=compute_dtype)
        cpu_weight_bytes = unique_tensor_bytes(model)
        return DenseTieringRuntime(
            mode="cpu-resident",
            execution_device=execution_device,
            compute_dtype=compute_dtype,
            total_weight_bytes=cpu_weight_bytes,
            host_budget_bytes=host_budget,
            host_budget_source=host_source,
            vram_budget_bytes=None,
            vram_budget_source="not-applicable",
            activation_reserve_bytes=0,
            static_vram_bytes=0,
            storage_evidence=evidence,
        )

    free_device_bytes, free_source = accelerator_free_memory_bytes(execution_device)
    vram_budget, vram_source = _effective_budget(
        requested=config.vram_budget_bytes,
        measured=free_device_bytes,
        requested_source="sealed",
        measured_source=free_source,
        require_measured=True,
    )
    if vram_budget is None:
        raise DenseTieringError(
            "dense_tiering_cannot_measure_accelerator_free_memory"
        )
    activation_reserve = (
        config.activation_reserve_bytes
        if config.activation_reserve_bytes is not None
        else min(
            _AUTO_ACCELERATOR_RESERVE_BYTES,
            max(0, vram_budget // 8),
        )
    )
    usable_weight_budget = vram_budget - activation_reserve
    if usable_weight_budget < 1:
        raise DenseTieringError(
            "dense_tiering_activation_reserve_exhausts_vram_budget:"
            f"budget={vram_budget}:reserve={activation_reserve}"
        )

    target_weight_bytes = unique_tensor_bytes(model, dtype=compute_dtype)
    if target_weight_bytes <= usable_weight_budget:
        started = time.perf_counter_ns()
        model.to(device=execution_device.device, dtype=compute_dtype)
        elapsed = max(0, time.perf_counter_ns() - started)
        return DenseTieringRuntime(
            mode="full-resident",
            execution_device=execution_device,
            compute_dtype=compute_dtype,
            total_weight_bytes=target_weight_bytes,
            host_budget_bytes=host_budget,
            host_budget_source=host_source,
            vram_budget_bytes=vram_budget,
            vram_budget_source=vram_source,
            activation_reserve_bytes=activation_reserve,
            static_vram_bytes=target_weight_bytes,
            storage_evidence=evidence,
            startup_ram_to_vram_bytes=target_weight_bytes,
            startup_ram_to_vram_nanoseconds=elapsed,
        )
    if not permit_bounded_tiering or not config.allow_bounded_layer_cache:
        raise DenseTieringError(
            "dense_tiering_full_stage_exceeds_vram_and_execution_mode_forbids_tiering:"
            f"required={target_weight_bytes}:budget={usable_weight_budget}"
        )

    layers = tuple(model.model.layers)
    if not layers:
        raise DenseTieringError("dense_tiering_stage_has_no_layers")
    layer_ids = {
        id(tensor)
        for layer in layers
        for tensor in _module_tensors(layer)
    }
    static_bytes = unique_tensor_bytes(
        model,
        dtype=compute_dtype,
        excluded_tensor_ids=layer_ids,
    )
    layer_units = tuple(
        DenseMemoryUnit(
            key=f"layer-{index}",
            module=layer,
            bytes=unique_tensor_bytes(layer, dtype=compute_dtype),
        )
        for index, layer in enumerate(layers)
    )
    largest_layer_bytes = max(unit.bytes for unit in layer_units)
    if static_bytes + largest_layer_bytes > usable_weight_budget:
        raise DenseTieringError(
            "dense_tiering_fixed_state_and_largest_layer_exceed_vram_budget:"
            f"fixed={static_bytes}:largestLayer={largest_layer_bytes}:"
            f"budget={usable_weight_budget}"
        )

    # Convert host weights once to their accelerator compute representation.
    # Module._apply visits parameters sequentially, so the explicit upper bound
    # is the loaded source stage plus the largest destination layer.
    needs_host_dtype_conversion = any(
        tensor.is_floating_point() and tensor.dtype != compute_dtype
        for tensor in _module_tensors(model)
    )
    host_conversion_peak = source_weight_bytes + (
        largest_layer_bytes if needs_host_dtype_conversion else 0
    )
    if host_budget is not None and host_conversion_peak > host_budget:
        raise DenseTieringError(
            "dense_tiering_host_conversion_peak_exceeds_budget:"
            f"required={host_conversion_peak}:budget={host_budget}"
        )
    model.to(device=torch.device("cpu"), dtype=compute_dtype)
    host_states = {
        unit.key: _capture_host_module_state(unit.module)
        for unit in layer_units
    }
    static_started = time.perf_counter_ns()
    _move_static_tensors(
        model,
        excluded_tensor_ids={
            id(tensor)
            for layer in layers
            for tensor in _module_tensors(layer)
        },
        device=execution_device.device,
        dtype=compute_dtype,
    )
    _synchronize(execution_device)
    static_elapsed = max(0, time.perf_counter_ns() - static_started)

    def move(unit: DenseMemoryUnit, to_accelerator: bool) -> int:
        host_state = host_states[unit.key]
        if to_accelerator:
            try:
                _install_accelerator_copy(
                    host_state,
                    device=execution_device.device,
                    dtype=compute_dtype,
                )
                _synchronize(execution_device)
            except BaseException:
                _restore_host_module_state(host_state)
                raise
            return unit.bytes
        # Inference weights are immutable. Synchronize the completed kernel,
        # restore the already-authoritative host objects and drop the GPU
        # references; a GPU->RAM copy would only duplicate PCIe traffic.
        _synchronize(execution_device)
        _restore_host_module_state(host_state)
        return 0

    cache = BoundedLayerResidency(
        layer_units,
        capacity_bytes=usable_weight_budget - static_bytes,
        move=move,
    )
    hooks: list[Any] = []
    for index, unit in enumerate(layer_units):
        next_key = layer_units[(index + 1) % len(layer_units)].key

        def before(
            _module: nn.Module,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            *,
            key: str = unit.key,
        ) -> tuple[tuple[Any, ...], dict[str, Any]]:
            cache.acquire(key)
            return (
                _move_nested(args, execution_device.device),
                _move_nested(kwargs, execution_device.device),
            )

        def after(
            _module: nn.Module,
            _args: tuple[Any, ...],
            _kwargs: dict[str, Any],
            output: Any,
            *,
            predicted: str = next_key,
        ) -> Any:
            if config.admit_next_layer:
                cache.admit_next(predicted)
            return output

        hooks.append(
            unit.module.register_forward_pre_hook(before, with_kwargs=True)
        )
        hooks.append(
            unit.module.register_forward_hook(after, with_kwargs=True)
        )

    # The first demand starts as a measured cache hit, not an allocator-risking
    # cold miss inside the first user request.
    cache.warmup(layer_units[0].key)
    return DenseTieringRuntime(
        mode="bounded-layer-cache",
        execution_device=execution_device,
        compute_dtype=compute_dtype,
        total_weight_bytes=target_weight_bytes,
        host_budget_bytes=host_budget,
        host_budget_source=host_source,
        vram_budget_bytes=vram_budget,
        vram_budget_source=vram_source,
        activation_reserve_bytes=activation_reserve,
        static_vram_bytes=static_bytes,
        storage_evidence=evidence,
        startup_ram_to_vram_bytes=static_bytes,
        startup_ram_to_vram_nanoseconds=static_elapsed,
        cache=cache,
        hooks=tuple(hooks),
    )


def unique_tensor_bytes(
    module: nn.Module,
    *,
    dtype: torch.dtype | None = None,
    excluded_tensor_ids: set[int] | frozenset[int] = frozenset(),
) -> int:
    """Exact unique parameter+buffer bytes, preserving tied storage aliases."""

    seen_storage: set[tuple[str, int | None, int, int]] = set()
    total = 0
    for tensor in _module_tensors(module):
        if id(tensor) in excluded_tensor_ids:
            continue
        storage = tensor.untyped_storage()
        identity = (
            tensor.device.type,
            tensor.device.index,
            int(storage.data_ptr()),
            int(storage.nbytes()),
        )
        if identity in seen_storage:
            continue
        seen_storage.add(identity)
        element_size = (
            torch.empty((), dtype=dtype).element_size()
            if dtype is not None and tensor.is_floating_point()
            else tensor.element_size()
        )
        total += tensor.numel() * element_size
    return total


def available_host_memory_bytes() -> int | None:
    """Return OS-reported currently available host memory."""

    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        try:
            ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        except (AttributeError, OSError):
            return None
        return int(status.available_physical) if ok else None
    try:
        with open("/proc/meminfo", encoding="ascii") as stream:
            for line in stream:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def accelerator_free_memory_bytes(
    execution_device: TorchExecutionDevice,
) -> tuple[int | None, str]:
    device = execution_device.device
    if device.type == "cuda":
        try:
            free, _total = torch.cuda.mem_get_info(device)
            return int(free), "torch-cuda-mem-get-info"
        except (AssertionError, RuntimeError):
            return None, "unavailable"
    if device.type == "xpu":
        try:
            free, _total = torch.xpu.mem_get_info(device)
            return int(free), "torch-xpu-mem-get-info"
        except (AttributeError, AssertionError, RuntimeError):
            return None, "unavailable"
    if device.type == "mps":
        try:
            recommended = int(torch.mps.recommended_max_memory())
            allocated = int(torch.mps.driver_allocated_memory())
            return max(0, recommended - allocated), "torch-mps-recommended-minus-driver"
        except (AttributeError, AssertionError, RuntimeError):
            return None, "unavailable"
    return None, "not-applicable"


def _effective_budget(
    *,
    requested: int,
    measured: int | None,
    requested_source: str,
    measured_source: str,
    require_measured: bool = False,
) -> tuple[int | None, str]:
    if measured is None and require_measured:
        return None, "unavailable"
    if requested > 0:
        if measured is None:
            return requested, requested_source
        return (
            min(requested, measured),
            f"{requested_source}-clamped-to-{measured_source}",
        )
    return measured, measured_source if measured is not None else "unavailable"


def _module_tensors(module: nn.Module) -> Iterable[torch.Tensor]:
    yield from module.parameters(recurse=True)
    yield from module.buffers(recurse=True)


def _capture_host_module_state(module: nn.Module) -> _HostModuleState:
    parameters: list[tuple[nn.Module, str, nn.Parameter]] = []
    buffers: list[tuple[nn.Module, str, torch.Tensor]] = []
    for child in module.modules():
        for name, parameter in child._parameters.items():
            if parameter is not None:
                if parameter.device.type != "cpu":
                    raise DenseTieringError(
                        "dense_tiering_authoritative_parameter_is_not_in_host_ram"
                    )
                parameters.append((child, name, parameter))
        for name, buffer in child._buffers.items():
            if buffer is not None:
                if buffer.device.type != "cpu":
                    raise DenseTieringError(
                        "dense_tiering_authoritative_buffer_is_not_in_host_ram"
                    )
                buffers.append((child, name, buffer))
    return _HostModuleState(tuple(parameters), tuple(buffers))


def _install_accelerator_copy(
    state: _HostModuleState,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    converted: dict[int, torch.Tensor] = {}
    for module, name, parameter in state.parameters:
        replacement = converted.get(id(parameter))
        if replacement is None:
            replacement = nn.Parameter(
                parameter.to(
                    device=device,
                    dtype=dtype if parameter.is_floating_point() else parameter.dtype,
                ),
                requires_grad=parameter.requires_grad,
            )
            converted[id(parameter)] = replacement
        module._parameters[name] = replacement
    for module, name, buffer in state.buffers:
        replacement = converted.get(id(buffer))
        if replacement is None:
            replacement = buffer.to(
                device=device,
                dtype=dtype if buffer.is_floating_point() else buffer.dtype,
            )
            converted[id(buffer)] = replacement
        module._buffers[name] = replacement


def _restore_host_module_state(state: _HostModuleState) -> None:
    for module, name, parameter in state.parameters:
        module._parameters[name] = parameter
    for module, name, buffer in state.buffers:
        module._buffers[name] = buffer


def _move_static_tensors(
    model: nn.Module,
    *,
    excluded_tensor_ids: set[int],
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    converted: dict[int, torch.Tensor] = {}
    for module in model.modules():
        for name, parameter in tuple(module._parameters.items()):
            if parameter is None or id(parameter) in excluded_tensor_ids:
                continue
            replacement = converted.get(id(parameter))
            if replacement is None:
                value = parameter.to(
                    device=device,
                    dtype=dtype if parameter.is_floating_point() else parameter.dtype,
                )
                replacement = nn.Parameter(
                    value,
                    requires_grad=parameter.requires_grad,
                )
                converted[id(parameter)] = replacement
            module._parameters[name] = replacement
        for name, buffer in tuple(module._buffers.items()):
            if buffer is None or id(buffer) in excluded_tensor_ids:
                continue
            replacement = converted.get(id(buffer))
            if replacement is None:
                replacement = buffer.to(
                    device=device,
                    dtype=dtype if buffer.is_floating_point() else buffer.dtype,
                )
                converted[id(buffer)] = replacement
            module._buffers[name] = replacement


def _move_nested(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, tuple):
        return tuple(_move_nested(item, device) for item in value)
    if isinstance(value, list):
        return [_move_nested(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move_nested(item, device) for key, item in value.items()}
    return value


def _synchronize(execution_device: TorchExecutionDevice) -> None:
    device = execution_device.device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "xpu":
        torch.xpu.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


__all__ = [
    "BoundedLayerResidency",
    "DenseMemoryUnit",
    "DenseTieringConfig",
    "DenseTieringError",
    "DenseTieringRuntime",
    "accelerator_free_memory_bytes",
    "add_dense_tiering_arguments",
    "available_host_memory_bytes",
    "configure_dense_tiering",
    "dense_tiering_config_from_args",
    "unique_tensor_bytes",
]
