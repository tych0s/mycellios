"""Authenticated, byte-bounded storage -> RAM residency for native GGUF.

This is the low-memory execution path for a dense native GGUF stage.  The
Transformers module is constructed on the ``meta`` device, fixed tensors stay
resident, and exactly one decoder layer is materialized from authenticated
GGUF offsets while it executes.  Layer weights are immutable and eviction
replaces them with meta placeholders; it never writes a cache file or copies
weights back to storage.

There is deliberately no prefetch in this revision.  Every storage read and
RAM -> accelerator copy is synchronous and measured around the operation that
actually performs it.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from .dense_tiering import (
    DenseTieringConfig,
    DenseTieringError,
    accelerator_free_memory_bytes,
    available_host_memory_bytes,
)
from .device import TorchExecutionDevice
from .model import (
    StageModelSpec,
    _checkpoint_name,
    _validate_checkpoint_coverage,
)
from .model_adapters import SelectiveStageAdapter
from .native_gguf import (
    GgufDocument,
    GgufTensor,
    NativeGgufError,
    NativeGgufStagePackage,
    _hf_checkpoint_name,
    _restore_huggingface_tensor_layout,
    _validate_derived_rope_factors,
    dequantize_gguf_tensor,
)


_LOCAL_LAYER = re.compile(r"^model\.layers\.(?P<index>[0-9]+)\.")
_GGUF_LAYER = re.compile(r"^blk\.(?P<index>[0-9]+)\.")
_AUTO_ACCELERATOR_RESERVE_BYTES = 512 * 1024 * 1024
_AUTHENTICATION_CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True, slots=True)
class NativeGgufDiskTierDecision:
    """Pre-allocation decision based only on authenticated tensor metadata."""

    enabled: bool
    full_resident_weight_bytes: int
    static_weight_bytes: int
    largest_layer_weight_bytes: int
    explicit_host_working_set_upper_bound_bytes: int
    host_budget_bytes: int | None
    host_budget_source: str


@dataclass(frozen=True, slots=True)
class _TensorAlias:
    module: nn.Module
    name: str
    kind: str
    requires_grad: bool


@dataclass(frozen=True, slots=True)
class _TensorTarget:
    checkpoint_name: str
    descriptor: GgufTensor
    aliases: tuple[_TensorAlias, ...]
    expected_shape: tuple[int, ...]
    resident_bytes: int
    layer_index: int | None


@dataclass(frozen=True, slots=True)
class _LayerUnit:
    index: int
    module: nn.Module
    targets: tuple[_TensorTarget, ...]
    resident_bytes: int


def decide_native_gguf_disk_tiering(
    parsed: GgufDocument,
    *,
    architecture: str,
    resident_dtype: torch.dtype,
    config: DenseTieringConfig,
    measured_available_bytes: int | None = None,
    derived_static_bytes: int = 0,
) -> NativeGgufDiskTierDecision:
    """Choose paging before constructing any resident model parameters.

    The bound is conservative and explicit: all fixed weights, the largest
    decoder layer, and the largest raw/decode/layout temporary can coexist.
    It is not labelled process RSS because allocator and kernel workspaces are
    outside this code's observability.
    """

    if resident_dtype not in (torch.float16, torch.float32, torch.bfloat16):
        raise DenseTieringError("native_gguf_disk_tiering_dtype_is_unsupported")
    if (
        not isinstance(derived_static_bytes, int)
        or isinstance(derived_static_bytes, bool)
        or derived_static_bytes < 0
    ):
        raise ValueError("derived_static_bytes must be a non-negative integer")
    element_bytes = torch.empty((), dtype=resident_dtype).element_size()
    static_bytes = derived_static_bytes
    layer_bytes: dict[int, int] = {}
    largest_temporary = 0
    for tensor in parsed.tensors:
        if tensor.name.startswith("rope_freqs."):
            largest_temporary = max(
                largest_temporary,
                _explicit_tensor_temporary_bytes(tensor, architecture),
            )
            continue
        resident = tensor.element_count * element_bytes
        match = _GGUF_LAYER.match(tensor.name)
        if match is None:
            static_bytes += resident
        else:
            index = int(match.group("index"))
            layer_bytes[index] = layer_bytes.get(index, 0) + resident
        largest_temporary = max(
            largest_temporary,
            _explicit_tensor_temporary_bytes(tensor, architecture),
        )
    if not layer_bytes:
        raise DenseTieringError("native_gguf_disk_tiering_stage_has_no_layers")
    largest_layer = max(layer_bytes.values())
    full = static_bytes + sum(layer_bytes.values())
    working_set = static_bytes + largest_layer + largest_temporary
    measured = (
        available_host_memory_bytes()
        if measured_available_bytes is None
        else measured_available_bytes
    )
    if config.host_ram_budget_bytes > 0:
        if measured is None:
            host_budget = config.host_ram_budget_bytes
            source = "sealed"
        else:
            host_budget = min(config.host_ram_budget_bytes, measured)
            source = "sealed-clamped-to-os-available"
    else:
        host_budget = measured
        source = "os-available" if measured is not None else "unavailable"
    resident_load_peak = full + largest_temporary
    enabled = (
        host_budget is not None
        and resident_load_peak > host_budget
    )
    if enabled and not config.allow_bounded_layer_cache:
        raise DenseTieringError(
            "native_gguf_disk_tiering_resident_load_exceeds_host_ram_and_tiering_is_disabled:"
            f"weights={full}:loadPeak={resident_load_peak}:budget={host_budget}"
        )
    if enabled and working_set > host_budget:
        # Keep the historical prefix because callers already use it as a
        # fail-closed launch diagnostic.
        raise DenseTieringError(
            "dense_tiering_authenticated_gguf_stage_exceeds_host_ram_budget:"
            f"workingSet={working_set}:fullStage={full}:budget={host_budget}"
        )
    return NativeGgufDiskTierDecision(
        enabled=enabled,
        full_resident_weight_bytes=full,
        static_weight_bytes=static_bytes,
        largest_layer_weight_bytes=largest_layer,
        explicit_host_working_set_upper_bound_bytes=working_set,
        host_budget_bytes=host_budget,
        host_budget_source=source,
    )


class NativeGgufDiskTierRuntime:
    """One-layer synchronous GGUF pager installed on a meta-constructed model."""

    mode = "storage-backed-layer-cache"

    def __init__(
        self,
        model: nn.Module,
        spec: StageModelSpec,
        package: NativeGgufStagePackage,
        parsed: GgufDocument,
        adapter: SelectiveStageAdapter,
        *,
        execution_device: TorchExecutionDevice,
        compute_dtype: torch.dtype,
        config: DenseTieringConfig,
        decision: NativeGgufDiskTierDecision,
        gguf_config: Mapping[str, Any],
    ) -> None:
        if not decision.enabled:
            raise ValueError("disk tier runtime requires an enabled decision")
        if spec.compile_mode is not None:
            raise DenseTieringError(
                "native_gguf_disk_tiering_is_incompatible_with_torch_compile"
            )
        if spec.quantize is not None:
            raise DenseTieringError(
                "native_gguf_disk_tiering_is_incompatible_with_dynamic_quantization"
            )
        self.model = model
        self.package = package
        self.parsed = parsed
        self.execution_device = execution_device
        self.compute_dtype = compute_dtype
        self.config = config
        self.decision = decision
        self.gguf_config = dict(gguf_config)
        self._lock = threading.RLock()
        self._forward_lock = threading.Lock()
        self._forward_owned = False
        self._closed = False
        self._active_layer: int | None = None
        self._hooks: list[Any] = []
        self._stream = parsed.path.open("rb")
        initial_stat = os.fstat(self._stream.fileno())
        self._file_identity = (
            int(initial_stat.st_dev),
            int(initial_stat.st_ino),
            int(initial_stat.st_size),
            int(initial_stat.st_mtime_ns),
        )
        self._range_digests: dict[str, str] = {}
        self._artifact_bytes_authenticated = 0
        self._file_bytes_authenticated = 0
        self._artifact_bytes_materialized = 0
        self._tensor_reads = 0
        self._storage_to_ram_ns = 0
        self._ram_to_vram_bytes = 0
        self._ram_to_vram_ns = 0
        self._layer_loads = 0
        self._layer_evictions = 0
        self._current_resident_layers = 0
        self._max_concurrent_resident_layers = 0
        self._current_ram_weight_bytes = 0
        self._peak_ram_weight_bytes = 0
        self._static_device_bytes = 0
        self._largest_actual_layer_bytes = 0
        try:
            self._authenticate_ranges()
            targets = _build_targets(
                model,
                spec,
                parsed,
                adapter=adapter,
                resident_dtype=compute_dtype,
            )
            static_targets = tuple(
                target for target in targets if target.layer_index is None
            )
            layers = tuple(model.model.layers)
            layer_units = tuple(
                _LayerUnit(
                    index=index,
                    module=layer,
                    targets=tuple(
                        target
                        for target in targets
                        if target.layer_index == index
                    ),
                    resident_bytes=sum(
                        target.resident_bytes
                        for target in targets
                        if target.layer_index == index
                    ),
                )
                for index, layer in enumerate(layers)
            )
            if any(
                not unit.targets or unit.resident_bytes < 1
                for unit in layer_units
            ):
                raise DenseTieringError(
                    "native_gguf_disk_tiering_layer_has_no_authenticated_weights"
                )
            self._layers = layer_units
            self._static_target_bytes = sum(
                target.resident_bytes for target in static_targets
            )
            self._rotary_bytes = sum(
                buffer.numel() * buffer.element_size()
                for buffer in model.model.rotary_emb.buffers()
            )
            planned_layers = tuple(unit.resident_bytes for unit in layer_units)
            if (
                self._static_target_bytes + self._rotary_bytes
                != decision.static_weight_bytes
                or max(planned_layers) != decision.largest_layer_weight_bytes
                or self._static_target_bytes
                + self._rotary_bytes
                + sum(planned_layers)
                != decision.full_resident_weight_bytes
            ):
                raise DenseTieringError(
                    "native_gguf_disk_tiering_metadata_and_model_plan_differ"
                )
            self._preflight_accelerator()
            actual_rotary_bytes = _materialize_rotary_embedding(
                model,
                execution_device.device,
            )
            if actual_rotary_bytes != self._rotary_bytes:
                raise DenseTieringError(
                    "native_gguf_disk_tiering_derived_static_size_changed:"
                    f"planned={self._rotary_bytes}:actual={actual_rotary_bytes}"
                )
            self._load_targets(static_targets, static=True)
            _assert_only_layer_meta_state(model)
            _assert_all_layer_state_meta(model)
            self._register_hooks()
        except BaseException:
            for hook in self._hooks:
                hook.remove()
            self._hooks.clear()
            self._stream.close()
            raise

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            static_ram = (
                self._static_target_bytes + self._rotary_bytes
                if self.execution_device.device.type == "cpu"
                else 0
            )
            return {
                "schema": "mycellios-dense-tiering/1",
                "mode": self.mode,
                "device": str(self.execution_device.device),
                "computeDtype": str(self.compute_dtype).removeprefix("torch."),
                "hostBudgetBytes": self.decision.host_budget_bytes,
                "hostBudgetSource": self.decision.host_budget_source,
                "vramBudgetBytes": self._vram_budget_bytes,
                "vramBudgetSource": self._vram_budget_source,
                "activationReserveBytes": self._activation_reserve_bytes,
                "totalWeightBytes": self.decision.full_resident_weight_bytes,
                "hostResidentWeightBytes": static_ram
                + (
                    self._largest_actual_layer_bytes
                    if self._active_layer is not None
                    and self.execution_device.device.type == "cpu"
                    else 0
                ),
                "staticVramWeightBytes": self._static_device_bytes,
                "currentVramWeightBytes": self._static_device_bytes
                + (
                    self._largest_actual_layer_bytes
                    if self._active_layer is not None
                    and self.execution_device.device.type != "cpu"
                    else 0
                ),
                "peakVramWeightBytes": self._static_device_bytes
                + (
                    self.decision.largest_layer_weight_bytes
                    if self.execution_device.device.type != "cpu"
                    else 0
                ),
                "currentRamWeightBytes": self._current_ram_weight_bytes,
                "peakRamWeightBytes": self._peak_ram_weight_bytes,
                "storageToRam": {
                    "schema": "mycellios-storage-to-ram/1",
                    "format": "gguf-authenticated-offsets",
                    "fileBytesAuthenticated": self._file_bytes_authenticated,
                    "artifactBytesAuthenticated": self._artifact_bytes_authenticated,
                    "artifactBytesMaterialized": self._artifact_bytes_materialized,
                    "tensorReadOperations": self._tensor_reads,
                    "materializeAndCopyNanoseconds": self._storage_to_ram_ns,
                    "physicalDiskBytes": None,
                    "osPageCacheHits": None,
                    "writes": 0,
                },
                "explicitHostWorkingSetUpperBoundBytes":
                    self.decision.explicit_host_working_set_upper_bound_bytes,
                "layerLoads": self._layer_loads,
                "layerEvictions": self._layer_evictions,
                "currentResidentLayers": self._current_resident_layers,
                "maxConcurrentResidentLayers":
                    self._max_concurrent_resident_layers,
                "prefetchEnabled": False,
                "overlapVerified": False,
                "ramToVramBytes": self._ram_to_vram_bytes,
                "ramToVramNanoseconds": self._ram_to_vram_ns,
            }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                self._evict_active()
                _assert_all_layer_state_meta(self.model)
            finally:
                for hook in self._hooks:
                    hook.remove()
                self._hooks.clear()
                self._stream.close()
                self._closed = True

    def _authenticate_ranges(self) -> None:
        before = self._current_file_identity()
        digest = hashlib.sha256()
        range_hashes = {
            descriptor.name: hashlib.sha256()
            for descriptor in self.parsed.tensors
        }
        range_counts = {descriptor.name: 0 for descriptor in self.parsed.tensors}
        ordered = tuple(
            sorted(self.parsed.tensors, key=lambda item: item.data_offset)
        )
        for left, right in zip(ordered, ordered[1:]):
            if left.data_offset + left.size_bytes > right.data_offset:
                raise NativeGgufError(
                    "native GGUF authentication ranges overlap"
                )
        self._stream.seek(0)
        file_offset = 0
        range_index = 0
        authentication_chunk_bytes = min(
            _AUTHENTICATION_CHUNK_BYTES,
            max(
                1,
                self.decision.host_budget_bytes
                or _AUTHENTICATION_CHUNK_BYTES,
            ),
        )
        while True:
            chunk = self._stream.read(authentication_chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
            chunk_end = file_offset + len(chunk)
            while (
                range_index < len(ordered)
                and ordered[range_index].data_offset
                + ordered[range_index].size_bytes
                <= file_offset
            ):
                range_index += 1
            scan_index = range_index
            while scan_index < len(ordered):
                descriptor = ordered[scan_index]
                tensor_start = descriptor.data_offset
                tensor_end = tensor_start + descriptor.size_bytes
                if tensor_start >= chunk_end:
                    break
                overlap_start = max(file_offset, tensor_start)
                overlap_end = min(chunk_end, tensor_end)
                relative_start = overlap_start - file_offset
                relative_end = overlap_end - file_offset
                range_hashes[descriptor.name].update(
                    chunk[relative_start:relative_end]
                )
                range_counts[descriptor.name] += overlap_end - overlap_start
                if tensor_end > chunk_end:
                    break
                scan_index += 1
            range_index = scan_index
            file_offset = chunk_end
        after = self._current_file_identity()
        if before != self._file_identity or after != self._file_identity:
            raise NativeGgufError(
                "native GGUF file identity changed during authentication"
            )
        if digest.hexdigest() != self.package.stage_gguf_sha256:
            raise NativeGgufError(
                "native GGUF changed before disk-tier authentication"
            )
        self._file_bytes_authenticated = file_offset
        for descriptor in ordered:
            if range_counts[descriptor.name] != descriptor.size_bytes:
                raise NativeGgufError(
                    f"GGUF tensor {descriptor.name!r} authentication range is incomplete"
                )
            self._range_digests[descriptor.name] = range_hashes[
                descriptor.name
            ].hexdigest()
            self._artifact_bytes_authenticated += descriptor.size_bytes
            if descriptor.name.startswith("rope_freqs."):
                raw = self._read_authenticated(descriptor)
                decoded = dequantize_gguf_tensor(descriptor, raw)
                _validate_derived_rope_factors(
                    decoded,
                    architecture=self.package.architecture,
                    config=self.gguf_config,
                )
                del decoded, raw

    def _preflight_accelerator(self) -> None:
        static = self._static_target_bytes + self._rotary_bytes
        if self.execution_device.device.type == "cpu":
            self._vram_budget_bytes = None
            self._vram_budget_source = "not-applicable"
            self._activation_reserve_bytes = 0
            self._static_device_bytes = 0
            return
        measured, measured_source = accelerator_free_memory_bytes(
            self.execution_device
        )
        if measured is None:
            raise DenseTieringError(
                "native_gguf_disk_tiering_cannot_measure_accelerator_free_memory"
            )
        if self.config.vram_budget_bytes > 0:
            budget = min(self.config.vram_budget_bytes, measured)
            source = f"sealed-clamped-to-{measured_source}"
        else:
            budget = measured
            source = measured_source
        reserve = (
            self.config.activation_reserve_bytes
            if self.config.activation_reserve_bytes is not None
            else min(_AUTO_ACCELERATOR_RESERVE_BYTES, max(0, budget // 8))
        )
        usable = budget - reserve
        required = static + self.decision.largest_layer_weight_bytes
        if required > usable:
            raise DenseTieringError(
                "native_gguf_disk_tiering_fixed_state_and_largest_layer_exceed_vram_budget:"
                f"required={required}:budget={usable}"
            )
        self._vram_budget_bytes = budget
        self._vram_budget_source = source
        self._activation_reserve_bytes = reserve
        self._static_device_bytes = static

    def _register_hooks(self) -> None:
        def model_before(
            _module: nn.Module,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> tuple[tuple[Any, ...], dict[str, Any]]:
            self._forward_lock.acquire()
            self._forward_owned = True
            return args, kwargs

        def model_after(
            _module: nn.Module,
            _args: tuple[Any, ...],
            _kwargs: dict[str, Any],
            output: Any,
        ) -> Any:
            try:
                with self._lock:
                    self._evict_active()
            finally:
                if self._forward_owned:
                    self._forward_owned = False
                    self._forward_lock.release()
            return output

        self._hooks.append(
            self.model.model.register_forward_pre_hook(
                model_before,
                with_kwargs=True,
            )
        )
        self._hooks.append(
            self.model.model.register_forward_hook(
                model_after,
                with_kwargs=True,
                always_call=True,
            )
        )
        for unit in self._layers:
            def layer_before(
                _module: nn.Module,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
                *,
                selected: _LayerUnit = unit,
            ) -> tuple[tuple[Any, ...], dict[str, Any]]:
                with self._lock:
                    if self._active_layer is not None:
                        raise DenseTieringError(
                            "native_gguf_disk_tiering_detected_overlapping_layers"
                        )
                    self._load_targets(selected.targets, static=False)
                    self._active_layer = selected.index
                    self._current_resident_layers = 1
                    self._max_concurrent_resident_layers = max(
                        self._max_concurrent_resident_layers,
                        self._current_resident_layers,
                    )
                    self._largest_actual_layer_bytes = selected.resident_bytes
                    self._layer_loads += 1
                return args, kwargs

            def layer_after(
                _module: nn.Module,
                _args: tuple[Any, ...],
                _kwargs: dict[str, Any],
                output: Any,
                *,
                selected: _LayerUnit = unit,
            ) -> Any:
                with self._lock:
                    if self._active_layer is None:
                        # A failing pre-hook never made this layer resident.
                        # The base-model always-call hook still releases the
                        # forward serialization lock.
                        return output
                    if self._active_layer != selected.index:
                        raise DenseTieringError(
                            "native_gguf_disk_tiering_layer_lifecycle_mismatch"
                        )
                    self._evict_unit(selected)
                return output

            self._hooks.append(
                unit.module.register_forward_pre_hook(
                    layer_before,
                    with_kwargs=True,
                )
            )
            self._hooks.append(
                unit.module.register_forward_hook(
                    layer_after,
                    with_kwargs=True,
                    always_call=True,
                )
            )

    def _load_targets(
        self,
        targets: Sequence[_TensorTarget],
        *,
        static: bool,
    ) -> None:
        installed: list[_TensorTarget] = []
        layer_ram = 0
        try:
            for target in targets:
                started = time.perf_counter_ns()
                raw = self._read_authenticated(target.descriptor)
                decoded = dequantize_gguf_tensor(target.descriptor, raw)
                restored = _restore_huggingface_tensor_layout(
                    target.descriptor.name,
                    decoded,
                    architecture=self.package.architecture,
                    config=self.gguf_config,
                )
                if tuple(restored.shape) != target.expected_shape:
                    raise NativeGgufError(
                        f"shape mismatch for {target.checkpoint_name!r}: "
                        f"GGUF {tuple(restored.shape)}, target {target.expected_shape}"
                    )
                host_value = restored.to(
                    device=torch.device("cpu"),
                    dtype=self.compute_dtype,
                )
                materialized = max(0, time.perf_counter_ns() - started)
                if self.execution_device.device.type == "cpu":
                    converted = host_value
                    transfer_elapsed = 0
                else:
                    transfer_started = time.perf_counter_ns()
                    converted = host_value.to(
                        device=self.execution_device.device,
                    )
                    _synchronize(self.execution_device)
                    transfer_elapsed = max(
                        0,
                        time.perf_counter_ns() - transfer_started,
                    )
                _install_target(target, converted)
                installed.append(target)
                layer_ram += (
                    target.resident_bytes
                    if self.execution_device.device.type == "cpu"
                    else 0
                )
                self._artifact_bytes_materialized += len(raw)
                self._tensor_reads += 1
                self._storage_to_ram_ns += materialized
                if self.execution_device.device.type != "cpu":
                    self._ram_to_vram_bytes += target.resident_bytes
                    self._ram_to_vram_ns += transfer_elapsed
                del converted, host_value, restored, decoded, raw
            if static:
                self._current_ram_weight_bytes = (
                    self._static_target_bytes + self._rotary_bytes
                    if self.execution_device.device.type == "cpu"
                    else 0
                )
            else:
                self._current_ram_weight_bytes = (
                    self._static_target_bytes + self._rotary_bytes + layer_ram
                    if self.execution_device.device.type == "cpu"
                    else 0
                )
            self._peak_ram_weight_bytes = max(
                self._peak_ram_weight_bytes,
                self._current_ram_weight_bytes,
            )
        except BaseException:
            for target in reversed(installed):
                _evict_target(target, self.compute_dtype)
            raise

    def _evict_unit(self, unit: _LayerUnit) -> None:
        _synchronize(self.execution_device)
        for target in unit.targets:
            _evict_target(target, self.compute_dtype)
        self._active_layer = None
        self._current_resident_layers = 0
        self._current_ram_weight_bytes = (
            self._static_target_bytes + self._rotary_bytes
            if self.execution_device.device.type == "cpu"
            else 0
        )
        self._layer_evictions += 1
        _assert_targets_meta(unit.targets)

    def _evict_active(self) -> None:
        if self._active_layer is None:
            return
        self._evict_unit(self._layers[self._active_layer])

    def _read_authenticated(self, descriptor: GgufTensor) -> bytes:
        self._assert_file_identity()
        raw = self._read_exact_unchecked(descriptor)
        expected = self._range_digests.get(descriptor.name)
        if expected is None or hashlib.sha256(raw).hexdigest() != expected:
            raise NativeGgufError(
                f"authenticated GGUF range {descriptor.name!r} changed on storage"
            )
        return raw

    def _read_exact_unchecked(self, descriptor: GgufTensor) -> bytes:
        self._stream.seek(descriptor.data_offset)
        raw = self._stream.read(descriptor.size_bytes)
        if len(raw) != descriptor.size_bytes:
            raise NativeGgufError(
                f"GGUF tensor {descriptor.name!r} is truncated"
            )
        return raw

    def _assert_file_identity(self) -> None:
        if self._current_file_identity() != self._file_identity:
            raise NativeGgufError(
                "authenticated GGUF file identity changed after startup"
            )

    def _current_file_identity(self) -> tuple[int, int, int, int]:
        stat = os.fstat(self._stream.fileno())
        return (
            int(stat.st_dev),
            int(stat.st_ino),
            int(stat.st_size),
            int(stat.st_mtime_ns),
        )


def _build_targets(
    model: nn.Module,
    spec: StageModelSpec,
    parsed: GgufDocument,
    *,
    adapter: SelectiveStageAdapter,
    resident_dtype: torch.dtype,
) -> tuple[_TensorTarget, ...]:
    gguf_by_checkpoint: dict[str, GgufTensor] = {}
    for tensor in parsed.tensors:
        if tensor.name.startswith("rope_freqs."):
            continue
        checkpoint = _hf_checkpoint_name(tensor.name)
        if checkpoint in gguf_by_checkpoint:
            raise NativeGgufError(
                f"multiple GGUF tensors map to {checkpoint!r}"
            )
        gguf_by_checkpoint[checkpoint] = tensor
    checkpoint_files = {
        checkpoint: parsed.path.name for checkpoint in gguf_by_checkpoint
    }
    state = model.state_dict(keep_vars=True)
    aliases_by_identity: dict[int, list[str]] = {}
    tensor_by_identity: dict[int, torch.Tensor] = {}
    for local_name, tensor in state.items():
        aliases_by_identity.setdefault(id(tensor), []).append(local_name)
        tensor_by_identity[id(tensor)] = tensor
    targets: list[_TensorTarget] = []
    loaded_checkpoints: set[str] = set()
    tied_embeddings = bool(getattr(model.config, "tie_word_embeddings", False))
    for identity, local_names in aliases_by_identity.items():
        local_name = local_names[0]
        tensor = tensor_by_identity[identity]
        assignments = adapter.checkpoint_assignments(
            local_name,
            tensor,
            layer_start=spec.layer_start,
            checkpoint_names=set(checkpoint_files),
        )
        if assignments is not None:
            raise NativeGgufError(
                "native GGUF disk tier does not accept sliced checkpoint assignments"
            )
        checkpoint = _checkpoint_name(
            local_name,
            spec,
            checkpoint_files,
            tied_embeddings=tied_embeddings,
        )
        if checkpoint is None:
            continue
        descriptor = gguf_by_checkpoint.get(checkpoint)
        if descriptor is None:
            raise NativeGgufError(
                f"GGUF tensor {checkpoint!r} required by {local_name!r} is missing"
            )
        aliases = tuple(_resolve_alias(model, name) for name in local_names)
        categories = {_layer_index(name) for name in local_names}
        if len(categories) != 1:
            raise NativeGgufError(
                "one tied GGUF tensor cannot span fixed and paged layer state"
            )
        targets.append(
            _TensorTarget(
                checkpoint_name=checkpoint,
                descriptor=descriptor,
                aliases=aliases,
                expected_shape=tuple(tensor.shape),
                resident_bytes=tensor.numel()
                * torch.empty((), dtype=resident_dtype).element_size(),
                layer_index=next(iter(categories)),
            )
        )
        loaded_checkpoints.add(checkpoint)
    _validate_checkpoint_coverage(
        checkpoint_files,
        loaded_checkpoints,
        spec,
        tied_embeddings=tied_embeddings,
    )
    unused = set(checkpoint_files) - loaded_checkpoints
    if unused:
        raise NativeGgufError(
            "native GGUF has authenticated tensors without a model destination: "
            f"{sorted(unused)}"
        )
    return tuple(
        sorted(
            targets,
            key=lambda target: (
                -1 if target.layer_index is None else target.layer_index,
                target.descriptor.data_offset,
            ),
        )
    )


def _resolve_alias(model: nn.Module, local_name: str) -> _TensorAlias:
    parts = local_name.split(".")
    owner: nn.Module = model
    for part in parts[:-1]:
        child = getattr(owner, part, None)
        if not isinstance(child, nn.Module):
            raise NativeGgufError(
                f"model tensor owner for {local_name!r} is invalid"
            )
        owner = child
    name = parts[-1]
    if name in owner._parameters and owner._parameters[name] is not None:
        parameter = owner._parameters[name]
        assert parameter is not None
        return _TensorAlias(
            owner,
            name,
            "parameter",
            bool(parameter.requires_grad),
        )
    if name in owner._buffers and owner._buffers[name] is not None:
        return _TensorAlias(owner, name, "buffer", False)
    raise NativeGgufError(f"model tensor slot {local_name!r} is invalid")


def _install_target(target: _TensorTarget, value: torch.Tensor) -> None:
    parameter: nn.Parameter | None = None
    for alias in target.aliases:
        if alias.kind == "parameter":
            if parameter is None:
                parameter = nn.Parameter(
                    value,
                    requires_grad=alias.requires_grad,
                )
            alias.module._parameters[alias.name] = parameter
        else:
            alias.module._buffers[alias.name] = value


def _evict_target(target: _TensorTarget, dtype: torch.dtype) -> None:
    placeholder = torch.empty(
        target.expected_shape,
        dtype=dtype,
        device="meta",
    )
    parameter: nn.Parameter | None = None
    for alias in target.aliases:
        if alias.kind == "parameter":
            if parameter is None:
                parameter = nn.Parameter(
                    placeholder,
                    requires_grad=alias.requires_grad,
                )
            alias.module._parameters[alias.name] = parameter
        else:
            alias.module._buffers[alias.name] = placeholder


def _materialize_rotary_embedding(
    model: nn.Module,
    device: torch.device,
) -> int:
    rotary = getattr(getattr(model, "model", None), "rotary_emb", None)
    if not isinstance(rotary, nn.Module):
        raise NativeGgufError("native GGUF model has no certified rotary embedding")
    try:
        replacement = type(rotary)(rotary.config, device=device)
    except (AttributeError, TypeError, RuntimeError) as error:
        raise NativeGgufError(
            "native GGUF rotary embedding cannot be materialized independently"
        ) from error
    model.model.rotary_emb = replacement
    return sum(
        buffer.numel() * buffer.element_size()
        for buffer in replacement.buffers()
    )


def _assert_only_layer_meta_state(model: nn.Module) -> None:
    invalid = [
        name
        for name, tensor in (
            *tuple(model.named_parameters(remove_duplicate=False)),
            *tuple(model.named_buffers(remove_duplicate=False)),
        )
        if tensor.device.type == "meta" and _layer_index(name) is None
    ]
    if invalid:
        raise NativeGgufError(
            "fixed native GGUF tensors remain unmaterialized: "
            f"{sorted(invalid)}"
        )


def _assert_all_layer_state_meta(model: nn.Module) -> None:
    resident = [
        name
        for name, tensor in (
            *tuple(model.named_parameters(remove_duplicate=False)),
            *tuple(model.named_buffers(remove_duplicate=False)),
        )
        if _layer_index(name) is not None and tensor.device.type != "meta"
    ]
    if resident:
        raise DenseTieringError(
            "native_gguf_disk_tiering_left_layers_resident_outside_execution:"
            f"{sorted(resident)}"
        )


def _assert_targets_meta(targets: Sequence[_TensorTarget]) -> None:
    resident: list[str] = []
    for target in targets:
        for alias in target.aliases:
            tensor = (
                alias.module._parameters.get(alias.name)
                if alias.kind == "parameter"
                else alias.module._buffers.get(alias.name)
            )
            if tensor is not None and tensor.device.type != "meta":
                resident.append(f"{target.checkpoint_name}:{alias.name}")
    if resident:
        raise DenseTieringError(
            "native_gguf_disk_tiering_failed_to_evict_layer_targets:"
            f"{resident}"
        )


def _layer_index(local_name: str) -> int | None:
    match = _LOCAL_LAYER.match(local_name)
    return None if match is None else int(match.group("index"))


def _explicit_tensor_temporary_bytes(
    tensor: GgufTensor,
    architecture: str,
) -> int:
    decoded = tensor.element_count * 4
    restored_copy = (
        decoded
        if architecture == "llama"
        and (
            tensor.name.endswith(".attn_q.weight")
            or tensor.name.endswith(".attn_q.bias")
            or tensor.name.endswith(".attn_k.weight")
            or tensor.name.endswith(".attn_k.bias")
        )
        else 0
    )
    return tensor.size_bytes + decoded + restored_copy


def _synchronize(execution_device: TorchExecutionDevice) -> None:
    device = execution_device.device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "xpu":
        torch.xpu.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


__all__ = [
    "NativeGgufDiskTierDecision",
    "NativeGgufDiskTierRuntime",
    "decide_native_gguf_disk_tiering",
]
