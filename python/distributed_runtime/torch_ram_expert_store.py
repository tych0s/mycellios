"""Torch RAM-to-device expert storage for predictive MoE macro-stages.

``TorchRamExpertStore`` is the physical companion to
``RamBackedExpertScheduler``.  Authoritative expert tensors are owned as
contiguous, pageable CPU tensors.  CUDA workers may opt into exactly two
reusable pinned staging slots -- one execution slot and one prefetch slot --
instead of pinning a second copy of the whole expert inventory.  Transfers use
a dedicated stream and record events; neither staging slot is overwritten
until its previous device copy has completed.  ``ensure_ready`` makes the
current compute stream wait for those events before the scheduler exposes an
authoritative route.  Without CUDA the same API uses independent contiguous
CPU copies and never attempts a pinned allocation, which keeps CPU
certification exact rather than pretending asynchronous transfer exists.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Literal, Mapping, Protocol, Sequence

import torch

from .ram_expert_cache import (
    ExpertKey,
    ExpertRecord,
    MacroStageExpertInventory,
    PredictiveCacheConfig,
)


@dataclass(frozen=True)
class TorchExpertBundle:
    key: ExpertKey
    content_id: str
    tensors: tuple[tuple[str, torch.Tensor], ...]

    @property
    def byte_size(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for _, tensor in self.tensors
        )

    @property
    def device(self) -> torch.device:
        return self.tensors[0][1].device

    def tensor(self, name: str = "weight") -> torch.Tensor:
        for tensor_name, tensor in self.tensors:
            if tensor_name == name:
                return tensor
        raise KeyError(f"tensor {name!r} is absent from expert {self.key}")

    def packed_gate_up(self) -> torch.Tensor:
        """Return the zero-copy packed gate/up matrix used by HF grouped MoE.

        Physical canonical expert copies deliberately place gate and up in one
        allocation and expose each half as a view.  Reconstructing the complete
        view avoids a per-token ``torch.cat`` allocation that would otherwise
        violate the byte-bounded device cache and alter the grouped GEMM path.
        """

        gate = self.tensor("gate_proj.weight")
        up = self.tensor("up_proj.weight")
        if (
            gate.ndim != 2
            or tuple(up.shape) != tuple(gate.shape)
            or gate.dtype != up.dtype
            or gate.device != up.device
            or not gate.is_contiguous()
            or not up.is_contiguous()
            or gate.untyped_storage().data_ptr() != up.untyped_storage().data_ptr()
            or up.storage_offset() != gate.storage_offset() + gate.numel()
        ):
            raise RuntimeError(
                f"expert {self.key} gate/up tensors are not one packed allocation"
            )
        return gate.as_strided(
            (int(gate.shape[0]) * 2, int(gate.shape[1])),
            gate.stride(),
            gate.storage_offset(),
        )


@dataclass(frozen=True)
class TorchExpertStoreSnapshot:
    requested_device: str
    effective_device: str
    cpu_fallback: bool
    pin_memory_requested: bool
    pin_memory_enabled: bool
    separate_prefetch_stream: bool
    ram_keys: tuple[ExpertKey, ...]
    active_keys: tuple[ExpertKey, ...]
    prefetch_keys: tuple[ExpertKey, ...]
    ephemeral_keys: tuple[ExpertKey, ...]
    ram_bytes: int
    active_bytes: int
    prefetch_bytes: int
    ephemeral_bytes: int
    copied_bytes: int
    prefetch_copies: int
    fallback_copies: int
    readiness_waits: int
    ram_adopted_without_clone: bool
    bounded_pinned_staging_requested: bool
    bounded_pinned_staging_enabled: bool
    pinned_slot_bytes: int
    pinned_capacity_bytes: int
    pinned_occupied_bytes: int
    pinned_execution_key: ExpertKey | None
    pinned_prefetch_key: ExpertKey | None
    pageable_to_pinned_bytes: int
    pinned_to_device_bytes: int
    pinned_staging_hits: int
    pinned_staging_misses: int
    pinned_reuse_stalls: int
    pinned_reuse_stall_ms: float


class _TransferCompletion(Protocol):
    def query(self) -> bool: ...

    def synchronize(self) -> None: ...


StagingRole = Literal["execution", "prefetch"]


@dataclass
class _PinnedStagingSlot:
    role: StagingRole
    storage: torch.Tensor
    key: ExpertKey | None = None
    content_id: str | None = None
    byte_size: int = 0
    completion: _TransferCompletion | None = None


@dataclass(frozen=True)
class BoundedPinnedStagingSnapshot:
    slot_bytes: int
    capacity_bytes: int
    occupied_bytes: int
    execution_key: ExpertKey | None
    prefetch_key: ExpertKey | None
    pageable_to_pinned_bytes: int
    staging_hits: int
    staging_misses: int
    reuse_stalls: int
    reuse_stall_ms: float


class BoundedPinnedExpertStaging:
    """Exactly two reusable pinned slots for pageable expert weights.

    The raw byte buffers are allocated once and never grow. Tensor views are
    packed in decreasing element-size order, so no alignment padding is needed
    and every bundle at or below ``largest_expert_bytes`` fits the sealed slot
    budget. ``allocator`` and ``clock_ns`` are injectable only so lifecycle and
    capacity can be certified on a CPU-only host; production uses real pinned
    allocations and CUDA events.
    """

    _ROLES: tuple[StagingRole, ...] = ("execution", "prefetch")

    def __init__(
        self,
        largest_expert_bytes: int,
        *,
        allocator: Callable[[int], torch.Tensor] | None = None,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        if (
            not isinstance(largest_expert_bytes, int)
            or isinstance(largest_expert_bytes, bool)
            or largest_expert_bytes < 1
        ):
            raise ValueError("largest_expert_bytes must be a positive integer")
        allocate = allocator or self._allocate_pinned_bytes
        require_pinned = allocator is None
        self.slot_bytes = largest_expert_bytes
        self._clock_ns = clock_ns
        self._slots: dict[StagingRole, _PinnedStagingSlot] = {}
        for role in self._ROLES:
            storage = allocate(largest_expert_bytes)
            self._validate_storage(
                storage,
                largest_expert_bytes,
                require_pinned=require_pinned,
            )
            self._slots[role] = _PinnedStagingSlot(role=role, storage=storage)
        self._pageable_to_pinned_bytes = 0
        self._hits = 0
        self._misses = 0
        self._reuse_stalls = 0
        self._reuse_stall_ns = 0

    def stage(
        self,
        role: StagingRole,
        source: TorchExpertBundle,
    ) -> TorchExpertBundle:
        slot = self._slot(role)
        if source.byte_size > self.slot_bytes:
            raise MemoryError(
                f"expert {source.key} requires {source.byte_size} pinned bytes "
                f"but each bounded staging slot has {self.slot_bytes} bytes"
            )
        hit = (
            slot.key == source.key
            and slot.content_id == source.content_id
            and slot.byte_size == source.byte_size
        )
        if hit:
            # Expert bytes are immutable. The same transfer stream may read
            # them again without a host wait because stream order preserves the
            # previous read before the next one.
            self._hits += 1
            return self._bundle_views(slot, source, copy_from_pageable=False)

        self._wait_before_overwrite(slot)
        staged = self._bundle_views(slot, source, copy_from_pageable=True)
        slot.key = source.key
        slot.content_id = source.content_id
        slot.byte_size = source.byte_size
        self._misses += 1
        self._pageable_to_pinned_bytes += source.byte_size
        return staged

    def mark_in_flight(
        self,
        role: StagingRole,
        *,
        key: ExpertKey,
        completion: _TransferCompletion,
    ) -> None:
        slot = self._slot(role)
        if slot.key != key:
            raise RuntimeError(
                f"cannot bind {key} completion to {role} slot containing {slot.key}"
            )
        if not callable(getattr(completion, "query", None)) or not callable(
            getattr(completion, "synchronize", None)
        ):
            raise TypeError("staging completion must provide query and synchronize")
        slot.completion = completion

    def snapshot(self) -> BoundedPinnedStagingSnapshot:
        execution = self._slots["execution"]
        prefetch = self._slots["prefetch"]
        return BoundedPinnedStagingSnapshot(
            slot_bytes=self.slot_bytes,
            capacity_bytes=self.slot_bytes * len(self._slots),
            occupied_bytes=sum(slot.byte_size for slot in self._slots.values()),
            execution_key=execution.key,
            prefetch_key=prefetch.key,
            pageable_to_pinned_bytes=self._pageable_to_pinned_bytes,
            staging_hits=self._hits,
            staging_misses=self._misses,
            reuse_stalls=self._reuse_stalls,
            reuse_stall_ms=self._reuse_stall_ns / 1_000_000,
        )

    @staticmethod
    def _allocate_pinned_bytes(byte_count: int) -> torch.Tensor:
        return torch.empty(
            byte_count,
            dtype=torch.uint8,
            device="cpu",
            pin_memory=True,
        )

    @staticmethod
    def _validate_storage(
        storage: torch.Tensor,
        expected_bytes: int,
        *,
        require_pinned: bool,
    ) -> None:
        if (
            not isinstance(storage, torch.Tensor)
            or storage.device.type != "cpu"
            or storage.dtype != torch.uint8
            or not storage.is_contiguous()
            or storage.numel() != expected_bytes
            or (require_pinned and not storage.is_pinned())
        ):
            raise ValueError(
                "pinned staging allocator must return an exact contiguous CPU uint8 buffer"
            )

    def _slot(self, role: StagingRole) -> _PinnedStagingSlot:
        try:
            return self._slots[role]
        except KeyError as exc:
            raise ValueError(f"unsupported staging role {role!r}") from exc

    def _wait_before_overwrite(self, slot: _PinnedStagingSlot) -> None:
        completion = slot.completion
        if completion is None:
            return
        if not completion.query():
            started = self._clock_ns()
            completion.synchronize()
            self._reuse_stall_ns += max(0, self._clock_ns() - started)
            self._reuse_stalls += 1
        slot.completion = None

    @staticmethod
    def _layout(source: TorchExpertBundle) -> tuple[tuple[int, int], ...]:
        # Torch scalar element sizes are powers of two. Larger types first keep
        # every following offset naturally aligned without extra padding.
        ordered = sorted(
            enumerate(source.tensors),
            key=lambda item: (-item[1][1].element_size(), item[0]),
        )
        offsets: list[tuple[int, int]] = []
        cursor = 0
        for index, (_, tensor) in ordered:
            if tensor.device.type != "cpu" or not tensor.is_contiguous():
                raise ValueError(
                    "authoritative staging sources must be contiguous CPU tensors"
                )
            if tensor.layout != torch.strided or tensor.is_meta:
                raise ValueError(
                    "authoritative staging sources must be materialized strided tensors"
                )
            if cursor % tensor.element_size() != 0:
                raise ValueError("expert tensor layout cannot be packed without padding")
            offsets.append((index, cursor))
            cursor += tensor.numel() * tensor.element_size()
        if cursor != source.byte_size:
            raise RuntimeError("expert staging layout byte accounting changed")
        return tuple(offsets)

    def _bundle_views(
        self,
        slot: _PinnedStagingSlot,
        source: TorchExpertBundle,
        *,
        copy_from_pageable: bool,
    ) -> TorchExpertBundle:
        by_index: dict[int, torch.Tensor] = {}
        for index, offset in self._layout(source):
            _, tensor = source.tensors[index]
            byte_count = tensor.numel() * tensor.element_size()
            view = (
                slot.storage.narrow(0, offset, byte_count)
                .view(tensor.dtype)
                .view(tuple(tensor.shape))
            )
            if copy_from_pageable:
                view.copy_(tensor, non_blocking=False)
            by_index[index] = view
        return TorchExpertBundle(
            key=source.key,
            content_id=source.content_id,
            tensors=tuple(
                (name, by_index[index])
                for index, (name, _) in enumerate(source.tensors)
            ),
        )


TensorSource = torch.Tensor | Mapping[str, torch.Tensor] | TorchExpertBundle


class TorchRamExpertStore:
    """Real Torch storage backend controlled by a RAM expert scheduler.

    The cache capacities are logical weight budgets.  CUDA runtime workspaces,
    attention tensors and KV cache must be reserved before choosing these
    values; this class cannot infer the rest of a process's VRAM use.
    """

    enforces_device_capacity = True

    def __init__(
        self,
        inventory: MacroStageExpertInventory,
        tensors: Mapping[ExpertKey, TensorSource],
        cache_config: PredictiveCacheConfig,
        *,
        device: str | torch.device = "cuda",
        pin_memory: bool = True,
        bounded_pinned_staging: bool = False,
        allow_cpu_fallback: bool = True,
        _adopt_cpu_bundles: bool = False,
    ) -> None:
        if not isinstance(inventory, MacroStageExpertInventory):
            raise TypeError("inventory must be MacroStageExpertInventory")
        if not isinstance(cache_config, PredictiveCacheConfig):
            raise TypeError("cache_config must be PredictiveCacheConfig")
        self.inventory = inventory
        self.cache_config = cache_config
        self.requested_device = torch.device(device)
        if self.requested_device.type not in {"cpu", "cuda"}:
            raise ValueError("TorchRamExpertStore supports only cpu or cuda")
        self.effective_device, self.cpu_fallback = self._select_device(
            self.requested_device,
            allow_cpu_fallback=allow_cpu_fallback,
        )
        self.pin_memory_requested = bool(pin_memory)
        self.pin_memory_enabled = False
        self.bounded_pinned_staging_requested = bool(bounded_pinned_staging)
        if self.pin_memory_requested and self.bounded_pinned_staging_requested:
            raise ValueError(
                "full-inventory pinning and bounded pinned staging are mutually exclusive"
            )

        expected_keys = {record.key for record in inventory.records}
        supplied_keys = set(tensors)
        missing = expected_keys - supplied_keys
        extra = supplied_keys - expected_keys
        if missing or extra:
            raise ValueError(
                f"tensor inventory mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
            )

        if _adopt_cpu_bundles:
            ram = {
                record.key: self._adopt_owned_bundle(record, tensors[record.key])
                for record in inventory.records
            }
        else:
            ram = {
                record.key: self._normalize_bundle(record, tensors[record.key])
                for record in inventory.records
            }
        self.ram_adopted_without_clone = bool(_adopt_cpu_bundles)
        if (
            self.pin_memory_requested
            and self.effective_device.type == "cuda"
            and torch.cuda.is_available()
        ):
            pinned: dict[ExpertKey, TorchExpertBundle] = {}
            try:
                for key, bundle in ram.items():
                    pinned[key] = TorchExpertBundle(
                        key=bundle.key,
                        content_id=bundle.content_id,
                        tensors=tuple(
                            (
                                name,
                                tensor
                                if tensor.is_pinned()
                                else tensor.pin_memory(),
                            )
                            for name, tensor in bundle.tensors
                        ),
                    )
            except RuntimeError:
                pinned = {}
            if len(pinned) == len(ram):
                if any(
                    not tensor.is_pinned()
                    for bundle in ram.values()
                    for _, tensor in bundle.tensors
                ):
                    self.ram_adopted_without_clone = False
                ram = pinned
                self.pin_memory_enabled = True

        self._ram = ram
        self._active: dict[ExpertKey, TorchExpertBundle] = {}
        self._prefetch: dict[ExpertKey, TorchExpertBundle] = {}
        self._ephemeral: dict[ExpertKey, TorchExpertBundle] = {}
        self._events: dict[ExpertKey, torch.cuda.Event] = {}
        # Device bundles are allocated on the transfer stream and consumed on
        # one or more compute streams.  CUDA's caching allocator only knows the
        # allocation stream unless every consumer is recorded explicitly.  We
        # also retain those streams so eviction can wait for their queued GEMMs
        # before dropping the last store-owned reference.  That wait preserves
        # the sealed two-buffer physical budget instead of allowing a pending
        # free to force a transient third expert allocation.
        self._compute_streams: dict[ExpertKey, list[torch.cuda.Stream]] = {}
        self._prefetch_stream = (
            torch.cuda.Stream(device=self.effective_device)
            if self.effective_device.type == "cuda"
            else None
        )
        self._bounded_pinned_staging: BoundedPinnedExpertStaging | None = None
        if (
            self.bounded_pinned_staging_requested
            and self.effective_device.type == "cuda"
        ):
            largest_expert_bytes = max(record.byte_size for record in inventory.records)
            try:
                self._bounded_pinned_staging = BoundedPinnedExpertStaging(
                    largest_expert_bytes
                )
            except (RuntimeError, ValueError) as error:
                # A CUDA worker must not silently violate a sealed bounded
                # staging contract by falling back to pageable transfers.
                raise RuntimeError(
                    "cannot allocate the bounded two-slot pinned expert staging pool"
                ) from error
        self._copied_bytes = 0
        self._pinned_to_device_bytes = 0
        self._prefetch_copies = 0
        self._fallback_copies = 0
        self._readiness_waits = 0

    @classmethod
    def adopt_owned_cpu_bundles(
        cls,
        inventory: MacroStageExpertInventory,
        bundles: Mapping[ExpertKey, TorchExpertBundle],
        cache_config: PredictiveCacheConfig,
        *,
        device: str | torch.device = "cuda",
        pin_memory: bool = False,
        bounded_pinned_staging: bool = False,
        allow_cpu_fallback: bool = True,
    ) -> "TorchRamExpertStore":
        """Adopt certified CPU buffers without making a second RAM copy.

        This is an explicit ownership-transfer API for loaders that already
        produced detached, contiguous, floating CPU tensors.  The caller must
        relinquish mutation of every supplied tensor after this call.  General
        callers should keep using the normal constructor, which clones inputs
        defensively.  Requesting pinned memory may still require one deliberate
        pageable-to-pinned copy when the source buffers are not pinned.
        ``bounded_pinned_staging=True`` instead keeps authoritative bundles
        pageable and allocates exactly two reusable largest-expert slots.
        """

        return cls(
            inventory,
            bundles,
            cache_config,
            device=device,
            pin_memory=pin_memory,
            bounded_pinned_staging=bounded_pinned_staging,
            allow_cpu_fallback=allow_cpu_fallback,
            _adopt_cpu_bundles=True,
        )

    def validate_cache_config(self, config: PredictiveCacheConfig) -> None:
        if config != self.cache_config:
            raise ValueError("scheduler and Torch store cache configs must match")

    def prefetch(self, record: ExpertRecord) -> None:
        self._validate_record(record)
        key = record.key
        if key in self._active or key in self._prefetch:
            return
        if self._prefetch_bytes() + record.byte_size > self.cache_config.prefetch_reserve_bytes:
            raise MemoryError("physical prefetch buffer would exceed its byte budget")
        self._ensure_device_capacity(record.byte_size, operation="prefetch")
        self._prefetch[key] = self._copy_to_effective_device(
            key,
            staging_role="prefetch",
        )
        self._prefetch_copies += 1

    def load_for_execution(self, record: ExpertRecord) -> None:
        self._validate_record(record)
        key = record.key
        if key in self._active or key in self._ephemeral:
            return
        if key in self._prefetch:
            self._ephemeral[key] = self._prefetch.pop(key)
            return
        self._ensure_device_capacity(record.byte_size, operation="execution load")
        self._ephemeral[key] = self._copy_to_effective_device(
            key,
            staging_role="execution",
        )
        self._fallback_copies += 1

    def commit_loaded(self, key: ExpertKey, *, cached: bool) -> None:
        if key in self._active:
            return
        bundle = self._ephemeral.get(key)
        if bundle is None:
            raise KeyError(f"expert {key} was not loaded for execution")
        if not cached:
            return
        projected_bytes = self._active_bytes() + bundle.byte_size
        if projected_bytes > self.cache_config.active_capacity_bytes:
            raise MemoryError("physical active cache would exceed its byte budget")
        self._active[key] = self._ephemeral.pop(key)

    def promote(self, keys: Sequence[ExpertKey]) -> None:
        for key in keys:
            if key in self._active:
                continue
            bundle = self._prefetch.get(key)
            if bundle is None:
                raise KeyError(f"expert {key} is absent from physical prefetch buffer")
            if self._active_bytes() + bundle.byte_size > self.cache_config.active_capacity_bytes:
                raise MemoryError("physical active cache would exceed its byte budget")
            self._active[key] = self._prefetch.pop(key)

    def evict(self, keys: Sequence[ExpertKey]) -> None:
        for key in keys:
            if key in self._active:
                self._wait_for_compute(key)
                self._wait_host(key)
                self._active.pop(key, None)
                self._drop_event_if_unused(key)

    def discard_prefetch(self, keys: Sequence[ExpertKey]) -> None:
        for key in keys:
            if key in self._prefetch:
                self._wait_for_compute(key)
                self._wait_host(key)
                self._prefetch.pop(key, None)
                self._drop_event_if_unused(key)

    def ensure_ready(self, keys: Sequence[ExpertKey]) -> None:
        compute_stream: torch.cuda.Stream | None = None
        for key in keys:
            bundle = self._bundle_without_wait(key)
            event = self._events.get(key)
            if event is not None:
                if compute_stream is None:
                    compute_stream = torch.cuda.current_stream(self.effective_device)
                compute_stream.wait_event(event)
                self._readiness_waits += 1
            if self.effective_device.type == "cuda":
                if compute_stream is None:
                    compute_stream = torch.cuda.current_stream(self.effective_device)
                self._record_compute_stream(key, bundle, compute_stream)

    def release_uncached(self, keys: Sequence[ExpertKey]) -> None:
        for key in keys:
            # The caller invokes this after launching the expert operation.
            # Wait for every stream on which the bundle was exposed, rather
            # than assuming release happens on that same thread/current stream.
            self._wait_for_compute(key)
            self._wait_host(key)
            self._ephemeral.pop(key, None)
            self._drop_event_if_unused(key)

    def bundle(self, key: ExpertKey, *, ensure_ready: bool = True) -> TorchExpertBundle:
        if ensure_ready:
            self.ensure_ready((key,))
        return self._bundle_without_wait(key)

    def bundles_for_route(
        self,
        keys: Sequence[ExpertKey],
    ) -> tuple[TorchExpertBundle, ...]:
        self.ensure_ready(keys)
        return tuple(self._bundle_without_wait(key) for key in keys)

    def ram_bundle(self, key: ExpertKey) -> TorchExpertBundle:
        try:
            return self._ram[key]
        except KeyError as exc:
            raise KeyError(f"expert {key} is absent from authoritative RAM") from exc

    def snapshot(self) -> TorchExpertStoreSnapshot:
        bounded = (
            self._bounded_pinned_staging.snapshot()
            if self._bounded_pinned_staging is not None
            else None
        )
        return TorchExpertStoreSnapshot(
            requested_device=str(self.requested_device),
            effective_device=str(self.effective_device),
            cpu_fallback=self.cpu_fallback,
            pin_memory_requested=self.pin_memory_requested,
            pin_memory_enabled=self.pin_memory_enabled,
            separate_prefetch_stream=self._prefetch_stream is not None,
            ram_keys=tuple(sorted(self._ram)),
            active_keys=tuple(sorted(self._active)),
            prefetch_keys=tuple(sorted(self._prefetch)),
            ephemeral_keys=tuple(sorted(self._ephemeral)),
            ram_bytes=sum(bundle.byte_size for bundle in self._ram.values()),
            active_bytes=self._active_bytes(),
            prefetch_bytes=self._prefetch_bytes(),
            ephemeral_bytes=sum(
                bundle.byte_size for bundle in self._ephemeral.values()
            ),
            copied_bytes=self._copied_bytes,
            prefetch_copies=self._prefetch_copies,
            fallback_copies=self._fallback_copies,
            readiness_waits=self._readiness_waits,
            ram_adopted_without_clone=self.ram_adopted_without_clone,
            bounded_pinned_staging_requested=(
                self.bounded_pinned_staging_requested
            ),
            bounded_pinned_staging_enabled=bounded is not None,
            pinned_slot_bytes=bounded.slot_bytes if bounded else 0,
            pinned_capacity_bytes=bounded.capacity_bytes if bounded else 0,
            pinned_occupied_bytes=bounded.occupied_bytes if bounded else 0,
            pinned_execution_key=bounded.execution_key if bounded else None,
            pinned_prefetch_key=bounded.prefetch_key if bounded else None,
            pageable_to_pinned_bytes=(
                bounded.pageable_to_pinned_bytes if bounded else 0
            ),
            pinned_to_device_bytes=self._pinned_to_device_bytes,
            pinned_staging_hits=bounded.staging_hits if bounded else 0,
            pinned_staging_misses=bounded.staging_misses if bounded else 0,
            pinned_reuse_stalls=bounded.reuse_stalls if bounded else 0,
            pinned_reuse_stall_ms=bounded.reuse_stall_ms if bounded else 0.0,
        )

    def _adopt_owned_bundle(
        self,
        record: ExpertRecord,
        source: TensorSource,
    ) -> TorchExpertBundle:
        if not isinstance(source, TorchExpertBundle):
            raise TypeError(
                "owned bundle adoption requires TorchExpertBundle values"
            )
        if source.key != record.key or source.content_id != record.content_id:
            raise ValueError(f"owned bundle identity mismatch for {record.key}")
        if not source.tensors:
            raise ValueError(f"owned bundle {record.key} cannot be empty")
        names: set[str] = set()
        for name, tensor in source.tensors:
            if not isinstance(name, str) or not name or name in names:
                raise ValueError(
                    f"owned bundle {record.key} has invalid tensor names"
                )
            names.add(name)
            if not isinstance(tensor, torch.Tensor):
                raise TypeError("owned bundle values must be torch.Tensor")
            if (
                tensor.device.type != "cpu"
                or tensor.is_meta
                or tensor.layout != torch.strided
                or not tensor.is_floating_point()
                or not tensor.is_contiguous()
                or tensor.requires_grad
                or tensor.grad_fn is not None
            ):
                raise ValueError(
                    f"owned bundle {record.key} tensors must be detached, contiguous "
                    "floating CPU tensors"
                )
        if source.byte_size != record.byte_size:
            raise ValueError(
                f"expert {record.key} tensor bytes {source.byte_size} "
                f"do not match inventory bytes {record.byte_size}"
            )
        return source

    def _normalize_bundle(
        self,
        record: ExpertRecord,
        source: TensorSource,
    ) -> TorchExpertBundle:
        if isinstance(source, torch.Tensor):
            items = (("weight", source),)
        elif isinstance(source, TorchExpertBundle):
            if source.key != record.key or source.content_id != record.content_id:
                raise ValueError(f"expert bundle identity mismatch for {record.key}")
            if not source.tensors:
                raise ValueError(f"expert bundle {record.key} cannot be empty")
            items = source.tensors
        elif isinstance(source, Mapping) and source:
            items = tuple(sorted(source.items()))
        else:
            raise TypeError(
                "each expert source must be a Tensor, TorchExpertBundle, "
                "or non-empty tensor mapping"
            )

        normalized: list[tuple[str, torch.Tensor]] = []
        for name, tensor in items:
            if not isinstance(name, str) or not name:
                raise ValueError("expert tensor names cannot be empty")
            if not isinstance(tensor, torch.Tensor):
                raise TypeError("expert bundle values must be torch.Tensor")
            cpu_tensor = tensor.detach().to(device="cpu").contiguous().clone()
            cpu_tensor.requires_grad_(False)
            normalized.append((name, cpu_tensor))

        bundle = TorchExpertBundle(
            key=record.key,
            content_id=record.content_id,
            tensors=tuple(normalized),
        )
        if bundle.byte_size != record.byte_size:
            raise ValueError(
                f"expert {record.key} tensor bytes {bundle.byte_size} "
                f"do not match inventory bytes {record.byte_size}"
            )
        return bundle

    def _copy_to_effective_device(
        self,
        key: ExpertKey,
        *,
        staging_role: StagingRole,
    ) -> TorchExpertBundle:
        source = self._ram[key]
        if self.effective_device.type == "cpu":
            copied = self._copy_execution_bundle(
                source,
                device=self.effective_device,
                non_blocking=False,
            )
        else:
            assert self._prefetch_stream is not None
            transfer_source = (
                self._bounded_pinned_staging.stage(staging_role, source)
                if self._bounded_pinned_staging is not None
                else source
            )
            with torch.cuda.stream(self._prefetch_stream):
                copied = self._copy_execution_bundle(
                    transfer_source,
                    device=self.effective_device,
                    non_blocking=(
                        self.pin_memory_enabled
                        or self._bounded_pinned_staging is not None
                    ),
                )
                event = torch.cuda.Event(blocking=False, interprocess=False)
                event.record(self._prefetch_stream)
                self._events[key] = event
            if self._bounded_pinned_staging is not None:
                self._bounded_pinned_staging.mark_in_flight(
                    staging_role,
                    key=key,
                    completion=event,
                )
                self._pinned_to_device_bytes += copied.byte_size
        self._copied_bytes += copied.byte_size
        return copied

    @staticmethod
    def _copy_execution_bundle(
        source: TorchExpertBundle,
        *,
        device: torch.device,
        non_blocking: bool,
    ) -> TorchExpertBundle:
        names = tuple(name for name, _ in source.tensors)
        canonical = (
            "gate_proj.weight",
            "up_proj.weight",
            "down_proj.weight",
        )
        if names != canonical:
            copied_items: list[tuple[str, torch.Tensor]] = []
            for name, tensor in source.tensors:
                destination = torch.empty_like(tensor, device=device)
                destination.copy_(tensor, non_blocking=non_blocking)
                copied_items.append((name, destination))
            return TorchExpertBundle(
                key=source.key,
                content_id=source.content_id,
                tensors=tuple(copied_items),
            )

        gate = source.tensor("gate_proj.weight")
        up = source.tensor("up_proj.weight")
        down = source.tensor("down_proj.weight")
        if (
            gate.ndim != 2
            or tuple(up.shape) != tuple(gate.shape)
            or gate.dtype != up.dtype
            or gate.device != up.device
        ):
            raise ValueError(f"expert {source.key} gate/up geometry cannot be packed")
        packed = torch.empty(
            (int(gate.shape[0]) * 2, int(gate.shape[1])),
            dtype=gate.dtype,
            device=device,
        )
        packed[: gate.shape[0]].copy_(gate, non_blocking=non_blocking)
        packed[gate.shape[0] :].copy_(up, non_blocking=non_blocking)
        copied_down = torch.empty_like(down, device=device)
        copied_down.copy_(down, non_blocking=non_blocking)
        copied = TorchExpertBundle(
            key=source.key,
            content_id=source.content_id,
            tensors=(
                ("gate_proj.weight", packed[: gate.shape[0]]),
                ("up_proj.weight", packed[gate.shape[0] :]),
                ("down_proj.weight", copied_down),
            ),
        )
        # This also verifies adjacency and storage bounds before the bundle is
        # admitted to the scheduler's byte-accounted device state.
        copied.packed_gate_up()
        return copied

    def _validate_record(self, record: ExpertRecord) -> None:
        authoritative = self.inventory.record(record.key)
        if authoritative != record:
            raise ValueError(f"expert record identity mismatch for {record.key}")
        bundle = self._ram[record.key]
        if bundle.content_id != record.content_id or bundle.byte_size != record.byte_size:
            raise ValueError(f"authoritative RAM bundle mismatch for {record.key}")

    def _bundle_without_wait(self, key: ExpertKey) -> TorchExpertBundle:
        for storage in (self._active, self._ephemeral, self._prefetch):
            bundle = storage.get(key)
            if bundle is not None:
                return bundle
        raise KeyError(f"expert {key} is not loaded on the execution device")

    def _wait_host(self, key: ExpertKey) -> None:
        event = self._events.get(key)
        if event is not None:
            event.synchronize()

    def _record_compute_stream(
        self,
        key: ExpertKey,
        bundle: TorchExpertBundle,
        stream: torch.cuda.Stream,
    ) -> None:
        # ``record_stream`` is the allocator-lifetime boundary for tensors
        # created on ``_prefetch_stream`` and later consumed elsewhere.
        for _, tensor in bundle.tensors:
            tensor.record_stream(stream)
        streams = self._compute_streams.setdefault(key, [])
        if not any(existing == stream for existing in streams):
            streams.append(stream)

    def _wait_for_compute(self, key: ExpertKey) -> None:
        streams = self._compute_streams.get(key)
        if not streams:
            return
        # Synchronize before removing the bookkeeping entry. If CUDA reports a
        # failure, the bundle remains owned and a later cleanup can fail closed
        # again instead of silently freeing storage with unknown consumers.
        for stream in streams:
            stream.synchronize()
        self._compute_streams.pop(key, None)

    def _drop_event_if_unused(self, key: ExpertKey) -> None:
        if key not in self._active and key not in self._prefetch and key not in self._ephemeral:
            self._events.pop(key, None)

    def _active_bytes(self) -> int:
        return sum(bundle.byte_size for bundle in self._active.values())

    def _prefetch_bytes(self) -> int:
        return sum(bundle.byte_size for bundle in self._prefetch.values())

    def _device_bytes(self) -> int:
        return (
            self._active_bytes()
            + self._prefetch_bytes()
            + sum(bundle.byte_size for bundle in self._ephemeral.values())
        )

    def _ensure_device_capacity(
        self,
        additional_bytes: int,
        *,
        operation: str,
    ) -> None:
        projected_bytes = self._device_bytes() + additional_bytes
        if projected_bytes > self.cache_config.capacity_bytes:
            raise MemoryError(
                f"physical {operation} would require {projected_bytes} bytes "
                f"but capacity is {self.cache_config.capacity_bytes} bytes"
            )

    @staticmethod
    def _select_device(
        requested: torch.device,
        *,
        allow_cpu_fallback: bool,
    ) -> tuple[torch.device, bool]:
        if requested.type == "cpu":
            return torch.device("cpu"), False
        index = requested.index
        cuda_available = torch.cuda.is_available()
        index_available = (
            cuda_available
            and (index is None or 0 <= index < torch.cuda.device_count())
        )
        if index_available:
            return requested, False
        if not allow_cpu_fallback:
            raise RuntimeError(f"requested CUDA device {requested} is unavailable")
        return torch.device("cpu"), True
