"""Predictive VRAM expert cache for RAM-backed MoE macro-stages.

The module is deliberately independent from the physical stage runner.  It
models the part of a future macro-stage that is easy to get subtly wrong:

* RAM is the authoritative store for several contiguous MoE layers;
* VRAM is split into an active expert cache and a logical prefetch buffer;
* predictions may prepare weights, but the target model router is always
  authoritative;
* a prediction miss loads the exact routed expert from RAM before execution;
* eviction combines decaying hotness with LRU ordering; and
* a break-even calculation decides whether PCIe misses are cheaper than the
  network waits removed by co-locating layers in one macro-stage.

No tensor backend is assumed.  The caller supplies ``ram_loader`` and performs
the actual tensor copy; this scheduler only decides which immutable expert
artifact must be present and records the exact fallback path.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable, Protocol, Sequence


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite(name: str, value: float, *, minimum: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and >= {minimum}") from exc
    if not math.isfinite(number) or number < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return number


@dataclass(frozen=True, order=True)
class ExpertKey:
    layer: int
    expert: int

    def __post_init__(self) -> None:
        if not _is_int(self.layer) or self.layer < 0:
            raise ValueError("layer must be a non-negative integer")
        if not _is_int(self.expert) or self.expert < 0:
            raise ValueError("expert must be a non-negative integer")


@dataclass(frozen=True)
class ExpertRecord:
    """Immutable identity and RAM footprint of one routed expert."""

    key: ExpertKey
    byte_size: int
    content_id: str

    def __post_init__(self) -> None:
        if not _is_int(self.byte_size) or self.byte_size < 1:
            raise ValueError("byte_size must be a positive integer")
        if not isinstance(self.content_id, str) or not self.content_id.strip():
            raise ValueError("content_id cannot be empty")


class UnknownExpertError(KeyError):
    """Raised when the authoritative router names an absent RAM artifact."""


class ExpertWeightBackend(Protocol):
    """Physical storage hooks driven by :class:`RamBackedExpertScheduler`."""

    enforces_device_capacity: bool

    def prefetch(self, record: ExpertRecord) -> None: ...

    def load_for_execution(self, record: ExpertRecord) -> None: ...

    def commit_loaded(self, key: ExpertKey, *, cached: bool) -> None: ...

    def promote(self, keys: Sequence[ExpertKey]) -> None: ...

    def evict(self, keys: Sequence[ExpertKey]) -> None: ...

    def discard_prefetch(self, keys: Sequence[ExpertKey]) -> None: ...

    def ensure_ready(self, keys: Sequence[ExpertKey]) -> None: ...

    def release_uncached(self, keys: Sequence[ExpertKey]) -> None: ...


class _CallbackWeightBackend:
    """Compatibility adapter for metadata-only callers and existing tests."""

    enforces_device_capacity = False

    def __init__(self, loader: Callable[[ExpertRecord], None]) -> None:
        self._loader = loader
        self._prefetched: set[ExpertKey] = set()

    def prefetch(self, record: ExpertRecord) -> None:
        self._loader(record)
        self._prefetched.add(record.key)

    def load_for_execution(self, record: ExpertRecord) -> None:
        if record.key in self._prefetched:
            self._prefetched.remove(record.key)
            return
        self._loader(record)

    def commit_loaded(self, key: ExpertKey, *, cached: bool) -> None:
        del key, cached

    def promote(self, keys: Sequence[ExpertKey]) -> None:
        self._prefetched.difference_update(keys)

    def evict(self, keys: Sequence[ExpertKey]) -> None:
        del keys

    def discard_prefetch(self, keys: Sequence[ExpertKey]) -> None:
        self._prefetched.difference_update(keys)

    def ensure_ready(self, keys: Sequence[ExpertKey]) -> None:
        del keys

    def release_uncached(self, keys: Sequence[ExpertKey]) -> None:
        del keys


class MacroStageExpertInventory:
    """Inventory for a contiguous RAM-backed range of MoE layers."""

    def __init__(self, records: Sequence[ExpertRecord]) -> None:
        if not records:
            raise ValueError("records cannot be empty")
        by_key: dict[ExpertKey, ExpertRecord] = {}
        for record in records:
            if not isinstance(record, ExpertRecord):
                raise TypeError("records must contain ExpertRecord values")
            if record.key in by_key:
                raise ValueError(f"duplicate expert record {record.key}")
            by_key[record.key] = record

        layers = tuple(sorted({key.layer for key in by_key}))
        expected_layers = tuple(range(layers[0], layers[-1] + 1))
        if layers != expected_layers:
            raise ValueError("macro-stage layers must be contiguous")

        self._records = by_key
        self._layers = layers
        self._keys_by_layer = {
            layer: tuple(sorted(key for key in by_key if key.layer == layer))
            for layer in layers
        }

    @classmethod
    def uniform(
        cls,
        *,
        layer_start: int,
        layer_end: int,
        num_experts: int,
        expert_bytes: int,
        content_prefix: str = "expert",
    ) -> "MacroStageExpertInventory":
        if not _is_int(layer_start) or layer_start < 0:
            raise ValueError("layer_start must be a non-negative integer")
        if not _is_int(layer_end) or layer_end <= layer_start:
            raise ValueError("layer_end must be greater than layer_start")
        if not _is_int(num_experts) or num_experts < 1:
            raise ValueError("num_experts must be a positive integer")
        if not _is_int(expert_bytes) or expert_bytes < 1:
            raise ValueError("expert_bytes must be a positive integer")
        if not isinstance(content_prefix, str) or not content_prefix.strip():
            raise ValueError("content_prefix cannot be empty")
        return cls(
            tuple(
                ExpertRecord(
                    key=ExpertKey(layer, expert),
                    byte_size=expert_bytes,
                    content_id=f"{content_prefix}:l{layer}:e{expert}",
                )
                for layer in range(layer_start, layer_end)
                for expert in range(num_experts)
            )
        )

    @property
    def layers(self) -> tuple[int, ...]:
        return self._layers

    @property
    def layer_start(self) -> int:
        return self._layers[0]

    @property
    def layer_end(self) -> int:
        return self._layers[-1] + 1

    @property
    def records(self) -> tuple[ExpertRecord, ...]:
        return tuple(self._records[key] for key in sorted(self._records))

    @property
    def total_bytes(self) -> int:
        return sum(record.byte_size for record in self._records.values())

    def keys_for_layer(self, layer: int) -> tuple[ExpertKey, ...]:
        try:
            return self._keys_by_layer[layer]
        except KeyError as exc:
            raise UnknownExpertError(f"layer {layer} is outside this macro-stage") from exc

    def record(self, key: ExpertKey) -> ExpertRecord:
        try:
            return self._records[key]
        except KeyError as exc:
            raise UnknownExpertError(f"expert {key} is absent from RAM inventory") from exc


@dataclass(frozen=True, kw_only=True)
class PredictiveCacheConfig:
    """Physical cache budgets and decimal PCIe payload bandwidth.

    ``pcie_bandwidth_gbytes_per_second`` is GB/s (10^9 bytes/s), never Gbit/s.
    The deliberately long name prevents an eightfold unit ambiguity at API
    boundaries.
    """

    capacity_bytes: int
    prefetch_reserve_bytes: int
    pcie_bandwidth_gbytes_per_second: float
    hotness_decay: float = 0.95
    min_prefetch_confidence: float = 0.0

    def __post_init__(self) -> None:
        if not _is_int(self.capacity_bytes) or self.capacity_bytes < 2:
            raise ValueError("capacity_bytes must be an integer >= 2")
        if (
            not _is_int(self.prefetch_reserve_bytes)
            or self.prefetch_reserve_bytes < 1
        ):
            raise ValueError("prefetch_reserve_bytes must be a positive integer")
        if self.prefetch_reserve_bytes * 2 > self.capacity_bytes:
            raise ValueError(
                "prefetch_reserve_bytes cannot exceed half of capacity_bytes"
            )
        bandwidth = _finite(
            "pcie_bandwidth_gbytes_per_second",
            self.pcie_bandwidth_gbytes_per_second,
            minimum=0.0,
        )
        if bandwidth == 0:
            raise ValueError("pcie_bandwidth_gbytes_per_second must be positive")
        decay = _finite("hotness_decay", self.hotness_decay, minimum=0.0)
        if not 0 < decay <= 1:
            raise ValueError("hotness_decay must be in (0, 1]")
        confidence = _finite(
            "min_prefetch_confidence",
            self.min_prefetch_confidence,
            minimum=0.0,
        )
        if confidence > 1:
            raise ValueError("min_prefetch_confidence must be in [0, 1]")
        object.__setattr__(self, "pcie_bandwidth_gbytes_per_second", bandwidth)
        object.__setattr__(self, "hotness_decay", decay)
        object.__setattr__(self, "min_prefetch_confidence", confidence)

    @property
    def active_capacity_bytes(self) -> int:
        return self.capacity_bytes - self.prefetch_reserve_bytes

    @property
    def pcie_bytes_per_ms(self) -> float:
        return self.pcie_bandwidth_gbytes_per_second * 1_000_000.0


@dataclass(frozen=True)
class PrefetchTicket:
    ticket_id: int
    key: ExpertKey
    submitted_at_ms: float
    deadline_ms: float
    confidence: float
    byte_size: int


@dataclass(frozen=True)
class PrefetchStageResult:
    staged: tuple[ExpertKey, ...]
    already_resident: tuple[ExpertKey, ...]
    expired: tuple[ExpertKey, ...]
    below_confidence: tuple[ExpertKey, ...]
    deferred: tuple[ExpertKey, ...]
    transferred_bytes: int
    completion_ms: float


@dataclass(frozen=True)
class PrefetchPromotion:
    promoted: tuple[ExpertKey, ...]
    evicted: tuple[ExpertKey, ...]
    active_bytes: int
    prefetch_bytes: int


@dataclass(frozen=True)
class RouteResolution:
    """Exact physical resolution of one authoritative MoE router decision."""

    layer: int
    authoritative_experts: tuple[ExpertKey, ...]
    predicted_experts: tuple[ExpertKey, ...]
    invalid_predictions: tuple[int, ...]
    cache_hits: tuple[ExpertKey, ...]
    prefetch_hits: tuple[ExpertKey, ...]
    ram_fallbacks: tuple[ExpertKey, ...]
    uncached_after_use: tuple[ExpertKey, ...]
    evicted: tuple[ExpertKey, ...]
    prediction_false_positives: tuple[ExpertKey, ...]
    prediction_false_negatives: tuple[ExpertKey, ...]
    fallback_bytes: int
    exact: bool = True


@dataclass(frozen=True)
class ExpertCacheSnapshot:
    active_keys: tuple[ExpertKey, ...]
    prefetch_keys: tuple[ExpertKey, ...]
    queued_prefetch_keys: tuple[ExpertKey, ...]
    active_bytes: int
    prefetch_bytes: int
    capacity_bytes: int
    cache_hits: int
    prefetch_hits: int
    ram_misses: int
    ram_fallback_bytes: int
    evictions: int


@dataclass
class _CacheEntry:
    record: ExpertRecord
    hotness: float
    hotness_tick: int
    last_access_tick: int


class RamBackedExpertScheduler:
    """Byte-bounded predictive expert scheduler with exact RAM fallback."""

    def __init__(
        self,
        inventory: MacroStageExpertInventory,
        config: PredictiveCacheConfig,
        *,
        ram_loader: Callable[[ExpertRecord], None] | None = None,
        weight_backend: ExpertWeightBackend | None = None,
    ) -> None:
        if not isinstance(inventory, MacroStageExpertInventory):
            raise TypeError("inventory must be MacroStageExpertInventory")
        if not isinstance(config, PredictiveCacheConfig):
            raise TypeError("config must be PredictiveCacheConfig")
        if (ram_loader is None) == (weight_backend is None):
            raise ValueError(
                "provide exactly one of ram_loader or weight_backend"
            )
        if ram_loader is not None and not callable(ram_loader):
            raise TypeError("ram_loader must be callable")
        if weight_backend is not None:
            for method in (
                "prefetch",
                "load_for_execution",
                "commit_loaded",
                "promote",
                "evict",
                "discard_prefetch",
                "ensure_ready",
                "release_uncached",
            ):
                if not callable(getattr(weight_backend, method, None)):
                    raise TypeError(f"weight_backend is missing {method}()")
            validator = getattr(weight_backend, "validate_cache_config", None)
            if callable(validator):
                validator(config)
        self.inventory = inventory
        self.config = config
        self._weight_backend = (
            weight_backend
            if weight_backend is not None
            else _CallbackWeightBackend(ram_loader)  # type: ignore[arg-type]
        )
        self._active: dict[ExpertKey, _CacheEntry] = {}
        self._prefetch: dict[ExpertKey, _CacheEntry] = {}
        self._tickets: dict[ExpertKey, PrefetchTicket] = {}
        self._ticket_sequence = 0
        self._tick = 0
        self._cache_hits = 0
        self._prefetch_hits = 0
        self._ram_misses = 0
        self._ram_fallback_bytes = 0
        self._evictions = 0

    def submit_prefetch(
        self,
        key: ExpertKey,
        *,
        deadline_ms: float,
        confidence: float,
        now_ms: float,
    ) -> PrefetchTicket:
        record = self.inventory.record(key)
        now = _finite("now_ms", now_ms)
        deadline = _finite("deadline_ms", deadline_ms)
        if deadline < now:
            raise ValueError("deadline_ms cannot precede now_ms")
        normalized_confidence = _finite("confidence", confidence)
        if normalized_confidence > 1:
            raise ValueError("confidence must be in [0, 1]")

        existing = self._tickets.get(key)
        if existing is not None:
            ticket = PrefetchTicket(
                ticket_id=existing.ticket_id,
                key=key,
                submitted_at_ms=min(existing.submitted_at_ms, now),
                deadline_ms=min(existing.deadline_ms, deadline),
                confidence=max(existing.confidence, normalized_confidence),
                byte_size=record.byte_size,
            )
        else:
            self._ticket_sequence += 1
            ticket = PrefetchTicket(
                ticket_id=self._ticket_sequence,
                key=key,
                submitted_at_ms=now,
                deadline_ms=deadline,
                confidence=normalized_confidence,
                byte_size=record.byte_size,
            )
        self._tickets[key] = ticket
        return ticket

    def stage_prefetch(
        self,
        *,
        now_ms: float,
        byte_budget: int | None = None,
    ) -> PrefetchStageResult:
        now = _finite("now_ms", now_ms)
        if byte_budget is None:
            transfer_budget = math.inf
        elif not _is_int(byte_budget) or byte_budget < 0:
            raise ValueError("byte_budget must be a non-negative integer")
        else:
            transfer_budget = float(byte_budget)

        staged: list[ExpertKey] = []
        already: list[ExpertKey] = []
        expired: list[ExpertKey] = []
        below_confidence: list[ExpertKey] = []
        transferred = 0
        cursor_ms = now
        remaining_buffer = (
            self.config.prefetch_reserve_bytes - self._prefetch_bytes()
        )

        tickets = sorted(
            self._tickets.values(),
            key=lambda ticket: (
                ticket.deadline_ms,
                -ticket.confidence,
                ticket.ticket_id,
            ),
        )
        for ticket in tickets:
            key = ticket.key
            if key in self._active or key in self._prefetch:
                already.append(key)
                self._tickets.pop(key, None)
                continue
            if ticket.confidence < self.config.min_prefetch_confidence:
                below_confidence.append(key)
                self._tickets.pop(key, None)
                continue
            if ticket.deadline_ms < now:
                expired.append(key)
                self._tickets.pop(key, None)
                continue
            if (
                ticket.byte_size > remaining_buffer
                or transferred + ticket.byte_size > transfer_budget
            ):
                continue

            completion = (
                cursor_ms
                + ticket.byte_size / self.config.pcie_bytes_per_ms
            )
            if completion > ticket.deadline_ms:
                expired.append(key)
                self._tickets.pop(key, None)
                continue

            record = self.inventory.record(key)
            self._weight_backend.prefetch(record)
            self._tick += 1
            self._prefetch[key] = _CacheEntry(
                record=record,
                hotness=max(ticket.confidence, 0.01),
                hotness_tick=self._tick,
                last_access_tick=self._tick,
            )
            staged.append(key)
            transferred += ticket.byte_size
            remaining_buffer -= ticket.byte_size
            cursor_ms = completion
            self._tickets.pop(key, None)

        deferred = tuple(
            ticket.key
            for ticket in sorted(
                self._tickets.values(),
                key=lambda ticket: (
                    ticket.deadline_ms,
                    -ticket.confidence,
                    ticket.ticket_id,
                ),
            )
        )
        return PrefetchStageResult(
            staged=tuple(staged),
            already_resident=tuple(already),
            expired=tuple(expired),
            below_confidence=tuple(below_confidence),
            deferred=deferred,
            transferred_bytes=transferred,
            completion_ms=cursor_ms,
        )

    def promote_prefetch(
        self,
        keys: Iterable[ExpertKey] | None = None,
        *,
        protected_keys: Iterable[ExpertKey] = (),
    ) -> PrefetchPromotion:
        if keys is None:
            selected = tuple(self._prefetch)
        else:
            selected = tuple(dict.fromkeys(keys))
            for key in selected:
                if key not in self._prefetch:
                    raise KeyError(f"expert {key} is not in the prefetch buffer")

        protected = set(selected)
        for key in protected_keys:
            self.inventory.record(key)
            protected.add(key)
        promoted: list[ExpertKey] = []
        evicted: list[ExpertKey] = []
        for key in sorted(
            selected,
            key=lambda candidate: (
                -self._effective_hotness(self._prefetch[candidate]),
                candidate,
            ),
        ):
            entry = self._prefetch[key]
            inserted, removed = self._insert_active(
                entry.record,
                initial_hotness=entry.hotness,
                protected=protected,
            )
            evicted.extend(removed)
            if inserted:
                if removed:
                    self._weight_backend.evict(removed)
                self._weight_backend.promote((key,))
                promoted.append(key)
                self._prefetch.pop(key, None)

        return PrefetchPromotion(
            promoted=tuple(promoted),
            evicted=tuple(evicted),
            active_bytes=self._active_bytes(),
            prefetch_bytes=self._prefetch_bytes(),
        )

    def discard_prefetch(self) -> tuple[ExpertKey, ...]:
        discarded = tuple(self._prefetch)
        if discarded:
            self._weight_backend.discard_prefetch(discarded)
        self._prefetch.clear()
        return discarded

    def resolve_route(
        self,
        *,
        layer: int,
        authoritative_expert_ids: Sequence[int],
        predicted_expert_ids: Sequence[int] = (),
    ) -> RouteResolution:
        if not _is_int(layer) or layer < 0:
            raise ValueError("layer must be a non-negative integer")
        self.inventory.keys_for_layer(layer)
        actual = self._authoritative_keys(layer, authoritative_expert_ids)
        predicted, invalid_predictions = self._prediction_keys(
            layer,
            predicted_expert_ids,
        )

        # ``load_for_execution`` materializes a real device bundle.  Reserve
        # room for the complete authoritative route before the first copy so
        # active + prefetch + ephemeral storage can never transiently exceed
        # the physical budget.  Predictions and inactive cache entries are
        # expendable; experts selected by the authoritative router are not.
        evicted = list(self._reserve_route_capacity(actual))

        staged_actual = tuple(key for key in actual if key in self._prefetch)
        promotion = (
            self.promote_prefetch(staged_actual, protected_keys=actual)
            if staged_actual
            else None
        )
        promoted = set(promotion.promoted if promotion is not None else ())
        evicted.extend(promotion.evicted if promotion is not None else ())

        cache_hits: list[ExpertKey] = []
        prefetch_hits: list[ExpertKey] = []
        fallbacks: list[ExpertKey] = []
        uncached: list[ExpertKey] = []
        fallback_bytes = 0
        protected = set(actual)

        for key in actual:
            if key in self._active:
                self._touch(key)
                if key in promoted:
                    prefetch_hits.append(key)
                    self._prefetch_hits += 1
                else:
                    cache_hits.append(key)
                    self._cache_hits += 1
                continue

            record = self.inventory.record(key)
            if key in self._prefetch:
                # The transfer already happened, but all authoritative experts
                # may not fit in the active cache together.  Consume the exact
                # prefetched artifact as an ephemeral execution bundle and
                # remove its logical prefetch entry.  It is a prefetch hit, not
                # a RAM/PCIe miss, and leaving the entry behind would make the
                # next route attempt to promote a non-existent physical copy.
                prefetched_entry = self._prefetch.pop(key)
                self._weight_backend.load_for_execution(record)
                inserted, removed = self._insert_active(
                    record,
                    initial_hotness=(
                        self._effective_hotness(prefetched_entry) + 1.0
                    ),
                    protected=protected,
                )
                evicted.extend(removed)
                if removed:
                    self._weight_backend.evict(removed)
                self._weight_backend.commit_loaded(key, cached=inserted)
                prefetch_hits.append(key)
                self._prefetch_hits += 1
                if not inserted:
                    uncached.append(key)
                continue

            # This callback is the exactness boundary.  A prediction never
            # substitutes another expert: the immutable routed artifact is
            # loaded from RAM, or its exception aborts the route resolution.
            self._weight_backend.load_for_execution(record)
            fallbacks.append(key)
            fallback_bytes += record.byte_size
            self._ram_misses += 1
            self._ram_fallback_bytes += record.byte_size
            inserted, removed = self._insert_active(
                record,
                initial_hotness=1.0,
                protected=protected,
            )
            evicted.extend(removed)
            if removed:
                self._weight_backend.evict(removed)
            self._weight_backend.commit_loaded(key, cached=inserted)
            if not inserted:
                uncached.append(key)

        self._weight_backend.ensure_ready(actual)

        actual_set = set(actual)
        predicted_set = set(predicted)
        return RouteResolution(
            layer=layer,
            authoritative_experts=actual,
            predicted_experts=predicted,
            invalid_predictions=invalid_predictions,
            cache_hits=tuple(cache_hits),
            prefetch_hits=tuple(prefetch_hits),
            ram_fallbacks=tuple(fallbacks),
            uncached_after_use=tuple(uncached),
            evicted=tuple(dict.fromkeys(evicted)),
            prediction_false_positives=tuple(sorted(predicted_set - actual_set)),
            prediction_false_negatives=tuple(sorted(actual_set - predicted_set)),
            fallback_bytes=fallback_bytes,
        )

    def release_route(self, resolution: RouteResolution) -> None:
        """Release exact fallback tensors that were too large to cache."""

        if not isinstance(resolution, RouteResolution):
            raise TypeError("resolution must be RouteResolution")
        self._weight_backend.release_uncached(resolution.uncached_after_use)

    def snapshot(self) -> ExpertCacheSnapshot:
        return ExpertCacheSnapshot(
            active_keys=tuple(sorted(self._active)),
            prefetch_keys=tuple(sorted(self._prefetch)),
            queued_prefetch_keys=tuple(sorted(self._tickets)),
            active_bytes=self._active_bytes(),
            prefetch_bytes=self._prefetch_bytes(),
            capacity_bytes=self.config.capacity_bytes,
            cache_hits=self._cache_hits,
            prefetch_hits=self._prefetch_hits,
            ram_misses=self._ram_misses,
            ram_fallback_bytes=self._ram_fallback_bytes,
            evictions=self._evictions,
        )

    def _authoritative_keys(
        self,
        layer: int,
        expert_ids: Sequence[int],
    ) -> tuple[ExpertKey, ...]:
        if not expert_ids:
            raise ValueError("authoritative_expert_ids cannot be empty")
        keys: list[ExpertKey] = []
        seen: set[int] = set()
        for expert in expert_ids:
            if not _is_int(expert) or expert < 0:
                raise ValueError("authoritative expert ids must be non-negative integers")
            if expert in seen:
                raise ValueError("authoritative expert ids must be distinct")
            seen.add(expert)
            key = ExpertKey(layer, expert)
            self.inventory.record(key)
            keys.append(key)
        return tuple(keys)

    def _prediction_keys(
        self,
        layer: int,
        expert_ids: Sequence[int],
    ) -> tuple[tuple[ExpertKey, ...], tuple[int, ...]]:
        keys: list[ExpertKey] = []
        invalid: list[int] = []
        seen: set[int] = set()
        available = {key.expert for key in self.inventory.keys_for_layer(layer)}
        for expert in expert_ids:
            if not _is_int(expert) or expert < 0 or expert not in available:
                if _is_int(expert):
                    invalid.append(int(expert))
                continue
            if expert not in seen:
                seen.add(expert)
                keys.append(ExpertKey(layer, expert))
        return tuple(keys), tuple(invalid)

    def _reserve_route_capacity(
        self,
        actual: Sequence[ExpertKey],
    ) -> tuple[ExpertKey, ...]:
        """Make physical room for an exact route before any fallback copy.

        The execution API exposes every authoritative bundle together, so the
        route itself must fit in the configured device budget.  When it does,
        speculative prefetch entries are discarded first and then inactive
        cache entries are evicted by the same hotness/LRU policy used for
        insertion.  Entries in ``actual`` are protected throughout.
        """

        if not bool(
            getattr(self._weight_backend, "enforces_device_capacity", False)
        ):
            return ()

        protected = set(actual)
        route_bytes = sum(self.inventory.record(key).byte_size for key in actual)
        if route_bytes > self.config.capacity_bytes:
            raise MemoryError(
                "authoritative expert route requires "
                f"{route_bytes} bytes but physical capacity is "
                f"{self.config.capacity_bytes} bytes"
            )

        missing_bytes = sum(
            self.inventory.record(key).byte_size
            for key in actual
            if key not in self._active and key not in self._prefetch
        )
        bytes_to_free = max(
            0,
            self._active_bytes()
            + self._prefetch_bytes()
            + missing_bytes
            - self.config.capacity_bytes,
        )
        if bytes_to_free == 0:
            return ()

        freed_bytes = 0
        discarded_prefetch: list[ExpertKey] = []
        prefetch_candidates = sorted(
            (key for key in self._prefetch if key not in protected),
            key=lambda key: (
                self._effective_hotness(self._prefetch[key]),
                self._prefetch[key].last_access_tick,
                key,
            ),
        )
        for key in prefetch_candidates:
            if freed_bytes >= bytes_to_free:
                break
            discarded_prefetch.append(key)
            freed_bytes += self._prefetch[key].record.byte_size
        if discarded_prefetch:
            self._weight_backend.discard_prefetch(discarded_prefetch)
            for key in discarded_prefetch:
                self._prefetch.pop(key)

        evicted: list[ExpertKey] = []
        active_candidates = sorted(
            (key for key in self._active if key not in protected),
            key=lambda key: (
                self._effective_hotness(self._active[key]),
                self._active[key].last_access_tick,
                key,
            ),
        )
        for key in active_candidates:
            if freed_bytes >= bytes_to_free:
                break
            evicted.append(key)
            freed_bytes += self._active[key].record.byte_size

        if freed_bytes < bytes_to_free:
            # ``route_bytes <= capacity`` proves that non-authoritative
            # allocations are sufficient to satisfy this reservation.  Keep a
            # defensive guard in case a future storage state breaks that
            # invariant.
            raise RuntimeError("unable to reserve physical capacity for route")
        if evicted:
            self._weight_backend.evict(evicted)
            for key in evicted:
                self._active.pop(key)
                self._evictions += 1
        return tuple(evicted)

    def _insert_active(
        self,
        record: ExpertRecord,
        *,
        initial_hotness: float,
        protected: set[ExpertKey],
    ) -> tuple[bool, tuple[ExpertKey, ...]]:
        key = record.key
        if key in self._active:
            self._touch(key)
            return True, ()
        capacity = self.config.active_capacity_bytes
        if record.byte_size > capacity:
            return False, ()

        bytes_to_free = max(
            0,
            self._active_bytes() + record.byte_size - capacity,
        )
        victims: list[ExpertKey] = []
        freed_bytes = 0
        candidates = sorted(
            (
                candidate
                for candidate in self._active
                if candidate not in protected
            ),
            key=lambda candidate: (
                self._effective_hotness(self._active[candidate]),
                self._active[candidate].last_access_tick,
                candidate,
            ),
        )
        for victim in candidates:
            if freed_bytes >= bytes_to_free:
                break
            victims.append(victim)
            freed_bytes += self._active[victim].record.byte_size
        if freed_bytes < bytes_to_free:
            return False, ()

        for victim in victims:
            self._active.pop(victim)
            self._evictions += 1

        self._tick += 1
        self._active[key] = _CacheEntry(
            record=record,
            hotness=max(initial_hotness, 0.01),
            hotness_tick=self._tick,
            last_access_tick=self._tick,
        )
        return True, tuple(victims)

    def _touch(self, key: ExpertKey) -> None:
        self._tick += 1
        entry = self._active[key]
        entry.hotness = self._effective_hotness(entry) + 1.0
        entry.hotness_tick = self._tick
        entry.last_access_tick = self._tick

    def _effective_hotness(self, entry: _CacheEntry) -> float:
        age = max(0, self._tick - entry.hotness_tick)
        return entry.hotness * (self.config.hotness_decay**age)

    def _active_bytes(self) -> int:
        return sum(entry.record.byte_size for entry in self._active.values())

    def _prefetch_bytes(self) -> int:
        return sum(entry.record.byte_size for entry in self._prefetch.values())


@dataclass(frozen=True)
class RamPcieBreakEven:
    layers: int
    active_expert_bytes: int
    expected_miss_bytes: float
    raw_pcie_transfer_ms: float
    hidden_by_prefetch_ms: float
    effective_pcie_stall_ms: float
    network_wait_saved_ms: float
    margin_ms: float
    required_cache_hit_rate: float
    beneficial: bool


def estimate_ram_pcie_break_even(
    *,
    layers: int,
    experts_per_token: int,
    expert_bytes: int,
    cache_hit_rate: float,
    pcie_bandwidth_gbytes_per_second: float,
    rtt_ms: float,
    round_trips_avoided: int = 0,
    one_way_hops_avoided: int | None = None,
    prefetch_overlap_ms: float = 0.0,
) -> RamPcieBreakEven:
    """Compare expert RAM misses with network waits removed by a macro-stage.

    ``one_way_hops_avoided`` models a persistent streaming pipeline, where an
    internal layer boundary costs roughly half an RTT.  ``round_trips_avoided``
    models RPC or horizontal expert-parallel collectives.  Both may be present.
    Attention/router/shared weights are assumed pinned; only routed expert
    bytes participate in the PCIe miss calculation.  PCIe bandwidth is decimal
    GB/s (10^9 bytes/s), expressed by the unambiguous
    ``pcie_bandwidth_gbytes_per_second`` parameter.
    """

    for name, value in (
        ("layers", layers),
        ("experts_per_token", experts_per_token),
        ("expert_bytes", expert_bytes),
        ("round_trips_avoided", round_trips_avoided),
    ):
        minimum = 1 if name in {"layers", "experts_per_token", "expert_bytes"} else 0
        if not _is_int(value) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if one_way_hops_avoided is None:
        one_way_hops = max(0, layers - 1)
    elif not _is_int(one_way_hops_avoided) or one_way_hops_avoided < 0:
        raise ValueError("one_way_hops_avoided must be a non-negative integer")
    else:
        one_way_hops = one_way_hops_avoided

    hit_rate = _finite("cache_hit_rate", cache_hit_rate)
    if hit_rate > 1:
        raise ValueError("cache_hit_rate must be in [0, 1]")
    bandwidth = _finite(
        "pcie_bandwidth_gbytes_per_second",
        pcie_bandwidth_gbytes_per_second,
    )
    if bandwidth == 0:
        raise ValueError("pcie_bandwidth_gbytes_per_second must be positive")
    rtt = _finite("rtt_ms", rtt_ms)
    overlap = _finite("prefetch_overlap_ms", prefetch_overlap_ms)

    active_bytes = layers * experts_per_token * expert_bytes
    miss_bytes = active_bytes * (1.0 - hit_rate)
    bytes_per_ms = bandwidth * 1_000_000.0
    raw_transfer_ms = miss_bytes / bytes_per_ms
    hidden_ms = min(raw_transfer_ms, overlap)
    effective_stall_ms = raw_transfer_ms - hidden_ms
    network_saved_ms = (
        round_trips_avoided * rtt + one_way_hops * rtt / 2.0
    )
    margin_ms = network_saved_ms - effective_stall_ms

    tolerable_raw_transfer_ms = network_saved_ms + overlap
    required_hit_rate = 1.0 - (
        tolerable_raw_transfer_ms * bytes_per_ms / active_bytes
    )
    required_hit_rate = min(1.0, max(0.0, required_hit_rate))
    return RamPcieBreakEven(
        layers=layers,
        active_expert_bytes=active_bytes,
        expected_miss_bytes=miss_bytes,
        raw_pcie_transfer_ms=raw_transfer_ms,
        hidden_by_prefetch_ms=hidden_ms,
        effective_pcie_stall_ms=effective_stall_ms,
        network_wait_saved_ms=network_saved_ms,
        margin_ms=margin_ms,
        required_cache_hit_rate=required_hit_rate,
        beneficial=effective_stall_ms < network_saved_ms,
    )
