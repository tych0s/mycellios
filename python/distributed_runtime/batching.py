"""Fair, compatibility-safe adaptive batching primitives.

The current wire protocol carries one request id per frame, so this module does
not pretend that admission batching is already one physical model forward.  It
provides the single-owner queue needed by a future multi-request frame/runner:
only work with an identical compatibility key can share a batch, and at most
one item per request is selected in a scheduling round.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Hashable
from dataclasses import dataclass
import math
import time
from typing import Generic, TypeVar


PayloadT = TypeVar("PayloadT")


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite_non_negative(name: str, value: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and non-negative") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


@dataclass(frozen=True)
class InferenceBatchKey:
    """Strict default key for token-exact physical inference batching.

    Equal keys prove that model/stage/backend, tensor shape and KV position are
    equal.  Backends with correct ragged-attention support may supply their own
    less restrictive hashable key instead.
    """

    model_id: str
    stage_id: str
    input_tokens: int
    cache_tokens: int
    hidden_size: int
    dtype: str
    codec: str
    backend: str

    def __post_init__(self) -> None:
        for name in ("model_id", "stage_id", "dtype", "codec", "backend"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} cannot be empty")
        for name, minimum in (
            ("input_tokens", 1),
            ("cache_tokens", 0),
            ("hidden_size", 1),
        ):
            value = getattr(self, name)
            if not _is_int(value) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class FairBatchConfig:
    max_batch_size: int = 8
    initial_batch_size: int = 2
    target_batch_latency_ms: float = 50.0
    phase_weights: tuple[tuple[str, int], ...] = (
        ("decode", 4),
        ("verify", 2),
        ("prefill", 1),
    )
    phase_max_wait_ms: tuple[tuple[str, float], ...] = (
        ("decode", 0.0),
        ("verify", 0.0),
        ("prefill", 2.0),
    )
    starvation_ms: float = 20.0
    ewma_alpha: float = 0.25
    growth_interval: int = 3
    growth_headroom_ratio: float = 0.80

    def __post_init__(self) -> None:
        if not _is_int(self.max_batch_size) or self.max_batch_size < 1:
            raise ValueError("max_batch_size must be a positive integer")
        if (
            not _is_int(self.initial_batch_size)
            or not 1 <= self.initial_batch_size <= self.max_batch_size
        ):
            raise ValueError(
                "initial_batch_size must be between 1 and max_batch_size"
            )
        target = _finite_non_negative(
            "target_batch_latency_ms", self.target_batch_latency_ms
        )
        if target == 0:
            raise ValueError("target_batch_latency_ms must be positive")
        starvation = _finite_non_negative("starvation_ms", self.starvation_ms)
        if starvation == 0:
            raise ValueError("starvation_ms must be positive")
        alpha = _finite_non_negative("ewma_alpha", self.ewma_alpha)
        if not 0 < alpha <= 1:
            raise ValueError("ewma_alpha must be in (0, 1]")
        headroom = _finite_non_negative(
            "growth_headroom_ratio", self.growth_headroom_ratio
        )
        if not 0 < headroom < 1:
            raise ValueError("growth_headroom_ratio must be in (0, 1)")
        if not _is_int(self.growth_interval) or self.growth_interval < 1:
            raise ValueError("growth_interval must be a positive integer")

        weighted_phases: list[str] = []
        weights_seen: set[str] = set()
        for phase, weight in self.phase_weights:
            if not isinstance(phase, str) or not phase.strip():
                raise ValueError("phase names cannot be empty")
            if phase in weights_seen:
                raise ValueError("phase_weights cannot contain duplicate phases")
            if not _is_int(weight) or weight < 1:
                raise ValueError("phase weights must be positive integers")
            weights_seen.add(phase)
            weighted_phases.append(phase)
        if not weighted_phases:
            raise ValueError("phase_weights cannot be empty")

        waits_seen: set[str] = set()
        normalized_waits: list[tuple[str, float]] = []
        for phase, wait_ms in self.phase_max_wait_ms:
            if phase in waits_seen:
                raise ValueError("phase_max_wait_ms cannot contain duplicate phases")
            waits_seen.add(phase)
            normalized_waits.append(
                (phase, _finite_non_negative(f"wait for phase {phase}", wait_ms))
            )
        if waits_seen != weights_seen:
            raise ValueError(
                "phase_max_wait_ms must define exactly the phases in phase_weights"
            )

        object.__setattr__(self, "target_batch_latency_ms", target)
        object.__setattr__(self, "starvation_ms", starvation)
        object.__setattr__(self, "ewma_alpha", alpha)
        object.__setattr__(self, "growth_headroom_ratio", headroom)
        object.__setattr__(self, "phase_max_wait_ms", tuple(normalized_waits))


@dataclass(frozen=True)
class BatchWork(Generic[PayloadT]):
    request_id: int
    phase: str
    compatibility_key: Hashable
    token_count: int
    payload: PayloadT
    enqueued_at: float


@dataclass(frozen=True)
class BatchSelection(Generic[PayloadT]):
    items: tuple[BatchWork[PayloadT], ...]
    selected_at: float
    adaptive_limit: int
    starved: bool

    @property
    def phase(self) -> str:
        return self.items[0].phase

    @property
    def compatibility_key(self) -> Hashable:
        return self.items[0].compatibility_key

    @property
    def payloads(self) -> tuple[PayloadT, ...]:
        return tuple(item.payload for item in self.items)

    @property
    def max_wait_ms(self) -> float:
        return max(
            0.0,
            max((self.selected_at - item.enqueued_at) * 1_000 for item in self.items),
        )


@dataclass(frozen=True)
class BatchProfileStats:
    phase: str
    compatibility_key: Hashable
    adaptive_limit: int
    observations: int
    ewma_latency_ms: float | None
    ewma_bytes: float | None
    target_violations: int


@dataclass(frozen=True)
class FairBatchStats:
    queued_items: int
    enqueued_items: int
    dequeued_items: int
    cancelled_items: int
    batches: int
    max_queue_depth: int
    average_wait_ms: float
    maximum_wait_ms: float
    average_batch_size: float
    fill_ratio: float
    starvation_selections: int
    transferred_bytes: int
    phase_depths: tuple[tuple[str, int], ...]
    phase_batches: tuple[tuple[str, int], ...]
    profiles: tuple[BatchProfileStats, ...]


@dataclass
class _BatchProfile:
    adaptive_limit: int
    observations: int = 0
    ewma_latency_ms: float | None = None
    ewma_bytes: float | None = None
    target_violations: int = 0
    consecutive_headroom: int = 0


class _PhaseQueue(Generic[PayloadT]):
    def __init__(self) -> None:
        self.flows: dict[int, deque[BatchWork[PayloadT]]] = {}
        self.rotation: deque[int] = deque()
        self.size = 0

    def push(self, work: BatchWork[PayloadT]) -> None:
        flow = self.flows.get(work.request_id)
        if flow is None:
            flow = deque()
            self.flows[work.request_id] = flow
            self.rotation.append(work.request_id)
        flow.append(work)
        self.size += 1

    def head(self, request_id: int) -> BatchWork[PayloadT]:
        return self.flows[request_id][0]

    def round_robin_head(self) -> BatchWork[PayloadT]:
        return self.head(self.rotation[0])

    def oldest_head(self) -> BatchWork[PayloadT]:
        return min((flow[0] for flow in self.flows.values()), key=lambda item: item.enqueued_at)

    def pop_request(self, request_id: int) -> BatchWork[PayloadT]:
        flow = self.flows[request_id]
        work = flow.popleft()
        self.size -= 1
        self.rotation.remove(request_id)
        if flow:
            self.rotation.append(request_id)
        else:
            del self.flows[request_id]
        return work

    def cancel_request(self, request_id: int) -> tuple[BatchWork[PayloadT], ...]:
        flow = self.flows.pop(request_id, None)
        if flow is None:
            return ()
        self.rotation.remove(request_id)
        removed = tuple(flow)
        self.size -= len(removed)
        return removed


class FairAdaptiveBatchQueue(Generic[PayloadT]):
    """Single-owner weighted-fair queue with exact compatibility grouping."""

    def __init__(self, config: FairBatchConfig | None = None) -> None:
        self.config = config or FairBatchConfig()
        self._phase_queues = {
            phase: _PhaseQueue[PayloadT]() for phase, _ in self.config.phase_weights
        }
        self._max_wait_by_phase = dict(self.config.phase_max_wait_ms)
        self._phase_cycle = tuple(
            phase
            for phase, weight in self.config.phase_weights
            for _ in range(weight)
        )
        self._phase_cursor = 0
        self._profiles: dict[tuple[str, Hashable], _BatchProfile] = {}
        self._enqueued_items = 0
        self._dequeued_items = 0
        self._cancelled_items = 0
        self._batches = 0
        self._max_queue_depth = 0
        self._total_wait_ms = 0.0
        self._maximum_wait_ms = 0.0
        self._starvation_selections = 0
        self._transferred_bytes = 0
        self._phase_batches = {phase: 0 for phase in self._phase_queues}

    def __len__(self) -> int:
        return sum(queue.size for queue in self._phase_queues.values())

    @property
    def empty(self) -> bool:
        return len(self) == 0

    def enqueue(
        self,
        payload: PayloadT,
        *,
        request_id: int,
        phase: str,
        compatibility_key: Hashable,
        token_count: int = 1,
        now: float | None = None,
    ) -> BatchWork[PayloadT]:
        if not _is_int(request_id) or request_id < 0:
            raise ValueError("request_id must be a non-negative integer")
        if phase not in self._phase_queues:
            raise ValueError(f"unknown batch phase {phase!r}")
        try:
            hash(compatibility_key)
        except TypeError as exc:
            raise ValueError("compatibility_key must be hashable") from exc
        if not _is_int(token_count) or token_count < 1:
            raise ValueError("token_count must be a positive integer")
        enqueued_at = time.monotonic() if now is None else _finite_non_negative("now", now)
        work = BatchWork(
            request_id=request_id,
            phase=phase,
            compatibility_key=compatibility_key,
            token_count=token_count,
            payload=payload,
            enqueued_at=enqueued_at,
        )
        self._phase_queues[phase].push(work)
        self._enqueued_items += 1
        self._max_queue_depth = max(self._max_queue_depth, len(self))
        return work

    def cancel_request(self, request_id: int) -> tuple[BatchWork[PayloadT], ...]:
        if not _is_int(request_id) or request_id < 0:
            raise ValueError("request_id must be a non-negative integer")
        removed = tuple(
            work
            for phase_queue in self._phase_queues.values()
            for work in phase_queue.cancel_request(request_id)
        )
        self._cancelled_items += len(removed)
        return tuple(sorted(removed, key=lambda work: work.enqueued_at))

    def next_ready_in_ms(self, *, now: float | None = None) -> float | None:
        if self.empty:
            return None
        selected_at = time.monotonic() if now is None else _finite_non_negative("now", now)
        if any(self._phase_ready(phase, selected_at) for phase in self._phase_queues):
            return 0.0
        waits = []
        for phase, phase_queue in self._phase_queues.items():
            if phase_queue.size == 0:
                continue
            age_ms = max(0.0, (selected_at - phase_queue.oldest_head().enqueued_at) * 1_000)
            waits.append(max(0.0, self._max_wait_by_phase[phase] - age_ms))
            waits.append(max(0.0, float(self.config.starvation_ms) - age_ms))
        return min(waits)

    def pop_batch(
        self,
        *,
        now: float | None = None,
        force: bool = False,
    ) -> BatchSelection[PayloadT] | None:
        if self.empty:
            return None
        selected_at = time.monotonic() if now is None else _finite_non_negative("now", now)
        oldest = self._oldest_head()
        starved = (
            (selected_at - oldest.enqueued_at) * 1_000
            >= float(self.config.starvation_ms)
        )
        if starved:
            phase = oldest.phase
            anchor_request = oldest.request_id
        else:
            ready_phases = {
                phase
                for phase in self._phase_queues
                if self._phase_ready(phase, selected_at)
            }
            phase = self._choose_phase(ready_phases)
            if phase is None and force:
                phase = self._choose_phase(
                    {
                        name
                        for name, phase_queue in self._phase_queues.items()
                        if phase_queue.size
                    }
                )
            if phase is None:
                return None
            anchor_request = self._phase_queues[phase].rotation[0]

        phase_queue = self._phase_queues[phase]
        anchor = phase_queue.head(anchor_request)
        profile = self._profile(phase, anchor.compatibility_key)
        selected = [phase_queue.pop_request(anchor_request)]
        selected_requests = {anchor_request}
        for request_id in tuple(phase_queue.rotation):
            if len(selected) >= profile.adaptive_limit:
                break
            if request_id in selected_requests:
                continue
            candidate = phase_queue.head(request_id)
            if candidate.compatibility_key != anchor.compatibility_key:
                continue
            selected.append(phase_queue.pop_request(request_id))
            selected_requests.add(request_id)

        waits = [max(0.0, (selected_at - item.enqueued_at) * 1_000) for item in selected]
        self._dequeued_items += len(selected)
        self._batches += 1
        self._phase_batches[phase] += 1
        self._total_wait_ms += sum(waits)
        self._maximum_wait_ms = max(self._maximum_wait_ms, max(waits))
        if starved:
            self._starvation_selections += 1
        return BatchSelection(
            items=tuple(selected),
            selected_at=selected_at,
            adaptive_limit=profile.adaptive_limit,
            starved=starved,
        )

    def record_batch(
        self,
        selection: BatchSelection[PayloadT],
        *,
        latency_ms: float,
        transferred_bytes: int = 0,
    ) -> None:
        if not selection.items:
            raise ValueError("selection cannot be empty")
        phases = {item.phase for item in selection.items}
        keys = {item.compatibility_key for item in selection.items}
        request_ids = {item.request_id for item in selection.items}
        if len(phases) != 1 or len(keys) != 1:
            raise ValueError("selection contains incompatible work")
        if len(request_ids) != len(selection.items):
            raise ValueError("a physical batch cannot repeat a request id")
        latency = _finite_non_negative("latency_ms", latency_ms)
        if latency == 0:
            raise ValueError("latency_ms must be positive")
        if not _is_int(transferred_bytes) or transferred_bytes < 0:
            raise ValueError("transferred_bytes must be a non-negative integer")

        profile = self._profile(selection.phase, selection.compatibility_key)
        alpha = float(self.config.ewma_alpha)
        profile.observations += 1
        profile.ewma_latency_ms = self._ewma(profile.ewma_latency_ms, latency, alpha)
        profile.ewma_bytes = self._ewma(
            profile.ewma_bytes, float(transferred_bytes), alpha
        )
        self._transferred_bytes += transferred_bytes

        target = float(self.config.target_batch_latency_ms)
        if latency > target:
            profile.target_violations += 1
            profile.consecutive_headroom = 0
            profile.adaptive_limit = max(1, profile.adaptive_limit // 2)
        elif (
            latency <= target * float(self.config.growth_headroom_ratio)
            and len(selection.items) >= selection.adaptive_limit
        ):
            profile.consecutive_headroom += 1
            if profile.consecutive_headroom >= int(self.config.growth_interval):
                profile.adaptive_limit = min(
                    int(self.config.max_batch_size), profile.adaptive_limit + 1
                )
                profile.consecutive_headroom = 0
        else:
            profile.consecutive_headroom = 0

    def stats(self) -> FairBatchStats:
        average_wait = (
            self._total_wait_ms / self._dequeued_items if self._dequeued_items else 0.0
        )
        average_batch = self._dequeued_items / self._batches if self._batches else 0.0
        fill_ratio = (
            self._dequeued_items / (self._batches * int(self.config.max_batch_size))
            if self._batches
            else 0.0
        )
        profiles = tuple(
            BatchProfileStats(
                phase=phase,
                compatibility_key=key,
                adaptive_limit=profile.adaptive_limit,
                observations=profile.observations,
                ewma_latency_ms=profile.ewma_latency_ms,
                ewma_bytes=profile.ewma_bytes,
                target_violations=profile.target_violations,
            )
            for (phase, key), profile in self._profiles.items()
        )
        return FairBatchStats(
            queued_items=len(self),
            enqueued_items=self._enqueued_items,
            dequeued_items=self._dequeued_items,
            cancelled_items=self._cancelled_items,
            batches=self._batches,
            max_queue_depth=self._max_queue_depth,
            average_wait_ms=average_wait,
            maximum_wait_ms=self._maximum_wait_ms,
            average_batch_size=average_batch,
            fill_ratio=fill_ratio,
            starvation_selections=self._starvation_selections,
            transferred_bytes=self._transferred_bytes,
            phase_depths=tuple(
                (phase, phase_queue.size)
                for phase, phase_queue in self._phase_queues.items()
            ),
            phase_batches=tuple(self._phase_batches.items()),
            profiles=profiles,
        )

    def _phase_ready(self, phase: str, now: float) -> bool:
        phase_queue = self._phase_queues[phase]
        if phase_queue.size == 0:
            return False
        oldest_age_ms = max(
            0.0, (now - phase_queue.oldest_head().enqueued_at) * 1_000
        )
        if oldest_age_ms >= self._max_wait_by_phase[phase]:
            return True
        anchor = phase_queue.round_robin_head()
        profile = self._profile(phase, anchor.compatibility_key)
        compatible_requests = sum(
            1
            for request_id in phase_queue.rotation
            if phase_queue.head(request_id).compatibility_key
            == anchor.compatibility_key
        )
        return compatible_requests >= profile.adaptive_limit

    def _choose_phase(self, allowed: set[str]) -> str | None:
        if not allowed:
            return None
        for offset in range(len(self._phase_cycle)):
            index = (self._phase_cursor + offset) % len(self._phase_cycle)
            phase = self._phase_cycle[index]
            if phase in allowed:
                self._phase_cursor = (index + 1) % len(self._phase_cycle)
                return phase
        return None

    def _oldest_head(self) -> BatchWork[PayloadT]:
        return min(
            (
                phase_queue.oldest_head()
                for phase_queue in self._phase_queues.values()
                if phase_queue.size
            ),
            key=lambda item: item.enqueued_at,
        )

    def _profile(self, phase: str, key: Hashable) -> _BatchProfile:
        return self._profiles.setdefault(
            (phase, key),
            _BatchProfile(adaptive_limit=int(self.config.initial_batch_size)),
        )

    @staticmethod
    def _ewma(previous: float | None, current: float, alpha: float) -> float:
        if previous is None:
            return current
        return alpha * current + (1.0 - alpha) * previous


__all__ = [
    "BatchProfileStats",
    "BatchSelection",
    "BatchWork",
    "FairAdaptiveBatchQueue",
    "FairBatchConfig",
    "FairBatchStats",
    "InferenceBatchKey",
]
