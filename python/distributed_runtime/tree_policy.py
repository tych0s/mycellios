"""Observation-driven policy for exact RTT-aware sparse tree waves.

This module deliberately owns no sockets, tensors, or scheduler state.  It
compares measured classic and tree cost per emitted token, applies the sealed
physical limits for the next wave, and only then recommends a tree shape.

Tree execution is never an implicit experiment.  Callers that want to collect
the minimum evidence for a shape must explicitly request :meth:`next_probe`
and later record (or cancel) that probe.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Integral, Real
from typing import Any


MAX_POLICY_COUNT = (1 << 63) - 1


def _count(name: str, value: object, *, minimum: int = 0) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    normalized = int(value)
    if normalized < minimum or normalized > MAX_POLICY_COUNT:
        raise ValueError(
            f"{name} must be an integer between {minimum} and {MAX_POLICY_COUNT}"
        )
    return normalized


def _increment_count(name: str, current: int) -> int:
    """Return one sealed increment without mutating the caller's state."""

    normalized = _count(name, current)
    if normalized == MAX_POLICY_COUNT:
        raise ValueError(f"{name} exceeds the sealed policy range")
    return normalized + 1


def _measurement(
    name: str,
    value: object,
    *,
    minimum: float = 0.0,
    maximum: float | None = None,
) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    if maximum is not None and normalized > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return normalized


@dataclass(frozen=True, slots=True)
class TreeWaveShape:
    """One sealed sparse-tree topology, independent of chat length.

    ``total_candidate_tokens`` is the amount of candidate work across all
    leaves.  It must contain at least one token per leaf and the deepest leaf,
    and cannot exceed a dense ``width * depth`` tree wave.

    KV bytes deliberately do not belong to this identity: the same topology
    costs more memory as a conversation grows.  A caller supplies that
    per-decision value with :class:`TreeWaveCandidate`, while observations stay
    attached to the stable topology.
    """

    width: int
    depth: int
    total_candidate_tokens: int

    def __post_init__(self) -> None:
        width = _count("width", self.width, minimum=1)
        depth = _count("depth", self.depth, minimum=1)
        candidates = _count(
            "total_candidate_tokens", self.total_candidate_tokens, minimum=1
        )
        if candidates < max(width, depth):
            raise ValueError(
                "total_candidate_tokens must cover every leaf and the deepest path"
            )
        if candidates > width * depth:
            raise ValueError("total_candidate_tokens cannot exceed width * depth")
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "depth", depth)
        object.__setattr__(self, "total_candidate_tokens", candidates)


@dataclass(frozen=True, slots=True)
class TreeWaveCandidate:
    """A topology plus its conservative KV projection for this decision."""

    shape: TreeWaveShape
    projected_kv_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.shape, TreeWaveShape):
            raise ValueError("shape must be a TreeWaveShape")
        object.__setattr__(
            self,
            "projected_kv_bytes",
            _count("projected_kv_bytes", self.projected_kv_bytes),
        )


@dataclass(frozen=True, slots=True)
class ClassicObservation:
    """One measured classic wave, normalized by ``emitted_tokens``."""

    latency_seconds: float
    transferred_bytes: int
    emitted_tokens: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "latency_seconds",
            _measurement("latency_seconds", self.latency_seconds),
        )
        object.__setattr__(
            self,
            "transferred_bytes",
            _count("transferred_bytes", self.transferred_bytes),
        )
        object.__setattr__(
            self,
            "emitted_tokens",
            _count("emitted_tokens", self.emitted_tokens, minimum=1),
        )


@dataclass(frozen=True, slots=True)
class TreeWaveObservation:
    """One completed exact tree wave for a specific sealed shape."""

    shape: TreeWaveShape
    latency_seconds: float
    transferred_bytes: int
    emitted_tokens: int
    accepted_tokens: int

    def __post_init__(self) -> None:
        if not isinstance(self.shape, TreeWaveShape):
            raise ValueError("shape must be a TreeWaveShape")
        latency = _measurement("latency_seconds", self.latency_seconds)
        byte_count = _count("transferred_bytes", self.transferred_bytes)
        emitted = _count("emitted_tokens", self.emitted_tokens, minimum=1)
        accepted = _count("accepted_tokens", self.accepted_tokens)
        if accepted > self.shape.depth:
            raise ValueError("accepted_tokens cannot exceed shape.depth")
        if emitted != accepted + 1:
            raise ValueError(
                "exact tree observations require emitted_tokens == "
                "accepted_tokens + 1"
            )
        object.__setattr__(self, "latency_seconds", latency)
        object.__setattr__(self, "transferred_bytes", byte_count)
        object.__setattr__(self, "emitted_tokens", emitted)
        object.__setattr__(self, "accepted_tokens", accepted)


@dataclass(frozen=True, slots=True)
class TreeWaveBudget:
    """Per-decision capacity, intersected with the configured hard limits."""

    available_kv_bytes: int
    max_branches: int
    max_candidate_tokens: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "available_kv_bytes",
            _count("available_kv_bytes", self.available_kv_bytes),
        )
        object.__setattr__(
            self, "max_branches", _count("max_branches", self.max_branches)
        )
        object.__setattr__(
            self,
            "max_candidate_tokens",
            _count("max_candidate_tokens", self.max_candidate_tokens),
        )


@dataclass(frozen=True, slots=True)
class TreeWavePolicyConfig:
    candidate_shapes: tuple[TreeWaveShape, ...]
    min_classic_observations: int = 4
    min_tree_observations: int = 3
    seconds_per_byte: float = 0.0
    minimum_speedup: float = 1.05
    max_projected_kv_bytes: int = (1 << 63) - 1
    max_branches: int = 64
    max_candidate_tokens: int = 1024
    max_depth: int = 16
    rtt_reference_ms: float = 25.0
    rtt_ewma_alpha: float = 0.2
    measurement_ewma_alpha: float = 0.25
    hysteresis_fraction: float = 0.05
    cooldown_decisions: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_shapes, tuple) or not self.candidate_shapes:
            raise ValueError("candidate_shapes must be a non-empty tuple")
        normalized: list[TreeWaveShape] = []
        for shape in self.candidate_shapes:
            if not isinstance(shape, TreeWaveShape):
                raise ValueError("candidate_shapes must contain TreeWaveShape values")
            if shape not in normalized:
                normalized.append(shape)
        normalized.sort(
            key=lambda shape: (
                shape.total_candidate_tokens,
                shape.width,
                shape.depth,
            )
        )
        object.__setattr__(self, "candidate_shapes", tuple(normalized))

        for name, value, minimum in (
            ("min_classic_observations", self.min_classic_observations, 1),
            ("min_tree_observations", self.min_tree_observations, 1),
            ("max_projected_kv_bytes", self.max_projected_kv_bytes, 0),
            ("max_branches", self.max_branches, 0),
            ("max_candidate_tokens", self.max_candidate_tokens, 0),
            ("max_depth", self.max_depth, 1),
            ("cooldown_decisions", self.cooldown_decisions, 0),
        ):
            object.__setattr__(self, name, _count(name, value, minimum=minimum))

        byte_cost = _measurement("seconds_per_byte", self.seconds_per_byte)
        speedup = _measurement(
            "minimum_speedup", self.minimum_speedup, minimum=1.0
        )
        reference = _measurement(
            "rtt_reference_ms", self.rtt_reference_ms, minimum=0.0
        )
        if reference == 0.0:
            raise ValueError("rtt_reference_ms must be > 0")
        rtt_alpha = _measurement(
            "rtt_ewma_alpha", self.rtt_ewma_alpha, minimum=0.0, maximum=1.0
        )
        if rtt_alpha == 0.0:
            raise ValueError("rtt_ewma_alpha must be > 0")
        measurement_alpha = _measurement(
            "measurement_ewma_alpha",
            self.measurement_ewma_alpha,
            minimum=0.0,
            maximum=1.0,
        )
        if measurement_alpha == 0.0:
            raise ValueError("measurement_ewma_alpha must be > 0")
        hysteresis = _measurement(
            "hysteresis_fraction",
            self.hysteresis_fraction,
            minimum=0.0,
            maximum=1.0,
        )
        object.__setattr__(self, "seconds_per_byte", byte_cost)
        object.__setattr__(self, "minimum_speedup", speedup)
        object.__setattr__(self, "rtt_reference_ms", reference)
        object.__setattr__(self, "rtt_ewma_alpha", rtt_alpha)
        object.__setattr__(self, "measurement_ewma_alpha", measurement_alpha)
        object.__setattr__(self, "hysteresis_fraction", hysteresis)


@dataclass(frozen=True, slots=True)
class ClassicMetrics:
    observations: int
    emitted_tokens: int
    total_latency_seconds: float
    total_bytes: int
    latency_seconds_per_token: float | None
    bytes_per_token: float | None
    cost_per_token: float | None


@dataclass(frozen=True, slots=True)
class TreeShapeMetrics:
    shape: TreeWaveShape
    observations: int
    emitted_tokens: int
    accepted_tokens: int
    total_latency_seconds: float
    total_bytes: int
    acceptance_rate: float | None
    latency_seconds_per_token: float | None
    bytes_per_token: float | None
    cost_per_token: float | None
    predicted_speedup: float | None
    ready: bool


@dataclass(frozen=True, slots=True)
class TreeWaveDecision:
    mode: str
    shape: TreeWaveShape | None
    projected_kv_bytes: int | None
    reason: str
    classic_cost_per_token: float | None
    selected_cost_per_token: float | None
    predicted_speedup: float | None
    rtt_ewma_ms: float | None
    depth_cap: int

    @property
    def enabled(self) -> bool:
        return self.mode == "tree"


@dataclass(frozen=True, slots=True)
class TreeWavePolicyStats:
    classic: ClassicMetrics
    shapes: tuple[TreeShapeMetrics, ...]
    rtt_observations: int
    rtt_ewma_ms: float | None
    depth_cap: int
    selected_shape: TreeWaveShape | None
    pending_probe: TreeWaveShape | None
    cooldown_remaining: int
    decisions: int
    probes_issued: int

    def to_dict(self) -> dict[str, Any]:
        """Return only JSON-compatible containers and finite values/``None``."""

        return asdict(self)


@dataclass(slots=True)
class _Measurements:
    observations: int = 0
    emitted_tokens: int = 0
    accepted_tokens: int = 0
    total_latency_seconds: float = 0.0
    total_bytes: int = 0
    latency_tpot_ewma: float | None = None
    bytes_per_token_ewma: float | None = None

    def record(
        self,
        *,
        latency_seconds: float,
        transferred_bytes: int,
        emitted_tokens: int,
        accepted_tokens: int,
        alpha: float,
    ) -> None:
        latency_tpot = latency_seconds / emitted_tokens
        bytes_per_token = transferred_bytes / emitted_tokens
        next_observations = self.observations + 1
        next_emitted = self.emitted_tokens + emitted_tokens
        next_accepted = self.accepted_tokens + accepted_tokens
        next_bytes = self.total_bytes + transferred_bytes
        if any(
            value > MAX_POLICY_COUNT
            for value in (next_observations, next_emitted, next_accepted, next_bytes)
        ):
            raise ValueError("measurement totals exceed the sealed policy range")
        next_latency = self.total_latency_seconds + latency_seconds
        if not math.isfinite(next_latency):
            raise ValueError("total_latency_seconds must remain finite")

        if self.latency_tpot_ewma is None:
            next_latency_ewma = latency_tpot
            next_bytes_ewma = bytes_per_token
        else:
            next_latency_ewma = (
                alpha * latency_tpot + (1.0 - alpha) * self.latency_tpot_ewma
            )
            assert self.bytes_per_token_ewma is not None
            next_bytes_ewma = (
                alpha * bytes_per_token
                + (1.0 - alpha) * self.bytes_per_token_ewma
            )
        if not math.isfinite(next_latency_ewma) or not math.isfinite(next_bytes_ewma):
            raise ValueError("per-token EWMA must remain finite")

        # Commit only after every aggregate has been checked so a rejected
        # observation cannot leave a partially mutated controller.
        self.observations = next_observations
        self.emitted_tokens = next_emitted
        self.accepted_tokens = next_accepted
        self.total_latency_seconds = next_latency
        self.total_bytes = next_bytes
        self.latency_tpot_ewma = next_latency_ewma
        self.bytes_per_token_ewma = next_bytes_ewma


class RTTAwareTreeWaveController:
    """Fail-closed selector for classic versus measured exact tree waves."""

    def __init__(self, config: TreeWavePolicyConfig) -> None:
        if not isinstance(config, TreeWavePolicyConfig):
            raise ValueError("config must be a TreeWavePolicyConfig")
        self.config = config
        self._classic = _Measurements()
        self._tree = {shape: _Measurements() for shape in config.candidate_shapes}
        self._rtt_observations = 0
        self._rtt_ewma_ms: float | None = None
        self._selected_shape: TreeWaveShape | None = None
        self._pending_probe: TreeWaveShape | None = None
        self._cooldown_remaining = 0
        self._decisions = 0
        self._probes_issued = 0

    @property
    def rtt_ewma_ms(self) -> float | None:
        return self._rtt_ewma_ms

    @property
    def selected_shape(self) -> TreeWaveShape | None:
        return self._selected_shape

    def reset(self) -> None:
        self._classic = _Measurements()
        self._tree = {
            shape: _Measurements() for shape in self.config.candidate_shapes
        }
        self._rtt_observations = 0
        self._rtt_ewma_ms = None
        self._selected_shape = None
        self._pending_probe = None
        self._cooldown_remaining = 0
        self._decisions = 0
        self._probes_issued = 0

    def record_rtt(self, rtt_ms: float) -> None:
        measured = _measurement("rtt_ms", rtt_ms)
        next_observations = _increment_count(
            "rtt_observations", self._rtt_observations
        )
        if self._rtt_ewma_ms is None:
            next_ewma = measured
        else:
            alpha = self.config.rtt_ewma_alpha
            next_ewma = (
                alpha * measured + (1.0 - alpha) * self._rtt_ewma_ms
            )
        next_ewma = _measurement("rtt_ewma_ms", next_ewma)
        # Commit together only after both the counter and derived EWMA pass.
        self._rtt_observations = next_observations
        self._rtt_ewma_ms = next_ewma

    def rtt_depth_cap(self, rtt_ms: float | None = None) -> int:
        """Logarithmic, monotone cap; no RTT evidence permits depth one only."""

        if rtt_ms is None:
            measured = self._rtt_ewma_ms
        else:
            measured = _measurement("rtt_ms", rtt_ms)
        if measured is None:
            return 1
        ratio = measured / self.config.rtt_reference_ms
        if not math.isfinite(ratio):
            return self.config.max_depth
        raw = 1 + math.floor(math.log2(1.0 + ratio))
        return max(1, min(raw, self.config.max_depth))

    def record_classic(self, observation: ClassicObservation) -> None:
        if not isinstance(observation, ClassicObservation):
            raise ValueError("observation must be a ClassicObservation")
        self._classic.record(
            latency_seconds=observation.latency_seconds,
            transferred_bytes=observation.transferred_bytes,
            emitted_tokens=observation.emitted_tokens,
            accepted_tokens=0,
            alpha=self.config.measurement_ewma_alpha,
        )

    def record_tree(self, observation: TreeWaveObservation) -> None:
        if not isinstance(observation, TreeWaveObservation):
            raise ValueError("observation must be a TreeWaveObservation")
        measurements = self._tree.get(observation.shape)
        if measurements is None:
            raise ValueError("observation shape is not configured")
        measurements.record(
            latency_seconds=observation.latency_seconds,
            transferred_bytes=observation.transferred_bytes,
            emitted_tokens=observation.emitted_tokens,
            accepted_tokens=observation.accepted_tokens,
            alpha=self.config.measurement_ewma_alpha,
        )
        if self._pending_probe == observation.shape:
            self._pending_probe = None

    def cancel_probe(self, shape: TreeWaveShape | None = None) -> bool:
        """Release an explicitly issued probe that could not be completed."""

        if shape is not None and not isinstance(shape, TreeWaveShape):
            raise ValueError("shape must be a TreeWaveShape or None")
        if self._pending_probe is None:
            return False
        if shape is not None and shape != self._pending_probe:
            return False
        self._pending_probe = None
        return True

    def next_probe(
        self,
        *,
        candidates: tuple[TreeWaveCandidate, ...] | None = None,
        budget: TreeWaveBudget | None = None,
    ) -> TreeWaveCandidate | None:
        """Issue at most one controlled probe after classic warm-up.

        A pending probe blocks another issuance until a matching observation is
        recorded or :meth:`cancel_probe` is called.  Calling ``decide`` never
        issues probes.
        """

        checked_budget = self._budget(budget)
        projected = self._candidates(candidates)
        if self._pending_probe is not None:
            return None
        if self._classic.observations < self.config.min_classic_observations:
            return None
        for shape in self.config.candidate_shapes:
            candidate = projected.get(shape)
            if candidate is None or not self._eligible(candidate, checked_budget):
                continue
            if self._tree[shape].observations < self.config.min_tree_observations:
                next_probes_issued = _increment_count(
                    "probes_issued", self._probes_issued
                )
                self._pending_probe = shape
                self._probes_issued = next_probes_issued
                return candidate
        return None

    def decide(
        self,
        *,
        candidates: tuple[TreeWaveCandidate, ...] | None = None,
        budget: TreeWaveBudget | None = None,
    ) -> TreeWaveDecision:
        checked_budget = self._budget(budget)
        projected = self._candidates(candidates)
        next_decisions = _increment_count("decisions", self._decisions)
        self._decisions = next_decisions
        classic = self._classic_metrics()
        cap = self.rtt_depth_cap()

        if self._classic.observations < self.config.min_classic_observations:
            return self._classic_decision("classic_warmup", classic, cap)
        if classic.cost_per_token is None or classic.cost_per_token <= 0.0:
            return self._drop_to_classic("classic_zero_cost", classic, cap)

        eligible = [
            metrics
            for metrics in self._shape_metrics(classic.cost_per_token)
            if metrics.ready
            and metrics.shape in projected
            and self._eligible(projected[metrics.shape], checked_budget)
            and metrics.cost_per_token is not None
            and metrics.cost_per_token > 0.0
            and metrics.predicted_speedup is not None
        ]
        if not eligible:
            any_eligible = any(
                shape in projected
                and self._eligible(projected[shape], checked_budget)
                for shape in self.config.candidate_shapes
            )
            reason = "tree_warmup" if any_eligible else "no_eligible_shape"
            return self._drop_to_classic(reason, classic, cap)

        eligible.sort(
            key=lambda item: (
                float(item.cost_per_token),
                projected[item.shape].projected_kv_bytes,
                item.shape.total_candidate_tokens,
                item.shape.width,
                item.shape.depth,
            )
        )
        best = eligible[0]
        current = next(
            (item for item in eligible if item.shape == self._selected_shape), None
        )

        # Safety overrides cooldown: a selected shape that is now ineligible,
        # unmeasured, zero-cost, or below the minimum speedup drops immediately.
        if self._selected_shape is not None and (
            current is None
            or current.predicted_speedup is None
            or current.predicted_speedup < self.config.minimum_speedup
        ):
            return self._drop_to_classic("selected_shape_slowdown", classic, cap)

        if self._selected_shape is None:
            entry_threshold = self.config.minimum_speedup * (
                1.0 + self.config.hysteresis_fraction
            )
            if (
                best.predicted_speedup is None
                or best.predicted_speedup < entry_threshold
            ):
                reason = (
                    "hysteresis_hold"
                    if best.predicted_speedup is not None
                    and best.predicted_speedup >= self.config.minimum_speedup
                    else "not_beneficial"
                )
                return self._classic_decision(reason, classic, cap)
            if self._cooldown_remaining > 0:
                self._cooldown_remaining -= 1
                return self._classic_decision("cooldown_hold", classic, cap)
            self._selected_shape = best.shape
            self._cooldown_remaining = self.config.cooldown_decisions
            return self._tree_decision(
                "beneficial", classic, best, projected[best.shape], cap
            )

        assert current is not None
        assert current.cost_per_token is not None
        if best.shape != current.shape:
            improvement_needed = current.cost_per_token / (
                1.0 + self.config.hysteresis_fraction
            )
            materially_better = float(best.cost_per_token) < improvement_needed
            if materially_better and self._cooldown_remaining == 0:
                self._selected_shape = best.shape
                self._cooldown_remaining = self.config.cooldown_decisions
                return self._tree_decision(
                    "better_shape", classic, best, projected[best.shape], cap
                )

        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            reason = "cooldown_hold"
        else:
            reason = "hysteresis_hold"
        return self._tree_decision(
            reason, classic, current, projected[current.shape], cap
        )

    def stats(self) -> TreeWavePolicyStats:
        classic = self._classic_metrics()
        return TreeWavePolicyStats(
            classic=classic,
            shapes=self._shape_metrics(classic.cost_per_token),
            rtt_observations=self._rtt_observations,
            rtt_ewma_ms=self._rtt_ewma_ms,
            depth_cap=self.rtt_depth_cap(),
            selected_shape=self._selected_shape,
            pending_probe=self._pending_probe,
            cooldown_remaining=self._cooldown_remaining,
            decisions=self._decisions,
            probes_issued=self._probes_issued,
        )

    def _budget(self, budget: TreeWaveBudget | None) -> TreeWaveBudget:
        if budget is not None and not isinstance(budget, TreeWaveBudget):
            raise ValueError("budget must be a TreeWaveBudget or None")
        configured = TreeWaveBudget(
            available_kv_bytes=self.config.max_projected_kv_bytes,
            max_branches=self.config.max_branches,
            max_candidate_tokens=self.config.max_candidate_tokens,
        )
        if budget is None:
            return configured
        return TreeWaveBudget(
            available_kv_bytes=min(
                configured.available_kv_bytes, budget.available_kv_bytes
            ),
            max_branches=min(configured.max_branches, budget.max_branches),
            max_candidate_tokens=min(
                configured.max_candidate_tokens, budget.max_candidate_tokens
            ),
        )

    def _candidates(
        self,
        candidates: tuple[TreeWaveCandidate, ...] | None,
    ) -> dict[TreeWaveShape, TreeWaveCandidate]:
        """Validate current projections without inventing a zero-byte default.

        Missing projections are simply ineligible.  This is deliberately
        fail-closed: a controller can collect classic observations before a
        backend can quote KV, but it cannot enable or probe a tree until the
        current context has been projected explicitly.
        """

        if candidates is None:
            return {}
        if not isinstance(candidates, tuple):
            raise ValueError("candidates must be a tuple or None")
        configured = set(self.config.candidate_shapes)
        result: dict[TreeWaveShape, TreeWaveCandidate] = {}
        for candidate in candidates:
            if not isinstance(candidate, TreeWaveCandidate):
                raise ValueError("candidates must contain TreeWaveCandidate values")
            if candidate.shape not in configured:
                raise ValueError("candidate shape is not configured")
            if candidate.shape in result:
                raise ValueError("candidates cannot repeat a shape")
            result[candidate.shape] = candidate
        return result

    def _eligible(
        self,
        candidate: TreeWaveCandidate,
        budget: TreeWaveBudget,
    ) -> bool:
        shape = candidate.shape
        return (
            shape.width <= budget.max_branches
            and shape.total_candidate_tokens <= budget.max_candidate_tokens
            and candidate.projected_kv_bytes <= budget.available_kv_bytes
            and shape.depth <= self.config.max_depth
            and shape.depth <= self.rtt_depth_cap()
        )

    def _classic_metrics(self) -> ClassicMetrics:
        latency = self._classic.latency_tpot_ewma
        byte_rate = self._classic.bytes_per_token_ewma
        cost = self._cost(latency, byte_rate)
        return ClassicMetrics(
            observations=self._classic.observations,
            emitted_tokens=self._classic.emitted_tokens,
            total_latency_seconds=self._classic.total_latency_seconds,
            total_bytes=self._classic.total_bytes,
            latency_seconds_per_token=latency,
            bytes_per_token=byte_rate,
            cost_per_token=cost,
        )

    def _shape_metrics(
        self, classic_cost_per_token: float | None
    ) -> tuple[TreeShapeMetrics, ...]:
        result: list[TreeShapeMetrics] = []
        for shape in self.config.candidate_shapes:
            measured = self._tree[shape]
            cost = self._cost(
                measured.latency_tpot_ewma, measured.bytes_per_token_ewma
            )
            speedup: float | None = None
            if (
                classic_cost_per_token is not None
                and classic_cost_per_token > 0.0
                and cost is not None
                and cost > 0.0
            ):
                speedup = classic_cost_per_token / cost
            denominator = measured.observations * shape.depth
            acceptance = (
                measured.accepted_tokens / denominator if denominator > 0 else None
            )
            result.append(
                TreeShapeMetrics(
                    shape=shape,
                    observations=measured.observations,
                    emitted_tokens=measured.emitted_tokens,
                    accepted_tokens=measured.accepted_tokens,
                    total_latency_seconds=measured.total_latency_seconds,
                    total_bytes=measured.total_bytes,
                    acceptance_rate=acceptance,
                    latency_seconds_per_token=measured.latency_tpot_ewma,
                    bytes_per_token=measured.bytes_per_token_ewma,
                    cost_per_token=cost,
                    predicted_speedup=speedup,
                    ready=(
                        self._classic.observations
                        >= self.config.min_classic_observations
                        and measured.observations
                        >= self.config.min_tree_observations
                    ),
                )
            )
        return tuple(result)

    def _cost(
        self, latency_tpot: float | None, bytes_per_token: float | None
    ) -> float | None:
        if latency_tpot is None or bytes_per_token is None:
            return None
        cost = latency_tpot + self.config.seconds_per_byte * bytes_per_token
        return cost if math.isfinite(cost) and cost >= 0.0 else None

    def _drop_to_classic(
        self, reason: str, classic: ClassicMetrics, cap: int
    ) -> TreeWaveDecision:
        if self._selected_shape is not None:
            self._selected_shape = None
            self._cooldown_remaining = self.config.cooldown_decisions
        return self._classic_decision(reason, classic, cap)

    def _classic_decision(
        self, reason: str, classic: ClassicMetrics, cap: int
    ) -> TreeWaveDecision:
        return TreeWaveDecision(
            mode="classic",
            shape=None,
            projected_kv_bytes=None,
            reason=reason,
            classic_cost_per_token=classic.cost_per_token,
            selected_cost_per_token=classic.cost_per_token,
            predicted_speedup=1.0 if classic.cost_per_token is not None else None,
            rtt_ewma_ms=self._rtt_ewma_ms,
            depth_cap=cap,
        )

    def _tree_decision(
        self,
        reason: str,
        classic: ClassicMetrics,
        selected: TreeShapeMetrics,
        candidate: TreeWaveCandidate,
        cap: int,
    ) -> TreeWaveDecision:
        if candidate.shape != selected.shape:
            raise RuntimeError("tree decision candidate does not match its metrics")
        return TreeWaveDecision(
            mode="tree",
            shape=selected.shape,
            projected_kv_bytes=candidate.projected_kv_bytes,
            reason=reason,
            classic_cost_per_token=classic.cost_per_token,
            selected_cost_per_token=selected.cost_per_token,
            predicted_speedup=selected.predicted_speedup,
            rtt_ewma_ms=self._rtt_ewma_ms,
            depth_cap=cap,
        )


__all__ = [
    "MAX_POLICY_COUNT",
    "ClassicMetrics",
    "ClassicObservation",
    "RTTAwareTreeWaveController",
    "TreeShapeMetrics",
    "TreeWaveBudget",
    "TreeWaveCandidate",
    "TreeWaveDecision",
    "TreeWaveObservation",
    "TreeWavePolicyConfig",
    "TreeWavePolicyStats",
    "TreeWaveShape",
]
