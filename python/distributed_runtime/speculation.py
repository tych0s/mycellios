"""Model-independent speculative decoding policy helpers.

The distributed engine owns verification and KV rollback.  This module keeps
draft generation and the adaptive go/no-go decision deliberately independent
from a model backend so the same controller can be used with n-gram lookup,
MTP/EAGLE heads, or an external draft model.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from numbers import Integral
from typing import Protocol, runtime_checkable


MAX_DRAFT_TOKENS = 16


def _is_int(value: object) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def _validate_count(name: str, value: int, *, minimum: int = 0) -> int:
    if not _is_int(value) or int(value) < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _validate_measurement(name: str, value: float, *, allow_zero: bool) -> float:
    try:
        measured = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(measured) or measured < 0 or (not allow_zero and measured == 0):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return measured


@runtime_checkable
class DraftProvider(Protocol):
    """Minimal interface implemented by any speculative draft source."""

    strategy: str
    max_draft_tokens: int

    def draft(
        self,
        token_history: Sequence[int],
        max_tokens: int | None = None,
    ) -> tuple[int, ...]:
        """Return up to ``max_tokens`` deterministic candidate tokens."""


@dataclass(frozen=True)
class NgramDraftProvider:
    """Copy a continuation from the longest repeated suffix in token history.

    Every earlier position in ``token_history`` is searched; this is not a
    sliding-window lookup.  ``max_match_tokens`` only caps the suffix width and
    therefore bounds lookup cost.  Ties use the most recent prior occurrence,
    making the result deterministic and biased toward recent continuations.
    """

    max_draft_tokens: int = 8
    min_match_tokens: int = 2
    max_match_tokens: int | None = 8
    strategy: str = "ngram"

    def __post_init__(self) -> None:
        if (
            not _is_int(self.max_draft_tokens)
            or not 1 <= int(self.max_draft_tokens) <= MAX_DRAFT_TOKENS
        ):
            raise ValueError(
                f"max_draft_tokens must be an integer between 1 and {MAX_DRAFT_TOKENS}"
            )
        if not _is_int(self.min_match_tokens) or int(self.min_match_tokens) < 1:
            raise ValueError("min_match_tokens must be a positive integer")
        if self.max_match_tokens is not None:
            if not _is_int(self.max_match_tokens) or int(self.max_match_tokens) < int(
                self.min_match_tokens
            ):
                raise ValueError(
                    "max_match_tokens must be None or an integer >= min_match_tokens"
                )
        if self.strategy != "ngram":
            raise ValueError("NgramDraftProvider strategy must be 'ngram'")

    def draft(
        self,
        token_history: Sequence[int],
        max_tokens: int | None = None,
    ) -> tuple[int, ...]:
        if isinstance(token_history, (str, bytes, bytearray)):
            raise ValueError("token_history must be a sequence of integer token ids")
        observed: list[int] = []
        for token in token_history:
            if not _is_int(token) or int(token) < 0:
                raise ValueError("token_history must contain non-negative integer token ids")
            observed.append(int(token))

        if max_tokens is None:
            limit = int(self.max_draft_tokens)
        else:
            limit = _validate_count("max_tokens", max_tokens)
            limit = min(limit, int(self.max_draft_tokens))
        if limit == 0 or len(observed) <= int(self.min_match_tokens):
            return ()

        largest_width = len(observed) - 1
        if self.max_match_tokens is not None:
            largest_width = min(largest_width, int(self.max_match_tokens))

        # Search suffix widths from most to least specific.  Search start
        # positions from newest to oldest so ties have a stable recency rule.
        history_size = len(observed)
        for width in range(largest_width, int(self.min_match_tokens) - 1, -1):
            suffix_start = history_size - width
            suffix = observed[suffix_start:]
            for start in range(suffix_start - 1, -1, -1):
                continuation_start = start + width
                if observed[start:continuation_start] != suffix:
                    continue
                continuation_end = min(history_size, continuation_start + limit)
                continuation = observed[continuation_start:continuation_end]
                if continuation:
                    return tuple(continuation)
        return ()


def _automatic_candidate_sizes(max_draft_tokens: int) -> tuple[int, ...]:
    candidates = [size for size in (1, 2, 4, 8, 16) if size <= max_draft_tokens]
    if max_draft_tokens not in candidates:
        candidates.append(max_draft_tokens)
    return tuple(sorted(candidates))


@dataclass(frozen=True)
class AdaptiveSpeculationConfig:
    """Configuration for an observation-driven speculative decoding gate."""

    max_draft_tokens: int = 8
    candidate_sizes: tuple[int, ...] = ()
    min_token_history: int = 8
    min_classic_observations: int = 4
    min_verify_observations: int = 3
    seconds_per_byte: float = 0.0
    minimum_speedup: float = 1.0
    # Opt-in delay-adaptive depth cap (UCB-SpecStop, arXiv 2606.20591): the
    # optimal draft depth is a monotone threshold in communication delay and
    # grows only logarithmically with it.  Defaults keep behaviour unchanged.
    delay_adaptive: bool = False
    # RTT at which a two-token draft first becomes admissible; below it the
    # delay cap stays at 1.
    delay_reference_ms: float = 25.0
    rtt_ewma_alpha: float = 0.2

    def __post_init__(self) -> None:
        if (
            not _is_int(self.max_draft_tokens)
            or not 1 <= int(self.max_draft_tokens) <= MAX_DRAFT_TOKENS
        ):
            raise ValueError(
                f"max_draft_tokens must be an integer between 1 and {MAX_DRAFT_TOKENS}"
            )

        if not self.candidate_sizes:
            normalized = _automatic_candidate_sizes(int(self.max_draft_tokens))
        else:
            normalized_values: list[int] = []
            for value in self.candidate_sizes:
                if (
                    not _is_int(value)
                    or not 1 <= int(value) <= int(self.max_draft_tokens)
                ):
                    raise ValueError(
                        "candidate_sizes must contain integers between 1 and max_draft_tokens"
                    )
                normalized_values.append(int(value))
            normalized = tuple(sorted(set(normalized_values)))
        object.__setattr__(self, "candidate_sizes", normalized)

        for name, value, minimum in (
            ("min_token_history", self.min_token_history, 0),
            ("min_classic_observations", self.min_classic_observations, 1),
            ("min_verify_observations", self.min_verify_observations, 1),
        ):
            _validate_count(name, value, minimum=minimum)
        byte_cost = _validate_measurement(
            "seconds_per_byte", self.seconds_per_byte, allow_zero=True
        )
        threshold = _validate_measurement(
            "minimum_speedup", self.minimum_speedup, allow_zero=False
        )
        if threshold < 1.0:
            raise ValueError("minimum_speedup must be >= 1")
        object.__setattr__(self, "seconds_per_byte", byte_cost)
        object.__setattr__(self, "minimum_speedup", threshold)

        if not isinstance(self.delay_adaptive, bool):
            raise ValueError("delay_adaptive must be a bool")
        reference = _validate_measurement(
            "delay_reference_ms", self.delay_reference_ms, allow_zero=False
        )
        alpha = _validate_measurement(
            "rtt_ewma_alpha", self.rtt_ewma_alpha, allow_zero=False
        )
        if alpha > 1.0:
            raise ValueError("rtt_ewma_alpha must be in (0, 1]")
        object.__setattr__(self, "delay_reference_ms", reference)
        object.__setattr__(self, "rtt_ewma_alpha", alpha)


@dataclass(frozen=True)
class SpeculationStats:
    classic_observations: int
    classic_generated_tokens: int
    classic_latency_seconds: float
    classic_bytes: int
    verification_observations: int
    proposed_tokens: int
    accepted_tokens: int
    emitted_tokens: int
    verification_latency_seconds: float
    verification_bytes: int

    @property
    def acceptance_rate(self) -> float | None:
        if self.proposed_tokens == 0:
            return None
        return self.accepted_tokens / self.proposed_tokens


@dataclass(frozen=True)
class CandidateEstimate:
    candidate_size: int
    observations: int
    proposed_tokens: int
    accepted_tokens: int
    acceptance_rate: float | None
    expected_emitted_tokens: float | None
    mean_verification_latency_seconds: float | None
    mean_verification_bytes: float | None
    predicted_latency_speedup: float | None
    predicted_byte_efficiency: float | None
    predicted_speedup: float | None
    ready: bool


@dataclass(frozen=True)
class SpeculationDecision:
    enabled: bool
    candidate_size: int
    predicted_speedup: float | None
    predicted_latency_speedup: float | None
    predicted_byte_efficiency: float | None
    expected_emitted_tokens: float | None
    reason: str

    @property
    def draft_tokens(self) -> int:
        """Alias useful at engine call sites."""

        return self.candidate_size


@dataclass
class _ClassicMeasurements:
    observations: int = 0
    generated_tokens: int = 0
    latency_seconds: float = 0.0
    transferred_bytes: int = 0


@dataclass
class _VerificationMeasurements:
    observations: int = 0
    proposed_tokens: int = 0
    accepted_tokens: int = 0
    emitted_tokens: int = 0
    latency_seconds: float = 0.0
    transferred_bytes: int = 0


class AdaptiveSpeculationController:
    """Choose a verified draft size only when measured cost predicts a win.

    ``decide`` never turns warm-up into an implicit experiment: until classic
    and verification history meet their configured sample counts it returns a
    disabled decision.  A scheduler that wants controlled exploration can call
    ``next_probe_size`` and explicitly run that size as a probe.
    """

    def __init__(self, config: AdaptiveSpeculationConfig | None = None) -> None:
        self.config = config or AdaptiveSpeculationConfig()
        self._classic = _ClassicMeasurements()
        self._verification: dict[int, _VerificationMeasurements] = {}
        self._rtt_ewma_ms: float | None = None

    def reset(self) -> None:
        self._classic = _ClassicMeasurements()
        self._verification.clear()
        self._rtt_ewma_ms = None

    def record_rtt(self, rtt_ms: float) -> None:
        rtt = _validate_measurement("rtt_ms", rtt_ms, allow_zero=True)
        if self._rtt_ewma_ms is None:
            self._rtt_ewma_ms = rtt
        else:
            alpha = float(self.config.rtt_ewma_alpha)
            self._rtt_ewma_ms = alpha * rtt + (1.0 - alpha) * self._rtt_ewma_ms

    @property
    def rtt_ewma_ms(self) -> float | None:
        return self._rtt_ewma_ms

    def delay_draft_cap(self, rtt_ms: float | None = None) -> int:
        """Depth cap for the measured delay; never raises depth on its own.

        cap = clamp(1 + floor(log2(1 + rtt / delay_reference_ms)), 1, max).
        Monotone threshold with logarithmic growth in delay (UCB-SpecStop,
        arXiv 2606.20591).  Without RTT evidence the cap is max_draft_tokens.
        """

        if rtt_ms is None:
            if self._rtt_ewma_ms is None:
                return int(self.config.max_draft_tokens)
            rtt = self._rtt_ewma_ms
        else:
            rtt = _validate_measurement("rtt_ms", rtt_ms, allow_zero=True)
        ratio = rtt / float(self.config.delay_reference_ms)
        # A pathological (but validation-passing) tiny delay_reference_ms can
        # overflow the ratio to +inf; floor(inf) raises instead of clamping, so
        # saturate to the ceiling here rather than crash the decision path.
        if not math.isfinite(ratio):
            return int(self.config.max_draft_tokens)
        raw = 1 + math.floor(math.log2(1.0 + ratio))
        return max(1, min(raw, int(self.config.max_draft_tokens)))

    def record_classic(
        self,
        *,
        latency_seconds: float,
        transferred_bytes: int,
        generated_tokens: int = 1,
    ) -> None:
        latency = _validate_measurement(
            "latency_seconds", latency_seconds, allow_zero=False
        )
        byte_count = _validate_count("transferred_bytes", transferred_bytes)
        tokens = _validate_count("generated_tokens", generated_tokens, minimum=1)
        self._classic.observations += 1
        self._classic.generated_tokens += tokens
        self._classic.latency_seconds += latency
        self._classic.transferred_bytes += byte_count

    def record_verification(
        self,
        *,
        proposed_tokens: int,
        accepted_tokens: int,
        latency_seconds: float,
        transferred_bytes: int,
    ) -> None:
        proposed = _validate_count("proposed_tokens", proposed_tokens, minimum=1)
        if proposed > int(self.config.max_draft_tokens):
            raise ValueError("proposed_tokens cannot exceed max_draft_tokens")
        accepted = _validate_count("accepted_tokens", accepted_tokens)
        if accepted > proposed:
            raise ValueError("accepted_tokens cannot exceed proposed_tokens")
        latency = _validate_measurement(
            "latency_seconds", latency_seconds, allow_zero=False
        )
        byte_count = _validate_count("transferred_bytes", transferred_bytes)

        measurements = self._verification.setdefault(
            proposed, _VerificationMeasurements()
        )
        measurements.observations += 1
        measurements.proposed_tokens += proposed
        measurements.accepted_tokens += accepted
        # Exact speculative decoding emits every accepted draft plus either a
        # correction token or the target model's bonus token.
        measurements.emitted_tokens += accepted + 1
        measurements.latency_seconds += latency
        measurements.transferred_bytes += byte_count

    def stats(self) -> SpeculationStats:
        verification = tuple(self._verification.values())
        return SpeculationStats(
            classic_observations=self._classic.observations,
            classic_generated_tokens=self._classic.generated_tokens,
            classic_latency_seconds=self._classic.latency_seconds,
            classic_bytes=self._classic.transferred_bytes,
            verification_observations=sum(item.observations for item in verification),
            proposed_tokens=sum(item.proposed_tokens for item in verification),
            accepted_tokens=sum(item.accepted_tokens for item in verification),
            emitted_tokens=sum(item.emitted_tokens for item in verification),
            verification_latency_seconds=sum(
                item.latency_seconds for item in verification
            ),
            verification_bytes=sum(item.transferred_bytes for item in verification),
        )

    def candidate_estimate(self, candidate_size: int) -> CandidateEstimate:
        candidate = _validate_count("candidate_size", candidate_size, minimum=1)
        if candidate > int(self.config.max_draft_tokens):
            raise ValueError("candidate_size cannot exceed max_draft_tokens")
        measurements = self._verification.get(candidate, _VerificationMeasurements())
        ready = (
            self._classic.observations >= int(self.config.min_classic_observations)
            and measurements.observations >= int(self.config.min_verify_observations)
        )
        if measurements.proposed_tokens == 0:
            acceptance_rate = None
            expected_emitted = None
        else:
            acceptance_rate = (
                measurements.accepted_tokens / measurements.proposed_tokens
            )
            expected_emitted = 1.0 + candidate * acceptance_rate

        if measurements.observations == 0:
            mean_verify_latency = None
            mean_verify_bytes = None
        else:
            mean_verify_latency = (
                measurements.latency_seconds / measurements.observations
            )
            mean_verify_bytes = (
                measurements.transferred_bytes / measurements.observations
            )

        predicted_latency_speedup: float | None = None
        predicted_byte_efficiency: float | None = None
        predicted_speedup: float | None = None
        if ready and expected_emitted is not None:
            classic_latency_per_token = (
                self._classic.latency_seconds / self._classic.generated_tokens
            )
            classic_bytes_per_token = (
                self._classic.transferred_bytes / self._classic.generated_tokens
            )
            assert mean_verify_latency is not None
            assert mean_verify_bytes is not None
            predicted_latency_speedup = (
                classic_latency_per_token * expected_emitted / mean_verify_latency
            )
            if mean_verify_bytes == 0:
                predicted_byte_efficiency = math.inf
            else:
                predicted_byte_efficiency = (
                    classic_bytes_per_token * expected_emitted / mean_verify_bytes
                )
            classic_cost_per_token = (
                classic_latency_per_token
                + float(self.config.seconds_per_byte) * classic_bytes_per_token
            )
            verification_cost = (
                mean_verify_latency
                + float(self.config.seconds_per_byte) * mean_verify_bytes
            )
            predicted_speedup = (
                classic_cost_per_token * expected_emitted / verification_cost
            )

        return CandidateEstimate(
            candidate_size=candidate,
            observations=measurements.observations,
            proposed_tokens=measurements.proposed_tokens,
            accepted_tokens=measurements.accepted_tokens,
            acceptance_rate=acceptance_rate,
            expected_emitted_tokens=expected_emitted,
            mean_verification_latency_seconds=mean_verify_latency,
            mean_verification_bytes=mean_verify_bytes,
            predicted_latency_speedup=predicted_latency_speedup,
            predicted_byte_efficiency=predicted_byte_efficiency,
            predicted_speedup=predicted_speedup,
            ready=ready,
        )

    def candidate_estimates(self) -> tuple[CandidateEstimate, ...]:
        return tuple(
            self.candidate_estimate(size) for size in self.config.candidate_sizes
        )

    def next_probe_size(
        self,
        *,
        history_tokens: int,
        available_draft_tokens: int,
    ) -> int | None:
        """Return the smallest eligible candidate still needing observations."""

        history, available = self._validate_decision_inputs(
            history_tokens, available_draft_tokens
        )
        available = self._capped_available(available)
        if history < int(self.config.min_token_history) or available == 0:
            return None
        if self._classic.observations < int(self.config.min_classic_observations):
            return None
        for candidate in self.config.candidate_sizes:
            if candidate > available:
                break
            observations = self._verification.get(
                candidate, _VerificationMeasurements()
            ).observations
            if observations < int(self.config.min_verify_observations):
                return candidate
        return None

    def decide(
        self,
        *,
        history_tokens: int,
        available_draft_tokens: int,
    ) -> SpeculationDecision:
        history, available = self._validate_decision_inputs(
            history_tokens, available_draft_tokens
        )
        available = self._capped_available(available)
        if history < int(self.config.min_token_history):
            return self._disabled("history_too_short")
        if available == 0:
            return self._disabled("no_draft_tokens")
        if self._classic.observations < int(self.config.min_classic_observations):
            return self._disabled("classic_warmup")

        eligible = [
            estimate
            for estimate in self.candidate_estimates()
            if estimate.candidate_size <= available and estimate.ready
        ]
        if not eligible:
            return self._disabled("verification_warmup")

        # Prefer measured speedup, resolving exact ties toward the smaller and
        # therefore lower-risk verification wave.
        best = max(
            eligible,
            key=lambda estimate: (
                float(estimate.predicted_speedup),
                -estimate.candidate_size,
            ),
        )
        assert best.predicted_speedup is not None
        enabled = best.predicted_speedup > float(self.config.minimum_speedup)
        return SpeculationDecision(
            enabled=enabled,
            candidate_size=best.candidate_size if enabled else 0,
            predicted_speedup=best.predicted_speedup,
            predicted_latency_speedup=best.predicted_latency_speedup,
            predicted_byte_efficiency=best.predicted_byte_efficiency,
            expected_emitted_tokens=best.expected_emitted_tokens,
            reason="beneficial" if enabled else "not_beneficial",
        )

    def _capped_available(self, available: int) -> int:
        if not self.config.delay_adaptive:
            return available
        # Delay cap only bounds depth from above; acceptance evidence still
        # selects the actual draft size.
        return min(available, self.delay_draft_cap())

    def _validate_decision_inputs(
        self, history_tokens: int, available_draft_tokens: int
    ) -> tuple[int, int]:
        history = _validate_count("history_tokens", history_tokens)
        available = _validate_count(
            "available_draft_tokens", available_draft_tokens
        )
        return history, min(available, int(self.config.max_draft_tokens))

    @staticmethod
    def _disabled(reason: str) -> SpeculationDecision:
        return SpeculationDecision(
            enabled=False,
            candidate_size=0,
            predicted_speedup=None,
            predicted_latency_speedup=None,
            predicted_byte_efficiency=None,
            expected_emitted_tokens=None,
            reason=reason,
        )


__all__ = [
    "MAX_DRAFT_TOKENS",
    "AdaptiveSpeculationConfig",
    "AdaptiveSpeculationController",
    "CandidateEstimate",
    "DraftProvider",
    "NgramDraftProvider",
    "SpeculationDecision",
    "SpeculationStats",
]
