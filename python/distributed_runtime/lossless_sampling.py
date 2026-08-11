"""Distribution-preserving speculative sampling primitives.

The target distribution is always authoritative. A proposed draft token is
accepted with ``min(1, p(x) / q(x))``; rejection samples from normalized
``max(p - q, 0)``. Counter-based randomness makes checkpoint/rollback exact.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from numbers import Integral, Real
from typing import Sequence


MAX_VOCABULARY = 1_000_000
RNG_SCHEMA = "mycellios-sampling-rng/1"
_DOMAIN = b"mycellios-lossless-sampling-rng/1\0"
_TWO_TO_53 = 1 << 53


@dataclass(frozen=True)
class SamplingRngCheckpoint:
    schema: str
    seed_hex: str
    counter: int


class CounterSamplingRng:
    """Portable SHA-256 counter stream; never relies on process-global RNG."""

    def __init__(self, seed: bytes, counter: int = 0) -> None:
        if not isinstance(seed, bytes) or len(seed) != 32:
            raise ValueError("sampling RNG seed must be exactly 32 bytes")
        if (
            not isinstance(counter, Integral)
            or isinstance(counter, bool)
            or not 0 <= int(counter) < 2**64
        ):
            raise ValueError("sampling RNG counter must be uint64")
        self._seed = seed
        self._counter = int(counter)

    @classmethod
    def from_checkpoint(cls, value: SamplingRngCheckpoint) -> "CounterSamplingRng":
        if not isinstance(value, SamplingRngCheckpoint) or value.schema != RNG_SCHEMA:
            raise ValueError("sampling RNG checkpoint is invalid")
        try:
            seed = bytes.fromhex(value.seed_hex)
        except ValueError as error:
            raise ValueError("sampling RNG checkpoint seed is invalid") from error
        if seed.hex() != value.seed_hex:
            raise ValueError("sampling RNG checkpoint seed is not canonical")
        return cls(seed, value.counter)

    def checkpoint(self) -> SamplingRngCheckpoint:
        return SamplingRngCheckpoint(RNG_SCHEMA, self._seed.hex(), self._counter)

    def uniform(self) -> float:
        if self._counter >= 2**64:
            raise OverflowError("sampling RNG counter exhausted")
        block = hashlib.sha256(
            _DOMAIN + self._seed + self._counter.to_bytes(8, "big")
        ).digest()
        self._counter += 1
        return (int.from_bytes(block[:8], "big") >> 11) / _TWO_TO_53

    def categorical(self, probabilities: Sequence[float]) -> int:
        distribution = _validate_distribution(probabilities)
        draw = self.uniform()
        cumulative = 0.0
        for index, probability in enumerate(distribution):
            cumulative += probability
            if draw < cumulative:
                return index
        return len(distribution) - 1


@dataclass(frozen=True)
class LosslessSamplingDecision:
    token_id: int
    accepted_draft: bool
    acceptance_probability: float


def probabilities_from_logits(
    logits: Sequence[float],
    *,
    temperature: float,
    top_p: float = 1.0,
) -> tuple[float, ...]:
    values = _finite_vector(logits, "sampling logits")
    if not isinstance(temperature, Real) or isinstance(temperature, bool):
        raise ValueError("sampling temperature must be finite and positive")
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("sampling temperature must be finite and positive")
    if not isinstance(top_p, Real) or isinstance(top_p, bool):
        raise ValueError("sampling top_p must be in (0, 1]")
    top_p = float(top_p)
    if not math.isfinite(top_p) or not 0.0 < top_p <= 1.0:
        raise ValueError("sampling top_p must be in (0, 1]")
    scaled = tuple(value / temperature for value in values)
    maximum = max(scaled)
    weights = tuple(math.exp(value - maximum) for value in scaled)
    total = math.fsum(weights)
    probabilities = tuple(weight / total for weight in weights)
    if top_p == 1.0:
        return probabilities
    ordered = sorted(range(len(probabilities)), key=lambda index: (-probabilities[index], index))
    selected: list[int] = []
    cumulative = 0.0
    for index in ordered:
        selected.append(index)
        cumulative += probabilities[index]
        if cumulative >= top_p:
            break
    selected_total = math.fsum(probabilities[index] for index in selected)
    selected_set = frozenset(selected)
    return tuple(
        probability / selected_total if index in selected_set else 0.0
        for index, probability in enumerate(probabilities)
    )


def lossless_speculative_sample(
    target_probabilities: Sequence[float],
    draft_probabilities: Sequence[float],
    proposed_token_id: int,
    rng: CounterSamplingRng,
) -> LosslessSamplingDecision:
    target = _validate_distribution(target_probabilities)
    draft = _validate_distribution(draft_probabilities)
    if len(target) != len(draft):
        raise ValueError("target and draft vocabulary sizes differ")
    if (
        not isinstance(proposed_token_id, Integral)
        or isinstance(proposed_token_id, bool)
        or not 0 <= int(proposed_token_id) < len(target)
    ):
        raise ValueError("proposed token id is outside the vocabulary")
    token_id = int(proposed_token_id)
    proposal_probability = draft[token_id]
    if proposal_probability <= 0.0:
        raise ValueError("proposed token has zero draft probability")
    acceptance = min(1.0, target[token_id] / proposal_probability)
    if rng.uniform() < acceptance:
        return LosslessSamplingDecision(token_id, True, acceptance)
    residual = tuple(max(0.0, p - q) for p, q in zip(target, draft, strict=True))
    residual_total = math.fsum(residual)
    if residual_total <= 0.0:
        raise RuntimeError("lossless sampling residual is empty after rejection")
    corrected = tuple(value / residual_total for value in residual)
    return LosslessSamplingDecision(rng.categorical(corrected), False, acceptance)


def _finite_vector(values: Sequence[float], label: str) -> tuple[float, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"{label} must be a sequence")
    if not 1 <= len(values) <= MAX_VOCABULARY:
        raise ValueError(f"{label} size is invalid")
    result: list[float] = []
    for value in values:
        if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(float(value)):
            raise ValueError(f"{label} must contain only finite numbers")
        result.append(float(value))
    return tuple(result)


def _validate_distribution(values: Sequence[float]) -> tuple[float, ...]:
    distribution = _finite_vector(values, "sampling distribution")
    if any(value < 0.0 for value in distribution):
        raise ValueError("sampling distribution cannot contain negative values")
    total = math.fsum(distribution)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("sampling distribution total must be positive")
    normalized = tuple(value / total for value in distribution)
    if abs(math.fsum(normalized) - 1.0) > 1e-12:
        raise ValueError("sampling distribution normalization is unstable")
    return normalized


__all__ = [
    "CounterSamplingRng",
    "LosslessSamplingDecision",
    "RNG_SCHEMA",
    "SamplingRngCheckpoint",
    "lossless_speculative_sample",
    "probabilities_from_logits",
]
