from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Literal


RECOVERY_EVENT_SCHEMA = "mycellios-recovery-event/1"
RecoveryRole = Literal["root", "head", "middle", "tail", "direct", "relay", "coordinator"]
RecoveryFailureClass = Literal[
    "process-exit", "health-failed", "direct-link-lost", "relay-link-lost",
    "coordinator-lost", "generation-superseded", "checkpoint-corrupt",
    "identity-mismatch", "retry-exhausted",
]
RecoveryCheckpointKind = Literal["none", "stream-offset", "visible-token-prefix", "activation-kv"]
RecoveryReplayScope = Literal["none", "stream-offset", "visible-token-prefix", "full-request"]
RecoveryOutcome = Literal["resume", "exact-replay", "safe-retry", "terminal"]

_ROLES = {"root", "head", "middle", "tail", "direct", "relay", "coordinator"}
_FAILURES = {
    "process-exit", "health-failed", "direct-link-lost", "relay-link-lost",
    "coordinator-lost", "generation-superseded", "checkpoint-corrupt",
    "identity-mismatch", "retry-exhausted",
}
_CHECKPOINTS = {"none", "stream-offset", "visible-token-prefix", "activation-kv"}


@dataclass(frozen=True, slots=True)
class RecoveryFailureObservation:
    generation: int
    active_generation: int
    role: RecoveryRole
    failure_class: RecoveryFailureClass
    checkpoint_kind: RecoveryCheckpointKind
    checkpoint_compatible: bool
    compatible_standby_available: bool
    visible_tokens: int
    downtime_ms: float
    discarded_waves: int = 0
    discarded_bytes: int = 0

    def __post_init__(self) -> None:
        _nonnegative_int(self.generation, "generation")
        _nonnegative_int(self.active_generation, "active_generation")
        if self.generation > self.active_generation:
            raise ValueError("recovery failure generation is from the future")
        if self.role not in _ROLES:
            raise ValueError("recovery role is invalid")
        if self.failure_class not in _FAILURES:
            raise ValueError("recovery failure class is invalid")
        if self.checkpoint_kind not in _CHECKPOINTS:
            raise ValueError("recovery checkpoint kind is invalid")
        if not isinstance(self.checkpoint_compatible, bool):
            raise TypeError("checkpoint_compatible must be boolean")
        if not isinstance(self.compatible_standby_available, bool):
            raise TypeError("compatible_standby_available must be boolean")
        _nonnegative_int(self.visible_tokens, "visible_tokens")
        _nonnegative_float(self.downtime_ms, "downtime_ms")
        _nonnegative_int(self.discarded_waves, "discarded_waves")
        _nonnegative_int(self.discarded_bytes, "discarded_bytes")


@dataclass(frozen=True, slots=True)
class RecoveryEvent:
    generation: int
    role: RecoveryRole
    failure_class: RecoveryFailureClass
    outcome: RecoveryOutcome
    checkpoint_kind: RecoveryCheckpointKind
    replay_scope: RecoveryReplayScope
    downtime_ms: float
    replayed_tokens: int
    discarded_waves: int
    discarded_bytes: int

    def to_document(self) -> dict[str, object]:
        values = asdict(self)
        return {
            "schema": RECOVERY_EVENT_SCHEMA,
            "generation": values["generation"],
            "role": values["role"],
            "failureClass": values["failure_class"],
            "outcome": values["outcome"],
            "checkpointKind": values["checkpoint_kind"],
            "replayScope": values["replay_scope"],
            "downtimeMs": values["downtime_ms"],
            "replayedTokens": values["replayed_tokens"],
            "discardedWaves": values["discarded_waves"],
            "discardedBytes": values["discarded_bytes"],
        }


def classify_recovery_failure(observation: RecoveryFailureObservation) -> RecoveryEvent:
    def terminal(failure_class: RecoveryFailureClass | None = None) -> RecoveryEvent:
        return RecoveryEvent(
            generation=observation.generation,
            role=observation.role,
            failure_class=failure_class or observation.failure_class,
            outcome="terminal",
            checkpoint_kind=observation.checkpoint_kind,
            replay_scope="none",
            downtime_ms=observation.downtime_ms,
            replayed_tokens=0,
            discarded_waves=observation.discarded_waves,
            discarded_bytes=observation.discarded_bytes,
        )

    if observation.generation < observation.active_generation:
        return terminal("generation-superseded")
    if (
        observation.failure_class in {
            "checkpoint-corrupt", "identity-mismatch", "retry-exhausted"
        }
        or not observation.checkpoint_compatible
    ):
        return terminal()
    if observation.checkpoint_kind == "stream-offset":
        outcome: RecoveryOutcome = "resume"
        replay_scope: RecoveryReplayScope = "stream-offset"
        replayed_tokens = 0
    elif (
        observation.checkpoint_kind == "activation-kv"
        and observation.compatible_standby_available
    ):
        outcome = "resume"
        replay_scope = "none"
        replayed_tokens = 0
    elif (
        observation.checkpoint_kind == "visible-token-prefix"
        and observation.compatible_standby_available
        and observation.visible_tokens > 0
    ):
        outcome = "exact-replay"
        replay_scope = "visible-token-prefix"
        replayed_tokens = observation.visible_tokens
    elif observation.visible_tokens == 0 and observation.compatible_standby_available:
        outcome = "safe-retry"
        replay_scope = "full-request"
        replayed_tokens = 0
    else:
        return terminal()
    return RecoveryEvent(
        generation=observation.generation,
        role=observation.role,
        failure_class=observation.failure_class,
        outcome=outcome,
        checkpoint_kind=observation.checkpoint_kind,
        replay_scope=replay_scope,
        downtime_ms=observation.downtime_ms,
        replayed_tokens=replayed_tokens,
        discarded_waves=observation.discarded_waves,
        discarded_bytes=observation.discarded_bytes,
    )


def _nonnegative_int(value: object, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _nonnegative_float(value: object, name: str) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise ValueError(f"{name} must be finite and non-negative")


__all__ = [
    "RECOVERY_EVENT_SCHEMA",
    "RecoveryEvent",
    "RecoveryFailureObservation",
    "classify_recovery_failure",
]
