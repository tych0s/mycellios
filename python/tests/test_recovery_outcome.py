from __future__ import annotations

import json
from pathlib import Path
import unittest

from distributed_runtime.recovery_outcome import (
    RECOVERY_EVENT_SCHEMA,
    RecoveryFailureObservation,
    classify_recovery_failure,
)


class RecoveryOutcomeTests(unittest.TestCase):
    def test_shared_typescript_python_vectors_are_identical(self) -> None:
        fixture = Path(__file__).parents[2] / "tests" / "fixtures" / "recovery-outcomes.json"
        vectors = json.loads(fixture.read_text(encoding="utf-8"))
        for vector in vectors:
            value = vector["observation"]
            observation = RecoveryFailureObservation(
                generation=value["generation"],
                active_generation=value["activeGeneration"],
                role=value["role"],
                failure_class=value["failureClass"],
                checkpoint_kind=value["checkpointKind"],
                checkpoint_compatible=value["checkpointCompatible"],
                compatible_standby_available=value["compatibleStandbyAvailable"],
                visible_tokens=value["visibleTokens"],
                downtime_ms=value["downtimeMs"],
                discarded_waves=value["discardedWaves"],
                discarded_bytes=value["discardedBytes"],
            )
            with self.subTest(vector=vector["name"]):
                self.assertEqual(
                    classify_recovery_failure(observation).to_document(),
                    vector["event"],
                )

    def test_exact_visible_prefix_replay_matches_receipt_document(self) -> None:
        event = classify_recovery_failure(RecoveryFailureObservation(
            generation=7,
            active_generation=7,
            role="middle",
            failure_class="process-exit",
            checkpoint_kind="visible-token-prefix",
            checkpoint_compatible=True,
            compatible_standby_available=True,
            visible_tokens=13,
            downtime_ms=42.5,
            discarded_waves=2,
            discarded_bytes=4096,
        )).to_document()
        self.assertEqual(event, {
            "schema": RECOVERY_EVENT_SCHEMA,
            "generation": 7,
            "role": "middle",
            "failureClass": "process-exit",
            "outcome": "exact-replay",
            "checkpointKind": "visible-token-prefix",
            "replayScope": "visible-token-prefix",
            "downtimeMs": 42.5,
            "replayedTokens": 13,
            "discardedWaves": 2,
            "discardedBytes": 4096,
        })

    def test_superseded_generation_and_incompatible_checkpoint_fail_closed(self) -> None:
        base = dict(
            generation=4,
            active_generation=5,
            role="tail",
            failure_class="health-failed",
            checkpoint_kind="visible-token-prefix",
            checkpoint_compatible=True,
            compatible_standby_available=True,
            visible_tokens=3,
            downtime_ms=1.0,
        )
        superseded = classify_recovery_failure(
            RecoveryFailureObservation(**base)
        )
        self.assertEqual(superseded.outcome, "terminal")
        self.assertEqual(superseded.failure_class, "generation-superseded")
        incompatible = classify_recovery_failure(RecoveryFailureObservation(
            **{**base, "generation": 5, "checkpoint_compatible": False}
        ))
        self.assertEqual(incompatible.outcome, "terminal")
        self.assertEqual(incompatible.replay_scope, "none")

    def test_future_generation_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "from the future"):
            RecoveryFailureObservation(
                generation=2,
                active_generation=1,
                role="root",
                failure_class="health-failed",
                checkpoint_kind="none",
                checkpoint_compatible=True,
                compatible_standby_available=False,
                visible_tokens=0,
                downtime_ms=0,
            )


if __name__ == "__main__":
    unittest.main()
