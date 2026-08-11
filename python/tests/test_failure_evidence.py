from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import unittest

from distributed_runtime.engine import DistributedPipelineEngine
from distributed_runtime.failure_evidence import StageFailureEvidence
from distributed_runtime.model import StageModelSpec
from distributed_runtime.protocol import TensorCodec
from distributed_runtime.stage import (
    StageProcessConfig,
    stage_failure_payload,
    validate_stage_config,
)


class StageFailureEvidenceTests(unittest.TestCase):
    def test_canonical_evidence_round_trips_and_rejects_tampering(self) -> None:
        evidence = StageFailureEvidence(
            generation=7,
            route_id="route-a",
            stage_role="middle",
            stage_index=2,
            executor_id="a" * 32,
            layer_start=4,
            layer_end=8,
            failure_class="health-failed",
            error_type="TimeoutError",
            message="downstream timed out",
        )
        payload = evidence.to_payload()
        self.assertEqual(StageFailureEvidence.parse_payload(payload), evidence)
        with self.assertRaisesRegex(ValueError, "canonical"):
            StageFailureEvidence.parse_payload(payload + b" ")
        mutated = payload.replace(b'"generation":7', b'"generation":8')
        self.assertNotEqual(
            StageFailureEvidence.parse_payload(mutated).generation,
            evidence.generation,
        )

    def test_stage_config_requires_complete_position_bound_identity(self) -> None:
        config = _tail_config()
        validate_stage_config(config)
        payload = stage_failure_payload(config, TimeoutError("lost return"))
        evidence = StageFailureEvidence.parse_payload(payload)
        self.assertEqual(evidence.stage_role, "tail")
        self.assertEqual(evidence.stage_index, 2)
        self.assertEqual(evidence.executor_id, "b" * 32)
        with self.assertRaisesRegex(ValueError, "configured completely"):
            validate_stage_config(replace(config, stage_executor_id=None))
        with self.assertRaisesRegex(ValueError, "physical stage position"):
            validate_stage_config(replace(config, stage_role="head"))

    def test_root_rejects_cross_route_generation_executor_role_and_range(self) -> None:
        engine = DistributedPipelineEngine.__new__(DistributedPipelineEngine)
        engine.config = SimpleNamespace(
            route_id="route-a",
            boundaries=(0, 4, 8, 12),
            stage_executor_ids=("a" * 32, "b" * 32, "c" * 32),
        )
        frame = SimpleNamespace(deployment_generation=7)
        evidence = StageFailureEvidence(
            generation=7,
            route_id="route-a",
            stage_role="tail",
            stage_index=2,
            executor_id="c" * 32,
            layer_start=8,
            layer_end=12,
            failure_class="health-failed",
            error_type="TimeoutError",
            message="lost return",
        )
        engine._validate_stage_failure_evidence(evidence, frame)
        cases = (
            (replace(evidence, generation=8), "generation"),
            (replace(evidence, route_id="route-b"), "route"),
            (replace(evidence, executor_id="d" * 32), "executor"),
            (replace(evidence, stage_role="middle"), "role"),
            (replace(evidence, layer_start=7), "layer range"),
        )
        for invalid, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                engine._validate_stage_failure_evidence(invalid, frame)


def _tail_config() -> StageProcessConfig:
    return StageProcessConfig(
        spec=StageModelSpec("fake", 4, 8, 8, 1),
        pipeline_id=1,
        listen_host="127.0.0.1",
        listen_port=20_001,
        next_host=None,
        next_port=None,
        next_layer_end=None,
        return_host="127.0.0.1",
        return_port=20_002,
        codec=TensorCodec.FP16,
        one_way_delay_ms=0,
        bandwidth_mbps=0,
        deployment_generation=7,
        route_id="route-a",
        stage_index=2,
        stage_role="tail",
        stage_executor_id="b" * 32,
    )


if __name__ == "__main__":
    unittest.main()
