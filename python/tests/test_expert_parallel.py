from __future__ import annotations

import copy
from importlib import metadata as importlib_metadata
import os
from pathlib import Path
import tempfile
import unittest

import torch

from distributed_runtime.expert_parallel import (
    CERTIFIED_ACCELERATE_VERSION,
    CERTIFIED_TORCH_VERSION,
    CERTIFIED_TRANSFORMERS_VERSION,
    EXPERT_PARALLEL_CELL_SCHEMA,
    EXPERT_PARALLEL_INSPECTION_SCHEMA,
    QWEN3_MOE_EP_PLAN,
    UnsupportedExpertParallelCellError,
    build_expert_parallel_cell_manifest,
    inspect_expert_parallel_checkpoint,
    runtime_versions,
    validate_expert_parallel_cell_manifest,
)

try:
    import accelerate  # noqa: F401
    import transformers  # noqa: F401

    HAS_HF_EP_RUNTIME = True
except ImportError:
    HAS_HF_EP_RUNTIME = False


@unittest.skipUnless(HAS_HF_EP_RUNTIME, "HF EP runtime packages are required")
class ExpertParallelInspectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from distributed_runtime.expert_parallel_cli import (
            create_tiny_qwen3_moe_checkpoint,
        )

        cls._temporary = tempfile.TemporaryDirectory(prefix="gdlp-ep-unit-")
        cls.checkpoint = create_tiny_qwen3_moe_checkpoint(
            Path(cls._temporary.name) / "checkpoint"
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def test_certified_runtime_has_accelerate_without_replacing_torch(self) -> None:
        torch_version_before = torch.__version__
        versions = runtime_versions()
        self.assertEqual(versions["transformers"], CERTIFIED_TRANSFORMERS_VERSION)
        self.assertEqual(versions["accelerate"], CERTIFIED_ACCELERATE_VERSION)
        self.assertEqual(versions["torch"].split("+", 1)[0], CERTIFIED_TORCH_VERSION)
        self.assertEqual(importlib_metadata.version("accelerate"), "1.14.0")
        self.assertEqual(torch.__version__, torch_version_before)
        self.assertEqual(torch.__version__, "2.13.0+cpu")

    def test_inspector_exposes_plan_divisibility_scope_and_memory(self) -> None:
        inspection = self._inspection()
        document = inspection.to_document()
        self.assertEqual(document["schema"], EXPERT_PARALLEL_INSPECTION_SCHEMA)
        self.assertEqual(document["model"]["modelType"], "qwen3_moe")
        self.assertEqual(document["experts"]["globalCount"], 4)
        self.assertEqual(document["experts"]["localCount"], 2)
        self.assertEqual(document["experts"]["topK"], 2)
        self.assertEqual(document["experts"]["baseModelEpPlan"], QWEN3_MOE_EP_PLAN)
        self.assertEqual(document["distributed"]["backend"], "gloo")
        self.assertEqual(document["distributed"]["deviceKind"], "cpu")
        self.assertEqual(document["distributed"]["worldSize"], 2)
        self.assertEqual(document["distributed"]["scope"], "low-latency-cell-only")
        self.assertFalse(document["distributed"]["wanAllowed"])
        self.assertEqual(document["memory"]["checkpointParameterBytes"], 13_792)
        self.assertEqual(document["memory"]["checkpointExpertBytes"], 6_144)
        self.assertEqual(document["memory"]["projectedRankLocalParameterBytes"], 10_720)
        self.assertIsNone(document["memory"]["rankLocalParameterBytes"])
        self.assertFalse(document["capabilities"]["partialStage"])
        self.assertFalse(document["capabilities"]["stageRunner"])
        self.assertFalse(document["capabilities"]["expertParallelAcrossWan"])

    def test_inspector_fails_closed_on_non_divisible_experts(self) -> None:
        with self.assertRaisesRegex(
            UnsupportedExpertParallelCellError,
            "num_experts must be divisible",
        ):
            inspect_expert_parallel_checkpoint(
                self.checkpoint,
                world_size=3,
                backend="gloo",
                device_kind="cpu",
            )

    def test_inspector_fails_closed_on_backend_device_mismatch(self) -> None:
        with self.assertRaisesRegex(
            UnsupportedExpertParallelCellError,
            "requires backend 'gloo'",
        ):
            inspect_expert_parallel_checkpoint(
                self.checkpoint,
                world_size=2,
                backend="nccl",
                device_kind="cpu",
            )

    def test_manifest_seals_rank_layout_and_disallows_stage_or_wan(self) -> None:
        inspection = self._inspection()
        rank_zero = build_expert_parallel_cell_manifest(
            inspection,
            rank=0,
            device="cpu",
            rank_local_parameter_bytes=10_720,
            rank_local_expert_parameter_bytes=3_072,
        )
        rank_one = build_expert_parallel_cell_manifest(
            inspection,
            rank=1,
            device="cpu",
            rank_local_parameter_bytes=10_720,
            rank_local_expert_parameter_bytes=3_072,
        )
        zero = rank_zero.to_document()
        one = rank_one.to_document()
        self.assertEqual(zero["schema"], EXPERT_PARALLEL_CELL_SCHEMA)
        self.assertEqual(rank_zero.cell_contract_id, rank_one.cell_contract_id)
        self.assertNotEqual(rank_zero.executor_id, rank_one.executor_id)
        self.assertEqual(zero["rank"]["localExpertRange"], {"start": 0, "end": 2})
        self.assertEqual(one["rank"]["localExpertRange"], {"start": 2, "end": 4})
        self.assertFalse(zero["capabilities"]["partialStage"])
        self.assertFalse(zero["capabilities"]["stageRunner"])
        self.assertFalse(zero["capabilities"]["expertParallelAcrossWan"])
        self.assertTrue(zero["capabilities"]["replicatedNonExpertParameters"])
        self.assertEqual(
            validate_expert_parallel_cell_manifest(zero).executor_id,
            rank_zero.executor_id,
        )

        tampered = copy.deepcopy(zero)
        tampered["distributed"]["wanAllowed"] = True
        with self.assertRaisesRegex(ValueError, "cannot cross the WAN"):
            validate_expert_parallel_cell_manifest(tampered)

        tampered = copy.deepcopy(zero)
        tampered["rank"]["localExpertRange"]["end"] = 4
        with self.assertRaisesRegex(ValueError, "local expert range"):
            validate_expert_parallel_cell_manifest(tampered)

        tampered = copy.deepcopy(zero)
        tampered["memory"]["rankLocalParameterBytes"] += 1
        with self.assertRaisesRegex(ValueError, "executor identity"):
            validate_expert_parallel_cell_manifest(tampered)

    def _inspection(self):
        return inspect_expert_parallel_checkpoint(
            self.checkpoint,
            world_size=2,
            backend="gloo",
            device_kind="cpu",
            model_source="gdlp/unit-qwen3-moe",
        )


@unittest.skipUnless(
    HAS_HF_EP_RUNTIME
    and os.environ.get("RUN_DISTRIBUTED_EP_TESTS", "").strip() == "1",
    "set RUN_DISTRIBUTED_EP_TESTS=1 to run the physical HF expert-parallel gate",
)
class ExpertParallelPhysicalTests(unittest.TestCase):
    def test_two_cpu_gloo_ranks_match_monolithic_logits_and_tokens(self) -> None:
        from distributed_runtime.expert_parallel_cli import (
            run_physical_expert_parallel_gate,
        )

        evidence = run_physical_expert_parallel_gate(timeout_seconds=120)
        self.assertEqual(evidence["status"], "pass")
        self.assertEqual(evidence["evidence"], "loopback-physical-multiprocess")
        self.assertEqual(evidence["distributed"]["processes"], 2)
        self.assertEqual(evidence["distributed"]["backend"], "gloo")
        self.assertEqual(evidence["experts"]["ranges"], [[0, 2], [2, 4]])
        self.assertEqual(evidence["parity"]["maxAbsLogitError"], 0.0)
        self.assertTrue(evidence["parity"]["logitsExactlyEqual"])
        self.assertTrue(evidence["parity"]["tokensEqual"])
        self.assertTrue(evidence["parity"]["ranksExactlyEqual"])
        self.assertEqual(evidence["memory"]["denseParameterBytes"], 13_792)
        self.assertEqual(evidence["memory"]["rankLocalParameterBytes"], [10_720, 10_720])
        self.assertEqual(
            evidence["memory"]["rankLocalExpertParameterBytes"],
            [3_072, 3_072],
        )
        self.assertTrue(evidence["limitations"]["fullModelCellOnly"])
        self.assertFalse(evidence["limitations"]["partialStage"])
        self.assertFalse(evidence["limitations"]["stageRunner"])
        self.assertFalse(evidence["limitations"]["expertParallelAcrossWan"])


if __name__ == "__main__":
    unittest.main()
