from __future__ import annotations

from types import SimpleNamespace
import unittest

from distributed_runtime.engine import PipelineEngineConfig
from distributed_runtime.stage import validate_speculative_runner


class PhysicalTreeLimitWiringTests(unittest.TestCase):
    def test_engine_limits_are_disabled_by_default_or_enabled_as_one_contract(self) -> None:
        common = {"model_name": "fixture", "boundaries": (0, 2, 4)}
        disabled = PipelineEngineConfig(**common)
        self.assertEqual(disabled.max_speculative_branches, 0)
        self.assertEqual(disabled.max_speculative_branch_tokens, 0)
        self.assertEqual(disabled.max_speculative_kv_bytes, 0)

        enabled = PipelineEngineConfig(
            **common,
            max_speculative_branches=8,
            max_speculative_branch_tokens=32_768,
            max_speculative_kv_bytes=512 * 1024 * 1024,
        )
        self.assertEqual(enabled.max_speculative_branches, 8)
        self.assertEqual(enabled.max_speculative_branch_tokens, 32_768)
        self.assertEqual(enabled.max_speculative_kv_bytes, 512 * 1024 * 1024)

        for updates in (
            {"max_speculative_branches": 1},
            {
                "max_speculative_branches": 65,
                "max_speculative_branch_tokens": 1,
                "max_speculative_kv_bytes": 1,
            },
            {
                "max_speculative_branches": 1,
                "max_speculative_branch_tokens": 1_048_577,
                "max_speculative_kv_bytes": 1,
            },
            {
                "max_speculative_branches": 1,
                "max_speculative_branch_tokens": 1,
                "max_speculative_kv_bytes": (1 << 40) + 1,
            },
        ):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                PipelineEngineConfig(**common, **updates)

    def test_single_request_backend_is_rejected_before_stage_ready(self) -> None:
        enabled = PipelineEngineConfig(
            model_name="fixture",
            boundaries=(0, 2, 4),
            max_speculative_branches=2,
            max_speculative_branch_tokens=128,
            max_speculative_kv_bytes=1024,
        )
        runner = SimpleNamespace(max_active_requests=1)
        with self.assertRaisesRegex(ValueError, "multi-request"):
            # run_stage_process and the root engine both call this immediately
            # after constructing a backend and before accepting pipeline READY.
            validate_speculative_runner(enabled, runner)


if __name__ == "__main__":
    unittest.main()
