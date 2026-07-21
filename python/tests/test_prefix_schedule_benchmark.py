from __future__ import annotations

import unittest

from distributed_runtime.prefix_schedule_benchmark import (
    SCHEMA,
    build_prefix_schedule_benchmark,
)


class PrefixScheduleBenchmarkTests(unittest.TestCase):
    def test_report_is_stable_bounded_and_explicitly_not_wall_clock(self) -> None:
        first = build_prefix_schedule_benchmark()
        second = build_prefix_schedule_benchmark()
        self.assertEqual(first, second)
        self.assertEqual(first["schema"], SCHEMA)
        self.assertEqual(first["evidenceClass"], "deterministic-cpu-planner-only")
        self.assertIn("not wall-clock", first["interpretation"])
        self.assertRegex(first["rowsSha256"], r"^[0-9a-f]{64}$")
        for row in first["rows"]:
            self.assertGreaterEqual(row["savedTokenSteps"], 0)
            self.assertLessEqual(
                row["uniquePrefixTokenSteps"], row["flatLeafTokenSteps"]
            )
            self.assertEqual(row["lanes"], row["leafCount"])

    def test_representative_shared_trunk_reduces_token_step_work_materially(self) -> None:
        report = build_prefix_schedule_benchmark()
        rows = {row["scenario"]: row for row in report["rows"]}
        shared = rows["shared-trunk-8x8"]
        self.assertEqual(shared["flatLeafTokenSteps"], 72)
        self.assertEqual(shared["uniquePrefixTokenSteps"], 23)
        self.assertEqual(shared["savedTokenSteps"], 49)
        self.assertGreater(shared["computeReductionPercent"], 68)
        self.assertEqual(rows["single-leaf-control"]["savedTokenSteps"], 0)


if __name__ == "__main__":
    unittest.main()
