from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import statistics
import tempfile
import unittest

from distributed_runtime.resident_expert_rpc_benchmark import main, run_benchmark


class ResidentExpertRpcBenchmarkTests(unittest.TestCase):
    def test_compare_mode_runs_abba_with_bit_exact_lower_application_frames(self) -> None:
        report = run_benchmark(
            iterations=2,
            warmup=1,
            positions=4,
            hidden_size=64,
            experts=2,
            dtype_name="float16",
            transport="compare",
        )

        self.assertEqual(
            report["schema"],
            "gdlp-resident-expert-rpc-loopback-benchmark/3",
        )
        self.assertEqual(
            report["abbaOrder"],
            ["v1", "coalesced", "coalesced", "v1"],
        )
        self.assertEqual(report["parity"], "bit-exact identity transport")
        self.assertTrue(report["input"]["finite"])
        self.assertEqual(len(report["input"]["sha256"]), 64)
        self.assertLessEqual(report["startedAtUtc"], report["endedAtUtc"])
        self.assertIn("gitWorktreeDirty", report["environment"])
        measurements = report["measurements"]
        self.assertEqual(len(measurements), 8)
        for cycle in range(2):
            self.assertEqual(
                [
                    row["mode"]
                    for row in measurements
                    if row["cycle"] == cycle
                ],
                ["v1", "coalesced", "coalesced", "v1"],
            )
        self.assertTrue(all(row["elapsedNs"] > 0 for row in measurements))
        results = report["results"]
        self.assertEqual(results["v1"]["measuredCalls"], 4)
        self.assertEqual(results["coalesced"]["measuredCalls"], 4)
        self.assertLess(
            results["coalesced"]["meanMeasuredApplicationFrameBytesPerCall"],
            results["v1"]["meanMeasuredApplicationFrameBytesPerCall"],
        )
        self.assertGreater(
            report["comparison"]["meanApplicationFrameReductionPercent"],
            0,
        )
        for mode in ("v1", "coalesced"):
            rows = [row for row in measurements if row["mode"] == mode]
            elapsed_ms = [row["elapsedNs"] / 1_000_000 for row in rows]
            self.assertEqual(
                sum(
                    row["applicationBytesSent"]
                    + row["applicationBytesReceived"]
                    for row in rows
                ),
                results[mode]["measuredApplicationFrameBytes"],
            )
            self.assertAlmostEqual(
                statistics.fmean(elapsed_ms),
                results[mode]["latencyMs"]["mean"],
            )
            self.assertEqual(min(elapsed_ms), results[mode]["latencyMs"]["min"])
            self.assertEqual(max(elapsed_ms), results[mode]["latencyMs"]["max"])

    def test_coalesced_single_mode_uses_v3_with_raw_measurements(self) -> None:
        report = run_benchmark(
            iterations=1,
            warmup=0,
            positions=2,
            hidden_size=4,
            experts=2,
            dtype_name="float16",
            transport="coalesced",
        )

        self.assertEqual(
            report["schema"],
            "gdlp-resident-expert-rpc-loopback-benchmark/3",
        )
        self.assertEqual(report["kind"], "single-transport")
        self.assertEqual(report["configuration"]["transport"], "coalesced")
        self.assertEqual(len(report["measurements"]), 1)
        self.assertEqual(report["measurements"][0]["mode"], "coalesced")
        self.assertEqual(report["measuredCalls"], 1)
        self.assertEqual(report["correctness"]["checkedCalls"], 1)

    def test_cli_default_v1_writes_same_v3_json_as_stdout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "benchmark.json"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                result = main(
                    [
                        "--iterations",
                        "1",
                        "--warmup",
                        "0",
                        "--positions",
                        "2",
                        "--hidden-size",
                        "4",
                        "--experts",
                        "2",
                        "--dtype",
                        "float16",
                        "--json-out",
                        str(output),
                    ]
                )

            self.assertEqual(result, 0)
            stdout_document = json.loads(stdout.getvalue())
            output_document = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(stdout_document, output_document)
            self.assertEqual(
                output_document["schema"],
                "gdlp-resident-expert-rpc-loopback-benchmark/3",
            )
            self.assertEqual(output_document["kind"], "single-transport")
            self.assertEqual(output_document["transport"], "v1")
            self.assertEqual(len(output_document["measurements"]), 1)
            self.assertIn("endedAtUtc", output_document)


if __name__ == "__main__":
    unittest.main()
