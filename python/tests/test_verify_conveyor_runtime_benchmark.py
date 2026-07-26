from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from typing import Any

from distributed_runtime.verify_conveyor_runtime_benchmark import (
    ControlledReferenceDraftProvider,
    build_interleaved_order,
    parse_acceptances,
    parse_delays,
    parse_stage_counts,
    parse_windows,
    publish_json_no_overwrite,
    summarize_paired_runs,
    validate_arm_evidence,
)


def _flatten_windows(value: Any) -> list[int]:
    """Normalize order records while keeping the public order API testable."""

    if isinstance(value, dict):
        return [int(value["window"])]
    if hasattr(value, "window"):
        return [int(value.window)]
    if isinstance(value, (tuple, list)):
        flattened: list[int] = []
        for item in value:
            flattened.extend(_flatten_windows(item))
        return flattened
    return [int(value)]


def _valid_arm(*, window: int = 4, byte_cap: int = 4096) -> dict[str, Any]:
    return {
        "pair_id": 0,
        "window": window,
        "token_sha256": "reference-hash",
        "healthy_before_shutdown": True,
        "speculation_stats": {
            "proposed_tokens": 24,
            "accepted_tokens": 20,
            "acceptance_rate": 20 / 24,
            "selected_candidate_sizes": {"4": 6},
        },
        "speculative_window_stats": {
            "configured_waves_per_request": window,
            "configured_bytes_per_request": byte_cap,
            "current_waves": 0,
            "current_bytes": 0,
            "current_reserved_bytes": 0,
            "high_water_waves": 3 if window > 1 else 1,
            "high_water_bytes": 3072 if window > 1 else 0,
            "high_water_reserved_bytes": 3072 if window > 1 else 0,
            "max_request_waves": 3 if window > 1 else 1,
            "max_request_bytes": 3072 if window > 1 else 0,
            "max_request_reserved_bytes": 3072 if window > 1 else 0,
            "discarded_proposed_tokens": 4,
            "discarded_bytes": 512,
        },
        "shutdown": {
            "exact": True,
            "close_error": None,
            "hard_fallback_used": False,
            "cleanup_errors": [],
        },
    }


class VerifyConveyorRuntimeBenchmarkTests(unittest.TestCase):
    def test_parse_windows_requires_a_unique_w1_baseline_and_conveyor_arm(self) -> None:
        self.assertEqual(parse_windows("1,2,16"), (1, 2, 16))

        for raw in ("", "1", "2,4", "1,1", "1,2,2", "0,2", "1,17"):
            with self.subTest(raw=raw), self.assertRaises(argparse.ArgumentTypeError):
                parse_windows(raw)

    def test_parse_sweeps_are_finite_unique_and_in_domain(self) -> None:
        self.assertEqual(parse_delays("0,12.5,50"), (0.0, 12.5, 50.0))
        self.assertEqual(parse_acceptances("0,0.5,1"), (0.0, 0.5, 1.0))
        self.assertEqual(parse_stage_counts("2,3,8"), (2, 3, 8))

        for raw in ("", "-1,0", "0,0", "nan", "inf"):
            with self.subTest(parser="delays", raw=raw):
                with self.assertRaises(argparse.ArgumentTypeError):
                    parse_delays(raw)
        for raw in ("", "-0.1,1", "0,1.1", "0.5,0.5", "nan", "inf"):
            with self.subTest(parser="acceptances", raw=raw):
                with self.assertRaises(argparse.ArgumentTypeError):
                    parse_acceptances(raw)
        for raw in ("", "1,2", "2,2", "0,3", "2.5,3"):
            with self.subTest(parser="stage-counts", raw=raw):
                with self.assertRaises(argparse.ArgumentTypeError):
                    parse_stage_counts(raw)

    def test_interleaved_order_is_deterministic_balanced_and_abba(self) -> None:
        first = build_interleaved_order(windows=(1, 4), rounds=4, seed=23)
        repeated = build_interleaved_order(windows=(1, 4), rounds=4, seed=23)

        self.assertEqual(first, repeated)
        windows = _flatten_windows(first)
        self.assertEqual(windows, [1, 4, 4, 1, 1, 4, 4, 1])
        self.assertEqual(windows.count(1), windows.count(4))
        for offset in range(0, len(windows), 2):
            self.assertEqual(set(windows[offset : offset + 2]), {1, 4})

    def test_reference_provider_alpha_one_returns_the_exact_greedy_suffix(self) -> None:
        provider = ControlledReferenceDraftProvider(
            prompt_tokens=(10, 11),
            reference_tokens=(20, 21, 22, 23, 24),
            max_draft_tokens=3,
            alpha=1.0,
            seed=17,
            vocab_size=128,
        )

        self.assertEqual(provider.draft((10, 11), max_tokens=3), (20, 21, 22))
        self.assertEqual(provider.draft((10, 11, 20), max_tokens=3), (21, 22, 23))
        self.assertEqual(
            provider.draft((10, 11, 20, 21, 22), max_tokens=2),
            (23, 24),
        )

    def test_reference_provider_alpha_zero_is_wrong_but_reproducible(self) -> None:
        arguments = {
            "prompt_tokens": (10, 11),
            "reference_tokens": (20, 21, 22, 23),
            "max_draft_tokens": 4,
            "alpha": 0.0,
            "seed": 19,
            "vocab_size": 128,
        }
        first = ControlledReferenceDraftProvider(**arguments)
        repeated = ControlledReferenceDraftProvider(**arguments)

        first_draft = first.draft((10, 11), max_tokens=4)
        self.assertEqual(first_draft, first.draft((10, 11), max_tokens=4))
        self.assertEqual(first_draft, repeated.draft((10, 11), max_tokens=4))
        self.assertEqual(len(first_draft), 4)
        self.assertTrue(all(0 <= token < 128 for token in first_draft))
        self.assertTrue(
            all(actual != exact for actual, exact in zip(first_draft, (20, 21, 22, 23)))
        )

    def test_arm_evidence_accepts_exact_bounded_and_fully_drained_run(self) -> None:
        evidence = validate_arm_evidence(
            _valid_arm(),
            reference_sha256="reference-hash",
            expected_window=4,
            expected_inflight_bytes=4096,
        )

        self.assertTrue(evidence["exact"])
        self.assertTrue(evidence["token_hash_exact"])
        self.assertTrue(evidence["conveyor_exercised"])
        self.assertTrue(evidence["resource_limits_exact"])
        self.assertTrue(evidence["drained_to_zero"])
        self.assertTrue(evidence["clean_shutdown"])

    def test_arm_evidence_fails_on_hash_or_unexercised_conveyor(self) -> None:
        mismatched = _valid_arm()
        mismatched["token_sha256"] = "different-hash"
        mismatch_evidence = validate_arm_evidence(
            mismatched,
            reference_sha256="reference-hash",
            expected_window=4,
            expected_inflight_bytes=4096,
        )
        self.assertFalse(mismatch_evidence["exact"])
        self.assertFalse(mismatch_evidence["token_hash_exact"])

        idle = _valid_arm()
        idle["speculative_window_stats"]["high_water_waves"] = 1
        idle["speculative_window_stats"]["max_request_waves"] = 1
        idle_evidence = validate_arm_evidence(
            idle,
            reference_sha256="reference-hash",
            expected_window=4,
            expected_inflight_bytes=4096,
        )
        self.assertFalse(idle_evidence["exact"])
        self.assertFalse(idle_evidence["conveyor_exercised"])

    def test_arm_evidence_fails_closed_on_wave_byte_or_cleanup_leaks(self) -> None:
        cases: list[tuple[str, dict[str, Any], str]] = []

        waves = _valid_arm()
        waves["speculative_window_stats"]["high_water_waves"] = 5
        waves["speculative_window_stats"]["max_request_waves"] = 5
        cases.append(("waves", waves, "resource_limits_exact"))

        bytes_over = _valid_arm()
        bytes_over["speculative_window_stats"]["high_water_bytes"] = 4097
        bytes_over["speculative_window_stats"]["high_water_reserved_bytes"] = 4097
        bytes_over["speculative_window_stats"]["max_request_bytes"] = 4097
        bytes_over["speculative_window_stats"]["max_request_reserved_bytes"] = 4097
        cases.append(("bytes", bytes_over, "resource_limits_exact"))

        not_drained = _valid_arm()
        not_drained["speculative_window_stats"]["current_waves"] = 1
        not_drained["speculative_window_stats"]["current_bytes"] = 128
        not_drained["speculative_window_stats"]["current_reserved_bytes"] = 128
        cases.append(("drain", not_drained, "drained_to_zero"))

        dirty_shutdown = _valid_arm()
        dirty_shutdown["shutdown"]["exact"] = False
        dirty_shutdown["shutdown"]["close_error"] = "child did not exit"
        cases.append(("shutdown", dirty_shutdown, "clean_shutdown"))

        for name, arm, failed_check in cases:
            with self.subTest(name=name):
                evidence = validate_arm_evidence(
                    arm,
                    reference_sha256="reference-hash",
                    expected_window=4,
                    expected_inflight_bytes=4096,
                )
                self.assertFalse(evidence["exact"])
                self.assertFalse(evidence[failed_check])

    def test_paired_summary_uses_within_pair_deltas_not_unpaired_means(self) -> None:
        runs = [
            {
                "pair_id": 0,
                "window": 1,
                "ttft_ms": 100.0,
                "tpot_ms": 20.0,
                "total_ms": 200.0,
                "tokens_per_second": 50.0,
            },
            {
                "pair_id": 0,
                "window": 4,
                "ttft_ms": 90.0,
                "tpot_ms": 15.0,
                "total_ms": 160.0,
                "tokens_per_second": 62.5,
            },
            {
                "pair_id": 1,
                "window": 4,
                "ttft_ms": 110.0,
                "tpot_ms": 25.0,
                "total_ms": 230.0,
                "tokens_per_second": 43.0,
            },
            {
                "pair_id": 1,
                "window": 1,
                "ttft_ms": 120.0,
                "tpot_ms": 30.0,
                "total_ms": 260.0,
                "tokens_per_second": 38.0,
            },
        ]

        summary = summarize_paired_runs(
            runs,
            baseline_window=1,
            contender_window=4,
        )

        self.assertEqual(summary["pair_count"], 2)
        self.assertEqual(
            [pair["pair_id"] for pair in summary["pairs"]],
            [0, 1],
        )
        self.assertAlmostEqual(summary["deltas"]["ttft_ms"]["mean"], -10.0)
        self.assertAlmostEqual(summary["deltas"]["tpot_ms"]["mean"], -5.0)
        self.assertAlmostEqual(summary["deltas"]["total_ms"]["mean"], -35.0)
        self.assertAlmostEqual(
            summary["deltas"]["tokens_per_second"]["mean"],
            8.75,
        )
        self.assertGreater(summary["speedup"]["total_ms_geometric_mean"], 1.0)

    def test_paired_summary_rejects_missing_or_duplicate_arms(self) -> None:
        complete = [
            {
                "pair_id": 0,
                "window": 1,
                "ttft_ms": 10.0,
                "tpot_ms": 5.0,
                "total_ms": 20.0,
                "tokens_per_second": 50.0,
            },
            {
                "pair_id": 0,
                "window": 4,
                "ttft_ms": 8.0,
                "tpot_ms": 4.0,
                "total_ms": 16.0,
                "tokens_per_second": 62.5,
            },
        ]
        for bad in (
            complete[:1],
            [*complete, deepcopy(complete[1])],
        ):
            with self.subTest(runs=bad), self.assertRaises(ValueError):
                summarize_paired_runs(
                    bad,
                    baseline_window=1,
                    contender_window=4,
                )

    def test_json_publication_is_atomic_and_never_overwrites(self) -> None:
        payload = {
            "schema": "mycellios-native-verify-conveyor-runtime-ab/1",
            "success": True,
            "label": "medición loopback",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "nested" / "evidence.json"

            published = publish_json_no_overwrite(destination, payload)

            self.assertEqual(Path(published), destination)
            self.assertEqual(
                json.loads(destination.read_text(encoding="utf-8")),
                payload,
            )
            with self.assertRaises(FileExistsError):
                publish_json_no_overwrite(destination, {"success": False})
            self.assertEqual(
                json.loads(destination.read_text(encoding="utf-8")),
                payload,
            )
            self.assertEqual(
                [path for path in destination.parent.iterdir() if path != destination],
                [],
            )


if __name__ == "__main__":
    unittest.main()
