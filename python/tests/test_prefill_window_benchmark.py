from __future__ import annotations

import argparse
from types import SimpleNamespace
import unittest

from distributed_runtime.prefill_window_benchmark import (
    LOOPBACK_EVIDENCE,
    SIMULATION_EVIDENCE,
    _layer_boundaries,
    _token_parity,
    _window_exercise,
    _shutdown_evidence,
    balanced_closed_form_ms,
    build_simulation_report,
    parse_args,
    parse_service_ms,
    parse_windows,
    simulate_pipeline,
    stop_and_wait_closed_form_ms,
    unlimited_window_closed_form_ms,
)


class PrefillWindowSimulationTests(unittest.TestCase):
    def test_balanced_pipeline_matches_all_three_closed_forms(self) -> None:
        services = (20.0, 20.0, 20.0)
        stop_and_wait = simulate_pipeline(
            chunks=12,
            chunk_tokens=16,
            service_ms=services,
            window=1,
        )
        full = simulate_pipeline(
            chunks=12,
            chunk_tokens=16,
            service_ms=services,
            window=3,
        )

        self.assertEqual(stop_and_wait.ttft_ms, 720.0)
        self.assertEqual(full.ttft_ms, 280.0)
        self.assertEqual(
            stop_and_wait.ttft_ms,
            stop_and_wait_closed_form_ms(12, services),
        )
        self.assertEqual(
            full.ttft_ms,
            unlimited_window_closed_form_ms(12, services),
        )
        for window in (1, 2, 3, 8):
            with self.subTest(window=window):
                simulated = simulate_pipeline(
                    chunks=12,
                    chunk_tokens=16,
                    service_ms=services,
                    window=window,
                )
                self.assertEqual(
                    simulated.ttft_ms,
                    balanced_closed_form_ms(
                        chunks=12,
                        stages=3,
                        service_ms=20.0,
                        window=window,
                    ),
                )

    def test_bottleneck_limits_the_credit_window_gain(self) -> None:
        services = (8.0, 44.0, 8.0)
        stop_and_wait = simulate_pipeline(
            chunks=12,
            chunk_tokens=16,
            service_ms=services,
            window=1,
        )
        window_three = simulate_pipeline(
            chunks=12,
            chunk_tokens=16,
            service_ms=services,
            window=3,
        )

        self.assertEqual(stop_and_wait.ttft_ms, 720.0)
        self.assertEqual(window_three.ttft_ms, 544.0)
        self.assertAlmostEqual(stop_and_wait.ttft_ms / window_three.ttft_ms, 1.3235294118)
        self.assertLess(
            stop_and_wait.ttft_ms / window_three.ttft_ms,
            720.0 / 280.0,
        )

    def test_report_labels_simulation_and_never_claims_physical_evidence(self) -> None:
        report = build_simulation_report(
            chunks=12,
            chunk_tokens=16,
            windows=(1, 3),
            balanced_services_ms=(20, 20, 20),
            bottleneck_services_ms=(8, 44, 8),
        )

        self.assertEqual(report["evidence_class"], SIMULATION_EVIDENCE)
        self.assertFalse(report["physical_multi_pc"])
        self.assertFalse(report["physical_gpu"])
        balanced, bottleneck = report["scenarios"]
        self.assertEqual(balanced["windows"][1]["ttft_ms"], 280.0)
        self.assertEqual(bottleneck["windows"][1]["ttft_ms"], 544.0)

    def test_token_parity_is_exact_and_reports_every_mismatch(self) -> None:
        exact = _token_parity(
            (
                {"samples": ({"token_ids": [10, 20]},)},
                {"samples": ({"token_ids": [10, 20]},)},
            )
        )
        mismatch = _token_parity(
            (
                {"samples": ({"token_ids": [10, 20]},)},
                {"samples": ({"token_ids": [10, 21]},)},
            )
        )

        self.assertTrue(exact["exact"])
        self.assertFalse(mismatch["exact"])
        self.assertEqual(mismatch["mismatches"], [{"sample": 1, "token_ids": [10, 21]}])

    def test_window_must_reach_its_sealed_credit_to_count(self) -> None:
        exact = _window_exercise(
            window=3,
            prompt_tokens=192,
            chunk_tokens=16,
            prefill_inflight_bytes=1024,
            stats={
                "max_request_chunks": 3,
                "configured_bytes_per_request": 1024,
                "max_request_reserved_bytes": 900,
            },
        )
        unused = _window_exercise(
            window=3,
            prompt_tokens=192,
            chunk_tokens=16,
            prefill_inflight_bytes=1024,
            stats={
                "max_request_chunks": 2,
                "configured_bytes_per_request": 1024,
                "max_request_reserved_bytes": 900,
            },
        )
        short_prompt = _window_exercise(
            window=3,
            prompt_tokens=16,
            chunk_tokens=16,
            prefill_inflight_bytes=1024,
            stats={
                "max_request_chunks": 1,
                "configured_bytes_per_request": 1024,
                "max_request_reserved_bytes": 900,
            },
        )
        zero_cap = _window_exercise(
            window=3,
            prompt_tokens=192,
            chunk_tokens=16,
            prefill_inflight_bytes=0,
            stats={
                "max_request_chunks": 3,
                "configured_bytes_per_request": 0,
                "max_request_reserved_bytes": 900,
            },
        )
        overflow = _window_exercise(
            window=3,
            prompt_tokens=192,
            chunk_tokens=16,
            prefill_inflight_bytes=1024,
            stats={
                "max_request_chunks": 3,
                "configured_bytes_per_request": 1024,
                "max_request_reserved_bytes": 1025,
            },
        )

        self.assertTrue(exact["exact"])
        self.assertFalse(unused["exact"])
        self.assertTrue(short_prompt["exact"])
        self.assertFalse(zero_cap["exact"])
        self.assertFalse(overflow["exact"])

    def test_runtime_shutdown_requires_zero_exit_codes_and_no_fatal_metrics(self) -> None:
        clean = _shutdown_evidence(
            SimpleNamespace(
                _processes=(SimpleNamespace(name="stage-1", pid=10, exitcode=0),),
                stage_metrics=(),
            )
        )
        crashed = _shutdown_evidence(
            SimpleNamespace(
                _processes=(SimpleNamespace(name="stage-1", pid=10, exitcode=1),),
                stage_metrics=({"fatal_error": "connection reset"},),
            )
        )
        missing = _shutdown_evidence(SimpleNamespace(_processes=(), stage_metrics=()))

        self.assertTrue(clean["exact"])
        self.assertFalse(crashed["exact"])
        self.assertFalse(missing["exact"])

    def test_boundary_scenarios_cover_every_layer_without_overlap(self) -> None:
        balanced = _layer_boundaries(30, 3, bottleneck=False)
        bottleneck = _layer_boundaries(30, 3, bottleneck=True)

        self.assertEqual(balanced, (0, 10, 20, 30))
        self.assertEqual(bottleneck[0], 0)
        self.assertEqual(bottleneck[-1], 30)
        counts = tuple(right - left for left, right in zip(bottleneck, bottleneck[1:]))
        self.assertEqual(sum(counts), 30)
        self.assertGreater(counts[1], counts[0])
        self.assertGreater(counts[1], counts[2])

    def test_cli_parsers_are_fail_closed(self) -> None:
        self.assertEqual(parse_windows("1,3"), (1, 3))
        self.assertEqual(parse_service_ms("8,44,8"), (8.0, 44.0, 8.0))
        for raw in ("", "2,3", "1,1", "1,65"):
            with self.subTest(windows=raw), self.assertRaises(argparse.ArgumentTypeError):
                parse_windows(raw)
        for raw in ("", "20", "20,0", "20,nan"):
            with self.subTest(services=raw), self.assertRaises(argparse.ArgumentTypeError):
                parse_service_ms(raw)
        with self.assertRaises(SystemExit):
            parse_args(["--runtime-scenarios", "inventado"])

    def test_loopback_evidence_label_is_unambiguous(self) -> None:
        self.assertEqual(LOOPBACK_EVIDENCE, "MEDIDO_EN_LOOPBACK_CON_RED_EMULADA")


if __name__ == "__main__":
    unittest.main()
