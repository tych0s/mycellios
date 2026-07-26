from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import patch

from distributed_runtime.draft_quality_runtime_benchmark import (
    ARM_NAMES,
    DEFAULT_K_SWEEP,
    MIN_MEASURED_REQUESTS_PER_ARM,
    aggregate_arm_runs,
    allocate_requests,
    build_balanced_arm_orders,
    classify_quality_gate,
    classify_sweep_quality_gate,
    counter_delta,
    derive_request_metrics,
    load_corpus,
    one_sided_mean_interval,
    parse_args,
    parse_k_sweep,
    publish_json_no_overwrite,
    run,
    validate_run_evidence,
)


def _snapshot(
    *,
    classic: int,
    verification: int,
    proposed: int,
    accepted: int,
    completed: int,
    discarded: int = 0,
) -> dict[str, dict[str, int]]:
    return {
        "speculation": {
            "classic_observations": classic,
            "verification_observations": verification,
            "proposed_tokens": proposed,
            "accepted_tokens": accepted,
        },
        "window": {
            "dispatched_waves": completed,
            "completed_waves": completed,
            "committed_waves": completed,
            "condemned_waves": 0,
            "drained_waves": 0,
            "rejection_collapses": 0,
            "rejected_proposed_tokens": 0,
            "condemned_proposed_tokens": 0,
            "discarded_proposed_tokens": discarded,
            "rejected_wave_bytes": 0,
            "tombstone_bytes": 0,
            "discarded_bytes": discarded * 16,
            "current_waves": 0,
            "current_bytes": 0,
            "current_reserved_bytes": 0,
        },
    }


def _sample(
    *,
    g: float | None = 3.5,
    exact: bool = True,
    proposed: int = 4,
    accepted: int = 3,
    completed: int = 1,
    classic: int = 0,
) -> dict[str, Any]:
    traversals = completed + classic
    return {
        "token_hash_exact": exact,
        "drained_after_request": True,
        "ttft_ms": 10.0,
        "tpot_ms": 5.0,
        "total_ms": 100.0,
        "tokens_per_second": 40.0,
        "g_route": g,
        "counter_deltas": {
            "proposed_tokens": proposed,
            "accepted_tokens": accepted,
            "completed_waves": completed,
            "classic_observations": classic,
        }
        if g is not None
        else {},
        "discard_deltas": {
            "discarded_proposed_tokens": 1 if proposed else 0,
            "discarded_bytes": 16 if proposed else 0,
        }
        if g is not None
        else {},
        "route_traversals": traversals if g is not None else None,
    }


def _valid_run(
    arm: str = "draft_w1",
    *,
    requests: int = 4,
    window: int | None = None,
    byte_cap: int | None = None,
) -> dict[str, Any]:
    if window is None:
        window = 4 if arm == "draft_conveyor" else 1
    if byte_cap is None:
        byte_cap = 4096 if arm == "draft_conveyor" else 0
    speculative = arm != "classic"
    high_water = 2 if arm == "draft_conveyor" else (1 if speculative else 0)
    provider = (
        {"strategy": "ngram", "maxDraftTokens": 4}
        if speculative
        else None
    )
    observed_bytes = (
        min(byte_cap, 2048)
        if byte_cap > 0
        else (11552 if speculative else 0)
    )
    return {
        "arm": arm,
        "samples": [
            _sample(g=3.5 if speculative else None, proposed=4 if speculative else 0)
            for _ in range(requests)
        ],
        "warmups": [],
        "healthy_before_shutdown": True,
        "speculation_stats": {
            "configured": speculative,
            "proposed_tokens": requests * 4 if speculative else 0,
            "accepted_tokens": requests * 3 if speculative else 0,
            "provider": provider,
        },
        "speculative_window_stats": {
            "configured_waves_per_request": window,
            "configured_bytes_per_request": byte_cap,
            "current_waves": 0,
            "current_bytes": 0,
            "current_reserved_bytes": 0,
            "high_water_waves": high_water,
            "high_water_bytes": observed_bytes,
            "high_water_reserved_bytes": observed_bytes,
            "max_request_waves": high_water,
            "max_request_bytes": observed_bytes,
            "max_request_reserved_bytes": observed_bytes,
            "completed_waves": requests if speculative else 0,
            "dispatched_waves": requests if speculative else 0,
        },
        "shutdown": {
            "exact": True,
            "close_error": None,
            "hard_fallback_used": False,
            "cleanup_errors": [],
        },
    }


class DraftQualityRuntimeBenchmarkTests(unittest.TestCase):
    def test_k_sweep_defaults_and_validation(self) -> None:
        self.assertEqual(parse_k_sweep("1,2,4,8,16"), DEFAULT_K_SWEEP)
        for raw in ("", "0,1", "1,17", "1,1", "1.5,2"):
            with self.subTest(raw=raw), self.assertRaises(
                argparse.ArgumentTypeError
            ):
                parse_k_sweep(raw)

    def test_balanced_orders_are_mirrored_and_pairwise_balanced(self) -> None:
        orders = build_balanced_arm_orders(repetitions=4, seed=29)
        self.assertEqual(len(orders), 4)
        for index in range(0, len(orders), 2):
            forward = orders[index]
            reverse = orders[index + 1]
            self.assertEqual(set(forward), set(ARM_NAMES))
            self.assertEqual(reverse, tuple(reversed(forward)))
            positions = {
                arm: forward.index(arm) + reverse.index(arm)
                for arm in ARM_NAMES
            }
            self.assertEqual(set(positions.values()), {2})

        for invalid in (0, 1, 3):
            with self.subTest(repetitions=invalid), self.assertRaises(ValueError):
                build_balanced_arm_orders(repetitions=invalid, seed=1)

    def test_request_allocation_preserves_at_least_seven_samples(self) -> None:
        self.assertEqual(allocate_requests(7, 2), (4, 3))
        self.assertEqual(allocate_requests(9, 4), (3, 2, 2, 2))
        with self.assertRaises(ValueError):
            allocate_requests(6, 2)
        with self.assertRaises(ValueError):
            allocate_requests(7, 8)

    def test_corpus_json_is_bounded_normalized_and_sealed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.json"
            path.write_text(
                json.dumps({"prompts": ["first", "segundo"]}),
                encoding="utf-8",
            )
            prompts, evidence = load_corpus(path)
            self.assertEqual(prompts, ("first", "segundo"))
            self.assertEqual(evidence["count"], 2)
            self.assertRegex(evidence["sha256"], r"^sha256:[0-9a-f]{64}$")

            path.write_text(json.dumps([" padded "]), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_corpus(path)

    def test_counter_delta_fails_if_a_counter_resets(self) -> None:
        self.assertEqual(counter_delta({"x": 2}, {"x": 7}, "x"), 5)
        with self.assertRaisesRegex(ValueError, "decreased"):
            counter_delta({"x": 7}, {"x": 2}, "x")
        for value in (True, -1, 1.5, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                counter_delta({"x": 0}, {"x": value}, "x")

    def test_g_route_uses_physical_waves_plus_classic_not_acceptance(self) -> None:
        before = _snapshot(
            classic=10,
            verification=10,
            proposed=50,
            accepted=40,
            completed=7,
        )
        after = _snapshot(
            classic=11,
            verification=110,
            proposed=150,
            accepted=140,
            completed=9,
            discarded=4,
        )

        metrics = derive_request_metrics(
            before,
            after,
            output_token_count=10,
            speculative=True,
        )

        # useful=9 and route traversals=2 completed waves + 1 classic.
        self.assertEqual(metrics["route_traversals"], 3)
        self.assertEqual(metrics["g_route"], 3.0)
        self.assertEqual(metrics["acceptance_rate"], 1.0)
        self.assertAlmostEqual(metrics["draft_route_coverage"], 2 / 3)
        self.assertEqual(metrics["counter_deltas"]["accepted_tokens"], 100)
        self.assertEqual(metrics["discard_deltas"]["discarded_proposed_tokens"], 4)

    def test_g_route_fails_closed_without_a_traversal_or_on_bad_acceptance(self) -> None:
        empty = _snapshot(
            classic=0,
            verification=0,
            proposed=0,
            accepted=0,
            completed=0,
        )
        with self.assertRaisesRegex(ValueError, "without an observed traversal"):
            derive_request_metrics(
                empty,
                empty,
                output_token_count=4,
                speculative=True,
            )

        bad = deepcopy(empty)
        bad["speculation"]["proposed_tokens"] = 1
        bad["speculation"]["accepted_tokens"] = 2
        bad["window"]["completed_waves"] = 1
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            derive_request_metrics(
                empty,
                bad,
                output_token_count=4,
                speculative=True,
            )

    def test_classic_request_does_not_invent_a_route_g(self) -> None:
        empty = _snapshot(
            classic=0,
            verification=0,
            proposed=0,
            accepted=0,
            completed=0,
        )
        metrics = derive_request_metrics(
            empty,
            empty,
            output_token_count=8,
            speculative=False,
        )
        self.assertEqual(metrics["useful_output_tokens"], 7)
        self.assertIsNone(metrics["route_traversals"])
        self.assertIsNone(metrics["g_route"])

    def test_student_gate_has_go_stop_gray_and_invalid_regions(self) -> None:
        go = classify_quality_gate([3.5] * MIN_MEASURED_REQUESTS_PER_ARM)
        stop = classify_quality_gate([1.5] * MIN_MEASURED_REQUESTS_PER_ARM)
        gray = classify_quality_gate([2.5] * MIN_MEASURED_REQUESTS_PER_ARM)
        invalid = classify_quality_gate([4.0] * 6)
        inexact = classify_quality_gate(
            [4.0] * MIN_MEASURED_REQUESTS_PER_ARM,
            evidence_exact=False,
        )

        self.assertEqual(go["outcome"], "GO")
        self.assertEqual(stop["outcome"], "STOP")
        self.assertEqual(gray["outcome"], "GRAY")
        self.assertEqual(invalid["outcome"], "INVALID")
        self.assertEqual(inexact["outcome"], "INVALID")
        interval = one_sided_mean_interval([2.0, 3.0, 4.0, 5.0])
        self.assertLess(interval["lower"], interval["mean"])
        self.assertGreater(interval["upper"], interval["mean"])
        self.assertEqual(interval["confidence"], 0.95)

    def test_sweep_gate_stops_only_after_the_full_pre_registered_sweep(self) -> None:
        full_stop = [
            {
                "k": k,
                "w1_quality_gate": {"outcome": "STOP", "valid": True},
            }
            for k in DEFAULT_K_SWEEP
        ]
        partial_stop = full_stop[:-1]
        mixed = deepcopy(full_stop)
        mixed[2]["w1_quality_gate"]["outcome"] = "GRAY"
        has_go = deepcopy(partial_stop)
        has_go[0]["w1_quality_gate"]["outcome"] = "GO"

        self.assertEqual(
            classify_sweep_quality_gate(full_stop)["outcome"],
            "STOP",
        )
        self.assertEqual(
            classify_sweep_quality_gate(partial_stop)["outcome"],
            "INCOMPLETE",
        )
        self.assertEqual(
            classify_sweep_quality_gate(mixed)["outcome"],
            "GRAY",
        )
        self.assertEqual(
            classify_sweep_quality_gate(has_go)["outcome"],
            "GO",
        )

    def test_run_evidence_accepts_exact_bounded_and_drained_arms(self) -> None:
        cases = (
            ("classic", 1, 0),
            ("draft_w1", 1, 0),
            ("draft_conveyor", 4, 4096),
        )
        for arm, window, byte_cap in cases:
            with self.subTest(arm=arm):
                evidence = validate_run_evidence(
                    _valid_run(arm, window=window, byte_cap=byte_cap),
                    expected_window=window,
                    expected_inflight_bytes=byte_cap,
                    expected_requests=4,
                )
                self.assertTrue(evidence["exact"], evidence["violations"])

    def test_w1_zero_byte_ceiling_allows_real_verify_transport_bytes(self) -> None:
        run_value = _valid_run("draft_w1", window=1, byte_cap=0)
        self.assertGreater(
            run_value["speculative_window_stats"]["high_water_bytes"],
            0,
        )

        evidence = validate_run_evidence(
            run_value,
            expected_window=1,
            expected_inflight_bytes=0,
            expected_requests=4,
        )

        self.assertTrue(evidence["resource_limits_exact"])
        self.assertTrue(evidence["exact"], evidence["violations"])

    def test_run_evidence_rejects_hash_leak_limit_and_unexercised_conveyor(self) -> None:
        cases: list[tuple[str, dict[str, Any], str]] = []
        mismatched = _valid_run("draft_conveyor")
        mismatched["samples"][0]["token_hash_exact"] = False
        cases.append(("hash", mismatched, "exact_output_hashes"))

        leaking = _valid_run("draft_conveyor")
        leaking["samples"][0]["drained_after_request"] = False
        leaking["speculative_window_stats"]["current_waves"] = 1
        cases.append(("drain", leaking, "drained_to_zero"))

        over_limit = _valid_run("draft_conveyor")
        over_limit["speculative_window_stats"]["high_water_waves"] = 5
        cases.append(("limit", over_limit, "resource_limits_exact"))

        idle = _valid_run("draft_conveyor")
        idle["speculative_window_stats"]["high_water_waves"] = 1
        idle["speculative_window_stats"]["max_request_waves"] = 1
        cases.append(("idle", idle, "execution_exercised"))

        missing_return = _valid_run("draft_conveyor")
        missing_return["speculative_window_stats"]["dispatched_waves"] += 1
        cases.append(("return", missing_return, "all_returns_accounted"))

        dirty = _valid_run("draft_w1")
        dirty["shutdown"]["exact"] = False
        dirty["shutdown"]["close_error"] = "child alive"
        cases.append(("shutdown", dirty, "clean_shutdown"))

        for name, arm, failed_check in cases:
            with self.subTest(name=name):
                evidence = validate_run_evidence(
                    arm,
                    expected_window=arm.get("window", 4),
                    expected_inflight_bytes=arm.get("inflight_bytes", 4096),
                    expected_requests=4,
                )
                self.assertFalse(evidence["exact"])
                self.assertFalse(evidence[failed_check])

    def test_draft_model_provider_cleanup_is_part_of_evidence(self) -> None:
        run_value = _valid_run("draft_w1")
        run_value["speculation_stats"]["provider"] = {
            "strategy": "draft-model",
            "draftFailures": 0,
            "bypassedCalls": 0,
            "circuitOpen": False,
            "cachedKvBytes": 0,
        }
        evidence = validate_run_evidence(
            run_value,
            expected_window=1,
            expected_inflight_bytes=0,
            expected_requests=4,
        )
        self.assertTrue(evidence["provider_clean"])

        run_value["speculation_stats"]["provider"]["cachedKvBytes"] = 64
        evidence = validate_run_evidence(
            run_value,
            expected_window=1,
            expected_inflight_bytes=0,
            expected_requests=4,
        )
        self.assertFalse(evidence["provider_clean"])
        self.assertFalse(evidence["exact"])

    def test_aggregate_uses_all_requests_and_weighted_counter_totals(self) -> None:
        first = _valid_run("draft_w1", requests=4)
        second = _valid_run("draft_w1", requests=3)
        first["evidence"] = {"exact": True}
        second["evidence"] = {"exact": True}

        aggregate = aggregate_arm_runs(
            (first, second),
            arm_name="draft_w1",
            expected_requests=7,
        )

        self.assertTrue(aggregate["evidence_exact"])
        self.assertEqual(aggregate["measured_requests"], 7)
        self.assertEqual(aggregate["g_route"]["count"], 7)
        self.assertEqual(aggregate["counter_deltas"]["proposed_tokens"], 28)
        self.assertEqual(aggregate["counter_deltas"]["accepted_tokens"], 21)
        self.assertEqual(aggregate["acceptance_rate"], 0.75)

    def test_aggregate_allows_zero_coverage_partition_but_not_idle_arm(self) -> None:
        exercised = _valid_run("draft_w1", requests=4)
        idle = _valid_run("draft_w1", requests=3)
        exercised["evidence"] = validate_run_evidence(
            exercised,
            expected_window=1,
            expected_inflight_bytes=0,
            expected_requests=4,
        )
        idle["speculation_stats"]["proposed_tokens"] = 0
        idle["speculative_window_stats"]["dispatched_waves"] = 0
        idle["speculative_window_stats"]["completed_waves"] = 0
        idle["speculative_window_stats"]["high_water_waves"] = 0
        idle["speculative_window_stats"]["max_request_waves"] = 0
        idle["evidence"] = validate_run_evidence(
            idle,
            expected_window=1,
            expected_inflight_bytes=0,
            expected_requests=3,
        )

        aggregate = aggregate_arm_runs(
            (exercised, idle),
            arm_name="draft_w1",
            expected_requests=7,
        )
        entirely_idle = aggregate_arm_runs(
            (idle,),
            arm_name="draft_w1",
            expected_requests=3,
        )

        self.assertTrue(aggregate["integrity_exact"])
        self.assertTrue(aggregate["execution_exercised"])
        self.assertTrue(aggregate["evidence_exact"])
        self.assertTrue(entirely_idle["integrity_exact"])
        self.assertFalse(entirely_idle["execution_exercised"])
        self.assertFalse(entirely_idle["evidence_exact"])

    def test_cli_preregisters_default_sweep_and_rejects_short_runs(self) -> None:
        args = parse_args([])
        self.assertEqual(args.k_sweep, DEFAULT_K_SWEEP)
        self.assertEqual(args.requests_per_arm, MIN_MEASURED_REQUESTS_PER_ARM)
        self.assertEqual(args.order_repetitions, 2)
        self.assertGreaterEqual(args.output_tokens, 2 * max(args.k_sweep) + 2)

        with self.assertRaises(SystemExit):
            parse_args(["--requests-per-arm", "6"])
        with self.assertRaises(SystemExit):
            parse_args(["--order-repetitions", "3"])
        with self.assertRaises(SystemExit):
            parse_args(["--output-tokens", "32"])

    def test_draft_model_cli_requires_the_full_sealed_factory_contract(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(["--provider", "draft-model"])
        with self.assertRaises(SystemExit):
            parse_args(["--draft-model-source", "unexpected"])
        with self.assertRaises(SystemExit):
            parse_args(["--draft-model-revision", "unexpected"])

        args = parse_args(
            [
                "--provider",
                "draft-model",
                "--draft-model-source",
                "local-draft",
                "--draft-model-artifact-identity",
                "sha256:" + "a" * 64,
                "--draft-model-parameter-bytes",
                "100",
                "--draft-model-memory-reservation-bytes",
                "200",
            ]
        )
        self.assertEqual(args.provider, "draft-model")
        self.assertEqual(args.draft_model_memory_reservation_bytes, 200)

    def test_run_seals_the_exact_body_without_runtime_or_model_access(self) -> None:
        measured = {
            "evidence_class": "test",
            "success": True,
            "scenarios": [],
        }
        args = parse_args([])
        with patch(
            "distributed_runtime.draft_quality_runtime_benchmark.run_runtime_benchmark",
            return_value=measured,
        ), patch(
            "distributed_runtime.draft_quality_runtime_benchmark.time.time",
            return_value=123.0,
        ), patch(
            "distributed_runtime.draft_quality_runtime_benchmark.platform.python_version",
            return_value="3.test",
        ), patch(
            "distributed_runtime.draft_quality_runtime_benchmark.platform.platform",
            return_value="platform-test",
        ):
            result = run(args)

        body = {key: value for key, value in result.items() if key != "seal"}
        canonical = json.dumps(
            body,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
        self.assertEqual(
            result["seal"]["digest"],
            f"sha256:{hashlib.sha256(canonical).hexdigest()}",
        )

    def test_publisher_never_overwrites_existing_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.json"
            publish_json_no_overwrite(path, {"first": True})
            with self.assertRaises(FileExistsError):
                publish_json_no_overwrite(path, {"second": True})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"first": True})


if __name__ == "__main__":
    unittest.main()
