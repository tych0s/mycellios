from __future__ import annotations

import argparse
import unittest

from distributed_runtime.verify_conveyor_benchmark import (
    EVIDENCE_CLASS,
    build_benchmark_report,
    parse_alphas,
    parse_args,
    parse_rtts_ms,
    parse_windows,
    simulate_verify_conveyor,
    synthetic_greedy_oracle,
    token_sequence_sha256,
)


class VerifyConveyorBenchmarkTests(unittest.TestCase):
    def test_synthetic_greedy_oracle_and_hash_are_reproducible(self) -> None:
        first = synthetic_greedy_oracle(
            seed=7,
            sample_index=2,
            token_count=32,
        )
        repeated = synthetic_greedy_oracle(
            seed=7,
            sample_index=2,
            token_count=32,
        )
        other_sample = synthetic_greedy_oracle(
            seed=7,
            sample_index=3,
            token_count=32,
        )

        self.assertEqual(first, repeated)
        self.assertEqual(token_sequence_sha256(first), token_sequence_sha256(repeated))
        self.assertNotEqual(first, other_sample)
        self.assertNotEqual(
            token_sequence_sha256(first),
            token_sequence_sha256(other_sample),
        )

    def test_full_sweep_requires_exact_hash_parity_for_every_arm(self) -> None:
        report = build_benchmark_report(
            alphas=(0.0, 0.6, 1.0),
            rtts_ms=(0.0, 50.0),
            draft_lengths=(1, 4),
            windows=(1, 2, 4),
            output_tokens=40,
            samples=3,
            verify_ms_per_token=1.0,
            bytes_per_token=128,
            inflight_bytes=4 * 4 * 128,
            seed=11,
        )

        self.assertTrue(report["success"])
        self.assertEqual(
            report["evidence_class"],
            "SIMULACION_DETERMINISTA_RTT_Y_COMPUTO_EMULADOS",
        )
        self.assertFalse(report["claim_boundary"]["physical_tokens_per_second_claimed"])
        for case in report["cases"]:
            for arm in case["arms"]:
                with self.subTest(
                    alpha=case["alpha"],
                    rtt=case["rtt_ms_emulated"],
                    k=case["draft_length_k"],
                    window=arm["window"],
                ):
                    self.assertTrue(arm["parity"]["exact"])
                    self.assertEqual(
                        arm["parity"]["oracle_sha256_by_sample"],
                        arm["parity"]["output_sha256_by_sample"],
                    )
                    self.assertTrue(arm["high_water"]["exact"])
                    self.assertTrue(arm["bonus_bridge"]["exact"])
                    self.assertTrue(arm["drain_barrier"]["exact"])
        self.assertIn(
            "k drafts plus one bonus", report["native_verify_semantics"]["first_wave"]
        )
        self.assertIn(
            "drain every tombstone",
            report["native_verify_semantics"]["rejection"],
        )

    def test_alpha_one_uses_conveyor_without_discarding_work(self) -> None:
        baseline = simulate_verify_conveyor(
            alpha=1.0,
            rtt_ms=100.0,
            k=4,
            window=1,
            output_tokens=64,
            verify_ms_per_token=1.0,
            bytes_per_token=64,
            inflight_bytes=4096,
            seed=5,
            sample_index=0,
        )
        conveyor = simulate_verify_conveyor(
            alpha=1.0,
            rtt_ms=100.0,
            k=4,
            window=8,
            output_tokens=64,
            verify_ms_per_token=1.0,
            bytes_per_token=64,
            inflight_bytes=4096,
            seed=5,
            sample_index=0,
        )

        self.assertEqual(
            token_sequence_sha256(baseline.token_ids),
            token_sequence_sha256(conveyor.token_ids),
        )
        self.assertLess(conveyor.finish_ms, baseline.finish_ms)
        self.assertEqual(conveyor.stats["discarded_draft_tokens"], 0)
        self.assertEqual(conveyor.stats["waves_rejected"], 0)
        self.assertGreater(conveyor.stats["max_inflight_waves"], 1)

    def test_w1_full_verify_commits_k_drafts_plus_bonus(self) -> None:
        sample = simulate_verify_conveyor(
            alpha=1.0,
            rtt_ms=25.0,
            k=4,
            window=1,
            output_tokens=5,
            verify_ms_per_token=1.0,
            bytes_per_token=64,
            inflight_bytes=5 * 64,
            seed=19,
            sample_index=0,
        )

        self.assertEqual(sample.stats["waves_dispatched"], 1)
        self.assertEqual(sample.stats["first_seeded_verify_waves"], 1)
        self.assertEqual(sample.stats["continuation_verify_waves"], 0)
        self.assertEqual(sample.stats["accepted_draft_tokens"], 4)
        self.assertEqual(sample.stats["bonus_tokens_committed"], 1)
        self.assertEqual(sample.stats["verify_input_positions_processed"], 5)
        self.assertEqual(len(sample.token_ids), 5)

    def test_successor_consumes_predecessor_bonus_as_first_target_bridge(self) -> None:
        sample = simulate_verify_conveyor(
            alpha=1.0,
            rtt_ms=100.0,
            k=4,
            window=3,
            output_tokens=12,
            verify_ms_per_token=1.0,
            bytes_per_token=64,
            inflight_bytes=4096,
            seed=23,
            sample_index=0,
        )

        self.assertEqual(sample.stats["first_seeded_verify_waves"], 1)
        self.assertEqual(sample.stats["continuation_verify_waves"], 2)
        self.assertEqual(sample.stats["bonus_bridges_published"], 2)
        self.assertEqual(sample.stats["bonus_bridges_consumed"], 2)
        self.assertEqual(sample.stats["bonus_tokens_committed"], 1)
        self.assertEqual(sample.stats["classic_tail_waves"], 0)
        self.assertEqual(
            sample.token_ids,
            synthetic_greedy_oracle(
                seed=23,
                sample_index=0,
                token_count=12,
            ),
        )

    def test_rejection_drains_descendants_and_counts_discarded_work(self) -> None:
        sample = simulate_verify_conveyor(
            alpha=0.0,
            rtt_ms=100.0,
            k=4,
            window=4,
            output_tokens=24,
            verify_ms_per_token=1.0,
            bytes_per_token=64,
            inflight_bytes=4096,
            seed=5,
            sample_index=0,
        )
        oracle = synthetic_greedy_oracle(
            seed=5,
            sample_index=0,
            token_count=24,
        )

        self.assertEqual(sample.token_ids, oracle)
        self.assertGreater(sample.stats["waves_condemned_drained"], 0)
        self.assertGreater(sample.stats["discarded_draft_tokens"], 0)
        self.assertEqual(sample.stats["accepted_draft_tokens"], 0)
        self.assertTrue(sample.stats["resource_limits_exact"])
        self.assertGreater(sample.stats["drain_barriers_started"], 0)
        self.assertEqual(
            sample.stats["drain_barriers_started"],
            sample.stats["drain_barriers_completed"],
        )

        for reject_index, event in enumerate(sample.events):
            if event.kind != "resolve_reject":
                continue
            drain_complete_index = next(
                (
                    index
                    for index in range(reject_index + 1, len(sample.events))
                    if sample.events[index].kind == "drain_complete"
                    and sample.events[index].generation == event.generation + 1
                ),
                None,
            )
            if drain_complete_index is None:
                # A W=1-like rejection with no descendants needs no barrier.
                continue
            next_dispatch_index = next(
                (
                    index
                    for index in range(reject_index + 1, len(sample.events))
                    if sample.events[index].kind
                    in {"dispatch_verify", "dispatch_classic"}
                    and sample.events[index].generation == event.generation + 1
                ),
                None,
            )
            self.assertIsNotNone(next_dispatch_index)
            self.assertGreater(next_dispatch_index, drain_complete_index)

    def test_byte_cap_is_an_enforced_high_water_limit(self) -> None:
        k = 4
        bytes_per_token = 100
        # First wave: visible seed + k drafts. Successor: k drafts.
        cap = (2 * k + 1) * bytes_per_token
        sample = simulate_verify_conveyor(
            alpha=1.0,
            rtt_ms=100.0,
            k=k,
            window=8,
            output_tokens=48,
            verify_ms_per_token=1.0,
            bytes_per_token=bytes_per_token,
            inflight_bytes=cap,
            seed=13,
            sample_index=0,
        )

        self.assertEqual(sample.stats["max_inflight_waves"], 2)
        self.assertEqual(sample.stats["max_inflight_bytes"], cap)
        self.assertTrue(sample.stats["resource_limits_exact"])

    def test_cli_parsers_fail_closed(self) -> None:
        self.assertEqual(parse_alphas("0,0.8,1"), (0.0, 0.8, 1.0))
        self.assertEqual(parse_rtts_ms("0,25,100"), (0.0, 25.0, 100.0))
        self.assertEqual(parse_windows("1,2,8"), (1, 2, 8))
        for parser, bad_values in (
            (parse_alphas, ("", "-0.1,1", "0.8,0.7", "0.8,0.8")),
            (parse_rtts_ms, ("", "-1,0", "25,0", "25,25")),
            (parse_windows, ("", "2,4", "1,1", "1,65")),
        ):
            for raw in bad_values:
                with self.subTest(parser=parser.__name__, raw=raw):
                    with self.assertRaises(argparse.ArgumentTypeError):
                        parser(raw)
        with self.assertRaises(SystemExit):
            parse_args(
                [
                    "--draft-lengths",
                    "4",
                    "--bytes-per-token",
                    "1024",
                    "--inflight-bytes",
                    "1024",
                ]
            )

    def test_evidence_label_does_not_claim_physical_measurement(self) -> None:
        self.assertEqual(
            EVIDENCE_CLASS,
            "SIMULACION_DETERMINISTA_RTT_Y_COMPUTO_EMULADOS",
        )


if __name__ == "__main__":
    unittest.main()
