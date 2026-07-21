from __future__ import annotations

import hashlib
import json
import unittest
from dataclasses import replace

from distributed_runtime.prefix_wave_benchmark import (
    CAP_FLAT_BATCHED_EXACT_V1,
    CAP_FLAT_LEGACY_V6,
    CAP_PREFIX_MONOLITHIC_EXACT_BATCH_V1,
    CAP_PREFIX_STREAMING_EXACT_BATCH_V1,
    CAP_PREFIX_WAVE_V7,
    CAPABILITY_CONFIG_SCHEMA,
    DEFAULT_DEPLOYMENT_CAPABILITIES,
    EVIDENCE_CLASS,
    KNOWN_DEPLOYMENT_CAPABILITIES,
    SCHEMA,
    STRATEGY_ORDER,
    build_prefix_wave_benchmark,
    canonical_prefix_wave_benchmark_json,
    select_fastest_prefix_wave_strategy,
)
from distributed_runtime.prefix_wave_latency import (
    PrefixWaveCostModel,
    PrefixWaveStrategy,
    optimize_stream_chunking,
    simulate_prefix_wave,
)


EXPECTED_PAYLOAD_SHA256 = (
    "93c155bcf8e3d058f71d3d31902e5bf114de8e2150ce34070d05a4a687ca07fc"
)


class PrefixWaveBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = build_prefix_wave_benchmark()

    def test_report_is_canonical_deterministic_and_explicitly_theoretical(
        self,
    ) -> None:
        second = build_prefix_wave_benchmark()
        self.assertEqual(second, self.report)
        self.assertEqual(self.report["schema"], SCHEMA)
        self.assertEqual(self.report["evidenceClass"], EVIDENCE_CLASS)
        self.assertEqual(EVIDENCE_CLASS, "theoretical_simulation")
        self.assertIn("no wall-clock", self.report["interpretation"])
        self.assertIn("tokens-per-second", self.report["interpretation"])
        self.assertTrue(
            any("PREPARE/COMMIT" in item for item in self.report["assumptions"])
        )
        self.assertTrue(
            any("uncertified" in item for item in self.report["assumptions"])
        )
        self.assertNotIn("\"tps", canonical_prefix_wave_benchmark_json().lower())

        canonical = canonical_prefix_wave_benchmark_json(self.report)
        self.assertEqual(json.loads(canonical), self.report)
        self.assertNotIn("\n", canonical)
        self.assertEqual(canonical, canonical_prefix_wave_benchmark_json(second))

        payload = dict(self.report)
        recorded_hash = payload.pop("canonicalPayloadSha256")
        canonical_payload = canonical_prefix_wave_benchmark_json(payload).encode(
            "utf-8"
        )
        self.assertEqual(
            hashlib.sha256(canonical_payload).hexdigest(), recorded_hash
        )
        self.assertEqual(recorded_hash, EXPECTED_PAYLOAD_SHA256)

        self.assertEqual(
            [scenario["scenario"] for scenario in self.report["scenarios"]],
            [
                "shared8x8",
                "hierarchical16x6",
                "clusteredVariable",
                "disjoint8x8",
            ],
        )
        self.assertEqual(
            [profile["profile"] for profile in self.report["inputs"]["profiles"]],
            ["lan", "domesticWanDirect", "domesticWanConservative"],
        )
        capability_config = self.report["inputs"]["deploymentCapabilityConfig"]
        self.assertEqual(capability_config["schema"], CAPABILITY_CONFIG_SCHEMA)
        self.assertEqual(
            capability_config["certifiedCapabilities"],
            list(DEFAULT_DEPLOYMENT_CAPABILITIES),
        )
        self.assertEqual(
            tuple(capability_config["strategyRequirements"]),
            tuple(strategy.value for strategy in STRATEGY_ORDER),
        )
        for profile in self.report["inputs"]["profiles"]:
            model = profile["costModel"]
            self.assertEqual(len(model["stageComputeMsPerToken"]), 4)
            self.assertEqual(len(model["hopOneWayPropagationMs"]), 3)
            self.assertEqual(len(model["hopBandwidthMbps"]), 3)
            self.assertGreater(model["activationBytesPerToken"], 0)
            self.assertGreater(model["returnBandwidthMbps"], 0)
            self.assertGreater(model["forkControlFrameBytes"], 0)

    def test_every_strategy_conserves_planner_work_and_reports_transport(self) -> None:
        strategy_order = (
            PrefixWaveStrategy.FLAT_LEGACY.value,
            PrefixWaveStrategy.FLAT_BATCHED.value,
            PrefixWaveStrategy.MONOLITHIC_PACKED.value,
            PrefixWaveStrategy.STREAMING_SEGMENTED.value,
        )
        for scenario in self.report["scenarios"]:
            planner = scenario["planner"]
            self.assertEqual(
                planner["flatTokenSteps"] - planner["uniqueTokenSteps"],
                planner["savedTokenSteps"],
            )
            self.assertLessEqual(
                planner["segmentCalls"], planner["nodeCalls"]
            )
            for profile in scenario["profiles"]:
                input_cost_model = next(
                    item["costModel"]
                    for item in self.report["inputs"]["profiles"]
                    if item["profile"] == profile["profile"]
                )
                with self.subTest(
                    scenario=scenario["scenario"], profile=profile["profile"]
                ):
                    strategies = profile["strategies"]
                    self.assertEqual(tuple(strategies), strategy_order)
                    flat = strategies[PrefixWaveStrategy.FLAT_LEGACY.value]
                    flat_batched = strategies[
                        PrefixWaveStrategy.FLAT_BATCHED.value
                    ]
                    monolithic = strategies[
                        PrefixWaveStrategy.MONOLITHIC_PACKED.value
                    ]
                    streaming = strategies[
                        PrefixWaveStrategy.STREAMING_SEGMENTED.value
                    ]
                    self.assertEqual(
                        flat["executedTokenSteps"], planner["flatTokenSteps"]
                    )
                    self.assertEqual(
                        flat_batched["executedTokenSteps"],
                        planner["flatTokenSteps"],
                    )
                    self.assertEqual(
                        monolithic["executedTokenSteps"],
                        planner["uniqueTokenSteps"],
                    )
                    self.assertEqual(
                        streaming["executedTokenSteps"],
                        planner["uniqueTokenSteps"],
                    )
                    for record in strategies.values():
                        self.assertEqual(
                            record["flatEquivalentTokenSteps"],
                            planner["flatTokenSteps"],
                        )
                        self.assertEqual(
                            record["routeAggregateFrameCount"],
                            record["forwardFrameCountAcrossHops"]
                            + record["reverseFrameCount"],
                        )
                        self.assertEqual(
                            record["routeAggregateWireBytes"],
                            record["forwardWireBytesAcrossHops"]
                            + record["reverseWireBytes"],
                        )
                        self.assertGreater(record["latencyMs"], 0)
                        self.assertGreater(record["speedupVsFlatLatency"], 0)
                        self.assertEqual(
                            record["logicalRequestResponseBarriers"], 1
                        )
                        self.assertEqual(
                            record["physicalForkCountPerStage"],
                            planner["leafCount"],
                        )
                        self.assertEqual(
                            record["forkControlFrameCountAcrossHops"],
                            planner["leafCount"] * 3,
                        )
                        self.assertEqual(
                            record["forkControlWireBytesAcrossHops"],
                            record["forkControlFrameCountAcrossHops"]
                            * input_cost_model["forkControlFrameBytes"],
                        )
                    self.assertEqual(flat["speedupVsFlatLatency"], 1.0)
                    self.assertIsNone(flat["selectedChainTokensPerRecord"])
                    self.assertIsNone(
                        flat_batched["selectedChainTokensPerRecord"]
                    )
                    self.assertIsNone(
                        monolithic["selectedChainTokensPerRecord"]
                    )
                    self.assertIsInstance(
                        streaming["selectedChainTokensPerRecord"], int
                    )
                    chunk_candidates = streaming["chunkCandidates"]
                    best_chunk = min(
                        chunk_candidates,
                        key=lambda candidate: (
                            candidate["latencyMs"],
                            -candidate["chainTokensPerRecord"],
                        ),
                    )
                    self.assertEqual(
                        streaming["selectedChainTokensPerRecord"],
                        best_chunk["chainTokensPerRecord"],
                    )
                    theoretical_best = min(
                        strategy_order,
                        key=lambda strategy: (
                            strategies[strategy]["latencyMs"],
                            strategy_order.index(strategy),
                        ),
                    )
                    self.assertEqual(
                        profile["theoreticalBestStrategy"], theoretical_best
                    )
                    self.assertEqual(
                        profile["theoreticalBestLatencyMs"],
                        strategies[theoretical_best]["latencyMs"],
                    )
                    eligibility = profile["deploymentEligibility"]
                    self.assertTrue(
                        eligibility[PrefixWaveStrategy.FLAT_LEGACY.value][
                            "deployable"
                        ]
                    )
                    for strategy in strategy_order[1:]:
                        self.assertFalse(eligibility[strategy]["deployable"])
                        self.assertTrue(eligibility[strategy]["missingCapabilities"])
                    self.assertEqual(
                        profile["deployableStrategy"],
                        PrefixWaveStrategy.FLAT_LEGACY.value,
                    )
                    self.assertEqual(
                        profile["deployableLatencyMs"], flat["latencyMs"]
                    )

    def test_kv_sweep_covers_every_alignment_and_conserves_physical_blocks(
        self,
    ) -> None:
        kv_inputs = self.report["inputs"]["kv"]
        block_tokens = kv_inputs["blockTokens"]
        bytes_per_block = kv_inputs["bytesPerBlock"]
        saw_positive_saving = False
        for scenario in self.report["scenarios"]:
            sweep = scenario["kvAlignmentSweep"]
            samples = sweep["samples"]
            self.assertEqual(len(samples), block_tokens)
            self.assertEqual(
                [sample["alignmentOffsetTokens"] for sample in samples],
                list(range(block_tokens)),
            )
            for sample in samples:
                self.assertEqual(
                    sample["parentRemainderTokens"],
                    sample["parentTokens"] % block_tokens,
                )
                self.assertEqual(
                    sample["prefixIncrementalBlocks"]
                    + sample["savedIncrementalBlocks"],
                    sample["flatIncrementalBlocks"],
                )
                self.assertEqual(
                    sample["prefixIncrementalBytes"],
                    sample["prefixIncrementalBlocks"] * bytes_per_block,
                )
                self.assertEqual(
                    sample["flatIncrementalBytes"],
                    sample["flatIncrementalBlocks"] * bytes_per_block,
                )
                self.assertEqual(
                    sample["savedIncrementalBytes"],
                    sample["savedIncrementalBlocks"] * bytes_per_block,
                )
                saw_positive_saving |= sample["savedIncrementalBlocks"] > 0

            key = lambda sample: (
                sample["savedIncrementalBlocks"],
                sample["blockReductionPercent"],
                sample["alignmentOffsetTokens"],
            )
            ordered = sorted(samples, key=key)
            self.assertEqual(sweep["worst"], min(samples, key=key))
            self.assertEqual(sweep["median"], ordered[len(ordered) // 2])
            expected_best = max(
                samples,
                key=lambda sample: (
                    sample["savedIncrementalBlocks"],
                    sample["blockReductionPercent"],
                    -sample["alignmentOffsetTokens"],
                ),
            )
            self.assertEqual(sweep["best"], expected_best)
            self.assertGreaterEqual(
                sweep["best"]["savedIncrementalBlocks"],
                sweep["worst"]["savedIncrementalBlocks"],
            )
        self.assertTrue(saw_positive_saving)

    def test_selector_never_prefers_stream_when_it_does_not_win(self) -> None:
        paths = ((1, 2, 3), (1, 2, 4), (7, 8, 9))
        model = PrefixWaveCostModel(
            stage_compute_ms_per_token=(0.2, 0.2),
            stage_kernel_launch_ms=(0.05, 0.05),
            hop_one_way_propagation_ms=(1.0,),
            hop_bandwidth_mbps=(100.0,),
            activation_bytes_per_token=1_024,
            return_one_way_propagation_ms=1.0,
            return_bandwidth_mbps=100.0,
        )
        flat = simulate_prefix_wave(
            paths, model, strategy=PrefixWaveStrategy.FLAT_LEGACY
        )
        flat_batched = simulate_prefix_wave(
            paths, model, strategy=PrefixWaveStrategy.FLAT_BATCHED
        )
        monolithic = simulate_prefix_wave(
            paths, model, strategy=PrefixWaveStrategy.MONOLITHIC_PACKED
        )
        streaming = optimize_stream_chunking(paths, model).simulation

        flat_ten = replace(flat, makespan_ms=10.0)
        batched_ten = replace(flat_batched, makespan_ms=10.0)
        mono_twelve = replace(monolithic, makespan_ms=12.0)
        stream_ten = replace(streaming, makespan_ms=10.0)
        self.assertEqual(
            select_fastest_prefix_wave_strategy(
                flat_ten, batched_ten, mono_twelve, stream_ten
            ).strategy,
            PrefixWaveStrategy.FLAT_LEGACY,
        )

        batched_nine = replace(flat_batched, makespan_ms=9.0)
        mono_nine = replace(monolithic, makespan_ms=9.0)
        stream_nine = replace(streaming, makespan_ms=9.0)
        self.assertEqual(
            select_fastest_prefix_wave_strategy(
                flat_ten, batched_nine, mono_nine, stream_nine
            ).strategy,
            PrefixWaveStrategy.FLAT_BATCHED,
        )

        stream_eight = replace(streaming, makespan_ms=8.0)
        self.assertEqual(
            select_fastest_prefix_wave_strategy(
                flat_ten, batched_nine, mono_nine, stream_eight
            ).strategy,
            PrefixWaveStrategy.STREAMING_SEGMENTED,
        )
        self.assertEqual(
            select_fastest_prefix_wave_strategy(
                flat_ten,
                batched_nine,
                mono_nine,
                stream_eight,
                eligible_strategies=(PrefixWaveStrategy.FLAT_LEGACY,),
            ).strategy,
            PrefixWaveStrategy.FLAT_LEGACY,
        )
        with self.assertRaisesRegex(ValueError, "expected flat_legacy"):
            select_fastest_prefix_wave_strategy(
                monolithic, flat_batched, flat, streaming
            )
        with self.assertRaisesRegex(ValueError, "at least one"):
            select_fastest_prefix_wave_strategy(
                flat, flat_batched, monolithic, streaming, eligible_strategies=()
            )

    def test_deployment_selection_is_capability_gated_and_fail_closed(self) -> None:
        self.assertEqual(
            KNOWN_DEPLOYMENT_CAPABILITIES,
            (
                CAP_FLAT_LEGACY_V6,
                CAP_FLAT_BATCHED_EXACT_V1,
                CAP_PREFIX_WAVE_V7,
                CAP_PREFIX_MONOLITHIC_EXACT_BATCH_V1,
                CAP_PREFIX_STREAMING_EXACT_BATCH_V1,
            ),
        )
        fully_certified = build_prefix_wave_benchmark(
            deployment_capabilities=KNOWN_DEPLOYMENT_CAPABILITIES
        )
        for scenario in fully_certified["scenarios"]:
            for profile in scenario["profiles"]:
                self.assertEqual(
                    profile["deployableStrategy"],
                    profile["theoreticalBestStrategy"],
                )
                self.assertTrue(
                    all(
                        item["deployable"]
                        for item in profile["deploymentEligibility"].values()
                    )
                )

        prefix_transport_only = build_prefix_wave_benchmark(
            deployment_capabilities=(
                CAP_FLAT_LEGACY_V6,
                CAP_PREFIX_WAVE_V7,
            )
        )
        first_profile = prefix_transport_only["scenarios"][0]["profiles"][0]
        self.assertEqual(
            first_profile["deployableStrategy"],
            PrefixWaveStrategy.FLAT_LEGACY.value,
        )
        self.assertFalse(
            first_profile["deploymentEligibility"][
                PrefixWaveStrategy.MONOLITHIC_PACKED.value
            ]["deployable"]
        )

        with self.assertRaisesRegex(ValueError, "flat_legacy fallback"):
            build_prefix_wave_benchmark(
                deployment_capabilities=(CAP_FLAT_BATCHED_EXACT_V1,)
            )
        with self.assertRaisesRegex(ValueError, "unknown deployment capability"):
            build_prefix_wave_benchmark(
                deployment_capabilities=(CAP_FLAT_LEGACY_V6, "invented")
            )
        with self.assertRaisesRegex(ValueError, "duplicate deployment capability"):
            build_prefix_wave_benchmark(
                deployment_capabilities=(CAP_FLAT_LEGACY_V6, CAP_FLAT_LEGACY_V6)
            )


if __name__ == "__main__":
    unittest.main()
