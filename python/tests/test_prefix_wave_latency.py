from __future__ import annotations

import unittest

from distributed_runtime.prefix_wave_latency import (
    PrefixWaveCostModel,
    PrefixWaveStrategy,
    compare_prefix_wave_strategies,
    optimize_stream_chunking,
    simulate_prefix_wave,
)


PATHS = (
    (11, 12, 13, 21),
    (11, 12, 13, 22),
    (11, 12, 14, 23),
    (11, 12, 14, 24),
)


def _cost_model(
    *,
    propagation_ms: float = 1.0,
    compute_ms: float = 2.0,
    launch_ms: float = 0.1,
    bandwidth_mbps: float = 10_000.0,
    activation_bytes: int = 100,
    codec_fixed_ms: float = 0.01,
) -> PrefixWaveCostModel:
    return PrefixWaveCostModel(
        stage_compute_ms_per_token=(compute_ms, compute_ms, compute_ms),
        stage_kernel_launch_ms=(launch_ms, launch_ms, launch_ms),
        hop_one_way_propagation_ms=(propagation_ms, propagation_ms),
        hop_bandwidth_mbps=(bandwidth_mbps, bandwidth_mbps),
        activation_bytes_per_token=activation_bytes,
        pack_fixed_ms_per_record=codec_fixed_ms,
        unpack_fixed_ms_per_record=codec_fixed_ms,
        pack_bytes_per_ms=1_000_000_000.0,
        hash_bytes_per_ms=1_000_000_000.0,
        fork_apply_ms_per_edge=0.001,
    )


class PrefixWaveWorkTests(unittest.TestCase):
    def test_shared_trunk_is_computed_once(self) -> None:
        comparison = compare_prefix_wave_strategies(PATHS, _cost_model())

        self.assertEqual(comparison.flat_legacy.executed_token_steps, 20)
        self.assertEqual(comparison.monolithic_packed.executed_token_steps, 9)
        self.assertEqual(comparison.streaming_segmented.executed_token_steps, 9)
        self.assertEqual(len(comparison.monolithic_packed.work_units), 1)
        self.assertEqual(len(comparison.streaming_segmented.work_units), 5)
        self.assertAlmostEqual(
            comparison.streaming_segmented.shared_compute_reduction_ratio,
            0.55,
        )
        self.assertGreater(
            comparison.streaming_segmented.forward_wire_bytes,
            comparison.monolithic_packed.forward_wire_bytes,
        )

    def test_grouping_every_segment_is_equivalent_to_monolithic_cost(self) -> None:
        model = _cost_model()
        monolithic = simulate_prefix_wave(
            PATHS,
            model,
            strategy=PrefixWaveStrategy.MONOLITHIC_PACKED,
        )
        grouped_stream = simulate_prefix_wave(
            PATHS,
            model,
            strategy=PrefixWaveStrategy.STREAMING_SEGMENTED,
            stream_groups_per_record=100,
        )

        self.assertEqual(len(grouped_stream.work_units), 1)
        self.assertEqual(grouped_stream.executed_token_steps, 9)
        self.assertEqual(grouped_stream.forward_wire_bytes, monolithic.forward_wire_bytes)
        self.assertEqual(grouped_stream.kernel_launch_count, monolithic.kernel_launch_count)
        self.assertAlmostEqual(grouped_stream.makespan_ms, monolithic.makespan_ms)
        self.assertEqual(grouped_stream.cost, monolithic.cost)


class PrefixWaveCausalityTests(unittest.TestCase):
    def test_segment_stream_has_one_logical_barrier_and_stage_overlap(self) -> None:
        comparison = compare_prefix_wave_strategies(PATHS, _cost_model())
        streamed = comparison.streaming_segmented

        self.assertEqual(streamed.logical_request_response_barriers, 1)
        self.assertEqual(streamed.per_node_acknowledgements, 0)
        self.assertFalse(streamed.forward_progress_waits_for_reverse)
        self.assertGreater(streamed.inter_stage_compute_overlap_ms, 0.0)
        self.assertEqual(
            comparison.monolithic_packed.inter_stage_compute_overlap_ms,
            0.0,
        )

        stage_one_first = next(
            event
            for event in streamed.events
            if event.category == "compute"
            and event.stage_index == 1
            and event.unit_index == 0
        )
        root_last = max(
            event.end_ms
            for event in streamed.events
            if event.category == "compute" and event.stage_index == 0
        )
        self.assertLess(stage_one_first.start_ms, root_last)

    def test_ordered_stream_preserves_fork_and_activation_causality(self) -> None:
        streamed = simulate_prefix_wave(
            PATHS,
            _cost_model(),
            strategy=PrefixWaveStrategy.STREAMING_SEGMENTED,
        )

        for stage_index in range(3):
            compute = [
                event
                for event in streamed.events
                if event.category == "compute" and event.stage_index == stage_index
            ]
            self.assertEqual(
                [event.unit_index for event in compute],
                sorted(event.unit_index for event in compute),
            )
            for event in compute:
                matching_forks = [
                    candidate
                    for candidate in streamed.events
                    if candidate.category == "fork"
                    and candidate.stage_index == stage_index
                    and candidate.unit_index == event.unit_index
                ]
                for fork in matching_forks:
                    self.assertLessEqual(fork.end_ms, event.start_ms)

        for unit in streamed.work_units:
            for downstream_stage in (1, 2):
                arrival = next(
                    event
                    for event in streamed.events
                    if event.category == "propagation"
                    and event.direction == "forward"
                    and event.hop_index == downstream_stage - 1
                    and event.unit_index == unit.unit_index
                )
                compute = next(
                    event
                    for event in streamed.events
                    if event.category == "compute"
                    and event.stage_index == downstream_stage
                    and event.unit_index == unit.unit_index
                )
                self.assertGreaterEqual(compute.start_ms, arrival.end_ms)

        for resource in ("forward-link-0", "forward-link-1"):
            transmissions = sorted(
                (
                    event
                    for event in streamed.events
                    if event.category == "bandwidth" and event.resource == resource
                ),
                key=lambda event: event.start_ms,
            )
            for previous, current in zip(transmissions, transmissions[1:]):
                self.assertLessEqual(previous.end_ms, current.start_ms)

    def test_propagation_cost_is_one_route_round_trip_not_one_per_segment(self) -> None:
        def network_only(propagation_ms: float):
            return simulate_prefix_wave(
                PATHS,
                _cost_model(
                    propagation_ms=propagation_ms,
                    compute_ms=0.0,
                    launch_ms=0.0,
                    bandwidth_mbps=1_000_000_000.0,
                    activation_bytes=1,
                    codec_fixed_ms=0.0,
                ),
                strategy=PrefixWaveStrategy.STREAMING_SEGMENTED,
            )

        low = network_only(5.0)
        high = network_only(15.0)
        self.assertGreater(len(low.work_units), 1)
        self.assertAlmostEqual(low.propagation_floor_ms, 20.0)
        self.assertAlmostEqual(high.propagation_floor_ms, 60.0)
        # Two hops forward plus the same two hops back: increasing each
        # one-way hop by 10 ms adds 40 ms, independent of segment count.
        self.assertAlmostEqual(high.makespan_ms - low.makespan_ms, 40.0, places=6)


class PrefixWaveTradeoffTests(unittest.TestCase):
    def test_chunk_optimizer_selects_pipeline_or_amortization_from_costs(self) -> None:
        long_chains = tuple(
            (10, 11, 12, 13, 100 + leaf, 200 + leaf, 300 + leaf, 400 + leaf)
            for leaf in range(8)
        )
        pipeline_bound = PrefixWaveCostModel(
            stage_compute_ms_per_token=(2.0,) * 4,
            stage_kernel_launch_ms=(0.1,) * 4,
            hop_one_way_propagation_ms=(0.1,) * 3,
            hop_bandwidth_mbps=(10_000.0,) * 3,
            activation_bytes_per_token=100,
            pack_fixed_ms_per_record=0.01,
            unpack_fixed_ms_per_record=0.01,
            pack_bytes_per_ms=1_000_000_000.0,
            hash_bytes_per_ms=1_000_000_000.0,
        )
        overhead_bound = PrefixWaveCostModel(
            stage_compute_ms_per_token=(0.001,) * 4,
            stage_kernel_launch_ms=(1.0,) * 4,
            hop_one_way_propagation_ms=(0.0,) * 3,
            hop_bandwidth_mbps=(10_000.0,) * 3,
            activation_bytes_per_token=100,
            pack_fixed_ms_per_record=1.0,
            unpack_fixed_ms_per_record=1.0,
            pack_bytes_per_ms=1_000_000_000.0,
            hash_bytes_per_ms=1_000_000_000.0,
        )

        fine = optimize_stream_chunking(long_chains, pipeline_bound)
        coarse = optimize_stream_chunking(long_chains, overhead_bound)
        self.assertEqual(fine.selected_chain_tokens_per_record, 1)
        self.assertEqual(coarse.selected_chain_tokens_per_record, 5)
        self.assertEqual(
            {item.simulation.executed_token_steps for item in fine.candidates},
            {37},
        )
        self.assertLess(
            fine.simulation.makespan_ms,
            fine.candidates[-1].simulation.makespan_ms,
        )
        self.assertLess(
            coarse.simulation.makespan_ms,
            coarse.candidates[0].simulation.makespan_ms,
        )

    def test_streaming_wins_when_stage_pipeline_is_material(self) -> None:
        comparison = compare_prefix_wave_strategies(PATHS, _cost_model())
        self.assertLess(
            comparison.streaming_segmented.makespan_ms,
            comparison.monolithic_packed.makespan_ms,
        )
        self.assertLess(
            comparison.streaming_segmented.makespan_ms,
            comparison.flat_legacy.makespan_ms,
        )

    def test_monolithic_wins_when_record_and_launch_overhead_dominate(self) -> None:
        comparison = compare_prefix_wave_strategies(
            PATHS,
            _cost_model(
                propagation_ms=0.0,
                compute_ms=0.001,
                launch_ms=1.0,
                bandwidth_mbps=1_000_000_000.0,
                activation_bytes=1,
                codec_fixed_ms=1.0,
            ),
        )
        self.assertLess(
            comparison.monolithic_packed.makespan_ms,
            comparison.streaming_segmented.makespan_ms,
        )

    def test_cost_breakdown_is_derived_from_timeline(self) -> None:
        simulation = simulate_prefix_wave(
            PATHS,
            _cost_model(),
            strategy=PrefixWaveStrategy.STREAMING_SEGMENTED,
        )
        by_category: dict[str, float] = {}
        for event in simulation.events:
            by_category[event.category] = (
                by_category.get(event.category, 0.0) + event.duration_ms
            )
        self.assertAlmostEqual(
            simulation.cost.propagation_ms, by_category["propagation"]
        )
        self.assertAlmostEqual(
            simulation.cost.bandwidth_ms, by_category["bandwidth"]
        )
        self.assertAlmostEqual(simulation.cost.compute_ms, by_category["compute"])
        self.assertAlmostEqual(
            simulation.cost.kernel_launch_ms, by_category["kernel_launch"]
        )
        self.assertGreater(simulation.cost.pack_ms, 0.0)
        self.assertGreater(simulation.cost.hash_ms, 0.0)
        self.assertGreater(
            simulation.cost.total_resource_work_ms, simulation.makespan_ms
        )


class PrefixWaveValidationTests(unittest.TestCase):
    def test_cost_model_rejects_incoherent_route_shapes(self) -> None:
        with self.assertRaisesRegex(ValueError, "same length"):
            PrefixWaveCostModel(
                stage_compute_ms_per_token=(1.0, 1.0),
                stage_kernel_launch_ms=(1.0,),
                hop_one_way_propagation_ms=(1.0,),
                hop_bandwidth_mbps=(10.0,),
                activation_bytes_per_token=16,
            )
        with self.assertRaisesRegex(ValueError, "adjacent stage pair"):
            PrefixWaveCostModel(
                stage_compute_ms_per_token=(1.0, 1.0),
                stage_kernel_launch_ms=(1.0, 1.0),
                hop_one_way_propagation_ms=(),
                hop_bandwidth_mbps=(),
                activation_bytes_per_token=16,
            )

    def test_invalid_stream_or_tree_inputs_fail_closed(self) -> None:
        model = _cost_model()
        with self.assertRaisesRegex(ValueError, "stream_groups_per_record"):
            simulate_prefix_wave(
                PATHS,
                model,
                strategy=PrefixWaveStrategy.STREAMING_SEGMENTED,
                stream_groups_per_record=0,
            )
        with self.assertRaisesRegex(ValueError, "shared_root_tokens"):
            simulate_prefix_wave(
                PATHS,
                model,
                strategy=PrefixWaveStrategy.STREAMING_SEGMENTED,
                shared_root_tokens=10**9,
            )
        with self.assertRaisesRegex(ValueError, "prefix"):
            simulate_prefix_wave(
                ((1,), (1, 2)),
                model,
                strategy=PrefixWaveStrategy.STREAMING_SEGMENTED,
            )
        with self.assertRaisesRegex(ValueError, "unknown"):
            simulate_prefix_wave(PATHS, model, strategy="not-a-strategy")


if __name__ == "__main__":
    unittest.main()
