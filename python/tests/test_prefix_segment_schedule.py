from __future__ import annotations

import random
import unittest

from distributed_runtime.prefix_segment_schedule import (
    PrefixComputeSegment,
    PrefixSegmentFork,
    plan_prefix_segment_schedule,
    target_argmax_by_prefix_from_flat_leaf_results,
    target_argmax_by_prefix_from_node_outputs,
    target_argmax_by_prefix_from_segment_outputs,
)


def _execute_symbolically(schedule) -> dict[int, tuple[int, ...]]:
    lanes: dict[int, tuple[int, ...]] = {0: ()}
    compute_seen_by_frontier: set[int] = set()
    for sequence_index, operation in enumerate(schedule.operations):
        if operation.sequence_index != sequence_index:
            raise AssertionError("operation indices are not contiguous")
        if isinstance(operation, PrefixSegmentFork):
            if operation.frontier_index in compute_seen_by_frontier:
                raise AssertionError("a frontier fork followed a mutating compute")
            prefix = schedule.node(operation.divergence_node_id).prefix
            if prefix != operation.prefix:
                raise AssertionError("fork prefix and divergence node disagree")
            if lanes[operation.source_lane_id] != prefix:
                raise AssertionError("fork source is not at the divergence prefix")
            if operation.target_lane_id in lanes:
                raise AssertionError("fork target lane was reused")
            lanes[operation.target_lane_id] = prefix
            continue
        if not isinstance(operation, PrefixComputeSegment):
            raise AssertionError(f"unknown operation {operation!r}")
        compute_seen_by_frontier.add(operation.frontier_index)
        parent = schedule.node(operation.parent_node_id)
        if lanes[operation.lane_id] != parent.prefix:
            raise AssertionError("segment lane does not hold its parent prefix")
        if len(operation.node_ids) != len(operation.tokens):
            raise AssertionError("segment token and node vectors differ in length")
        cursor = parent.prefix
        for node_id, token in zip(operation.node_ids, operation.tokens):
            node = schedule.node(node_id)
            cursor = (*cursor, token)
            if node.prefix != cursor or node.token != token:
                raise AssertionError("segment does not follow contiguous trie nodes")
        lanes[operation.lane_id] = cursor
    return lanes


def _assert_segments_maximal(test: unittest.TestCase, schedule) -> None:
    for segment in schedule.segments:
        parent = schedule.node(segment.parent_node_id)
        test.assertTrue(parent.node_id == 0 or len(parent.child_node_ids) > 1)
        for node_id in segment.node_ids[:-1]:
            test.assertEqual(len(schedule.node(node_id).child_node_ids), 1)
        test.assertNotEqual(len(schedule.node(segment.end_node_id).child_node_ids), 1)
        test.assertEqual(segment.start_depth, parent.depth)
        test.assertEqual(segment.end_depth, parent.depth + segment.token_count)


def _assert_groups_contiguous(test: unittest.TestCase, schedule) -> None:
    for group in schedule.compatible_frontier_groups:
        operations = schedule.operations[
            group.first_operation_index : group.last_operation_index + 1
        ]
        test.assertEqual(
            tuple(operation.segment_id for operation in operations),
            group.segment_ids,
        )
        for operation in operations:
            test.assertIsInstance(operation, PrefixComputeSegment)
            test.assertEqual(operation.compatible_group_id, group.group_id)
            test.assertEqual(operation.frontier_index, group.frontier_index)
            test.assertEqual(operation.start_depth, group.start_depth)
            test.assertEqual(operation.token_count, group.token_count)


class PrefixSegmentScheduleTests(unittest.TestCase):
    def test_compresses_maximal_chains_and_batches_compatible_frontier_segments(
        self,
    ) -> None:
        schedule = plan_prefix_segment_schedule(
            ((1, 2, 3, 4), (8, 9), (1, 2, 6, 7), (1, 2, 3, 5)),
            shared_prefix_tokens=(99,),
        )
        self.assertEqual(
            schedule.candidate_paths,
            ((1, 2, 3, 4), (1, 2, 3, 5), (1, 2, 6, 7), (8, 9)),
        )
        self.assertEqual(schedule.cost.flat_token_steps, 18)
        self.assertEqual(schedule.cost.unique_token_steps, 10)
        self.assertEqual(schedule.cost.saved_token_steps, 8)
        self.assertEqual(schedule.cost.node_calls, 10)
        self.assertEqual(schedule.cost.segment_calls, 7)
        self.assertEqual(schedule.cost.fork_count, 3)
        self.assertEqual(schedule.cost.carrier_fork_count, 1)
        self.assertEqual(schedule.cost.total_physical_fork_count, 4)
        self.assertEqual(schedule.cost.lane_count, 4)
        self.assertEqual(schedule.cost.frontier_count, 4)
        self.assertEqual(schedule.cost.compatible_frontier_group_count, 5)
        self.assertAlmostEqual(schedule.cost.compute_reduction_ratio, 8 / 18)
        self.assertAlmostEqual(schedule.cost.call_reduction_ratio, 3 / 10)

        self.assertEqual(
            [
                (group.frontier_index, group.start_depth, group.token_count, group.segment_count)
                for group in schedule.compatible_frontier_groups
            ],
            [
                (0, 0, 1, 1),
                (1, 1, 2, 2),
                (2, 3, 1, 1),
                (2, 3, 2, 1),
                (3, 4, 1, 2),
            ],
        )
        lanes = _execute_symbolically(schedule)
        for leaf in schedule.leaves:
            self.assertEqual(lanes[leaf.lane_id], leaf.full_path)
        self.assertEqual(
            schedule.reconstruct_full_leaf_paths(),
            tuple((99,) + path for path in schedule.candidate_paths),
        )
        self.assertEqual(
            schedule.reconstruct_candidate_paths(), schedule.candidate_paths
        )
        _assert_segments_maximal(self, schedule)
        _assert_groups_contiguous(self, schedule)

    def test_schedule_is_identical_under_candidate_input_permutations(self) -> None:
        paths = ((7, 8, 9), (7, 8, 10), (7, 11, 12), (13, 14, 15))
        expected = plan_prefix_segment_schedule(
            paths, shared_prefix_tokens=(101, 102)
        )
        for seed in range(50):
            shuffled = list(paths)
            random.Random(seed).shuffle(shuffled)
            self.assertEqual(
                plan_prefix_segment_schedule(
                    shuffled, shared_prefix_tokens=(101, 102)
                ),
                expected,
            )

    def test_random_prefix_free_tries_preserve_cost_and_execution_invariants(
        self,
    ) -> None:
        generator = random.Random(0x5E6E17)
        for _case in range(250):
            candidate_depth = generator.randint(1, 8)
            width = generator.randint(1, min(32, 16**candidate_depth))
            paths: set[tuple[int, ...]] = set()
            while len(paths) < width:
                paths.add(
                    tuple(generator.randrange(16) for _ in range(candidate_depth))
                )
            shared = tuple(generator.randrange(100, 120) for _ in range(generator.randint(0, 3)))
            schedule = plan_prefix_segment_schedule(
                tuple(paths),
                shared_prefix_tokens=shared,
                max_paths=64,
                max_depth=16,
                max_nodes=2_048,
            )
            lanes = _execute_symbolically(schedule)
            for leaf in schedule.leaves:
                self.assertEqual(lanes[leaf.lane_id], leaf.full_path)
            self.assertEqual(schedule.reconstruct_candidate_paths(), schedule.candidate_paths)
            unique_full_prefixes = {
                full_path[:depth]
                for path in paths
                for full_path in (shared + path,)
                for depth in range(1, len(full_path) + 1)
            }
            self.assertEqual(schedule.cost.unique_token_steps, len(unique_full_prefixes))
            self.assertEqual(
                schedule.cost.flat_token_steps,
                sum(len(shared) + len(path) for path in paths),
            )
            self.assertEqual(
                schedule.cost.saved_token_steps,
                schedule.cost.flat_token_steps - schedule.cost.unique_token_steps,
            )
            self.assertEqual(schedule.cost.node_calls, len(unique_full_prefixes))
            self.assertLessEqual(schedule.cost.segment_calls, schedule.cost.node_calls)
            self.assertEqual(schedule.cost.fork_count, len(paths) - 1)
            self.assertEqual(schedule.cost.total_physical_fork_count, len(paths))
            self.assertEqual(schedule.cost.lane_count, len(paths))
            computed_nodes = [
                node_id for segment in schedule.segments for node_id in segment.node_ids
            ]
            self.assertEqual(len(computed_nodes), len(set(computed_nodes)))
            self.assertEqual(set(computed_nodes), set(range(1, len(schedule.nodes))))
            _assert_segments_maximal(self, schedule)
            _assert_groups_contiguous(self, schedule)

    def test_invalid_or_unbounded_inputs_fail_closed(self) -> None:
        invalid = (
            ((), (), "must not be empty"),
            (((1,), (1,)), (), "duplicates"),
            (((1,), (1, 2)), (), "prefix"),
            (((-1,),), (), "token id"),
            (((True,),), (), "token id"),
            (((1,),), (False,), "token id"),
        )
        for paths, shared, pattern in invalid:
            with self.subTest(paths=paths, shared=shared):
                with self.assertRaisesRegex(ValueError, pattern):
                    plan_prefix_segment_schedule(
                        paths, shared_prefix_tokens=shared
                    )
        with self.assertRaisesRegex(ValueError, "contains 2 paths"):
            plan_prefix_segment_schedule(((1,), (2,)), max_paths=1)
        with self.assertRaisesRegex(ValueError, "depth"):
            plan_prefix_segment_schedule(
                ((1,),), shared_prefix_tokens=(8, 9), max_depth=2
            )
        with self.assertRaisesRegex(ValueError, "nodes"):
            plan_prefix_segment_schedule(
                ((1, 2), (1, 3)),
                shared_prefix_tokens=(9,),
                max_nodes=4,
            )
        for name in ("max_paths", "max_depth", "max_nodes"):
            with self.subTest(limit=name):
                kwargs = {name: True}
                with self.assertRaisesRegex(ValueError, name):
                    plan_prefix_segment_schedule(((1,),), **kwargs)


class PrefixSegmentTargetMappingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.schedule = plan_prefix_segment_schedule(
            ((1, 2, 3), (1, 2, 4), (1, 5, 6), (7, 8)),
            shared_prefix_tokens=(90, 91),
        )
        shared_count = len(self.schedule.shared_prefix_tokens)
        self.expected = {
            prefix: 1_000 + index
            for index, prefix in enumerate(
                sorted(
                    {()}
                    | {
                        path[:depth]
                        for path in self.schedule.candidate_paths
                        for depth in range(1, len(path) + 1)
                    },
                    key=lambda item: (len(item), item),
                )
            )
        }
        self.node_targets = {}
        for node in self.schedule.nodes[1:]:
            if node.depth < shared_count:
                self.node_targets[node.node_id] = 77
            else:
                self.node_targets[node.node_id] = self.expected[
                    node.prefix[shared_count:]
                ]

    def test_segment_node_and_flat_results_produce_the_same_relative_map(self) -> None:
        segment_targets = {
            segment.segment_id: tuple(
                self.node_targets[node_id] for node_id in segment.node_ids
            )
            for segment in self.schedule.segments
        }
        flat_targets = {
            path: tuple(
                self.expected[path[:depth]] for depth in range(len(path) + 1)
            )
            for path in self.schedule.candidate_paths
        }
        self.assertEqual(
            target_argmax_by_prefix_from_node_outputs(
                self.schedule, self.node_targets
            ),
            self.expected,
        )
        self.assertEqual(
            target_argmax_by_prefix_from_segment_outputs(
                self.schedule, segment_targets
            ),
            self.expected,
        )
        self.assertEqual(
            target_argmax_by_prefix_from_flat_leaf_results(
                self.schedule, flat_targets
            ),
            self.expected,
        )

    def test_missing_malformed_and_contradictory_targets_fail_closed(self) -> None:
        missing_node = dict(self.node_targets)
        missing_node.pop(next(iter(missing_node)))
        with self.assertRaisesRegex(ValueError, "do not cover the schedule"):
            target_argmax_by_prefix_from_node_outputs(self.schedule, missing_node)

        segment_targets = {
            segment.segment_id: tuple(
                self.node_targets[node_id] for node_id in segment.node_ids
            )
            for segment in self.schedule.segments
        }
        missing_segment = dict(segment_targets)
        missing_segment.pop(next(iter(missing_segment)))
        with self.assertRaisesRegex(ValueError, "segment targets do not cover"):
            target_argmax_by_prefix_from_segment_outputs(
                self.schedule, missing_segment
            )
        malformed_segment = dict(segment_targets)
        segment_id = next(iter(malformed_segment))
        malformed_segment[segment_id] = (*malformed_segment[segment_id], 123)
        with self.assertRaisesRegex(ValueError, "returned .* expected"):
            target_argmax_by_prefix_from_segment_outputs(
                self.schedule, malformed_segment
            )

        flat_targets = {
            path: tuple(
                self.expected[path[:depth]] for depth in range(len(path) + 1)
            )
            for path in self.schedule.candidate_paths
        }
        first, second = self.schedule.candidate_paths[:2]
        contradictory = dict(flat_targets)
        contradictory[second] = (flat_targets[first][0] + 1, *flat_targets[second][1:])
        with self.assertRaisesRegex(ValueError, "contradictory target argmax"):
            target_argmax_by_prefix_from_flat_leaf_results(
                self.schedule, contradictory
            )

    def test_empty_shared_prefix_requires_explicit_root_target(self) -> None:
        schedule = plan_prefix_segment_schedule(((1, 2), (1, 3)))
        node_targets = {node.node_id: 200 + node.node_id for node in schedule.nodes[1:]}
        with self.assertRaisesRegex(ValueError, "root_target_argmax is required"):
            target_argmax_by_prefix_from_node_outputs(schedule, node_targets)
        result = target_argmax_by_prefix_from_node_outputs(
            schedule, node_targets, root_target_argmax=199
        )
        self.assertEqual(result[()], 199)
        with self.assertRaisesRegex(ValueError, "must be omitted"):
            target_argmax_by_prefix_from_node_outputs(
                self.schedule, self.node_targets, root_target_argmax=199
            )


if __name__ == "__main__":
    unittest.main()
