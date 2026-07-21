from __future__ import annotations

import random
import unittest

from distributed_runtime.prefix_closed_schedule import (
    PrefixComputeStep,
    PrefixDivergenceStep,
    PrefixForkStep,
    plan_prefix_closed_schedule,
)


def _execute_symbolically(schedule) -> dict[int, tuple[int, ...]]:
    lanes: dict[int, tuple[int, ...]] = {0: ()}
    for index, operation in enumerate(schedule.operations):
        if operation.sequence_index != index:
            raise AssertionError("operation indices are not contiguous")
        if isinstance(operation, PrefixDivergenceStep):
            if lanes[operation.lane_id] != operation.prefix:
                raise AssertionError("divergence lane is not at its prefix")
        elif isinstance(operation, PrefixForkStep):
            if lanes[operation.source_lane_id] != operation.prefix:
                raise AssertionError("fork source is not at its immutable prefix")
            if operation.target_lane_id in lanes:
                raise AssertionError("fork target lane was reused")
            lanes[operation.target_lane_id] = operation.prefix
        elif isinstance(operation, PrefixComputeStep):
            parent = operation.prefix[:-1]
            if lanes[operation.lane_id] != parent:
                raise AssertionError("compute lane does not hold its parent prefix")
            lanes[operation.lane_id] = (*parent, operation.token)
        else:  # pragma: no cover - closed union guard
            raise AssertionError(f"unknown operation {operation!r}")
    return lanes


class PrefixClosedScheduleTests(unittest.TestCase):
    def test_shared_trunks_are_computed_once_and_reconstruct_every_leaf(self) -> None:
        schedule = plan_prefix_closed_schedule(
            ((10, 20, 31), (10, 40), (10, 20, 30)),
            shared_root_tokens=1,
        )
        self.assertEqual(
            schedule.candidate_paths,
            ((10, 20, 30), (10, 20, 31), (10, 40)),
        )
        self.assertEqual(schedule.cost.flat_leaf_token_steps, 11)
        self.assertEqual(schedule.cost.unique_prefix_token_steps, 6)
        self.assertEqual(schedule.cost.saved_token_steps, 5)
        self.assertAlmostEqual(schedule.cost.compute_reduction_ratio, 5 / 11)
        self.assertEqual(schedule.cost.prefix_closed_fork_steps, 2)
        self.assertEqual(schedule.cost.prefix_closed_lane_count, 3)

        lanes = _execute_symbolically(schedule)
        for leaf in schedule.leaves:
            self.assertEqual(lanes[leaf.lane_id], leaf.path)
        computed_prefixes = [
            operation.prefix
            for operation in schedule.operations
            if isinstance(operation, PrefixComputeStep)
        ]
        self.assertEqual(len(computed_prefixes), len(set(computed_prefixes)))
        self.assertEqual(
            set(computed_prefixes),
            {node.prefix for node in schedule.nodes if node.prefix},
        )

    def test_plan_is_byte_for_byte_deterministic_under_input_permutations(self) -> None:
        paths = ((7, 8, 9), (7, 8, 10), (7, 11), (12, 13))
        expected = plan_prefix_closed_schedule(paths, shared_root_tokens=1)
        for seed in range(25):
            shuffled = list(paths)
            random.Random(seed).shuffle(shuffled)
            self.assertEqual(
                plan_prefix_closed_schedule(shuffled, shared_root_tokens=1),
                expected,
            )

    def test_random_fixed_depth_tries_preserve_every_path_and_cost_identity(self) -> None:
        generator = random.Random(0xC0DEC0DE)
        for _case in range(200):
            depth = generator.randint(1, 8)
            width = generator.randint(1, min(32, 16**depth))
            paths: set[tuple[int, ...]] = set()
            while len(paths) < width:
                paths.add(tuple(generator.randrange(16) for _ in range(depth)))
            root_tokens = generator.randint(0, 3)
            schedule = plan_prefix_closed_schedule(
                tuple(paths),
                shared_root_tokens=root_tokens,
                max_paths=64,
                max_depth=16,
                max_nodes=1_024,
            )
            lanes = _execute_symbolically(schedule)
            for leaf in schedule.leaves:
                self.assertEqual(lanes[leaf.lane_id], leaf.path)
            unique_prefixes = {
                path[:prefix_length]
                for path in paths
                for prefix_length in range(1, len(path) + 1)
            }
            self.assertEqual(
                schedule.cost.unique_prefix_node_count,
                len(unique_prefixes),
            )
            self.assertEqual(
                schedule.cost.flat_leaf_token_steps,
                sum(len(path) + root_tokens for path in paths),
            )
            self.assertEqual(
                schedule.cost.unique_prefix_token_steps,
                len(unique_prefixes) + root_tokens,
            )
            self.assertEqual(
                schedule.cost.saved_token_steps,
                schedule.cost.flat_leaf_token_steps
                - schedule.cost.unique_prefix_token_steps,
            )
            self.assertEqual(
                schedule.cost.prefix_closed_lane_count,
                len(paths),
            )

    def test_single_leaf_has_no_artificial_compute_saving(self) -> None:
        schedule = plan_prefix_closed_schedule(((1, 2, 3),), shared_root_tokens=1)
        self.assertEqual(schedule.cost.saved_token_steps, 0)
        self.assertEqual(schedule.cost.compute_reduction_ratio, 0.0)
        self.assertEqual(schedule.cost.flat_leaf_fork_steps, 1)
        self.assertEqual(schedule.cost.prefix_closed_fork_steps, 0)
        self.assertEqual(_execute_symbolically(schedule)[0], (1, 2, 3))

    def test_invalid_or_unbounded_leaf_sets_fail_closed(self) -> None:
        invalid = (
            ((), "must not be empty"),
            (((1,), (1,)), "duplicates"),
            (((1,), (1, 2)), "prefix"),
            (((-1,),), "candidate token"),
            (((True,),), "candidate token"),
        )
        for paths, pattern in invalid:
            with self.subTest(paths=paths):
                with self.assertRaisesRegex(ValueError, pattern):
                    plan_prefix_closed_schedule(paths)
        with self.assertRaisesRegex(ValueError, "contains 2 paths"):
            plan_prefix_closed_schedule(((1,), (2,)), max_paths=1)
        with self.assertRaisesRegex(ValueError, "depth"):
            plan_prefix_closed_schedule(((1, 2),), max_depth=1)
        with self.assertRaisesRegex(ValueError, "nodes"):
            plan_prefix_closed_schedule(((1, 2), (1, 3)), max_nodes=3)


if __name__ == "__main__":
    unittest.main()
