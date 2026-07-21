from __future__ import annotations

from dataclasses import replace
import random
import unittest

from distributed_runtime.prefix_segment_executor import (
    EndPrefixLeaf,
    PrefixPhysicalCompute,
    PrefixPhysicalComputeBatch,
    PrefixPhysicalFork,
    PrefixSegmentExecutionError,
    PromotePrefixLeaf,
    execute_prefix_segment_schedule,
)
from distributed_runtime.prefix_segment_schedule import (
    PrefixComputeSegment,
    PrefixSegmentFork,
    plan_prefix_segment_schedule,
    target_argmax_by_prefix_from_flat_leaf_results,
)


def _target(path: tuple[int, ...]) -> int:
    value = 0x811C9DC5
    for token in path:
        value = ((value ^ token) * 0x01000193) & 0xFFFFFFFF
    return value


class _SymbolicBackend:
    def __init__(
        self,
        parent_physical_request_id: int,
        *,
        base_state: tuple[int, ...] = (70_001, 70_002),
        fail_on: str | None = None,
        malformed_compute: bool = False,
    ) -> None:
        self.parent_physical_request_id = parent_physical_request_id
        self.base_state = base_state
        self.states: dict[int, tuple[int, ...]] = {
            parent_physical_request_id: base_state
        }
        self.events: list[tuple[object, ...]] = []
        self.callback_operations: list[
            PrefixPhysicalFork | PrefixPhysicalCompute | PrefixPhysicalComputeBatch
        ] = []
        self.compute_call_count = 0
        self.fail_on = fail_on
        self.malformed_compute = malformed_compute

    def fork(self, operation: PrefixPhysicalFork) -> None:
        self.callback_operations.append(operation)
        self.events.append(
            (
                "carrier" if operation.carrier else "fork",
                operation.execution_index,
                operation.schedule_sequence_index,
                operation.source_lane_id,
                operation.target_lane_id,
                operation.fork_source_physical_request_id,
                operation.target_physical_request_id,
                operation.logical_owner_request_id,
            )
        )
        if self.fail_on == ("carrier" if operation.carrier else "fork"):
            raise RuntimeError("injected fork failure")
        if operation.fork_source_physical_request_id not in self.states:
            raise AssertionError("physical FORK source does not exist")
        if operation.target_physical_request_id in self.states:
            raise AssertionError("physical FORK target already exists")
        source_state = self.states[operation.fork_source_physical_request_id]
        if operation.carrier:
            if operation.fork_source_physical_request_id != self.parent_physical_request_id:
                raise AssertionError("carrier did not fork the real physical parent")
            if operation.prefix != ():
                raise AssertionError("carrier must start at the virtual root")
        elif source_state[len(self.base_state) :] != operation.prefix:
            raise AssertionError("nested FORK source is at the wrong prefix")
        self.states[operation.target_physical_request_id] = tuple(source_state)

    def compute(self, operation: PrefixPhysicalCompute) -> tuple[int, ...]:
        self.callback_operations.append(operation)
        self.compute_call_count += 1
        self.events.append(
            (
                "compute",
                operation.execution_index,
                operation.schedule_sequence_index,
                operation.segment_id,
                operation.lane_id,
                operation.physical_request_id,
                operation.tokens,
                operation.logical_owner_request_id,
            )
        )
        if self.fail_on == "compute":
            raise RuntimeError("injected compute failure")
        targets = self._apply_compute(operation)
        if self.malformed_compute:
            return (*targets, 123)
        return targets

    def _apply_compute(
        self,
        operation: PrefixPhysicalCompute,
    ) -> tuple[int, ...]:
        if operation.physical_request_id == self.parent_physical_request_id:
            raise AssertionError("executor attempted to compute on the real parent")
        state = self.states[operation.physical_request_id]
        targets: list[int] = []
        for token in operation.tokens:
            state = (*state, token)
            targets.append(_target(state[len(self.base_state) :]))
        self.states[operation.physical_request_id] = state
        return tuple(targets)


class _SymbolicBatchBackend(_SymbolicBackend):
    def __init__(
        self,
        parent_physical_request_id: int,
        *,
        batch_mode: str = "valid",
    ) -> None:
        super().__init__(parent_physical_request_id)
        self.batch_mode = batch_mode
        self.batch_calls: list[PrefixPhysicalComputeBatch] = []

    def compute_batch(
        self,
        batch: PrefixPhysicalComputeBatch,
    ) -> tuple[tuple[int, ...], ...]:
        self.callback_operations.append(batch)
        self.batch_calls.append(batch)
        self.events.append(
            (
                "batch",
                batch.execution_index,
                batch.frontier_index,
                batch.compatible_group_id,
                batch.segment_ids,
                batch.start_depth,
                batch.token_count,
            )
        )
        if self.batch_mode == "raise":
            raise RuntimeError("injected batch failure")
        vectors = tuple(
            self._apply_compute(operation) for operation in batch.operations
        )
        if self.batch_mode == "missing_segment":
            return vectors[:-1]
        if self.batch_mode == "extra_target":
            return ((*vectors[0], 123), *vectors[1:])
        if self.batch_mode == "out_of_range":
            return (((1 << 32), *vectors[0][1:]), *vectors[1:])
        return vectors


def _execute(schedule, *, backend=None, owner=900_001, parent=800_001):
    if backend is None:
        backend = _SymbolicBackend(parent)
    lane_ids = tuple(1_000_000 + index for index in range(schedule.cost.lane_count))
    result = execute_prefix_segment_schedule(
        schedule,
        logical_owner_request_id=owner,
        intact_parent_physical_request_id=parent,
        lane_physical_request_ids=lane_ids,
        backend=backend,
    )
    return result, backend, lane_ids


class PrefixSegmentExecutorTests(unittest.TestCase):
    def test_forged_schedule_metadata_is_rejected_before_carrier_mutation(self) -> None:
        schedule = plan_prefix_segment_schedule(
            ((1, 2, 3), (1, 2, 4)), shared_prefix_tokens=(9,)
        )
        forged_leaf = replace(
            schedule.leaves[0],
            candidate_path=(0xDEADBEEF,),
        )
        forged = replace(schedule, leaves=(forged_leaf, *schedule.leaves[1:]))
        backend = _SymbolicBackend(77)

        with self.assertRaises(PrefixSegmentExecutionError) as rejected:
            execute_prefix_segment_schedule(
                forged,
                logical_owner_request_id=88,
                intact_parent_physical_request_id=77,
                lane_physical_request_ids=(100, 101),
                backend=backend,
            )

        self.assertTrue(rejected.exception.fallback_allowed)
        self.assertEqual(rejected.exception.phase, "preflight")
        self.assertEqual(backend.events, [])
        self.assertEqual(backend.states, {77: backend.base_state})

    def test_base_exception_after_callback_mutation_is_wrapped_route_fatal(self) -> None:
        schedule = plan_prefix_segment_schedule(
            ((1, 2), (1, 3)), shared_prefix_tokens=(9,)
        )

        class InterruptCarrier(_SymbolicBackend):
            def fork(self, operation: PrefixPhysicalFork) -> None:
                super().fork(operation)
                if operation.carrier:
                    raise KeyboardInterrupt("after carrier mutation")

        class InterruptCompute(_SymbolicBackend):
            def compute(self, operation: PrefixPhysicalCompute) -> tuple[int, ...]:
                super().compute(operation)
                raise KeyboardInterrupt("after compute mutation")

        class InterruptBatch(_SymbolicBatchBackend):
            def compute_batch(
                self, batch: PrefixPhysicalComputeBatch
            ) -> tuple[tuple[int, ...], ...]:
                super().compute_batch(batch)
                raise KeyboardInterrupt("after batch mutation")

        cases = (
            (InterruptCarrier(77), "carrier_fork"),
            (InterruptCompute(77), "compute"),
            (InterruptBatch(77), "compute_batch"),
        )
        for backend, phase in cases:
            with self.subTest(phase=phase):
                with self.assertRaises(PrefixSegmentExecutionError) as failure:
                    execute_prefix_segment_schedule(
                        schedule,
                        logical_owner_request_id=88,
                        intact_parent_physical_request_id=77,
                        lane_physical_request_ids=(100, 101),
                        backend=backend,
                    )
                self.assertTrue(failure.exception.route_fatal)
                self.assertFalse(failure.exception.fallback_allowed)
                self.assertEqual(failure.exception.phase, phase)

    def test_carrier_and_nested_forks_use_physical_sources_but_keep_real_owner(
        self,
    ) -> None:
        schedule = plan_prefix_segment_schedule(
            ((1, 2, 3), (1, 2, 4), (1, 8, 9), (7, 6)),
            shared_prefix_tokens=(91, 92),
        )
        owner = 44
        parent = 55
        result, backend, lane_ids = _execute(
            schedule, owner=owner, parent=parent
        )

        self.assertFalse(result.parent_was_compute_target)
        self.assertEqual(backend.states[parent], backend.base_state)
        self.assertEqual(result.logical_owner_request_id, owner)
        self.assertEqual(result.intact_parent_physical_request_id, parent)
        self.assertEqual(
            tuple(binding.physical_request_id for binding in result.lane_bindings),
            lane_ids,
        )
        self.assertEqual(len(result.operations), 1 + len(schedule.operations))

        carrier = result.operations[0]
        self.assertIsInstance(carrier, PrefixPhysicalFork)
        assert isinstance(carrier, PrefixPhysicalFork)
        self.assertTrue(carrier.carrier)
        self.assertEqual(carrier.fork_source_physical_request_id, parent)
        self.assertEqual(carrier.target_physical_request_id, lane_ids[0])
        self.assertEqual(carrier.logical_owner_request_id, owner)

        nested = [
            operation
            for operation in result.operations
            if isinstance(operation, PrefixPhysicalFork) and not operation.carrier
        ]
        self.assertTrue(nested)
        for operation in nested:
            self.assertIn(operation.fork_source_physical_request_id, lane_ids)
            self.assertIn(operation.target_physical_request_id, lane_ids)
            self.assertNotEqual(operation.fork_source_physical_request_id, owner)
            self.assertEqual(operation.logical_owner_request_id, owner)
        for leaf in result.leaves:
            self.assertEqual(
                backend.states[leaf.physical_request_id],
                backend.base_state + leaf.full_path,
            )

    def test_callbacks_follow_carrier_then_exact_planner_order(self) -> None:
        schedule = plan_prefix_segment_schedule(
            ((1, 2, 3, 4), (1, 2, 3, 5), (1, 2, 8, 9), (7, 6)),
            shared_prefix_tokens=(99,),
        )
        result, backend, lane_ids = _execute(schedule)
        expected: list[tuple[object, ...]] = [
            ("carrier", 0, None, None, 0, 800_001, lane_ids[0], 900_001)
        ]
        for execution_index, operation in enumerate(schedule.operations, start=1):
            if isinstance(operation, PrefixSegmentFork):
                expected.append(
                    (
                        "fork",
                        execution_index,
                        operation.sequence_index,
                        operation.source_lane_id,
                        operation.target_lane_id,
                        lane_ids[operation.source_lane_id],
                        lane_ids[operation.target_lane_id],
                        900_001,
                    )
                )
            else:
                assert isinstance(operation, PrefixComputeSegment)
                expected.append(
                    (
                        "compute",
                        execution_index,
                        operation.sequence_index,
                        operation.segment_id,
                        operation.lane_id,
                        lane_ids[operation.lane_id],
                        operation.tokens,
                        900_001,
                    )
                )
        self.assertEqual(backend.events, expected)
        self.assertEqual(backend.compute_call_count, len(schedule.segments))

        for frontier in range(schedule.cost.frontier_count):
            frontier_operations = [
                operation
                for operation in result.operations[1:]
                if operation.frontier_index == frontier
            ]
            seen_compute = False
            for operation in frontier_operations:
                if isinstance(operation, PrefixPhysicalCompute):
                    seen_compute = True
                elif isinstance(operation, PrefixPhysicalFork):
                    self.assertFalse(seen_compute)

    def test_compatible_segments_use_one_exact_batch_call_per_group(self) -> None:
        schedule = plan_prefix_segment_schedule(
            ((1, 2, 3, 4), (1, 2, 3, 5), (1, 2, 6, 7), (8, 9)),
            shared_prefix_tokens=(99,),
        )
        parent = 810_000
        backend = _SymbolicBatchBackend(parent)
        result, backend, _lane_ids = _execute(
            schedule, backend=backend, parent=parent
        )

        self.assertEqual(backend.compute_call_count, 0)
        self.assertEqual(
            len(backend.batch_calls), len(schedule.compatible_frontier_groups)
        )
        groups_by_id = {
            group.group_id: group for group in schedule.compatible_frontier_groups
        }
        multi_segment_batches = []
        for batch in backend.batch_calls:
            group = groups_by_id[batch.compatible_group_id]
            self.assertEqual(batch.frontier_index, group.frontier_index)
            self.assertEqual(batch.start_depth, group.start_depth)
            self.assertEqual(batch.token_count, group.token_count)
            self.assertEqual(batch.segment_ids, group.segment_ids)
            self.assertTrue(
                all(
                    operation.frontier_index == batch.frontier_index
                    and operation.compatible_group_id == batch.compatible_group_id
                    and operation.start_depth == batch.start_depth
                    and operation.token_count == batch.token_count
                    for operation in batch.operations
                )
            )
            if len(batch.operations) > 1:
                multi_segment_batches.append(batch)
        self.assertTrue(multi_segment_batches)

        # Public execution semantics stay scalar: one trace operation and one
        # output record per segment even though the backend saw one group call.
        self.assertEqual(len(result.operations), 1 + len(schedule.operations))
        self.assertEqual(len(result.segment_outputs), len(schedule.segments))
        self.assertEqual(
            [
                operation.segment_id
                for operation in result.operations
                if isinstance(operation, PrefixPhysicalCompute)
            ],
            [segment.segment_id for segment in schedule.segments],
        )

        callback_positions = {
            id(operation): position
            for position, operation in enumerate(backend.callback_operations)
        }
        for frontier in range(schedule.cost.frontier_count):
            fork_positions = [
                callback_positions[id(operation)]
                for operation in backend.callback_operations
                if isinstance(operation, PrefixPhysicalFork)
                and not operation.carrier
                and operation.frontier_index == frontier
            ]
            batch_positions = [
                callback_positions[id(operation)]
                for operation in backend.batch_calls
                if operation.frontier_index == frontier
            ]
            if fork_positions and batch_positions:
                self.assertLess(max(fork_positions), min(batch_positions))

    def test_backend_without_batch_keeps_sequential_callback_behavior(self) -> None:
        schedule = plan_prefix_segment_schedule(
            ((1, 2, 3), (1, 2, 4), (1, 5, 6), (7, 8)),
            shared_prefix_tokens=(90,),
        )
        result, backend, _lane_ids = _execute(schedule)
        self.assertFalse(hasattr(backend, "compute_batch"))
        self.assertEqual(backend.compute_call_count, len(schedule.segments))
        self.assertEqual(
            sum(
                isinstance(operation, PrefixPhysicalCompute)
                for operation in backend.callback_operations
            ),
            len(schedule.segments),
        )
        self.assertEqual(len(result.segment_outputs), len(schedule.segments))

    def test_malformed_or_failed_batch_is_route_fatal(self) -> None:
        schedule = plan_prefix_segment_schedule(
            ((1, 2, 3), (1, 2, 4), (1, 5, 6), (7, 8)),
            shared_prefix_tokens=(90,),
        )
        for mode in ("raise", "missing_segment", "extra_target", "out_of_range"):
            with self.subTest(mode=mode):
                parent = 720_000
                backend = _SymbolicBatchBackend(parent, batch_mode=mode)
                with self.assertRaises(PrefixSegmentExecutionError) as failure:
                    execute_prefix_segment_schedule(
                        schedule,
                        logical_owner_request_id=730_000,
                        intact_parent_physical_request_id=parent,
                        lane_physical_request_ids=tuple(
                            740_000 + lane
                            for lane in range(schedule.cost.lane_count)
                        ),
                        backend=backend,
                    )
                self.assertTrue(failure.exception.route_fatal)
                self.assertFalse(failure.exception.fallback_allowed)
                self.assertEqual(failure.exception.phase, "compute_batch")
                self.assertIsInstance(
                    failure.exception.attempted_operation,
                    PrefixPhysicalComputeBatch,
                )
                self.assertTrue(backend.batch_calls)
                self.assertIsInstance(
                    backend.callback_operations[0], PrefixPhysicalFork
                )
                assert isinstance(
                    backend.callback_operations[0], PrefixPhysicalFork
                )
                self.assertTrue(backend.callback_operations[0].carrier)

    def test_returns_segment_and_node_outputs_equal_to_flat_leaf_execution(self) -> None:
        schedule = plan_prefix_segment_schedule(
            ((1, 2, 3), (1, 2, 4), (1, 5, 6), (7, 8, 9)),
            shared_prefix_tokens=(90, 91),
        )
        result, _backend, _lane_ids = _execute(schedule)
        flat = {
            path: tuple(
                _target(schedule.shared_prefix_tokens + path[:depth])
                for depth in range(len(path) + 1)
            )
            for path in schedule.candidate_paths
        }
        expected = target_argmax_by_prefix_from_flat_leaf_results(schedule, flat)
        self.assertEqual(result.target_argmax_by_prefix(), expected)
        self.assertEqual(len(result.segment_outputs), len(schedule.segments))
        self.assertEqual(len(result.node_outputs), len(schedule.nodes) - 1)
        self.assertEqual(
            {output.segment_id for output in result.segment_outputs},
            set(range(len(schedule.segments))),
        )
        self.assertEqual(
            {output.node_id for output in result.node_outputs},
            set(range(1, len(schedule.nodes))),
        )

    def test_secondary_leaf_is_a_valid_promotion_carrier(self) -> None:
        schedule = plan_prefix_segment_schedule(
            ((1, 2, 3), (1, 2, 4), (1, 5, 6), (7, 8, 9)),
            shared_prefix_tokens=(90,),
        )
        result, backend, _lane_ids = _execute(schedule, owner=700, parent=701)
        secondary = next(leaf for leaf in result.leaves if leaf.lane_id != 0)
        plan = result.plan_leaf_commit(secondary.candidate_path)

        self.assertEqual(plan.winner, secondary)
        self.assertEqual(len(plan.loser_ends), len(result.leaves) - 1)
        self.assertTrue(all(isinstance(item, EndPrefixLeaf) for item in plan.operations[:-1]))
        self.assertIsInstance(plan.operations[-1], PromotePrefixLeaf)
        self.assertEqual(plan.promotion.logical_owner_request_id, 700)
        self.assertEqual(plan.promotion.intact_parent_physical_request_id, 701)
        self.assertEqual(
            plan.promotion.winner_physical_request_id,
            secondary.physical_request_id,
        )
        self.assertNotIn(
            secondary.physical_request_id,
            {item.physical_request_id for item in plan.loser_ends},
        )
        self.assertEqual(
            {item.physical_request_id for item in plan.loser_ends},
            {
                leaf.physical_request_id
                for leaf in result.leaves
                if leaf is not secondary
            },
        )

        # Symbolically apply the plan: the real physical parent was intact up
        # to this point and can receive any leaf, not only lane 0.
        self.assertEqual(backend.states[701], backend.base_state)
        for end in plan.loser_ends:
            backend.states.pop(end.physical_request_id)
        backend.states[plan.promotion.intact_parent_physical_request_id] = tuple(
            backend.states[plan.promotion.winner_physical_request_id]
        )
        self.assertEqual(
            backend.states[701], backend.base_state + secondary.full_path
        )

    def test_preflight_error_allows_fallback_but_any_callback_attempt_is_fatal(
        self,
    ) -> None:
        schedule = plan_prefix_segment_schedule(
            ((1, 2), (1, 3)), shared_prefix_tokens=(9,)
        )
        backend = _SymbolicBackend(77)
        with self.assertRaises(PrefixSegmentExecutionError) as rejected:
            execute_prefix_segment_schedule(
                schedule,
                logical_owner_request_id=88,
                intact_parent_physical_request_id=77,
                lane_physical_request_ids=(100, 100),
                backend=backend,
            )
        self.assertTrue(rejected.exception.fallback_allowed)
        self.assertFalse(rejected.exception.route_fatal)
        self.assertEqual(rejected.exception.phase, "preflight")
        self.assertEqual(backend.events, [])
        self.assertEqual(backend.states, {77: backend.base_state})

        failing_compute = _SymbolicBackend(77, fail_on="compute")
        with self.assertRaises(PrefixSegmentExecutionError) as mutated:
            execute_prefix_segment_schedule(
                schedule,
                logical_owner_request_id=88,
                intact_parent_physical_request_id=77,
                lane_physical_request_ids=(100, 101),
                backend=failing_compute,
            )
        self.assertTrue(mutated.exception.route_fatal)
        self.assertFalse(mutated.exception.fallback_allowed)
        self.assertEqual(mutated.exception.phase, "compute")
        self.assertGreaterEqual(mutated.exception.completed_operation_count, 1)
        self.assertTrue(failing_compute.events)

        failing_carrier = _SymbolicBackend(77, fail_on="carrier")
        with self.assertRaises(PrefixSegmentExecutionError) as uncertain_fork:
            execute_prefix_segment_schedule(
                schedule,
                logical_owner_request_id=88,
                intact_parent_physical_request_id=77,
                lane_physical_request_ids=(100, 101),
                backend=failing_carrier,
            )
        self.assertTrue(uncertain_fork.exception.route_fatal)
        self.assertEqual(uncertain_fork.exception.phase, "carrier_fork")
        self.assertEqual(uncertain_fork.exception.completed_operation_count, 0)

        malformed = _SymbolicBackend(77, malformed_compute=True)
        with self.assertRaises(PrefixSegmentExecutionError) as malformed_result:
            execute_prefix_segment_schedule(
                schedule,
                logical_owner_request_id=88,
                intact_parent_physical_request_id=77,
                lane_physical_request_ids=(100, 101),
                backend=malformed,
            )
        self.assertTrue(malformed_result.exception.route_fatal)
        self.assertEqual(malformed_result.exception.phase, "compute")

    def test_random_symbolic_execution_matches_flat_prefix_map_and_keeps_parent(
        self,
    ) -> None:
        generator = random.Random(0xE7EC0702)
        for case in range(200):
            depth = generator.randint(1, 7)
            width = generator.randint(1, min(20, 8**depth))
            paths: set[tuple[int, ...]] = set()
            while len(paths) < width:
                paths.add(tuple(generator.randrange(8) for _ in range(depth)))
            shared = tuple(
                generator.randrange(20, 40)
                for _ in range(generator.randint(1, 3))
            )
            schedule = plan_prefix_segment_schedule(
                tuple(paths),
                shared_prefix_tokens=shared,
                max_paths=32,
                max_depth=12,
                max_nodes=1_024,
            )
            parent = 50_000 + case
            owner = 80_000 + case
            backend = _SymbolicBackend(parent)
            lane_ids = tuple(
                2_000_000 + case * 64 + lane
                for lane in range(schedule.cost.lane_count)
            )
            result = execute_prefix_segment_schedule(
                schedule,
                logical_owner_request_id=owner,
                intact_parent_physical_request_id=parent,
                lane_physical_request_ids=lane_ids,
                backend=backend,
            )
            flat = {
                path: tuple(
                    _target(shared + path[:prefix_depth])
                    for prefix_depth in range(len(path) + 1)
                )
                for path in schedule.candidate_paths
            }
            self.assertEqual(
                result.target_argmax_by_prefix(),
                target_argmax_by_prefix_from_flat_leaf_results(schedule, flat),
            )
            self.assertEqual(backend.states[parent], backend.base_state)
            self.assertFalse(result.parent_was_compute_target)
            for leaf in result.leaves:
                self.assertEqual(
                    backend.states[leaf.physical_request_id],
                    backend.base_state + leaf.full_path,
                )

    def test_random_batched_execution_matches_sequential_and_flat(self) -> None:
        generator = random.Random(0xBA7C4ED)
        for case in range(150):
            depth = generator.randint(1, 7)
            width = generator.randint(1, min(20, 8**depth))
            paths: set[tuple[int, ...]] = set()
            while len(paths) < width:
                paths.add(tuple(generator.randrange(8) for _ in range(depth)))
            shared = tuple(
                generator.randrange(20, 40)
                for _ in range(generator.randint(1, 3))
            )
            schedule = plan_prefix_segment_schedule(
                tuple(paths),
                shared_prefix_tokens=shared,
                max_paths=32,
                max_depth=12,
                max_nodes=1_024,
            )
            parent = 3_000_000 + case
            lane_ids = tuple(
                4_000_000 + case * 64 + lane
                for lane in range(schedule.cost.lane_count)
            )
            sequential_backend = _SymbolicBackend(parent)
            sequential = execute_prefix_segment_schedule(
                schedule,
                logical_owner_request_id=5_000_000 + case,
                intact_parent_physical_request_id=parent,
                lane_physical_request_ids=lane_ids,
                backend=sequential_backend,
            )
            batched_backend = _SymbolicBatchBackend(parent)
            batched = execute_prefix_segment_schedule(
                schedule,
                logical_owner_request_id=5_000_000 + case,
                intact_parent_physical_request_id=parent,
                lane_physical_request_ids=lane_ids,
                backend=batched_backend,
            )
            flat = {
                path: tuple(
                    _target(shared + path[:prefix_depth])
                    for prefix_depth in range(len(path) + 1)
                )
                for path in schedule.candidate_paths
            }
            expected = target_argmax_by_prefix_from_flat_leaf_results(
                schedule, flat
            )
            self.assertEqual(sequential.target_argmax_by_prefix(), expected)
            self.assertEqual(batched.target_argmax_by_prefix(), expected)
            self.assertEqual(
                tuple(
                    (output.segment_id, output.node_ids, output.target_argmax)
                    for output in batched.segment_outputs
                ),
                tuple(
                    (output.segment_id, output.node_ids, output.target_argmax)
                    for output in sequential.segment_outputs
                ),
            )
            self.assertEqual(
                tuple(type(operation) for operation in batched.operations),
                tuple(type(operation) for operation in sequential.operations),
            )
            self.assertEqual(
                len(batched_backend.batch_calls),
                len(schedule.compatible_frontier_groups),
            )
            self.assertEqual(batched_backend.compute_call_count, 0)
            self.assertEqual(batched_backend.states[parent], batched_backend.base_state)


if __name__ == "__main__":
    unittest.main()
