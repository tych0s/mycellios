from __future__ import annotations

import itertools
import unittest
from collections import Counter
from dataclasses import dataclass

from distributed_runtime.prefix_closed_schedule import (
    PrefixComputeStep,
    PrefixDivergenceStep,
    PrefixForkStep,
    plan_prefix_closed_schedule,
)
from distributed_runtime.prefix_cow_projection import (
    CowKVPhysicalCost,
    project_prefix_closed_cow_kv,
)


UINT32_MAX = (1 << 32) - 1
UINT64_MAX = (1 << 64) - 1


@dataclass
class _OracleBlock:
    used_tokens: int
    ref_count: int = 1


class _RefcountOracle:
    """Small block-identity simulator independent of the production formula."""

    def __init__(self, parent_tokens: int, block_tokens: int) -> None:
        self.block_tokens = block_tokens
        self.blocks: dict[int, _OracleBlock] = {}
        self.requests: dict[object, list[int]] = {"parent": []}
        self.next_block_id = 0
        self.peak_live_blocks = 0
        self.copied_tail_blocks = 0
        self.append_allocated_blocks = 0
        self.fork_count = 0
        self._append("parent", parent_tokens, account=False)
        self.parent_block_ids = tuple(self.requests["parent"])
        self.parent_contents = tuple(
            self.blocks[block_id].used_tokens
            for block_id in self.parent_block_ids
        )
        self.parent_blocks = len(self.blocks)

    def fork(self, child: object, parent: object) -> None:
        if child in self.requests:
            raise AssertionError(f"oracle request {child!r} already exists")
        child_blocks: list[int] = []
        for block_id in self.requests[parent]:
            block = self.blocks[block_id]
            if block.used_tokens == self.block_tokens:
                block.ref_count += 1
                child_blocks.append(block_id)
            else:
                copied_id = self._allocate(block.used_tokens)
                child_blocks.append(copied_id)
                self.copied_tail_blocks += 1
        self.requests[child] = child_blocks
        self.fork_count += 1
        self._observe_peak()

    def append(self, request: object, token_count: int) -> None:
        self._append(request, token_count, account=True)

    def _append(self, request: object, token_count: int, *, account: bool) -> None:
        remaining = token_count
        while remaining:
            request_blocks = self.requests[request]
            if request_blocks:
                tail = self.blocks[request_blocks[-1]]
            else:
                tail = None
            if tail is None or tail.used_tokens == self.block_tokens:
                block_id = self._allocate(0)
                request_blocks.append(block_id)
                tail = self.blocks[block_id]
                if account:
                    self.append_allocated_blocks += 1
            if tail.ref_count != 1:
                raise AssertionError("oracle attempted to mutate a shared partial block")
            written = min(remaining, self.block_tokens - tail.used_tokens)
            tail.used_tokens += written
            remaining -= written
        self._observe_peak()

    def _allocate(self, used_tokens: int) -> int:
        block_id = self.next_block_id
        self.next_block_id += 1
        self.blocks[block_id] = _OracleBlock(used_tokens)
        return block_id

    def _observe_peak(self) -> None:
        self.peak_live_blocks = max(self.peak_live_blocks, len(self.blocks))

    def assert_parent_intact(self) -> None:
        if tuple(self.requests["parent"]) != self.parent_block_ids:
            raise AssertionError("oracle mutated the parent's block table")
        contents = tuple(
            self.blocks[block_id].used_tokens
            for block_id in self.parent_block_ids
        )
        if contents != self.parent_contents:
            raise AssertionError("oracle mutated the parent's KV contents")
        expected_refs = Counter(
            block_id
            for block_table in self.requests.values()
            for block_id in block_table
        )
        actual_refs = {
            block_id: block.ref_count for block_id, block in self.blocks.items()
        }
        if dict(expected_refs) != actual_refs:
            raise AssertionError(
                f"oracle refcounts disagree: {dict(expected_refs)!r} != "
                f"{actual_refs!r}"
            )
        for block_id, block in self.blocks.items():
            if block.used_tokens < self.block_tokens and block.ref_count != 1:
                raise AssertionError(
                    f"oracle partial block {block_id} is unexpectedly shared"
                )

    @property
    def incremental_blocks(self) -> int:
        return len(self.blocks) - self.parent_blocks


def _oracle_prefix_closed(
    parent_tokens: int,
    block_tokens: int,
    paths: tuple[tuple[int, ...], ...],
) -> _RefcountOracle:
    schedule = plan_prefix_closed_schedule(paths, shared_root_tokens=1)
    oracle = _RefcountOracle(parent_tokens, block_tokens)
    oracle.fork(0, "parent")
    oracle.append(0, 1)
    for operation in schedule.operations:
        if isinstance(operation, PrefixDivergenceStep):
            continue
        if isinstance(operation, PrefixForkStep):
            oracle.fork(operation.target_lane_id, operation.source_lane_id)
            continue
        if isinstance(operation, PrefixComputeStep):
            oracle.append(operation.lane_id, 1)
            continue
        raise AssertionError(f"unknown schedule operation {operation!r}")
    oracle.assert_parent_intact()
    return oracle


def _oracle_naive_flat(
    parent_tokens: int,
    block_tokens: int,
    paths: tuple[tuple[int, ...], ...],
) -> _RefcountOracle:
    oracle = _RefcountOracle(parent_tokens, block_tokens)
    for leaf_id, path in enumerate(paths):
        request = ("leaf", leaf_id)
        oracle.fork(request, "parent")
        oracle.append(request, 1 + len(path))
    oracle.assert_parent_intact()
    return oracle


def _prefix_free_path_sets() -> tuple[tuple[tuple[int, ...], ...], ...]:
    # Every non-empty subset of the full binary depth-three frontier exercises
    # all equal-depth trie shapes.  Adding every prefix-free subset through
    # depth two covers variable-length antichains as well.
    front = tuple(itertools.product(range(2), repeat=3))
    shallow = tuple(
        path
        for depth in range(1, 3)
        for path in itertools.product(range(2), repeat=depth)
    )
    path_sets: set[tuple[tuple[int, ...], ...]] = set()
    for universe in (front, shallow):
        for mask in range(1, 1 << len(universe)):
            paths = tuple(
                path for index, path in enumerate(universe) if mask & (1 << index)
            )
            if any(
                len(left) < len(right) and right[: len(left)] == left
                for left in paths
                for right in paths
            ):
                continue
            path_sets.add(tuple(sorted(paths)))
    return tuple(sorted(path_sets))


class PrefixClosedCowProjectionTests(unittest.TestCase):
    def test_boundary_and_partial_tail_examples_are_exact(self) -> None:
        boundary = project_prefix_closed_cow_kv(
            parent_tokens=8,
            block_tokens=4,
            bytes_per_block=64,
            pending_token=99,
            candidate_paths=((1,), (2,)),
        )
        self.assertEqual(boundary.parent_blocks, 2)
        self.assertEqual(boundary.prefix_closed.incremental_blocks, 2)
        self.assertEqual(boundary.prefix_closed.copied_tail_blocks, 1)
        self.assertEqual(boundary.prefix_closed.append_allocated_blocks, 1)
        self.assertEqual(boundary.prefix_closed.peak_live_blocks, 4)
        self.assertEqual(boundary.naive_flat.incremental_blocks, 2)
        self.assertEqual(boundary.saved_incremental_blocks, 0)

        tail_filled_by_pending = project_prefix_closed_cow_kv(
            parent_tokens=7,
            block_tokens=4,
            bytes_per_block=64,
            pending_token=99,
            candidate_paths=((1,), (2,)),
        )
        self.assertEqual(tail_filled_by_pending.prefix_closed.incremental_blocks, 3)
        self.assertEqual(tail_filled_by_pending.prefix_closed.copied_tail_blocks, 1)
        self.assertEqual(
            tail_filled_by_pending.prefix_closed.append_allocated_blocks, 2
        )
        self.assertEqual(tail_filled_by_pending.naive_flat.incremental_blocks, 4)
        self.assertEqual(tail_filled_by_pending.saved_incremental_blocks, 1)
        self.assertEqual(tail_filled_by_pending.saved_incremental_bytes, 64)
        self.assertEqual(
            tail_filled_by_pending.physical_block_reduction_ratio, 0.25
        )

    def test_shared_prefix_that_completes_a_block_becomes_zero_copy(self) -> None:
        projection = project_prefix_closed_cow_kv(
            parent_tokens=8,
            block_tokens=4,
            bytes_per_block=128,
            pending_token=77,
            candidate_paths=((10, 11, 12, 1), (10, 11, 12, 2)),
        )
        self.assertEqual(projection.prefix_closed.copied_tail_blocks, 0)
        self.assertEqual(projection.prefix_closed.append_allocated_blocks, 3)
        self.assertEqual(projection.prefix_closed.incremental_blocks, 3)
        self.assertEqual(projection.prefix_closed.peak_live_blocks, 5)
        self.assertEqual(projection.naive_flat.incremental_blocks, 4)
        self.assertEqual(projection.saved_incremental_blocks, 1)

    def test_nested_divergences_inside_one_partial_block_do_not_fake_savings(self) -> None:
        projection = project_prefix_closed_cow_kv(
            parent_tokens=0,
            block_tokens=4,
            bytes_per_block=32,
            pending_token=5,
            candidate_paths=(
                (1, 1, 1),
                (1, 1, 2),
                (1, 2, 1),
                (1, 2, 2),
            ),
        )
        self.assertEqual(projection.prefix_closed.append_allocated_blocks, 1)
        self.assertEqual(projection.prefix_closed.copied_tail_blocks, 3)
        self.assertEqual(projection.prefix_closed.incremental_blocks, 4)
        self.assertEqual(projection.naive_flat.incremental_blocks, 4)
        self.assertEqual(projection.saved_incremental_blocks, 0)

    def test_exhaustive_small_tries_match_block_identity_refcount_oracle(self) -> None:
        path_sets = _prefix_free_path_sets()
        self.assertGreater(len(path_sets), 250)
        checked = 0
        for block_tokens in range(1, 5):
            for parent_tokens in range(0, 9):
                for paths in path_sets:
                    with self.subTest(
                        block_tokens=block_tokens,
                        parent_tokens=parent_tokens,
                        paths=paths,
                    ):
                        projection = project_prefix_closed_cow_kv(
                            parent_tokens=parent_tokens,
                            block_tokens=block_tokens,
                            bytes_per_block=37,
                            pending_token=3,
                            candidate_paths=paths,
                        )
                        prefix_oracle = _oracle_prefix_closed(
                            parent_tokens, block_tokens, paths
                        )
                        naive_oracle = _oracle_naive_flat(
                            parent_tokens, block_tokens, projection.candidate_paths
                        )
                        self.assertEqual(
                            projection.prefix_closed.incremental_blocks,
                            prefix_oracle.incremental_blocks,
                        )
                        self.assertEqual(
                            projection.prefix_closed.peak_live_blocks,
                            prefix_oracle.peak_live_blocks,
                        )
                        self.assertEqual(
                            projection.prefix_closed.copied_tail_blocks,
                            prefix_oracle.copied_tail_blocks,
                        )
                        self.assertEqual(
                            projection.prefix_closed.append_allocated_blocks,
                            prefix_oracle.append_allocated_blocks,
                        )
                        self.assertEqual(
                            projection.prefix_closed.fork_count,
                            prefix_oracle.fork_count,
                        )
                        self.assertEqual(
                            projection.naive_flat.incremental_blocks,
                            naive_oracle.incremental_blocks,
                        )
                        self.assertEqual(
                            projection.naive_flat.peak_live_blocks,
                            naive_oracle.peak_live_blocks,
                        )
                        self.assertEqual(
                            projection.naive_flat.copied_tail_blocks,
                            naive_oracle.copied_tail_blocks,
                        )
                        self.assertEqual(
                            projection.naive_flat.append_allocated_blocks,
                            naive_oracle.append_allocated_blocks,
                        )
                        self.assertGreaterEqual(
                            projection.saved_incremental_blocks, 0
                        )
                        checked += 1
        self.assertGreater(checked, 9_000)

    def test_projection_is_independent_of_candidate_input_order(self) -> None:
        paths = ((8, 1, 3), (8, 1, 4), (8, 2), (9, 5))
        expected = project_prefix_closed_cow_kv(
            parent_tokens=13,
            block_tokens=4,
            bytes_per_block=96,
            pending_token=42,
            candidate_paths=paths,
        )
        for permutation in itertools.permutations(paths):
            self.assertEqual(
                project_prefix_closed_cow_kv(
                    parent_tokens=13,
                    block_tokens=4,
                    bytes_per_block=96,
                    pending_token=42,
                    candidate_paths=permutation,
                ),
                expected,
            )

    def test_single_leaf_matches_flat_baseline_for_every_alignment(self) -> None:
        for block_tokens in range(1, 9):
            for parent_tokens in range(0, 2 * block_tokens + 1):
                projection = project_prefix_closed_cow_kv(
                    parent_tokens=parent_tokens,
                    block_tokens=block_tokens,
                    bytes_per_block=16,
                    pending_token=0,
                    candidate_paths=((1, 2, 3),),
                )
                self.assertEqual(projection.prefix_closed, projection.naive_flat)
                self.assertEqual(projection.saved_incremental_blocks, 0)

    def test_invalid_values_limits_and_byte_overflow_fail_closed(self) -> None:
        base = {
            "parent_tokens": 0,
            "block_tokens": 4,
            "bytes_per_block": 64,
            "pending_token": 1,
            "candidate_paths": ((2,),),
        }
        invalid = (
            ("parent_tokens", -1),
            ("parent_tokens", True),
            ("block_tokens", 0),
            ("block_tokens", False),
            ("bytes_per_block", 0),
            ("bytes_per_block", 1.5),
            ("pending_token", -1),
            ("pending_token", UINT32_MAX + 1),
        )
        for name, value in invalid:
            with self.subTest(name=name, value=value):
                values = dict(base)
                values[name] = value
                with self.assertRaisesRegex(ValueError, name):
                    project_prefix_closed_cow_kv(**values)

        with self.assertRaisesRegex(ValueError, "sequence-length"):
            project_prefix_closed_cow_kv(
                parent_tokens=UINT32_MAX,
                block_tokens=4,
                bytes_per_block=64,
                pending_token=1,
                candidate_paths=((2,),),
            )
        with self.assertRaisesRegex(ValueError, "uint64"):
            project_prefix_closed_cow_kv(
                parent_tokens=1,
                block_tokens=1,
                bytes_per_block=UINT64_MAX,
                pending_token=1,
                candidate_paths=((2,),),
            )
        two_paths = dict(base)
        two_paths["candidate_paths"] = ((1,), (2,))
        with self.assertRaisesRegex(ValueError, "contains 2 paths"):
            project_prefix_closed_cow_kv(**two_paths, max_paths=1)
        prefixed_paths = dict(base)
        prefixed_paths["candidate_paths"] = ((1,), (1, 2))
        with self.assertRaisesRegex(ValueError, "prefix"):
            project_prefix_closed_cow_kv(**prefixed_paths)

    def test_cost_record_rejects_internally_inconsistent_accounting(self) -> None:
        with self.assertRaisesRegex(ValueError, "copied tails"):
            CowKVPhysicalCost(
                incremental_blocks=3,
                incremental_bytes=192,
                peak_live_blocks=4,
                peak_live_bytes=256,
                copied_tail_blocks=1,
                append_allocated_blocks=1,
                fork_count=1,
                leaf_lane_count=1,
            )


if __name__ == "__main__":
    unittest.main()
