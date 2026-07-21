from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import random
import unittest

from distributed_runtime.prefix_segment_schedule import (
    PrefixSegmentSchedule,
    plan_prefix_segment_schedule,
)
from distributed_runtime.prefix_token_arena import (
    PrefixTokenArena,
    PrefixTokenArenaError,
    PrefixWaveChunking,
    build_prefix_token_arena,
    validate_prefix_token_arena,
)
from distributed_runtime.prefix_wave_topology import (
    PrefixWaveTopology,
    build_prefix_wave_topology,
)


IDENTITY = {
    "nonce": 0x1020_3040_5060_7080,
    "parent_request_id": 90_000,
    "step": 37,
}


ABSTRACT_PATHS = (
    (0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 1),
    (0, 0, 1, 0, 0),
    (0, 1, 0, 0),
    (1, 0, 0, 0, 0, 0),
    (1, 1, 0, 0, 0),
    (2, 0, 0, 0),
    (2, 1, 0, 0),
)


def _schedule(
    paths=ABSTRACT_PATHS,
    *,
    shared=(700, 701, 702),
) -> PrefixSegmentSchedule:
    return plan_prefix_segment_schedule(
        paths,
        shared_prefix_tokens=shared,
        max_paths=64,
        max_depth=32,
        max_nodes=4_096,
    )


def _topology(
    schedule: PrefixSegmentSchedule,
    *,
    identity: dict[str, int] = IDENTITY,
    first_lane_request_id: int = 100_000,
) -> PrefixWaveTopology:
    lane_request_ids = tuple(
        range(first_lane_request_id, first_lane_request_id + len(schedule.leaves))
    )
    return build_prefix_wave_topology(
        schedule,
        lane_request_ids=lane_request_ids,
        **identity,
    )


def _arena(
    schedule: PrefixSegmentSchedule | None = None,
    *,
    topology: PrefixWaveTopology | None = None,
    identity: dict[str, int] = IDENTITY,
    chunking: PrefixWaveChunking = PrefixWaveChunking(),
) -> PrefixTokenArena:
    source = schedule or _schedule()
    sealed = topology or _topology(source, identity=identity)
    return build_prefix_token_arena(
        source,
        sealed,
        chunking=chunking,
        **identity,
    )


def _relabel_paths(
    paths: tuple[tuple[int, ...], ...], seed: int
) -> tuple[tuple[int, ...], ...]:
    generator = random.Random(seed)
    children: dict[tuple[int, ...], set[int]] = {}
    for path in paths:
        for depth, child in enumerate(path):
            children.setdefault(path[:depth], set()).add(child)
    labels: dict[tuple[tuple[int, ...], int], int] = {}
    for parent, raw_children in children.items():
        ordered = sorted(raw_children)
        replacements = generator.sample(range(1_000, 4_000_000), len(ordered))
        for child, replacement in zip(ordered, replacements):
            labels[(parent, child)] = replacement
    return tuple(
        tuple(labels[(path[:depth], child)] for depth, child in enumerate(path))
        for path in paths
    )


def _assert_exact_token_reconstruction(
    testcase: unittest.TestCase, arena: PrefixTokenArena
) -> None:
    reconstructed: dict[int, int] = {}
    for record in arena.records:
        token_ids = arena.token_ids_for_record(record.record_id)
        testcase.assertEqual(len(token_ids), len(record.canonical_node_ids))
        for node_id, token_id in zip(record.canonical_node_ids, token_ids):
            testcase.assertNotIn(node_id, reconstructed)
            reconstructed[node_id] = token_id
    testcase.assertEqual(
        tuple(reconstructed[node_id] for node_id in range(1, len(reconstructed) + 1)),
        arena.token_ids_by_canonical_node,
    )


def _assert_cut_through_state_alignment(
    testcase: unittest.TestCase,
    arena: PrefixTokenArena,
    schedule: PrefixSegmentSchedule,
    topology: PrefixWaveTopology,
) -> None:
    """Execute canonical records symbolically and join terminal lanes locally.

    The source schedule may choose different transient lane inheritance for
    token-relabelled isomorphic subtrees.  Every machine must therefore execute
    the canonical topology, not replay source-schedule lane ids.  This proof
    checks that record slicing preserves each canonical physical lane and that
    only the terminal bijection is needed to recover root-local candidates.
    """

    lane_node: dict[int, int] = {0: 0}
    opened_frontiers: set[int] = set()
    for record in arena.records:
        if record.frontier_index not in opened_frontiers:
            testcase.assertEqual(
                record.frontier_index,
                len(opened_frontiers),
            )
            for fork in topology.forks:
                if fork.frontier_index != record.frontier_index:
                    continue
                testcase.assertEqual(
                    lane_node[fork.source_lane_id],
                    fork.divergence_node_id,
                )
                testcase.assertNotIn(fork.target_lane_id, lane_node)
                lane_node[fork.target_lane_id] = fork.divergence_node_id
            opened_frontiers.add(record.frontier_index)

        for segment_id, node_ids in zip(
            record.segment_ids,
            record.canonical_nodes_by_segment(),
        ):
            segment = topology.segments[segment_id]
            testcase.assertEqual(segment.frontier_index, record.frontier_index)
            testcase.assertEqual(
                node_ids,
                segment.node_ids[
                    record.slice_offset : record.slice_offset
                    + record.slice_tokens
                ],
            )
            expected_parent = (
                segment.parent_node_id
                if record.slice_offset == 0
                else segment.node_ids[record.slice_offset - 1]
            )
            testcase.assertEqual(lane_node[segment.lane_id], expected_parent)
            cursor = expected_parent
            for node_id in node_ids:
                testcase.assertEqual(topology.nodes[node_id].parent_node_id, cursor)
                testcase.assertEqual(topology.nodes[node_id].lane_id, segment.lane_id)
                cursor = node_id
            lane_node[segment.lane_id] = cursor

    testcase.assertEqual(opened_frontiers, set(range(len(topology.frontiers))))
    for canonical_leaf_id, leaf in enumerate(topology.leaves):
        testcase.assertEqual(lane_node[leaf.lane_id], leaf.node_id)
        schedule_leaf_id = arena.schedule_leaf_ids_by_canonical_leaf_id[
            canonical_leaf_id
        ]
        schedule_leaf = schedule.leaves[schedule_leaf_id]
        testcase.assertEqual(
            arena.schedule_lane_ids_by_canonical_lane_id[leaf.lane_id],
            schedule_leaf.lane_id,
        )
        testcase.assertEqual(
            arena.physical_request_ids_by_canonical_lane_id[leaf.lane_id],
            topology.lane_request_ids[leaf.lane_id],
        )
        cursor = leaf.node_id
        candidate_reversed: list[int] = []
        while cursor:
            candidate_reversed.append(arena.token_ids_by_canonical_node[cursor - 1])
            parent = topology.nodes[cursor].parent_node_id
            testcase.assertIsNotNone(parent)
            cursor = int(parent)
        testcase.assertEqual(
            tuple(reversed(candidate_reversed)),
            (*schedule.shared_prefix_tokens, *schedule_leaf.candidate_path),
        )


class PrefixTokenArenaMappingTests(unittest.TestCase):
    def test_arena_maps_every_canonical_object_and_reconstructs_tokens_once(self) -> None:
        schedule = _schedule()
        topology = _topology(schedule)
        arena = _arena(schedule, topology=topology)

        self.assertEqual(arena.topology_digest, topology.digest)
        self.assertEqual(len(arena.topology_digest), 32)
        self.assertEqual(len(arena.wave_digest), 32)
        self.assertEqual(len(arena.local_arena_digest), 32)
        self.assertEqual(
            set(arena.schedule_node_ids_by_canonical_node_id),
            set(range(len(schedule.nodes))),
        )
        self.assertEqual(
            set(arena.schedule_leaf_ids_by_canonical_leaf_id),
            set(range(len(schedule.leaves))),
        )
        self.assertEqual(
            set(arena.schedule_lane_ids_by_canonical_lane_id),
            set(range(schedule.cost.lane_count)),
        )
        self.assertEqual(
            arena.physical_request_ids_by_canonical_lane_id,
            topology.lane_request_ids,
        )
        self.assertEqual(
            len(arena.token_ids_by_canonical_node), len(schedule.nodes) - 1
        )
        _assert_exact_token_reconstruction(self, arena)
        _assert_cut_through_state_alignment(self, arena, schedule, topology)
        validate_prefix_token_arena(arena, schedule, topology, **IDENTITY)

        with self.assertRaises(FrozenInstanceError):
            arena.step = 99  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            arena.records[0].group_id = 99  # type: ignore[misc]

    def test_records_slice_only_one_group_and_obey_all_chunk_bounds(self) -> None:
        schedule = _schedule()
        topology = _topology(schedule)
        chunking = PrefixWaveChunking(
            max_slice_tokens=3,
            max_segments_per_record=2,
            max_nodes_per_record=4,
        )
        arena = _arena(schedule, topology=topology, chunking=chunking)
        self.assertGreater(len(arena.records), len(topology.compatible_groups))

        for record in arena.records:
            self.assertLessEqual(record.slice_tokens, 3)
            self.assertLessEqual(record.segment_count, 2)
            self.assertLessEqual(record.node_count, 4)
            self.assertEqual(
                record.node_count, record.segment_count * record.slice_tokens
            )
            self.assertTrue(record.segment_ids)
            for segment_id, node_ids in zip(
                record.segment_ids, record.canonical_nodes_by_segment()
            ):
                segment = topology.segments[segment_id]
                self.assertEqual(segment.compatible_group_id, record.group_id)
                self.assertEqual(segment.frontier_index, record.frontier_index)
                self.assertEqual(
                    node_ids,
                    segment.node_ids[
                        record.slice_offset : record.slice_offset
                        + record.slice_tokens
                    ],
                )
        _assert_exact_token_reconstruction(self, arena)

    def test_random_prefix_trees_preserve_bijections_order_and_exact_coverage(self) -> None:
        generator = random.Random(0xA8E1A)
        for case in range(250):
            leaf_count = generator.randint(1, 10)
            depth = generator.randint(1, 8)
            if depth == 1 and leaf_count > 7:
                depth = 2
            paths: set[tuple[int, ...]] = set()
            while len(paths) < leaf_count:
                paths.add(tuple(generator.randrange(7) for _ in range(depth)))
            shuffled = list(paths)
            generator.shuffle(shuffled)
            shared = tuple(
                generator.randrange(1_000, 20_000)
                for _ in range(generator.randint(0, 4))
            )
            schedule = _schedule(tuple(shuffled), shared=shared)
            topology = _topology(
                schedule,
                identity={
                    "nonce": case,
                    "parent_request_id": 50_000 + case,
                    "step": case % 100,
                },
                first_lane_request_id=1_000_000 + case * 32,
            )
            identity = {
                "nonce": case,
                "parent_request_id": 50_000 + case,
                "step": case % 100,
            }
            chunking = PrefixWaveChunking(
                max_slice_tokens=generator.randint(1, 8),
                max_segments_per_record=generator.randint(1, 6),
                max_nodes_per_record=generator.randint(1, 24),
            )
            arena = _arena(
                schedule,
                topology=topology,
                identity=identity,
                chunking=chunking,
            )
            validate_prefix_token_arena(arena, schedule, topology, **identity)
            self.assertEqual(
                tuple(record.record_id for record in arena.records),
                tuple(range(len(arena.records))),
            )
            self.assertEqual(
                sorted(
                    node_id
                    for record in arena.records
                    for node_id in record.canonical_node_ids
                ),
                list(range(1, len(topology.nodes))),
            )
            _assert_exact_token_reconstruction(self, arena)
            _assert_cut_through_state_alignment(
                self,
                arena,
                schedule,
                topology,
            )


class PrefixTokenArenaDigestTests(unittest.TestCase):
    def test_descriptor_and_wave_digest_ignore_token_relabeling_and_input_order(self) -> None:
        baseline_descriptor = None
        baseline_wave_digest = None
        local_digests: set[bytes] = set()
        token_arenas: set[tuple[int, ...]] = set()
        for seed in range(50):
            paths = list(_relabel_paths(ABSTRACT_PATHS, seed))
            random.Random(seed ^ 0x51CE).shuffle(paths)
            shared = tuple(random.Random(seed).sample(range(5_000_000, 6_000_000), 3))
            schedule = _schedule(tuple(paths), shared=shared)
            topology = _topology(schedule)
            arena = _arena(schedule, topology=topology)
            if baseline_descriptor is None:
                baseline_descriptor = arena.structural_descriptor_bytes
                baseline_wave_digest = arena.wave_digest
            self.assertEqual(arena.structural_descriptor_bytes, baseline_descriptor)
            self.assertEqual(arena.wave_digest, baseline_wave_digest)
            local_digests.add(arena.local_arena_digest)
            token_arenas.add(arena.token_ids_by_canonical_node)
        self.assertGreater(len(local_digests), 45)
        self.assertGreater(len(token_arenas), 45)

    def test_chunking_changes_wave_digest_without_changing_topology_digest(self) -> None:
        schedule = _schedule()
        topology = _topology(schedule)
        variants = (
            PrefixWaveChunking(1, 1, 1),
            PrefixWaveChunking(2, 2, 4),
            PrefixWaveChunking(4, 8, 32),
            PrefixWaveChunking(8, 64, 4_096),
        )
        arenas = tuple(
            _arena(schedule, topology=topology, chunking=chunking)
            for chunking in variants
        )
        self.assertEqual({arena.topology_digest for arena in arenas}, {topology.digest})
        self.assertEqual(len({arena.wave_digest for arena in arenas}), len(variants))
        self.assertEqual(
            len({arena.structural_descriptor_bytes for arena in arenas}), len(variants)
        )

    def test_transaction_identity_changes_wave_digest_but_not_records(self) -> None:
        schedule = _schedule()
        identities = (
            IDENTITY,
            {**IDENTITY, "nonce": IDENTITY["nonce"] + 1},
            {**IDENTITY, "parent_request_id": IDENTITY["parent_request_id"] + 1},
            {**IDENTITY, "step": IDENTITY["step"] + 1},
        )
        arenas = []
        for identity in identities:
            topology = _topology(schedule, identity=identity)
            arenas.append(_arena(schedule, topology=topology, identity=identity))
        self.assertEqual(
            len({arena.structural_descriptor_bytes for arena in arenas}), 1
        )
        self.assertEqual(len({arena.records for arena in arenas}), 1)
        self.assertEqual(len({arena.wave_digest for arena in arenas}), len(identities))


class PrefixTokenArenaFailClosedTests(unittest.TestCase):
    def test_wrong_identity_stale_digest_and_structural_mismatch_are_rejected(self) -> None:
        schedule = _schedule()
        topology = _topology(schedule)
        with self.assertRaisesRegex(PrefixTokenArenaError, "identity"):
            build_prefix_token_arena(
                schedule,
                topology,
                **{**IDENTITY, "nonce": IDENTITY["nonce"] + 1},
            )
        with self.assertRaisesRegex(PrefixTokenArenaError, "sealed 32-byte"):
            build_prefix_token_arena(
                schedule,
                replace(topology, digest=b""),
                **IDENTITY,
            )
        with self.assertRaisesRegex(PrefixTokenArenaError, "stale"):
            build_prefix_token_arena(
                schedule,
                replace(topology, digest=b"x" * 32),
                **IDENTITY,
            )

        different = _schedule(
            (
                (10, 11, 12),
                (10, 11, 13),
                (20, 21, 22),
                (20, 21, 23),
                (30, 31, 32),
                (30, 31, 33),
                (40, 41, 42),
                (40, 41, 43),
            ),
            shared=(700, 701, 702),
        )
        with self.assertRaisesRegex(PrefixTokenArenaError, "same canonical"):
            build_prefix_token_arena(different, topology, **IDENTITY)

    def test_hostile_local_arena_mutations_all_fail_validation(self) -> None:
        schedule = _schedule()
        topology = _topology(schedule)
        arena = _arena(schedule, topology=topology)
        first = arena.records[0]
        hostile = (
            replace(arena, topology_digest=b"z" * 32),
            replace(arena, wave_digest=b"w" * 32),
            replace(arena, local_arena_digest=b"l" * 32),
            replace(
                arena,
                token_ids_by_canonical_node=(
                    999,
                    *arena.token_ids_by_canonical_node[1:],
                ),
            ),
            replace(
                arena,
                schedule_node_ids_by_canonical_node_id=tuple(
                    reversed(arena.schedule_node_ids_by_canonical_node_id)
                ),
            ),
            replace(arena, structural_descriptor_bytes=arena.structural_descriptor_bytes + b"x"),
            replace(
                arena,
                records=(replace(first, slice_offset=first.slice_offset + 1), *arena.records[1:]),
            ),
        )
        for mutation in hostile:
            with self.subTest(mutation=mutation):
                with self.assertRaisesRegex(PrefixTokenArenaError, "stale"):
                    validate_prefix_token_arena(
                        mutation, schedule, topology, **IDENTITY
                    )

    def test_hostile_record_cannot_read_virtual_root_or_out_of_bounds_token(self) -> None:
        arena = _arena()
        first = arena.records[0]
        for node_id in (0, len(arena.token_ids_by_canonical_node) + 1):
            bad_record = replace(first, canonical_node_ids=(node_id,))
            bad = replace(arena, records=(bad_record, *arena.records[1:]))
            with self.subTest(node_id=node_id):
                with self.assertRaisesRegex(PrefixTokenArenaError, "canonical_node_id"):
                    bad.token_ids_for_record(0)

    def test_constructor_types_and_chunk_bounds_fail_closed(self) -> None:
        schedule = _schedule()
        topology = _topology(schedule)
        with self.assertRaisesRegex(PrefixTokenArenaError, "schedule"):
            build_prefix_token_arena(object(), topology, **IDENTITY)  # type: ignore[arg-type]
        with self.assertRaisesRegex(PrefixTokenArenaError, "topology"):
            build_prefix_token_arena(schedule, object(), **IDENTITY)  # type: ignore[arg-type]
        with self.assertRaisesRegex(PrefixTokenArenaError, "chunking"):
            build_prefix_token_arena(
                schedule, topology, chunking=object(), **IDENTITY  # type: ignore[arg-type]
            )
        for kwargs in (
            {"max_slice_tokens": 0},
            {"max_segments_per_record": 0},
            {"max_nodes_per_record": 0},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(PrefixTokenArenaError):
                    PrefixWaveChunking(**kwargs)


if __name__ == "__main__":
    unittest.main()
