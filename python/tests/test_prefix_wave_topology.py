from __future__ import annotations

from dataclasses import replace
import hashlib
import random
import struct
import unittest

import distributed_runtime.prefix_wave_topology as pwt
from distributed_runtime.prefix_segment_schedule import plan_prefix_segment_schedule
from distributed_runtime.prefix_wave_topology import (
    PrefixWaveFork,
    PrefixWaveSegment,
    PrefixWaveTopologyError,
    PrefixWaveTopologyLimits,
    build_prefix_wave_topology,
    decode_prefix_wave_topology,
    encode_decoded_prefix_wave_topology,
    encode_prefix_wave_topology,
    prefix_wave_topology_mapping,
    validate_prefix_wave_topology,
)


IDENTITY = {
    "nonce": 0x0123456789ABCDEF,
    "parent_request_id": 10_000,
    "step": 73,
    "lane_request_ids": (20_000, 20_001, 20_002, 20_003),
}


def _schedule():
    return plan_prefix_segment_schedule(
        ((1, 2, 3, 4), (1, 2, 3, 5), (1, 2, 6), (8, 9)),
        shared_prefix_tokens=(90, 91),
    )


def _packet() -> bytes:
    return encode_prefix_wave_topology(_schedule(), **IDENTITY)


def _reseal(value: bytes | bytearray) -> bytes:
    mutable = bytearray(value)
    fields = list(pwt._HEADER.unpack_from(mutable))
    fields[-1] = b"\0" * 32
    unsigned = pwt._HEADER.pack(*fields)
    digest = hashlib.sha256(
        pwt._DIGEST_DOMAIN + unsigned + mutable[pwt.HEADER_BYTES :]
    ).digest()
    fields[-1] = digest
    mutable[: pwt.HEADER_BYTES] = pwt._HEADER.pack(*fields)
    return bytes(mutable)


def _table_offsets(packet: bytes) -> dict[str, int]:
    fields = pwt._HEADER.unpack_from(packet)
    lane_count = fields[8]
    node_count = fields[9]
    fork_count = fields[10]
    segment_count = fields[11]
    frontier_count = fields[12]
    group_count = fields[13]
    leaf_count = fields[14]
    offsets = {"lanes": pwt.HEADER_BYTES}
    offsets["nodes"] = offsets["lanes"] + lane_count * pwt.LANE_BYTES
    offsets["forks"] = offsets["nodes"] + node_count * pwt.NODE_BYTES
    offsets["segments"] = offsets["forks"] + fork_count * pwt.FORK_BYTES
    offsets["frontiers"] = (
        offsets["segments"] + segment_count * pwt.SEGMENT_BYTES
    )
    offsets["groups"] = (
        offsets["frontiers"] + frontier_count * pwt.FRONTIER_BYTES
    )
    offsets["leaves"] = offsets["groups"] + group_count * pwt.GROUP_BYTES
    offsets["node_refs"] = offsets["leaves"] + leaf_count * pwt.LEAF_BYTES
    return offsets


def _execute_symbolically(topology) -> dict[int, int]:
    lane_node = {0: 0}
    operations = sorted(
        (*topology.forks, *topology.segments), key=lambda item: item.sequence_index
    )
    if [item.sequence_index for item in operations] != list(range(len(operations))):
        raise AssertionError("operation sequence has a gap")
    compute_frontiers: set[int] = set()
    for operation in operations:
        if isinstance(operation, PrefixWaveFork):
            if operation.frontier_index in compute_frontiers:
                raise AssertionError("fork followed mutating compute in one frontier")
            if lane_node[operation.source_lane_id] != operation.divergence_node_id:
                raise AssertionError("fork source is not at the divergence node")
            if operation.target_lane_id in lane_node:
                raise AssertionError("fork target lane was reused")
            lane_node[operation.target_lane_id] = operation.divergence_node_id
            continue
        if not isinstance(operation, PrefixWaveSegment):
            raise AssertionError("unknown PrefixWave operation")
        compute_frontiers.add(operation.frontier_index)
        if lane_node[operation.lane_id] != operation.parent_node_id:
            raise AssertionError("segment lane does not hold its parent")
        cursor = operation.parent_node_id
        for node_id in operation.node_ids:
            if topology.nodes[node_id].parent_node_id != cursor:
                raise AssertionError("segment is not a contiguous tree chain")
            if topology.nodes[node_id].lane_id != operation.lane_id:
                raise AssertionError("node changed lanes inside a segment")
            cursor = node_id
        lane_node[operation.lane_id] = cursor
    return lane_node


def _relabel_paths(
    paths: tuple[tuple[int, ...], ...], seed: int
) -> tuple[tuple[int, ...], ...]:
    generator = random.Random(seed)
    children: dict[tuple[int, ...], set[int]] = {}
    for path in paths:
        for depth, child in enumerate(path):
            children.setdefault(path[:depth], set()).add(child)
    edge_labels: dict[tuple[tuple[int, ...], int], int] = {}
    for parent, raw_children in children.items():
        ordered = sorted(raw_children)
        labels = generator.sample(range(1_000, 50_000), len(ordered))
        for child, label in zip(ordered, labels):
            edge_labels[(parent, child)] = label
    return tuple(
        tuple(edge_labels[(path[:depth], child)] for depth, child in enumerate(path))
        for path in paths
    )


class PrefixWaveTopologyCanonicalTests(unittest.TestCase):
    def test_roundtrip_binds_complete_structure_and_is_symbolically_executable(self) -> None:
        packet = _packet()
        topology = decode_prefix_wave_topology(packet)
        self.assertEqual(encode_decoded_prefix_wave_topology(topology), packet)
        self.assertEqual(decode_prefix_wave_topology(bytearray(packet)), topology)
        self.assertEqual(
            decode_prefix_wave_topology(memoryview(packet).toreadonly()), topology
        )
        self.assertEqual(len(topology.digest), 32)
        self.assertEqual(
            topology.digest_hex,
            "da5ab2373b59afb5f0fc833452a16415d96f85f5dd8188771f4200791acd6f45",
        )
        self.assertEqual(
            hashlib.sha256(packet).hexdigest(),
            "e10857605562c622670fbca0fb1808a02e194879b39a380df67a11b01af44c53",
        )
        self.assertEqual(
            topology.lane_request_ids, tuple(sorted(topology.lane_request_ids))
        )

        lane_nodes = _execute_symbolically(topology)
        for leaf in topology.leaves:
            self.assertEqual(lane_nodes[leaf.lane_id], leaf.node_id)
            self.assertTrue(topology.nodes[leaf.node_id].terminal)
            self.assertEqual(topology.nodes[leaf.node_id].depth, leaf.depth)
        for frontier in topology.frontiers:
            operations = sorted(
                (
                    operation
                    for operation in (*topology.forks, *topology.segments)
                    if operation.frontier_index == frontier.frontier_index
                ),
                key=lambda operation: operation.sequence_index,
            )
            kinds = tuple(isinstance(operation, PrefixWaveFork) for operation in operations)
            self.assertEqual(kinds, tuple(sorted(kinds, reverse=True)))

    def test_arbitrary_token_relabeling_does_not_change_topology_bytes(self) -> None:
        abstract_paths = (
            (0, 0, 0),
            (0, 0, 1),
            (0, 1, 0, 0),
            (1, 0),
            (1, 1, 0),
            (2, 0, 0, 0, 0),
        )
        identity = {
            "nonce": 99,
            "parent_request_id": 800,
            "step": 17,
            "lane_request_ids": tuple(range(900, 906)),
        }
        expected = None
        observed_mappings = set()
        for seed in range(50):
            paths = _relabel_paths(abstract_paths, seed)
            shared = tuple(random.Random(seed ^ 0xBAD5EED).sample(range(50_001, 60_000), 3))
            schedule = plan_prefix_segment_schedule(
                paths, shared_prefix_tokens=shared, max_depth=16
            )
            packet = encode_prefix_wave_topology(schedule, **identity)
            if expected is None:
                expected = packet
            self.assertEqual(packet, expected)
            observed_mappings.add(
                prefix_wave_topology_mapping(schedule).schedule_leaf_ids_by_canonical_leaf_id
            )
        self.assertGreater(len(observed_mappings), 1)

    def test_external_identity_fields_each_change_the_authenticated_digest(self) -> None:
        baseline = decode_prefix_wave_topology(_packet())
        variants = (
            {**IDENTITY, "nonce": IDENTITY["nonce"] + 1},
            {**IDENTITY, "parent_request_id": IDENTITY["parent_request_id"] + 1},
            {**IDENTITY, "step": IDENTITY["step"] + 1},
            {
                **IDENTITY,
                "lane_request_ids": (*IDENTITY["lane_request_ids"][:-1], 20_004),
            },
        )
        for kwargs in variants:
            with self.subTest(kwargs=kwargs):
                changed = decode_prefix_wave_topology(
                    encode_prefix_wave_topology(_schedule(), **kwargs)
                )
                self.assertNotEqual(changed.digest, baseline.digest)

    def test_mapping_covers_source_nodes_and_leaves_without_entering_wire(self) -> None:
        schedule = _schedule()
        mapping = prefix_wave_topology_mapping(schedule)
        self.assertEqual(
            set(mapping.schedule_node_ids_by_canonical_node_id),
            set(range(len(schedule.nodes))),
        )
        self.assertEqual(
            set(mapping.schedule_leaf_ids_by_canonical_leaf_id),
            set(range(len(schedule.leaves))),
        )
        # Mapping metadata is deliberately absent from the authenticated object.
        topology = decode_prefix_wave_topology(_packet())
        self.assertFalse(hasattr(topology, "schedule_node_ids_by_canonical_node_id"))


class PrefixWaveTopologyInputValidationTests(unittest.TestCase):
    def test_lane_ids_are_counted_unique_ordered_and_disjoint_from_parent(self) -> None:
        invalid = (
            ((20_000, 20_001, 20_002), "requires 4"),
            ((20_000, 20_002, 20_001, 20_003), "strictly increasing"),
            ((20_000, 20_001, 20_001, 20_003), "strictly increasing"),
            ((10_000, 20_001, 20_002, 20_003), "alias"),
            ((20_000, True, 20_002, 20_003), "integer"),
        )
        for lanes, pattern in invalid:
            with self.subTest(lanes=lanes):
                with self.assertRaisesRegex(PrefixWaveTopologyError, pattern):
                    encode_prefix_wave_topology(
                        _schedule(), **{**IDENTITY, "lane_request_ids": lanes}
                    )

    def test_source_schedule_must_be_exact_canonical_planner_output(self) -> None:
        schedule = _schedule()
        bad_fork = replace(schedule.forks[0], sequence_index=999)
        malformed = replace(schedule, forks=(bad_fork, *schedule.forks[1:]))
        with self.assertRaisesRegex(PrefixWaveTopologyError, "canonical planner"):
            encode_prefix_wave_topology(malformed, **IDENTITY)

    def test_custom_limits_apply_to_encode_decode_and_object_validation(self) -> None:
        packet = _packet()
        small = PrefixWaveTopologyLimits(max_lanes=3, max_leaves=3, max_forks=2)
        with self.assertRaisesRegex(PrefixWaveTopologyError, "lane_count"):
            decode_prefix_wave_topology(packet, limits=small)
        with self.assertRaisesRegex(PrefixWaveTopologyError, "lane_count"):
            encode_prefix_wave_topology(_schedule(), **IDENTITY, limits=small)
        with self.assertRaisesRegex(PrefixWaveTopologyError, "lane_count"):
            validate_prefix_wave_topology(
                decode_prefix_wave_topology(packet), limits=small
            )

    def test_stale_digest_cannot_be_reused_after_identity_mutation(self) -> None:
        topology = decode_prefix_wave_topology(_packet())
        changed = replace(topology, nonce=topology.nonce + 1)
        with self.assertRaisesRegex(PrefixWaveTopologyError, "digest is stale"):
            validate_prefix_wave_topology(changed)
        with self.assertRaisesRegex(PrefixWaveTopologyError, "digest is stale"):
            encode_decoded_prefix_wave_topology(changed)


class PrefixWaveTopologyHostileDecoderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.packet = _packet()
        self.offsets = _table_offsets(self.packet)

    def test_header_digest_length_and_bounded_count_fail_closed(self) -> None:
        mutations = []

        bad_magic = bytearray(self.packet)
        bad_magic[0:4] = b"EVIL"
        mutations.append((bad_magic, "magic"))

        bad_version = bytearray(self.packet)
        bad_version[4] += 1
        mutations.append((bad_version, "version"))

        bad_reserved = bytearray(self.packet)
        bad_reserved[5] = 1
        mutations.append((bad_reserved, "reserved"))

        bad_digest = bytearray(self.packet)
        bad_digest[-1] ^= 1
        mutations.append((bad_digest, "SHA-256"))

        for packet, pattern in mutations:
            with self.subTest(pattern=pattern):
                with self.assertRaisesRegex(PrefixWaveTopologyError, pattern):
                    decode_prefix_wave_topology(packet)

        with self.assertRaisesRegex(PrefixWaveTopologyError, "truncated"):
            decode_prefix_wave_topology(self.packet[:-1])
        with self.assertRaisesRegex(PrefixWaveTopologyError, "trailing"):
            decode_prefix_wave_topology(self.packet + b"x")

        count_bomb = bytearray(self.packet)
        struct.pack_into("<I", count_bomb, 24 + 2 * 4, pwt.MAX_LANES)
        with self.assertRaisesRegex(PrefixWaveTopologyError, "lane_count"):
            decode_prefix_wave_topology(count_bomb)

    def test_semantic_tables_are_rejected_even_with_a_valid_attacker_digest(self) -> None:
        cases: list[tuple[bytearray, str]] = []

        bad_lane_order = bytearray(self.packet)
        first = struct.unpack_from("<Q", bad_lane_order, self.offsets["lanes"])[0]
        struct.pack_into("<Q", bad_lane_order, self.offsets["lanes"] + 8, first)
        cases.append((bad_lane_order, "strictly increasing"))

        bad_depth = bytearray(self.packet)
        struct.pack_into(
            "<I", bad_depth, self.offsets["nodes"] + pwt.NODE_BYTES + 8, 99
        )
        cases.append((bad_depth, "node topology"))

        bad_parent = bytearray(self.packet)
        struct.pack_into(
            "<I", bad_parent, self.offsets["nodes"] + pwt.NODE_BYTES, 1
        )
        cases.append((bad_parent, "invalid parent"))

        bad_node_lane = bytearray(self.packet)
        struct.pack_into(
            "<I", bad_node_lane, self.offsets["nodes"] + pwt.NODE_BYTES + 4, 99
        )
        cases.append((bad_node_lane, "lane order"))

        bad_terminal = bytearray(self.packet)
        bad_terminal[self.offsets["nodes"] + pwt.NODE_BYTES + 12] ^= 1
        cases.append((bad_terminal, "terminal"))

        bad_fork_order = bytearray(self.packet)
        struct.pack_into("<I", bad_fork_order, self.offsets["forks"], 0xFFFF)
        cases.append((bad_fork_order, "fork plan"))

        bad_segment = bytearray(self.packet)
        # parent node id is the sixth uint32 field after sequence/frontier/group/lane.
        struct.pack_into(
            "<I", bad_segment, self.offsets["segments"] + 4 * 4, 0xFFFF
        )
        cases.append((bad_segment, "compute segments"))

        bad_frontier = bytearray(self.packet)
        struct.pack_into("<I", bad_frontier, self.offsets["frontiers"] + 4, 0)
        cases.append((bad_frontier, "frontier"))

        bad_group = bytearray(self.packet)
        struct.pack_into("<I", bad_group, self.offsets["groups"] + 4, 99)
        cases.append((bad_group, "groups"))

        bad_leaf = bytearray(self.packet)
        struct.pack_into("<I", bad_leaf, self.offsets["leaves"] + 8, 99)
        cases.append((bad_leaf, "leaf"))

        for packet, pattern in cases:
            with self.subTest(pattern=pattern):
                with self.assertRaisesRegex(PrefixWaveTopologyError, pattern):
                    decode_prefix_wave_topology(_reseal(packet))

    def test_segment_reference_overflow_is_rejected_before_slicing(self) -> None:
        malformed = bytearray(self.packet)
        # Segment field six is ref_offset; keep a valid digest to reach bounds.
        struct.pack_into(
            "<I", malformed, self.offsets["segments"] + 5 * 4, 0xFFFFFFFF
        )
        with self.assertRaisesRegex(PrefixWaveTopologyError, "overflows"):
            decode_prefix_wave_topology(_reseal(malformed))

    def test_object_level_validator_rejects_missing_fork_before_compute(self) -> None:
        topology = decode_prefix_wave_topology(self.packet)
        malformed = replace(topology, forks=topology.forks[1:], digest=b"")
        with self.assertRaisesRegex(PrefixWaveTopologyError, "fork_count"):
            validate_prefix_wave_topology(malformed)


if __name__ == "__main__":
    unittest.main()
