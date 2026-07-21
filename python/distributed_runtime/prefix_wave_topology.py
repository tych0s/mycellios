"""Bounded canonical wire descriptor for a PrefixWave execution topology.

This module deliberately carries *structure only*.  Token ids, activations,
logits and KV data belong to separate, explicitly-bound payloads.  Keeping the
topology small lets every stage authenticate and validate the complete fork and
compute order before it mutates a cache.

The canonical form is independent of token values.  It derives an unordered
rooted-tree signature from :class:`PrefixSegmentSchedule`, orders isomorphic
subtrees by that signature, and regenerates lanes, forks, segments, frontiers
and compatible groups from the resulting tree.  Therefore token-renamed
isomorphic plans have identical bytes when their external transaction identity
is identical.

The decoder is intentionally fail-closed: fixed-width counts are checked before
allocation, the SHA-256 integrity-binds the complete header and body, and the
decoded descriptors must equal a freshly regenerated canonical schedule before
callers may use them for compute.  SHA-256 here is not peer authentication: an
active attacker can recompute an unkeyed digest, so the transport still needs
an authenticated channel or MAC before this descriptor crosses a trust
boundary.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from itertools import groupby
import hashlib
from numbers import Integral
import struct

from .prefix_segment_schedule import (
    PrefixSegmentSchedule,
    plan_prefix_segment_schedule,
)


MAGIC = b"PWTP"
VERSION = 1

UINT32_MAX = (1 << 32) - 1
UINT64_MAX = (1 << 64) - 1
PARENT_NONE = UINT32_MAX

MAX_LANES = 4_096
MAX_NODES = 65_536
MAX_FORKS = MAX_LANES - 1
MAX_SEGMENTS = MAX_NODES - 1
MAX_FRONTIERS = 257
MAX_GROUPS = MAX_SEGMENTS
MAX_LEAVES = MAX_LANES
MAX_NODE_REFS = MAX_NODES - 1
MAX_DEPTH = 256
MAX_PACKET_BYTES = 16 * 1024 * 1024

# magic, version, flags, header bytes, nonce, parent request id, followed by:
# step, shared-prefix depth, lane/node/fork/segment/frontier/group/leaf counts,
# segment-node-reference count, body bytes, reserved, and SHA-256.
_HEADER = struct.Struct("<4sBBHQQ12I32s")
_LANE = struct.Struct("<Q")
_NODE = struct.Struct("<IIIB3x")
_FORK = struct.Struct("<IIIIII")
_SEGMENT = struct.Struct("<IIIIIIIII")
_FRONTIER = struct.Struct("<IIIIIIII")
_GROUP = struct.Struct("<IIIIIII")
_LEAF = struct.Struct("<III")
_NODE_REF = struct.Struct("<I")

HEADER_BYTES = _HEADER.size
LANE_BYTES = _LANE.size
NODE_BYTES = _NODE.size
FORK_BYTES = _FORK.size
SEGMENT_BYTES = _SEGMENT.size
FRONTIER_BYTES = _FRONTIER.size
GROUP_BYTES = _GROUP.size
LEAF_BYTES = _LEAF.size
NODE_REF_BYTES = _NODE_REF.size

_DIGEST_DOMAIN = b"GDLP/PREFIX-WAVE-TOPOLOGY/canonical-v1\0"
_ZERO_DIGEST = b"\0" * 32
_TERMINAL_FLAG = 1


class PrefixWaveTopologyError(ValueError):
    """A topology or packet violates the sealed PrefixWave contract."""


def _bounded(name: str, value: object, minimum: int, maximum: int) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise PrefixWaveTopologyError(f"{name} must be an integer")
    normalized = int(value)
    if normalized < minimum or normalized > maximum:
        raise PrefixWaveTopologyError(f"{name} must be in [{minimum}, {maximum}]")
    return normalized


@dataclass(frozen=True, slots=True)
class PrefixWaveTopologyLimits:
    max_lanes: int = 64
    max_nodes: int = 4_096
    max_forks: int = 63
    max_segments: int = 4_095
    max_frontiers: int = MAX_FRONTIERS
    max_groups: int = 4_095
    max_leaves: int = 64
    max_node_refs: int = 4_095
    max_depth: int = MAX_DEPTH
    max_packet_bytes: int = 2 * 1024 * 1024

    def __post_init__(self) -> None:
        _bounded("max_lanes", self.max_lanes, 1, MAX_LANES)
        _bounded("max_nodes", self.max_nodes, 2, MAX_NODES)
        _bounded("max_forks", self.max_forks, 0, MAX_FORKS)
        _bounded("max_segments", self.max_segments, 1, MAX_SEGMENTS)
        _bounded("max_frontiers", self.max_frontiers, 1, MAX_FRONTIERS)
        _bounded("max_groups", self.max_groups, 1, MAX_GROUPS)
        _bounded("max_leaves", self.max_leaves, 1, MAX_LEAVES)
        _bounded("max_node_refs", self.max_node_refs, 1, MAX_NODE_REFS)
        _bounded("max_depth", self.max_depth, 1, MAX_DEPTH)
        _bounded(
            "max_packet_bytes",
            self.max_packet_bytes,
            HEADER_BYTES + NODE_BYTES,
            MAX_PACKET_BYTES,
        )


DEFAULT_LIMITS = PrefixWaveTopologyLimits()


@dataclass(frozen=True, slots=True)
class PrefixWaveNode:
    node_id: int
    parent_node_id: int | None
    lane_id: int
    depth: int
    terminal: bool


@dataclass(frozen=True, slots=True)
class PrefixWaveFork:
    fork_id: int
    sequence_index: int
    frontier_index: int
    divergence_node_id: int
    source_lane_id: int
    target_lane_id: int
    target_child_node_id: int


@dataclass(frozen=True, slots=True)
class PrefixWaveSegment:
    segment_id: int
    sequence_index: int
    frontier_index: int
    compatible_group_id: int
    lane_id: int
    parent_node_id: int
    node_ids: tuple[int, ...]
    start_depth: int
    end_depth: int

    @property
    def token_count(self) -> int:
        return len(self.node_ids)


@dataclass(frozen=True, slots=True)
class PrefixWaveFrontier:
    frontier_index: int
    first_operation_index: int
    operation_count: int
    first_fork_id: int
    fork_count: int
    first_segment_id: int
    segment_count: int
    first_group_id: int
    group_count: int


@dataclass(frozen=True, slots=True)
class PrefixWaveCompatibleGroup:
    group_id: int
    frontier_index: int
    start_depth: int
    token_count: int
    first_segment_id: int
    segment_count: int
    first_operation_index: int
    last_operation_index: int


@dataclass(frozen=True, slots=True)
class PrefixWaveLeaf:
    leaf_id: int
    node_id: int
    lane_id: int
    depth: int


@dataclass(frozen=True, slots=True)
class PrefixWaveTopology:
    nonce: int
    parent_request_id: int
    step: int
    shared_prefix_depth: int
    lane_request_ids: tuple[int, ...]
    nodes: tuple[PrefixWaveNode, ...]
    forks: tuple[PrefixWaveFork, ...]
    segments: tuple[PrefixWaveSegment, ...]
    frontiers: tuple[PrefixWaveFrontier, ...]
    compatible_groups: tuple[PrefixWaveCompatibleGroup, ...]
    leaves: tuple[PrefixWaveLeaf, ...]
    digest: bytes = field(default=b"", repr=False)

    @property
    def digest_hex(self) -> str:
        return self.digest.hex()

    @property
    def operation_count(self) -> int:
        return len(self.forks) + len(self.segments)


@dataclass(frozen=True, slots=True)
class PrefixWaveTopologyMapping:
    """Token-payload mapping; it is local metadata and never enters the wire."""

    schedule_node_ids_by_canonical_node_id: tuple[int, ...]
    schedule_leaf_ids_by_canonical_leaf_id: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _CanonicalStructure:
    nodes: tuple[PrefixWaveNode, ...]
    forks: tuple[PrefixWaveFork, ...]
    segments: tuple[PrefixWaveSegment, ...]
    frontiers: tuple[PrefixWaveFrontier, ...]
    groups: tuple[PrefixWaveCompatibleGroup, ...]
    leaves: tuple[PrefixWaveLeaf, ...]
    original_node_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _SegmentSpec:
    parent_node_id: int
    lane_id: int
    node_ids: tuple[int, ...]
    start_depth: int

    @property
    def token_count(self) -> int:
        return len(self.node_ids)


def build_prefix_wave_topology(
    schedule: PrefixSegmentSchedule,
    *,
    nonce: int,
    parent_request_id: int,
    step: int,
    lane_request_ids: Sequence[int],
    limits: PrefixWaveTopologyLimits = DEFAULT_LIMITS,
) -> PrefixWaveTopology:
    """Build and integrity-bind a token-independent canonical topology."""

    _require_limits(limits)
    _validate_source_schedule(schedule)
    lane_ids = _lane_request_ids(lane_request_ids)
    parents = tuple(node.parent_node_id for node in schedule.nodes)
    terminals = frozenset(node.node_id for node in schedule.nodes if node.terminal)
    structure = _canonical_structure(
        parents,
        terminals,
        shared_prefix_depth=len(schedule.shared_prefix_tokens),
        maximum_depth=limits.max_depth,
    )
    if len(lane_ids) != len(structure.leaves):
        raise PrefixWaveTopologyError(
            f"lane_request_ids has {len(lane_ids)} entries; "
            f"canonical topology requires {len(structure.leaves)}"
        )
    parent = _bounded("parent_request_id", parent_request_id, 0, UINT64_MAX)
    if parent in lane_ids:
        raise PrefixWaveTopologyError(
            "lane request ids must not alias parent_request_id"
        )
    topology = PrefixWaveTopology(
        nonce=_bounded("nonce", nonce, 0, UINT64_MAX),
        parent_request_id=parent,
        step=_bounded("step", step, 0, UINT32_MAX),
        shared_prefix_depth=len(schedule.shared_prefix_tokens),
        lane_request_ids=lane_ids,
        nodes=structure.nodes,
        forks=structure.forks,
        segments=structure.segments,
        frontiers=structure.frontiers,
        compatible_groups=structure.groups,
        leaves=structure.leaves,
    )
    packet, digest = _serialize(topology, limits=limits)
    del packet
    return replace(topology, digest=digest)


def prefix_wave_topology_mapping(
    schedule: PrefixSegmentSchedule,
    *,
    max_depth: int = MAX_DEPTH,
) -> PrefixWaveTopologyMapping:
    """Return how a separate token arena maps onto canonical node ids."""

    _validate_source_schedule(schedule)
    depth_limit = _bounded("max_depth", max_depth, 1, MAX_DEPTH)
    structure = _canonical_structure(
        tuple(node.parent_node_id for node in schedule.nodes),
        frozenset(node.node_id for node in schedule.nodes if node.terminal),
        shared_prefix_depth=len(schedule.shared_prefix_tokens),
        maximum_depth=depth_limit,
    )
    leaf_id_by_node_id = {leaf.node_id: leaf.leaf_id for leaf in schedule.leaves}
    schedule_leaf_ids = tuple(
        leaf_id_by_node_id[structure.original_node_ids[leaf.node_id]]
        for leaf in structure.leaves
    )
    return PrefixWaveTopologyMapping(
        schedule_node_ids_by_canonical_node_id=structure.original_node_ids,
        schedule_leaf_ids_by_canonical_leaf_id=schedule_leaf_ids,
    )


def encode_prefix_wave_topology(
    schedule: PrefixSegmentSchedule,
    *,
    nonce: int,
    parent_request_id: int,
    step: int,
    lane_request_ids: Sequence[int],
    limits: PrefixWaveTopologyLimits = DEFAULT_LIMITS,
) -> bytes:
    topology = build_prefix_wave_topology(
        schedule,
        nonce=nonce,
        parent_request_id=parent_request_id,
        step=step,
        lane_request_ids=lane_request_ids,
        limits=limits,
    )
    packet, digest = _serialize(topology, limits=limits)
    if digest != topology.digest:
        raise RuntimeError("PrefixWave topology digest changed during encoding")
    return packet


def encode_decoded_prefix_wave_topology(
    topology: PrefixWaveTopology,
    *,
    limits: PrefixWaveTopologyLimits = DEFAULT_LIMITS,
) -> bytes:
    """Re-encode a decoded object, rejecting stale identity/digest pairs."""

    validate_prefix_wave_topology(topology, limits=limits)
    packet, digest = _serialize(topology, limits=limits)
    if topology.digest and topology.digest != digest:
        raise PrefixWaveTopologyError(
            "topology digest is stale for its transaction identity or descriptors"
        )
    return packet


def decode_prefix_wave_topology(
    packet: bytes | bytearray | memoryview,
    *,
    limits: PrefixWaveTopologyLimits = DEFAULT_LIMITS,
) -> PrefixWaveTopology:
    """Decode, integrity-check and fully validate a canonical topology packet."""

    _require_limits(limits)
    raw = _immutable_bytes(packet)
    if len(raw) < HEADER_BYTES:
        raise PrefixWaveTopologyError("truncated PrefixWave topology header")
    if len(raw) > limits.max_packet_bytes:
        raise PrefixWaveTopologyError("PrefixWave topology packet exceeds sealed limit")

    unpacked = _HEADER.unpack_from(raw)
    (
        magic,
        version,
        flags,
        header_bytes,
        nonce,
        parent_request_id,
        step,
        shared_prefix_depth,
        lane_count,
        node_count,
        fork_count,
        segment_count,
        frontier_count,
        group_count,
        leaf_count,
        node_ref_count,
        body_bytes,
        reserved,
        digest,
    ) = unpacked
    if magic != MAGIC:
        raise PrefixWaveTopologyError("invalid PrefixWave topology magic")
    if version != VERSION:
        raise PrefixWaveTopologyError("unsupported PrefixWave topology version")
    if flags != 0 or reserved != 0:
        raise PrefixWaveTopologyError("non-zero reserved PrefixWave topology field")
    if header_bytes != HEADER_BYTES:
        raise PrefixWaveTopologyError("non-canonical PrefixWave header length")

    counts = _validated_counts(
        lane_count=lane_count,
        node_count=node_count,
        fork_count=fork_count,
        segment_count=segment_count,
        frontier_count=frontier_count,
        group_count=group_count,
        leaf_count=leaf_count,
        node_ref_count=node_ref_count,
        limits=limits,
    )
    expected_body = _body_size(*counts)
    if body_bytes != expected_body:
        raise PrefixWaveTopologyError("declared topology body length is not canonical")
    expected_packet = _checked_add(HEADER_BYTES, expected_body, limits.max_packet_bytes)
    if len(raw) != expected_packet:
        qualifier = "truncated" if len(raw) < expected_packet else "trailing"
        raise PrefixWaveTopologyError(f"{qualifier} PrefixWave topology bytes")

    unsigned_header = _HEADER.pack(*unpacked[:-1], _ZERO_DIGEST)
    expected_digest = hashlib.sha256(
        _DIGEST_DOMAIN + unsigned_header + raw[HEADER_BYTES:]
    ).digest()
    if digest != expected_digest:
        raise PrefixWaveTopologyError("PrefixWave topology SHA-256 mismatch")

    cursor = HEADER_BYTES
    lanes: list[int] = []
    for _ in range(lane_count):
        lanes.append(_LANE.unpack_from(raw, cursor)[0])
        cursor += LANE_BYTES

    nodes: list[PrefixWaveNode] = []
    for node_id in range(node_count):
        parent, lane_id, depth, node_flags = _NODE.unpack_from(raw, cursor)
        cursor += NODE_BYTES
        if node_flags & ~_TERMINAL_FLAG:
            raise PrefixWaveTopologyError("unknown PrefixWave node flags")
        nodes.append(
            PrefixWaveNode(
                node_id=node_id,
                parent_node_id=None if parent == PARENT_NONE else parent,
                lane_id=lane_id,
                depth=depth,
                terminal=bool(node_flags & _TERMINAL_FLAG),
            )
        )

    forks: list[PrefixWaveFork] = []
    for fork_id in range(fork_count):
        values = _FORK.unpack_from(raw, cursor)
        cursor += FORK_BYTES
        forks.append(PrefixWaveFork(fork_id, *values))

    raw_segments: list[tuple[int, ...]] = []
    for _ in range(segment_count):
        raw_segments.append(_SEGMENT.unpack_from(raw, cursor))
        cursor += SEGMENT_BYTES

    frontiers: list[PrefixWaveFrontier] = []
    for frontier_index in range(frontier_count):
        frontiers.append(
            PrefixWaveFrontier(frontier_index, *_FRONTIER.unpack_from(raw, cursor))
        )
        cursor += FRONTIER_BYTES

    groups: list[PrefixWaveCompatibleGroup] = []
    for group_id in range(group_count):
        groups.append(
            PrefixWaveCompatibleGroup(group_id, *_GROUP.unpack_from(raw, cursor))
        )
        cursor += GROUP_BYTES

    leaves: list[PrefixWaveLeaf] = []
    for leaf_id in range(leaf_count):
        leaves.append(PrefixWaveLeaf(leaf_id, *_LEAF.unpack_from(raw, cursor)))
        cursor += LEAF_BYTES

    node_refs: list[int] = []
    for _ in range(node_ref_count):
        node_refs.append(_NODE_REF.unpack_from(raw, cursor)[0])
        cursor += NODE_REF_BYTES
    if cursor != len(raw):
        raise RuntimeError("PrefixWave decoder cursor did not consume sealed body")

    segments: list[PrefixWaveSegment] = []
    for segment_id, values in enumerate(raw_segments):
        (
            sequence_index,
            frontier_index,
            group_id,
            lane_id,
            parent_node_id,
            ref_offset,
            ref_count,
            start_depth,
            end_depth,
        ) = values
        ref_end = _checked_add(ref_offset, ref_count, node_ref_count)
        segments.append(
            PrefixWaveSegment(
                segment_id=segment_id,
                sequence_index=sequence_index,
                frontier_index=frontier_index,
                compatible_group_id=group_id,
                lane_id=lane_id,
                parent_node_id=parent_node_id,
                node_ids=tuple(node_refs[ref_offset:ref_end]),
                start_depth=start_depth,
                end_depth=end_depth,
            )
        )

    topology = PrefixWaveTopology(
        nonce=nonce,
        parent_request_id=parent_request_id,
        step=step,
        shared_prefix_depth=shared_prefix_depth,
        lane_request_ids=tuple(lanes),
        nodes=tuple(nodes),
        forks=tuple(forks),
        segments=tuple(segments),
        frontiers=tuple(frontiers),
        compatible_groups=tuple(groups),
        leaves=tuple(leaves),
        digest=digest,
    )
    validate_prefix_wave_topology(topology, limits=limits)
    canonical, canonical_digest = _serialize(topology, limits=limits)
    if canonical_digest != digest or canonical != raw:
        raise PrefixWaveTopologyError("PrefixWave topology packet is not canonical")
    return topology


def validate_prefix_wave_topology(
    topology: PrefixWaveTopology,
    *,
    limits: PrefixWaveTopologyLimits = DEFAULT_LIMITS,
) -> None:
    """Prove tree, fork and compute order before a stage starts execution."""

    _require_limits(limits)
    if not isinstance(topology, PrefixWaveTopology):
        raise PrefixWaveTopologyError("topology must be a PrefixWaveTopology")
    _bounded("nonce", topology.nonce, 0, UINT64_MAX)
    parent = _bounded("parent_request_id", topology.parent_request_id, 0, UINT64_MAX)
    _bounded("step", topology.step, 0, UINT32_MAX)
    shared_depth = _bounded(
        "shared_prefix_depth", topology.shared_prefix_depth, 0, limits.max_depth
    )
    lanes = _lane_request_ids(topology.lane_request_ids)
    if parent in lanes:
        raise PrefixWaveTopologyError(
            "lane request ids must not alias parent_request_id"
        )
    counts = _validated_counts(
        lane_count=len(lanes),
        node_count=len(topology.nodes),
        fork_count=len(topology.forks),
        segment_count=len(topology.segments),
        frontier_count=len(topology.frontiers),
        group_count=len(topology.compatible_groups),
        leaf_count=len(topology.leaves),
        node_ref_count=sum(len(segment.node_ids) for segment in topology.segments),
        limits=limits,
    )
    if len(lanes) != len(topology.leaves):
        raise PrefixWaveTopologyError("one canonical request id is required per leaf lane")
    if topology.digest and len(topology.digest) != 32:
        raise PrefixWaveTopologyError("topology digest must contain exactly 32 bytes")

    for node_id, node in enumerate(topology.nodes):
        if not isinstance(node, PrefixWaveNode) or node.node_id != node_id:
            raise PrefixWaveTopologyError("node ids must be canonical and contiguous")
    terminals = frozenset(node.node_id for node in topology.nodes if node.terminal)
    expected = _canonical_structure(
        tuple(node.parent_node_id for node in topology.nodes),
        terminals,
        shared_prefix_depth=shared_depth,
        maximum_depth=limits.max_depth,
    )
    if topology.nodes != expected.nodes:
        raise PrefixWaveTopologyError("node topology or lane order is not canonical")
    if topology.forks != expected.forks:
        raise PrefixWaveTopologyError(
            "fork plan is incomplete, out of order, or follows mutating compute"
        )
    if topology.segments != expected.segments:
        raise PrefixWaveTopologyError("compute segments are not canonical contiguous chains")
    if topology.frontiers != expected.frontiers:
        raise PrefixWaveTopologyError("frontier ranges or operation order are not canonical")
    if topology.compatible_groups != expected.groups:
        raise PrefixWaveTopologyError("compatible frontier groups are not canonical")
    if topology.leaves != expected.leaves:
        raise PrefixWaveTopologyError("leaf nodes, lanes, or depths are not canonical")
    _body_size(*counts)
    # Object-level validation is also an integrity boundary.  A frozen
    # dataclass can still be copied with ``replace()``, so accepting any
    # correctly-sized digest here would let a stale transaction identity pass
    # callers which validate without immediately re-encoding.  Recompute from
    # the already-proved canonical tables without recursively validating.
    if topology.digest:
        _, expected_digest = _serialize(
            topology,
            limits=limits,
            _already_validated=True,
        )
        if topology.digest != expected_digest:
            raise PrefixWaveTopologyError(
                "topology digest is stale for its transaction identity or descriptors"
            )


def _validate_source_schedule(schedule: object) -> PrefixSegmentSchedule:
    if not isinstance(schedule, PrefixSegmentSchedule):
        raise PrefixWaveTopologyError("schedule must be a PrefixSegmentSchedule")
    try:
        expected = plan_prefix_segment_schedule(
            schedule.candidate_paths,
            shared_prefix_tokens=schedule.shared_prefix_tokens,
            max_paths=MAX_LEAVES,
            max_depth=MAX_DEPTH,
            max_nodes=MAX_NODES,
        )
    except (TypeError, ValueError, RuntimeError) as error:
        raise PrefixWaveTopologyError(f"invalid source schedule: {error}") from error
    if schedule != expected:
        raise PrefixWaveTopologyError(
            "source PrefixSegmentSchedule is not the canonical planner output"
        )
    return schedule


def _canonical_structure(
    parents: Sequence[int | None],
    terminals: frozenset[int],
    *,
    shared_prefix_depth: int,
    maximum_depth: int,
) -> _CanonicalStructure:
    node_count = len(parents)
    if node_count < 2:
        raise PrefixWaveTopologyError("PrefixWave tree requires a root and one token node")
    if parents[0] is not None:
        raise PrefixWaveTopologyError("node zero must be the virtual root")
    children: list[list[int]] = [[] for _ in range(node_count)]
    for node_id in range(1, node_count):
        parent = parents[node_id]
        if (
            not isinstance(parent, Integral)
            or isinstance(parent, bool)
            or int(parent) < 0
            or int(parent) >= node_count
            or int(parent) == node_id
        ):
            raise PrefixWaveTopologyError(f"node {node_id} has an invalid parent")
        children[int(parent)].append(node_id)

    depth: list[int | None] = [None] * node_count
    depth[0] = 0
    queue: deque[int] = deque((0,))
    by_depth: dict[int, list[int]] = defaultdict(list)
    by_depth[0].append(0)
    while queue:
        node_id = queue.popleft()
        assert depth[node_id] is not None
        for child in children[node_id]:
            if depth[child] is not None:
                raise PrefixWaveTopologyError("PrefixWave tree contains a cycle")
            child_depth = depth[node_id] + 1
            if child_depth > maximum_depth:
                raise PrefixWaveTopologyError("PrefixWave tree exceeds maximum depth")
            depth[child] = child_depth
            by_depth[child_depth].append(child)
            queue.append(child)
    if any(item is None for item in depth):
        raise PrefixWaveTopologyError("PrefixWave tree is disconnected or cyclic")

    if not terminals or 0 in terminals:
        raise PrefixWaveTopologyError("terminal set must contain non-root leaves")
    if any(node < 0 or node >= node_count for node in terminals):
        raise PrefixWaveTopologyError("terminal node id is outside the tree")
    for node_id, node_children in enumerate(children):
        terminal = node_id in terminals
        if terminal and node_children:
            raise PrefixWaveTopologyError("terminal nodes cannot have children")
        if not terminal and not node_children:
            raise PrefixWaveTopologyError("every physical leaf must be terminal")

    shared = _bounded("shared_prefix_depth", shared_prefix_depth, 0, maximum_depth)
    cursor = 0
    for _ in range(shared):
        if len(children[cursor]) != 1:
            raise PrefixWaveTopologyError(
                "shared prefix must remain unary until its declared depth"
            )
        cursor = children[cursor][0]
    if any(int(depth[leaf]) <= shared for leaf in terminals):
        raise PrefixWaveTopologyError(
            "every candidate leaf must extend beyond the shared prefix"
        )

    # Assign comparable structural ranks bottom-up.  Ranks are local to a
    # depth, which is sufficient because every child is exactly one level down.
    rank: dict[int, int] = {}
    for current_depth in range(max(by_depth), -1, -1):
        keys: dict[int, tuple[bool, tuple[int, ...]]] = {}
        for node_id in by_depth[current_depth]:
            keys[node_id] = (
                node_id in terminals,
                tuple(sorted(rank[child] for child in children[node_id])),
            )
        rank_by_key = {key: index for index, key in enumerate(sorted(set(keys.values())))}
        for node_id, key in keys.items():
            rank[node_id] = rank_by_key[key]
    ordered_children = tuple(
        tuple(sorted(node_children, key=lambda child: (rank[child], child)))
        for node_children in children
    )

    original_by_canonical: list[int] = []
    canonical_queue: deque[int] = deque((0,))
    while canonical_queue:
        original = canonical_queue.popleft()
        original_by_canonical.append(original)
        canonical_queue.extend(ordered_children[original])
    canonical_by_original = {
        original: canonical for canonical, original in enumerate(original_by_canonical)
    }
    canonical_children = tuple(
        tuple(canonical_by_original[child] for child in ordered_children[original])
        for original in original_by_canonical
    )
    canonical_parent: list[int | None] = [None] * node_count
    canonical_depth: list[int] = [0] * node_count
    canonical_path: list[tuple[int, ...]] = [()] * node_count
    canonical_terminals: set[int] = set()
    for canonical, original in enumerate(original_by_canonical):
        if original in terminals:
            canonical_terminals.add(canonical)
        for child_index, child in enumerate(canonical_children[canonical]):
            canonical_parent[child] = canonical
            canonical_depth[child] = canonical_depth[canonical] + 1
            canonical_path[child] = canonical_path[canonical] + (child_index,)

    lane_by_node: dict[int, int] = {0: 0}
    forks: list[PrefixWaveFork] = []
    segments: list[PrefixWaveSegment] = []
    groups: list[PrefixWaveCompatibleGroup] = []
    frontiers: list[PrefixWaveFrontier] = []
    next_lane_id = 1
    next_sequence = 0
    frontier_nodes: tuple[int, ...] = (0,)
    frontier_index = 0

    while frontier_nodes:
        first_operation = next_sequence
        first_fork = len(forks)
        first_segment = len(segments)
        first_group = len(groups)
        specs: list[_SegmentSpec] = []

        # The complete fork phase is emitted before any lane in this frontier
        # can be mutated by a segment.
        for parent in sorted(frontier_nodes, key=lambda item: canonical_path[item]):
            node_children = canonical_children[parent]
            if not node_children:
                raise PrefixWaveTopologyError("terminal entered a compute frontier")
            source_lane = lane_by_node[parent]
            lane_by_node[node_children[0]] = source_lane
            for child in node_children[1:]:
                target_lane = next_lane_id
                next_lane_id += 1
                lane_by_node[child] = target_lane
                forks.append(
                    PrefixWaveFork(
                        fork_id=len(forks),
                        sequence_index=next_sequence,
                        frontier_index=frontier_index,
                        divergence_node_id=parent,
                        source_lane_id=source_lane,
                        target_lane_id=target_lane,
                        target_child_node_id=child,
                    )
                )
                next_sequence += 1

        for parent in sorted(frontier_nodes, key=lambda item: canonical_path[item]):
            for first_child in canonical_children[parent]:
                lane_id = lane_by_node[first_child]
                chain = [first_child]
                cursor_node = first_child
                while len(canonical_children[cursor_node]) == 1:
                    cursor_node = canonical_children[cursor_node][0]
                    lane_by_node[cursor_node] = lane_id
                    chain.append(cursor_node)
                specs.append(
                    _SegmentSpec(
                        parent_node_id=parent,
                        lane_id=lane_id,
                        node_ids=tuple(chain),
                        start_depth=canonical_depth[parent],
                    )
                )

        specs.sort(
            key=lambda spec: (
                spec.start_depth,
                spec.token_count,
                canonical_path[spec.node_ids[0]],
            )
        )
        next_frontier: list[int] = []
        for compatibility, iterator in groupby(
            specs, key=lambda spec: (spec.start_depth, spec.token_count)
        ):
            compatible_specs = tuple(iterator)
            group_id = len(groups)
            group_first_segment = len(segments)
            group_first_operation = next_sequence
            for spec in compatible_specs:
                segment = PrefixWaveSegment(
                    segment_id=len(segments),
                    sequence_index=next_sequence,
                    frontier_index=frontier_index,
                    compatible_group_id=group_id,
                    lane_id=spec.lane_id,
                    parent_node_id=spec.parent_node_id,
                    node_ids=spec.node_ids,
                    start_depth=spec.start_depth,
                    end_depth=spec.start_depth + spec.token_count,
                )
                segments.append(segment)
                next_sequence += 1
                endpoint = spec.node_ids[-1]
                if len(canonical_children[endpoint]) > 1:
                    next_frontier.append(endpoint)
            groups.append(
                PrefixWaveCompatibleGroup(
                    group_id=group_id,
                    frontier_index=frontier_index,
                    start_depth=compatibility[0],
                    token_count=compatibility[1],
                    first_segment_id=group_first_segment,
                    segment_count=len(compatible_specs),
                    first_operation_index=group_first_operation,
                    last_operation_index=next_sequence - 1,
                )
            )

        frontiers.append(
            PrefixWaveFrontier(
                frontier_index=frontier_index,
                first_operation_index=first_operation,
                operation_count=next_sequence - first_operation,
                first_fork_id=first_fork,
                fork_count=len(forks) - first_fork,
                first_segment_id=first_segment,
                segment_count=len(segments) - first_segment,
                first_group_id=first_group,
                group_count=len(groups) - first_group,
            )
        )
        frontier_nodes = tuple(sorted(next_frontier, key=lambda item: canonical_path[item]))
        frontier_index += 1

    if set(lane_by_node) != set(range(node_count)):
        raise PrefixWaveTopologyError("canonical planner did not assign every node lane")
    if next_lane_id != len(canonical_terminals):
        raise PrefixWaveTopologyError("canonical lane count does not equal leaf count")
    computed = tuple(node for segment in segments for node in segment.node_ids)
    if len(computed) != node_count - 1 or set(computed) != set(range(1, node_count)):
        raise PrefixWaveTopologyError("canonical planner did not compute each node once")

    nodes = tuple(
        PrefixWaveNode(
            node_id=node_id,
            parent_node_id=canonical_parent[node_id],
            lane_id=lane_by_node[node_id],
            depth=canonical_depth[node_id],
            terminal=node_id in canonical_terminals,
        )
        for node_id in range(node_count)
    )
    leaf_nodes = sorted(canonical_terminals, key=lambda node: canonical_path[node])
    leaves = tuple(
        PrefixWaveLeaf(
            leaf_id=leaf_id,
            node_id=node_id,
            lane_id=lane_by_node[node_id],
            depth=canonical_depth[node_id],
        )
        for leaf_id, node_id in enumerate(leaf_nodes)
    )
    return _CanonicalStructure(
        nodes=nodes,
        forks=tuple(forks),
        segments=tuple(segments),
        frontiers=tuple(frontiers),
        groups=tuple(groups),
        leaves=leaves,
        original_node_ids=tuple(original_by_canonical),
    )


def _serialize(
    topology: PrefixWaveTopology,
    *,
    limits: PrefixWaveTopologyLimits,
    _already_validated: bool = False,
) -> tuple[bytes, bytes]:
    if not _already_validated:
        validate_prefix_wave_topology(topology, limits=limits)
    node_refs = tuple(node for segment in topology.segments for node in segment.node_ids)
    counts = (
        len(topology.lane_request_ids),
        len(topology.nodes),
        len(topology.forks),
        len(topology.segments),
        len(topology.frontiers),
        len(topology.compatible_groups),
        len(topology.leaves),
        len(node_refs),
    )
    body_bytes = _body_size(*counts)
    if HEADER_BYTES + body_bytes > limits.max_packet_bytes:
        raise PrefixWaveTopologyError("PrefixWave topology packet exceeds sealed limit")

    body = bytearray(body_bytes)
    cursor = 0
    for request_id in topology.lane_request_ids:
        _LANE.pack_into(body, cursor, request_id)
        cursor += LANE_BYTES
    for node in topology.nodes:
        _NODE.pack_into(
            body,
            cursor,
            PARENT_NONE if node.parent_node_id is None else node.parent_node_id,
            node.lane_id,
            node.depth,
            _TERMINAL_FLAG if node.terminal else 0,
        )
        cursor += NODE_BYTES
    for fork in topology.forks:
        _FORK.pack_into(
            body,
            cursor,
            fork.sequence_index,
            fork.frontier_index,
            fork.divergence_node_id,
            fork.source_lane_id,
            fork.target_lane_id,
            fork.target_child_node_id,
        )
        cursor += FORK_BYTES
    ref_offset = 0
    for segment in topology.segments:
        _SEGMENT.pack_into(
            body,
            cursor,
            segment.sequence_index,
            segment.frontier_index,
            segment.compatible_group_id,
            segment.lane_id,
            segment.parent_node_id,
            ref_offset,
            len(segment.node_ids),
            segment.start_depth,
            segment.end_depth,
        )
        cursor += SEGMENT_BYTES
        ref_offset += len(segment.node_ids)
    for frontier in topology.frontiers:
        _FRONTIER.pack_into(
            body,
            cursor,
            frontier.first_operation_index,
            frontier.operation_count,
            frontier.first_fork_id,
            frontier.fork_count,
            frontier.first_segment_id,
            frontier.segment_count,
            frontier.first_group_id,
            frontier.group_count,
        )
        cursor += FRONTIER_BYTES
    for group in topology.compatible_groups:
        _GROUP.pack_into(
            body,
            cursor,
            group.frontier_index,
            group.start_depth,
            group.token_count,
            group.first_segment_id,
            group.segment_count,
            group.first_operation_index,
            group.last_operation_index,
        )
        cursor += GROUP_BYTES
    for leaf in topology.leaves:
        _LEAF.pack_into(body, cursor, leaf.node_id, leaf.lane_id, leaf.depth)
        cursor += LEAF_BYTES
    for node_id in node_refs:
        _NODE_REF.pack_into(body, cursor, node_id)
        cursor += NODE_REF_BYTES
    if cursor != body_bytes:
        raise RuntimeError("PrefixWave encoder cursor did not fill sealed body")

    unsigned_header = _HEADER.pack(
        MAGIC,
        VERSION,
        0,
        HEADER_BYTES,
        topology.nonce,
        topology.parent_request_id,
        topology.step,
        topology.shared_prefix_depth,
        *counts,
        body_bytes,
        0,
        _ZERO_DIGEST,
    )
    digest = hashlib.sha256(_DIGEST_DOMAIN + unsigned_header + body).digest()
    header = _HEADER.pack(
        MAGIC,
        VERSION,
        0,
        HEADER_BYTES,
        topology.nonce,
        topology.parent_request_id,
        topology.step,
        topology.shared_prefix_depth,
        *counts,
        body_bytes,
        0,
        digest,
    )
    return header + body, digest


def _validated_counts(
    *,
    lane_count: int,
    node_count: int,
    fork_count: int,
    segment_count: int,
    frontier_count: int,
    group_count: int,
    leaf_count: int,
    node_ref_count: int,
    limits: PrefixWaveTopologyLimits,
) -> tuple[int, int, int, int, int, int, int, int]:
    lanes = _bounded("lane_count", lane_count, 1, limits.max_lanes)
    nodes = _bounded("node_count", node_count, 2, limits.max_nodes)
    forks = _bounded("fork_count", fork_count, 0, limits.max_forks)
    segments = _bounded("segment_count", segment_count, 1, limits.max_segments)
    frontiers = _bounded("frontier_count", frontier_count, 1, limits.max_frontiers)
    groups = _bounded("group_count", group_count, 1, limits.max_groups)
    leaves = _bounded("leaf_count", leaf_count, 1, limits.max_leaves)
    refs = _bounded("node_ref_count", node_ref_count, 1, limits.max_node_refs)
    if lanes != leaves:
        raise PrefixWaveTopologyError("lane_count must equal leaf_count")
    if forks != lanes - 1:
        raise PrefixWaveTopologyError("fork_count must equal lane_count minus one")
    if refs != nodes - 1:
        raise PrefixWaveTopologyError("node refs must cover every non-root node exactly once")
    if segments > refs or groups > segments or frontiers > segments:
        raise PrefixWaveTopologyError("topology counts violate structural bounds")
    return lanes, nodes, forks, segments, frontiers, groups, leaves, refs


def _body_size(
    lane_count: int,
    node_count: int,
    fork_count: int,
    segment_count: int,
    frontier_count: int,
    group_count: int,
    leaf_count: int,
    node_ref_count: int,
) -> int:
    total = 0
    for count, width in (
        (lane_count, LANE_BYTES),
        (node_count, NODE_BYTES),
        (fork_count, FORK_BYTES),
        (segment_count, SEGMENT_BYTES),
        (frontier_count, FRONTIER_BYTES),
        (group_count, GROUP_BYTES),
        (leaf_count, LEAF_BYTES),
        (node_ref_count, NODE_REF_BYTES),
    ):
        total = _checked_add(total, _checked_mul(count, width, MAX_PACKET_BYTES), MAX_PACKET_BYTES)
    return total


def _lane_request_ids(values: Sequence[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes, bytearray, memoryview)) or not isinstance(
        values, Sequence
    ):
        raise PrefixWaveTopologyError("lane_request_ids must be a sequence")
    normalized = tuple(
        _bounded("lane request id", value, 0, UINT64_MAX) for value in values
    )
    if any(left >= right for left, right in zip(normalized, normalized[1:])):
        raise PrefixWaveTopologyError(
            "lane request ids must be strictly increasing"
        )
    return normalized


def _immutable_bytes(value: bytes | bytearray | memoryview) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, (bytearray, memoryview)):
        try:
            return bytes(memoryview(value).cast("B"))
        except (TypeError, ValueError) as error:
            raise PrefixWaveTopologyError("packet must be a contiguous byte buffer") from error
    raise PrefixWaveTopologyError("packet must be bytes-like")


def _checked_add(left: int, right: int, maximum: int) -> int:
    if left < 0 or right < 0 or left > maximum - right:
        raise PrefixWaveTopologyError("PrefixWave topology size overflows sealed range")
    return left + right


def _checked_mul(left: int, right: int, maximum: int) -> int:
    if left < 0 or right < 0 or (left and right > maximum // left):
        raise PrefixWaveTopologyError("PrefixWave topology size overflows sealed range")
    return left * right


def _require_limits(value: object) -> PrefixWaveTopologyLimits:
    if not isinstance(value, PrefixWaveTopologyLimits):
        raise PrefixWaveTopologyError("limits must be PrefixWaveTopologyLimits")
    return value


__all__ = [
    "DEFAULT_LIMITS",
    "HEADER_BYTES",
    "MAGIC",
    "PrefixWaveCompatibleGroup",
    "PrefixWaveFork",
    "PrefixWaveFrontier",
    "PrefixWaveLeaf",
    "PrefixWaveNode",
    "PrefixWaveSegment",
    "PrefixWaveTopology",
    "PrefixWaveTopologyError",
    "PrefixWaveTopologyLimits",
    "PrefixWaveTopologyMapping",
    "VERSION",
    "build_prefix_wave_topology",
    "decode_prefix_wave_topology",
    "encode_decoded_prefix_wave_topology",
    "encode_prefix_wave_topology",
    "prefix_wave_topology_mapping",
    "validate_prefix_wave_topology",
]
