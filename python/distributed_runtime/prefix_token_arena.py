"""Root-local token arena and cut-through records for PrefixWave v1.

``prefix_wave_topology`` intentionally contains no token ids.  This module
joins that canonical structural topology to the token-bearing
``PrefixSegmentSchedule`` *at the root only*.  The result has two distinct
seals with deliberately different meanings:

* ``wave_digest`` binds the sealed topology, chunking parameters and
  token-free cut-through descriptors.  It does **not** bind token ids.
* ``local_arena_digest`` additionally binds the root-local tokens and mapping
  tables.  It is a cache/debugging integrity key only.  It must never be sent
  to peers or treated as peer authentication.

Neither digest is a MAC.  A transport crossing a trust boundary still needs
an authenticated channel or a keyed protocol authenticator.

The v1 stream slices one compatible group at a time.  Within a group, every
record covers a contiguous token-depth slice of one or more contiguous
segments.  Canonical node ids are flattened segment-major; token payloads are
looked up separately from the local arena.  Consequently the structural
descriptor is stable under arbitrary token relabeling.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from numbers import Integral
import struct

from .prefix_segment_schedule import PrefixSegmentSchedule
from .prefix_wave_topology import (
    DEFAULT_LIMITS,
    MAX_DEPTH,
    MAX_NODES,
    MAX_SEGMENTS,
    PrefixWaveTopology,
    PrefixWaveTopologyError,
    PrefixWaveTopologyLimits,
    build_prefix_wave_topology,
    encode_decoded_prefix_wave_topology,
    prefix_wave_topology_mapping,
)


UINT32_MAX = (1 << 32) - 1
UINT64_MAX = (1 << 64) - 1
VERSION = 1

DEFAULT_MAX_SLICE_TOKENS = 8
DEFAULT_MAX_SEGMENTS_PER_RECORD = 64
DEFAULT_MAX_NODES_PER_RECORD = 4_096

_DESCRIPTOR_MAGIC = b"PWCR"
_DESCRIPTOR_HEADER = struct.Struct("<4sB3xIIII")
_RECORD_HEADER = struct.Struct("<7I")
_U32 = struct.Struct("<I")
_U64 = struct.Struct("<Q")

_WAVE_DIGEST_DOMAIN = b"GDLP/PREFIX-WAVE/cut-through-v1\0"
_LOCAL_ARENA_DIGEST_DOMAIN = b"GDLP/PREFIX-TOKEN-ARENA/root-local-v1\0"


class PrefixTokenArenaError(ValueError):
    """A local arena does not exactly match its sealed PrefixWave topology."""


def _bounded(name: str, value: object, minimum: int, maximum: int) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise PrefixTokenArenaError(f"{name} must be an integer")
    normalized = int(value)
    if normalized < minimum or normalized > maximum:
        raise PrefixTokenArenaError(f"{name} must be in [{minimum}, {maximum}]")
    return normalized


@dataclass(frozen=True, slots=True)
class PrefixWaveChunking:
    """Sealed v1 chunking policy for cut-through compatible groups.

    ``max_nodes_per_record`` includes all segment-major nodes in one record.
    If a complete compatible group does not fit, its contiguous segment range
    is partitioned deterministically.  No record ever mixes group ids.
    """

    max_slice_tokens: int = DEFAULT_MAX_SLICE_TOKENS
    max_segments_per_record: int = DEFAULT_MAX_SEGMENTS_PER_RECORD
    max_nodes_per_record: int = DEFAULT_MAX_NODES_PER_RECORD

    def __post_init__(self) -> None:
        _bounded("max_slice_tokens", self.max_slice_tokens, 1, MAX_DEPTH)
        _bounded(
            "max_segments_per_record",
            self.max_segments_per_record,
            1,
            MAX_SEGMENTS,
        )
        _bounded(
            "max_nodes_per_record",
            self.max_nodes_per_record,
            1,
            MAX_NODES - 1,
        )


DEFAULT_CHUNKING = PrefixWaveChunking()


@dataclass(frozen=True, slots=True)
class PrefixCutThroughRecord:
    """One token-free structural record in canonical execution order.

    ``canonical_node_ids`` is segment-major.  Every segment contributes
    exactly ``slice_tokens`` ids, so no offset table is needed on the wire.
    Token ids are intentionally absent and must be obtained from the
    root-local :class:`PrefixTokenArena`.
    """

    record_id: int
    frontier_index: int
    group_id: int
    slice_offset: int
    slice_tokens: int
    segment_ids: tuple[int, ...]
    canonical_node_ids: tuple[int, ...]

    @property
    def segment_count(self) -> int:
        return len(self.segment_ids)

    @property
    def node_count(self) -> int:
        return len(self.canonical_node_ids)

    def canonical_nodes_by_segment(self) -> tuple[tuple[int, ...], ...]:
        width = self.slice_tokens
        return tuple(
            self.canonical_node_ids[offset : offset + width]
            for offset in range(0, len(self.canonical_node_ids), width)
        )


@dataclass(frozen=True, slots=True)
class PrefixTokenArena:
    """Immutable root-local join of tokens, mappings and stream descriptors.

    Tuple indices are canonical ids unless their field says otherwise.
    ``token_ids_by_canonical_node`` omits the virtual root: canonical node
    ``n`` is stored at index ``n - 1``.  ``local_arena_digest`` is explicitly
    local-only and unauthenticated; peers receive neither it nor this token
    table.
    """

    nonce: int
    parent_request_id: int
    step: int
    topology_digest: bytes
    chunking: PrefixWaveChunking
    token_ids_by_canonical_node: tuple[int, ...]
    schedule_node_ids_by_canonical_node_id: tuple[int, ...]
    schedule_leaf_ids_by_canonical_leaf_id: tuple[int, ...]
    # Terminal-lane bijection.  Transient inheritance inside isomorphic
    # subtrees can differ because the canonical topology deliberately ignores
    # token ordering; a node-by-node lane mapping therefore does not exist.
    schedule_lane_ids_by_canonical_lane_id: tuple[int, ...]
    physical_request_ids_by_canonical_lane_id: tuple[int, ...]
    records: tuple[PrefixCutThroughRecord, ...]
    structural_descriptor_bytes: bytes
    wave_digest: bytes
    local_arena_digest: bytes

    @property
    def wave_digest_hex(self) -> str:
        return self.wave_digest.hex()

    @property
    def local_arena_digest_hex(self) -> str:
        return self.local_arena_digest.hex()

    def token_ids_for_record(self, record_id: int) -> tuple[int, ...]:
        """Return the separate root-only token payload for one descriptor."""

        normalized = _bounded("record_id", record_id, 0, len(self.records) - 1)
        record = self.records[normalized]
        if record.record_id != normalized:
            raise PrefixTokenArenaError("record ids are not canonical and contiguous")
        token_ids: list[int] = []
        for node_id in record.canonical_node_ids:
            normalized_node = _bounded(
                "canonical_node_id",
                node_id,
                1,
                len(self.token_ids_by_canonical_node),
            )
            token_ids.append(
                self.token_ids_by_canonical_node[normalized_node - 1]
            )
        return tuple(token_ids)


def build_prefix_token_arena(
    schedule: PrefixSegmentSchedule,
    topology: PrefixWaveTopology,
    *,
    nonce: int,
    parent_request_id: int,
    step: int,
    chunking: PrefixWaveChunking = DEFAULT_CHUNKING,
    limits: PrefixWaveTopologyLimits = DEFAULT_LIMITS,
) -> PrefixTokenArena:
    """Join one schedule to its exact sealed topology and build stream records.

    The transaction identity is supplied independently and compared with the
    topology.  This prevents a valid topology for another nonce, parent or
    decoding step from being reused accidentally.
    """

    expected_nonce = _bounded("nonce", nonce, 0, UINT64_MAX)
    expected_parent = _bounded("parent_request_id", parent_request_id, 0, UINT64_MAX)
    expected_step = _bounded("step", step, 0, UINT32_MAX)
    if not isinstance(chunking, PrefixWaveChunking):
        raise PrefixTokenArenaError("chunking must be a PrefixWaveChunking")
    if not isinstance(limits, PrefixWaveTopologyLimits):
        raise PrefixTokenArenaError("limits must be PrefixWaveTopologyLimits")
    if not isinstance(schedule, PrefixSegmentSchedule):
        raise PrefixTokenArenaError("schedule must be a PrefixSegmentSchedule")
    if not isinstance(topology, PrefixWaveTopology):
        raise PrefixTokenArenaError("topology must be a PrefixWaveTopology")
    if (
        topology.nonce != expected_nonce
        or topology.parent_request_id != expected_parent
        or topology.step != expected_step
    ):
        raise PrefixTokenArenaError(
            "PrefixWave topology transaction identity does not match the local wave"
        )
    if not isinstance(topology.digest, bytes) or len(topology.digest) != 32:
        raise PrefixTokenArenaError("topology must carry a sealed 32-byte digest")

    # Re-encoding proves that the object is canonical and that its digest still
    # binds its current identity and descriptors.  Rebuilding from the source
    # schedule then proves the two objects describe the same structural tree.
    try:
        encode_decoded_prefix_wave_topology(topology, limits=limits)
        expected_topology = build_prefix_wave_topology(
            schedule,
            nonce=expected_nonce,
            parent_request_id=expected_parent,
            step=expected_step,
            lane_request_ids=topology.lane_request_ids,
            limits=limits,
        )
        mapping = prefix_wave_topology_mapping(
            schedule, max_depth=limits.max_depth
        )
    except (PrefixWaveTopologyError, TypeError, ValueError, RuntimeError) as error:
        raise PrefixTokenArenaError(f"invalid PrefixWave topology join: {error}") from error
    if topology != expected_topology:
        raise PrefixTokenArenaError(
            "schedule and topology do not describe the same canonical PrefixWave"
        )

    node_mapping = mapping.schedule_node_ids_by_canonical_node_id
    leaf_mapping = mapping.schedule_leaf_ids_by_canonical_leaf_id
    if len(node_mapping) != len(topology.nodes):
        raise PrefixTokenArenaError("canonical node mapping has the wrong length")
    if len(leaf_mapping) != len(topology.leaves):
        raise PrefixTokenArenaError("canonical leaf mapping has the wrong length")

    token_ids: list[int] = []
    for canonical_node_id, schedule_node_id in enumerate(node_mapping):
        if not 0 <= schedule_node_id < len(schedule.nodes):
            raise PrefixTokenArenaError("canonical node mapping is out of bounds")
        canonical_node = topology.nodes[canonical_node_id]
        schedule_node = schedule.nodes[schedule_node_id]
        if (
            schedule_node.depth != canonical_node.depth
            or schedule_node.terminal != canonical_node.terminal
        ):
            raise PrefixTokenArenaError("canonical node mapping changed node semantics")
        if canonical_node.parent_node_id is None:
            if schedule_node.parent_node_id is not None:
                raise PrefixTokenArenaError("canonical root mapping is invalid")
        elif (
            schedule_node.parent_node_id
            != node_mapping[canonical_node.parent_node_id]
        ):
            raise PrefixTokenArenaError("canonical node mapping changed a parent edge")
        if canonical_node_id:
            if schedule_node.token is None:
                raise PrefixTokenArenaError("non-root canonical node has no token id")
            token_ids.append(_bounded("token id", schedule_node.token, 0, UINT32_MAX))

    # Canonical sibling order ignores token values, whereas the source
    # schedule chooses primary lanes by token order.  Their transient lane ids
    # can consequently cross at a divergence.  Terminal leaves still define
    # one stable, complete lane bijection, which is the mapping needed to join
    # canonical physical request ids back to root-local candidate results.
    canonical_lane_to_schedule: dict[int, int] = {}
    schedule_lane_to_canonical: dict[int, int] = {}
    for canonical_leaf_id, schedule_leaf_id in enumerate(leaf_mapping):
        if not 0 <= schedule_leaf_id < len(schedule.leaves):
            raise PrefixTokenArenaError("canonical leaf mapping is out of bounds")
        canonical_leaf = topology.leaves[canonical_leaf_id]
        schedule_leaf = schedule.leaves[schedule_leaf_id]
        if node_mapping[canonical_leaf.node_id] != schedule_leaf.node_id:
            raise PrefixTokenArenaError("canonical leaf mapping changed its terminal node")
        _bind_lane_bijection(
            canonical_leaf.lane_id,
            schedule_leaf.lane_id,
            canonical_lane_to_schedule,
            schedule_lane_to_canonical,
        )
    lane_count = len(topology.lane_request_ids)
    if set(canonical_lane_to_schedule) != set(range(lane_count)):
        raise PrefixTokenArenaError("canonical terminal-lane mapping is incomplete")
    if set(schedule_lane_to_canonical) != set(range(schedule.cost.lane_count)):
        raise PrefixTokenArenaError("schedule terminal-lane mapping is incomplete")
    if len(schedule_lane_to_canonical) != lane_count:
        raise PrefixTokenArenaError(
            "canonical-to-schedule terminal-lane mapping is not bijective"
        )
    schedule_lanes = tuple(
        canonical_lane_to_schedule[lane_id] for lane_id in range(lane_count)
    )

    records = _build_records(topology, chunking)
    _validate_record_coverage(topology, records, chunking)
    descriptor = _encode_structural_descriptors(chunking, records)
    wave_digest = hashlib.sha256(
        _WAVE_DIGEST_DOMAIN
        + topology.digest
        + _U64.pack(len(descriptor))
        + descriptor
    ).digest()
    physical_request_ids = tuple(topology.lane_request_ids)
    local_digest = _local_arena_digest(
        wave_digest=wave_digest,
        token_ids=tuple(token_ids),
        node_mapping=node_mapping,
        leaf_mapping=leaf_mapping,
        lane_mapping=schedule_lanes,
        physical_request_ids=physical_request_ids,
    )
    return PrefixTokenArena(
        nonce=expected_nonce,
        parent_request_id=expected_parent,
        step=expected_step,
        topology_digest=topology.digest,
        chunking=chunking,
        token_ids_by_canonical_node=tuple(token_ids),
        schedule_node_ids_by_canonical_node_id=node_mapping,
        schedule_leaf_ids_by_canonical_leaf_id=leaf_mapping,
        schedule_lane_ids_by_canonical_lane_id=schedule_lanes,
        physical_request_ids_by_canonical_lane_id=physical_request_ids,
        records=records,
        structural_descriptor_bytes=descriptor,
        wave_digest=wave_digest,
        local_arena_digest=local_digest,
    )


def validate_prefix_token_arena(
    arena: PrefixTokenArena,
    schedule: PrefixSegmentSchedule,
    topology: PrefixWaveTopology,
    *,
    nonce: int,
    parent_request_id: int,
    step: int,
    limits: PrefixWaveTopologyLimits = DEFAULT_LIMITS,
) -> None:
    """Rebuild and compare every local field; any mutation fails closed."""

    if not isinstance(arena, PrefixTokenArena):
        raise PrefixTokenArenaError("arena must be a PrefixTokenArena")
    try:
        expected = build_prefix_token_arena(
            schedule,
            topology,
            nonce=nonce,
            parent_request_id=parent_request_id,
            step=step,
            chunking=arena.chunking,
            limits=limits,
        )
    except PrefixTokenArenaError:
        raise
    except (TypeError, ValueError, RuntimeError) as error:
        raise PrefixTokenArenaError(f"invalid PrefixTokenArena: {error}") from error
    if arena != expected:
        raise PrefixTokenArenaError(
            "PrefixTokenArena fields, descriptors, mappings, or digests are stale"
        )


def _bind_lane_bijection(
    canonical_lane: int,
    schedule_lane: int,
    canonical_to_schedule: dict[int, int],
    schedule_to_canonical: dict[int, int],
) -> None:
    previous_schedule = canonical_to_schedule.setdefault(canonical_lane, schedule_lane)
    if previous_schedule != schedule_lane:
        raise PrefixTokenArenaError(
            "one canonical lane maps to multiple schedule lanes"
        )
    previous_canonical = schedule_to_canonical.setdefault(schedule_lane, canonical_lane)
    if previous_canonical != canonical_lane:
        raise PrefixTokenArenaError(
            "multiple canonical lanes map to one schedule lane"
        )


def _build_records(
    topology: PrefixWaveTopology,
    chunking: PrefixWaveChunking,
) -> tuple[PrefixCutThroughRecord, ...]:
    records: list[PrefixCutThroughRecord] = []
    for group in topology.compatible_groups:
        group_segments = topology.segments[
            group.first_segment_id : group.first_segment_id + group.segment_count
        ]
        if not group_segments:
            raise PrefixTokenArenaError("compatible group contains no segments")
        offset = 0
        while offset < group.token_count:
            slice_tokens = min(
                chunking.max_slice_tokens,
                chunking.max_nodes_per_record,
                group.token_count - offset,
            )
            segments_per_record = min(
                chunking.max_segments_per_record,
                chunking.max_nodes_per_record // slice_tokens,
            )
            if segments_per_record < 1:
                raise PrefixTokenArenaError(
                    "chunking cannot fit one segment slice in a record"
                )
            for first in range(0, len(group_segments), segments_per_record):
                selected = group_segments[first : first + segments_per_record]
                node_ids = tuple(
                    node_id
                    for segment in selected
                    for node_id in segment.node_ids[offset : offset + slice_tokens]
                )
                records.append(
                    PrefixCutThroughRecord(
                        record_id=len(records),
                        frontier_index=group.frontier_index,
                        group_id=group.group_id,
                        slice_offset=offset,
                        slice_tokens=slice_tokens,
                        segment_ids=tuple(segment.segment_id for segment in selected),
                        canonical_node_ids=node_ids,
                    )
                )
            offset += slice_tokens
    return tuple(records)


def _validate_record_coverage(
    topology: PrefixWaveTopology,
    records: tuple[PrefixCutThroughRecord, ...],
    chunking: PrefixWaveChunking,
) -> None:
    expected = _build_records(topology, chunking)
    if records != expected:
        raise PrefixTokenArenaError("cut-through records are not canonical")
    if not records:
        raise PrefixTokenArenaError("PrefixWave requires at least one cut-through record")
    observed_nodes = tuple(
        node_id for record in records for node_id in record.canonical_node_ids
    )
    if len(observed_nodes) != len(topology.nodes) - 1:
        raise PrefixTokenArenaError("cut-through records do not cover every token node")
    if set(observed_nodes) != set(range(1, len(topology.nodes))):
        raise PrefixTokenArenaError(
            "cut-through records duplicate, omit, or reference invalid token nodes"
        )
    previous_group = -1
    previous_offset = -1
    for record_id, record in enumerate(records):
        if record.record_id != record_id:
            raise PrefixTokenArenaError("record ids must be canonical and contiguous")
        if record.group_id < previous_group:
            raise PrefixTokenArenaError("cut-through group order regressed")
        if record.group_id != previous_group:
            previous_group = record.group_id
            previous_offset = -1
        if record.slice_offset < previous_offset:
            raise PrefixTokenArenaError("cut-through slice order regressed")
        previous_offset = record.slice_offset
        if len(record.canonical_node_ids) != len(record.segment_ids) * record.slice_tokens:
            raise PrefixTokenArenaError("record node matrix is not rectangular")
        if len(record.canonical_node_ids) > chunking.max_nodes_per_record:
            raise PrefixTokenArenaError("record exceeds max_nodes_per_record")
        if len(record.segment_ids) > chunking.max_segments_per_record:
            raise PrefixTokenArenaError("record exceeds max_segments_per_record")


def _encode_structural_descriptors(
    chunking: PrefixWaveChunking,
    records: tuple[PrefixCutThroughRecord, ...],
) -> bytes:
    """Encode only canonical structure; token ids never enter these bytes."""

    parts = [
        _DESCRIPTOR_HEADER.pack(
            _DESCRIPTOR_MAGIC,
            VERSION,
            chunking.max_slice_tokens,
            chunking.max_segments_per_record,
            chunking.max_nodes_per_record,
            len(records),
        )
    ]
    for record_id, record in enumerate(records):
        if not isinstance(record, PrefixCutThroughRecord):
            raise PrefixTokenArenaError("records must be PrefixCutThroughRecord values")
        values = (
            record.record_id,
            record.frontier_index,
            record.group_id,
            record.slice_offset,
            record.slice_tokens,
            len(record.segment_ids),
            len(record.canonical_node_ids),
        )
        for name, value in zip(
            (
                "record_id",
                "frontier_index",
                "group_id",
                "slice_offset",
                "slice_tokens",
                "segment_count",
                "node_count",
            ),
            values,
        ):
            minimum = (
                1
                if name in {"slice_tokens", "segment_count", "node_count"}
                else 0
            )
            _bounded(name, value, minimum, UINT32_MAX)
        if record.record_id != record_id:
            raise PrefixTokenArenaError("record ids are not canonical")
        parts.append(_RECORD_HEADER.pack(*values))
        for segment_id in record.segment_ids:
            parts.append(
                _U32.pack(_bounded("segment_id", segment_id, 0, UINT32_MAX))
            )
        for node_id in record.canonical_node_ids:
            parts.append(
                _U32.pack(
                    _bounded("canonical_node_id", node_id, 1, UINT32_MAX)
                )
            )
    return b"".join(parts)


def _local_arena_digest(
    *,
    wave_digest: bytes,
    token_ids: tuple[int, ...],
    node_mapping: tuple[int, ...],
    leaf_mapping: tuple[int, ...],
    lane_mapping: tuple[int, ...],
    physical_request_ids: tuple[int, ...],
) -> bytes:
    payload = bytearray()
    payload.extend(wave_digest)
    for values in (token_ids, node_mapping, leaf_mapping, lane_mapping):
        payload.extend(_U32.pack(len(values)))
        for value in values:
            payload.extend(
                _U32.pack(_bounded("local arena value", value, 0, UINT32_MAX))
            )
    payload.extend(_U32.pack(len(physical_request_ids)))
    for request_id in physical_request_ids:
        payload.extend(
            _U64.pack(
                _bounded("physical request id", request_id, 0, UINT64_MAX)
            )
        )
    return hashlib.sha256(_LOCAL_ARENA_DIGEST_DOMAIN + payload).digest()


__all__ = [
    "DEFAULT_CHUNKING",
    "DEFAULT_MAX_NODES_PER_RECORD",
    "DEFAULT_MAX_SEGMENTS_PER_RECORD",
    "DEFAULT_MAX_SLICE_TOKENS",
    "PrefixCutThroughRecord",
    "PrefixTokenArena",
    "PrefixTokenArenaError",
    "PrefixWaveChunking",
    "VERSION",
    "build_prefix_token_arena",
    "validate_prefix_token_arena",
]
