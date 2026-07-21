"""Canonical, incremental ``PREFIX_SEGMENT`` stream guard for PrefixWave v1.

This module is deliberately independent from the live socket protocol and from
the stage runtime.  It gives that future integration a small fail-closed core:

* a root-only :class:`~prefix_token_arena.PrefixTokenArena` is reduced to a
  token-free immutable manifest, while remote stages reconstruct the identical
  manifest from records, topology digest and chunking alone;
* every frame binds the wave nonce and digest, the record index and total, the
  complete :class:`~prefix_token_arena.PrefixCutThroughRecord`, the activation
  codec and shape, and a digest of an owned payload snapshot;
* a stream accepts records only in the canonical contiguous order; and
* the mutation callback is the conservative safety boundary.  Validation
  failures before the first callback permit same-route fallback.  Once a
  callback has been entered, every failure is ``route_fatal`` because the
  backend may already have changed KV state.

The SHA-256 values below are corruption/integrity seals, **not** MACs.  Anyone
who can replace a frame can recompute them.  A deployment crossing a trust
boundary must authenticate the transport (for example mTLS, Noise, or
WireGuard plus a keyed protocol authenticator).  This module intentionally does
not pretend that an unkeyed digest authenticates a peer.

Token ids never occur in the manifest or structural wire descriptor.  The
opaque payload is expected to contain activations produced by the preceding
stage, not the root-local candidate-token arena.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
import hashlib
import hmac
from numbers import Integral
import struct
import threading
from typing import Generic, TypeVar

from .prefix_token_arena import (
    PrefixCutThroughRecord,
    PrefixTokenArena,
    PrefixWaveChunking,
)
from .prefix_wave_topology import MAX_NODES, MAX_SEGMENTS


VERSION = 1
UINT16_MAX = (1 << 16) - 1
UINT32_MAX = (1 << 32) - 1
UINT64_MAX = (1 << 64) - 1

DEFAULT_MAX_PAYLOAD_BYTES = 256 * 1024 * 1024
DEFAULT_MAX_PACKET_BYTES = DEFAULT_MAX_PAYLOAD_BYTES + 16 * 1024 * 1024
DEFAULT_MAX_HIDDEN_SIZE = 1_048_576

_MAGIC = b"PWSG"
_FLAGS = 0
_RESERVED = 0

# magic, version, flags, codec, nonce, wave digest, contract digest,
# record index/total, record metadata (six uint32 values), rank/reserved,
# payload length, payload digest, frame binding digest.
_FIXED_HEADER = struct.Struct(
    "<4sBBHQ32s32sIIIIIIIIHHQ32s32s"
)
_BINDING_PREFIX = struct.Struct(
    "<4sBBHQ32s32sIIIIIIIIHHQ"
)
_U16 = struct.Struct("<H")
_U32 = struct.Struct("<I")
_U64 = struct.Struct("<Q")

# Public PrefixTokenArena v1 currently uses this canonical descriptor.  The
# stream builder reproduces it rather than trusting a possibly forged arena
# dataclass.  If the arena format changes, construction fails closed until this
# contract receives an explicit version update.
_ARENA_DESCRIPTOR_MAGIC = b"PWCR"
_ARENA_DESCRIPTOR_HEADER = struct.Struct("<4sB3xIIII")
_ARENA_RECORD_HEADER = struct.Struct("<7I")
_ARENA_WAVE_DIGEST_DOMAIN = b"GDLP/PREFIX-WAVE/cut-through-v1\0"

_CONTRACT_DIGEST_DOMAIN = b"GDLP/PREFIX-SEGMENT/contract-v1\0"
_PAYLOAD_DIGEST_DOMAIN = b"GDLP/PREFIX-SEGMENT/payload-v1\0"
_FRAME_BINDING_DOMAIN = b"GDLP/PREFIX-SEGMENT/frame-v1\0"


class PrefixSegmentCodecError(ValueError):
    """A manifest or wire packet violates the canonical v1 contract."""


class PrefixSegmentStreamState(Enum):
    """Observable lifecycle of one single-use wave receiver."""

    OPEN = "open"
    MUTATING = "mutating"
    COMPLETE = "complete"
    ABORTED = "aborted"
    FATAL = "fatal"


class PrefixSegmentStreamError(RuntimeError):
    """Fail-closed stream failure with an explicit fallback boundary."""

    def __init__(
        self,
        message: str,
        *,
        route_fatal: bool,
        phase: str,
        expected_record_index: int,
        received_record_index: int | None,
        completed_record_count: int,
    ) -> None:
        super().__init__(message)
        self.route_fatal = route_fatal
        self.phase = phase
        self.expected_record_index = expected_record_index
        self.received_record_index = received_record_index
        self.completed_record_count = completed_record_count

    @property
    def fallback_allowed(self) -> bool:
        return not self.route_fatal


def _bounded(name: str, value: object, minimum: int, maximum: int) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise PrefixSegmentCodecError(f"{name} must be an integer")
    normalized = int(value)
    if normalized < minimum or normalized > maximum:
        raise PrefixSegmentCodecError(
            f"{name} must be in [{minimum}, {maximum}]"
        )
    return normalized


def _digest(name: str, value: object) -> bytes:
    if not isinstance(value, bytes) or len(value) != 32:
        raise PrefixSegmentCodecError(f"{name} must be immutable 32-byte data")
    return value


@dataclass(frozen=True, slots=True)
class PrefixSegmentStreamLimits:
    """Allocation limits checked before copying or slicing hostile packets."""

    max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES
    max_packet_bytes: int = DEFAULT_MAX_PACKET_BYTES
    max_records: int = MAX_NODES - 1
    max_hidden_size: int = DEFAULT_MAX_HIDDEN_SIZE

    def __post_init__(self) -> None:
        payload = _bounded(
            "max_payload_bytes", self.max_payload_bytes, 1, UINT64_MAX
        )
        packet = _bounded(
            "max_packet_bytes", self.max_packet_bytes, 1, UINT64_MAX
        )
        _bounded("max_records", self.max_records, 1, MAX_NODES - 1)
        _bounded(
            "max_hidden_size", self.max_hidden_size, 1, UINT32_MAX
        )
        if packet < payload + _FIXED_HEADER.size:
            raise PrefixSegmentCodecError(
                "max_packet_bytes cannot hold max_payload_bytes plus a frame header"
            )


DEFAULT_LIMITS = PrefixSegmentStreamLimits()


@dataclass(frozen=True, slots=True)
class PrefixSegmentStreamManifest:
    """Token-free, immutable receiving contract derived from a local arena.

    Shape is canonical for v1: ``(segment_count, slice_tokens, hidden_size)``.
    ``codec_id`` is an opaque, route-negotiated uint16 value.  The tensor codec
    still owns semantic payload validation; this guard seals its identity and
    shape and protects the stream state without importing tensor runtimes.
    """

    nonce: int
    wave_digest: bytes
    topology_digest: bytes
    chunking: PrefixWaveChunking
    codec_id: int
    hidden_size: int
    records: tuple[PrefixCutThroughRecord, ...]
    shapes: tuple[tuple[int, ...], ...]
    contract_digest: bytes

    @property
    def record_total(self) -> int:
        return len(self.records)


@dataclass(frozen=True, slots=True)
class PrefixSegmentFrame:
    """One fully owned and validated token-free structural frame."""

    nonce: int
    wave_digest: bytes
    contract_digest: bytes
    record_index: int
    record_total: int
    record: PrefixCutThroughRecord
    codec_id: int
    shape: tuple[int, ...]
    payload: bytes
    payload_digest: bytes
    binding_digest: bytes

    @property
    def descriptor_contains_tokens(self) -> bool:
        """Always false; explicit audit hook for callers and tests."""

        return False


@dataclass(frozen=True, slots=True)
class PrefixSegmentStreamSnapshot:
    state: PrefixSegmentStreamState
    next_record_index: int
    record_total: int
    completed_record_count: int
    mutation_started: bool
    fatal_reason: str | None

    @property
    def fallback_allowed(self) -> bool:
        return not self.mutation_started and self.state is not PrefixSegmentStreamState.FATAL


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class PrefixSegmentApplyResult(Generic[T]):
    frame: PrefixSegmentFrame
    mutation_result: T
    completed_record_count: int
    complete: bool


def build_prefix_segment_stream_manifest(
    arena: PrefixTokenArena,
    *,
    expected_nonce: int,
    expected_wave_digest: bytes,
    codec_id: int,
    hidden_size: int,
    limits: PrefixSegmentStreamLimits = DEFAULT_LIMITS,
) -> PrefixSegmentStreamManifest:
    """Reduce a sealed root arena to the exact token-free wire contract.

    The expected identity is deliberately supplied independently, just like
    the identity used to build ``PrefixTokenArena``.  Reading it back from a
    possibly stale/replaced arena would not detect an arena from another live
    reservation.
    """

    if not isinstance(limits, PrefixSegmentStreamLimits):
        raise PrefixSegmentCodecError("limits must be PrefixSegmentStreamLimits")
    _validate_arena_for_stream(arena, limits=limits)
    normalized_nonce = _bounded(
        "expected_nonce", expected_nonce, 0, UINT64_MAX
    )
    normalized_wave_digest = _digest(
        "expected_wave_digest", expected_wave_digest
    )
    if arena.nonce != normalized_nonce:
        raise PrefixSegmentCodecError(
            "arena nonce does not match the independently supplied wave identity"
        )
    if not hmac.compare_digest(arena.wave_digest, normalized_wave_digest):
        raise PrefixSegmentCodecError(
            "arena wave digest does not match the independently supplied identity"
        )
    return build_prefix_segment_stream_manifest_from_records(
        tuple(arena.records),
        nonce=normalized_nonce,
        wave_digest=normalized_wave_digest,
        topology_digest=arena.topology_digest,
        chunking=arena.chunking,
        codec_id=codec_id,
        hidden_size=hidden_size,
        limits=limits,
    )


def build_prefix_segment_stream_manifest_from_records(
    records: tuple[PrefixCutThroughRecord, ...],
    *,
    nonce: int,
    wave_digest: bytes,
    topology_digest: bytes,
    chunking: PrefixWaveChunking,
    codec_id: int,
    hidden_size: int,
    limits: PrefixSegmentStreamLimits = DEFAULT_LIMITS,
) -> PrefixSegmentStreamManifest:
    """Build the same manifest on a remote stage without a token arena.

    ``records``, ``topology_digest``, ``chunking``, ``nonce`` and
    ``wave_digest`` are token-free values from the already validated
    ``PREFIX_PREPARE`` transaction.  The function recomputes the complete PWCR
    descriptor and proves that the supplied wave digest seals exactly this
    topology/descriptor pair before returning a usable stream contract.
    """

    if not isinstance(limits, PrefixSegmentStreamLimits):
        raise PrefixSegmentCodecError("limits must be PrefixSegmentStreamLimits")
    normalized_nonce = _bounded("nonce", nonce, 0, UINT64_MAX)
    normalized_wave_digest = _digest("wave_digest", wave_digest)
    normalized_topology_digest = _digest("topology_digest", topology_digest)
    if not isinstance(chunking, PrefixWaveChunking):
        raise PrefixSegmentCodecError("chunking must be PrefixWaveChunking")
    _validate_records_and_chunking(records, chunking=chunking, limits=limits)
    structural_descriptor = _encode_arena_structural_descriptor(
        chunking, records
    )
    expected_wave_digest = _wave_digest(
        topology_digest=normalized_topology_digest,
        structural_descriptor=structural_descriptor,
    )
    if not hmac.compare_digest(normalized_wave_digest, expected_wave_digest):
        raise PrefixSegmentCodecError(
            "records, topology_digest, or chunking do not match wave_digest"
        )
    normalized_codec = _bounded("codec_id", codec_id, 0, UINT16_MAX)
    normalized_hidden = _bounded(
        "hidden_size", hidden_size, 1, limits.max_hidden_size
    )
    shapes = tuple(
        (record.segment_count, record.slice_tokens, normalized_hidden)
        for record in records
    )
    contract_digest = _contract_digest(
        nonce=normalized_nonce,
        wave_digest=normalized_wave_digest,
        topology_digest=normalized_topology_digest,
        chunking=chunking,
        codec_id=normalized_codec,
        hidden_size=normalized_hidden,
        records=records,
        shapes=shapes,
    )
    manifest = PrefixSegmentStreamManifest(
        nonce=normalized_nonce,
        wave_digest=normalized_wave_digest,
        topology_digest=normalized_topology_digest,
        chunking=chunking,
        codec_id=normalized_codec,
        hidden_size=normalized_hidden,
        records=records,
        shapes=shapes,
        contract_digest=contract_digest,
    )
    validate_prefix_segment_stream_manifest(manifest, limits=limits)
    return manifest


def validate_prefix_segment_stream_manifest(
    manifest: PrefixSegmentStreamManifest,
    *,
    limits: PrefixSegmentStreamLimits = DEFAULT_LIMITS,
) -> None:
    """Recompute the complete token-free contract; mutations fail closed."""

    if not isinstance(limits, PrefixSegmentStreamLimits):
        raise PrefixSegmentCodecError("limits must be PrefixSegmentStreamLimits")
    if not isinstance(manifest, PrefixSegmentStreamManifest):
        raise PrefixSegmentCodecError(
            "manifest must be PrefixSegmentStreamManifest"
        )
    nonce = _bounded("nonce", manifest.nonce, 0, UINT64_MAX)
    wave_digest = _digest("wave_digest", manifest.wave_digest)
    topology_digest = _digest("topology_digest", manifest.topology_digest)
    if not isinstance(manifest.chunking, PrefixWaveChunking):
        raise PrefixSegmentCodecError("manifest chunking must be PrefixWaveChunking")
    codec_id = _bounded("codec_id", manifest.codec_id, 0, UINT16_MAX)
    hidden_size = _bounded(
        "hidden_size", manifest.hidden_size, 1, limits.max_hidden_size
    )
    _validate_records_and_chunking(
        manifest.records,
        chunking=manifest.chunking,
        limits=limits,
    )
    record_total = len(manifest.records)
    if not isinstance(manifest.shapes, tuple) or len(manifest.shapes) != record_total:
        raise PrefixSegmentCodecError("manifest shapes do not cover every record")
    for record_index, (record, shape) in enumerate(
        zip(manifest.records, manifest.shapes)
    ):
        expected_shape = (record.segment_count, record.slice_tokens, hidden_size)
        if not isinstance(shape, tuple) or shape != expected_shape:
            raise PrefixSegmentCodecError(
                f"record {record_index} shape is not canonical {expected_shape}"
            )
        for dimension in shape:
            _bounded("shape dimension", dimension, 1, UINT32_MAX)
    structural_descriptor = _encode_arena_structural_descriptor(
        manifest.chunking, manifest.records
    )
    expected_wave_digest = _wave_digest(
        topology_digest=topology_digest,
        structural_descriptor=structural_descriptor,
    )
    if not hmac.compare_digest(wave_digest, expected_wave_digest):
        raise PrefixSegmentCodecError(
            "manifest records, topology_digest, or chunking do not match wave_digest"
        )
    expected_digest = _contract_digest(
        nonce=nonce,
        wave_digest=wave_digest,
        topology_digest=topology_digest,
        chunking=manifest.chunking,
        codec_id=codec_id,
        hidden_size=hidden_size,
        records=manifest.records,
        shapes=manifest.shapes,
    )
    if not isinstance(manifest.contract_digest, bytes) or len(manifest.contract_digest) != 32:
        raise PrefixSegmentCodecError(
            "contract_digest must be immutable 32-byte data"
        )
    if not hmac.compare_digest(manifest.contract_digest, expected_digest):
        raise PrefixSegmentCodecError("manifest contract digest is stale")


def encode_prefix_segment_frame(
    manifest: PrefixSegmentStreamManifest,
    record: PrefixCutThroughRecord,
    payload: bytes | bytearray | memoryview,
    *,
    limits: PrefixSegmentStreamLimits = DEFAULT_LIMITS,
) -> bytes:
    """Encode one canonical record and an owned payload snapshot.

    The caller supplies the actual record object so accidentally pairing a
    payload with another record is rejected before any bytes are emitted.
    """

    validate_prefix_segment_stream_manifest(manifest, limits=limits)
    return _encode_prefix_segment_frame_validated(
        manifest, record, payload, limits=limits
    )


def _encode_prefix_segment_frame_validated(
    manifest: PrefixSegmentStreamManifest,
    record: PrefixCutThroughRecord,
    payload: bytes | bytearray | memoryview,
    *,
    limits: PrefixSegmentStreamLimits,
) -> bytes:
    if not isinstance(record, PrefixCutThroughRecord):
        raise PrefixSegmentCodecError("record must be PrefixCutThroughRecord")
    record_index = _bounded(
        "record_index", record.record_id, 0, manifest.record_total - 1
    )
    if record != manifest.records[record_index]:
        raise PrefixSegmentCodecError(
            "record does not exactly match its sealed manifest entry"
        )
    owned_payload = _owned_octets(
        "payload", payload, maximum=limits.max_payload_bytes, allow_empty=False
    )
    shape = manifest.shapes[record_index]
    variable = _variable_descriptor(shape=shape, record=record)
    payload_digest = _payload_digest(owned_payload)
    prefix = _binding_prefix(
        manifest=manifest,
        record=record,
        shape=shape,
        payload_length=len(owned_payload),
    )
    binding_digest = hashlib.sha256(
        _FRAME_BINDING_DOMAIN + prefix + variable + payload_digest
    ).digest()
    header = _FIXED_HEADER.pack(
        _MAGIC,
        VERSION,
        _FLAGS,
        manifest.codec_id,
        manifest.nonce,
        manifest.wave_digest,
        manifest.contract_digest,
        record_index,
        manifest.record_total,
        record.frontier_index,
        record.group_id,
        record.slice_offset,
        record.slice_tokens,
        record.segment_count,
        record.node_count,
        len(shape),
        _RESERVED,
        len(owned_payload),
        payload_digest,
        binding_digest,
    )
    packet = header + variable + owned_payload
    if len(packet) > limits.max_packet_bytes:
        raise PrefixSegmentCodecError(
            f"PREFIX_SEGMENT packet has {len(packet)} bytes, limit is "
            f"{limits.max_packet_bytes}"
        )
    return packet


def decode_prefix_segment_frame(
    packet: bytes | bytearray | memoryview,
    manifest: PrefixSegmentStreamManifest,
    *,
    limits: PrefixSegmentStreamLimits = DEFAULT_LIMITS,
) -> PrefixSegmentFrame:
    """Decode, own and validate exactly one canonical frame."""

    validate_prefix_segment_stream_manifest(manifest, limits=limits)
    return _decode_prefix_segment_frame_validated(
        packet, manifest, limits=limits
    )


def _decode_prefix_segment_frame_validated(
    packet: bytes | bytearray | memoryview,
    manifest: PrefixSegmentStreamManifest,
    *,
    limits: PrefixSegmentStreamLimits,
) -> PrefixSegmentFrame:
    owned_packet = _owned_octets(
        "packet", packet, maximum=limits.max_packet_bytes, allow_empty=False
    )
    if len(owned_packet) < _FIXED_HEADER.size:
        raise PrefixSegmentCodecError("PREFIX_SEGMENT packet is truncated")
    (
        magic,
        version,
        flags,
        codec_id,
        nonce,
        wave_digest,
        contract_digest,
        record_index,
        record_total,
        frontier_index,
        group_id,
        slice_offset,
        slice_tokens,
        segment_count,
        node_count,
        rank,
        reserved,
        payload_length,
        payload_digest,
        binding_digest,
    ) = _FIXED_HEADER.unpack_from(owned_packet)

    if magic != _MAGIC:
        raise PrefixSegmentCodecError("PREFIX_SEGMENT magic mismatch")
    if version != VERSION:
        raise PrefixSegmentCodecError(
            f"unsupported PREFIX_SEGMENT version {version}"
        )
    if flags != _FLAGS or reserved != _RESERVED:
        raise PrefixSegmentCodecError("PREFIX_SEGMENT reserved bits are non-zero")
    if codec_id != manifest.codec_id:
        raise PrefixSegmentCodecError("PREFIX_SEGMENT codec does not match the wave")
    if nonce != manifest.nonce:
        raise PrefixSegmentCodecError("PREFIX_SEGMENT nonce belongs to another wave")
    if not hmac.compare_digest(wave_digest, manifest.wave_digest):
        raise PrefixSegmentCodecError(
            "PREFIX_SEGMENT wave digest belongs to another wave"
        )
    if not hmac.compare_digest(contract_digest, manifest.contract_digest):
        raise PrefixSegmentCodecError(
            "PREFIX_SEGMENT contract digest belongs to another stream contract"
        )
    if record_total != manifest.record_total:
        raise PrefixSegmentCodecError("PREFIX_SEGMENT record total changed")
    if record_index >= record_total:
        raise PrefixSegmentCodecError("PREFIX_SEGMENT record index is out of bounds")
    expected_record = manifest.records[record_index]
    expected_shape = manifest.shapes[record_index]
    observed_metadata = (
        frontier_index,
        group_id,
        slice_offset,
        slice_tokens,
        segment_count,
        node_count,
    )
    expected_metadata = (
        expected_record.frontier_index,
        expected_record.group_id,
        expected_record.slice_offset,
        expected_record.slice_tokens,
        expected_record.segment_count,
        expected_record.node_count,
    )
    if observed_metadata != expected_metadata:
        raise PrefixSegmentCodecError(
            "PREFIX_SEGMENT structural record metadata changed"
        )
    if rank != len(expected_shape):
        raise PrefixSegmentCodecError("PREFIX_SEGMENT activation rank changed")
    if payload_length < 1 or payload_length > limits.max_payload_bytes:
        raise PrefixSegmentCodecError(
            "PREFIX_SEGMENT payload length is outside configured bounds"
        )

    variable_length = (
        rank * _U32.size
        + segment_count * _U32.size
        + node_count * _U32.size
    )
    expected_packet_length = _FIXED_HEADER.size + variable_length + payload_length
    if len(owned_packet) > expected_packet_length:
        raise PrefixSegmentCodecError("PREFIX_SEGMENT packet has trailing bytes")
    if len(owned_packet) < expected_packet_length:
        raise PrefixSegmentCodecError("PREFIX_SEGMENT packet is truncated")

    cursor = _FIXED_HEADER.size
    shape = _unpack_u32_tuple(owned_packet, cursor, rank)
    cursor += rank * _U32.size
    segment_ids = _unpack_u32_tuple(owned_packet, cursor, segment_count)
    cursor += segment_count * _U32.size
    canonical_node_ids = _unpack_u32_tuple(owned_packet, cursor, node_count)
    cursor += node_count * _U32.size
    payload = owned_packet[cursor : cursor + payload_length]

    if shape != expected_shape:
        raise PrefixSegmentCodecError("PREFIX_SEGMENT activation shape changed")
    if segment_ids != expected_record.segment_ids:
        raise PrefixSegmentCodecError("PREFIX_SEGMENT segment ids changed")
    if canonical_node_ids != expected_record.canonical_node_ids:
        raise PrefixSegmentCodecError("PREFIX_SEGMENT canonical node ids changed")
    observed_payload_digest = _payload_digest(payload)
    if not hmac.compare_digest(payload_digest, observed_payload_digest):
        raise PrefixSegmentCodecError("PREFIX_SEGMENT payload SHA-256 mismatch")

    variable = owned_packet[_FIXED_HEADER.size:cursor]
    prefix = _binding_prefix(
        manifest=manifest,
        record=expected_record,
        shape=shape,
        payload_length=payload_length,
    )
    observed_binding = hashlib.sha256(
        _FRAME_BINDING_DOMAIN + prefix + variable + payload_digest
    ).digest()
    if not hmac.compare_digest(binding_digest, observed_binding):
        raise PrefixSegmentCodecError("PREFIX_SEGMENT frame binding SHA-256 mismatch")

    return PrefixSegmentFrame(
        nonce=nonce,
        wave_digest=wave_digest,
        contract_digest=contract_digest,
        record_index=record_index,
        record_total=record_total,
        record=expected_record,
        codec_id=codec_id,
        shape=shape,
        payload=payload,
        payload_digest=payload_digest,
        binding_digest=binding_digest,
    )


class PrefixSegmentStreamEncoder:
    """Reusable O(frame-size) encoder after one complete manifest preflight."""

    def __init__(
        self,
        manifest: PrefixSegmentStreamManifest,
        *,
        limits: PrefixSegmentStreamLimits = DEFAULT_LIMITS,
    ) -> None:
        validate_prefix_segment_stream_manifest(manifest, limits=limits)
        self._manifest = manifest
        self._limits = limits

    @property
    def manifest(self) -> PrefixSegmentStreamManifest:
        return self._manifest

    def encode(
        self,
        record: PrefixCutThroughRecord,
        payload: bytes | bytearray | memoryview,
    ) -> bytes:
        return _encode_prefix_segment_frame_validated(
            self._manifest, record, payload, limits=self._limits
        )


class PrefixSegmentStreamGuard:
    """Single-use ordered receiver with one explicit mutation callback.

    ``accept`` never calls ``mutation`` until the complete packet, manifest and
    order have passed validation.  It sets ``mutation_started`` immediately
    before entering the callback.  Callback exceptions, including
    ``BaseException`` subclasses, are conservatively fatal because the guard
    cannot prove that the backend remained unchanged.
    """

    def __init__(
        self,
        manifest: PrefixSegmentStreamManifest,
        *,
        limits: PrefixSegmentStreamLimits = DEFAULT_LIMITS,
    ) -> None:
        validate_prefix_segment_stream_manifest(manifest, limits=limits)
        self._manifest = manifest
        self._limits = limits
        self._state = PrefixSegmentStreamState.OPEN
        self._next_record_index = 0
        self._completed_record_count = 0
        self._mutation_started = False
        self._fatal_reason: str | None = None
        self._lock = threading.Lock()

    @property
    def manifest(self) -> PrefixSegmentStreamManifest:
        return self._manifest

    def snapshot(self) -> PrefixSegmentStreamSnapshot:
        with self._lock:
            return PrefixSegmentStreamSnapshot(
                state=self._state,
                next_record_index=self._next_record_index,
                record_total=self._manifest.record_total,
                completed_record_count=self._completed_record_count,
                mutation_started=self._mutation_started,
                fatal_reason=self._fatal_reason,
            )

    def accept(
        self,
        packet: bytes | bytearray | memoryview,
        mutation: Callable[[PrefixSegmentFrame], T],
    ) -> PrefixSegmentApplyResult[T]:
        """Validate the next frame, cross the mutation boundary, and advance."""

        received_index: int | None = None
        with self._lock:
            self._ensure_open_locked()
            if not callable(mutation):
                self._fail_locked(
                    "PREFIX_SEGMENT mutation boundary must be callable",
                    phase="preflight",
                    received_record_index=None,
                )
                raise AssertionError("unreachable")
            try:
                frame = _decode_prefix_segment_frame_validated(
                    packet, self._manifest, limits=self._limits
                )
                received_index = frame.record_index
            except PrefixSegmentCodecError as error:
                self._fail_locked(
                    f"PREFIX_SEGMENT validation failed: {error}",
                    phase="validation",
                    received_record_index=received_index,
                    cause=error,
                )
                raise AssertionError("unreachable")
            except BaseException as error:
                self._fail_locked(
                    f"PREFIX_SEGMENT validation failed unexpectedly: {error}",
                    phase="validation",
                    received_record_index=received_index,
                    cause=error,
                )
                raise AssertionError("unreachable")

            if frame.record_index != self._next_record_index:
                relation = (
                    "duplicate/replayed"
                    if frame.record_index < self._next_record_index
                    else "out-of-order/gapped"
                )
                self._fail_locked(
                    f"{relation} PREFIX_SEGMENT record {frame.record_index}; "
                    f"expected {self._next_record_index}",
                    phase="order",
                    received_record_index=frame.record_index,
                )
                raise AssertionError("unreachable")

            # From this assignment onward the callback may have changed KV even
            # if it never returns.  Never permit same-route fallback again.
            self._mutation_started = True
            self._state = PrefixSegmentStreamState.MUTATING

        try:
            mutation_result = mutation(frame)
        except BaseException as error:
            with self._lock:
                self._state = PrefixSegmentStreamState.FATAL
                self._fatal_reason = (
                    f"PREFIX_SEGMENT mutation callback failed at record "
                    f"{frame.record_index}: {error}"
                )
                failure = self._error_locked(
                    self._fatal_reason,
                    route_fatal=True,
                    phase="mutation",
                    received_record_index=frame.record_index,
                )
            raise failure from error

        with self._lock:
            if self._state is not PrefixSegmentStreamState.MUTATING:
                # A concurrent or re-entrant caller observed the in-flight
                # state.  The callback already ran, so the route is poisoned.
                self._state = PrefixSegmentStreamState.FATAL
                if self._fatal_reason is None:
                    self._fatal_reason = (
                        "PREFIX_SEGMENT stream state changed during mutation"
                    )
                raise self._error_locked(
                    self._fatal_reason,
                    route_fatal=True,
                    phase="finalize",
                    received_record_index=frame.record_index,
                )
            self._completed_record_count += 1
            self._next_record_index += 1
            complete = self._next_record_index == self._manifest.record_total
            self._state = (
                PrefixSegmentStreamState.COMPLETE
                if complete
                else PrefixSegmentStreamState.OPEN
            )
            return PrefixSegmentApplyResult(
                frame=frame,
                mutation_result=mutation_result,
                completed_record_count=self._completed_record_count,
                complete=complete,
            )

    def _ensure_open_locked(self) -> None:
        if self._state is PrefixSegmentStreamState.OPEN:
            return
        route_fatal = self._mutation_started or self._state is PrefixSegmentStreamState.FATAL
        message = f"PREFIX_SEGMENT stream is already {self._state.value}"
        if route_fatal:
            self._state = PrefixSegmentStreamState.FATAL
            self._fatal_reason = self._fatal_reason or message
        raise self._error_locked(
            message,
            route_fatal=route_fatal,
            phase="state",
            received_record_index=None,
        )

    def _fail_locked(
        self,
        message: str,
        *,
        phase: str,
        received_record_index: int | None,
        cause: BaseException | None = None,
    ) -> None:
        route_fatal = self._mutation_started
        self._state = (
            PrefixSegmentStreamState.FATAL
            if route_fatal
            else PrefixSegmentStreamState.ABORTED
        )
        if route_fatal:
            self._fatal_reason = message
        failure = self._error_locked(
            message,
            route_fatal=route_fatal,
            phase=phase,
            received_record_index=received_record_index,
        )
        if cause is None:
            raise failure
        raise failure from cause

    def _error_locked(
        self,
        message: str,
        *,
        route_fatal: bool,
        phase: str,
        received_record_index: int | None,
    ) -> PrefixSegmentStreamError:
        return PrefixSegmentStreamError(
            message,
            route_fatal=route_fatal,
            phase=phase,
            expected_record_index=self._next_record_index,
            received_record_index=received_record_index,
            completed_record_count=self._completed_record_count,
        )


def _validate_arena_for_stream(
    arena: PrefixTokenArena,
    *,
    limits: PrefixSegmentStreamLimits,
) -> None:
    if not isinstance(arena, PrefixTokenArena):
        raise PrefixSegmentCodecError("arena must be PrefixTokenArena")
    nonce = _bounded("arena nonce", arena.nonce, 0, UINT64_MAX)
    topology_digest = _digest("arena topology_digest", arena.topology_digest)
    wave_digest = _digest("arena wave_digest", arena.wave_digest)
    if not isinstance(arena.chunking, PrefixWaveChunking):
        raise PrefixSegmentCodecError("arena chunking must be PrefixWaveChunking")
    _validate_records_and_chunking(
        arena.records,
        chunking=arena.chunking,
        limits=limits,
    )

    canonical_descriptor = _encode_arena_structural_descriptor(
        arena.chunking, arena.records
    )
    if not isinstance(arena.structural_descriptor_bytes, bytes):
        raise PrefixSegmentCodecError(
            "arena structural descriptor must be immutable bytes"
        )
    if arena.structural_descriptor_bytes != canonical_descriptor:
        raise PrefixSegmentCodecError(
            "arena structural descriptor does not match its token-free records"
        )
    expected_wave_digest = _wave_digest(
        topology_digest=topology_digest,
        structural_descriptor=canonical_descriptor,
    )
    if not hmac.compare_digest(wave_digest, expected_wave_digest):
        raise PrefixSegmentCodecError("arena wave digest is stale")
    # ``nonce`` is intentionally checked independently even though the sealed
    # topology and therefore wave digest also bind it.
    del nonce


def _validate_records_and_chunking(
    records: object,
    *,
    chunking: PrefixWaveChunking,
    limits: PrefixSegmentStreamLimits,
) -> None:
    if not isinstance(records, tuple):
        raise PrefixSegmentCodecError("records must be an immutable tuple")
    _bounded("record_total", len(records), 1, limits.max_records)
    max_slice_tokens = _bounded(
        "max_slice_tokens", chunking.max_slice_tokens, 1, UINT32_MAX
    )
    max_segments_per_record = _bounded(
        "max_segments_per_record",
        chunking.max_segments_per_record,
        1,
        UINT32_MAX,
    )
    max_nodes_per_record = _bounded(
        "max_nodes_per_record", chunking.max_nodes_per_record, 1, UINT32_MAX
    )
    observed_nodes: list[int] = []
    observed_segments: set[int] = set()
    previous_group = -1
    previous_frontier = -1
    previous_slice_offset = -1
    for record_index, record in enumerate(records):
        _validate_record(record, expected_id=record_index)
        if record.slice_tokens > max_slice_tokens:
            raise PrefixSegmentCodecError(
                f"record {record_index} exceeds chunking max_slice_tokens"
            )
        if record.segment_count > max_segments_per_record:
            raise PrefixSegmentCodecError(
                f"record {record_index} exceeds chunking max_segments_per_record"
            )
        if record.node_count > max_nodes_per_record:
            raise PrefixSegmentCodecError(
                f"record {record_index} exceeds chunking max_nodes_per_record"
            )
        if record.group_id < previous_group or record.frontier_index < previous_frontier:
            raise PrefixSegmentCodecError(
                "record group/frontier order is not canonical"
            )
        if record.group_id != previous_group:
            if record.group_id != previous_group + 1:
                raise PrefixSegmentCodecError("record group ids are not contiguous")
            previous_group = record.group_id
            previous_slice_offset = -1
        elif record.frontier_index != previous_frontier:
            raise PrefixSegmentCodecError(
                "records in one compatible group changed frontier"
            )
        if record.slice_offset < previous_slice_offset:
            raise PrefixSegmentCodecError("record slice order regressed")
        previous_frontier = record.frontier_index
        previous_slice_offset = record.slice_offset
        observed_nodes.extend(record.canonical_node_ids)
        observed_segments.update(record.segment_ids)

    if sorted(observed_nodes) != list(range(1, len(observed_nodes) + 1)):
        raise PrefixSegmentCodecError(
            "records duplicate, omit, or reorder the canonical node id space"
        )
    if observed_segments != set(range(len(observed_segments))):
        raise PrefixSegmentCodecError(
            "records do not cover a contiguous canonical segment id space"
        )


def _validate_record(record: object, *, expected_id: int) -> None:
    if not isinstance(record, PrefixCutThroughRecord):
        raise PrefixSegmentCodecError(
            f"record {expected_id} must be PrefixCutThroughRecord"
        )
    record_id = _bounded("record_id", record.record_id, 0, UINT32_MAX)
    if record_id != expected_id:
        raise PrefixSegmentCodecError("record ids must be canonical and contiguous")
    _bounded("frontier_index", record.frontier_index, 0, UINT32_MAX)
    _bounded("group_id", record.group_id, 0, UINT32_MAX)
    _bounded("slice_offset", record.slice_offset, 0, UINT32_MAX)
    slice_tokens = _bounded("slice_tokens", record.slice_tokens, 1, UINT32_MAX)
    if not isinstance(record.segment_ids, tuple) or not isinstance(
        record.canonical_node_ids, tuple
    ):
        raise PrefixSegmentCodecError("record id vectors must be immutable tuples")
    segment_count = _bounded(
        "segment_count", len(record.segment_ids), 1, MAX_SEGMENTS
    )
    node_count = _bounded(
        "node_count", len(record.canonical_node_ids), 1, MAX_NODES - 1
    )
    if node_count != segment_count * slice_tokens:
        raise PrefixSegmentCodecError("record node matrix is not rectangular")
    for segment_id in record.segment_ids:
        _bounded("segment_id", segment_id, 0, UINT32_MAX)
    for node_id in record.canonical_node_ids:
        _bounded("canonical_node_id", node_id, 1, UINT32_MAX)


def _encode_arena_structural_descriptor(
    chunking: PrefixWaveChunking,
    records: tuple[PrefixCutThroughRecord, ...],
) -> bytes:
    max_slice_tokens = _bounded(
        "max_slice_tokens", chunking.max_slice_tokens, 1, UINT32_MAX
    )
    max_segments = _bounded(
        "max_segments_per_record",
        chunking.max_segments_per_record,
        1,
        UINT32_MAX,
    )
    max_nodes = _bounded(
        "max_nodes_per_record", chunking.max_nodes_per_record, 1, UINT32_MAX
    )
    parts = [
        _ARENA_DESCRIPTOR_HEADER.pack(
            _ARENA_DESCRIPTOR_MAGIC,
            VERSION,
            max_slice_tokens,
            max_segments,
            max_nodes,
            len(records),
        )
    ]
    for record_index, record in enumerate(records):
        _validate_record(record, expected_id=record_index)
        parts.append(
            _ARENA_RECORD_HEADER.pack(
                record.record_id,
                record.frontier_index,
                record.group_id,
                record.slice_offset,
                record.slice_tokens,
                record.segment_count,
                record.node_count,
            )
        )
        parts.extend(_U32.pack(value) for value in record.segment_ids)
        parts.extend(_U32.pack(value) for value in record.canonical_node_ids)
    return b"".join(parts)


def _contract_digest(
    *,
    nonce: int,
    wave_digest: bytes,
    topology_digest: bytes,
    chunking: PrefixWaveChunking,
    codec_id: int,
    hidden_size: int,
    records: tuple[PrefixCutThroughRecord, ...],
    shapes: tuple[tuple[int, ...], ...],
) -> bytes:
    parts = [
        _CONTRACT_DIGEST_DOMAIN,
        _U64.pack(nonce),
        wave_digest,
        topology_digest,
        _U32.pack(chunking.max_slice_tokens),
        _U32.pack(chunking.max_segments_per_record),
        _U32.pack(chunking.max_nodes_per_record),
        _U16.pack(codec_id),
        _U32.pack(hidden_size),
        _U32.pack(len(records)),
    ]
    for record, shape in zip(records, shapes):
        parts.append(
            _ARENA_RECORD_HEADER.pack(
                record.record_id,
                record.frontier_index,
                record.group_id,
                record.slice_offset,
                record.slice_tokens,
                record.segment_count,
                record.node_count,
            )
        )
        parts.append(_U16.pack(len(shape)))
        parts.extend(_U32.pack(dimension) for dimension in shape)
        parts.extend(_U32.pack(value) for value in record.segment_ids)
        parts.extend(_U32.pack(value) for value in record.canonical_node_ids)
    return hashlib.sha256(b"".join(parts)).digest()


def _wave_digest(
    *,
    topology_digest: bytes,
    structural_descriptor: bytes,
) -> bytes:
    return hashlib.sha256(
        _ARENA_WAVE_DIGEST_DOMAIN
        + topology_digest
        + _U64.pack(len(structural_descriptor))
        + structural_descriptor
    ).digest()


def _binding_prefix(
    *,
    manifest: PrefixSegmentStreamManifest,
    record: PrefixCutThroughRecord,
    shape: tuple[int, ...],
    payload_length: int,
) -> bytes:
    return _BINDING_PREFIX.pack(
        _MAGIC,
        VERSION,
        _FLAGS,
        manifest.codec_id,
        manifest.nonce,
        manifest.wave_digest,
        manifest.contract_digest,
        record.record_id,
        manifest.record_total,
        record.frontier_index,
        record.group_id,
        record.slice_offset,
        record.slice_tokens,
        record.segment_count,
        record.node_count,
        len(shape),
        _RESERVED,
        payload_length,
    )


def _variable_descriptor(
    *,
    shape: tuple[int, ...],
    record: PrefixCutThroughRecord,
) -> bytes:
    parts = [_U32.pack(dimension) for dimension in shape]
    parts.extend(_U32.pack(segment_id) for segment_id in record.segment_ids)
    parts.extend(_U32.pack(node_id) for node_id in record.canonical_node_ids)
    return b"".join(parts)


def _payload_digest(payload: bytes) -> bytes:
    return hashlib.sha256(
        _PAYLOAD_DIGEST_DOMAIN + _U64.pack(len(payload)) + payload
    ).digest()


def _owned_octets(
    name: str,
    value: object,
    *,
    maximum: int,
    allow_empty: bool,
) -> bytes:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise PrefixSegmentCodecError(f"{name} must be a bytes-like value")
    try:
        view = memoryview(value)
    except (TypeError, ValueError) as error:
        raise PrefixSegmentCodecError(f"{name} is not a readable buffer") from error
    if view.ndim != 1 or not view.c_contiguous:
        raise PrefixSegmentCodecError(f"{name} must be a contiguous one-dimensional buffer")
    try:
        octets = view.cast("B")
    except (TypeError, ValueError) as error:
        raise PrefixSegmentCodecError(f"{name} cannot be viewed as octets") from error
    length = octets.nbytes
    if (not allow_empty and length == 0) or length > maximum:
        qualifier = "non-empty and " if not allow_empty else ""
        raise PrefixSegmentCodecError(
            f"{name} must be {qualifier}at most {maximum} bytes"
        )
    # Always own a snapshot.  The source bytearray/tensor staging buffer may be
    # reused as soon as the caller returns.
    return octets.tobytes()


def _unpack_u32_tuple(data: bytes, offset: int, count: int) -> tuple[int, ...]:
    end = offset + count * _U32.size
    return tuple(value[0] for value in struct.iter_unpack("<I", data[offset:end]))


__all__ = [
    "DEFAULT_LIMITS",
    "DEFAULT_MAX_HIDDEN_SIZE",
    "DEFAULT_MAX_PACKET_BYTES",
    "DEFAULT_MAX_PAYLOAD_BYTES",
    "PrefixSegmentApplyResult",
    "PrefixSegmentCodecError",
    "PrefixSegmentFrame",
    "PrefixSegmentStreamError",
    "PrefixSegmentStreamEncoder",
    "PrefixSegmentStreamGuard",
    "PrefixSegmentStreamLimits",
    "PrefixSegmentStreamManifest",
    "PrefixSegmentStreamSnapshot",
    "PrefixSegmentStreamState",
    "VERSION",
    "build_prefix_segment_stream_manifest",
    "build_prefix_segment_stream_manifest_from_records",
    "decode_prefix_segment_frame",
    "encode_prefix_segment_frame",
    "validate_prefix_segment_stream_manifest",
]
