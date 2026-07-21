"""Canonical binary packing for one exact physical tree wave.

The live GDLP protocol intentionally does not import this module yet.  It is a
stand-alone codec used to prove the representation, bounds and compression
policy before a future wire-version change.

The format is deliberately narrow:

* descriptors are sorted by strictly increasing request id;
* every descriptor owns one non-empty, contiguous slab;
* slab offsets are canonical (no holes, aliases or overlaps);
* all slabs have the same bytes-per-token shape;
* CRC32 protects the table, logical arena and stored arena independently;
* SHA-256 binds the complete header, table and *logical* arena;
* compressed representations are only selected after a measured break-even
  calculation that includes compression, decompression and transfer time.

Uncompressed packets backed by immutable ``bytes`` decode to memoryview slices
without copying their slabs.  Mutable or ambiguously-backed buffers are copied
before validation so a caller cannot mutate data after its digest was checked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
import hashlib
import math
from numbers import Integral, Real
import statistics
import struct
import time
from typing import Sequence
import zlib


MAGIC = b"PTWV"
VERSION = 1

# Offsets and lengths are uint32 on purpose.  A single exact tree wave larger
# than 256 MiB is not acceptable for the intended 4 GB devices even though the
# field could address more.  Raising these hard bounds requires a format audit.
UINT8_MAX = (1 << 8) - 1
UINT16_MAX = (1 << 16) - 1
UINT32_MAX = (1 << 32) - 1
UINT64_MAX = (1 << 64) - 1
MAX_PACKED_ENTRIES = 64
MAX_PACKED_SLAB_BYTES = 64 * 1024 * 1024
MAX_PACKED_LOGICAL_BYTES = 256 * 1024 * 1024
MAX_PACKED_TOKEN_COUNT = 4096
PACKET_DEFLATE_LEVEL = 1

# Header fields:
# magic, version, compression, header bytes, entry count, descriptor bytes,
# element width, reserved8, flags, table bytes, logical bytes, stored bytes,
# table CRC, logical CRC, stored CRC, reserved32, SHA-256.
_HEADER = struct.Struct("<4sBBHIIBBHQQQIIII32s")
# request id, step, token count, logical offset, logical length.
_DESCRIPTOR = struct.Struct("<QIIII")
_PLANE_LENGTH = struct.Struct("<I")
HEADER_BYTES = _HEADER.size
DESCRIPTOR_BYTES = _DESCRIPTOR.size
_DIGEST_DOMAIN = b"GDLP2/PTWV/canonical-v1\0"


class PackedTreeWaveError(ValueError):
    """The packet or one of its inputs violates the sealed format."""


class CompressionMode(IntEnum):
    NONE = 0
    RAW_DEFLATE = 1
    BYTE_PLANE_DEFLATE = 2


@dataclass(frozen=True, slots=True)
class PackedTreeWaveLimits:
    max_entries: int = MAX_PACKED_ENTRIES
    max_slab_bytes: int = MAX_PACKED_SLAB_BYTES
    max_logical_bytes: int = MAX_PACKED_LOGICAL_BYTES
    max_token_count: int = MAX_PACKED_TOKEN_COUNT

    def __post_init__(self) -> None:
        _bounded_int(
            "max_entries", self.max_entries, minimum=1, maximum=MAX_PACKED_ENTRIES
        )
        _bounded_int(
            "max_slab_bytes",
            self.max_slab_bytes,
            minimum=1,
            maximum=MAX_PACKED_SLAB_BYTES,
        )
        _bounded_int(
            "max_logical_bytes",
            self.max_logical_bytes,
            minimum=1,
            maximum=MAX_PACKED_LOGICAL_BYTES,
        )
        _bounded_int(
            "max_token_count",
            self.max_token_count,
            minimum=1,
            maximum=MAX_PACKED_TOKEN_COUNT,
        )
        if self.max_slab_bytes > self.max_logical_bytes:
            raise ValueError("max_slab_bytes cannot exceed max_logical_bytes")


@dataclass(frozen=True, slots=True)
class PackedTreeLeaf:
    request_id: int
    step: int
    token_count: int
    slab: bytes | bytearray | memoryview


@dataclass(frozen=True, slots=True)
class DecodedPackedTreeLeaf:
    request_id: int
    step: int
    token_count: int
    offset: int
    slab: memoryview


@dataclass(frozen=True, slots=True)
class DecodedPackedTreeWave:
    leaves: tuple[DecodedPackedTreeLeaf, ...]
    compression: CompressionMode
    logical_bytes: int
    stored_bytes: int
    digest: bytes
    zero_copy_slabs: bool
    copied_input: bool
    # Keep the packet/decompressed arena alive even if callers retain only the
    # decoded object.  Individual memoryviews also own their backing object.
    _owner: object = field(repr=False, compare=False)

    @property
    def digest_hex(self) -> str:
        return self.digest.hex()


@dataclass(frozen=True, slots=True)
class CompressionPolicy:
    """Local measurement policy used before compression may be selected."""

    bandwidth_mbps: float
    repeats: int = 3
    minimum_savings_seconds: float = 0.0
    modes: tuple[CompressionMode, ...] = (
        CompressionMode.RAW_DEFLATE,
        CompressionMode.BYTE_PLANE_DEFLATE,
    )

    def __post_init__(self) -> None:
        _finite_real("bandwidth_mbps", self.bandwidth_mbps, positive=True)
        _bounded_int("repeats", self.repeats, minimum=1, maximum=100)
        _finite_real(
            "minimum_savings_seconds", self.minimum_savings_seconds, positive=False
        )
        if not self.modes:
            raise ValueError("modes cannot be empty")
        normalized: list[CompressionMode] = []
        for value in self.modes:
            try:
                mode = CompressionMode(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"unknown compression mode {value!r}") from error
            if mode == CompressionMode.NONE:
                raise ValueError("NONE is the fallback and cannot be benchmarked")
            if mode in normalized:
                raise ValueError("compression modes cannot contain duplicates")
            normalized.append(mode)
        object.__setattr__(self, "modes", tuple(normalized))


@dataclass(frozen=True, slots=True)
class CompressionBreakEven:
    raw_bytes: int
    compressed_bytes: int
    compression_seconds: float
    decompression_seconds: float
    bandwidth_mbps: float
    raw_transfer_seconds: float
    compressed_transfer_seconds: float
    compressed_total_seconds: float
    net_savings_seconds: float
    beneficial: bool


@dataclass(frozen=True, slots=True)
class CompressionAssessment:
    mode: CompressionMode
    element_width: int
    payload: bytes = field(repr=False)
    break_even: CompressionBreakEven
    byte_exact: bool


@dataclass(frozen=True, slots=True)
class CompressionSelection:
    mode: CompressionMode
    element_width: int
    payload: bytes = field(repr=False)
    source_sha256: bytes
    assessments: tuple[CompressionAssessment, ...]

    @property
    def selected_assessment(self) -> CompressionAssessment | None:
        return next(
            (item for item in self.assessments if item.mode == self.mode), None
        )


@dataclass(frozen=True, slots=True)
class PackedTreeWaveEncoding:
    packet: bytes = field(repr=False)
    compression: CompressionSelection
    entry_count: int
    logical_bytes: int

    @property
    def stored_bytes(self) -> int:
        return len(self.compression.payload)


def _bounded_int(
    name: str,
    value: object,
    *,
    minimum: int = 0,
    maximum: int,
) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    normalized = int(value)
    if normalized < minimum or normalized > maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return normalized


def _finite_real(name: str, value: object, *, positive: bool) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0 or (
        positive and normalized <= 0
    ):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return normalized


DEFAULT_LIMITS = PackedTreeWaveLimits()


def _checked_add(name: str, left: int, right: int, *, maximum: int) -> int:
    if left < 0 or right < 0 or left > maximum - right:
        raise PackedTreeWaveError(f"{name} overflows its sealed range")
    return left + right


def _checked_mul(name: str, left: int, right: int, *, maximum: int) -> int:
    if left < 0 or right < 0 or (left and right > maximum // left):
        raise PackedTreeWaveError(f"{name} overflows its sealed range")
    return left * right


def _byte_view(value: object, name: str) -> memoryview:
    try:
        view = memoryview(value)
    except TypeError as error:
        raise TypeError(f"{name} must be bytes-like") from error
    if not view.c_contiguous:
        raise PackedTreeWaveError(f"{name} must be C-contiguous")
    try:
        return view.cast("B")
    except TypeError as error:
        raise PackedTreeWaveError(f"{name} cannot be viewed as bytes") from error


def _has_immutable_bytes_backing(value: object) -> bool:
    owner = value
    while isinstance(owner, memoryview):
        owner = owner.obj
    return isinstance(owner, bytes)


def _crc32(value: bytes | memoryview) -> int:
    return zlib.crc32(value) & UINT32_MAX


def _raw_deflate(value: bytes, level: int) -> bytes:
    compressor = zlib.compressobj(level, zlib.DEFLATED, -zlib.MAX_WBITS)
    return compressor.compress(value) + compressor.flush()


def _raw_inflate_bounded(
    value: bytes | memoryview, *, expected_bytes: int, maximum_bytes: int
) -> bytes:
    _bounded_int(
        "expected_bytes", expected_bytes, minimum=1, maximum=maximum_bytes
    )
    decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
    try:
        decoded = decompressor.decompress(value, expected_bytes + 1)
    except zlib.error as error:
        raise PackedTreeWaveError("invalid raw-deflate payload") from error
    if len(decoded) > expected_bytes or decompressor.unconsumed_tail:
        raise PackedTreeWaveError("raw-deflate payload exceeds declared output")
    if not decompressor.eof:
        raise PackedTreeWaveError("truncated raw-deflate payload")
    if decompressor.unused_data:
        raise PackedTreeWaveError("raw-deflate payload has trailing data")
    try:
        tail = decompressor.flush()
    except zlib.error as error:
        raise PackedTreeWaveError("invalid raw-deflate payload") from error
    decoded += tail
    if len(decoded) != expected_bytes:
        raise PackedTreeWaveError(
            "raw-deflate output length does not match the declaration"
        )
    return decoded


def compress_lossless(
    value: bytes | bytearray | memoryview,
    mode: CompressionMode,
    *,
    element_width: int = 0,
    level: int = 1,
) -> bytes:
    """Return one deterministic, byte-exact compression payload.

    This primitive does not decide whether the result should be used.  Packet
    encoding can only activate it through :func:`select_compression_for_bandwidth`.
    """

    try:
        normalized_mode = CompressionMode(mode)
    except (TypeError, ValueError) as error:
        raise ValueError(f"unknown compression mode {mode!r}") from error
    _bounded_int("level", level, minimum=0, maximum=9)
    raw = bytes(_byte_view(value, "value"))
    if not raw:
        raise ValueError("cannot compress an empty payload")
    if normalized_mode == CompressionMode.NONE:
        if element_width not in (0, 1):
            raise ValueError("NONE does not use an element width")
        return raw
    if normalized_mode == CompressionMode.RAW_DEFLATE:
        if element_width not in (0, 1):
            raise ValueError("RAW_DEFLATE does not use an element width")
        return _raw_deflate(raw, level)
    if normalized_mode != CompressionMode.BYTE_PLANE_DEFLATE:
        raise ValueError(f"unsupported compression mode {normalized_mode}")
    width = _bounded_int("element_width", element_width, minimum=2, maximum=4)
    if width not in (2, 4):
        raise ValueError("byte-plane compression supports 2- or 4-byte elements")
    if len(raw) % width:
        raise ValueError("payload length must be divisible by element_width")
    planes = tuple(_raw_deflate(raw[index::width], level) for index in range(width))
    return b"".join(_PLANE_LENGTH.pack(len(plane)) for plane in planes) + b"".join(
        planes
    )


def decompress_lossless(
    value: bytes | bytearray | memoryview,
    mode: CompressionMode,
    *,
    expected_bytes: int,
    element_width: int = 0,
    maximum_bytes: int = MAX_PACKED_LOGICAL_BYTES,
) -> bytes:
    """Bounded inverse of :func:`compress_lossless`."""

    try:
        normalized_mode = CompressionMode(mode)
    except (TypeError, ValueError) as error:
        raise ValueError(f"unknown compression mode {mode!r}") from error
    maximum = _bounded_int(
        "maximum_bytes",
        maximum_bytes,
        minimum=1,
        maximum=MAX_PACKED_LOGICAL_BYTES,
    )
    expected = _bounded_int(
        "expected_bytes", expected_bytes, minimum=1, maximum=maximum
    )
    encoded = _byte_view(value, "value")
    if normalized_mode == CompressionMode.NONE:
        if element_width not in (0, 1):
            raise ValueError("NONE does not use an element width")
        if encoded.nbytes != expected:
            raise PackedTreeWaveError("raw payload length does not match declaration")
        return bytes(encoded)
    if normalized_mode == CompressionMode.RAW_DEFLATE:
        if element_width not in (0, 1):
            raise ValueError("RAW_DEFLATE does not use an element width")
        return _raw_inflate_bounded(
            encoded, expected_bytes=expected, maximum_bytes=maximum
        )
    if normalized_mode != CompressionMode.BYTE_PLANE_DEFLATE:
        raise ValueError(f"unsupported compression mode {normalized_mode}")
    width = _bounded_int("element_width", element_width, minimum=2, maximum=4)
    if width not in (2, 4):
        raise ValueError("byte-plane compression supports 2- or 4-byte elements")
    if expected % width:
        raise PackedTreeWaveError(
            "declared byte-plane output is not divisible by element width"
        )
    lengths_bytes = _checked_mul(
        "byte-plane length table", width, _PLANE_LENGTH.size, maximum=UINT32_MAX
    )
    if encoded.nbytes < lengths_bytes:
        raise PackedTreeWaveError("truncated byte-plane length table")
    lengths = tuple(
        _PLANE_LENGTH.unpack_from(encoded, index * _PLANE_LENGTH.size)[0]
        for index in range(width)
    )
    if any(length == 0 for length in lengths):
        raise PackedTreeWaveError("byte-plane streams cannot be empty")
    cursor = lengths_bytes
    for length in lengths:
        cursor = _checked_add(
            "byte-plane payload", cursor, length, maximum=UINT32_MAX
        )
    if cursor != encoded.nbytes:
        qualifier = "truncated" if cursor > encoded.nbytes else "trailing"
        raise PackedTreeWaveError(f"{qualifier} byte-plane payload")
    plane_bytes = expected // width
    planes: list[bytes] = []
    cursor = lengths_bytes
    for length in lengths:
        end = cursor + length
        planes.append(
            _raw_inflate_bounded(
                encoded[cursor:end],
                expected_bytes=plane_bytes,
                maximum_bytes=maximum,
            )
        )
        cursor = end
    decoded = bytearray(expected)
    for index, plane in enumerate(planes):
        decoded[index::width] = plane
    return bytes(decoded)


def compression_break_even(
    *,
    raw_bytes: int,
    compressed_bytes: int,
    compression_seconds: float,
    decompression_seconds: float,
    bandwidth_mbps: float,
    minimum_savings_seconds: float = 0.0,
) -> CompressionBreakEven:
    """Compare total end-to-end seconds, not compression ratio alone.

    The fixed packed header and descriptor table are identical in both choices
    and therefore cancel.  ``raw_bytes`` and ``compressed_bytes`` describe only
    the arena whose transfer size changes.
    """

    raw_size = _bounded_int(
        "raw_bytes", raw_bytes, minimum=1, maximum=MAX_PACKED_LOGICAL_BYTES
    )
    compressed_size = _bounded_int(
        "compressed_bytes",
        compressed_bytes,
        minimum=1,
        # Candidate compression can expand incompressible data slightly.  It
        # will never be selected in that case, but it still needs an honest
        # break-even assessment instead of failing before the comparison.
        maximum=UINT32_MAX,
    )
    compress_s = _finite_real(
        "compression_seconds", compression_seconds, positive=False
    )
    decompress_s = _finite_real(
        "decompression_seconds", decompression_seconds, positive=False
    )
    bandwidth = _finite_real("bandwidth_mbps", bandwidth_mbps, positive=True)
    margin = _finite_real(
        "minimum_savings_seconds", minimum_savings_seconds, positive=False
    )
    bytes_per_second = bandwidth * 1_000_000.0 / 8.0
    raw_transfer = raw_size / bytes_per_second
    compressed_transfer = compressed_size / bytes_per_second
    compressed_total = compress_s + decompress_s + compressed_transfer
    savings = raw_transfer - compressed_total
    return CompressionBreakEven(
        raw_bytes=raw_size,
        compressed_bytes=compressed_size,
        compression_seconds=compress_s,
        decompression_seconds=decompress_s,
        bandwidth_mbps=bandwidth,
        raw_transfer_seconds=raw_transfer,
        compressed_transfer_seconds=compressed_transfer,
        compressed_total_seconds=compressed_total,
        net_savings_seconds=savings,
        beneficial=compressed_size < raw_size and savings > margin,
    )


def _measure_mode(
    raw: bytes,
    *,
    mode: CompressionMode,
    element_width: int,
    policy: CompressionPolicy,
) -> CompressionAssessment:
    compressed_samples: list[float] = []
    decompressed_samples: list[float] = []
    canonical_payload: bytes | None = None
    for _ in range(policy.repeats):
        started = time.perf_counter_ns()
        payload = compress_lossless(
            raw,
            mode,
            element_width=element_width if mode == CompressionMode.BYTE_PLANE_DEFLATE else 0,
            # Packet compression has one fixed level so a selected mode has a
            # deterministic representation on every encoder using this format.
            level=PACKET_DEFLATE_LEVEL,
        )
        compressed_at = time.perf_counter_ns()
        decoded = decompress_lossless(
            payload,
            mode,
            expected_bytes=len(raw),
            element_width=element_width if mode == CompressionMode.BYTE_PLANE_DEFLATE else 0,
            maximum_bytes=len(raw),
        )
        finished = time.perf_counter_ns()
        if decoded != raw:
            raise PackedTreeWaveError(f"{mode.name} failed byte-exact roundtrip")
        if canonical_payload is None:
            canonical_payload = payload
        elif payload != canonical_payload:
            raise PackedTreeWaveError(f"{mode.name} produced non-deterministic bytes")
        compressed_samples.append((compressed_at - started) / 1_000_000_000.0)
        decompressed_samples.append((finished - compressed_at) / 1_000_000_000.0)
    assert canonical_payload is not None
    break_even = compression_break_even(
        raw_bytes=len(raw),
        compressed_bytes=len(canonical_payload),
        compression_seconds=statistics.median(compressed_samples),
        decompression_seconds=statistics.median(decompressed_samples),
        bandwidth_mbps=policy.bandwidth_mbps,
        minimum_savings_seconds=policy.minimum_savings_seconds,
    )
    return CompressionAssessment(
        mode=mode,
        element_width=element_width if mode == CompressionMode.BYTE_PLANE_DEFLATE else 0,
        payload=canonical_payload,
        break_even=break_even,
        byte_exact=True,
    )


def select_compression_for_bandwidth(
    value: bytes | bytearray | memoryview,
    *,
    element_width: int,
    policy: CompressionPolicy,
) -> CompressionSelection:
    """Measure candidates and select only one with positive end-to-end savings."""

    if not isinstance(policy, CompressionPolicy):
        raise TypeError("policy must be CompressionPolicy")
    raw = bytes(_byte_view(value, "value"))
    if not raw:
        raise ValueError("cannot select compression for an empty payload")
    width = _bounded_int("element_width", element_width, minimum=1, maximum=8)
    assessments: list[CompressionAssessment] = []
    for mode in policy.modes:
        if mode == CompressionMode.BYTE_PLANE_DEFLATE:
            if width not in (2, 4) or len(raw) % width:
                continue
        assessments.append(
            _measure_mode(raw, mode=mode, element_width=width, policy=policy)
        )
    beneficial = [item for item in assessments if item.break_even.beneficial]
    selected = min(
        beneficial,
        key=lambda item: (item.break_even.compressed_total_seconds, int(item.mode)),
        default=None,
    )
    if selected is None:
        mode = CompressionMode.NONE
        payload = raw
        selected_width = 0
    else:
        mode = selected.mode
        payload = selected.payload
        selected_width = selected.element_width
    return CompressionSelection(
        mode=mode,
        element_width=selected_width,
        payload=payload,
        source_sha256=hashlib.sha256(raw).digest(),
        assessments=tuple(assessments),
    )


def _validated_entries(
    leaves: Sequence[PackedTreeLeaf], limits: PackedTreeWaveLimits
) -> tuple[tuple[int, int, int, bytes], ...]:
    if isinstance(leaves, (str, bytes, bytearray, memoryview)):
        raise TypeError("leaves must be a sequence of PackedTreeLeaf")
    materialized = tuple(leaves)
    if not materialized:
        raise PackedTreeWaveError("a packed tree wave needs at least one leaf")
    if len(materialized) > limits.max_entries:
        raise PackedTreeWaveError(
            f"entry count exceeds sealed limit {limits.max_entries}"
        )
    normalized: list[tuple[int, int, int, bytes]] = []
    seen_request_ids: set[int] = set()
    logical_total = 0
    for ordinal, leaf in enumerate(materialized):
        if not isinstance(leaf, PackedTreeLeaf):
            raise TypeError(f"leaves[{ordinal}] must be PackedTreeLeaf")
        request_id = _bounded_int(
            f"leaves[{ordinal}].request_id",
            leaf.request_id,
            minimum=1,
            maximum=UINT64_MAX,
        )
        if request_id in seen_request_ids:
            raise PackedTreeWaveError(f"duplicate request id {request_id}")
        seen_request_ids.add(request_id)
        step = _bounded_int(
            f"leaves[{ordinal}].step", leaf.step, maximum=UINT32_MAX
        )
        token_count = _bounded_int(
            f"leaves[{ordinal}].token_count",
            leaf.token_count,
            minimum=1,
            maximum=limits.max_token_count,
        )
        slab_view = _byte_view(leaf.slab, f"leaves[{ordinal}].slab")
        if slab_view.nbytes < 1 or slab_view.nbytes > limits.max_slab_bytes:
            raise PackedTreeWaveError(
                f"leaves[{ordinal}].slab exceeds sealed size bounds"
            )
        if slab_view.nbytes % token_count:
            raise PackedTreeWaveError(
                f"leaves[{ordinal}].slab is not divisible by token_count"
            )
        if slab_view.nbytes > limits.max_logical_bytes - logical_total:
            raise PackedTreeWaveError(
                f"logical slab arena exceeds sealed limit {limits.max_logical_bytes}"
            )
        logical_total += slab_view.nbytes
        normalized.append((request_id, step, token_count, bytes(slab_view)))
    normalized.sort(key=lambda item: item[0])
    bytes_per_token = len(normalized[0][3]) // normalized[0][2]
    if any(len(item[3]) // item[2] != bytes_per_token for item in normalized[1:]):
        raise PackedTreeWaveError("all slabs must have the same bytes per token")
    return tuple(normalized)


def _zero_digest_header(fields: Sequence[object]) -> bytes:
    values = list(fields)
    values[-1] = b"\0" * 32
    return _HEADER.pack(*values)


def pack_packed_tree_wave(
    leaves: Sequence[PackedTreeLeaf],
    *,
    limits: PackedTreeWaveLimits = DEFAULT_LIMITS,
    compression_policy: CompressionPolicy | None = None,
    element_width: int = 0,
) -> PackedTreeWaveEncoding:
    """Build one canonical packet and retain compression decision evidence."""

    if not isinstance(limits, PackedTreeWaveLimits):
        raise TypeError("limits must be PackedTreeWaveLimits")
    entries = _validated_entries(leaves, limits)
    table_parts: list[bytes] = []
    slab_parts: list[bytes] = []
    offset = 0
    for request_id, step, token_count, slab in entries:
        length = len(slab)
        offset_end = _checked_add(
            "logical slab arena", offset, length, maximum=UINT32_MAX
        )
        if offset_end > limits.max_logical_bytes:
            raise PackedTreeWaveError(
                f"logical slab arena exceeds sealed limit {limits.max_logical_bytes}"
            )
        table_parts.append(
            _DESCRIPTOR.pack(request_id, step, token_count, offset, length)
        )
        slab_parts.append(slab)
        offset = offset_end
    table = b"".join(table_parts)
    logical = b"".join(slab_parts)
    expected_table_bytes = _checked_mul(
        "descriptor table", len(entries), DESCRIPTOR_BYTES, maximum=UINT64_MAX
    )
    if len(table) != expected_table_bytes:
        raise AssertionError("internal descriptor table length mismatch")
    if compression_policy is None:
        if element_width not in (0, 1):
            raise ValueError("element_width is only used by compression policy")
        selection = CompressionSelection(
            mode=CompressionMode.NONE,
            element_width=0,
            payload=logical,
            source_sha256=hashlib.sha256(logical).digest(),
            assessments=(),
        )
    else:
        selection = select_compression_for_bandwidth(
            logical, element_width=element_width, policy=compression_policy
        )
    wire = selection.payload
    if selection.source_sha256 != hashlib.sha256(logical).digest():
        raise AssertionError("compression selection is not bound to logical arena")
    if selection.mode != CompressionMode.NONE:
        selected = selection.selected_assessment
        if selected is None or not selected.break_even.beneficial:
            raise AssertionError("compression cannot activate without break-even proof")
        if len(wire) >= len(logical):
            raise AssertionError("selected compression must reduce stored bytes")
    table_crc = _crc32(table)
    logical_crc = _crc32(logical)
    wire_crc = _crc32(wire)
    fields: list[object] = [
        MAGIC,
        VERSION,
        int(selection.mode),
        HEADER_BYTES,
        len(entries),
        DESCRIPTOR_BYTES,
        selection.element_width,
        0,
        0,
        len(table),
        len(logical),
        len(wire),
        table_crc,
        logical_crc,
        wire_crc,
        0,
        b"\0" * 32,
    ]
    header_without_digest = _HEADER.pack(*fields)
    digest_hasher = hashlib.sha256()
    digest_hasher.update(_DIGEST_DOMAIN)
    digest_hasher.update(header_without_digest)
    digest_hasher.update(table)
    digest_hasher.update(logical)
    digest_hasher.update(wire)
    digest = digest_hasher.digest()
    fields[-1] = digest
    packet = _HEADER.pack(*fields) + table + wire
    return PackedTreeWaveEncoding(
        packet=packet,
        compression=selection,
        entry_count=len(entries),
        logical_bytes=len(logical),
    )


def encode_packed_tree_wave(
    leaves: Sequence[PackedTreeLeaf],
    *,
    limits: PackedTreeWaveLimits = DEFAULT_LIMITS,
    compression_policy: CompressionPolicy | None = None,
    element_width: int = 0,
) -> bytes:
    return pack_packed_tree_wave(
        leaves,
        limits=limits,
        compression_policy=compression_policy,
        element_width=element_width,
    ).packet


def decode_packed_tree_wave(
    packet: bytes | bytearray | memoryview,
    *,
    limits: PackedTreeWaveLimits = DEFAULT_LIMITS,
) -> DecodedPackedTreeWave:
    """Validate one complete packet and expose safe slab memoryviews."""

    if not isinstance(limits, PackedTreeWaveLimits):
        raise TypeError("limits must be PackedTreeWaveLimits")
    supplied = _byte_view(packet, "packet")
    maximum_packet_bytes = (
        HEADER_BYTES
        + limits.max_entries * DESCRIPTOR_BYTES
        + limits.max_logical_bytes
    )
    if supplied.nbytes < HEADER_BYTES:
        raise PackedTreeWaveError("truncated packed-tree-wave header")
    if supplied.nbytes > maximum_packet_bytes:
        raise PackedTreeWaveError("packed-tree-wave packet exceeds sealed limits")
    immutable_backing = _has_immutable_bytes_backing(packet)
    copied_input = not immutable_backing
    owner: object
    if immutable_backing:
        raw = supplied
        owner = packet
    else:
        owner = bytes(supplied)
        raw = memoryview(owner)
    fields = list(_HEADER.unpack_from(raw, 0))
    (
        magic,
        version,
        compression_value,
        header_bytes,
        entry_count,
        descriptor_bytes,
        element_width,
        reserved8,
        flags,
        table_bytes,
        logical_bytes,
        wire_bytes,
        table_crc,
        logical_crc,
        wire_crc,
        reserved32,
        digest,
    ) = fields
    if magic != MAGIC:
        raise PackedTreeWaveError("invalid packed-tree-wave magic")
    if version != VERSION:
        raise PackedTreeWaveError(f"unsupported packed-tree-wave version {version}")
    try:
        compression = CompressionMode(compression_value)
    except ValueError as error:
        raise PackedTreeWaveError(
            f"unknown packed-tree-wave compression {compression_value}"
        ) from error
    if header_bytes != HEADER_BYTES or descriptor_bytes != DESCRIPTOR_BYTES:
        raise PackedTreeWaveError("non-canonical packed-tree-wave structure sizes")
    if reserved8 != 0 or flags != 0 or reserved32 != 0:
        raise PackedTreeWaveError("reserved packed-tree-wave fields must be zero")
    if entry_count < 1 or entry_count > limits.max_entries:
        raise PackedTreeWaveError("packed-tree-wave entry count exceeds limits")
    expected_table_bytes = _checked_mul(
        "descriptor table", entry_count, DESCRIPTOR_BYTES, maximum=UINT64_MAX
    )
    if table_bytes != expected_table_bytes:
        raise PackedTreeWaveError("descriptor table length is not canonical")
    if logical_bytes < 1 or logical_bytes > limits.max_logical_bytes:
        raise PackedTreeWaveError("logical slab arena exceeds limits")
    if wire_bytes < 1 or wire_bytes > limits.max_logical_bytes:
        raise PackedTreeWaveError("stored slab arena exceeds limits")
    if compression == CompressionMode.NONE:
        if element_width != 0 or wire_bytes != logical_bytes:
            raise PackedTreeWaveError("non-canonical uncompressed arena metadata")
    elif compression == CompressionMode.RAW_DEFLATE:
        if element_width != 0 or wire_bytes >= logical_bytes:
            raise PackedTreeWaveError("non-canonical raw-deflate arena metadata")
    elif compression == CompressionMode.BYTE_PLANE_DEFLATE:
        if element_width not in (2, 4) or logical_bytes % element_width:
            raise PackedTreeWaveError("invalid byte-plane arena metadata")
        if wire_bytes >= logical_bytes:
            raise PackedTreeWaveError("byte-plane arena did not reduce stored bytes")
    table_start = HEADER_BYTES
    wire_start = _checked_add(
        "packet table end", table_start, table_bytes, maximum=UINT64_MAX
    )
    packet_end = _checked_add(
        "packet end", wire_start, wire_bytes, maximum=UINT64_MAX
    )
    if packet_end != raw.nbytes:
        qualifier = "truncated" if packet_end > raw.nbytes else "trailing"
        raise PackedTreeWaveError(f"{qualifier} packed-tree-wave bytes")
    table = raw[table_start:wire_start]
    wire = raw[wire_start:packet_end]

    # Structural validation intentionally precedes checksums.  It never
    # allocates from descriptor values and produces a precise rejection for a
    # CRC-valid malicious table as well as an accidentally corrupted one.
    descriptors: list[tuple[int, int, int, int, int]] = []
    previous_request_id = -1
    expected_offset = 0
    bytes_per_token: int | None = None
    for ordinal in range(entry_count):
        descriptor_offset = ordinal * DESCRIPTOR_BYTES
        request_id, step, token_count, offset, length = _DESCRIPTOR.unpack_from(
            table, descriptor_offset
        )
        if request_id == 0 or request_id <= previous_request_id:
            reason = "duplicate" if request_id == previous_request_id else "unordered"
            raise PackedTreeWaveError(f"{reason} request id in descriptor table")
        previous_request_id = request_id
        if token_count < 1 or token_count > limits.max_token_count:
            raise PackedTreeWaveError("descriptor token count exceeds limits")
        if length < 1 or length > limits.max_slab_bytes:
            raise PackedTreeWaveError("descriptor slab length exceeds limits")
        end = _checked_add(
            "descriptor slab", offset, length, maximum=UINT32_MAX
        )
        if offset != expected_offset:
            reason = "overlaps" if offset < expected_offset else "leaves a gap"
            raise PackedTreeWaveError(f"descriptor slab {reason}")
        if end > logical_bytes:
            raise PackedTreeWaveError("descriptor slab exceeds logical arena")
        if length % token_count:
            raise PackedTreeWaveError("descriptor slab is not divisible by token count")
        current_bytes_per_token = length // token_count
        if bytes_per_token is None:
            bytes_per_token = current_bytes_per_token
        elif current_bytes_per_token != bytes_per_token:
            raise PackedTreeWaveError("descriptor slabs disagree on bytes per token")
        descriptors.append((request_id, step, token_count, offset, length))
        expected_offset = end
    if expected_offset != logical_bytes:
        raise PackedTreeWaveError("descriptor slabs do not cover the logical arena")

    if _crc32(table) != table_crc:
        raise PackedTreeWaveError("descriptor table CRC mismatch")
    if _crc32(wire) != wire_crc:
        raise PackedTreeWaveError("stored slab arena CRC mismatch")
    if compression == CompressionMode.NONE:
        logical_view = wire
        logical_owner = owner
        zero_copy = immutable_backing
    else:
        logical_owner = decompress_lossless(
            wire,
            compression,
            expected_bytes=logical_bytes,
            element_width=element_width,
            maximum_bytes=limits.max_logical_bytes,
        )
        logical_view = memoryview(logical_owner)
        zero_copy = False
    if _crc32(logical_view) != logical_crc:
        raise PackedTreeWaveError("logical slab arena CRC mismatch")
    header_without_digest = _zero_digest_header(fields)
    digest_hasher = hashlib.sha256()
    digest_hasher.update(_DIGEST_DOMAIN)
    digest_hasher.update(header_without_digest)
    digest_hasher.update(table)
    digest_hasher.update(logical_view)
    digest_hasher.update(wire)
    actual_digest = digest_hasher.digest()
    if actual_digest != digest:
        raise PackedTreeWaveError("packed-tree-wave SHA-256 mismatch")
    leaves = tuple(
        DecodedPackedTreeLeaf(
            request_id=request_id,
            step=step,
            token_count=token_count,
            offset=offset,
            slab=logical_view[offset : offset + length],
        )
        for request_id, step, token_count, offset, length in descriptors
    )
    return DecodedPackedTreeWave(
        leaves=leaves,
        compression=compression,
        logical_bytes=logical_bytes,
        stored_bytes=wire_bytes,
        digest=digest,
        zero_copy_slabs=zero_copy,
        copied_input=copied_input,
        _owner=logical_owner,
    )


__all__ = [
    "CompressionAssessment",
    "CompressionBreakEven",
    "CompressionMode",
    "CompressionPolicy",
    "CompressionSelection",
    "DecodedPackedTreeLeaf",
    "DecodedPackedTreeWave",
    "DEFAULT_LIMITS",
    "DESCRIPTOR_BYTES",
    "HEADER_BYTES",
    "MAGIC",
    "MAX_PACKED_ENTRIES",
    "MAX_PACKED_LOGICAL_BYTES",
    "MAX_PACKED_SLAB_BYTES",
    "MAX_PACKED_TOKEN_COUNT",
    "PACKET_DEFLATE_LEVEL",
    "PackedTreeLeaf",
    "PackedTreeWaveEncoding",
    "PackedTreeWaveError",
    "PackedTreeWaveLimits",
    "VERSION",
    "compress_lossless",
    "compression_break_even",
    "decode_packed_tree_wave",
    "decompress_lossless",
    "encode_packed_tree_wave",
    "pack_packed_tree_wave",
    "select_compression_for_bandwidth",
]
