"""Persistent fail-closed TCP transport for resident expert owners.

The wire format is a fixed binary prefix, one canonical JSON header and an
optional raw little-endian tensor payload.  It never serializes Python objects
or uses pickle.  The transport implements the structural ``ExpertOwner``
contract consumed by ``ResidentExpertMesh`` while remaining agnostic to the
operator behind each inventory entry (linear, SwiGLU, or another exact owner).

This protocol currently provides framing and identity checks, not transport
security.  It has no authentication, TLS or Noise handshake and must not be
exposed to the public WAN.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass
import json
import math
import socket
import struct
import sys
import threading
from typing import Any, Mapping, Sequence

import torch

from .ram_expert_cache import ExpertKey
from .resident_expert_mesh import (
    ExpertOwner,
    ExpertResidentSlotUnavailableError,
    ExpertRouteUnavailableError,
    OwnerCoalescedExpertBatchItem,
    OwnerExpertBatchItem,
    OwnerExpertBatchResult,
)


RESIDENT_EXPERT_RPC_SCHEMA = "gdlp-resident-expert-rpc/1"
RESIDENT_EXPERT_RPC_COALESCED_CAPABILITY = (
    "gdlp-resident-expert-exact-input-coalescing/1"
)

_MAGIC = b"GDLPRPC1"
_FRAME_PREFIX = struct.Struct("!8sIQ")
_HARD_MAX_HEADER_BYTES = 4 * 1024 * 1024
_HARD_MAX_PAYLOAD_BYTES = 2 * 1024 * 1024 * 1024
_HARD_MAX_BATCH_ITEMS = 4_096
_HARD_MAX_INVENTORY_ITEMS = 65_536
_HARD_MAX_TENSOR_ELEMENTS = 1_000_000_000
_HARD_MAX_HOST_TRANSIENT_BYTES = (
    _HARD_MAX_HEADER_BYTES + 2 * _HARD_MAX_PAYLOAD_BYTES
)
_MAX_IDENTIFIER_CHARS = 512
# Server decoding must materialize Python tuple refs for the mesh ABI.  At its
# peak each assignment occupies its uint32 array cell, one tuple pointer and a
# Python integer object.  This is deliberately conservative for cached ints.
_ROW_REF_DECODE_BYTES_PER_ASSIGNMENT = 4 + struct.calcsize("P") + sys.getsizeof(0)
_COVERAGE_BITMAP_OVERHEAD_BYTES = sys.getsizeof(bytearray())


def _coverage_bitmap_bytes(row_count: int) -> int:
    """Conservative Python allocation for one byte of row coverage state."""

    # CPython keeps one trailing NUL byte in non-empty bytearray allocations.
    return _COVERAGE_BITMAP_OVERHEAD_BYTES + row_count + (1 if row_count else 0)


def _mark_covered_row(bitmap: bytearray, row: int) -> int:
    """Mark ``row`` and return one only when this is its first reference."""

    if bitmap[row]:
        return 0
    bitmap[row] = 1
    return 1

_Buffer = bytes | bytearray | memoryview
_Payload = _Buffer | Sequence[_Buffer]

_DTYPE_BY_NAME: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}
_NAME_BY_DTYPE = {value: key for key, value in _DTYPE_BY_NAME.items()}


class ResidentExpertRpcError(RuntimeError):
    """Base class for resident-expert transport failures."""


class ResidentExpertRpcContractError(ResidentExpertRpcError):
    """A local or peer value violates the sealed RPC contract."""


class ResidentExpertRpcRemoteError(ResidentExpertRpcError):
    """The peer atomically rejected a complete request."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.remote_message = message
        ResidentExpertRpcError.__init__(
            self,
            f"resident expert RPC remote error {code!r}: {message}",
        )


class ResidentExpertRpcResidentSlotUnavailable(
    ResidentExpertRpcRemoteError,
    ExpertResidentSlotUnavailableError,
):
    """Structured atomic rejection for a replica evicted after handshake."""

    def __init__(self, key: ExpertKey, message: str) -> None:
        self.key = key
        ResidentExpertRpcRemoteError.__init__(
            self,
            "resident_slot_unavailable",
            message,
        )


class ResidentExpertRpcFrameTooLarge(ResidentExpertRpcError):
    """A frame prefix exceeds a negotiated allocation limit."""


class ResidentExpertRpcTruncatedFrame(ResidentExpertRpcError):
    """The connection ended after only part of a frame arrived."""


class ResidentExpertRpcTimeout(ResidentExpertRpcError):
    """A frame did not complete before the configured socket timeout."""


class _IdleTimeout(ResidentExpertRpcTimeout):
    pass


def _positive_int(name: str, value: object, *, maximum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} exceeds the hard protocol maximum {maximum}")
    return value


def _nonnegative_int(name: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_float(name: str, value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite and positive") from error
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def _identifier(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} cannot be empty")
    normalized = value.strip()
    if len(normalized) > _MAX_IDENTIFIER_CHARS:
        raise ValueError(f"{name} is too long")
    return normalized


def _exact_keys(document: Mapping[str, object], expected: set[str], name: str) -> None:
    actual = set(document)
    if actual != expected:
        raise ResidentExpertRpcContractError(
            f"{name} keys mismatch: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


@dataclass(frozen=True)
class ResidentExpertRpcLimits:
    max_header_bytes: int = 256 * 1024
    max_payload_bytes: int = 256 * 1024 * 1024
    max_batch_items: int = 128
    max_inventory_items: int = 4_096
    max_tensor_elements: int = 16_777_216
    # Receive-side framing budget.  A tensor frame retains its wire buffer while
    # materializing Torch tensors, so the sealed lower bound is header + 2x
    # payload.  Owner/operator allocations are deliberately outside this cap.
    max_host_transient_bytes: int = (
        256 * 1024 + 2 * 256 * 1024 * 1024
    )
    io_timeout_seconds: float = 5.0

    def __post_init__(self) -> None:
        _positive_int(
            "max_header_bytes",
            self.max_header_bytes,
            maximum=_HARD_MAX_HEADER_BYTES,
        )
        _positive_int(
            "max_payload_bytes",
            self.max_payload_bytes,
            maximum=_HARD_MAX_PAYLOAD_BYTES,
        )
        _positive_int(
            "max_batch_items",
            self.max_batch_items,
            maximum=_HARD_MAX_BATCH_ITEMS,
        )
        _positive_int(
            "max_inventory_items",
            self.max_inventory_items,
            maximum=_HARD_MAX_INVENTORY_ITEMS,
        )
        _positive_int(
            "max_tensor_elements",
            self.max_tensor_elements,
            maximum=_HARD_MAX_TENSOR_ELEMENTS,
        )
        _positive_int(
            "max_host_transient_bytes",
            self.max_host_transient_bytes,
            maximum=_HARD_MAX_HOST_TRANSIENT_BYTES,
        )
        _positive_float("io_timeout_seconds", self.io_timeout_seconds)

    def negotiated_with(
        self,
        peer: "ResidentExpertRpcLimits",
    ) -> "ResidentExpertRpcLimits":
        return ResidentExpertRpcLimits(
            max_header_bytes=min(self.max_header_bytes, peer.max_header_bytes),
            max_payload_bytes=min(self.max_payload_bytes, peer.max_payload_bytes),
            max_batch_items=min(self.max_batch_items, peer.max_batch_items),
            max_inventory_items=min(
                self.max_inventory_items,
                peer.max_inventory_items,
            ),
            max_tensor_elements=min(
                self.max_tensor_elements,
                peer.max_tensor_elements,
            ),
            max_host_transient_bytes=min(
                self.max_host_transient_bytes,
                peer.max_host_transient_bytes,
            ),
            io_timeout_seconds=min(
                self.io_timeout_seconds,
                peer.io_timeout_seconds,
            ),
        )


@dataclass(frozen=True, order=True)
class ResidentExpertRpcInventoryEntry:
    key: ExpertKey
    content_id: str
    # Per-position input and output widths. Batch/assignment count is dynamic.
    shape: tuple[int, int]
    dtype: str

    def __post_init__(self) -> None:
        if not isinstance(self.key, ExpertKey):
            raise TypeError("inventory key must be ExpertKey")
        object.__setattr__(
            self,
            "content_id",
            _identifier("content_id", self.content_id),
        )
        if (
            not isinstance(self.shape, tuple)
            or len(self.shape) != 2
            or any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 1
                for value in self.shape
            )
        ):
            raise ValueError("inventory shape must be (input_width, output_width)")
        if self.dtype not in _DTYPE_BY_NAME:
            raise ValueError(f"unsupported inventory dtype {self.dtype!r}")

    @property
    def input_width(self) -> int:
        return self.shape[0]

    @property
    def output_width(self) -> int:
        return self.shape[1]

    def to_document(self) -> dict[str, object]:
        return {
            "contentId": self.content_id,
            "dtype": self.dtype,
            "expert": self.key.expert,
            "layer": self.key.layer,
            "shape": list(self.shape),
        }


@dataclass(frozen=True)
class ResidentExpertRpcTelemetry:
    connections: int
    handshakes: int
    batch_round_trips: int
    batch_items: int
    capability_round_trips: int
    coalesced_round_trips: int
    coalesced_items: int
    coalesced_shared_input_bytes: int
    coalesced_row_index_bytes: int
    coalesced_v1_equivalent_input_bytes: int
    close_round_trips: int
    frames_sent: int
    frames_received: int
    bytes_sent: int
    bytes_received: int
    peak_host_transient_bytes: int
    host_transient_preflight_rejections: int
    errors: int

    @property
    def round_trips(self) -> int:
        return (
            self.handshakes
            + self.capability_round_trips
            + self.batch_round_trips
            + self.close_round_trips
        )

    @property
    def coalesced_input_bytes_saved(self) -> int:
        return max(
            0,
            self.coalesced_v1_equivalent_input_bytes
            - self.coalesced_shared_input_bytes
            - self.coalesced_row_index_bytes,
        )


class _TelemetryCounters:
    def __init__(self) -> None:
        self.connections = 0
        self.handshakes = 0
        self.batch_round_trips = 0
        self.batch_items = 0
        self.capability_round_trips = 0
        self.coalesced_round_trips = 0
        self.coalesced_items = 0
        self.coalesced_shared_input_bytes = 0
        self.coalesced_row_index_bytes = 0
        self.coalesced_v1_equivalent_input_bytes = 0
        self.close_round_trips = 0
        self.frames_sent = 0
        self.frames_received = 0
        self.bytes_sent = 0
        self.bytes_received = 0
        self.peak_host_transient_bytes = 0
        self.host_transient_preflight_rejections = 0
        self.errors = 0

    def sent(self, byte_count: int) -> None:
        self.frames_sent += 1
        self.bytes_sent += byte_count

    def received(self, byte_count: int) -> None:
        self.frames_received += 1
        self.bytes_received += byte_count

    def host_transient_accepted(self, byte_count: int) -> None:
        self.peak_host_transient_bytes = max(
            self.peak_host_transient_bytes,
            byte_count,
        )

    def host_transient_rejected(self) -> None:
        self.host_transient_preflight_rejections += 1

    def snapshot(self) -> ResidentExpertRpcTelemetry:
        return ResidentExpertRpcTelemetry(
            connections=self.connections,
            handshakes=self.handshakes,
            batch_round_trips=self.batch_round_trips,
            batch_items=self.batch_items,
            capability_round_trips=self.capability_round_trips,
            coalesced_round_trips=self.coalesced_round_trips,
            coalesced_items=self.coalesced_items,
            coalesced_shared_input_bytes=self.coalesced_shared_input_bytes,
            coalesced_row_index_bytes=self.coalesced_row_index_bytes,
            coalesced_v1_equivalent_input_bytes=(
                self.coalesced_v1_equivalent_input_bytes
            ),
            close_round_trips=self.close_round_trips,
            frames_sent=self.frames_sent,
            frames_received=self.frames_received,
            bytes_sent=self.bytes_sent,
            bytes_received=self.bytes_received,
            peak_host_transient_bytes=self.peak_host_transient_bytes,
            host_transient_preflight_rejections=(
                self.host_transient_preflight_rejections
            ),
            errors=self.errors,
        )


def _limits_document(limits: ResidentExpertRpcLimits) -> dict[str, object]:
    return {
        "ioTimeoutSeconds": limits.io_timeout_seconds,
        "maxBatchItems": limits.max_batch_items,
        "maxHeaderBytes": limits.max_header_bytes,
        "maxHostTransientBytes": limits.max_host_transient_bytes,
        "maxInventoryItems": limits.max_inventory_items,
        "maxPayloadBytes": limits.max_payload_bytes,
        "maxTensorElements": limits.max_tensor_elements,
    }


def _limits_from_document(value: object) -> ResidentExpertRpcLimits:
    if not isinstance(value, dict):
        raise ResidentExpertRpcContractError("RPC limits must be an object")
    _exact_keys(
        value,
        {
            "ioTimeoutSeconds",
            "maxBatchItems",
            "maxHeaderBytes",
            "maxHostTransientBytes",
            "maxInventoryItems",
            "maxPayloadBytes",
            "maxTensorElements",
        },
        "RPC limits",
    )
    try:
        return ResidentExpertRpcLimits(
            max_header_bytes=value["maxHeaderBytes"],  # type: ignore[arg-type]
            max_host_transient_bytes=value["maxHostTransientBytes"],  # type: ignore[arg-type]
            max_payload_bytes=value["maxPayloadBytes"],  # type: ignore[arg-type]
            max_batch_items=value["maxBatchItems"],  # type: ignore[arg-type]
            max_inventory_items=value["maxInventoryItems"],  # type: ignore[arg-type]
            max_tensor_elements=value["maxTensorElements"],  # type: ignore[arg-type]
            io_timeout_seconds=value["ioTimeoutSeconds"],  # type: ignore[arg-type]
        )
    except (TypeError, ValueError) as error:
        raise ResidentExpertRpcContractError(str(error)) from error


def _canonical_header(document: Mapping[str, object]) -> bytes:
    try:
        return json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ResidentExpertRpcContractError(
            f"RPC header is not canonical JSON: {error}"
        ) from error


def _send_frame(
    connection: socket.socket,
    header: Mapping[str, object],
    payload: _Payload,
    limits: ResidentExpertRpcLimits,
) -> int:
    header_bytes = _canonical_header(header)
    if len(header_bytes) > limits.max_header_bytes:
        raise ResidentExpertRpcFrameTooLarge(
            f"RPC header requires {len(header_bytes)} bytes; "
            f"limit is {limits.max_header_bytes}"
        )
    payload_views = _payload_views(payload)
    payload_size = sum(len(view) for view in payload_views)
    if payload_size > limits.max_payload_bytes:
        raise ResidentExpertRpcFrameTooLarge(
            f"RPC payload requires {payload_size} bytes; "
            f"limit is {limits.max_payload_bytes}"
        )
    _preflight_host_transient(
        header_size=len(header_bytes),
        payload_size=payload_size,
        limits=limits,
        telemetry=None,
    )
    prefix = _FRAME_PREFIX.pack(_MAGIC, len(header_bytes), payload_size)
    connection.sendall(prefix)
    connection.sendall(header_bytes)
    for view in payload_views:
        if view:
            connection.sendall(view)
    return len(prefix) + len(header_bytes) + payload_size


def _payload_views(payload: _Payload) -> tuple[memoryview, ...]:
    if isinstance(payload, (bytes, bytearray, memoryview)):
        parts: Sequence[_Buffer] = (payload,)
    else:
        parts = payload
    views: list[memoryview] = []
    for part in parts:
        if not isinstance(part, (bytes, bytearray, memoryview)):
            raise TypeError("RPC payload parts must implement the buffer protocol")
        view = memoryview(part)
        if not view.c_contiguous:
            raise ResidentExpertRpcContractError(
                "RPC payload parts must be C-contiguous"
            )
        views.append(view.cast("B"))
    return tuple(views)


def _recv_exact(
    connection: socket.socket,
    byte_count: int,
    *,
    allow_clean_eof: bool = False,
    allow_idle_timeout: bool = False,
) -> bytearray | None:
    buffer = bytearray(byte_count)
    view = memoryview(buffer)
    received = 0
    while received < byte_count:
        try:
            chunk_size = connection.recv_into(view[received:], byte_count - received)
        except socket.timeout as error:
            if allow_idle_timeout and received == 0:
                raise _IdleTimeout("RPC connection is idle") from error
            raise ResidentExpertRpcTimeout(
                f"RPC frame timed out after {received} of {byte_count} bytes"
            ) from error
        if not chunk_size:
            if allow_clean_eof and received == 0:
                return None
            raise ResidentExpertRpcTruncatedFrame(
                f"RPC frame ended after {received} of {byte_count} bytes"
            )
        received += chunk_size
    return buffer


def _preflight_host_transient(
    *,
    header_size: int,
    payload_size: int,
    limits: ResidentExpertRpcLimits,
    telemetry: _TelemetryCounters | None,
) -> int:
    # The receive path owns one exact wire buffer and clones its tensor bytes
    # once into Torch storage.  This is a framing lower bound, not total process
    # RSS: parsed JSON and owner/operator allocations remain outside the budget.
    required = header_size + 2 * payload_size
    if required > limits.max_host_transient_bytes:
        if telemetry is not None:
            telemetry.host_transient_rejected()
        raise ResidentExpertRpcFrameTooLarge(
            f"RPC frame requires at least {required} host-transient bytes "
            f"(header + wire payload + tensor materialization); limit is "
            f"{limits.max_host_transient_bytes}"
        )
    if telemetry is not None:
        telemetry.host_transient_accepted(required)
    return required


def _recv_frame(
    connection: socket.socket,
    limits: ResidentExpertRpcLimits,
    *,
    allow_clean_eof: bool = False,
    allow_idle_timeout: bool = False,
    telemetry: _TelemetryCounters | None = None,
) -> tuple[dict[str, object], bytearray, int] | None:
    prefix = _recv_exact(
        connection,
        _FRAME_PREFIX.size,
        allow_clean_eof=allow_clean_eof,
        allow_idle_timeout=allow_idle_timeout,
    )
    if prefix is None:
        return None
    magic, header_size, payload_size = _FRAME_PREFIX.unpack(prefix)
    if magic != _MAGIC:
        raise ResidentExpertRpcContractError("invalid resident expert RPC magic")
    if header_size > limits.max_header_bytes:
        raise ResidentExpertRpcFrameTooLarge(
            f"RPC header declares {header_size} bytes; "
            f"limit is {limits.max_header_bytes}"
        )
    if payload_size > limits.max_payload_bytes:
        raise ResidentExpertRpcFrameTooLarge(
            f"RPC payload declares {payload_size} bytes; "
            f"limit is {limits.max_payload_bytes}"
        )
    _preflight_host_transient(
        header_size=header_size,
        payload_size=payload_size,
        limits=limits,
        telemetry=telemetry,
    )
    header_bytes = _recv_exact(connection, header_size)
    payload = _recv_exact(connection, payload_size) if payload_size else bytearray()
    assert header_bytes is not None and payload is not None
    try:
        decoded: Any = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ResidentExpertRpcContractError(
            f"invalid resident expert RPC JSON header: {error}"
        ) from error
    if not isinstance(decoded, dict):
        raise ResidentExpertRpcContractError("resident expert RPC header must be an object")
    if _canonical_header(decoded) != header_bytes:
        raise ResidentExpertRpcContractError("resident expert RPC header is not canonical")
    return decoded, payload, _FRAME_PREFIX.size + header_size + payload_size


def _dtype_name(tensor: torch.Tensor) -> str:
    try:
        return _NAME_BY_DTYPE[tensor.dtype]
    except KeyError as error:
        raise ResidentExpertRpcContractError(
            f"unsupported tensor dtype {tensor.dtype}"
        ) from error


def _tensor_payload(tensor: torch.Tensor) -> tuple[torch.Tensor, memoryview]:
    if sys.byteorder != "little":
        raise ResidentExpertRpcContractError(
            "resident expert RPC currently requires a little-endian host"
        )
    if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
        raise ResidentExpertRpcContractError("RPC tensors must use strided Torch layout")
    cpu = tensor.detach().to(device="cpu").contiguous()
    raw = memoryview(cpu.view(torch.uint8).numpy()).cast("B")
    return cpu, raw


def _tensor_from_payload(
    payload: _Buffer,
    *,
    offset: int,
    nbytes: int,
    shape: tuple[int, int],
    dtype_name: str,
) -> torch.Tensor:
    dtype = _DTYPE_BY_NAME[dtype_name]
    expected = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
    if nbytes != expected:
        raise ResidentExpertRpcContractError("tensor byte length does not match shape/dtype")
    if offset < 0 or nbytes < 0 or offset + nbytes > len(payload):
        raise ResidentExpertRpcTruncatedFrame("tensor segment is truncated")
    segment = memoryview(payload)[offset : offset + nbytes]
    # clone() is intentional: the returned tensor must outlive the reusable
    # frame buffer.  The old bytearray(slice) copy before frombuffer is gone.
    return torch.frombuffer(segment, dtype=dtype).clone().reshape(shape)


def _uint32_payload(values: Sequence[int]) -> tuple[array[int], memoryview]:
    """Encode row references without building one aggregate request payload."""

    packed = array("I", values)
    if packed.itemsize != 4:
        raise ResidentExpertRpcContractError(
            "resident expert RPC requires four-byte unsigned integers"
        )
    if sys.byteorder != "little":
        packed.byteswap()
    return packed, memoryview(packed).cast("B")


def _uint32_from_payload(
    payload: _Buffer,
    *,
    offset: int,
    count: int,
) -> tuple[int, ...]:
    nbytes = count * 4
    if offset < 0 or offset + nbytes > len(payload):
        raise ResidentExpertRpcTruncatedFrame("row-index segment is truncated")
    decoded = array("I")
    if decoded.itemsize != 4:
        raise ResidentExpertRpcContractError(
            "resident expert RPC requires four-byte unsigned integers"
        )
    decoded.frombytes(memoryview(payload)[offset : offset + nbytes])
    if sys.byteorder != "little":
        decoded.byteswap()
    return tuple(int(value) for value in decoded)


def _inventory_from_document(
    value: object,
    limits: ResidentExpertRpcLimits,
) -> tuple[ResidentExpertRpcInventoryEntry, ...]:
    if not isinstance(value, list):
        raise ResidentExpertRpcContractError("RPC inventory must be a list")
    if len(value) > limits.max_inventory_items:
        raise ResidentExpertRpcFrameTooLarge("RPC inventory exceeds negotiated limit")
    entries: list[ResidentExpertRpcInventoryEntry] = []
    seen: set[ExpertKey] = set()
    for item in value:
        if not isinstance(item, dict):
            raise ResidentExpertRpcContractError("RPC inventory entry must be an object")
        _exact_keys(
            item,
            {"contentId", "dtype", "expert", "layer", "shape"},
            "RPC inventory entry",
        )
        shape_value = item["shape"]
        if not isinstance(shape_value, list) or len(shape_value) != 2:
            raise ResidentExpertRpcContractError("RPC inventory shape is invalid")
        try:
            entry = ResidentExpertRpcInventoryEntry(
                key=ExpertKey(
                    _nonnegative_int("layer", item["layer"]),
                    _nonnegative_int("expert", item["expert"]),
                ),
                content_id=_identifier("content_id", item["contentId"]),
                shape=(
                    _positive_int("input_width", shape_value[0]),
                    _positive_int("output_width", shape_value[1]),
                ),
                dtype=_identifier("dtype", item["dtype"]),
            )
        except (TypeError, ValueError) as error:
            raise ResidentExpertRpcContractError(str(error)) from error
        if entry.key in seen:
            raise ResidentExpertRpcContractError("RPC inventory repeats an expert")
        seen.add(entry.key)
        entries.append(entry)
    if tuple(entries) != tuple(sorted(entries)):
        raise ResidentExpertRpcContractError("RPC inventory order is not canonical")
    return tuple(entries)


def _error_header(request_id: int, code: str, message: str) -> dict[str, object]:
    return {
        "code": _identifier("error code", code),
        "message": str(message)[:512],
        "requestId": _nonnegative_int("request_id", request_id),
        "schema": RESIDENT_EXPERT_RPC_SCHEMA,
        "type": "error",
    }


def _resident_slot_message(key: ExpertKey) -> str:
    return json.dumps(
        {"expert": key.expert, "layer": key.layer},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _resident_slot_key(message: str) -> ExpertKey:
    try:
        document = json.loads(message)
        if not isinstance(document, dict):
            raise ValueError("resident slot detail must be an object")
        _exact_keys(document, {"expert", "layer"}, "resident slot detail")
        return ExpertKey(
            _nonnegative_int("resident slot layer", document["layer"]),
            _nonnegative_int("resident slot expert", document["expert"]),
        )
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise ResidentExpertRpcContractError(
            "RPC resident-slot error detail is invalid"
        ) from error


def _raise_if_error(
    header: Mapping[str, object],
    payload: _Buffer,
    *,
    expected_request_id: int,
) -> None:
    if header.get("type") != "error":
        return
    _exact_keys(
        header,
        {"code", "message", "requestId", "schema", "type"},
        "RPC error",
    )
    if payload:
        raise ResidentExpertRpcContractError("RPC error response cannot contain payload")
    if header["schema"] != RESIDENT_EXPERT_RPC_SCHEMA:
        raise ResidentExpertRpcContractError("RPC error schema mismatch")
    try:
        request_id = _nonnegative_int("request_id", header["requestId"])
        code = _identifier("error code", header["code"])
        message = _identifier("error message", header["message"])
    except ValueError as error:
        raise ResidentExpertRpcContractError(str(error)) from error
    if request_id != expected_request_id:
        raise ResidentExpertRpcContractError(
            "RPC error response request identity mismatch"
        )
    if code == "resident_slot_unavailable":
        raise ResidentExpertRpcResidentSlotUnavailable(
            _resident_slot_key(message),
            message,
        )
    raise ResidentExpertRpcRemoteError(code, message)


class ResidentExpertRpcClient:
    """Persistent TCP ``ExpertOwner`` client with a handshake inventory cache."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        client_node_id: str,
        expected_node_id: str,
        limits: ResidentExpertRpcLimits | None = None,
        connect_timeout_seconds: float = 5.0,
    ) -> None:
        self.host = _identifier("host", host)
        self.port = _positive_int("port", port, maximum=65_535)
        self.client_node_id = _identifier("client_node_id", client_node_id)
        self.node_id = _identifier("expected_node_id", expected_node_id)
        self.limits = limits or ResidentExpertRpcLimits()
        self._negotiated_limits = self.limits
        self._telemetry = _TelemetryCounters()
        self._lock = threading.Lock()
        self._request_id = 1
        self._closed = False
        self._handshake_complete = False
        # Capability discovery is lazy so pure v1 clients do not pay another
        # round trip and fault-injection/legacy peers keep their original flow.
        self._coalesced_supported: bool | None = None
        self._inventory: dict[ExpertKey, ResidentExpertRpcInventoryEntry] = {}
        self._resident_inventory: frozenset[tuple[ExpertKey, str]] = frozenset()
        timeout = _positive_float(
            "connect_timeout_seconds",
            connect_timeout_seconds,
        )
        self._socket = socket.create_connection((self.host, self.port), timeout=timeout)
        self._socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._socket.settimeout(self.limits.io_timeout_seconds)
        self._telemetry.connections = 1
        try:
            self._perform_handshake()
        except BaseException:
            self._abort_socket()
            raise

    @property
    def inventory(self) -> tuple[ResidentExpertRpcInventoryEntry, ...]:
        return tuple(self._inventory[key] for key in sorted(self._inventory))

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def supports_exact_input_coalescing(self) -> bool | None:
        """Cached extension state; ``None`` means it has not been probed yet."""

        return self._coalesced_supported

    @property
    def coalesced_row_index_bytes_per_assignment(self) -> int:
        return 4

    def probe_exact_input_coalescing(self) -> bool:
        """Discover the optional extension without executing an expert batch."""

        with self._lock:
            self._require_open()
            self._probe_coalesced_capability_locked()
            return bool(self._coalesced_supported)

    def telemetry_snapshot(self) -> ResidentExpertRpcTelemetry:
        return self._telemetry.snapshot()

    def has_expert(self, key: ExpertKey, content_id: str) -> bool:
        # Deliberately local: the immutable handshake inventory is the cache.
        entry = self._inventory.get(key)
        return entry is not None and entry.content_id == content_id

    def is_expert_resident(self, key: ExpertKey, content_id: str) -> bool:
        # This is a distinct sealed promise, not an alias for availability.
        # The server may advertise only inventory that its physical owner
        # declared resident during handshake construction.
        with self._lock:
            return (key, content_id) in self._resident_inventory

    def _record_lost_residency(
        self,
        error: ResidentExpertRpcResidentSlotUnavailable,
    ) -> None:
        self._resident_inventory = frozenset(
            (key, content_id)
            for key, content_id in self._resident_inventory
            if key != error.key
        )

    def execute_batch(
        self,
        items: Sequence[OwnerExpertBatchItem],
    ) -> tuple[OwnerExpertBatchResult, ...]:
        with self._lock:
            self._require_open()
            batch = tuple(items)
            limits = self._negotiated_limits
            if not batch:
                raise ResidentExpertRpcContractError("RPC expert batch cannot be empty")
            if len(batch) > limits.max_batch_items:
                raise ResidentExpertRpcFrameTooLarge("RPC expert batch exceeds item limit")
            if any(not isinstance(item, OwnerExpertBatchItem) for item in batch):
                raise TypeError("items must contain OwnerExpertBatchItem values")
            ordered = tuple(sorted(batch, key=lambda item: item.key))
            keys = tuple(item.key for item in ordered)
            if len(set(keys)) != len(keys):
                raise ResidentExpertRpcContractError("RPC expert batch repeats a key")
            layers = {key.layer for key in keys}
            if len(layers) != 1:
                raise ResidentExpertRpcContractError(
                    "one RPC expert batch must contain exactly one layer"
                )

            prepared: list[
                tuple[OwnerExpertBatchItem, tuple[int, int], str, int]
            ] = []
            item_headers: list[dict[str, object]] = []
            expected: dict[ExpertKey, tuple[str, tuple[int, int], str]] = {}
            offset = 0
            for item in ordered:
                entry = self._inventory.get(item.key)
                if entry is None or entry.content_id != item.content_id:
                    raise ExpertRouteUnavailableError(
                        f"RPC owner {self.node_id!r} lacks exact content for {item.key}"
                    )
                if (item.key, item.content_id) not in self._resident_inventory:
                    raise ResidentExpertRpcResidentSlotUnavailable(
                        item.key,
                        _resident_slot_message(item.key),
                    )
                activation = item.activations
                if (
                    not isinstance(activation, torch.Tensor)
                    or activation.layout != torch.strided
                ):
                    raise ResidentExpertRpcContractError(
                        "RPC activations must use strided Torch layout"
                    )
                shape = tuple(int(value) for value in activation.shape)
                if (
                    activation.ndim != 2
                    or shape[0] < 1
                    or shape[1] != entry.input_width
                ):
                    raise ResidentExpertRpcContractError(
                        f"activation shape for {item.key} does not match inventory"
                    )
                dtype_name = _dtype_name(activation)
                if dtype_name != entry.dtype:
                    raise ResidentExpertRpcContractError(
                        f"activation dtype for {item.key} does not match inventory"
                    )
                if activation.numel() > limits.max_tensor_elements:
                    raise ResidentExpertRpcFrameTooLarge(
                        f"activation tensor for {item.key} exceeds element limit"
                    )
                nbytes = activation.numel() * activation.element_size()
                if nbytes > limits.max_payload_bytes - offset:
                    raise ResidentExpertRpcFrameTooLarge(
                        "RPC expert batch exceeds aggregate payload limit"
                    )
                item_headers.append(
                    {
                        "contentId": item.content_id,
                        "dtype": dtype_name,
                        "expert": item.key.expert,
                        "layer": item.key.layer,
                        "nbytes": nbytes,
                        "offset": offset,
                        "shape": list(shape),
                    }
                )
                expected[item.key] = (
                    entry.content_id,
                    (shape[0], entry.output_width),
                    entry.dtype,
                )
                prepared.append((item, shape, dtype_name, nbytes))
                offset += nbytes

            request_id = self._next_request_id()
            header = {
                "items": item_headers,
                "layer": next(iter(layers)),
                "requestId": request_id,
                "schema": RESIDENT_EXPERT_RPC_SCHEMA,
                "type": "execute-batch",
            }
            _preflight_host_transient(
                header_size=len(_canonical_header(header)),
                payload_size=offset,
                limits=limits,
                telemetry=None,
            )
            # Only stage/serialize after every tensor, aggregate payload and
            # peer receive budget have passed allocation-independent preflight.
            payload_parts: list[memoryview] = []
            for item, _shape, _dtype, nbytes in prepared:
                _cpu, raw = _tensor_payload(item.activations)
                if len(raw) != nbytes:
                    raise ResidentExpertRpcContractError(
                        "RPC activation changed during serialization"
                    )
                payload_parts.append(raw)
            try:
                self._telemetry.sent(
                    _send_frame(self._socket, header, payload_parts, limits)
                )
                # sendall() has consumed every view. Release request staging
                # before response receive/materialization so the two payload
                # peaks cannot overlap causally.
                payload_parts.clear()
                del raw, _cpu
                self._telemetry.batch_items += len(batch)
                frame = _recv_frame(
                    self._socket,
                    limits,
                    telemetry=self._telemetry,
                )
                if frame is None:
                    raise ResidentExpertRpcTruncatedFrame(
                        "RPC owner closed before returning the batch"
                    )
                response, response_payload, wire_bytes = frame
                self._telemetry.received(wire_bytes)
                self._telemetry.batch_round_trips += 1
                if response.get("type") == "error":
                    self._telemetry.errors += 1
                    _raise_if_error(
                        response,
                        response_payload,
                        expected_request_id=request_id,
                    )
                return self._decode_batch_response(
                    response,
                    response_payload,
                    request_id=request_id,
                    expected=expected,
                )
            except ResidentExpertRpcResidentSlotUnavailable as error:
                if error.key not in keys:
                    self._abort_socket()
                    raise ResidentExpertRpcContractError(
                        "RPC residency error references a key outside the sent batch"
                    ) from error
                self._record_lost_residency(error)
                raise
            except ResidentExpertRpcRemoteError:
                # A complete, correlated atomic error leaves framing aligned.
                raise
            except BaseException:
                # Any other post-send failure makes stream alignment uncertain.
                self._abort_socket()
                raise

    def execute_coalesced_batch(
        self,
        shared_activations: torch.Tensor,
        items: Sequence[OwnerCoalescedExpertBatchItem],
    ) -> tuple[OwnerExpertBatchResult, ...]:
        """Send one owner-shared hidden matrix plus little-endian row refs.

        A peer that only implements the original v1 request set is detected by
        a correlated capability probe.  In that case this method reconstructs
        the original per-expert matrices locally and uses ``execute_batch``;
        the persistent stream remains aligned and exact.
        """

        fallback: tuple[OwnerExpertBatchItem, ...] | None = None
        with self._lock:
            self._require_open()
            limits = self._negotiated_limits
            batch = tuple(items)
            if not batch:
                raise ResidentExpertRpcContractError(
                    "RPC coalesced expert batch cannot be empty"
                )
            if len(batch) > limits.max_batch_items:
                raise ResidentExpertRpcFrameTooLarge(
                    "RPC coalesced expert batch exceeds item limit"
                )
            if any(
                not isinstance(item, OwnerCoalescedExpertBatchItem)
                for item in batch
            ):
                raise TypeError(
                    "items must contain OwnerCoalescedExpertBatchItem values"
                )
            ordered = tuple(sorted(batch, key=lambda item: item.key))
            keys = tuple(item.key for item in ordered)
            if len(set(keys)) != len(keys):
                raise ResidentExpertRpcContractError(
                    "RPC coalesced expert batch repeats a key"
                )
            layers = {key.layer for key in keys}
            if len(layers) != 1:
                raise ResidentExpertRpcContractError(
                    "one RPC coalesced batch must contain exactly one layer"
                )
            if (
                not isinstance(shared_activations, torch.Tensor)
                or shared_activations.layout != torch.strided
                or shared_activations.ndim != 2
                or shared_activations.shape[0] < 1
            ):
                raise ResidentExpertRpcContractError(
                    "RPC shared activations must be a non-empty strided matrix"
                )
            shared_shape = tuple(int(value) for value in shared_activations.shape)
            if shared_activations.numel() > limits.max_tensor_elements:
                raise ResidentExpertRpcFrameTooLarge(
                    "RPC shared activation tensor exceeds element limit"
                )
            dtype_name = _dtype_name(shared_activations)
            shared_nbytes = (
                shared_activations.numel() * shared_activations.element_size()
            )

            expected: dict[ExpertKey, tuple[str, tuple[int, int], str]] = {}
            assignment_count = 0
            for item in ordered:
                entry = self._inventory.get(item.key)
                if entry is None or entry.content_id != item.content_id:
                    raise ExpertRouteUnavailableError(
                        f"RPC owner {self.node_id!r} lacks exact content for {item.key}"
                    )
                if (item.key, item.content_id) not in self._resident_inventory:
                    raise ResidentExpertRpcResidentSlotUnavailable(
                        item.key,
                        _resident_slot_message(item.key),
                    )
                if shared_shape[1] != entry.input_width or dtype_name != entry.dtype:
                    raise ResidentExpertRpcContractError(
                        f"shared activation metadata for {item.key} does not match inventory"
                    )
                for row in item.row_indices:
                    if row >= shared_shape[0]:
                        raise ResidentExpertRpcContractError(
                            f"row reference for {item.key} exceeds shared activations"
                        )
                assignment_count += len(item.row_indices)
                if assignment_count > limits.max_tensor_elements:
                    raise ResidentExpertRpcFrameTooLarge(
                        "RPC coalesced row references exceed element limit"
                    )
                expected[item.key] = (
                    entry.content_id,
                    (len(item.row_indices), entry.output_width),
                    entry.dtype,
                )
            coverage_bitmap_bytes = _coverage_bitmap_bytes(shared_shape[0])
            if coverage_bitmap_bytes > limits.max_host_transient_bytes:
                raise ResidentExpertRpcFrameTooLarge(
                    "RPC coalesced row-coverage bitmap exceeds the negotiated "
                    "host-transient limit"
                )
            covered_rows = bytearray(shared_shape[0])
            covered_count = 0
            for item in ordered:
                for row in item.row_indices:
                    covered_count += _mark_covered_row(covered_rows, row)
            if covered_count != shared_shape[0]:
                raise ResidentExpertRpcContractError(
                    "RPC coalesced shared activations must have exact row coverage"
                )
            del covered_rows
            row_index_bytes = assignment_count * 4
            v1_equivalent_input_bytes = (
                assignment_count
                * shared_shape[1]
                * shared_activations.element_size()
            )
            coalesced_input_bytes = shared_nbytes + row_index_bytes
            # The public coalesced ABI is also the final no-regression guard.
            # When every shared row has only one assignment (or tiny H makes
            # uint32 metadata erase reuse), keep the exact v1 transport even if
            # the peer advertises the extension.
            use_extension = coalesced_input_bytes < v1_equivalent_input_bytes
            if use_extension:
                self._probe_coalesced_capability_locked()
                use_extension = bool(self._coalesced_supported)
            if not use_extension:
                fallback = tuple(
                    OwnerExpertBatchItem(
                        item.key,
                        item.content_id,
                        shared_activations.index_select(
                            0,
                            torch.tensor(
                                item.row_indices,
                                dtype=torch.long,
                                device=shared_activations.device,
                            ),
                        ),
                        require_resident=item.require_resident,
                    )
                    for item in ordered
                )
            else:
                payload_size = shared_nbytes + row_index_bytes
                if payload_size > limits.max_payload_bytes:
                    raise ResidentExpertRpcFrameTooLarge(
                        "RPC coalesced batch exceeds aggregate payload limit"
                    )
                item_headers: list[dict[str, object]] = []
                cursor = shared_nbytes
                for item in ordered:
                    nbytes = len(item.row_indices) * 4
                    item_headers.append(
                        {
                            "contentId": item.content_id,
                            "expert": item.key.expert,
                            "layer": item.key.layer,
                            # RPC inventory is resident-only, exactly like the
                            # v1 request which has no fallback flag.
                            "requireResident": True,
                            "rowCount": len(item.row_indices),
                            "rowNbytes": nbytes,
                            "rowOffset": cursor,
                        }
                    )
                    cursor += nbytes
                request_id = self._next_request_id()
                header = {
                    "activation": {
                        "dtype": dtype_name,
                        "nbytes": shared_nbytes,
                        "offset": 0,
                        "shape": list(shared_shape),
                    },
                    "items": item_headers,
                    "layer": next(iter(layers)),
                    "requestId": request_id,
                    "schema": RESIDENT_EXPERT_RPC_SCHEMA,
                    "type": "execute-coalesced",
                }
                row_materialization_bytes = (
                    assignment_count * _ROW_REF_DECODE_BYTES_PER_ASSIGNMENT
                )
                projected_output_bytes = sum(
                    math.prod(shape)
                    * torch.empty(
                        (),
                        dtype=_DTYPE_BY_NAME[expected_dtype],
                    ).element_size()
                    for _content_id, shape, expected_dtype in expected.values()
                )
                peer_required_host_transient = max(
                    len(_canonical_header(header))
                    + 2 * payload_size
                    + row_materialization_bytes
                    + coverage_bitmap_bytes,
                    shared_nbytes
                    + row_materialization_bytes
                    + projected_output_bytes,
                )
                if (
                    peer_required_host_transient
                    > limits.max_host_transient_bytes
                ):
                    raise ResidentExpertRpcFrameTooLarge(
                        "RPC peer would require at least "
                        f"{peer_required_host_transient} host-transient bytes "
                        "including decoded row references; limit is "
                        f"{limits.max_host_transient_bytes}"
                    )
                _preflight_host_transient(
                    header_size=len(_canonical_header(header)),
                    payload_size=payload_size,
                    limits=limits,
                    telemetry=None,
                )
                _shared_cpu, shared_raw = _tensor_payload(shared_activations)
                packed_rows: list[array[int]] = []
                payload_parts: list[memoryview] = [shared_raw]
                for item in ordered:
                    packed, raw = _uint32_payload(item.row_indices)
                    packed_rows.append(packed)
                    payload_parts.append(raw)
                try:
                    self._telemetry.sent(
                        _send_frame(self._socket, header, payload_parts, limits)
                    )
                    payload_parts.clear()
                    packed_rows.clear()
                    del packed, raw, shared_raw, _shared_cpu
                    self._telemetry.batch_items += len(batch)
                    self._telemetry.coalesced_items += len(batch)
                    self._telemetry.coalesced_shared_input_bytes += shared_nbytes
                    self._telemetry.coalesced_row_index_bytes += row_index_bytes
                    self._telemetry.coalesced_v1_equivalent_input_bytes += (
                        v1_equivalent_input_bytes
                    )
                    frame = _recv_frame(
                        self._socket,
                        limits,
                        telemetry=self._telemetry,
                    )
                    if frame is None:
                        raise ResidentExpertRpcTruncatedFrame(
                            "RPC owner closed before returning the coalesced batch"
                        )
                    response, response_payload, wire_bytes = frame
                    self._telemetry.received(wire_bytes)
                    self._telemetry.batch_round_trips += 1
                    self._telemetry.coalesced_round_trips += 1
                    if response.get("type") == "error":
                        self._telemetry.errors += 1
                        _raise_if_error(
                            response,
                            response_payload,
                            expected_request_id=request_id,
                        )
                    return self._decode_batch_response(
                        response,
                        response_payload,
                        request_id=request_id,
                        expected=expected,
                        response_type="execute-coalesced-ok",
                    )
                except ResidentExpertRpcResidentSlotUnavailable as error:
                    if error.key not in keys:
                        self._abort_socket()
                        raise ResidentExpertRpcContractError(
                            "RPC residency error references a key outside the sent batch"
                        ) from error
                    self._record_lost_residency(error)
                    raise
                except ResidentExpertRpcRemoteError:
                    raise
                except BaseException:
                    self._abort_socket()
                    raise

        assert fallback is not None
        return self.execute_batch(fallback)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                if self._handshake_complete:
                    request_id = self._next_request_id()
                    header = {
                        "requestId": request_id,
                        "schema": RESIDENT_EXPERT_RPC_SCHEMA,
                        "type": "close",
                    }
                    self._telemetry.sent(
                        _send_frame(
                            self._socket,
                            header,
                            b"",
                            self._negotiated_limits,
                        )
                    )
                    frame = _recv_frame(
                        self._socket,
                        self._negotiated_limits,
                        telemetry=self._telemetry,
                    )
                    if frame is not None:
                        response, payload, wire_bytes = frame
                        self._telemetry.received(wire_bytes)
                        _exact_keys(
                            response,
                            {"requestId", "schema", "type"},
                            "RPC close response",
                        )
                        if (
                            response["schema"] != RESIDENT_EXPERT_RPC_SCHEMA
                            or response["type"] != "close-ok"
                            or response["requestId"] != request_id
                            or payload
                        ):
                            raise ResidentExpertRpcContractError(
                                "invalid RPC close response"
                            )
                        self._telemetry.close_round_trips += 1
            finally:
                self._abort_socket()

    def __enter__(self) -> "ResidentExpertRpcClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _perform_handshake(self) -> None:
        header = {
            "clientNodeId": self.client_node_id,
            "expectedNodeId": self.node_id,
            "limits": _limits_document(self.limits),
            "schema": RESIDENT_EXPERT_RPC_SCHEMA,
            "type": "hello",
        }
        self._telemetry.sent(_send_frame(self._socket, header, b"", self.limits))
        frame = _recv_frame(
            self._socket,
            self.limits,
            telemetry=self._telemetry,
        )
        if frame is None:
            raise ResidentExpertRpcTruncatedFrame("RPC owner closed during handshake")
        response, payload, wire_bytes = frame
        self._telemetry.received(wire_bytes)
        if response.get("type") == "error":
            self._telemetry.errors += 1
            _raise_if_error(response, payload, expected_request_id=0)
        _exact_keys(
            response,
            {"inventory", "limits", "nodeId", "schema", "type"},
            "RPC hello response",
        )
        if payload:
            raise ResidentExpertRpcContractError("RPC hello response cannot contain payload")
        if response["schema"] != RESIDENT_EXPERT_RPC_SCHEMA:
            raise ResidentExpertRpcContractError("RPC handshake schema mismatch")
        if response["type"] != "hello-ok":
            raise ResidentExpertRpcContractError("RPC handshake response type mismatch")
        if response["nodeId"] != self.node_id:
            raise ResidentExpertRpcContractError(
                f"RPC node identity mismatch: {response['nodeId']!r} != {self.node_id!r}"
            )
        negotiated = _limits_from_document(response["limits"])
        expected_negotiated = self.limits.negotiated_with(negotiated)
        if negotiated != expected_negotiated:
            raise ResidentExpertRpcContractError(
                "RPC server advertised limits above the client contract"
            )
        inventory = _inventory_from_document(response["inventory"], negotiated)
        if not inventory:
            raise ResidentExpertRpcContractError("RPC owner inventory cannot be empty")
        self._inventory = {entry.key: entry for entry in inventory}
        self._resident_inventory = frozenset(
            (entry.key, entry.content_id) for entry in inventory
        )
        self._negotiated_limits = negotiated
        self._socket.settimeout(negotiated.io_timeout_seconds)
        self._handshake_complete = True
        self._telemetry.handshakes = 1

    def _probe_coalesced_capability_locked(self) -> None:
        if self._coalesced_supported is not None:
            return
        request_id = self._next_request_id()
        header = {
            "requestId": request_id,
            "schema": RESIDENT_EXPERT_RPC_SCHEMA,
            "type": "capabilities",
        }
        try:
            self._telemetry.sent(
                _send_frame(
                    self._socket,
                    header,
                    b"",
                    self._negotiated_limits,
                )
            )
            frame = _recv_frame(
                self._socket,
                self._negotiated_limits,
                telemetry=self._telemetry,
            )
            if frame is None:
                raise ResidentExpertRpcTruncatedFrame(
                    "RPC owner closed during capability discovery"
                )
            response, payload, wire_bytes = frame
            self._telemetry.received(wire_bytes)
            self._telemetry.capability_round_trips += 1
            if response.get("type") == "error":
                try:
                    _raise_if_error(
                        response,
                        payload,
                        expected_request_id=request_id,
                    )
                except ResidentExpertRpcRemoteError as error:
                    if error.code == "unsupported_request":
                        self._coalesced_supported = False
                        return
                    raise
            _exact_keys(
                response,
                {"capabilities", "requestId", "schema", "type"},
                "RPC capabilities response",
            )
            capabilities = response["capabilities"]
            if (
                response["schema"] != RESIDENT_EXPERT_RPC_SCHEMA
                or response["type"] != "capabilities-ok"
                or response["requestId"] != request_id
                or payload
                or not isinstance(capabilities, list)
                or any(not isinstance(value, str) for value in capabilities)
                or capabilities != sorted(set(capabilities))
            ):
                raise ResidentExpertRpcContractError(
                    "invalid RPC capabilities response"
                )
            self._coalesced_supported = (
                RESIDENT_EXPERT_RPC_COALESCED_CAPABILITY in capabilities
            )
        except ResidentExpertRpcRemoteError:
            raise
        except BaseException:
            self._abort_socket()
            raise

    def _decode_batch_response(
        self,
        header: Mapping[str, object],
        payload: _Buffer,
        *,
        request_id: int,
        expected: Mapping[ExpertKey, tuple[str, tuple[int, int], str]],
        response_type: str = "execute-batch-ok",
    ) -> tuple[OwnerExpertBatchResult, ...]:
        _exact_keys(
            header,
            {"items", "requestId", "schema", "type"},
            "RPC batch response",
        )
        if (
            header["schema"] != RESIDENT_EXPERT_RPC_SCHEMA
            or header["type"] != response_type
            or header["requestId"] != request_id
        ):
            raise ResidentExpertRpcContractError("RPC batch response identity mismatch")
        items = header["items"]
        if not isinstance(items, list) or len(items) != len(expected):
            raise ResidentExpertRpcContractError("RPC batch response is incomplete")
        decoded_meta: list[tuple[ExpertKey, tuple[int, int], str, int, int]] = []
        seen: set[ExpertKey] = set()
        cursor = 0
        prior_key: ExpertKey | None = None
        for item in items:
            if not isinstance(item, dict):
                raise ResidentExpertRpcContractError("RPC result item must be an object")
            _exact_keys(
                item,
                {
                    "contentId",
                    "dtype",
                    "expert",
                    "layer",
                    "nbytes",
                    "offset",
                    "shape",
                },
                "RPC result item",
            )
            key = ExpertKey(
                _nonnegative_int("layer", item["layer"]),
                _nonnegative_int("expert", item["expert"]),
            )
            if (
                key in seen
                or key not in expected
                or (prior_key is not None and key <= prior_key)
            ):
                raise ResidentExpertRpcContractError("RPC batch returned an invalid key")
            prior_key = key
            seen.add(key)
            content_id, expected_shape, expected_dtype = expected[key]
            shape = _shape_from_document(item["shape"])
            dtype_name = _identifier("dtype", item["dtype"])
            offset = _nonnegative_int("offset", item["offset"])
            nbytes = _nonnegative_int("nbytes", item["nbytes"])
            if (
                item["contentId"] != content_id
                or shape != expected_shape
                or dtype_name != expected_dtype
                or offset != cursor
            ):
                raise ResidentExpertRpcContractError(
                    f"RPC result metadata mismatch for {key}"
                )
            expected_nbytes = (
                math.prod(shape)
                * torch.empty((), dtype=_DTYPE_BY_NAME[dtype_name]).element_size()
            )
            if nbytes != expected_nbytes or nbytes > len(payload) - offset:
                raise ResidentExpertRpcContractError(
                    f"RPC result byte length mismatch for {key}"
                )
            decoded_meta.append((key, shape, dtype_name, offset, nbytes))
            cursor += nbytes
        if seen != set(expected) or cursor != len(payload):
            raise ResidentExpertRpcContractError("RPC batch response is not atomic/complete")
        # Materialize only after all metadata covers the already-budgeted wire
        # buffer exactly; malformed late items cannot force partial tensor clones.
        results = [
            OwnerExpertBatchResult(
                key,
                _tensor_from_payload(
                    payload,
                    offset=offset,
                    nbytes=nbytes,
                    shape=shape,
                    dtype_name=dtype_name,
                ),
            )
            for key, shape, dtype_name, offset, nbytes in decoded_meta
        ]
        return tuple(results)

    def _next_request_id(self) -> int:
        request_id = self._request_id
        self._request_id += 1
        return request_id

    def _require_open(self) -> None:
        if self._closed or not self._handshake_complete:
            raise ResidentExpertRpcError("resident expert RPC client is closed")

    def _abort_socket(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._socket.close()


class ResidentExpertRpcServer:
    """Single-process persistent owner server with atomic batch responses."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        node_id: str,
        owner: ExpertOwner,
        inventory: Sequence[ResidentExpertRpcInventoryEntry],
        limits: ResidentExpertRpcLimits | None = None,
        accept_poll_seconds: float = 0.1,
        enable_coalesced_extension: bool = True,
    ) -> None:
        self.host = _identifier("host", host)
        if not isinstance(port, int) or isinstance(port, bool) or not 0 <= port <= 65_535:
            raise ValueError("port must be an integer in [0, 65535]")
        self.port = port
        self.node_id = _identifier("node_id", node_id)
        self.owner = owner
        if getattr(owner, "node_id", None) != self.node_id:
            raise ValueError("owner node_id does not match RPC server node_id")
        if not callable(getattr(owner, "has_expert", None)) or not callable(
            getattr(owner, "execute_batch", None)
        ) or not callable(
            getattr(owner, "is_expert_resident", None)
        ):
            raise TypeError("owner must implement ExpertOwner")
        self.limits = limits or ResidentExpertRpcLimits()
        self.accept_poll_seconds = _positive_float(
            "accept_poll_seconds",
            accept_poll_seconds,
        )
        if not isinstance(enable_coalesced_extension, bool):
            raise TypeError("enable_coalesced_extension must be boolean")
        self.enable_coalesced_extension = enable_coalesced_extension
        entries = tuple(sorted(inventory))
        if not entries:
            raise ValueError("RPC server inventory cannot be empty")
        if len(entries) > self.limits.max_inventory_items:
            raise ValueError("RPC server inventory exceeds configured limit")
        if len({entry.key for entry in entries}) != len(entries):
            raise ValueError("RPC server inventory repeats an expert")
        for entry in entries:
            if not isinstance(entry, ResidentExpertRpcInventoryEntry):
                raise TypeError("inventory must contain RPC inventory entries")
            if not owner.has_expert(entry.key, entry.content_id):
                raise ValueError(f"owner lacks inventory content for {entry.key}")
            if not owner.is_expert_resident(entry.key, entry.content_id):
                raise ValueError(
                    f"owner inventory content is not physically resident for {entry.key}"
                )
        self.inventory = entries
        self._inventory = {entry.key: entry for entry in entries}
        self._listener: socket.socket | None = None
        self._active_connection: socket.socket | None = None
        self._state_lock = threading.Lock()
        self._closed = False
        self._telemetry = _TelemetryCounters()

    @property
    def address(self) -> tuple[str, int]:
        with self._state_lock:
            listener = self._listener
        if listener is None:
            raise RuntimeError("RPC server is not bound")
        host, port = listener.getsockname()[:2]
        return str(host), int(port)

    def telemetry_snapshot(self) -> ResidentExpertRpcTelemetry:
        return self._telemetry.snapshot()

    def bind(self) -> tuple[str, int]:
        with self._state_lock:
            if self._closed:
                raise RuntimeError("RPC server is closed")
            if self._listener is None:
                listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                try:
                    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    listener.bind((self.host, self.port))
                    listener.listen()
                    listener.settimeout(self.accept_poll_seconds)
                except BaseException:
                    listener.close()
                    raise
                self._listener = listener
            listener = self._listener
            assert listener is not None
            host, port = listener.getsockname()[:2]
            return str(host), int(port)

    def serve_forever(self, stop_event: object) -> None:
        if not callable(getattr(stop_event, "is_set", None)):
            raise TypeError("stop_event must provide is_set()")
        self.bind()
        try:
            while not stop_event.is_set():
                with self._state_lock:
                    if self._closed:
                        return
                    listener = self._listener
                if listener is None:
                    return
                try:
                    connection, _ = listener.accept()
                except socket.timeout:
                    continue
                except OSError:
                    with self._state_lock:
                        intentional_close = (
                            self._closed or self._listener is not listener
                        )
                    if intentional_close:
                        return
                    raise
                with self._state_lock:
                    if self._closed:
                        accept_after_close = True
                    else:
                        self._active_connection = connection
                        accept_after_close = False
                if accept_after_close:
                    connection.close()
                    return
                self._telemetry.connections += 1
                connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                connection.settimeout(self.limits.io_timeout_seconds)
                try:
                    with connection:
                        self._serve_connection(connection, stop_event)
                except (ResidentExpertRpcError, OSError):
                    with self._state_lock:
                        intentional_close = self._closed
                    if not intentional_close:
                        self._telemetry.errors += 1
                finally:
                    with self._state_lock:
                        if self._active_connection is connection:
                            self._active_connection = None
        finally:
            self.close()

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            listener = self._listener
            connection = self._active_connection
            self._listener = None
            self._active_connection = None
        if connection is not None:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            connection.close()
        if listener is not None:
            listener.close()

    def _serve_connection(self, connection: socket.socket, stop_event: object) -> None:
        frame = _recv_frame(
            connection,
            self.limits,
            allow_clean_eof=True,
            telemetry=self._telemetry,
        )
        if frame is None:
            return
        hello, payload, wire_bytes = frame
        self._telemetry.received(wire_bytes)
        try:
            negotiated = self._accept_handshake(hello, payload)
        except Exception as error:
            self._telemetry.errors += 1
            self._telemetry.sent(
                _send_frame(
                    connection,
                    _error_header(0, "handshake_rejected", str(error)),
                    b"",
                    self.limits,
                )
            )
            return
        response = {
            "inventory": [entry.to_document() for entry in self.inventory],
            "limits": _limits_document(negotiated),
            "nodeId": self.node_id,
            "schema": RESIDENT_EXPERT_RPC_SCHEMA,
            "type": "hello-ok",
        }
        self._telemetry.sent(_send_frame(connection, response, b"", negotiated))
        self._telemetry.handshakes += 1
        connection.settimeout(negotiated.io_timeout_seconds)

        while not stop_event.is_set():
            try:
                frame = _recv_frame(
                    connection,
                    negotiated,
                    allow_clean_eof=True,
                    allow_idle_timeout=True,
                    telemetry=self._telemetry,
                )
            except _IdleTimeout:
                continue
            if frame is None:
                return
            header, payload, wire_bytes = frame
            self._telemetry.received(wire_bytes)
            request_id = header.get("requestId", 0)
            if (
                not isinstance(request_id, int)
                or isinstance(request_id, bool)
                or request_id < 0
            ):
                request_id = 0
            frame_type = header.get("type")
            if frame_type == "close":
                try:
                    _exact_keys(
                        header,
                        {"requestId", "schema", "type"},
                        "RPC close request",
                    )
                    if header["schema"] != RESIDENT_EXPERT_RPC_SCHEMA or payload:
                        raise ResidentExpertRpcContractError("invalid RPC close request")
                    response = {
                        "requestId": request_id,
                        "schema": RESIDENT_EXPERT_RPC_SCHEMA,
                        "type": "close-ok",
                    }
                    self._telemetry.sent(
                        _send_frame(connection, response, b"", negotiated)
                    )
                    self._telemetry.close_round_trips += 1
                    return
                except Exception as error:
                    self._send_error(
                        connection,
                        negotiated,
                        request_id,
                        "invalid_close",
                        error,
                    )
                    return
            if frame_type == "capabilities" and self.enable_coalesced_extension:
                try:
                    _exact_keys(
                        header,
                        {"requestId", "schema", "type"},
                        "RPC capabilities request",
                    )
                    if header["schema"] != RESIDENT_EXPERT_RPC_SCHEMA or payload:
                        raise ResidentExpertRpcContractError(
                            "invalid RPC capabilities request"
                        )
                    response = {
                        "capabilities": [
                            RESIDENT_EXPERT_RPC_COALESCED_CAPABILITY
                        ],
                        "requestId": request_id,
                        "schema": RESIDENT_EXPERT_RPC_SCHEMA,
                        "type": "capabilities-ok",
                    }
                    self._telemetry.sent(
                        _send_frame(connection, response, b"", negotiated)
                    )
                    self._telemetry.capability_round_trips += 1
                except Exception as error:
                    self._send_error(
                        connection,
                        negotiated,
                        request_id,
                        "capability_rejected",
                        error,
                    )
                continue
            if (
                frame_type == "execute-coalesced"
                and self.enable_coalesced_extension
            ):
                try:
                    (
                        response_header,
                        response_payload,
                        item_count,
                        shared_input_bytes,
                        row_index_bytes,
                        v1_equivalent_input_bytes,
                    ) = self._execute_coalesced_batch(
                        header,
                        payload,
                        negotiated,
                    )
                except Exception as error:
                    self._send_error(
                        connection,
                        negotiated,
                        request_id,
                        "coalesced_batch_rejected",
                        error,
                    )
                    self._telemetry.batch_round_trips += 1
                    self._telemetry.coalesced_round_trips += 1
                    continue
                self._telemetry.sent(
                    _send_frame(
                        connection,
                        response_header,
                        response_payload,
                        negotiated,
                    )
                )
                self._telemetry.batch_round_trips += 1
                self._telemetry.batch_items += item_count
                self._telemetry.coalesced_round_trips += 1
                self._telemetry.coalesced_items += item_count
                self._telemetry.coalesced_shared_input_bytes += shared_input_bytes
                self._telemetry.coalesced_row_index_bytes += row_index_bytes
                self._telemetry.coalesced_v1_equivalent_input_bytes += (
                    v1_equivalent_input_bytes
                )
                continue
            if frame_type != "execute-batch":
                self._send_error(
                    connection,
                    negotiated,
                    request_id,
                    "unsupported_request",
                    ResidentExpertRpcContractError("unsupported RPC request type"),
                )
                continue
            try:
                response_header, response_payload, item_count = self._execute_batch(
                    header,
                    payload,
                    negotiated,
                )
            except Exception as error:
                self._send_error(
                    connection,
                    negotiated,
                    request_id,
                    "batch_rejected",
                    error,
                )
                self._telemetry.batch_round_trips += 1
                continue
            self._telemetry.sent(
                _send_frame(
                    connection,
                    response_header,
                    response_payload,
                    negotiated,
                )
            )
            self._telemetry.batch_round_trips += 1
            self._telemetry.batch_items += item_count

    def _accept_handshake(
        self,
        header: Mapping[str, object],
        payload: _Buffer,
    ) -> ResidentExpertRpcLimits:
        _exact_keys(
            header,
            {"clientNodeId", "expectedNodeId", "limits", "schema", "type"},
            "RPC hello request",
        )
        if payload:
            raise ResidentExpertRpcContractError("RPC hello request cannot contain payload")
        if header["schema"] != RESIDENT_EXPERT_RPC_SCHEMA or header["type"] != "hello":
            raise ResidentExpertRpcContractError("RPC hello identity mismatch")
        _identifier("client_node_id", header["clientNodeId"])
        expected = _identifier("expected_node_id", header["expectedNodeId"])
        if expected != self.node_id:
            raise ResidentExpertRpcContractError(
                f"expected node {expected!r}, server is {self.node_id!r}"
            )
        peer_limits = _limits_from_document(header["limits"])
        negotiated = self.limits.negotiated_with(peer_limits)
        if len(self.inventory) > negotiated.max_inventory_items:
            raise ResidentExpertRpcFrameTooLarge(
                "RPC inventory exceeds negotiated client limit"
            )
        return negotiated

    @torch.no_grad()
    def _execute_coalesced_batch(
        self,
        header: Mapping[str, object],
        payload: _Buffer,
        limits: ResidentExpertRpcLimits,
    ) -> tuple[
        dict[str, object],
        tuple[memoryview, ...],
        int,
        int,
        int,
        int,
    ]:
        _exact_keys(
            header,
            {
                "activation",
                "items",
                "layer",
                "requestId",
                "schema",
                "type",
            },
            "RPC coalesced batch request",
        )
        if (
            header["schema"] != RESIDENT_EXPERT_RPC_SCHEMA
            or header["type"] != "execute-coalesced"
        ):
            raise ResidentExpertRpcContractError(
                "RPC coalesced batch identity mismatch"
            )
        request_id = _nonnegative_int("request_id", header["requestId"])
        layer = _nonnegative_int("layer", header["layer"])
        activation = header["activation"]
        if not isinstance(activation, dict):
            raise ResidentExpertRpcContractError(
                "RPC coalesced activation metadata must be an object"
            )
        _exact_keys(
            activation,
            {"dtype", "nbytes", "offset", "shape"},
            "RPC coalesced activation metadata",
        )
        shared_shape = _shape_from_document(activation["shape"])
        dtype_name = _identifier("dtype", activation["dtype"])
        if dtype_name not in _DTYPE_BY_NAME:
            raise ResidentExpertRpcContractError(
                f"unsupported coalesced activation dtype {dtype_name!r}"
            )
        activation_offset = _nonnegative_int("activation offset", activation["offset"])
        shared_nbytes = _nonnegative_int("activation nbytes", activation["nbytes"])
        expected_shared_nbytes = (
            math.prod(shared_shape)
            * torch.empty((), dtype=_DTYPE_BY_NAME[dtype_name]).element_size()
        )
        if (
            activation_offset != 0
            or shared_nbytes != expected_shared_nbytes
            or shared_nbytes > len(payload)
        ):
            raise ResidentExpertRpcContractError(
                "RPC coalesced shared activation byte length mismatch"
            )
        if math.prod(shared_shape) > limits.max_tensor_elements:
            raise ResidentExpertRpcFrameTooLarge(
                "RPC coalesced shared activations exceed element limit"
            )

        items = header["items"]
        if not isinstance(items, list) or not items:
            raise ResidentExpertRpcContractError(
                "RPC coalesced batch items must be non-empty"
            )
        if len(items) > limits.max_batch_items:
            raise ResidentExpertRpcFrameTooLarge(
                "RPC coalesced batch exceeds item limit"
            )
        decoded_meta: list[
            tuple[
                ExpertKey,
                str,
                bool,
                int,
                int,
                ResidentExpertRpcInventoryEntry,
            ]
        ] = []
        request_meta: dict[ExpertKey, tuple[int, ResidentExpertRpcInventoryEntry]] = {}
        assignment_count = 0
        cursor = shared_nbytes
        prior_key: ExpertKey | None = None
        for item in items:
            if not isinstance(item, dict):
                raise ResidentExpertRpcContractError(
                    "RPC coalesced batch item must be an object"
                )
            _exact_keys(
                item,
                {
                    "contentId",
                    "expert",
                    "layer",
                    "requireResident",
                    "rowCount",
                    "rowNbytes",
                    "rowOffset",
                },
                "RPC coalesced batch item",
            )
            key = ExpertKey(
                _nonnegative_int("item layer", item["layer"]),
                _nonnegative_int("expert", item["expert"]),
            )
            if key.layer != layer:
                raise ResidentExpertRpcContractError(
                    "RPC coalesced batch crosses layer boundary"
                )
            if prior_key is not None and key <= prior_key:
                raise ResidentExpertRpcContractError(
                    "RPC coalesced item order is not canonical"
                )
            prior_key = key
            entry = self._inventory.get(key)
            content_id = _identifier("content_id", item["contentId"])
            if entry is None or entry.content_id != content_id:
                raise ExpertRouteUnavailableError(
                    f"RPC owner {self.node_id!r} lacks exact content for {key}"
                )
            if entry.input_width != shared_shape[1] or entry.dtype != dtype_name:
                raise ResidentExpertRpcContractError(
                    f"RPC shared activation metadata mismatch for {key}"
                )
            if not self.owner.has_expert(key, content_id):
                raise ExpertResidentSlotUnavailableError(
                    key,
                    f"RPC owner implementation lost exact content for {key}"
                )
            # Inventory is a resident routing promise. Revalidate it for every
            # request before tensor materialization, even if a malformed client
            # sends requireResident=false; the flag is still preserved for the
            # owner implementation rather than silently rewritten.
            if not self.owner.is_expert_resident(key, content_id):
                raise ExpertResidentSlotUnavailableError(
                    key,
                    f"RPC owner implementation lost resident content for {key}"
                )
            require_resident = item["requireResident"]
            if not isinstance(require_resident, bool):
                raise ResidentExpertRpcContractError(
                    "RPC coalesced requireResident must be boolean"
                )
            if not require_resident:
                raise ResidentExpertRpcContractError(
                    "RPC coalesced resident inventory cannot request RAM fallback"
                )
            row_count = _positive_int("row count", item["rowCount"])
            row_offset = _nonnegative_int("row offset", item["rowOffset"])
            row_nbytes = _nonnegative_int("row nbytes", item["rowNbytes"])
            if (
                row_offset != cursor
                or row_nbytes != row_count * 4
                or row_nbytes > len(payload) - row_offset
            ):
                raise ResidentExpertRpcContractError(
                    f"RPC row-index metadata mismatch for {key}"
                )
            assignment_count += row_count
            if assignment_count > limits.max_tensor_elements:
                raise ResidentExpertRpcFrameTooLarge(
                    "RPC coalesced row references exceed element limit"
                )
            decoded_meta.append(
                (
                    key,
                    content_id,
                    require_resident,
                    row_count,
                    row_offset,
                    entry,
                )
            )
            request_meta[key] = (row_count, entry)
            cursor += row_nbytes
        if cursor != len(payload):
            raise ResidentExpertRpcContractError(
                "RPC coalesced batch payload has trailing bytes"
            )

        element_size = torch.empty(
            (),
            dtype=_DTYPE_BY_NAME[dtype_name],
        ).element_size()
        row_materialization_bytes = (
            assignment_count * _ROW_REF_DECODE_BYTES_PER_ASSIGNMENT
        )
        coverage_bitmap_bytes = _coverage_bitmap_bytes(shared_shape[0])
        projected_output_bytes = sum(
            row_count * entry.output_width * element_size
            for row_count, entry in request_meta.values()
        )
        required_host_transient = max(
            len(_canonical_header(header))
            + 2 * len(payload)
            + row_materialization_bytes
            + coverage_bitmap_bytes,
            shared_nbytes
            + row_materialization_bytes
            + projected_output_bytes,
        )
        if required_host_transient > limits.max_host_transient_bytes:
            self._telemetry.host_transient_rejected()
            raise ResidentExpertRpcFrameTooLarge(
                "RPC coalesced frame requires at least "
                f"{required_host_transient} host-transient bytes including "
                "decoded row references; limit is "
                f"{limits.max_host_transient_bytes}"
            )
        self._telemetry.host_transient_accepted(required_host_transient)

        # Only after the whole metadata table and its decoded-memory budget pass
        # do we materialize Python row tuples. Malformed late items cannot leave
        # an earlier partial owner batch.
        decoded_items: list[OwnerCoalescedExpertBatchItem] = []
        covered_rows = bytearray(shared_shape[0])
        covered_count = 0
        for (
            key,
            content_id,
            require_resident,
            row_count,
            row_offset,
            _entry,
        ) in decoded_meta:
            row_indices = _uint32_from_payload(
                payload,
                offset=row_offset,
                count=row_count,
            )
            if any(row >= shared_shape[0] for row in row_indices):
                raise ResidentExpertRpcContractError(
                    f"RPC row reference exceeds shared activations for {key}"
                )
            for row in row_indices:
                covered_count += _mark_covered_row(covered_rows, row)
            decoded_items.append(
                OwnerCoalescedExpertBatchItem(
                    key,
                    content_id,
                    row_indices,
                    require_resident=require_resident,
                )
            )
        if covered_count != shared_shape[0]:
            raise ResidentExpertRpcContractError(
                "RPC coalesced shared activations do not have exact row coverage"
            )
        del covered_rows

        # All identity, residency, row-map and aggregate-coverage checks have
        # passed. Only now clone the one shared tensor and invoke the owner.
        shared = _tensor_from_payload(
            payload,
            offset=0,
            nbytes=shared_nbytes,
            shape=shared_shape,
            dtype_name=dtype_name,
        )
        if isinstance(payload, bytearray):
            # _recv_frame owns this mutable wire buffer and every tensor/row
            # view above has now been cloned. Release it before owner outputs
            # are allocated so request wire + response cannot overlap.
            payload.clear()
        coalesced_execute = getattr(self.owner, "execute_coalesced_batch", None)
        if callable(coalesced_execute):
            results = coalesced_execute(shared, tuple(decoded_items))
        else:
            fallback_items = tuple(
                OwnerExpertBatchItem(
                    item.key,
                    item.content_id,
                    shared.index_select(
                        0,
                        torch.tensor(item.row_indices, dtype=torch.long),
                    ),
                    require_resident=item.require_resident,
                )
                for item in decoded_items
            )
            results = self.owner.execute_batch(fallback_items)
        response_header, response_payload = self._encode_batch_results(
            request_id=request_id,
            response_type="execute-coalesced-ok",
            results=results,
            request_meta=request_meta,
            limits=limits,
        )
        row_index_bytes = assignment_count * 4
        v1_equivalent_input_bytes = (
            assignment_count
            * shared_shape[1]
            * torch.empty((), dtype=_DTYPE_BY_NAME[dtype_name]).element_size()
        )
        return (
            response_header,
            response_payload,
            len(decoded_items),
            shared_nbytes,
            row_index_bytes,
            v1_equivalent_input_bytes,
        )

    @torch.no_grad()
    def _execute_batch(
        self,
        header: Mapping[str, object],
        payload: _Buffer,
        limits: ResidentExpertRpcLimits,
    ) -> tuple[dict[str, object], tuple[memoryview, ...], int]:
        _exact_keys(
            header,
            {"items", "layer", "requestId", "schema", "type"},
            "RPC batch request",
        )
        if header["schema"] != RESIDENT_EXPERT_RPC_SCHEMA:
            raise ResidentExpertRpcContractError("RPC batch schema mismatch")
        request_id = _nonnegative_int("request_id", header["requestId"])
        layer = _nonnegative_int("layer", header["layer"])
        items = header["items"]
        if not isinstance(items, list) or not items:
            raise ResidentExpertRpcContractError("RPC batch items must be non-empty")
        if len(items) > limits.max_batch_items:
            raise ResidentExpertRpcFrameTooLarge("RPC batch exceeds item limit")

        decoded_meta: list[
            tuple[ExpertKey, str, tuple[int, int], str, int, int]
        ] = []
        request_meta: dict[ExpertKey, tuple[int, ResidentExpertRpcInventoryEntry]] = {}
        cursor = 0
        prior_key: ExpertKey | None = None
        for item in items:
            if not isinstance(item, dict):
                raise ResidentExpertRpcContractError("RPC batch item must be an object")
            _exact_keys(
                item,
                {
                    "contentId",
                    "dtype",
                    "expert",
                    "layer",
                    "nbytes",
                    "offset",
                    "shape",
                },
                "RPC batch item",
            )
            key = ExpertKey(
                _nonnegative_int("item layer", item["layer"]),
                _nonnegative_int("expert", item["expert"]),
            )
            if key.layer != layer:
                raise ResidentExpertRpcContractError("RPC batch crosses layer boundary")
            if prior_key is not None and key <= prior_key:
                raise ResidentExpertRpcContractError("RPC batch item order is not canonical")
            prior_key = key
            entry = self._inventory.get(key)
            content_id = _identifier("content_id", item["contentId"])
            if entry is None or entry.content_id != content_id:
                raise ExpertRouteUnavailableError(
                    f"RPC owner {self.node_id!r} lacks exact content for {key}"
                )
            if not self.owner.has_expert(key, content_id):
                raise ExpertResidentSlotUnavailableError(
                    key,
                    f"RPC owner implementation lost exact content for {key}"
                )
            # Handshake residence is a sealed routing promise, but physical
            # residency may become stale. Revalidate it for every execution
            # before materializing any activation or invoking the owner.
            if not self.owner.is_expert_resident(key, content_id):
                raise ExpertResidentSlotUnavailableError(
                    key,
                    f"RPC owner implementation lost resident content for {key}"
                )
            shape = _shape_from_document(item["shape"])
            dtype_name = _identifier("dtype", item["dtype"])
            offset = _nonnegative_int("offset", item["offset"])
            nbytes = _nonnegative_int("nbytes", item["nbytes"])
            if (
                shape[0] < 1
                or shape[1] != entry.input_width
                or dtype_name != entry.dtype
                or offset != cursor
            ):
                raise ResidentExpertRpcContractError(
                    f"RPC activation metadata mismatch for {key}"
                )
            if math.prod(shape) > limits.max_tensor_elements:
                raise ResidentExpertRpcFrameTooLarge("RPC activation exceeds element limit")
            expected_nbytes = (
                math.prod(shape)
                * torch.empty((), dtype=_DTYPE_BY_NAME[dtype_name]).element_size()
            )
            if nbytes != expected_nbytes or nbytes > len(payload) - offset:
                raise ResidentExpertRpcContractError(
                    f"RPC activation byte length mismatch for {key}"
                )
            decoded_meta.append(
                (
                    key,
                    content_id,
                    shape,
                    dtype_name,
                    offset,
                    nbytes,
                )
            )
            request_meta[key] = (shape[0], entry)
            cursor += nbytes
        if cursor != len(payload):
            raise ResidentExpertRpcContractError("RPC batch payload has trailing bytes")

        # Metadata, identity, residence and aggregate coverage all pass before
        # any large tensor allocation is made from the receive buffer.
        decoded = [
            OwnerExpertBatchItem(
                key,
                content_id,
                _tensor_from_payload(
                    payload,
                    offset=offset,
                    nbytes=nbytes,
                    shape=shape,
                    dtype_name=dtype_name,
                ),
                # Every RPC inventory entry is a sealed resident promise. Force
                # the stricter owner path without adding a wire-schema field.
                require_resident=True,
            )
            for (
                key,
                content_id,
                shape,
                dtype_name,
                offset,
                nbytes,
            ) in decoded_meta
        ]
        if isinstance(payload, bytearray):
            # The per-item tensors are independent clones. Free the receive
            # frame before expert compute/output allocation (see coalesced path).
            payload.clear()

        # One owner call for the complete layer batch. Nothing is serialized
        # until every result has passed identity and shape validation.
        results = self.owner.execute_batch(tuple(decoded))
        response_header, response_payload = self._encode_batch_results(
            request_id=request_id,
            response_type="execute-batch-ok",
            results=results,
            request_meta=request_meta,
            limits=limits,
        )
        return response_header, response_payload, len(decoded)

    def _encode_batch_results(
        self,
        *,
        request_id: int,
        response_type: str,
        results: Sequence[OwnerExpertBatchResult],
        request_meta: Mapping[
            ExpertKey,
            tuple[int, ResidentExpertRpcInventoryEntry],
        ],
        limits: ResidentExpertRpcLimits,
    ) -> tuple[dict[str, object], tuple[memoryview, ...]]:
        if len(results) != len(request_meta):
            raise ResidentExpertRpcContractError("owner returned an incomplete batch")
        by_key: dict[ExpertKey, torch.Tensor] = {}
        for result in results:
            if not isinstance(result, OwnerExpertBatchResult):
                raise TypeError("owner results must be OwnerExpertBatchResult values")
            if result.key in by_key or result.key not in request_meta:
                raise ResidentExpertRpcContractError("owner returned an invalid batch key")
            by_key[result.key] = result.output
        if set(by_key) != set(request_meta):
            raise ResidentExpertRpcContractError("owner omitted a batch result")

        response_items: list[dict[str, object]] = []
        prepared_outputs: list[tuple[torch.Tensor, int]] = []
        cursor = 0
        for key in sorted(request_meta):
            row_count, entry = request_meta[key]
            output = by_key[key]
            expected_shape = (row_count, entry.output_width)
            if (
                not isinstance(output, torch.Tensor)
                or output.layout != torch.strided
            ):
                raise ResidentExpertRpcContractError(
                    f"owner output for {key} must use strided Torch layout"
                )
            if tuple(output.shape) != expected_shape or _dtype_name(output) != entry.dtype:
                raise ResidentExpertRpcContractError(
                    f"owner output metadata mismatch for {key}"
                )
            if output.numel() > limits.max_tensor_elements:
                raise ResidentExpertRpcFrameTooLarge("RPC output exceeds element limit")
            nbytes = output.numel() * output.element_size()
            if nbytes > limits.max_payload_bytes - cursor:
                raise ResidentExpertRpcFrameTooLarge(
                    "RPC batch response exceeds aggregate payload limit"
                )
            response_items.append(
                {
                    "contentId": entry.content_id,
                    "dtype": entry.dtype,
                    "expert": key.expert,
                    "layer": key.layer,
                    "nbytes": nbytes,
                    "offset": cursor,
                    "shape": list(expected_shape),
                }
            )
            prepared_outputs.append((output, nbytes))
            cursor += nbytes

        response_header = {
            "items": response_items,
            "requestId": request_id,
            "schema": RESIDENT_EXPERT_RPC_SCHEMA,
            "type": response_type,
        }
        _preflight_host_transient(
            header_size=len(_canonical_header(response_header)),
            payload_size=cursor,
            limits=limits,
            telemetry=None,
        )
        # Avoid a second payload-sized join: each contiguous CPU tensor is sent
        # directly as a buffer view after the whole response passes preflight.
        response_payload: list[memoryview] = []
        for output, nbytes in prepared_outputs:
            _cpu, raw = _tensor_payload(output)
            if len(raw) != nbytes:
                raise ResidentExpertRpcContractError(
                    "RPC owner output changed during serialization"
                )
            response_payload.append(raw)
        return response_header, tuple(response_payload)

    def _send_error(
        self,
        connection: socket.socket,
        limits: ResidentExpertRpcLimits,
        request_id: int,
        code: str,
        error: Exception,
    ) -> None:
        message = str(error)
        if isinstance(error, ExpertResidentSlotUnavailableError):
            code = "resident_slot_unavailable"
            message = _resident_slot_message(error.key)
        self._telemetry.errors += 1
        self._telemetry.sent(
            _send_frame(
                connection,
                _error_header(request_id, code, message),
                b"",
                limits,
            )
        )


def _shape_from_document(value: object) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise ResidentExpertRpcContractError("RPC tensor shape must have rank two")
    try:
        return (
            _positive_int("tensor rows", value[0]),
            _positive_int("tensor width", value[1]),
        )
    except ValueError as error:
        raise ResidentExpertRpcContractError(str(error)) from error


__all__ = [
    "RESIDENT_EXPERT_RPC_COALESCED_CAPABILITY",
    "RESIDENT_EXPERT_RPC_SCHEMA",
    "ResidentExpertRpcClient",
    "ResidentExpertRpcContractError",
    "ResidentExpertRpcError",
    "ResidentExpertRpcFrameTooLarge",
    "ResidentExpertRpcInventoryEntry",
    "ResidentExpertRpcLimits",
    "ResidentExpertRpcRemoteError",
    "ResidentExpertRpcResidentSlotUnavailable",
    "ResidentExpertRpcServer",
    "ResidentExpertRpcTelemetry",
    "ResidentExpertRpcTimeout",
    "ResidentExpertRpcTruncatedFrame",
]
