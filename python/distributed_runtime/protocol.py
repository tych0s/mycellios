from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import IntEnum
import math
import socket
import struct
import threading
import time
from typing import Callable
import zlib

import torch

MAGIC = b"GDLP"
# Transactional sparse-tree capacity quotes extend the v4 frame vocabulary.
# v6 adds an end-to-end COMMIT result: the root cannot emit FORK until the last
# stage proves that every preceding stage consumed and revalidated COMMIT.
# Mixed deployments therefore fail during HELLO instead of silently weakening
# the all-stage mutation barrier.
VERSION = 6
HEADER = struct.Struct("<4sBBHQIIII")
HEADER_BYTES = HEADER.size
MAX_PAYLOAD_BYTES = 64 * 1024 * 1024
QUANT_GROUP_SIZE = 64
UINT16_MAX = (1 << 16) - 1
UINT32_MAX = (1 << 32) - 1
UINT64_MAX = (1 << 64) - 1
MAX_TREE_PREPARE_PATHS = 64
MAX_TREE_PREPARE_PATH_TOKENS = 1_048_575


class FrameType(IntEnum):
    HELLO = 1
    READY = 2
    BEGIN = 3
    ACTIVATION = 4
    TOKEN = 5
    END = 6
    CANCEL = 7
    ERROR = 8
    SHUTDOWN = 9
    # Non-final prefill chunks flow through the same tensor data plane but return
    # a lightweight ACK instead of projecting logits.  This lets long prompts be
    # interleaved with decode without materialising a second protocol.
    PREFILL = 10
    PREFILL_ACK = 11
    # TRUNCATE is ordered on the stage stream and crops every local KV cache to
    # token_count before the following activation can overtake it.
    TRUNCATE = 12
    # VERIFY carries a multi-position target wave.  The last stage returns the
    # greedy target token for every position so the root can accept a draft prefix.
    VERIFY = 13
    VERIFY_RESULT = 14
    # PING follows the forward stage chain and the last stage returns PONG over
    # the direct-return socket. The elapsed time measures the same no-compute
    # route used by a decode wave, rather than one arbitrary peer-to-peer link.
    PING = 15
    PONG = 16
    # Exact speculative trees use independent request IDs as virtual leaves.
    # FORK(child,parent) clones the parent's stage-local KV state into `child`;
    # PROMOTE(parent,child) atomically moves the selected child's state back to
    # `parent`. The second uint64 request ID is the only payload field.
    FORK = 17
    PROMOTE = 18
    # TREE_PREPARE is a mutation-free capacity quote. Every stage appends its
    # result to the bounded binary aggregate and the final stage returns
    # TREE_PREPARE_RESULT on the direct-return socket. COMMIT revalidates and
    # arms the quote; CANCEL releases it without allocating KV.
    TREE_PREPARE = 19
    TREE_PREPARE_RESULT = 20
    TREE_RESERVATION_COMMIT = 21
    TREE_RESERVATION_CANCEL = 22
    TREE_RESERVATION_COMMIT_RESULT = 23


class TreePrepareStatus(IntEnum):
    READY = 0
    REJECT = 1


class TreePrepareRejection(IntEnum):
    NONE = 0
    TREE_DISABLED = 1
    RESERVATION_BUSY = 2
    BRANCH_COUNT = 3
    BRANCH_TOKENS = 4
    KV_BYTES = 5


class TensorCodec(IntEnum):
    FP32 = 0
    FP16 = 1
    INT8 = 2
    INT8_GROUPED = 3
    INT8_HADAMARD = 4
    # Opt-in variants: the complete grouped payload (scales + int8 data) is
    # wrapped with zlib level 1.  Framing is unchanged; only the payload size
    # on the wire becomes data-dependent.
    INT8_GROUPED_DEFLATE = 5
    INT8_HADAMARD_DEFLATE = 6


_DEFLATE_BASE = {
    TensorCodec.INT8_GROUPED_DEFLATE: TensorCodec.INT8_GROUPED,
    TensorCodec.INT8_HADAMARD_DEFLATE: TensorCodec.INT8_HADAMARD,
}

TENSOR_FRAME_TYPES = frozenset(
    (FrameType.ACTIVATION, FrameType.PREFILL, FrameType.VERIFY)
)
ROUTE_PROBE_FRAME_TYPES = frozenset((FrameType.PING, FrameType.PONG))
BRANCH_CONTROL_FRAME_TYPES = frozenset((FrameType.FORK, FrameType.PROMOTE))
BRANCH_REQUEST_ID = struct.Struct("<Q")
TREE_PREPARE_FRAME_TYPES = frozenset(
    (FrameType.TREE_PREPARE, FrameType.TREE_PREPARE_RESULT)
)
TREE_RESERVATION_COMMIT_FRAME_TYPES = frozenset(
    (FrameType.TREE_RESERVATION_COMMIT,)
)
TREE_RESERVATION_CANCEL_FRAME_TYPES = frozenset(
    (FrameType.TREE_RESERVATION_CANCEL,)
)
TREE_RESERVATION_RESULT_FRAME_TYPES = frozenset(
    (FrameType.TREE_RESERVATION_COMMIT_RESULT,)
)
TREE_RESERVATION_FRAME_TYPES = (
    TREE_RESERVATION_COMMIT_FRAME_TYPES
    | TREE_RESERVATION_CANCEL_FRAME_TYPES
    | TREE_RESERVATION_RESULT_FRAME_TYPES
)
# nonce, status, rejection, visited stage count, rejecting layer start,
# required capacity, configured limit and sum of all visited-stage projections.
TREE_PREPARE_PREFIX = struct.Struct("<QBBHIQQQ")
TREE_PATH_LENGTH = struct.Struct("<I")
TREE_RESERVATION_NONCE = struct.Struct("<Q")


@dataclass(frozen=True)
class Frame:
    frame_type: FrameType
    flags: int
    request_id: int
    step: int
    token_count: int
    hidden_size: int
    payload: bytes | bytearray


@dataclass(frozen=True)
class TreePrepareQuote:
    """Closed, bounded sparse-tree capacity quote carried across the route."""

    nonce: int
    path_lengths: tuple[int, ...]
    status: TreePrepareStatus = TreePrepareStatus.READY
    rejection: TreePrepareRejection = TreePrepareRejection.NONE
    stage_count: int = 0
    rejecting_layer_start: int = 0
    required: int = 0
    limit: int = 0
    total_projected_bytes: int = 0


@dataclass(frozen=True)
class EncodedTensorPayload:
    """A bytes view that keeps its tensor/bytes backing storage alive.

    The original data plane always materialised a Python ``bytes`` object after
    first converting every activation to CPU FP32.  That doubled host traffic
    for the common FP16 wire codec and added another full payload copy before
    ``sendall``.  This owner-backed view lets the synchronous socket write read
    directly from a contiguous CPU tensor.  Quantized/deflated codecs retain
    their existing byte-exact implementation until a certified GPU codec exists.
    """

    view: memoryview
    owner: object
    staging_dtype: torch.dtype | None
    source_device: str

    @property
    def nbytes(self) -> int:
        return self.view.nbytes


class LinkEmulatorError(RuntimeError):
    """An asynchronous emulated-link send failed or could not be drained."""


@dataclass(frozen=True)
class _ScheduledLinkFrame:
    sequence: int
    sock: socket.socket
    header: bytes
    payload: bytes
    wire_bytes: int
    deliver_at: float


class LinkEmulator:
    """A bounded, ordered store-and-forward link emulator.

    Bandwidth reserves a serialization interval for every frame. Propagation
    starts when that interval ends, so several already-serialized frames can be
    in flight at once. A single owned worker writes due frames to the TCP socket
    in FIFO order; there is never one sleeping thread per frame.

    ``send_frame`` remains synchronous when the emulator is disabled (both
    values are zero). With an active emulator it queues an immutable snapshot
    and returns after admission. Call :meth:`flush` at a lifecycle boundary
    when the caller must observe a background socket error before proceeding.
    The worker retires automatically after an idle interval, while
    :meth:`close` provides deterministic ownership for tests and long-lived
    runtimes.
    """

    def __init__(
        self,
        one_way_delay_ms: float = 0.0,
        bandwidth_mbps: float = 0.0,
        *,
        max_queued_frames: int = 64,
        max_queued_bytes: int = MAX_PAYLOAD_BYTES + HEADER_BYTES,
        enqueue_timeout_seconds: float | None = 30.0,
        idle_worker_seconds: float = 1.0,
        clock: Callable[[], float] | None = None,
    ) -> None:
        delay = float(one_way_delay_ms)
        bandwidth = float(bandwidth_mbps)
        if not math.isfinite(delay) or not math.isfinite(bandwidth):
            raise ValueError("link delay and bandwidth must be finite")
        if max_queued_frames <= 0:
            raise ValueError("max_queued_frames must be positive")
        if max_queued_bytes < HEADER_BYTES:
            raise ValueError(f"max_queued_bytes must be at least {HEADER_BYTES} bytes")
        if enqueue_timeout_seconds is not None and enqueue_timeout_seconds < 0:
            raise ValueError("enqueue_timeout_seconds must be non-negative")
        if not math.isfinite(idle_worker_seconds) or idle_worker_seconds <= 0:
            raise ValueError("idle_worker_seconds must be positive and finite")

        self.one_way_delay_ms = max(0.0, delay)
        self.bandwidth_mbps = max(0.0, bandwidth)
        self.max_queued_frames = int(max_queued_frames)
        self.max_queued_bytes = int(max_queued_bytes)
        self.enqueue_timeout_seconds = enqueue_timeout_seconds
        self.idle_worker_seconds = float(idle_worker_seconds)
        self._clock = clock or time.monotonic
        self._condition = threading.Condition()
        self._queue: deque[_ScheduledLinkFrame] = deque()
        self._queued_bytes = 0
        self._next_serial_finish = self._clock()
        self._submitted_sequence = 0
        self._completed_sequence = 0
        self._inflight: _ScheduledLinkFrame | None = None
        self._bound_socket: socket.socket | None = None
        self._worker: threading.Thread | None = None
        self._closed = False
        self._failure: BaseException | None = None

    @property
    def enabled(self) -> bool:
        return self.one_way_delay_ms > 0 or self.bandwidth_mbps > 0

    @property
    def pending_frames(self) -> int:
        with self._condition:
            return len(self._queue) + (1 if self._inflight is not None else 0)

    @property
    def pending_bytes(self) -> int:
        with self._condition:
            inflight = self._inflight.wire_bytes if self._inflight else 0
            return self._queued_bytes + inflight

    @property
    def propagation_seconds(self) -> float:
        """Return the parallel propagation component of the link model."""

        return self.one_way_delay_ms / 1_000

    def serialization_seconds(self, wire_bytes: int) -> float:
        """Return the serial wire-time component for one complete frame."""

        if wire_bytes < 0:
            raise ValueError("wire_bytes must be non-negative")
        if self.bandwidth_mbps <= 0:
            return 0.0
        return (wire_bytes * 8) / (self.bandwidth_mbps * 1_000_000)

    def single_frame_delay_seconds(self, wire_bytes: int) -> float:
        return self.propagation_seconds + self.serialization_seconds(wire_bytes)

    def wait_before_send(self, wire_bytes: int) -> None:
        """Reproduce the retired serial-delay model for historical benchmarks.

        Production sends must use :func:`send_frame`, which queues propagation
        asynchronously. This compatibility shim exists only so the versioned
        benchmark can compare the former behaviour against the corrected one.
        """

        delay = self.single_frame_delay_seconds(wire_bytes)
        if delay > 0:
            time.sleep(delay)

    def send(
        self,
        sock: socket.socket,
        header: bytes,
        payload: bytes | bytearray | memoryview,
    ) -> None:
        """Queue one validated frame while preserving a bounded snapshot."""

        with self._condition:
            self._raise_failure_locked()
            if self._closed:
                raise LinkEmulatorError("emulated link is closed")
        payload_bytes = (
            payload.nbytes if isinstance(payload, memoryview) else len(payload)
        )
        wire_bytes = len(header) + payload_bytes
        if wire_bytes > self.max_queued_bytes:
            raise ValueError(
                "frame exceeds emulated link queue byte capacity "
                f"({wire_bytes} > {self.max_queued_bytes})"
            )
        owner, newly_claimed = _claim_link_emulator_socket(sock, self)
        if owner is not self:
            # Once one sender owns a TCP stream every later frame must enter
            # that same FIFO, including control frames whose call site does
            # not explicitly pass the emulator. Otherwise an inline END or
            # SHUTDOWN could overtake a delayed ACTIVATION.
            owner.send(sock, header, payload)
            return
        deadline = (
            None
            if self.enqueue_timeout_seconds is None
            else time.monotonic() + self.enqueue_timeout_seconds
        )
        try:
            with self._condition:
                self._raise_failure_locked()
                if self._closed:
                    raise LinkEmulatorError("emulated link is closed")
                if self._bound_socket is None:
                    self._bound_socket = sock
                elif self._bound_socket is not sock:
                    raise LinkEmulatorError(
                        "one LinkEmulator instance cannot own multiple TCP sockets"
                    )
                while (
                    len(self._queue) + (1 if self._inflight is not None else 0)
                    >= self.max_queued_frames
                    or self._queued_bytes
                    + (
                        self._inflight.wire_bytes
                        if self._inflight is not None
                        else 0
                    )
                    + wire_bytes
                    > self.max_queued_bytes
                ):
                    self._raise_failure_locked()
                    if self._closed:
                        raise LinkEmulatorError(
                            "emulated link closed while queueing"
                        )
                    remaining = (
                        None if deadline is None else deadline - time.monotonic()
                    )
                    if remaining is not None and remaining <= 0:
                        raise LinkEmulatorError(
                            "timed out waiting for bounded emulated-link queue capacity"
                        )
                    self._condition.wait(timeout=remaining)

                # Snapshot mutable payloads before returning from send_frame.
                payload_snapshot = (
                    payload if isinstance(payload, bytes) else bytes(payload)
                )
                now = self._clock()
                serial_start = max(now, self._next_serial_finish)
                serialization = 0.0
                if self.bandwidth_mbps > 0:
                    serialization = (
                        wire_bytes * 8
                    ) / (self.bandwidth_mbps * 1_000_000)
                serial_finish = serial_start + serialization
                self._next_serial_finish = serial_finish
                self._submitted_sequence += 1
                self._queue.append(
                    _ScheduledLinkFrame(
                        sequence=self._submitted_sequence,
                        sock=sock,
                        header=header,
                        payload=payload_snapshot,
                        wire_bytes=wire_bytes,
                        deliver_at=(
                            serial_finish + self.one_way_delay_ms / 1_000
                        ),
                    )
                )
                self._queued_bytes += wire_bytes
                self._ensure_worker_locked()
                self._condition.notify_all()
        except BaseException:
            # A rejected first frame must not poison this socket: otherwise a
            # later lifecycle frame without an explicit emulator would be
            # redirected into an owner that never admitted any work.
            if newly_claimed:
                with self._condition:
                    admitted = (
                        self._bound_socket is sock
                        and (
                            self._submitted_sequence > self._completed_sequence
                            or self._inflight is not None
                            or bool(self._queue)
                        )
                    )
                    if not admitted and self._bound_socket is sock:
                        self._bound_socket = None
                if not admitted:
                    _release_link_emulator_socket(sock, self)
            raise

    def flush(self, timeout_seconds: float | None = None) -> None:
        """Wait until every frame admitted before this call is on the socket."""

        if timeout_seconds is not None and timeout_seconds < 0:
            raise ValueError("timeout_seconds must be non-negative")
        deadline = (
            None if timeout_seconds is None else time.monotonic() + timeout_seconds
        )
        with self._condition:
            target = self._submitted_sequence
            while self._completed_sequence < target:
                self._raise_failure_locked()
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise LinkEmulatorError("timed out draining emulated-link frames")
                self._condition.wait(timeout=remaining)
            self._raise_failure_locked()

    def close(
        self,
        *,
        drain: bool = True,
        timeout_seconds: float | None = None,
    ) -> None:
        """Close the owned sender, optionally discarding queued frames."""

        if timeout_seconds is not None and timeout_seconds < 0:
            raise ValueError("timeout_seconds must be non-negative")
        with self._condition:
            self._closed = True
            aborted_socket = None
            if not drain:
                self._queue.clear()
                self._queued_bytes = 0
                aborted_socket = (
                    self._inflight.sock
                    if self._inflight is not None
                    else self._bound_socket
                )
            self._condition.notify_all()
        if aborted_socket is not None:
            try:
                aborted_socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        if drain:
            self.flush(timeout_seconds=timeout_seconds)

        with self._condition:
            worker = self._worker
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=timeout_seconds)
            if worker.is_alive():
                raise LinkEmulatorError("timed out stopping emulated-link sender")
        with self._condition:
            self._raise_failure_locked()
        _release_link_emulator_socket(self._bound_socket, self)
        self._bound_socket = None

    def notify_clock_advanced(self) -> None:
        """Wake the worker after advancing an injected deterministic clock."""

        with self._condition:
            self._condition.notify_all()

    def __enter__(self) -> LinkEmulator:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _ensure_worker_locked(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._worker = threading.Thread(
            target=self._run_sender,
            name=f"mycellios-link-sender-{id(self):x}",
            daemon=True,
        )
        self._worker.start()

    def _run_sender(self) -> None:
        while True:
            with self._condition:
                item: _ScheduledLinkFrame | None = None
                while item is None:
                    if self._failure is not None:
                        self._worker = None
                        self._condition.notify_all()
                        return
                    if self._closed and not self._queue:
                        self._worker = None
                        self._condition.notify_all()
                        return
                    if not self._queue:
                        notified = self._condition.wait(
                            timeout=self.idle_worker_seconds
                        )
                        if not notified and not self._queue and not self._closed:
                            self._worker = None
                            self._condition.notify_all()
                            return
                        continue
                    candidate = self._queue[0]
                    remaining = candidate.deliver_at - self._clock()
                    if remaining > 0:
                        self._condition.wait(timeout=remaining)
                        continue
                    item = self._queue.popleft()
                    self._queued_bytes -= item.wire_bytes
                    self._inflight = item
                    self._condition.notify_all()

            try:
                item.sock.sendall(item.header)
                if item.payload:
                    item.sock.sendall(item.payload)
            except BaseException as error:
                with self._condition:
                    self._failure = error
                    self._queue.clear()
                    self._queued_bytes = 0
                    self._inflight = None
                    self._worker = None
                    self._condition.notify_all()
                try:
                    item.sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                return

            with self._condition:
                self._completed_sequence = item.sequence
                self._inflight = None
                self._condition.notify_all()

    def _raise_failure_locked(self) -> None:
        if self._failure is not None:
            raise LinkEmulatorError("emulated-link sender failed") from self._failure


_LINK_EMULATOR_OWNERS: dict[object, LinkEmulator] = {}
_LINK_EMULATOR_OWNERS_LOCK = threading.Lock()


def _claim_link_emulator_socket(
    sock: object,
    emulator: LinkEmulator,
) -> tuple[LinkEmulator, bool]:
    with _LINK_EMULATOR_OWNERS_LOCK:
        owner = _LINK_EMULATOR_OWNERS.get(sock)
        if owner is None:
            _LINK_EMULATOR_OWNERS[sock] = emulator
            return emulator, True
        return owner, False


def _owned_link_emulator(sock: object) -> LinkEmulator | None:
    with _LINK_EMULATOR_OWNERS_LOCK:
        return _LINK_EMULATOR_OWNERS.get(sock)


def _release_link_emulator_socket(
    sock: object | None,
    emulator: LinkEmulator,
) -> None:
    if sock is None:
        return
    with _LINK_EMULATOR_OWNERS_LOCK:
        if _LINK_EMULATOR_OWNERS.get(sock) is emulator:
            del _LINK_EMULATOR_OWNERS[sock]


def close_emulated_link(sock: object, timeout: float = 5.0) -> None:
    """Drain and release the emulator that owns ``sock``, if any."""

    if timeout < 0:
        raise ValueError("timeout must be non-negative")
    owner = _owned_link_emulator(sock)
    if owner is not None:
        owner.close(timeout_seconds=timeout)


def configure_socket(sock: socket.socket) -> None:
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)


def send_frame(
    sock: socket.socket,
    frame_type: FrameType,
    request_id: int,
    *,
    step: int = 0,
    token_count: int = 0,
    hidden_size: int = 0,
    flags: int = 0,
    payload: bytes | bytearray | memoryview = b"",
    emulator: LinkEmulator | None = None,
) -> int:
    try:
        normalized_type = FrameType(frame_type)
    except (TypeError, ValueError) as error:
        raise ValueError(f"unknown frame type {frame_type}") from error
    _require_unsigned("flags", flags, UINT16_MAX)
    _require_unsigned("request_id", request_id, UINT64_MAX)
    _require_unsigned("step", step, UINT32_MAX)
    _require_unsigned("token_count", token_count, UINT32_MAX)
    _require_unsigned("hidden_size", hidden_size, UINT32_MAX)
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("payload must be bytes-like")
    if isinstance(payload, memoryview):
        if not payload.contiguous:
            raise ValueError("payload memoryview must be contiguous")
        payload_size = payload.nbytes
    else:
        payload_size = len(payload)
    if payload_size > MAX_PAYLOAD_BYTES:
        raise ValueError(f"payload exceeds {MAX_PAYLOAD_BYTES} bytes")
    _validate_frame_metadata(
        normalized_type,
        flags=flags,
        step=step,
        token_count=token_count,
        hidden_size=hidden_size,
        payload_size=payload_size,
    )
    header = HEADER.pack(
        MAGIC,
        VERSION,
        int(normalized_type),
        flags,
        request_id,
        step,
        token_count,
        hidden_size,
        payload_size,
    )
    selected_emulator = (
        emulator
        if emulator is not None and emulator.enabled
        else _owned_link_emulator(sock)
    )
    if selected_emulator is not None:
        selected_emulator.send(sock, header, payload)
    else:
        sock.sendall(header)
        if payload:
            sock.sendall(payload)
    return len(header) + payload_size


def recv_frame(sock: socket.socket) -> Frame:
    raw_header = recv_exact(sock, HEADER_BYTES)
    magic, version, type_value, flags, request_id, step, token_count, hidden_size, size = (
        HEADER.unpack(raw_header)
    )
    if magic != MAGIC:
        raise ValueError("invalid frame magic")
    if version != VERSION:
        raise ValueError(f"unsupported protocol version {version}")
    if size > MAX_PAYLOAD_BYTES:
        raise ValueError(f"payload exceeds {MAX_PAYLOAD_BYTES} bytes")
    try:
        frame_type = FrameType(type_value)
    except ValueError as error:
        raise ValueError(f"unknown frame type {type_value}") from error
    _validate_frame_metadata(
        frame_type,
        flags=flags,
        step=step,
        token_count=token_count,
        hidden_size=hidden_size,
        payload_size=size,
    )
    payload = recv_exact(sock, size) if size else b""
    return Frame(
        frame_type=frame_type,
        flags=flags,
        request_id=request_id,
        step=step,
        token_count=token_count,
        hidden_size=hidden_size,
        payload=payload,
    )


def recv_exact(sock: socket.socket, size: int) -> bytearray:
    if size < 0:
        raise ValueError("receive size cannot be negative")
    data = bytearray(size)
    view = memoryview(data)
    received = 0
    while received < size:
        count = sock.recv_into(view[received:])
        if count == 0:
            raise EOFError("socket closed while receiving a frame")
        received += count
    return data


def encode_tensor_payload(
    tensor: torch.Tensor, codec: TensorCodec
) -> EncodedTensorPayload:
    if tensor.numel() == 0:
        raise ValueError("cannot encode an empty tensor")
    if not tensor.is_floating_point():
        raise ValueError("activation tensors must use a real floating dtype")
    base_codec = _DEFLATE_BASE.get(codec)
    if base_codec is not None:
        compressed = zlib.compress(encode_tensor(tensor, base_codec), level=1)
        return _owned_bytes_payload(compressed, tensor)
    if codec == TensorCodec.FP32:
        contiguous = _transport_staging_tensor(tensor, torch.float32)
        return _owned_tensor_payload(contiguous, tensor)
    if codec == TensorCodec.FP16:
        detached = tensor.detach()
        max_value = float(detached.abs().max().item())
        if not math.isfinite(max_value) or max_value > torch.finfo(torch.float16).max:
            raise ValueError("tensor contains values outside the finite FP16 range")
        # Convert GPU/BF16/FP32 activations directly to the wire dtype.  Going
        # through a CPU FP32 tensor first needlessly transfers twice as many
        # bytes over PCIe on the most common production path.
        contiguous = _transport_staging_tensor(detached, torch.float16)
        return _owned_tensor_payload(contiguous, tensor)

    # Keep the historical CPU/FP32 quantization reference bit-for-bit stable.
    # A future accelerator codec must be advertised and parity-certified before
    # it can replace this branch on heterogeneous hardware.
    contiguous = _transport_staging_tensor(tensor, torch.float32)
    if codec == TensorCodec.INT8:
        if not bool(torch.isfinite(contiguous).all().item()):
            raise ValueError("cannot quantize a tensor containing non-finite values")
        max_value = float(contiguous.abs().max().item())
        scale = max_value / 127.0 if max_value > 0 else 1.0
        quantized = torch.clamp(torch.round(contiguous / scale), -127, 127).to(torch.int8)
        encoded = struct.pack("<f", scale) + quantized.numpy().tobytes(order="C")
        return _owned_bytes_payload(encoded, tensor)
    if codec in (TensorCodec.INT8_GROUPED, TensorCodec.INT8_HADAMARD):
        if not bool(torch.isfinite(contiguous).all().item()):
            raise ValueError("cannot quantize a tensor containing non-finite values")
        hidden_size = int(contiguous.shape[-1])
        rows = contiguous.reshape(-1, hidden_size)
        row_count = int(rows.shape[0])
        hadamard = codec == TensorCodec.INT8_HADAMARD
        blocks = _quantization_blocks(hidden_size, hadamard=hadamard)
        scale_columns: list[torch.Tensor] = []
        data_columns: list[torch.Tensor] = []
        # Runs of equal-size blocks are batched into one [rows, blocks, size]
        # kernel; uneven tails (hidden sizes not divisible by the group size,
        # or the power-of-two Hadamard decomposition) produce at most a few
        # extra runs instead of a Python loop over every row and block.
        for start, end, size in _uniform_block_runs(blocks):
            segment = rows[:, start:end].reshape(row_count, -1, size)
            if hadamard:
                segment = _batched_fwht(segment)
            amax = segment.abs().amax(dim=-1)
            # float64 reproduces the historical per-block `float(max) / 127.0`
            # exactly; the wire scale is its float32 rounding, which is also
            # the divisor torch used for the float32 quantization.
            scales = torch.where(
                amax > 0,
                amax.to(torch.float64) / 127.0,
                torch.ones((), dtype=torch.float64),
            ).to(torch.float32)
            quantized = torch.clamp(
                torch.round(segment / scales.unsqueeze(-1)), -127, 127
            ).to(torch.int8)
            scale_columns.append(scales)
            data_columns.append(quantized.reshape(row_count, -1))
        scale_matrix = torch.cat(scale_columns, dim=1).contiguous()
        quantized = torch.cat(data_columns, dim=1).contiguous()
        encoded = scale_matrix.numpy().tobytes(order="C") + quantized.numpy().tobytes(
            order="C"
        )
        return _owned_bytes_payload(encoded, tensor)
    raise ValueError(f"unsupported tensor codec {codec}")


def encode_tensor(tensor: torch.Tensor, codec: TensorCodec) -> bytes:
    """Compatibility wrapper for callers that need an owned ``bytes`` value."""

    return encode_tensor_payload(tensor, codec).view.tobytes()


def _transport_staging_tensor(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    detached = tensor.detach()
    if detached.device.type == "cpu" and detached.dtype == dtype:
        # The returned memoryview must be a snapshot. A runner is allowed to
        # reuse or mutate its scratch tensor as soon as encode returns, while a
        # LinkEmulator or a slow socket can keep the payload alive much longer.
        return detached.contiguous().clone()
    return detached.to(device="cpu", dtype=dtype).contiguous()


def _owned_tensor_payload(
    staging: torch.Tensor, source: torch.Tensor
) -> EncodedTensorPayload:
    if staging.device.type != "cpu" or not staging.is_contiguous():
        raise ValueError("transport staging tensor must be contiguous CPU memory")
    view = memoryview(staging.numpy()).cast("B")
    return EncodedTensorPayload(
        view=view,
        owner=staging,
        staging_dtype=staging.dtype,
        source_device=str(source.device),
    )


def _owned_bytes_payload(
    encoded: bytes, source: torch.Tensor
) -> EncodedTensorPayload:
    return EncodedTensorPayload(
        view=memoryview(encoded),
        owner=encoded,
        staging_dtype=None,
        source_device=str(source.device),
    )


def decode_tensor(frame: Frame) -> torch.Tensor:
    if frame.frame_type not in TENSOR_FRAME_TYPES:
        raise ValueError("frame does not carry a tensor activation")
    if frame.token_count < 1 or frame.hidden_size < 1:
        raise ValueError("activation shape must be positive")
    elements = frame.token_count * frame.hidden_size
    try:
        codec = TensorCodec(frame.flags)
    except ValueError as error:
        raise ValueError(f"unsupported tensor codec {frame.flags}") from error
    # The network path receives directly into a bytearray, so torch can adopt the
    # buffer without another full activation copy. Hand-built immutable frames get
    # one defensive copy to provide the same writable-buffer guarantee.
    payload_owner = (
        frame.payload if isinstance(frame.payload, bytearray) else bytearray(frame.payload)
    )
    base_codec = _DEFLATE_BASE.get(codec)
    if base_codec is not None:
        payload_owner = _inflate_grouped_payload(
            payload_owner,
            codec,
            token_count=frame.token_count,
            hidden_size=frame.hidden_size,
        )
        codec = base_codec
    payload_view = memoryview(payload_owner)
    if codec == TensorCodec.FP32:
        expected = elements * 4
        if len(payload_view) != expected:
            raise ValueError(f"FP32 payload has {len(payload_view)} bytes, expected {expected}")
        tensor = torch.frombuffer(payload_view, dtype=torch.float32)
    elif codec == TensorCodec.FP16:
        expected = elements * 2
        if len(payload_view) != expected:
            raise ValueError(f"FP16 payload has {len(payload_view)} bytes, expected {expected}")
        tensor = torch.frombuffer(payload_view, dtype=torch.float16).to(torch.float32)
    elif codec == TensorCodec.INT8:
        expected = elements + 4
        if len(payload_view) != expected:
            raise ValueError(f"INT8 payload has {len(payload_view)} bytes, expected {expected}")
        scale = struct.unpack("<f", payload_view[:4])[0]
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("INT8 payload has an invalid quantization scale")
        tensor = torch.frombuffer(payload_view[4:], dtype=torch.int8).to(torch.float32) * scale
    elif codec in (TensorCodec.INT8_GROUPED, TensorCodec.INT8_HADAMARD):
        hadamard = codec == TensorCodec.INT8_HADAMARD
        blocks = _quantization_blocks(frame.hidden_size, hadamard=hadamard)
        scale_count = frame.token_count * len(blocks)
        scale_bytes = scale_count * 4
        expected = elements + scale_bytes
        if len(payload_view) != expected:
            raise ValueError(
                f"{codec.name} payload has {len(payload_view)} bytes, expected {expected}"
            )
        scales = torch.frombuffer(
            payload_view[:scale_bytes], dtype=torch.float32
        ).reshape(frame.token_count, len(blocks))
        if not bool(torch.isfinite(scales).all().item()) or not bool(
            (scales > 0).all().item()
        ):
            raise ValueError(f"{codec.name} payload has an invalid quantization scale")
        quantized = (
            torch.frombuffer(payload_view[scale_bytes:], dtype=torch.int8)
            .to(torch.float32)
            .reshape(frame.token_count, frame.hidden_size)
        )
        columns: list[torch.Tensor] = []
        block_offset = 0
        for start, end, size in _uniform_block_runs(blocks):
            count = (end - start) // size
            segment = quantized[:, start:end].reshape(frame.token_count, count, size)
            segment = segment * scales[:, block_offset : block_offset + count].unsqueeze(-1)
            block_offset += count
            if hadamard:
                segment = _batched_fwht(segment)
            columns.append(segment.reshape(frame.token_count, -1))
        tensor = torch.cat(columns, dim=1)
    else:
        raise ValueError(f"unsupported tensor codec {codec}")
    return tensor.reshape(1, frame.token_count, frame.hidden_size)


def token_payload(token_id: int) -> bytes:
    _require_unsigned("token_id", token_id, UINT32_MAX)
    return struct.pack("<I", token_id)


def branch_request_payload(request_id: int) -> bytes:
    """Encode the second request ID in a FORK/PROMOTE control frame."""

    _require_unsigned("branch_request_id", request_id, UINT64_MAX)
    return BRANCH_REQUEST_ID.pack(request_id)


def decode_branch_request_id(frame: Frame) -> int:
    """Decode the parent/child ID carried by an exact branch control frame.

    FORK uses ``frame.request_id`` as the child and returns the parent. PROMOTE
    uses ``frame.request_id`` as the parent and returns the selected child.
    Keeping both wire forms canonical avoids ambiguous retries or hidden branch
    metadata in the otherwise unused header fields.
    """

    if frame.frame_type not in BRANCH_CONTROL_FRAME_TYPES:
        raise ValueError("frame is not a branch lifecycle control")
    if len(frame.payload) != BRANCH_REQUEST_ID.size:
        raise ValueError("branch control payload must contain exactly one uint64")
    return BRANCH_REQUEST_ID.unpack(frame.payload)[0]


def tree_prepare_payload(
    nonce: int,
    path_lengths: list[int] | tuple[int, ...],
) -> bytes:
    """Create the canonical root-originated TREE_PREPARE payload.

    The ordered lengths bind the later FORKs to the proposed leaf shape without
    putting candidate token IDs on the control plane. A nonce is unique for the
    pipeline session and is never accepted as an implicit retry token.
    """

    return encode_tree_prepare_quote(
        TreePrepareQuote(nonce=nonce, path_lengths=tuple(path_lengths))
    )


def encode_tree_prepare_quote(quote: TreePrepareQuote) -> bytes:
    if not isinstance(quote, TreePrepareQuote):
        raise TypeError("quote must be a TreePrepareQuote")
    _require_unsigned("tree reservation nonce", quote.nonce, UINT64_MAX)
    path_lengths = tuple(quote.path_lengths)
    if not 1 <= len(path_lengths) <= MAX_TREE_PREPARE_PATHS:
        raise ValueError(
            "tree prepare path count must be between 1 and "
            f"{MAX_TREE_PREPARE_PATHS}"
        )
    for path_length in path_lengths:
        _require_unsigned(
            "tree path length", path_length, MAX_TREE_PREPARE_PATH_TOKENS
        )
        if path_length < 1:
            raise ValueError("tree path lengths must be positive")
    try:
        status = TreePrepareStatus(quote.status)
    except (TypeError, ValueError) as error:
        raise ValueError(f"unknown tree prepare status {quote.status}") from error
    try:
        rejection = TreePrepareRejection(quote.rejection)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"unknown tree prepare rejection {quote.rejection}"
        ) from error
    _require_unsigned("tree quote stage_count", quote.stage_count, UINT16_MAX)
    _require_unsigned(
        "tree quote rejecting_layer_start", quote.rejecting_layer_start, UINT32_MAX
    )
    _require_unsigned("tree quote required", quote.required, UINT64_MAX)
    _require_unsigned("tree quote limit", quote.limit, UINT64_MAX)
    _require_unsigned(
        "tree quote total_projected_bytes",
        quote.total_projected_bytes,
        UINT64_MAX,
    )
    if status is TreePrepareStatus.READY:
        if rejection is not TreePrepareRejection.NONE:
            raise ValueError("READY tree quote cannot carry a rejection reason")
        if quote.rejecting_layer_start or quote.required or quote.limit:
            raise ValueError("READY tree quote must use zero rejection fields")
    elif rejection is TreePrepareRejection.NONE:
        raise ValueError("REJECT tree quote requires a structured rejection reason")

    prefix = TREE_PREPARE_PREFIX.pack(
        quote.nonce,
        int(status),
        int(rejection),
        quote.stage_count,
        quote.rejecting_layer_start,
        quote.required,
        quote.limit,
        quote.total_projected_bytes,
    )
    lengths = struct.pack(f"<{len(path_lengths)}I", *path_lengths)
    return prefix + lengths


def decode_tree_prepare(frame: Frame) -> TreePrepareQuote:
    if frame.frame_type not in TREE_PREPARE_FRAME_TYPES:
        raise ValueError("frame is not a tree capacity quote")
    if not 1 <= frame.token_count <= MAX_TREE_PREPARE_PATHS:
        raise ValueError("tree prepare frame has an invalid path count")
    expected = TREE_PREPARE_PREFIX.size + frame.token_count * TREE_PATH_LENGTH.size
    if len(frame.payload) != expected:
        raise ValueError(
            f"tree prepare payload has {len(frame.payload)} bytes, expected {expected}"
        )
    (
        nonce,
        status_value,
        rejection_value,
        stage_count,
        rejecting_layer_start,
        required,
        limit,
        total_projected_bytes,
    ) = TREE_PREPARE_PREFIX.unpack_from(frame.payload)
    try:
        status = TreePrepareStatus(status_value)
    except ValueError as error:
        raise ValueError(f"unknown tree prepare status {status_value}") from error
    try:
        rejection = TreePrepareRejection(rejection_value)
    except ValueError as error:
        raise ValueError(
            f"unknown tree prepare rejection {rejection_value}"
        ) from error
    path_lengths = struct.unpack_from(
        f"<{frame.token_count}I", frame.payload, TREE_PREPARE_PREFIX.size
    )
    quote = TreePrepareQuote(
        nonce=nonce,
        path_lengths=tuple(path_lengths),
        status=status,
        rejection=rejection,
        stage_count=stage_count,
        rejecting_layer_start=rejecting_layer_start,
        required=required,
        limit=limit,
        total_projected_bytes=total_projected_bytes,
    )
    # Re-encoding is both a semantic validator and a canonical representation
    # check. It rejects zero paths, inconsistent status/reason combinations and
    # every value outside the sealed control-plane bounds.
    if encode_tree_prepare_quote(quote) != bytes(frame.payload):
        raise ValueError("tree prepare payload is not canonical")
    return quote


def tree_reservation_payload(nonce: int) -> bytes:
    _require_unsigned("tree reservation nonce", nonce, UINT64_MAX)
    return TREE_RESERVATION_NONCE.pack(nonce)


def decode_tree_reservation_nonce(frame: Frame) -> int:
    if frame.frame_type not in TREE_RESERVATION_FRAME_TYPES:
        raise ValueError("frame is not a tree reservation control")
    if len(frame.payload) != TREE_RESERVATION_NONCE.size:
        raise ValueError("tree reservation payload must contain exactly one uint64")
    return TREE_RESERVATION_NONCE.unpack(frame.payload)[0]


def decode_token(frame: Frame) -> int:
    if frame.frame_type != FrameType.TOKEN or len(frame.payload) != 4:
        raise ValueError("invalid token frame")
    return struct.unpack("<I", frame.payload)[0]


def verify_result_payload(token_ids: list[int] | tuple[int, ...]) -> bytes:
    if not token_ids:
        raise ValueError("verification result cannot be empty")
    for token_id in token_ids:
        _require_unsigned("token_id", token_id, UINT32_MAX)
    return struct.pack(f"<{len(token_ids)}I", *token_ids)


def decode_verify_result(frame: Frame) -> tuple[int, ...]:
    if frame.frame_type != FrameType.VERIFY_RESULT or frame.token_count < 1:
        raise ValueError("invalid verification result frame")
    expected = frame.token_count * 4
    if len(frame.payload) != expected:
        raise ValueError(
            f"verification result has {len(frame.payload)} bytes, expected {expected}"
        )
    return struct.unpack(f"<{frame.token_count}I", frame.payload)


def _require_unsigned(name: str, value: int, maximum: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value < 0 or value > maximum:
        raise ValueError(f"{name} must be between 0 and {maximum}")


def _activation_payload_bytes(flags: int, token_count: int, hidden_size: int) -> int:
    try:
        codec = TensorCodec(flags)
    except (TypeError, ValueError) as error:
        raise ValueError(f"unsupported tensor codec {flags}") from error
    if codec in _DEFLATE_BASE:
        raise ValueError(f"{codec.name} payload size is data-dependent")
    if token_count < 1 or hidden_size < 1:
        raise ValueError("activation shape must be positive")
    elements = token_count * hidden_size
    if codec == TensorCodec.FP32:
        return elements * 4
    if codec == TensorCodec.FP16:
        return elements * 2
    if codec == TensorCodec.INT8:
        return elements + 4
    blocks = _quantization_blocks(
        hidden_size,
        hadamard=codec == TensorCodec.INT8_HADAMARD,
    )
    return elements + token_count * len(blocks) * 4


def _deflate_bound(size: int) -> int:
    # Upper bound on zlib output for `size` input bytes: deflate stored-block
    # worst case plus the two-byte zlib header and the Adler-32 trailer.
    return size + (size >> 12) + (size >> 14) + (size >> 25) + 13 + 6


def _validate_frame_metadata(
    frame_type: FrameType,
    *,
    flags: int,
    step: int,
    token_count: int,
    hidden_size: int,
    payload_size: int,
) -> None:
    if payload_size > MAX_PAYLOAD_BYTES:
        raise ValueError(f"payload exceeds {MAX_PAYLOAD_BYTES} bytes")
    if frame_type in ROUTE_PROBE_FRAME_TYPES:
        # request_id and step deliberately remain available for correlating a
        # PONG with its PING.  Every other header field has exactly one wire
        # representation so probes cannot smuggle codec/shape metadata that
        # different implementations might interpret inconsistently.
        if flags != 0 or token_count != 0 or hidden_size != 0 or payload_size != 0:
            raise ValueError(
                f"{frame_type.name} frames require "
                "flags=token_count=hidden_size=0 and cannot carry a payload"
            )
        return
    if frame_type in BRANCH_CONTROL_FRAME_TYPES:
        if flags != 0 or step != 0 or token_count != 0 or hidden_size != 0:
            raise ValueError(
                f"{frame_type.name} frames require "
                "flags=step=token_count=hidden_size=0"
            )
        if payload_size != BRANCH_REQUEST_ID.size:
            raise ValueError(
                f"{frame_type.name} payload must contain exactly one uint64"
            )
        return
    if frame_type in TREE_PREPARE_FRAME_TYPES:
        if flags != 0 or hidden_size != 0:
            raise ValueError(
                f"{frame_type.name} frames require flags=hidden_size=0"
            )
        if not 1 <= token_count <= MAX_TREE_PREPARE_PATHS:
            raise ValueError(
                f"{frame_type.name} token_count must be between 1 and "
                f"{MAX_TREE_PREPARE_PATHS}"
            )
        expected = TREE_PREPARE_PREFIX.size + token_count * TREE_PATH_LENGTH.size
        if payload_size != expected:
            raise ValueError(
                f"{frame_type.name} payload has {payload_size} bytes, expected {expected}"
            )
        return
    if frame_type in TREE_RESERVATION_COMMIT_FRAME_TYPES:
        if flags != 0 or hidden_size != 0:
            raise ValueError(
                f"{frame_type.name} frames require flags=hidden_size=0"
            )
        if token_count > UINT16_MAX - 1:
            raise ValueError(
                f"{frame_type.name} token_count must contain a uint16 "
                "predecessor count"
            )
        if payload_size != TREE_RESERVATION_NONCE.size:
            raise ValueError(
                f"{frame_type.name} payload must contain exactly one uint64"
            )
        return
    if frame_type in TREE_RESERVATION_CANCEL_FRAME_TYPES:
        if flags != 0 or token_count != 0 or hidden_size != 0:
            raise ValueError(
                f"{frame_type.name} frames require "
                "flags=token_count=hidden_size=0"
            )
        if payload_size != TREE_RESERVATION_NONCE.size:
            raise ValueError(
                f"{frame_type.name} payload must contain exactly one uint64"
            )
        return
    if frame_type in TREE_RESERVATION_RESULT_FRAME_TYPES:
        if flags != 0 or hidden_size != 0:
            raise ValueError(
                f"{frame_type.name} frames require flags=hidden_size=0"
            )
        if not 1 <= token_count <= UINT16_MAX:
            raise ValueError(
                f"{frame_type.name} token_count must contain a positive "
                "uint16 stage count"
            )
        if payload_size != TREE_RESERVATION_NONCE.size:
            raise ValueError(
                f"{frame_type.name} payload must contain exactly one uint64"
            )
        return
    if frame_type in TENSOR_FRAME_TYPES:
        try:
            codec = TensorCodec(flags)
        except (TypeError, ValueError) as error:
            raise ValueError(f"unsupported tensor codec {flags}") from error
        base_codec = _DEFLATE_BASE.get(codec)
        if base_codec is not None:
            # Compressed payload sizes are data-dependent; bound them by the
            # zlib worst case so a hostile header cannot reserve huge buffers.
            inflated = _activation_payload_bytes(int(base_codec), token_count, hidden_size)
            if inflated > MAX_PAYLOAD_BYTES:
                # Every non-deflate codec enforces inflated == payload_size <=
                # MAX_PAYLOAD_BYTES; the same ceiling must apply to the
                # inflated size here, or token_count from an untrusted header
                # would set the decompression budget.
                raise ValueError(
                    f"activation would inflate to {inflated} bytes, exceeding "
                    f"{MAX_PAYLOAD_BYTES}"
                )
            bound = _deflate_bound(inflated)
            if payload_size < 1 or payload_size > bound:
                raise ValueError(
                    f"activation payload has {payload_size} bytes, expected between "
                    f"1 and {bound}"
                )
            return
        expected = _activation_payload_bytes(flags, token_count, hidden_size)
        if expected != payload_size:
            raise ValueError(
                f"activation payload has {payload_size} bytes, expected {expected}"
            )
        return
    if frame_type == FrameType.TOKEN:
        if payload_size != 4:
            raise ValueError("token payload must contain exactly four bytes")
        return
    if frame_type == FrameType.VERIFY_RESULT:
        if token_count < 1 or payload_size != token_count * 4:
            raise ValueError(
                "verification result payload must contain one uint32 per token"
            )
        return
    if frame_type != FrameType.ERROR and payload_size != 0:
        raise ValueError(f"{frame_type.name} frames cannot carry a payload")


def _quantization_blocks(hidden_size: int, *, hadamard: bool) -> tuple[tuple[int, int], ...]:
    if hidden_size < 1:
        raise ValueError("hidden size must be positive")
    blocks: list[tuple[int, int]] = []
    start = 0
    while start < hidden_size:
        remaining = hidden_size - start
        if hadamard:
            # Decompose a non-power-of-two tail into power-of-two blocks.  This
            # preserves every dimension without transmitting padding and keeps H
            # exactly self-inverse up to floating-point rounding.
            size = 1 << int(math.floor(math.log2(min(QUANT_GROUP_SIZE, remaining))))
        else:
            size = min(QUANT_GROUP_SIZE, remaining)
        blocks.append((start, start + size))
        start += size
    return tuple(blocks)


def _uniform_block_runs(
    blocks: tuple[tuple[int, int], ...]
) -> tuple[tuple[int, int, int], ...]:
    # Collapse consecutive equal-size blocks into (start, end, block_size) runs
    # so the uneven tail stays exact while the bulk is batched in one kernel.
    runs: list[tuple[int, int, int]] = []
    for start, end in blocks:
        size = end - start
        if runs and runs[-1][2] == size and runs[-1][1] == start:
            runs[-1] = (runs[-1][0], end, size)
        else:
            runs.append((start, end, size))
    return tuple(runs)


def _batched_fwht(values: torch.Tensor) -> torch.Tensor:
    # Same butterfly schedule as _normalized_fwht applied to the last axis of a
    # [..., size] batch.  Blocks are contiguous, so the flat (-1, stride * 2)
    # view groups exactly the same element pairs as the per-vector transform.
    size = int(values.shape[-1])
    if size < 1 or size & (size - 1):
        raise ValueError("Hadamard block size must be a power of two")
    output = values.to(torch.float32).contiguous().clone()
    stride = 1
    while stride < size:
        view = output.reshape(-1, stride * 2)
        left = view[:, :stride].clone()
        right = view[:, stride:].clone()
        view[:, :stride] = left + right
        view[:, stride:] = left - right
        stride *= 2
    return output / math.sqrt(size)


def _inflate_grouped_payload(
    payload: bytes | bytearray,
    codec: TensorCodec,
    *,
    token_count: int,
    hidden_size: int,
) -> bytearray:
    base_codec = _DEFLATE_BASE[codec]
    blocks = _quantization_blocks(
        hidden_size,
        hadamard=base_codec == TensorCodec.INT8_HADAMARD,
    )
    expected = token_count * hidden_size + token_count * len(blocks) * 4
    if expected > MAX_PAYLOAD_BYTES:
        # token_count/hidden_size come from an untrusted header; without this
        # cap the max_length passed to decompress would track the attacker's
        # shape instead of the trusted payload ceiling.
        raise ValueError(
            f"{codec.name} payload would inflate to {expected} bytes, exceeding "
            f"{MAX_PAYLOAD_BYTES}"
        )
    inflater = zlib.decompressobj()
    try:
        # max_length caps the buffer: anything past expected + 1 stays in
        # unconsumed_tail and is rejected below together with short streams.
        inflated = inflater.decompress(bytes(payload), expected + 1)
    except zlib.error as error:
        raise ValueError(f"{codec.name} payload is not valid zlib data") from error
    if (
        len(inflated) != expected
        or not inflater.eof
        or inflater.unconsumed_tail
        or inflater.unused_data
    ):
        raise ValueError(f"{codec.name} payload must inflate to exactly {expected} bytes")
    return bytearray(inflated)


def _normalized_fwht(values: torch.Tensor) -> torch.Tensor:
    if values.ndim != 1 or values.numel() < 1:
        raise ValueError("Hadamard input must be a non-empty vector")
    return _batched_fwht(values)
