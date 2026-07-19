from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
import socket
import struct
import time
import zlib

import torch

MAGIC = b"GDLP"
VERSION = 2
HEADER = struct.Struct("<4sBBHQIIII")
HEADER_BYTES = HEADER.size
MAX_PAYLOAD_BYTES = 64 * 1024 * 1024
QUANT_GROUP_SIZE = 64
UINT16_MAX = (1 << 16) - 1
UINT32_MAX = (1 << 32) - 1
UINT64_MAX = (1 << 64) - 1


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
class LinkEmulator:
    one_way_delay_ms: float = 0.0
    bandwidth_mbps: float = 0.0

    def wait_before_send(self, payload_bytes: int) -> None:
        seconds = max(0.0, self.one_way_delay_ms) / 1_000
        if self.bandwidth_mbps > 0:
            seconds += (payload_bytes * 8) / (self.bandwidth_mbps * 1_000_000)
        if seconds > 0:
            time.sleep(seconds)


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
    if emulator is not None:
        emulator.wait_before_send(len(header) + payload_size)
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


def encode_tensor(tensor: torch.Tensor, codec: TensorCodec) -> bytes:
    if tensor.numel() == 0:
        raise ValueError("cannot encode an empty tensor")
    base_codec = _DEFLATE_BASE.get(codec)
    if base_codec is not None:
        return zlib.compress(encode_tensor(tensor, base_codec), level=1)
    contiguous = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if codec == TensorCodec.FP32:
        return contiguous.numpy().tobytes(order="C")
    if codec == TensorCodec.FP16:
        max_value = float(contiguous.abs().max().item())
        if not math.isfinite(max_value) or max_value > torch.finfo(torch.float16).max:
            raise ValueError("tensor contains values outside the finite FP16 range")
        return contiguous.to(dtype=torch.float16).numpy().tobytes(order="C")
    if codec == TensorCodec.INT8:
        if not bool(torch.isfinite(contiguous).all().item()):
            raise ValueError("cannot quantize a tensor containing non-finite values")
        max_value = float(contiguous.abs().max().item())
        scale = max_value / 127.0 if max_value > 0 else 1.0
        quantized = torch.clamp(torch.round(contiguous / scale), -127, 127).to(torch.int8)
        return struct.pack("<f", scale) + quantized.numpy().tobytes(order="C")
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
        return scale_matrix.numpy().tobytes(order="C") + quantized.numpy().tobytes(
            order="C"
        )
    raise ValueError(f"unsupported tensor codec {codec}")


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
    token_count: int,
    hidden_size: int,
    payload_size: int,
) -> None:
    if payload_size > MAX_PAYLOAD_BYTES:
        raise ValueError(f"payload exceeds {MAX_PAYLOAD_BYTES} bytes")
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
