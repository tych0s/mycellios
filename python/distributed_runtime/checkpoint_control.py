"""Private workspace-local control plane for live activation KV checkpoints.

The socket is never exposed on the network. A background thread only performs
bounded framing; the stage's inference thread executes every runner mutation,
so capture/restore cannot race a CUDA forward or speculative KV transition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import queue
import socket
import threading
from typing import Any

MAX_CONTROL_HEADER_BYTES = 4096
MAX_CHECKPOINT_BYTES = 512 * 1024 * 1024
CHECKPOINT_SOCKET_NAME = "activation-checkpoint.sock"


@dataclass
class _ControlRequest:
    operation: str
    request_id: int
    max_bytes: int
    committed_position: int | None = None
    payload: bytes | None = None
    result: bytes | None = None
    result_position: int | None = None
    error: str | None = None
    completed: threading.Event = field(default_factory=threading.Event)


class StageCheckpointControl:
    def __init__(self, workspace: str, runner: Any) -> None:
        root = Path(workspace).resolve(strict=True)
        if not root.is_dir():
            raise ValueError("checkpoint control workspace is not a directory")
        self.path = root / CHECKPOINT_SOCKET_NAME
        if self.path.exists() or self.path.is_symlink():
            raise ValueError("checkpoint control socket path already exists")
        self.runner = runner
        self.requests: queue.Queue[_ControlRequest] = queue.Queue(maxsize=8)
        self.stopping = threading.Event()
        self.wake_reader, self.wake_writer = socket.socketpair()
        self.wake_reader.setblocking(False)
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.listener.bind(str(self.path))
        os.chmod(self.path, 0o600)
        self.listener.listen(4)
        self.listener.settimeout(0.2)
        self.thread = threading.Thread(
            target=self._serve,
            name="stage-checkpoint-control",
            daemon=True,
        )
        self.thread.start()

    @property
    def wake_socket(self) -> socket.socket:
        return self.wake_reader

    def service_pending(self) -> int:
        try:
            while self.wake_reader.recv(1024):
                pass
        except (BlockingIOError, OSError):
            pass
        serviced = 0
        while True:
            try:
                request = self.requests.get_nowait()
            except queue.Empty:
                return serviced
            try:
                if request.operation == "capture":
                    request.result = self.runner.activation_checkpoint_payload(
                        request.request_id, max_bytes=request.max_bytes
                    )
                    request.result_position = self.runner.sequence_length(request.request_id)
                elif request.operation == "restore":
                    if request.payload is None or request.committed_position is None:
                        raise ValueError("checkpoint restore request is incomplete")
                    self.runner.restore_activation_checkpoint_payload(
                        request.request_id,
                        request.payload,
                        request.committed_position,
                        max_bytes=request.max_bytes,
                    )
                    request.result = b""
                else:
                    raise ValueError("checkpoint control operation is invalid")
            except BaseException as error:
                request.error = f"{type(error).__name__}: {error}"[:1024]
            finally:
                request.payload = None
                request.completed.set()
                serviced += 1

    def close(self) -> None:
        if self.stopping.is_set():
            return
        self.stopping.set()
        try:
            self.listener.close()
        except OSError:
            pass
        try:
            self.wake_writer.send(b"x")
        except OSError:
            pass
        while True:
            try:
                request = self.requests.get_nowait()
            except queue.Empty:
                break
            request.payload = None
            request.error = "RuntimeError: checkpoint control is closing"
            request.completed.set()
        self.thread.join(timeout=1.0)
        self.wake_reader.close()
        self.wake_writer.close()
        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            pass

    def _serve(self) -> None:
        while not self.stopping.is_set():
            try:
                connection, _ = self.listener.accept()
            except (TimeoutError, OSError):
                continue
            with connection:
                try:
                    self._handle(connection)
                except BaseException as error:
                    try:
                        self._send_header(connection, {"ok": False, "error": str(error)[:1024]})
                    except OSError:
                        pass

    def _handle(self, connection: socket.socket) -> None:
        connection.settimeout(30.0)
        header = _read_header(connection)
        operation = header.get("operation")
        expected_keys = (
            {"operation", "requestId", "maxBytes", "payloadBytes", "committedPosition"}
            if operation == "restore"
            else {"operation", "requestId", "maxBytes"}
        )
        if set(header) != expected_keys:
            raise ValueError("checkpoint control header fields are invalid")
        request_id = _bounded_integer(header.get("requestId"), 0, (1 << 64) - 1, "requestId")
        max_bytes = _bounded_integer(header.get("maxBytes"), 1, MAX_CHECKPOINT_BYTES, "maxBytes")
        request = _ControlRequest(operation=str(operation), request_id=request_id, max_bytes=max_bytes)
        if operation == "restore":
            payload_bytes = _bounded_integer(header.get("payloadBytes"), 1, max_bytes, "payloadBytes")
            request.committed_position = _bounded_integer(
                header.get("committedPosition"), 1, (1 << 53) - 1, "committedPosition"
            )
            request.payload = _read_exact(connection, payload_bytes)
        elif operation != "capture":
            raise ValueError("checkpoint control operation is invalid")
        try:
            self.requests.put(request, timeout=1.0)
        except queue.Full as error:
            raise RuntimeError("checkpoint control queue is full") from error
        self.wake_writer.send(b"x")
        if not request.completed.wait(timeout=30.0):
            raise TimeoutError("checkpoint control stage operation timed out")
        if request.error is not None:
            self._send_header(connection, {"ok": False, "error": request.error})
            return
        result = request.result or b""
        self._send_header(connection, {
            "ok": True,
            "payloadBytes": len(result),
            **(
                {"committedPosition": request.result_position}
                if request.operation == "capture"
                else {}
            ),
        })
        if result:
            connection.sendall(result)

    @staticmethod
    def _send_header(connection: socket.socket, value: dict[str, Any]) -> None:
        connection.sendall(json.dumps(value, separators=(",", ":")).encode("utf-8") + b"\n")


def checkpoint_control_from_environment(runner: Any) -> StageCheckpointControl | None:
    workspace = os.environ.get("MYCELLIOS_EXECUTOR_WORKSPACE")
    if not workspace or not hasattr(socket, "AF_UNIX"):
        return None
    if not callable(getattr(runner, "activation_checkpoint_payload", None)):
        return None
    if not callable(getattr(runner, "restore_activation_checkpoint_payload", None)):
        return None
    return StageCheckpointControl(workspace, runner)


def _read_header(connection: socket.socket) -> dict[str, Any]:
    data = bytearray()
    while len(data) <= MAX_CONTROL_HEADER_BYTES:
        chunk = connection.recv(1)
        if not chunk:
            raise EOFError("checkpoint control header ended early")
        if chunk == b"\n":
            try:
                value = json.loads(data.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError("checkpoint control header is invalid JSON") from error
            if not isinstance(value, dict):
                raise ValueError("checkpoint control header must be an object")
            return value
        data.extend(chunk)
    raise ValueError("checkpoint control header is too large")


def _read_exact(connection: socket.socket, length: int) -> bytes:
    chunks: list[bytes] = []
    remaining = length
    while remaining:
        chunk = connection.recv(min(remaining, 256 * 1024))
        if not chunk:
            raise EOFError("checkpoint control payload ended early")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _bounded_integer(value: Any, minimum: int, maximum: int, name: str) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"checkpoint control {name} is invalid")
    return value
