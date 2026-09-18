"""Private workspace-local control plane for live activation KV checkpoints.

Unix uses an owner-only socket; Windows uses authenticated IPv4 loopback with
a signed endpoint descriptor. A background thread only performs bounded framing;
the stage's inference thread executes every runner mutation,
so capture/restore cannot race a CUDA forward or speculative KV transition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import hmac
import json
import os
from pathlib import Path
import queue
import re
import select
import socket
import threading
import time
from typing import Any

MAX_CONTROL_HEADER_BYTES = 4096
MAX_CHECKPOINT_BYTES = 512 * 1024 * 1024
CHECKPOINT_SOCKET_NAME = "activation-checkpoint.sock"
CHECKPOINT_ENDPOINT_NAME = "activation-checkpoint.endpoint.json"
CHECKPOINT_ENDPOINT_SCHEMA = "mycellios-checkpoint-control/1"
CHECKPOINT_TOKEN_ENV = "MYCELLIOS_CHECKPOINT_CONTROL_TOKEN"
CONTROL_REQUEST_TIMEOUT_SECONDS = 30.0


def _unix_socket_available() -> bool:
    return os.name != "nt" and hasattr(socket, "AF_UNIX")


@dataclass
class _ControlRequest:
    operation: str
    request_id: int
    max_bytes: int
    connection: socket.socket
    deadline: float = 0.0
    committed_position: int | None = None
    payload: bytes | None = None
    result: bytes | None = None
    result_position: int | None = None
    error: str | None = None
    completed: threading.Event = field(default_factory=threading.Event)
    _state_lock: threading.Lock = field(default_factory=threading.Lock)
    _started: bool = False

    def claim(self) -> bool:
        with self._state_lock:
            if self._started or self.completed.is_set():
                return False
            self._started = True
            return True

    def cancel_pending(self, message: str) -> None:
        with self._state_lock:
            # A runner mutation already claimed by the stage cannot be rolled back.
            if not self._started:
                self.payload = None
                self.error = message
                self.completed.set()

    def check_caller(self) -> None:
        if time.monotonic() >= self.deadline:
            raise TimeoutError("checkpoint control stage operation timed out")
        readable, _, _ = select.select([self.connection], [], [], 0)
        if readable:
            if not self.connection.recv(1, socket.MSG_PEEK):
                raise EOFError("checkpoint control caller disconnected")
            raise ValueError("checkpoint control request has trailing bytes")


class StageCheckpointControl:
    def __init__(self, workspace: str, runner: Any, *, auth_token: str | None = None) -> None:
        root = Path(workspace).resolve(strict=True)
        if not root.is_dir():
            raise ValueError("checkpoint control workspace is not a directory")
        self._unix = _unix_socket_available()
        self._token = None if self._unix else (
            auth_token if auth_token is not None else os.environ.get(CHECKPOINT_TOKEN_ENV)
        )
        if not self._unix and (
            not isinstance(self._token, str) or re.fullmatch(r"[0-9a-f]{64}", self._token) is None
        ):
            raise ValueError("checkpoint control requires a 256-bit launch authentication token")
        self.path = root / (CHECKPOINT_SOCKET_NAME if self._unix else CHECKPOINT_ENDPOINT_NAME)
        if self.path.exists() or self.path.is_symlink():
            raise ValueError("checkpoint control socket path already exists")
        self.runner = runner
        self.requests: queue.Queue[_ControlRequest] = queue.Queue(maxsize=8)
        self.stopping = threading.Event()
        self._connection_lock = threading.Lock()
        self._connection: socket.socket | None = None
        self.wake_reader, self.wake_writer = socket.socketpair()
        self.wake_reader.setblocking(False)
        try:
            self.listener = socket.socket(
                socket.AF_UNIX if self._unix else socket.AF_INET, socket.SOCK_STREAM
            )
        except BaseException:
            self.wake_reader.close()
            self.wake_writer.close()
            raise
        owns_path = False
        try:
            if self._unix:
                self.listener.bind(str(self.path))
                owns_path = True
                os.chmod(self.path, 0o600)
            else:
                # A numeric loopback address avoids DNS and never listens on LAN interfaces.
                self.listener.bind(("127.0.0.1", 0))
            self.listener.listen(4)
            self.listener.settimeout(0.2)
            if not self._unix:
                port = self.listener.getsockname()[1]
                descriptor = {
                    "schema": CHECKPOINT_ENDPOINT_SCHEMA,
                    "host": "127.0.0.1",
                    "port": port,
                    "signature": hmac.new(
                        bytes.fromhex(self._token),
                        f"{CHECKPOINT_ENDPOINT_SCHEMA}\n127.0.0.1\n{port}".encode("utf-8"),
                        hashlib.sha256,
                    ).hexdigest(),
                }
                # The secret stays in the launch environment, never in the workspace.
                descriptor_fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                owns_path = True
                with os.fdopen(descriptor_fd, "w", encoding="utf-8") as endpoint:
                    json.dump(descriptor, endpoint, separators=(",", ":"))
        except BaseException:
            self.listener.close()
            self.wake_reader.close()
            self.wake_writer.close()
            if owns_path:
                self.path.unlink(missing_ok=True)
            raise
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
            with self._connection_lock:
                if self.stopping.is_set():
                    request.cancel_pending("RuntimeError: checkpoint control is closing")
                if not request.completed.is_set():
                    try:
                        request.check_caller()
                    except (OSError, ValueError, EOFError) as error:
                        request.cancel_pending(f"{type(error).__name__}: {error}")
                if not request.claim():
                    continue
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
        with self._connection_lock:
            if self.stopping.is_set():
                return
            self.stopping.set()
            if self._connection is not None:
                try:
                    self._connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                self._connection.close()
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
            request.cancel_pending("RuntimeError: checkpoint control is closing")
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
            deadline = time.monotonic() + CONTROL_REQUEST_TIMEOUT_SECONDS
            with connection:
                with self._connection_lock:
                    if self.stopping.is_set():
                        return
                    self._connection = connection
                try:
                    self._handle(connection, deadline)
                except BaseException as error:
                    try:
                        # Reporting invalid/expired input is best effort: an
                        # unread response must not extend this client's budget.
                        connection.setblocking(False)
                        self._send_header(connection, {"ok": False, "error": str(error)[:1024]})
                    except OSError:
                        pass
                finally:
                    with self._connection_lock:
                        self._connection = None

    def _handle(self, connection: socket.socket, deadline: float) -> None:
        header = _read_header(connection, deadline)
        if self._token is not None:
            token = header.pop("token", None)
            if (
                not isinstance(token, str)
                or re.fullmatch(r"[0-9a-f]{64}", token) is None
                or not hmac.compare_digest(token, self._token)
            ):
                raise ValueError("checkpoint control authentication failed")
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
        request = _ControlRequest(
            operation=str(operation), request_id=request_id, max_bytes=max_bytes,
            connection=connection, deadline=deadline,
        )
        if operation == "restore":
            payload_bytes = _bounded_integer(header.get("payloadBytes"), 1, max_bytes, "payloadBytes")
            request.committed_position = _bounded_integer(
                header.get("committedPosition"), 1, (1 << 53) - 1, "committedPosition"
            )
            request.payload = _read_exact(connection, payload_bytes, deadline)
        elif operation != "capture":
            raise ValueError("checkpoint control operation is invalid")
        request.check_caller()
        with self._connection_lock:
            if self.stopping.is_set():
                raise RuntimeError("checkpoint control is closing")
            try:
                self.requests.put_nowait(request)
            except queue.Full as error:
                raise RuntimeError("checkpoint control queue is full") from error
        try:
            self.wake_writer.send(b"x")
            while not request.completed.wait(timeout=min(0.05, _remaining_seconds(deadline))):
                if self.stopping.is_set():
                    raise RuntimeError("checkpoint control is closing")
                request.check_caller()
        except BaseException as error:
            request.cancel_pending(f"{type(error).__name__}: {error}")
            raise
        if request.error is not None:
            # Cancellation may already have exhausted the absolute deadline.
            connection.setblocking(False)
            self._send_header(connection, {"ok": False, "error": request.error})
            return
        result = request.result or b""
        connection.settimeout(_remaining_seconds(deadline))
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
            connection.settimeout(_remaining_seconds(deadline))
            connection.sendall(result)

    @staticmethod
    def _send_header(connection: socket.socket, value: dict[str, Any]) -> None:
        connection.sendall(json.dumps(value, separators=(",", ":")).encode("utf-8") + b"\n")


def checkpoint_control_from_environment(runner: Any) -> StageCheckpointControl | None:
    workspace = os.environ.get("MYCELLIOS_EXECUTOR_WORKSPACE")
    if not workspace:
        return None
    if not callable(getattr(runner, "activation_checkpoint_payload", None)):
        return None
    if not callable(getattr(runner, "restore_activation_checkpoint_payload", None)):
        return None
    return StageCheckpointControl(workspace, runner)


def _remaining_seconds(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("checkpoint control request timed out")
    return remaining


def _read_header(connection: socket.socket, deadline: float) -> dict[str, Any]:
    data = bytearray()
    while len(data) <= MAX_CONTROL_HEADER_BYTES:
        connection.settimeout(_remaining_seconds(deadline))
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


def _read_exact(connection: socket.socket, length: int, deadline: float) -> bytes:
    chunks: list[bytes] = []
    remaining = length
    while remaining:
        connection.settimeout(_remaining_seconds(deadline))
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
