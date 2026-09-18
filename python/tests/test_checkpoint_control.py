from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path
import select
import socket
from tempfile import TemporaryDirectory
import threading
import unittest
from unittest.mock import patch

from distributed_runtime.checkpoint_control import (
    CHECKPOINT_ENDPOINT_NAME,
    CHECKPOINT_ENDPOINT_SCHEMA,
    CHECKPOINT_TOKEN_ENV,
    StageCheckpointControl,
    checkpoint_control_from_environment,
)

_TOKEN = "7d" * 32


class _Runner:
    def __init__(self) -> None:
        self.restored: tuple[int, bytes, int, int] | None = None

    def activation_checkpoint_payload(self, request_id: int, *, max_bytes: int) -> bytes:
        payload = f"kv:{request_id}".encode()
        if len(payload) > max_bytes:
            raise ValueError("too large")
        return payload

    def sequence_length(self, request_id: int) -> int:
        return 31

    def restore_activation_checkpoint_payload(
        self, request_id: int, payload: bytes, committed_position: int, *, max_bytes: int
    ) -> None:
        self.restored = request_id, payload, committed_position, max_bytes


class StageCheckpointControlTest(unittest.TestCase):
    def test_capture_and_restore_run_only_when_stage_thread_services_queue(self) -> None:
        with TemporaryDirectory() as temporary:
            runner = _Runner()
            control = StageCheckpointControl(temporary, runner, auth_token=_TOKEN)
            try:
                capture: dict[str, object] = {}

                def capture_client() -> None:
                    with _client(control, {
                        "operation": "capture", "requestId": 17, "maxBytes": 64,
                    }) as client:
                        capture["header"] = _header(client)
                        capture["payload"] = client.recv(64)

                thread = threading.Thread(target=capture_client)
                thread.start()
                readable, _, _ = select.select([control.wake_socket], [], [], 2.0)
                self.assertEqual(readable, [control.wake_socket])
                self.assertEqual(control.service_pending(), 1)
                thread.join(2.0)
                self.assertFalse(thread.is_alive())
                self.assertEqual(capture, {
                    "header": {"ok": True, "payloadBytes": 5, "committedPosition": 31},
                    "payload": b"kv:17",
                })

                restore = threading.Thread(
                    target=_restore_client,
                    args=(control, b"safe-kv"),
                )
                restore.start()
                readable, _, _ = select.select([control.wake_socket], [], [], 2.0)
                self.assertEqual(readable, [control.wake_socket])
                self.assertEqual(control.service_pending(), 1)
                restore.join(2.0)
                self.assertFalse(restore.is_alive())
                self.assertEqual(runner.restored, (23, b"safe-kv", 11, 64))
            finally:
                control.close()
            self.assertFalse(Path(temporary, "activation-checkpoint.sock").exists())
            self.assertFalse(Path(temporary, CHECKPOINT_ENDPOINT_NAME).exists())

    def test_rejects_oversized_or_truncated_restore_before_runner(self) -> None:
        with TemporaryDirectory() as temporary:
            runner = _Runner()
            control = StageCheckpointControl(temporary, runner, auth_token=_TOKEN)
            try:
                with _client(control, {
                    "operation": "restore", "requestId": 1, "maxBytes": 4,
                    "payloadBytes": 5, "committedPosition": 1,
                }) as client:
                    self.assertEqual(_header(client)["ok"], False)
                with _client(control, {
                    "operation": "restore", "requestId": 1, "maxBytes": 4,
                    "payloadBytes": 4, "committedPosition": 1,
                }, b"X") as client:
                    client.shutdown(socket.SHUT_WR)
                    self.assertEqual(_header(client)["ok"], False)
                self.assertTrue(control.requests.empty())
                self.assertIsNone(runner.restored)
            finally:
                control.close()

    def test_rejects_unknown_header_fields_without_reaching_runner(self) -> None:
        with TemporaryDirectory() as temporary:
            runner = _Runner()
            control = StageCheckpointControl(temporary, runner, auth_token=_TOKEN)
            try:
                with _client(control, {
                    "operation": "restore", "requestId": 1, "maxBytes": 4,
                    "payloadBytes": 1, "committedPosition": 1, "unexpected": True,
                }) as client:
                    self.assertEqual(_header(client)["ok"], False)
                self.assertIsNone(runner.restored)
            finally:
                control.close()

    @patch("distributed_runtime.checkpoint_control._unix_socket_available", return_value=False)
    def test_loopback_descriptor_is_signed_and_contains_no_secret(self, _availability) -> None:
        with TemporaryDirectory() as temporary:
            control = StageCheckpointControl(temporary, _Runner(), auth_token=_TOKEN)
            try:
                descriptor_text = control.path.read_text(encoding="utf-8")
                self.assertNotIn(_TOKEN, descriptor_text)
                descriptor = json.loads(descriptor_text)
                self.assertEqual(set(descriptor), {"schema", "host", "port", "signature"})
                self.assertEqual(descriptor["schema"], CHECKPOINT_ENDPOINT_SCHEMA)
                self.assertEqual(descriptor["host"], "127.0.0.1")
                self.assertEqual(control.listener.getsockname(), ("127.0.0.1", descriptor["port"]))
                expected = hmac.new(
                    bytes.fromhex(_TOKEN),
                    f"{CHECKPOINT_ENDPOINT_SCHEMA}\n127.0.0.1\n{descriptor['port']}".encode(),
                    hashlib.sha256,
                ).hexdigest()
                self.assertEqual(descriptor["signature"], expected)
            finally:
                control.close()
            self.assertFalse(control.path.exists())

    @patch("distributed_runtime.checkpoint_control._unix_socket_available", return_value=False)
    def test_loopback_capture_and_restore_with_authentication(self, _availability) -> None:
        self.test_capture_and_restore_run_only_when_stage_thread_services_queue()

    @patch("distributed_runtime.checkpoint_control._unix_socket_available", return_value=False)
    def test_loopback_rejects_missing_wrong_and_malformed_tokens_before_queue(self, _availability) -> None:
        with TemporaryDirectory() as temporary:
            runner = _Runner()
            control = StageCheckpointControl(temporary, runner, auth_token=_TOKEN)
            try:
                for token in (None, "a" * 64, True, {"token": _TOKEN}, "\u2603"):
                    with self.subTest(token=token), _client(control, {
                        "operation": "restore", "requestId": 1, "maxBytes": 4,
                        "payloadBytes": 1, "committedPosition": 1,
                        **({"token": token} if token is not None else {}),
                    }, authenticate=False) as client:
                        # No payload: authentication must reject before waiting for it.
                        self.assertFalse(_header(client)["ok"])
                    self.assertTrue(control.requests.empty())
                self.assertIsNone(runner.restored)
            finally:
                control.close()

    @patch("distributed_runtime.checkpoint_control._unix_socket_available", return_value=False)
    def test_loopback_requires_valid_launch_secret_and_preserves_existing_path(self, _availability) -> None:
        with TemporaryDirectory() as temporary, patch.dict(os.environ, {}, clear=True):
            for token in (None, "", "guessable", "g" * 64):
                with self.subTest(token=token), self.assertRaisesRegex(ValueError, "authentication token"):
                    StageCheckpointControl(temporary, _Runner(), auth_token=token)
            self.assertEqual(list(Path(temporary).iterdir()), [])
            endpoint = Path(temporary, CHECKPOINT_ENDPOINT_NAME)
            endpoint.write_text("existing")
            with self.assertRaisesRegex(ValueError, "already exists"):
                StageCheckpointControl(temporary, _Runner(), auth_token=_TOKEN)
            self.assertEqual(endpoint.read_text(), "existing")

    @patch("distributed_runtime.checkpoint_control._unix_socket_available", return_value=False)
    def test_environment_enables_authenticated_loopback_control(self, _availability) -> None:
        with TemporaryDirectory() as temporary, patch.dict(os.environ, {
            "MYCELLIOS_EXECUTOR_WORKSPACE": temporary,
            CHECKPOINT_TOKEN_ENV: _TOKEN,
        }):
            control = checkpoint_control_from_environment(_Runner())
            self.assertIsNotNone(control)
            try:
                self.assertEqual(control.listener.family, socket.AF_INET)
            finally:
                control.close()

    @patch("distributed_runtime.checkpoint_control._unix_socket_available", return_value=False)
    def test_close_interrupts_incomplete_client_and_releases_listener(self, _availability) -> None:
        with TemporaryDirectory() as temporary:
            control = StageCheckpointControl(temporary, _Runner(), auth_token=_TOKEN)
            address = control.listener.getsockname()
            accepted = threading.Event()
            handle = control._handle

            def handle_incomplete(connection: socket.socket, deadline: float) -> None:
                accepted.set()
                handle(connection, deadline)

            try:
                with patch.object(control, "_handle", side_effect=handle_incomplete):
                    with socket.create_connection(address, timeout=2.0) as client:
                        client.sendall(b'{"operation":')
                        self.assertTrue(accepted.wait(2.0))
                        control.close()
                        control.close()
                        self.assertFalse(control.thread.is_alive())
                        self.assertFalse(control.path.exists())
                        with self.assertRaises(OSError):
                            socket.create_connection(address, timeout=0.2)
            finally:
                control.close()

    @patch("distributed_runtime.checkpoint_control._unix_socket_available", return_value=False)
    def test_descriptor_creation_failure_closes_all_sockets(self, _availability) -> None:
        reader, writer = socket.socketpair()
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        with TemporaryDirectory() as temporary:
            with patch("distributed_runtime.checkpoint_control.socket.socketpair", return_value=(reader, writer)):
                with patch("distributed_runtime.checkpoint_control.socket.socket", return_value=listener):
                    with patch("distributed_runtime.checkpoint_control.os.open", side_effect=PermissionError("denied")):
                        with self.assertRaises(PermissionError):
                            StageCheckpointControl(temporary, _Runner(), auth_token=_TOKEN)
            self.assertEqual(list(Path(temporary).iterdir()), [])
            self.assertEqual([reader.fileno(), writer.fileno(), listener.fileno()], [-1, -1, -1])

    @patch("distributed_runtime.checkpoint_control._unix_socket_available", return_value=False)
    def test_timed_out_restore_cannot_mutate_when_stage_resumes(self, _availability) -> None:
        with TemporaryDirectory() as temporary:
            runner = _Runner()
            control = StageCheckpointControl(temporary, runner, auth_token=_TOKEN)
            try:
                with patch("distributed_runtime.checkpoint_control.CONTROL_REQUEST_TIMEOUT_SECONDS", 0.2):
                    with _client(control, {
                        "operation": "restore", "requestId": 1, "maxBytes": 4,
                        "payloadBytes": 1, "committedPosition": 1,
                    }, b"X") as client:
                        self.assertEqual(select.select([control.wake_socket], [], [], 2.0)[0], [control.wake_socket])
                        response = _header(client)
                        self.assertFalse(response["ok"])
                        self.assertIn("timed out", response["error"])
                self.assertEqual(control.service_pending(), 0)
                self.assertIsNone(runner.restored)
            finally:
                control.close()

    @patch("distributed_runtime.checkpoint_control._unix_socket_available", return_value=False)
    def test_stage_rejects_expired_restore_before_background_timeout_poll(self, _availability) -> None:
        with TemporaryDirectory() as temporary:
            runner = _Runner()
            control = StageCheckpointControl(temporary, runner, auth_token=_TOKEN)
            try:
                with _client(control, {
                    "operation": "restore", "requestId": 1, "maxBytes": 4,
                    "payloadBytes": 1, "committedPosition": 1,
                }, b"X") as client:
                    self.assertEqual(select.select([control.wake_socket], [], [], 2.0)[0], [control.wake_socket])
                    with control.requests.mutex:
                        control.requests.queue[0].deadline = 0.0
                    self.assertEqual(control.service_pending(), 0)
                    self.assertFalse(_header(client)["ok"])
                self.assertIsNone(runner.restored)
            finally:
                control.close()

    @patch("distributed_runtime.checkpoint_control._unix_socket_available", return_value=False)
    def test_disconnected_restore_cannot_mutate_when_stage_resumes(self, _availability) -> None:
        with TemporaryDirectory() as temporary:
            runner = _Runner()
            control = StageCheckpointControl(temporary, runner, auth_token=_TOKEN)
            try:
                with _client(control, {
                    "operation": "restore", "requestId": 1, "maxBytes": 4,
                    "payloadBytes": 1, "committedPosition": 1,
                }, b"X") as client:
                    self.assertEqual(select.select([control.wake_socket], [], [], 2.0)[0], [control.wake_socket])
                    with control.requests.mutex:
                        request = control.requests.queue[0]
                    client.shutdown(socket.SHUT_RDWR)
                self.assertTrue(request.completed.wait(2.0))
                self.assertIn("disconnected", request.error)
                self.assertEqual(control.service_pending(), 0)
                self.assertIsNone(runner.restored)
            finally:
                control.close()

    @patch("distributed_runtime.checkpoint_control._unix_socket_available", return_value=False)
    def test_close_cancels_queued_restore_before_runner(self, _availability) -> None:
        with TemporaryDirectory() as temporary:
            runner = _Runner()
            control = StageCheckpointControl(temporary, runner, auth_token=_TOKEN)
            try:
                with _client(control, {
                    "operation": "restore", "requestId": 1, "maxBytes": 4,
                    "payloadBytes": 1, "committedPosition": 1,
                }, b"X"):
                    self.assertEqual(select.select([control.wake_socket], [], [], 2.0)[0], [control.wake_socket])
                    control.close()
                self.assertEqual(control.service_pending(), 0)
                self.assertIsNone(runner.restored)
                self.assertTrue(control.requests.empty())
            finally:
                control.close()

    @patch("distributed_runtime.checkpoint_control._unix_socket_available", return_value=False)
    def test_slow_unauthenticated_header_expires_and_next_client_is_serviced(self, _availability) -> None:
        self._assert_slow_framing_releases_server(b"{", b" ")

    @patch("distributed_runtime.checkpoint_control._unix_socket_available", return_value=False)
    def test_slow_authenticated_payload_expires_before_runner_and_releases_server(self, _availability) -> None:
        header = {
            "operation": "restore", "requestId": 1, "maxBytes": 64,
            "payloadBytes": 64, "committedPosition": 1, "token": _TOKEN,
        }
        self._assert_slow_framing_releases_server(json.dumps(header).encode() + b"\n", b"X")

    def _assert_slow_framing_releases_server(self, initial: bytes, fragment: bytes) -> None:
        with TemporaryDirectory() as temporary:
            runner = _Runner()
            control = StageCheckpointControl(temporary, runner, auth_token=_TOKEN)
            stop_dripping = threading.Event()
            fragment_sent = threading.Event()
            dripper = None
            try:
                with (
                    patch("distributed_runtime.checkpoint_control.CONTROL_REQUEST_TIMEOUT_SECONDS", 0.2),
                    socket.create_connection(control.listener.getsockname(), timeout=1.5) as attacker,
                ):
                    attacker.sendall(initial)

                    def drip() -> None:
                        while not stop_dripping.wait(0.025):
                            try:
                                attacker.sendall(fragment)
                                fragment_sent.set()
                            except OSError:
                                return

                    dripper = threading.Thread(target=drip)
                    dripper.start()
                    self.assertTrue(fragment_sent.wait(1.0))
                    try:
                        response = _header(attacker)
                        self.assertFalse(response["ok"])
                        self.assertIn("timed out", response["error"])
                    except (ConnectionResetError, EOFError):
                        # Rejected TCP input may reset on close with unread bytes.
                        pass
                    finally:
                        stop_dripping.set()
                        dripper.join(1.0)
                    self.assertFalse(dripper.is_alive())
                    self.assertTrue(control.requests.empty())
                    self.assertIsNone(runner.restored)
                    # Keep the attacker's socket open while another real client
                    # uses this same serial listener and the stage-owned queue.
                    with (
                        patch("distributed_runtime.checkpoint_control.CONTROL_REQUEST_TIMEOUT_SECONDS", 30.0),
                        _client(control, {"operation": "capture", "requestId": 17, "maxBytes": 64}) as legitimate,
                    ):
                        self.assertEqual(select.select([control.wake_socket], [], [], 2.0)[0], [control.wake_socket])
                        self.assertEqual(control.service_pending(), 1)
                        self.assertEqual(_header(legitimate), {
                            "ok": True, "payloadBytes": 5, "committedPosition": 31,
                        })
                        self.assertEqual(legitimate.recv(64), b"kv:17")
            finally:
                stop_dripping.set()
                if dripper is not None:
                    dripper.join(1.0)
                control.close()

    @unittest.skipUnless(os.name != "nt" and hasattr(socket, "AF_UNIX"), "requires Unix sockets")
    def test_unix_socket_preserves_owner_only_transport(self) -> None:
        with TemporaryDirectory() as temporary:
            control = StageCheckpointControl(temporary, _Runner())
            try:
                self.assertEqual(control.listener.family, socket.AF_UNIX)
                self.assertEqual(control.path.stat().st_mode & 0o777, 0o600)
                self.assertFalse(Path(temporary, CHECKPOINT_ENDPOINT_NAME).exists())
            finally:
                control.close()


def _restore_client(control: StageCheckpointControl, payload: bytes) -> None:
    with _client(control, {
        "operation": "restore",
        "requestId": 23,
        "maxBytes": 64,
        "payloadBytes": len(payload),
        "committedPosition": 11,
    }, payload) as client:
        response = _header(client)
        if response != {"ok": True, "payloadBytes": 0}:
            raise AssertionError(response)


def _client(
    control: StageCheckpointControl,
    header: dict[str, object],
    payload: bytes = b"",
    *, authenticate: bool = True,
) -> socket.socket:
    client = socket.socket(control.listener.family, socket.SOCK_STREAM)
    client.settimeout(2.0)
    if control.listener.family == socket.AF_INET:
        descriptor = json.loads(control.path.read_text(encoding="utf-8"))
        client.connect((descriptor["host"], descriptor["port"]))
        if authenticate:
            header = {**header, "token": _TOKEN}
    else:
        client.connect(str(control.path))
    client.sendall(json.dumps(header, separators=(",", ":")).encode() + b"\n" + payload)
    return client


def _header(client: socket.socket) -> dict[str, object]:
    data = bytearray()
    while True:
        byte = client.recv(1)
        if byte == b"\n":
            return json.loads(data)
        if not byte:
            raise EOFError
        data.extend(byte)


if __name__ == "__main__":
    unittest.main()
