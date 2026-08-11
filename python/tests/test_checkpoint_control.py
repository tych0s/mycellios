from __future__ import annotations

import json
from pathlib import Path
import select
import socket
from tempfile import TemporaryDirectory
import threading
import unittest

from distributed_runtime.checkpoint_control import StageCheckpointControl


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
            control = StageCheckpointControl(temporary, runner)
            try:
                capture: dict[str, object] = {}

                def capture_client() -> None:
                    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                        client.connect(str(control.path))
                        client.sendall(b'{"operation":"capture","requestId":17,"maxBytes":64}\n')
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
                    args=(control.path, b"safe-kv"),
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

    def test_rejects_oversized_or_truncated_restore_before_runner(self) -> None:
        with TemporaryDirectory() as temporary:
            runner = _Runner()
            control = StageCheckpointControl(temporary, runner)
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                    client.connect(str(control.path))
                    client.sendall(
                        b'{"operation":"restore","requestId":1,"maxBytes":4,'
                        b'"payloadBytes":5,"committedPosition":1}\n'
                    )
                    self.assertEqual(_header(client)["ok"], False)
                self.assertIsNone(runner.restored)
            finally:
                control.close()

    def test_rejects_unknown_header_fields_without_reaching_runner(self) -> None:
        with TemporaryDirectory() as temporary:
            runner = _Runner()
            control = StageCheckpointControl(temporary, runner)
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                    client.connect(str(control.path))
                    client.sendall(
                        b'{"operation":"restore","requestId":1,"maxBytes":4,'
                        b'"payloadBytes":1,"committedPosition":1,"unexpected":true}\nX'
                    )
                    self.assertEqual(_header(client)["ok"], False)
                self.assertIsNone(runner.restored)
            finally:
                control.close()


def _restore_client(path: Path, payload: bytes) -> None:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(str(path))
        header = {
            "operation": "restore",
            "requestId": 23,
            "maxBytes": 64,
            "payloadBytes": len(payload),
            "committedPosition": 11,
        }
        client.sendall(json.dumps(header, separators=(",", ":")).encode() + b"\n" + payload)
        response = _header(client)
        if response != {"ok": True, "payloadBytes": 0}:
            raise AssertionError(response)


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
