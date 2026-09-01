"""Integración real del relé: procesos, WebSocket y TCP de verdad.

`python/tests/test_salad_relay.py` prueba la ruta de seguridad del HELLO como
funciones puras. Esto comprueba lo otro: que el relé **arranca y empareja**, que
un HELLO firmado entra, que uno sin firmar NO, y que un tercer endpoint no puede
tirar una tubería viva. Sin esto, un error de montaje (un import, un argumento,
un orden) sólo aparecería en un despliegue de pago.

Se salta si falta `aiohttp`, que el relé sólo necesita en tiempo de ejecución.
"""
from __future__ import annotations

import hashlib
import hmac
import importlib.util
import json
import os
import socket
import struct
import subprocess
import sys
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RELAY = REPO / "deploy" / "salad" / "relay.py"
SECRET = "integration-secret"
HELLO, HELLO_ACK, DATA, ACK, ACKW = 0x10, 0x11, 0x20, 0x21, 0x22

_HAS_AIOHTTP = importlib.util.find_spec("aiohttp") is not None


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _sign(room: str, eid: bytes, gen: int) -> bytes:
    mac = hmac.new(SECRET.encode(), digestmod=hashlib.sha256)
    mac.update(room.encode())
    mac.update(b"\x00")
    mac.update(eid)
    mac.update(struct.pack(">Q", gen))
    return mac.digest()


def _hello(room: str, eid: bytes, gen: int, recv: int, signature: bytes) -> bytes:
    rb = room.encode()
    return (bytes([HELLO]) + struct.pack(">H", len(rb)) + rb + eid
            + struct.pack(">Q", gen) + struct.pack(">Q", recv) + signature)


@unittest.skipUnless(_HAS_AIOHTTP, "el relé necesita aiohttp en ejecución")
class RelayHandshakeIntegrationTests(unittest.TestCase):
    """Arranca un relé de verdad y le habla por WebSocket."""

    @classmethod
    def setUpClass(cls):
        cls.port = _free_port()
        env = dict(os.environ)
        env.update({
            "GDLP_PORT": str(cls.port),
            "GDLP_BIND": "127.0.0.1",
            "GDLP_SKIP_PIP": "1",
            "GDLP_RELAY_SECRET": SECRET,
            "PYTHONUNBUFFERED": "1",
        })
        cls.proc = subprocess.Popen(
            [sys.executable, str(RELAY)], env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        deadline = time.time() + 40
        while time.time() < deadline:
            if cls.proc.poll() is not None:
                raise RuntimeError(f"el relé murió al arrancar:\n{cls.proc.stdout.read()[:2000]}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{cls.port}/healthz", timeout=2) as r:
                    if r.status == 200:
                        return
            except (urllib.error.URLError, OSError):
                time.sleep(0.4)
        raise RuntimeError("el relé no llegó a responder /healthz")

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate()
        try:
            cls.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            cls.proc.kill()

    def _status(self):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/status", timeout=5) as r:
            return json.loads(r.read())

    def _connect(self, room, eid, gen=1, recv=0, signature=None):
        """Abre un WS y manda el HELLO. Devuelve (ws, primer_mensaje_o_None)."""
        from aiohttp import ClientSession, WSMsgType
        import asyncio

        async def _run():
            session = ClientSession()
            ws = await session.ws_connect(f"http://127.0.0.1:{self.port}/relay", timeout=10)
            sig = _sign(room, eid, gen) if signature is None else signature
            await ws.send_bytes(_hello(room, eid, gen, recv, sig))
            try:
                msg = await asyncio.wait_for(ws.receive(), timeout=6)
            except asyncio.TimeoutError:
                msg = None
            kind = None if msg is None else msg.type
            data = None if msg is None or kind != WSMsgType.BINARY else msg.data
            await ws.close()
            await session.close()
            return kind, data

        return asyncio.new_event_loop().run_until_complete(_run())

    def test_the_relay_is_up_and_reports_status(self):
        state = self._status()
        self.assertEqual(state["role"], "relay")
        self.assertTrue(state.get("ready"))

    def test_a_correctly_signed_hello_is_accepted(self):
        kind, data = self._connect("f0", b"\x11" * 16)
        self.assertIsNotNone(data, "el relé cerró un HELLO válido")
        self.assertEqual(data[0], HELLO_ACK)

    def test_an_unsigned_hello_is_refused(self):
        from aiohttp import WSMsgType

        kind, data = self._connect("f0", b"\x22" * 16, signature=b"")
        # El relé cierra el WS: o llega un CLOSE, o no llega nada.
        self.assertNotEqual(
            (kind, data and data[0]), (WSMsgType.BINARY, HELLO_ACK),
            "un HELLO sin firma obtuvo HELLO_ACK: el relé está abierto",
        )

    def test_a_signature_from_another_room_is_refused(self):
        from aiohttp import WSMsgType

        eid = b"\x33" * 16
        kind, data = self._connect("f0", eid, signature=_sign("ret", eid, 1))
        self.assertNotEqual((kind, data and data[0]), (WSMsgType.BINARY, HELLO_ACK))

    def test_a_third_endpoint_cannot_evict_a_live_pair(self):
        """El fallo original: un tercer HELLO con `gen` alto vaciaba la sala."""
        from aiohttp import WSMsgType

        room = "pair"
        first, second = b"\xA1" * 16, b"\xA2" * 16
        self.assertEqual(self._connect(room, first)[1][0], HELLO_ACK)
        self.assertEqual(self._connect(room, second)[1][0], HELLO_ACK)
        before = self._status()["rooms"][room]["slots"]
        self.assertEqual(before, 2)

        # Tercero, firmado correctamente pero sobrante, con generación altísima.
        kind, data = self._connect(room, b"\xA3" * 16, gen=2 ** 62)
        self.assertNotEqual(
            (kind, data and data[0]), (WSMsgType.BINARY, HELLO_ACK),
            "la sala admitió un tercer endpoint",
        )
        after = self._status()["rooms"][room]
        self.assertEqual(after["slots"], 2, "el tercero alteró la sala")


if __name__ == "__main__":
    unittest.main()
