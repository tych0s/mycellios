"""Pruebas del relé TCP-sobre-WS de `deploy/salad/relay.py`.

Hasta ahora `deploy/salad/` (3.956 líneas, la pieza que sostiene TODO el
multi-nodo real) no tenía ni una prueba. Estas cubren la ruta crítica de
seguridad del HELLO, que era explotable a distancia sin autenticación.

A propósito NO importan `distributed_runtime`: `relay.py` sólo necesita la
biblioteca estándar en su nivel superior (aiohttp se importa dentro de
`_main`), así que estas pruebas corren aunque no haya torch instalado.
"""
import importlib.util
import os
import struct
import unittest
from pathlib import Path

_RELAY_PATH = Path(__file__).resolve().parents[2] / "deploy" / "salad" / "relay.py"
_BRIDGE_PATH = Path(__file__).resolve().parents[2] / "deploy" / "salad" / "bridge.py"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _relay(secret=b"", allow_anonymous=False):
    module = _load("gdlp_relay_under_test", _RELAY_PATH)
    module.RELAY_SECRET = secret
    module.ALLOW_ANONYMOUS = allow_anonymous
    return module


def _hello(room, eid, gen, recv, signature=b""):
    rb = room.encode()
    return (bytes([0x10]) + struct.pack(">H", len(rb)) + rb + eid
            + struct.pack(">Q", gen) + struct.pack(">Q", recv) + signature)


class HelloParsingTests(unittest.TestCase):
    def test_round_trips_every_field(self):
        relay = _relay(allow_anonymous=True)
        eid = bytes(range(16))
        frame = _hello("f0", eid, 7, 4096, b"\xAB" * 32)
        room, got_eid, gen, recv, signature = relay.parse_hello(frame)
        self.assertEqual(room, "f0")
        self.assertEqual(got_eid, eid)
        self.assertEqual(gen, 7)
        self.assertEqual(recv, 4096)
        self.assertEqual(signature, b"\xAB" * 32)

    def test_missing_signature_reads_as_empty(self):
        relay = _relay(allow_anonymous=True)
        _, _, _, _, signature = relay.parse_hello(_hello("f0", b"\x01" * 16, 1, 0))
        self.assertEqual(signature, b"")


class AuthorizationTests(unittest.TestCase):
    def test_rejects_unsigned_hello_when_a_secret_is_configured(self):
        relay = _relay(secret=b"topsecret")
        self.assertEqual(
            relay.authorize_hello("f0", b"\x01" * 16, 1, b""),
            "hmac_invalido",
        )

    def test_rejects_signature_minted_for_another_room(self):
        relay = _relay(secret=b"topsecret")
        eid = b"\x01" * 16
        stolen = relay.hello_signature("ret", eid, 1)
        self.assertEqual(relay.authorize_hello("f0", eid, 1, stolen), "hmac_invalido")

    def test_rejects_signature_replayed_at_a_higher_generation(self):
        # Sin ligar `gen` a la firma, un atacante que capturase un HELLO válido
        # podría reenviarlo con gen alto y provocar el reset de la sala.
        relay = _relay(secret=b"topsecret")
        eid = b"\x01" * 16
        captured = relay.hello_signature("f0", eid, 1)
        self.assertEqual(relay.authorize_hello("f0", eid, 2, captured), "hmac_invalido")

    def test_accepts_a_correctly_signed_hello(self):
        relay = _relay(secret=b"topsecret")
        eid = b"\x01" * 16
        good = relay.hello_signature("f0", eid, 3)
        self.assertIsNone(relay.authorize_hello("f0", eid, 3, good))

    def test_fails_closed_when_no_secret_is_configured(self):
        # El relé vive en una URL pública: sin secreto debe negarse a servir,
        # no aceptar a cualquiera.
        relay = _relay(secret=b"", allow_anonymous=False)
        self.assertEqual(
            relay.authorize_hello("f0", b"\x01" * 16, 1, b""),
            "falta_secreto",
        )

    def test_anonymous_mode_is_opt_in_for_local_tests(self):
        relay = _relay(secret=b"", allow_anonymous=True)
        self.assertIsNone(relay.authorize_hello("f0", b"\x01" * 16, 1, b""))


class AdmissionOrderTests(unittest.TestCase):
    """El fallo original: el reset por generación corría ANTES de comprobar si
    había hueco, así que un tercer endpoint destruía los buffers de los dos
    legítimos y sólo después era rechazado."""

    def _room_with_live_traffic(self, relay):
        room = relay.Room("f0")
        first, denied = relay.admit_slot(room, b"\x01" * 16, 1)
        self.assertIsNone(denied)
        second, denied = relay.admit_slot(room, b"\x02" * 16, 1)
        self.assertIsNone(denied)
        first.out_buf += b"payload-en-vuelo"
        second.out_buf += b"respuesta-en-vuelo"
        return room, first, second

    def test_third_endpoint_is_rejected(self):
        relay = _relay(allow_anonymous=True)
        room, _, _ = self._room_with_live_traffic(relay)
        slot, denied = relay.admit_slot(room, b"\x03" * 16, 1)
        self.assertIsNone(slot)
        self.assertEqual(denied, "sala_llena")

    def test_third_endpoint_cannot_wipe_the_buffers_of_the_legitimate_pair(self):
        relay = _relay(allow_anonymous=True)
        room, first, second = self._room_with_live_traffic(relay)
        slot, denied = relay.admit_slot(room, b"\x03" * 16, 9_999)
        self.assertEqual(denied, "sala_llena")
        self.assertIsNone(slot)
        self.assertEqual(bytes(first.out_buf), b"payload-en-vuelo")
        self.assertEqual(bytes(second.out_buf), b"respuesta-en-vuelo")
        self.assertEqual(room.gen, 1, "una generación ajena no debe avanzar la sala")

    def test_poisoning_the_generation_is_not_possible_from_outside(self):
        # gen = 2^63 dejaba la sala inservible: ningún run posterior podía
        # superarla, así que nunca volvía a resetearse.
        relay = _relay(allow_anonymous=True)
        room, _, _ = self._room_with_live_traffic(relay)
        relay.admit_slot(room, b"\x03" * 16, 2 ** 63)
        self.assertEqual(room.gen, 1)

    def test_legitimate_reconnect_keeps_its_slot_and_buffer(self):
        relay = _relay(allow_anonymous=True)
        room, first, _ = self._room_with_live_traffic(relay)
        again, denied = relay.admit_slot(room, b"\x01" * 16, 1)
        self.assertIsNone(denied)
        self.assertIs(again, first)
        self.assertEqual(bytes(again.out_buf), b"payload-en-vuelo")

    def test_a_genuinely_new_run_still_resets_the_room(self):
        # La función legítima del reset no debe perderse con el arreglo.
        relay = _relay(allow_anonymous=True)
        room, first, second = self._room_with_live_traffic(relay)
        slot, denied = relay.admit_slot(room, b"\x01" * 16, 2)
        self.assertIsNone(denied)
        self.assertIs(slot, first)
        self.assertEqual(room.gen, 2)
        self.assertEqual(bytes(first.out_buf), b"")
        self.assertEqual(bytes(second.out_buf), b"")
        self.assertEqual(second.fwd_ptr, 0)


class BufferBoundTests(unittest.TestCase):
    def test_a_bound_is_configured_by_default(self):
        relay = _relay(allow_anonymous=True)
        self.assertGreater(relay.MAX_BUFFER_BYTES, 0)
        # El contenedor del relé tiene 2 GB; la cota debe dejar margen.
        self.assertLess(relay.MAX_BUFFER_BYTES, 2 * 1024 * 1024 * 1024)


class BridgeInteropTests(unittest.TestCase):
    """El puente y el relé calculan la firma por separado: si divergen, el
    despliegue entero deja de emparejar y el fallo sólo se ve en producción."""

    def test_bridge_and_relay_agree_on_the_signature(self):
        os.environ["GDLP_RELAY_SECRET"] = "shared-secret"
        try:
            bridge = _load("gdlp_bridge_under_test", _BRIDGE_PATH)
            relay = _relay(secret=b"shared-secret")
            eid = bytes(range(16))
            self.assertEqual(
                bridge.hello_signature("f0", eid, 5),
                relay.hello_signature("f0", eid, 5),
            )
        finally:
            os.environ.pop("GDLP_RELAY_SECRET", None)

    def test_a_bridge_hello_is_accepted_by_the_relay(self):
        os.environ["GDLP_RELAY_SECRET"] = "shared-secret"
        try:
            bridge = _load("gdlp_bridge_under_test", _BRIDGE_PATH)
            relay = _relay(secret=b"shared-secret")
            eid = bytes(range(16))
            frame = bridge.enc_hello("f0", eid, 5, 1234)
            room, got_eid, gen, recv, signature = relay.parse_hello(frame)
            self.assertEqual((room, got_eid, gen, recv), ("f0", eid, 5, 1234))
            self.assertIsNone(relay.authorize_hello(room, got_eid, gen, signature))
        finally:
            os.environ.pop("GDLP_RELAY_SECRET", None)

    def test_bridge_omits_the_signature_when_no_secret_is_set(self):
        os.environ.pop("GDLP_RELAY_SECRET", None)
        bridge = _load("gdlp_bridge_under_test", _BRIDGE_PATH)
        frame = bridge.enc_hello("f0", bytes(range(16)), 1, 0)
        self.assertEqual(len(frame), 1 + 2 + 2 + 16 + 8 + 8)


if __name__ == "__main__":
    unittest.main()
