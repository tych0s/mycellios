"""El emulador de enlace debe modelar la propagación como PROPAGACIÓN.

Bug histórico (bug 1 del §7 de `docs/01_TRASPASO_PARA_CONTINUAR.md`, vivo desde
que existe el emulador): `wait_before_send` hacía `time.sleep` en el hilo emisor,
así que el retardo de ida se cobraba en SERIE. Con W ventanas en vuelo el
emulador cobraba W×retardo donde la red cobra retardo UNA vez, de modo que
cualquier medida de la cinta especulativa daba un falso negativo garantizado:
parecía que solapar no servía aunque la implementación fuese perfecta.

Estas pruebas fijan las dos propiedades que lo distinguen:
  - PARIDAD a W=1: un único frame sigue tardando lo mismo que antes.
  - SOLAPE a W>1: N frames NO cuestan N×retardo.
Y la propiedad que no se puede perder por el camino: el ORDEN de los bytes.
"""
import socket
import struct
import threading
import time
import unittest

from distributed_runtime.protocol import (
    FrameType,
    LinkEmulator,
    close_emulated_link,
    recv_frame,
    send_frame,
)

DELAY_MS = 60.0
DELAY_S = DELAY_MS / 1000.0
# El reloj de Windows y el planificador del SO meten ruido; los umbrales dejan
# margen de sobra para que la prueba distinga MECANISMOS, no milisegundos.
TOLERANCE_S = 0.35


class _Link:
    """Par de sockets conectados por loopback, con lector en segundo plano."""

    def __init__(self):
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        self.client = socket.create_connection(listener.getsockname())
        self.server, _ = listener.accept()
        listener.close()
        self.received = []
        self._reader = None

    def read_frames(self, count):
        """Lee `count` frames en un hilo y anota CUÁNDO llegó cada uno."""
        started = time.monotonic()

        def _run():
            for _ in range(count):
                frame = recv_frame(self.server)
                self.received.append((time.monotonic() - started, frame))

        self._reader = threading.Thread(target=_run, daemon=True)
        self._reader.start()
        return self._reader

    def close(self):
        close_emulated_link(self.client)
        for sock in (self.client, self.server):
            try:
                sock.close()
            except OSError:
                pass


def _send(link, emulator, step, payload=b""):
    # ERROR es el único tipo con payload libre y `step` libre, así que sirve de
    # portador neutro para medir el transporte sin pelearse con el validador.
    return send_frame(
        link.client,
        FrameType.ERROR,
        request_id=1,
        step=step,
        payload=payload,
        emulator=emulator,
    )


class PropagationIsNotSerialTests(unittest.TestCase):
    def setUp(self):
        self.link = _Link()
        self.addCleanup(self.link.close)

    def test_one_frame_still_takes_one_delay(self):
        """PARIDAD a W=1: el arreglo no debe abaratar el caso de un solo frame."""
        emulator = LinkEmulator(one_way_delay_ms=DELAY_MS)
        reader = self.link.read_frames(1)
        _send(self.link, emulator, step=0)
        reader.join(timeout=10)
        self.assertEqual(len(self.link.received), 1)
        arrival, _ = self.link.received[0]
        self.assertGreaterEqual(
            arrival, DELAY_S * 0.5,
            "el frame llegó demasiado pronto: la propagación no se está aplicando",
        )
        self.assertLess(arrival, DELAY_S + TOLERANCE_S)

    def test_the_sender_is_not_blocked_by_propagation(self):
        """El emisor debe volver enseguida: la red no le hace esperar a que llegue."""
        emulator = LinkEmulator(one_way_delay_ms=DELAY_MS)
        reader = self.link.read_frames(4)
        started = time.monotonic()
        for step in range(4):
            _send(self.link, emulator, step=step)
        elapsed_sending = time.monotonic() - started
        # Con el bug, enviar 4 frames costaba 4×60 = 240 ms de reloj DEL EMISOR.
        self.assertLess(
            elapsed_sending, DELAY_S * 2,
            f"enviar 4 frames bloqueó al emisor {elapsed_sending * 1000:.0f} ms: "
            "la propagación se sigue cobrando en serie",
        )
        reader.join(timeout=10)
        self.assertEqual(len(self.link.received), 4)

    def test_frames_in_flight_overlap_instead_of_queueing(self):
        """SOLAPE: N frames en vuelo no cuestan N retardos."""
        emulator = LinkEmulator(one_way_delay_ms=DELAY_MS)
        windows = 4
        reader = self.link.read_frames(windows)
        for step in range(windows):
            _send(self.link, emulator, step=step)
        reader.join(timeout=15)
        self.assertEqual(len(self.link.received), windows)
        last_arrival, _ = self.link.received[-1]
        serial_cost = DELAY_S * windows
        self.assertLess(
            last_arrival, serial_cost * 0.6,
            f"el último de {windows} frames llegó a {last_arrival * 1000:.0f} ms; "
            f"en serie serían {serial_cost * 1000:.0f} ms — no hay solape",
        )

    def test_delivery_order_matches_send_order(self):
        """Un hilo de entrega por enlace: reordenar partiría el stream."""
        emulator = LinkEmulator(one_way_delay_ms=5.0)
        count = 25
        reader = self.link.read_frames(count)
        for step in range(count):
            _send(self.link, emulator, step=step)
        reader.join(timeout=15)
        self.assertEqual(len(self.link.received), count)
        self.assertEqual([frame.step for _, frame in self.link.received], list(range(count)))

    def test_payloads_survive_the_delayed_path_intact(self):
        emulator = LinkEmulator(one_way_delay_ms=5.0)
        payloads = [struct.pack(">I", n) * (n + 1) for n in range(8)]
        reader = self.link.read_frames(len(payloads))
        for step, payload in enumerate(payloads):
            _send(self.link, emulator, step=step, payload=payload)
        reader.join(timeout=15)
        self.assertEqual([frame.payload for _, frame in self.link.received], payloads)

    def test_a_socket_with_an_emulated_link_keeps_using_it(self):
        """Mezclar `sendall` directo con el hilo de entrega intercalaría bytes."""
        emulator = LinkEmulator(one_way_delay_ms=5.0)
        reader = self.link.read_frames(3)
        _send(self.link, emulator, step=0)
        _send(self.link, None, step=1)                    # sin emulador
        _send(self.link, LinkEmulator(), step=2)          # emulador sin retardo
        reader.join(timeout=15)
        self.assertEqual([frame.step for _, frame in self.link.received], [0, 1, 2])


class WithoutEmulationNothingChangesTests(unittest.TestCase):
    """El camino de producción (sin retardo configurado) queda intacto."""

    def setUp(self):
        self.link = _Link()
        self.addCleanup(self.link.close)

    def test_no_emulator_sends_synchronously(self):
        reader = self.link.read_frames(1)
        _send(self.link, None, step=0, payload=b"hola")
        reader.join(timeout=5)
        self.assertEqual(len(self.link.received), 1)
        arrival, frame = self.link.received[0]
        self.assertLess(arrival, 0.25)
        self.assertEqual(frame.payload, b"hola")

    def test_a_zero_delay_emulator_does_not_spawn_a_delivery_thread(self):
        before = threading.active_count()
        reader = self.link.read_frames(1)
        _send(self.link, LinkEmulator(), step=0)
        reader.join(timeout=5)
        # +1 por el hilo lector de la propia prueba; el emulador no debe añadir otro.
        self.assertLessEqual(threading.active_count(), before + 1)


class CostModelTests(unittest.TestCase):
    def test_serialization_and_propagation_are_reported_separately(self):
        emulator = LinkEmulator(one_way_delay_ms=100.0, bandwidth_mbps=8.0)
        self.assertAlmostEqual(emulator.propagation_seconds, 0.1)
        # 1 MB a 8 Mbps = 1 s.
        self.assertAlmostEqual(emulator.serialization_seconds(1_000_000), 1.0)

    def test_bandwidth_is_optional(self):
        self.assertEqual(LinkEmulator(one_way_delay_ms=10.0).serialization_seconds(10_000), 0.0)

    def test_negative_delay_is_clamped(self):
        self.assertEqual(LinkEmulator(one_way_delay_ms=-5.0).propagation_seconds, 0.0)


if __name__ == "__main__":
    unittest.main()
