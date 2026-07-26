"""La clasificación de pilas y la banda sin resolución del perfilador T2."""
from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest

_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[2] / "scripts" / "profile_scheduler.py"
)
_spec = importlib.util.spec_from_file_location("profile_scheduler", _SCRIPT)
assert _spec and _spec.loader
profile_scheduler = importlib.util.module_from_spec(_spec)
# `@dataclass` resuelve anotaciones mirando `sys.modules[cls.__module__]`, así
# que el módulo tiene que estar registrado ANTES de ejecutarlo.
sys.modules["profile_scheduler"] = profile_scheduler
_spec.loader.exec_module(profile_scheduler)


class ClassificationTests(unittest.TestCase):
    def test_socket_wait_is_not_gil_contention(self) -> None:
        # Un hilo bloqueado en un socket no compite por el GIL. Contarlo como
        # ocupado inflaría el ciclo de trabajo y podría disparar T2 en falso.
        self.assertEqual(
            profile_scheduler._classify('select.select (selectors.py:323)'),
            "waiting",
        )
        self.assertEqual(
            profile_scheduler._classify('socket.recv (socket.py:12)'),
            "waiting",
        )

    def test_torch_frames_count_as_native(self) -> None:
        # Dentro de torch el GIL normalmente está suelto: es trabajo, pero no
        # trabajo que serialice a los demás hilos.
        self.assertEqual(
            profile_scheduler._classify("forward (torch/nn/modules/linear.py:114)"),
            "native",
        )

    def test_plain_python_frames_count_against_the_gil(self) -> None:
        self.assertEqual(
            profile_scheduler._classify(
                "_dispatch_wave (distributed_runtime/engine.py:4555)"
            ),
            "python",
        )


class DumpParsingTests(unittest.TestCase):
    def test_keeps_the_deepest_frame_per_thread(self) -> None:
        dump = "\n".join(
            [
                'Thread 1234 ("MainThread")',
                "    _dispatch_wave (distributed_runtime/engine.py:4555)",
                "    run (distributed_runtime/engine.py:900)",
                'Thread 5678 ("stage-io")',
                "    select.select (selectors.py:323)",
                "    poll (asyncio/base_events.py:1)",
            ]
        )
        threads = profile_scheduler._parse_dump(dump)
        self.assertEqual(len(threads), 2)
        self.assertIn("MainThread#1234", threads)
        # El frame activo es el primero, no el último.
        self.assertIn("engine.py:4555", threads["MainThread#1234"])
        self.assertIn("selectors.py:323", threads["stage-io#5678"])


class DutyCycleTests(unittest.TestCase):
    def test_duty_cycle_counts_only_bytecode(self) -> None:
        profile = profile_scheduler.ThreadProfile(thread="t")
        profile.python_samples = 10
        profile.native_samples = 10
        profile.waiting_samples = 80
        # Ocupado el 20 %, pero solo el 10 % serializa el GIL.
        self.assertAlmostEqual(profile.gil_duty_cycle, 0.10)
        self.assertAlmostEqual(profile.busy_fraction, 0.20)

    def test_empty_profile_does_not_divide_by_zero(self) -> None:
        profile = profile_scheduler.ThreadProfile(thread="t")
        self.assertEqual(profile.gil_duty_cycle, 0.0)
        self.assertEqual(profile.busy_fraction, 0.0)

    def test_inconclusive_band_brackets_the_threshold(self) -> None:
        # La banda tiene que CONTENER el umbral: si no, habría valores que
        # disparan T2 sin pasar por la comprobación de resolución.
        low, high = profile_scheduler.T2_INCONCLUSIVE_BAND
        self.assertLess(low, profile_scheduler.T2_THRESHOLD)
        self.assertGreater(high, profile_scheduler.T2_THRESHOLD)


if __name__ == "__main__":
    unittest.main()
