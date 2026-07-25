"""El régimen de medida va pegado al dato, no en la memoria de quien lo leyó."""
from __future__ import annotations

import os
import unittest

from distributed_runtime.compute_timing import (
    DISPATCH,
    ENV_VAR,
    SYNC,
    ComputeSample,
    ComputeTimer,
    resolve_timing_mode,
)


class TimingModeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._previous = os.environ.get(ENV_VAR)
        os.environ.pop(ENV_VAR, None)

    def tearDown(self) -> None:
        os.environ.pop(ENV_VAR, None)
        if self._previous is not None:
            os.environ[ENV_VAR] = self._previous

    def test_default_mode_is_dispatch(self) -> None:
        self.assertEqual(resolve_timing_mode(), DISPATCH)

    def test_env_var_selects_sync(self) -> None:
        os.environ[ENV_VAR] = "sync"
        self.assertEqual(resolve_timing_mode(), SYNC)

    def test_invalid_mode_fails_closed(self) -> None:
        # Falla cerrado a propósito: un modo mal escrito que cayera en silencio
        # a `dispatch` produciría una campaña entera etiquetada como kernel time
        # cuando en realidad midió despacho. Ese es el fallo caro.
        os.environ[ENV_VAR] = "syncronized"
        with self.assertRaises(ValueError):
            resolve_timing_mode()


class SampleNamingTests(unittest.TestCase):
    def test_dispatch_sample_refuses_the_compute_ms_name(self) -> None:
        """La aserción central: sin sincronizar NO puede llamarse compute_ms."""
        sample = ComputeSample(elapsed_ms=3.2, mode=DISPATCH)
        self.assertFalse(sample.is_kernel_time)
        self.assertEqual(sample.metric_key(), "compute_dispatch_ms")

    def test_sync_sample_earns_the_compute_ms_name(self) -> None:
        sample = ComputeSample(elapsed_ms=33.4, mode=SYNC)
        self.assertTrue(sample.is_kernel_time)
        self.assertEqual(sample.metric_key(), "compute_ms")


class ComputeTimerTests(unittest.TestCase):
    def test_timer_measures_and_labels(self) -> None:
        timer = ComputeTimer(DISPATCH)
        with timer.measure() as measurement:
            sum(range(10_000))
        self.assertGreater(measurement.elapsed_ms, 0)
        self.assertEqual(measurement.metric_key(), "compute_dispatch_ms")

    def test_no_sample_for_a_failed_forward(self) -> None:
        """Un forward que lanza no aporta muestra: su duración parcial no es cómputo."""
        timer = ComputeTimer(DISPATCH)
        with self.assertRaises(RuntimeError):
            with timer.measure() as measurement:
                raise RuntimeError("forward exploded")
        self.assertEqual(measurement.elapsed_ms, 0.0)

    def test_sync_mode_is_a_noop_without_a_device(self) -> None:
        # En CPU no hay barrera que poner, pero el modo se conserva: el dato
        # sigue siendo kernel time porque en CPU despacho y ejecución coinciden.
        timer = ComputeTimer(SYNC)
        with timer.measure() as measurement:
            sum(range(1_000))
        self.assertEqual(measurement.metric_key(), "compute_ms")


if __name__ == "__main__":
    unittest.main()
