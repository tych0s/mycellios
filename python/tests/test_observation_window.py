"""Regresión de las dos retractaciones por ventana de observación.

Los dos primeros tests reconstruyen con sus números reales los escenarios que
produjeron el ×7,7 retractado y el «goodput cero» de Exp10. Si alguien vuelve a
publicar una métrica bajo esas condiciones, fallan.
"""
from __future__ import annotations

import unittest

from distributed_runtime.observation_window import (
    DEFAULT_WINDOW_FACTOR,
    ObservationWindow,
    ObservationWindowError,
    diagnose,
    publish_throughput,
)


class RetractedMeasurementRegressionTests(unittest.TestCase):
    def test_the_retracted_long_output_measurement_is_rejected(self) -> None:
        """El ×7,7: 256 tokens a 1,8 tok/s = ~142 s, medidos en ventana de 60 s."""
        window = ObservationWindow(
            duration_seconds=60.0,
            slowest_request_seconds=142.0,
            completed_requests=3,
            started_requests=40,
        )
        self.assertFalse(window.is_adequate)
        with self.assertRaises(ObservationWindowError) as caught:
            publish_throughput("tokens_per_second", 1.8, window)
        # El mensaje tiene que explicar el defecto, no solo negarse.
        self.assertIn("truncamiento", str(caught.exception))

    def test_exp10_zero_goodput_is_rejected_as_arithmetic_expectation(self) -> None:
        """Exp10: ~1 280 s por respuesta en una ventana de 60 s.

        Cero completadas aquí NO es evidencia de colapso: es lo que da la
        aritmética en un sistema sano. Publicarlo como goodput cero fue la
        conclusión que este test impide repetir.
        """
        window = ObservationWindow.from_observations(
            duration_seconds=60.0,
            request_durations_seconds=[],  # ninguna acabó
            started_requests=100,
        )
        self.assertEqual(window.completed_requests, 0)
        self.assertEqual(window.truncation_ratio, 1.0)
        with self.assertRaises(ObservationWindowError):
            publish_throughput("goodput_requests_per_second", 0.0, window)

    def test_all_truncated_does_not_fake_a_valid_window(self) -> None:
        """Si nada acabó, la ventana no puede rellenarse con su propia duración.

        Sería el fallo más traicionero posible: hacer pasar la validación justo
        en el escenario que se quiere atrapar.
        """
        window = ObservationWindow.from_observations(
            duration_seconds=600.0,
            request_durations_seconds=[],
            started_requests=10,
        )
        self.assertEqual(window.slowest_request_seconds, 0.0)
        self.assertIsNotNone(window.failure_reason())


class AdequateWindowTests(unittest.TestCase):
    def test_a_window_five_times_the_slowest_request_publishes(self) -> None:
        window = ObservationWindow(
            duration_seconds=800.0,
            slowest_request_seconds=142.0,
            completed_requests=38,
            started_requests=40,
        )
        self.assertTrue(window.is_adequate)
        published = publish_throughput("tokens_per_second", 1.8, window)
        self.assertEqual(published["value"], 1.8)
        # La ventana viaja SIEMPRE con el dato: ése es el punto del módulo.
        self.assertIn("observation_window", published)
        self.assertTrue(published["observation_window"]["adequate"])

    def test_truncation_ratio_is_reported_even_when_valid(self) -> None:
        window = ObservationWindow(
            duration_seconds=800.0,
            slowest_request_seconds=100.0,
            completed_requests=90,
            started_requests=100,
        )
        described = window.describe()
        self.assertEqual(described["truncated_requests"], 10)
        self.assertAlmostEqual(described["truncation_ratio"], 0.1)

    def test_required_window_uses_the_configured_factor(self) -> None:
        window = ObservationWindow(
            duration_seconds=100.0,
            slowest_request_seconds=10.0,
            completed_requests=5,
            started_requests=5,
            factor=DEFAULT_WINDOW_FACTOR,
        )
        self.assertEqual(window.required_seconds, 50.0)
        self.assertTrue(window.is_adequate)

    def test_diagnose_explains_rather_than_only_flagging(self) -> None:
        bad = ObservationWindow(
            duration_seconds=60.0,
            slowest_request_seconds=142.0,
            completed_requests=0,
            started_requests=40,
        )
        text = diagnose(bad)
        self.assertIn("NO VÁLIDA", text)
        self.assertIn("142", text)

        good = ObservationWindow(
            duration_seconds=800.0,
            slowest_request_seconds=100.0,
            completed_requests=40,
            started_requests=40,
        )
        self.assertIn("válida", diagnose(good))


if __name__ == "__main__":
    unittest.main()
