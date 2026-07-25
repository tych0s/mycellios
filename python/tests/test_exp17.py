"""EXP17 se niega a publicar lo que no cumple los criterios preregistrados."""
from __future__ import annotations

import unittest

from distributed_runtime.exp17 import (
    ARM_C_MIN_TOKENS_PER_SECOND,
    Arm,
    ArmResult,
    evaluate,
    publish,
)
from distributed_runtime.observation_window import (
    ObservationWindow,
    ObservationWindowError,
)


def _window(*, duration: float = 900.0, slowest: float = 120.0) -> ObservationWindow:
    return ObservationWindow(
        duration_seconds=duration,
        slowest_request_seconds=slowest,
        completed_requests=20,
        started_requests=20,
    )


def _arm(
    arm: Arm,
    *,
    tps: float = 4.0,
    exact: float = 1.0,
    hops: float = 2.0,
    network: float = 180.0,
    forward: float = 25.0,
    residual: float = 3.0,
    runs: int = 5,
    footprint: int | None = 512 * 1024 * 1024,
    window: ObservationWindow | None = None,
) -> ArmResult:
    return ArmResult(
        arm=arm,
        tokens_per_second_p50=tps,
        exact_rate=exact,
        hop_coefficient=hops,
        network_ms_per_token=network,
        forward_ms_per_token=forward,
        python_residual_ms_per_token=residual,
        runs=runs,
        window=window or _window(),
        base_footprint_bytes=footprint,
    )


def _full_set(**overrides: ArmResult) -> dict[Arm, ArmResult]:
    results = {
        Arm.A_BASE: _arm(Arm.A_BASE, tps=1.8, hops=4.0, network=496.0),
        Arm.B_GEOGRAPHY: _arm(Arm.B_GEOGRAPHY, tps=2.6, hops=4.0, network=330.0),
        Arm.C_TOPOLOGY: _arm(Arm.C_TOPOLOGY, tps=4.0, hops=2.0, network=180.0),
        Arm.D_ENGINE: _arm(Arm.D_ENGINE, tps=5.0, hops=2.0, forward=18.0, residual=3.0),
    }
    results.update({result.arm: result for result in overrides.values()})
    return results


class ExactnessTests(unittest.TestCase):
    def test_a_faster_arm_that_changes_tokens_is_rejected(self) -> None:
        """Un brazo más rápido que no es token-exacto es OTRO sistema, no mejora."""
        results = _full_set(c=_arm(Arm.C_TOPOLOGY, tps=9.0, exact=0.98))
        verdict = evaluate(results)
        self.assertFalse(verdict.passed)
        self.assertTrue(any("exact_rate" in failure for failure in verdict.failures))


class ActivationTests(unittest.TestCase):
    def test_activation_failure_blocks_even_a_good_number(self) -> None:
        """El fallo de exp13: un brazo rápido cuyo mecanismo nunca se activó.

        C rinde por encima del criterio, pero el coeficiente de saltos sigue en
        4: el relé no salió del camino. Atribuir la mejora a la topología sería
        exactamente el error que este arnés existe para impedir.
        """
        results = _full_set(c=_arm(Arm.C_TOPOLOGY, tps=6.0, hops=4.0))
        verdict = evaluate(results)
        self.assertFalse(verdict.passed)
        self.assertTrue(
            any("ACTIVACIÓN" in failure for failure in verdict.failures),
            verdict.failures,
        )

    def test_hop_coefficient_within_tolerance_passes(self) -> None:
        results = _full_set(c=_arm(Arm.C_TOPOLOGY, tps=4.0, hops=2.2))
        self.assertTrue(evaluate(results).passed, evaluate(results).failures)


class RefutationTests(unittest.TestCase):
    def test_slow_arm_c_refutes_the_network_dominance_thesis(self) -> None:
        results = _full_set(c=_arm(Arm.C_TOPOLOGY, tps=2.0))
        verdict = evaluate(results)
        self.assertTrue(verdict.refuted)
        self.assertFalse(verdict.passed)
        # El resumen tiene que decir REFUTADA, no limitarse a no publicar: un
        # experimento que refuta la tesis es un resultado, no un fallo.
        self.assertIn("REFUTADA", verdict.summary())
        self.assertTrue(
            any("2.00 tok/s" in reason for reason in verdict.refutation_reasons),
            verdict.refutation_reasons,
        )

    def test_heavy_compute_refutes_it_too(self) -> None:
        results = _full_set(c=_arm(Arm.C_TOPOLOGY, tps=4.0, forward=200.0))
        verdict = evaluate(results)
        self.assertTrue(verdict.refuted)

    def test_heavy_python_residual_refutes_it_too(self) -> None:
        # Éste es el que reabriría la decisión de Rust: si la orquestación pesa
        # más de 20 ms/token, T1 deja de estar 6x por encima del peor caso.
        results = _full_set(c=_arm(Arm.C_TOPOLOGY, tps=4.0, residual=35.0))
        self.assertTrue(evaluate(results).refuted)


class WindowAndFootprintTests(unittest.TestCase):
    def test_short_window_blocks_publication(self) -> None:
        bad = _window(duration=60.0, slowest=142.0)
        results = _full_set(c=_arm(Arm.C_TOPOLOGY, window=bad))
        self.assertFalse(evaluate(results).passed)

    def test_arm_d_without_footprint_cannot_evaluate_t5(self) -> None:
        results = _full_set(d=_arm(Arm.D_ENGINE, forward=18.0, footprint=None))
        verdict = evaluate(results)
        self.assertFalse(verdict.passed)
        self.assertTrue(any("huella base" in f for f in verdict.failures))

    def test_too_few_runs_is_not_a_result(self) -> None:
        results = _full_set(c=_arm(Arm.C_TOPOLOGY, runs=2))
        self.assertFalse(evaluate(results).passed)


class WarningTests(unittest.TestCase):
    def test_geography_arm_slower_than_base_warns_loudly(self) -> None:
        # No bloquea, pero obliga a mirar: o la sonda miente o el RTT no domina.
        results = _full_set(b=_arm(Arm.B_GEOGRAPHY, tps=1.2, hops=4.0))
        verdict = evaluate(results)
        self.assertTrue(any("PEOR" in warning for warning in verdict.warnings))


class PublishTests(unittest.TestCase):
    def test_publish_refuses_a_failing_set(self) -> None:
        results = _full_set(c=_arm(Arm.C_TOPOLOGY, tps=1.0))
        with self.assertRaises(ObservationWindowError):
            publish(results)

    def test_publish_emits_criteria_alongside_results(self) -> None:
        report = publish(_full_set())
        self.assertEqual(report["experiment"], "EXP17")
        # Los criterios viajan CON el resultado: quien lo lea puede comprobar
        # que no se movieron después de ver los datos.
        self.assertEqual(
            report["preregistered"]["arm_c_min_tokens_per_second"],
            ARM_C_MIN_TOKENS_PER_SECOND,
        )
        self.assertIn("observation_window", report["arms"]["C"])
        self.assertEqual(len(report["arms"]), 4)

    def test_missing_arm_is_not_publishable(self) -> None:
        results = _full_set()
        del results[Arm.D_ENGINE]
        with self.assertRaises(ObservationWindowError):
            publish(results)


if __name__ == "__main__":
    unittest.main()
