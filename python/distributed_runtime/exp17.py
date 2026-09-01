"""EXP17: geografía y topología, con criterios preregistrados.

QUÉ PREGUNTA RESPONDE
---------------------
Cuánto vale (a) elegir el nodo por RTT medido y (b) sacar el relé del plano de
datos, separando ambos efectos. Es el experimento que convierte la ventaja
documental en ventaja de producto — y el que puede refutar la recomendación de
no migrar a Rust.

POR QUÉ ESTÁ PREREGISTRADO
--------------------------
Nueve cifras de la campaña anterior no sobrevivieron a auditoría. La causa no
fue mala fe ni mal código: fue decidir después de ver los datos qué contaba como
éxito. Los criterios de abajo se fijan ANTES de correr y el arnés se niega a
declarar un resultado que no los cumpla, incluido el caso incómodo de que el
resultado sea bueno por la razón equivocada.

LOS CUATRO BRAZOS
-----------------
    A  4 tramos, nodo actual                     -> base
    B  4 tramos, mejor nodo por RTT MEDIDO       -> aísla geografía
    C  2 tramos (relé fuera del plano de datos)  -> aísla topología
    D  C + etapa nakshatra/GGUF                  -> aísla motor nativo

A/B se corre en el MISMO nodo e intercalado, no en bloques: el ruido del banco
es ±20-25 % y dos bloques consecutivos confunden deriva temporal con efecto.

CRITERIOS (fijados antes de correr)
-----------------------------------
1. `exact_rate == 1.0` en los cuatro brazos. Sin exactitud no hay resultado:
   un brazo más rápido que cambia los tokens no es más rápido, es otro sistema.
2. Brazo C >= 3,2 tok/s p50.
3. Coeficiente de saltos por token de C ~ 2,0 (no 4,0). Es la PRUEBA DE
   ACTIVACIÓN: sin ella, un C rápido podría serlo por cualquier otra causa y
   estaríamos atribuyendo la mejora a una topología que nunca cambió.
4. Desglose publicado en tres líneas MEDIDAS y separadas: red / forward /
   residual. Nada de restas.
5. Brazo D: suelo no-red por debajo de 45 ms, con huella base reportada.

CONDICIÓN DE REFUTACIÓN
-----------------------
Si C < 2,5 tok/s, o el cómputo medido supera 150 ms/token, o el residual de
Python supera 20 ms/token, la tesis de dominancia de red queda herida y hay que
reabrir la pregunta del MOTOR (no la del lenguaje de orquestación). Está aquí
por escrito para que no se pueda reinterpretar luego.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .observation_window import ObservationWindow, ObservationWindowError


class Arm(str, Enum):
    A_BASE = "A"
    B_GEOGRAPHY = "B"
    C_TOPOLOGY = "C"
    D_ENGINE = "D"


#: Criterios preregistrados. Cambiarlos después de ver datos invalida el experimento.
MIN_EXACT_RATE = 1.0
ARM_C_MIN_TOKENS_PER_SECOND = 3.2
ARM_C_EXPECTED_HOP_COEFFICIENT = 2.0
HOP_COEFFICIENT_TOLERANCE = 0.35
ARM_D_MAX_NON_NETWORK_FLOOR_MS = 45.0
MIN_INTERLEAVED_RUNS = 5

#: Condición de refutación.
REFUTATION_MIN_TOKENS_PER_SECOND = 2.5
REFUTATION_MAX_COMPUTE_MS_PER_TOKEN = 150.0
REFUTATION_MAX_PYTHON_RESIDUAL_MS_PER_TOKEN = 20.0


@dataclass(frozen=True)
class ArmResult:
    """Resultado de un brazo. Los tres tiempos son MEDIDOS, no restados."""

    arm: Arm
    tokens_per_second_p50: float
    exact_rate: float
    hop_coefficient: float
    network_ms_per_token: float
    forward_ms_per_token: float
    python_residual_ms_per_token: float
    runs: int
    window: ObservationWindow
    base_footprint_bytes: int | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def non_network_floor_ms(self) -> float:
        return self.forward_ms_per_token + self.python_residual_ms_per_token

    def budget_residual_ms(self, measured_total_ms: float) -> float:
        """Cuánto del total NO explican las tres líneas medidas.

        Un residual grande significa que el desglose no cierra y que alguna de
        las tres partidas está mal instrumentada. Es la comprobación que evita
        publicar un desglose que suma cualquier cosa.
        """
        return measured_total_ms - (
            self.network_ms_per_token
            + self.forward_ms_per_token
            + self.python_residual_ms_per_token
        )


@dataclass(frozen=True)
class Verdict:
    passed: bool
    failures: tuple[str, ...]
    warnings: tuple[str, ...]
    refuted: bool
    refutation_reasons: tuple[str, ...]

    def summary(self) -> str:
        if self.refuted:
            return "REFUTADA la tesis de dominancia de red: " + "; ".join(
                self.refutation_reasons
            )
        if not self.passed:
            return "NO CONCLUYENTE: " + "; ".join(self.failures)
        return "criterios preregistrados cumplidos"


def _check_common(result: ArmResult) -> list[str]:
    failures: list[str] = []
    if result.exact_rate < MIN_EXACT_RATE:
        failures.append(
            f"brazo {result.arm.value}: exact_rate {result.exact_rate:.4f} < "
            f"{MIN_EXACT_RATE}. Un brazo que cambia los tokens no es comparable"
        )
    if result.runs < MIN_INTERLEAVED_RUNS:
        failures.append(
            f"brazo {result.arm.value}: {result.runs} corridas < "
            f"{MIN_INTERLEAVED_RUNS}; con ruido de ±20-25 % no alcanza"
        )
    reason = result.window.failure_reason()
    if reason is not None:
        failures.append(f"brazo {result.arm.value}: ventana inválida — {reason}")
    return failures


def evaluate(results: dict[Arm, ArmResult]) -> Verdict:
    """Aplica los criterios preregistrados. No negocia."""
    failures: list[str] = []
    warnings: list[str] = []
    refutations: list[str] = []

    missing = [arm.value for arm in Arm if arm not in results]
    if missing:
        failures.append(f"faltan brazos: {', '.join(missing)}")

    for result in results.values():
        failures.extend(_check_common(result))

    arm_c = results.get(Arm.C_TOPOLOGY)
    if arm_c is not None:
        # Prueba de activación ANTES que el criterio de rendimiento. Si la
        # topología no cambió, un C rápido no prueba nada sobre topología, y
        # aceptarlo sería repetir el fallo de exp13.
        deviation = abs(arm_c.hop_coefficient - ARM_C_EXPECTED_HOP_COEFFICIENT)
        if deviation > HOP_COEFFICIENT_TOLERANCE:
            failures.append(
                f"PRUEBA DE ACTIVACIÓN fallida: el coeficiente de saltos de C es "
                f"{arm_c.hop_coefficient:.2f}, se esperaba "
                f"~{ARM_C_EXPECTED_HOP_COEFFICIENT}. El relé NO salió del camino "
                f"crítico, así que cualquier mejora medida no es atribuible a la "
                f"topología"
            )
        if arm_c.tokens_per_second_p50 < ARM_C_MIN_TOKENS_PER_SECOND:
            failures.append(
                f"brazo C: {arm_c.tokens_per_second_p50:.2f} tok/s < "
                f"{ARM_C_MIN_TOKENS_PER_SECOND} preregistrado"
            )
        if arm_c.tokens_per_second_p50 < REFUTATION_MIN_TOKENS_PER_SECOND:
            refutations.append(
                f"C dio {arm_c.tokens_per_second_p50:.2f} tok/s, por debajo del "
                f"umbral de refutación {REFUTATION_MIN_TOKENS_PER_SECOND}"
            )
        if arm_c.forward_ms_per_token > REFUTATION_MAX_COMPUTE_MS_PER_TOKEN:
            refutations.append(
                f"cómputo medido {arm_c.forward_ms_per_token:.1f} ms/token > "
                f"{REFUTATION_MAX_COMPUTE_MS_PER_TOKEN}: el cuello no es la red"
            )
        if (
            arm_c.python_residual_ms_per_token
            > REFUTATION_MAX_PYTHON_RESIDUAL_MS_PER_TOKEN
        ):
            refutations.append(
                f"residual de Python {arm_c.python_residual_ms_per_token:.1f} "
                f"ms/token > {REFUTATION_MAX_PYTHON_RESIDUAL_MS_PER_TOKEN}: "
                f"la orquestación pesa más de lo asumido"
            )

    arm_d = results.get(Arm.D_ENGINE)
    if arm_d is not None:
        if arm_d.non_network_floor_ms >= ARM_D_MAX_NON_NETWORK_FLOOR_MS:
            failures.append(
                f"brazo D: suelo no-red {arm_d.non_network_floor_ms:.1f} ms >= "
                f"{ARM_D_MAX_NON_NETWORK_FLOOR_MS} preregistrado"
            )
        if arm_d.base_footprint_bytes is None:
            failures.append(
                "brazo D: falta la huella base (RSS+VRAM antes de pesos); sin "
                "ella no se puede evaluar el umbral T5"
            )

    arm_a = results.get(Arm.A_BASE)
    arm_b = results.get(Arm.B_GEOGRAPHY)
    if arm_a is not None and arm_b is not None:
        gain = arm_b.tokens_per_second_p50 / max(arm_a.tokens_per_second_p50, 1e-9)
        if gain < 1.0:
            warnings.append(
                f"el mejor nodo por RTT rindió PEOR que la base (x{gain:.2f}): "
                f"o la sonda mide mal, o el RTT no domina en este punto de "
                f"operación. Investigar antes de publicar nada"
            )

    return Verdict(
        passed=not failures and not refutations,
        failures=tuple(failures),
        warnings=tuple(warnings),
        refuted=bool(refutations),
        refutation_reasons=tuple(refutations),
    )


def publish(results: dict[Arm, ArmResult]) -> dict[str, object]:
    """Informe publicable, o excepción. Nunca un resultado a medias."""
    verdict = evaluate(results)
    if not verdict.passed:
        raise ObservationWindowError(
            f"EXP17 no publica: {verdict.summary()}"
        )
    return {
        "experiment": "EXP17",
        "preregistered": {
            "min_exact_rate": MIN_EXACT_RATE,
            "arm_c_min_tokens_per_second": ARM_C_MIN_TOKENS_PER_SECOND,
            "arm_c_expected_hop_coefficient": ARM_C_EXPECTED_HOP_COEFFICIENT,
            "arm_d_max_non_network_floor_ms": ARM_D_MAX_NON_NETWORK_FLOOR_MS,
            "min_interleaved_runs": MIN_INTERLEAVED_RUNS,
        },
        "arms": {
            arm.value: {
                "tokens_per_second_p50": result.tokens_per_second_p50,
                "exact_rate": result.exact_rate,
                "hop_coefficient": result.hop_coefficient,
                "breakdown_ms_per_token": {
                    "network": result.network_ms_per_token,
                    "forward": result.forward_ms_per_token,
                    "python_residual": result.python_residual_ms_per_token,
                },
                "runs": result.runs,
                "observation_window": result.window.describe(),
                "base_footprint_bytes": result.base_footprint_bytes,
                "notes": list(result.notes),
            }
            for arm, result in sorted(results.items(), key=lambda item: item[0].value)
        },
        "warnings": list(verdict.warnings),
    }
