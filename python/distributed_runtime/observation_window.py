"""La ventana de observación viaja con la métrica, o la métrica no se publica.

POR QUÉ EXISTE ESTE MÓDULO
--------------------------
Dos cifras estrella de este proyecto se cayeron en auditoría por la MISMA causa,
y ninguna de las dos era un error de código:

* «las respuestas largas hunden el sistema ×7,7» — RETRACTADO. La ventana de
  medida eran 40-60 s y una respuesta de 256 tokens a 1,8 tok/s tarda ~142 s.
  Se estaba midiendo cuántas respuestas caben en una ventana más corta que una
  sola respuesta.
* «bajo sobrecarga el goodput cae a CERO» (Exp10) — sospechoso del mismo
  defecto. Con λ=4, salidas de 256 tokens y ~100 peticiones en vuelo, el tiempo
  por respuesta es ~1 280 s. **Cero completadas en una ventana de 60 s es la
  expectativa aritmética de un sistema perfectamente sano**, no evidencia de
  colapso.

En los dos casos el número era correcto como aritmética y falso como
conclusión, porque la ventana no se declaró junto al dato. Un revisor que ve
`goodput = 0` no tiene forma de saber que la ventana era más corta que una
respuesta.

LA REGLA
--------
Toda métrica de throughput o goodput lleva su ventana de observación, y la
ventana debe dimensionarse contra **la petición más lenta**, no contra el reloj
del banco. El factor por defecto es 5×: con menos, el sesgo de truncamiento
domina, porque las peticiones que empezaron dentro de la ventana y no acabaron
cuentan como cero en el numerador y sí ocupan el denominador.

Este módulo no estima nada: se limita a rechazar la publicación de una métrica
cuya ventana no aguanta el escrutinio. Falla ruidoso a propósito.
"""
from __future__ import annotations

from dataclasses import dataclass, field

#: Cuántas veces la petición más lenta debe caber en la ventana.
#: 5× no es un número mágico: es el punto donde el sesgo de truncamiento baja del
#: 20 % para una distribución de servicio razonable. Súbelo si la cola es larga.
DEFAULT_WINDOW_FACTOR = 5.0


class ObservationWindowError(ValueError):
    """La ventana no permite publicar la métrica. Se falla, no se avisa."""


@dataclass(frozen=True)
class ObservationWindow:
    """Ventana bajo la que se midió una métrica agregada.

    `slowest_request_seconds` es el dato que casi siempre falta. Si no se conoce
    de antemano, se deriva de la corrida (`from_observations`); lo que NO vale
    es omitirlo, porque entonces la validación no puede hacer su trabajo.
    """

    duration_seconds: float
    slowest_request_seconds: float
    completed_requests: int
    started_requests: int
    factor: float = DEFAULT_WINDOW_FACTOR
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def required_seconds(self) -> float:
        return self.slowest_request_seconds * self.factor

    @property
    def is_adequate(self) -> bool:
        return self.duration_seconds >= self.required_seconds

    @property
    def truncated_requests(self) -> int:
        """Peticiones que empezaron y no acabaron dentro de la ventana."""
        return max(0, self.started_requests - self.completed_requests)

    @property
    def truncation_ratio(self) -> float:
        if self.started_requests <= 0:
            return 0.0
        return self.truncated_requests / self.started_requests

    def failure_reason(self) -> str | None:
        """Por qué esta ventana no permite publicar. `None` si sí permite."""
        if self.duration_seconds <= 0:
            return "la ventana de observación no tiene duración"
        if self.slowest_request_seconds <= 0:
            return (
                "no se registró la duración de la petición más lenta: sin ese "
                "dato la ventana no se puede validar"
            )
        if not self.is_adequate:
            return (
                f"ventana de {self.duration_seconds:.1f}s frente a una petición "
                f"más lenta de {self.slowest_request_seconds:.1f}s: hacen falta "
                f"{self.required_seconds:.1f}s ({self.factor:g}x). Una ventana "
                f"más corta que la petición mide truncamiento, no rendimiento — "
                f"es la causa exacta de las dos retractaciones anteriores"
            )
        if self.completed_requests == 0:
            return (
                "cero peticiones completadas: el agregado sería 0 por "
                "construcción, no por comportamiento del sistema"
            )
        return None

    def describe(self) -> dict[str, object]:
        """Bloque que acompaña SIEMPRE a la métrica publicada."""
        return {
            "window_seconds": round(self.duration_seconds, 3),
            "slowest_request_seconds": round(self.slowest_request_seconds, 3),
            "required_window_seconds": round(self.required_seconds, 3),
            "window_factor": self.factor,
            "started_requests": self.started_requests,
            "completed_requests": self.completed_requests,
            "truncated_requests": self.truncated_requests,
            "truncation_ratio": round(self.truncation_ratio, 4),
            "adequate": self.is_adequate,
            "notes": list(self.notes),
        }

    @classmethod
    def from_observations(
        cls,
        *,
        duration_seconds: float,
        request_durations_seconds: list[float] | tuple[float, ...],
        started_requests: int,
        factor: float = DEFAULT_WINDOW_FACTOR,
        notes: tuple[str, ...] = (),
    ) -> "ObservationWindow":
        """Deriva la ventana de las duraciones observadas.

        Aviso que importa: si TODAS las peticiones se truncaron, no hay ninguna
        duración observada y `slowest_request_seconds` sale 0 — que es justo el
        caso en que la validación debe fallar. No se rellena con la duración de
        la ventana, porque eso haría pasar la validación precisamente en el
        escenario que se quiere atrapar.
        """
        completed = len(request_durations_seconds)
        slowest = max(request_durations_seconds) if request_durations_seconds else 0.0
        return cls(
            duration_seconds=duration_seconds,
            slowest_request_seconds=slowest,
            completed_requests=completed,
            started_requests=started_requests,
            factor=factor,
            notes=notes,
        )


def publish_throughput(
    metric_name: str,
    value: float,
    window: ObservationWindow,
) -> dict[str, object]:
    """Envuelve una métrica agregada con su ventana, o lanza.

    Se usa en el punto de PUBLICACIÓN, no en el de cálculo: el objetivo es que
    ningún número de throughput pueda salir de una corrida hacia un informe sin
    llevar pegadas las condiciones en que se midió.
    """
    reason = window.failure_reason()
    if reason is not None:
        raise ObservationWindowError(
            f"no se puede publicar {metric_name!r}: {reason}"
        )
    return {
        "metric": metric_name,
        "value": value,
        "observation_window": window.describe(),
    }


def diagnose(window: ObservationWindow) -> str:
    """Texto legible para el operador. Explica el defecto, no solo lo señala."""
    reason = window.failure_reason()
    if reason is None:
        return (
            f"ventana válida: {window.duration_seconds:.1f}s >= "
            f"{window.required_seconds:.1f}s requeridos; "
            f"{window.completed_requests}/{window.started_requests} completadas "
            f"({window.truncation_ratio:.1%} truncadas)"
        )
    return f"VENTANA NO VÁLIDA: {reason}"
