"""Temporización honesta del forward: despacho frente a kernel.

Contexto (auditoría 2026-07-25). El término C —el coste de cómputo por token—
nunca se ha medido. Existe una cota por resta contra el techo roofline que dice
que 28,9 ms de los 33,35 del forward (el 87 %) están sin explicar, y esa cota
es el número que decide si tiene sentido tocar el motor o no.

El problema al ir a medirlo: `stage.py` cronometra alrededor de
`runner.forward_hidden` con `time.perf_counter()` y **nadie sincroniza CUDA**.
En GPU los kernels se encolan y se devuelven de inmediato, así que ese reloj mide
el tiempo de DESPACHO, no el de ejecución. Puede salir un orden de magnitud por
debajo del real.

La respuesta de este módulo no es "sincronizar siempre" —una barrera en el camino
caliente cuesta rendimiento y contaminaría justo la medida de throughput que
queremos— sino **separar los dos regímenes y no dejar que se confundan por el
nombre**:

* modo `dispatch` (por defecto): barato, sin barrera. Lo que mide es el coste de
  encolar. Se publica como `compute_dispatch_ms`, jamás como `compute_ms`.
* modo `sync`: sincroniza el dispositivo antes de parar el reloj. Caro, para
  campañas de medición. Se publica como `compute_ms` porque entonces sí lo es.

La regla que hace falta institucionalizar tras dos retractaciones por
instrumentación: **un número lleva en el nombre el régimen en que se midió.**
Un `compute_ms` sin sincronizar es exactamente la clase de dato que sobrevive
seis meses en un informe y luego se cae en una auditoría.

Se activa con `GDLP_COMPUTE_TIMING=sync` o pasando `mode` explícitamente.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

DISPATCH = "dispatch"
SYNC = "sync"
_VALID_MODES = (DISPATCH, SYNC)

ENV_VAR = "GDLP_COMPUTE_TIMING"


def resolve_timing_mode(explicit: str | None = None) -> str:
    """Modo efectivo. Por defecto `dispatch`, que es el barato y el honesto."""
    value = explicit if explicit is not None else os.environ.get(ENV_VAR, DISPATCH)
    normalized = str(value).strip().lower()
    if normalized not in _VALID_MODES:
        raise ValueError(
            f"{ENV_VAR} must be one of {_VALID_MODES}; received {value!r}"
        )
    return normalized


@dataclass(frozen=True)
class ComputeSample:
    """Una medida de forward, con el régimen adherido al dato."""

    elapsed_ms: float
    mode: str

    @property
    def is_kernel_time(self) -> bool:
        """`True` solo si el reloj esperó de verdad a que el dispositivo acabara."""
        return self.mode == SYNC

    def metric_key(self) -> str:
        """Nombre del campo bajo el que puede publicarse SIN mentir."""
        return "compute_ms" if self.is_kernel_time else "compute_dispatch_ms"


def _synchronize(device: object | None) -> None:
    """Barrera del dispositivo, si lo hay. Sin torch o sin GPU es un no-op."""
    try:
        import torch
    except ImportError:  # pragma: no cover - entornos sin torch
        return
    if torch.cuda.is_available():
        # `device=None` sincroniza el dispositivo actual, que es lo que queremos
        # cuando el llamante no sabe (o no le importa) en cuál cayó la etapa.
        torch.cuda.synchronize(device)  # type: ignore[arg-type]
        return
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        synchronize = getattr(getattr(torch, "mps", None), "synchronize", None)
        if callable(synchronize):
            synchronize()


class ComputeTimer:
    """Cronómetro de un forward que sabe en qué régimen está midiendo.

    Uso::

        timer = ComputeTimer(mode)
        with timer.measure(device) as sample:
            results = runner.forward_hidden(...)
        metrics[sample.metric_key()] += sample.elapsed_ms

    En modo `sync` la barrera se pone **después** del trabajo y **antes** de
    parar el reloj. No se pone una barrera de entrada a propósito: si hay trabajo
    previo en vuelo, ese coste pertenece al forward anterior y cargárselo a éste
    solo movería el error de sitio.
    """

    def __init__(self, mode: str | None = None) -> None:
        self._mode = resolve_timing_mode(mode)

    @property
    def mode(self) -> str:
        return self._mode

    def measure(self, device: object | None = None) -> "_Measurement":
        return _Measurement(self._mode, device)


class _Measurement:
    def __init__(self, mode: str, device: object | None) -> None:
        self._mode = mode
        self._device = device
        self._started = 0.0
        self.sample = ComputeSample(elapsed_ms=0.0, mode=mode)

    def __enter__(self) -> "_Measurement":
        self._started = time.perf_counter()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        if exc_type is not None:
            # Un forward que ha lanzado no produce una muestra válida: publicar
            # su duración parcial contaminaría la distribución con abortos.
            return False
        if self._mode == SYNC:
            _synchronize(self._device)
        elapsed_ms = (time.perf_counter() - self._started) * 1_000
        self.sample = ComputeSample(elapsed_ms=elapsed_ms, mode=self._mode)
        return False

    def metric_key(self) -> str:
        return self.sample.metric_key()

    @property
    def elapsed_ms(self) -> float:
        return self.sample.elapsed_ms
