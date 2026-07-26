#!/usr/bin/env python
"""Fase 0: mide el ciclo de trabajo del hilo del scheduler. Resuelve T2.

QUÉ DECIDE
----------
T2 del cuadro de umbrales de la decisión de Rust:

    «ciclo de trabajo del hilo del scheduler > 70 % a B=32, con la GPU ociosa y
     el agregado saturado -> el GIL es un cuello real -> núcleo de scheduling
     en Rust.»

El valor que circula hoy (~1 %) es **estimado por aritmética**, no medido: sale
de suponer 3-5 ms de Python por ronda a B=32 y compararlo con el agregado
observado de 40-98 tok/s. Una estimación no puede disparar ni descartar un
umbral; por eso este guion existe y por eso es el más barato de los tres de
Fase 0 —treinta minutos.

CÓMO FUNCIONA
-------------
Muestrea la pila del proceso con `py-spy dump` a intervalos regulares y clasifica
cada muestra de cada hilo en:

* `python`  — el hilo está ejecutando bytecode: cuenta contra el GIL.
* `native`  — está dentro de C/torch/BLAS: normalmente con el GIL SUELTO.
* `waiting` — bloqueado en sockets, locks, colas o durmiendo: no compite.

El ciclo de trabajo es `python / (python + native + waiting)` por hilo. Lo que
dispara T2 no es que el hilo esté ocupado, sino que esté ocupado **en bytecode**,
que es lo único que serializa el GIL.

LÍMITE HONESTO DE ESTE MÉTODO
-----------------------------
`py-spy dump` es un muestreo, no una traza. Con 200 muestras el error de una
fracción cercana a 0,5 ronda ±3,5 puntos; cerca de 0,01 o de 0,99 es mucho menor.
No sirve para distinguir 68 % de 72 %: sirve para distinguir 1 % de 70 %, que es
exactamente la escala de la decisión. Si el resultado cae entre 60 % y 80 %, hay
que subir el número de muestras o pasar a una traza real.

USO
---
    python scripts/profile_scheduler.py --pid 12345 --samples 200
    python scripts/profile_scheduler.py --pid 12345 --json out.json

Requiere `py-spy` (`pip install py-spy`). En Windows puede hacer falta una
consola con privilegios de administrador para adjuntarse a otro proceso.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field

#: Umbral preregistrado. Por encima, el GIL es un cuello real.
T2_THRESHOLD = 0.70

#: Zona en la que el muestreo no tiene resolución para decidir.
T2_INCONCLUSIVE_BAND = (0.60, 0.80)

#: Marcas de pila que significan "bloqueado, no compite por el GIL".
_WAITING_MARKERS = (
    "select.select",
    "poll",
    "epoll",
    "socket.accept",
    "socket.recv",
    "sock_recv",
    "acquire",
    "wait",
    "sleep",
    "join",
    "get (queue",
    "queue.get",
    "_worker",
)

#: Marcas de pila que significan "dentro de código nativo, GIL normalmente suelto".
_NATIVE_MARKERS = (
    "torch/",
    "torch\\",
    "aten::",
    "numpy",
    "_C.",
    "libtorch",
    "mkl",
    "blas",
)


@dataclass
class ThreadProfile:
    thread: str
    python_samples: int = 0
    native_samples: int = 0
    waiting_samples: int = 0
    top_frames: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return self.python_samples + self.native_samples + self.waiting_samples

    @property
    def gil_duty_cycle(self) -> float:
        """Fracción del tiempo ejecutando bytecode. Es lo que serializa el GIL."""
        return self.python_samples / self.total if self.total else 0.0

    @property
    def busy_fraction(self) -> float:
        """Fracción no bloqueada, con o sin GIL."""
        if not self.total:
            return 0.0
        return (self.python_samples + self.native_samples) / self.total


@dataclass
class SchedulerReport:
    pid: int
    samples_requested: int
    samples_collected: int
    interval_seconds: float
    threads: list[dict[str, object]]
    hottest_thread: str | None
    hottest_gil_duty_cycle: float | None
    t2_triggered: bool | None
    t2_inconclusive: bool
    notes: list[str]


def _classify(frame_text: str) -> str:
    lowered = frame_text.lower()
    for marker in _WAITING_MARKERS:
        if marker in lowered:
            return "waiting"
    for marker in _NATIVE_MARKERS:
        if marker in lowered:
            return "native"
    return "python"


def _parse_dump(output: str) -> dict[str, str]:
    """Extrae, por hilo, el frame más profundo (el que está ejecutándose).

    `py-spy dump` imprime bloques `Thread <id> ("<nombre>")` seguidos de la pila
    con el frame activo primero. Nos quedamos con ese primer frame porque es el
    que describe qué está haciendo el hilo AHORA; los de más arriba son
    contexto.
    """
    threads: dict[str, str] = {}
    current: str | None = None
    for line in output.splitlines():
        stripped = line.strip()
        match = re.match(r'Thread\s+(\S+)\s*(?:\("([^"]*)"\))?', stripped)
        if match:
            identifier, name = match.group(1), match.group(2)
            current = f"{name or 'unnamed'}#{identifier}"
            continue
        if current and current not in threads and stripped:
            threads[current] = stripped
    return threads


def collect(pid: int, samples: int, interval: float) -> SchedulerReport:
    notes: list[str] = []
    if shutil.which("py-spy") is None:
        raise SystemExit(
            "py-spy no está instalado. Instálalo con:  pip install py-spy\n"
            "En Windows puede requerir una consola de administrador para "
            "adjuntarse a otro proceso."
        )

    profiles: dict[str, ThreadProfile] = defaultdict(lambda: ThreadProfile(thread=""))
    collected = 0
    failures = 0
    for _ in range(samples):
        try:
            result = subprocess.run(
                ["py-spy", "dump", "--pid", str(pid), "--nonblocking"],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except subprocess.TimeoutExpired:
            failures += 1
            continue
        if result.returncode != 0:
            failures += 1
            if failures <= 1:
                notes.append(f"py-spy devolvió {result.returncode}: {result.stderr.strip()[:200]}")
            if failures > samples // 4:
                raise SystemExit(
                    "py-spy falla de forma sostenida. ¿Existe el PID? "
                    "¿Hacen falta privilegios elevados?"
                )
            continue

        for thread, frame in _parse_dump(result.stdout).items():
            profile = profiles[thread]
            profile.thread = thread
            kind = _classify(frame)
            if kind == "python":
                profile.python_samples += 1
            elif kind == "native":
                profile.native_samples += 1
            else:
                profile.waiting_samples += 1
            profile.top_frames[frame] = profile.top_frames.get(frame, 0) + 1
        collected += 1
        time.sleep(interval)

    if collected == 0:
        raise SystemExit("no se recogió ninguna muestra; no hay nada que concluir")
    if failures:
        notes.append(f"{failures} muestras fallaron y se descartaron")

    ranked = sorted(profiles.values(), key=lambda p: p.gil_duty_cycle, reverse=True)
    hottest = ranked[0] if ranked else None
    duty = hottest.gil_duty_cycle if hottest else None

    inconclusive = False
    triggered: bool | None = None
    if duty is not None:
        low, high = T2_INCONCLUSIVE_BAND
        if low <= duty <= high:
            inconclusive = True
            notes.append(
                f"el ciclo de trabajo ({duty:.1%}) cae en la banda sin "
                f"resolución [{low:.0%}, {high:.0%}]: sube --samples o pasa a "
                f"una traza real antes de concluir nada"
            )
        else:
            triggered = duty > T2_THRESHOLD

    if collected < 100:
        notes.append(
            f"solo {collected} muestras: el error de muestreo es grande. "
            f"Para decidir un umbral usa >=200."
        )
    notes.append(
        "recuerda las condiciones del umbral: B=32, GPU ociosa y agregado "
        "saturado. Fuera de ese punto de operación el número no dice nada."
    )

    return SchedulerReport(
        pid=pid,
        samples_requested=samples,
        samples_collected=collected,
        interval_seconds=interval,
        threads=[
            {
                "thread": profile.thread,
                "gil_duty_cycle": round(profile.gil_duty_cycle, 4),
                "busy_fraction": round(profile.busy_fraction, 4),
                "python": profile.python_samples,
                "native": profile.native_samples,
                "waiting": profile.waiting_samples,
                "top_frames": sorted(
                    profile.top_frames.items(), key=lambda item: -item[1]
                )[:5],
            }
            for profile in ranked
        ],
        hottest_thread=hottest.thread if hottest else None,
        hottest_gil_duty_cycle=None if duty is None else round(duty, 4),
        t2_triggered=triggered,
        t2_inconclusive=inconclusive,
        notes=notes,
    )


def render(report: SchedulerReport) -> str:
    lines = [
        "== Fase 0: ciclo de trabajo del scheduler (umbral T2) ==",
        f"pid       : {report.pid}",
        f"muestras  : {report.samples_collected}/{report.samples_requested} "
        f"cada {report.interval_seconds}s",
        "",
        "hilo                                   GIL%   ocupado%  py/nat/wait",
    ]
    for entry in report.threads[:8]:
        lines.append(
            f"  {str(entry['thread'])[:36]:<36} "
            f"{float(entry['gil_duty_cycle']):>5.1%} "
            f"{float(entry['busy_fraction']):>9.1%}  "
            f"{entry['python']}/{entry['native']}/{entry['waiting']}"
        )
    lines += ["", f"hilo más caliente: {report.hottest_thread}"]
    if report.hottest_gil_duty_cycle is not None:
        lines.append(f"ciclo de trabajo GIL: {report.hottest_gil_duty_cycle:.1%}")
    if report.t2_inconclusive:
        lines.append("T2: SIN RESOLUCIÓN — no concluyas con estos datos")
    elif report.t2_triggered is None:
        lines.append("T2: no evaluable")
    elif report.t2_triggered:
        lines += [
            "T2: DISPARADO (>70%)",
            "  -> el GIL serializa el scheduler. Un núcleo de scheduling en "
            "Rust pasa a estar justificado; conviene confirmarlo con una traza.",
        ]
    else:
        lines += [
            "T2: NO disparado",
            "  -> el hilo del scheduler no está limitado por el GIL. "
            "Reescribirlo no compra rendimiento.",
        ]
    lines += ["", "avisos:"]
    for note in report.notes:
        lines.append(f"  - {note}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pid", type=int, required=True, help="PID del proceso raíz")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--interval", type=float, default=0.05)
    parser.add_argument("--json", help="escribe el informe completo a un fichero")
    args = parser.parse_args(argv)

    report = collect(args.pid, args.samples, args.interval)
    print(render(report))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(asdict(report), handle, indent=2, ensure_ascii=False)
        print(f"\nJSON -> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
