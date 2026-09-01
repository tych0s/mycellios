#!/usr/bin/env python
"""Re-análisis de Exp10: ¿el goodput cero era colapso o artefacto de ventana?

QUÉ HACE
--------
Recalcula, a partir de los datos ya publicados en
`docs/benchmarks/salad-exp10-salida-larga-2026-07-24/result.json`, el tiempo de
servicio que cada brazo NECESITABA y lo compara con la ventana en que se midió.
No re-corre nada: audita la corrida existente.

POR QUÉ
-------
Exp10 concluyó que «bajo sobrecarga el sistema no degrada: se va a CERO», y esa
frase se convirtió en uno de los dos requisitos duros del proyecto. Pero el ×7,7
de salidas largas ya se retractó por un artefacto de ventana, y Exp10 tiene la
misma forma: ventana corta frente a peticiones largas.

La pregunta que este guion contesta con aritmética, no con opinión: **¿podía una
sola petición de 256 tokens haber terminado dentro de la ventana de medida?**
Si la respuesta es no, el cero no es evidencia de nada sobre el sistema.

QUÉ NO HACE
-----------
No dice que la recomendación de Exp10 fuera errónea. El control de admisión
sigue siendo necesario —aceptar trabajo que no se puede servir y dejar que
expire es peor que rechazarlo rápido—. Lo que se corrige es la EVIDENCIA: bajo
10× de sobrecarga sostenida, una cola sin cota que acaba en timeouts es libro de
texto, no una patología propia que nos distinga de nadie.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

from distributed_runtime.observation_window import (  # noqa: E402
    ObservationWindow,
    diagnose,
)

DEFAULT_RESULT = (
    pathlib.Path(__file__).resolve().parents[1]
    / "docs"
    / "benchmarks"
    / "salad-exp10-salida-larga-2026-07-24"
    / "result.json"
)

#: `bench_agent.py:76` -> `dur = int(params.get("dur", "60"))`, warmup 10 s.
BENCH_DEFAULT_DURATION_S = 60.0
BENCH_WARMUP_S = 10.0


def analyze(result_path: pathlib.Path) -> list[dict[str, object]]:
    data = json.loads(result_path.read_text(encoding="utf-8"))
    arms = data["arms"]

    # Capacidad de referencia: el mejor agregado observado en la corrida. Es
    # generoso a propósito —usar una capacidad ALTA hace el argumento más
    # difícil de sostener, no más fácil.
    capacity_tok_s = max(
        float(entry["summary"][0]["agg_tok_s"])
        for arm in arms.values()
        for entry in arm["bench"]
    )

    rows: list[dict[str, object]] = []
    for name, arm in arms.items():
        entry = arm["bench"][0]
        summary = entry["summary"][0]
        tokens = int(arm["arm_cfg"]["bench_tokens"])
        offered_lambda = float(summary["lambda"])
        inflight = float(summary["inflight"])
        errors = int(summary["errors"])
        agg = float(summary["agg_tok_s"])

        offered_tok_s = offered_lambda * tokens
        overload = offered_tok_s / capacity_tok_s if capacity_tok_s else float("inf")
        # Tasa de servicio en peticiones/s a esa longitud de salida.
        service_req_s = capacity_tok_s / tokens if tokens else 0.0
        # Little: W = L / X. Con la cola creciendo, es el tiempo que tarda una
        # petición en salir por el otro lado.
        required_seconds = inflight / service_req_s if service_req_s else float("inf")

        measurement_window = BENCH_DEFAULT_DURATION_S - BENCH_WARMUP_S
        window = ObservationWindow(
            duration_seconds=measurement_window,
            slowest_request_seconds=required_seconds,
            completed_requests=0 if agg == 0 else 1,
            started_requests=max(1, int(offered_lambda * measurement_window)),
            notes=(
                f"ventana derivada del defecto de bench_agent.py "
                f"({BENCH_DEFAULT_DURATION_S:g}s - {BENCH_WARMUP_S:g}s de warmup); "
                f"result.json NO registra la ventana, que es el defecto de fondo",
            ),
        )

        rows.append(
            {
                "arm": name,
                "tokens": tokens,
                "offered_tok_s": round(offered_tok_s, 1),
                "capacity_tok_s": round(capacity_tok_s, 1),
                "overload_factor": round(overload, 2),
                "inflight": round(inflight, 1),
                "service_req_s": round(service_req_s, 4),
                "required_seconds_per_request": round(required_seconds, 1),
                "measurement_window_s": measurement_window,
                "window_covers_one_request": required_seconds <= measurement_window,
                "reported_agg_tok_s": agg,
                "errors": errors,
                "verdict": diagnose(window),
            }
        )
    return rows


def render(rows: list[dict[str, object]]) -> str:
    lines = [
        "== Re-análisis de Exp10 ==",
        "",
        f"{'brazo':<10} {'tok':>5} {'sobrecarga':>11} {'s/petición':>11} "
        f"{'ventana':>8} {'¿cabe?':>7} {'tok/s':>7} {'errores':>8}",
    ]
    for row in rows:
        lines.append(
            f"{row['arm']:<10} {row['tokens']:>5} "
            f"{float(row['overload_factor']):>10.2f}x "
            f"{float(row['required_seconds_per_request']):>11.1f} "
            f"{float(row['measurement_window_s']):>7.0f}s "
            f"{('SÍ' if row['window_covers_one_request'] else 'NO'):>7} "
            f"{float(row['reported_agg_tok_s']):>7.1f} "
            f"{row['errors']:>8}"
        )

    lines += ["", "Lectura:"]
    for row in rows:
        if not row["window_covers_one_request"]:
            lines.append(
                f"  - {row['arm']}: una sola petición necesitaba "
                f"~{float(row['required_seconds_per_request']):.0f}s y la ventana "
                f"eran {float(row['measurement_window_s']):.0f}s. Ni la PRIMERA "
                f"petición podía terminar dentro de la medida: el cero está "
                f"garantizado por construcción, no por el comportamiento del "
                f"sistema."
            )
    clean = [r for r in rows if float(r["overload_factor"]) <= 1.0]
    if clean:
        lines.append(
            "  - brazos NO saturados (los únicos interpretables): "
            + ", ".join(str(r["arm"]) for r in clean)
        )
    lines += [
        "",
        "Qué sobrevive y qué no:",
        "  SOBREVIVE  la recomendación: hace falta control de admisión. Aceptar",
        "             trabajo que no se puede servir y dejar que expire es peor",
        "             que rechazarlo rápido.",
        "  NO SOBREVIVE la evidencia tal como se contó. Bajo 10x de sobrecarga",
        "             sostenida, una cola sin cota que acaba en timeouts es libro",
        "             de texto — no una patología propia que nos distinga.",
        "",
        "Protocolo corregido para re-medir:",
        "  1. ventana >= 5x el tiempo de servicio esperado (usa la columna s/petición);",
        "  2. barrer lambda por DEBAJO y por encima de la capacidad, no solo a 10x;",
        "  3. registrar la ventana en result.json (hoy no está: ése es el defecto raíz);",
        "  4. reportar goodput junto a peticiones truncadas y rechazadas por separado.",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--result", type=pathlib.Path, default=DEFAULT_RESULT)
    parser.add_argument("--json", help="escribe el re-análisis a un fichero")
    args = parser.parse_args(argv)

    if not args.result.exists():
        raise SystemExit(f"no existe {args.result}")
    rows = analyze(args.result)
    print(render(rows))
    if args.json:
        pathlib.Path(args.json).write_text(
            json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nJSON -> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
