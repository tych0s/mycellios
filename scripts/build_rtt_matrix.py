#!/usr/bin/env python
"""Ensambla la matriz RTT all-pairs y la emite como `links` para el planificador.

QUÉ CIERRA
----------
El eslabón que faltaba. El planificador ya sabe consumir aristas dirigidas y
penaliza con un centinela caro lo que no está medido (`src/core/rtt.ts`,
`UNMEASURED_RTT_MS`), pero nadie le daba medidas nodo↔nodo: `config.links` iba
vacío y todas las aristas caían al centinela.

Este guion recoge `/stats` de cada sonda, cruza los resultados y emite el bloque
`links` que consume `auto-distribute`. A partir de ahí, el orden de la cadena lo
decide `latencyGreedyOrder` con datos reales en vez de con el orden de
declaración.

REGLAS DE HONESTIDAD
--------------------
1. **Una arista sin muestras NO se emite.** Emitirla con un valor inventado
   sería peor que omitirla: omitida cae al centinela y el planificador la evita;
   inventada, la elige.
2. **Se usa la mediana**, no la media: la cola derecha de un enlace WAN es
   pesada y la media persigue la cola.
3. **Se emite `p90` como jitter**, para que el modelo de coste lo tenga
   separado de la latencia.
4. **RTT medido -> one-way = RTT/2.** La sonda mide ida y vuelta; el
   planificador espera one-way. Confundirlos duplica el coste de cada salto.
5. **Asimetría preservada.** A→B y B→A se emiten por separado si ambos se
   midieron. Promediarlas ocultaría rutas asimétricas, que existen.

USO
---
    python scripts/build_rtt_matrix.py \\
        --probe n0=http://host-a:8000 --probe n1=http://host-b:8000 \\
        --out links.json
    python scripts/build_rtt_matrix.py --from-file stats-dump.json --out links.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import urllib.error
import urllib.request

DEFAULT_BANDWIDTH_MBPS = 100.0


def fetch_stats(base_url: str, timeout: float = 15.0) -> dict:
    url = base_url.rstrip("/") + "/stats"
    request = urllib.request.Request(url, headers={"User-Agent": "gdlp-matrix"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def build_links(
    stats_by_node: dict[str, dict],
    *,
    bandwidth_mbps: float = DEFAULT_BANDWIDTH_MBPS,
    require_complete: bool = True,
) -> tuple[list[dict], list[str]]:
    """Devuelve (links, avisos). Solo aristas realmente medidas."""
    links: list[dict] = []
    warnings: list[str] = []

    for source, stats in stats_by_node.items():
        peers = stats.get("peers") or {}
        if not peers:
            warnings.append(
                f"{source}: sin datos de pares. Todas sus aristas salientes "
                f"quedarán sin medir y el planificador lo evitará"
            )
            continue
        for target, entry in peers.items():
            median = entry.get("p50_ms")
            samples = entry.get("n") or 0
            if median is None or samples <= 0:
                warnings.append(
                    f"{source}->{target}: sin muestras"
                    + (f" ({entry['error']})" if entry.get("error") else "")
                    + ". Se OMITE: una arista inventada se elegiría; omitida, se evita"
                )
                continue
            if require_complete and not entry.get("complete", False):
                warnings.append(
                    f"{source}->{target}: solo {samples} muestras, medida "
                    f"incompleta. Se omite (usa --allow-partial para incluirla)"
                )
                continue
            # La sonda mide ida y vuelta; el planificador espera one-way.
            one_way = median / 2.0
            p90 = entry.get("p90_ms")
            jitter = max(0.0, (p90 - median) / 2.0) if p90 is not None else 0.0
            links.append(
                {
                    "from": source,
                    "to": target,
                    "oneWayLatencyMs": round(one_way, 3),
                    "jitterP95Ms": round(jitter, 3),
                    "bandwidthMbps": bandwidth_mbps,
                    "lossRate": 0,
                    "availability": 0.999,
                }
            )

    nodes = sorted(stats_by_node)
    for source in nodes:
        for target in nodes:
            if source == target:
                continue
            if not any(
                link["from"] == source and link["to"] == target for link in links
            ):
                warnings.append(
                    f"HUECO en la matriz: {source}->{target} sin medir; el "
                    f"planificador cobrará el centinela por esa arista"
                )

    asymmetric = []
    for link in links:
        reverse = next(
            (
                other
                for other in links
                if other["from"] == link["to"] and other["to"] == link["from"]
            ),
            None,
        )
        if reverse is None:
            continue
        forward, backward = link["oneWayLatencyMs"], reverse["oneWayLatencyMs"]
        if max(forward, backward) > 1.5 * max(min(forward, backward), 0.001):
            pair = tuple(sorted((link["from"], link["to"])))
            if pair not in asymmetric:
                asymmetric.append(pair)
                warnings.append(
                    f"ruta ASIMÉTRICA {pair[0]}<->{pair[1]}: "
                    f"{forward:.1f} vs {backward:.1f} ms one-way. Se preservan "
                    f"ambas direcciones; promediarlas ocultaría el efecto"
                )

    return links, warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--probe",
        action="append",
        default=[],
        metavar="ID=URL",
        help="sonda a consultar; repetible",
    )
    parser.add_argument("--from-file", type=pathlib.Path, help="JSON {id: stats}")
    parser.add_argument("--out", type=pathlib.Path, help="fichero de salida")
    parser.add_argument("--bandwidth-mbps", type=float, default=DEFAULT_BANDWIDTH_MBPS)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="incluye aristas con menos muestras de las requeridas",
    )
    args = parser.parse_args(argv)

    stats_by_node: dict[str, dict] = {}
    if args.from_file:
        stats_by_node = json.loads(args.from_file.read_text(encoding="utf-8"))
    for spec in args.probe:
        if "=" not in spec:
            raise SystemExit(f"--probe espera ID=URL, recibido {spec!r}")
        node_id, url = spec.split("=", 1)
        try:
            stats_by_node[node_id] = fetch_stats(url)
        except (urllib.error.URLError, OSError, ValueError) as error:
            print(f"[matrix] {node_id}: no responde ({error})", file=sys.stderr)
    if not stats_by_node:
        raise SystemExit("no hay estadísticas que ensamblar")

    links, warnings = build_links(
        stats_by_node,
        bandwidth_mbps=args.bandwidth_mbps,
        require_complete=not args.allow_partial,
    )

    print(f"== matriz RTT: {len(links)} aristas medidas ==")
    for link in sorted(links, key=lambda item: item["oneWayLatencyMs"]):
        print(
            f"  {link['from']:>10} -> {link['to']:<10} "
            f"{link['oneWayLatencyMs']:>7.1f} ms one-way "
            f"(jitter {link['jitterP95Ms']:.1f})"
        )
    if warnings:
        print("\navisos:")
        for warning in warnings:
            print(f"  - {warning}")

    payload = {"links": links, "warnings": warnings}
    if args.out:
        args.out.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
