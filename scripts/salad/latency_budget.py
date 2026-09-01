#!/usr/bin/env python
"""¿Qué hace falta para pasar de 1-2 tok/s a 3-5 tok/s por usuario?

El cuello está medido:  tok/s = tokens_por_traversía / (saltos × RTT + cómputo)

Una "traversía" es un recorrido completo de la cadena (root -> etapas -> root).
Hoy sale UN token por traversía. Este script calcula, para cada palanca, qué
haría falta y si alcanza el objetivo.

Palancas modeladas:
 1. Menos saltos            (cadenas cortas / streaming de pesos en un nodo)
 2. Más tokens por traversía (decodificación especulativa: cota EXACTA
                              E = (1-a^(g+1))/(1-a), techo 1/(1-a))
 3. Menos RTT               (colocación por cercanía)
 4. Combinaciones

Uso:
  python latency_budget.py --rtt-ms 60 --hops 2 --compute-ms 60 --objetivo 4
"""
from __future__ import annotations
import argparse


def spec_tokens(alpha: float, gamma: int) -> float:
    """Tokens esperados por iteración con borrador en cadena (cota exacta)."""
    if alpha >= 1.0:
        return gamma + 1.0
    return (1.0 - alpha ** (gamma + 1)) / (1.0 - alpha)


def tok_s(tokens_per_traversal: float, hops: int, rtt_ms: float,
          compute_ms: float, verify_overhead_ms: float = 0.0) -> float:
    """Tokens por segundo de UN usuario (single-stream)."""
    t = hops * rtt_ms + compute_ms + verify_overhead_ms
    return tokens_per_traversal / (t / 1000.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--medido-tok-s", type=float, default=1.8,
                    help="tok/s single-stream MEDIDO (calibra el modelo; Exp6: 1,77-1,81)")
    ap.add_argument("--hops", type=int, default=4,
                    help="tramos de red por token: root->relé->etapa->relé->root = 4")
    ap.add_argument("--compute-ms", type=float, default=60.0,
                    help="cómputo total por token (el resto se atribuye a la red)")
    ap.add_argument("--objetivo", type=float, default=4.0, help="tok/s deseados")
    args = ap.parse_args()

    # CALIBRACIÓN: derivar el RTT por tramo de la medida real, no suponerlo.
    total_ms = 1000.0 / args.medido_tok_s
    red_ms = max(total_ms - args.compute_ms, 1.0)
    args.rtt_ms = red_ms / args.hops
    print(f"[calibrado con la medida real: {args.medido_tok_s:.2f} tok/s = "
          f"{total_ms:.0f} ms/token; cómputo {args.compute_ms:.0f} ms; "
          f"red {red_ms:.0f} ms en {args.hops} tramos = {args.rtt_ms:.0f} ms/tramo]")

    base = tok_s(1.0, args.hops, args.rtt_ms, args.compute_ms)
    presupuesto_ms = 1000.0 / args.objetivo
    print("=" * 70)
    print(f"HOY: {args.hops} saltos x {args.rtt_ms:.0f} ms + {args.compute_ms:.0f} ms cómputo "
          f"= {args.hops*args.rtt_ms+args.compute_ms:.0f} ms/token -> {base:.2f} tok/s")
    print(f"OBJETIVO: {args.objetivo:.0f} tok/s = {presupuesto_ms:.0f} ms por token")
    print(f"FALTA UN FACTOR DE {args.objetivo/base:.1f}x")
    print("=" * 70)

    print("\n[1] MENOS SALTOS (misma física, cadena más corta o pesos en un nodo)")
    for h in (args.hops, 1, 0):
        v = tok_s(1.0, h, args.rtt_ms, args.compute_ms)
        etiqueta = {0: "0 saltos (todo el modelo en un nodo, pesos por streaming)",
                    1: "1 salto", args.hops: f"{args.hops} saltos (hoy)"}.get(h, f"{h} saltos")
        marca = " <-- ALCANZA" if v >= args.objetivo else ""
        print(f"   {etiqueta:<52} {v:5.2f} tok/s{marca}")
    print("   NOTA: con 0 saltos el techo lo pone leer los pesos desde RAM/NVMe,")
    print("         no la red. Hay que medirlo aparte (ancho de banda x tamaño).")

    print("\n[2] MÁS TOKENS POR TRAVERSÍA (especulativa; cota exacta)")
    print(f"   {'alfa':>6} {'g=2':>8} {'g=4':>8} {'g=8':>8}   (tok/s resultante)")
    for a in (0.3, 0.5, 0.6, 0.7, 0.8, 0.9):
        fila = []
        for g in (2, 4, 8):
            n = spec_tokens(a, g)
            # el borrador cuesta: se asume local y barato, pero la verificación
            # procesa g+1 posiciones -> algo más de cómputo por traversía
            extra = args.compute_ms * 0.12 * g
            fila.append(tok_s(n, args.hops, args.rtt_ms, args.compute_ms, extra))
        marca = " <-- ALCANZA" if max(fila) >= args.objetivo else ""
        print(f"   {a:>6.1f} {fila[0]:8.2f} {fila[1]:8.2f} {fila[2]:8.2f}{marca}")
    print("   alfa = tasa de aceptación del borrador. Es LA variable a medir")
    print("   antes de invertir: con alfa<0,4 no compensa.")

    print("\n[3] MENOS RTT (colocar nodos cerca; no siempre posible)")
    for r in (args.rtt_ms, 30.0, 10.0, 5.0):
        v = tok_s(1.0, args.hops, r, args.compute_ms)
        marca = " <-- ALCANZA" if v >= args.objetivo else ""
        print(f"   RTT {r:>5.0f} ms -> {v:5.2f} tok/s{marca}")

    print("\n[4] COMBINADO (lo realista): especulativa + cadena corta")
    for h in (2, 1):
        for a in (0.5, 0.7):
            n = spec_tokens(a, 4)
            v = tok_s(n, h, args.rtt_ms, args.compute_ms, args.compute_ms * 0.48)
            marca = " <-- ALCANZA" if v >= args.objetivo else ""
            print(f"   {h} salto(s) + especulativa alfa={a} g=4 -> {v:5.2f} tok/s{marca}")
    print("=" * 70)


if __name__ == "__main__":
    main()
