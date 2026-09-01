#!/usr/bin/env python
"""Ajusta el MODELO DE COSTE POR PASO del pipeline a partir de medidas reales.

Modelo:  t_paso(B) = t_fijo + t_seq * B        (B = B_eff, secuencias fusionadas)
         tok/s     = B / t_paso(B)

Interpretación:
  * t_fijo  = coste que NO depende del lote (RTT WAN, serialización, Python,
              lanzamiento de kernels, planificación).
  * t_seq   = coste MARGINAL de añadir una secuencia al lote. Si el batching
              amortizara los pesos como dice el kernel (S0: e(8)=7,8), t_seq
              debería ser ~0. Si sale grande, la fusión NO está amortizando
              (padding del ragged, atención por-secuencia, etc.).

Techo asintótico (B->inf) = 1/t_seq tok/s. Si t_seq>0, subir B_eff tiene
rendimientos decrecientes y el techo NO es el roofline de pesos.

Uso: pasar medidas 'tok_s:B_eff' (>=2 puntos, del MISMO nodo/experimento).
  python fit_step_model.py --points 70.5:1.132 98.2:3.467 --label "Exp6 lam6"
"""
from __future__ import annotations
import argparse


def fit(points):
    """Mínimos cuadrados de t_paso = a + b*B, con t_paso = B/tok_s."""
    xs = [b for _, b in points]
    ys = [b / tok * 1000.0 for tok, b in points]  # ms por paso
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        raise SystemExit("los puntos tienen el mismo B_eff: no se puede ajustar")
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    a = my - b * mx
    return a, b, xs, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", nargs="+", required=True, help="tok_s:B_eff (>=2)")
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    pts = []
    for p in args.points:
        tok, beff = p.split(":")
        pts.append((float(tok), float(beff)))
    if len(pts) < 2:
        raise SystemExit("hacen falta >=2 puntos")

    t_fijo, t_seq, xs, ys = fit(pts)
    print("=" * 62)
    if args.label:
        print(f"AJUSTE: {args.label}")
    print(f"  t_paso(B) = {t_fijo:.2f} ms + {t_seq:.2f} ms * B")
    print("-" * 62)
    print("  datos (B_eff -> ms/paso medido vs modelo):")
    for (tok, b), y in zip(pts, ys):
        print(f"    B={b:<6.3f} medido {y:6.2f} ms   modelo {t_fijo + t_seq*b:6.2f} ms   ({tok:.1f} tok/s)")
    print("-" * 62)
    if t_seq > 0:
        techo = 1000.0 / t_seq
        print(f"  TECHO ASINTÓTICO (B->inf): {techo:.0f} tok/s")
        print("  proyección al subir la fusión:")
        for b in (2, 4, 8, 16, 32):
            t = t_fijo + t_seq * b
            print(f"    B={b:<3} -> {b/t*1000:6.1f} tok/s")
        print()
        print(f"  >> El coste MARGINAL por secuencia es {t_seq:.2f} ms.")
        if t_seq > 1.0:
            print("     Es ALTO: la fusión NO está amortizando los pesos como debería")
            print("     (el kernel medido en S0 daba e(8)=7,8, es decir ~gratis).")
            print("     Sospechas: padding del ragged (se computa relleno), atención")
            print("     por-secuencia, o copias/concatenaciones de KV por paso.")
            print("     -> Optimizar ESTO sube el techo; subir B_eff solo se acerca a él.")
        else:
            print("     Es BAJO: la fusión sí amortiza; subir B_eff paga casi lineal.")
    else:
        print("  t_seq<=0: la fusión es gratis o los datos son ruidosos;")
        print("  subir B_eff debería pagar ~lineal. Repetir con más puntos.")
    print("=" * 62)


if __name__ == "__main__":
    main()
