#!/usr/bin/env python
"""¿Se pueden gastar GPUs baratas para COMPRAR LATENCIA?

La idea (del fundador): con 20-100 GPUs, en vez de dedicarlas todas a servir más
usuarios, dedicar algunas a explorar VARIAS HIPÓTESIS EN PARALELO y quedarse con
la que acierta. Es decodificación especulativa en ÁRBOL, repartida entre cadenas.

Modelo (deliberadamente simple y declarado como aproximación):
  * Cadena lineal: se proponen g tokens en fila. Se aceptan hasta el primer
    fallo. Cota EXACTA y demostrada:  E = (1 - a^(g+1)) / (1 - a)
  * Árbol de anchura w: se proponen w continuaciones ALTERNATIVAS por posición.
    La probabilidad de que al menos una acierte en un nivel sube de a a
    p_w = 1 - (1-a)^w  (aproximación: supone ramas independientes, lo cual
    SOBREESTIMA — las ramas de un mismo borrador están correlacionadas).
    Con eso, E_arbol = (1 - p_w^(g+1)) / (1 - p_w).

Coste: el árbol multiplica el CÓMPUTO por (aproximadamente) el número de nodos
del árbol, pero NO multiplica las traversías: se verifica todo en UNA pasada.
Por eso convierte GPUs en latencia — que es exactamente lo que se busca.

Uso:
  python tree_speculation_model.py --alfa 0.5 --medido-tok-s 1.8
"""
from __future__ import annotations
import argparse


def e_chain(a: float, g: int) -> float:
    return (g + 1.0) if a >= 1.0 else (1.0 - a ** (g + 1)) / (1.0 - a)


def e_tree(a: float, g: int, w: int) -> float:
    p = 1.0 - (1.0 - a) ** w          # al menos una rama acierta en el nivel
    return e_chain(p, g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alfa", type=float, default=0.5, help="tasa de acierto de UNA rama")
    ap.add_argument("--medido-tok-s", type=float, default=1.8, help="single-stream medido hoy")
    ap.add_argument("--coste-verif", type=float, default=0.12,
                    help="fracción extra de tiempo de traversía por token verificado")
    ap.add_argument("--objetivo", type=float, default=4.0)
    args = ap.parse_args()

    base_ms = 1000.0 / args.medido_tok_s
    print("=" * 74)
    print(f"alfa por rama = {args.alfa}   ·   hoy = {args.medido_tok_s:.2f} tok/s "
          f"({base_ms:.0f} ms/traversía)   ·   objetivo = {args.objetivo:.0f} tok/s")
    print("=" * 74)
    print(f"{'anchura':>8} {'g=2':>16} {'g=4':>16} {'g=8':>16}")
    print(f"{'del árbol':>8} {'tok/s (nodos)':>16} {'tok/s (nodos)':>16} {'tok/s (nodos)':>16}")
    print("-" * 74)
    for w in (1, 2, 4, 8, 16):
        celdas = []
        for g in (2, 4, 8):
            E = e_tree(args.alfa, g, w)
            nodos = w * g                       # posiciones verificadas en la pasada
            # la traversía se alarga algo por verificar más posiciones
            t_ms = base_ms * (1.0 + args.coste_verif * nodos / 4.0)
            tps = E / (t_ms / 1000.0)
            marca = "*" if tps >= args.objetivo else " "
            celdas.append(f"{tps:6.2f}{marca} ({nodos:3d})")
        etiqueta = "lineal" if w == 1 else f"w={w}"
        print(f"{etiqueta:>8} {celdas[0]:>16} {celdas[1]:>16} {celdas[2]:>16}")
    print("-" * 74)
    print("* alcanza el objetivo · (nodos) = posiciones verificadas por traversía,")
    print("  proporcional al cómputo redundante = GPUs que se 'gastan' en latencia.")
    print()
    print("LECTURA: el árbol ayuda MUCHO cuando alfa es bajo (rescata borradores")
    print("malos) y poco cuando alfa ya es alto (la cadena sola basta).")
    print("AVISO: el modelo del árbol SOBREESTIMA (supone ramas independientes;")
    print("en la práctica están correlacionadas). Tómese como cota superior.")
    print("=" * 74)


if __name__ == "__main__":
    main()
