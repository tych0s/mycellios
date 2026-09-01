#!/usr/bin/env python
"""¿Qué se puede hacer con la RED para ir más rápido? Opciones, cuantificadas.

Punto de partida MEDIDO: 1,80 tok/s de un usuario = 556 ms/token, de los cuales
~60 ms son cómputo y ~496 ms son red repartidos en 4 tramos de internet
(raíz -> relé -> etapa -> relé -> raíz) = ~124 ms por tramo.

Este script compara las opciones REALES de topología y transporte. No inventa
constantes: parte de la medida y sólo cambia lo que cada opción cambia.
"""
from __future__ import annotations

import sys

# la consola de Windows por defecto es cp1252 y no traga ni acentos ni ✅
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

COMPUTE_MS = 60.0          # cómputo por token (estimado del presupuesto medido)
RTT_MEDIDO = 124.0         # ms por tramo, derivado de 1,80 tok/s con 4 tramos
OBJETIVO = 4.0


def toks(hops: int, rtt_ms: float, compute_ms: float = COMPUTE_MS,
         tokens_per_traversal: float = 1.0) -> float:
    return tokens_per_traversal / ((hops * rtt_ms + compute_ms) / 1000.0)


def fila(nombre: str, hops: int, rtt: float, tpt: float = 1.0, nota: str = "") -> None:
    v = toks(hops, rtt, tokens_per_traversal=tpt)
    marca = " ✅" if v >= OBJETIVO else "  "
    print(f"  {nombre:<52} {v:6.2f} tok/s{marca}  {nota}")


def main() -> None:
    print("=" * 78)
    print(f"HOY (medido): 4 tramos x {RTT_MEDIDO:.0f} ms + {COMPUTE_MS:.0f} ms cómputo "
          f"= {toks(4, RTT_MEDIDO):.2f} tok/s")
    print("=" * 78)

    print("\nA) QUITAR EL RODEO DEL RELÉ  (nodos que se hablan directos)")
    fila("hoy: raíz->relé->etapa->relé->raíz (4 tramos)", 4, RTT_MEDIDO)
    fila("directo: raíz->etapa->raíz (2 tramos)", 2, RTT_MEDIDO, nota="x1,8 sin tocar nada más")
    print("     Exp16: un tramo nodo->nodo por la URL pública cuesta 0,96-1,24x lo que")
    print("     cuesta uno nodo->relé => no hace falta perforación de NAT.")

    print("\nB) ELEGIR NODOS CERCANOS  (RTT por tramo MEDIDO en Exp15, no proyectado)")
    for rtt, etiqueta in ((54, "Elmsford NY — el mejor medido"),
                          (65, "mediana EE.UU. = mismo país que el relé"),
                          (132, "mediana EE.UU. en la SEGUNDA tirada (Exp16)"),
                          (148, "Coventry, Reino Unido"),
                          (190, "Cachoeirinha, Brasil"),
                          (437, "Kaohsiung, Taiwán — la mala tirada")):
        fila(f"RTT {rtt:>3} ms — {etiqueta}", 2, rtt,
             nota="(ya con conexión directa)")
    print("     Mismo país vs distinto: x2,9. Mejor nodo vs peor: x8,1.")
    print("     OJO: las dos tiradas con el MISMO country_codes=['us'] dieron medianas de")
    print("     65 y 132 ms -> la etiqueta de país no basta, hay que MEDIR el RTT.")

    print("\nC) COMBINADO: las dos palancas se necesitan mutuamente")
    fila("sólo quitar el relé, mala tirada de nodos", 2, 177.0)
    fila("sólo elegir nodos, manteniendo el relé", 4, 65.0)
    fila("quitar el relé + elegir nodos (65 ms)", 2, 65.0, nota="<- objetivo cumplido")
    fila("+ el mejor nodo medido (54 ms)", 2, 54.0)

    print("\nD) SIN RED: un nodo con todo el modelo (pesos por streaming desde RAM)")
    print("     el techo ya no es la red sino leer los pesos; depende del modelo:")
    for gb, bw, nombre in ((3.5, 25.0, "modelo 7B en int4, RAM DDR4 ~25 GB/s"),
                           (7.0, 25.0, "modelo 14B en int4, RAM DDR4 ~25 GB/s"),
                           (3.5, 50.0, "modelo 7B en int4, RAM DDR5 ~50 GB/s")):
        ms = gb / bw * 1000.0
        v = 1000.0 / ms
        marca = " ✅" if v >= OBJETIVO else "  "
        print(f"  {nombre:<52} {v:6.2f} tok/s{marca}  ({ms:.0f} ms/token leyendo pesos)")
    print("     OJO: sólo aplica si el modelo CABE en la RAM de un nodo.")

    print("\n" + "=" * 78)
    print("LECTURA:")
    print("  · Conexiones 'en paralelo' NO ayudan por sí solas: el problema es la")
    print("    LATENCIA, no el ancho de banda (medido: comprimir no cambió nada).")
    print("    Abrir más sockets no acorta el viaje de ida y vuelta.")
    print("  · Lo que SÍ es paralelo y sirve: mandar la misma activación por DOS")
    print("    caminos y quedarse con la que llegue antes (mata la variabilidad,")
    print("    no la latencia base). Nos sobra ancho de banda para hacerlo.")
    print("  · La palanca grande es ELEGIR NODOS CERCANOS, y crece con el fleet:")
    print("    con 100 voluntarios sondeas veinte y te quedas con los cuatro mejores.")
    print("  · Pero NO por la etiqueta de país: dos tiradas idénticas dieron 65 y 132 ms.")
    print("    Hay que medir el RTT de verdad antes de formar la cadena (Exp15/16).")
    print("  · Y sirve tanto para la media como para la COLA: la peor tirada medida")
    print("    (Taiwán, 437 ms) hunde una cadena de 4 tramos a 0,55 tok/s. Un solo nodo")
    print("    lejano arrastra a todos los demás.")
    print("=" * 78)


if __name__ == "__main__":
    main()
