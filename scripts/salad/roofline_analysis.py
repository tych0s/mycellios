#!/usr/bin/env python
"""Techo teórico (roofline) del decode distribuido vs lo MEDIDO.

El decode autoregresivo es memory-bandwidth-bound: cada paso debe LEER todos los
pesos de la etapa. Con fusión B_eff, un solo forward produce B_eff tokens, así
que el coste de leer pesos se amortiza entre ellos.

  t_forward_min = bytes_pesos_etapa / ancho_banda_gpu
  tok/s_max     = B_eff / t_forward_min      (por etapa; el pipeline va al ritmo
                                              de la etapa más lenta)

Comparar con lo medido dice si el cuello es la GPU (cerca del techo) o el
software/red (lejos). Sin números inventados: todo se pasa por CLI.
"""
from __future__ import annotations
import argparse

# Anchos de banda de memoria (GB/s) — especificación del fabricante
GPUS = {
    "gtx1050ti": 112.0,
    "gtx1650": 192.0,
    "rtx3060": 360.0,
    "rtx4090": 1008.0,
}


def human(x, unit=""):
    return f"{x:,.2f}{unit}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params-b", type=float, default=0.6, help="parámetros del modelo en miles de millones")
    ap.add_argument("--bytes-per-param", type=float, default=2.0, help="2=fp16, 1=int8, 0.5=int4")
    ap.add_argument("--stages", type=int, default=2)
    ap.add_argument("--gpu", default="gtx1050ti", choices=sorted(GPUS))
    ap.add_argument("--bw-efficiency", type=float, default=0.75,
                    help="fracción realista del ancho de banda pico alcanzable (0.7-0.85 típico)")
    ap.add_argument("--beff", type=float, required=True, help="secuencias fusionadas por forward MEDIDO")
    ap.add_argument("--measured-tok-s", type=float, required=True, help="tok/s agregado MEDIDO")
    args = ap.parse_args()

    bw = GPUS[args.gpu] * 1e9 * args.bw_efficiency
    total_bytes = args.params_b * 1e9 * args.bytes_per_param
    stage_bytes = total_bytes / args.stages

    t_fwd_min = stage_bytes / bw                     # s por forward (solo leer pesos)
    fwd_s_max = 1.0 / t_fwd_min
    tok_s_max = fwd_s_max * args.beff

    fwd_s_meas = args.measured_tok_s / args.beff
    t_fwd_meas = 1.0 / fwd_s_meas
    util = args.measured_tok_s / tok_s_max
    overhead_ms = (t_fwd_meas - t_fwd_min) * 1e3

    print("=" * 66)
    print(f"MODELO {args.params_b}B x {args.bytes_per_param}B/param = {human(total_bytes/1e9,' GB')} pesos")
    print(f"REPARTO {args.stages} etapas -> {human(stage_bytes/1e9,' GB')}/etapa")
    print(f"GPU {args.gpu}: {GPUS[args.gpu]} GB/s pico x {args.bw_efficiency} = {human(bw/1e9,' GB/s')} efectivo")
    print("-" * 66)
    print("TECHO TEÓRICO (solo lectura de pesos, todo lo demás gratis):")
    print(f"  t_forward_min = {human(t_fwd_min*1e3,' ms')}   -> {human(fwd_s_max,' forwards/s')}")
    print(f"  con B_eff={args.beff} fusionadas -> TECHO {human(tok_s_max,' tok/s')}")
    print("-" * 66)
    print("MEDIDO:")
    print(f"  {human(args.measured_tok_s,' tok/s')} con B_eff={args.beff} -> {human(fwd_s_meas,' forwards/s')}"
          f" ({human(t_fwd_meas*1e3,' ms')}/forward)")
    print("-" * 66)
    print(f"UTILIZACIÓN DEL TECHO: {util*100:.1f}%")
    print(f"SOBRECOSTE por forward: {human(overhead_ms,' ms')} "
          f"({(t_fwd_meas/t_fwd_min):.1f}x el mínimo teórico)")
    print()
    if util < 0.35:
        print(">> DIAGNÓSTICO: MUY lejos del techo. El cuello NO es la GPU ni el ancho")
        print("   de banda de pesos, sino SOBRECOSTE POR PASO (Python, serialización,")
        print("   TCP/WS por frame, planificación, sincronía entre etapas, RTT WAN).")
        print("   Palancas: subir B_eff (fusionar más), reducir pasos (especulación /")
        print("   multi-token), y recortar el coste fijo por forward.")
    elif util < 0.7:
        print(">> DIAGNÓSTICO: a media distancia. Hay margen tanto en sobrecoste como")
        print("   en fusión; medir el desglose por etapa antes de optimizar.")
    else:
        print(">> DIAGNÓSTICO: cerca del techo de ancho de banda. Para ir más rápido")
        print("   hay que CAMBIAR el techo: cuantizar pesos (int8/int4 = leer menos")
        print("   bytes) o GPUs con más ancho de banda.")
    print()
    print("Si se cuantizan los pesos, el techo escala inversamente a los bytes:")
    for bpp, name in ((1.0, "int8"), (0.5, "int4")):
        print(f"  {name}: techo x{args.bytes_per_param/bpp:.0f} = "
              f"{human(tok_s_max*args.bytes_per_param/bpp,' tok/s')} "
              f"(solo ayuda si la utilización actual fuera alta)")
    print("=" * 66)


if __name__ == "__main__":
    main()
