#!/usr/bin/env python
"""Corre la suite de Python y aplica un TRINQUETE de fallos conocidos.

Por qué existe. El workflow reusable de CI detecta el stack del repositorio y
elige Node, así que el job `Test (Python)` sale **skipped**: los ~850 tests de
Python —todo el camino caliente del runtime— no se ejecutan nunca en CI. Se
diagnosticó varias veces y quedó anotado en `docs/REGISTRO_VERIFICACIONES.md`
§9, pero el diagnóstico no ejecuta tests.

Por qué un trinquete y no "todo verde o rojo". Hay fallos preexistentes reales.
Encender CI en rojo desde el primer día entrena a quien revisa a ignorar los
checks, que es exactamente cómo se cuela el fallo que sí importa (el mismo
razonamiento que el registro §9 aplica al bloqueo de Semgrep). Así que:

  - Un fallo **en** la lista de abajo no rompe la corrida, pero se anuncia.
  - Un fallo **fuera** de la lista rompe la corrida. Esa es la regresión.
  - Un test de la lista que **empieza a pasar** también rompe la corrida, para
    obligar a quitarlo. La lista sólo puede encoger.

Uso:
    python scripts/run-python-tests.py            # desde la raíz del repo
    python scripts/run-python-tests.py --strict   # ignora la lista: todo debe pasar
"""
from __future__ import annotations

import argparse
import os
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO / "python"

# En Windows la lista está VACÍA: allí cualquier fallo rompe. En Linux hay
# TRES entradas, y las tres son la misma deuda (ver abajo).
#
# CADA ENTRADA NECESITA: por qué falla y qué haría falta para quitarla.
#
# ⚠️ HISTORIA, porque es una lección de método y no conviene repetirla. Aquí
# hubo tres entradas y se describieron MAL DOS VECES:
#
#   1. (25-07) Como "fallos preexistentes reales" del códec deflate, medidos en
#      Windows. Daban a entender que el códec estaba roto.
#   2. (25-07, tarde) Corregido a "específicos de Windows" cuando la primera
#      corrida en CI —runner limpio de Linux, PR #19— los pasó los tres.
#   3. (26-07) Y ahora resulta que **tampoco fallan en Windows**: los 1.092
#      tests pasan aquí, y pasan con el `protocol.py` de antes y el de después
#      de la fusión (A/B directo), así que el cambio de código no fue la causa.
#
# Lo que cambió es el ENTORNO. Estas corridas usan el venv sellado —Python
# 3.12.13, numpy 1.26.4, torch 2.13.0+cpu, zlib 1.3.1—; las del 25-07 se
# hicieron con el venv re-aprovisionado en vivo por la automatización, que es un
# riesgo ya anotado en la memoria del proyecto ("congelar el venv antes de
# medir"). Y el test que más ruido dio compara un ratio de compresión contra 1,0
# habiendo medido 1,06: a esa distancia del umbral, cualquier cambio de zlib o
# de numpy vuelca el signo.
#
# La lección, que es la misma que la de los seis P2 del PR #19: un resultado
# atribuido a la causa equivocada. Primero al código, luego a la plataforma, y
# era el entorno. **Antes de declarar un fallo "conocido", fijar el entorno.**
#
# ⚠️ LAS TRES ENTRADAS DE LINUX SON DEUDA REAL, Y LAS DESTAPÓ ESTE MISMO GUION.
# Con la regla vieja —eximir por prefijo de clase— llevaban al menos dos corridas
# de CI fallando y saliendo como «intermitentes que han fallado esta vez, no
# rompen la construcción». Se comprobó en el log de la corrida VERDE `f4e5bce`:
# fallaban exactamente esas. No las causó fijar las dependencias; estaban
# tapadas. Es el punto ciego que motivó el arreglo, encontrado por el arreglo.
#
# Y son UNA SOLA deuda con tres caras: los tres comparan una célula
# tensor-parallel contra la referencia de un proceso con `rtol=atol=1e-5` sobre
# float32, los tres pasan en Windows y fallan en Linux, y los tres se irán
# juntos cuando alguien resuelva el orden de acumulación. Que caigan tests de
# DOS clases distintas por el mismo umbral es lo que descarta que sea un bug de
# una ruta concreta.
KNOWN_FAILURES_BY_PLATFORM: dict[str, dict[str, str]] = {
    "linux": {
        "test_cell_stage.TensorParallelCellStageTests"
        ".test_local_cell_fork_has_exact_bytes_independent_kv_and_reference_promotion":
            "Tolerancia numérica: la salida de la célula tensor-parallel contra "
            "la referencia de un solo proceso falla 1 de 16 elementos, con "
            "4,196e-05 absoluto frente a 1e-05 permitido (relativo 1,777e-05). "
            "PASA en Windows con el MISMO torch sellado (2.13.0+cpu), así que "
            "apunta a orden de acumulación distinto por despachar a un BLAS "
            "distinto — float32 no es asociativo y la célula reduce sumas "
            "parciales en otro orden que la referencia. "
            "PARA QUITARLA: comprobar primero que célula y referencia hacen las "
            "MISMAS operaciones. Si lo hacen, la tolerancia de 1e-5 es "
            "demasiado justa para recomposición tensor-parallel en float32 y "
            "hay que justificar una nueva. NO ensanchar la tolerancia antes de "
            "esa comprobación: taparía un fallo de orden real igual de bien.",
        "test_external_cell.ExternalTensorParallelCellTests"
        ".test_external_rank_cli_and_anchor_rank_zero_execute_one_logical_stage":
            "El mismo patrón y probablemente la misma causa: "
            "`torch.allclose(actual_prompt, expected_prompt, rtol=1e-5, "
            "atol=1e-5)` en `test_external_cell.py:246`, célula tensor-parallel "
            "contra referencia de un proceso. También pasa en Windows. "
            "PARA QUITARLA: la misma comprobación que la anterior; si comparten "
            "causa, se van las tres juntas.",
        "test_cell_stage.TensorParallelCellStageTests"
        ".test_wire_loop_wraps_two_member_cell_as_one_logical_gdlp_stage":
            "TERCERO de la misma familia, y por eso se DECLARA en vez de "
            "investigarse aparte: `torch.allclose(decode_tensor(prefill), "
            "expected_prompt, rtol=1e-5, atol=1e-5)` en "
            "`test_cell_stage.py:821`. Célula tensor-parallel contra "
            "referencia de un proceso, la MISMA tolerancia justa de 1e-5 "
            "sobre float32, y el mismo comportamiento: pasa en Windows y "
            "falla en Linux. Que sean TRES tests de DOS clases distintas "
            "cayendo por el mismo umbral refuerza el diagnóstico —orden de "
            "acumulación, no un bug de una ruta concreta— y refuerza que el "
            "arreglo correcto es UNO solo. "
            "PARA QUITARLA: la misma comprobación que las otras dos.",
    },
}

KNOWN_FAILURES = KNOWN_FAILURES_BY_PLATFORM.get(sys.platform, {})

# INTERMITENTES: pueden pasar o fallar en la misma máquina sin que cambie nada.
# Todos levantan procesos o sockets de verdad, así que dependen de puertos, de
# arranque y del planificador del SO. Mezclarlos con los deterministas rompería
# el trinquete: oscilarían entre "regresión" y "ya no falla" en cada corrida y
# la señal se perdería. Aquí no rompen la construcción en ningún sentido, pero
# se cuentan y se anuncian, para que una racha rara sea visible.
#
# En Windows la causa está diagnosticada: `torch.distributed` (c10d) intenta
# ligar a `kubernetes.docker.internal` —entrada que Docker Desktop mete en el
# fichero `hosts`— y falla con WinError 10049.
#
# ⚠️ Pero en Linux TAMBIÉN fallan a veces: la primera corrida en CI (PR #19)
# tumbó dos de estas clases. O sea que no era sólo Docker Desktop — levantar
# ranks reales es intermitente de por sí. Se quedan como intermitentes en TODAS
# las plataformas hasta que alguien los estabilice (puertos fijos, esperas
# explícitas) en vez de esperar a que el entorno mejore solo.
# Se listan por PREFIJO (clase o módulo), no test a test. La causa es común a
# todo el grupo —levantar ranks o procesos de verdad—, así que enumerarlos uno a
# uno sólo garantiza que la lista se quede corta: en esta misma sesión aparecieron
# cinco casos nuevos de las mismas dos clases entre dos corridas consecutivas.
FLAKY_PREFIXES = {
    "test_external_cell.ExternalTensorParallelCellTests":
        "Levanta ranks reales de torch.distributed (c10d).",
    "test_cell_stage.TensorParallelCellStageTests":
        "Células tensor-parallel con ranks y sockets reales.",
    "test_resident_expert_rpc.ResidentExpertRpcTests":
        "RPC persistente entre procesos reales.",
    "test_ram_backed_moe_stage.RamBackedMoeStageRunnerTests":
        "Genera procesos hijo reales.",
}


# El prefijo NO basta para eximir. Un prefijo dice "esta clase levanta procesos",
# no "este fallo concreto es de arranque". Eximir por prefijo a secas convierte
# esas cuatro clases en un punto ciego permanente: una regresión determinista en
# un `assert` de `test_cell_stage` —un cálculo que empieza a dar mal— se restaría
# de `unexpected` igual que un puerto ocupado, y el trinquete la anunciaría como
# "intermitente que ha fallado esta vez" en vez de romper la construcción.
#
# Eso es peor que no tener trinquete, porque da una señal de seguridad falsa
# justo en el código de ranks y sockets, que es el que menos se mira a mano.
#
# Así que la exención pide DOS cosas: la clase está en la lista Y el traceback
# se parece al modo de fallo diagnosticado (arranque de proceso o de socket).
# Cualquier otro fallo en esas mismas clases cuenta como regresión.
FLAKY_SIGNATURES = (
    "WinError 10049",              # bind a kubernetes.docker.internal (Docker Desktop)
    "WinError 10048",              # puerto ya en uso
    "Address already in use",
    "Cannot assign requested address",
    "Connection refused",
    "ConnectionResetError",
    "ConnectionAbortedError",
    "BrokenPipeError",
    "ProcessExitedException",      # torch.multiprocessing: el hijo murió
    "ProcessRaisedException",
    "torch.distributed",
    "c10d",
    "The client socket has failed to connect",
    "Timed out initializing process group",
    "timed out waiting for",
    "rendezvous",
)


def _is_flaky(test_id: str, traceback_text: str) -> bool:
    """Exime sólo si la clase es de las conocidas Y el fallo es de arranque."""
    if not any(test_id.startswith(prefix) for prefix in FLAKY_PREFIXES):
        return False
    return any(signature in traceback_text for signature in FLAKY_SIGNATURES)


def _test_id(test) -> str:
    """Id canónico: sin la parametrización de subtest y sin el paquete `tests.`.

    `unittest discover -s tests` etiqueta los tests como `test_modulo.Clase.caso`,
    pero `python -m unittest tests.test_modulo` los etiqueta con el prefijo
    `tests.`. Normalizar aquí evita que la lista deje de casar según cómo se haya
    invocado la suite — que es justo el fallo que hacía que el trinquete de esta
    misma sesión anunciara seis regresiones inexistentes.
    """
    raw = test.id().split(" ")[0]
    return raw[len("tests."):] if raw.startswith("tests.") else raw


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true",
                        help="Ignora KNOWN_FAILURES: cualquier fallo rompe.")
    args = parser.parse_args()

    os.chdir(PYTHON_DIR)
    sys.path.insert(0, str(PYTHON_DIR))

    suite = unittest.defaultTestLoader.discover("tests")
    runner = unittest.TextTestRunner(verbosity=1, stream=sys.stdout)
    result = runner.run(suite)

    # Un módulo que no importa aparece como un "test" que falla al cargarse, y
    # sus tests reales NO se cuentan. Sin mirar esto, un conteo de tests miente:
    # sin aiohttp la suite parecía de 808 cuando es de 852.
    load_errors = [
        _test_id(test) for test, _ in result.errors
        if "_FailedTest" in type(test).__name__
    ]

    # id -> traceback, porque la exención de intermitentes se decide por el modo
    # de fallo y no sólo por el nombre. Un mismo test puede salir en `failures` y
    # en `errors`; se concatenan para no perder la firma que lo clasifica.
    tracebacks: dict[str, str] = {}
    for test, trace in result.failures + result.errors:
        name = _test_id(test)
        tracebacks[name] = tracebacks.get(name, "") + trace

    failed = set(tracebacks)
    flaky_failed = sorted(name for name in failed if _is_flaky(name, tracebacks[name]))
    unexpected = sorted(failed - set(KNOWN_FAILURES) - set(flaky_failed))
    fixed = sorted(set(KNOWN_FAILURES) - failed)
    # Fallos en una clase intermitente que NO son de arranque. Se señalan aparte
    # porque son el caso que antes se colaba: rompen igual que cualquier otra
    # regresión, pero quien lea el log necesita saber por qué esta vez sí cuenta.
    deterministic_in_flaky_class = [
        name for name in unexpected
        if any(name.startswith(prefix) for prefix in FLAKY_PREFIXES)
    ]

    print()
    print("=" * 70)
    print(f"corridos={result.testsRun}  fallos={len(result.failures)}  "
          f"errores={len(result.errors)}  saltados={len(result.skipped)}")

    if load_errors:
        print()
        print("MÓDULOS QUE NO IMPORTAN (sus tests no se han contado):")
        for name in load_errors:
            print(f"  - {name}")
        print("  Suele ser una dependencia ausente. Arréglalo antes de fiarte")
        print("  del número de tests corridos.")

    if args.strict:
        ok = not (result.failures or result.errors)
        print("modo estricto: " + ("OK" if ok else "FALLA"))
        return 0 if ok else 1

    if unexpected:
        print()
        print("REGRESIÓN — fallos que no estaban en la lista conocida:")
        for name in unexpected:
            print(f"  - {name}")

    if deterministic_in_flaky_class:
        print()
        print("  ⚠️ De esos, estos son de una clase marcada como intermitente,")
        print("     pero su traceback NO es de arranque de proceso ni de socket:")
        for name in deterministic_in_flaky_class:
            print(f"       - {name}")
        print("     Trátalos como regresión de verdad. Si resulta ser un modo de")
        print("     fallo de arranque nuevo, añade su firma a FLAKY_SIGNATURES —")
        print("     nunca ensanches la exención a la clase entera.")

    if fixed:
        print()
        print("YA NO FALLAN — quítalos de KNOWN_FAILURES en este script:")
        for name in fixed:
            print(f"  - {name}")
        print("  (la lista sólo puede encoger; si no se poda, deja de servir)")

    if flaky_failed:
        print()
        print(f"intermitentes que han fallado esta vez ({len(flaky_failed)}), "
              "no rompen la construcción:")
        for name in flaky_failed:
            print(f"  - {name}")

    if not unexpected and not fixed:
        print()
        print(f"sin regresiones. {len(KNOWN_FAILURES)} fallos conocidos siguen ahí.")

    return 1 if (unexpected or fixed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
