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

# El trinquete llegó a cero el 26-07-2026: la suite completa pasó sin fallos.
# La estructura se conserva para que cualquier excepción futura tenga que
# documentarse explícitamente y, cuando se arregle, vuelva a obligar a podarla.
KNOWN_FAILURES: dict[str, str] = {}

# INTERMITENTES: pueden pasar o fallar en la misma máquina sin que cambie nada.
# Todos levantan procesos o sockets de verdad, así que dependen de puertos, de
# arranque y del planificador del SO. Mezclarlos con los deterministas rompería
# el trinquete: oscilarían entre "regresión" y "ya no falla" en cada corrida y
# la señal se perdería. Aquí no rompen la construcción en ningún sentido, pero
# se cuentan y se anuncian, para que una racha rara sea visible.
#
# En el host de desarrollo la causa está diagnosticada: `torch.distributed`
# (c10d) intenta ligar a `kubernetes.docker.internal` —entrada que Docker
# Desktop mete en el fichero `hosts`— y falla con WinError 10049. En un runner
# limpio de CI no deberían fallar; si nunca fallan durante un tiempo, hay que
# ascenderlos a normales y quitarlos de aquí.
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


def _is_flaky(test_id: str) -> bool:
    return any(test_id.startswith(prefix) for prefix in FLAKY_PREFIXES)


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

    failed = {_test_id(test) for test, _ in result.failures + result.errors}
    flaky_failed = sorted(name for name in failed if _is_flaky(name))
    unexpected = sorted(failed - set(KNOWN_FAILURES) - set(flaky_failed))
    fixed = sorted(set(KNOWN_FAILURES) - failed)

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
