#!/usr/bin/env python
"""Lanza UN nodo GPU en Salad, mide C, recoge el resultado y borra el contenedor.

Alcance deliberadamente mínimo: un contenedor, minutos, sin relé ni
coordinación multi-nodo. Es el experimento de Fase 0 que produce el término C y
el veredicto T3 — el único que puede refutar la recomendación de no migrar a
Rust — y no necesita nada de la maquinaria de pipeline.

CONTROL DE GASTO
----------------
* Una sola instancia, clase de GPU barata por defecto (~0,02 $/h).
* `restart_policy: never`: si el proceso sale, la instancia no revive.
* TTL dentro del propio contenedor: se autodestruye aunque este guion muera.
* `finally` que BORRA el contenedor pase lo que pase, incluido Ctrl-C.

Uso:
    SALAD_API_KEY=... python scripts/salad/run_gpu_profile.py --wait-seconds 900
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import pathlib
import sys
import time
import urllib.error
import urllib.request

API = "https://api.salad.com/api/public"
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
UA = "gdlp-gpu-profile"

#: GTX 1650 (4 GB) — 0,015 $/h. Suficiente para un forward sintético de decode.
DEFAULT_GPU_CLASS = "0f60d6f5-642a-44a5-af51-3ac34d7602c1"
DEFAULT_IMAGE = "pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime"


def key() -> str:
    value = os.environ.get("SALAD_API_KEY")
    if not value:
        sys.exit("falta SALAD_API_KEY")
    return value


def call(method: str, path: str, body: dict | None = None, timeout: float = 60.0):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        API + path,
        data=data,
        method=method,
        headers={
            "Salad-Api-Key": key(),
            "Content-Type": "application/json",
            "User-Agent": UA,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode()
            return response.status, (json.loads(raw) if raw else {})
    except urllib.error.HTTPError as error:
        raw = error.read().decode()
        try:
            return error.code, json.loads(raw)
        except ValueError:
            return error.code, {"raw": raw}
    except (urllib.error.URLError, OSError) as error:
        # `HTTPError` sola NO basta: un corte de red o un timeout durante el
        # teardown escapaba y abortaba el borrado, dejando la GPU facturando.
        # Es el fallo que ya se corrigió en `latency_map.py` y que seguía vivo
        # en los orquestadores.
        return 0, {"error": str(error)}


def delete_container_verified(base: str, name: str, attempts: int = 4) -> bool:
    """Borra y **verifica listando**. Un 2xx no prueba que el recurso muriera.

    Regla del proyecto para cualquier script que cree recursos de pago: borrado
    aislado por recurso, con reintentos, y comprobación positiva de que ya no
    está. Confiar en el código de respuesta es lo que deja facturas abiertas.
    """
    for attempt in range(attempts):
        call("DELETE", f"{base}/{name}")
        status, listing = call("GET", base)
        if status == 200:
            names = {item.get("name") for item in listing.get("items", [])}
            if name not in names:
                return True
        time.sleep(2 * (attempt + 1))
    print(
        f"[run] ⚠️ NO se pudo confirmar el borrado de {name}. Bórralo a mano:\n"
        f"  curl -X DELETE -H \"Salad-Api-Key: $SALAD_API_KEY\" \\\n"
        f"    {API}{base}/{name}",
        file=sys.stderr,
    )
    return False


#: Tope por variable de entorno. `pipeline_orchestrate.py::chunk_env` usa 990 y
#: lleva meses en producción; el primer intento de esta sonda metió dos ficheros
#: en base64 sin trocear y el contenedor murió con `start_time == finish_time`,
#: sin llegar a ejecutar nada.
ENV_CHUNK = 990


def chunk_env(env: dict[str, str], name: str, raw: bytes) -> int:
    encoded = base64.b64encode(raw).decode()
    parts = [encoded[i : i + ENV_CHUNK] for i in range(0, len(encoded), ENV_CHUNK)]
    for index, part in enumerate(parts):
        env[f"{name}_{index}"] = part
    env[f"{name}_PARTS"] = str(len(parts))
    return len(parts)


def build_command() -> list[str]:
    """Bootstrap en Python puro: reensambla los trozos y arranca la sonda.

    Se usa `python -c` y no `bash`: la imagen base de PyTorch trae Python por
    definición, pero no hay garantía de `bash` ni de `base64` como binarios, y
    depender de ellos añade dos modos de fallo que no aportan nada.

    Se inyecta por entorno en vez de construir una imagen propia porque, para un
    experimento de minutos, un build y un push cuestan más que la medida — y una
    imagen nueva es otra variable que explicar si el número sale raro.
    """
    loader = (
        "import base64,os,pathlib;"
        "d=pathlib.Path('/opt/gdlp');d.mkdir(parents=True,exist_ok=True);"
        "join=lambda n:base64.b64decode("
        "''.join(os.environ[f'{n}_{i}'] for i in range(int(os.environ[f'{n}_PARTS']))));"
        "(d/'profile_forward.py').write_bytes(join('GDLP_FORWARD'));"
        "(d/'gpu_profile_probe.py').write_bytes(join('GDLP_PROBE'));"
        "os.chdir(d);"
        "exec(compile((d/'gpu_profile_probe.py').read_text(),'<probe>','exec'),"
        "{'__name__':'__main__','__file__':str(d/'gpu_profile_probe.py')})"
    )
    return ["python", "-c", loader]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--org", default="nodecodex")
    parser.add_argument("--project", default="mycellios")
    parser.add_argument("--name", default="gdlp-cprofile")
    parser.add_argument("--gpu-class", default=DEFAULT_GPU_CLASS)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    # 200 iteraciones son baratas SIN profiler, pero la pasada de
    # `torch.profiler` usa `iters//4` y registra ~2.037 eventos por iteración en
    # un modelo real: con 50 iteraciones son >100.000 eventos y la pasada tarda
    # más de media hora. Para modelo real, 40 iteraciones (10 perfiladas) dan la
    # misma fracción CPU/kernel en minutos.
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument(
        "--hf-model",
        help="modelo REAL de Hugging Face; sin esto se usa la pila sintética",
    )
    parser.add_argument("--ttl", type=int, default=1800)
    parser.add_argument(
        "--wait-seconds",
        type=int,
        default=2100,
        # La imagen de PyTorch pesa ~7 GB y el gateway devuelve 503 con el
        # contenedor ya en `running` mientras se descomprime: **12-22 min** de
        # 503 mudo que son indistinguibles de un cuelgue. Esperar menos de eso
        # tira el contenedor justo antes de que empiece a medir.
        help="espera máxima; el pull de la imagen ya se come 12-22 min",
    )
    parser.add_argument("--out", type=pathlib.Path)
    parser.add_argument(
        "--keep",
        action="store_true",
        help="NO borrar el contenedor al terminar (deja gasto abierto)",
    )
    args = parser.parse_args(argv)

    probe = REPO_ROOT / "deploy" / "salad" / "gpu_profile_probe.py"
    forward = REPO_ROOT / "scripts" / "profile_forward.py"
    for path in (probe, forward):
        if not path.exists():
            sys.exit(f"falta {path}")

    base = f"/organizations/{args.org}/projects/{args.project}/containers"
    env: dict[str, str] = {
        "GDLP_PROBE_TTL": str(args.ttl),
        "GDLP_PROFILE_ITERS": str(args.iters),
        "GDLP_PROFILE_LAYERS": str(args.layers),
        "GDLP_PROFILE_HIDDEN": str(args.hidden),
        **({"GDLP_HF_MODEL": args.hf_model} if args.hf_model else {}),
    }
    parts = chunk_env(env, "GDLP_FORWARD", forward.read_bytes())
    parts += chunk_env(env, "GDLP_PROBE", probe.read_bytes())
    print(f"[run] guiones troceados en {parts} variables de entorno")

    body = {
        "name": args.name,
        "display_name": args.name,
        "container": {
            "image": args.image,
            "resources": {
                "cpu": 2,
                "memory": 8192,
                "gpu_classes": [args.gpu_class],
            },
            "environment_variables": env,
            "command": build_command(),
        },
        "replicas": 1,
        # Sin `autostart_policy` el contenedor se crea y se queda `stopped` con
        # `start_time == finish_time`, que es lo que pasó en los dos primeros
        # intentos: parece un fallo de arranque cuando en realidad nunca se pidió
        # arrancarlo. `pipeline_orchestrate.py:377` ya lo ponía.
        "autostart_policy": True,
        "restart_policy": "never",
        "networking": {
            "protocol": "http",
            "port": 8000,
            "auth": False,
            "client_request_timeout": 100,
        },
    }

    # Un contenedor con el mismo nombre de una corrida anterior seguiría
    # facturando y devolvería datos viejos. Se borra antes de crear.
    call("DELETE", f"{base}/{args.name}")
    status, created = call("POST", base, body)
    if status not in (200, 201, 202):
        sys.exit(f"no se pudo crear el contenedor ({status}): {json.dumps(created)[:500]}")
    print(f"[run] contenedor creado: {args.name}")

    result: dict | None = None
    try:
        deadline = time.time() + args.wait_seconds
        url = None
        while time.time() < deadline:
            status, detail = call("GET", f"{base}/{args.name}")
            url = url or (detail.get("networking") or {}).get("dns")
            state = ((detail.get("current_state") or {}).get("status")) or "?"
            instances = (detail.get("current_state") or {}).get("instance_status_counts", {})
            print(f"[run] estado={state} {json.dumps(instances)} url={url}", flush=True)
            if url:
                try:
                    request = urllib.request.Request(
                        f"https://{url}/", headers={"User-Agent": UA}
                    )
                    with urllib.request.urlopen(request, timeout=20) as response:
                        payload = json.loads(response.read().decode())
                    print(f"[run] fase={payload.get('phase')}", flush=True)
                    if payload.get("phase") in ("done", "failed"):
                        result = payload
                        break
                except Exception:
                    pass  # aún arrancando: el gateway tarda en enrutar
            time.sleep(20)
        if result is None:
            print("[run] SIN RESULTADO dentro del plazo", file=sys.stderr)
    finally:
        if args.keep:
            print(f"[run] --keep: el contenedor {args.name} SIGUE FACTURANDO")
        elif delete_container_verified(base, args.name):
            print("[run] contenedor borrado y VERIFICADO ausente de la lista")

    if result is None:
        return 1
    if result.get("phase") == "failed":
        print("\n=== FALLO EN EL NODO ===\n" + str(result.get("error"))[:3000])
        return 1

    print("\n" + str(result.get("render", "")))
    if args.out:
        args.out.write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
