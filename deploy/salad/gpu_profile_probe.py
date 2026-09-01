#!/usr/bin/env python
"""Fase 0 en GPU real: mide C y decide T3. Se autodestruye para no facturar.

QUÉ MIDE
--------
Lo mismo que `scripts/profile_forward.py`, pero dentro de un contenedor de
Salad con GPU de verdad, que es donde el número cuenta. En CPU el perfilador se
niega —correctamente— a evaluar T3, porque despacho y ejecución coinciden.

  * `wall_ms`  con barrera de dispositivo  -> **el término C**
  * `dispatch_ms` sin barrera              -> coste de encolar desde Python
  * `torch.profiler` self-CPU vs self-CUDA -> **T3**: ¿domina el intérprete?

C nunca se ha medido en este proyecto. La cota existente (28,9 ms de 33,35, el
87 % del forward) sale de restar contra el techo roofline, y una resta no es una
medida. Ese factor decide si tiene sentido tocar el motor.

SEGURIDAD DE FACTURACIÓN
------------------------
Un contenedor de GPU olvidado factura para siempre. Este proceso tiene un plazo
de muerte ABSOLUTO (`GDLP_PROBE_TTL`) que corre desde el arranque y se cumple
pase lo que pase: si el orquestador muere de forma dura, si la red se cae, si
nadie recoge el resultado. Con `restart_policy=never`, salir del proceso libera
la instancia. El TTL no es una cortesía: es lo único que separa un experimento
de 20 céntimos de una factura abierta.

Env: GDLP_PROBE_TTL, GDLP_PORT, GDLP_PROFILE_ITERS, GDLP_PROFILE_LAYERS,
     GDLP_PROFILE_HIDDEN.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import traceback

PORT = int(os.environ.get("GDLP_PORT", "8000"))
TTL = float(os.environ.get("GDLP_PROBE_TTL", "1800"))
ITERS = int(os.environ.get("GDLP_PROFILE_ITERS", "200"))
LAYERS = int(os.environ.get("GDLP_PROFILE_LAYERS", "8"))
HIDDEN = int(os.environ.get("GDLP_PROFILE_HIDDEN", "2048"))
#: Modelo real de Hugging Face. Vacío = pila sintética (valida el instrumento).
HF_MODEL = os.environ.get("GDLP_HF_MODEL", "").strip() or None

STATE: dict = {
    "phase": "starting",
    "started": time.time(),
    "ttl_seconds": TTL,
    "result": None,
    "error": None,
    "gpu": None,
    "hf_model": HF_MODEL,
    "pip": None,
}


def _gpu_facts() -> dict:
    """Qué GPU tocó. Sin esto el número no es interpretable ni reproducible."""
    facts: dict = {"nvidia_smi": None, "torch": None}
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
        facts["nvidia_smi"] = (out.stdout or out.stderr).strip()
    except Exception as error:
        facts["nvidia_smi"] = f"no disponible: {error}"
    try:
        import torch

        facts["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name(0)
            if torch.cuda.is_available() else None,
            "capability": list(torch.cuda.get_device_capability(0))
            if torch.cuda.is_available() else None,
        }
    except Exception as error:
        facts["torch"] = f"no disponible: {error}"
    return facts


def _ensure_transformers() -> None:
    """La imagen de PyTorch no trae `transformers`. Se instala si hace falta.

    Solo cuando se pide modelo real: para la pila sintética es peso muerto y
    varios minutos de arranque que se pagan en una GPU facturando.
    """
    if not HF_MODEL:
        return
    try:
        import transformers  # noqa: F401

        return
    except ImportError:
        pass
    STATE["phase"] = "installing"
    print("[gpu-probe] instalando transformers…", flush=True)
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-cache-dir",
         "transformers", "accelerate", "safetensors"],
        capture_output=True, text=True, timeout=900,
    )
    STATE["pip"] = (result.stdout or "")[-2000:] + (result.stderr or "")[-2000:]
    if result.returncode != 0:
        raise RuntimeError(f"pip falló con {result.returncode}")


def _run_profile() -> None:
    """Ejecuta el perfilado. Cualquier fallo queda en `error`, no tumba el HTTP.

    El servidor tiene que seguir en pie aunque el perfilado falle: si el proceso
    muere, el contenedor se reinicia o se queda sin nadie que informe de por qué
    falló, y el diagnóstico se pierde justo cuando más falta hace.
    """
    try:
        STATE["gpu"] = _gpu_facts()
        _ensure_transformers()
        STATE["phase"] = "profiling"
        sys.path.insert(0, "/opt/gdlp")
        from profile_forward import profile_forward, render  # type: ignore

        breakdown = profile_forward(
            layers=LAYERS,
            hidden=HIDDEN,
            iterations=ITERS,
            warmup=max(8, ITERS // 10),
            device="auto",
            use_profiler=True,
            hf_model=HF_MODEL,
        )
        from dataclasses import asdict

        STATE["result"] = asdict(breakdown)
        STATE["render"] = render(breakdown)
        STATE["phase"] = "done"
        print(render(breakdown), flush=True)
    except Exception:
        STATE["error"] = traceback.format_exc()
        STATE["phase"] = "failed"
        print(STATE["error"], flush=True)


def _serve() -> None:
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            body = json.dumps(
                {**STATE, "elapsed": round(time.time() - STATE["started"], 1)},
                ensure_ascii=False,
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # silencio: los logs útiles son los del perfilado
            return

    # El gateway de Salad entra por IPv6: hay que escuchar en dual-stack.
    import socket

    class DualStack(HTTPServer):
        address_family = socket.AF_INET6

    DualStack(("::", PORT), Handler).serve_forever()


def main() -> int:
    threading.Thread(target=_serve, daemon=True).start()
    print(f"[gpu-probe] HTTP en :{PORT}, TTL {TTL:.0f}s", flush=True)
    threading.Thread(target=_run_profile, daemon=True).start()

    # Plazo de muerte absoluto desde el arranque. Se cumple aunque el perfilado
    # siga corriendo: un perfilado que no termina en el TTL es un perfilado roto,
    # y prefiero perder la medida a dejar una GPU facturando.
    while time.time() - STATE["started"] < TTL:
        time.sleep(10)
    print(f"[gpu-probe] TTL de {TTL:.0f}s agotado; salgo para dejar de facturar", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
