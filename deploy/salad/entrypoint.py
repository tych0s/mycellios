"""Entrypoint OBSERVABLE para contenedores mycellios en SaladCloud.

LECCIÓN nº2 (fallo del 24-07): el nodo estuvo 8 min en 503 sin visibilidad
porque el servidor HTTP arrancaba DESPUÉS del trabajo pesado. Aquí el servidor
de estado se levanta en el segundo 0 (solo stdlib, sin imports pesados), captura
todo el stdout/stderr, y solo entonces ejecuta el modo. Así el gateway SIEMPRE
responde y `/status`, `/log`, `/results.json` cuentan qué pasa — incluido un
fallo de importación o de descarga.

Modos (env GDLP_MODE):
  probe        — calibración S0 autocontenida (huella, timers, exactitud, e(B), ruido).
  server       — servidor de inferencia distribuida de un nodo (para S1/S4).
  worker       — worker que DIALA A UN COORDINADOR (GDLP_COORDINATOR_URL) — S2/S3.
                 (Salad solo acepta ingress por gateway; outbound es libre → el
                  nodo marca hacia fuera, nunca escucha TCP entrante.)
Config común por env: GDLP_MODEL, GDLP_ARGS (JSON con overrides), y las que cada modo documente.
Todos los artefactos se sirven por el gateway y el orquestador los recoge y empuja al repo.
"""
from __future__ import annotations

import io
import json
import os
import socket
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ----------------------------------------------------------------------------
# Estado global observable (se sirve por HTTP desde el segundo 0).
# ----------------------------------------------------------------------------
STATE: dict = {
    "status": "booting",
    "mode": os.environ.get("GDLP_MODE", "probe"),
    "phase": "start",
    "started_ts": time.time(),
    "error": None,
}
_LOG: list[str] = []
_LOG_LOCK = threading.Lock()
RESULTS: dict = {}


def log(message: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {message}"
    with _LOG_LOCK:
        _LOG.append(line)
        if len(_LOG) > 5000:
            del _LOG[:1000]
    print(line, flush=True)
    STATE["ts"] = time.time()


class _Tee(io.TextIOBase):
    """Duplica lo escrito a stdout real y al buffer observable."""

    def __init__(self, real):
        self._real = real

    def write(self, s):  # noqa: D401
        try:
            self._real.write(s)
            self._real.flush()
        except Exception:
            pass
        if s and s.strip():
            with _LOG_LOCK:
                _LOG.append(s.rstrip("\n"))
        return len(s)


class _Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload, content_type="application/json"):
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except Exception:
            pass

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/status", "/health", "/healthz"):
            self._send(200, {**STATE, "log_lines": len(_LOG)})
        elif path in ("/results", "/results.json"):
            self._send(200, {"status": STATE["status"], "results": RESULTS})
        elif path in ("/log", "/logs"):
            with _LOG_LOCK:
                text = "\n".join(_LOG[-2000:])
            self._send(200, text.encode(), content_type="text/plain; charset=utf-8")
        else:
            self._send(404, {"error": "not found", "paths": ["/status", "/results.json", "/log"]})

    def log_message(self, *args):  # silencio el logging por request
        pass


class _DualStackServer(ThreadingHTTPServer):
    # El gateway de Salad conecta por IPv6; dual-stack (:: acepta IPv4+IPv6).
    address_family = socket.AF_INET6

    def server_bind(self):
        try:
            self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        except OSError:
            pass
        super().server_bind()


def _serve():
    port = int(os.environ.get("GDLP_PORT", "8000"))
    _DualStackServer(("::", port), _Handler).serve_forever()


def _run_mode() -> None:
    """Ejecuta el modo pedido; cualquier excepción queda en STATE['error']."""
    mode = STATE["mode"]
    try:
        STATE["status"] = "running"
        if mode == "probe":
            from distributed_runtime.salad_probe import run_probe  # inyectada por la imagen

            STATE["phase"] = "probe"
            run_probe(STATE, RESULTS, log)
        elif mode == "server":
            STATE["phase"] = "server"
            _run_server()
        elif mode == "worker":
            STATE["phase"] = "worker"
            _run_worker()
        else:
            raise ValueError(f"GDLP_MODE desconocido: {mode!r}")
        STATE["status"] = "done"
        STATE["done_ts"] = time.time()
        log(f"modo {mode} COMPLETADO")
    except BaseException as exc:  # noqa: BLE001 — queremos verlo por HTTP, no morir mudo
        STATE["status"] = "error"
        STATE["error"] = f"{type(exc).__name__}: {exc}"
        log("ERROR:\n" + traceback.format_exc())


def _run_server() -> None:
    """Arranca distributed_runtime.server con los args de GDLP_ARGS (JSON list).
    El server sirve su propio OpenAI-compatible en otro puerto; este entrypoint
    sigue reportando estado en GDLP_PORT. Placeholder para S1/S4 — la orquestación
    concreta (qué puerto expone el gateway) se fija al cablear esas fases."""
    import runpy
    import sys

    argv = json.loads(os.environ.get("GDLP_ARGS", "[]"))
    log(f"lanzando server con argv={argv}")
    sys.argv = ["distributed_runtime.server", *argv]
    runpy.run_module("distributed_runtime.server", run_name="__main__")


def _run_worker() -> None:
    """Worker dial-out. IMPORTANTE (mapeo de federación 24-07): el rol que marca
    hacia fuera y tunela el pipeline por un único WSS es el `WorkerAgent` de Node
    (`src/worker/agent.ts`), NO un CLI de Python. Requiere la Imagen B (Node+Python)
    y un `container-worker` headless que cablee WorkerAgent+distributedExecutor
    desde env — ver DISENO_INTEGRACION_SALAD_MYCELLIOS.md §1.1. Este modo Python
    queda deshabilitado a propósito para no fingir una conexión que no le toca."""
    url = os.environ.get("GDLP_COORDINATOR_URL")
    log(f"worker: el dial-out real es el WorkerAgent de Node (Imagen B), no este entrypoint Python. coordinador={url}")
    STATE["phase"] = "worker-needs-image-B"
    raise NotImplementedError(
        "worker dial-out = WorkerAgent (Node) en Imagen B; ver §1.1 del diseño. Este es el entrypoint Python (Imagen A)."
    )


def main() -> None:
    import sys

    sys.stdout = _Tee(sys.__stdout__)
    sys.stderr = _Tee(sys.__stderr__)
    threading.Thread(target=_serve, daemon=True).start()
    log(f"entrypoint arriba; modo={STATE['mode']}; sirviendo status en :{os.environ.get('GDLP_PORT', '8000')}")
    STATE["status"] = "starting-mode"
    worker = threading.Thread(target=_run_mode, daemon=True)
    worker.start()
    # El proceso vive mientras sirve estado (el orquestador hace teardown).
    while True:
        time.sleep(30)


if __name__ == "__main__":
    main()
