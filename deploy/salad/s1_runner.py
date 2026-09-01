"""S1 AUTOCONTENIDO — strict vs ragged EN SERVICIO, en GPU real de Salad.

Inyecta el runtime mycellios (tarball base64 en env PAYLOAD_*), arranca el server
de inferencia distribuida REAL en GPU, y mide el throughput agregado open-loop
con el grouping ESTRICTO (por defecto) vs RAGGED (`GDLP_RAGGED_GROUPING=1`) +
ventana adaptativa. Es el número que convierte el e(8)=7,82 (kernel) en tok/s de
servicio real. Observable (dual-stack IPv6) y con teardown por el orquestador.
"""
import base64, io, json, os, socket, statistics, subprocess, sys, tarfile, threading, time, traceback, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATE = {"status": "booting", "phase": "http-up", "started_ts": time.time(), "error": None}
RESULTS = {}
_LOG, _LK = [], threading.Lock()


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    with _LK:
        _LOG.append(line)
        if len(_LOG) > 8000:
            del _LOG[:2000]
    print(line, flush=True)


class H(BaseHTTPRequestHandler):
    def _s(self, c, p, ct="application/json"):
        b = p if isinstance(p, bytes) else json.dumps(p).encode()
        self.send_response(c); self.send_header("Content-Type", ct); self.send_header("Content-Length", str(len(b))); self.end_headers()
        try: self.wfile.write(b)
        except Exception: pass

    def do_GET(self):
        p = self.path.split("?")[0]
        if p in ("/", "/status", "/health", "/healthz"): self._s(200, {**STATE, "log_lines": len(_LOG)})
        elif p in ("/results", "/results.json"): self._s(200, {"status": STATE["status"], "results": RESULTS})
        elif p in ("/log", "/logs"):
            with _LK: t = "\n".join(_LOG[-3000:])
            self._s(200, t.encode(), "text/plain; charset=utf-8")
        else: self._s(404, {"error": "not found"})

    def log_message(self, *a): pass


class _DS(ThreadingHTTPServer):
    address_family = socket.AF_INET6
    def server_bind(self):
        try: self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        except OSError: pass
        super().server_bind()


def serve():
    _DS(("::", int(os.environ.get("GDLP_PORT", "8000"))), H).serve_forever()


def sh(cmd, **kw):
    log("$ " + " ".join(cmd) if isinstance(cmd, list) else "$ " + cmd)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, **kw)
    for ln in p.stdout:
        log(ln.rstrip())
    return p.wait()


def extract_payload():
    STATE["phase"] = "extract-runtime"
    n = int(os.environ["PAYLOAD_PARTS"])
    b = base64.b64decode("".join(os.environ[f"PAYLOAD_{i}"] for i in range(n)))
    os.makedirs("/opt/mycellios", exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(b), mode="r:gz") as t:
        t.extractall("/opt/mycellios")
    log(f"runtime extraído ({len(b)//1024} KB) → /opt/mycellios")


def wait_health(port, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if b'"status": "ready"' in r.read():
                    return True
        except Exception:
            pass
        time.sleep(3)
    return False


def run_arm(ragged: bool, model: str, boundaries: str, window_ms: float, lam: float, dur: int, port: int) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = "/opt/mycellios"
    env["HF_HOME"] = "/opt/hf"
    if ragged:
        env["GDLP_RAGGED_GROUPING"] = "1"
    else:
        env.pop("GDLP_RAGGED_GROUPING", None)
    srv = subprocess.Popen(
        [sys.executable, "-m", "distributed_runtime.server", "--model", model, "--stages", "2",
         "--boundaries", boundaries, "--threads-per-stage", "2", "--codec", "fp16", "--device", "cuda",
         "--max-batch-size", "8", "--max-active-sequences", "8", "--prefill-chunk-tokens", "128",
         "--no-speculation-probes", "--root-batch-window-ms", str(window_ms), "--port", str(port)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
    reader = threading.Thread(target=lambda: [log(f"srv| {l.rstrip()}") for l in srv.stdout], daemon=True)
    reader.start()
    tag = "ragged" if ragged else "strict"
    try:
        if not wait_health(port):
            return {"arm": tag, "error": "server no ready"}
        log(f"[{tag}] server listo; warmup closed C=8 + open-loop λ={lam}")
        sh([sys.executable, "-m", "distributed_runtime.api_benchmark", "--base-url", f"http://127.0.0.1:{port}",
            "--mode", "closed", "--concurrencies", "8", "--iterations", "2", "--warmups", "1", "--output-tokens", "16"],
           env=env)
        out = f"/tmp/open_{tag}.json"
        sh([sys.executable, "-m", "distributed_runtime.api_benchmark", "--base-url", f"http://127.0.0.1:{port}",
            "--mode", "open", "--lambda-rps", str(lam), "--duration-seconds", str(dur), "--warmup-seconds", "10",
            "--output-tokens", "16", "--seed", "7", "--prewarm-connections", "16", "--json-out", out], env=env)
        d = json.load(open(out))
        r = d["rows"][0]; ss = r["steady_state"]
        hb = (d.get("server_health_before") or {}).get("root_batching") or {}
        ha = (d.get("server_after_measurement") or {}).get("root_batching") or {}
        fwd = (ha.get("model_forward_calls", 0) - hb.get("model_forward_calls", 0)) or 1
        items = ha.get("ready_items", 0) - hb.get("ready_items", 0)
        return {"arm": tag, "agg_tok_s": ss["aggregate_completion_tokens_per_second"],
                "b_eff": round(items / fwd, 3), "errors": r["errors"], "offered": r["offered_requests"],
                "inflight": round(ss.get("average_inflight_requests", 0), 2)}
    finally:
        srv.terminate()
        try: srv.wait(timeout=15)
        except Exception: srv.kill()
        time.sleep(3)


def run():
    threading.Thread(target=serve, daemon=True).start()
    log("HTTP status arriba (dual-stack)")
    try:
        STATE["status"] = "running"; STATE["phase"] = "pip"
        sh([sys.executable, "-m", "pip", "install", "--no-cache-dir", "transformers==5.14.1", "accelerate==1.14.0",
            "safetensors==0.8.0", "numpy==1.26.4", "sentencepiece==0.2.2", "aiohttp==3.14.1"])
        extract_payload()
        import torch  # noqa
        model = os.environ.get("GDLP_MODEL", "Qwen/Qwen3-0.6B")
        from transformers import AutoConfig
        nl = AutoConfig.from_pretrained(model).num_hidden_layers
        mid = nl // 2
        boundaries = f"0,{mid},{nl}"
        RESULTS["config"] = {"model": model, "layers": nl, "boundaries": boundaries, "gpu": torch.cuda.get_device_name(0)}
        log(f"modelo {model} ({nl} capas → boundaries {boundaries}) en {torch.cuda.get_device_name(0)}")
        lam = float(os.environ.get("GDLP_LAMBDA", "3.0"))
        dur = int(os.environ.get("GDLP_DURATION", "60"))
        win = float(os.environ.get("GDLP_WINDOW_MS", "50"))
        STATE["phase"] = "A/B intercalado"
        # A/B intercalado: strict, ragged, strict, ragged
        arms = []
        for i, ragged in enumerate((False, True, False, True)):
            log(f"=== brazo {i} {'RAGGED' if ragged else 'STRICT'} ===")
            arms.append(run_arm(ragged, model, boundaries, win, lam, dur, 8081))
            RESULTS["arms"] = arms
        # síntesis
        strict = [a["agg_tok_s"] for a in arms if a.get("arm") == "strict" and "agg_tok_s" in a]
        ragged = [a["agg_tok_s"] for a in arms if a.get("arm") == "ragged" and "agg_tok_s" in a]
        if strict and ragged:
            RESULTS["summary"] = {
                "strict_agg_median": round(statistics.median(strict), 2),
                "ragged_agg_median": round(statistics.median(ragged), 2),
                "ragged_over_strict": round(statistics.median(ragged) / statistics.median(strict), 3),
                "strict_beff": statistics.median([a["b_eff"] for a in arms if a.get("arm") == "strict"]),
                "ragged_beff": statistics.median([a["b_eff"] for a in arms if a.get("arm") == "ragged"]),
            }
            log(f"RESUMEN: {RESULTS['summary']}")
        STATE["status"] = "done"; STATE["done_ts"] = time.time()
        log("S1 COMPLETADO")
    except BaseException as e:
        STATE["status"] = "error"; STATE["error"] = f"{type(e).__name__}: {e}"
        log("ERROR:\n" + traceback.format_exc())
    while True:
        time.sleep(30)


if __name__ == "__main__":
    run()
