"""Agente de benchmark CALIENTE para SaladCloud — se queda vivo y ejecuta muchos
experimentos sin re-arrancar (invierte el peaje de arranque UNA vez).

Arranque (una vez por sesión): servidor HTTP observable (dual-stack IPv6) →
pip install → extraer runtime mycellios (PAYLOAD_* en env) → pre-cargar modelo.
Luego SIRVE trabajos por HTTP y NO muere:
  GET  /status /log /ready
  GET  /run?job=probe|s1&<params>   → ejecuta y devuelve JSON con el resultado
El orquestador lanza el agente una vez, le manda N trabajos, y hace teardown al
final de la sesión. Cada trabajo tras el primero es casi instantáneo (todo cacheado).
"""
import base64, io, json, os, socket, statistics, subprocess, sys, tarfile, threading, time, traceback, urllib.parse, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATE = {"status": "booting", "phase": "http-up", "started_ts": time.time(), "error": None, "ready": False, "jobs_done": 0}
_LOG, _LK = [], threading.Lock()
_JOB_LK = threading.Lock()  # un trabajo GPU a la vez


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    with _LK:
        _LOG.append(line)
        if len(_LOG) > 12000:
            del _LOG[:3000]
    print(line, flush=True)


def sh(cmd, env=None):
    log("$ " + " ".join(cmd))
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
    for ln in p.stdout:
        log(ln.rstrip())
    return p.wait()


# ------------------------- trabajos -------------------------

def job_probe(params):
    """Calibración e(B)/exactitud/ruido en el modelo dado (reusa el proceso caliente)."""
    import torch, hashlib
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.cache_utils import DynamicCache
    name = params.get("model", "Qwen/Qwen3-0.6B")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if dev == "cuda" else torch.float32
    tok, model = _load(name, dtype, dev)
    out = {"model": name, "device": dev}

    @torch.inference_mode()
    def curve(prefill=48, reps=15):
        r = {}
        for B in (1, 2, 4, 8):
            ids = torch.randint(0, 1000, (B, prefill), device=dev); nt = torch.randint(0, 1000, (B, 1), device=dev)
            for _ in range(3):
                c = DynamicCache(); model(input_ids=ids, past_key_values=c, use_cache=True); model(input_ids=nt, past_key_values=c, use_cache=True)
            ts = []
            for _ in range(reps):
                c = DynamicCache(); model(input_ids=ids, past_key_values=c, use_cache=True)
                if dev == "cuda": torch.cuda.synchronize()
                t0 = time.perf_counter(); model(input_ids=nt, past_key_values=c, use_cache=True)
                if dev == "cuda": torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1000)
            r[B] = statistics.median(ts)
        return r
    cv = curve(); out["cost_curve_ms"] = cv; out["e_of_B"] = {B: round(B * cv[1] / cv[B], 3) for B in cv}
    return out


def job_s1(params):
    """strict vs ragged en servicio real (server + api_benchmark), A/B intercalado."""
    import torch
    from transformers import AutoConfig
    name = params.get("model", "Qwen/Qwen3-0.6B")
    nl = AutoConfig.from_pretrained(name).num_hidden_layers
    boundaries = f"0,{nl // 2},{nl}"
    lam = float(params.get("lambda", "3.0")); dur = int(params.get("dur", "60")); win = float(params.get("window", "50"))
    arms = []
    for i, ragged in enumerate((False, True, False, True)):
        arms.append(_s1_arm(ragged, name, boundaries, win, lam, dur, 8081))
    strict = [a["agg"] for a in arms if a["arm"] == "strict" and "agg" in a]
    ragged = [a["agg"] for a in arms if a["arm"] == "ragged" and "agg" in a]
    summary = {}
    if strict and ragged:
        summary = {"strict_agg": round(statistics.median(strict), 2), "ragged_agg": round(statistics.median(ragged), 2),
                   "ragged_over_strict": round(statistics.median(ragged) / statistics.median(strict), 3),
                   "strict_beff": statistics.median([a["beff"] for a in arms if a["arm"] == "strict"]),
                   "ragged_beff": statistics.median([a["beff"] for a in arms if a["arm"] == "ragged"])}
    return {"model": name, "boundaries": boundaries, "lambda": lam, "arms": arms, "summary": summary}


def _wait_health(port, timeout=180):
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


def _s1_arm(ragged, model, boundaries, win, lam, dur, port):
    env = dict(os.environ); env["PYTHONPATH"] = "/opt/mycellios"; env["HF_HOME"] = "/opt/hf"
    if ragged: env["GDLP_RAGGED_GROUPING"] = "1"
    else: env.pop("GDLP_RAGGED_GROUPING", None)
    srv = subprocess.Popen([sys.executable, "-m", "distributed_runtime.server", "--model", model, "--stages", "2",
        "--boundaries", boundaries, "--threads-per-stage", "2", "--codec", "fp16", "--device", "cuda",
        "--max-batch-size", "8", "--max-active-sequences", "8", "--prefill-chunk-tokens", "128",
        "--no-speculation-probes", "--root-batch-window-ms", str(win), "--port", str(port)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
    threading.Thread(target=lambda: [log(f"srv| {l.rstrip()}") for l in srv.stdout], daemon=True).start()
    tag = "ragged" if ragged else "strict"
    try:
        if not _wait_health(port):
            return {"arm": tag, "error": "no ready"}
        sh([sys.executable, "-m", "distributed_runtime.api_benchmark", "--base-url", f"http://127.0.0.1:{port}",
            "--mode", "closed", "--concurrencies", "8", "--iterations", "2", "--warmups", "1", "--output-tokens", "16"], env=env)
        of = f"/tmp/o_{tag}.json"
        sh([sys.executable, "-m", "distributed_runtime.api_benchmark", "--base-url", f"http://127.0.0.1:{port}",
            "--mode", "open", "--lambda-rps", str(lam), "--duration-seconds", str(dur), "--warmup-seconds", "10",
            "--output-tokens", "16", "--seed", "7", "--prewarm-connections", "16", "--json-out", of], env=env)
        d = json.load(open(of)); r = d["rows"][0]; ss = r["steady_state"]
        hb = (d.get("server_health_before") or {}).get("root_batching") or {}
        ha = (d.get("server_after_measurement") or {}).get("root_batching") or {}
        fwd = (ha.get("model_forward_calls", 0) - hb.get("model_forward_calls", 0)) or 1
        items = ha.get("ready_items", 0) - hb.get("ready_items", 0)
        return {"arm": tag, "agg": ss["aggregate_completion_tokens_per_second"], "beff": round(items / fwd, 3),
                "errors": r["errors"], "inflight": round(ss.get("average_inflight_requests", 0), 2)}
    finally:
        srv.terminate()
        try: srv.wait(timeout=15)
        except Exception: srv.kill()
        time.sleep(3)


_MODELS = {}


def _load(name, dtype, dev):
    if name not in _MODELS:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        log(f"cargando {name} en {dev} …")
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(dev).eval()
        _MODELS[name] = (tok, model)
    return _MODELS[name]


JOBS = {"probe": job_probe, "s1": job_s1}

# --- jobs ASÍNCRONOS: el gateway (Cloudflare) corta peticiones HTTP a ~100 s,
# así que un trabajo largo (s1 ~7 min) se lanza en segundo plano y el cliente
# pregunta por el resultado con /job?id=… ---
_JOB_STATE = {}
_JOB_SEQ = [0]


def _launch(job, params):
    _JOB_SEQ[0] += 1
    jid = f"{job}-{_JOB_SEQ[0]}-{int(time.time())}"
    _JOB_STATE[jid] = {"status": "running", "job": job, "params": params, "result": None, "error": None, "started": time.time()}

    def worker():
        with _JOB_LK:  # un trabajo GPU a la vez
            log(f"=== JOB {jid} params={params} ===")
            try:
                res = JOBS[job](params)
                _JOB_STATE[jid].update(status="done", result=res)
                STATE["jobs_done"] += 1
                log(f"JOB {jid} OK: {res.get('summary') or res.get('e_of_B')}")
            except BaseException as e:
                _JOB_STATE[jid].update(status="error", error=f"{type(e).__name__}: {e}")
                log(f"JOB {jid} ERROR:\n" + traceback.format_exc())

    threading.Thread(target=worker, daemon=True).start()
    return jid


# ------------------------- HTTP -------------------------

class H(BaseHTTPRequestHandler):
    def _s(self, c, p, ct="application/json"):
        b = p if isinstance(p, bytes) else json.dumps(p).encode()
        self.send_response(c); self.send_header("Content-Type", ct); self.send_header("Content-Length", str(len(b))); self.end_headers()
        try: self.wfile.write(b)
        except Exception: pass

    def do_GET(self):
        u = urllib.parse.urlparse(self.path); p = u.path; q = dict(urllib.parse.parse_qsl(u.query))
        if p in ("/", "/status", "/health", "/healthz"): self._s(200, {**STATE, "log_lines": len(_LOG)})
        elif p == "/ready": self._s(200 if STATE["ready"] else 503, {"ready": STATE["ready"]})
        elif p in ("/log", "/logs"):
            with _LK: t = "\n".join(_LOG[-3000:])
            self._s(200, t.encode(), "text/plain; charset=utf-8")
        elif p == "/run":
            # ASÍNCRONO: lanza y devuelve el id al instante (evita el timeout ~100s del gateway).
            job = q.pop("job", None)
            if job not in JOBS: self._s(400, {"error": f"job desconocido {job}"}); return
            jid = _launch(job, q)
            self._s(200, {"ok": True, "job_id": jid, "status": "running"})
        elif p == "/job":
            jid = q.get("id")
            if jid not in _JOB_STATE: self._s(404, {"error": "job_id desconocido"}); return
            self._s(200, _JOB_STATE[jid])
        else: self._s(404, {"error": "not found"})

    def log_message(self, *a): pass


class _DS(ThreadingHTTPServer):
    address_family = socket.AF_INET6
    daemon_threads = True
    def server_bind(self):
        try: self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        except OSError: pass
        super().server_bind()


def _serve():
    _DS(("::", int(os.environ.get("GDLP_PORT", "8000"))), H).serve_forever()


def _extract_payload():
    n = int(os.environ["PAYLOAD_PARTS"])
    b = base64.b64decode("".join(os.environ[f"PAYLOAD_{i}"] for i in range(n)))
    os.makedirs("/opt/mycellios", exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(b), mode="r:gz") as t:
        t.extractall("/opt/mycellios")
    log(f"runtime extraído ({len(b)//1024} KB)")


def main():
    threading.Thread(target=_serve, daemon=True).start()
    log("agente arriba (dual-stack); preparando entorno…")
    try:
        STATE["status"] = "preparing"; STATE["phase"] = "pip"
        sh([sys.executable, "-m", "pip", "install", "--no-cache-dir", "transformers==5.14.1", "accelerate==1.14.0",
            "safetensors==0.8.0", "numpy==1.26.4", "sentencepiece==0.2.2", "aiohttp==3.14.1"])
        STATE["phase"] = "extract"; _extract_payload()
        os.environ["PYTHONPATH"] = "/opt/mycellios"; os.environ["HF_HOME"] = "/opt/hf"
        import torch
        STATE["phase"] = "preload-model"
        m = os.environ.get("GDLP_MODEL", "Qwen/Qwen3-0.6B")
        _load(m, torch.float16 if torch.cuda.is_available() else torch.float32, "cuda" if torch.cuda.is_available() else "cpu")
        STATE["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        STATE["ready"] = True; STATE["status"] = "ready"
        log(f"AGENTE LISTO en {STATE['gpu']}. Esperando trabajos en /run?job=…")
    except BaseException as e:
        STATE["status"] = "error"; STATE["error"] = f"{type(e).__name__}: {e}"
        log("ERROR arranque:\n" + traceback.format_exc())
    while True:
        time.sleep(30)


if __name__ == "__main__":
    main()
