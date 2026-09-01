"""Agente de un nodo-etapa GPU del enjambre Salad (root o stage), CALIENTE.

Arranca UNA vez (pip + extrae runtime + escribe bridge + precarga modelo a cache)
y queda en "booted" ESPERANDO. El orquestador, cuando TODOS los nodos están
booted, hace POST /start ~a la vez en todos (BARRERA) con la config del run
(relay, endpoints del puente, flags de mycellios). Así:
  - se elimina el desfase de arranque (los WS del túnel no quedan ociosos
    esperando a un nodo lento -> el gateway los cierra a los ~min);
  - se puede /stop + /start con OTRA config sin redesplegar (runs rápidos).

Puerto gateway (8000): SIEMPRE observabilidad (/status,/log,/ready), control
(/start,/stop), benchmark on-node (/run,/job) y, para el root, proxy hacia el
server interno (127.0.0.1:INTERNAL) para /health y /v1/*. Dual-stack IPv6.
"""
import base64, io, json, os, socket, subprocess, sys, tarfile, threading, time, traceback
import urllib.request, urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

INTERNAL = 8071
GW_PORT = int(os.environ.get("GDLP_PORT", "8000"))
_LOG, _LK = [], threading.Lock()
_START_LK = threading.Lock()
_PROCS = {"bridge": None, "mycellios": None}
_CFG = {"role": os.environ.get("GDLP_ROLE", "stage")}
STATE = {"status": "booting", "phase": None, "booted": False, "ready": False,
         "started": time.time(), "gpu": None, "error": None, "role": _CFG["role"]}


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    with _LK:
        _LOG.append(line)
        if len(_LOG) > 5000:
            del _LOG[:1200]
    print(line, flush=True)


def sh(cmd, **kw):
    log("$ " + " ".join(cmd))
    return subprocess.run(cmd, check=True, **kw)


def _pipe_reader(proc, tag):
    for line in proc.stdout:
        log(f"{tag}| {line.rstrip()}")
        if tag == "stage" and "stage_ready" in line:
            STATE["mycellios_up"] = True


# --------------------- proxy (root) ---------------------

def _proxy(method, path, body, headers):
    req = urllib.request.Request(f"http://127.0.0.1:{INTERNAL}{path}", data=body, method=method)
    for k, v in headers.items():
        if k.lower() in ("content-type", "accept"):
            req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.status, r.read(), r.headers.get("Content-Type", "application/json")


# --------------------- benchmark on-node ---------------------
_JOB_STATE, _JOB_SEQ, _JOB_LK = {}, [0], threading.Lock()


def job_bench(params):
    mode = params.get("mode", "open")
    out = "/tmp/bench.json"
    base = f"http://127.0.0.1:{INTERNAL}"
    cmd = [sys.executable, "-m", "distributed_runtime.api_benchmark", "--base-url", base,
           "--mode", mode, "--output-tokens", params.get("tokens", "16"), "--seed", "7", "--json-out", out]
    if mode == "open":
        cmd += ["--lambda-rps", params.get("lam", "3"), "--duration-seconds", params.get("dur", "30"),
                "--warmup-seconds", params.get("warmup", "8"), "--prewarm-connections", "16"]
    else:
        cmd += ["--concurrencies", params.get("conc", "8"), "--iterations", params.get("iters", "3"), "--warmups", "1"]
    env = dict(os.environ); env["PYTHONPATH"] = "/opt/mycellios"; env["HF_HOME"] = "/opt/hf"
    log("$ " + " ".join(cmd))
    r = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for ln in (r.stdout or "").splitlines()[-40:]:
        log("bench| " + ln)
    d = json.load(open(out))
    # B_eff = secuencias fusionadas por forward. DELATA si el grouping ragged está
    # ACTIVO de verdad (strict ~1.0; ragged > 1). Sin esto, un flag que no llega al
    # proceso daría "ragged == strict" trivialmente.
    hb = (d.get("server_health_before") or {}).get("root_batching") or {}
    ha = (d.get("server_after_measurement") or {}).get("root_batching") or {}
    fwd = (ha.get("model_forward_calls", 0) - hb.get("model_forward_calls", 0)) or 1
    items = ha.get("ready_items", 0) - hb.get("ready_items", 0)
    beff = round(items / fwd, 3)
    summary = [{"lambda": row.get("lambda_rps"),
                "agg_tok_s": row.get("steady_state", {}).get("aggregate_completion_tokens_per_second"),
                "inflight": row.get("steady_state", {}).get("average_inflight_requests"),
                "errors": row.get("errors")} for row in d.get("rows", [])]
    return {"mode": mode, "params": params, "summary": summary, "beff": beff,
            "ragged_requested": _CFG.get("ragged"), "my_args": _CFG.get("my_args"), "full": d}


JOBS = {"bench": job_bench}


def _launch_job(job, params):
    _JOB_SEQ[0] += 1
    jid = f"{job}-{_JOB_SEQ[0]}"
    _JOB_STATE[jid] = {"status": "running", "job": job, "params": params, "result": None, "error": None}

    def worker():
        with _JOB_LK:
            log(f"=== JOB {jid} params={params} ===")
            try:
                _JOB_STATE[jid].update(status="done", result=JOBS[job](params))
                log(f"JOB {jid} OK")
            except BaseException as e:
                _JOB_STATE[jid].update(status="error", error=f"{type(e).__name__}: {e}")
                log(f"JOB {jid} ERROR:\n" + traceback.format_exc())

    threading.Thread(target=worker, daemon=True).start()
    return jid


# --------------------- start / stop (barrera + caliente) ---------------------

def _start(cfg):
    if _PROCS["mycellios"] is not None or _PROCS["bridge"] is not None:
        # idempotente: reinicia limpio en vez de dejar el nodo sucio (un proceso
        # viejo reteniendo el puerto -> "Address already in use" en la etapa).
        log("START sobre procesos vivos -> _stop() previo")
        _stop()
    with _START_LK:
        if not STATE["booted"]:
            return {"error": "aún no booted"}
        _CFG.update(cfg)
        role = cfg.get("role", _CFG["role"])
        STATE["role"] = role
        STATE["mycellios_up"] = False
        STATE["ready"] = False
        STATE["status"] = "starting"
        env = dict(os.environ)
        env["PYTHONPATH"] = "/opt/mycellios"; env["HF_HOME"] = "/opt/hf"
        env["PYTHONUNBUFFERED"] = "1"; env["GDLP_SKIP_PIP"] = "1"
        env["RELAY_URL"] = cfg["relay_url"]
        env["GDLP_BRIDGE_ENDPOINTS"] = json.dumps(cfg["bridge_endpoints"])
        if cfg.get("ragged"):
            env["GDLP_RAGGED_GROUPING"] = "1"
        else:
            env.pop("GDLP_RAGGED_GROUPING", None)
        # 1) puente
        log(f"START rol={role} endpoints={json.dumps(cfg['bridge_endpoints'])}")
        b = subprocess.Popen([sys.executable, "/opt/bridge.py"], env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        _PROCS["bridge"] = b
        threading.Thread(target=_pipe_reader, args=(b, "bridge"), daemon=True).start()
        time.sleep(2)
        # 2) mycellios
        mod = "distributed_runtime.server" if role == "root" else "distributed_runtime.stage_cli"
        cmd = [sys.executable, "-m", mod] + cfg["my_args"]
        m = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        _PROCS["mycellios"] = m
        threading.Thread(target=_pipe_reader, args=(m, "srv" if role == "root" else "stage"), daemon=True).start()
        # 3) readiness en un hilo (no bloquea la respuesta HTTP)
        threading.Thread(target=_await_ready, args=(role,), daemon=True).start()
        return {"ok": True, "role": role, "status": "starting"}


def _await_ready(role):
    t0 = time.time()
    if role == "root":
        while time.time() - t0 < 420:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{INTERNAL}/health", timeout=5) as r:
                    if b'"status": "ready"' in r.read():
                        STATE["mycellios_up"] = True
                        break
            except Exception:
                pass
            time.sleep(3)
    else:
        while time.time() - t0 < 420 and not STATE.get("mycellios_up"):
            time.sleep(2)
    if STATE.get("mycellios_up"):
        STATE["ready"] = True; STATE["status"] = "ready"; log(f"NODO READY (rol={role})")
    else:
        STATE["status"] = "error"; STATE["error"] = "mycellios no llegó a ready"; log("ERROR: no ready")


def _stop():
    with _START_LK:
        for k in ("mycellios", "bridge"):
            p = _PROCS.get(k)
            if p is not None:
                try:
                    p.terminate(); p.wait(timeout=8)
                except Exception:
                    try: p.kill()
                    except Exception: pass
                _PROCS[k] = None
        STATE["ready"] = False; STATE["mycellios_up"] = False; STATE["status"] = "booted"
        log("STOP: mycellios+puente parados; vuelto a booted")
        return {"ok": True, "status": "booted"}


# --------------------- HTTP ---------------------

class H(BaseHTTPRequestHandler):
    def _s(self, code, payload, ct="application/json"):
        b = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(code); self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(b))); self.end_headers()
        try: self.wfile.write(b)
        except Exception: pass

    def _obs(self, p):
        if p in ("/status", "/", "/healthz"):
            self._s(200, {**STATE, "log_lines": len(_LOG)}); return True
        if p == "/ready":
            self._s(200 if STATE["ready"] else 503, {"ready": STATE["ready"], "status": STATE["status"]}); return True
        if p == "/booted":
            self._s(200 if STATE["booted"] else 503, {"booted": STATE["booted"], "status": STATE["status"]}); return True
        if p in ("/log", "/logs"):
            with _LK: t = "\n".join(_LOG[-3500:])
            self._s(200, t.encode(), "text/plain; charset=utf-8"); return True
        return False

    def do_GET(self):
        u = urllib.parse.urlparse(self.path); p = u.path
        if self._obs(p):
            return
        if p == "/run":
            q = dict(urllib.parse.parse_qsl(u.query)); job = q.pop("job", None)
            if job not in JOBS: self._s(400, {"error": f"job desconocido {job}"}); return
            self._s(200, {"ok": True, "job_id": _launch_job(job, q), "status": "running"}); return
        if p == "/job":
            q = dict(urllib.parse.parse_qsl(u.query)); jid = q.get("id")
            if jid not in _JOB_STATE: self._s(404, {"error": "job_id desconocido"}); return
            self._s(200, _JOB_STATE[jid]); return
        if STATE["role"] == "root" and STATE["ready"]:
            try:
                st, b, ct = _proxy("GET", self.path, None, dict(self.headers)); self._s(st, b, ct)
            except Exception as e:
                self._s(503, {"error": f"proxy: {e}"})
        else:
            self._s(404, {"error": "no disponible (stage o no-ready)"})

    def do_POST(self):
        u = urllib.parse.urlparse(self.path); p = u.path
        n = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(n) if n else b""
        if p == "/start":
            try:
                cfg = json.loads(raw or b"{}")
                self._s(200, _start(cfg))
            except Exception as e:
                self._s(400, {"error": f"{type(e).__name__}: {e}"})
            return
        if p == "/stop":
            self._s(200, _stop()); return
        if STATE["role"] == "root" and STATE["ready"]:
            try:
                st, b, ct = _proxy("POST", self.path, raw, dict(self.headers)); self._s(st, b, ct)
            except Exception as e:
                self._s(503, {"error": f"proxy: {e}"})
        else:
            self._s(404, {"error": "no disponible"})

    def log_message(self, *a):
        pass


class _DS(ThreadingHTTPServer):
    address_family = socket.AF_INET6
    daemon_threads = True

    def server_bind(self):
        try: self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        except OSError: pass
        super().server_bind()


def _serve():
    _DS(("::", GW_PORT), H).serve_forever()


# --------------------- boot ---------------------

def _extract_payload():
    n = int(os.environ["PAYLOAD_PARTS"])
    b = base64.b64decode("".join(os.environ[f"PAYLOAD_{i}"] for i in range(n)))
    os.makedirs("/opt/mycellios", exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(b), mode="r:gz") as t:
        t.extractall("/opt/mycellios")
    log(f"runtime extraído ({len(b)//1024} KB)")


def _write_bridge():
    with open("/opt/bridge.py", "wb") as f:
        f.write(base64.b64decode(os.environ["GDLP_BRIDGE_B64"]))


def _boot():
    STATE["status"] = "preparing"; STATE["phase"] = "pip"
    sh([sys.executable, "-m", "pip", "install", "--no-cache-dir",
        "transformers==5.14.1", "accelerate==1.14.0", "safetensors==0.8.0",
        "numpy==1.26.4", "sentencepiece==0.2.2", "aiohttp==3.14.1"])
    STATE["phase"] = "extract"; _extract_payload(); _write_bridge()
    import torch
    STATE["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    log(f"GPU={STATE['gpu']} cuda={torch.cuda.is_available()}")
    # precarga: descarga el modelo a /opt/hf (server/stage lo cargan luego desde disco)
    model = os.environ.get("GDLP_MODEL")
    if model:
        STATE["phase"] = "preload"
        try:
            from huggingface_hub import snapshot_download
            os.environ["HF_HOME"] = "/opt/hf"
            snapshot_download(model, cache_dir="/opt/hf/hub")
            log(f"modelo {model} precargado a cache")
        except Exception as e:
            log(f"preload aviso ({e}); el server lo descargará")
    STATE["booted"] = True; STATE["status"] = "booted"; STATE["phase"] = None
    log("NODO BOOTED. Esperando /start del orquestador (barrera).")


def main():
    threading.Thread(target=_serve, daemon=True).start()
    log(f"agente nodo arriba (rol={_CFG['role']}, dual-stack :{GW_PORT})")
    try:
        _boot()
    except BaseException as e:
        STATE["status"] = "error"; STATE["error"] = f"{type(e).__name__}: {e}"
        log("ERROR boot:\n" + traceback.format_exc())
    while True:
        time.sleep(30)


if __name__ == "__main__":
    main()
