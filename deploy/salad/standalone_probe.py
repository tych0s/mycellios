"""Sonda S0 AUTOCONTENIDA y OBSERVABLE — vía SIN-imagen (solo API key).

Para no bloquear el arranque del programa en construir/pushear una imagen Docker
(que requiere Docker+registro del fundador), esta sonda corre sobre una imagen
pytorch estándar e instala transformers en caliente PERO de forma VISIBLE:

  1. Arranca el servidor HTTP de estado (SOLO stdlib) en el segundo 0.
  2. Instala deps por subprocess, volcando stdout/stderr a /log en vivo.
  3. Corre la calibración S0 (huella, timers, exactitud, e(B), ruido).

Así el gateway SIEMPRE responde y /log muestra el pip en directo — se acabó el
"8 min en 503 sin logs". Se inyecta via env (base64) y se ejecuta con
`python -c "<loader>"`. Cuando exista la imagen pre-horneada, se usa esa (más
rápida y reproducible); esta es la vía de arranque inmediato.
"""
import base64, hashlib, io, json, os, platform, socket, statistics, subprocess, sys, threading, time, traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATE = {"status": "booting", "phase": "http-up", "started_ts": time.time(), "error": None}
RESULTS = {}
_LOG = []
_LK = threading.Lock()


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    with _LK:
        _LOG.append(line)
        if len(_LOG) > 6000:
            del _LOG[:1500]
    print(line, flush=True)


class H(BaseHTTPRequestHandler):
    def _s(self, code, payload, ct="application/json"):
        b = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        try:
            self.wfile.write(b)
        except Exception:
            pass

    def do_GET(self):
        p = self.path.split("?")[0]
        if p in ("/", "/status", "/health", "/healthz"):
            self._s(200, {**STATE, "log_lines": len(_LOG)})
        elif p in ("/results", "/results.json"):
            self._s(200, {"status": STATE["status"], "results": RESULTS})
        elif p in ("/log", "/logs"):
            with _LK:
                t = "\n".join(_LOG[-2500:])
            self._s(200, t.encode(), "text/plain; charset=utf-8")
        else:
            self._s(404, {"error": "not found"})

    def log_message(self, *a):
        pass


class _DualStack(ThreadingHTTPServer):
    # El gateway de Salad conecta por IPv6; escuchar dual-stack (:: acepta IPv4+IPv6).
    address_family = socket.AF_INET6

    def server_bind(self):
        try:
            self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        except OSError:
            pass
        super().server_bind()


def serve():
    port = int(os.environ.get("GDLP_PORT", "8000"))
    _DualStack(("::", port), H).serve_forever()


def pip_install():
    STATE["phase"] = "pip"
    pkgs = ["transformers==5.14.1", "accelerate==1.14.0", "safetensors==0.8.0",
            "numpy==1.26.4", "sentencepiece==0.2.2"]
    log(f"pip install {pkgs} …")
    proc = subprocess.Popen([sys.executable, "-m", "pip", "install", "--no-cache-dir", *pkgs],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for ln in proc.stdout:
        log("pip| " + ln.rstrip())
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"pip fallo rc={rc}")
    log("pip OK")


def probe():
    import torch
    cuda = torch.cuda.is_available()
    RESULTS["timer_census"] = _timers()
    log(f"timers: {RESULTS['timer_census']}")
    RESULTS["fingerprint"] = {
        "cuda": cuda, "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if cuda else 0,
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "cpu": platform.processor() or platform.machine(), "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
    }
    log(f"fp: {RESULTS['fingerprint']}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.cache_utils import DynamicCache
    name = os.environ.get("GDLP_MODEL", "Qwen/Qwen3-0.6B")
    dev = "cuda" if cuda else "cpu"
    dtype = torch.float16 if cuda else torch.float32
    STATE["phase"] = "loading-model"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(dev).eval()
    RESULTS["model"] = {"name": name, "device": dev, "dtype": str(dtype)}
    log(f"modelo {name} en {dev}")

    def greedy(prompt, n=32):
        import torch as T
        with T.inference_mode():
            ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
            c = DynamicCache()
            o = model(input_ids=ids, past_key_values=c, use_cache=True)
            cur = int(T.argmax(o.logits[0, -1])); seq = [cur]
            for _ in range(n - 1):
                o = model(input_ids=T.tensor([[cur]], device=dev), past_key_values=c, use_cache=True)
                cur = int(T.argmax(o.logits[0, -1])); seq.append(cur)
        return seq

    STATE["phase"] = "exactness"
    ex = {}
    for p in ("The capital of France is", "def fibonacci(n):", "Explain briefly what a GPU is."):
        s = greedy(p); ex[p] = {"ids": s, "sha256": hashlib.sha256(json.dumps(s).encode()).hexdigest()}
    RESULTS["exactness"] = ex
    log("exactitud hecha")

    def curve(prefill=48, reps=15):
        import torch as T
        out = {}
        with T.inference_mode():
            for B in (1, 2, 4, 8):
                ids = T.randint(0, 1000, (B, prefill), device=dev)
                nt = T.randint(0, 1000, (B, 1), device=dev)
                for _ in range(3):
                    c = DynamicCache(); model(input_ids=ids, past_key_values=c, use_cache=True)
                    model(input_ids=nt, past_key_values=c, use_cache=True)
                ts = []
                for _ in range(reps):
                    c = DynamicCache(); model(input_ids=ids, past_key_values=c, use_cache=True)
                    if cuda: T.cuda.synchronize()
                    t0 = time.perf_counter()
                    model(input_ids=nt, past_key_values=c, use_cache=True)
                    if cuda: T.cuda.synchronize()
                    ts.append((time.perf_counter() - t0) * 1000)
                out[B] = statistics.median(ts)
        return out

    STATE["phase"] = "cost-curve"
    cv = curve()
    RESULTS["cost_curve_ms"] = cv
    RESULTS["e_of_B"] = {B: round(B * cv[1] / cv[B], 3) for B in cv}
    log(f"c(B)={cv} e(B)={RESULTS['e_of_B']}")

    STATE["phase"] = "noise"
    n = {"c1": [], "c8": []}
    for _ in range(10):
        c = curve(reps=5); n["c1"].append(c[1]); n["c8"].append(c[8])
    RESULTS["noise"] = {k: {"median_ms": round(statistics.median(v), 3),
                            "cv_pct": round(100 * statistics.pstdev(v) / statistics.median(v), 2)}
                        for k, v in n.items()}
    log(f"ruido: {RESULTS['noise']}")


def _timers():
    w = []
    for _ in range(200):
        t0 = time.perf_counter(); time.sleep(0.001); w.append((time.perf_counter() - t0) * 1000)
    w.sort()
    return {"sleep1ms_p50_ms": round(w[100], 4), "sleep1ms_p95_ms": round(w[190], 4),
            "sleep1ms_min_ms": round(w[0], 4), "sleep1ms_max_ms": round(w[-1], 4)}


def run():
    threading.Thread(target=serve, daemon=True).start()
    log("HTTP status arriba")
    try:
        STATE["status"] = "running"
        pip_install()
        probe()
        STATE["status"] = "done"; STATE["done_ts"] = time.time()
        log("S0 COMPLETADO")
    except BaseException as e:
        STATE["status"] = "error"; STATE["error"] = f"{type(e).__name__}: {e}"
        log("ERROR:\n" + traceback.format_exc())
    while True:
        time.sleep(30)


if __name__ == "__main__":
    run()
