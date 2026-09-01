"""Sonda AUTOCONTENIDA que mide C, el término de cómputo por token, en un nodo GPU real.

Es la ejecución remota de `docs/benchmarks/compute-term-c-2026-07-25/measure_c.py`,
que en portátil salió NO RESOLUBLE porque el ruido térmico (dispersión intra-punto
del 220 %) superaba al efecto. Un nodo Linux dedicado tiene reloj estable y no
hace boost, que es justo lo que faltaba.

MÉTODO. Se barre el retardo de ida emulado y se ajusta `t = pendiente·RTT + C`.
La ordenada en el origen **es** C; la pendiente son los saltos efectivos por
token, que valida el convenio «1 tramo = 1 RTT» contra la medida.

Sólo es honesto desde el 25-07-2026, cuando se arregló el `LinkEmulator`: el
modelo viejo cobraba la propagación en serie e inflaba el coste por exactamente
las ventanas en vuelo.

DOS PUERTAS, no una. `R² ≥ 0,90` **y** dispersión por punto **≤ 40 %**. Un R²
alto sobre p50 ruidosas es casualidad, no medida — es el error que ya costó dos
retractaciones en esta campaña.

Barridos INTERCALADOS (alternando el orden en cada pasada): recorrer los retardos
de menor a mayor deja la deriva del host correlacionada con el retardo, y entonces
el ajuste mide la deriva.

Se inyecta por env (base64) junto al runtime en `PAYLOAD_*`, sobre una imagen
pytorch estándar. Sirve `/status`, `/log` y `/results` desde el segundo 0 para que
el gateway nunca dé 503 mudo.
"""
import base64, io, json, os, socket, statistics, subprocess, sys, tarfile, threading, time, traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATE = {"status": "booting", "phase": "http-up", "started_ts": time.time(), "error": None}
RESULTS = {}
_LOG = []
_LK = threading.Lock()

DELAYS_MS = (0.0, 10.0, 25.0, 50.0)
R2_FLOOR = 0.90
SPREAD_CEILING = 0.40
RUNTIME_DIR = "/opt/mycellios"
# Margen para que los procesos-etapa de la corrida anterior liberen la VRAM.
SETTLE_SECONDS = int(os.environ.get("GDLP_SETTLE_SECONDS", "10"))


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
    # El gateway de Salad conecta por IPv6; hay que escuchar dual-stack.
    address_family = socket.AF_INET6

    def server_bind(self):
        try:
            self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        except OSError:
            pass
        super().server_bind()


def serve():
    _DualStack(("::", int(os.environ.get("GDLP_PORT", "8000"))), H).serve_forever()


def pip_install():
    STATE["phase"] = "pip"
    pkgs = ["transformers==5.14.1", "accelerate==1.14.0", "safetensors==0.8.0",
            "numpy==1.26.4", "sentencepiece==0.2.2", "huggingface_hub"]
    log(f"pip install {pkgs}")
    proc = subprocess.Popen([sys.executable, "-m", "pip", "install", "--no-cache-dir", *pkgs],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for ln in proc.stdout:
        log("pip| " + ln.rstrip())
    if proc.wait() != 0:
        raise RuntimeError("pip falló")
    log("pip OK")


def extract_payload():
    STATE["phase"] = "extract"
    n = int(os.environ["PAYLOAD_PARTS"])
    blob = base64.b64decode("".join(os.environ[f"PAYLOAD_{i}"] for i in range(n)))
    os.makedirs(RUNTIME_DIR, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        tar.extractall(RUNTIME_DIR)
    log(f"runtime extraído ({len(blob) // 1024} KB)")


def fit_line(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx if sxx else 0.0
    intercept = my - slope * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    return slope, intercept, (1 - ss_res / ss_tot if ss_tot else 1.0)


def run_benchmark(delay_ms, model, stages, output_tokens, iterations):
    """Una corrida del banco. Reintenta el OOM de CUDA, que aquí es transitorio.

    El banco lanza un proceso por etapa. En una tarjeta de 4 GB, la VRAM de la
    corrida anterior puede no estar liberada todavía cuando arranca la siguiente
    —el proceso padre ha terminado pero los hijos aún están muriendo—, así que
    el segundo arranque revienta con OOM aunque el modelo quepa de sobra. No es
    falta de memoria: es una carrera. Se deja asentar y se reintenta.
    """
    cmd = [sys.executable, "-m", "distributed_runtime.benchmark",
           "--model", model, "--stages", str(stages),
           "--output-tokens", str(output_tokens),
           "--iterations", str(iterations), "--warmups", "1",
           "--one-way-delay-ms", str(delay_ms),
           "--compact-json"]
    env = dict(os.environ)
    env["PYTHONPATH"] = RUNTIME_DIR
    env["HF_HOME"] = "/opt/hf"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    last = ""
    for attempt in range(3):
        out = f"/tmp/bench-{delay_ms}-{attempt}-{int(time.time())}.json"
        done = subprocess.run(cmd + ["--json-out", out], env=env,
                              capture_output=True, text=True, timeout=1800)
        if done.returncode == 0:
            with open(out, encoding="utf-8") as handle:
                report = json.load(handle)
            os.unlink(out)
            # Cada corrida deja asentar la VRAM antes de la siguiente.
            time.sleep(SETTLE_SECONDS)
            return float(report["latency_ms"]["tpot"]["p50"]), float(report["correctness"]["rate"])
        last = done.stderr[-900:]
        oom = "out of memory" in last.lower() or "CUDA" in last
        log(f"banco rc={done.returncode} retardo={delay_ms} intento {attempt + 1}/3"
            f"{' (OOM, dejando asentar)' if oom else ''}")
        time.sleep(SETTLE_SECONDS * (attempt + 2))
    log(f"stderr final: {last}")
    raise RuntimeError(f"el banco falló con retardo {delay_ms} tras 3 intentos")


def measure():
    model = os.environ.get("GDLP_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct")
    stages = int(os.environ.get("GDLP_STAGES", "2"))
    sweeps = int(os.environ.get("GDLP_SWEEPS", "5"))
    output_tokens = int(os.environ.get("GDLP_OUTPUT_TOKENS", "32"))
    iterations = int(os.environ.get("GDLP_ITERATIONS", "3"))

    import torch
    cuda = torch.cuda.is_available()
    RESULTS["fingerprint"] = {
        "cuda": cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if cuda else 0,
        "torch": torch.__version__,
        "cpu_count": os.cpu_count(),
    }
    log(f"huella: {RESULTS['fingerprint']}")

    STATE["phase"] = "measuring"
    samples = {d: [] for d in DELAYS_MS}
    exactness = []
    for sweep in range(sweeps):
        order = list(DELAYS_MS) if sweep % 2 == 0 else list(reversed(DELAYS_MS))
        for delay in order:
            tpot, exact = run_benchmark(delay, model, stages, output_tokens, iterations)
            samples[delay].append(tpot)
            exactness.append(exact)
            log(f"barrido {sweep + 1}/{sweeps} retardo={delay:5.1f} -> {tpot:8.2f} ms/token (exact={exact})")
            RESULTS["progress"] = {"sweep": sweep + 1, "of": sweeps}

    rows = []
    for delay in DELAYS_MS:
        values = samples[delay]
        p50 = statistics.median(values)
        spread = (max(values) - min(values)) / p50 if p50 else 0.0
        rows.append({"one_way_delay_ms": delay, "ms_per_token_p50": round(p50, 3),
                     "samples": [round(v, 3) for v in values],
                     "spread_fraction": round(spread, 4)})

    slope, intercept, r2 = fit_line([r["one_way_delay_ms"] for r in rows],
                                    [r["ms_per_token_p50"] for r in rows])
    worst = max(r["spread_fraction"] for r in rows)
    resolvable = bool(r2 >= R2_FLOOR and slope > 0 and worst <= SPREAD_CEILING)

    RESULTS.update({
        "model": model, "stages": stages, "sweeps": sweeps,
        "output_tokens": output_tokens, "iterations": iterations,
        "rows": rows,
        "fit": {"compute_term_c_ms": round(intercept, 2),
                "effective_hops_per_token": round(slope, 3),
                "r2": round(r2, 5)},
        "noise": {"worst_spread_fraction": round(worst, 4),
                  "spread_ceiling": SPREAD_CEILING, "r2_floor": R2_FLOOR},
        "token_exactness_rate": min(exactness) if exactness else None,
        "resolvable": resolvable,
    })
    log(f"C={intercept:.2f} ms  saltos={slope:.3f}  R2={r2:.4f}  dispersion={worst * 100:.0f}%")
    log("VEREDICTO: " + ("RESOLUBLE" if resolvable else "NO RESOLUBLE — no citar C"))


def main():
    threading.Thread(target=serve, daemon=True).start()
    log("servidor de estado arriba")
    try:
        extract_payload()
        pip_install()
        STATE["status"] = "running"
        measure()
        STATE["status"] = "done"
        STATE["phase"] = None
    except BaseException as error:
        STATE["status"] = "error"
        STATE["error"] = f"{type(error).__name__}: {error}"
        RESULTS["traceback"] = traceback.format_exc()[-4000:]
        log("FALLO: " + STATE["error"])
        log(traceback.format_exc()[-3000:])
    # Se queda vivo para que el orquestador pueda recoger /results antes del teardown.
    while True:
        time.sleep(30)


main()
