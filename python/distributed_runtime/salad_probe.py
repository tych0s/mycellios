"""Sonda de calibración S0 para un nodo SaladCloud (módulo importable).

Mide, en un solo nodo GPU real:
  (a) huella del nodo (GPU, host CPU, VRAM, driver, plataforma),
  (b) censo de timers (anexo A2: ¿la cuantización ~12-15 ms vista en Windows
      aparece en Linux y reconfigura las olas?),
  (c) exactitud greedy — ids + sha256 de 32 tokens por prompt — para comparar
      contra la referencia CPU del testbed (anexo A6/S0: ¿"exacto" sobrevive
      entre hardwares?),
  (d) curva de coste c(B), B∈{1,2,4,8}, y eficiencia de fusión e(B)=B·c(1)/c(B)
      — EL número que decide cuánto vale el batching (2.6) en GPU real,
  (e) ruido (10 repeticiones separadas de c(1)/c(8)) → CV intra-nodo (ley L4).

No hace red: el entrypoint observable sirve `RESULTS` por HTTP. Diseñado para
correr bajo `deploy/salad/entrypoint.py` (modo probe), que le pasa el dict
compartido `results`, el dict `state` y un `log` callable.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import statistics
import time
from typing import Any, Callable

PROMPTS = (
    "The capital of France is",
    "def fibonacci(n):",
    "Explain briefly what a GPU is.",
)


def _timer_census() -> dict[str, float]:
    waits = []
    for _ in range(200):
        t0 = time.perf_counter()
        time.sleep(0.001)
        waits.append((time.perf_counter() - t0) * 1000)
    waits.sort()
    return {
        "sleep1ms_p50_ms": round(waits[100], 4),
        "sleep1ms_p95_ms": round(waits[190], 4),
        "sleep1ms_min_ms": round(waits[0], 4),
        "sleep1ms_max_ms": round(waits[-1], 4),
    }


def run_probe(state: dict, results: dict, log: Callable[[str], None]) -> None:
    results["timer_census"] = _timer_census()
    log(f"censo de timers: {results['timer_census']}")

    import torch

    cuda = torch.cuda.is_available()
    results["fingerprint"] = {
        "cuda": cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if cuda else 0,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
    }
    log(f"huella: {results['fingerprint']}")

    model_name = os.environ.get("GDLP_MODEL", "Qwen/Qwen3-0.6B")
    state["phase"] = "loading-model"
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.cache_utils import DynamicCache

    dev = "cuda" if cuda else "cpu"
    dtype = torch.float16 if cuda else torch.float32
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(dev).eval()
    results["model"] = {"name": model_name, "device": dev, "dtype": str(dtype)}
    log(f"modelo {model_name} en {dev} ({dtype})")

    @torch.inference_mode()
    def greedy_ids(prompt: str, n: int = 32) -> list[int]:
        ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
        cache = DynamicCache()
        out = model(input_ids=ids, past_key_values=cache, use_cache=True)
        cur = int(torch.argmax(out.logits[0, -1]))
        seq = [cur]
        for _ in range(n - 1):
            o = model(input_ids=torch.tensor([[cur]], device=dev), past_key_values=cache, use_cache=True)
            cur = int(torch.argmax(o.logits[0, -1]))
            seq.append(cur)
        return seq

    state["phase"] = "exactness"
    exact: dict[str, Any] = {}
    for prompt in PROMPTS:
        seq = greedy_ids(prompt)
        exact[prompt] = {"ids": seq, "sha256": hashlib.sha256(json.dumps(seq).encode()).hexdigest()}
    results["exactness"] = exact
    log("exactitud (32 tokens/prompt) hecha")

    @torch.inference_mode()
    def cost_curve(prefill_len: int = 48, reps: int = 15) -> dict[int, float]:
        curve: dict[int, float] = {}
        for batch in (1, 2, 4, 8):
            ids = torch.randint(0, 1000, (batch, prefill_len), device=dev)
            newtok = torch.randint(0, 1000, (batch, 1), device=dev)
            for _ in range(3):  # warm
                c = DynamicCache()
                model(input_ids=ids, past_key_values=c, use_cache=True)
                model(input_ids=newtok, past_key_values=c, use_cache=True)
            times = []
            for _ in range(reps):
                c = DynamicCache()
                model(input_ids=ids, past_key_values=c, use_cache=True)
                if cuda:
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                model(input_ids=newtok, past_key_values=c, use_cache=True)
                if cuda:
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1000)
            curve[batch] = statistics.median(times)
        return curve

    state["phase"] = "cost-curve"
    curve = cost_curve()
    results["cost_curve_ms"] = curve
    results["e_of_B"] = {b: round(b * curve[1] / curve[b], 3) for b in curve}
    log(f"c(B) ms={curve}  e(B)={results['e_of_B']}")

    state["phase"] = "noise"
    noise = {"c1": [], "c8": []}
    for _ in range(10):
        c = cost_curve(reps=5)
        noise["c1"].append(c[1])
        noise["c8"].append(c[8])
    results["noise"] = {
        k: {
            "median_ms": round(statistics.median(v), 3),
            "cv_pct": round(100 * statistics.pstdev(v) / statistics.median(v), 2),
        }
        for k, v in noise.items()
    }
    log(f"ruido: {results['noise']}")
    results["completed_ts"] = time.time()
