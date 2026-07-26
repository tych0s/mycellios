#!/usr/bin/env python
"""Fase 0: mide C, el coste de cómputo por token, y lo desglosa.

QUÉ LAGUNA CIERRA
-----------------
En GPU real un forward de decode desnudo cuesta ~33,35 ms cuando el techo de
ancho de banda de pesos (roofline) son ~4,44 ms. Los ~28,9 ms restantes —el
87 %— están acotados POR RESTA contra el roofline, no perfilados. Nunca se ha
corrido `torch.profiler` ni `nsys` en un nodo GPU de este proyecto.

Ese número decide una reescritura: si la mayor parte de esos 28,9 ms es
intérprete de Python, cambiar de motor (o de lenguaje) cotiza. Si son kernels
pequeños e ineficientes o lanzamiento de kernels, la respuesta es otra
completamente —`compile`, `StaticCache`, CUDA graphs— y no pasa por reescribir
la orquestación.

QUÉ MIDE
--------
Tres tiempos separados, no uno restado de otro:

  1. `wall_ms`        — reloj de pared con barrera de dispositivo. Es C.
  2. `dispatch_ms`    — reloj SIN barrera. Lo que tarda Python en encolar.
  3. `profiler_*_ms`  — desglose de `torch.profiler`: self-CPU (intérprete y
                        lanzamiento) frente a self-CUDA (kernel de verdad).

La resta `wall - dispatch` NO es "tiempo de GPU": en un forward encolado, parte
del despacho solapa con la ejecución. Por eso el desglose autoritativo es el del
profiler y no una resta, que es exactamente el método que produjo la cota de
28,9 ms que venimos a sustituir.

CÓMO SE LEE (umbrales preregistrados)
-------------------------------------
El informe imprime el veredicto contra los umbrales de la decisión de Rust:

  T3  fracción de intérprete Python dentro del forward > 40 %
      -> el motor cotiza; evaluar compile/StaticCache/CUDA graphs ANTES que
         cualquier cambio de lenguaje.
  T1  residual de orquestación fuera del forward > 25 ms/token
      -> este guion NO lo mide (mide dentro del forward). Sale del banco.

USO
---
    python scripts/profile_forward.py --help
    python scripts/profile_forward.py --layers 4 --hidden 512 --iters 50
    python scripts/profile_forward.py --device cuda --iters 200 --json out.json

Sin argumentos corre un modelo sintético en CPU, que sirve para validar el
guion pero NO para decidir nada: el número que importa es el de un nodo GPU
real con el modelo real.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass


@dataclass
class ForwardBreakdown:
    """Desglose de un forward. Cada campo dice cómo se obtuvo."""

    device: str
    model_label: str
    iterations: int
    warmup: int
    # Reloj de pared CON barrera: esto es C.
    wall_ms_p50: float
    wall_ms_p95: float
    # Reloj SIN barrera: coste de encolar desde Python.
    dispatch_ms_p50: float
    # Desglose del profiler. `None` si el profiler no estaba disponible.
    profiler_self_cpu_ms: float | None
    profiler_self_device_ms: float | None
    profiler_top_ops: list[dict[str, object]]
    # Veredicto contra los umbrales preregistrados.
    python_interpreter_fraction: float | None
    t3_triggered: bool | None
    notes: list[str]


#: Operaciones que son ESPERA, no cómputo. Se excluyen del self-CPU.
_SYNCHRONIZATION_MARKERS = (
    "synchronize",
    "cudaStreamSynchronize",
    "cudaEventSynchronize",
    "cudaMemcpyAsync",  # el memcpy síncrono bloquea esperando al dispositivo
    "Event",
)


def _is_synchronization(op_name: str) -> bool:
    lowered = op_name.lower()
    return any(marker.lower() in lowered for marker in _SYNCHRONIZATION_MARKERS)


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def _synchronize(torch_module, device: str) -> None:
    if device.startswith("cuda") and torch_module.cuda.is_available():
        torch_module.cuda.synchronize()
    elif device == "mps":
        synchronize = getattr(getattr(torch_module, "mps", None), "synchronize", None)
        if callable(synchronize):
            synchronize()


def _load_real_model(torch_module, model_id: str, device: str):
    """Carga un modelo real de Hugging Face y prepara un paso de decode.

    Es la diferencia entre validar el instrumento y medir el sistema. La pila
    sintética hace un forward de una capa densa; el modelo real hace atención
    con caché KV, que es **el régimen donde vive el problema**: en decode cada
    paso lee todos los pesos para producir un token, y ese es el término que la
    cota por resta contra el roofline nunca explicó.

    Devuelve `(step, descripcion)` donde `step()` ejecuta UN paso de decode
    incremental con la caché ya poblada — no un prefill, que es paralelo y no
    describe el coste por token.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch_module.float16 if device.startswith("cuda") else torch_module.float32
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()

    prompt = tokenizer("The capital of France is", return_tensors="pt").to(device)
    with torch_module.inference_mode():
        primed = model(**prompt, use_cache=True)
    cache = primed.past_key_values
    next_token = primed.logits[:, -1:].argmax(-1)

    primed_length = cache.get_seq_length()

    def step():
        # `use_cache=True` con un solo token de entrada: exactamente un paso de
        # decode.
        #
        # La caché se RECORTA a su longitud original después de cada paso. Sin
        # esto crece un token por iteración —270 iteraciones son 270 tokens— y
        # entonces no se mide "el coste de un paso de decode" sino "el coste
        # medio de una secuencia que se alarga", que sube con la longitud y
        # contamina la mediana. Además, dejar crecer la caché reasigna memoria
        # de forma repetida y añade un coste que no es del modelo.
        out = model(input_ids=next_token, past_key_values=cache, use_cache=True)
        cache.crop(primed_length)
        return out

    description = (
        f"{model_id} · {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params · "
        f"{dtype}"
    )
    return step, description


def _build_synthetic_model(torch_module, layers: int, hidden: int, device: str):
    """Pila de bloques lineales. NO es un transformer.

    Sirve para validar que el guion mide lo que dice medir. Para decidir nada
    hay que apuntarlo al modelo real con `--module`; el informe lo repite en
    `notes` para que ningún resultado sintético acabe citado como evidencia.
    """
    nn = torch_module.nn
    blocks = []
    for _ in range(layers):
        blocks.append(nn.Linear(hidden, hidden * 4))
        blocks.append(nn.GELU())
        blocks.append(nn.Linear(hidden * 4, hidden))
    return nn.Sequential(*blocks).to(device).eval()


def profile_forward(
    *,
    layers: int,
    hidden: int,
    iterations: int,
    warmup: int,
    device: str,
    use_profiler: bool,
    hf_model: str | None = None,
) -> ForwardBreakdown:
    try:
        import torch
    except ImportError:
        raise SystemExit(
            "torch no está disponible en este intérprete. "
            "Usa el venv del proyecto: runtime/distribution-venv/python.exe"
        )

    notes: list[str] = []
    if hf_model is None:
        notes.append(
            "modelo SINTÉTICO (pila de lineales): valida el guion, no decide nada. "
            "Para decidir, usar --hf-model con el modelo real."
        )

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        notes.append("se pidió cuda pero no hay GPU visible; se cae a cpu")
        device = "cpu"
    if device == "cpu":
        notes.append(
            "en CPU despacho y ejecución coinciden, así que la fracción de "
            "intérprete NO es extrapolable a GPU. T3 no se evalúa."
        )

    if hf_model:
        step, description = _load_real_model(torch, hf_model, device)
        notes.append(f"modelo REAL: {description}; un paso de decode con caché KV")
    else:
        model = _build_synthetic_model(torch, layers, hidden, device)
        # Un solo token: es el régimen de decode, que es donde vive el problema.
        sample = torch.randn(1, 1, hidden, device=device)

        def step():
            return model(sample)

    with torch.inference_mode():
        for _ in range(warmup):
            step()
        _synchronize(torch, device)

        wall_samples: list[float] = []
        dispatch_samples: list[float] = []
        for _ in range(iterations):
            started = time.perf_counter()
            step()
            dispatch_samples.append((time.perf_counter() - started) * 1_000)
            _synchronize(torch, device)
            wall_samples.append((time.perf_counter() - started) * 1_000)

        self_cpu_ms: float | None = None
        self_device_ms: float | None = None
        top_ops: list[dict[str, object]] = []
        if use_profiler:
            try:
                from torch.profiler import ProfilerActivity, profile

                activities = [ProfilerActivity.CPU]
                if device.startswith("cuda"):
                    activities.append(ProfilerActivity.CUDA)
                with profile(activities=activities, record_shapes=False) as prof:
                    for _ in range(max(8, iterations // 4)):
                        step()
                    _synchronize(torch, device)
                events = prof.key_averages()
                # Las primitivas de sincronización aparecen como self-CPU pero
                # son ESPERA a la GPU, no trabajo de intérprete. En la primera
                # corrida en GPU real `cudaDeviceSynchronize` fue 394,8 ms de
                # 465,2 (el 85 %), lo que inflaba la "fracción de intérprete"
                # de un 7 % real a un 33,5 % falso. Contar la espera como
                # cómputo es exactamente el error que este guion existe para
                # no cometer.
                wait_ms = (
                    sum(
                        e.self_cpu_time_total
                        for e in events
                        if _is_synchronization(e.key)
                    )
                    / 1_000
                )
                self_cpu_ms = (
                    sum(e.self_cpu_time_total for e in events) / 1_000 - wait_ms
                )
                notes.append(
                    f"self-CPU excluye {wait_ms:.1f} ms de sincronización "
                    f"(espera a la GPU, no intérprete)"
                )
                device_total = 0.0
                for event in events:
                    value = getattr(event, "self_device_time_total", None)
                    if value is None:
                        value = getattr(event, "self_cuda_time_total", 0)
                    device_total += value or 0
                self_device_ms = device_total / 1_000
                ranked = sorted(
                    events, key=lambda e: e.self_cpu_time_total, reverse=True
                )[:10]
                top_ops = [
                    {
                        "op": event.key,
                        "self_cpu_ms": round(event.self_cpu_time_total / 1_000, 4),
                        "count": event.count,
                    }
                    for event in ranked
                ]
            except Exception as error:  # pragma: no cover - depende del entorno
                notes.append(f"torch.profiler no disponible o falló: {error}")

    fraction: float | None = None
    t3: bool | None = None
    if (
        device.startswith("cuda")
        and self_cpu_ms is not None
        and self_device_ms is not None
        and (self_cpu_ms + self_device_ms) > 0
    ):
        fraction = self_cpu_ms / (self_cpu_ms + self_device_ms)
        t3 = fraction > 0.40

    return ForwardBreakdown(
        device=device,
        model_label=hf_model or f"sintetico L{layers} H{hidden}",
        iterations=iterations,
        warmup=warmup,
        wall_ms_p50=round(_percentile(wall_samples, 0.50), 4),
        wall_ms_p95=round(_percentile(wall_samples, 0.95), 4),
        dispatch_ms_p50=round(_percentile(dispatch_samples, 0.50), 4),
        profiler_self_cpu_ms=None if self_cpu_ms is None else round(self_cpu_ms, 4),
        profiler_self_device_ms=(
            None if self_device_ms is None else round(self_device_ms, 4)
        ),
        profiler_top_ops=top_ops,
        python_interpreter_fraction=None if fraction is None else round(fraction, 4),
        t3_triggered=t3,
        notes=notes,
    )


def render(breakdown: ForwardBreakdown) -> str:
    lines = [
        "== Fase 0: desglose del forward (término C) ==",
        f"dispositivo      : {breakdown.device}",
        f"modelo           : {breakdown.model_label}",
        f"iteraciones      : {breakdown.iterations} (warmup {breakdown.warmup})",
        "",
        f"C (wall, p50)    : {breakdown.wall_ms_p50:.4f} ms   <- ESTE es el término C",
        f"C (wall, p95)    : {breakdown.wall_ms_p95:.4f} ms",
        f"despacho (p50)   : {breakdown.dispatch_ms_p50:.4f} ms   (sin barrera)",
    ]
    if breakdown.profiler_self_cpu_ms is not None:
        lines += [
            "",
            f"profiler self-CPU: {breakdown.profiler_self_cpu_ms:.4f} ms  (intérprete + lanzamiento)",
            f"profiler self-dev: {breakdown.profiler_self_device_ms:.4f} ms  (kernel)",
        ]
    if breakdown.python_interpreter_fraction is not None:
        lines += [
            "",
            f"fracción intérprete: {breakdown.python_interpreter_fraction:.1%}",
            f"T3 (>40%) disparado: {'SÍ' if breakdown.t3_triggered else 'NO'}",
        ]
        if breakdown.t3_triggered:
            lines.append(
                "  -> evaluar compile/StaticCache/CUDA graphs ANTES de considerar "
                "cualquier cambio de motor o de lenguaje."
            )
        else:
            lines.append(
                "  -> el intérprete NO domina el forward; el cuello está en otra parte."
            )
    else:
        lines += ["", "fracción intérprete: NO EVALUADA (hace falta CUDA + profiler)"]
    if breakdown.profiler_top_ops:
        lines += ["", "operaciones por self-CPU:"]
        for entry in breakdown.profiler_top_ops[:5]:
            lines.append(
                f"  {entry['self_cpu_ms']:>9.4f} ms  x{entry['count']:<6} {entry['op']}"
            )
    lines += ["", "avisos:"]
    for note in breakdown.notes:
        lines.append(f"  - {note}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    parser.add_argument(
        "--hf-model",
        help="id de Hugging Face del modelo REAL; sin esto se usa la pila sintética",
    )
    parser.add_argument("--no-profiler", action="store_true")
    parser.add_argument("--json", help="escribe el desglose completo a un fichero")
    args = parser.parse_args(argv)

    breakdown = profile_forward(
        layers=args.layers,
        hidden=args.hidden,
        iterations=args.iters,
        warmup=args.warmup,
        device=args.device,
        use_profiler=not args.no_profiler,
        hf_model=args.hf_model,
    )
    print(render(breakdown))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(asdict(breakdown), handle, indent=2, ensure_ascii=False)
        print(f"\nJSON -> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
