"""Reproducible validation of the exact prefill credit window.

This module deliberately separates two evidence classes:

* ``SIMULACION_DETERMINISTA`` is an exact deterministic tandem-pipeline model.
* ``MEDIDO_EN_LOOPBACK_CON_RED_EMULADA`` runs the real GDLP pipeline processes
  and TCP framing on one host.  It is not evidence from several physical PCs.

The benchmark compares stop-and-wait (W=1) with one or more larger credit
windows, and refuses to report success when generated token ids differ.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any, Iterable, Sequence


SCHEMA = "gdlp-prefill-credit-window-benchmark/1"
SIMULATION_EVIDENCE = "SIMULACION_DETERMINISTA"
LOOPBACK_EVIDENCE = "MEDIDO_EN_LOOPBACK_CON_RED_EMULADA"
DEFAULT_PREFILL_INFLIGHT_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class PipelineSimulation:
    chunks: int
    chunk_tokens: int
    service_ms: tuple[float, ...]
    window: int
    completion_ms: tuple[float, ...]

    @property
    def ttft_ms(self) -> float:
        return self.completion_ms[-1]

    @property
    def prompt_tokens_per_second_to_first_token(self) -> float:
        return self.chunks * self.chunk_tokens * 1_000.0 / self.ttft_ms


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be at least 0")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return parsed


def parse_windows(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("windows must contain integers") from error
    if not values or any(value < 1 or value > 64 for value in values):
        raise argparse.ArgumentTypeError("windows must be between 1 and 64")
    if values[0] != 1:
        raise argparse.ArgumentTypeError("windows must start with the W=1 baseline")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("windows must not contain duplicates")
    return values


def parse_service_ms(raw: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("service times must be numbers") from error
    if len(values) < 2 or any(not math.isfinite(value) or value <= 0 for value in values):
        raise argparse.ArgumentTypeError(
            "service times need at least two finite positive values"
        )
    return values


def simulate_pipeline(
    *,
    chunks: int,
    chunk_tokens: int,
    service_ms: Sequence[float],
    window: int,
) -> PipelineSimulation:
    """Simulate a deterministic closed tandem pipeline exactly.

    ``window`` is the maximum number of chunks sent but not yet acknowledged.
    A credit becomes reusable only when the corresponding chunk leaves the
    final stage.  The recurrence therefore includes both stage capacity and
    the end-to-end credit dependency.
    """

    if chunks < 1 or chunk_tokens < 1 or window < 1:
        raise ValueError("chunks, chunk_tokens and window must be positive")
    services = tuple(float(value) for value in service_ms)
    if len(services) < 2 or any(
        not math.isfinite(value) or value <= 0 for value in services
    ):
        raise ValueError("service_ms needs at least two finite positive values")

    stages = len(services)
    completion = [[0.0 for _ in range(stages)] for _ in range(chunks)]
    for chunk_index in range(chunks):
        for stage_index, service in enumerate(services):
            prior_stage = completion[chunk_index][stage_index - 1] if stage_index else 0.0
            prior_chunk = completion[chunk_index - 1][stage_index] if chunk_index else 0.0
            credit_return = (
                completion[chunk_index - window][-1]
                if stage_index == 0 and chunk_index >= window
                else 0.0
            )
            completion[chunk_index][stage_index] = (
                max(prior_stage, prior_chunk, credit_return) + service
            )
    return PipelineSimulation(
        chunks=chunks,
        chunk_tokens=chunk_tokens,
        service_ms=services,
        window=window,
        completion_ms=tuple(row[-1] for row in completion),
    )


def stop_and_wait_closed_form_ms(chunks: int, service_ms: Sequence[float]) -> float:
    """Exact W=1 makespan: every chunk pays the whole route serially."""

    return float(chunks) * sum(float(value) for value in service_ms)


def unlimited_window_closed_form_ms(chunks: int, service_ms: Sequence[float]) -> float:
    """Exact unlimited-credit makespan for a deterministic tandem pipeline."""

    services = tuple(float(value) for value in service_ms)
    return sum(services) + (chunks - 1) * max(services)


def balanced_closed_form_ms(
    *, chunks: int, stages: int, service_ms: float, window: int
) -> float:
    """Exact finite-W formula when all stages have equal service time.

    With ``K=min(W,S)``, completion slots are::

        T/s = S + (N-1) + floor((N-1)/K) * (S-K)
    """

    if min(chunks, stages, window) < 1 or not math.isfinite(service_ms) or service_ms <= 0:
        raise ValueError("balanced formula inputs must be finite and positive")
    credits = min(window, stages)
    slots = (
        stages
        + (chunks - 1)
        + ((chunks - 1) // credits) * (stages - credits)
    )
    return service_ms * slots


def _describe(values: Iterable[float]) -> dict[str, float | int]:
    materialized = sorted(float(value) for value in values)
    if not materialized:
        raise ValueError("cannot describe an empty sample")

    def percentile(fraction: float) -> float:
        if len(materialized) == 1:
            return materialized[0]
        position = (len(materialized) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return materialized[lower]
        weight = position - lower
        return materialized[lower] * (1 - weight) + materialized[upper] * weight

    return {
        "count": len(materialized),
        "mean": statistics.fmean(materialized),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "min": materialized[0],
        "max": materialized[-1],
    }


def build_simulation_report(
    *,
    chunks: int,
    chunk_tokens: int,
    windows: Sequence[int],
    balanced_services_ms: Sequence[float],
    bottleneck_services_ms: Sequence[float],
) -> dict[str, Any]:
    if not windows or windows[0] != 1:
        raise ValueError("simulation windows must start with W=1")
    scenarios: list[dict[str, Any]] = []
    for name, services in (
        ("equilibrado", tuple(float(value) for value in balanced_services_ms)),
        ("cuello_de_botella", tuple(float(value) for value in bottleneck_services_ms)),
    ):
        runs = [
            simulate_pipeline(
                chunks=chunks,
                chunk_tokens=chunk_tokens,
                service_ms=services,
                window=window,
            )
            for window in windows
        ]
        baseline = runs[0].ttft_ms
        scenarios.append(
            {
                "name": name,
                "service_ms_per_stage": list(services),
                "route_service_sum_ms": sum(services),
                "bottleneck_service_ms": max(services),
                "closed_forms_ms": {
                    "w1_stop_and_wait": stop_and_wait_closed_form_ms(chunks, services),
                    "unlimited_credit": unlimited_window_closed_form_ms(chunks, services),
                },
                "windows": [
                    {
                        "window": run.window,
                        "ttft_ms": run.ttft_ms,
                        "prompt_tokens_per_second_to_first_token": (
                            run.prompt_tokens_per_second_to_first_token
                        ),
                        "speedup_vs_w1": baseline / run.ttft_ms,
                        "chunk_completion_ms": list(run.completion_ms),
                    }
                    for run in runs
                ],
            }
        )
    return {
        "evidence_class": SIMULATION_EVIDENCE,
        "physical_multi_pc": False,
        "physical_gpu": False,
        "network": "not_used",
        "chunks": chunks,
        "chunk_tokens": chunk_tokens,
        "prompt_tokens": chunks * chunk_tokens,
        "scenarios": scenarios,
        "interpretation": (
            "Deterministic queueing result only; it validates the credit-window "
            "algorithm and its theoretical ceiling, not model or WAN speed."
        ),
    }


def _layer_boundaries(total_layers: int, stages: int, *, bottleneck: bool) -> tuple[int, ...]:
    if stages < 2 or stages > total_layers:
        raise ValueError("stages must be between 2 and the number of model layers")
    if not bottleneck:
        boundaries = tuple(round(index * total_layers / stages) for index in range(stages + 1))
    else:
        # Give every stage one layer, then put 70% of the remaining layers in
        # the middle stage. Distribute the rest deterministically around it.
        counts = [1 for _ in range(stages)]
        remaining = total_layers - stages
        middle = stages // 2
        heavy = min(remaining, max(1, round(total_layers * 0.70) - 1))
        counts[middle] += heavy
        remaining -= heavy
        cursor = 0
        while remaining > 0:
            if cursor != middle:
                counts[cursor] += 1
                remaining -= 1
            cursor = (cursor + 1) % stages
        values = [0]
        for count in counts:
            values.append(values[-1] + count)
        boundaries = tuple(values)
    if any(right <= left for left, right in zip(boundaries, boundaries[1:])):
        raise ValueError("derived boundaries are not strictly increasing")
    return boundaries


def _build_prompt_ids(tokenizer: Any, prompt: str, prompt_tokens: int) -> Any:
    import torch

    repeated = prompt.strip()
    if not repeated:
        raise ValueError("prompt cannot be empty")
    text = repeated
    while True:
        encoded = tokenizer(text, return_tensors="pt", add_special_tokens=True)[
            "input_ids"
        ].to(dtype=torch.long, device="cpu")
        if int(encoded.shape[1]) >= prompt_tokens:
            return encoded[:, :prompt_tokens].contiguous()
        text = f"{text} {repeated}"


def _run_runtime_window(
    *,
    model_snapshot: str,
    input_ids: Any,
    boundaries: tuple[int, ...],
    window: int,
    chunk_tokens: int,
    prefill_inflight_bytes: int,
    output_tokens: int,
    iterations: int,
    warmups: int,
    threads_per_stage: int,
    one_way_delay_ms: float,
    bandwidth_mbps: float,
    client_id_seed: int,
) -> dict[str, Any]:
    from .engine import (
        DistributedPipelineEngine,
        GenerationInput,
        PipelineEngineConfig,
        PipelineShutdownError,
    )
    from .protocol import TensorCodec

    startup_started = time.perf_counter()
    engine = DistributedPipelineEngine(
        PipelineEngineConfig(
            model_name=model_snapshot,
            boundaries=boundaries,
            codec=TensorCodec.FP32,
            threads_per_stage=threads_per_stage,
            startup_timeout_seconds=180.0,
            socket_timeout_seconds=180.0,
            one_way_delay_ms=one_way_delay_ms,
            bandwidth_mbps=bandwidth_mbps,
            max_active_sequences=1,
            max_pending_requests=2,
            prefill_chunk_tokens=chunk_tokens,
            prefill_inflight_chunks=window,
            prefill_inflight_bytes=prefill_inflight_bytes,
            speculative_max_draft_tokens=0,
            root_batch_window_ms=0.0,
            route_probe_interval_seconds=0.0,
        )
    )
    startup_ms = (time.perf_counter() - startup_started) * 1_000.0
    samples: list[dict[str, Any]] = []
    close_error: str | None = None
    try:
        total = warmups + iterations
        for index in range(total):
            measured = index >= warmups
            wall_started = time.perf_counter()
            output = engine.generate(
                [
                    GenerationInput(
                        client_id=client_id_seed + index,
                        input_ids=input_ids.clone(),
                        max_new_tokens=output_tokens,
                    )
                ]
            )[0]
            wall_ms = (time.perf_counter() - wall_started) * 1_000.0
            if measured:
                samples.append(
                    {
                        "ttft_ms": float(output.ttft_ms),
                        "total_ms": float(output.total_ms),
                        "wall_ms": wall_ms,
                        "token_ids": list(output.token_ids),
                        "finish_reason": output.finish_reason,
                        "prompt_tokens_per_second_to_first_token": (
                            int(input_ids.shape[1]) * 1_000.0 / float(output.ttft_ms)
                        ),
                    }
                )
        stats = dict(engine.prefill_window_stats)
        healthy = bool(engine.healthy)
    finally:
        try:
            engine.close()
        except PipelineShutdownError as error:
            close_error = str(error)
    shutdown = _shutdown_evidence(engine)
    shutdown["close_error"] = close_error
    exercise = _window_exercise(
        window=window,
        prompt_tokens=int(input_ids.shape[1]),
        chunk_tokens=chunk_tokens,
        prefill_inflight_bytes=prefill_inflight_bytes,
        stats=stats,
    )
    return {
        "window": window,
        "startup_ms_excluded_from_latency": startup_ms,
        "healthy_before_shutdown": healthy,
        "shutdown": shutdown,
        "window_exercised": exercise,
        "prefill_window_stats": stats,
        "samples": samples,
        "ttft_ms": _describe(sample["ttft_ms"] for sample in samples),
        "total_ms": _describe(sample["total_ms"] for sample in samples),
        "prompt_tokens_per_second_to_first_token": _describe(
            sample["prompt_tokens_per_second_to_first_token"] for sample in samples
        ),
    }


def _token_parity(window_runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    token_sequences = [
        tuple(int(token) for token in sample["token_ids"])
        for run in window_runs
        for sample in run["samples"]
    ]
    if not token_sequences:
        raise ValueError("runtime report has no measured token sequences")
    baseline = token_sequences[0]
    mismatches = [
        {"sample": index, "token_ids": list(tokens)}
        for index, tokens in enumerate(token_sequences)
        if tokens != baseline
    ]
    return {
        "exact": not mismatches,
        "baseline_token_ids": list(baseline),
        "sequences_checked": len(token_sequences),
        "mismatches": mismatches,
    }


def _window_exercise(
    *,
    window: int,
    prompt_tokens: int,
    chunk_tokens: int,
    prefill_inflight_bytes: int,
    stats: dict[str, Any],
) -> dict[str, Any]:
    expected = min(window, math.ceil(prompt_tokens / chunk_tokens))
    observed = int(stats.get("max_request_chunks", -1))
    configured_bytes = int(stats.get("configured_bytes_per_request", -1))
    observed_reserved_bytes = int(stats.get("max_request_reserved_bytes", -1))
    byte_cap_exact = (
        configured_bytes == prefill_inflight_bytes
        and prefill_inflight_bytes > 0
        and 0 < observed_reserved_bytes <= prefill_inflight_bytes
    )
    return {
        "exact": observed == expected and byte_cap_exact,
        "expected_max_request_chunks": expected,
        "observed_max_request_chunks": observed,
        "configured_bytes_per_request": configured_bytes,
        "expected_bytes_per_request": prefill_inflight_bytes,
        "observed_max_request_reserved_bytes": observed_reserved_bytes,
        "reserved_bytes_within_cap": byte_cap_exact,
        "stats_include_warmups": True,
    }


def _shutdown_evidence(engine: Any) -> dict[str, Any]:
    engine_report = getattr(engine, "shutdown_status", None)
    if engine_report is not None:
        return {
            "exact": bool(engine_report.get("clean", False)),
            "child_processes": engine_report.get("child_processes", []),
            "threads_alive": engine_report.get("threads_alive", {}),
            "hard_fallback_used": bool(
                engine_report.get("hard_fallback_used", False)
            ),
            "fatal_stage_metrics": engine_report.get("fatal_stage_metrics", []),
            "cleanup_errors": engine_report.get("cleanup_errors", []),
        }
    exit_codes = [
        {
            "name": str(getattr(process, "name", "unknown")),
            "pid": getattr(process, "pid", None),
            "exit_code": getattr(process, "exitcode", None),
        }
        for process in getattr(engine, "_processes", ())
    ]
    fatal_metrics = [
        metric
        for metric in getattr(engine, "stage_metrics", ())
        if isinstance(metric, dict) and "fatal_error" in metric
    ]
    exact = bool(exit_codes) and all(
        item["exit_code"] == 0 for item in exit_codes
    ) and not fatal_metrics
    return {
        "exact": exact,
        "child_processes": exit_codes,
        "fatal_stage_metrics": fatal_metrics,
    }


def run_runtime_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoConfig

    from .model import load_tokenizer, model_snapshot_identity, resolve_model_snapshot

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.manual_seed(args.seed)
    model_snapshot = resolve_model_snapshot(args.model)
    config = AutoConfig.from_pretrained(model_snapshot, local_files_only=True)
    total_layers = int(config.num_hidden_layers)
    tokenizer = load_tokenizer(model_snapshot)
    input_ids = _build_prompt_ids(tokenizer, args.prompt, args.runtime_prompt_tokens)
    maximum_context = int(getattr(config, "max_position_embeddings", 0) or 0)
    if maximum_context and int(input_ids.shape[1]) + args.output_tokens > maximum_context:
        raise ValueError("runtime prompt plus output exceeds the model context")

    scenarios: list[dict[str, Any]] = []
    for scenario_index, scenario_name in enumerate(args.runtime_scenarios):
        bottleneck = scenario_name == "cuello_de_botella"
        boundaries = _layer_boundaries(total_layers, args.stages, bottleneck=bottleneck)
        runs = []
        for window_index, window in enumerate(args.windows):
            print(
                f"runtime scenario={scenario_name} W={window}: loading and measuring...",
                file=sys.stderr,
                flush=True,
            )
            runs.append(
                _run_runtime_window(
                    model_snapshot=model_snapshot,
                    input_ids=input_ids,
                    boundaries=boundaries,
                    window=window,
                    chunk_tokens=args.chunk_tokens,
                    prefill_inflight_bytes=args.prefill_inflight_bytes,
                    output_tokens=args.output_tokens,
                    iterations=args.iterations,
                    warmups=args.warmups,
                    threads_per_stage=args.threads_per_stage,
                    one_way_delay_ms=args.one_way_delay_ms,
                    bandwidth_mbps=args.bandwidth_mbps,
                    client_id_seed=(scenario_index + 1) * 100_000 + window_index * 1_000,
                )
            )
        parity = _token_parity(runs)
        exercise_exact = all(run["window_exercised"]["exact"] for run in runs)
        shutdown_exact = all(run["shutdown"]["exact"] for run in runs)
        baseline_p50 = float(runs[0]["ttft_ms"]["p50"])
        for run in runs:
            run["ttft_speedup_vs_w1_p50"] = baseline_p50 / float(
                run["ttft_ms"]["p50"]
            )
        scenarios.append(
            {
                "name": (
                    "capas_equilibradas_sin_calibrar_servicio"
                    if scenario_name == "equilibrado"
                    else "etapa_central_con_mas_capas"
                ),
                "boundaries": list(boundaries),
                "layer_counts": [
                    right - left for left, right in zip(boundaries, boundaries[1:])
                ],
                "token_parity": parity,
                "credit_window_exercised": exercise_exact,
                "clean_shutdown": shutdown_exact,
                "windows": runs,
            }
        )

    exact = all(
        scenario["token_parity"]["exact"]
        and scenario["credit_window_exercised"]
        and scenario["clean_shutdown"]
        for scenario in scenarios
    )
    return {
        "evidence_class": LOOPBACK_EVIDENCE,
        "success": exact,
        "physical_multi_pc": False,
        "physical_gpu": False,
        "real_runtime_processes": True,
        "real_tcp_framing": True,
        "network": {
            "transport": "TCP 127.0.0.1",
            "one_way_delay_ms_emulated_per_send": args.one_way_delay_ms,
            "bandwidth_mbps_emulated": args.bandwidth_mbps,
            "warning": (
                "Delay/bandwidth are injected with sleep before loopback sends; "
                "they do not reproduce NIC, router, WAN jitter or multi-host contention."
            ),
        },
        "configuration": {
            "model": args.model,
            "model_snapshot": model_snapshot,
            "pipeline_snapshot_identity": model_snapshot_identity(model_snapshot),
            "prompt_tokens": int(input_ids.shape[1]),
            "chunk_tokens": args.chunk_tokens,
            "prefill_inflight_bytes_per_request": args.prefill_inflight_bytes,
            "chunks": math.ceil(int(input_ids.shape[1]) / args.chunk_tokens),
            "output_tokens": args.output_tokens,
            "codec": "fp32_exact",
            "stages": args.stages,
            "windows": list(args.windows),
            "warmups": args.warmups,
            "iterations": args.iterations,
            "threads_per_stage": args.threads_per_stage,
            "startup_excluded": True,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "logical_cpus": os.cpu_count(),
        },
        "scenarios": scenarios,
        "limits": [
            "Single host and loopback only; this is not a two-PC physical result.",
            "CPU/GPU service time is whatever this host executes; only link delay and bandwidth are emulated.",
            "Prompt throughput to first token is not autoregressive decode tok/s.",
            "Layer-balanced boundaries do not guarantee calibrated equal stage service.",
        ],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare exact prefill stop-and-wait with a credit window."
    )
    parser.add_argument("--mode", choices=("simulation", "runtime", "both"), default="simulation")
    parser.add_argument("--windows", type=parse_windows, default=parse_windows("1,3"))
    parser.add_argument("--chunks", type=positive_int, default=12)
    parser.add_argument("--chunk-tokens", type=positive_int, default=16)
    parser.add_argument(
        "--prefill-inflight-bytes",
        type=positive_int,
        default=DEFAULT_PREFILL_INFLIGHT_BYTES,
        help="Sealed per-request byte credit; productive default is 64 MiB.",
    )
    parser.add_argument("--balanced-services-ms", type=parse_service_ms, default=parse_service_ms("20,20,20"))
    parser.add_argument("--bottleneck-services-ms", type=parse_service_ms, default=parse_service_ms("8,44,8"))
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--prompt", default="Explain why exact distributed inference keeps token parity.")
    parser.add_argument("--runtime-prompt-tokens", type=positive_int, default=192)
    parser.add_argument("--output-tokens", type=positive_int, default=1)
    parser.add_argument("--stages", type=positive_int, default=3)
    parser.add_argument(
        "--runtime-scenarios",
        type=lambda value: tuple(item.strip() for item in value.split(",") if item.strip()),
        default=("equilibrado", "cuello_de_botella"),
    )
    parser.add_argument("--warmups", type=nonnegative_int, default=1)
    parser.add_argument("--iterations", type=positive_int, default=3)
    parser.add_argument("--threads-per-stage", type=positive_int, default=1)
    parser.add_argument("--one-way-delay-ms", type=nonnegative_float, default=20.0)
    parser.add_argument("--bandwidth-mbps", type=nonnegative_float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--compact-json", action="store_true")
    args = parser.parse_args(argv)
    allowed_scenarios = {"equilibrado", "cuello_de_botella"}
    if not args.runtime_scenarios or any(
        item not in allowed_scenarios for item in args.runtime_scenarios
    ):
        parser.error("runtime-scenarios must use equilibrado and/or cuello_de_botella")
    if len(args.balanced_services_ms) != len(args.bottleneck_services_ms):
        parser.error("both simulation scenarios must have the same number of stages")
    if args.prefill_inflight_bytes > 1024 * 1024 * 1024:
        parser.error("prefill-inflight-bytes cannot exceed 1 GiB")
    return args


def run(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at_unix_seconds": time.time(),
        "claim_boundary": {
            "physical_multi_pc": False,
            "physical_gpu": False,
            "wan_measured": False,
            "statement": (
                "This artifact cannot validate multi-PC, GPU or real-WAN performance."
            ),
        },
    }
    if args.mode in ("simulation", "both"):
        result["simulation"] = build_simulation_report(
            chunks=args.chunks,
            chunk_tokens=args.chunk_tokens,
            windows=args.windows,
            balanced_services_ms=args.balanced_services_ms,
            bottleneck_services_ms=args.bottleneck_services_ms,
        )
    if args.mode in ("runtime", "both"):
        result["runtime"] = run_runtime_benchmark(args)
    result["success"] = bool(result.get("runtime", {}).get("success", True))
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run(args)
    rendered = json.dumps(
        result,
        ensure_ascii=False,
        indent=None if args.compact_json else 2,
        sort_keys=True,
    )
    print(rendered)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")
    return 0 if result["success"] else 2


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    raise SystemExit(main())
