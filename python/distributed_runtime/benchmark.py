from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import socket
import statistics
import sys
import time
from typing import Any, Iterable

import torch
from transformers import AutoConfig

from .model import (
    StageModelSpec,
    StageRunner,
    load_tokenizer,
    model_snapshot_identity,
    reference_generate,
    resolve_model_snapshot,
)
from .protocol import (
    HEADER_BYTES,
    FrameType,
    LinkEmulator,
    TensorCodec,
    configure_socket,
    decode_token,
    encode_tensor_payload,
    recv_frame,
    send_frame,
)
from .stage import StageProcessConfig, drain_metrics, run_stage_process


DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
HOST = "127.0.0.1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark a real, persistent TCP layer pipeline against exact "
            "greedy monolithic inference."
        )
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--output-tokens", type=positive_int, default=8)
    parser.add_argument("--stages", type=int, choices=(2, 3, 4), default=2)
    parser.add_argument(
        "--boundaries",
        help="Optional comma-separated layer boundaries, for example 0,15,30.",
    )
    parser.add_argument(
        "--codec",
        choices=(
            "fp32",
            "fp16",
            "int8",
            "int8-grouped",
            "int8-hadamard",
            "int8-grouped-deflate",
            "int8-hadamard-deflate",
        ),
        default="fp32",
    )
    parser.add_argument("--concurrency", type=positive_int, default=1)
    parser.add_argument("--warmups", type=nonnegative_int, default=1)
    parser.add_argument("--iterations", type=positive_int, default=3)
    parser.add_argument("--threads-per-stage", type=positive_int, default=1)
    parser.add_argument("--reference-threads", type=positive_int)
    parser.add_argument(
        "--stage-quantize",
        choices=("none", "dynamic-int8"),
        default="none",
        help=(
            "Opt-in APPROXIMATE mode: dynamically quantize every stage Linear "
            "to INT8. Greedy tokens may drift from the FP32 reference; combine "
            "with --allow-token-drift to report the drift without failing."
        ),
    )
    parser.add_argument(
        "--stage-compile",
        choices=("none", "default", "reduce-overhead"),
        default="none",
        help=(
            "Opt-in torch.compile (inductor CPU) of every stage forward. "
            "Requires a working C++ toolchain on the host."
        ),
    )
    parser.add_argument(
        "--kv-cache",
        choices=("arena", "dynamic"),
        default="arena",
        help=(
            "KV cache layout for every stage. 'arena' appends new positions into "
            "preallocated storage; 'dynamic' restores Transformers' concatenating "
            "cache. Both are token-exact, so the flag exists to run the A/B."
        ),
    )
    parser.add_argument(
        "--decode-attention",
        choices=("grouped-prefix", "stock"),
        default="grouped-prefix",
        help=(
            "Decode attention for every stage. 'grouped-prefix' regroups the query "
            "into its KV groups and reads the cache in place; 'stock' expands the "
            "cached keys/values by the group factor as Transformers does."
        ),
    )
    parser.add_argument(
        "--one-way-delay-ms",
        type=link_delays,
        default=(0.0,),
        help=(
            "One-way delay in ms. A single value applies to every link; a comma-separated "
            "list sets each link separately, so co-located stages can carry a LAN delay "
            "while the boundary between homes carries a WAN one (e.g. '30,0.5,0.5,30')."
        ),
    )
    parser.add_argument("--bandwidth-mbps", type=nonnegative_float, default=0.0)
    parser.add_argument("--startup-timeout-seconds", type=positive_float, default=180.0)
    parser.add_argument("--socket-timeout-seconds", type=positive_float, default=180.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--allow-token-drift",
        action="store_true",
        help="Report lossy-codec token drift without returning a failing exit status.",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--compact-json", action="store_true")
    return parser.parse_args(argv)


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
        raise argparse.ArgumentTypeError("must be a finite number greater than 0")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be a finite number at least 0")
    return parsed


def link_delays(value: str) -> tuple[float, ...]:
    """One delay for every link, or a single delay applied to all of them.

    A swarm's links are not uniform: several stages can share a house, where the hop is
    a LAN round-trip, while the boundary between houses crosses the WAN. Expressing that
    needs a delay per link, not one number for the whole chain.
    """
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("expected at least one delay")
    return tuple(nonnegative_float(part) for part in parts)


def delay_for_link(delays: tuple[float, ...], index: int) -> float:
    """A single value covers every link; a list is indexed, and must be long enough."""
    if len(delays) == 1:
        return delays[0]
    if index >= len(delays):
        raise ValueError(
            f"--one-way-delay-ms lists {len(delays)} links but the pipeline uses more; "
            "pass one delay per link or a single value for all of them"
        )
    return delays[index]


def codec_from_name(name: str) -> TensorCodec:
    return {
        "fp32": TensorCodec.FP32,
        "fp16": TensorCodec.FP16,
        "int8": TensorCodec.INT8,
        "int8-grouped": TensorCodec.INT8_GROUPED,
        "int8-hadamard": TensorCodec.INT8_HADAMARD,
        "int8-grouped-deflate": TensorCodec.INT8_GROUPED_DEFLATE,
        "int8-hadamard-deflate": TensorCodec.INT8_HADAMARD_DEFLATE,
    }[name]


def layer_boundaries(total_layers: int, stages: int, raw: str | None) -> list[int]:
    if raw:
        try:
            boundaries = [int(item.strip()) for item in raw.split(",")]
        except ValueError as error:
            raise ValueError("boundaries must contain only integers") from error
    else:
        boundaries = [round(index * total_layers / stages) for index in range(stages + 1)]
    if len(boundaries) != stages + 1:
        raise ValueError(f"expected {stages + 1} boundaries, got {len(boundaries)}")
    if boundaries[0] != 0 or boundaries[-1] != total_layers:
        raise ValueError(f"boundaries must start at 0 and end at {total_layers}")
    if any(right <= left for left, right in zip(boundaries, boundaries[1:])):
        raise ValueError("every stage must own at least one contiguous layer")
    return boundaries


def reserve_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((HOST, 0))
        return int(sock.getsockname()[1])
    finally:
        sock.close()


def wait_until_listening(process: Any, ready_event: Any, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if ready_event.wait(0.1):
            return
        if process.exitcode is not None:
            raise RuntimeError(
                f"stage process {process.pid} exited during startup with code {process.exitcode}"
            )
    raise TimeoutError(f"stage process {process.pid} did not listen within {timeout_seconds}s")


def percentile(values: Iterable[float], fraction: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def describe(values: Iterable[float]) -> dict[str, float | int | None]:
    materialized = [float(value) for value in values]
    if not materialized:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "min": None, "max": None}
    return {
        "count": len(materialized),
        "mean": statistics.fmean(materialized),
        "p50": percentile(materialized, 0.50),
        "p95": percentile(materialized, 0.95),
        "min": min(materialized),
        "max": max(materialized),
    }


def send_activation(
    sock: socket.socket,
    emulator: LinkEmulator,
    codec: TensorCodec,
    request_id: int,
    step: int,
    hidden: torch.Tensor,
) -> tuple[int, float]:
    encode_started = time.perf_counter()
    encoded = encode_tensor_payload(hidden, codec)
    encode_ms = (time.perf_counter() - encode_started) * 1_000
    bytes_out = send_frame(
        sock,
        FrameType.ACTIVATION,
        request_id,
        step=step,
        token_count=int(hidden.shape[1]),
        hidden_size=int(hidden.shape[2]),
        flags=int(codec),
        payload=encoded.view,
        emulator=emulator,
    )
    return bytes_out, encode_ms


def receive_token_round(
    return_socket: socket.socket,
    request_ids: list[int],
    expected_step: int,
) -> dict[int, tuple[int, float]]:
    pending = set(request_ids)
    received: dict[int, tuple[int, float]] = {}
    while pending:
        frame = recv_frame(return_socket)
        arrived = time.perf_counter()
        if frame.frame_type == FrameType.ERROR:
            raise RuntimeError(frame.payload.decode("utf-8", errors="replace"))
        if frame.frame_type != FrameType.TOKEN:
            raise RuntimeError(f"unexpected return frame {frame.frame_type.name}")
        if frame.request_id not in pending:
            raise RuntimeError(f"duplicate or unknown token for request {frame.request_id}")
        if frame.step != expected_step:
            raise RuntimeError(
                f"request {frame.request_id} returned step {frame.step}, expected {expected_step}"
            )
        received[frame.request_id] = (decode_token(frame), arrived)
        pending.remove(frame.request_id)
    return received


def run_batch(
    *,
    runner: StageRunner,
    downstream: socket.socket,
    return_socket: socket.socket,
    emulator: LinkEmulator,
    codec: TensorCodec,
    input_ids: torch.Tensor,
    expected_tokens: list[int],
    output_tokens: int,
    concurrency: int,
    batch_index: int,
    measured: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    request_ids = [batch_index * 1_000_000 + index + 1 for index in range(concurrency)]
    batch_started = time.perf_counter()
    root_metrics = {
        request_id: {"compute_ms": 0.0, "encode_ms": 0.0, "frames": 0, "bytes_out": 0}
        for request_id in request_ids
    }
    generated: dict[int, list[int]] = {request_id: [] for request_id in request_ids}
    arrivals: dict[int, list[float]] = {request_id: [] for request_id in request_ids}

    for request_id in request_ids:
        runner.begin(request_id)
        send_frame(downstream, FrameType.BEGIN, request_id)
        compute_started = time.perf_counter()
        hidden = runner.forward_ids(request_id, input_ids)
        root_metrics[request_id]["compute_ms"] += (time.perf_counter() - compute_started) * 1_000
        bytes_out, encode_ms = send_activation(
            downstream, emulator, codec, request_id, 0, hidden
        )
        root_metrics[request_id]["encode_ms"] += encode_ms
        root_metrics[request_id]["bytes_out"] += bytes_out
        root_metrics[request_id]["frames"] += 1

    returned = receive_token_round(return_socket, request_ids, 0)
    for request_id, (token, arrived) in returned.items():
        generated[request_id].append(token)
        arrivals[request_id].append(arrived)

    for step in range(1, output_tokens):
        for request_id in request_ids:
            next_input = torch.tensor([[generated[request_id][-1]]], dtype=torch.long)
            compute_started = time.perf_counter()
            hidden = runner.forward_ids(request_id, next_input)
            root_metrics[request_id]["compute_ms"] += (
                time.perf_counter() - compute_started
            ) * 1_000
            bytes_out, encode_ms = send_activation(
                downstream, emulator, codec, request_id, step, hidden
            )
            root_metrics[request_id]["encode_ms"] += encode_ms
            root_metrics[request_id]["bytes_out"] += bytes_out
            root_metrics[request_id]["frames"] += 1
        returned = receive_token_round(return_socket, request_ids, step)
        for request_id, (token, arrived) in returned.items():
            generated[request_id].append(token)
            arrivals[request_id].append(arrived)

    batch_finished = max(times[-1] for times in arrivals.values())
    request_results: list[dict[str, Any]] = []
    for request_id in request_ids:
        times = arrivals[request_id]
        intervals = [
            (right - left) * 1_000 for left, right in zip(times, times[1:])
        ]
        response_ms = (times[-1] - batch_started) * 1_000
        request_results.append(
            {
                "request_id": request_id,
                "measured": measured,
                "tokens": generated[request_id],
                "exact_match": generated[request_id] == expected_tokens,
                "ttft_ms": (times[0] - batch_started) * 1_000,
                "tpot_ms": statistics.fmean(intervals) if intervals else 0.0,
                "response_ms": response_ms,
                "output_tps_including_ttft": output_tokens / max(response_ms / 1_000, 1e-12),
                "root_stage": root_metrics[request_id],
            }
        )
        runner.end(request_id)
        send_frame(downstream, FrameType.END, request_id)

    wall_ms = (batch_finished - batch_started) * 1_000
    batch_result = {
        "batch_index": batch_index,
        "measured": measured,
        "requests": concurrency,
        "output_tokens": concurrency * output_tokens,
        "wall_ms": wall_ms,
        "aggregate_output_tps_including_ttft": (
            concurrency * output_tokens / max(wall_ms / 1_000, 1e-12)
        ),
        "exact_matches": sum(result["exact_match"] for result in request_results),
    }
    return batch_result, request_results


def summarize_stage_metrics(
    stage_metrics: list[dict[str, Any]], measured_request_ids: set[int]
) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for metric in stage_metrics:
        if "fatal_error" in metric or int(metric.get("request_id", -1)) not in measured_request_ids:
            continue
        grouped[int(metric["stage"])].append(metric)
    summaries: list[dict[str, Any]] = []
    for layer_start, metrics in sorted(grouped.items()):
        summaries.append(
            {
                "layer_start": layer_start,
                "layer_end": int(metrics[0]["layer_end"]),
                "loader": str(metrics[0]["loader"]),
                "retained_parameter_bytes": int(metrics[0]["parameter_bytes"]),
                "requests": len(metrics),
                "frames": sum(int(metric["frames"]) for metric in metrics),
                "compute_ms": describe(float(metric["compute_ms"]) for metric in metrics),
                "bytes_out": sum(int(metric["bytes_out"]) for metric in metrics),
            }
        )
    return summaries


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.manual_seed(args.seed)
    # Resolve once so config, tokenizer, reference and every spawned stage use the
    # same immutable commit even if a Hub branch moves while the benchmark runs.
    stage_model_name = resolve_model_snapshot(args.model)
    pipeline_id = model_snapshot_identity(stage_model_name)
    config = AutoConfig.from_pretrained(stage_model_name)
    total_layers = int(config.num_hidden_layers)
    hidden_size = int(config.hidden_size)
    boundaries = layer_boundaries(total_layers, args.stages, args.boundaries)
    codec = codec_from_name(args.codec)
    reference_threads = args.reference_threads or args.threads_per_stage
    stage_quantize = None if args.stage_quantize == "none" else args.stage_quantize
    stage_compile = None if args.stage_compile == "none" else args.stage_compile

    tokenizer = load_tokenizer(stage_model_name)
    tokenized = tokenizer(args.prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = tokenized["input_ids"].to(dtype=torch.long, device="cpu")
    maximum_context = int(getattr(config, "max_position_embeddings", 0) or 0)
    if maximum_context and input_ids.shape[1] + args.output_tokens > maximum_context:
        raise ValueError(
            f"prompt plus output ({input_ids.shape[1] + args.output_tokens}) exceeds "
            f"model context {maximum_context}"
        )

    print("Generating monolithic greedy reference...", file=sys.stderr, flush=True)
    expected_tokens, reference_metrics = reference_generate(
        stage_model_name, input_ids, args.output_tokens, reference_threads
    )

    return_listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    return_listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    return_listener.bind((HOST, 0))
    return_listener.listen(1)
    return_listener.settimeout(args.socket_timeout_seconds)
    return_port = int(return_listener.getsockname()[1])
    listen_ports = [reserve_port() for _ in range(args.stages - 1)]

    context = mp.get_context("spawn")
    metrics_queue = context.Queue()
    processes: list[Any] = []
    downstream: socket.socket | None = None
    return_socket: socket.socket | None = None
    runner: StageRunner | None = None
    stage_metrics: list[dict[str, Any]] = []
    shutdown_sent = False
    emulator: LinkEmulator | None = None
    all_batches: list[dict[str, Any]] = []
    all_requests: list[dict[str, Any]] = []
    fatal_errors: list[dict[str, Any]] = []
    root_stage_info: dict[str, Any] = {}
    try:
        child_configs: list[StageProcessConfig] = []
        for child_index in range(args.stages - 1):
            stage_index = child_index + 1
            next_exists = stage_index + 1 < args.stages
            child_configs.append(
                StageProcessConfig(
                    spec=StageModelSpec(
                        model_name=stage_model_name,
                        layer_start=boundaries[stage_index],
                        layer_end=boundaries[stage_index + 1],
                        total_layers=total_layers,
                        threads=args.threads_per_stage,
                        quantize=stage_quantize,
                        compile_mode=stage_compile,
                        kv_cache=args.kv_cache,
                        decode_attention=args.decode_attention,
                    ),
                    pipeline_id=pipeline_id,
                    listen_host=HOST,
                    listen_port=listen_ports[child_index],
                    next_host=HOST if next_exists else None,
                    next_port=listen_ports[child_index + 1] if next_exists else None,
                    next_layer_end=boundaries[stage_index + 2] if next_exists else None,
                    return_host=HOST,
                    return_port=return_port,
                    codec=codec,
                    one_way_delay_ms=delay_for_link(args.one_way_delay_ms, child_index + 1),
                    bandwidth_mbps=args.bandwidth_mbps,
                    connect_timeout_seconds=args.startup_timeout_seconds,
                )
            )

        # Last-to-first startup guarantees that every downstream listener exists
        # before the persistent handshake begins, while avoiding simultaneous model loads.
        for stage_config in reversed(child_configs):
            ready_event = context.Event()
            process = context.Process(
                target=run_stage_process,
                args=(stage_config, ready_event, metrics_queue),
                name=f"distribution-stage-{stage_config.spec.layer_start}",
            )
            process.start()
            processes.append(process)
            print(
                f"Loading stage [{stage_config.spec.layer_start},{stage_config.spec.layer_end}) "
                f"in process {process.pid}...",
                file=sys.stderr,
                flush=True,
            )
            wait_until_listening(process, ready_event, args.startup_timeout_seconds)

        print(
            f"Loading root stage [0,{boundaries[1]})...", file=sys.stderr, flush=True
        )
        runner = StageRunner(
            StageModelSpec(
                model_name=stage_model_name,
                layer_start=0,
                layer_end=boundaries[1],
                total_layers=total_layers,
                threads=args.threads_per_stage,
                quantize=stage_quantize,
                compile_mode=stage_compile,
                kv_cache=args.kv_cache,
                decode_attention=args.decode_attention,
            )
        )
        root_stage_info = {
            "layer_start": 0,
            "layer_end": boundaries[1],
            "loader": runner.loader,
            "retained_parameter_bytes": runner.parameter_bytes,
        }
        if runner.hidden_size != hidden_size:
            raise RuntimeError(
                f"loaded model hidden size {runner.hidden_size} differs from config {hidden_size}"
            )

        downstream = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        configure_socket(downstream)
        downstream.settimeout(args.socket_timeout_seconds)
        downstream.connect((HOST, listen_ports[0]))
        send_frame(
            downstream,
            FrameType.HELLO,
            pipeline_id,
            step=boundaries[1],
            token_count=boundaries[2],
            hidden_size=hidden_size,
            flags=int(codec),
        )
        ready = recv_frame(downstream)
        if ready.frame_type != FrameType.READY:
            raise RuntimeError("pipeline did not return READY")
        if ready.request_id != pipeline_id:
            raise RuntimeError("pipeline READY returned a different model identity")
        return_socket, _ = return_listener.accept()
        configure_socket(return_socket)
        return_socket.settimeout(args.socket_timeout_seconds)
        emulator = LinkEmulator(
            delay_for_link(args.one_way_delay_ms, 0),
            args.bandwidth_mbps,
        )

        total_batches = args.warmups + args.iterations
        for batch_offset in range(total_batches):
            measured = batch_offset >= args.warmups
            label = "measure" if measured else "warmup"
            print(
                f"Running {label} batch {batch_offset + 1}/{total_batches} "
                f"(concurrency={args.concurrency})...",
                file=sys.stderr,
                flush=True,
            )
            batch, requests = run_batch(
                runner=runner,
                downstream=downstream,
                return_socket=return_socket,
                emulator=emulator,
                codec=codec,
                input_ids=input_ids,
                expected_tokens=expected_tokens,
                output_tokens=args.output_tokens,
                concurrency=args.concurrency,
                batch_index=batch_offset + 1,
                measured=measured,
            )
            all_batches.append(batch)
            all_requests.extend(requests)

        send_frame(downstream, FrameType.SHUTDOWN, 0)
        shutdown_sent = True
        emulator.close(timeout_seconds=args.socket_timeout_seconds)
        for process in processes:
            process.join(timeout=args.socket_timeout_seconds)
        stage_metrics.extend(drain_metrics(metrics_queue))
        fatal_errors = [metric for metric in stage_metrics if "fatal_error" in metric]
        failed_processes = [
            {"pid": process.pid, "name": process.name, "exit_code": process.exitcode}
            for process in processes
            if process.exitcode != 0
        ]
        if failed_processes or fatal_errors:
            raise RuntimeError(
                f"stage failure: processes={failed_processes}, errors={fatal_errors}"
            )
    finally:
        if downstream is not None and not shutdown_sent:
            try:
                send_frame(downstream, FrameType.SHUTDOWN, 0)
            except BaseException:
                pass
        if emulator is not None:
            try:
                emulator.close(timeout_seconds=args.socket_timeout_seconds)
            except BaseException:
                pass
        for sock in (return_socket, downstream, return_listener):
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
        for process in processes:
            if process.is_alive():
                process.join(timeout=2.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
        if runner is not None:
            del runner

    measured_batches = [batch for batch in all_batches if batch["measured"]]
    measured_requests = [request for request in all_requests if request["measured"]]
    measured_ids = {int(request["request_id"]) for request in measured_requests}
    exact_count = sum(bool(request["exact_match"]) for request in measured_requests)
    mismatch_examples = [
        {"request_id": request["request_id"], "tokens": request["tokens"]}
        for request in measured_requests
        if not request["exact_match"]
    ][:5]
    total_root_bytes = sum(
        int(request["root_stage"]["bytes_out"]) for request in measured_requests
    )
    measured_child_metrics = [
        metric
        for metric in stage_metrics
        if "fatal_error" not in metric and int(metric.get("request_id", -1)) in measured_ids
    ]
    counted_network_bytes = total_root_bytes + sum(
        int(metric["bytes_out"]) for metric in measured_child_metrics
    )
    # Stage metrics count activation/token frames and forwarded BEGIN frames. Root
    # BEGIN/END frames and forwarded END frames are deliberately off the hot metrics
    # path, so account for those fixed headers here rather than understating traffic.
    uncounted_control_bytes = len(measured_requests) * HEADER_BYTES * (
        2 + max(0, args.stages - 2)
    )
    total_network_bytes = counted_network_bytes + uncounted_control_bytes
    result = {
        "schema_version": 1,
        "success": exact_count == len(measured_requests),
        "configuration": {
            "model": args.model,
            "pipeline_id": pipeline_id,
            "prompt": args.prompt,
            "prompt_tokens": int(input_ids.shape[1]),
            "output_tokens": args.output_tokens,
            "stages": args.stages,
            "boundaries": boundaries,
            "codec": args.codec,
            "concurrency": args.concurrency,
            "kv_cache": args.kv_cache,
            "decode_attention": args.decode_attention,
            "warmups": args.warmups,
            "iterations": args.iterations,
            "threads_per_stage": args.threads_per_stage,
            "reference_threads": reference_threads,
            "stage_quantize": args.stage_quantize,
            "stage_compile": args.stage_compile,
            "token_equivalence_mode": (
                "approximate" if stage_quantize is not None else "exact-reference"
            ),
            "one_way_delay_ms": (
                args.one_way_delay_ms[0]
                if len(args.one_way_delay_ms) == 1
                else list(args.one_way_delay_ms)
            ),
            "bandwidth_mbps": args.bandwidth_mbps,
            "persistent_tcp": True,
            "direct_token_return": True,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "logical_cpus": os.cpu_count(),
        },
        "reference": {
            "tokens": expected_tokens,
            "text": tokenizer.decode(expected_tokens, skip_special_tokens=False),
            "metrics": reference_metrics,
        },
        "correctness": {
            "exact_matches": exact_count,
            "requests": len(measured_requests),
            "rate": exact_count / max(len(measured_requests), 1),
            "mismatch_examples": mismatch_examples,
        },
        "latency_ms": {
            "ttft": describe(request["ttft_ms"] for request in measured_requests),
            "tpot": describe(request["tpot_ms"] for request in measured_requests),
            "response": describe(request["response_ms"] for request in measured_requests),
        },
        "throughput": {
            "per_user_output_tps_including_ttft": describe(
                request["output_tps_including_ttft"] for request in measured_requests
            ),
            "aggregate_output_tps_including_ttft": describe(
                batch["aggregate_output_tps_including_ttft"] for batch in measured_batches
            ),
        },
        "network": {
            "measured_bytes": total_network_bytes,
            "activation_token_and_forwarded_begin_bytes": counted_network_bytes,
            "other_request_control_bytes": uncounted_control_bytes,
            "bytes_per_generated_token": total_network_bytes
            / max(len(measured_requests) * args.output_tokens, 1),
            "handshake_and_shutdown_excluded": True,
        },
        "root_stage": {
            **root_stage_info,
            "compute_ms": describe(
                request["root_stage"]["compute_ms"] for request in measured_requests
            ),
            "encode_ms": describe(
                request["root_stage"]["encode_ms"] for request in measured_requests
            ),
            "bytes_out": total_root_bytes,
        },
        "child_stages": summarize_stage_metrics(stage_metrics, measured_ids),
        "batches": measured_batches,
        "requests": measured_requests,
    }
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_benchmark(args)
    indent = None if args.compact_json else 2
    rendered = json.dumps(result, indent=indent, sort_keys=True)
    print(rendered)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")
    if not result["success"] and not args.allow_token_drift:
        print(
            "Distributed tokens differ from the monolithic greedy reference.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
