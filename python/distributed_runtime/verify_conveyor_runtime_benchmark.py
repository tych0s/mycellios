"""Paired A/B evidence for the native exact VERIFY conveyor.

This benchmark runs the real :class:`DistributedPipelineEngine`: a root stage,
local child processes and the production TCP protocol.  The network delay is
injected by ``PipelineEngineConfig.one_way_delay_ms`` on loopback, so the
result is useful runtime evidence but is deliberately *not* physical multi-PC,
GPU or WAN evidence.

Every contender is compared with the historical W=1 path using the same model,
prompt, target reference, draft seed and runtime settings.  The only execution
contract fields changed inside a pair are the VERIFY wave credit and its byte
ceiling.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import tempfile
import time
from typing import Any


SCHEMA = "mycellios-native-verify-conveyor-runtime-ab/1"
EVIDENCE_CLASS = "MEDIDO_EN_LOOPBACK_CON_RED_EMULADA"
CANONICALIZATION = "sorted-json-utf8-no-whitespace/1"
DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
DEFAULT_INFLIGHT_BYTES = 64 * 1024 * 1024
MAX_WINDOW = 16


def _csv_values(raw: str, label: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError(f"{label} cannot be empty")
    return values


def parse_windows(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item) for item in _csv_values(raw, "windows"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("windows must contain integers") from error
    if (
        values[0] != 1
        or len(values) < 2
        or any(value < 1 or value > MAX_WINDOW for value in values)
    ):
        raise argparse.ArgumentTypeError(
            f"windows must start with W=1 and include a W>1 arm up to {MAX_WINDOW}"
        )
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("windows must not contain duplicates")
    return values


def parse_delays(raw: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item) for item in _csv_values(raw, "delays"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("delays must contain numbers") from error
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise argparse.ArgumentTypeError("delays must be finite and non-negative")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("delays must not contain duplicates")
    return values


def parse_acceptances(raw: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item) for item in _csv_values(raw, "acceptances"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("acceptances must contain numbers") from error
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise argparse.ArgumentTypeError("acceptances must be finite values from 0 to 1")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("acceptances must not contain duplicates")
    return values


def parse_stage_counts(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item) for item in _csv_values(raw, "stage counts"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("stage counts must contain integers") from error
    if any(value < 2 or value > 64 for value in values):
        raise argparse.ArgumentTypeError("stage counts must be integers from 2 to 64")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("stage counts must not contain duplicates")
    return values


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return parsed


def build_interleaved_order(
    *,
    windows: Sequence[int],
    rounds: int,
    seed: int,
) -> tuple[tuple[int, ...], ...]:
    """Return balanced rotated arms; two arms produce AB/BA/AB/BA.

    ``seed`` is sealed into the benchmark order contract even though the first
    block intentionally starts at the baseline.  Provider randomness uses the
    same seed within each returned pair.
    """

    normalized = tuple(int(value) for value in windows)
    if (
        rounds < 1
        or len(normalized) < 2
        or normalized[0] != 1
        or len(set(normalized)) != len(normalized)
    ):
        raise ValueError("interleaving needs rounds >= 1 and unique windows starting at 1")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    orders: list[tuple[int, ...]] = []
    for pair_id in range(rounds):
        block = pair_id // 2
        offset = block % len(normalized) if len(normalized) > 2 else 0
        rotated = normalized[offset:] + normalized[:offset]
        if pair_id % 2:
            rotated = tuple(reversed(rotated))
        orders.append(rotated)
    return tuple(orders)


class ControlledReferenceDraftProvider:
    """Deterministic benchmark-only drafter with a sealed target acceptance.

    Correct tokens are read from the real monolithic greedy reference.  A
    stable SHA-256 draw selects correct versus deliberately different tokens
    independently by absolute continuation position.  This controls drafter
    quality without changing the target model or production runtime.
    """

    strategy = "benchmark-controlled-reference"
    defer_until_selected = False

    def __init__(
        self,
        *,
        prompt_tokens: Sequence[int],
        reference_tokens: Sequence[int],
        max_draft_tokens: int,
        alpha: float,
        seed: int,
        vocab_size: int,
    ) -> None:
        self.prompt_tokens = self._token_tuple(prompt_tokens, "prompt_tokens")
        self.reference_tokens = self._token_tuple(reference_tokens, "reference_tokens")
        if not self.prompt_tokens:
            raise ValueError("prompt_tokens cannot be empty")
        if not self.reference_tokens:
            raise ValueError("reference_tokens cannot be empty")
        if (
            isinstance(max_draft_tokens, bool)
            or not isinstance(max_draft_tokens, int)
            or not 1 <= max_draft_tokens <= 16
        ):
            raise ValueError("max_draft_tokens must be an integer from 1 to 16")
        if not math.isfinite(float(alpha)) or not 0 <= float(alpha) <= 1:
            raise ValueError("alpha must be finite and between 0 and 1")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        if (
            isinstance(vocab_size, bool)
            or not isinstance(vocab_size, int)
            or vocab_size < 2
        ):
            raise ValueError("vocab_size must be an integer of at least 2")
        if any(token >= vocab_size for token in (*self.prompt_tokens, *self.reference_tokens)):
            raise ValueError("benchmark token is outside vocab_size")
        self.max_draft_tokens = max_draft_tokens
        self.alpha = float(alpha)
        self.seed = seed
        self.vocab_size = vocab_size
        self.calls = 0
        self.tokens_offered = 0
        self.correct_tokens_offered = 0
        self.incorrect_tokens_offered = 0

    @staticmethod
    def _token_tuple(values: Sequence[int], label: str) -> tuple[int, ...]:
        if isinstance(values, (str, bytes, bytearray)):
            raise ValueError(f"{label} must be a sequence of token ids")
        normalized: list[int] = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{label} must contain non-negative integers")
            normalized.append(value)
        return tuple(normalized)

    def _correct_at(self, position: int) -> bool:
        if self.alpha == 0:
            return False
        if self.alpha == 1:
            return True
        material = f"{self.seed}:{position}".encode("ascii")
        draw = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
        return draw / float(1 << 64) < self.alpha

    def _wrong_token(self, exact: int, position: int) -> int:
        material = f"wrong:{self.seed}:{position}".encode("ascii")
        offset = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
        return (exact + 1 + offset % (self.vocab_size - 1)) % self.vocab_size

    def draft(
        self,
        token_history: Sequence[int],
        max_tokens: int | None = None,
    ) -> tuple[int, ...]:
        history = self._token_tuple(token_history, "token_history")
        if len(history) < len(self.prompt_tokens):
            raise ValueError("token_history is shorter than the sealed prompt")
        if history[: len(self.prompt_tokens)] != self.prompt_tokens:
            raise ValueError("token_history does not begin with the sealed prompt")
        if max_tokens is None:
            limit = self.max_draft_tokens
        elif isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 0:
            raise ValueError("max_tokens must be a non-negative integer")
        else:
            limit = min(max_tokens, self.max_draft_tokens)
        position = len(history) - len(self.prompt_tokens)
        remaining = self.reference_tokens[position : position + limit]
        offered: list[int] = []
        correct = 0
        for relative, exact in enumerate(remaining):
            absolute = position + relative
            if self._correct_at(absolute):
                offered.append(exact)
                correct += 1
            else:
                offered.append(self._wrong_token(exact, absolute))
        self.calls += 1
        self.tokens_offered += len(offered)
        self.correct_tokens_offered += correct
        self.incorrect_tokens_offered += len(offered) - correct
        return tuple(offered)

    def execution_snapshot(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "maxDraftTokens": self.max_draft_tokens,
            "targetAcceptance": self.alpha,
            "seed": self.seed,
            "calls": self.calls,
            "tokensOffered": self.tokens_offered,
            "correctTokensOffered": self.correct_tokens_offered,
            "incorrectTokensOffered": self.incorrect_tokens_offered,
            "referenceTokens": len(self.reference_tokens),
            "deferredUntilSelected": False,
        }


def _describe(values: Sequence[float]) -> dict[str, float | int]:
    ordered = sorted(float(value) for value in values)
    if not ordered or any(not math.isfinite(value) for value in ordered):
        raise ValueError("metric sample must contain finite values")

    def percentile(fraction: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        position = (len(ordered) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "min": ordered[0],
        "max": ordered[-1],
    }


def _int_stat(stats: Mapping[str, Any], name: str) -> int | None:
    value = stats.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def validate_arm_evidence(
    arm: Mapping[str, Any],
    *,
    reference_sha256: str,
    expected_window: int,
    expected_inflight_bytes: int,
) -> dict[str, Any]:
    """Fail closed unless parity, credit bounds, drain and cleanup are exact."""

    stats_value = arm.get("speculative_window_stats")
    stats = stats_value if isinstance(stats_value, Mapping) else {}
    configured_waves = _int_stat(stats, "configured_waves_per_request")
    configured_bytes = _int_stat(stats, "configured_bytes_per_request")
    wave_values = tuple(
        _int_stat(stats, name)
        for name in ("high_water_waves", "max_request_waves")
    )
    byte_values = tuple(
        _int_stat(stats, name)
        for name in (
            "high_water_bytes",
            "high_water_reserved_bytes",
            "max_request_bytes",
            "max_request_reserved_bytes",
        )
    )
    current_values = tuple(
        _int_stat(stats, name)
        for name in ("current_waves", "current_bytes", "current_reserved_bytes")
    )
    speculation = arm.get("speculation_stats")
    proposed = (
        speculation.get("proposed_tokens")
        if isinstance(speculation, Mapping)
        else None
    )
    token_hash_exact = (
        isinstance(reference_sha256, str)
        and bool(reference_sha256)
        and arm.get("token_sha256") == reference_sha256
        and all(
            value == reference_sha256
            for value in arm.get("token_hashes", ())
        )
    )
    wave_bounds = (
        configured_waves == expected_window
        and all(value is not None and value <= expected_window for value in wave_values)
    )
    byte_bounds = configured_bytes == expected_inflight_bytes
    if expected_inflight_bytes > 0:
        byte_bounds = byte_bounds and all(
            value is not None and value <= expected_inflight_bytes
            for value in byte_values
        )
    else:
        byte_bounds = byte_bounds and all(value is not None for value in byte_values)
    resource_limits_exact = wave_bounds and byte_bounds
    high_water = wave_values[0]
    max_request = wave_values[1]
    conveyor_exercised = (
        isinstance(proposed, int)
        and not isinstance(proposed, bool)
        and proposed > 0
        and high_water is not None
        and max_request is not None
        and (
            (expected_window == 1 and high_water <= 1 and max_request <= 1)
            or (expected_window > 1 and high_water > 1 and max_request > 1)
        )
    )
    drained_to_zero = all(value == 0 for value in current_values)
    shutdown_value = arm.get("shutdown")
    shutdown = shutdown_value if isinstance(shutdown_value, Mapping) else {}
    clean_shutdown = (
        shutdown.get("exact") is True
        and shutdown.get("close_error") is None
        and shutdown.get("hard_fallback_used") is False
        and shutdown.get("cleanup_errors") == []
        and arm.get("healthy_before_shutdown") is True
    )
    checks = {
        "token_hash_exact": token_hash_exact,
        "conveyor_exercised": conveyor_exercised,
        "resource_limits_exact": resource_limits_exact,
        "drained_to_zero": drained_to_zero,
        "clean_shutdown": clean_shutdown,
    }
    return {
        "exact": all(checks.values()),
        **checks,
        "expected_window": expected_window,
        "expected_inflight_bytes": expected_inflight_bytes,
        "violations": [name for name, exact in checks.items() if not exact],
    }


def summarize_paired_runs(
    flat_runs: Sequence[Mapping[str, Any]],
    *,
    baseline_window: int,
    contender_window: int,
) -> dict[str, Any]:
    """Calculate within-pair deltas so load drift cannot become the result."""

    grouped: dict[int, dict[int, Mapping[str, Any]]] = {}
    for run in flat_runs:
        pair_id = run.get("pair_id")
        window = run.get("window")
        if (
            isinstance(pair_id, bool)
            or not isinstance(pair_id, int)
            or isinstance(window, bool)
            or not isinstance(window, int)
        ):
            raise ValueError("paired runs require integer pair_id and window")
        if window not in (baseline_window, contender_window):
            continue
        arms = grouped.setdefault(pair_id, {})
        if window in arms:
            raise ValueError(f"pair {pair_id} contains duplicate W={window} arms")
        arms[window] = run
    if not grouped:
        raise ValueError("paired summary has no matching runs")
    metrics = ("ttft_ms", "tpot_ms", "total_ms", "tokens_per_second")
    pairs: list[dict[str, Any]] = []
    deltas: dict[str, list[float]] = {metric: [] for metric in metrics}
    speedups: list[float] = []
    for pair_id, arms in sorted(grouped.items()):
        if set(arms) != {baseline_window, contender_window}:
            raise ValueError(f"pair {pair_id} is missing one comparison arm")
        baseline = arms[baseline_window]
        contender = arms[contender_window]
        pair_deltas: dict[str, float] = {}
        for metric in metrics:
            left = float(baseline[metric])
            right = float(contender[metric])
            if not math.isfinite(left) or not math.isfinite(right):
                raise ValueError(f"pair {pair_id} has non-finite {metric}")
            pair_deltas[metric] = right - left
            deltas[metric].append(pair_deltas[metric])
        baseline_total = float(baseline["total_ms"])
        contender_total = float(contender["total_ms"])
        if baseline_total <= 0 or contender_total <= 0:
            raise ValueError("total_ms must be positive for paired speedup")
        total_speedup = baseline_total / contender_total
        speedups.append(total_speedup)
        pairs.append(
            {
                "pair_id": pair_id,
                "baseline_window": baseline_window,
                "contender_window": contender_window,
                "deltas": pair_deltas,
                "total_ms_speedup": total_speedup,
            }
        )
    geometric = math.exp(statistics.fmean(math.log(value) for value in speedups))
    return {
        "pair_count": len(pairs),
        "baseline_window": baseline_window,
        "contender_window": contender_window,
        "pairs": pairs,
        "deltas": {
            metric: _describe(values)
            for metric, values in deltas.items()
        },
        "speedup": {
            "total_ms_geometric_mean": geometric,
            "samples": speedups,
        },
    }


def publish_json_no_overwrite(path: Path | str, payload: Mapping[str, Any]) -> Path:
    """Publish one JSON artifact atomically without replacing prior evidence."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="x",
            encoding="utf-8",
            newline="\n",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(rendered)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination


def _build_prompt_ids(tokenizer: Any, prompt: str, prompt_tokens: int) -> Any:
    import torch

    seed = prompt.strip()
    if not seed:
        raise ValueError("prompt cannot be empty")
    text = seed
    while True:
        encoded = tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=True,
        )["input_ids"].to(dtype=torch.long, device="cpu")
        if int(encoded.shape[1]) >= prompt_tokens:
            return encoded[:, :prompt_tokens].contiguous()
        text = f"{text} {seed}"


def _balanced_boundaries(total_layers: int, stages: int) -> tuple[int, ...]:
    if stages < 2 or stages > total_layers:
        raise ValueError(
            f"stage count {stages} must be between 2 and model layers {total_layers}"
        )
    boundaries = tuple(
        round(index * total_layers / stages)
        for index in range(stages + 1)
    )
    if any(right <= left for left, right in zip(boundaries, boundaries[1:])):
        raise ValueError("derived layer boundaries are not strictly increasing")
    return boundaries


def _shutdown_evidence(engine: Any, close_error: str | None) -> dict[str, Any]:
    report = getattr(engine, "shutdown_status", None)
    if isinstance(report, Mapping):
        return {
            "exact": bool(report.get("clean", False)) and close_error is None,
            "child_processes": report.get("child_processes", []),
            "threads_alive": report.get("threads_alive", {}),
            "hard_fallback_used": bool(report.get("hard_fallback_used", False)),
            "fatal_stage_metrics": report.get("fatal_stage_metrics", []),
            "cleanup_errors": list(report.get("cleanup_errors", [])),
            "close_error": close_error,
        }
    processes = tuple(getattr(engine, "_processes", ()))
    child_processes = [
        {
            "name": str(getattr(process, "name", "unknown")),
            "pid": getattr(process, "pid", None),
            "exit_code": getattr(process, "exitcode", None),
        }
        for process in processes
    ]
    fatal = [
        metric
        for metric in getattr(engine, "stage_metrics", ())
        if isinstance(metric, Mapping) and "fatal_error" in metric
    ]
    exact = (
        bool(child_processes)
        and all(item["exit_code"] == 0 for item in child_processes)
        and not fatal
        and close_error is None
    )
    return {
        "exact": exact,
        "child_processes": child_processes,
        "threads_alive": {},
        "hard_fallback_used": False,
        "fatal_stage_metrics": fatal,
        "cleanup_errors": [],
        "close_error": close_error,
    }


class _FixedBenchmarkController:
    """Mixin marker used only to make the benchmark policy obvious in reports."""


def _fixed_controller(k: int) -> Any:
    from .speculation import (
        AdaptiveSpeculationConfig,
        AdaptiveSpeculationController,
        SpeculationDecision,
    )

    class FixedBenchmarkController(
        _FixedBenchmarkController,
        AdaptiveSpeculationController,
    ):
        def decide(
            self,
            *,
            history_tokens: int,
            available_draft_tokens: int,
        ) -> Any:
            del history_tokens
            selected = min(k, int(available_draft_tokens))
            if selected <= 0:
                return SpeculationDecision(
                    enabled=False,
                    candidate_size=0,
                    predicted_speedup=None,
                    predicted_speedup_lower_bound=None,
                    predicted_latency_speedup=None,
                    predicted_byte_efficiency=None,
                    expected_emitted_tokens=None,
                    reason="benchmark_no_draft",
                )
            return SpeculationDecision(
                enabled=True,
                candidate_size=selected,
                predicted_speedup=1.0,
                predicted_speedup_lower_bound=1.0,
                predicted_latency_speedup=1.0,
                predicted_byte_efficiency=None,
                expected_emitted_tokens=None,
                reason="benchmark_fixed_k",
            )

        def next_probe_size(
            self,
            *,
            history_tokens: int,
            available_draft_tokens: int,
        ) -> int | None:
            del history_tokens, available_draft_tokens
            return None

    return FixedBenchmarkController(
        AdaptiveSpeculationConfig(
            max_draft_tokens=k,
            candidate_sizes=(k,),
            min_token_history=0,
            min_classic_observations=1,
            min_verify_observations=1,
            minimum_speedup=1.0,
        )
    )


def _token_hash(token_ids: Sequence[int]) -> str:
    from .server import output_token_ids_sha256

    return output_token_ids_sha256(tuple(int(token) for token in token_ids))


def _run_arm(
    *,
    model_snapshot: str,
    boundaries: tuple[int, ...],
    input_ids: Any,
    reference_tokens: tuple[int, ...],
    reference_sha256: str,
    vocab_size: int,
    alpha: float,
    pair_seed: int,
    pair_id: int,
    order_index: int,
    window: int,
    inflight_bytes: int,
    k: int,
    iterations: int,
    warmups: int,
    threads_per_stage: int,
    one_way_delay_ms: float,
    bandwidth_mbps: float,
    device: str,
    client_id_seed: int,
) -> dict[str, Any]:
    from .engine import (
        DistributedPipelineEngine,
        GenerationInput,
        PipelineEngineConfig,
        PipelineShutdownError,
    )
    from .protocol import TensorCodec

    effective_bytes = 0 if window == 1 else inflight_bytes
    provider = ControlledReferenceDraftProvider(
        prompt_tokens=tuple(int(token) for token in input_ids.reshape(-1).tolist()),
        reference_tokens=reference_tokens,
        max_draft_tokens=k,
        alpha=alpha,
        seed=pair_seed,
        vocab_size=vocab_size,
    )
    startup_started = time.perf_counter()
    engine = DistributedPipelineEngine(
        PipelineEngineConfig(
            model_name=model_snapshot,
            boundaries=boundaries,
            codec=TensorCodec.FP32,
            threads_per_stage=threads_per_stage,
            device=device,
            startup_timeout_seconds=180.0,
            socket_timeout_seconds=180.0,
            one_way_delay_ms=one_way_delay_ms,
            bandwidth_mbps=bandwidth_mbps,
            max_active_sequences=1,
            max_pending_requests=2,
            speculative_max_draft_tokens=k,
            speculative_inflight_waves=window,
            speculative_inflight_bytes=effective_bytes,
            speculation_minimum_speedup=1.0,
            speculation_probe=False,
            root_batch_window_ms=0.0,
            route_probe_interval_seconds=0.0,
        ),
        draft_provider=provider,
        speculation_controller=_fixed_controller(k),
    )
    startup_ms = (time.perf_counter() - startup_started) * 1_000.0
    samples: list[dict[str, Any]] = []
    warmup_hashes: list[str] = []
    close_error: str | None = None
    healthy_before_shutdown = False
    speculation_stats: dict[str, Any] = {}
    window_stats: dict[str, Any] = {}
    try:
        for index in range(warmups + iterations):
            wall_started = time.perf_counter()
            output = engine.generate(
                [
                    GenerationInput(
                        client_id=client_id_seed + index,
                        input_ids=input_ids.clone(),
                        max_new_tokens=len(reference_tokens),
                        eos_token_ids=frozenset(),
                    )
                ]
            )[0]
            wall_ms = (time.perf_counter() - wall_started) * 1_000.0
            token_hash = _token_hash(output.token_ids)
            if index < warmups:
                warmup_hashes.append(token_hash)
                continue
            total_ms = float(output.total_ms)
            tpot_ms = float(output.tpot_ms)
            samples.append(
                {
                    "sample": index - warmups,
                    "token_sha256": token_hash,
                    "finish_reason": output.finish_reason,
                    "ttft_ms": float(output.ttft_ms),
                    "tpot_ms": tpot_ms,
                    "total_ms": total_ms,
                    "wall_ms": wall_ms,
                    "tokens_per_second": (
                        len(output.token_ids) * 1_000.0 / total_ms
                        if total_ms > 0
                        else 0.0
                    ),
                    "decode_tokens_per_second": (
                        1_000.0 / tpot_ms if tpot_ms > 0 else None
                    ),
                    "output_tokens": len(output.token_ids),
                }
            )
        speculation_stats = dict(engine.speculation_stats)
        window_stats = dict(engine.speculative_window_stats)
        healthy_before_shutdown = bool(engine.healthy)
    finally:
        try:
            engine.close()
        except PipelineShutdownError as error:
            close_error = str(error)
    shutdown = _shutdown_evidence(engine, close_error)
    hashes = [str(sample["token_sha256"]) for sample in samples]
    all_hashes = [*warmup_hashes, *hashes]
    token_sha256 = (
        reference_sha256
        if all_hashes and all(value == reference_sha256 for value in all_hashes)
        else None
    )
    return {
        "pair_id": pair_id,
        "order_index": order_index,
        "window": window,
        "inflight_bytes": effective_bytes,
        "startup_ms_excluded_from_latency": startup_ms,
        "token_sha256": token_sha256,
        "token_hashes": hashes,
        "warmup_token_hashes": warmup_hashes,
        "healthy_before_shutdown": healthy_before_shutdown,
        "speculation_stats": speculation_stats,
        "speculative_window_stats": window_stats,
        "provider": provider.execution_snapshot(),
        "shutdown": shutdown,
        "samples": samples,
        "ttft_ms": float(_describe([item["ttft_ms"] for item in samples])["p50"]),
        "tpot_ms": float(_describe([item["tpot_ms"] for item in samples])["p50"]),
        "total_ms": float(_describe([item["total_ms"] for item in samples])["p50"]),
        "tokens_per_second": float(
            _describe([item["tokens_per_second"] for item in samples])["p50"]
        ),
        "metrics": {
            "ttft_ms": _describe([item["ttft_ms"] for item in samples]),
            "tpot_ms": _describe([item["tpot_ms"] for item in samples]),
            "total_ms": _describe([item["total_ms"] for item in samples]),
            "wall_ms": _describe([item["wall_ms"] for item in samples]),
            "tokens_per_second": _describe(
                [item["tokens_per_second"] for item in samples]
            ),
        },
    }


def run_runtime_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoConfig

    from .model import (
        load_tokenizer,
        model_snapshot_identity,
        reference_generate,
        resolve_model_snapshot,
    )

    model_snapshot = resolve_model_snapshot(args.model, args.revision)
    tokenizer = load_tokenizer(model_snapshot)
    input_ids = _build_prompt_ids(tokenizer, args.prompt, args.prompt_tokens)
    configuration = AutoConfig.from_pretrained(
        model_snapshot,
        local_files_only=True,
        trust_remote_code=False,
    )
    total_layers = int(getattr(configuration, "num_hidden_layers", 0))
    vocab_size = int(getattr(configuration, "vocab_size", 0))
    maximum_context = int(getattr(configuration, "max_position_embeddings", 0) or 0)
    if total_layers < 2 or vocab_size < 2:
        raise ValueError("model config lacks a usable layer count or vocabulary")
    if maximum_context and int(input_ids.shape[1]) + args.output_tokens > maximum_context:
        raise ValueError("prompt plus output exceeds the model context")
    if any(stages > total_layers for stages in args.stage_counts):
        raise ValueError("a requested stage count exceeds the model layer count")
    if args.output_tokens < args.draft_tokens * 2 + 2:
        raise ValueError(
            "output-tokens must be at least 2*k+2 so W>1 can be exercised"
        )
    reference_tokens_list, reference_metrics = reference_generate(
        model_snapshot,
        input_ids,
        args.output_tokens,
        args.threads_per_stage,
    )
    reference_tokens = tuple(reference_tokens_list)
    reference_sha256 = _token_hash(reference_tokens)
    snapshot_identity = model_snapshot_identity(args.model, args.revision)
    scenarios: list[dict[str, Any]] = []
    overall_success = True
    scenario_number = 0
    for stages in args.stage_counts:
        boundaries = _balanced_boundaries(total_layers, stages)
        for delay in args.delays_ms:
            for alpha in args.acceptances:
                scenario_number += 1
                scenario_id = (
                    f"stages-{stages}_delay-{delay:g}ms_alpha-{alpha:g}"
                )
                orders = build_interleaved_order(
                    windows=args.windows,
                    rounds=args.rounds,
                    seed=args.seed,
                )
                arms: list[dict[str, Any]] = []
                for pair_id, order in enumerate(orders):
                    pair_seed = args.seed + scenario_number * 1_000_003 + pair_id
                    for order_index, window in enumerate(order):
                        print(
                            (
                                f"{scenario_id} pair={pair_id} W={window}: "
                                "loading real loopback pipeline..."
                            ),
                            file=__import__("sys").stderr,
                            flush=True,
                        )
                        arm = _run_arm(
                            model_snapshot=model_snapshot,
                            boundaries=boundaries,
                            input_ids=input_ids,
                            reference_tokens=reference_tokens,
                            reference_sha256=reference_sha256,
                            vocab_size=vocab_size,
                            alpha=alpha,
                            pair_seed=pair_seed,
                            pair_id=pair_id,
                            order_index=order_index,
                            window=window,
                            inflight_bytes=args.inflight_bytes,
                            k=args.draft_tokens,
                            iterations=args.iterations,
                            warmups=args.warmups,
                            threads_per_stage=args.threads_per_stage,
                            one_way_delay_ms=delay,
                            bandwidth_mbps=args.bandwidth_mbps,
                            device=args.device,
                            client_id_seed=(
                                scenario_number * 1_000_000
                                + pair_id * 10_000
                                + order_index * 100
                            ),
                        )
                        expected_bytes = 0 if window == 1 else args.inflight_bytes
                        evidence = validate_arm_evidence(
                            arm,
                            reference_sha256=reference_sha256,
                            expected_window=window,
                            expected_inflight_bytes=expected_bytes,
                        )
                        arm["evidence"] = evidence
                        overall_success = overall_success and bool(evidence["exact"])
                        arms.append(arm)
                comparisons = [
                    summarize_paired_runs(
                        arms,
                        baseline_window=1,
                        contender_window=contender,
                    )
                    for contender in args.windows[1:]
                ]
                scenarios.append(
                    {
                        "scenario_id": scenario_id,
                        "stage_count": stages,
                        "boundaries": list(boundaries),
                        "one_way_delay_ms": delay,
                        "emulated_rtt_ms": delay * 2,
                        "target_draft_acceptance": alpha,
                        "pair_seed_rule": (
                            "same deterministic provider seed for every arm "
                            "inside one pair"
                        ),
                        "arm_order": [list(order) for order in orders],
                        "arms": arms,
                        "comparisons": comparisons,
                        "success": all(
                            bool(arm["evidence"]["exact"]) for arm in arms
                        ),
                    }
                )
    return {
        "evidence_class": EVIDENCE_CLASS,
        "claim_boundary": {
            "physical_multi_pc": False,
            "physical_gpu": False,
            "wan_measured": False,
            "physical_tokens_per_second_claimed": False,
            "statement": (
                "Real Mycellios processes and TCP on one host with an emulated "
                "link; never physical multi-PC, GPU or WAN performance."
            ),
        },
        "model": {
            "requested": args.model,
            "revision": args.revision,
            "snapshot_identity_uint64": snapshot_identity,
            "architecture": str(
                getattr(configuration, "model_type", type(configuration).__name__)
            ),
            "layers": total_layers,
            "vocab_size": vocab_size,
            "prompt_tokens": int(input_ids.shape[1]),
            "output_tokens": len(reference_tokens),
        },
        "reference": {
            "type": "monolithic_greedy_target",
            "token_sha256": reference_sha256,
            "hash_scheme": "gdlp-output-token-ids-v1",
            "metrics_not_used_as_distributed_performance_claim": reference_metrics,
        },
        "configuration": {
            "windows": list(args.windows),
            "draft_tokens_k": args.draft_tokens,
            "inflight_bytes_for_w_gt_1": args.inflight_bytes,
            "acceptances": list(args.acceptances),
            "one_way_delays_ms": list(args.delays_ms),
            "stage_counts": list(args.stage_counts),
            "rounds": args.rounds,
            "warmups_per_arm": args.warmups,
            "measured_iterations_per_arm": args.iterations,
            "threads_per_stage": args.threads_per_stage,
            "bandwidth_mbps": args.bandwidth_mbps,
            "device": args.device,
            "codec": "fp32",
            "seed": args.seed,
            "only_pair_differences": [
                "speculative_inflight_waves",
                "speculative_inflight_bytes",
            ],
        },
        "scenarios": scenarios,
        "success": overall_success and all(
            bool(scenario["success"]) for scenario in scenarios
        ),
        "limits": [
            "All pipeline stages are child processes on one physical host.",
            "Delay and bandwidth are runtime link-emulator inputs, not observed WAN.",
            "Draft acceptance is deterministic benchmark input, not a real drafter score.",
            "Startup/model-load time is recorded but excluded from generation latency.",
            "Physical throughput claims require the separate two-host GPU gate.",
        ],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    parser = argparse.ArgumentParser(
        description=(
            "Run a paired real-runtime W=1 versus W>1 exact VERIFY conveyor A/B "
            "on loopback with an emulated link."
        )
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision")
    parser.add_argument(
        "--prompt",
        default="Explain why exact distributed inference preserves every output token.",
    )
    parser.add_argument("--prompt-tokens", type=_positive_int, default=32)
    parser.add_argument("--output-tokens", type=_positive_int, default=32)
    parser.add_argument("--draft-tokens", type=_positive_int, default=4)
    parser.add_argument("--windows", type=parse_windows, default=parse_windows("1,4"))
    parser.add_argument(
        "--inflight-bytes",
        type=_positive_int,
        default=DEFAULT_INFLIGHT_BYTES,
    )
    parser.add_argument(
        "--acceptances",
        type=parse_acceptances,
        default=parse_acceptances("0.9,1"),
    )
    parser.add_argument(
        "--delays-ms",
        type=parse_delays,
        default=parse_delays("12.5"),
    )
    parser.add_argument(
        "--stage-counts",
        type=parse_stage_counts,
        default=parse_stage_counts("3"),
    )
    parser.add_argument("--rounds", type=_positive_int, default=2)
    parser.add_argument("--warmups", type=_nonnegative_int, default=0)
    parser.add_argument("--iterations", type=_positive_int, default=1)
    parser.add_argument("--threads-per-stage", type=_positive_int, default=1)
    parser.add_argument(
        "--bandwidth-mbps",
        type=_nonnegative_float,
        default=0.0,
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path(f"benchmarks/verify-conveyor-runtime-{timestamp}.json"),
    )
    parser.add_argument("--compact-json", action="store_true")
    args = parser.parse_args(argv)
    if args.draft_tokens > 16:
        parser.error("draft-tokens cannot exceed 16")
    if args.inflight_bytes > 1024 * 1024 * 1024:
        parser.error("inflight-bytes cannot exceed 1 GiB")
    if args.output_tokens < args.draft_tokens * 2 + 2:
        parser.error("output-tokens must be at least 2*k+2")
    return args


def run(args: argparse.Namespace) -> dict[str, Any]:
    measured = run_runtime_benchmark(args)
    body = {
        "schema": SCHEMA,
        "generated_at_unix_seconds": time.time(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        **measured,
    }
    canonical = json.dumps(
        body,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    return {
        **body,
        "seal": {
            "canonicalization": CANONICALIZATION,
            "algorithm": "sha256",
            "digest": f"sha256:{hashlib.sha256(canonical).hexdigest()}",
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run(args)
        rendered = json.dumps(
            result,
            ensure_ascii=False,
            indent=None if args.compact_json else 2,
            sort_keys=True,
            allow_nan=False,
        )
        print(rendered)
        publish_json_no_overwrite(args.json_out, result)
        print(f"Published immutable evidence: {args.json_out}", file=__import__("sys").stderr)
        return 0 if result["success"] else 2
    except BaseException as error:
        print(
            f"VERIFY conveyor runtime benchmark failed closed: "
            f"{type(error).__name__}: {error}",
            file=__import__("sys").stderr,
        )
        return 2


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    raise SystemExit(main())
