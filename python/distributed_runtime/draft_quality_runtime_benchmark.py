"""Real-runtime quality gate for native Mycellios draft providers.

The benchmark deliberately separates three execution arms:

``classic``
    Exact autoregressive distributed inference without a draft provider.
``draft_w1``
    Exact speculative inference with one VERIFY wave in flight.  This is the
    pre-registered quality-gate arm.
``draft_conveyor``
    The same provider and ``k`` with a bounded ``W > 1`` VERIFY conveyor.

The target :class:`DistributedPipelineEngine` remains the only token
authority.  Greedy monolithic references are used solely to establish an
expected output-token hash; their timings are never reported as distributed
performance evidence.

For every measured request the runner takes counter snapshots immediately
before and after ``generate``.  Route-level useful tokens per traversal are:

    g_route = (len(output.token_ids) - 1)
              / (delta completed_waves + delta classic_observations)

This definition is intentionally independent from controller
``accepted_tokens + verification_observations`` telemetry.  In a ``W > 1``
conveyor, overlapping waves may otherwise count the bridge token more than
once.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import statistics
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .verify_conveyor_runtime_benchmark import (
    _balanced_boundaries,
    _fixed_controller,
    _shutdown_evidence,
    _token_hash,
    publish_json_no_overwrite,
)

SCHEMA = "mycellios-native-draft-quality-runtime/1"
EVIDENCE_CLASS = "MEDIDO_RUNTIME_NATIVO_SINGLE_HOST"
CANONICALIZATION = "sorted-json-utf8-no-whitespace/1"
DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
DEFAULT_K_SWEEP = (1, 2, 4, 8, 16)
DEFAULT_CONVEYOR_WINDOW = 4
DEFAULT_INFLIGHT_BYTES = 64 * 1024 * 1024
MIN_MEASURED_REQUESTS_PER_ARM = 7
GATE_CONFIDENCE = 0.95
GATE_GO_LOWER_BOUND = 3.0
GATE_STOP_UPPER_BOUND = 2.0
ARM_NAMES = ("classic", "draft_w1", "draft_conveyor")

# A frozen, mixed corpus avoids selecting prompts after seeing acceptance.
# Repetition is present in some prompts because real workloads do contain it,
# but the corpus also includes prose, instructions, code and Spanish text.
DEFAULT_CORPUS = (
    "Explain in simple terms why exact verification keeps distributed inference correct.",
    "Summarize this rule: measure first, change one variable, measure again, and keep the exact output.",
    "Escribe tres pasos breves para comprobar que un nodo sigue sano después de una reconexión.",
    "Complete the sequence and explain the pattern: red blue red blue red blue.",
    "Given values = [2, 4, 8, 16], describe a loop that computes their sum without modifying the list.",
    "A network packet is sent, verified, acknowledged, and then the next packet is sent. Describe the bottleneck.",
    "Compare latency and throughput in one concise paragraph, including one practical example.",
    "Respond with a short checklist for diagnosing a slow GPU worker while preserving user data.",
)

_SPECULATION_COUNTERS = (
    "classic_observations",
    "verification_observations",
    "proposed_tokens",
    "accepted_tokens",
)
_WINDOW_COUNTERS = (
    "dispatched_waves",
    "completed_waves",
    "committed_waves",
    "condemned_waves",
    "drained_waves",
    "rejection_collapses",
    "rejected_proposed_tokens",
    "condemned_proposed_tokens",
    "discarded_proposed_tokens",
    "rejected_wave_bytes",
    "tombstone_bytes",
    "discarded_bytes",
)
_CURRENT_RESOURCE_COUNTERS = (
    "current_waves",
    "current_bytes",
    "current_reserved_bytes",
)
_HIGH_WATER_WAVE_COUNTERS = ("high_water_waves", "max_request_waves")
_HIGH_WATER_BYTE_COUNTERS = (
    "high_water_bytes",
    "high_water_reserved_bytes",
    "max_request_bytes",
    "max_request_reserved_bytes",
)

# Student-t critical values for a one-sided 95% interval.  For df > 30 we
# retain the df=30 value, which is slightly conservative.
_ONE_SIDED_T95 = (
    0.0,
    6.3138,
    2.9200,
    2.3534,
    2.1318,
    2.0150,
    1.9432,
    1.8946,
    1.8595,
    1.8331,
    1.8125,
    1.7959,
    1.7823,
    1.7709,
    1.7613,
    1.7531,
    1.7459,
    1.7396,
    1.7341,
    1.7291,
    1.7247,
    1.7207,
    1.7171,
    1.7139,
    1.7109,
    1.7081,
    1.7056,
    1.7033,
    1.7011,
    1.6991,
    1.6973,
)


def _csv_values(raw: str, label: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError(f"{label} cannot be empty")
    return values


def parse_k_sweep(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item) for item in _csv_values(raw, "k sweep"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("k sweep must contain integers") from error
    if any(value < 1 or value > 16 for value in values):
        raise argparse.ArgumentTypeError("k sweep values must be between 1 and 16")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("k sweep must not contain duplicates")
    return values


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def _nonnegative_int(raw: str) -> int:
    value = int(raw)
    if value < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return value


def _finite_nonnegative(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return value


def _window(raw: str) -> int:
    value = int(raw)
    if value < 2 or value > 16:
        raise argparse.ArgumentTypeError("conveyor window must be between 2 and 16")
    return value


def load_corpus(path: Path | str | None) -> tuple[tuple[str, ...], dict[str, Any]]:
    """Load a frozen built-in corpus or a bounded JSON list/object."""

    source = "builtin:draft-quality-corpus-v1"
    if path is None:
        prompts = DEFAULT_CORPUS
    else:
        corpus_path = Path(path)
        document = json.loads(corpus_path.read_text(encoding="utf-8"))
        if isinstance(document, Mapping):
            document = document.get("prompts")
        if not isinstance(document, list):
            raise ValueError("corpus JSON must be a list or an object with prompts")
        prompts = tuple(document)
        source = str(corpus_path.resolve())
    if not 1 <= len(prompts) <= 256:
        raise ValueError("corpus must contain between 1 and 256 prompts")
    normalized: list[str] = []
    for prompt in prompts:
        if (
            not isinstance(prompt, str)
            or not prompt
            or prompt != prompt.strip()
            or len(prompt) > 16_384
        ):
            raise ValueError(
                "corpus prompts must be normalized non-empty strings up to 16384 chars"
            )
        normalized.append(prompt)
    canonical = json.dumps(
        normalized,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return tuple(normalized), {
        "source": source,
        "count": len(normalized),
        "sha256": f"sha256:{hashlib.sha256(canonical).hexdigest()}",
        "prompts": normalized,
    }


def build_balanced_arm_orders(
    *,
    repetitions: int,
    seed: int,
) -> tuple[tuple[str, ...], ...]:
    """Return mirrored arm orders with exact pairwise order balance.

    Each adjacent pair is a rotated ``ABC`` order followed by its reverse
    ``CBA``.  Consequently every arm pair appears once in each direction and
    every arm's average order position is the middle position.
    """

    if (
        isinstance(repetitions, bool)
        or not isinstance(repetitions, int)
        or repetitions < 2
        or repetitions % 2
    ):
        raise ValueError("balanced arm repetitions must be a positive even integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    orders: list[tuple[str, ...]] = []
    base = ARM_NAMES
    for pair_index in range(repetitions // 2):
        offset = (seed + pair_index) % len(base)
        rotated = base[offset:] + base[:offset]
        orders.extend((rotated, tuple(reversed(rotated))))
    return tuple(orders)


def allocate_requests(total: int, repetitions: int) -> tuple[int, ...]:
    if (
        isinstance(total, bool)
        or not isinstance(total, int)
        or total < MIN_MEASURED_REQUESTS_PER_ARM
    ):
        raise ValueError(
            f"each arm needs at least {MIN_MEASURED_REQUESTS_PER_ARM} requests"
        )
    if (
        isinstance(repetitions, bool)
        or not isinstance(repetitions, int)
        or repetitions < 1
        or total < repetitions
    ):
        raise ValueError("repetitions must be positive and not exceed requests")
    quotient, remainder = divmod(total, repetitions)
    return tuple(
        quotient + (1 if index < remainder else 0)
        for index in range(repetitions)
    )


def _counter(mapping: Mapping[str, Any], name: str) -> int:
    value = mapping.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"counter {name} must be a non-negative integer")
    return value


def _optional_counter(mapping: Mapping[str, Any], name: str) -> int | None:
    value = mapping.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def counter_delta(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    name: str,
) -> int:
    previous = _counter(before, name)
    current = _counter(after, name)
    if current < previous:
        raise ValueError(f"counter {name} decreased during one request")
    return current - previous


def snapshot_runtime_counters(engine: Any) -> dict[str, dict[str, Any]]:
    speculation = engine.speculation_stats
    window = engine.speculative_window_stats
    if not isinstance(speculation, Mapping) or not isinstance(window, Mapping):
        raise TypeError("engine counter snapshots must be mappings")
    return {
        "speculation": dict(speculation),
        "window": dict(window),
    }


def derive_request_metrics(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    output_token_count: int,
    speculative: bool,
) -> dict[str, Any]:
    """Derive auditable per-request deltas and route-level ``g``."""

    if (
        isinstance(output_token_count, bool)
        or not isinstance(output_token_count, int)
        or output_token_count < 1
    ):
        raise ValueError("output_token_count must be a positive integer")
    before_speculation = before.get("speculation")
    after_speculation = after.get("speculation")
    before_window = before.get("window")
    after_window = after.get("window")
    if not all(
        isinstance(value, Mapping)
        for value in (
            before_speculation,
            after_speculation,
            before_window,
            after_window,
        )
    ):
        raise TypeError("request snapshots need speculation and window mappings")
    assert isinstance(before_speculation, Mapping)
    assert isinstance(after_speculation, Mapping)
    assert isinstance(before_window, Mapping)
    assert isinstance(after_window, Mapping)

    useful_tokens = output_token_count - 1
    if not speculative:
        return {
            "useful_output_tokens": useful_tokens,
            "route_traversals": None,
            "g_route": None,
            "acceptance_rate": None,
            "draft_route_coverage": None,
            "counter_deltas": {},
            "discard_deltas": {},
            "drained_after_request": all(
                _optional_counter(after_window, name) == 0
                for name in _CURRENT_RESOURCE_COUNTERS
            ),
        }

    speculation_deltas = {
        name: counter_delta(before_speculation, after_speculation, name)
        for name in _SPECULATION_COUNTERS
    }
    window_deltas = {
        name: counter_delta(before_window, after_window, name)
        for name in _WINDOW_COUNTERS
    }
    completed_waves = window_deltas["completed_waves"]
    classic_observations = speculation_deltas["classic_observations"]
    route_traversals = completed_waves + classic_observations
    if useful_tokens > 0 and route_traversals <= 0:
        raise ValueError(
            "a speculative request emitted useful tokens without an observed traversal"
        )
    proposed = speculation_deltas["proposed_tokens"]
    accepted = speculation_deltas["accepted_tokens"]
    if accepted > proposed:
        raise ValueError("accepted token delta cannot exceed proposed token delta")
    acceptance_rate = accepted / proposed if proposed else None
    g_route = useful_tokens / route_traversals if route_traversals else None
    route_coverage = (
        completed_waves / route_traversals if route_traversals else None
    )
    discard_deltas = {
        name: window_deltas[name]
        for name in (
            "condemned_waves",
            "drained_waves",
            "rejection_collapses",
            "rejected_proposed_tokens",
            "condemned_proposed_tokens",
            "discarded_proposed_tokens",
            "rejected_wave_bytes",
            "tombstone_bytes",
            "discarded_bytes",
        )
    }
    return {
        "useful_output_tokens": useful_tokens,
        "route_traversals": route_traversals,
        "g_route": g_route,
        "acceptance_rate": acceptance_rate,
        "draft_route_coverage": route_coverage,
        "counter_deltas": {
            **speculation_deltas,
            **window_deltas,
        },
        "discard_deltas": discard_deltas,
        "drained_after_request": all(
            _counter(after_window, name) == 0
            for name in _CURRENT_RESOURCE_COUNTERS
        ),
    }


def _describe(values: Sequence[float]) -> dict[str, float | int]:
    normalized = sorted(float(value) for value in values)
    if not normalized or any(not math.isfinite(value) for value in normalized):
        raise ValueError("metric samples must be finite and non-empty")

    def percentile(fraction: float) -> float:
        if len(normalized) == 1:
            return normalized[0]
        position = (len(normalized) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return normalized[lower]
        weight = position - lower
        return normalized[lower] * (1 - weight) + normalized[upper] * weight

    return {
        "count": len(normalized),
        "mean": statistics.fmean(normalized),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "min": normalized[0],
        "max": normalized[-1],
    }


def one_sided_mean_interval(values: Sequence[float]) -> dict[str, Any]:
    normalized = tuple(float(value) for value in values)
    if len(normalized) < 2 or any(
        not math.isfinite(value) for value in normalized
    ):
        raise ValueError("one-sided interval requires at least two finite samples")
    mean = statistics.fmean(normalized)
    standard_deviation = statistics.stdev(normalized)
    standard_error = standard_deviation / math.sqrt(len(normalized))
    degrees_of_freedom = len(normalized) - 1
    critical = _ONE_SIDED_T95[min(degrees_of_freedom, 30)]
    margin = critical * standard_error
    return {
        "method": "student-t-one-sided",
        "confidence": GATE_CONFIDENCE,
        "count": len(normalized),
        "degrees_of_freedom": degrees_of_freedom,
        "mean": mean,
        "sample_standard_deviation": standard_deviation,
        "standard_error": standard_error,
        "critical_value": critical,
        "lower": mean - margin,
        "upper": mean + margin,
    }


def classify_quality_gate(
    values: Sequence[float],
    *,
    evidence_exact: bool = True,
    minimum_samples: int = MIN_MEASURED_REQUESTS_PER_ARM,
) -> dict[str, Any]:
    """Apply the pre-registered W=1 useful-token traversal gate."""

    normalized = tuple(float(value) for value in values)
    if (
        not evidence_exact
        or len(normalized) < minimum_samples
        or any(not math.isfinite(value) or value < 0 for value in normalized)
    ):
        return {
            "outcome": "INVALID",
            "valid": False,
            "reason": (
                "evidence_not_exact"
                if not evidence_exact
                else "insufficient_or_invalid_samples"
            ),
            "sample_count": len(normalized),
            "minimum_samples": minimum_samples,
            "interval": None,
            "go_rule": f"one-sided 95% lower >= {GATE_GO_LOWER_BOUND:g}",
            "stop_rule": f"one-sided 95% upper < {GATE_STOP_UPPER_BOUND:g}",
        }
    interval = one_sided_mean_interval(normalized)
    if float(interval["lower"]) >= GATE_GO_LOWER_BOUND:
        outcome = "GO"
        reason = "lower_bound_meets_go_threshold"
    elif float(interval["upper"]) < GATE_STOP_UPPER_BOUND:
        outcome = "STOP"
        reason = "upper_bound_below_stop_threshold"
    else:
        outcome = "GRAY"
        reason = "confidence_interval_crosses_pre_registered_region"
    return {
        "outcome": outcome,
        "valid": True,
        "reason": reason,
        "sample_count": len(normalized),
        "minimum_samples": minimum_samples,
        "interval": interval,
        "go_rule": f"one-sided 95% lower >= {GATE_GO_LOWER_BOUND:g}",
        "stop_rule": f"one-sided 95% upper < {GATE_STOP_UPPER_BOUND:g}",
    }


def classify_sweep_quality_gate(
    scenarios: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Reduce per-k decisions without turning a partial STOP into a no-go."""

    decisions: list[tuple[int, str, bool]] = []
    seen: set[int] = set()
    for scenario in scenarios:
        k = scenario.get("k")
        gate = scenario.get("w1_quality_gate")
        if (
            isinstance(k, bool)
            or not isinstance(k, int)
            or k in seen
            or not isinstance(gate, Mapping)
        ):
            return {
                "outcome": "INCOMPLETE",
                "valid": False,
                "reason": "invalid_or_duplicate_scenario",
            }
        seen.add(k)
        decisions.append(
            (k, str(gate.get("outcome")), gate.get("valid") is True)
        )
    if any(outcome == "GO" and valid for _, outcome, valid in decisions):
        return {
            "outcome": "GO",
            "valid": True,
            "reason": "at_least_one_pre_registered_k_is_go",
            "tested_k": sorted(seen),
        }
    full_sweep = seen == set(DEFAULT_K_SWEEP) and len(decisions) == len(
        DEFAULT_K_SWEEP
    )
    all_valid = bool(decisions) and all(valid for _, _, valid in decisions)
    if not full_sweep or not all_valid:
        return {
            "outcome": "INCOMPLETE",
            "valid": False,
            "reason": (
                "partial_k_sweep" if not full_sweep else "invalid_k_evidence"
            ),
            "tested_k": sorted(seen),
            "required_k": list(DEFAULT_K_SWEEP),
        }
    if all(outcome == "STOP" for _, outcome, _ in decisions):
        return {
            "outcome": "STOP",
            "valid": True,
            "reason": "every_k_in_full_pre_registered_sweep_is_stop",
            "tested_k": sorted(seen),
        }
    return {
        "outcome": "GRAY",
        "valid": True,
        "reason": "full_valid_sweep_has_no_go_and_not_all_stop",
        "tested_k": sorted(seen),
    }


def _provider_snapshot_clean(
    provider: Mapping[str, Any] | None,
    *,
    arm_name: str,
) -> bool:
    if arm_name == "classic":
        return provider is None
    if not isinstance(provider, Mapping):
        return False
    if provider.get("strategy") == "draft-model":
        return (
            provider.get("draftFailures") == 0
            and provider.get("bypassedCalls") == 0
            and provider.get("circuitOpen") is False
            and provider.get("cachedKvBytes") == 0
        )
    return provider.get("strategy") == "ngram"


def validate_run_evidence(
    run: Mapping[str, Any],
    *,
    expected_window: int,
    expected_inflight_bytes: int,
    expected_requests: int,
) -> dict[str, Any]:
    """Fail closed on parity, resource bounds, drain, provider and shutdown."""

    arm_name = run.get("arm")
    if arm_name not in ARM_NAMES:
        raise ValueError("run arm is invalid")
    samples_value = run.get("samples")
    warmups_value = run.get("warmups")
    samples = samples_value if isinstance(samples_value, list) else []
    warmups = warmups_value if isinstance(warmups_value, list) else []
    exact_hashes = (
        len(samples) == expected_requests
        and all(sample.get("token_hash_exact") is True for sample in samples)
        and all(sample.get("token_hash_exact") is True for sample in warmups)
    )
    request_drain = all(
        sample.get("drained_after_request") is True for sample in samples
    ) and all(sample.get("drained_after_request") is True for sample in warmups)

    stats_value = run.get("speculative_window_stats")
    stats = stats_value if isinstance(stats_value, Mapping) else {}
    configured_waves = _optional_counter(stats, "configured_waves_per_request")
    configured_bytes = _optional_counter(stats, "configured_bytes_per_request")
    wave_values = tuple(
        _optional_counter(stats, name) for name in _HIGH_WATER_WAVE_COUNTERS
    )
    byte_values = tuple(
        _optional_counter(stats, name) for name in _HIGH_WATER_BYTE_COUNTERS
    )
    current_values = tuple(
        _optional_counter(stats, name) for name in _CURRENT_RESOURCE_COUNTERS
    )
    wave_bounds = (
        configured_waves == expected_window
        and all(value is not None and value <= expected_window for value in wave_values)
    )
    # A zero configured byte ceiling disables only the secondary reservation
    # cap; real W=1 VERIFY frames still transfer bytes.  Therefore zero must
    # never be interpreted as "no transport".  A positive conveyor ceiling is
    # the only case in which the observed byte high-water values are bounded.
    byte_bounds = configured_bytes == expected_inflight_bytes and all(
        value is not None
        and (
            value <= expected_inflight_bytes
            if expected_inflight_bytes > 0
            else True
        )
        for value in byte_values
    )
    resource_limits_exact = wave_bounds and byte_bounds
    drained_to_zero = request_drain and all(value == 0 for value in current_values)

    speculation_value = run.get("speculation_stats")
    speculation = (
        speculation_value if isinstance(speculation_value, Mapping) else {}
    )
    proposed = _optional_counter(speculation, "proposed_tokens")
    dispatched = _optional_counter(stats, "dispatched_waves")
    completed = _optional_counter(stats, "completed_waves")
    high_water = wave_values[0]
    max_request = wave_values[1]
    if arm_name == "classic":
        execution_exercised = (
            speculation.get("configured") is False
            and (proposed in (None, 0))
            and (dispatched in (None, 0))
            and (completed in (None, 0))
        )
        all_returns_accounted = dispatched in (None, 0) and completed in (None, 0)
    elif arm_name == "draft_w1":
        execution_exercised = (
            proposed is not None
            and proposed > 0
            and completed is not None
            and completed > 0
            and high_water is not None
            and high_water <= 1
            and max_request is not None
            and max_request <= 1
        )
        all_returns_accounted = dispatched is not None and dispatched == completed
    else:
        execution_exercised = (
            proposed is not None
            and proposed > 0
            and completed is not None
            and completed > 0
            and high_water is not None
            and high_water > 1
            and max_request is not None
            and max_request > 1
        )
        all_returns_accounted = dispatched is not None and dispatched == completed

    shutdown_value = run.get("shutdown")
    shutdown = shutdown_value if isinstance(shutdown_value, Mapping) else {}
    clean_shutdown = (
        shutdown.get("exact") is True
        and shutdown.get("close_error") is None
        and shutdown.get("hard_fallback_used") is False
        and shutdown.get("cleanup_errors") == []
        and run.get("healthy_before_shutdown") is True
    )
    provider_clean = _provider_snapshot_clean(
        speculation.get("provider")
        if isinstance(speculation.get("provider"), Mapping)
        else None,
        arm_name=str(arm_name),
    )
    checks = {
        "exact_output_hashes": exact_hashes,
        "execution_exercised": execution_exercised,
        "all_returns_accounted": all_returns_accounted,
        "resource_limits_exact": resource_limits_exact,
        "drained_to_zero": drained_to_zero,
        "provider_clean": provider_clean,
        "clean_shutdown": clean_shutdown,
    }
    return {
        "exact": all(checks.values()),
        **checks,
        "expected_window": expected_window,
        "expected_inflight_bytes": expected_inflight_bytes,
        "expected_requests": expected_requests,
        "violations": [name for name, value in checks.items() if not value],
    }


def aggregate_arm_runs(
    runs: Sequence[Mapping[str, Any]],
    *,
    arm_name: str,
    expected_requests: int,
) -> dict[str, Any]:
    matching = [run for run in runs if run.get("arm") == arm_name]
    if not matching:
        raise ValueError(f"no runs found for arm {arm_name}")
    samples = [
        sample
        for run in matching
        for sample in run.get("samples", ())
        if isinstance(sample, Mapping)
    ]
    evidence_items = [
        run.get("evidence")
        for run in matching
        if isinstance(run.get("evidence"), Mapping)
    ]
    integrity_checks = (
        "exact_output_hashes",
        "all_returns_accounted",
        "resource_limits_exact",
        "drained_to_zero",
        "provider_clean",
        "clean_shutdown",
    )
    integrity_exact = len(evidence_items) == len(matching) and all(
        evidence.get("exact") is True
        or all(evidence.get(check) is True for check in integrity_checks)
        for evidence in evidence_items
    )
    execution_flags = [
        evidence.get("exact") is True
        or evidence.get("execution_exercised") is True
        for evidence in evidence_items
    ]
    # A realistic mixed corpus may not give a linear drafter enough context on
    # every request partition.  That is a measured zero-coverage sample, not
    # corrupt evidence.  Draft execution must therefore be exercised by the
    # arm as a whole, while every run must still satisfy all integrity checks.
    execution_exercised = (
        all(execution_flags)
        if arm_name == "classic"
        else any(execution_flags)
    )
    exact = (
        len(samples) == expected_requests
        and integrity_exact
        and execution_exercised
    )
    total_deltas: dict[str, int] = {}
    discard_totals: dict[str, int] = {}
    for sample in samples:
        for name, value in sample.get("counter_deltas", {}).items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"invalid aggregated counter {name}")
            total_deltas[name] = total_deltas.get(name, 0) + value
        for name, value in sample.get("discard_deltas", {}).items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"invalid discard counter {name}")
            discard_totals[name] = discard_totals.get(name, 0) + value
    proposed = total_deltas.get("proposed_tokens", 0)
    accepted = total_deltas.get("accepted_tokens", 0)
    completed = total_deltas.get("completed_waves", 0)
    classic = total_deltas.get("classic_observations", 0)
    traversal_total = completed + classic
    g_values = [
        float(sample["g_route"])
        for sample in samples
        if sample.get("g_route") is not None
    ]
    return {
        "arm": arm_name,
        "run_count": len(matching),
        "measured_requests": len(samples),
        "evidence_exact": exact,
        "integrity_exact": integrity_exact,
        "execution_exercised": execution_exercised,
        "g_route": _describe(g_values) if g_values else None,
        "g_route_samples": g_values,
        "acceptance_rate": accepted / proposed if proposed else None,
        "draft_route_coverage": (
            completed / traversal_total if traversal_total else None
        ),
        "counter_deltas": total_deltas,
        "discard_deltas": discard_totals,
        "latency": {
            "ttft_ms": _describe([float(sample["ttft_ms"]) for sample in samples]),
            "tpot_ms": _describe([float(sample["tpot_ms"]) for sample in samples]),
            "total_ms": _describe([float(sample["total_ms"]) for sample in samples]),
            "tokens_per_second": _describe(
                [float(sample["tokens_per_second"]) for sample in samples]
            ),
        },
    }


def _encode_prompt(tokenizer: Any, prompt: str, maximum_tokens: int) -> Any:
    import torch

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
    )["input_ids"].to(dtype=torch.long, device="cpu")
    if encoded.ndim != 2 or int(encoded.shape[0]) != 1 or int(encoded.shape[1]) < 1:
        raise ValueError("tokenizer returned an invalid prompt tensor")
    if int(encoded.shape[1]) > maximum_tokens:
        encoded = encoded[:, :maximum_tokens]
    return encoded.contiguous()


def _greedy_reference(model: Any, input_ids: Any, output_tokens: int) -> tuple[int, ...]:
    import torch

    tokens: list[int] = []
    with torch.inference_mode():
        output = model(input_ids=input_ids, use_cache=True)
        cache = output.past_key_values
        for index in range(output_tokens):
            token = torch.argmax(output.logits[:, -1, :], dim=-1)
            tokens.append(int(token.item()))
            if index + 1 < output_tokens:
                output = model(
                    input_ids=token[:, None],
                    past_key_values=cache,
                    use_cache=True,
                )
                cache = output.past_key_values
    return tuple(tokens)


def _prepare_model_and_references(
    args: argparse.Namespace,
    prompts: Sequence[str],
) -> dict[str, Any]:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    from .device import resolve_torch_execution_device
    from .model import (
        load_tokenizer,
        model_snapshot_identity,
        resolve_model_snapshot,
    )

    snapshot = resolve_model_snapshot(args.model, args.revision)
    execution_device = resolve_torch_execution_device(args.device)
    tokenizer = load_tokenizer(snapshot)
    configuration = AutoConfig.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
    )
    total_layers = int(getattr(configuration, "num_hidden_layers", 0))
    maximum_context = int(
        getattr(configuration, "max_position_embeddings", 0) or 0
    )
    if total_layers < 2:
        raise ValueError("target model config lacks a usable layer count")
    if args.stage_count > total_layers:
        raise ValueError("stage count exceeds target model layers")
    inputs = tuple(
        _encode_prompt(tokenizer, prompt, args.prompt_tokens)
        for prompt in prompts
    )
    if maximum_context and any(
        int(input_ids.shape[1]) + args.output_tokens > maximum_context
        for input_ids in inputs
    ):
        raise ValueError("a corpus prompt plus output exceeds target context")

    torch.set_num_threads(args.threads_per_stage)
    reference_model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.float32,
    ).eval()
    try:
        references = tuple(
            _greedy_reference(reference_model, input_ids, args.output_tokens)
            for input_ids in inputs
        )
    finally:
        del reference_model
        gc.collect()
    return {
        "snapshot": snapshot,
        "tokenizer": tokenizer,
        "configuration": configuration,
        "total_layers": total_layers,
        "inputs": inputs,
        "references": references,
        "reference_hashes": tuple(_token_hash(tokens) for tokens in references),
        "snapshot_identity": model_snapshot_identity(args.model, args.revision),
        "execution_device": {
            "requested": args.device,
            "resolved": str(execution_device.device),
            "kind": execution_device.kind,
            "backend": execution_device.backend,
            "name": execution_device.name,
            "accelerated": execution_device.accelerated,
            "total_memory_bytes": execution_device.total_memory_bytes,
        },
    }


def _build_draft_provider(
    args: argparse.Namespace,
    *,
    k: int,
    target_tokenizer: Any,
) -> Any:
    if args.provider == "ngram":
        from .speculation import NgramDraftProvider

        return NgramDraftProvider(
            max_draft_tokens=k,
            min_match_tokens=args.ngram_min_match_tokens,
            max_match_tokens=args.ngram_max_match_tokens,
        )

    from .draft_model import (
        LocalDraftModelRuntimeConfig,
        load_local_draft_model_provider,
    )

    config = LocalDraftModelRuntimeConfig(
        source=args.draft_model_source,
        revision=args.draft_model_revision,
        artifact_identity=args.draft_model_artifact_identity,
        canonical_source=args.draft_model_canonical_source,
        canonical_revision=args.draft_model_canonical_revision,
        device=args.draft_model_device,
        dtype=args.draft_model_dtype,
        parameter_bytes=args.draft_model_parameter_bytes,
        memory_reservation_bytes=args.draft_model_memory_reservation_bytes,
        max_draft_tokens=k,
    )
    return load_local_draft_model_provider(
        config,
        target_tokenizer=target_tokenizer,
        max_cached_requests=1,
    )


def _run_arm(
    *,
    args: argparse.Namespace,
    model_snapshot: str,
    boundaries: tuple[int, ...],
    target_tokenizer: Any,
    inputs: Sequence[Any],
    reference_hashes: Sequence[str],
    prompt_indices: Sequence[int],
    k: int,
    arm_name: str,
    repetition: int,
    order_index: int,
    client_id_seed: int,
) -> dict[str, Any]:
    from .engine import (
        DistributedPipelineEngine,
        GenerationInput,
        PipelineEngineConfig,
        PipelineShutdownError,
    )
    from .protocol import TensorCodec

    speculative = arm_name != "classic"
    window = (
        args.conveyor_window
        if arm_name == "draft_conveyor"
        else 1
    )
    inflight_bytes = (
        args.inflight_bytes
        if arm_name == "draft_conveyor"
        else 0
    )
    provider = (
        _build_draft_provider(args, k=k, target_tokenizer=target_tokenizer)
        if speculative
        else None
    )
    config = PipelineEngineConfig(
        model_name=model_snapshot,
        boundaries=boundaries,
        codec=TensorCodec.FP32,
        threads_per_stage=args.threads_per_stage,
        device=args.device,
        startup_timeout_seconds=args.timeout_seconds,
        socket_timeout_seconds=args.timeout_seconds,
        one_way_delay_ms=args.one_way_delay_ms,
        bandwidth_mbps=args.bandwidth_mbps,
        max_active_sequences=1,
        max_pending_requests=2,
        speculative_max_draft_tokens=k if speculative else 0,
        speculative_inflight_waves=window,
        speculative_inflight_bytes=inflight_bytes,
        speculation_minimum_speedup=1.0,
        speculation_probe=False,
        root_batch_window_ms=0.0,
        route_probe_interval_seconds=0.0,
    )
    startup_started = time.perf_counter()
    engine = DistributedPipelineEngine(
        config,
        draft_provider=provider,
        speculation_controller=_fixed_controller(k) if speculative else None,
    )
    startup_ms = (time.perf_counter() - startup_started) * 1_000.0
    samples: list[dict[str, Any]] = []
    warmups: list[dict[str, Any]] = []
    close_error: str | None = None
    healthy_before_shutdown = False
    final_speculation: dict[str, Any] = {}
    final_window: dict[str, Any] = {}
    request_sequence = [
        *(
            prompt_indices[index % len(prompt_indices)]
            for index in range(args.warmups_per_run)
        ),
        *prompt_indices,
    ]
    try:
        for local_index, prompt_index in enumerate(request_sequence):
            before = snapshot_runtime_counters(engine)
            wall_started = time.perf_counter()
            output = engine.generate(
                [
                    GenerationInput(
                        client_id=client_id_seed + local_index,
                        input_ids=inputs[prompt_index].clone(),
                        max_new_tokens=args.output_tokens,
                        eos_token_ids=frozenset(),
                    )
                ]
            )[0]
            wall_ms = (time.perf_counter() - wall_started) * 1_000.0
            after = snapshot_runtime_counters(engine)
            metrics = derive_request_metrics(
                before,
                after,
                output_token_count=len(output.token_ids),
                speculative=speculative,
            )
            token_sha256 = _token_hash(output.token_ids)
            record = {
                "prompt_index": prompt_index,
                "reference_sha256": reference_hashes[prompt_index],
                "token_sha256": token_sha256,
                "token_hash_exact": token_sha256 == reference_hashes[prompt_index],
                "finish_reason": output.finish_reason,
                "output_tokens": len(output.token_ids),
                "ttft_ms": float(output.ttft_ms),
                "tpot_ms": float(output.tpot_ms),
                "total_ms": float(output.total_ms),
                "wall_ms": wall_ms,
                "tokens_per_second": (
                    len(output.token_ids) * 1_000.0 / float(output.total_ms)
                    if float(output.total_ms) > 0
                    else 0.0
                ),
                **metrics,
            }
            if local_index < args.warmups_per_run:
                warmups.append(record)
            else:
                record["sample"] = len(samples)
                samples.append(record)
        final_speculation = dict(engine.speculation_stats)
        final_window = dict(engine.speculative_window_stats)
        healthy_before_shutdown = bool(engine.healthy)
    finally:
        try:
            engine.close()
        except PipelineShutdownError as error:
            close_error = str(error)
    shutdown = _shutdown_evidence(engine, close_error)
    return {
        "arm": arm_name,
        "k": k,
        "window": window,
        "inflight_bytes": inflight_bytes,
        "repetition": repetition,
        "order_index": order_index,
        "startup_ms_excluded_from_request_latency": startup_ms,
        "healthy_before_shutdown": healthy_before_shutdown,
        "speculation_stats": final_speculation,
        "speculative_window_stats": final_window,
        "shutdown": shutdown,
        "warmups": warmups,
        "samples": samples,
    }


def run_runtime_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    prompts, corpus = load_corpus(args.corpus_json)
    prepared = _prepare_model_and_references(args, prompts)
    inputs = prepared["inputs"]
    reference_hashes = prepared["reference_hashes"]
    boundaries = _balanced_boundaries(
        int(prepared["total_layers"]),
        args.stage_count,
    )
    request_counts = allocate_requests(
        args.requests_per_arm,
        args.order_repetitions,
    )
    cursor = 0
    prompt_indices_by_repetition: list[tuple[int, ...]] = []
    for count in request_counts:
        indices = tuple(
            (cursor + offset) % len(prompts)
            for offset in range(count)
        )
        prompt_indices_by_repetition.append(indices)
        cursor += count

    scenarios: list[dict[str, Any]] = []
    all_valid = True
    for k_index, k in enumerate(args.k_sweep):
        runs: list[dict[str, Any]] = []
        # Rotate the already mirrored design per k without changing its
        # pairwise balance.
        k_orders = build_balanced_arm_orders(
            repetitions=args.order_repetitions,
            seed=args.seed + k_index,
        )
        for repetition, order in enumerate(k_orders):
            for order_index, arm_name in enumerate(order):
                print(
                    (
                        f"k={k} repetition={repetition} arm={arm_name}: "
                        "loading real single-host distributed pipeline..."
                    ),
                    file=__import__("sys").stderr,
                    flush=True,
                )
                run = _run_arm(
                    args=args,
                    model_snapshot=str(prepared["snapshot"]),
                    boundaries=boundaries,
                    target_tokenizer=prepared["tokenizer"],
                    inputs=inputs,
                    reference_hashes=reference_hashes,
                    prompt_indices=prompt_indices_by_repetition[repetition],
                    k=k,
                    arm_name=arm_name,
                    repetition=repetition,
                    order_index=order_index,
                    client_id_seed=(
                        (k_index + 1) * 10_000_000
                        + repetition * 100_000
                        + order_index * 10_000
                    ),
                )
                expected_window = (
                    args.conveyor_window
                    if arm_name == "draft_conveyor"
                    else 1
                )
                expected_bytes = (
                    args.inflight_bytes
                    if arm_name == "draft_conveyor"
                    else 0
                )
                run["evidence"] = validate_run_evidence(
                    run,
                    expected_window=expected_window,
                    expected_inflight_bytes=expected_bytes,
                    expected_requests=request_counts[repetition],
                )
                runs.append(run)
        aggregates = {
            arm_name: aggregate_arm_runs(
                runs,
                arm_name=arm_name,
                expected_requests=args.requests_per_arm,
            )
            for arm_name in ARM_NAMES
        }
        w1 = aggregates["draft_w1"]
        gate = classify_quality_gate(
            w1["g_route_samples"],
            evidence_exact=bool(w1["evidence_exact"]),
        )
        scenario_valid = (
            all(bool(value["evidence_exact"]) for value in aggregates.values())
            and gate["valid"] is True
        )
        all_valid = all_valid and scenario_valid
        scenarios.append(
            {
                "k": k,
                "arm_order": [list(order) for order in k_orders],
                "requests_by_repetition": list(request_counts),
                "runs": runs,
                "aggregates": aggregates,
                "w1_quality_gate": gate,
                "valid": scenario_valid,
            }
        )
    go_candidates = [
        {
            "k": scenario["k"],
            "lower": scenario["w1_quality_gate"]["interval"]["lower"],
            "mean": scenario["w1_quality_gate"]["interval"]["mean"],
        }
        for scenario in scenarios
        if scenario["w1_quality_gate"]["outcome"] == "GO"
    ]
    selected = (
        max(go_candidates, key=lambda item: (item["lower"], item["mean"], -item["k"]))
        if go_candidates
        else None
    )
    sweep_gate = classify_sweep_quality_gate(scenarios)
    configuration = prepared["configuration"]
    execution_device = prepared["execution_device"]
    return {
        "evidence_class": EVIDENCE_CLASS,
        "claim_boundary": {
            "real_target_runtime": True,
            "real_native_draft_provider": True,
            "physical_multi_pc": False,
            "multi_host_gpu": False,
            "single_host_gpu_measured": execution_device["accelerated"] is True,
            "wan_measured": False,
            "statement": (
                "Real Mycellios target and native drafter on one host. "
                "The resolved device is recorded below; this is never "
                "physical multi-PC or WAN throughput evidence."
            ),
        },
        "model": {
            "requested": args.model,
            "revision": args.revision,
            "snapshot_identity_uint64": prepared["snapshot_identity"],
            "architecture": str(
                getattr(configuration, "model_type", type(configuration).__name__)
            ),
            "layers": prepared["total_layers"],
            "stage_count": args.stage_count,
            "boundaries": list(boundaries),
            "output_tokens_per_request": args.output_tokens,
        },
        "corpus": corpus,
        "references": {
            "type": "monolithic_greedy_hash_only",
            "hash_scheme": "gdlp-output-token-ids-v1",
            "timings_recorded": False,
            "items": [
                {
                    "prompt_index": index,
                    "prompt_tokens": int(inputs[index].shape[1]),
                    "output_tokens": len(prepared["references"][index]),
                    "token_sha256": reference_hash,
                }
                for index, reference_hash in enumerate(reference_hashes)
            ],
        },
        "configuration": {
            "provider": args.provider,
            "k_sweep": list(args.k_sweep),
            "arms": list(ARM_NAMES),
            "conveyor_window": args.conveyor_window,
            "conveyor_inflight_bytes": args.inflight_bytes,
            "requests_per_arm": args.requests_per_arm,
            "order_repetitions": args.order_repetitions,
            "warmups_per_run": args.warmups_per_run,
            "balanced_order_contract": (
                "Each order is paired with its exact reverse; every arm pair "
                "appears once in each direction."
            ),
            "one_way_delay_ms": args.one_way_delay_ms,
            "emulated_rtt_ms": args.one_way_delay_ms * 2,
            "bandwidth_mbps": args.bandwidth_mbps,
            "device": execution_device,
            "threads_per_stage": args.threads_per_stage,
            "seed": args.seed,
            "g_route_formula": (
                "(len(output.token_ids)-1) / "
                "(delta completed_waves + delta classic_observations)"
            ),
            "w_gt_1_forbidden_formula": (
                "accepted_tokens + verification_observations"
            ),
            "pre_registered_gate": {
                "arm": "draft_w1",
                "confidence": GATE_CONFIDENCE,
                "method": "student-t-one-sided",
                "go": f"lower >= {GATE_GO_LOWER_BOUND:g}",
                "stop": f"upper < {GATE_STOP_UPPER_BOUND:g}",
                "otherwise": "GRAY",
                "minimum_measured_requests": MIN_MEASURED_REQUESTS_PER_ARM,
            },
        },
        "scenarios": scenarios,
        "selected_go_candidate": selected,
        "sweep_quality_gate": sweep_gate,
        "success": all_valid,
        "limits": [
            "Every pipeline stage is a process on one physical host.",
            "Delay and bandwidth are configured loopback emulation inputs.",
            "Reference generation establishes only exact token hashes.",
            "A STOP or GRAY quality decision is a valid measurement outcome.",
            "Physical performance claims require the separate two-host GPU gate.",
        ],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    parser = argparse.ArgumentParser(
        description=(
            "Measure real native n-gram or sibling-model draft quality with "
            "classic, W=1 and W>1 target-runtime arms."
        )
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--provider", choices=("ngram", "draft-model"), default="ngram")
    parser.add_argument(
        "--k-sweep",
        type=parse_k_sweep,
        default=DEFAULT_K_SWEEP,
    )
    parser.add_argument(
        "--conveyor-window",
        type=_window,
        default=DEFAULT_CONVEYOR_WINDOW,
    )
    parser.add_argument(
        "--inflight-bytes",
        type=_positive_int,
        default=DEFAULT_INFLIGHT_BYTES,
    )
    parser.add_argument("--corpus-json", type=Path)
    parser.add_argument("--prompt-tokens", type=_positive_int, default=128)
    parser.add_argument("--output-tokens", type=_positive_int, default=40)
    parser.add_argument(
        "--requests-per-arm",
        type=_positive_int,
        default=MIN_MEASURED_REQUESTS_PER_ARM,
    )
    parser.add_argument("--order-repetitions", type=_positive_int, default=2)
    parser.add_argument("--warmups-per-run", type=_nonnegative_int, default=0)
    parser.add_argument("--stage-count", type=_positive_int, default=3)
    parser.add_argument("--threads-per-stage", type=_positive_int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--one-way-delay-ms",
        type=_finite_nonnegative,
        default=12.5,
    )
    parser.add_argument(
        "--bandwidth-mbps",
        type=_finite_nonnegative,
        default=0.0,
    )
    parser.add_argument("--timeout-seconds", type=_positive_int, default=180)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--ngram-min-match-tokens", type=_positive_int, default=2)
    parser.add_argument("--ngram-max-match-tokens", type=_positive_int, default=8)
    parser.add_argument("--draft-model-source")
    parser.add_argument("--draft-model-revision")
    parser.add_argument("--draft-model-artifact-identity")
    parser.add_argument("--draft-model-canonical-source")
    parser.add_argument("--draft-model-canonical-revision")
    parser.add_argument("--draft-model-parameter-bytes", type=_positive_int)
    parser.add_argument("--draft-model-memory-reservation-bytes", type=_positive_int)
    parser.add_argument("--draft-model-device", default="auto")
    parser.add_argument(
        "--draft-model-dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path(f"benchmarks/draft-quality-runtime-{timestamp}.json"),
    )
    parser.add_argument("--compact-json", action="store_true")
    args = parser.parse_args(argv)
    if args.requests_per_arm < MIN_MEASURED_REQUESTS_PER_ARM:
        parser.error(
            f"requests-per-arm must be at least {MIN_MEASURED_REQUESTS_PER_ARM}"
        )
    if args.order_repetitions < 2 or args.order_repetitions % 2:
        parser.error("order-repetitions must be an even integer of at least 2")
    if args.order_repetitions > args.requests_per_arm:
        parser.error("order-repetitions cannot exceed requests-per-arm")
    if args.output_tokens < max(args.k_sweep) * 2 + 2:
        parser.error("output-tokens must be at least 2*max(k)+2")
    if args.inflight_bytes > 1024 * 1024 * 1024:
        parser.error("inflight-bytes cannot exceed 1 GiB")
    if args.ngram_max_match_tokens < args.ngram_min_match_tokens:
        parser.error("ngram-max-match-tokens must be >= ngram-min-match-tokens")
    required_draft_fields = (
        args.draft_model_source,
        args.draft_model_artifact_identity,
        args.draft_model_parameter_bytes,
        args.draft_model_memory_reservation_bytes,
    )
    supplied_draft_fields = (
        *required_draft_fields,
        args.draft_model_revision,
        args.draft_model_canonical_source,
        args.draft_model_canonical_revision,
    )
    if args.provider == "draft-model" and any(
        value is None for value in required_draft_fields
    ):
        parser.error(
            "draft-model requires source, artifact identity, parameter bytes "
            "and memory reservation bytes"
        )
    if args.provider == "ngram" and any(
        value is not None for value in supplied_draft_fields
    ):
        parser.error("draft-model coordinates require --provider draft-model")
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
        print(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=None if args.compact_json else 2,
                sort_keys=True,
                allow_nan=False,
            )
        )
        publish_json_no_overwrite(args.json_out, result)
        print(
            f"Published immutable evidence: {args.json_out}",
            file=__import__("sys").stderr,
        )
        return 0 if result["success"] else 2
    except Exception as error:  # noqa: BLE001 - benchmark CLI must fail closed
        print(
            (
                "Draft quality runtime benchmark failed closed: "
                f"{type(error).__name__}: {error}"
            ),
            file=__import__("sys").stderr,
        )
        return 2


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    raise SystemExit(main())
