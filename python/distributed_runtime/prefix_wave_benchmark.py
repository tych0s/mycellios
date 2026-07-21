"""Canonical theoretical benchmark for exact prefix-segment tree waves.

This report composes three independent pure models:

* :mod:`prefix_segment_schedule` supplies compressed-trie work conservation;
* :mod:`prefix_wave_latency` supplies a deterministic resource/event timeline;
* :mod:`prefix_cow_projection` supplies exact paged-KV block accounting.

Every number is calculated from inputs embedded in the report.  Nothing in
this module measures wall-clock time, a GPU, a network, throughput, or
tokens-per-second performance.  In particular, exact batching amortizes only
launch and record overhead here: token compute remains linear in total work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from .prefix_cow_projection import project_prefix_closed_cow_kv
from .prefix_segment_schedule import plan_prefix_segment_schedule
from .prefix_wave_latency import (
    PrefixWaveCostModel,
    PrefixWaveSimulation,
    PrefixWaveStrategy,
    optimize_stream_chunking,
    simulate_prefix_wave,
)


SCHEMA = "gdlp-prefix-wave-theoretical-benchmark/2"
EVIDENCE_CLASS = "theoretical_simulation"
CAPABILITY_CONFIG_SCHEMA = "gdlp-prefix-wave-deployment-capabilities/1"
CAP_FLAT_LEGACY_V6 = "flat-legacy-v6"
CAP_FLAT_BATCHED_EXACT_V1 = "flat-batched-exact-v1"
CAP_PREFIX_WAVE_V7 = "prefix-wave-v7"
CAP_PREFIX_MONOLITHIC_EXACT_BATCH_V1 = "prefix-monolithic-exact-batch-v1"
CAP_PREFIX_STREAMING_EXACT_BATCH_V1 = "prefix-streaming-exact-batch-v1"
KNOWN_DEPLOYMENT_CAPABILITIES = (
    CAP_FLAT_LEGACY_V6,
    CAP_FLAT_BATCHED_EXACT_V1,
    CAP_PREFIX_WAVE_V7,
    CAP_PREFIX_MONOLITHIC_EXACT_BATCH_V1,
    CAP_PREFIX_STREAMING_EXACT_BATCH_V1,
)
DEFAULT_DEPLOYMENT_CAPABILITIES = (CAP_FLAT_LEGACY_V6,)
STRATEGY_ORDER = (
    PrefixWaveStrategy.FLAT_LEGACY,
    PrefixWaveStrategy.FLAT_BATCHED,
    PrefixWaveStrategy.MONOLITHIC_PACKED,
    PrefixWaveStrategy.STREAMING_SEGMENTED,
)
STRATEGY_CAPABILITY_REQUIREMENTS = {
    PrefixWaveStrategy.FLAT_LEGACY: (CAP_FLAT_LEGACY_V6,),
    PrefixWaveStrategy.FLAT_BATCHED: (CAP_FLAT_BATCHED_EXACT_V1,),
    PrefixWaveStrategy.MONOLITHIC_PACKED: (
        CAP_PREFIX_WAVE_V7,
        CAP_PREFIX_MONOLITHIC_EXACT_BATCH_V1,
    ),
    PrefixWaveStrategy.STREAMING_SEGMENTED: (
        CAP_PREFIX_WAVE_V7,
        CAP_PREFIX_STREAMING_EXACT_BATCH_V1,
    ),
}
SHARED_ROOT_TOKENS = 1
STREAM_GROUPS_PER_RECORD = 1
KV_BLOCK_TOKENS = 16
KV_BYTES_PER_BLOCK = 2_097_152
KV_PARENT_BASE_TOKENS = 4_096
KV_PENDING_TOKEN = 0
FLOAT_DIGITS = 6


def _scenarios() -> tuple[tuple[str, tuple[tuple[int, ...], ...]], ...]:
    shared8x8 = tuple(
        (100, 101, 102, 103, 104, 105, 200 + branch, 300 + branch)
        for branch in range(8)
    )
    hierarchical = tuple(
        (
            400,
            401,
            *(
                500 + bit_index * 2 + bit
                for bit_index, bit in enumerate(
                    tuple((value >> shift) & 1 for shift in (3, 2, 1, 0))
                )
            ),
        )
        for value in range(16)
    )
    clustered = (
        (50, 51, 52, 60, 70),
        (50, 51, 52, 60, 71),
        (50, 51, 52, 61, 72),
        (50, 51, 53, 62),
        (50, 51, 53, 63),
        (50, 54, 64),
    )
    disjoint = tuple(
        tuple(1_000 + branch * 16 + offset for offset in range(8))
        for branch in range(8)
    )
    return (
        ("shared8x8", shared8x8),
        ("hierarchical16x6", hierarchical),
        ("clusteredVariable", clustered),
        ("disjoint8x8", disjoint),
    )


def _profiles() -> tuple[tuple[str, PrefixWaveCostModel], ...]:
    compute = (0.28, 0.31, 0.29, 0.30)
    launches = (0.08, 0.08, 0.08, 0.08)
    common = {
        "stage_compute_ms_per_token": compute,
        "stage_kernel_launch_ms": launches,
        "activation_bytes_per_token": 8_192,
    }
    return (
        (
            "lan",
            PrefixWaveCostModel(
                **common,
                hop_one_way_propagation_ms=(0.15, 0.15, 0.15),
                hop_bandwidth_mbps=(2_500.0, 2_500.0, 2_500.0),
                return_one_way_propagation_ms=0.2,
                return_bandwidth_mbps=2_500.0,
            ),
        ),
        (
            "domesticWanDirect",
            PrefixWaveCostModel(
                **common,
                hop_one_way_propagation_ms=(5.0, 5.0, 5.0),
                hop_bandwidth_mbps=(300.0, 300.0, 300.0),
                return_one_way_propagation_ms=15.0,
                return_bandwidth_mbps=300.0,
            ),
        ),
        (
            "domesticWanConservative",
            PrefixWaveCostModel(
                **common,
                hop_one_way_propagation_ms=(20.0, 20.0, 20.0),
                hop_bandwidth_mbps=(50.0, 50.0, 50.0),
                return_one_way_propagation_ms=60.0,
                return_bandwidth_mbps=50.0,
            ),
        ),
    )


def select_fastest_prefix_wave_strategy(
    flat_legacy: PrefixWaveSimulation,
    flat_batched: PrefixWaveSimulation,
    monolithic_packed: PrefixWaveSimulation,
    streaming_segmented: PrefixWaveSimulation,
    *,
    eligible_strategies: Sequence[PrefixWaveStrategy | str] | None = None,
) -> PrefixWaveSimulation:
    """Select minimum makespan from an explicit eligibility set.

    With no filter this is the theoretical selector.  A deployment decision
    must pass the strategies whose required capabilities are present in the
    route-wide manifest/config intersection.  Exact ties follow
    :data:`STRATEGY_ORDER`, so an uncertified strategy can never become
    deployable merely because its simulated latency is lower.
    """

    simulations = (
        (PrefixWaveStrategy.FLAT_LEGACY, flat_legacy),
        (PrefixWaveStrategy.FLAT_BATCHED, flat_batched),
        (PrefixWaveStrategy.MONOLITHIC_PACKED, monolithic_packed),
        (PrefixWaveStrategy.STREAMING_SEGMENTED, streaming_segmented),
    )
    for expected, simulation in simulations:
        if not isinstance(simulation, PrefixWaveSimulation):
            raise TypeError("strategy selector requires PrefixWaveSimulation inputs")
        if simulation.strategy is not expected:
            raise ValueError(
                f"expected {expected.value}, received {simulation.strategy.value}"
            )
        if not math.isfinite(simulation.makespan_ms) or simulation.makespan_ms < 0:
            raise ValueError("strategy makespan must be finite and non-negative")
    if eligible_strategies is None:
        eligible = set(STRATEGY_ORDER)
    else:
        if isinstance(eligible_strategies, (str, bytes)):
            raise TypeError("eligible_strategies must be a sequence of strategies")
        eligible = set()
        for value in eligible_strategies:
            try:
                eligible.add(PrefixWaveStrategy(value))
            except (TypeError, ValueError) as error:
                raise ValueError(f"unknown eligible strategy {value!r}") from error
    if not eligible:
        raise ValueError("at least one strategy must be eligible")
    return min(
        (
            simulation
            for expected, simulation in simulations
            if expected in eligible
        ),
        key=lambda simulation: (
            simulation.makespan_ms,
            STRATEGY_ORDER.index(simulation.strategy),
        ),
    )


def _normalize_deployment_capabilities(
    values: Sequence[str],
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("deployment_capabilities must be a sequence of strings")
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise TypeError("deployment capabilities must be strings")
        if value not in KNOWN_DEPLOYMENT_CAPABILITIES:
            raise ValueError(f"unknown deployment capability {value!r}")
        if value in seen:
            raise ValueError(f"duplicate deployment capability {value!r}")
        seen.add(value)
    if CAP_FLAT_LEGACY_V6 not in seen:
        raise ValueError("deployment config must retain the flat_legacy fallback")
    return tuple(
        capability
        for capability in KNOWN_DEPLOYMENT_CAPABILITIES
        if capability in seen
    )


def _strategy_deployability(
    deployment_capabilities: tuple[str, ...],
) -> dict[PrefixWaveStrategy, dict[str, Any]]:
    present = set(deployment_capabilities)
    result: dict[PrefixWaveStrategy, dict[str, Any]] = {}
    for strategy in STRATEGY_ORDER:
        required = STRATEGY_CAPABILITY_REQUIREMENTS[strategy]
        missing = tuple(value for value in required if value not in present)
        result[strategy] = {
            "deployable": not missing,
            "requiredCapabilities": list(required),
            "missingCapabilities": list(missing),
        }
    return result


def build_prefix_wave_benchmark(
    *,
    deployment_capabilities: Sequence[str] = DEFAULT_DEPLOYMENT_CAPABILITIES,
) -> dict[str, Any]:
    """Build the complete deterministic report and its canonical payload hash."""

    capabilities = _normalize_deployment_capabilities(deployment_capabilities)
    profiles = _profiles()
    scenario_rows: list[dict[str, Any]] = []
    for scenario_name, paths in _scenarios():
        schedule = plan_prefix_segment_schedule(
            paths,
            shared_prefix_tokens=(KV_PENDING_TOKEN,) * SHARED_ROOT_TOKENS,
        )
        profile_rows = [
            _profile_result(
                scenario_name=scenario_name,
                paths=paths,
                schedule=schedule,
                profile_name=profile_name,
                cost_model=cost_model,
                deployment_capabilities=capabilities,
            )
            for profile_name, cost_model in profiles
        ]
        scenario_rows.append(
            {
                "scenario": scenario_name,
                "candidatePaths": [list(path) for path in paths],
                "planner": _planner_record(schedule),
                "kvAlignmentSweep": _kv_alignment_sweep(paths),
                "profiles": profile_rows,
            }
        )

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "evidenceClass": EVIDENCE_CLASS,
        "interpretation": (
            "Deterministic cost/event simulation only. Latencies, speedups, "
            "frames, bytes and COW occupancy are calculated from the stated "
            "inputs; there is no wall-clock, GPU, two-host, throughput or "
            "tokens-per-second measurement."
        ),
        "assumptions": [
            (
                "Capacity is already reserved before the timed wave; the two "
                "route-wide PREPARE/COMMIT control traversals are not included."
            ),
            (
                "Every batched row models one launch per exact compatible "
                "work unit but charges linear compute for every token step; "
                "no batching throughput multiplier is assumed."
            ),
            (
                "The route is a warm ordered stream with the stated fixed "
                "compute, launch, propagation, bandwidth and codec costs; "
                "jitter, retransmission and queueing are not modeled."
            ),
            (
                "The timings cover one speculative wave, not steady-state "
                "multi-user throughput or a complete chat."
            ),
            (
                "All four rows verify the same fixed candidate tree. This "
                "benchmark chooses its transport/execution form; it does not "
                "prove that proposing that tree beats one-token classic "
                "decoding after acceptance and control costs."
            ),
            (
                "Only strategies whose complete route-wide requirements are "
                "present in deploymentCapabilityConfig are deployable. By "
                "default, batched flat and both prefix forms remain "
                "uncertified and flat_legacy is the fallback."
            ),
        ],
        "selectionRule": (
            "theoreticalBestStrategy minimizes modeled makespan across these "
            "four fixed strategy configurations; it is not a global optimum. "
            "deployableStrategy first filters by explicit route-wide capabilities. "
            "Exact ties prefer flat_legacy, flat_batched, monolithic_packed, "
            "then streaming_segmented."
        ),
        "canonicalization": {
            "format": "UTF-8 JSON, sorted keys, no insignificant whitespace",
            "hash": "SHA-256",
            "excludedField": "canonicalPayloadSha256",
        },
        "inputs": {
            "sharedRootTokens": SHARED_ROOT_TOKENS,
            "streamGroupsPerRecord": STREAM_GROUPS_PER_RECORD,
            "deploymentCapabilityConfig": {
                "schema": CAPABILITY_CONFIG_SCHEMA,
                "semantics": (
                    "Caller-supplied explicit set. Production must derive it "
                    "as the intersection of protocol configuration and all "
                    "stage executor manifests; absent capabilities fail closed."
                ),
                "certifiedCapabilities": list(capabilities),
                "strategyRequirements": {
                    strategy.value: list(STRATEGY_CAPABILITY_REQUIREMENTS[strategy])
                    for strategy in STRATEGY_ORDER
                },
            },
            "kv": {
                "blockTokens": KV_BLOCK_TOKENS,
                "bytesPerBlock": KV_BYTES_PER_BLOCK,
                "parentBaseTokens": KV_PARENT_BASE_TOKENS,
                "parentAlignmentOffsets": list(range(KV_BLOCK_TOKENS)),
                "pendingToken": KV_PENDING_TOKEN,
            },
            "profiles": [
                {
                    "profile": profile_name,
                    "costModel": _cost_model_record(cost_model),
                }
                for profile_name, cost_model in profiles
            ],
        },
        "scenarios": scenario_rows,
    }
    canonical_payload = _canonical_json(payload).encode("utf-8")
    return {
        **payload,
        "canonicalPayloadSha256": hashlib.sha256(canonical_payload).hexdigest(),
    }


def canonical_prefix_wave_benchmark_json(
    report: Mapping[str, Any] | None = None,
) -> str:
    """Return the stable canonical JSON representation of a report."""

    selected = build_prefix_wave_benchmark() if report is None else report
    if not isinstance(selected, Mapping):
        raise TypeError("report must be a mapping")
    return _canonical_json(selected)


def _profile_result(
    *,
    scenario_name: str,
    paths: tuple[tuple[int, ...], ...],
    schedule: Any,
    profile_name: str,
    cost_model: PrefixWaveCostModel,
    deployment_capabilities: tuple[str, ...],
) -> dict[str, Any]:
    flat = simulate_prefix_wave(
        paths,
        cost_model,
        strategy=PrefixWaveStrategy.FLAT_LEGACY,
        shared_root_tokens=SHARED_ROOT_TOKENS,
    )
    flat_batched = simulate_prefix_wave(
        paths,
        cost_model,
        strategy=PrefixWaveStrategy.FLAT_BATCHED,
        shared_root_tokens=SHARED_ROOT_TOKENS,
    )
    monolithic = simulate_prefix_wave(
        paths,
        cost_model,
        strategy=PrefixWaveStrategy.MONOLITHIC_PACKED,
        shared_root_tokens=SHARED_ROOT_TOKENS,
    )
    stream_optimization = optimize_stream_chunking(
        paths,
        cost_model,
        shared_root_tokens=SHARED_ROOT_TOKENS,
    )
    streaming = stream_optimization.simulation
    _assert_work_conservation(
        scenario_name=scenario_name,
        profile_name=profile_name,
        flat_steps=schedule.cost.flat_token_steps,
        unique_steps=schedule.cost.unique_token_steps,
        flat=flat,
        flat_batched=flat_batched,
        monolithic=monolithic,
        streaming=streaming,
    )
    theoretical_best = select_fastest_prefix_wave_strategy(
        flat,
        flat_batched,
        monolithic,
        streaming,
    )
    deployability = _strategy_deployability(deployment_capabilities)
    deployable = select_fastest_prefix_wave_strategy(
        flat,
        flat_batched,
        monolithic,
        streaming,
        eligible_strategies=tuple(
            strategy
            for strategy in STRATEGY_ORDER
            if deployability[strategy]["deployable"]
        ),
    )
    strategy_records = {
        PrefixWaveStrategy.FLAT_LEGACY.value: _simulation_record(
            flat,
            flat_makespan_ms=flat.makespan_ms,
            selected_chain_tokens_per_record=None,
        ),
        PrefixWaveStrategy.FLAT_BATCHED.value: _simulation_record(
            flat_batched,
            flat_makespan_ms=flat.makespan_ms,
            selected_chain_tokens_per_record=None,
        ),
        PrefixWaveStrategy.MONOLITHIC_PACKED.value: _simulation_record(
            monolithic,
            flat_makespan_ms=flat.makespan_ms,
            selected_chain_tokens_per_record=None,
        ),
        PrefixWaveStrategy.STREAMING_SEGMENTED.value: {
            **_simulation_record(
                streaming,
                flat_makespan_ms=flat.makespan_ms,
                selected_chain_tokens_per_record=(
                    stream_optimization.selected_chain_tokens_per_record
                ),
            ),
            "chunkCandidates": [
                {
                    "chainTokensPerRecord": candidate.chain_tokens_per_record,
                    "latencyMs": _rounded(candidate.simulation.makespan_ms),
                    "routeAggregateFrameCount": (
                        candidate.simulation.forward_frame_count
                        + candidate.simulation.reverse_frame_count
                    ),
                }
                for candidate in stream_optimization.candidates
            ],
        },
    }
    return {
        "profile": profile_name,
        "theoreticalBestStrategy": theoretical_best.strategy.value,
        "theoreticalBestLatencyMs": _rounded(theoretical_best.makespan_ms),
        "deployableStrategy": deployable.strategy.value,
        "deployableLatencyMs": _rounded(deployable.makespan_ms),
        "deploymentEligibility": {
            strategy.value: deployability[strategy]
            for strategy in STRATEGY_ORDER
        },
        "strategies": strategy_records,
    }


def _assert_work_conservation(
    *,
    scenario_name: str,
    profile_name: str,
    flat_steps: int,
    unique_steps: int,
    flat: PrefixWaveSimulation,
    flat_batched: PrefixWaveSimulation,
    monolithic: PrefixWaveSimulation,
    streaming: PrefixWaveSimulation,
) -> None:
    if flat.flat_equivalent_token_steps != flat_steps:
        raise RuntimeError(
            f"{scenario_name}/{profile_name}: flat-equivalent work changed"
        )
    if any(
        simulation.flat_equivalent_token_steps != flat_steps
        for simulation in (flat_batched, monolithic, streaming)
    ):
        raise RuntimeError(
            f"{scenario_name}/{profile_name}: strategy baselines disagree"
        )
    if flat.executed_token_steps != flat_steps:
        raise RuntimeError(f"{scenario_name}/{profile_name}: flat work was lost")
    if flat_batched.executed_token_steps != flat_steps:
        raise RuntimeError(
            f"{scenario_name}/{profile_name}: batched flat work was lost"
        )
    if any(
        simulation.executed_token_steps != unique_steps
        for simulation in (monolithic, streaming)
    ):
        raise RuntimeError(
            f"{scenario_name}/{profile_name}: prefix work was duplicated or lost"
        )


def _planner_record(schedule: Any) -> dict[str, Any]:
    cost = schedule.cost
    return {
        "leafCount": cost.leaf_count,
        "maximumDepth": cost.maximum_depth,
        "sharedPrefixTokenCount": cost.shared_prefix_token_count,
        "flatTokenSteps": cost.flat_token_steps,
        "uniqueTokenSteps": cost.unique_token_steps,
        "savedTokenSteps": cost.saved_token_steps,
        "computeReductionPercent": _rounded(100.0 * cost.compute_reduction_ratio),
        "nodeCalls": cost.node_calls,
        "segmentCalls": cost.segment_calls,
        "callReductionPercent": _rounded(100.0 * cost.call_reduction_ratio),
        "nestedForks": cost.fork_count,
        "totalPhysicalForksIncludingCarrier": cost.total_physical_fork_count,
        "lanes": cost.lane_count,
        "frontiers": cost.frontier_count,
        "compatibleFrontierGroups": cost.compatible_frontier_group_count,
        "maximumSegmentTokens": max(
            segment.token_count for segment in schedule.segments
        ),
    }


def _simulation_record(
    simulation: PrefixWaveSimulation,
    *,
    flat_makespan_ms: float,
    selected_chain_tokens_per_record: int | None,
) -> dict[str, Any]:
    forward_frames = simulation.forward_frame_count
    reverse_frames = simulation.reverse_frame_count
    forward_bytes = simulation.forward_wire_bytes
    reverse_bytes = simulation.reverse_wire_bytes
    return {
        "latencyMs": _rounded(simulation.makespan_ms),
        "firstResultMs": _rounded(simulation.first_result_ms),
        "speedupVsFlatLatency": _rounded(
            flat_makespan_ms / simulation.makespan_ms
        ),
        "executedTokenSteps": simulation.executed_token_steps,
        "flatEquivalentTokenSteps": simulation.flat_equivalent_token_steps,
        "kernelLaunchCount": simulation.kernel_launch_count,
        "physicalForkCountPerStage": simulation.physical_fork_count,
        "forkControlFrameCountAcrossHops": simulation.fork_control_frame_count,
        "forkControlWireBytesAcrossHops": simulation.fork_control_wire_bytes,
        "forwardFrameCountAcrossHops": forward_frames,
        "reverseFrameCount": reverse_frames,
        "routeAggregateFrameCount": forward_frames + reverse_frames,
        "forwardWireBytesAcrossHops": forward_bytes,
        "reverseWireBytes": reverse_bytes,
        "routeAggregateWireBytes": forward_bytes + reverse_bytes,
        "logicalRequestResponseBarriers": (
            simulation.logical_request_response_barriers
        ),
        "selectedChainTokensPerRecord": selected_chain_tokens_per_record,
    }


def _kv_alignment_sweep(
    paths: tuple[tuple[int, ...], ...],
) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    for alignment_offset in range(KV_BLOCK_TOKENS):
        parent_tokens = KV_PARENT_BASE_TOKENS + alignment_offset
        projection = project_prefix_closed_cow_kv(
            parent_tokens=parent_tokens,
            block_tokens=KV_BLOCK_TOKENS,
            bytes_per_block=KV_BYTES_PER_BLOCK,
            pending_token=KV_PENDING_TOKEN,
            candidate_paths=paths,
        )
        samples.append(
            {
                "alignmentOffsetTokens": alignment_offset,
                "parentTokens": parent_tokens,
                "parentRemainderTokens": parent_tokens % KV_BLOCK_TOKENS,
                "flatIncrementalBlocks": (
                    projection.naive_flat.incremental_blocks
                ),
                "flatIncrementalBytes": projection.naive_flat.incremental_bytes,
                "prefixIncrementalBlocks": (
                    projection.prefix_closed.incremental_blocks
                ),
                "prefixIncrementalBytes": (
                    projection.prefix_closed.incremental_bytes
                ),
                "savedIncrementalBlocks": projection.saved_incremental_blocks,
                "savedIncrementalBytes": projection.saved_incremental_bytes,
                "blockReductionPercent": _rounded(
                    100.0 * projection.physical_block_reduction_ratio
                ),
                "prefixCopiedTailBlocks": (
                    projection.prefix_closed.copied_tail_blocks
                ),
                "prefixAppendAllocatedBlocks": (
                    projection.prefix_closed.append_allocated_blocks
                ),
            }
        )

    ordered = sorted(
        samples,
        key=lambda sample: (
            sample["savedIncrementalBlocks"],
            sample["blockReductionPercent"],
            sample["alignmentOffsetTokens"],
        ),
    )
    worst = min(
        samples,
        key=lambda sample: (
            sample["savedIncrementalBlocks"],
            sample["blockReductionPercent"],
            sample["alignmentOffsetTokens"],
        ),
    )
    best = max(
        samples,
        key=lambda sample: (
            sample["savedIncrementalBlocks"],
            sample["blockReductionPercent"],
            -sample["alignmentOffsetTokens"],
        ),
    )
    return {
        "selectionMetric": (
            "saved incremental blocks, then reduction percent; deterministic "
            "alignment-offset tie break. Median is the upper middle sample."
        ),
        "worst": dict(worst),
        "median": dict(ordered[len(ordered) // 2]),
        "best": dict(best),
        "samples": samples,
    }


def _cost_model_record(cost_model: PrefixWaveCostModel) -> dict[str, Any]:
    return {
        "stageComputeMsPerToken": list(cost_model.stage_compute_ms_per_token),
        "stageKernelLaunchMs": list(cost_model.stage_kernel_launch_ms),
        "hopOneWayPropagationMs": list(
            cost_model.hop_one_way_propagation_ms
        ),
        "hopBandwidthMbps": list(cost_model.hop_bandwidth_mbps),
        "activationBytesPerToken": cost_model.activation_bytes_per_token,
        "returnOneWayPropagationMs": cost_model.return_one_way_propagation_ms,
        "returnBandwidthMbps": cost_model.return_bandwidth_mbps,
        "resultBytesPerToken": cost_model.result_bytes_per_token,
        "flatRecordHeaderBytes": cost_model.flat_record_header_bytes,
        "prefixRecordHeaderBytes": cost_model.prefix_record_header_bytes,
        "resultRecordHeaderBytes": cost_model.result_record_header_bytes,
        "forkControlFrameBytes": cost_model.fork_control_frame_bytes,
        "nodeMetadataBytes": cost_model.node_metadata_bytes,
        "forkMetadataBytes": cost_model.fork_metadata_bytes,
        "packFixedMsPerRecord": cost_model.pack_fixed_ms_per_record,
        "unpackFixedMsPerRecord": cost_model.unpack_fixed_ms_per_record,
        "packBytesPerMs": cost_model.pack_bytes_per_ms,
        "hashBytesPerMs": cost_model.hash_bytes_per_ms,
        "forkApplyMsPerEdge": cost_model.fork_apply_ms_per_edge,
    }


def _rounded(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("benchmark values must be finite")
    return round(float(value), FLOAT_DIGITS)


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="indent for reading; omit for canonical JSON",
    )
    args = parser.parse_args(argv)
    report = build_prefix_wave_benchmark()
    if args.pretty:
        print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))
    else:
        print(canonical_prefix_wave_benchmark_json(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CAPABILITY_CONFIG_SCHEMA",
    "CAP_FLAT_BATCHED_EXACT_V1",
    "CAP_FLAT_LEGACY_V6",
    "CAP_PREFIX_MONOLITHIC_EXACT_BATCH_V1",
    "CAP_PREFIX_STREAMING_EXACT_BATCH_V1",
    "CAP_PREFIX_WAVE_V7",
    "DEFAULT_DEPLOYMENT_CAPABILITIES",
    "EVIDENCE_CLASS",
    "KNOWN_DEPLOYMENT_CAPABILITIES",
    "SCHEMA",
    "STRATEGY_CAPABILITY_REQUIREMENTS",
    "STRATEGY_ORDER",
    "build_prefix_wave_benchmark",
    "canonical_prefix_wave_benchmark_json",
    "main",
    "select_fastest_prefix_wave_strategy",
]
