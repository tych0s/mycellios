"""Pure event/cost model for exact tree-wave transport choices.

This module does not own sockets, tensors, KV caches, or request identifiers.
It answers one deliberately narrow architecture question: once a speculative
tree has been reserved route-wide, what is the latency trade-off between:

``flat_legacy``
    One record per complete leaf.  Shared prefixes are recomputed, but records
    can pipeline between stages.

``monolithic_packed``
    One prefix-closed tree-attention record.  Every unique prefix is computed
    once and each stage launches once, but the next stage cannot start until
    the complete record has been produced, serialized, received and checked.

``streaming_segmented``
    A topologically ordered sequence of prefix-closed tree-attention records.
    Each record contains one or more adjacent trie frontiers.  A downstream
    stage may start segment ``k`` while its predecessor computes ``k + 1``.

The streaming schedule has no application acknowledgement dependency between
segments.  It is therefore one logical request/response barrier, not one RTT
per trie node.  That claim assumes a warm ordered byte stream, a route-wide
capacity reservation, immutable wave/segment identities, and speculative KV
state which can be discarded on any framing, digest, stage, or connection
failure.  TCP acknowledgements and congestion control still exist, of course;
they are transport mechanics rather than application barriers.

Times are deterministic model inputs, not measured performance.  Category
totals are *resource work*, so they are not additive wall time when stages or
links overlap.  The event timeline is exposed to make that distinction
auditable.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from numbers import Integral, Real
from typing import Sequence

from .prefix_segment_schedule import plan_prefix_segment_schedule


class PrefixWaveStrategy(str, Enum):
    FLAT_LEGACY = "flat_legacy"
    MONOLITHIC_PACKED = "monolithic_packed"
    STREAMING_SEGMENTED = "streaming_segmented"


@dataclass(frozen=True, slots=True)
class PrefixWaveCostModel:
    """Explicit deterministic costs for one linear virtual-stage route.

    There are ``len(stage_compute_ms_per_token)`` stages and one fewer network
    hops.  A link is full duplex in the model: forward activations and reverse
    results have independent serialization resources, as a normal TCP
    connection does at the physical-link level.

    ``pack_bytes_per_ms`` and ``hash_bytes_per_ms`` are local memory/codec
    throughputs.  Decode uses the same copy throughput as encode.  GPU work and
    the stage's forward codec path are conservatively serialized; reverse
    result relay is a separate lightweight resource and may overlap them.
    """

    stage_compute_ms_per_token: tuple[float, ...]
    stage_kernel_launch_ms: tuple[float, ...]
    hop_one_way_propagation_ms: tuple[float, ...]
    hop_bandwidth_mbps: tuple[float, ...]
    activation_bytes_per_token: int
    result_bytes_per_leaf: int = 16
    record_header_bytes: int = 96
    node_metadata_bytes: int = 12
    fork_metadata_bytes: int = 16
    pack_fixed_ms_per_record: float = 0.02
    unpack_fixed_ms_per_record: float = 0.02
    pack_bytes_per_ms: float = 2_000_000.0
    hash_bytes_per_ms: float = 5_000_000.0
    fork_apply_ms_per_edge: float = 0.002

    def __post_init__(self) -> None:
        stage_compute = _finite_tuple(
            "stage_compute_ms_per_token",
            self.stage_compute_ms_per_token,
            positive=False,
            allow_empty=False,
        )
        stage_launch = _finite_tuple(
            "stage_kernel_launch_ms",
            self.stage_kernel_launch_ms,
            positive=False,
            allow_empty=False,
        )
        if len(stage_compute) != len(stage_launch):
            raise ValueError(
                "stage_compute_ms_per_token and stage_kernel_launch_ms must "
                "have the same length"
            )
        propagation = _finite_tuple(
            "hop_one_way_propagation_ms",
            self.hop_one_way_propagation_ms,
            positive=False,
            allow_empty=True,
        )
        bandwidth = _finite_tuple(
            "hop_bandwidth_mbps",
            self.hop_bandwidth_mbps,
            positive=True,
            allow_empty=True,
        )
        expected_hops = len(stage_compute) - 1
        if len(propagation) != expected_hops or len(bandwidth) != expected_hops:
            raise ValueError(
                "hop cost tuples must contain exactly one entry per adjacent "
                "stage pair"
            )
        for name in (
            "activation_bytes_per_token",
            "result_bytes_per_leaf",
        ):
            _positive_integer(name, getattr(self, name))
        for name in (
            "record_header_bytes",
            "node_metadata_bytes",
            "fork_metadata_bytes",
        ):
            _nonnegative_integer(name, getattr(self, name))
        for name in (
            "pack_fixed_ms_per_record",
            "unpack_fixed_ms_per_record",
            "fork_apply_ms_per_edge",
        ):
            _finite_number(name, getattr(self, name), positive=False)
        for name in ("pack_bytes_per_ms", "hash_bytes_per_ms"):
            _finite_number(name, getattr(self, name), positive=True)

        object.__setattr__(self, "stage_compute_ms_per_token", stage_compute)
        object.__setattr__(self, "stage_kernel_launch_ms", stage_launch)
        object.__setattr__(self, "hop_one_way_propagation_ms", propagation)
        object.__setattr__(self, "hop_bandwidth_mbps", bandwidth)

    @property
    def stage_count(self) -> int:
        return len(self.stage_compute_ms_per_token)

    @property
    def hop_count(self) -> int:
        return len(self.hop_bandwidth_mbps)


@dataclass(frozen=True, slots=True)
class PrefixWaveWorkUnit:
    unit_index: int
    label: str
    first_depth: int
    last_depth: int
    token_steps: int
    node_count: int
    fork_edges: int
    terminal_leaf_count: int
    activation_frame_bytes: int
    result_frame_bytes: int


@dataclass(frozen=True, slots=True)
class PrefixWaveEvent:
    category: str
    resource: str
    direction: str
    unit_index: int
    start_ms: float
    end_ms: float
    stage_index: int | None = None
    hop_index: int | None = None
    byte_count: int = 0

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms


@dataclass(frozen=True, slots=True)
class PrefixWaveCostBreakdown:
    propagation_ms: float
    bandwidth_ms: float
    compute_ms: float
    kernel_launch_ms: float
    fork_ms: float
    pack_ms: float
    unpack_ms: float
    hash_ms: float

    @property
    def total_resource_work_ms(self) -> float:
        return (
            self.propagation_ms
            + self.bandwidth_ms
            + self.compute_ms
            + self.kernel_launch_ms
            + self.fork_ms
            + self.pack_ms
            + self.unpack_ms
            + self.hash_ms
        )


@dataclass(frozen=True, slots=True)
class PrefixWaveSimulation:
    strategy: PrefixWaveStrategy
    work_units: tuple[PrefixWaveWorkUnit, ...]
    events: tuple[PrefixWaveEvent, ...]
    makespan_ms: float
    first_result_ms: float
    last_result_ms: float
    propagation_floor_ms: float
    inter_stage_compute_overlap_ms: float
    cost: PrefixWaveCostBreakdown
    flat_equivalent_token_steps: int
    executed_token_steps: int
    kernel_launch_count: int
    forward_frame_count: int
    reverse_frame_count: int
    forward_wire_bytes: int
    reverse_wire_bytes: int
    logical_request_response_barriers: int = 1
    per_node_acknowledgements: int = 0
    forward_progress_waits_for_reverse: bool = False

    @property
    def shared_compute_reduction_ratio(self) -> float:
        if self.flat_equivalent_token_steps == 0:
            return 0.0
        return 1.0 - self.executed_token_steps / self.flat_equivalent_token_steps


@dataclass(frozen=True, slots=True)
class PrefixWaveComparison:
    flat_legacy: PrefixWaveSimulation
    monolithic_packed: PrefixWaveSimulation
    streaming_segmented: PrefixWaveSimulation


@dataclass(frozen=True, slots=True)
class PrefixWaveChunkingCandidate:
    chain_tokens_per_record: int
    simulation: PrefixWaveSimulation


@dataclass(frozen=True, slots=True)
class PrefixWaveChunkingOptimization:
    candidates: tuple[PrefixWaveChunkingCandidate, ...]
    selected_chain_tokens_per_record: int
    simulation: PrefixWaveSimulation


@dataclass(frozen=True, slots=True)
class _SegmentBatch:
    group_id: int
    frontier_index: int
    first_depth: int
    last_depth: int
    token_steps: int
    node_count: int
    fork_edges: int
    terminal_leaf_count: int


def compare_prefix_wave_strategies(
    candidate_paths: Sequence[Sequence[int]],
    cost_model: PrefixWaveCostModel,
    *,
    shared_root_tokens: int = 1,
    stream_chain_tokens_per_record: int | None = 1,
    stream_groups_per_record: int = 1,
) -> PrefixWaveComparison:
    """Simulate all three strategies with one validated workload."""

    return PrefixWaveComparison(
        flat_legacy=simulate_prefix_wave(
            candidate_paths,
            cost_model,
            strategy=PrefixWaveStrategy.FLAT_LEGACY,
            shared_root_tokens=shared_root_tokens,
        ),
        monolithic_packed=simulate_prefix_wave(
            candidate_paths,
            cost_model,
            strategy=PrefixWaveStrategy.MONOLITHIC_PACKED,
            shared_root_tokens=shared_root_tokens,
        ),
        streaming_segmented=simulate_prefix_wave(
            candidate_paths,
            cost_model,
            strategy=PrefixWaveStrategy.STREAMING_SEGMENTED,
            shared_root_tokens=shared_root_tokens,
            stream_chain_tokens_per_record=stream_chain_tokens_per_record,
            stream_groups_per_record=stream_groups_per_record,
        ),
    )


def optimize_stream_chunking(
    candidate_paths: Sequence[Sequence[int]],
    cost_model: PrefixWaveCostModel,
    *,
    shared_root_tokens: int = 1,
) -> PrefixWaveChunkingOptimization:
    """Select the predicted fastest deterministic chain-slice size.

    This is intentionally an exhaustive bounded controller rather than a
    fitted formula.  Candidate trees are already capped to small depths, and
    every candidate uses the same exact topology and cost inputs.  Ties prefer
    the larger slice (fewer records and less operational complexity).
    """

    root_tokens = _bounded_nonnegative_integer(
        "shared_root_tokens", shared_root_tokens, maximum=63
    )
    schedule = plan_prefix_segment_schedule(
        candidate_paths,
        shared_prefix_tokens=(0,) * root_tokens,
    )
    maximum_chain = max(segment.token_count for segment in schedule.segments)
    candidates = tuple(
        PrefixWaveChunkingCandidate(
            chain_tokens_per_record=chain_tokens,
            simulation=simulate_prefix_wave(
                schedule.candidate_paths,
                cost_model,
                strategy=PrefixWaveStrategy.STREAMING_SEGMENTED,
                shared_root_tokens=root_tokens,
                stream_chain_tokens_per_record=chain_tokens,
            ),
        )
        for chain_tokens in range(1, maximum_chain + 1)
    )
    selected = min(
        candidates,
        key=lambda item: (
            item.simulation.makespan_ms,
            -item.chain_tokens_per_record,
        ),
    )
    return PrefixWaveChunkingOptimization(
        candidates=candidates,
        selected_chain_tokens_per_record=selected.chain_tokens_per_record,
        simulation=selected.simulation,
    )


def simulate_prefix_wave(
    candidate_paths: Sequence[Sequence[int]],
    cost_model: PrefixWaveCostModel,
    *,
    strategy: PrefixWaveStrategy | str,
    shared_root_tokens: int = 1,
    stream_chain_tokens_per_record: int | None = 1,
    stream_groups_per_record: int = 1,
) -> PrefixWaveSimulation:
    """Build the exact work units and run the deterministic event recurrence.

    The recurrence deliberately contains no edge from a reverse/result event
    to a later forward segment.  Consequently the segmented strategy cannot
    accidentally acquire a hidden per-node acknowledgement RTT.
    """

    try:
        selected = PrefixWaveStrategy(strategy)
    except (TypeError, ValueError) as error:
        raise ValueError(f"unknown prefix-wave strategy {strategy!r}") from error
    # The structural planner's default full-path bound is 64 and candidates
    # are non-empty.  Bound before allocating the synthetic prefix tuple.
    root_tokens = _bounded_nonnegative_integer(
        "shared_root_tokens", shared_root_tokens, maximum=63
    )
    chain_tokens = (
        None
        if stream_chain_tokens_per_record is None
        else _positive_integer(
            "stream_chain_tokens_per_record",
            stream_chain_tokens_per_record,
        )
    )
    groups_per_record = _positive_integer(
        "stream_groups_per_record", stream_groups_per_record
    )
    # Only topology matters to this latency model.  A deterministic zero-valued
    # shared prefix gives the compressed-trie planner the requested length
    # without pretending that the simulator owns real token IDs.
    schedule = plan_prefix_segment_schedule(
        candidate_paths,
        shared_prefix_tokens=(0,) * root_tokens,
    )
    segment_batches = _segment_batches(
        schedule,
        maximum_chain_tokens=(
            chain_tokens
            if selected is PrefixWaveStrategy.STREAMING_SEGMENTED
            else None
        ),
    )
    units = _work_units(
        selected,
        schedule.candidate_paths,
        segment_batches,
        shared_root_tokens=root_tokens,
        stream_groups_per_record=groups_per_record,
        cost_model=cost_model,
    )
    events, result_times = _simulate_units(units, cost_model)
    if len(result_times) != len(schedule.candidate_paths):
        raise RuntimeError("event simulation did not return every terminal leaf")

    event_tuple = tuple(
        sorted(
            events,
            key=lambda item: (
                item.start_ms,
                item.end_ms,
                item.resource,
                item.unit_index,
                item.category,
            ),
        )
    )
    cost = _cost_breakdown(event_tuple)
    flat_steps = schedule.cost.flat_token_steps
    executed_steps = sum(unit.token_steps for unit in units)
    hop_count = cost_model.hop_count
    return PrefixWaveSimulation(
        strategy=selected,
        work_units=units,
        events=event_tuple,
        makespan_ms=max(result_times),
        first_result_ms=min(result_times),
        last_result_ms=max(result_times),
        propagation_floor_ms=2.0
        * sum(cost_model.hop_one_way_propagation_ms),
        inter_stage_compute_overlap_ms=_inter_stage_compute_overlap(event_tuple),
        cost=cost,
        flat_equivalent_token_steps=flat_steps,
        executed_token_steps=executed_steps,
        kernel_launch_count=len(units) * cost_model.stage_count,
        forward_frame_count=len(units) * hop_count,
        reverse_frame_count=sum(
            unit.terminal_leaf_count > 0 for unit in units
        )
        * hop_count,
        forward_wire_bytes=sum(unit.activation_frame_bytes for unit in units)
        * hop_count,
        reverse_wire_bytes=sum(
            unit.result_frame_bytes
            for unit in units
            if unit.terminal_leaf_count > 0
        )
        * hop_count,
    )


def _segment_batches(
    schedule,
    *,
    maximum_chain_tokens: int | None,
) -> tuple[_SegmentBatch, ...]:
    """Map structural compatible groups to canonical wire/kernel batches.

    Every FORK for a dependency frontier is attached to its first compatible
    group.  The segment planner guarantees those FORKs precede *all* compute
    groups in that frontier, so this preserves the source-lane-before-mutation
    invariant on an ordered stream.
    """

    segments = {segment.segment_id: segment for segment in schedule.segments}
    first_group_by_frontier: dict[int, int] = {}
    for group in schedule.compatible_frontier_groups:
        first_group_by_frontier.setdefault(group.frontier_index, group.group_id)
    forks_by_frontier: dict[int, int] = {}
    for fork in schedule.forks:
        forks_by_frontier[fork.frontier_index] = (
            forks_by_frontier.get(fork.frontier_index, 0) + 1
        )

    batches: list[_SegmentBatch] = []
    for group in schedule.compatible_frontier_groups:
        grouped_segments = tuple(segments[item] for item in group.segment_ids)
        group_terminal_count = sum(
            schedule.node(segment.end_node_id).terminal
            for segment in grouped_segments
        )
        chain_limit = maximum_chain_tokens or group.token_count
        for offset in range(0, group.token_count, chain_limit):
            chain_count = min(chain_limit, group.token_count - offset)
            node_count = len(grouped_segments) * chain_count
            is_first_slice = offset == 0
            is_last_slice = offset + chain_count == group.token_count
            batches.append(
                _SegmentBatch(
                    group_id=group.group_id,
                    frontier_index=group.frontier_index,
                    first_depth=group.start_depth + offset + 1,
                    last_depth=group.start_depth + offset + chain_count,
                    token_steps=node_count,
                    node_count=node_count,
                    fork_edges=(
                        forks_by_frontier.get(group.frontier_index, 0)
                        if is_first_slice
                        and first_group_by_frontier[group.frontier_index]
                        == group.group_id
                        else 0
                    ),
                    terminal_leaf_count=(
                        group_terminal_count if is_last_slice else 0
                    ),
                )
            )
    if sum(item.node_count for item in batches) != schedule.cost.unique_token_steps:
        raise RuntimeError("segment batches lost unique prefix work")
    if sum(item.fork_edges for item in batches) != schedule.cost.fork_count:
        raise RuntimeError("segment batches lost a topological FORK")
    if sum(item.terminal_leaf_count for item in batches) != schedule.cost.leaf_count:
        raise RuntimeError("segment batches lost a terminal leaf")
    return tuple(batches)


def _work_units(
    strategy: PrefixWaveStrategy,
    paths: tuple[tuple[int, ...], ...],
    segment_batches: tuple[_SegmentBatch, ...],
    *,
    shared_root_tokens: int,
    stream_groups_per_record: int,
    cost_model: PrefixWaveCostModel,
) -> tuple[PrefixWaveWorkUnit, ...]:
    raw_units: list[tuple[str, int, int, int, int, int, int]] = []
    if strategy is PrefixWaveStrategy.FLAT_LEGACY:
        for leaf_index, path in enumerate(paths):
            token_steps = len(path) + shared_root_tokens
            raw_units.append(
                (
                    f"leaf-{leaf_index}",
                    0 if shared_root_tokens else 1,
                    len(path),
                    token_steps,
                    token_steps,
                    1,
                    1,
                )
            )
    elif strategy is PrefixWaveStrategy.MONOLITHIC_PACKED:
        raw_units.append(
            (
                "prefix-wave",
                segment_batches[0].first_depth,
                segment_batches[-1].last_depth,
                sum(item.token_steps for item in segment_batches),
                sum(item.node_count for item in segment_batches),
                sum(item.fork_edges for item in segment_batches),
                sum(item.terminal_leaf_count for item in segment_batches),
            )
        )
    else:
        for start in range(0, len(segment_batches), stream_groups_per_record):
            group = segment_batches[start : start + stream_groups_per_record]
            raw_units.append(
                (
                    f"segment-{len(raw_units)}",
                    min(item.first_depth for item in group),
                    max(item.last_depth for item in group),
                    sum(item.token_steps for item in group),
                    sum(item.node_count for item in group),
                    sum(item.fork_edges for item in group),
                    sum(item.terminal_leaf_count for item in group),
                )
            )

    units: list[PrefixWaveWorkUnit] = []
    for index, (
        label,
        first_depth,
        last_depth,
        token_steps,
        node_count,
        fork_edges,
        terminal_count,
    ) in enumerate(raw_units):
        prefix_metadata = (
            0
            if strategy is PrefixWaveStrategy.FLAT_LEGACY
            else node_count * cost_model.node_metadata_bytes
            + fork_edges * cost_model.fork_metadata_bytes
        )
        activation_bytes = (
            cost_model.record_header_bytes
            + token_steps * cost_model.activation_bytes_per_token
            + prefix_metadata
        )
        result_bytes = (
            cost_model.record_header_bytes
            + terminal_count * cost_model.result_bytes_per_leaf
            if terminal_count
            else 0
        )
        units.append(
            PrefixWaveWorkUnit(
                unit_index=index,
                label=label,
                first_depth=first_depth,
                last_depth=last_depth,
                token_steps=token_steps,
                node_count=node_count,
                fork_edges=fork_edges,
                terminal_leaf_count=terminal_count,
                activation_frame_bytes=activation_bytes,
                result_frame_bytes=result_bytes,
            )
        )
    return tuple(units)


def _simulate_units(
    units: tuple[PrefixWaveWorkUnit, ...],
    model: PrefixWaveCostModel,
) -> tuple[list[PrefixWaveEvent], list[float]]:
    events: list[PrefixWaveEvent] = []
    stage_available = [0.0] * model.stage_count
    forward_link_available = [0.0] * model.hop_count
    reverse_link_available = [0.0] * model.hop_count
    reverse_codec_available = [0.0] * model.stage_count
    result_times: list[float] = []

    for unit in units:
        arrival = 0.0
        for stage_index in range(model.stage_count):
            resource = f"stage-{stage_index}-forward"
            cursor = max(arrival, stage_available[stage_index])
            if stage_index:
                cursor = _codec_decode(
                    events,
                    resource=resource,
                    direction="forward",
                    unit_index=unit.unit_index,
                    stage_index=stage_index,
                    cursor=cursor,
                    byte_count=unit.activation_frame_bytes,
                    model=model,
                )
            if unit.fork_edges:
                cursor = _append_event(
                    events,
                    category="fork",
                    resource=resource,
                    direction="local",
                    unit_index=unit.unit_index,
                    stage_index=stage_index,
                    start_ms=cursor,
                    duration_ms=(
                        unit.fork_edges * model.fork_apply_ms_per_edge
                    ),
                )
            cursor = _append_event(
                events,
                category="kernel_launch",
                resource=resource,
                direction="local",
                unit_index=unit.unit_index,
                stage_index=stage_index,
                start_ms=cursor,
                duration_ms=model.stage_kernel_launch_ms[stage_index],
            )
            cursor = _append_event(
                events,
                category="compute",
                resource=resource,
                direction="local",
                unit_index=unit.unit_index,
                stage_index=stage_index,
                start_ms=cursor,
                duration_ms=(
                    unit.token_steps
                    * model.stage_compute_ms_per_token[stage_index]
                ),
            )

            if stage_index < model.stage_count - 1:
                cursor = _codec_encode(
                    events,
                    resource=resource,
                    direction="forward",
                    unit_index=unit.unit_index,
                    stage_index=stage_index,
                    cursor=cursor,
                    byte_count=unit.activation_frame_bytes,
                    model=model,
                )
                stage_available[stage_index] = cursor
                arrival = _network_hop(
                    events,
                    direction="forward",
                    unit_index=unit.unit_index,
                    hop_index=stage_index,
                    cursor=cursor,
                    byte_count=unit.activation_frame_bytes,
                    link_available=forward_link_available,
                    model=model,
                )
            else:
                if unit.terminal_leaf_count:
                    cursor = _codec_encode(
                        events,
                        resource=resource,
                        direction="reverse",
                        unit_index=unit.unit_index,
                        stage_index=stage_index,
                        cursor=cursor,
                        byte_count=unit.result_frame_bytes,
                        model=model,
                    )
                stage_available[stage_index] = cursor

        if not unit.terminal_leaf_count:
            continue
        result_arrival = cursor
        for hop_index in range(model.hop_count - 1, -1, -1):
            result_arrival = _network_hop(
                events,
                direction="reverse",
                unit_index=unit.unit_index,
                hop_index=hop_index,
                cursor=result_arrival,
                byte_count=unit.result_frame_bytes,
                link_available=reverse_link_available,
                model=model,
            )
            upstream_stage = hop_index
            reverse_resource = f"stage-{upstream_stage}-reverse"
            reverse_cursor = max(
                result_arrival, reverse_codec_available[upstream_stage]
            )
            reverse_cursor = _codec_decode(
                events,
                resource=reverse_resource,
                direction="reverse",
                unit_index=unit.unit_index,
                stage_index=upstream_stage,
                cursor=reverse_cursor,
                byte_count=unit.result_frame_bytes,
                model=model,
            )
            if upstream_stage:
                reverse_cursor = _codec_encode(
                    events,
                    resource=reverse_resource,
                    direction="reverse",
                    unit_index=unit.unit_index,
                    stage_index=upstream_stage,
                    cursor=reverse_cursor,
                    byte_count=unit.result_frame_bytes,
                    model=model,
                )
            reverse_codec_available[upstream_stage] = reverse_cursor
            result_arrival = reverse_cursor
        result_times.extend(
            [result_arrival] * unit.terminal_leaf_count
        )
    return events, result_times


def _codec_encode(
    events: list[PrefixWaveEvent],
    *,
    resource: str,
    direction: str,
    unit_index: int,
    stage_index: int,
    cursor: float,
    byte_count: int,
    model: PrefixWaveCostModel,
) -> float:
    cursor = _append_event(
        events,
        category="pack",
        resource=resource,
        direction=direction,
        unit_index=unit_index,
        stage_index=stage_index,
        start_ms=cursor,
        duration_ms=(
            model.pack_fixed_ms_per_record
            + byte_count / model.pack_bytes_per_ms
        ),
        byte_count=byte_count,
    )
    return _append_event(
        events,
        category="hash",
        resource=resource,
        direction=direction,
        unit_index=unit_index,
        stage_index=stage_index,
        start_ms=cursor,
        duration_ms=byte_count / model.hash_bytes_per_ms,
        byte_count=byte_count,
    )


def _codec_decode(
    events: list[PrefixWaveEvent],
    *,
    resource: str,
    direction: str,
    unit_index: int,
    stage_index: int,
    cursor: float,
    byte_count: int,
    model: PrefixWaveCostModel,
) -> float:
    cursor = _append_event(
        events,
        category="unpack",
        resource=resource,
        direction=direction,
        unit_index=unit_index,
        stage_index=stage_index,
        start_ms=cursor,
        duration_ms=(
            model.unpack_fixed_ms_per_record
            + byte_count / model.pack_bytes_per_ms
        ),
        byte_count=byte_count,
    )
    return _append_event(
        events,
        category="hash",
        resource=resource,
        direction=direction,
        unit_index=unit_index,
        stage_index=stage_index,
        start_ms=cursor,
        duration_ms=byte_count / model.hash_bytes_per_ms,
        byte_count=byte_count,
    )


def _network_hop(
    events: list[PrefixWaveEvent],
    *,
    direction: str,
    unit_index: int,
    hop_index: int,
    cursor: float,
    byte_count: int,
    link_available: list[float],
    model: PrefixWaveCostModel,
) -> float:
    tx_start = max(cursor, link_available[hop_index])
    bandwidth_ms = (
        byte_count * 8.0 / (model.hop_bandwidth_mbps[hop_index] * 1_000.0)
    )
    tx_end = _append_event(
        events,
        category="bandwidth",
        resource=f"{direction}-link-{hop_index}",
        direction=direction,
        unit_index=unit_index,
        hop_index=hop_index,
        start_ms=tx_start,
        duration_ms=bandwidth_ms,
        byte_count=byte_count,
    )
    link_available[hop_index] = tx_end
    return _append_event(
        events,
        category="propagation",
        resource=f"{direction}-propagation-{hop_index}",
        direction=direction,
        unit_index=unit_index,
        hop_index=hop_index,
        start_ms=tx_end,
        duration_ms=model.hop_one_way_propagation_ms[hop_index],
        byte_count=byte_count,
    )


def _append_event(
    events: list[PrefixWaveEvent],
    *,
    category: str,
    resource: str,
    direction: str,
    unit_index: int,
    start_ms: float,
    duration_ms: float,
    stage_index: int | None = None,
    hop_index: int | None = None,
    byte_count: int = 0,
) -> float:
    end_ms = start_ms + duration_ms
    events.append(
        PrefixWaveEvent(
            category=category,
            resource=resource,
            direction=direction,
            unit_index=unit_index,
            stage_index=stage_index,
            hop_index=hop_index,
            start_ms=start_ms,
            end_ms=end_ms,
            byte_count=byte_count,
        )
    )
    return end_ms


def _cost_breakdown(
    events: tuple[PrefixWaveEvent, ...],
) -> PrefixWaveCostBreakdown:
    totals: dict[str, float] = {}
    for event in events:
        totals[event.category] = totals.get(event.category, 0.0) + event.duration_ms
    return PrefixWaveCostBreakdown(
        propagation_ms=totals.get("propagation", 0.0),
        bandwidth_ms=totals.get("bandwidth", 0.0),
        compute_ms=totals.get("compute", 0.0),
        kernel_launch_ms=totals.get("kernel_launch", 0.0),
        fork_ms=totals.get("fork", 0.0),
        pack_ms=totals.get("pack", 0.0),
        unpack_ms=totals.get("unpack", 0.0),
        hash_ms=totals.get("hash", 0.0),
    )


def _inter_stage_compute_overlap(events: tuple[PrefixWaveEvent, ...]) -> float:
    compute = [event for event in events if event.category == "compute"]
    overlap = 0.0
    for left in compute:
        if left.stage_index is None:
            continue
        for right in compute:
            if (
                right.stage_index != left.stage_index + 1
                or right.unit_index == left.unit_index
            ):
                continue
            overlap += max(
                0.0,
                min(left.end_ms, right.end_ms)
                - max(left.start_ms, right.start_ms),
            )
    return overlap


def _finite_tuple(
    name: str,
    values: object,
    *,
    positive: bool,
    allow_empty: bool,
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(
        values, Sequence
    ):
        raise ValueError(f"{name} must be a sequence")
    normalized = tuple(
        _finite_number(f"{name}[{index}]", value, positive=positive)
        for index, value in enumerate(values)
    )
    if not normalized and not allow_empty:
        raise ValueError(f"{name} must not be empty")
    return normalized


def _finite_number(name: str, value: object, *, positive: bool) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0 or (
        positive and normalized <= 0.0
    ):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return normalized


def _positive_integer(name: str, value: object) -> int:
    normalized = _nonnegative_integer(name, value)
    if normalized < 1:
        raise ValueError(f"{name} must be positive")
    return normalized


def _bounded_nonnegative_integer(name: str, value: object, *, maximum: int) -> int:
    normalized = _nonnegative_integer(name, value)
    if normalized > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return normalized


def _nonnegative_integer(name: str, value: object) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool) or int(value) < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


__all__ = [
    "PrefixWaveComparison",
    "PrefixWaveChunkingCandidate",
    "PrefixWaveChunkingOptimization",
    "PrefixWaveCostBreakdown",
    "PrefixWaveCostModel",
    "PrefixWaveEvent",
    "PrefixWaveSimulation",
    "PrefixWaveStrategy",
    "PrefixWaveWorkUnit",
    "compare_prefix_wave_strategies",
    "optimize_stream_chunking",
    "simulate_prefix_wave",
]
