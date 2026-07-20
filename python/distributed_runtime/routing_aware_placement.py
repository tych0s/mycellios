"""Deterministic co-routing-aware placement for resident MoE experts.

The planner consumes only explicit route traces.  A top-k set is treated as a
weighted hyperedge, so the greedy coverage phase rewards placements that make
the whole observed set reachable through fewer owners.  A marginal baseline
uses the same records, node budgets, links and trace-derived per-expert heat but
deliberately ignores co-occurrence.

This is an offline projection.  It does not make transformer layers parallel
and it is not a runtime or WAN benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Mapping, Sequence

from .ram_expert_cache import ExpertKey, ExpertRecord
from .resident_expert_mesh import (
    MeshLinkProfile,
    MeshNodeProfile,
    ResidentExpertReplica,
)


ROUTING_AWARE_PLACEMENT_SCHEMA = "gdlp-routing-aware-placement/1"
_FLOAT_TOLERANCE = 1e-12
_COVERAGE_REPAIR_MAX_KEYS = 36
_COVERAGE_REPAIR_MAX_STATES = 200_000


class RoutingAwarePlacementError(RuntimeError):
    """Base class for routing-aware placement failures."""


class RoutingAwarePlacementUnavailableError(RoutingAwarePlacementError):
    """The sealed traces cannot be covered by an exact resident route."""


class RoutingAwarePlacementSearchLimitError(RoutingAwarePlacementError):
    """A bounded irregular bin-packing search could not prove feasibility."""


class _CoverageRepairSearchLimit(RuntimeError):
    pass


def _nonnegative_integer(name: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_integer(name: str, value: object) -> int:
    result = _nonnegative_integer(name, value)
    if result == 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _positive_float(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and positive")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite and positive") from error
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def _node_id(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} cannot be empty")
    return value.strip()


@dataclass(frozen=True, order=True)
class RoutingTraceSample:
    """One measured top-k set and its explicit projected position mass."""

    layer: int
    expert_ids: tuple[int, ...]
    frequency: float
    weight: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "layer", _nonnegative_integer("layer", self.layer))
        if not isinstance(self.expert_ids, tuple) or not self.expert_ids:
            raise ValueError("expert_ids must be a non-empty tuple")
        normalized = tuple(
            _nonnegative_integer("expert_id", expert_id)
            for expert_id in self.expert_ids
        )
        if len(set(normalized)) != len(normalized):
            raise ValueError("a top-k route set cannot repeat an expert")
        object.__setattr__(self, "expert_ids", tuple(sorted(normalized)))
        frequency = _positive_float("frequency", self.frequency)
        weight = _positive_float("weight", self.weight)
        if not math.isfinite(frequency * weight):
            raise ValueError("frequency multiplied by weight must remain finite")
        object.__setattr__(self, "frequency", frequency)
        object.__setattr__(self, "weight", weight)

    @property
    def effective_positions(self) -> float:
        return self.frequency * self.weight


@dataclass(frozen=True)
class TraceOwnerProjection:
    layer: int
    expert_ids: tuple[int, ...]
    effective_positions: float
    owner_ids: tuple[str, ...]
    owner_for_expert: tuple[tuple[int, str], ...]
    owner_contacts_per_position: int
    rpc_v1_activation_bytes_per_position: int
    coalesced_exact_activation_bytes_per_position: int
    coalesced_exact_row_index_bytes_per_position: int
    owner_partial_activation_byte_floor_per_position: int
    projected_link_ms_per_position: float


@dataclass(frozen=True)
class PlacementProjection:
    weighted_positions: float
    weighted_owner_contacts: float
    mean_owner_contacts_per_position: float
    max_owner_contacts_per_position: int
    rpc_v1_activation_bytes: float
    coalesced_exact_activation_bytes: float
    coalesced_exact_row_index_bytes: float
    owner_partial_activation_byte_floor: float
    projected_link_ms: float
    peak_experts_per_position: int
    peak_experts_per_position_by_node: tuple[tuple[str, int], ...]
    weighted_expert_activations_by_node: tuple[tuple[str, float], ...]
    used_vram_bytes_by_node: tuple[tuple[str, int], ...]
    total_replica_bytes: int
    trace_routes: tuple[TraceOwnerProjection, ...]


@dataclass(frozen=True)
class RoutingAwarePlacementPlan:
    schema: str
    required_experts: tuple[ExpertKey, ...]
    placements: tuple[ResidentExpertReplica, ...]
    projection: PlacementProjection
    marginal_baseline_placements: tuple[ResidentExpertReplica, ...]
    marginal_baseline_projection: PlacementProjection

    @property
    def comparison_uses_equal_replica_bytes(self) -> bool:
        return (
            self.projection.total_replica_bytes
            == self.marginal_baseline_projection.total_replica_bytes
        )

    @property
    def mean_owner_contacts_saved_per_position(self) -> float:
        return (
            self.marginal_baseline_projection.mean_owner_contacts_per_position
            - self.projection.mean_owner_contacts_per_position
        )

    @property
    def weighted_owner_contacts_saved(self) -> float:
        return (
            self.marginal_baseline_projection.weighted_owner_contacts
            - self.projection.weighted_owner_contacts
        )

    @property
    def rpc_v1_activation_bytes_saved(self) -> float:
        return (
            self.marginal_baseline_projection.rpc_v1_activation_bytes
            - self.projection.rpc_v1_activation_bytes
        )

    @property
    def coalesced_exact_activation_bytes_saved(self) -> float:
        return (
            self.marginal_baseline_projection.coalesced_exact_activation_bytes
            - self.projection.coalesced_exact_activation_bytes
        )


@dataclass(frozen=True)
class _TraceEdge:
    layer: int
    keys: tuple[ExpertKey, ...]
    effective_positions: float


@dataclass(frozen=True)
class _TraceRouteState:
    max_owner_ms: float
    request_bytes: int
    response_bytes: int
    owners: tuple[str, ...]
    assignments: tuple[tuple[int, str], ...]


@dataclass(frozen=True)
class _PlanningContext:
    coordinator_id: str
    records: Mapping[ExpertKey, ExpertRecord]
    nodes: Mapping[str, MeshNodeProfile]
    candidate_node_ids: tuple[str, ...]
    fixed_contact_cost_ms: Mapping[str, float]
    per_expert_transfer_ms: Mapping[str, float]
    shared_input_transfer_ms: Mapping[str, float]
    per_expert_output_transfer_ms: Mapping[str, float]
    row_index_transfer_ms: Mapping[str, float]
    contact_cost_ms: Mapping[str, float]
    edges: tuple[_TraceEdge, ...]
    marginal_mass: Mapping[ExpertKey, float]
    activation_bytes_per_position: int
    max_positions_per_wave: int


def plan_routing_aware_expert_placement(
    *,
    coordinator_id: str,
    experts: Sequence[ExpertRecord],
    nodes: Sequence[MeshNodeProfile],
    links: Sequence[MeshLinkProfile],
    traces: Sequence[RoutingTraceSample],
    activation_bytes_per_position: int,
    max_positions_per_wave: int,
    candidate_node_ids: Sequence[str] | None = None,
) -> RoutingAwarePlacementPlan:
    """Build a co-routing plan and a memory-matched marginal control plan."""

    context = _build_context(
        coordinator_id=coordinator_id,
        experts=experts,
        nodes=nodes,
        links=links,
        traces=traces,
        activation_bytes_per_position=activation_bytes_per_position,
        max_positions_per_wave=max_positions_per_wave,
        candidate_node_ids=candidate_node_ids,
    )
    required = tuple(sorted(context.marginal_mass))

    aware, aware_remaining = _place_required_coverage(context, routing_aware=True)
    aware = _add_routing_aware_replicas(context, aware, aware_remaining)
    aware_projection = _project(context, aware)

    marginal, marginal_remaining = _place_required_coverage(
        context,
        routing_aware=False,
    )
    mandatory_bytes = sum(context.records[key].byte_size for key in required)
    extra_replica_bytes = aware_projection.total_replica_bytes - mandatory_bytes
    marginal = _add_marginal_replicas(
        context,
        marginal,
        marginal_remaining,
        byte_allowance=extra_replica_bytes,
    )
    marginal_projection = _project(context, marginal)
    if (
        marginal_projection.projected_link_ms
        < aware_projection.projected_link_ms - _FLOAT_TOLERANCE
    ):
        # Co-location is a means, not the objective. Parallel owners can beat
        # fewer contacts when compute dominates, so never publish the affinity
        # heuristic as the faster plan when the matched control is better.
        aware = _copy_placements(marginal)
        aware_projection = marginal_projection

    return RoutingAwarePlacementPlan(
        schema=ROUTING_AWARE_PLACEMENT_SCHEMA,
        required_experts=required,
        placements=_replica_tuple(context, aware),
        projection=aware_projection,
        marginal_baseline_placements=_replica_tuple(context, marginal),
        marginal_baseline_projection=marginal_projection,
    )


def _build_context(
    *,
    coordinator_id: str,
    experts: Sequence[ExpertRecord],
    nodes: Sequence[MeshNodeProfile],
    links: Sequence[MeshLinkProfile],
    traces: Sequence[RoutingTraceSample],
    activation_bytes_per_position: int,
    max_positions_per_wave: int,
    candidate_node_ids: Sequence[str] | None,
) -> _PlanningContext:
    coordinator = _node_id("coordinator_id", coordinator_id)
    activation_bytes = _positive_integer(
        "activation_bytes_per_position",
        activation_bytes_per_position,
    )
    positions_per_wave = _positive_integer(
        "max_positions_per_wave",
        max_positions_per_wave,
    )

    records: dict[ExpertKey, ExpertRecord] = {}
    for record in experts:
        if not isinstance(record, ExpertRecord):
            raise TypeError("experts must contain ExpertRecord values")
        if record.key in records:
            raise ValueError(f"duplicate expert record {record.key}")
        records[record.key] = record
    if not records:
        raise ValueError("experts cannot be empty")

    node_map: dict[str, MeshNodeProfile] = {}
    for node in nodes:
        if not isinstance(node, MeshNodeProfile):
            raise TypeError("nodes must contain MeshNodeProfile values")
        if node.node_id in node_map:
            raise ValueError(f"duplicate node {node.node_id!r}")
        node_map[node.node_id] = node
    if coordinator not in node_map:
        raise ValueError("coordinator_id is absent from nodes")

    link_map: dict[tuple[str, str], MeshLinkProfile] = {}
    for link in links:
        if not isinstance(link, MeshLinkProfile):
            raise TypeError("links must contain MeshLinkProfile values")
        slot = (link.from_node, link.to_node)
        if slot in link_map:
            raise ValueError(f"duplicate link {slot}")
        if link.from_node not in node_map or link.to_node not in node_map:
            raise ValueError(f"link {slot} references an unknown node")
        link_map[slot] = link

    if candidate_node_ids is None:
        candidates = tuple(
            sorted(
                node_id
                for node_id, node in node_map.items()
                if node_id != coordinator
                and node.available
                and (coordinator, node_id) in link_map
                and link_map[(coordinator, node_id)].available
            )
        )
    else:
        raw_candidates = tuple(
            _node_id("candidate_node_id", node_id)
            for node_id in candidate_node_ids
        )
        if len(set(raw_candidates)) != len(raw_candidates):
            raise ValueError("candidate_node_ids cannot contain duplicates")
        candidates = tuple(sorted(raw_candidates))
        for node_id in candidates:
            if node_id == coordinator or node_id not in node_map:
                raise ValueError(f"invalid candidate node {node_id!r}")
            link = link_map.get((coordinator, node_id))
            if not node_map[node_id].available or link is None or not link.available:
                raise ValueError(f"candidate node {node_id!r} has no available route")
    if not candidates:
        raise RoutingAwarePlacementUnavailableError(
            "no available remote owner can hold required experts"
        )

    fixed_contact_cost: dict[str, float] = {}
    per_expert_transfer: dict[str, float] = {}
    shared_input_transfer: dict[str, float] = {}
    per_expert_output_transfer: dict[str, float] = {}
    row_index_transfer: dict[str, float] = {}
    contact_cost: dict[str, float] = {}
    for node_id in candidates:
        link = link_map[(coordinator, node_id)]
        per_direction_staging_ms = 0.0
        if link.host_device_staging_bytes_per_ms > 0:
            per_direction_staging_ms = (
                2 * activation_bytes / link.host_device_staging_bytes_per_ms
            )
        fixed_contact_cost[node_id] = (
            link.round_trip_ms + link.rpc_setup_ms
        )
        per_expert_transfer[node_id] = (
            activation_bytes / link.egress_bytes_per_ms
            + activation_bytes / link.ingress_bytes_per_ms
            + 2 * per_direction_staging_ms
        )
        shared_input_transfer[node_id] = (
            activation_bytes / link.egress_bytes_per_ms
            + per_direction_staging_ms
        )
        per_expert_output_transfer[node_id] = (
            activation_bytes / link.ingress_bytes_per_ms
            + per_direction_staging_ms
        )
        row_index_transfer[node_id] = 4 / link.egress_bytes_per_ms
        contact_cost[node_id] = (
            fixed_contact_cost[node_id]
            + per_expert_transfer[node_id]
            + node_map[node_id].expert_compute_ms_per_token
        )
        if not math.isfinite(contact_cost[node_id]) or contact_cost[node_id] <= 0:
            raise ValueError(f"link cost for {node_id!r} must remain finite")

    edges = _canonical_edges(traces, records)
    peak_experts_per_position = max(len(edge.keys) for edge in edges)
    coordinator_dynamic_bytes = positions_per_wave * peak_experts_per_position * (
        3 * activation_bytes
        + node_map[coordinator].expert_workspace_bytes_per_token
    )
    if (
        node_map[coordinator].reserved_vram_bytes + coordinator_dynamic_bytes
        > node_map[coordinator].resident_vram_budget_bytes
    ):
        raise RoutingAwarePlacementUnavailableError(
            "coordinator VRAM cannot hold the sealed routed activation peak"
        )
    marginal_values: dict[ExpertKey, list[float]] = {}
    for edge in edges:
        for key in edge.keys:
            marginal_values.setdefault(key, []).append(edge.effective_positions)
    marginal = {
        key: math.fsum(sorted(values))
        for key, values in sorted(marginal_values.items())
    }
    if not marginal:
        raise ValueError("traces cannot be empty")

    return _PlanningContext(
        coordinator_id=coordinator,
        records=records,
        nodes=node_map,
        candidate_node_ids=candidates,
        fixed_contact_cost_ms=fixed_contact_cost,
        per_expert_transfer_ms=per_expert_transfer,
        shared_input_transfer_ms=shared_input_transfer,
        per_expert_output_transfer_ms=per_expert_output_transfer,
        row_index_transfer_ms=row_index_transfer,
        contact_cost_ms=contact_cost,
        edges=edges,
        marginal_mass=marginal,
        activation_bytes_per_position=activation_bytes,
        max_positions_per_wave=positions_per_wave,
    )


def _canonical_edges(
    traces: Sequence[RoutingTraceSample],
    records: Mapping[ExpertKey, ExpertRecord],
) -> tuple[_TraceEdge, ...]:
    samples = tuple(traces)
    if not samples:
        raise ValueError("traces cannot be empty")
    grouped: dict[tuple[int, tuple[int, ...]], list[float]] = {}
    for trace in samples:
        if not isinstance(trace, RoutingTraceSample):
            raise TypeError("traces must contain RoutingTraceSample values")
        slot = (trace.layer, trace.expert_ids)
        grouped.setdefault(slot, []).append(trace.effective_positions)

    edges: list[_TraceEdge] = []
    for (layer, expert_ids), masses in sorted(grouped.items()):
        keys = tuple(ExpertKey(layer, expert_id) for expert_id in expert_ids)
        missing = tuple(key for key in keys if key not in records)
        if missing:
            raise RoutingAwarePlacementUnavailableError(
                f"sealed route trace references missing expert records: {missing}"
            )
        edges.append(
            _TraceEdge(
                layer=layer,
                keys=keys,
                effective_positions=math.fsum(sorted(masses)),
            )
        )
    return tuple(edges)


def _initial_remaining(context: _PlanningContext) -> dict[str, int]:
    return {
        node_id: (
            context.nodes[node_id].resident_vram_budget_bytes
            - context.nodes[node_id].reserved_vram_bytes
        )
        for node_id in context.candidate_node_ids
    }


def _dynamic_bytes_per_expert(
    context: _PlanningContext,
    node_id: str,
) -> int:
    return context.max_positions_per_wave * (
        3 * context.activation_bytes_per_position
        + context.nodes[node_id].expert_workspace_bytes_per_token
    )


def _resident_peak_for_node(
    context: _PlanningContext,
    placements: Mapping[ExpertKey, set[str]],
    node_id: str,
    *,
    added_key: ExpertKey | None = None,
) -> int:
    return max(
        (
            sum(
                1
                for key in edge.keys
                if node_id in placements.get(key, ()) or key == added_key
            )
            for edge in context.edges
        ),
        default=0,
    )


def _static_remaining(
    context: _PlanningContext,
    placements: Mapping[ExpertKey, set[str]],
) -> dict[str, int]:
    return {
        node_id: (
            context.nodes[node_id].resident_vram_budget_bytes
            - context.nodes[node_id].reserved_vram_bytes
            - sum(
                context.records[key].byte_size
                for key, owners in placements.items()
                if node_id in owners
            )
        )
        for node_id in context.candidate_node_ids
    }


def _place_required_coverage(
    context: _PlanningContext,
    *,
    routing_aware: bool,
) -> tuple[dict[ExpertKey, set[str]], dict[str, int]]:
    placements: dict[ExpertKey, set[str]] = {}
    remaining = _initial_remaining(context)
    required = tuple(sorted(context.marginal_mass))
    if sum(remaining.values()) < sum(
        context.records[key].byte_size for key in required
    ):
        raise RoutingAwarePlacementUnavailableError(
            "owner VRAM cannot cover every expert required by the sealed traces"
        )
    for key in required:
        if not any(
            remaining[node_id]
            >= context.records[key].byte_size
            + _dynamic_bytes_per_expert(context, node_id)
            for node_id in context.candidate_node_ids
        ):
            raise RoutingAwarePlacementUnavailableError(
                "owner VRAM cannot cover required expert "
                f"{key} on any exact route within VRAM"
            )

    coverage_search_was_inconclusive = False
    while len(placements) < len(required):
        best: tuple[tuple[object, ...], ExpertKey, str, int] | None = None
        for key in required:
            if key in placements:
                continue
            record = context.records[key]
            for node_id in context.candidate_node_ids:
                prior_peak = _resident_peak_for_node(
                    context,
                    placements,
                    node_id,
                )
                next_peak = _resident_peak_for_node(
                    context,
                    placements,
                    node_id,
                    added_key=key,
                )
                incremental_dynamic = (
                    next_peak - prior_peak
                ) * _dynamic_bytes_per_expert(context, node_id)
                placement_bytes = record.byte_size + incremental_dynamic
                if remaining[node_id] < placement_bytes:
                    continue
                simulated = dict(remaining)
                simulated[node_id] -= placement_bytes
                feasibility = _remaining_coverage_is_possible(
                    context,
                    placements,
                    key,
                    simulated,
                )
                if feasibility is False:
                    continue
                if feasibility is None:
                    coverage_search_was_inconclusive = True
                affinity = (
                    _hyperedge_affinity(context, placements, key, node_id)
                    if routing_aware
                    else 0.0
                )
                heat = context.marginal_mass[key]
                value = heat + affinity
                density = value / (
                    record.byte_size * context.contact_cost_ms[node_id]
                )
                rank: tuple[object, ...] = (
                    -record.byte_size,
                    -density,
                    -affinity,
                    -heat,
                    context.contact_cost_ms[node_id],
                    key,
                    node_id,
                )
                candidate = (rank, key, node_id, placement_bytes)
                if best is None or candidate[0] < best[0]:
                    best = candidate
        if best is None:
            uncovered = tuple(key for key in required if key not in placements)
            try:
                repaired = _search_required_coverage(
                    context,
                    routing_aware=routing_aware,
                )
            except _CoverageRepairSearchLimit:
                raise RoutingAwarePlacementSearchLimitError(
                    "bounded dynamic-VRAM coverage search could not prove a "
                    f"feasible placement for {uncovered}"
                ) from None
            if repaired is not None:
                return repaired, _static_remaining(context, repaired)
            if coverage_search_was_inconclusive:
                raise RoutingAwarePlacementSearchLimitError(
                    "irregular static packing remained inconclusive while "
                    f"covering {uncovered}"
                )
            raise RoutingAwarePlacementUnavailableError(
                f"no exact dynamic-VRAM placement can cover {uncovered}"
            )
        _, key, node_id, placement_bytes = best
        placements[key] = {node_id}
        remaining[node_id] -= placement_bytes
    return placements, _static_remaining(context, placements)


def _search_required_coverage(
    context: _PlanningContext,
    *,
    routing_aware: bool,
) -> dict[ExpertKey, set[str]] | None:
    """Exhaustively repair small greedy dead ends under placement-aware VRAM."""

    required = tuple(
        sorted(
            context.marginal_mass,
            key=lambda key: (
                -context.records[key].byte_size,
                -context.marginal_mass[key],
                key,
            ),
        )
    )
    if len(required) > _COVERAGE_REPAIR_MAX_KEYS:
        raise _CoverageRepairSearchLimit

    placements: dict[ExpertKey, set[str]] = {}
    remaining = _initial_remaining(context)
    states_evaluated = 0

    def search(index: int) -> dict[ExpertKey, set[str]] | None:
        nonlocal states_evaluated
        states_evaluated += 1
        if states_evaluated > _COVERAGE_REPAIR_MAX_STATES:
            raise _CoverageRepairSearchLimit
        if index == len(required):
            return _copy_placements(placements)

        key = required[index]
        record = context.records[key]
        candidates: list[tuple[tuple[object, ...], str, int]] = []
        equivalent_states: set[tuple[object, ...]] = set()
        for node_id in context.candidate_node_ids:
            prior_peak = _resident_peak_for_node(
                context,
                placements,
                node_id,
            )
            next_peak = _resident_peak_for_node(
                context,
                placements,
                node_id,
                added_key=key,
            )
            placement_bytes = record.byte_size + (
                next_peak - prior_peak
            ) * _dynamic_bytes_per_expert(context, node_id)
            if remaining[node_id] < placement_bytes:
                continue

            simulated = dict(remaining)
            simulated[node_id] -= placement_bytes
            future_items = tuple(
                context.records[future].byte_size
                for future in required[index + 1 :]
            )
            if _can_pack_exactly(
                future_items,
                tuple(simulated[value] for value in context.candidate_node_ids),
            ) is False:
                continue

            affinity = (
                _hyperedge_affinity(context, placements, key, node_id)
                if routing_aware
                else 0.0
            )
            node = context.nodes[node_id]
            edge_counts = tuple(
                sum(
                    1
                    for edge_key in edge.keys
                    if node_id in placements.get(edge_key, ())
                )
                for edge in context.edges
            )
            symmetry = (
                remaining[node_id],
                placement_bytes,
                context.contact_cost_ms[node_id],
                node.expert_compute_ms_per_token,
                node.expert_workspace_bytes_per_token,
                edge_counts,
            )
            if symmetry in equivalent_states:
                continue
            equivalent_states.add(symmetry)
            rank: tuple[object, ...] = (
                -affinity,
                context.contact_cost_ms[node_id],
                -remaining[node_id],
                node_id,
            )
            candidates.append((rank, node_id, placement_bytes))

        for _rank, node_id, placement_bytes in sorted(candidates):
            placements[key] = {node_id}
            remaining[node_id] -= placement_bytes
            result = search(index + 1)
            if result is not None:
                return result
            remaining[node_id] += placement_bytes
            del placements[key]
        return None

    return search(0)


def _remaining_coverage_is_possible(
    context: _PlanningContext,
    placements: Mapping[ExpertKey, set[str]],
    candidate_key: ExpertKey,
    remaining: Mapping[str, int],
) -> bool | None:
    uncovered = tuple(
        key
        for key in sorted(context.marginal_mass)
        if key not in placements and key != candidate_key
    )
    return _can_pack_exactly(
        tuple(context.records[key].byte_size for key in uncovered),
        tuple(remaining[node_id] for node_id in context.candidate_node_ids),
    )


def _can_pack_exactly(
    items: tuple[int, ...],
    capacities: tuple[int, ...],
) -> bool | None:
    """Prove small fragmented placements and handle large regular cases fast."""

    ordered_items = tuple(sorted(items, reverse=True))
    ordered_capacities = tuple(sorted((value for value in capacities if value > 0), reverse=True))
    if not ordered_items:
        return True
    if not ordered_capacities or sum(ordered_capacities) < sum(ordered_items):
        return False
    if ordered_items[0] > ordered_capacities[0]:
        return False
    if len(set(ordered_items)) == 1:
        size = ordered_items[0]
        return sum(capacity // size for capacity in ordered_capacities) >= len(
            ordered_items
        )

    # Best-fit decreasing is a cheap constructive proof for the large regular
    # inventories used by real MoE layers. It never returns a false success.
    greedy_capacities = list(ordered_capacities)
    greedy_succeeded = True
    for size in ordered_items:
        fitting = [
            (capacity - size, index)
            for index, capacity in enumerate(greedy_capacities)
            if capacity >= size
        ]
        if not fitting:
            greedy_succeeded = False
            break
        _remainder, index = min(fitting)
        greedy_capacities[index] -= size
    if greedy_succeeded:
        return True
    if len(ordered_items) > 64:
        # General bin packing is NP-hard. Do not turn an exhausted bounded
        # search into a false proof of impossibility.
        return None

    suffix_bytes = [0] * (len(ordered_items) + 1)
    for index in range(len(ordered_items) - 1, -1, -1):
        suffix_bytes[index] = suffix_bytes[index + 1] + ordered_items[index]

    @lru_cache(maxsize=None)
    def search(index: int, remaining_caps: tuple[int, ...]) -> bool:
        if index == len(ordered_items):
            return True
        if sum(remaining_caps) < suffix_bytes[index]:
            return False
        size = ordered_items[index]
        prior_capacity: int | None = None
        for slot, capacity in enumerate(remaining_caps):
            if capacity < size or capacity == prior_capacity:
                continue
            prior_capacity = capacity
            updated = list(remaining_caps)
            updated[slot] -= size
            canonical = tuple(sorted(updated, reverse=True))
            if search(index + 1, canonical):
                return True
        return False

    return search(0, ordered_capacities)


def _hyperedge_affinity(
    context: _PlanningContext,
    placements: Mapping[ExpertKey, set[str]],
    key: ExpertKey,
    node_id: str,
) -> float:
    placed_on_node = {
        placed_key
        for placed_key, owners in placements.items()
        if node_id in owners
    }
    values: list[float] = []
    for edge in context.edges:
        if key not in edge.keys or len(edge.keys) == 1:
            continue
        co_routed = len((set(edge.keys) - {key}) & placed_on_node)
        if co_routed == 0:
            continue
        completion = 1.0 if co_routed == len(edge.keys) - 1 else 0.0
        fraction = co_routed / (len(edge.keys) - 1)
        values.append(edge.effective_positions * (fraction + completion))
    return math.fsum(sorted(values))


def _add_routing_aware_replicas(
    context: _PlanningContext,
    placements: dict[ExpertKey, set[str]],
    remaining: dict[str, int],
) -> dict[ExpertKey, set[str]]:
    current = _project(context, placements)
    while True:
        best: tuple[
            tuple[object, ...],
            ExpertKey,
            str,
            PlacementProjection,
        ] | None = None
        for key in sorted(context.marginal_mass):
            record = context.records[key]
            for node_id in context.candidate_node_ids:
                if node_id in placements[key] or remaining[node_id] < record.byte_size:
                    continue
                simulated = _copy_placements(placements)
                simulated[key].add(node_id)
                try:
                    projection = _project(context, simulated)
                except RoutingAwarePlacementError:
                    continue
                gain = current.projected_link_ms - projection.projected_link_ms
                contacts_saved = (
                    current.weighted_owner_contacts
                    - projection.weighted_owner_contacts
                )
                if gain <= _FLOAT_TOLERANCE:
                    continue
                rank: tuple[object, ...] = (
                    -(gain / record.byte_size),
                    -gain,
                    -contacts_saved,
                    key,
                    node_id,
                )
                candidate = (rank, key, node_id, projection)
                if best is None or candidate[0] < best[0]:
                    best = candidate
        if best is None:
            return placements
        _, key, node_id, current = best
        placements[key].add(node_id)
        remaining[node_id] -= context.records[key].byte_size


def _add_marginal_replicas(
    context: _PlanningContext,
    placements: dict[ExpertKey, set[str]],
    remaining: dict[str, int],
    *,
    byte_allowance: int,
) -> dict[ExpertKey, set[str]]:
    used = 0
    while used < byte_allowance:
        best: tuple[
            tuple[object, ...],
            ExpertKey,
            str,
            PlacementProjection,
        ] | None = None
        for key in sorted(context.marginal_mass):
            record = context.records[key]
            if used + record.byte_size > byte_allowance:
                continue
            current_cost = min(
                context.contact_cost_ms[node_id] for node_id in placements[key]
            )
            for node_id in context.candidate_node_ids:
                if node_id in placements[key] or remaining[node_id] < record.byte_size:
                    continue
                simulated = _copy_placements(placements)
                simulated[key].add(node_id)
                try:
                    projection = _project(context, simulated)
                except RoutingAwarePlacementError:
                    continue
                link_gain = context.marginal_mass[key] * max(
                    0.0,
                    current_cost - context.contact_cost_ms[node_id],
                )
                heat_density = context.marginal_mass[key] / record.byte_size
                rank: tuple[object, ...] = (
                    -(link_gain / record.byte_size),
                    -heat_density,
                    context.contact_cost_ms[node_id],
                    key,
                    node_id,
                )
                candidate = (rank, key, node_id, projection)
                if best is None or candidate[0] < best[0]:
                    best = candidate
        if best is None:
            return placements
        _, key, node_id, _projection = best
        placements[key].add(node_id)
        size = context.records[key].byte_size
        remaining[node_id] -= size
        used += size
    return placements


def _copy_placements(
    placements: Mapping[ExpertKey, set[str]],
) -> dict[ExpertKey, set[str]]:
    return {key: set(owners) for key, owners in placements.items()}


def _replica_tuple(
    context: _PlanningContext,
    placements: Mapping[ExpertKey, set[str]],
) -> tuple[ResidentExpertReplica, ...]:
    return tuple(
        ResidentExpertReplica(
            key=key,
            node_id=node_id,
            content_id=context.records[key].content_id,
        )
        for key in sorted(placements)
        for node_id in sorted(placements[key])
    )


def _project(
    context: _PlanningContext,
    placements: Mapping[ExpertKey, set[str]],
) -> PlacementProjection:
    for key in context.marginal_mass:
        if not placements.get(key):
            raise RoutingAwarePlacementUnavailableError(
                f"required expert {key} has no exact resident route"
            )

    weighted_contacts: list[float] = []
    weighted_rpc_v1_bytes: list[float] = []
    weighted_coalesced_exact_bytes: list[float] = []
    weighted_coalesced_row_index_bytes: list[float] = []
    weighted_owner_partial_floor_bytes: list[float] = []
    weighted_link_ms: list[float] = []
    weighted_load: dict[str, list[float]] = {
        node_id: [] for node_id in context.candidate_node_ids
    }
    peak_load = {node_id: 0 for node_id in context.candidate_node_ids}
    routes: list[TraceOwnerProjection] = []
    for edge in context.edges:
        owners, owner_for_key, link_ms = _select_trace_owners(
            context,
            placements,
            edge,
        )
        per_owner_count = {owner: 0 for owner in owners}
        for owner in owner_for_key.values():
            per_owner_count[owner] += 1
        for node_id, count in per_owner_count.items():
            weighted_load[node_id].append(edge.effective_positions * count)
            peak_load[node_id] = max(peak_load[node_id], count)
        contacts = len(owners)
        rpc_v1_activation_bytes = (
            2 * context.activation_bytes_per_position * len(edge.keys)
        )
        # Exact input coalescing sends one hidden row per contacted owner, but
        # still returns one raw output per expert so the coordinator can keep
        # gate multiplication and canonical floating-point reduction intact.
        coalesced_exact_activation_bytes = 0
        coalesced_exact_row_index_bytes = 0
        for count in per_owner_count.values():
            v1_owner_bytes = 2 * context.activation_bytes_per_position * count
            exact_owner_bytes = (
                context.activation_bytes_per_position * (count + 1)
            )
            row_bytes = 4 * count
            if exact_owner_bytes + row_bytes < v1_owner_bytes:
                coalesced_exact_activation_bytes += exact_owner_bytes
                coalesced_exact_row_index_bytes += row_bytes
            else:
                coalesced_exact_activation_bytes += v1_owner_bytes
        # This lower floor would require weighted owner partials.  It is kept
        # separate because regrouping floating-point sums is not the sealed
        # reference-equivalent contract implemented by the exact transport.
        owner_partial_activation_byte_floor = (
            2 * context.activation_bytes_per_position * contacts
        )
        weighted_contacts.append(edge.effective_positions * contacts)
        weighted_rpc_v1_bytes.append(
            edge.effective_positions * rpc_v1_activation_bytes
        )
        weighted_coalesced_exact_bytes.append(
            edge.effective_positions * coalesced_exact_activation_bytes
        )
        weighted_coalesced_row_index_bytes.append(
            edge.effective_positions * coalesced_exact_row_index_bytes
        )
        weighted_owner_partial_floor_bytes.append(
            edge.effective_positions * owner_partial_activation_byte_floor
        )
        weighted_link_ms.append(edge.effective_positions * link_ms)
        routes.append(
            TraceOwnerProjection(
                layer=edge.layer,
                expert_ids=tuple(key.expert for key in edge.keys),
                effective_positions=edge.effective_positions,
                owner_ids=owners,
                owner_for_expert=tuple(
                    (key.expert, owner_for_key[key]) for key in edge.keys
                ),
                owner_contacts_per_position=contacts,
                rpc_v1_activation_bytes_per_position=rpc_v1_activation_bytes,
                coalesced_exact_activation_bytes_per_position=(
                    coalesced_exact_activation_bytes
                ),
                coalesced_exact_row_index_bytes_per_position=(
                    coalesced_exact_row_index_bytes
                ),
                owner_partial_activation_byte_floor_per_position=(
                    owner_partial_activation_byte_floor
                ),
                projected_link_ms_per_position=link_ms,
            )
        )

    weighted_positions = math.fsum(
        edge.effective_positions for edge in context.edges
    )
    total_contacts = math.fsum(weighted_contacts)
    total_replica_bytes = sum(
        context.records[key].byte_size * len(owners)
        for key, owners in placements.items()
    )
    used_vram: list[tuple[str, int]] = []
    for node_id in context.candidate_node_ids:
        expert_bytes = sum(
            context.records[key].byte_size
            for key, owners in placements.items()
            if node_id in owners
        )
        dynamic_bytes = context.max_positions_per_wave * peak_load[node_id] * (
            3 * context.activation_bytes_per_position
            + context.nodes[node_id].expert_workspace_bytes_per_token
        )
        used = (
            context.nodes[node_id].reserved_vram_bytes
            + expert_bytes
            + dynamic_bytes
        )
        if used > context.nodes[node_id].resident_vram_budget_bytes:
            raise RoutingAwarePlacementError(
                f"placement exceeds VRAM budget on {node_id!r}"
            )
        used_vram.append((node_id, used))

    weighted_by_node = tuple(
        (node_id, math.fsum(weighted_load[node_id]))
        for node_id in context.candidate_node_ids
    )
    peak_by_node = tuple(
        (node_id, peak_load[node_id])
        for node_id in context.candidate_node_ids
    )
    return PlacementProjection(
        weighted_positions=weighted_positions,
        weighted_owner_contacts=total_contacts,
        mean_owner_contacts_per_position=total_contacts / weighted_positions,
        max_owner_contacts_per_position=max(
            route.owner_contacts_per_position for route in routes
        ),
        rpc_v1_activation_bytes=math.fsum(weighted_rpc_v1_bytes),
        coalesced_exact_activation_bytes=math.fsum(
            weighted_coalesced_exact_bytes
        ),
        coalesced_exact_row_index_bytes=math.fsum(
            weighted_coalesced_row_index_bytes
        ),
        owner_partial_activation_byte_floor=math.fsum(
            weighted_owner_partial_floor_bytes
        ),
        projected_link_ms=math.fsum(weighted_link_ms),
        peak_experts_per_position=max(peak_load.values(), default=0),
        peak_experts_per_position_by_node=peak_by_node,
        weighted_expert_activations_by_node=weighted_by_node,
        used_vram_bytes_by_node=tuple(used_vram),
        total_replica_bytes=total_replica_bytes,
        trace_routes=tuple(routes),
    )


def _select_trace_owners(
    context: _PlanningContext,
    placements: Mapping[ExpertKey, set[str]],
    edge: _TraceEdge,
) -> tuple[tuple[str, ...], dict[ExpertKey, str], float]:
    full_mask = (1 << len(edge.keys)) - 1
    static_remaining = _static_remaining(context, placements)
    node_masks: list[tuple[str, int, int]] = []
    for node_id in context.candidate_node_ids:
        mask = 0
        for index, key in enumerate(edge.keys):
            if node_id in placements.get(key, set()):
                mask |= 1 << index
        if mask:
            max_dynamic_experts = max(
                0,
                static_remaining[node_id]
                // _dynamic_bytes_per_expert(context, node_id),
            )
            if max_dynamic_experts:
                node_masks.append((node_id, mask, max_dynamic_experts))

    states: dict[int, list[_TraceRouteState]] = {
        0: [_TraceRouteState(0.0, 0, 0, (), ())]
    }
    for node_id, node_mask, max_dynamic_experts in node_masks:
        updated = {mask: list(values) for mask, values in states.items()}
        for mask, route_states in states.items():
            for state in route_states:
                available = node_mask & ~mask
                subset = available
                while subset:
                    combined = mask | subset
                    count = subset.bit_count()
                    if count > max_dynamic_experts:
                        subset = (subset - 1) & available
                        continue
                    v1_transfer_ms = count * context.per_expert_transfer_ms[node_id]
                    coalesced_transfer_ms = (
                        context.shared_input_transfer_ms[node_id]
                        + count * context.per_expert_output_transfer_ms[node_id]
                        + count * context.row_index_transfer_ms[node_id]
                    )
                    use_coalesced = (
                        coalesced_transfer_ms
                        < v1_transfer_ms - _FLOAT_TOLERANCE
                    )
                    request_bytes = (
                        context.activation_bytes_per_position + 4 * count
                        if use_coalesced
                        else context.activation_bytes_per_position * count
                    )
                    response_bytes = context.activation_bytes_per_position * count
                    owner_ms = (
                        context.fixed_contact_cost_ms[node_id]
                        + (
                            coalesced_transfer_ms
                            if use_coalesced
                            else v1_transfer_ms
                        )
                        + count
                        * context.nodes[node_id].expert_compute_ms_per_token
                    )
                    candidate = _TraceRouteState(
                        max_owner_ms=max(state.max_owner_ms, owner_ms),
                        request_bytes=state.request_bytes + request_bytes,
                        response_bytes=state.response_bytes + response_bytes,
                        owners=state.owners + (node_id,),
                        assignments=tuple(
                            sorted(
                                (
                                    *state.assignments,
                                    *(
                                        (index, node_id)
                                        for index in range(len(edge.keys))
                                        if subset & (1 << index)
                                    ),
                                )
                            )
                        ),
                    )
                    _insert_route_state(updated.setdefault(combined, []), candidate)
                    subset = (subset - 1) & available
        states = updated
    complete = states.get(full_mask)
    if not complete:
        raise RoutingAwarePlacementUnavailableError(
            f"required trace at layer {edge.layer} has no complete exact route"
        )
    selected = min(complete, key=lambda state: _route_state_rank(context, state))
    link_ms = _route_state_projected_ms(context, selected)
    owners = selected.owners
    assignments = selected.assignments
    owner_for_key = {
        edge.keys[index]: owner for index, owner in assignments
    }
    if set(owner_for_key) != set(edge.keys):
        raise RoutingAwarePlacementUnavailableError(
            f"required trace at layer {edge.layer} has an incomplete owner route"
        )
    return owners, owner_for_key, link_ms


def _route_state_projected_ms(
    context: _PlanningContext,
    state: _TraceRouteState,
) -> float:
    coordinator = context.nodes[context.coordinator_id]
    directional_nic_ms = [0.0]
    if state.request_bytes > 0 and coordinator.aggregate_egress_bytes_per_ms > 0:
        directional_nic_ms.append(
            state.request_bytes / coordinator.aggregate_egress_bytes_per_ms
        )
    if state.response_bytes > 0 and coordinator.aggregate_ingress_bytes_per_ms > 0:
        directional_nic_ms.append(
            state.response_bytes / coordinator.aggregate_ingress_bytes_per_ms
        )
    nic_ms = max(directional_nic_ms)
    return max(state.max_owner_ms, nic_ms)


def _route_state_rank(
    context: _PlanningContext,
    state: _TraceRouteState,
) -> tuple[object, ...]:
    return (
        _route_state_projected_ms(context, state),
        state.request_bytes + state.response_bytes,
        len(state.owners),
        state.owners,
        state.assignments,
    )


def _insert_route_state(
    frontier: list[_TraceRouteState],
    candidate: _TraceRouteState,
) -> None:
    """Keep a Pareto frontier for parallel owner and shared-NIC bounds."""

    candidate_metrics = (
        candidate.max_owner_ms,
        candidate.request_bytes,
        candidate.response_bytes,
    )
    survivors: list[_TraceRouteState] = []
    for current in frontier:
        current_metrics = (
            current.max_owner_ms,
            current.request_bytes,
            current.response_bytes,
        )
        current_dominates = all(
            left <= right + _FLOAT_TOLERANCE
            for left, right in zip(current_metrics, candidate_metrics)
        )
        candidate_dominates = all(
            left <= right + _FLOAT_TOLERANCE
            for left, right in zip(candidate_metrics, current_metrics)
        )
        if current_dominates:
            if candidate_dominates and (
                len(candidate.owners),
                candidate.owners,
                candidate.assignments,
            ) < (
                len(current.owners),
                current.owners,
                current.assignments,
            ):
                continue
            return
        if not candidate_dominates:
            survivors.append(current)
    survivors.append(candidate)
    frontier[:] = survivors


__all__ = [
    "ROUTING_AWARE_PLACEMENT_SCHEMA",
    "PlacementProjection",
    "RoutingAwarePlacementError",
    "RoutingAwarePlacementPlan",
    "RoutingAwarePlacementSearchLimitError",
    "RoutingAwarePlacementUnavailableError",
    "RoutingTraceSample",
    "TraceOwnerProjection",
    "plan_routing_aware_expert_placement",
]
