"""Exact routing and cost model for a resident-expert activation mesh.

This module supplies the planner/executor used by the opt-in RAM-backed MoE
stage integration.  The target model's router remains authoritative,
while each selected expert may run from the coordinator GPU cache, after an
exact local RAM-to-device load, or on a remote node that already holds an
identity-matched resident replica.  Expert owners within one transformer layer
can run concurrently; transformer layers themselves remain sequential.
"""

from __future__ import annotations

from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
import heapq
from itertools import product
import math
import threading
from typing import Callable, Iterable, Mapping, Protocol, Sequence

import torch

from .ram_expert_cache import ExpertKey, ExpertRecord


RESIDENT_EXPERT_MESH_SCHEMA = "gdlp-resident-expert-mesh/1"
# Keep the exact Cartesian solver on the latency-critical top-k sized cases.
# Larger routed unions use the explicitly labelled local-search path; 100k
# full tensor-route evaluations made the offline sensitivity simulator itself
# a bottleneck without improving its chosen assignment.
_EXACT_ASSIGNMENT_COMBINATION_LIMIT = 1_024
# Heuristic planning must stay off the inference critical path even for long
# prefills.  The repair budget counts attempted row routes (not expensive full
# tensor-plan evaluations), while the local pass is deliberately single-sweep.
_HEURISTIC_FEASIBILITY_REPAIR_STATE_LIMIT = 65_536
_HEURISTIC_LOCAL_MOVE_PROBE_LIMIT = 4_096
# A public mesh may know thousands of identity-equivalent replicas for one
# expert.  Scoring every replica for every routed row is itself slower than an
# inference wave.  Keep a deterministic performance frontier on the critical
# path; capacity-driven extensions below retain enough owners for the rows of
# that expert, while the plan truthfully loses its global-optimality claim.
_HEURISTIC_REMOTE_CANDIDATE_LIMIT_PER_KEY = 64


def _list_scheduled_makespan_ms(
    owner_times: Mapping[str, float],
    max_inflight: int,
) -> float:
    """Deterministic LPT schedule for the bounded owner worker pool."""

    if not owner_times:
        return 0.0
    # The overwhelmingly common search state fits in one wave.  Its exact LPT
    # result is simply the slowest owner; avoid sorting/heap work for every
    # speculative add/remove probe.
    if len(owner_times) <= max_inflight:
        return max(owner_times.values())
    lane_count = min(max_inflight, len(owner_times))
    lanes = [(0.0, lane) for lane in range(lane_count)]
    heapq.heapify(lanes)
    for _owner_id, duration_ms in sorted(
        owner_times.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        available_ms, lane = heapq.heappop(lanes)
        heapq.heappush(lanes, (available_ms + duration_ms, lane))
    return max(available_ms for available_ms, _lane in lanes)


def _modeled_coordinator_overhead_ms(
    *,
    remote_owner_count: int,
    transport_payload_bytes: int,
    assignment_count: int,
    rpc_dispatch_ms_per_owner: float | None,
    serialization_bytes_per_ms: float | None,
    reduction_ms_per_assignment: float | None,
) -> float:
    """Return only calibrated serial coordinator work; missing terms stay 0."""

    return math.fsum(
        (
            (
                remote_owner_count * rpc_dispatch_ms_per_owner
                if rpc_dispatch_ms_per_owner is not None
                else 0.0
            ),
            (
                transport_payload_bytes / serialization_bytes_per_ms
                if serialization_bytes_per_ms is not None
                and transport_payload_bytes
                else 0.0
            ),
            (
                assignment_count * reduction_ms_per_assignment
                if reduction_ms_per_assignment is not None
                else 0.0
            ),
        )
    )


class ResidentExpertMeshError(RuntimeError):
    """Base class for closed mesh failures."""


class ExpertRouteUnavailableError(ResidentExpertMeshError):
    """No exact path exists for an expert selected by the target router."""


class ExpertResidentSlotUnavailableError(ExpertRouteUnavailableError):
    """A previously advertised exact replica is no longer physically resident."""

    def __init__(self, key: ExpertKey, message: str | None = None) -> None:
        if not isinstance(key, ExpertKey):
            raise TypeError("resident slot key must be ExpertKey")
        self.key = key
        super().__init__(
            message or f"physical resident slot is unavailable for {key}"
        )


class ExpertAssignmentSearchLimitError(ExpertRouteUnavailableError):
    """A bounded heuristic could not prove a feasible large assignment."""


class _RetryResidentSlots(RuntimeError):
    def __init__(self, slots: frozenset[tuple[str, ExpertKey]]) -> None:
        self.slots = slots
        super().__init__("retry exact layer after resident-slot loss")


def _integer(name: str, value: object, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(name: str, value: object, *, minimum: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and >= {minimum}") from exc
    if not math.isfinite(number) or number < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return number


def _name(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} cannot be empty")
    return value.strip()


@dataclass(frozen=True)
class MeshNodeProfile:
    node_id: str
    resident_vram_budget_bytes: int
    reserved_vram_bytes: int
    expert_compute_ms_per_token: float
    ram_to_device_gbytes_per_second: float = 0.0
    available: bool = True
    aggregate_ingress_mbps: float = 0.0
    aggregate_egress_mbps: float = 0.0
    expert_workspace_bytes_per_token: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", _name("node_id", self.node_id))
        budget = _integer(
            "resident_vram_budget_bytes",
            self.resident_vram_budget_bytes,
            minimum=1,
        )
        reserved = _integer("reserved_vram_bytes", self.reserved_vram_bytes)
        if reserved > budget:
            raise ValueError("reserved_vram_bytes exceeds the resident VRAM budget")
        object.__setattr__(
            self,
            "expert_compute_ms_per_token",
            _finite(
                "expert_compute_ms_per_token",
                self.expert_compute_ms_per_token,
            ),
        )
        object.__setattr__(
            self,
            "ram_to_device_gbytes_per_second",
            _finite(
                "ram_to_device_gbytes_per_second",
                self.ram_to_device_gbytes_per_second,
            ),
        )
        if not isinstance(self.available, bool):
            raise TypeError("available must be boolean")
        object.__setattr__(
            self,
            "aggregate_ingress_mbps",
            _finite("aggregate_ingress_mbps", self.aggregate_ingress_mbps),
        )
        object.__setattr__(
            self,
            "aggregate_egress_mbps",
            _finite("aggregate_egress_mbps", self.aggregate_egress_mbps),
        )
        _integer(
            "expert_workspace_bytes_per_token",
            self.expert_workspace_bytes_per_token,
        )

    @property
    def ram_to_device_bytes_per_ms(self) -> float:
        return self.ram_to_device_gbytes_per_second * 1_000_000.0

    @property
    def aggregate_ingress_bytes_per_ms(self) -> float:
        return self.aggregate_ingress_mbps * 125.0

    @property
    def aggregate_egress_bytes_per_ms(self) -> float:
        return self.aggregate_egress_mbps * 125.0


@dataclass(frozen=True)
class MeshLinkProfile:
    from_node: str
    to_node: str
    round_trip_ms: float
    bandwidth_mbps: float
    available: bool = True
    rpc_setup_ms_per_batch: float | None = None
    host_device_staging_gbytes_per_second: float = 0.0
    egress_bandwidth_mbps: float | None = None
    ingress_bandwidth_mbps: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "from_node", _name("from_node", self.from_node))
        object.__setattr__(self, "to_node", _name("to_node", self.to_node))
        if self.from_node == self.to_node:
            raise ValueError("mesh links must connect distinct nodes")
        object.__setattr__(
            self,
            "round_trip_ms",
            _finite("round_trip_ms", self.round_trip_ms),
        )
        bandwidth = _finite("bandwidth_mbps", self.bandwidth_mbps)
        if bandwidth <= 0:
            raise ValueError("bandwidth_mbps must be positive")
        object.__setattr__(self, "bandwidth_mbps", bandwidth)
        for field_name in ("egress_bandwidth_mbps", "ingress_bandwidth_mbps"):
            directional = getattr(self, field_name)
            if directional is None:
                object.__setattr__(self, field_name, bandwidth)
                continue
            normalized = _finite(field_name, directional)
            if normalized <= 0:
                raise ValueError(f"{field_name} must be positive")
            object.__setattr__(self, field_name, normalized)
        if not isinstance(self.available, bool):
            raise TypeError("available must be boolean")
        if self.rpc_setup_ms_per_batch is not None:
            object.__setattr__(
                self,
                "rpc_setup_ms_per_batch",
                _finite("rpc_setup_ms_per_batch", self.rpc_setup_ms_per_batch),
            )
        object.__setattr__(
            self,
            "host_device_staging_gbytes_per_second",
            _finite(
                "host_device_staging_gbytes_per_second",
                self.host_device_staging_gbytes_per_second,
            ),
        )

    @property
    def bytes_per_ms(self) -> float:
        # Backwards-compatible symmetric calibration. Directional transport
        # models use the two properties below.
        return self.bandwidth_mbps * 125.0

    @property
    def egress_bytes_per_ms(self) -> float:
        assert self.egress_bandwidth_mbps is not None
        return self.egress_bandwidth_mbps * 125.0

    @property
    def ingress_bytes_per_ms(self) -> float:
        assert self.ingress_bandwidth_mbps is not None
        return self.ingress_bandwidth_mbps * 125.0

    @property
    def host_device_staging_bytes_per_ms(self) -> float:
        return self.host_device_staging_gbytes_per_second * 1_000_000.0

    @property
    def rpc_setup_ms(self) -> float:
        return (
            float(self.rpc_setup_ms_per_batch)
            if self.rpc_setup_ms_per_batch is not None
            else 0.0
        )


@dataclass(frozen=True, order=True)
class ResidentExpertReplica:
    key: ExpertKey
    node_id: str
    content_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", _name("node_id", self.node_id))
        object.__setattr__(self, "content_id", _name("content_id", self.content_id))


@dataclass(frozen=True)
class ExplicitExpertDemand:
    """Demand supplied by measurements; missing keys receive no inferred heat."""

    key: ExpertKey
    active_probability: float
    expected_assignments_per_wave: float

    def __post_init__(self) -> None:
        probability = _finite("active_probability", self.active_probability)
        assignments = _finite(
            "expected_assignments_per_wave",
            self.expected_assignments_per_wave,
        )
        if probability > 1:
            raise ValueError("active_probability must be in [0, 1]")
        if assignments < probability:
            raise ValueError(
                "expected_assignments_per_wave cannot be below active_probability"
            )
        object.__setattr__(self, "active_probability", probability)
        object.__setattr__(self, "expected_assignments_per_wave", assignments)


@dataclass(frozen=True)
class AuthoritativeRouting:
    """Exact top-k result produced by the target model router."""

    expert_ids: torch.Tensor
    expert_weights: torch.Tensor
    _expert_ids_cpu: torch.Tensor = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.expert_ids.ndim != 2 or self.expert_ids.numel() < 1:
            raise ValueError("expert_ids must have shape [positions, top_k]")
        if self.expert_weights.shape != self.expert_ids.shape:
            raise ValueError("expert_weights must match expert_ids")
        if self.expert_ids.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise TypeError("expert_ids must use an integer dtype")
        # Seal one grouped CPU snapshot.  Planning may inspect every routed
        # assignment, but it must never synchronize CUDA once per top-k slot.
        expert_ids_cpu = (
            self.expert_ids.detach()
            .to(device="cpu", copy=True)
            .contiguous()
        )
        if bool(torch.any(expert_ids_cpu < 0)):
            raise ValueError("expert_ids cannot be negative")
        object.__setattr__(self, "_expert_ids_cpu", expert_ids_cpu)
        if not torch.is_floating_point(self.expert_weights):
            raise TypeError("expert_weights must use a floating dtype")
        if not torch.isfinite(self.expert_weights).all().item():
            raise ValueError("expert_weights must be finite")

    @property
    def token_count(self) -> int:
        return int(self.expert_ids.shape[0])

    @property
    def top_k(self) -> int:
        return int(self.expert_ids.shape[1])

    @property
    def authoritative_experts(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                {
                    int(expert_id)
                    for expert_id in self._expert_ids_cpu.reshape(-1).tolist()
                }
            )
        )


@dataclass(frozen=True, order=True)
class ExpertAssignment:
    token_index: int
    slot_index: int


@dataclass(frozen=True)
class OwnerExpertBatchItem:
    key: ExpertKey
    content_id: str
    activations: torch.Tensor
    require_resident: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.require_resident, bool):
            raise TypeError("require_resident must be boolean")


@dataclass(frozen=True)
class OwnerCoalescedExpertBatchItem:
    """One expert view into an owner-shared activation matrix.

    ``row_indices`` preserves the original assignment order for that expert.
    The coordinator keeps router gates and canonical reduction local, so input
    coalescing changes transport/staging only and not the model arithmetic.
    """

    key: ExpertKey
    content_id: str
    row_indices: tuple[int, ...]
    require_resident: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.key, ExpertKey):
            raise TypeError("coalesced expert key must be ExpertKey")
        if not isinstance(self.content_id, str) or not self.content_id.strip():
            raise ValueError("coalesced expert content_id cannot be empty")
        if not isinstance(self.row_indices, tuple) or not self.row_indices:
            raise ValueError("coalesced expert row_indices cannot be empty")
        if any(
            not isinstance(index, int) or isinstance(index, bool) or index < 0
            for index in self.row_indices
        ):
            raise ValueError("coalesced expert row indices must be non-negative integers")
        if not isinstance(self.require_resident, bool):
            raise TypeError("require_resident must be boolean")


@dataclass(frozen=True)
class OwnerExpertBatchResult:
    key: ExpertKey
    output: torch.Tensor


class ExpertOwner(Protocol):
    node_id: str

    def has_expert(self, key: ExpertKey, content_id: str) -> bool: ...

    def is_expert_resident(self, key: ExpertKey, content_id: str) -> bool: ...

    def execute_batch(
        self,
        items: Sequence[OwnerExpertBatchItem],
    ) -> tuple[OwnerExpertBatchResult, ...]: ...


class CoalescedExpertOwner(ExpertOwner, Protocol):
    """Optional exact-input extension; v1-only owners remain valid."""

    def execute_coalesced_batch(
        self,
        shared_activations: torch.Tensor,
        items: Sequence[OwnerCoalescedExpertBatchItem],
    ) -> tuple[OwnerExpertBatchResult, ...]: ...


@dataclass(frozen=True)
class ExpertDispatch:
    key: ExpertKey
    owner_id: str
    path: str
    assignments: tuple[ExpertAssignment, ...]
    exposed_ms: float
    activation_round_trip_bytes: int
    host_device_staging_bytes: int
    coordinator_dynamic_vram_bytes: int
    owner_dynamic_vram_bytes: int
    weight_loaded_bytes: int
    weight_avoided_bytes: int
    transport_calibration_required: bool
    workspace_calibration_required: bool


@dataclass(frozen=True)
class LayerMeshPlan:
    layer: int
    token_count: int
    top_k: int
    authoritative_expert_ids: tuple[tuple[int, ...], ...]
    dispatches: tuple[ExpertDispatch, ...]
    owner_exposed_ms: tuple[tuple[str, float], ...]
    owner_peak_vram_bytes: tuple[tuple[str, int], ...]
    per_owner_lower_bound_ms: float
    owner_scheduled_makespan_ms: float
    coordinator_nic_lower_bound_ms: float
    coordinator_overhead_ms: float
    exposed_ms: float
    activation_round_trip_bytes: int
    activation_request_bytes: int
    activation_response_bytes: int
    route_metadata_bytes: int
    transport_payload_bytes: int
    coalesced_owner_ids: tuple[str, ...]
    host_device_staging_bytes: int
    weight_loaded_bytes: int
    weight_avoided_bytes: int
    transport_calibration_required: bool
    coordination_calibration_required: bool
    workspace_calibration_required: bool
    max_inflight_owner_rpcs: int
    assignment_strategy: str
    assignment_optimality_proven: bool
    assignment_states_evaluated: int


@dataclass(frozen=True)
class MeshWaveProjection:
    schema: str
    layer_plans: tuple[LayerMeshPlan, ...]
    committed_tokens: float
    exposed_ms_per_wave: float
    activation_round_trip_bytes_per_wave: int
    activation_request_bytes_per_wave: int
    activation_response_bytes_per_wave: int
    route_metadata_bytes_per_wave: int
    transport_payload_bytes_per_wave: int
    host_device_staging_bytes_per_wave: int
    weight_loaded_bytes_per_wave: int
    weight_avoided_bytes_per_wave: int
    transport_calibration_required: bool
    workspace_calibration_required: bool
    tokens_per_second: float


@dataclass(frozen=True)
class HotReplicaPlan:
    placements: tuple[ResidentExpertReplica, ...]
    used_vram_bytes_by_node: tuple[tuple[str, int], ...]
    estimated_exposed_ms_saved_per_wave: float


@dataclass(frozen=True)
class _Candidate:
    key: ExpertKey
    owner_id: str
    path: str
    duration_ms: float
    activation_bytes: int
    host_device_staging_bytes: int
    coordinator_dynamic_vram_bytes: int
    owner_dynamic_vram_bytes: int
    weight_loaded_bytes: int
    weight_avoided_bytes: int
    transport_calibration_required: bool
    workspace_calibration_required: bool


@dataclass(frozen=True, order=True)
class _AssignmentUnit:
    """One authoritative router row/slot that may choose any exact replica."""

    key: ExpertKey
    ordinal: int
    assignment: ExpertAssignment


@dataclass(frozen=True)
class _AssignmentEvaluation:
    selected: tuple[tuple[_AssignmentUnit, _Candidate], ...]
    dispatches: tuple[ExpertDispatch, ...]
    owner_exposed_ms: tuple[tuple[str, float], ...]
    owner_peak_vram_bytes: tuple[tuple[str, int], ...]
    per_owner_lower_bound_ms: float
    owner_scheduled_makespan_ms: float
    coordinator_nic_lower_bound_ms: float
    coordinator_overhead_ms: float
    exposed_ms: float
    activation_round_trip_bytes: int
    activation_request_bytes: int
    activation_response_bytes: int
    route_metadata_bytes: int
    transport_payload_bytes: int
    coalesced_owner_ids: tuple[str, ...]
    host_device_staging_bytes: int
    weight_loaded_bytes: int
    weight_avoided_bytes: int
    transport_calibration_required: bool
    coordination_calibration_required: bool
    workspace_calibration_required: bool
    signature: tuple[tuple[int, int, int, int, int, str], ...]


class _IncrementalAssignmentState:
    """Reversible exact-cost state for bounded row-level search.

    Dynamic VRAM is linear in routed rows. Remote transport is grouped once per
    owner and local RAM weight cost once per owner/key/path, so all quantities
    needed by the heuristic can be updated without rebuilding and sorting the
    complete selection for every candidate row.
    """

    def __init__(
        self,
        *,
        coordinator_id: str,
        nodes: Mapping[str, MeshNodeProfile],
        links: Mapping[tuple[str, str], MeshLinkProfile],
        activation_bytes_per_token: int,
        static_vram_bytes: Mapping[str, int],
        v1_only_nodes: frozenset[str],
        row_index_bytes_by_node: Mapping[str, int],
        max_inflight_owner_rpcs: int,
        coordinator_rpc_dispatch_ms_per_owner: float | None,
        coordinator_serialization_bytes_per_ms: float | None,
        coordinator_reduction_ms_per_assignment: float | None,
    ) -> None:
        self.coordinator_id = coordinator_id
        self.nodes = nodes
        self.links = links
        self.activation_bytes_per_token = activation_bytes_per_token
        self.static_vram_bytes = static_vram_bytes
        self.v1_only_nodes = v1_only_nodes
        self.row_index_bytes_by_node = row_index_bytes_by_node
        self.max_inflight_owner_rpcs = max_inflight_owner_rpcs
        self.coordinator_rpc_dispatch_ms_per_owner = (
            coordinator_rpc_dispatch_ms_per_owner
        )
        self.coordinator_serialization_bytes_per_ms = (
            coordinator_serialization_bytes_per_ms
        )
        self.coordinator_reduction_ms_per_assignment = (
            coordinator_reduction_ms_per_assignment
        )
        self.selection: dict[_AssignmentUnit, _Candidate] = {}
        self.dynamic_vram_bytes = {node_id: 0 for node_id in nodes}
        self.route_counts: dict[tuple[ExpertKey, str, str], int] = {}
        self.remote_assignment_counts: dict[str, int] = {}
        self.remote_position_refcounts: dict[str, dict[int, int]] = {}
        self.remote_owner_metrics: dict[
            str,
            tuple[float, int, int, int],
        ] = {}
        self.total_request_bytes = 0
        self.total_response_bytes = 0
        self.total_metadata_bytes = 0
        self.local_owner_time_ms = 0.0
        self.weight_loaded_bytes = 0

    def _refresh_remote_owner_metrics(self, owner_id: str) -> None:
        previous = self.remote_owner_metrics.pop(owner_id, None)
        if previous is not None:
            _duration, request, response, metadata = previous
            self.total_request_bytes -= request
            self.total_response_bytes -= response
            self.total_metadata_bytes -= metadata

        assignment_total = self.remote_assignment_counts.get(owner_id, 0)
        if not assignment_total:
            return
        unique_position_total = len(self.remote_position_refcounts[owner_id])
        activation = self.activation_bytes_per_token
        v1_request_bytes = activation * assignment_total
        response_bytes = activation * assignment_total
        metadata_per_assignment = self.row_index_bytes_by_node.get(owner_id, 4)
        coalesced_request_bytes = activation * unique_position_total
        coalesced_metadata_bytes = metadata_per_assignment * assignment_total
        use_coalesced = (
            owner_id not in self.v1_only_nodes
            and coalesced_request_bytes + coalesced_metadata_bytes
            < v1_request_bytes
        )
        request_bytes = (
            coalesced_request_bytes if use_coalesced else v1_request_bytes
        )
        metadata_bytes = coalesced_metadata_bytes if use_coalesced else 0
        staging_bytes = (
            2 * activation * (unique_position_total + assignment_total)
            if use_coalesced
            else 4 * activation * assignment_total
        )
        link = self.links[(self.coordinator_id, owner_id)]
        duration_ms = (
            link.round_trip_ms
            + link.rpc_setup_ms
            + (request_bytes + metadata_bytes) / link.egress_bytes_per_ms
            + response_bytes / link.ingress_bytes_per_ms
            + (
                staging_bytes / link.host_device_staging_bytes_per_ms
                if link.host_device_staging_bytes_per_ms > 0
                else 0.0
            )
            + assignment_total
            * self.nodes[owner_id].expert_compute_ms_per_token
        )
        self.remote_owner_metrics[owner_id] = (
            duration_ms,
            request_bytes,
            response_bytes,
            metadata_bytes,
        )
        self.total_request_bytes += request_bytes
        self.total_response_bytes += response_bytes
        self.total_metadata_bytes += metadata_bytes

    def can_add(self, unit: _AssignmentUnit, candidate: _Candidate) -> bool:
        if unit in self.selection or candidate.key != unit.key:
            return False
        coordinator_added = candidate.coordinator_dynamic_vram_bytes
        if (
            self.static_vram_bytes[self.coordinator_id]
            + self.dynamic_vram_bytes[self.coordinator_id]
            + coordinator_added
            > self.nodes[self.coordinator_id].resident_vram_budget_bytes
        ):
            return False
        if candidate.owner_id == self.coordinator_id:
            return True
        return (
            self.static_vram_bytes[candidate.owner_id]
            + self.dynamic_vram_bytes[candidate.owner_id]
            + candidate.owner_dynamic_vram_bytes
            <= self.nodes[candidate.owner_id].resident_vram_budget_bytes
        )

    def add(self, unit: _AssignmentUnit, candidate: _Candidate) -> None:
        if not self.can_add(unit, candidate):
            raise ValueError("cannot add an infeasible incremental assignment")
        route_key = (unit.key, candidate.owner_id, candidate.path)
        previous_route_count = self.route_counts.get(route_key, 0)
        self.selection[unit] = candidate
        self.route_counts[route_key] = previous_route_count + 1
        self.dynamic_vram_bytes[self.coordinator_id] += (
            candidate.coordinator_dynamic_vram_bytes
        )
        if candidate.owner_id == self.coordinator_id:
            if previous_route_count == 0:
                self.local_owner_time_ms += candidate.duration_ms
                self.weight_loaded_bytes += candidate.weight_loaded_bytes
            else:
                self.local_owner_time_ms += self.nodes[
                    self.coordinator_id
                ].expert_compute_ms_per_token
            return

        self.dynamic_vram_bytes[candidate.owner_id] += (
            candidate.owner_dynamic_vram_bytes
        )
        self.remote_assignment_counts[candidate.owner_id] = (
            self.remote_assignment_counts.get(candidate.owner_id, 0) + 1
        )
        position_counts = self.remote_position_refcounts.setdefault(
            candidate.owner_id,
            {},
        )
        position_counts[unit.assignment.token_index] = (
            position_counts.get(unit.assignment.token_index, 0) + 1
        )
        self._refresh_remote_owner_metrics(candidate.owner_id)
        if previous_route_count == 0:
            self.weight_loaded_bytes += candidate.weight_loaded_bytes

    def remove(self, unit: _AssignmentUnit) -> _Candidate:
        try:
            candidate = self.selection.pop(unit)
        except KeyError as error:
            raise ValueError("incremental assignment is absent") from error
        route_key = (unit.key, candidate.owner_id, candidate.path)
        previous_route_count = self.route_counts[route_key]
        if previous_route_count == 1:
            del self.route_counts[route_key]
            self.weight_loaded_bytes -= candidate.weight_loaded_bytes
        else:
            self.route_counts[route_key] = previous_route_count - 1

        self.dynamic_vram_bytes[self.coordinator_id] -= (
            candidate.coordinator_dynamic_vram_bytes
        )
        if candidate.owner_id == self.coordinator_id:
            if previous_route_count == 1:
                self.local_owner_time_ms -= candidate.duration_ms
            else:
                self.local_owner_time_ms -= self.nodes[
                    self.coordinator_id
                ].expert_compute_ms_per_token
            if abs(self.local_owner_time_ms) <= 1e-12:
                self.local_owner_time_ms = 0.0
            return candidate

        self.dynamic_vram_bytes[candidate.owner_id] -= (
            candidate.owner_dynamic_vram_bytes
        )
        remaining_assignments = self.remote_assignment_counts[candidate.owner_id] - 1
        if remaining_assignments:
            self.remote_assignment_counts[candidate.owner_id] = remaining_assignments
        else:
            del self.remote_assignment_counts[candidate.owner_id]
        position_counts = self.remote_position_refcounts[candidate.owner_id]
        token_index = unit.assignment.token_index
        remaining_references = position_counts[token_index] - 1
        if remaining_references:
            position_counts[token_index] = remaining_references
        else:
            del position_counts[token_index]
        if not position_counts:
            del self.remote_position_refcounts[candidate.owner_id]
        self._refresh_remote_owner_metrics(candidate.owner_id)
        return candidate

    def objective(self) -> tuple[float, int, int]:
        """Return the heuristic objective matching the full cost model."""

        owner_times = {
            owner_id: metrics[0]
            for owner_id, metrics in self.remote_owner_metrics.items()
        }
        if self.local_owner_time_ms > 0:
            owner_times[self.coordinator_id] = self.local_owner_time_ms

        coordinator = self.nodes[self.coordinator_id]
        nic_bounds = []
        if self.total_request_bytes or self.total_metadata_bytes:
            if coordinator.aggregate_egress_bytes_per_ms > 0:
                nic_bounds.append(
                    (self.total_request_bytes + self.total_metadata_bytes)
                    / coordinator.aggregate_egress_bytes_per_ms
                )
        if (
            self.total_response_bytes
            and coordinator.aggregate_ingress_bytes_per_ms > 0
        ):
            nic_bounds.append(
                self.total_response_bytes
                / coordinator.aggregate_ingress_bytes_per_ms
            )
        exposed_ms = max(
            _list_scheduled_makespan_ms(
                owner_times,
                self.max_inflight_owner_rpcs,
            ),
            max(nic_bounds, default=0.0),
        ) + _modeled_coordinator_overhead_ms(
            remote_owner_count=len(self.remote_assignment_counts),
            transport_payload_bytes=(
                self.total_request_bytes
                + self.total_response_bytes
                + self.total_metadata_bytes
            ),
            assignment_count=len(self.selection),
            rpc_dispatch_ms_per_owner=(
                self.coordinator_rpc_dispatch_ms_per_owner
            ),
            serialization_bytes_per_ms=(
                self.coordinator_serialization_bytes_per_ms
            ),
            reduction_ms_per_assignment=(
                self.coordinator_reduction_ms_per_assignment
            ),
        )
        return (
            exposed_ms,
            self.total_request_bytes
            + self.total_response_bytes
            + self.total_metadata_bytes,
            self.weight_loaded_bytes,
        )


class ResidentExpertMesh:
    """Deterministic planner/executor for an exact resident-expert mesh."""

    def __init__(
        self,
        *,
        coordinator_id: str,
        experts: Sequence[ExpertRecord],
        nodes: Sequence[MeshNodeProfile],
        links: Sequence[MeshLinkProfile],
        local_ram_keys: Iterable[ExpertKey],
        local_gpu_keys: Iterable[ExpertKey],
        replicas: Sequence[ResidentExpertReplica],
        local_weight_buffer_bytes: int,
        activation_bytes_per_token: int,
        require_local_ram_fallback: bool = True,
        max_inflight_owner_rpcs: int = 32,
        coordinator_rpc_dispatch_ms_per_owner: float | None = None,
        coordinator_serialization_gbytes_per_second: float | None = None,
        coordinator_reduction_ms_per_assignment: float | None = None,
    ) -> None:
        self.coordinator_id = _name("coordinator_id", coordinator_id)
        self.local_weight_buffer_bytes = _integer(
            "local_weight_buffer_bytes",
            local_weight_buffer_bytes,
        )
        self.activation_bytes_per_token = _integer(
            "activation_bytes_per_token",
            activation_bytes_per_token,
            minimum=1,
        )
        if not isinstance(require_local_ram_fallback, bool):
            raise TypeError("require_local_ram_fallback must be boolean")
        self.require_local_ram_fallback = require_local_ram_fallback
        self.max_inflight_owner_rpcs = _integer(
            "max_inflight_owner_rpcs",
            max_inflight_owner_rpcs,
            minimum=1,
        )
        self.coordinator_rpc_dispatch_ms_per_owner = (
            None
            if coordinator_rpc_dispatch_ms_per_owner is None
            else _finite(
                "coordinator_rpc_dispatch_ms_per_owner",
                coordinator_rpc_dispatch_ms_per_owner,
            )
        )
        if coordinator_serialization_gbytes_per_second is None:
            self.coordinator_serialization_bytes_per_ms = None
        else:
            serialization_gbytes_per_second = _finite(
                "coordinator_serialization_gbytes_per_second",
                coordinator_serialization_gbytes_per_second,
            )
            if serialization_gbytes_per_second <= 0:
                raise ValueError(
                    "coordinator_serialization_gbytes_per_second must be positive"
                )
            self.coordinator_serialization_bytes_per_ms = (
                serialization_gbytes_per_second * 1_000_000.0
            )
        self.coordinator_reduction_ms_per_assignment = (
            None
            if coordinator_reduction_ms_per_assignment is None
            else _finite(
                "coordinator_reduction_ms_per_assignment",
                coordinator_reduction_ms_per_assignment,
            )
        )

        self._experts = self._index_experts(experts)
        self._nodes = self._index_nodes(nodes)
        if self.coordinator_id not in self._nodes:
            raise ValueError("coordinator_id is absent from nodes")
        self._links = self._index_links(links)
        self._local_ram_keys = frozenset(local_ram_keys)
        self._local_gpu_keys = frozenset(local_gpu_keys)
        self._validate_keys(self._local_ram_keys, "local_ram_keys")
        self._validate_keys(self._local_gpu_keys, "local_gpu_keys")

        replica_map: dict[ExpertKey, list[ResidentExpertReplica]] = {}
        seen_replica_slots: set[tuple[ExpertKey, str]] = set()
        for replica in replicas:
            if not isinstance(replica, ResidentExpertReplica):
                raise TypeError("replicas must contain ResidentExpertReplica values")
            slot = (replica.key, replica.node_id)
            if slot in seen_replica_slots:
                raise ValueError(f"duplicate resident replica {slot}")
            seen_replica_slots.add(slot)
            record = self._experts.get(replica.key)
            if record is None:
                raise ValueError(f"replica references unknown expert {replica.key}")
            if replica.node_id == self.coordinator_id:
                raise ValueError("coordinator residency belongs in local_gpu_keys")
            if replica.node_id not in self._nodes:
                raise ValueError(f"replica references unknown node {replica.node_id}")
            if replica.content_id != record.content_id:
                raise ValueError(f"replica content identity mismatch for {replica.key}")
            if (self.coordinator_id, replica.node_id) not in self._links:
                raise ValueError(f"replica {slot} has no coordinator link")
            replica_map.setdefault(replica.key, []).append(replica)
        self._replicas = {
            key: tuple(sorted(values, key=lambda value: value.node_id))
            for key, values in replica_map.items()
        }

        if self.require_local_ram_fallback and self._local_ram_keys != frozenset(
            self._experts
        ):
            raise ValueError(
                "fail-closed local RAM fallback requires every expert artifact"
            )
        if self._local_ram_keys:
            coordinator = self._nodes[self.coordinator_id]
            if coordinator.ram_to_device_bytes_per_ms <= 0:
                raise ValueError("local RAM experts require positive RAM-to-device bandwidth")
            largest = max(self._experts[key].byte_size for key in self._local_ram_keys)
            if self.local_weight_buffer_bytes < largest:
                raise ValueError("local weight buffer cannot hold the largest RAM expert")
        self._validate_resident_vram_budgets()
        # Reused across layers/waves. Creating and tearing down a thread pool
        # on every transformer layer is measurable overhead on the critical
        # path, especially on LAN links with sub-millisecond RTT.
        self._executor_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_inflight_owner_rpcs,
            thread_name_prefix=f"resident-mesh-{self.coordinator_id}",
        )
        self._closed = False

    @property
    def experts(self) -> tuple[ExpertRecord, ...]:
        return tuple(self._experts[key] for key in sorted(self._experts))

    @property
    def nodes(self) -> tuple[MeshNodeProfile, ...]:
        return tuple(self._nodes[node_id] for node_id in sorted(self._nodes))

    def node_profile(self, node_id: str) -> MeshNodeProfile:
        try:
            return self._nodes[node_id]
        except KeyError as error:
            raise ValueError(f"unknown mesh node {node_id!r}") from error

    @property
    def closed(self) -> bool:
        with self._executor_lock:
            return self._closed

    def close(self) -> None:
        """Close only the owned worker pool; ExpertOwner lifecycles stay external."""

        with self._executor_lock:
            if self._closed:
                return
            self._closed = True
            executor = self._executor
        executor.shutdown(wait=True, cancel_futures=False)

    def __enter__(self) -> "ResidentExpertMesh":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def expert_records_for_layer(self, layer: int) -> tuple[ExpertRecord, ...]:
        """Return the immutable identities a stage must match before attach."""

        _integer("layer", layer)
        return tuple(
            self._experts[key]
            for key in sorted(self._experts)
            if key.layer == layer
        )

    def replicas_for_layer(self, layer: int) -> tuple[ResidentExpertReplica, ...]:
        """Return canonical remote placements for owner preflight."""

        _integer("layer", layer)
        return tuple(
            replica
            for key in sorted(self._replicas)
            if key.layer == layer
            for replica in self._replicas[key]
        )

    @staticmethod
    def _owner_has_exact_content(
        owner: ExpertOwner | None,
        key: ExpertKey,
        content_id: str,
    ) -> bool:
        if owner is None:
            return False
        probe = getattr(owner, "has_expert", None)
        if not callable(probe):
            return False
        try:
            return probe(key, content_id) is True
        except Exception:
            return False

    @staticmethod
    def _owner_reports_resident(
        owner: ExpertOwner | None,
        key: ExpertKey,
        content_id: str,
    ) -> bool:
        """Require explicit physical residency; identity inventory is insufficient."""

        if owner is None:
            return False
        probe = getattr(owner, "is_expert_resident", None)
        if not callable(probe):
            return False
        try:
            return probe(key, content_id) is True
        except Exception:
            return False

    @staticmethod
    def _owner_supports_exact_input_coalescing(owner: object) -> bool:
        """Resolve an optional lazy transport capability before planning.

        RPC clients deliberately avoid an extra RTT during construction.  A
        mesh execution probes once, before sending any activations, so an old
        peer can be planned and measured as v1 without a mid-operation switch.
        """

        execute = getattr(owner, "execute_coalesced_batch", None)
        if not callable(execute):
            return False
        advertised = getattr(owner, "supports_exact_input_coalescing", None)
        if advertised is True:
            return True
        if advertised is False:
            return False
        probe = getattr(owner, "probe_exact_input_coalescing", None)
        if callable(probe):
            return probe() is True
        return False

    def _candidate_for_assignment_count(
        self,
        candidate: _Candidate,
        assignment_count: int,
    ) -> _Candidate:
        """Resize a one-row route while keeping fixed path costs single-shot."""

        _integer("assignment_count", assignment_count, minimum=1)
        record = self._experts[candidate.key]
        owner = self._nodes[candidate.owner_id]
        if candidate.path == "local-gpu":
            duration_ms = assignment_count * owner.expert_compute_ms_per_token
        elif candidate.path == "local-ram":
            duration_ms = (
                record.byte_size / owner.ram_to_device_bytes_per_ms
                + assignment_count * owner.expert_compute_ms_per_token
            )
        else:
            link = self._links[(self.coordinator_id, candidate.owner_id)]
            staging_bytes = (
                candidate.host_device_staging_bytes * assignment_count
            )
            duration_ms = (
                link.round_trip_ms
                + link.rpc_setup_ms
                + (
                    self.activation_bytes_per_token
                    * assignment_count
                    / link.egress_bytes_per_ms
                )
                + (
                    self.activation_bytes_per_token
                    * assignment_count
                    / link.ingress_bytes_per_ms
                )
                + (
                    staging_bytes / link.host_device_staging_bytes_per_ms
                    if link.host_device_staging_bytes_per_ms > 0
                    else 0.0
                )
                + assignment_count * owner.expert_compute_ms_per_token
            )
        return _Candidate(
            key=candidate.key,
            owner_id=candidate.owner_id,
            path=candidate.path,
            duration_ms=duration_ms,
            activation_bytes=candidate.activation_bytes * assignment_count,
            host_device_staging_bytes=(
                candidate.host_device_staging_bytes * assignment_count
            ),
            coordinator_dynamic_vram_bytes=(
                candidate.coordinator_dynamic_vram_bytes * assignment_count
            ),
            owner_dynamic_vram_bytes=(
                candidate.owner_dynamic_vram_bytes * assignment_count
            ),
            # A grouped owner/key/path executes one physical expert copy, no
            # matter how many routed rows are sent to it.
            weight_loaded_bytes=candidate.weight_loaded_bytes,
            weight_avoided_bytes=candidate.weight_avoided_bytes,
            transport_calibration_required=(
                candidate.transport_calibration_required
            ),
            workspace_calibration_required=(
                candidate.workspace_calibration_required
            ),
        )

    def _evaluate_assignment_selection(
        self,
        *,
        selection: Mapping[_AssignmentUnit, _Candidate],
        static_vram_bytes: Mapping[str, int],
        v1_only_nodes: frozenset[str],
        row_index_bytes_by_node: Mapping[str, int],
    ) -> _AssignmentEvaluation | None:
        """Evaluate one whole-layer assignment from grouped physical costs."""

        selected_items = tuple(sorted(selection.items()))
        route_units: dict[
            tuple[ExpertKey, str, str],
            list[_AssignmentUnit],
        ] = {}
        path_by_key_owner: dict[tuple[ExpertKey, str], str] = {}
        route_candidates: dict[tuple[ExpertKey, str, str], _Candidate] = {}
        for unit, candidate in selected_items:
            key_owner = (unit.key, candidate.owner_id)
            previous_path = path_by_key_owner.setdefault(
                key_owner,
                candidate.path,
            )
            # One owner cannot execute the same expert twice in one batch.
            # Local GPU always dominates local RAM when both are available,
            # so rejecting a mixed local path cannot remove an optimum.
            if previous_path != candidate.path:
                return None
            route_key = (unit.key, candidate.owner_id, candidate.path)
            route_units.setdefault(route_key, []).append(unit)
            route_candidates.setdefault(route_key, candidate)

        grouped_items: tuple[
            tuple[ExpertKey, _Candidate, tuple[ExpertAssignment, ...]],
            ...,
        ] = tuple(
            (
                route_key[0],
                self._candidate_for_assignment_count(
                    route_candidates[route_key],
                    len(units),
                ),
                tuple(unit.assignment for unit in sorted(units)),
            )
            for route_key, units in sorted(route_units.items())
        )

        dynamic_vram_bytes: dict[str, int] = {}
        for _key, candidate, _route_assignments in grouped_items:
            dynamic_vram_bytes[self.coordinator_id] = (
                dynamic_vram_bytes.get(self.coordinator_id, 0)
                +
                candidate.coordinator_dynamic_vram_bytes
            )
            if candidate.owner_id != self.coordinator_id:
                dynamic_vram_bytes[candidate.owner_id] = (
                    dynamic_vram_bytes.get(candidate.owner_id, 0)
                    +
                    candidate.owner_dynamic_vram_bytes
                )
        if any(
            static_vram_bytes[node_id] + dynamic_vram_bytes[node_id]
            > self._nodes[node_id].resident_vram_budget_bytes
            for node_id in dynamic_vram_bytes
        ):
            return None

        by_owner: dict[
            str,
            list[tuple[ExpertKey, _Candidate, tuple[ExpertAssignment, ...]]],
        ] = {}
        for key, candidate, route_assignments in grouped_items:
            by_owner.setdefault(candidate.owner_id, []).append(
                (key, candidate, route_assignments)
            )

        owner_times: dict[str, float] = {}
        grouped_transport: dict[str, tuple[int, int, int, int, bool]] = {}
        total_request_bytes = 0
        total_response_bytes = 0
        total_metadata_bytes = 0
        total_staging_bytes = 0
        coalesced_owner_ids: list[str] = []
        for owner_id, owner_items in sorted(by_owner.items()):
            if owner_id == self.coordinator_id:
                owner_times[owner_id] = math.fsum(
                    candidate.duration_ms
                    for _key, candidate, _assignments in owner_items
                )
                continue
            assignment_total = sum(
                len(route_assignments)
                for _key, _candidate, route_assignments in owner_items
            )
            unique_position_total = len(
                {
                    assignment.token_index
                    for _key, _candidate, route_assignments in owner_items
                    for assignment in route_assignments
                }
            )
            activation = self.activation_bytes_per_token
            v1_request_bytes = activation * assignment_total
            response_bytes = activation * assignment_total
            v1_staging_bytes = 4 * activation * assignment_total
            metadata_per_assignment = row_index_bytes_by_node.get(owner_id, 4)
            coalesced_request_bytes = activation * unique_position_total
            coalesced_metadata_bytes = metadata_per_assignment * assignment_total
            coalesced_staging_bytes = (
                2 * activation * (unique_position_total + assignment_total)
            )
            use_coalesced = (
                owner_id not in v1_only_nodes
                and coalesced_request_bytes + coalesced_metadata_bytes
                < v1_request_bytes
            )
            request_bytes = (
                coalesced_request_bytes if use_coalesced else v1_request_bytes
            )
            metadata_bytes = coalesced_metadata_bytes if use_coalesced else 0
            staging_bytes = (
                coalesced_staging_bytes if use_coalesced else v1_staging_bytes
            )
            link = self._links[(self.coordinator_id, owner_id)]
            owner_times[owner_id] = (
                link.round_trip_ms
                + link.rpc_setup_ms
                + (request_bytes + metadata_bytes) / link.egress_bytes_per_ms
                + response_bytes / link.ingress_bytes_per_ms
                + (
                    staging_bytes / link.host_device_staging_bytes_per_ms
                    if link.host_device_staging_bytes_per_ms > 0
                    else 0.0
                )
                + assignment_total
                * self._nodes[owner_id].expert_compute_ms_per_token
            )
            grouped_transport[owner_id] = (
                request_bytes,
                response_bytes,
                metadata_bytes,
                staging_bytes,
                use_coalesced,
            )
            total_request_bytes += request_bytes
            total_response_bytes += response_bytes
            total_metadata_bytes += metadata_bytes
            total_staging_bytes += staging_bytes
            if use_coalesced:
                coalesced_owner_ids.append(owner_id)

        active_owner_times = tuple(
            (node_id, duration)
            for node_id, duration in sorted(owner_times.items())
            if duration > 0
        )
        per_owner_lower_bound_ms = max(owner_times.values(), default=0.0)
        owner_scheduled_makespan_ms = _list_scheduled_makespan_ms(
            owner_times,
            self.max_inflight_owner_rpcs,
        )
        coordinator = self._nodes[self.coordinator_id]
        has_remote = bool(grouped_transport)
        nic_is_calibrated = (
            coordinator.aggregate_egress_bytes_per_ms > 0
            and coordinator.aggregate_ingress_bytes_per_ms > 0
        )
        known_nic_bounds = []
        if has_remote and coordinator.aggregate_egress_bytes_per_ms > 0:
            known_nic_bounds.append(
                (total_request_bytes + total_metadata_bytes)
                / coordinator.aggregate_egress_bytes_per_ms
            )
        if has_remote and coordinator.aggregate_ingress_bytes_per_ms > 0:
            known_nic_bounds.append(
                total_response_bytes
                / coordinator.aggregate_ingress_bytes_per_ms
            )
        coordinator_nic_lower_bound_ms = max(known_nic_bounds, default=0.0)
        coordination_calibration_required = (
            (
                has_remote
                and self.coordinator_rpc_dispatch_ms_per_owner is None
            )
            or (
                has_remote
                and self.coordinator_serialization_bytes_per_ms is None
            )
            or (
                bool(selected_items)
                and self.coordinator_reduction_ms_per_assignment is None
            )
        )
        coordinator_overhead_ms = _modeled_coordinator_overhead_ms(
            remote_owner_count=len(grouped_transport),
            transport_payload_bytes=(
                total_request_bytes
                + total_response_bytes
                + total_metadata_bytes
            ),
            assignment_count=len(selected_items),
            rpc_dispatch_ms_per_owner=(
                self.coordinator_rpc_dispatch_ms_per_owner
            ),
            serialization_bytes_per_ms=(
                self.coordinator_serialization_bytes_per_ms
            ),
            reduction_ms_per_assignment=(
                self.coordinator_reduction_ms_per_assignment
            ),
        )

        dispatches: list[ExpertDispatch] = []
        first_remote_key = {
            owner_id: min(
                key for key, _candidate, _assignments in owner_items
            )
            for owner_id, owner_items in by_owner.items()
            if owner_id != self.coordinator_id
        }
        for key, candidate, route_assignments in grouped_items:
            remote_group_head = (
                candidate.owner_id != self.coordinator_id
                and first_remote_key[candidate.owner_id] == key
            )
            if candidate.owner_id == self.coordinator_id:
                exposed_ms = candidate.duration_ms
                activation_bytes = 0
                staging_bytes = 0
            elif remote_group_head:
                request, response, _metadata, staging_bytes, _coalesced = (
                    grouped_transport[candidate.owner_id]
                )
                exposed_ms = owner_times[candidate.owner_id]
                activation_bytes = request + response
            else:
                exposed_ms = 0.0
                activation_bytes = 0
                staging_bytes = 0
            dispatches.append(
                ExpertDispatch(
                    key=key,
                    owner_id=candidate.owner_id,
                    path=candidate.path,
                    assignments=route_assignments,
                    exposed_ms=exposed_ms,
                    activation_round_trip_bytes=activation_bytes,
                    host_device_staging_bytes=staging_bytes,
                    coordinator_dynamic_vram_bytes=(
                        candidate.coordinator_dynamic_vram_bytes
                    ),
                    owner_dynamic_vram_bytes=candidate.owner_dynamic_vram_bytes,
                    weight_loaded_bytes=candidate.weight_loaded_bytes,
                    weight_avoided_bytes=candidate.weight_avoided_bytes,
                    transport_calibration_required=(
                        candidate.transport_calibration_required
                    ),
                    workspace_calibration_required=(
                        candidate.workspace_calibration_required
                    ),
                )
            )

        owner_peak_vram_bytes = tuple(
            (
                node_id,
                static_vram_bytes[node_id] + dynamic_vram_bytes[node_id],
            )
            for node_id in sorted(dynamic_vram_bytes)
            if dynamic_vram_bytes[node_id] > 0
        )
        path_rank = {"local-gpu": 0, "remote-resident": 1, "local-ram": 2}
        signature = tuple(
            (
                unit.key.layer,
                unit.key.expert,
                unit.assignment.token_index,
                unit.assignment.slot_index,
                path_rank[candidate.path],
                candidate.owner_id,
            )
            for unit, candidate in selected_items
        )
        activation_round_trip_bytes = total_request_bytes + total_response_bytes
        return _AssignmentEvaluation(
            selected=selected_items,
            dispatches=tuple(dispatches),
            owner_exposed_ms=active_owner_times,
            owner_peak_vram_bytes=owner_peak_vram_bytes,
            per_owner_lower_bound_ms=per_owner_lower_bound_ms,
            owner_scheduled_makespan_ms=owner_scheduled_makespan_ms,
            coordinator_nic_lower_bound_ms=coordinator_nic_lower_bound_ms,
            coordinator_overhead_ms=coordinator_overhead_ms,
            exposed_ms=max(
                owner_scheduled_makespan_ms,
                coordinator_nic_lower_bound_ms,
            ) + coordinator_overhead_ms,
            activation_round_trip_bytes=activation_round_trip_bytes,
            activation_request_bytes=total_request_bytes,
            activation_response_bytes=total_response_bytes,
            route_metadata_bytes=total_metadata_bytes,
            transport_payload_bytes=(
                activation_round_trip_bytes + total_metadata_bytes
            ),
            coalesced_owner_ids=tuple(coalesced_owner_ids),
            host_device_staging_bytes=total_staging_bytes,
            weight_loaded_bytes=sum(
                candidate.weight_loaded_bytes
                for _key, candidate, _assignments in grouped_items
            ),
            weight_avoided_bytes=sum(
                candidate.weight_avoided_bytes
                for _key, candidate, _assignments in grouped_items
            ),
            transport_calibration_required=any(
                candidate.transport_calibration_required
                for _key, candidate, _assignments in grouped_items
            ) or (has_remote and not nic_is_calibrated) or (
                has_remote and coordination_calibration_required
            ),
            coordination_calibration_required=(
                coordination_calibration_required
            ),
            workspace_calibration_required=any(
                candidate.workspace_calibration_required
                for _key, candidate, _assignments in grouped_items
            ),
            signature=signature,
        )

    @staticmethod
    def _assignment_is_better(
        candidate: _AssignmentEvaluation,
        current: _AssignmentEvaluation | None,
    ) -> bool:
        if current is None:
            return True
        if candidate.exposed_ms < current.exposed_ms - 1e-12:
            return True
        if not math.isclose(
            candidate.exposed_ms,
            current.exposed_ms,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            return False
        return (
            candidate.transport_payload_bytes,
            candidate.weight_loaded_bytes,
            candidate.signature,
        ) < (
            current.transport_payload_bytes,
            current.weight_loaded_bytes,
            current.signature,
        )

    @staticmethod
    def _heuristic_objective_is_better(
        candidate: tuple[float, int, int],
        current: tuple[float, int, int] | None,
    ) -> bool:
        if current is None:
            return True
        if candidate[0] < current[0] - 1e-12:
            return True
        if not math.isclose(
            candidate[0],
            current[0],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            return False
        return candidate[1:] < current[1:]

    def _incremental_state(
        self,
        *,
        static_vram_bytes: Mapping[str, int],
        v1_only_nodes: frozenset[str],
        row_index_bytes_by_node: Mapping[str, int],
    ) -> _IncrementalAssignmentState:
        return _IncrementalAssignmentState(
            coordinator_id=self.coordinator_id,
            nodes=self._nodes,
            links=self._links,
            activation_bytes_per_token=self.activation_bytes_per_token,
            static_vram_bytes=static_vram_bytes,
            v1_only_nodes=v1_only_nodes,
            row_index_bytes_by_node=row_index_bytes_by_node,
            max_inflight_owner_rpcs=self.max_inflight_owner_rpcs,
            coordinator_rpc_dispatch_ms_per_owner=(
                self.coordinator_rpc_dispatch_ms_per_owner
            ),
            coordinator_serialization_bytes_per_ms=(
                self.coordinator_serialization_bytes_per_ms
            ),
            coordinator_reduction_ms_per_assignment=(
                self.coordinator_reduction_ms_per_assignment
            ),
        )

    def _bounded_candidate_frontier(
        self,
        *,
        candidates: Sequence[_Candidate],
        assignment_count: int,
        static_vram_bytes: Mapping[str, int],
    ) -> tuple[tuple[_Candidate, ...], bool]:
        """Bound a huge replica set without making an exactness claim.

        Coordinator-local routes are always retained.  Remote routes are
        ranked by their sealed one-row cost and identity; after the normal
        frontier limit, more are retained until their individual dynamic-VRAM
        capacities can cover this expert's routed rows.  Cross-expert capacity
        interactions can still require a discarded route, which is why the
        caller labels every genuinely pruned plan heuristic and never turns a
        pruned exhaustive failure into an infeasibility proof.
        """

        local = tuple(
            candidate
            for candidate in candidates
            if candidate.owner_id == self.coordinator_id
        )
        remote = tuple(
            sorted(
                (
                    candidate
                    for candidate in candidates
                    if candidate.owner_id != self.coordinator_id
                ),
                key=lambda candidate: (
                    candidate.duration_ms,
                    candidate.transport_calibration_required,
                    candidate.workspace_calibration_required,
                    candidate.activation_bytes,
                    candidate.owner_dynamic_vram_bytes,
                    candidate.owner_id,
                ),
            )
        )
        frontier_limit = min(
            _HEURISTIC_REMOTE_CANDIDATE_LIMIT_PER_KEY,
            self.max_inflight_owner_rpcs,
        )
        if len(remote) <= frontier_limit:
            return tuple(local + remote), False

        retained_remote = list(remote[:frontier_limit])

        def row_capacity(candidate: _Candidate) -> int:
            coordinator_headroom = (
                self._nodes[self.coordinator_id].resident_vram_budget_bytes
                - static_vram_bytes[self.coordinator_id]
            )
            if candidate.coordinator_dynamic_vram_bytes:
                capacity = (
                    coordinator_headroom
                    // candidate.coordinator_dynamic_vram_bytes
                )
            else:
                capacity = assignment_count
            if candidate.owner_id != self.coordinator_id:
                owner_headroom = (
                    self._nodes[candidate.owner_id].resident_vram_budget_bytes
                    - static_vram_bytes[candidate.owner_id]
                )
                if candidate.owner_dynamic_vram_bytes:
                    capacity = min(
                        capacity,
                        owner_headroom // candidate.owner_dynamic_vram_bytes,
                    )
            return max(0, min(capacity, assignment_count))

        retained_capacity = sum(
            row_capacity(candidate)
            for candidate in (*local, *retained_remote)
        )
        next_index = len(retained_remote)
        while (
            retained_capacity < assignment_count
            and next_index < len(remote)
        ):
            candidate = remote[next_index]
            retained_remote.append(candidate)
            retained_capacity += row_capacity(candidate)
            next_index += 1

        retained = tuple(local + tuple(retained_remote))
        return retained, len(retained) < len(candidates)

    def _incremental_greedy_selection(
        self,
        *,
        ordered_units: Sequence[_AssignmentUnit],
        candidates_by_key: Mapping[ExpertKey, Sequence[_Candidate]],
        static_vram_bytes: Mapping[str, int],
        v1_only_nodes: frozenset[str],
        row_index_bytes_by_node: Mapping[str, int],
    ) -> tuple[_IncrementalAssignmentState | None, int]:
        state = self._incremental_state(
            static_vram_bytes=static_vram_bytes,
            v1_only_nodes=v1_only_nodes,
            row_index_bytes_by_node=row_index_bytes_by_node,
        )
        states_evaluated = 0
        for unit in ordered_units:
            best_candidate: _Candidate | None = None
            best_objective: tuple[float, int, int] | None = None
            for candidate in candidates_by_key[unit.key]:
                states_evaluated += 1
                if not state.can_add(unit, candidate):
                    continue
                state.add(unit, candidate)
                objective = state.objective()
                state.remove(unit)
                if self._heuristic_objective_is_better(
                    objective,
                    best_objective,
                ):
                    best_candidate = candidate
                    best_objective = objective
            if best_candidate is None:
                return None, states_evaluated
            state.add(unit, best_candidate)
        return state, states_evaluated

    def _bounded_feasible_selection(
        self,
        *,
        ordered_units: Sequence[_AssignmentUnit],
        candidates_by_key: Mapping[ExpertKey, Sequence[_Candidate]],
        static_vram_bytes: Mapping[str, int],
        v1_only_nodes: frozenset[str],
        row_index_bytes_by_node: Mapping[str, int],
    ) -> tuple[_IncrementalAssignmentState | None, int, bool]:
        """Iteratively search for feasibility within a truthful fixed budget.

        The final boolean is true only when every candidate branch was
        exhausted. A budget stop therefore remains an unresolved search limit,
        never a false proof that the sealed mesh is infeasible.
        """

        state = self._incremental_state(
            static_vram_bytes=static_vram_bytes,
            v1_only_nodes=v1_only_nodes,
            row_index_bytes_by_node=row_index_bytes_by_node,
        )
        unit_count = len(ordered_units)
        next_candidate_indexes = [0] * unit_count
        chosen: list[_Candidate | None] = [None] * unit_count
        depth = 0
        states_evaluated = 0
        while depth >= 0:
            if depth == unit_count:
                return state, states_evaluated, False
            unit = ordered_units[depth]
            pool = candidates_by_key[unit.key]
            advanced = False
            while next_candidate_indexes[depth] < len(pool):
                if (
                    states_evaluated
                    >= _HEURISTIC_FEASIBILITY_REPAIR_STATE_LIMIT
                ):
                    return None, states_evaluated, False
                candidate = pool[next_candidate_indexes[depth]]
                next_candidate_indexes[depth] += 1
                states_evaluated += 1
                if not state.can_add(unit, candidate):
                    continue
                state.add(unit, candidate)
                chosen[depth] = candidate
                depth += 1
                if depth < unit_count:
                    next_candidate_indexes[depth] = 0
                    chosen[depth] = None
                advanced = True
                break
            if advanced:
                continue

            next_candidate_indexes[depth] = 0
            if depth == 0:
                return None, states_evaluated, True
            depth -= 1
            previous = chosen[depth]
            if previous is None:
                raise AssertionError("bounded repair lost its parent assignment")
            state.remove(ordered_units[depth])
            chosen[depth] = None
        raise AssertionError("bounded repair escaped its iterative search")

    def _bounded_incremental_local_search(
        self,
        *,
        state: _IncrementalAssignmentState,
        ordered_units: Sequence[_AssignmentUnit],
        candidates_by_key: Mapping[ExpertKey, Sequence[_Candidate]],
        probe_limit: int = _HEURISTIC_LOCAL_MOVE_PROBE_LIMIT,
    ) -> int:
        """Apply at most one deterministic improving move per routed row."""

        probes = 0
        objective = state.objective()
        for unit in ordered_units:
            original = state.selection[unit]
            for candidate in candidates_by_key[unit.key]:
                if candidate == original:
                    continue
                if probes >= probe_limit:
                    return probes
                state.remove(unit)
                probes += 1
                accepted = False
                if state.can_add(unit, candidate):
                    state.add(unit, candidate)
                    candidate_objective = state.objective()
                    if self._heuristic_objective_is_better(
                        candidate_objective,
                        objective,
                    ):
                        objective = candidate_objective
                        accepted = True
                    else:
                        state.remove(unit)
                if not accepted:
                    state.add(unit, original)
                else:
                    break
        return probes

    def plan_layer(
        self,
        layer: int,
        routing: AuthoritativeRouting,
        *,
        unavailable_node_ids: Iterable[str] = (),
        unavailable_links: Iterable[tuple[str, str]] = (),
        unavailable_resident_slots: Iterable[tuple[str, ExpertKey]] = (),
        v1_only_node_ids: Iterable[str] = (),
        coalesced_row_index_bytes_by_node: Mapping[str, int] | None = None,
    ) -> LayerMeshPlan:
        _integer("layer", layer)
        if not isinstance(routing, AuthoritativeRouting):
            raise TypeError("routing must be AuthoritativeRouting")
        unavailable_nodes = frozenset(unavailable_node_ids)
        unavailable_link_set = frozenset(unavailable_links)
        unavailable_resident_set = frozenset(unavailable_resident_slots)
        v1_only_nodes = frozenset(v1_only_node_ids)
        if any(node_id not in self._nodes for node_id in v1_only_nodes):
            raise ValueError("v1_only_node_ids contains an unknown node")
        row_index_bytes_by_node = dict(
            coalesced_row_index_bytes_by_node or {}
        )
        for node_id, byte_size in row_index_bytes_by_node.items():
            if node_id not in self._nodes:
                raise ValueError(
                    "coalesced_row_index_bytes_by_node contains an unknown node"
                )
            _integer(
                "coalesced row-index bytes per assignment",
                byte_size,
            )
        coordinator = self._nodes[self.coordinator_id]
        if not coordinator.available or self.coordinator_id in unavailable_nodes:
            raise ExpertRouteUnavailableError("coordinator is unavailable")

        # ``AuthoritativeRouting`` made one grouped copy of all IDs.  Iterate
        # ordinary Python integers here; calling ``Tensor.item`` for every
        # position x top-k entry would serialize the CUDA stream repeatedly.
        authoritative_ids = tuple(
            tuple(int(expert_id) for expert_id in row)
            for row in routing._expert_ids_cpu.tolist()
        )
        assignments: dict[ExpertKey, list[ExpertAssignment]] = {}
        for flat_index, expert_id in enumerate(
            expert_id
            for row in authoritative_ids
            for expert_id in row
        ):
            token_index, slot_index = divmod(flat_index, routing.top_k)
            key = ExpertKey(layer, expert_id)
            if key not in self._experts:
                raise ExpertRouteUnavailableError(
                    f"target router selected unknown expert {key}"
                )
            assignments.setdefault(key, []).append(
                ExpertAssignment(token_index, slot_index)
            )

        static_vram_bytes = self._resident_vram_usage()
        candidates_by_key: dict[ExpertKey, tuple[_Candidate, ...]] = {}
        candidate_frontier_pruned = False
        path_rank = {"local-gpu": 0, "remote-resident": 1, "local-ram": 2}
        for key in sorted(assignments):
            raw_candidates = self._candidates(
                key,
                1,
                unavailable_nodes,
                unavailable_link_set,
                unavailable_resident_set,
            )
            candidates: list[_Candidate] = []
            for candidate in sorted(
                raw_candidates,
                key=lambda value: (
                    path_rank[value.path],
                    value.owner_id,
                ),
            ):
                if any(
                    existing.owner_id == candidate.owner_id
                    for existing in candidates
                ):
                    # A resident local-GPU path is strictly faster than the
                    # same owner's RAM-load fallback and has the same dynamic
                    # VRAM shape. Keep one executable path per key/owner; RAM
                    # reappears automatically if residency is unavailable.
                    continue
                additions = {
                    self.coordinator_id: candidate.coordinator_dynamic_vram_bytes
                }
                if candidate.owner_id != self.coordinator_id:
                    additions[candidate.owner_id] = (
                        candidate.owner_dynamic_vram_bytes
                    )
                if all(
                    static_vram_bytes[node_id]
                    + added
                    <= self._nodes[node_id].resident_vram_budget_bytes
                    for node_id, added in additions.items()
                ):
                    candidates.append(candidate)
            if not candidates:
                raise ExpertRouteUnavailableError(
                    f"no exact resident or RAM fallback path within sealed "
                    f"VRAM for {key}"
                )
            frontier, pruned = self._bounded_candidate_frontier(
                candidates=candidates,
                assignment_count=len(assignments[key]),
                static_vram_bytes=static_vram_bytes,
            )
            candidates_by_key[key] = frontier
            candidate_frontier_pruned = candidate_frontier_pruned or pruned

        ordered_units = tuple(
            _AssignmentUnit(key, ordinal, assignment)
            for key in sorted(assignments)
            for ordinal, assignment in enumerate(assignments[key])
        )
        combination_count = 1
        for unit in ordered_units:
            combination_count *= len(candidates_by_key[unit.key])
            if combination_count > _EXACT_ASSIGNMENT_COMBINATION_LIMIT:
                break

        best: _AssignmentEvaluation | None = None
        states_evaluated = 0
        if (
            combination_count <= _EXACT_ASSIGNMENT_COMBINATION_LIMIT
            and not candidate_frontier_pruned
        ):
            # ``itertools.product`` keeps exact enumeration iterative.  A
            # layer with thousands of one-choice experts is one state, not a
            # reason to recurse once per expert and overflow Python's stack.
            candidate_pools = tuple(
                candidates_by_key[unit.key] for unit in ordered_units
            )
            for chosen in product(*candidate_pools):
                states_evaluated += 1
                selection = dict(zip(ordered_units, chosen))
                evaluated = self._evaluate_assignment_selection(
                    selection=selection,
                    static_vram_bytes=static_vram_bytes,
                    v1_only_nodes=v1_only_nodes,
                    row_index_bytes_by_node=row_index_bytes_by_node,
                )
                if evaluated is not None and self._assignment_is_better(
                    evaluated,
                    best,
                ):
                    best = evaluated
            strategy = "exact-enumeration"
            optimality_proven = True
            if best is None:
                raise ExpertRouteUnavailableError(
                    "no complete exact assignment fits the sealed dynamic VRAM"
                )
        else:
            constrained_order = tuple(
                sorted(
                    ordered_units,
                    key=lambda unit: (
                        len(candidates_by_key[unit.key]),
                        unit,
                    ),
                )
            )
            state, greedy_states = self._incremental_greedy_selection(
                ordered_units=constrained_order,
                candidates_by_key=candidates_by_key,
                static_vram_bytes=static_vram_bytes,
                v1_only_nodes=v1_only_nodes,
                row_index_bytes_by_node=row_index_bytes_by_node,
            )
            states_evaluated += greedy_states
            repaired = False
            if state is None:
                state, repair_states, repair_exhaustive = (
                    self._bounded_feasible_selection(
                        ordered_units=constrained_order,
                        candidates_by_key=candidates_by_key,
                        static_vram_bytes=static_vram_bytes,
                        v1_only_nodes=v1_only_nodes,
                        row_index_bytes_by_node=row_index_bytes_by_node,
                    )
                )
                states_evaluated += repair_states
                if state is None:
                    if repair_exhaustive:
                        if candidate_frontier_pruned:
                            raise ExpertAssignmentSearchLimitError(
                                "the bounded replica frontier was exhausted; "
                                "full route feasibility remains unresolved"
                            )
                        raise ExpertRouteUnavailableError(
                            "bounded assignment repair exhaustively proved that "
                            "no complete route fits the sealed dynamic VRAM"
                        )
                    raise ExpertAssignmentSearchLimitError(
                        "bounded assignment repair exhausted "
                        f"{_HEURISTIC_FEASIBILITY_REPAIR_STATE_LIMIT} states; "
                        "route feasibility remains unresolved"
                    )
                repaired = True

            states_evaluated += self._bounded_incremental_local_search(
                state=state,
                ordered_units=ordered_units,
                candidates_by_key=candidates_by_key,
                probe_limit=(
                    min(
                        _HEURISTIC_LOCAL_MOVE_PROBE_LIMIT,
                        self.max_inflight_owner_rpcs**2,
                    )
                    if candidate_frontier_pruned
                    else _HEURISTIC_LOCAL_MOVE_PROBE_LIMIT
                ),
            )
            best = self._evaluate_assignment_selection(
                selection=state.selection,
                static_vram_bytes=static_vram_bytes,
                v1_only_nodes=v1_only_nodes,
                row_index_bytes_by_node=row_index_bytes_by_node,
            )
            if best is None:
                raise ResidentExpertMeshError(
                    "incremental assignment state disagrees with full validation"
                )
            if candidate_frontier_pruned:
                strategy = (
                    "heuristic-candidate-pruned-bounded-repair"
                    if repaired
                    else "heuristic-candidate-pruned-local-search"
                )
            else:
                strategy = (
                    "heuristic-bounded-repair"
                    if repaired
                    else "heuristic-incremental-local-search"
                )
            optimality_proven = False

        assert best is not None
        return LayerMeshPlan(
            layer=layer,
            token_count=routing.token_count,
            top_k=routing.top_k,
            authoritative_expert_ids=authoritative_ids,
            dispatches=best.dispatches,
            owner_exposed_ms=best.owner_exposed_ms,
            owner_peak_vram_bytes=best.owner_peak_vram_bytes,
            per_owner_lower_bound_ms=best.per_owner_lower_bound_ms,
            owner_scheduled_makespan_ms=(
                best.owner_scheduled_makespan_ms
            ),
            coordinator_nic_lower_bound_ms=best.coordinator_nic_lower_bound_ms,
            coordinator_overhead_ms=best.coordinator_overhead_ms,
            exposed_ms=best.exposed_ms,
            activation_round_trip_bytes=best.activation_round_trip_bytes,
            activation_request_bytes=best.activation_request_bytes,
            activation_response_bytes=best.activation_response_bytes,
            route_metadata_bytes=best.route_metadata_bytes,
            transport_payload_bytes=best.transport_payload_bytes,
            coalesced_owner_ids=best.coalesced_owner_ids,
            host_device_staging_bytes=best.host_device_staging_bytes,
            weight_loaded_bytes=best.weight_loaded_bytes,
            weight_avoided_bytes=best.weight_avoided_bytes,
            transport_calibration_required=best.transport_calibration_required,
            coordination_calibration_required=(
                best.coordination_calibration_required
            ),
            workspace_calibration_required=best.workspace_calibration_required,
            max_inflight_owner_rpcs=self.max_inflight_owner_rpcs,
            assignment_strategy=strategy,
            assignment_optimality_proven=optimality_proven,
            assignment_states_evaluated=states_evaluated,
        )

    def execute_layer(
        self,
        hidden: torch.Tensor,
        layer: int,
        routing: AuthoritativeRouting,
        owners: Mapping[str, ExpertOwner],
        *,
        unavailable_node_ids: Iterable[str] = (),
        unavailable_links: Iterable[tuple[str, str]] = (),
        unavailable_resident_slots: Iterable[tuple[str, ExpertKey]] = (),
    ) -> tuple[torch.Tensor, LayerMeshPlan]:
        unavailable_nodes = frozenset(unavailable_node_ids)
        unavailable_link_set = frozenset(unavailable_links)
        unavailable_resident = set(unavailable_resident_slots)
        while True:
            try:
                return self._execute_layer_once(
                    hidden,
                    layer,
                    routing,
                    owners,
                    unavailable_node_ids=unavailable_nodes,
                    unavailable_links=unavailable_link_set,
                    unavailable_resident_slots=unavailable_resident,
                )
            except _RetryResidentSlots as retry:
                new_slots = set(retry.slots) - unavailable_resident
                if not new_slots:
                    raise ResidentExpertMeshError(
                        "resident expert retry made no routing progress"
                    ) from retry
                unavailable_resident.update(new_slots)

    @torch.no_grad()
    def _execute_layer_once(
        self,
        hidden: torch.Tensor,
        layer: int,
        routing: AuthoritativeRouting,
        owners: Mapping[str, ExpertOwner],
        *,
        unavailable_node_ids: Iterable[str] = (),
        unavailable_links: Iterable[tuple[str, str]] = (),
        unavailable_resident_slots: Iterable[tuple[str, ExpertKey]] = (),
    ) -> tuple[torch.Tensor, LayerMeshPlan]:
        with self._executor_lock:
            if self._closed:
                raise ResidentExpertMeshError("resident expert mesh is closed")
        if hidden.ndim != 2 or hidden.shape[0] != routing.token_count:
            raise ValueError("hidden must have shape [routing positions, hidden_size]")
        if hidden.numel() < 1:
            raise ValueError("hidden cannot be empty")
        if not torch.is_floating_point(hidden):
            raise TypeError("hidden must use a floating dtype")
        if routing.expert_weights.device != hidden.device:
            raise ValueError("router weights must be on the hidden activation device")
        normalized_unavailable_nodes = frozenset(unavailable_node_ids)
        normalized_unavailable_links = frozenset(unavailable_links)
        actual_activation_bytes = int(hidden.shape[1]) * int(hidden.element_size())
        if actual_activation_bytes != self.activation_bytes_per_token:
            raise ValueError(
                "hidden activation bytes do not match the sealed mesh contract"
            )
        # Runtime execution is stricter than standalone projection. Discover
        # physical residency from each owner using a dedicated ABI; an exact
        # RAM inventory entry is never evidence that weights are in VRAM.
        runtime_unavailable_resident = set(unavailable_resident_slots)
        # Do not pay a capability RTT for every owner up front. Mark obvious
        # v1-only executors now; lazy RPC clients are probed only if the first
        # placement actually finds reusable input rows on that owner.
        runtime_v1_only_nodes = {
            node_id
            for node_id, owner in owners.items()
            if node_id in self._nodes
            and (
                not callable(getattr(owner, "execute_coalesced_batch", None))
                or getattr(owner, "supports_exact_input_coalescing", None)
                is False
            )
        }
        runtime_row_index_bytes_by_node: dict[str, int] = {}
        for node_id, owner in owners.items():
            if node_id not in self._nodes:
                continue
            row_index_bytes = getattr(
                owner,
                "coalesced_row_index_bytes_per_assignment",
                4,
            )
            if (
                not isinstance(row_index_bytes, int)
                or isinstance(row_index_bytes, bool)
                or row_index_bytes < 0
            ):
                raise ResidentExpertMeshError(
                    f"owner {node_id!r} advertises an invalid coalesced metadata cost"
                )
            runtime_row_index_bytes_by_node[node_id] = row_index_bytes
        for expert_id in routing.authoritative_experts:
            key = ExpertKey(layer, expert_id)
            record = self._experts.get(key)
            if record is None:
                continue
            if key in self._local_gpu_keys:
                local_owner = owners.get(self.coordinator_id)
                if not (
                    self._owner_has_exact_content(
                        local_owner,
                        key,
                        record.content_id,
                    )
                    and self._owner_reports_resident(
                        local_owner,
                        key,
                        record.content_id,
                    )
                ):
                    runtime_unavailable_resident.add((self.coordinator_id, key))
            for replica in self._replicas.get(key, ()):
                owner = owners.get(replica.node_id)
                if not (
                    self._owner_has_exact_content(owner, key, record.content_id)
                    and self._owner_reports_resident(
                        owner,
                        key,
                        record.content_id,
                    )
                ):
                    runtime_unavailable_resident.add((replica.node_id, key))

        # Residency may disappear between discovery and planning. Replan away
        # from every lost slot until the selected resident paths all have a
        # fresh positive attestation, or exact fallback becomes impossible.
        while True:
            plan = self.plan_layer(
                layer,
                routing,
                unavailable_node_ids=normalized_unavailable_nodes,
                unavailable_links=normalized_unavailable_links,
                unavailable_resident_slots=runtime_unavailable_resident,
                v1_only_node_ids=runtime_v1_only_nodes,
                coalesced_row_index_bytes_by_node=(
                    runtime_row_index_bytes_by_node
                ),
            )
            unsupported_coalesced = {
                owner_id
                for owner_id in plan.coalesced_owner_ids
                if not self._owner_supports_exact_input_coalescing(
                    owners.get(owner_id)
                )
            }
            if unsupported_coalesced - runtime_v1_only_nodes:
                runtime_v1_only_nodes.update(unsupported_coalesced)
                continue
            lost_residency: set[tuple[str, ExpertKey]] = set()
            for dispatch in plan.dispatches:
                if dispatch.path not in {"local-gpu", "remote-resident"}:
                    continue
                owner = owners.get(dispatch.owner_id)
                record = self._experts[dispatch.key]
                if not (
                    self._owner_has_exact_content(
                        owner,
                        dispatch.key,
                        record.content_id,
                    )
                    and self._owner_reports_resident(
                        owner,
                        dispatch.key,
                        record.content_id,
                    )
                ):
                    lost_residency.add((dispatch.owner_id, dispatch.key))
            if not lost_residency:
                break
            previous_size = len(runtime_unavailable_resident)
            runtime_unavailable_resident.update(lost_residency)
            if len(runtime_unavailable_resident) == previous_size:
                raise ExpertRouteUnavailableError(
                    "resident expert ownership changed during exact replanning"
                )
        if plan.workspace_calibration_required:
            raise ResidentExpertMeshError(
                "sealed mesh execution requires non-zero expert workspace "
                "bytes per token for every selected owner"
            )

        # Preflight every identity before any owner executes, so a missing or
        # stale replica cannot produce a partially applied layer.
        for dispatch in plan.dispatches:
            owner = owners.get(dispatch.owner_id)
            if owner is None:
                raise ExpertRouteUnavailableError(
                    f"owner executor {dispatch.owner_id!r} is missing"
                )
            record = self._experts[dispatch.key]
            if not self._owner_has_exact_content(
                owner,
                dispatch.key,
                record.content_id,
            ):
                raise ExpertRouteUnavailableError(
                    f"owner {dispatch.owner_id!r} lacks exact content for {dispatch.key}"
                )
            if (
                dispatch.path in {"local-gpu", "remote-resident"}
                and not self._owner_reports_resident(
                    owner,
                    dispatch.key,
                    record.content_id,
                )
            ):
                raise ExpertRouteUnavailableError(
                    f"owner {dispatch.owner_id!r} lost physical residency "
                    f"for {dispatch.key}"
                )

        by_owner: dict[str, list[ExpertDispatch]] = {}
        for dispatch in plan.dispatches:
            by_owner.setdefault(dispatch.owner_id, []).append(dispatch)

        # Build assignment tensors once per dispatch.  One expert may be split
        # across multiple replica owners, so the physical dispatch (not just
        # the logical key) identifies the exact row/slot subset used for both
        # activation selection and weighted reduction. Router gates are
        # gathered in one operation instead of extracting scalar tensors in
        # a Python loop.
        assignment_indexes: dict[
            ExpertDispatch,
            tuple[torch.Tensor, torch.Tensor],
        ] = {}
        for dispatch in plan.dispatches:
            assignment_indexes[dispatch] = (
                torch.tensor(
                    [assignment.token_index for assignment in dispatch.assignments],
                    dtype=torch.long,
                    device=hidden.device,
                ),
                torch.tensor(
                    [assignment.slot_index for assignment in dispatch.assignments],
                    dtype=torch.long,
                    device=hidden.device,
                ),
            )

        def run_owner(
            owner_id: str,
            owner_dispatches: Sequence[ExpertDispatch],
        ) -> list[tuple[ExpertDispatch, torch.Tensor]]:
            owner = owners[owner_id]
            dispatch_by_key: dict[ExpertKey, ExpertDispatch] = {}
            ordered_dispatches = tuple(
                sorted(owner_dispatches, key=lambda value: value.key)
            )
            for dispatch in ordered_dispatches:
                if dispatch.key in dispatch_by_key:
                    raise ResidentExpertMeshError(
                        f"owner batch repeats expert {dispatch.key}"
                    )
                dispatch_by_key[dispatch.key] = dispatch

            # Coalesce only when at least one token position is reused by two
            # experts on this owner.  Otherwise v1 has identical tensor bytes
            # and avoids the row-map overhead.  The shared rows are globally
            # ordered; each expert's references retain its original assignment
            # order, so the returned tensors are identical to v1 inputs.
            unique_positions = tuple(
                sorted(
                    {
                        assignment.token_index
                        for dispatch in ordered_dispatches
                        for assignment in dispatch.assignments
                    }
                )
            )
            assignment_count = sum(
                len(dispatch.assignments) for dispatch in ordered_dispatches
            )
            coalesced_execute = getattr(owner, "execute_coalesced_batch", None)
            supports_coalescing = self._owner_supports_exact_input_coalescing(owner)
            row_index_bytes = getattr(
                owner,
                "coalesced_row_index_bytes_per_assignment",
                0,
            )
            if (
                not isinstance(row_index_bytes, int)
                or isinstance(row_index_bytes, bool)
                or row_index_bytes < 0
            ):
                raise ResidentExpertMeshError(
                    f"owner {owner_id!r} advertises an invalid coalesced metadata cost"
                )
            saved_input_bytes = (
                assignment_count - len(unique_positions)
            ) * self.activation_bytes_per_token
            metadata_bytes = assignment_count * row_index_bytes
            use_coalesced = (
                callable(coalesced_execute)
                and supports_coalescing is True
                and saved_input_bytes > metadata_bytes
            )
            if use_coalesced:
                owner_rows = {
                    token_index: row_index
                    for row_index, token_index in enumerate(unique_positions)
                }
                shared_indices = torch.tensor(
                    unique_positions,
                    dtype=torch.long,
                    device=hidden.device,
                )
                shared_hidden = hidden.index_select(0, shared_indices)
                coalesced_items = tuple(
                    OwnerCoalescedExpertBatchItem(
                        dispatch.key,
                        self._experts[dispatch.key].content_id,
                        tuple(
                            owner_rows[assignment.token_index]
                            for assignment in dispatch.assignments
                        ),
                        require_resident=(
                            dispatch.path in {"local-gpu", "remote-resident"}
                        ),
                    )
                    for dispatch in ordered_dispatches
                )
                results = coalesced_execute(shared_hidden, coalesced_items)
                expected_count = len(coalesced_items)
            else:
                batch_items: list[OwnerExpertBatchItem] = []
                for dispatch in ordered_dispatches:
                    indices, _ = assignment_indexes[dispatch]
                    selected_hidden = hidden.index_select(0, indices)
                    record = self._experts[dispatch.key]
                    batch_items.append(
                        OwnerExpertBatchItem(
                            dispatch.key,
                            record.content_id,
                            selected_hidden,
                            require_resident=(
                                dispatch.path in {"local-gpu", "remote-resident"}
                            ),
                        )
                    )
                results = owner.execute_batch(tuple(batch_items))
                expected_count = len(batch_items)
            if len(results) != expected_count:
                raise ResidentExpertMeshError(
                    f"owner {owner_id!r} returned an incomplete expert batch"
                )
            values: list[tuple[ExpertDispatch, torch.Tensor]] = []
            returned_keys: set[ExpertKey] = set()
            for result in results:
                if not isinstance(result, OwnerExpertBatchResult):
                    raise ResidentExpertMeshError(
                        f"owner {owner_id!r} returned an invalid result type"
                    )
                if result.key in returned_keys or result.key not in dispatch_by_key:
                    raise ResidentExpertMeshError(
                        f"owner {owner_id!r} returned an invalid expert batch"
                    )
                returned_keys.add(result.key)
                values.append((dispatch_by_key[result.key], result.output))
            if returned_keys != set(dispatch_by_key):
                raise ResidentExpertMeshError(
                    f"owner {owner_id!r} omitted an expert batch result"
                )
            return values

        completed: list[tuple[ExpertDispatch, torch.Tensor]] = []
        failure_lock = threading.Lock()
        primary_failure: list[tuple[str, BaseException]] = []

        def guarded_run_owner(
            owner_id: str,
            owner_dispatches: Sequence[ExpertDispatch],
        ) -> list[tuple[ExpertDispatch, torch.Tensor]]:
            try:
                return run_owner(owner_id, owner_dispatches)
            except BaseException as error:
                with failure_lock:
                    if not primary_failure:
                        primary_failure.append((owner_id, error))
                raise

        with self._executor_lock:
            if self._closed:
                raise ResidentExpertMeshError("resident expert mesh is closed")
            predicted_owner_ms = dict(plan.owner_exposed_ms)
            execution_owner_ids = tuple(
                sorted(
                    by_owner,
                    key=lambda owner_id: (
                        -predicted_owner_ms.get(owner_id, 0.0),
                        owner_id,
                    ),
                )
            )
            futures = {
                owner_id: self._executor.submit(
                    guarded_run_owner,
                    owner_id,
                    by_owner[owner_id],
                )
                for owner_id in execution_owner_ids
            }
        wait(tuple(futures.values()), return_when=FIRST_EXCEPTION)
        if primary_failure:
            primary_owner_id, _ = primary_failure[0]
            primary_future = futures[primary_owner_id]
            for future in futures.values():
                if future is not primary_future and not future.done():
                    future.cancel()
            # A cancelled future may already be running. Drain every owner so
            # retry/teardown can never overlap an orphan RPC or GPU operation.
            failures: list[tuple[str, BaseException]] = []
            for owner_id, future in sorted(futures.items()):
                if future.cancelled():
                    continue
                try:
                    future.result()
                except BaseException as error:
                    failures.append((owner_id, error))
            fatal_failures = tuple(
                failure
                for failure in failures
                if not isinstance(
                    failure[1],
                    ExpertResidentSlotUnavailableError,
                )
            )
            if fatal_failures:
                raise fatal_failures[0][1]
            lost_slots: set[tuple[str, ExpertKey]] = set()
            for owner_id, error in failures:
                assert isinstance(error, ExpertResidentSlotUnavailableError)
                if not any(
                    dispatch.owner_id == owner_id
                    and dispatch.key == error.key
                    and dispatch.path in {"local-gpu", "remote-resident"}
                    for dispatch in plan.dispatches
                ):
                    raise ResidentExpertMeshError(
                        f"owner {owner_id!r} reported an unrelated resident slot loss"
                    ) from error
                lost_slots.add((owner_id, error.key))
            new_lost_slots = lost_slots - runtime_unavailable_resident
            if not new_lost_slots:
                raise ResidentExpertMeshError(
                    "resident expert ownership changed without replanning progress"
                ) from failures[0][1]
            # No reduction has started and every peer has quiesced. Re-run the
            # complete authoritative layer while excluding every newly stale
            # slot; partial owner outputs from this attempt are discarded.
            raise _RetryResidentSlots(frozenset(new_lost_slots)) from failures[0][1]
        for owner_id in sorted(futures):
            completed.extend(futures[owner_id].result())

        canonical_completed = sorted(
            completed,
            key=lambda value: (
                value[0].key,
                value[0].owner_id,
                value[0].path,
                value[0].assignments,
            ),
        )
        normalized: list[tuple[ExpertDispatch, torch.Tensor]] = []
        # Validate every result before allocating or mutating the reduction
        # target.  In particular, an RPC CPU tensor is copied explicitly back
        # to the hidden device only after the whole layer batch is accepted.
        for dispatch, expert_output in canonical_completed:
            if not isinstance(expert_output, torch.Tensor):
                raise ResidentExpertMeshError(
                    f"expert {dispatch.key} returned a non-tensor result"
                )
            if expert_output.ndim != 2 or tuple(expert_output.shape) != (
                len(dispatch.assignments),
                hidden.shape[1],
            ):
                raise ResidentExpertMeshError(
                    f"expert {dispatch.key} returned an invalid tensor shape"
                )
            if (
                not torch.is_floating_point(expert_output)
                or expert_output.dtype != hidden.dtype
            ):
                raise ResidentExpertMeshError(
                    f"expert {dispatch.key} returned an invalid tensor dtype"
                )
            normalized.append(
                (
                    dispatch,
                    expert_output.to(
                        device=hidden.device,
                        dtype=hidden.dtype,
                    ).contiguous(),
                )
            )

        output = torch.zeros_like(hidden)
        for dispatch, expert_output in normalized:
            indices, slots = assignment_indexes[dispatch]
            gates = routing.expert_weights[indices, slots]
            weighted = expert_output * gates.unsqueeze(-1)
            output.index_add_(0, indices, weighted.to(dtype=output.dtype))
        return output, plan

    def execute_layers(
        self,
        hidden: torch.Tensor,
        layers: Sequence[int],
        target_routers: Sequence[Callable[[torch.Tensor], AuthoritativeRouting]],
        owners: Mapping[str, ExpertOwner],
    ) -> tuple[torch.Tensor, tuple[LayerMeshPlan, ...]]:
        if len(layers) != len(target_routers):
            raise ValueError("layers and target_routers must have the same length")
        current = hidden
        plans: list[LayerMeshPlan] = []
        # This loop is intentionally sequential: layer N+1 cannot route until
        # the complete weighted output of layer N exists.
        for layer, router in zip(layers, target_routers):
            routing = router(current)
            current, plan = self.execute_layer(current, layer, routing, owners)
            plans.append(plan)
        return current, tuple(plans)

    def plan_hot_replicas(
        self,
        demand: Sequence[ExplicitExpertDemand],
        *,
        candidate_node_ids: Sequence[str] | None = None,
    ) -> HotReplicaPlan:
        by_key: dict[ExpertKey, ExplicitExpertDemand] = {}
        for item in demand:
            if not isinstance(item, ExplicitExpertDemand):
                raise TypeError("demand must contain ExplicitExpertDemand values")
            if item.key in by_key:
                raise ValueError(f"duplicate explicit demand for {item.key}")
            if item.key not in self._experts:
                raise ValueError(f"demand references unknown expert {item.key}")
            by_key[item.key] = item

        node_ids = (
            tuple(candidate_node_ids)
            if candidate_node_ids is not None
            else tuple(
                node_id for node_id in self._nodes if node_id != self.coordinator_id
            )
        )
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("candidate_node_ids cannot contain duplicates")
        for node_id in node_ids:
            if node_id == self.coordinator_id or node_id not in self._nodes:
                raise ValueError(f"invalid replica candidate node {node_id!r}")

        used = self._resident_vram_usage()
        existing_keys = set(self._replicas)
        candidates: list[tuple[float, float, ExpertKey, str]] = []
        coordinator = self._nodes[self.coordinator_id]
        for key, item in sorted(by_key.items()):
            if key in existing_keys or key not in self._local_ram_keys:
                continue
            record = self._experts[key]
            local_expected_ms = (
                item.active_probability
                * record.byte_size
                / coordinator.ram_to_device_bytes_per_ms
                + item.expected_assignments_per_wave
                * coordinator.expert_compute_ms_per_token
            )
            for node_id in sorted(node_ids):
                node = self._nodes[node_id]
                link = self._links.get((self.coordinator_id, node_id))
                if not node.available or link is None or not link.available:
                    continue
                remote_expected_ms = (
                    item.active_probability
                    * (link.round_trip_ms + link.rpc_setup_ms)
                    + item.expected_assignments_per_wave
                    * (
                        self.activation_bytes_per_token
                        / link.egress_bytes_per_ms
                        + self.activation_bytes_per_token
                        / link.ingress_bytes_per_ms
                        + (
                            4
                            * self.activation_bytes_per_token
                            / link.host_device_staging_bytes_per_ms
                            if link.host_device_staging_bytes_per_ms > 0
                            else 0.0
                        )
                        + node.expert_compute_ms_per_token
                    )
                )
                benefit = local_expected_ms - remote_expected_ms
                if benefit <= 0:
                    continue
                density = benefit / record.byte_size
                candidates.append((-density, -benefit, key, node_id))

        selected_keys: set[ExpertKey] = set()
        placements: list[ResidentExpertReplica] = []
        saved_ms = 0.0
        for negative_density, negative_benefit, key, node_id in sorted(candidates):
            if key in selected_keys:
                continue
            record = self._experts[key]
            node = self._nodes[node_id]
            if used[node_id] + record.byte_size > node.resident_vram_budget_bytes:
                continue
            used[node_id] += record.byte_size
            selected_keys.add(key)
            placements.append(
                ResidentExpertReplica(key, node_id, record.content_id)
            )
            saved_ms += -negative_benefit

        return HotReplicaPlan(
            placements=tuple(sorted(placements)),
            used_vram_bytes_by_node=tuple(sorted(used.items())),
            estimated_exposed_ms_saved_per_wave=saved_ms,
        )

    def _candidates(
        self,
        key: ExpertKey,
        assignment_count: int,
        unavailable_nodes: frozenset[str],
        unavailable_links: frozenset[tuple[str, str]],
        unavailable_resident_slots: frozenset[tuple[str, ExpertKey]],
    ) -> tuple[_Candidate, ...]:
        record = self._experts[key]
        coordinator = self._nodes[self.coordinator_id]
        local_compute = assignment_count * coordinator.expert_compute_ms_per_token
        # Conservative execution peak: selected input, down output and
        # weighted/reduction materialization (3H) coexist with operator
        # workspace. Network payload remains only input+output (2H).
        local_activation_bytes = (
            3 * self.activation_bytes_per_token * assignment_count
        )
        local_workspace_bytes = (
            coordinator.expert_workspace_bytes_per_token * assignment_count
        )
        candidates: list[_Candidate] = []
        if (
            key in self._local_gpu_keys
            and (self.coordinator_id, key) not in unavailable_resident_slots
        ):
            candidates.append(
                _Candidate(
                    key=key,
                    owner_id=self.coordinator_id,
                    path="local-gpu",
                    duration_ms=local_compute,
                    activation_bytes=0,
                    host_device_staging_bytes=0,
                    coordinator_dynamic_vram_bytes=(
                        local_activation_bytes + local_workspace_bytes
                    ),
                    owner_dynamic_vram_bytes=0,
                    weight_loaded_bytes=0,
                    weight_avoided_bytes=record.byte_size,
                    transport_calibration_required=False,
                    workspace_calibration_required=(
                        coordinator.expert_workspace_bytes_per_token <= 0
                    ),
                )
            )
        if key in self._local_ram_keys:
            candidates.append(
                _Candidate(
                    key=key,
                    owner_id=self.coordinator_id,
                    path="local-ram",
                    duration_ms=(
                        record.byte_size
                        / coordinator.ram_to_device_bytes_per_ms
                        + local_compute
                    ),
                    activation_bytes=0,
                    host_device_staging_bytes=0,
                    coordinator_dynamic_vram_bytes=(
                        local_activation_bytes + local_workspace_bytes
                    ),
                    owner_dynamic_vram_bytes=0,
                    weight_loaded_bytes=record.byte_size,
                    weight_avoided_bytes=0,
                    transport_calibration_required=False,
                    workspace_calibration_required=(
                        coordinator.expert_workspace_bytes_per_token <= 0
                    ),
                )
            )
        for replica in self._replicas.get(key, ()):
            node = self._nodes[replica.node_id]
            link_key = (self.coordinator_id, replica.node_id)
            link = self._links[link_key]
            if (
                not node.available
                or replica.node_id in unavailable_nodes
                or not link.available
                or link_key in unavailable_links
                or (replica.node_id, key) in unavailable_resident_slots
            ):
                continue
            activation_bytes = (
                2 * self.activation_bytes_per_token * assignment_count
            )
            # TCP RPC currently crosses the host/device boundary at both ends:
            # coordinator D2H input, owner H2D input, owner D2H output and
            # coordinator H2D output.  A zero calibration keeps the historical
            # optimistic ceiling but marks the plan as incomplete.
            staging_bytes = (
                4 * self.activation_bytes_per_token * assignment_count
            )
            staging_bytes_per_ms = link.host_device_staging_bytes_per_ms
            candidates.append(
                _Candidate(
                    key=key,
                    owner_id=replica.node_id,
                    path="remote-resident",
                    duration_ms=(
                        link.round_trip_ms
                        + link.rpc_setup_ms
                        + (
                            self.activation_bytes_per_token * assignment_count
                            / link.egress_bytes_per_ms
                        )
                        + (
                            self.activation_bytes_per_token * assignment_count
                            / link.ingress_bytes_per_ms
                        )
                        + (
                            staging_bytes / staging_bytes_per_ms
                            if staging_bytes_per_ms > 0
                            else 0.0
                        )
                        + assignment_count * node.expert_compute_ms_per_token
                    ),
                    activation_bytes=activation_bytes,
                    host_device_staging_bytes=staging_bytes,
                    coordinator_dynamic_vram_bytes=(
                        3 * self.activation_bytes_per_token * assignment_count
                    ),
                    owner_dynamic_vram_bytes=(
                        3 * self.activation_bytes_per_token * assignment_count
                        + node.expert_workspace_bytes_per_token * assignment_count
                    ),
                    weight_loaded_bytes=0,
                    weight_avoided_bytes=record.byte_size,
                    transport_calibration_required=(
                        staging_bytes_per_ms <= 0
                        or link.rpc_setup_ms_per_batch is None
                    ),
                    workspace_calibration_required=(
                        node.expert_workspace_bytes_per_token <= 0
                    ),
                )
            )
        return tuple(candidates)

    def _index_experts(
        self,
        experts: Sequence[ExpertRecord],
    ) -> dict[ExpertKey, ExpertRecord]:
        if not experts:
            raise ValueError("experts cannot be empty")
        result: dict[ExpertKey, ExpertRecord] = {}
        for record in experts:
            if not isinstance(record, ExpertRecord):
                raise TypeError("experts must contain ExpertRecord values")
            if record.key in result:
                raise ValueError(f"duplicate expert {record.key}")
            result[record.key] = record
        return result

    def _index_nodes(
        self,
        nodes: Sequence[MeshNodeProfile],
    ) -> dict[str, MeshNodeProfile]:
        if not nodes:
            raise ValueError("nodes cannot be empty")
        result: dict[str, MeshNodeProfile] = {}
        for node in nodes:
            if not isinstance(node, MeshNodeProfile):
                raise TypeError("nodes must contain MeshNodeProfile values")
            if node.node_id in result:
                raise ValueError(f"duplicate mesh node {node.node_id!r}")
            result[node.node_id] = node
        return result

    def _index_links(
        self,
        links: Sequence[MeshLinkProfile],
    ) -> dict[tuple[str, str], MeshLinkProfile]:
        result: dict[tuple[str, str], MeshLinkProfile] = {}
        for link in links:
            if not isinstance(link, MeshLinkProfile):
                raise TypeError("links must contain MeshLinkProfile values")
            key = (link.from_node, link.to_node)
            if key in result:
                raise ValueError(f"duplicate mesh link {key}")
            if link.from_node not in self._nodes or link.to_node not in self._nodes:
                raise ValueError(f"mesh link {key} references an unknown node")
            result[key] = link
        return result

    def _validate_keys(self, keys: frozenset[ExpertKey], name: str) -> None:
        for key in keys:
            if key not in self._experts:
                raise ValueError(f"{name} references unknown expert {key}")

    def _resident_vram_usage(self) -> dict[str, int]:
        used = {
            node_id: node.reserved_vram_bytes
            for node_id, node in self._nodes.items()
        }
        # The coordinator's bounded streaming slot competes with resident
        # experts for the same sealed VRAM budget.
        used[self.coordinator_id] += self.local_weight_buffer_bytes
        resident_slots: set[tuple[str, ExpertKey]] = set()
        for key in self._local_gpu_keys:
            resident_slots.add((self.coordinator_id, key))
        for key, replicas in self._replicas.items():
            for replica in replicas:
                resident_slots.add((replica.node_id, key))
        for node_id, key in resident_slots:
            used[node_id] += self._experts[key].byte_size
        return used

    def _validate_resident_vram_budgets(self) -> None:
        for node_id, used in self._resident_vram_usage().items():
            if used > self._nodes[node_id].resident_vram_budget_bytes:
                raise ValueError(
                    f"resident expert VRAM budget exceeded on {node_id}: "
                    f"{used} > {self._nodes[node_id].resident_vram_budget_bytes}"
                )


class InMemoryExpertOwner:
    """Small exact owner used by CPU tests; it does not model a transport."""

    supports_exact_input_coalescing = True
    coalesced_row_index_bytes_per_assignment = 0

    def __init__(
        self,
        node_id: str,
        experts: Mapping[ExpertKey, tuple[str, torch.Tensor]],
    ) -> None:
        self.node_id = _name("node_id", node_id)
        if not experts:
            raise ValueError("in-memory owner experts cannot be empty")
        normalized: dict[ExpertKey, tuple[str, torch.Tensor]] = {}
        for key, (content_id, weight) in experts.items():
            if not isinstance(key, ExpertKey):
                raise TypeError("owner expert keys must be ExpertKey values")
            identity = _name("content_id", content_id)
            if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
                raise TypeError("owner expert weights must be rank-two tensors")
            normalized[key] = (identity, weight.detach().clone())
        self._experts = normalized
        self.batch_calls = 0
        self.coalesced_batch_calls = 0

    def has_expert(self, key: ExpertKey, content_id: str) -> bool:
        current = self._experts.get(key)
        return current is not None and current[0] == content_id

    def is_expert_resident(self, key: ExpertKey, content_id: str) -> bool:
        # This test owner stores its complete executable tensor in-memory; its
        # inventory and physical resident set are intentionally identical.
        return self.has_expert(key, content_id)

    @torch.no_grad()
    def execute_batch(
        self,
        items: Sequence[OwnerExpertBatchItem],
    ) -> tuple[OwnerExpertBatchResult, ...]:
        if not items:
            raise ValueError("owner expert batch cannot be empty")
        self.batch_calls += 1
        results: list[OwnerExpertBatchResult] = []
        seen: set[ExpertKey] = set()
        for item in items:
            if item.key in seen:
                raise ValueError(f"owner expert batch repeats {item.key}")
            seen.add(item.key)
            current = self._experts.get(item.key)
            if current is None or current[0] != item.content_id:
                raise ExpertRouteUnavailableError(
                    f"owner {self.node_id!r} does not contain exact expert {item.key}"
                )
            weight = current[1].to(
                device=item.activations.device,
                dtype=item.activations.dtype,
            )
            if (
                item.activations.ndim != 2
                or item.activations.shape[1] != weight.shape[1]
            ):
                raise ValueError("expert activation shape does not match its weight")
            results.append(
                OwnerExpertBatchResult(
                    item.key,
                    item.activations @ weight.transpose(0, 1),
                )
            )
        return tuple(results)

    @torch.no_grad()
    def execute_coalesced_batch(
        self,
        shared_activations: torch.Tensor,
        items: Sequence[OwnerCoalescedExpertBatchItem],
    ) -> tuple[OwnerExpertBatchResult, ...]:
        if (
            not isinstance(shared_activations, torch.Tensor)
            or shared_activations.ndim != 2
            or shared_activations.shape[0] < 1
        ):
            raise ValueError("shared owner activations must be a non-empty matrix")
        batch = tuple(items)
        if not batch:
            raise ValueError("coalesced owner expert batch cannot be empty")
        if any(not isinstance(item, OwnerCoalescedExpertBatchItem) for item in batch):
            raise TypeError(
                "coalesced owner batch must contain OwnerCoalescedExpertBatchItem values"
            )
        keys = tuple(item.key for item in batch)
        if len(set(keys)) != len(keys) or keys != tuple(sorted(keys)):
            raise ValueError("coalesced owner expert batch must have unique canonical keys")
        if any(
            row >= shared_activations.shape[0]
            for item in batch
            for row in item.row_indices
        ):
            raise ValueError("coalesced owner row index exceeds shared activations")
        covered_rows = bytearray(int(shared_activations.shape[0]))
        covered_count = 0
        for item in batch:
            for row in item.row_indices:
                if not covered_rows[row]:
                    covered_rows[row] = 1
                    covered_count += 1
        if covered_count != len(covered_rows):
            raise ValueError(
                "coalesced owner row map must reference every shared activation"
            )

        self.batch_calls += 1
        self.coalesced_batch_calls += 1
        results: list[OwnerExpertBatchResult] = []
        for item in batch:
            current = self._experts.get(item.key)
            if current is None or current[0] != item.content_id:
                raise ExpertRouteUnavailableError(
                    f"owner {self.node_id!r} does not contain exact expert {item.key}"
                )
            weight = current[1].to(
                device=shared_activations.device,
                dtype=shared_activations.dtype,
            )
            if shared_activations.shape[1] != weight.shape[1]:
                raise ValueError("shared activation shape does not match expert weight")
            rows = torch.tensor(
                item.row_indices,
                dtype=torch.long,
                device=shared_activations.device,
            )
            selected = shared_activations.index_select(0, rows)
            results.append(
                OwnerExpertBatchResult(
                    item.key,
                    selected @ weight.transpose(0, 1),
                )
            )
        return tuple(results)


def project_mesh_wave(
    layer_plans: Sequence[LayerMeshPlan],
    *,
    committed_tokens: float,
) -> MeshWaveProjection:
    if not layer_plans:
        raise ValueError("layer_plans cannot be empty")
    committed = _finite("committed_tokens", committed_tokens)
    if committed <= 0:
        raise ValueError("committed_tokens must be positive")
    token_count = layer_plans[0].token_count
    if any(plan.token_count != token_count for plan in layer_plans):
        raise ValueError("every sequential layer must describe the same wave width")
    if committed > token_count:
        raise ValueError("committed_tokens cannot exceed routed wave positions")
    exposed_ms = sum(plan.exposed_ms for plan in layer_plans)
    if exposed_ms <= 0:
        raise ValueError("projected mesh wave time must be positive")
    return MeshWaveProjection(
        schema=RESIDENT_EXPERT_MESH_SCHEMA,
        layer_plans=tuple(layer_plans),
        committed_tokens=committed,
        exposed_ms_per_wave=exposed_ms,
        activation_round_trip_bytes_per_wave=sum(
            plan.activation_round_trip_bytes for plan in layer_plans
        ),
        activation_request_bytes_per_wave=sum(
            plan.activation_request_bytes for plan in layer_plans
        ),
        activation_response_bytes_per_wave=sum(
            plan.activation_response_bytes for plan in layer_plans
        ),
        route_metadata_bytes_per_wave=sum(
            plan.route_metadata_bytes for plan in layer_plans
        ),
        transport_payload_bytes_per_wave=sum(
            plan.transport_payload_bytes for plan in layer_plans
        ),
        host_device_staging_bytes_per_wave=sum(
            plan.host_device_staging_bytes for plan in layer_plans
        ),
        weight_loaded_bytes_per_wave=sum(
            plan.weight_loaded_bytes for plan in layer_plans
        ),
        weight_avoided_bytes_per_wave=sum(
            plan.weight_avoided_bytes for plan in layer_plans
        ),
        transport_calibration_required=any(
            plan.transport_calibration_required for plan in layer_plans
        ),
        workspace_calibration_required=any(
            plan.workspace_calibration_required for plan in layer_plans
        ),
        tokens_per_second=committed * 1_000.0 / exposed_ms,
    )
