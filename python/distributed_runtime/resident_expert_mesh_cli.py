"""Reproducible, closed-schema simulator for the resident expert mesh."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .ram_expert_cache import ExpertKey, ExpertRecord
from .resident_expert_mesh import (
    AuthoritativeRouting,
    MeshLinkProfile,
    MeshNodeProfile,
    ResidentExpertMesh,
    ResidentExpertReplica,
    project_mesh_wave,
)


SIMULATION_SCHEMA = "gdlp-resident-expert-mesh-simulation/1"
REPORT_SCHEMA = "gdlp-resident-expert-mesh-report/1"


def _record(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    return value


def _exact(value: Mapping[str, Any], keys: set[str], path: str) -> None:
    actual = set(value)
    if actual != keys:
        missing = sorted(keys - actual)
        extra = sorted(actual - keys)
        raise ValueError(f"{path} keys are invalid: missing={missing}, extra={extra}")


def _text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value.strip()


def _int(value: object, path: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{path} must be an integer >= {minimum}")
    return value


def _number(value: object, path: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{path} must be finite and >= {minimum}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} must be finite and >= {minimum}") from exc
    if not math.isfinite(number) or number < minimum:
        raise ValueError(f"{path} must be finite and >= {minimum}")
    return number


def _boolean(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{path} must be boolean")
    return value


@dataclass(frozen=True)
class ModelInput:
    sparse_layers: int
    experts_per_layer: int
    expert_bytes: int
    activation_elements: int
    activation_dtype_bytes: int
    top_k: int


@dataclass(frozen=True)
class WaveInput:
    positions: int
    positions_per_sequence: int
    concurrent_sequences: int
    committed_tokens: float
    routing_union_experts_per_layer: int


@dataclass(frozen=True)
class CoordinatorInput:
    node_id: str
    usable_vram_bytes: int
    reserved_vram_bytes: int
    weight_buffer_bytes: int
    ram_to_gpu_gbytes_per_second: float
    expert_compute_ms_per_token: float


@dataclass(frozen=True)
class OwnerInput:
    node_id: str
    usable_vram_bytes: int
    reserved_vram_bytes: int
    expert_compute_ms_per_token: float
    round_trip_ms: float | None = None
    bandwidth_mbps: float | None = None
    egress_bandwidth_mbps: float | None = None
    ingress_bandwidth_mbps: float | None = None
    supports_exact_input_coalescing: bool = True
    row_index_bytes_per_assignment: int = 4

    @property
    def expert_capacity_bytes(self) -> int:
        return self.usable_vram_bytes - self.reserved_vram_bytes


@dataclass(frozen=True)
class NetworkInput:
    round_trip_ms: float
    bandwidth_mbps: float
    egress_bandwidth_mbps: float
    ingress_bandwidth_mbps: float


@dataclass(frozen=True)
class SimulationInput:
    scenario_id: str
    model: ModelInput
    wave: WaveInput
    coordinator: CoordinatorInput
    owners: tuple[OwnerInput, ...]
    network: NetworkInput
    sensitivity_rtt_ms: tuple[float, ...]
    macro_wave_cold_tokens_per_second: float

    @classmethod
    def from_document(cls, value: object) -> "SimulationInput":
        root = _record(value, "input")
        _exact(
            root,
            {
                "schema",
                "scenarioId",
                "model",
                "wave",
                "coordinator",
                "owners",
                "network",
                "sensitivityRttMs",
                "comparison",
            },
            "input",
        )
        if root["schema"] != SIMULATION_SCHEMA:
            raise ValueError("input.schema is not supported")

        model_value = _record(root["model"], "input.model")
        _exact(
            model_value,
            {
                "sparseLayers",
                "expertsPerLayer",
                "expertBytes",
                "activationElements",
                "activationDtypeBytes",
                "topK",
            },
            "input.model",
        )
        model = ModelInput(
            sparse_layers=_int(
                model_value["sparseLayers"],
                "input.model.sparseLayers",
                minimum=1,
            ),
            experts_per_layer=_int(
                model_value["expertsPerLayer"],
                "input.model.expertsPerLayer",
                minimum=1,
            ),
            expert_bytes=_int(
                model_value["expertBytes"],
                "input.model.expertBytes",
                minimum=1,
            ),
            activation_elements=_int(
                model_value["activationElements"],
                "input.model.activationElements",
                minimum=1,
            ),
            activation_dtype_bytes=_int(
                model_value["activationDtypeBytes"],
                "input.model.activationDtypeBytes",
                minimum=1,
            ),
            top_k=_int(model_value["topK"], "input.model.topK", minimum=1),
        )
        if model.top_k > model.experts_per_layer:
            raise ValueError("input.model.topK exceeds expertsPerLayer")
        if model.activation_dtype_bytes not in (1, 2, 4, 8):
            raise ValueError("input.model.activationDtypeBytes must be 1, 2, 4 or 8")

        wave_value = _record(root["wave"], "input.wave")
        _exact(
            wave_value,
            {
                "positions",
                "positionsPerSequence",
                "concurrentSequences",
                "committedTokens",
                "routingUnionExpertsPerLayer",
            },
            "input.wave",
        )
        wave = WaveInput(
            positions=_int(
                wave_value["positions"],
                "input.wave.positions",
                minimum=1,
            ),
            positions_per_sequence=_int(
                wave_value["positionsPerSequence"],
                "input.wave.positionsPerSequence",
                minimum=1,
            ),
            concurrent_sequences=_int(
                wave_value["concurrentSequences"],
                "input.wave.concurrentSequences",
                minimum=1,
            ),
            committed_tokens=_number(
                wave_value["committedTokens"],
                "input.wave.committedTokens",
            ),
            routing_union_experts_per_layer=_int(
                wave_value["routingUnionExpertsPerLayer"],
                "input.wave.routingUnionExpertsPerLayer",
                minimum=1,
            ),
        )
        if wave.positions != wave.positions_per_sequence * wave.concurrent_sequences:
            raise ValueError(
                "input.wave.positions must equal positionsPerSequence * "
                "concurrentSequences"
            )
        if not 0 < wave.committed_tokens <= wave.positions_per_sequence:
            raise ValueError(
                "input.wave.committedTokens must be in (0, positionsPerSequence]"
            )
        if not model.top_k <= wave.routing_union_experts_per_layer:
            raise ValueError("routing union cannot be smaller than topK")
        if wave.routing_union_experts_per_layer > model.experts_per_layer:
            raise ValueError("routing union exceeds expertsPerLayer")
        if wave.routing_union_experts_per_layer > wave.positions * model.top_k:
            raise ValueError("routing union cannot be covered by the declared wave")

        coordinator_value = _record(root["coordinator"], "input.coordinator")
        _exact(
            coordinator_value,
            {
                "id",
                "usableVramBytes",
                "reservedVramBytes",
                "weightBufferBytes",
                "ramToGpuGbytesPerSecond",
                "expertComputeMsPerToken",
            },
            "input.coordinator",
        )
        coordinator = CoordinatorInput(
            node_id=_text(coordinator_value["id"], "input.coordinator.id"),
            usable_vram_bytes=_int(
                coordinator_value["usableVramBytes"],
                "input.coordinator.usableVramBytes",
                minimum=1,
            ),
            reserved_vram_bytes=_int(
                coordinator_value["reservedVramBytes"],
                "input.coordinator.reservedVramBytes",
            ),
            weight_buffer_bytes=_int(
                coordinator_value["weightBufferBytes"],
                "input.coordinator.weightBufferBytes",
                minimum=1,
            ),
            ram_to_gpu_gbytes_per_second=_number(
                coordinator_value["ramToGpuGbytesPerSecond"],
                "input.coordinator.ramToGpuGbytesPerSecond",
            ),
            expert_compute_ms_per_token=_number(
                coordinator_value["expertComputeMsPerToken"],
                "input.coordinator.expertComputeMsPerToken",
            ),
        )
        if coordinator.ram_to_gpu_gbytes_per_second <= 0:
            raise ValueError("input.coordinator.ramToGpuGbytesPerSecond must be positive")
        if coordinator.weight_buffer_bytes < model.expert_bytes:
            raise ValueError("coordinator weight buffer cannot hold one expert")
        if (
            coordinator.reserved_vram_bytes + coordinator.weight_buffer_bytes
            > coordinator.usable_vram_bytes
        ):
            raise ValueError("coordinator VRAM budget is exceeded")

        owners_value = root["owners"]
        if not isinstance(owners_value, list) or not owners_value:
            raise ValueError("input.owners must be a non-empty array")
        owners: list[OwnerInput] = []
        owner_ids: set[str] = set()
        for index, raw_owner in enumerate(owners_value):
            owner_value = _record(raw_owner, f"input.owners.{index}")
            owner_required = {
                "id",
                "usableVramBytes",
                "reservedVramBytes",
                "expertComputeMsPerToken",
            }
            owner_allowed = owner_required | {
                "roundTripMs",
                "bandwidthMbps",
                "egressBandwidthMbps",
                "ingressBandwidthMbps",
                "supportsExactInputCoalescing",
                "rowIndexBytesPerAssignment",
            }
            if not owner_required.issubset(owner_value) or not set(
                owner_value
            ).issubset(owner_allowed):
                raise ValueError(
                    f"input.owners.{index} keys mismatch: missing="
                    f"{sorted(owner_required - set(owner_value))}, extra="
                    f"{sorted(set(owner_value) - owner_allowed)}"
                )
            owner = OwnerInput(
                node_id=_text(owner_value["id"], f"input.owners.{index}.id"),
                usable_vram_bytes=_int(
                    owner_value["usableVramBytes"],
                    f"input.owners.{index}.usableVramBytes",
                    minimum=1,
                ),
                reserved_vram_bytes=_int(
                    owner_value["reservedVramBytes"],
                    f"input.owners.{index}.reservedVramBytes",
                ),
                expert_compute_ms_per_token=_number(
                    owner_value["expertComputeMsPerToken"],
                    f"input.owners.{index}.expertComputeMsPerToken",
                ),
                round_trip_ms=(
                    _number(
                        owner_value["roundTripMs"],
                        f"input.owners.{index}.roundTripMs",
                    )
                    if "roundTripMs" in owner_value
                    else None
                ),
                bandwidth_mbps=(
                    _number(
                        owner_value["bandwidthMbps"],
                        f"input.owners.{index}.bandwidthMbps",
                    )
                    if "bandwidthMbps" in owner_value
                    else None
                ),
                egress_bandwidth_mbps=(
                    _number(
                        owner_value["egressBandwidthMbps"],
                        f"input.owners.{index}.egressBandwidthMbps",
                    )
                    if "egressBandwidthMbps" in owner_value
                    else None
                ),
                ingress_bandwidth_mbps=(
                    _number(
                        owner_value["ingressBandwidthMbps"],
                        f"input.owners.{index}.ingressBandwidthMbps",
                    )
                    if "ingressBandwidthMbps" in owner_value
                    else None
                ),
                supports_exact_input_coalescing=_boolean(
                    owner_value.get("supportsExactInputCoalescing", True),
                    f"input.owners.{index}.supportsExactInputCoalescing",
                ),
                row_index_bytes_per_assignment=_int(
                    owner_value.get("rowIndexBytesPerAssignment", 4),
                    f"input.owners.{index}.rowIndexBytesPerAssignment",
                ),
            )
            if owner.node_id == coordinator.node_id or owner.node_id in owner_ids:
                raise ValueError("owner ids must be unique and distinct from coordinator")
            if owner.reserved_vram_bytes >= owner.usable_vram_bytes:
                raise ValueError(f"owner {owner.node_id!r} has no expert VRAM capacity")
            if any(
                value is not None and value <= 0
                for value in (
                    owner.bandwidth_mbps,
                    owner.egress_bandwidth_mbps,
                    owner.ingress_bandwidth_mbps,
                )
            ):
                raise ValueError(
                    f"owner {owner.node_id!r} bandwidth overrides must be positive"
                )
            owner_ids.add(owner.node_id)
            owners.append(owner)

        network_value = _record(root["network"], "input.network")
        network_required = {"roundTripMs", "bandwidthMbps"}
        network_allowed = network_required | {
            "egressBandwidthMbps",
            "ingressBandwidthMbps",
        }
        if not network_required.issubset(network_value) or not set(
            network_value
        ).issubset(network_allowed):
            raise ValueError(
                "input.network keys mismatch: requires roundTripMs and "
                "bandwidthMbps; only directional bandwidth overrides are optional"
            )
        base_bandwidth = _number(
            network_value["bandwidthMbps"],
            "input.network.bandwidthMbps",
        )
        network = NetworkInput(
            round_trip_ms=_number(
                network_value["roundTripMs"],
                "input.network.roundTripMs",
            ),
            bandwidth_mbps=base_bandwidth,
            egress_bandwidth_mbps=_number(
                network_value.get("egressBandwidthMbps", base_bandwidth),
                "input.network.egressBandwidthMbps",
            ),
            ingress_bandwidth_mbps=_number(
                network_value.get("ingressBandwidthMbps", base_bandwidth),
                "input.network.ingressBandwidthMbps",
            ),
        )
        if min(
            network.bandwidth_mbps,
            network.egress_bandwidth_mbps,
            network.ingress_bandwidth_mbps,
        ) <= 0:
            raise ValueError("input.network bandwidth values must be positive")

        sensitivity_value = root["sensitivityRttMs"]
        if not isinstance(sensitivity_value, list) or not sensitivity_value:
            raise ValueError("input.sensitivityRttMs must be a non-empty array")
        sensitivity = tuple(
            _number(item, f"input.sensitivityRttMs.{index}")
            for index, item in enumerate(sensitivity_value)
        )
        if len(set(sensitivity)) != len(sensitivity):
            raise ValueError("input.sensitivityRttMs cannot contain duplicates")

        comparison = _record(root["comparison"], "input.comparison")
        _exact(
            comparison,
            {"macroWaveColdTokensPerSecond"},
            "input.comparison",
        )
        macro_wave = _number(
            comparison["macroWaveColdTokensPerSecond"],
            "input.comparison.macroWaveColdTokensPerSecond",
        )
        if macro_wave <= 0:
            raise ValueError("MacroWave comparison throughput must be positive")

        return cls(
            scenario_id=_text(root["scenarioId"], "input.scenarioId"),
            model=model,
            wave=wave,
            coordinator=coordinator,
            owners=tuple(sorted(owners, key=lambda owner: owner.node_id)),
            network=network,
            sensitivity_rtt_ms=tuple(sorted(sensitivity)),
            macro_wave_cold_tokens_per_second=macro_wave,
        )

    def canonical_document(self) -> dict[str, Any]:
        return {
            "schema": SIMULATION_SCHEMA,
            "scenarioId": self.scenario_id,
            "model": {
                "sparseLayers": self.model.sparse_layers,
                "expertsPerLayer": self.model.experts_per_layer,
                "expertBytes": self.model.expert_bytes,
                "activationElements": self.model.activation_elements,
                "activationDtypeBytes": self.model.activation_dtype_bytes,
                "topK": self.model.top_k,
            },
            "wave": {
                "positions": self.wave.positions,
                "positionsPerSequence": self.wave.positions_per_sequence,
                "concurrentSequences": self.wave.concurrent_sequences,
                "committedTokens": self.wave.committed_tokens,
                "routingUnionExpertsPerLayer": (
                    self.wave.routing_union_experts_per_layer
                ),
            },
            "coordinator": {
                "id": self.coordinator.node_id,
                "usableVramBytes": self.coordinator.usable_vram_bytes,
                "reservedVramBytes": self.coordinator.reserved_vram_bytes,
                "weightBufferBytes": self.coordinator.weight_buffer_bytes,
                "ramToGpuGbytesPerSecond": (
                    self.coordinator.ram_to_gpu_gbytes_per_second
                ),
                "expertComputeMsPerToken": (
                    self.coordinator.expert_compute_ms_per_token
                ),
            },
            "owners": [
                {
                    "id": owner.node_id,
                    "usableVramBytes": owner.usable_vram_bytes,
                    "reservedVramBytes": owner.reserved_vram_bytes,
                    "expertComputeMsPerToken": owner.expert_compute_ms_per_token,
                    "roundTripMs": (
                        owner.round_trip_ms
                        if owner.round_trip_ms is not None
                        else self.network.round_trip_ms
                    ),
                    "bandwidthMbps": (
                        owner.bandwidth_mbps
                        if owner.bandwidth_mbps is not None
                        else self.network.bandwidth_mbps
                    ),
                    "egressBandwidthMbps": (
                        owner.egress_bandwidth_mbps
                        if owner.egress_bandwidth_mbps is not None
                        else self.network.egress_bandwidth_mbps
                    ),
                    "ingressBandwidthMbps": (
                        owner.ingress_bandwidth_mbps
                        if owner.ingress_bandwidth_mbps is not None
                        else self.network.ingress_bandwidth_mbps
                    ),
                    "supportsExactInputCoalescing": (
                        owner.supports_exact_input_coalescing
                    ),
                    "rowIndexBytesPerAssignment": (
                        owner.row_index_bytes_per_assignment
                    ),
                }
                for owner in self.owners
            ],
            "network": {
                "roundTripMs": self.network.round_trip_ms,
                "bandwidthMbps": self.network.bandwidth_mbps,
                "egressBandwidthMbps": self.network.egress_bandwidth_mbps,
                "ingressBandwidthMbps": self.network.ingress_bandwidth_mbps,
            },
            "sensitivityRttMs": list(self.sensitivity_rtt_ms),
            "comparison": {
                "macroWaveColdTokensPerSecond": (
                    self.macro_wave_cold_tokens_per_second
                )
            },
        }


@dataclass(frozen=True)
class _Placement:
    owners: tuple[OwnerInput, ...]
    replicas: tuple[ResidentExpertReplica, ...]
    resident_bytes_by_owner: tuple[tuple[str, int], ...]
    capacity_lower_bound_owners: int
    strategy: str
    minimum_owner_count_proven: bool
    search_states_evaluated: int
    search_limit_reached: bool
    owner_subset_strategy: str
    owner_subsets_evaluated: int
    owner_subset_total: int
    owner_subset_enumeration_complete: bool


@dataclass(frozen=True)
class _OwnerSubsetPlacement:
    owners: tuple[OwnerInput, ...]
    records_by_owner: tuple[tuple[ExpertRecord, ...], ...]
    strategy: str


def _owner_link_values(
    scenario: SimulationInput,
    owner: OwnerInput,
    requested_round_trip_ms: float,
) -> tuple[float, float, float, float]:
    owner_base_rtt = (
        owner.round_trip_ms
        if owner.round_trip_ms is not None
        else scenario.network.round_trip_ms
    )
    effective_rtt = max(
        0.0,
        requested_round_trip_ms
        + owner_base_rtt
        - scenario.network.round_trip_ms,
    )
    base_bandwidth = (
        owner.bandwidth_mbps
        if owner.bandwidth_mbps is not None
        else scenario.network.bandwidth_mbps
    )
    egress = (
        owner.egress_bandwidth_mbps
        if owner.egress_bandwidth_mbps is not None
        else (
            base_bandwidth
            if owner.bandwidth_mbps is not None
            else scenario.network.egress_bandwidth_mbps
        )
    )
    ingress = (
        owner.ingress_bandwidth_mbps
        if owner.ingress_bandwidth_mbps is not None
        else (
            base_bandwidth
            if owner.bandwidth_mbps is not None
            else scenario.network.ingress_bandwidth_mbps
        )
    )
    return effective_rtt, base_bandwidth, egress, ingress


def _owner_speed_score(scenario: SimulationInput, owner: OwnerInput) -> float:
    rtt, _base, egress, ingress = _owner_link_values(
        scenario,
        owner,
        scenario.network.round_trip_ms,
    )
    activation_bytes = (
        scenario.model.activation_elements
        * scenario.model.activation_dtype_bytes
    )
    return (
        rtt
        + owner.expert_compute_ms_per_token
        + activation_bytes / (egress * 125.0)
        + activation_bytes / (ingress * 125.0)
    )


def _expert_records(scenario: SimulationInput) -> tuple[ExpertRecord, ...]:
    return tuple(
        ExpertRecord(
            ExpertKey(layer, expert),
            scenario.model.expert_bytes,
            f"synthetic:{scenario.scenario_id}:layer-{layer}:expert-{expert}",
        )
        for layer in range(scenario.model.sparse_layers)
        for expert in range(scenario.model.experts_per_layer)
    )


def _place_all_experts(
    scenario: SimulationInput,
    records: Sequence[ExpertRecord],
) -> _Placement:
    # Dynamic input+output+reduction memory is placement-dependent. Reserving
    # top-k on every owner would reject valid split routes, so each candidate is
    # charged only for its peak observed coactivation count. Operator workspace
    # remains explicitly uncalibrated in this synthetic input schema.
    dynamic_bytes_per_assignment = (
        3
        * scenario.model.activation_elements
        * scenario.model.activation_dtype_bytes
    )
    # Capacity first preserves the minimum-owner packing bound. Among equal
    # domestic capacities, prefer the faster measured GPU/link instead of an
    # arbitrary lexical id.
    ordered = tuple(
        sorted(
            scenario.owners,
            key=lambda owner: (
                -owner.expert_capacity_bytes,
                _owner_speed_score(scenario, owner),
                owner.node_id,
            ),
        )
    )
    total_bytes = sum(record.byte_size for record in records)
    cumulative = 0
    lower_bound = 0
    for owner in ordered:
        cumulative += owner.expert_capacity_bytes
        lower_bound += 1
        if cumulative >= total_bytes:
            break
    if cumulative < total_bytes:
        raise ValueError("owner VRAM cannot hold every declared resident expert")

    # Turn the simulator's actual top-k observations into a deterministic
    # disjoint grouping.  Prefer the most frequent coactivation hyperedges;
    # any expert not captured by one becomes a singleton.  This preserves one
    # exact replica per key while avoiding the old round-robin anti-pattern
    # that systematically placed adjacent top-k experts on different owners.
    records_by_key = {record.key: record for record in records}
    placement_groups: list[tuple[ExpertRecord, ...]] = []
    for layer in range(scenario.model.sparse_layers):
        edge_frequency: dict[tuple[int, ...], int] = {}
        for token in range(scenario.wave.positions):
            edge = tuple(
                sorted(
                    (
                        layer + token * scenario.model.top_k + slot
                    )
                    % scenario.wave.routing_union_experts_per_layer
                    for slot in range(scenario.model.top_k)
                )
            )
            edge_frequency[edge] = edge_frequency.get(edge, 0) + 1
        grouped_experts: set[int] = set()
        for expert_ids, _frequency in sorted(
            edge_frequency.items(),
            key=lambda item: (-item[1], item[0]),
        ):
            if any(expert in grouped_experts for expert in expert_ids):
                continue
            group = tuple(
                records_by_key[ExpertKey(layer, expert)] for expert in expert_ids
            )
            placement_groups.append(group)
            grouped_experts.update(expert_ids)
        for expert in range(scenario.model.experts_per_layer):
            if expert not in grouped_experts:
                placement_groups.append(
                    (records_by_key[ExpertKey(layer, expert)],)
                )
    grouped_keys = tuple(
        record.key for group in placement_groups for record in group
    )
    if len(grouped_keys) != len(records_by_key) or set(grouped_keys) != set(
        records_by_key
    ):
        raise RuntimeError("co-routing placement groups lost or repeated an expert")

    observed_edges_by_layer = tuple(
        tuple(
            frozenset(
                ExpertKey(
                    layer,
                    (
                        layer + token * scenario.model.top_k + slot
                    )
                    % scenario.wave.routing_union_experts_per_layer,
                )
                for slot in range(scenario.model.top_k)
            )
            for token in range(scenario.wave.positions)
        )
        for layer in range(scenario.model.sparse_layers)
    )

    def peak_assignment_count(keys: set[ExpertKey]) -> int:
        return max(
            (
                sum(len(keys & edge) for edge in layer_edges)
                for layer_edges in observed_edges_by_layer
            ),
            default=0,
        )

    # The coactivation-first greedy is intentionally kept as the fast path: it
    # produces the established placement for the reference scenario.  It can,
    # however, paint itself into a corner on heterogeneous cards.  The bounded
    # exact repair below assigns individual experts and proves feasibility (or
    # infeasibility) for every state it fully explores.  A global bound keeps a
    # pathological input from turning this closed-schema simulator into an
    # unbounded bin-packing job.
    exact_key_limit = 64
    exact_state_limit = 200_000
    owner_subset_exact_limit = 256
    owner_subset_fallback_limit = 256
    exact_states_total = 0
    any_assignment_search_cut_off = False

    assignment_count_by_key: dict[ExpertKey, int] = {
        key: 0 for key in records_by_key
    }
    coactivation_by_pair: dict[tuple[ExpertKey, ExpertKey], int] = {}
    for layer_edges in observed_edges_by_layer:
        for edge in layer_edges:
            ordered_edge = tuple(sorted(edge))
            for key in ordered_edge:
                assignment_count_by_key[key] += 1
            for left_index, left in enumerate(ordered_edge):
                for right in ordered_edge[left_index + 1 :]:
                    pair = (left, right)
                    coactivation_by_pair[pair] = (
                        coactivation_by_pair.get(pair, 0) + 1
                    )

    exact_record_order = tuple(
        record for group in placement_groups for record in group
    )

    def exact_repair(
        selected: Sequence[OwnerInput],
    ) -> tuple[str, tuple[tuple[ExpertRecord, ...], ...] | None, int]:
        """Return ``feasible``, ``infeasible`` or ``cutoff`` plus assignments."""

        nonlocal exact_states_total
        # This necessary lower bound is cheap and works at any model size. At
        # least one resident expert is active, so some owner must reserve the
        # smallest observed per-expert route in addition to all static weights.
        # It proves the reference six-owner attempt impossible (those six GPUs
        # are already exactly full of weights) without invoking backtracking.
        minimum_route_bytes = (
            min(assignment_count_by_key.values(), default=0)
            * dynamic_bytes_per_assignment
        )
        if sum(owner.expert_capacity_bytes for owner in selected) < (
            total_bytes + minimum_route_bytes
        ):
            return "infeasible", None, 0
        if len(exact_record_order) > exact_key_limit:
            return "cutoff", None, 0
        remaining_state_budget = exact_state_limit - exact_states_total
        if remaining_state_budget <= 0:
            return "cutoff", None, 0

        owner_count = len(selected)
        owner_static = [0] * owner_count
        owner_layer_assignments = [
            [0] * scenario.model.sparse_layers for _ in selected
        ]
        owner_keys: list[set[ExpertKey]] = [set() for _ in selected]
        owner_records: list[list[ExpertRecord]] = [[] for _ in selected]
        dead_states: set[tuple[int, tuple[int, ...]]] = set()
        owner_masks = [0] * owner_count
        states = 0
        cut_off = False

        def projected_bytes(owner_index: int, record: ExpertRecord) -> int:
            layer = record.key.layer
            projected_layer_peak = max(
                max(owner_layer_assignments[owner_index], default=0),
                owner_layer_assignments[owner_index][layer]
                + assignment_count_by_key[record.key],
            )
            return (
                owner_static[owner_index]
                + record.byte_size
                + projected_layer_peak * dynamic_bytes_per_assignment
            )

        def affinity(owner_index: int, key: ExpertKey) -> int:
            score = 0
            for resident_key in owner_keys[owner_index]:
                pair = tuple(sorted((key, resident_key)))
                score += coactivation_by_pair.get(pair, 0)
            return score

        def search(record_index: int) -> bool:
            nonlocal states, cut_off
            if states >= remaining_state_budget:
                cut_off = True
                return False
            states += 1
            if record_index == len(exact_record_order):
                return True

            state = (record_index, tuple(owner_masks))
            if state in dead_states:
                return False

            record = exact_record_order[record_index]
            candidates = [
                owner_index
                for owner_index, owner in enumerate(selected)
                if projected_bytes(owner_index, record)
                <= owner.expert_capacity_bytes
            ]
            candidates.sort(
                key=lambda owner_index: (
                    -affinity(owner_index, record.key),
                    projected_bytes(owner_index, record)
                    / selected[owner_index].expert_capacity_bytes,
                    _owner_speed_score(scenario, selected[owner_index]),
                    owner_static[owner_index],
                    selected[owner_index].node_id,
                )
            )

            # Owners with identical capacity/performance/current contents are
            # symmetric for feasibility. Exploring only one avoids factorial
            # duplication without changing the result.
            seen_owner_states: set[tuple[object, ...]] = set()
            for owner_index in candidates:
                owner = selected[owner_index]
                owner_signature = (
                    owner.expert_capacity_bytes,
                    _owner_speed_score(scenario, owner),
                    owner_static[owner_index],
                    tuple(owner_layer_assignments[owner_index]),
                    owner_masks[owner_index],
                )
                if owner_signature in seen_owner_states:
                    continue
                seen_owner_states.add(owner_signature)

                layer = record.key.layer
                assignment_delta = assignment_count_by_key[record.key]
                owner_static[owner_index] += record.byte_size
                owner_layer_assignments[owner_index][layer] += assignment_delta
                owner_keys[owner_index].add(record.key)
                owner_records[owner_index].append(record)
                owner_masks[owner_index] |= 1 << record_index

                if search(record_index + 1):
                    return True

                owner_masks[owner_index] &= ~(1 << record_index)
                owner_records[owner_index].pop()
                owner_keys[owner_index].remove(record.key)
                owner_layer_assignments[owner_index][layer] -= assignment_delta
                owner_static[owner_index] -= record.byte_size
                if cut_off:
                    return False

            dead_states.add(state)
            return False

        feasible = search(0)
        exact_states_total += states
        if feasible:
            return (
                "feasible",
                tuple(tuple(items) for items in owner_records),
                states,
            )
        return ("cutoff" if cut_off else "infeasible"), None, states

    ordered_index = {
        owner.node_id: index for index, owner in enumerate(ordered)
    }

    def normalize_subset(
        selected: Sequence[OwnerInput],
    ) -> tuple[OwnerInput, ...]:
        return tuple(sorted(selected, key=lambda owner: ordered_index[owner.node_id]))

    def greedy_placement(
        selected: Sequence[OwnerInput],
    ) -> _OwnerSubsetPlacement | None:
        used = {owner.node_id: 0 for owner in selected}
        placed_keys = {owner.node_id: set() for owner in selected}
        records_by_owner = {owner.node_id: [] for owner in selected}
        failed = False

        def can_place(owner: OwnerInput, chunk: Sequence[ExpertRecord]) -> bool:
            projected_keys = placed_keys[owner.node_id] | {
                record.key for record in chunk
            }
            projected_static = used[owner.node_id] + sum(
                record.byte_size for record in chunk
            )
            projected_dynamic = (
                peak_assignment_count(projected_keys)
                * dynamic_bytes_per_assignment
            )
            return (
                projected_static + projected_dynamic
                <= owner.expert_capacity_bytes
            )

        for group in placement_groups:
            candidates = tuple(
                owner
                for owner in selected
                if can_place(owner, group)
            )
            chunks: tuple[tuple[ExpertRecord, ...], ...]
            if candidates:
                chunks = (group,)
            else:
                # Fragmentation may make a complete coactivation group
                # impossible even though every replica still fits. Split only
                # that group and preserve exact coverage.
                chunks = tuple((record,) for record in group)
            for chunk in chunks:
                chunk_bytes = sum(record.byte_size for record in chunk)
                chunk_candidates = tuple(
                    owner
                    for owner in selected
                    if can_place(owner, chunk)
                )
                if not chunk_candidates:
                    failed = True
                    break
                owner = min(
                    chunk_candidates,
                    key=lambda item: (
                        used[item.node_id] / item.expert_capacity_bytes,
                        _owner_speed_score(scenario, item),
                        used[item.node_id],
                        item.node_id,
                    ),
                )
                used[owner.node_id] += chunk_bytes
                placed_keys[owner.node_id].update(
                    record.key for record in chunk
                )
                records_by_owner[owner.node_id].extend(chunk)
            if failed:
                break
        if failed:
            return None
        return _OwnerSubsetPlacement(
            owners=tuple(selected),
            records_by_owner=tuple(
                tuple(records_by_owner[owner.node_id]) for owner in selected
            ),
            strategy="trace-coactivation-disjoint-greedy/1",
        )

    attempt_cache: dict[
        tuple[str, ...],
        tuple[str, _OwnerSubsetPlacement | None],
    ] = {}

    def attempt_subset(
        selected_value: Sequence[OwnerInput],
    ) -> tuple[str, _OwnerSubsetPlacement | None]:
        selected = normalize_subset(selected_value)
        signature = tuple(owner.node_id for owner in selected)
        cached = attempt_cache.get(signature)
        if cached is not None:
            return cached

        greedy = greedy_placement(selected)
        if greedy is not None:
            result = ("feasible", greedy)
            attempt_cache[signature] = result
            return result

        outcome, repaired, _states = exact_repair(selected)
        if outcome != "feasible":
            result = (outcome, None)
            attempt_cache[signature] = result
            return result
        if repaired is None:
            raise RuntimeError("exact placement repair returned no assignment")
        result = (
            "feasible",
            _OwnerSubsetPlacement(
                owners=selected,
                records_by_owner=repaired,
                strategy="trace-coactivation-bounded-exact-repair/1",
            ),
        )
        attempt_cache[signature] = result
        return result

    def projected_subset_score(
        placement: _OwnerSubsetPlacement,
    ) -> tuple[object, ...]:
        """Rank feasible subsets by the resident remote critical path.

        This deliberately mirrors the simulator's one owner batch per layer:
        owners advance in parallel, while layers remain sequential.  It is a
        placement ranking, not a replacement for the full mesh projection.
        """

        activation_bytes = (
            scenario.model.activation_elements
            * scenario.model.activation_dtype_bytes
        )
        layer_times: list[float] = []
        active_owner_ids: set[str] = set()
        for layer, layer_edges in enumerate(observed_edges_by_layer):
            owner_times: list[float] = []
            for owner, owner_records in zip(
                placement.owners,
                placement.records_by_owner,
            ):
                layer_keys = {
                    record.key for record in owner_records if record.key.layer == layer
                }
                assignment_total = sum(
                    assignment_count_by_key[key] for key in layer_keys
                )
                if assignment_total <= 0:
                    continue
                active_owner_ids.add(owner.node_id)
                unique_positions = sum(
                    1 for edge in layer_edges if layer_keys & edge
                )
                v1_request_bytes = assignment_total * activation_bytes
                coalesced_request_bytes = unique_positions * activation_bytes
                coalesced_metadata_bytes = (
                    assignment_total * owner.row_index_bytes_per_assignment
                )
                use_coalesced = (
                    owner.supports_exact_input_coalescing
                    and coalesced_request_bytes + coalesced_metadata_bytes
                    < v1_request_bytes
                )
                request_bytes = (
                    coalesced_request_bytes + coalesced_metadata_bytes
                    if use_coalesced
                    else v1_request_bytes
                )
                response_bytes = assignment_total * activation_bytes
                rtt, _base, egress, ingress = _owner_link_values(
                    scenario,
                    owner,
                    scenario.network.round_trip_ms,
                )
                owner_times.append(
                    rtt
                    + assignment_total * owner.expert_compute_ms_per_token
                    + request_bytes / (egress * 125.0)
                    + response_bytes / (ingress * 125.0)
                )
            layer_times.append(max(owner_times, default=0.0))

        owner_ids = tuple(sorted(owner.node_id for owner in placement.owners))
        assignment_signature = tuple(
            sorted(
                (
                    record.key.layer,
                    record.key.expert,
                    owner.node_id,
                )
                for owner, owner_records in zip(
                    placement.owners,
                    placement.records_by_owner,
                )
                for record in owner_records
            )
        )
        return (
            math.fsum(layer_times),
            math.fsum(
                _owner_speed_score(scenario, owner)
                for owner in placement.owners
                if owner.node_id in active_owner_ids
            ),
            owner_ids,
            assignment_signature,
        )

    def owner_subsets(
        owner_count: int,
    ) -> tuple[tuple[tuple[OwnerInput, ...], ...], int, bool, str]:
        subset_total = math.comb(len(ordered), owner_count)
        prefix = tuple(ordered[:owner_count])
        if subset_total <= owner_subset_exact_limit:
            return (
                tuple(combinations(ordered, owner_count)),
                subset_total,
                True,
                "exact-enumeration/1",
            )

        candidates: list[tuple[OwnerInput, ...]] = []
        signatures: set[tuple[str, ...]] = set()

        def add(selected_value: Sequence[OwnerInput]) -> None:
            if len(candidates) >= owner_subset_fallback_limit:
                return
            selected = normalize_subset(selected_value)
            signature = tuple(owner.node_id for owner in selected)
            if len(selected) != owner_count or signature in signatures:
                return
            signatures.add(signature)
            candidates.append(selected)

        add(prefix)
        fastest = normalize_subset(
            sorted(
                ordered,
                key=lambda owner: (
                    _owner_speed_score(scenario, owner),
                    -owner.expert_capacity_bytes,
                    owner.node_id,
                ),
            )[:owner_count]
        )
        add(fastest)
        for base in (prefix, fastest):
            base_ids = {owner.node_id for owner in base}
            incumbents = sorted(
                base,
                key=lambda owner: (
                    -_owner_speed_score(scenario, owner),
                    owner.expert_capacity_bytes,
                    owner.node_id,
                ),
            )
            outsiders = sorted(
                (owner for owner in ordered if owner.node_id not in base_ids),
                key=lambda owner: (
                    _owner_speed_score(scenario, owner),
                    -owner.expert_capacity_bytes,
                    owner.node_id,
                ),
            )
            for outsider in outsiders:
                for incumbent in incumbents:
                    add(
                        tuple(
                            owner
                            for owner in base
                            if owner.node_id != incumbent.node_id
                        )
                        + (outsider,)
                    )
                    if len(candidates) >= owner_subset_fallback_limit:
                        break
                if len(candidates) >= owner_subset_fallback_limit:
                    break
            if len(candidates) >= owner_subset_fallback_limit:
                break
        return (
            tuple(candidates),
            subset_total,
            False,
            "deterministic-capacity-speed-neighborhood/1",
        )

    unresolved_lower_count = False
    any_subset_search_limited = False
    for owner_count in range(lower_bound, len(ordered) + 1):
        subsets, subset_total, enumeration_complete, subset_strategy = (
            owner_subsets(owner_count)
        )
        prefix = tuple(ordered[:owner_count])
        prefix_signature = tuple(owner.node_id for owner in prefix)
        ordered_subsets = tuple(
            sorted(
                subsets,
                key=lambda selected: (
                    tuple(owner.node_id for owner in selected)
                    != prefix_signature,
                    math.fsum(
                        _owner_speed_score(scenario, owner) for owner in selected
                    ),
                    tuple(owner.node_id for owner in selected),
                ),
            )
        )
        evaluated: list[tuple[str, _OwnerSubsetPlacement | None]] = []
        prefix_outcome, prefix_placement = attempt_subset(prefix)
        evaluated.append((prefix_outcome, prefix_placement))
        if prefix_outcome == "infeasible":
            # The k largest capacities component-wise dominate every other
            # k-owner subset. Exact infeasibility of this prefix therefore
            # proves that no alternate subset of the same size can fit.
            continue

        for selected in ordered_subsets:
            if tuple(owner.node_id for owner in selected) == prefix_signature:
                continue
            evaluated.append(attempt_subset(selected))

        if any(outcome == "cutoff" for outcome, _placement in evaluated):
            any_assignment_search_cut_off = True
        feasible = tuple(
            placement
            for outcome, placement in evaluated
            if outcome == "feasible" and placement is not None
        )
        if not feasible:
            unresolved_lower_count = True
            if not enumeration_complete:
                any_subset_search_limited = True
            continue

        chosen = min(feasible, key=projected_subset_score)
        if not enumeration_complete:
            any_subset_search_limited = True
        by_owner_id = {
            owner.node_id: owner_records
            for owner, owner_records in zip(
                chosen.owners,
                chosen.records_by_owner,
            )
        }
        output_owners = tuple(sorted(chosen.owners, key=lambda owner: owner.node_id))
        replicas = tuple(
            sorted(
                ResidentExpertReplica(
                    record.key,
                    owner.node_id,
                    record.content_id,
                )
                for owner in output_owners
                for record in by_owner_id[owner.node_id]
            )
        )
        resident = tuple(
            (
                owner.node_id,
                sum(
                    record.byte_size for record in by_owner_id[owner.node_id]
                ),
            )
            for owner in output_owners
        )
        return _Placement(
            owners=output_owners,
            replicas=replicas,
            resident_bytes_by_owner=resident,
            capacity_lower_bound_owners=lower_bound,
            strategy=chosen.strategy,
            minimum_owner_count_proven=not unresolved_lower_count,
            search_states_evaluated=exact_states_total,
            search_limit_reached=(
                any_assignment_search_cut_off or any_subset_search_limited
            ),
            owner_subset_strategy=subset_strategy,
            owner_subsets_evaluated=len(evaluated),
            owner_subset_total=subset_total,
            owner_subset_enumeration_complete=enumeration_complete,
        )

    if any_assignment_search_cut_off or unresolved_lower_count:
        raise ValueError(
            "deterministic expert placement did not find a feasible assignment; "
            "bounded exact repair was not completed "
            f"(limits: {exact_key_limit} experts and {exact_state_limit} states)"
        )
    raise ValueError(
        "deterministic expert placement proved that owner VRAM budgets cannot "
        "hold the declared experts plus their observed activation routes"
    )


def _synthetic_routing(
    scenario: SimulationInput,
    layer: int,
) -> AuthoritativeRouting:
    ids = [
        [
            (
                layer
                + token * scenario.model.top_k
                + slot
            )
            % scenario.wave.routing_union_experts_per_layer
            for slot in range(scenario.model.top_k)
        ]
        for token in range(scenario.wave.positions)
    ]
    routing = AuthoritativeRouting(
        torch.tensor(ids, dtype=torch.long),
        torch.full(
            (scenario.wave.positions, scenario.model.top_k),
            1.0 / scenario.model.top_k,
            dtype=torch.float64,
        ),
    )
    union = {int(value) for row in ids for value in row}
    if len(union) != scenario.wave.routing_union_experts_per_layer:
        raise RuntimeError("synthetic routing did not cover the declared expert union")
    return routing


def _simulate_once(
    scenario: SimulationInput,
    placement: _Placement,
    records: Sequence[ExpertRecord],
    *,
    round_trip_ms: float,
) -> dict[str, Any]:
    coordinator = scenario.coordinator
    nodes = [
        MeshNodeProfile(
            coordinator.node_id,
            coordinator.usable_vram_bytes,
            coordinator.reserved_vram_bytes,
            coordinator.expert_compute_ms_per_token,
            coordinator.ram_to_gpu_gbytes_per_second,
        )
    ]
    nodes.extend(
        MeshNodeProfile(
            owner.node_id,
            owner.usable_vram_bytes,
            owner.reserved_vram_bytes,
            owner.expert_compute_ms_per_token,
        )
        for owner in placement.owners
    )
    links = tuple(
        MeshLinkProfile(
            coordinator.node_id,
            owner.node_id,
            _owner_link_values(scenario, owner, round_trip_ms)[0],
            _owner_link_values(scenario, owner, round_trip_ms)[1],
            egress_bandwidth_mbps=(
                _owner_link_values(scenario, owner, round_trip_ms)[2]
            ),
            ingress_bandwidth_mbps=(
                _owner_link_values(scenario, owner, round_trip_ms)[3]
            ),
        )
        for owner in placement.owners
    )
    mesh = ResidentExpertMesh(
        coordinator_id=coordinator.node_id,
        experts=records,
        nodes=tuple(nodes),
        links=links,
        local_ram_keys=tuple(record.key for record in records),
        local_gpu_keys=(),
        replicas=placement.replicas,
        local_weight_buffer_bytes=coordinator.weight_buffer_bytes,
        activation_bytes_per_token=(
            scenario.model.activation_elements
            * scenario.model.activation_dtype_bytes
        ),
    )
    layer_plans = tuple(
        mesh.plan_layer(
            layer,
            _synthetic_routing(scenario, layer),
            v1_only_node_ids=tuple(
                owner.node_id
                for owner in placement.owners
                if not owner.supports_exact_input_coalescing
            ),
            coalesced_row_index_bytes_by_node={
                owner.node_id: owner.row_index_bytes_per_assignment
                for owner in placement.owners
            },
        )
        for layer in range(scenario.model.sparse_layers)
    )
    projection = project_mesh_wave(
        layer_plans,
        committed_tokens=scenario.wave.committed_tokens,
    )
    remote_dispatches = tuple(
        dispatch
        for plan in layer_plans
        for dispatch in plan.dispatches
        if dispatch.path == "remote-resident"
    )
    local_ram_dispatches = tuple(
        dispatch
        for plan in layer_plans
        for dispatch in plan.dispatches
        if dispatch.path == "local-ram"
    )
    active_owner_ids = tuple(
        sorted({dispatch.owner_id for dispatch in remote_dispatches})
    )
    tokens_per_second = projection.tokens_per_second
    baseline = scenario.macro_wave_cold_tokens_per_second
    peak_vram_bytes_by_node: dict[str, int] = {}
    for plan in layer_plans:
        for node_id, byte_size in plan.owner_peak_vram_bytes:
            peak_vram_bytes_by_node[node_id] = max(
                peak_vram_bytes_by_node.get(node_id, 0),
                byte_size,
            )
    max_inflight_owner_rpcs = {
        plan.max_inflight_owner_rpcs for plan in layer_plans
    }
    if len(max_inflight_owner_rpcs) != 1:
        raise RuntimeError("layer plans disagree on max inflight owner RPCs")
    scheduling_by_layer = [
        {
            "layer": plan.layer,
            "perOwnerLowerBoundMs": plan.per_owner_lower_bound_ms,
            "ownerScheduledMakespanMs": plan.owner_scheduled_makespan_ms,
            "coordinatorNicLowerBoundMs": plan.coordinator_nic_lower_bound_ms,
            "coordinatorOverheadMs": plan.coordinator_overhead_ms,
            "exposedMs": plan.exposed_ms,
            "maxInflightOwnerRpcs": plan.max_inflight_owner_rpcs,
            "coordinationCalibrationRequired": (
                plan.coordination_calibration_required
            ),
        }
        for plan in layer_plans
    ]
    return {
        "roundTripMs": round_trip_ms,
        "exposedMsPerWave": projection.exposed_ms_per_wave,
        "exposedMsPerCommittedToken": (
            projection.exposed_ms_per_wave / scenario.wave.committed_tokens
        ),
        "tokensPerSecondPerSequence": tokens_per_second,
        "activationRoundTripBytesPerWave": (
            projection.activation_round_trip_bytes_per_wave
        ),
        "activationRequestBytesPerWave": (
            projection.activation_request_bytes_per_wave
        ),
        "activationResponseBytesPerWave": (
            projection.activation_response_bytes_per_wave
        ),
        "routeMetadataBytesPerWave": projection.route_metadata_bytes_per_wave,
        "transportPayloadBytesPerWave": (
            projection.transport_payload_bytes_per_wave
        ),
        "coalescedOwnersByLayer": [
            list(plan.coalesced_owner_ids) for plan in layer_plans
        ],
        "assignmentStrategyByLayer": [
            {
                "strategy": plan.assignment_strategy,
                "optimalityProven": plan.assignment_optimality_proven,
                "statesEvaluated": plan.assignment_states_evaluated,
            }
            for plan in layer_plans
        ],
        "transportAssumption": {
            "source": "declared-owner-profile-not-negotiated-runtime",
            "owners": {
                owner.node_id: {
                    "exactInputCoalescing": (
                        owner.supports_exact_input_coalescing
                    ),
                    "rowIndexBytesPerAssignment": (
                        owner.row_index_bytes_per_assignment
                    ),
                    "roundTripMs": _owner_link_values(
                        scenario,
                        owner,
                        round_trip_ms,
                    )[0],
                    "egressBandwidthMbps": _owner_link_values(
                        scenario,
                        owner,
                        round_trip_ms,
                    )[2],
                    "ingressBandwidthMbps": _owner_link_values(
                        scenario,
                        owner,
                        round_trip_ms,
                    )[3],
                }
                for owner in placement.owners
            },
        },
        "hostDeviceStagingBytesPerWave": (
            projection.host_device_staging_bytes_per_wave
        ),
        "weightLoadedBytesPerWave": projection.weight_loaded_bytes_per_wave,
        "weightAvoidedBytesPerWave": projection.weight_avoided_bytes_per_wave,
        "transportCalibrationRequired": (
            projection.transport_calibration_required
        ),
        "workspaceCalibrationRequired": (
            projection.workspace_calibration_required
        ),
        "maxInflightOwnerRpcs": next(iter(max_inflight_owner_rpcs)),
        "ownerScheduledMakespanMsPerWave": math.fsum(
            plan.owner_scheduled_makespan_ms for plan in layer_plans
        ),
        "coordinatorNicLowerBoundMsPerWave": sum(
            plan.coordinator_nic_lower_bound_ms for plan in layer_plans
        ),
        "coordinatorOverheadMsPerWave": math.fsum(
            plan.coordinator_overhead_ms for plan in layer_plans
        ),
        "coordinationCalibrationRequired": any(
            plan.coordination_calibration_required for plan in layer_plans
        ),
        "schedulingByLayer": scheduling_by_layer,
        "peakVramBytesByNode": dict(sorted(peak_vram_bytes_by_node.items())),
        "remoteResidentDispatches": len(remote_dispatches),
        "localRamDispatches": len(local_ram_dispatches),
        "activeRemoteOwnerIds": list(active_owner_ids),
        "ratioVsMacroWaveColdPerSequence": tokens_per_second / baseline,
    }


def simulate_at_rtt(value: object, round_trip_ms: float) -> dict[str, Any]:
    scenario = (
        value if isinstance(value, SimulationInput) else SimulationInput.from_document(value)
    )
    rtt = _number(round_trip_ms, "round_trip_ms")
    records = _expert_records(scenario)
    placement = _place_all_experts(scenario, records)
    return _simulate_once(scenario, placement, records, round_trip_ms=rtt)


def simulate_document(value: object) -> dict[str, Any]:
    scenario = SimulationInput.from_document(value)
    records = _expert_records(scenario)
    placement = _place_all_experts(scenario, records)
    primary = _simulate_once(
        scenario,
        placement,
        records,
        round_trip_ms=scenario.network.round_trip_ms,
    )
    sensitivity = [
        _simulate_once(
            scenario,
            placement,
            records,
            round_trip_ms=round_trip_ms,
        )
        for round_trip_ms in scenario.sensitivity_rtt_ms
    ]
    canonical = json.dumps(
        scenario.canonical_document(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    total_experts = scenario.model.sparse_layers * scenario.model.experts_per_layer
    return {
        "schema": REPORT_SCHEMA,
        "scenarioId": scenario.scenario_id,
        "inputSha256": hashlib.sha256(canonical).hexdigest(),
        "model": {
            "sparseLayers": scenario.model.sparse_layers,
            "expertsPerLayer": scenario.model.experts_per_layer,
            "totalExperts": total_experts,
            "expertBytes": scenario.model.expert_bytes,
            "totalResidentExpertBytes": total_experts * scenario.model.expert_bytes,
            "activationBytesPerPosition": (
                scenario.model.activation_elements
                * scenario.model.activation_dtype_bytes
            ),
            "topK": scenario.model.top_k,
        },
        "wave": {
            "positions": scenario.wave.positions,
            "positionsPerSequence": scenario.wave.positions_per_sequence,
            "concurrentSequences": scenario.wave.concurrent_sequences,
            "committedTokens": scenario.wave.committed_tokens,
            "routingUnionExpertsPerLayer": (
                scenario.wave.routing_union_experts_per_layer
            ),
            "routingKind": "deterministic-synthetic-union-coverage",
            "routingCoverageVerified": True,
            "throughputNormalization": "per-sequence-committed-tokens",
        },
        "placement": {
            "strategy": placement.strategy,
            "minimumOwnerCountProven": placement.minimum_owner_count_proven,
            "searchStatesEvaluated": placement.search_states_evaluated,
            "searchLimitReached": placement.search_limit_reached,
            "ownerSubsetStrategy": placement.owner_subset_strategy,
            "ownerSubsetsEvaluated": placement.owner_subsets_evaluated,
            "ownerSubsetTotal": placement.owner_subset_total,
            "ownerSubsetEnumerationComplete": (
                placement.owner_subset_enumeration_complete
            ),
            "candidateOwnerCount": len(scenario.owners),
            "capacityLowerBoundOwners": placement.capacity_lower_bound_owners,
            "ownersRequired": len(placement.owners),
            "ownerIds": [owner.node_id for owner in placement.owners],
            "residentReplicaCount": len(placement.replicas),
            "residentBytesByOwner": dict(placement.resident_bytes_by_owner),
        },
        "primary": primary,
        "sensitivity": sensitivity,
        "comparison": {
            "macroWaveColdTokensPerSecondPerSequence": (
                scenario.macro_wave_cold_tokens_per_second
            ),
            "scope": "partial-sparse-expert-ceiling-only",
        },
        "limitations": {
            "includes": [
                "sparse-expert-compute",
                "resident-mesh-rtt",
                "activation-transfer",
                "activation-input-output-vram-reserve",
                "local-ram-expert-load-fallback",
            ],
            "excludes": [
                "attention",
                "router-compute",
                "shared-and-dense-mlp",
                "normalization",
                "kv-cache",
                "serialization-and-queueing",
                "host-device-staging-time-without-calibration",
                "operator-workspace-without-calibration",
                "stage-to-stage-wan",
                "failures-and-retries",
            ],
            "isGlmPrediction": False,
            "isPartialUpperBound": True,
        },
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    primary = _record(report["primary"], "report.primary")
    placement = _record(report["placement"], "report.placement")
    model = _record(report["model"], "report.model")
    wave = _record(report["wave"], "report.wave")
    comparison = _record(report["comparison"], "report.comparison")
    lines = [
        f"# Resident Expert Mesh — {report['scenarioId']}",
        "",
        "> Synthetic partial ceiling for the sparse-expert segment. This is not a "
        "GLM prediction: it excludes attention, dense/shared MLP, KV, queues, and "
        "inter-stage WAN.",
        "",
        "## Configuration and placement",
        "",
        f"- Sparse layers: **{model['sparseLayers']}**",
        f"- Total resident experts: **{model['totalExperts']}**",
        f"- Candidate/required owners: **{placement['candidateOwnerCount']} / "
        f"{placement['ownersRequired']}**",
        f"- Physical positions/wave: **{wave['positions']}** = "
        f"**{wave['positionsPerSequence']}** × **{wave['concurrentSequences']} chats**",
        f"- Committed tokens used for per-sequence tok/s: "
        f"**{wave['committedTokens']}**",
        f"- Synthetic union covered per layer: **{wave['routingUnionExpertsPerLayer']}**",
        "",
        "## Primary result",
        "",
        "| RTT | ms/wave | ms/committed token | partial tok/s/sequence | "
        "request / response / row-map | avoided weights | loaded weights | "
        "active remote owners |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        _markdown_result_row(primary),
        "",
        "## Scheduling and coordination",
        "",
        f"- Maximum concurrent owner RPCs: **{primary['maxInflightOwnerRpcs']}**",
        "- Scheduled owner makespan/wave: **"
        f"{_fmt(primary['ownerScheduledMakespanMsPerWave'])} ms**",
        "- Modeled coordinator overhead/wave: **"
        f"{_fmt(primary['coordinatorOverheadMsPerWave'])} ms**",
        "- Coordination calibration required: **"
        f"{'yes' if primary['coordinationCalibrationRequired'] else 'no'}**",
        "",
        "## Actual simulator sensitivity",
        "",
        "| RTT | ms/wave | ms/token | partial tok/s/sequence | "
        "× versus cold MacroWave | "
        "avoided weights | loaded weights |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in report["sensitivity"]:
        item = _record(result, "report.sensitivity")
        lines.append(
            f"| {_fmt(item['roundTripMs'])} ms | "
            f"{_fmt(item['exposedMsPerWave'])} | "
            f"{_fmt(item['exposedMsPerCommittedToken'])} | "
            f"{_fmt(item['tokensPerSecondPerSequence'])} | "
            f"{_fmt(item['ratioVsMacroWaveColdPerSequence'])}× | "
            f"{_bytes(int(item['weightAvoidedBytesPerWave']))} | "
            f"{_bytes(int(item['weightLoadedBytesPerWave']))} |"
        )
    lines.extend(
        [
            "",
            "## Honest comparison",
            "",
            "The scenario's cold MacroWave reference is **"
            f"{comparison['macroWaveColdTokensPerSecondPerSequence']} tok/s per "
            "sequence**. The ratio shown above compares that complete synthetic-model "
            "result with a ceiling that times only sparse experts; it must not be "
            "interpreted as final speedup or expected GLM performance.",
            "",
            "The mesh avoids copying weights when RTT is low. As RTT grows, the "
            "scheduler falls back exactly to local RAM and loaded weights increase. "
            "Layers remain sequential, so per-layer WAN cost does not disappear.",
            "",
            "Placement groups co-activated top-k sets first. When multiple experts "
            "owned by the same node reuse positions, the request sends each row once "
            "and the row map preserves correspondence; the response remains separate "
            "per expert to preserve canonical reduction.",
            "",
            "This input does not calibrate host<->device staging, aggregate NIC, or "
            "operator workspace. The JSON reports staging bytes and marks "
            "`transportCalibrationRequired` / `coordinationCalibrationRequired` / "
            "`workspaceCalibrationRequired`; "
            "while those flags remain active, tok/s is derived from a lower bound on "
            "time (an upper bound on performance), not a complete physical projection.",
            "",
        ]
    )
    return "\n".join(lines)


def _markdown_result_row(item: Mapping[str, Any]) -> str:
    return (
        f"| {_fmt(item['roundTripMs'])} ms | "
        f"{_fmt(item['exposedMsPerWave'])} | "
        f"{_fmt(item['exposedMsPerCommittedToken'])} | "
        f"{_fmt(item['tokensPerSecondPerSequence'])} | "
        f"{_bytes(int(item['activationRequestBytesPerWave']))} / "
        f"{_bytes(int(item['activationResponseBytesPerWave']))} / "
        f"{_bytes(int(item['routeMetadataBytesPerWave']))} | "
        f"{_bytes(int(item['weightAvoidedBytesPerWave']))} | "
        f"{_bytes(int(item['weightLoadedBytesPerWave']))} | "
        f"{len(item['activeRemoteOwnerIds'])} |"
    )


def _fmt(value: object) -> str:
    return f"{float(value):.2f}"


def _bytes(value: int) -> str:
    units = ((1024**3, "GiB"), (1024**2, "MiB"), (1024, "KiB"))
    for divisor, suffix in units:
        if value >= divisor:
            return f"{value / divisor:.2f} {suffix}"
    return f"{value} B"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate a sealed resident expert mesh without launching a runner."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    return parser.parse_args(argv)


def execute_cli(argv: Sequence[str], cwd: str | Path | None = None) -> str:
    args = parse_args(argv)
    base = Path.cwd() if cwd is None else Path(cwd)
    input_path = args.input if args.input.is_absolute() else base / args.input
    try:
        document = json.loads(input_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read simulation input {input_path}: {exc}") from exc
    report = simulate_document(document)
    if args.format == "markdown":
        return render_markdown(report)
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = list(argv) if argv is not None else None
    parsed = parse_args(args)
    rendered = execute_cli(
        [str(parsed.input), "--format", parsed.format],
        Path.cwd(),
    )
    print(rendered, end="" if rendered.endswith("\n") else "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
