from __future__ import annotations

import sys
import threading
import time
import unittest
from unittest import mock

import torch

from distributed_runtime.ram_expert_cache import ExpertKey, ExpertRecord
from distributed_runtime.resident_expert_mesh import (
    AuthoritativeRouting,
    ExplicitExpertDemand,
    ExpertResidentSlotUnavailableError,
    ExpertRouteUnavailableError,
    InMemoryExpertOwner,
    MeshLinkProfile,
    MeshNodeProfile,
    OwnerExpertBatchResult,
    ResidentExpertMesh,
    ResidentExpertReplica,
    project_mesh_wave,
)


MIB = 1024 * 1024


def records(*, layers: int = 1, experts: int = 4, byte_size: int = 10 * MIB):
    return tuple(
        ExpertRecord(
            ExpertKey(layer, expert),
            byte_size,
            f"sha256:l{layer}:e{expert}",
        )
        for layer in range(layers)
        for expert in range(experts)
    )


def node(
    node_id: str,
    *,
    budget: int = 64 * MIB,
    compute_ms: float = 0.2,
    ram_gbytes_s: float = 1.0,
    available: bool = True,
    workspace_bytes_per_token: int = 128,
    ingress_mbps: float = 0.0,
    egress_mbps: float = 0.0,
) -> MeshNodeProfile:
    return MeshNodeProfile(
        node_id=node_id,
        resident_vram_budget_bytes=budget,
        reserved_vram_bytes=0,
        expert_compute_ms_per_token=compute_ms,
        ram_to_device_gbytes_per_second=ram_gbytes_s,
        available=available,
        aggregate_ingress_mbps=ingress_mbps,
        aggregate_egress_mbps=egress_mbps,
        expert_workspace_bytes_per_token=workspace_bytes_per_token,
    )


def link(*, rtt_ms: float, to_node: str = "owner-1") -> MeshLinkProfile:
    return MeshLinkProfile(
        from_node="root",
        to_node=to_node,
        round_trip_ms=rtt_ms,
        bandwidth_mbps=10_000,
    )


class ResidentExpertMeshTests(unittest.TestCase):
    def test_input_coalescing_preserves_exact_reduction_and_directional_bytes(self) -> None:
        inventory = records(experts=2)
        mesh = ResidentExpertMesh(
            coordinator_id="root",
            experts=inventory,
            nodes=(node("root"), node("owner-1")),
            links=(link(rtt_ms=0.0),),
            local_ram_keys=(),
            local_gpu_keys=(),
            replicas=tuple(
                ResidentExpertReplica(record.key, "owner-1", record.content_id)
                for record in inventory
            ),
            local_weight_buffer_bytes=0,
            activation_bytes_per_token=12,
            require_local_ram_fallback=False,
        )
        owner = InMemoryExpertOwner(
            "owner-1",
            {
                inventory[0].key: (inventory[0].content_id, torch.eye(3)),
                inventory[1].key: (inventory[1].content_id, 2 * torch.eye(3)),
            },
        )
        hidden = torch.tensor(
            [[1.0, 2.0, 3.0], [-2.0, 0.5, 4.0]],
            dtype=torch.float32,
        )
        routing = AuthoritativeRouting(
            torch.tensor([[0, 1], [0, 1]], dtype=torch.long),
            torch.tensor([[0.25, 0.75], [0.6, 0.4]], dtype=torch.float32),
        )

        actual, plan = mesh.execute_layer(
            hidden,
            0,
            routing,
            {"owner-1": owner},
        )

        expected_scale = routing.expert_weights[:, 0] + (
            2 * routing.expert_weights[:, 1]
        )
        torch.testing.assert_close(
            actual,
            hidden * expected_scale.unsqueeze(-1),
            rtol=0,
            atol=0,
        )
        self.assertEqual(owner.batch_calls, 1)
        self.assertEqual(owner.coalesced_batch_calls, 1)
        self.assertEqual(plan.coalesced_owner_ids, ("owner-1",))
        self.assertEqual(plan.activation_request_bytes, 24)
        self.assertEqual(plan.activation_response_bytes, 48)
        # The in-memory owner advertises a zero-byte local row-map ABI; the
        # planner and executor must use that owner-specific value consistently.
        self.assertEqual(plan.route_metadata_bytes, 0)
        self.assertEqual(plan.activation_round_trip_bytes, 72)
        self.assertEqual(plan.transport_payload_bytes, 72)
        self.assertEqual(plan.host_device_staging_bytes, 144)

        class V1OnlyOwner:
            node_id = "owner-1"

            def __init__(self, inner: InMemoryExpertOwner) -> None:
                self.inner = inner

            def has_expert(self, key, content_id):
                return self.inner.has_expert(key, content_id)

            def is_expert_resident(self, key, content_id):
                return self.inner.is_expert_resident(key, content_id)

            def execute_batch(self, items):
                return self.inner.execute_batch(items)

        v1_owner = V1OnlyOwner(
            InMemoryExpertOwner(
                "owner-1",
                {
                    inventory[0].key: (inventory[0].content_id, torch.eye(3)),
                    inventory[1].key: (
                        inventory[1].content_id,
                        2 * torch.eye(3),
                    ),
                },
            )
        )
        fallback, fallback_plan = mesh.execute_layer(
            hidden,
            0,
            routing,
            {"owner-1": v1_owner},
        )
        torch.testing.assert_close(fallback, actual, rtol=0, atol=0)
        self.assertEqual(fallback_plan.coalesced_owner_ids, ())
        self.assertEqual(fallback_plan.activation_request_bytes, 48)
        self.assertEqual(fallback_plan.activation_response_bytes, 48)
        self.assertEqual(fallback_plan.route_metadata_bytes, 0)
        self.assertEqual(fallback_plan.transport_payload_bytes, 96)

    def test_two_owner_cpu_execution_matches_monolithic_target_top_k(self) -> None:
        inventory = records()
        weights = {
            ExpertKey(0, 0): torch.tensor(
                [[1.0, 0.0, 0.5], [0.0, 1.0, 0.0], [0.5, 0.0, 1.0]]
            ),
            ExpertKey(0, 1): torch.tensor(
                [[0.0, 1.0, 0.0], [1.0, 0.0, 0.5], [0.0, 0.5, 1.0]]
            ),
            ExpertKey(0, 2): torch.tensor(
                [[1.0, 0.5, 0.0], [0.0, 1.0, 1.0], [0.5, 0.0, 1.0]]
            ),
            ExpertKey(0, 3): torch.tensor(
                [[0.5, 0.0, 1.0], [1.0, 0.5, 0.0], [0.0, 1.0, 0.5]]
            ),
        }
        mesh = ResidentExpertMesh(
            coordinator_id="root",
            experts=inventory,
            nodes=(node("root"), node("owner-1")),
            links=(link(rtt_ms=0.05),),
            local_ram_keys=tuple(weights),
            local_gpu_keys=(ExpertKey(0, 0),),
            replicas=tuple(
                ResidentExpertReplica(record.key, "owner-1", record.content_id)
                for record in inventory
                if record.key.expert != 0
            ),
            local_weight_buffer_bytes=10 * MIB,
            activation_bytes_per_token=3 * 4,
        )
        owners = {
            "root": InMemoryExpertOwner(
                "root",
                {
                    key: (f"sha256:l{key.layer}:e{key.expert}", weight)
                    for key, weight in weights.items()
                },
            ),
            "owner-1": InMemoryExpertOwner(
                "owner-1",
                {
                    key: (f"sha256:l{key.layer}:e{key.expert}", weight)
                    for key, weight in weights.items()
                    if key.expert != 0
                },
            ),
        }
        hidden = torch.tensor(
            [[1.0, 2.0, 3.0], [2.0, -1.0, 0.5], [0.25, 1.5, -2.0]],
            dtype=torch.float32,
        )
        routing = AuthoritativeRouting(
            expert_ids=torch.tensor([[0, 1], [1, 2], [0, 3]], dtype=torch.long),
            expert_weights=torch.tensor(
                [[0.75, 0.25], [0.4, 0.6], [0.2, 0.8]],
                dtype=torch.float32,
            ),
        )

        expected = torch.zeros_like(hidden)
        for token_index in range(hidden.shape[0]):
            for slot_index in range(routing.top_k):
                expert = int(routing.expert_ids[token_index, slot_index].item())
                gate = routing.expert_weights[token_index, slot_index]
                expected[token_index] += gate * (
                    hidden[token_index] @ weights[ExpertKey(0, expert)].transpose(0, 1)
                )

        actual, plan = mesh.execute_layer(hidden, 0, routing, owners)

        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
        self.assertEqual(plan.authoritative_expert_ids, ((0, 1), (1, 2), (0, 3)))
        self.assertEqual(
            {dispatch.key.expert for dispatch in plan.dispatches},
            {0, 1, 2, 3},
        )
        self.assertEqual(
            {dispatch.owner_id for dispatch in plan.dispatches},
            {"root", "owner-1"},
        )
        self.assertEqual(
            {dispatch.path for dispatch in plan.dispatches if dispatch.key.expert != 0},
            {"remote-resident"},
        )
        self.assertEqual(plan.weight_loaded_bytes, 0)
        self.assertEqual(plan.weight_avoided_bytes, 4 * 10 * MIB)
        self.assertEqual(owners["root"].batch_calls, 1)
        self.assertEqual(owners["owner-1"].batch_calls, 1)

    def test_rtt_selects_remote_resident_or_exact_local_ram(self) -> None:
        inventory = records(experts=1)
        key = ExpertKey(0, 0)
        routing = AuthoritativeRouting(
            torch.tensor([[0]], dtype=torch.long),
            torch.tensor([[1.0]], dtype=torch.float32),
        )

        def make_mesh(rtt_ms: float) -> ResidentExpertMesh:
            return ResidentExpertMesh(
                coordinator_id="root",
                experts=inventory,
                nodes=(node("root"), node("owner-1")),
                links=(link(rtt_ms=rtt_ms),),
                local_ram_keys=(key,),
                local_gpu_keys=(),
                replicas=(
                    ResidentExpertReplica(key, "owner-1", inventory[0].content_id),
                ),
                local_weight_buffer_bytes=10 * MIB,
                activation_bytes_per_token=12,
            )

        low = make_mesh(0.1).plan_layer(0, routing)
        high = make_mesh(100.0).plan_layer(0, routing)

        self.assertEqual(low.dispatches[0].path, "remote-resident")
        self.assertEqual(low.activation_round_trip_bytes, 24)
        self.assertEqual(low.weight_avoided_bytes, 10 * MIB)
        self.assertEqual(high.dispatches[0].path, "local-ram")
        self.assertEqual(high.activation_round_trip_bytes, 0)
        self.assertEqual(high.weight_loaded_bytes, 10 * MIB)

    def test_unavailable_replica_falls_back_without_changing_expert_or_fails_closed(self) -> None:
        inventory = records(experts=1)
        key = inventory[0].key
        routing = AuthoritativeRouting(
            torch.tensor([[0]], dtype=torch.long),
            torch.tensor([[1.0]], dtype=torch.float32),
        )
        fallback = ResidentExpertMesh(
            coordinator_id="root",
            experts=inventory,
            nodes=(node("root"), node("owner-1")),
            links=(link(rtt_ms=0.1),),
            local_ram_keys=(key,),
            local_gpu_keys=(),
            replicas=(ResidentExpertReplica(key, "owner-1", inventory[0].content_id),),
            local_weight_buffer_bytes=10 * MIB,
            activation_bytes_per_token=12,
        )
        fallback_plan = fallback.plan_layer(
            0,
            routing,
            unavailable_node_ids=("owner-1",),
        )
        self.assertEqual(fallback_plan.dispatches[0].key, key)
        self.assertEqual(fallback_plan.dispatches[0].path, "local-ram")

        no_fallback = ResidentExpertMesh(
            coordinator_id="root",
            experts=inventory,
            nodes=(node("root"), node("owner-1")),
            links=(link(rtt_ms=0.1),),
            local_ram_keys=(),
            local_gpu_keys=(),
            replicas=(ResidentExpertReplica(key, "owner-1", inventory[0].content_id),),
            local_weight_buffer_bytes=0,
            activation_bytes_per_token=12,
            require_local_ram_fallback=False,
        )
        with self.assertRaisesRegex(ExpertRouteUnavailableError, "no exact"):
            no_fallback.plan_layer(
                0,
                routing,
                unavailable_node_ids=("owner-1",),
            )

    def test_hot_replica_greedy_is_deterministic_explicit_and_budgeted(self) -> None:
        inventory = records(experts=3, byte_size=60 * MIB)
        mesh = ResidentExpertMesh(
            coordinator_id="root",
            experts=inventory,
            nodes=(
                node("root", budget=64 * MIB, ram_gbytes_s=1.0),
                node("owner-a", budget=100 * MIB),
                node("owner-b", budget=100 * MIB),
            ),
            links=(
                link(rtt_ms=0.1, to_node="owner-a"),
                link(rtt_ms=0.1, to_node="owner-b"),
            ),
            local_ram_keys=tuple(record.key for record in inventory),
            local_gpu_keys=(),
            replicas=(),
            local_weight_buffer_bytes=60 * MIB,
            activation_bytes_per_token=12,
        )
        demand = (
            ExplicitExpertDemand(ExpertKey(0, 0), 0.95, 4.0),
            ExplicitExpertDemand(ExpertKey(0, 1), 0.8, 3.0),
            # Expert 2 is deliberately absent: no implicit hotness is allowed.
        )

        first = mesh.plan_hot_replicas(
            demand,
            candidate_node_ids=("owner-b", "owner-a"),
        )
        second = mesh.plan_hot_replicas(
            tuple(reversed(demand)),
            candidate_node_ids=("owner-a", "owner-b"),
        )

        self.assertEqual(first, second)
        self.assertEqual(
            {placement.key for placement in first.placements},
            {ExpertKey(0, 0), ExpertKey(0, 1)},
        )
        self.assertNotIn(ExpertKey(0, 2), {p.key for p in first.placements})
        self.assertGreater(first.estimated_exposed_ms_saved_per_wave, 0)
        usage = dict(first.used_vram_bytes_by_node)
        self.assertLessEqual(usage["owner-a"], 100 * MIB)
        self.assertLessEqual(usage["owner-b"], 100 * MIB)
        self.assertEqual(usage["owner-a"], 60 * MIB)
        self.assertEqual(usage["owner-b"], 60 * MIB)

    def test_owners_are_parallel_per_layer_but_layers_project_sequentially(self) -> None:
        inventory = records(layers=2, experts=2)
        mesh = ResidentExpertMesh(
            coordinator_id="root",
            experts=inventory,
            nodes=(node("root"), node("owner-1")),
            links=(link(rtt_ms=0.1),),
            local_ram_keys=tuple(record.key for record in inventory),
            local_gpu_keys=(ExpertKey(0, 0), ExpertKey(1, 0)),
            replicas=tuple(
                ResidentExpertReplica(record.key, "owner-1", record.content_id)
                for record in inventory
                if record.key.expert == 1
            ),
            local_weight_buffer_bytes=10 * MIB,
            activation_bytes_per_token=12,
        )
        routing = AuthoritativeRouting(
            torch.tensor([[0, 1], [0, 1]], dtype=torch.long),
            torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32),
        )
        layer_zero = mesh.plan_layer(0, routing)
        layer_one = mesh.plan_layer(1, routing)
        projection = project_mesh_wave(
            (layer_zero, layer_one),
            committed_tokens=1.5,
        )

        self.assertEqual(len(layer_zero.owner_exposed_ms), 2)
        self.assertAlmostEqual(
            layer_zero.exposed_ms,
            max(dict(layer_zero.owner_exposed_ms).values()),
        )
        self.assertAlmostEqual(
            projection.exposed_ms_per_wave,
            layer_zero.exposed_ms + layer_one.exposed_ms,
        )
        self.assertAlmostEqual(
            projection.tokens_per_second,
            1.5 * 1000 / projection.exposed_ms_per_wave,
        )
        self.assertEqual(
            projection.activation_round_trip_bytes_per_wave,
            layer_zero.activation_round_trip_bytes
            + layer_one.activation_round_trip_bytes,
        )

    def test_exact_assignment_avoids_greedy_owner_collision(self) -> None:
        inventory = records(experts=2, byte_size=1 * MIB)
        mesh = ResidentExpertMesh(
            coordinator_id="root",
            experts=inventory,
            nodes=(
                node("root"),
                node("owner-a", compute_ms=1.0),
                node("owner-b", compute_ms=1.5),
            ),
            links=(
                MeshLinkProfile("root", "owner-a", 0.0, 1_000_000.0),
                MeshLinkProfile("root", "owner-b", 0.0, 1_000_000.0),
            ),
            local_ram_keys=(),
            local_gpu_keys=(),
            replicas=(
                ResidentExpertReplica(
                    inventory[0].key,
                    "owner-a",
                    inventory[0].content_id,
                ),
                ResidentExpertReplica(
                    inventory[0].key,
                    "owner-b",
                    inventory[0].content_id,
                ),
                ResidentExpertReplica(
                    inventory[1].key,
                    "owner-a",
                    inventory[1].content_id,
                ),
            ),
            local_weight_buffer_bytes=0,
            activation_bytes_per_token=12,
            require_local_ram_fallback=False,
        )
        routing = AuthoritativeRouting(
            torch.tensor([[0, 1]], dtype=torch.long),
            torch.tensor([[0.5, 0.5]], dtype=torch.float32),
        )

        plan = mesh.plan_layer(0, routing)

        self.assertEqual(
            {dispatch.key.expert: dispatch.owner_id for dispatch in plan.dispatches},
            {0: "owner-b", 1: "owner-a"},
        )
        self.assertEqual(plan.assignment_strategy, "exact-enumeration")
        self.assertTrue(plan.assignment_optimality_proven)
        self.assertEqual(plan.assignment_states_evaluated, 2)
        self.assertLess(plan.exposed_ms, 2.0)

    def test_one_expert_rows_split_across_replicas_with_exact_reduction(self) -> None:
        inventory = records(experts=1, byte_size=10)
        key = inventory[0].key
        mesh = ResidentExpertMesh(
            coordinator_id="root",
            experts=inventory,
            nodes=(
                node("root", budget=48, workspace_bytes_per_token=1),
                node("owner-a", budget=35, workspace_bytes_per_token=1),
                node("owner-b", budget=35, workspace_bytes_per_token=1),
            ),
            links=(
                MeshLinkProfile("root", "owner-a", 0.0, 1_000_000.0),
                MeshLinkProfile("root", "owner-b", 0.0, 1_000_000.0),
            ),
            local_ram_keys=(),
            local_gpu_keys=(),
            replicas=(
                ResidentExpertReplica(key, "owner-a", inventory[0].content_id),
                ResidentExpertReplica(key, "owner-b", inventory[0].content_id),
            ),
            local_weight_buffer_bytes=0,
            activation_bytes_per_token=8,
            require_local_ram_fallback=False,
        )
        weight = 2 * torch.eye(2)
        owners = {
            owner_id: InMemoryExpertOwner(
                owner_id,
                {key: (inventory[0].content_id, weight)},
            )
            for owner_id in ("owner-a", "owner-b")
        }
        hidden = torch.tensor([[1.0, -2.0], [3.0, 0.5]])
        routing = AuthoritativeRouting(
            torch.tensor([[0], [0]], dtype=torch.long),
            torch.tensor([[0.25], [0.75]], dtype=torch.float32),
        )

        actual, plan = mesh.execute_layer(hidden, 0, routing, owners)

        expected = hidden * (2 * routing.expert_weights[:, 0]).unsqueeze(-1)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(len(plan.dispatches), 2)
        self.assertEqual(
            {dispatch.owner_id for dispatch in plan.dispatches},
            {"owner-a", "owner-b"},
        )
        self.assertTrue(
            all(len(dispatch.assignments) == 1 for dispatch in plan.dispatches)
        )
        self.assertEqual(plan.assignment_states_evaluated, 4)
        self.assertTrue(plan.assignment_optimality_proven)
        self.assertEqual(plan.weight_loaded_bytes, 0)
        # One resident physical copy is exercised on each owner; row count
        # must not multiply the fixed weight metric within either dispatch.
        self.assertEqual(plan.weight_avoided_bytes, 20)
        self.assertEqual(
            dict(plan.owner_peak_vram_bytes),
            {"owner-a": 35, "owner-b": 35, "root": 48},
        )
        self.assertEqual(owners["owner-a"].batch_calls, 1)
        self.assertEqual(owners["owner-b"].batch_calls, 1)

    def test_rows_split_across_replicas_when_parallelism_reduces_makespan(self) -> None:
        inventory = records(experts=1, byte_size=10)
        key = inventory[0].key
        mesh = ResidentExpertMesh(
            coordinator_id="root",
            experts=inventory,
            nodes=(
                node("root", budget=200, workspace_bytes_per_token=1),
                node(
                    "owner-a",
                    budget=200,
                    compute_ms=10.0,
                    workspace_bytes_per_token=1,
                ),
                node(
                    "owner-b",
                    budget=200,
                    compute_ms=10.0,
                    workspace_bytes_per_token=1,
                ),
            ),
            links=(
                MeshLinkProfile("root", "owner-a", 0.0, 1_000_000_000.0),
                MeshLinkProfile("root", "owner-b", 0.0, 1_000_000_000.0),
            ),
            local_ram_keys=(),
            local_gpu_keys=(),
            replicas=(
                ResidentExpertReplica(key, "owner-a", inventory[0].content_id),
                ResidentExpertReplica(key, "owner-b", inventory[0].content_id),
            ),
            local_weight_buffer_bytes=0,
            activation_bytes_per_token=8,
            require_local_ram_fallback=False,
        )
        routing = AuthoritativeRouting(
            torch.zeros((4, 1), dtype=torch.long),
            torch.ones((4, 1), dtype=torch.float32),
        )

        plan = mesh.plan_layer(0, routing)

        self.assertEqual(
            sorted(len(dispatch.assignments) for dispatch in plan.dispatches),
            [2, 2],
        )
        self.assertLess(plan.exposed_ms, 30.0)
        self.assertEqual(plan.assignment_states_evaluated, 16)
        self.assertTrue(plan.assignment_optimality_proven)
        # Two rows land on each replica, but each resident physical weight is
        # still accounted once rather than once per row.
        self.assertEqual(plan.weight_avoided_bytes, 20)

    def test_grouped_rows_charge_local_ram_weight_once(self) -> None:
        inventory = records(experts=1, byte_size=10)
        key = inventory[0].key
        mesh = ResidentExpertMesh(
            coordinator_id="root",
            experts=inventory,
            nodes=(
                node("root", budget=85, workspace_bytes_per_token=1),
            ),
            links=(),
            local_ram_keys=(key,),
            local_gpu_keys=(),
            replicas=(),
            local_weight_buffer_bytes=10,
            activation_bytes_per_token=8,
        )
        routing = AuthoritativeRouting(
            torch.zeros((3, 1), dtype=torch.long),
            torch.ones((3, 1), dtype=torch.float32),
        )

        plan = mesh.plan_layer(0, routing)

        self.assertEqual(len(plan.dispatches), 1)
        self.assertEqual(len(plan.dispatches[0].assignments), 3)
        self.assertEqual(plan.weight_loaded_bytes, 10)
        self.assertEqual(plan.weight_avoided_bytes, 0)
        self.assertEqual(dict(plan.owner_peak_vram_bytes), {"root": 85})

    def test_exact_single_choice_over_many_experts_is_iterative(self) -> None:
        expert_count = 1_101
        inventory = records(experts=expert_count, byte_size=1)
        all_keys = tuple(record.key for record in inventory)
        mesh = ResidentExpertMesh(
            coordinator_id="root",
            experts=inventory,
            nodes=(
                node("root", budget=6_000, workspace_bytes_per_token=1),
            ),
            links=(),
            local_ram_keys=(),
            local_gpu_keys=all_keys,
            replicas=(),
            local_weight_buffer_bytes=0,
            activation_bytes_per_token=1,
            require_local_ram_fallback=False,
        )
        routing = AuthoritativeRouting(
            torch.arange(expert_count, dtype=torch.long).reshape(-1, 1),
            torch.ones((expert_count, 1), dtype=torch.float32),
        )

        plan = mesh.plan_layer(0, routing)

        self.assertEqual(plan.assignment_strategy, "exact-enumeration")
        self.assertTrue(plan.assignment_optimality_proven)
        self.assertEqual(plan.assignment_states_evaluated, 1)
        self.assertEqual(len(plan.dispatches), expert_count)

    def test_bounded_repair_escapes_incremental_greedy_dead_end(self) -> None:
        inventory = records(experts=11, byte_size=1)
        profiles = (
            node("root", budget=100, compute_ms=100.0, workspace_bytes_per_token=1),
            # These three owners each have dynamic room for exactly one row.
            node("a", budget=7, compute_ms=0.0, workspace_bytes_per_token=1),
            node("b", budget=5, compute_ms=100.0, workspace_bytes_per_token=1),
            node("c", budget=6, compute_ms=1.0, workspace_bytes_per_token=1),
            node("d", budget=100, compute_ms=2.0, workspace_bytes_per_token=1),
            node("e", budget=100, compute_ms=3.0, workspace_bytes_per_token=1),
        )
        links = tuple(
            MeshLinkProfile(
                "root",
                owner_id,
                0.0,
                1_000_000_000.0,
                rpc_setup_ms_per_batch=0.0,
                host_device_staging_gbytes_per_second=1.0,
            )
            for owner_id in ("a", "b", "c", "d", "e")
        )
        placements = []
        owner_sets = {
            0: ("a", "b"),
            1: ("a", "c"),
            2: ("a", "c"),
            **{expert: ("d", "e") for expert in range(3, 11)},
        }
        for record in inventory:
            for owner_id in owner_sets[record.key.expert]:
                placements.append(
                    ResidentExpertReplica(record.key, owner_id, record.content_id)
                )
        mesh = ResidentExpertMesh(
            coordinator_id="root",
            experts=inventory,
            nodes=profiles,
            links=links,
            local_ram_keys=(),
            local_gpu_keys=(),
            replicas=tuple(placements),
            local_weight_buffer_bytes=0,
            activation_bytes_per_token=1,
            require_local_ram_fallback=False,
        )
        routing = AuthoritativeRouting(
            torch.arange(11, dtype=torch.long).reshape(-1, 1),
            torch.ones((11, 1), dtype=torch.float32),
        )

        plan = mesh.plan_layer(0, routing)

        owners = {dispatch.key.expert: dispatch.owner_id for dispatch in plan.dispatches}
        # Greedy chooses fast owner a for expert 0, then c for expert 1 and
        # reaches a dead end on expert 2. The bounded repair must backtrack to
        # the only feasible matching instead of reporting a false failure.
        self.assertEqual(owners[0], "b")
        self.assertEqual({owners[1], owners[2]}, {"a", "c"})
        self.assertEqual(plan.assignment_strategy, "heuristic-bounded-repair")
        self.assertFalse(plan.assignment_optimality_proven)
        self.assertGreater(plan.assignment_states_evaluated, 0)

    def test_large_prefill_uses_one_full_plan_evaluation(self) -> None:
        row_count = 800
        inventory = records(experts=1, byte_size=1)
        key = inventory[0].key
        mesh = ResidentExpertMesh(
            coordinator_id="root",
            experts=inventory,
            nodes=(
                node("root", budget=100_000, workspace_bytes_per_token=1),
                node("owner-a", budget=100_000, workspace_bytes_per_token=1),
                node("owner-b", budget=100_000, workspace_bytes_per_token=1),
            ),
            links=(
                MeshLinkProfile("root", "owner-a", 0.0, 1_000_000_000.0),
                MeshLinkProfile("root", "owner-b", 0.0, 1_000_000_000.0),
            ),
            local_ram_keys=(),
            local_gpu_keys=(),
            replicas=(
                ResidentExpertReplica(key, "owner-a", inventory[0].content_id),
                ResidentExpertReplica(key, "owner-b", inventory[0].content_id),
            ),
            local_weight_buffer_bytes=0,
            activation_bytes_per_token=1,
            require_local_ram_fallback=False,
        )
        routing = AuthoritativeRouting(
            torch.zeros((row_count, 1), dtype=torch.long),
            torch.ones((row_count, 1), dtype=torch.float32),
        )

        with mock.patch.object(
            mesh,
            "_evaluate_assignment_selection",
            wraps=mesh._evaluate_assignment_selection,
        ) as full_evaluation:
            plan = mesh.plan_layer(0, routing)

        self.assertEqual(full_evaluation.call_count, 1)
        self.assertEqual(
            plan.assignment_strategy,
            "heuristic-incremental-local-search",
        )
        self.assertFalse(plan.assignment_optimality_proven)
        self.assertLessEqual(plan.assignment_states_evaluated, 4 * row_count)
        self.assertEqual(
            sorted(len(dispatch.assignments) for dispatch in plan.dispatches),
            [row_count // 2, row_count // 2],
        )

    def test_thousand_replicas_use_bounded_truthful_candidate_frontier(self) -> None:
        replica_count = 1_000
        row_count = 128
        inventory = records(experts=1, byte_size=1)
        key = inventory[0].key
        owner_ids = tuple(f"owner-{index:04d}" for index in range(replica_count))
        profiles = (
            node(
                "root",
                budget=10_000,
                compute_ms=100.0,
                workspace_bytes_per_token=1,
            ),
            *(
                node(
                    owner_id,
                    budget=1_000,
                    compute_ms=0.0,
                    workspace_bytes_per_token=1,
                )
                for owner_id in owner_ids
            ),
        )
        links = tuple(
            MeshLinkProfile(
                "root",
                owner_id,
                0.0,
                1_000_000_000.0,
                rpc_setup_ms_per_batch=0.0,
                host_device_staging_gbytes_per_second=10.0,
            )
            for owner_id in owner_ids
        )
        replicas = tuple(
            ResidentExpertReplica(key, owner_id, inventory[0].content_id)
            for owner_id in owner_ids
        )
        mesh = ResidentExpertMesh(
            coordinator_id="root",
            experts=inventory,
            nodes=profiles,
            links=links,
            local_ram_keys=(),
            local_gpu_keys=(),
            replicas=replicas,
            local_weight_buffer_bytes=0,
            activation_bytes_per_token=1,
            require_local_ram_fallback=False,
            max_inflight_owner_rpcs=32,
        )
        routing = AuthoritativeRouting(
            torch.zeros((row_count, 1), dtype=torch.long),
            torch.ones((row_count, 1), dtype=torch.float32),
        )

        try:
            plan = mesh.plan_layer(0, routing)
        finally:
            mesh.close()

        self.assertEqual(
            plan.assignment_strategy,
            "heuristic-candidate-pruned-local-search",
        )
        self.assertFalse(plan.assignment_optimality_proven)
        # Structural bound: 32 candidates per greedy row plus at most 32^2
        # local probes. This rejects the old 128 x 1,000 scoring surface
        # without relying on host timing.
        self.assertLessEqual(
            plan.assignment_states_evaluated,
            row_count * 32 + 32**2,
        )
        self.assertEqual(plan.max_inflight_owner_rpcs, 32)

    def test_pruned_frontier_extends_until_row_capacity_is_represented(self) -> None:
        owner_count = 100
        row_count = 80
        inventory = records(experts=1, byte_size=1)
        key = inventory[0].key
        owner_ids = tuple(f"owner-{index:03d}" for index in range(owner_count))
        mesh = ResidentExpertMesh(
            coordinator_id="root",
            experts=inventory,
            nodes=(
                node("root", budget=1_000, workspace_bytes_per_token=1),
                *(
                    # One resident byte plus exactly one four-byte routed row.
                    node(
                        owner_id,
                        budget=5,
                        compute_ms=0.0,
                        workspace_bytes_per_token=1,
                    )
                    for owner_id in owner_ids
                ),
            ),
            links=tuple(
                MeshLinkProfile(
                    "root",
                    owner_id,
                    0.0,
                    1_000_000_000.0,
                    rpc_setup_ms_per_batch=0.0,
                    host_device_staging_gbytes_per_second=10.0,
                )
                for owner_id in owner_ids
            ),
            local_ram_keys=(),
            local_gpu_keys=(),
            replicas=tuple(
                ResidentExpertReplica(key, owner_id, inventory[0].content_id)
                for owner_id in owner_ids
            ),
            local_weight_buffer_bytes=0,
            activation_bytes_per_token=1,
            require_local_ram_fallback=False,
            max_inflight_owner_rpcs=8,
        )
        routing = AuthoritativeRouting(
            torch.zeros((row_count, 1), dtype=torch.long),
            torch.ones((row_count, 1), dtype=torch.float32),
        )

        try:
            plan = mesh.plan_layer(0, routing)
        finally:
            mesh.close()

        self.assertEqual(len(plan.dispatches), row_count)
        self.assertEqual(len(plan.owner_exposed_ms), row_count)
        self.assertTrue(plan.assignment_strategy.startswith("heuristic-candidate-pruned"))
        self.assertFalse(plan.assignment_optimality_proven)

    def test_bounded_owner_waves_match_execution_and_calibrated_overhead(self) -> None:
        tracker_lock = threading.Lock()
        active = 0
        peak = 0

        class TrackingOwner:
            supports_exact_input_coalescing = False

            def __init__(self, node_id: str, record: ExpertRecord) -> None:
                self.node_id = node_id
                self.record = record

            def has_expert(self, key: ExpertKey, content_id: str) -> bool:
                return key == self.record.key and content_id == self.record.content_id

            def is_expert_resident(self, key: ExpertKey, content_id: str) -> bool:
                return self.has_expert(key, content_id)

            def execute_batch(self, items):
                nonlocal active, peak
                with tracker_lock:
                    active += 1
                    peak = max(peak, active)
                try:
                    time.sleep(0.02)
                    return tuple(
                        OwnerExpertBatchResult(item.key, item.activations)
                        for item in items
                    )
                finally:
                    with tracker_lock:
                        active -= 1

        owner_count = 6
        inventory = records(experts=owner_count, byte_size=1)
        owner_ids = tuple(f"owner-{index}" for index in range(owner_count))
        profiles = (
            node(
                "root",
                budget=1_000,
                workspace_bytes_per_token=1,
                ingress_mbps=1_000_000_000.0,
                egress_mbps=1_000_000_000.0,
            ),
            *(
                node(
                    owner_id,
                    budget=100,
                    compute_ms=0.0,
                    workspace_bytes_per_token=1,
                )
                for owner_id in owner_ids
            ),
        )
        links = tuple(
            MeshLinkProfile(
                "root",
                owner_id,
                10.0,
                1_000_000_000.0,
                rpc_setup_ms_per_batch=0.0,
                host_device_staging_gbytes_per_second=10.0,
            )
            for owner_id in owner_ids
        )
        replicas = tuple(
            ResidentExpertReplica(record.key, owner_id, record.content_id)
            for record, owner_id in zip(inventory, owner_ids)
        )
        owners = {
            owner_id: TrackingOwner(owner_id, record)
            for record, owner_id in zip(inventory, owner_ids)
        }
        mesh = ResidentExpertMesh(
            coordinator_id="root",
            experts=inventory,
            nodes=profiles,
            links=links,
            local_ram_keys=(),
            local_gpu_keys=(),
            replicas=replicas,
            local_weight_buffer_bytes=0,
            activation_bytes_per_token=4,
            require_local_ram_fallback=False,
            max_inflight_owner_rpcs=2,
            coordinator_rpc_dispatch_ms_per_owner=0.2,
            coordinator_serialization_gbytes_per_second=1.0,
            coordinator_reduction_ms_per_assignment=0.1,
        )
        hidden = torch.arange(owner_count, dtype=torch.float32).reshape(-1, 1)
        routing = AuthoritativeRouting(
            torch.arange(owner_count, dtype=torch.long).reshape(-1, 1),
            torch.ones((owner_count, 1), dtype=torch.float32),
        )

        try:
            actual, plan = mesh.execute_layer(hidden, 0, routing, owners)
        finally:
            mesh.close()

        torch.testing.assert_close(actual, hidden, rtol=0, atol=0)
        self.assertEqual(peak, 2)
        self.assertEqual(plan.max_inflight_owner_rpcs, 2)
        self.assertAlmostEqual(
            plan.owner_scheduled_makespan_ms,
            3 * plan.per_owner_lower_bound_ms,
            places=9,
        )
        expected_overhead = (
            owner_count * 0.2
            + plan.transport_payload_bytes / 1_000_000.0
            + owner_count * 0.1
        )
        self.assertAlmostEqual(
            plan.coordinator_overhead_ms,
            expected_overhead,
            places=12,
        )
        self.assertAlmostEqual(
            plan.exposed_ms,
            max(
                plan.owner_scheduled_makespan_ms,
                plan.coordinator_nic_lower_bound_ms,
            )
            + expected_overhead,
            places=12,
        )
        self.assertFalse(plan.coordination_calibration_required)
        self.assertFalse(plan.transport_calibration_required)

    def test_missing_coordinator_cost_profile_is_marked_uncalibrated(self) -> None:
        inventory = records(experts=1, byte_size=1)
        record = inventory[0]
        mesh = ResidentExpertMesh(
            coordinator_id="root",
            experts=inventory,
            nodes=(
                node(
                    "root",
                    budget=100,
                    workspace_bytes_per_token=1,
                    ingress_mbps=1_000_000.0,
                    egress_mbps=1_000_000.0,
                ),
                node(
                    "owner-1",
                    budget=100,
                    workspace_bytes_per_token=1,
                ),
            ),
            links=(
                MeshLinkProfile(
                    "root",
                    "owner-1",
                    1.0,
                    1_000_000.0,
                    rpc_setup_ms_per_batch=0.0,
                    host_device_staging_gbytes_per_second=1.0,
                ),
            ),
            local_ram_keys=(),
            local_gpu_keys=(),
            replicas=(
                ResidentExpertReplica(
                    record.key,
                    "owner-1",
                    record.content_id,
                ),
            ),
            local_weight_buffer_bytes=0,
            activation_bytes_per_token=4,
            require_local_ram_fallback=False,
        )
        routing = AuthoritativeRouting(
            torch.zeros((1, 1), dtype=torch.long),
            torch.ones((1, 1), dtype=torch.float32),
        )

        try:
            plan = mesh.plan_layer(0, routing)
        finally:
            mesh.close()

        self.assertTrue(plan.coordination_calibration_required)
        self.assertTrue(plan.transport_calibration_required)

    def test_owner_row_map_cost_controls_planning_and_execution_choice(self) -> None:
        inventory = records(experts=2)
        mesh = ResidentExpertMesh(
            coordinator_id="root",
            experts=inventory,
            nodes=(node("root"), node("owner-1")),
            links=(link(rtt_ms=0.0),),
            local_ram_keys=(),
            local_gpu_keys=(),
            replicas=tuple(
                ResidentExpertReplica(record.key, "owner-1", record.content_id)
                for record in inventory
            ),
            local_weight_buffer_bytes=0,
            activation_bytes_per_token=12,
            require_local_ram_fallback=False,
        )
        routing = AuthoritativeRouting(
            torch.tensor([[0, 1], [0, 1]], dtype=torch.long),
            torch.full((2, 2), 0.5, dtype=torch.float32),
        )

        plan = mesh.plan_layer(
            0,
            routing,
            coalesced_row_index_bytes_by_node={"owner-1": 100},
        )

        self.assertEqual(plan.coalesced_owner_ids, ())
        self.assertEqual(plan.route_metadata_bytes, 0)
        self.assertEqual(plan.activation_request_bytes, 48)

    def test_replica_identity_and_vram_budget_are_fail_closed(self) -> None:
        inventory = records(experts=1)
        key = inventory[0].key
        with self.assertRaisesRegex(ValueError, "content identity mismatch"):
            ResidentExpertMesh(
                coordinator_id="root",
                experts=inventory,
                nodes=(node("root"), node("owner-1")),
                links=(link(rtt_ms=1),),
                local_ram_keys=(key,),
                local_gpu_keys=(),
                replicas=(ResidentExpertReplica(key, "owner-1", "wrong"),),
                local_weight_buffer_bytes=10 * MIB,
                activation_bytes_per_token=12,
            )
        with self.assertRaisesRegex(ValueError, "VRAM budget exceeded"):
            ResidentExpertMesh(
                coordinator_id="root",
                experts=inventory,
                nodes=(node("root"), node("owner-1", budget=5 * MIB)),
                links=(link(rtt_ms=1),),
                local_ram_keys=(key,),
                local_gpu_keys=(),
                replicas=(
                    ResidentExpertReplica(key, "owner-1", inventory[0].content_id),
                ),
                local_weight_buffer_bytes=10 * MIB,
                activation_bytes_per_token=12,
            )

    def test_coordinator_aggregate_nic_is_a_layer_lower_bound(self) -> None:
        inventory = records(experts=2)
        mesh = ResidentExpertMesh(
            coordinator_id="root",
            experts=inventory,
            nodes=(
                node("root", ingress_mbps=0.5, egress_mbps=0.5),
                node("owner-1"),
                node("owner-2"),
            ),
            links=(
                link(rtt_ms=0.0, to_node="owner-1"),
                link(rtt_ms=0.0, to_node="owner-2"),
            ),
            local_ram_keys=(),
            local_gpu_keys=(),
            replicas=(
                ResidentExpertReplica(
                    inventory[0].key,
                    "owner-1",
                    inventory[0].content_id,
                ),
                ResidentExpertReplica(
                    inventory[1].key,
                    "owner-2",
                    inventory[1].content_id,
                ),
            ),
            local_weight_buffer_bytes=0,
            activation_bytes_per_token=12,
            require_local_ram_fallback=False,
        )
        routing = AuthoritativeRouting(
            torch.tensor([[0, 1]], dtype=torch.long),
            torch.tensor([[0.5, 0.5]], dtype=torch.float32),
        )

        plan = mesh.plan_layer(0, routing)

        self.assertGreater(
            plan.coordinator_nic_lower_bound_ms,
            plan.per_owner_lower_bound_ms,
        )
        self.assertEqual(plan.exposed_ms, plan.coordinator_nic_lower_bound_ms)
        self.assertTrue(plan.transport_calibration_required)

    def test_asymmetric_link_prices_request_and_response_directions(self) -> None:
        inventory = records(experts=1)
        key = inventory[0].key
        mesh = ResidentExpertMesh(
            coordinator_id="root",
            experts=inventory,
            nodes=(node("root"), node("owner-1", compute_ms=0.2)),
            links=(
                MeshLinkProfile(
                    from_node="root",
                    to_node="owner-1",
                    round_trip_ms=0.0,
                    bandwidth_mbps=1_000.0,
                    egress_bandwidth_mbps=100.0,
                    ingress_bandwidth_mbps=1_000.0,
                ),
            ),
            local_ram_keys=(),
            local_gpu_keys=(),
            replicas=(
                ResidentExpertReplica(key, "owner-1", inventory[0].content_id),
            ),
            local_weight_buffer_bytes=0,
            activation_bytes_per_token=12,
            require_local_ram_fallback=False,
        )
        routing = AuthoritativeRouting(
            torch.tensor([[0]], dtype=torch.long),
            torch.tensor([[1.0]], dtype=torch.float32),
        )

        plan = mesh.plan_layer(0, routing)

        expected = 0.2 + 12 / (100 * 125.0) + 12 / (1_000 * 125.0)
        self.assertAlmostEqual(dict(plan.owner_exposed_ms)["owner-1"], expected)
        self.assertEqual(plan.activation_request_bytes, 12)
        self.assertEqual(plan.activation_response_bytes, 12)

    def test_execution_rejects_unsealed_workspace_and_wrong_output_dtype(self) -> None:
        inventory = records(experts=1)
        key = inventory[0].key
        routing = AuthoritativeRouting(
            torch.tensor([[0]], dtype=torch.long),
            torch.tensor([[1.0]], dtype=torch.float32),
        )
        hidden = torch.ones((1, 3), dtype=torch.float32)

        unsealed = ResidentExpertMesh(
            coordinator_id="root",
            experts=inventory,
            nodes=(
                node("root"),
                node("owner-1", workspace_bytes_per_token=0),
            ),
            links=(link(rtt_ms=0.0),),
            local_ram_keys=(),
            local_gpu_keys=(),
            replicas=(
                ResidentExpertReplica(key, "owner-1", inventory[0].content_id),
            ),
            local_weight_buffer_bytes=0,
            activation_bytes_per_token=12,
            require_local_ram_fallback=False,
        )
        owner = InMemoryExpertOwner(
            "owner-1",
            {key: (inventory[0].content_id, torch.eye(3))},
        )
        with self.assertRaisesRegex(Exception, "workspace"):
            unsealed.execute_layer(hidden, 0, routing, {"owner-1": owner})

        sealed = ResidentExpertMesh(
            coordinator_id="root",
            experts=inventory,
            nodes=(node("root"), node("owner-1")),
            links=(link(rtt_ms=0.0),),
            local_ram_keys=(),
            local_gpu_keys=(),
            replicas=(
                ResidentExpertReplica(key, "owner-1", inventory[0].content_id),
            ),
            local_weight_buffer_bytes=0,
            activation_bytes_per_token=12,
            require_local_ram_fallback=False,
        )

        class WrongDtypeOwner:
            node_id = "owner-1"

            @staticmethod
            def has_expert(candidate: ExpertKey, content_id: str) -> bool:
                return candidate == key and content_id == inventory[0].content_id

            @staticmethod
            def is_expert_resident(
                candidate: ExpertKey,
                content_id: str,
            ) -> bool:
                return candidate == key and content_id == inventory[0].content_id

            @staticmethod
            def execute_batch(items):
                return tuple(
                    OwnerExpertBatchResult(
                        item.key,
                        item.activations.to(dtype=torch.float64),
                    )
                    for item in items
                )

        with self.assertRaisesRegex(Exception, "dtype"):
            sealed.execute_layer(
                hidden,
                0,
                routing,
                {"owner-1": WrongDtypeOwner()},
            )

    def test_worker_pool_is_reused_and_closed_without_closing_owner(self) -> None:
        inventory = records(experts=1)
        key = inventory[0].key
        mesh = ResidentExpertMesh(
            coordinator_id="root",
            experts=inventory,
            nodes=(node("root"),),
            links=(),
            local_ram_keys=(key,),
            local_gpu_keys=(key,),
            replicas=(),
            local_weight_buffer_bytes=10 * MIB,
            activation_bytes_per_token=12,
        )

        class RecordingOwner(InMemoryExpertOwner):
            def __init__(self):
                super().__init__(
                    "root",
                    {key: (inventory[0].content_id, torch.eye(3))},
                )
                self.thread_ids: list[int] = []
                self.close_calls = 0

            def execute_batch(self, items):
                self.thread_ids.append(threading.get_ident())
                return super().execute_batch(items)

            def close(self) -> None:
                self.close_calls += 1

        owner = RecordingOwner()
        routing = AuthoritativeRouting(
            torch.tensor([[0]], dtype=torch.long),
            torch.tensor([[1.0]], dtype=torch.float32),
        )
        hidden = torch.ones((1, 3), dtype=torch.float32)

        mesh.execute_layer(hidden, 0, routing, {"root": owner})
        mesh.execute_layer(hidden, 0, routing, {"root": owner})
        self.assertEqual(len(set(owner.thread_ids)), 1)

        mesh.close()
        self.assertTrue(mesh.closed)
        self.assertEqual(owner.close_calls, 0)
        with self.assertRaisesRegex(Exception, "closed"):
            mesh.execute_layer(hidden, 0, routing, {"root": owner})

    def test_execution_replans_cold_local_and_remote_resident_paths_to_ram(self) -> None:
        inventory = records(experts=2)
        keys = tuple(record.key for record in inventory)
        mesh = ResidentExpertMesh(
            coordinator_id="root",
            experts=inventory,
            nodes=(
                node("root", compute_ms=0.5, ram_gbytes_s=0.1),
                node("owner-1", compute_ms=0.01),
            ),
            links=(link(rtt_ms=0.01),),
            local_ram_keys=keys,
            local_gpu_keys=(keys[0],),
            replicas=(
                ResidentExpertReplica(
                    keys[1],
                    "owner-1",
                    inventory[1].content_id,
                ),
            ),
            local_weight_buffer_bytes=10 * MIB,
            activation_bytes_per_token=12,
        )
        routing = AuthoritativeRouting(
            torch.tensor([[0, 1]], dtype=torch.long),
            torch.tensor([[0.4, 0.6]], dtype=torch.float32),
        )
        projected = mesh.plan_layer(0, routing)
        self.assertEqual(
            {dispatch.path for dispatch in projected.dispatches},
            {"local-gpu", "remote-resident"},
        )

        class ColdOwner(InMemoryExpertOwner):
            @staticmethod
            def is_expert_resident(key: ExpertKey, content_id: str) -> bool:
                del key, content_id
                return False

        weights = {key: torch.eye(3) * (key.expert + 1) for key in keys}
        root = ColdOwner(
            "root",
            {
                key: (inventory[key.expert].content_id, weight)
                for key, weight in weights.items()
            },
        )
        remote = ColdOwner(
            "owner-1",
            {keys[1]: (inventory[1].content_id, weights[keys[1]])},
        )
        hidden = torch.tensor([[0.25, -0.5, 1.0]], dtype=torch.float32)

        actual, executed = mesh.execute_layer(
            hidden,
            0,
            routing,
            {"root": root, "owner-1": remote},
        )

        expected = hidden * (0.4 * 1.0 + 0.6 * 2.0)
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
        self.assertEqual(
            {dispatch.path for dispatch in executed.dispatches},
            {"local-ram"},
        )
        self.assertEqual(root.batch_calls, 1)
        self.assertEqual(remote.batch_calls, 0)

    def test_owner_failure_cancels_and_drains_peers_before_primary_reraise(self) -> None:
        inventory = records(experts=2)
        keys = tuple(record.key for record in inventory)
        mesh = ResidentExpertMesh(
            coordinator_id="root",
            experts=inventory,
            nodes=(node("root"), node("fail"), node("slow")),
            links=(
                link(rtt_ms=0.01, to_node="fail"),
                link(rtt_ms=0.01, to_node="slow"),
            ),
            local_ram_keys=(),
            local_gpu_keys=(),
            replicas=(
                ResidentExpertReplica(keys[0], "fail", inventory[0].content_id),
                ResidentExpertReplica(keys[1], "slow", inventory[1].content_id),
            ),
            local_weight_buffer_bytes=0,
            activation_bytes_per_token=12,
            require_local_ram_fallback=False,
        )
        slow_started = threading.Event()
        slow_finished = threading.Event()
        primary = RuntimeError("primary owner failure")

        class FailingOwner:
            node_id = "fail"

            @staticmethod
            def has_expert(key: ExpertKey, content_id: str) -> bool:
                return key == keys[0] and content_id == inventory[0].content_id

            is_expert_resident = has_expert

            @staticmethod
            def execute_batch(items):
                del items
                if not slow_started.wait(timeout=1.0):
                    raise AssertionError("slow peer did not start")
                raise primary

        class SlowOwner:
            node_id = "slow"

            @staticmethod
            def has_expert(key: ExpertKey, content_id: str) -> bool:
                return key == keys[1] and content_id == inventory[1].content_id

            is_expert_resident = has_expert

            @staticmethod
            def execute_batch(items):
                slow_started.set()
                time.sleep(0.05)
                slow_finished.set()
                return tuple(
                    OwnerExpertBatchResult(item.key, item.activations)
                    for item in items
                )

        routing = AuthoritativeRouting(
            torch.tensor([[0, 1]], dtype=torch.long),
            torch.tensor([[0.5, 0.5]], dtype=torch.float32),
        )
        with self.assertRaises(RuntimeError) as caught:
            mesh.execute_layer(
                torch.ones((1, 3), dtype=torch.float32),
                0,
                routing,
                {"fail": FailingOwner(), "slow": SlowOwner()},
            )

        self.assertIs(caught.exception, primary)
        self.assertTrue(slow_finished.is_set())
        mesh.close()

    def test_many_stale_replicas_retry_iteratively_until_local_ram(self) -> None:
        inventory = records(experts=1, byte_size=64)
        record = inventory[0]
        key = record.key
        replica_count = 96
        replica_ids = tuple(
            f"stale-{index:03d}" for index in range(replica_count)
        )
        mesh = ResidentExpertMesh(
            coordinator_id="root",
            experts=inventory,
            nodes=(
                node(
                    "root",
                    budget=4096,
                    compute_ms=100.0,
                    ram_gbytes_s=0.000001,
                    workspace_bytes_per_token=4,
                ),
                *(
                    node(
                        node_id,
                        budget=1024,
                        compute_ms=0.0,
                        workspace_bytes_per_token=4,
                    )
                    for node_id in replica_ids
                ),
            ),
            links=tuple(
                link(rtt_ms=0.0, to_node=node_id) for node_id in replica_ids
            ),
            local_ram_keys=(key,),
            local_gpu_keys=(),
            replicas=tuple(
                ResidentExpertReplica(key, node_id, record.content_id)
                for node_id in replica_ids
            ),
            local_weight_buffer_bytes=record.byte_size,
            activation_bytes_per_token=12,
        )

        class StaleOwner:
            def __init__(self, node_id: str) -> None:
                self.node_id = node_id
                self.batch_calls = 0

            @staticmethod
            def has_expert(candidate: ExpertKey, content_id: str) -> bool:
                return candidate == key and content_id == record.content_id

            is_expert_resident = has_expert

            def execute_batch(self, items):
                self.batch_calls += 1
                self.assert_batch(items)
                raise ExpertResidentSlotUnavailableError(key)

            @staticmethod
            def assert_batch(items) -> None:
                if len(items) != 1 or items[0].key != key:
                    raise AssertionError("unexpected stale replica batch")

        stale_owners = {
            node_id: StaleOwner(node_id) for node_id in replica_ids
        }
        root = InMemoryExpertOwner(
            "root",
            {key: (record.content_id, torch.eye(3))},
        )
        owners = {"root": root, **stale_owners}
        hidden = torch.tensor([[0.25, -0.5, 1.0]], dtype=torch.float32)
        routing = AuthoritativeRouting(
            torch.tensor([[0]], dtype=torch.long),
            torch.tensor([[1.0]], dtype=torch.float32),
        )

        previous_recursion_limit = sys.getrecursionlimit()
        try:
            # A recursive retry consumes this budget long before all replicas
            # are excluded; the iterative implementation keeps constant depth.
            sys.setrecursionlimit(80)
            actual, plan = mesh.execute_layer(hidden, 0, routing, owners)
        finally:
            sys.setrecursionlimit(previous_recursion_limit)
            mesh.close()

        torch.testing.assert_close(actual, hidden, rtol=0, atol=0)
        self.assertEqual(
            {dispatch.path for dispatch in plan.dispatches},
            {"local-ram"},
        )
        self.assertEqual(
            sum(owner.batch_calls for owner in stale_owners.values()),
            replica_count,
        )
        self.assertTrue(
            all(owner.batch_calls == 1 for owner in stale_owners.values())
        )
        self.assertEqual(root.batch_calls, 1)

    def test_simultaneous_residency_loss_does_not_mask_fatal_owner_error(self) -> None:
        inventory = records(experts=2, byte_size=64)
        keys = tuple(record.key for record in inventory)
        mesh = ResidentExpertMesh(
            coordinator_id="root",
            experts=inventory,
            nodes=(
                node(
                    "root",
                    budget=4096,
                    compute_ms=100.0,
                    ram_gbytes_s=0.000001,
                    workspace_bytes_per_token=4,
                ),
                node(
                    "fatal",
                    budget=1024,
                    compute_ms=0.0,
                    workspace_bytes_per_token=4,
                ),
                node(
                    "slot",
                    budget=1024,
                    compute_ms=0.0,
                    workspace_bytes_per_token=4,
                ),
            ),
            links=(
                link(rtt_ms=0.0, to_node="fatal"),
                link(rtt_ms=0.0, to_node="slot"),
            ),
            local_ram_keys=keys,
            local_gpu_keys=(),
            replicas=(
                ResidentExpertReplica(keys[0], "slot", inventory[0].content_id),
                ResidentExpertReplica(keys[1], "fatal", inventory[1].content_id),
            ),
            local_weight_buffer_bytes=64,
            activation_bytes_per_token=12,
        )
        rendezvous = threading.Barrier(2)
        fatal = RuntimeError("fatal owner failure")

        class SlotOwner:
            node_id = "slot"
            batch_calls = 0

            @staticmethod
            def has_expert(candidate: ExpertKey, content_id: str) -> bool:
                return (
                    candidate == keys[0]
                    and content_id == inventory[0].content_id
                )

            is_expert_resident = has_expert

            @classmethod
            def execute_batch(cls, items):
                cls.batch_calls += 1
                rendezvous.wait(timeout=5.0)
                raise ExpertResidentSlotUnavailableError(items[0].key)

        class FatalOwner:
            node_id = "fatal"
            batch_calls = 0

            @staticmethod
            def has_expert(candidate: ExpertKey, content_id: str) -> bool:
                return (
                    candidate == keys[1]
                    and content_id == inventory[1].content_id
                )

            is_expert_resident = has_expert

            @classmethod
            def execute_batch(cls, items):
                del items
                cls.batch_calls += 1
                rendezvous.wait(timeout=5.0)
                raise fatal

        root = InMemoryExpertOwner(
            "root",
            {
                record.key: (record.content_id, torch.eye(3))
                for record in inventory
            },
        )
        routing = AuthoritativeRouting(
            torch.tensor([[0, 1]], dtype=torch.long),
            torch.tensor([[0.5, 0.5]], dtype=torch.float32),
        )

        try:
            with self.assertRaises(RuntimeError) as caught:
                mesh.execute_layer(
                    torch.ones((1, 3), dtype=torch.float32),
                    0,
                    routing,
                    {"root": root, "slot": SlotOwner(), "fatal": FatalOwner()},
                )
        finally:
            mesh.close()

        self.assertIs(caught.exception, fatal)
        self.assertEqual(SlotOwner.batch_calls, 1)
        self.assertEqual(FatalOwner.batch_calls, 1)
        self.assertEqual(root.batch_calls, 0)


if __name__ == "__main__":
    unittest.main()
