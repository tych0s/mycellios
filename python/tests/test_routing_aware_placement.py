from __future__ import annotations

import unittest
from unittest import mock

from distributed_runtime.ram_expert_cache import ExpertKey, ExpertRecord
from distributed_runtime.resident_expert_mesh import MeshLinkProfile, MeshNodeProfile
from distributed_runtime.routing_aware_placement import (
    ROUTING_AWARE_PLACEMENT_SCHEMA,
    RoutingAwarePlacementUnavailableError,
    RoutingTraceSample,
    _can_pack_exactly,
    plan_routing_aware_expert_placement,
)


def _node(
    node_id: str,
    budget: int,
    *,
    compute_ms: float = 0.1,
    ingress_mbps: float = 0.0,
    egress_mbps: float = 0.0,
) -> MeshNodeProfile:
    return MeshNodeProfile(
        node_id=node_id,
        resident_vram_budget_bytes=budget,
        reserved_vram_bytes=0,
        expert_compute_ms_per_token=compute_ms,
        aggregate_ingress_mbps=ingress_mbps,
        aggregate_egress_mbps=egress_mbps,
    )


def _link(node_id: str, *, rtt_ms: float = 1.0) -> MeshLinkProfile:
    return MeshLinkProfile(
        from_node="root",
        to_node=node_id,
        round_trip_ms=rtt_ms,
        bandwidth_mbps=1_000.0,
    )


def _record(expert: int, *, byte_size: int = 10) -> ExpertRecord:
    return ExpertRecord(
        key=ExpertKey(0, expert),
        byte_size=byte_size,
        content_id=f"sha256:layer-0-expert-{expert}",
    )


class RoutingAwarePlacementTests(unittest.TestCase):
    def test_exact_repair_recovers_dynamic_vram_greedy_dead_end(self) -> None:
        import distributed_runtime.routing_aware_placement as placement_module

        sizes = (10, 8, 6, 5, 4, 3)
        traces = (
            (2, 3, 4, 5),
            (0,),
            (1, 2, 3, 5),
            (0, 1, 2, 5),
        )
        with mock.patch.object(
            placement_module,
            "_search_required_coverage",
            wraps=placement_module._search_required_coverage,
        ) as repair:
            plan = plan_routing_aware_expert_placement(
                coordinator_id="root",
                experts=tuple(
                    _record(expert, byte_size=size)
                    for expert, size in enumerate(sizes)
                ),
                nodes=(
                    _node("root", 100),
                    _node("n0", 17),
                    _node("n1", 12),
                    _node("n2", 28),
                ),
                links=(_link("n0"), _link("n1"), _link("n2")),
                traces=tuple(
                    RoutingTraceSample(0, expert_ids, frequency=1.0)
                    for expert_ids in traces
                ),
                activation_bytes_per_position=1,
                max_positions_per_wave=1,
            )

        self.assertTrue(repair.called)
        self.assertEqual(
            {placement.key for placement in plan.placements},
            {ExpertKey(0, expert) for expert in range(len(sizes))},
        )
        budgets = {"n0": 17, "n1": 12, "n2": 28}
        for node_id, used in plan.projection.used_vram_bytes_by_node:
            self.assertLessEqual(used, budgets[node_id])

    def test_large_irregular_pack_is_inconclusive_not_impossible(self) -> None:
        feasible_pattern = (2, 2, 2, 3, 5, 6)
        self.assertIsNone(
            _can_pack_exactly(feasible_pattern * 11, (10,) * 22)
        )

    def test_dynamic_reserve_allows_topk_split_across_tight_owners(self) -> None:
        plan = plan_routing_aware_expert_placement(
            coordinator_id="root",
            experts=(_record(0), _record(1)),
            nodes=(
                _node("root", 6),
                _node("owner-a", 13),
                _node("owner-b", 13),
            ),
            links=(_link("owner-a"), _link("owner-b")),
            traces=(RoutingTraceSample(0, (0, 1), frequency=1.0),),
            activation_bytes_per_position=1,
            max_positions_per_wave=1,
        )

        self.assertEqual(
            dict(plan.projection.used_vram_bytes_by_node),
            {"owner-a": 13, "owner-b": 13},
        )
        self.assertEqual(
            set(plan.projection.trace_routes[0].owner_ids),
            {"owner-a", "owner-b"},
        )

    def test_dynamic_reserve_scales_with_max_positions_per_wave(self) -> None:
        with self.assertRaisesRegex(
            RoutingAwarePlacementUnavailableError,
            "within VRAM",
        ):
            plan_routing_aware_expert_placement(
                coordinator_id="root",
                experts=(_record(0),),
                nodes=(_node("root", 12), _node("owner-a", 21)),
                links=(_link("owner-a"),),
                traces=(RoutingTraceSample(0, (0,), frequency=1.0),),
                activation_bytes_per_position=1,
                max_positions_per_wave=4,
            )

    def test_partial_coordinator_egress_calibration_is_a_real_bound(self) -> None:
        plan = plan_routing_aware_expert_placement(
            coordinator_id="root",
            experts=(_record(0, byte_size=1),),
            nodes=(
                _node("root", 3_000, ingress_mbps=0.0, egress_mbps=0.01),
                _node("owner-a", 3_001),
            ),
            links=(_link("owner-a", rtt_ms=0.01),),
            traces=(RoutingTraceSample(0, (0,), frequency=1.0),),
            activation_bytes_per_position=1_000,
            max_positions_per_wave=1,
        )

        self.assertGreaterEqual(
            plan.projection.trace_routes[0].projected_link_ms_per_position,
            800.0,
        )

    def test_hypergraph_greedy_beats_equal_memory_marginal_baseline(self) -> None:
        # Every required expert has identical marginal heat. A marginal greedy
        # therefore packs keys (0, 1) and (2, 3), while the measured route sets
        # say that (0, 2) and (1, 3) should share owners.
        records = tuple(_record(expert) for expert in range(5))
        nodes = (
            _node(
                "root",
                100,
                ingress_mbps=1_000.0,
                egress_mbps=0.1,
            ),
            _node("owner-a", 116),
            _node("owner-b", 116),
        )
        links = (_link("owner-a"), _link("owner-b"))
        traces = (
            RoutingTraceSample(0, (0, 2), frequency=100.0, weight=1.0),
            RoutingTraceSample(0, (1, 3), frequency=50.0, weight=2.0),
        )

        plan = plan_routing_aware_expert_placement(
            coordinator_id="root",
            experts=records,
            nodes=nodes,
            links=links,
            traces=traces,
            activation_bytes_per_position=16,
            max_positions_per_wave=1,
            candidate_node_ids=("owner-b", "owner-a"),
        )
        reordered = plan_routing_aware_expert_placement(
            coordinator_id="root",
            experts=tuple(reversed(records)),
            nodes=tuple(reversed(nodes)),
            links=tuple(reversed(links)),
            traces=tuple(reversed(traces)),
            activation_bytes_per_position=16,
            max_positions_per_wave=1,
            candidate_node_ids=("owner-a", "owner-b"),
        )

        self.assertEqual(plan, reordered)
        self.assertEqual(plan.schema, ROUTING_AWARE_PLACEMENT_SCHEMA)
        self.assertEqual(
            plan.required_experts,
            tuple(ExpertKey(0, expert) for expert in range(4)),
        )
        self.assertNotIn(ExpertKey(0, 4), {item.key for item in plan.placements})
        self.assertEqual(
            {(item.key.expert, item.node_id) for item in plan.placements},
            {(0, "owner-a"), (2, "owner-a"), (1, "owner-b"), (3, "owner-b")},
        )
        self.assertEqual(
            {
                (item.key.expert, item.node_id)
                for item in plan.marginal_baseline_placements
            },
            {(0, "owner-a"), (1, "owner-a"), (2, "owner-b"), (3, "owner-b")},
        )
        for item in plan.placements:
            self.assertEqual(
                item.content_id,
                f"sha256:layer-0-expert-{item.key.expert}",
            )

        self.assertTrue(plan.comparison_uses_equal_replica_bytes)
        self.assertEqual(plan.projection.total_replica_bytes, 40)
        self.assertEqual(plan.marginal_baseline_projection.total_replica_bytes, 40)
        self.assertEqual(
            dict(plan.projection.used_vram_bytes_by_node),
            {"owner-a": 116, "owner-b": 116},
        )
        self.assertEqual(plan.projection.weighted_positions, 200.0)
        self.assertEqual(plan.projection.mean_owner_contacts_per_position, 1.0)
        self.assertEqual(
            plan.marginal_baseline_projection.mean_owner_contacts_per_position,
            2.0,
        )
        self.assertEqual(plan.mean_owner_contacts_saved_per_position, 1.0)
        self.assertEqual(
            plan.weighted_owner_contacts_saved,
            200.0,
        )
        self.assertEqual(
            plan.projection.rpc_v1_activation_bytes,
            12_800.0,
        )
        self.assertEqual(
            plan.marginal_baseline_projection.rpc_v1_activation_bytes,
            12_800.0,
        )
        self.assertEqual(plan.rpc_v1_activation_bytes_saved, 0.0)
        self.assertEqual(
            plan.projection.coalesced_exact_activation_bytes,
            9_600.0,
        )
        self.assertEqual(
            plan.marginal_baseline_projection.coalesced_exact_activation_bytes,
            12_800.0,
        )
        self.assertEqual(
            plan.coalesced_exact_activation_bytes_saved,
            3_200.0,
        )
        self.assertEqual(
            plan.projection.coalesced_exact_row_index_bytes,
            1_600.0,
        )
        self.assertEqual(
            plan.projection.owner_partial_activation_byte_floor,
            6_400.0,
        )
        self.assertLess(
            plan.projection.projected_link_ms,
            plan.marginal_baseline_projection.projected_link_ms,
        )
        self.assertEqual(plan.projection.peak_experts_per_position, 2)
        self.assertEqual(
            plan.marginal_baseline_projection.peak_experts_per_position,
            1,
        )

    def test_spare_vram_can_hold_an_exact_beneficial_replica(self) -> None:
        plan = plan_routing_aware_expert_placement(
            coordinator_id="root",
            experts=tuple(_record(expert) for expert in range(3)),
            nodes=(
                _node("root", 100),
                _node("owner-a", 116),
                _node("owner-b", 116),
            ),
            links=(_link("owner-a"), _link("owner-b")),
            traces=(
                RoutingTraceSample(0, (0, 1), frequency=100.0),
                RoutingTraceSample(0, (0, 2), frequency=80.0),
            ),
            activation_bytes_per_position=16,
            max_positions_per_wave=1,
        )

        expert_zero = tuple(
            placement
            for placement in plan.placements
            if placement.key == ExpertKey(0, 0)
        )
        self.assertEqual(
            {placement.node_id for placement in expert_zero},
            {"owner-a", "owner-b"},
        )
        self.assertTrue(
            all(
                placement.content_id == "sha256:layer-0-expert-0"
                for placement in expert_zero
            )
        )
        for node_id, used in plan.projection.used_vram_bytes_by_node:
            self.assertLessEqual(used, 116, node_id)

    def test_parallel_makespan_includes_owner_compute(self) -> None:
        plan = plan_routing_aware_expert_placement(
            coordinator_id="root",
            experts=(_record(0, byte_size=1), _record(1, byte_size=1)),
            nodes=(
                _node("root", 7),
                _node("slow", 8, compute_ms=100.0),
                _node("fast-a", 7, compute_ms=1.0),
                _node("fast-b", 7, compute_ms=1.0),
            ),
            links=(
                _link("slow", rtt_ms=0.9),
                _link("fast-a", rtt_ms=1.0),
                _link("fast-b", rtt_ms=1.0),
            ),
            traces=(RoutingTraceSample(0, (0, 1), frequency=1.0),),
            activation_bytes_per_position=1,
            max_positions_per_wave=1,
        )

        route = plan.projection.trace_routes[0]
        self.assertNotIn("slow", route.owner_ids)
        self.assertEqual(set(route.owner_ids), {"fast-a", "fast-b"})
        self.assertLess(route.projected_link_ms_per_position, 3.0)

    def test_feasible_fragmented_sizes_get_exact_coverage(self) -> None:
        sizes = (4, 4, 6, 6)
        plan = plan_routing_aware_expert_placement(
            coordinator_id="root",
            experts=tuple(
                _record(expert, byte_size=size)
                for expert, size in enumerate(sizes)
            ),
            nodes=(
                _node("root", 4),
                _node("owner-a", 13),
                _node("owner-b", 13),
            ),
            links=(_link("owner-a"), _link("owner-b")),
            traces=tuple(
                RoutingTraceSample(0, (expert,), frequency=1.0)
                for expert in range(4)
            ),
            activation_bytes_per_position=1,
            max_positions_per_wave=1,
        )

        static_by_owner = {"owner-a": 0, "owner-b": 0}
        for placement in plan.placements:
            static_by_owner[placement.node_id] += sizes[placement.key.expert]
        self.assertEqual(static_by_owner, {"owner-a": 10, "owner-b": 10})
        self.assertEqual(
            dict(plan.projection.used_vram_bytes_by_node),
            {"owner-a": 13, "owner-b": 13},
        )

    def test_required_expert_without_owner_route_fails_closed(self) -> None:
        with self.assertRaisesRegex(
            RoutingAwarePlacementUnavailableError,
            "no available remote owner",
        ):
            plan_routing_aware_expert_placement(
                coordinator_id="root",
                experts=(_record(0),),
                nodes=(_node("root", 100), _node("owner-a", 100)),
                links=(),
                traces=(RoutingTraceSample(0, (0,), frequency=1.0),),
                activation_bytes_per_position=16,
                max_positions_per_wave=1,
            )

        with self.assertRaisesRegex(
            RoutingAwarePlacementUnavailableError,
            "missing expert records",
        ):
            plan_routing_aware_expert_placement(
                coordinator_id="root",
                experts=(_record(0),),
                nodes=(_node("root", 100), _node("owner-a", 100)),
                links=(_link("owner-a"),),
                traces=(RoutingTraceSample(0, (1,), frequency=1.0),),
                activation_bytes_per_position=16,
                max_positions_per_wave=1,
            )

        with self.assertRaisesRegex(
            RoutingAwarePlacementUnavailableError,
            "VRAM cannot cover",
        ):
            plan_routing_aware_expert_placement(
                coordinator_id="root",
                experts=(_record(0, byte_size=21),),
                nodes=(_node("root", 100), _node("owner-a", 68)),
                links=(_link("owner-a"),),
                traces=(RoutingTraceSample(0, (0,), frequency=1.0),),
                activation_bytes_per_position=16,
                max_positions_per_wave=1,
            )


if __name__ == "__main__":
    unittest.main()
