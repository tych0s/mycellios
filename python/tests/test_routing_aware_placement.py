from __future__ import annotations

import itertools
from types import SimpleNamespace
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
    def test_static_remaining_visits_each_replica_once(self) -> None:
        import distributed_runtime.routing_aware_placement as placement_module

        class CountingOwners(set[str]):
            def __init__(self, values: set[str]) -> None:
                super().__init__(values)
                self.iterations = 0
                self.contains_checks = 0

            def __iter__(self):  # type: ignore[no-untyped-def]
                for value in super().__iter__():
                    self.iterations += 1
                    yield value

            def __contains__(self, value: object) -> bool:
                self.contains_checks += 1
                return super().__contains__(value)

        class CountingPlacements(dict[ExpertKey, CountingOwners]):
            def __init__(self, values: dict[ExpertKey, CountingOwners]) -> None:
                super().__init__(values)
                self.items_calls = 0

            def items(self):  # type: ignore[no-untyped-def]
                self.items_calls += 1
                return super().items()

        owner_count = 96
        owner_ids = tuple(f"owner-{index:03d}" for index in range(owner_count))
        keys = tuple(ExpertKey(0, index) for index in range(owner_count))
        records = {
            key: _record(index, byte_size=index + 1)
            for index, key in enumerate(keys)
        }
        raw_owners = {
            key: {
                owner_ids[index],
                owner_ids[(index + 1) % owner_count],
                "non-candidate-owner",
            }
            for index, key in enumerate(keys)
        }
        placements = CountingPlacements(
            {key: CountingOwners(owners) for key, owners in raw_owners.items()}
        )
        context = SimpleNamespace(
            candidate_node_ids=owner_ids,
            records=records,
            nodes={
                owner_id: SimpleNamespace(
                    resident_vram_budget_bytes=100_000 + index,
                    reserved_vram_bytes=index,
                )
                for index, owner_id in enumerate(owner_ids)
            },
        )
        expected = {
            owner_id: (
                context.nodes[owner_id].resident_vram_budget_bytes
                - context.nodes[owner_id].reserved_vram_bytes
                - sum(
                    records[key].byte_size
                    for key, owners in raw_owners.items()
                    if owner_id in owners
                )
            )
            for owner_id in owner_ids
        }

        actual = placement_module._static_remaining(context, placements)

        self.assertEqual(actual, expected)
        self.assertEqual(placements.items_calls, 1)
        self.assertEqual(
            sum(owners.iterations for owners in placements.values()),
            3 * owner_count,
        )
        self.assertEqual(
            sum(owners.contains_checks for owners in placements.values()),
            0,
        )

    def test_static_remaining_preserves_unknown_placement_fail_closed(self) -> None:
        import distributed_runtime.routing_aware_placement as placement_module

        owner_id = "owner-a"
        context = SimpleNamespace(
            candidate_node_ids=(owner_id,),
            records={ExpertKey(0, 0): _record(0, byte_size=7)},
            nodes={
                owner_id: SimpleNamespace(
                    resident_vram_budget_bytes=100,
                    reserved_vram_bytes=10,
                )
            },
        )
        # As before, an unknown expert replicated only to an irrelevant owner
        # is ignored; claiming a real candidate owner must expose bad state.
        ignored = placement_module._static_remaining(
            context,
            {ExpertKey(9, 9): {"non-candidate-owner"}},
        )
        self.assertEqual(ignored, {owner_id: 90})
        with self.assertRaises(KeyError):
            placement_module._static_remaining(
                context,
                {ExpertKey(9, 9): {owner_id}},
            )

    def test_trace_owner_problem_scans_only_real_replica_memberships(self) -> None:
        import distributed_runtime.routing_aware_placement as placement_module

        class CountingCandidates(tuple[str, ...]):
            def __new__(cls, values: tuple[str, ...]):  # type: ignore[no-untyped-def]
                instance = super().__new__(cls, values)
                instance.iterations = 0
                return instance

            def __iter__(self):  # type: ignore[no-untyped-def]
                for value in super().__iter__():
                    self.iterations += 1
                    yield value

        class CountingOwners(set[str]):
            def __init__(self, values: set[str]) -> None:
                super().__init__(values)
                self.iterations = 0
                self.contains_checks = 0

            def __iter__(self):  # type: ignore[no-untyped-def]
                for value in super().__iter__():
                    self.iterations += 1
                    yield value

            def __contains__(self, value: object) -> bool:
                self.contains_checks += 1
                return super().__contains__(value)

        owner_count = 96
        raw_candidates = tuple(
            f"owner-{index:03d}" for index in range(owner_count)
        )
        candidate_ids = CountingCandidates(raw_candidates)
        keys = tuple(ExpertKey(0, index) for index in range(owner_count))
        records = {key: _record(index, byte_size=1) for index, key in enumerate(keys)}
        placements = {
            key: CountingOwners(
                {
                    raw_candidates[index],
                    raw_candidates[(index + 1) % owner_count],
                    "non-candidate-owner",
                }
            )
            for index, key in enumerate(keys)
        }
        context = SimpleNamespace(
            candidate_node_ids=candidate_ids,
            records=records,
            nodes={
                owner_id: SimpleNamespace(
                    resident_vram_budget_bytes=5,
                    reserved_vram_bytes=0,
                    expert_workspace_bytes_per_token=0,
                )
                for owner_id in raw_candidates
            },
            max_positions_per_wave=1,
            activation_bytes_per_position=1,
        )
        edge = placement_module._TraceEdge(
            layer=0,
            keys=keys,
            effective_positions=1.0,
        )

        owner_ids, domains, capacities = placement_module._trace_owner_problem(
            context,
            placements,
            edge,
        )

        self.assertEqual(owner_ids, raw_candidates)
        self.assertEqual(
            domains,
            tuple(
                tuple(
                    sorted(
                        (
                            raw_candidates[index],
                            raw_candidates[(index + 1) % owner_count],
                        )
                    )
                )
                for index in range(owner_count)
            ),
        )
        self.assertEqual(capacities, {owner_id: 1 for owner_id in raw_candidates})
        # Five owner-wide passes are independent of expert count: initialize
        # and emit static remaining, capacities, canonical order, and used
        # owners. Replica sets are visited once for resident bytes and once for
        # the expert domains.
        self.assertEqual(candidate_ids.iterations, 5 * owner_count)
        self.assertEqual(
            sum(owners.iterations for owners in placements.values()),
            6 * owner_count,
        )
        self.assertEqual(
            sum(owners.contains_checks for owners in placements.values()),
            0,
        )
        with self.assertRaisesRegex(
            RoutingAwarePlacementUnavailableError,
            "no complete exact route",
        ):
            placement_module._trace_owner_problem(
                context,
                {keys[0]: {"non-candidate-owner"}},
                placement_module._TraceEdge(
                    layer=0,
                    keys=(keys[0],),
                    effective_positions=1.0,
                ),
            )

    def test_iterative_matching_handles_topk1102_augmenting_chain(self) -> None:
        import distributed_runtime.routing_aware_placement as placement_module

        chain_length = 1_101
        owners = tuple(f"owner-{index}" for index in range(chain_length + 1))
        # The final expert starts at owner 0 and can reach the only free owner
        # only by traversing an alternating chain deeper than Python's default
        # recursion limit.  The matching implementation must remain iterative.
        candidates = tuple(
            (owners[index], owners[index + 1])
            for index in range(chain_length)
        ) + ((owners[0], owners[-1]),)
        capacities = {owner: 1 for owner in owners}
        context = SimpleNamespace(
            contact_cost_ms={
                owner: float(index + 1) for index, owner in enumerate(owners)
            }
        )

        assignment = placement_module._find_feasible_trace_assignment(
            context,
            candidates,
            capacities,
        )

        self.assertIsNotNone(assignment)
        assert assignment is not None
        self.assertEqual(len(assignment), len(candidates))
        self.assertTrue(
            all(owner in candidates[index] for index, owner in enumerate(assignment))
        )
        self.assertTrue(
            all(assignment.count(owner) <= capacities[owner] for owner in owners)
        )

    def test_dense_search_limits_apply_before_state_materialization(self) -> None:
        import distributed_runtime.routing_aware_placement as placement_module

        owner_count = 1_101
        owners = tuple(f"owner-{index}" for index in range(owner_count))
        candidates = tuple(
            (owners[index], owners[(index + 1) % owner_count])
            for index in range(owner_count)
        )
        capacities = {owner: 1 for owner in owners}
        dense_width = len(candidates) + len(owners)

        with (
            mock.patch.object(
                placement_module,
                "_TRACE_OWNER_EXACT_MAX_RESIDENT_STATE_CELLS",
                dense_width,
            ),
            mock.patch.object(
                placement_module,
                "_TRACE_OWNER_BEAM_MAX_RESIDENT_STATE_CELLS",
                dense_width,
            ),
        ):
            exact_state, exact_transitions, exact_completed = (
                placement_module._solve_trace_owners_exact(
                    SimpleNamespace(),
                    candidates,
                    capacities,
                )
            )
            beam_state, beam_transitions = placement_module._solve_trace_owners_beam(
                SimpleNamespace(),
                candidates,
                capacities,
            )

        self.assertIsNone(exact_state)
        self.assertEqual(exact_transitions, 0)
        self.assertFalse(exact_completed)
        self.assertIsNone(beam_state)
        self.assertEqual(beam_transitions, 0)

        # The independent resident-state counter also stops a branching layer
        # before it can append a third dense state.
        small_candidates = (("a", "b"), ("a", "b"))
        small_capacities = {"a": 2, "b": 2}
        with (
            mock.patch.object(
                placement_module,
                "_TRACE_OWNER_EXACT_MAX_RESIDENT_STATES",
                2,
            ),
            mock.patch.object(
                placement_module,
                "_TRACE_OWNER_EXACT_MAX_RESIDENT_STATE_CELLS",
                1_000,
            ),
        ):
            state, transitions, completed = placement_module._solve_trace_owners_exact(
                SimpleNamespace(),
                small_candidates,
                small_capacities,
            )
        with (
            mock.patch.object(
                placement_module,
                "_TRACE_OWNER_BEAM_MAX_RESIDENT_STATES",
                2,
            ),
            mock.patch.object(
                placement_module,
                "_TRACE_OWNER_BEAM_MAX_RESIDENT_STATE_CELLS",
                1_000,
            ),
        ):
            bounded_beam_state, bounded_beam_transitions = (
                placement_module._solve_trace_owners_beam(
                    SimpleNamespace(),
                    small_candidates,
                    small_capacities,
                )
            )
        self.assertIsNone(state)
        self.assertEqual(transitions, 2)
        self.assertFalse(completed)
        self.assertIsNone(bounded_beam_state)
        self.assertEqual(bounded_beam_transitions, 2)

    def test_wide_route_skips_dense_search_by_cumulative_copy_budget(self) -> None:
        import distributed_runtime.routing_aware_placement as placement_module

        expert_count = 1_200
        owners = tuple(f"owner-{index:04d}" for index in range(expert_count))
        candidates = tuple(
            tuple(sorted((owners[index], owners[(index + 1) % expert_count])))
            for index in range(expert_count)
        )
        capacities = {owner: 1 for owner in owners}
        copied_transition_limit = placement_module._trace_owner_transition_limit(
            expert_count=expert_count,
            owner_count=len(owners),
            max_transitions=placement_module._TRACE_OWNER_EXACT_MAX_TRANSITIONS,
            max_copied_state_cells=(
                placement_module._TRACE_OWNER_EXACT_MAX_COPIED_STATE_CELLS
            ),
        )
        self.assertLess(copied_transition_limit, expert_count)

        exact_state, exact_transitions, exact_completed = (
            placement_module._solve_trace_owners_exact(
                SimpleNamespace(),
                candidates,
                capacities,
            )
        )
        beam_state, beam_transitions = placement_module._solve_trace_owners_beam(
            SimpleNamespace(),
            candidates,
            capacities,
        )
        self.assertIsNone(exact_state)
        self.assertEqual(exact_transitions, 0)
        self.assertFalse(exact_completed)
        self.assertIsNone(beam_state)
        self.assertEqual(beam_transitions, 0)

        nodes = {
            owner: SimpleNamespace(expert_compute_ms_per_token=0.01)
            for owner in owners
        }
        nodes["root"] = SimpleNamespace(
            aggregate_egress_bytes_per_ms=1_000_000.0,
            aggregate_ingress_bytes_per_ms=1_000_000.0,
        )
        context = SimpleNamespace(
            coordinator_id="root",
            activation_bytes_per_position=1,
            nodes=nodes,
            contact_cost_ms={
                owner: float(index + 1) for index, owner in enumerate(owners)
            },
            fixed_contact_cost_ms={owner: 0.1 for owner in owners},
            per_expert_transfer_ms={owner: 0.01 for owner in owners},
            shared_input_transfer_ms={owner: 0.01 for owner in owners},
            per_expert_output_transfer_ms={owner: 0.01 for owner in owners},
            row_index_transfer_ms={owner: 0.001 for owner in owners},
        )
        edge = placement_module._TraceEdge(
            layer=0,
            keys=tuple(range(expert_count)),
            effective_positions=1.0,
        )
        with mock.patch.object(
            placement_module,
            "_trace_owner_problem",
            return_value=(owners, candidates, capacities),
        ):
            selection = placement_module._select_trace_owner_route(
                context,
                {},
                edge,
            )

        self.assertEqual(selection.strategy, "bounded-beam-local-search")
        self.assertFalse(selection.optimal)
        self.assertEqual(selection.states_evaluated, 1)
        assignment = tuple(owner for _index, owner in selection.state.assignments)
        self.assertEqual(len(assignment), expert_count)
        self.assertTrue(
            all(owner in candidates[index] for index, owner in enumerate(assignment))
        )
        self.assertTrue(
            all(assignment.count(owner) <= capacities[owner] for owner in owners)
        )

    def test_remote_route_does_not_charge_owner_workspace_to_coordinator(self) -> None:
        plan = plan_routing_aware_expert_placement(
            coordinator_id="root",
            experts=(_record(0, byte_size=1),),
            nodes=(
                MeshNodeProfile(
                    node_id="root",
                    resident_vram_budget_bytes=3,
                    reserved_vram_bytes=0,
                    expert_compute_ms_per_token=0.1,
                    expert_workspace_bytes_per_token=100,
                ),
                _node("owner-a", 4),
            ),
            links=(_link("owner-a"),),
            traces=(RoutingTraceSample(0, (0,), frequency=1.0),),
            activation_bytes_per_position=1,
            max_positions_per_wave=1,
        )

        self.assertEqual(plan.projection.used_vram_bytes_by_node, (("owner-a", 4),))

    def test_exact_count_dp_matches_small_bruteforce_owner_search(self) -> None:
        import distributed_runtime.routing_aware_placement as placement_module

        records = tuple(_record(expert, byte_size=1) for expert in range(3))
        context = placement_module._build_context(
            coordinator_id="root",
            experts=records,
            nodes=(
                _node("root", 100),
                _node("owner-a", 100, compute_ms=0.1),
                _node("owner-b", 100, compute_ms=0.2),
                _node("owner-c", 100, compute_ms=0.3),
            ),
            links=(
                _link("owner-a", rtt_ms=1.0),
                _link("owner-b", rtt_ms=0.8),
                _link("owner-c", rtt_ms=0.6),
            ),
            traces=(RoutingTraceSample(0, (0, 1, 2), frequency=1.0),),
            activation_bytes_per_position=4,
            max_positions_per_wave=1,
            candidate_node_ids=None,
        )
        edge = context.edges[0]
        placements = {
            record.key: {"owner-a", "owner-b", "owner-c"}
            for record in records
        }
        selection = placement_module._select_trace_owner_route(
            context,
            placements,
            edge,
        )
        _owner_ids, candidates, capacities = placement_module._trace_owner_problem(
            context,
            placements,
            edge,
        )
        brute_states = []
        for assignment in itertools.product(*candidates):
            if any(
                assignment.count(owner) > capacity
                for owner, capacity in capacities.items()
            ):
                continue
            brute_states.append(
                placement_module._build_trace_route_state(context, assignment)
            )
        brute = min(
            brute_states,
            key=lambda state: placement_module._route_state_rank(context, state),
        )

        self.assertEqual(selection.state, brute)
        self.assertEqual(selection.strategy, "exact-count-dp")
        self.assertTrue(selection.optimal)
        self.assertGreater(selection.states_evaluated, 0)

    def test_topk18_single_owner_uses_constant_state_fast_path(self) -> None:
        import distributed_runtime.routing_aware_placement as placement_module

        with (
            mock.patch.object(
                placement_module,
                "_solve_trace_owners_exact",
                side_effect=AssertionError("exact DP must not run"),
            ),
            mock.patch.object(
                placement_module,
                "_solve_trace_owners_heuristic",
                side_effect=AssertionError("heuristic must not run"),
            ),
        ):
            plan = plan_routing_aware_expert_placement(
                coordinator_id="root",
                experts=tuple(_record(expert, byte_size=1) for expert in range(18)),
                nodes=(_node("root", 54), _node("owner-a", 72)),
                links=(_link("owner-a"),),
                traces=(RoutingTraceSample(0, tuple(range(18)), frequency=1.0),),
                activation_bytes_per_position=1,
                max_positions_per_wave=1,
            )

        route = plan.projection.trace_routes[0]
        self.assertEqual(route.owner_selection_strategy, "single-owner-linear")
        self.assertTrue(route.owner_selection_optimal)
        self.assertEqual(route.owner_selection_states_evaluated, 1)
        self.assertEqual(route.owner_ids, ("owner-a",))
        self.assertEqual(
            route.owner_for_expert,
            tuple((expert, "owner-a") for expert in range(18)),
        )

    def test_bounded_owner_search_reports_non_optimal_strategy(self) -> None:
        import distributed_runtime.routing_aware_placement as placement_module

        records = tuple(_record(expert, byte_size=1) for expert in range(4))
        context = placement_module._build_context(
            coordinator_id="root",
            experts=records,
            nodes=(
                _node("root", 100),
                _node("owner-a", 100),
                _node("owner-b", 100),
            ),
            links=(_link("owner-a"), _link("owner-b")),
            traces=(RoutingTraceSample(0, (0, 1, 2, 3), frequency=1.0),),
            activation_bytes_per_position=1,
            max_positions_per_wave=1,
            candidate_node_ids=None,
        )
        edge = context.edges[0]
        placements = {
            record.key: {"owner-a", "owner-b"} for record in records
        }
        with mock.patch.object(
            placement_module,
            "_TRACE_OWNER_EXACT_MAX_TRANSITIONS",
            1,
        ):
            first = placement_module._select_trace_owner_route(
                context,
                placements,
                edge,
            )
            second = placement_module._select_trace_owner_route(
                context,
                placements,
                edge,
            )

        self.assertEqual(first, second)
        self.assertEqual(first.strategy, "bounded-beam-local-search")
        self.assertFalse(first.optimal)
        self.assertEqual(len(first.state.assignments), 4)

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

    def test_transport_mode_is_optimized_with_shared_coordinator_nic(self) -> None:
        # Coalescing is faster in isolation because it stages one shared input,
        # but for this deliberately tiny activation its row map increases the
        # request. The shared egress NIC makes v1 the globally faster mode:
        # max(32.03 ms owner, 40 ms NIC) rather than 60 ms NIC.
        plan = plan_routing_aware_expert_placement(
            coordinator_id="root",
            experts=(
                _record(0, byte_size=1),
                _record(1, byte_size=1),
            ),
            nodes=(
                _node("root", 24, egress_mbps=0.0016),
                _node("owner-a", 26),
            ),
            links=(
                MeshLinkProfile(
                    from_node="root",
                    to_node="owner-a",
                    round_trip_ms=0.01,
                    bandwidth_mbps=1_000_000.0,
                    host_device_staging_gbytes_per_second=0.000001,
                ),
            ),
            traces=(RoutingTraceSample(0, (0, 1), frequency=1.0),),
            activation_bytes_per_position=4,
            max_positions_per_wave=1,
        )

        route = plan.projection.trace_routes[0]
        self.assertEqual(route.owner_selection_strategy, "single-owner-linear")
        self.assertTrue(route.owner_selection_optimal)
        self.assertEqual(
            route.transport_mode_by_owner,
            (("owner-a", "rpc-v1"),),
        )
        self.assertEqual(route.projected_link_ms_per_position, 40.0)
        self.assertEqual(route.coalesced_exact_activation_bytes_per_position, 16)
        self.assertEqual(route.coalesced_exact_row_index_bytes_per_position, 0)

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
