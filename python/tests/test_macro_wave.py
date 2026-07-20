from __future__ import annotations

from dataclasses import replace
import unittest

from distributed_runtime.macro_wave import (
    BranchIdentity,
    KVSnapshotState,
    KVVersion,
    MacroBranchState,
    MacroWaveCostConfig,
    MacroWaveCostController,
    MacroWaveCostObservation,
    MacroWaveState,
    MacroWaveTree,
    WaveIdentity,
    verify_greedy_exact,
)


class MacroWaveIdentityAndTreeTests(unittest.TestCase):
    def test_wave_and_branch_identities_are_stable_and_prefix_bound(self) -> None:
        first = WaveIdentity.for_prefix("request-7", 3, (10, 20), KVVersion(9))
        repeated = WaveIdentity.for_prefix("request-7", 3, (10, 20), KVVersion(9))
        changed_prefix = WaveIdentity.for_prefix("request-7", 3, (10, 21), KVVersion(9))
        changed_parent = WaveIdentity.for_prefix("request-7", 3, (10, 20), KVVersion(10))

        self.assertEqual(first, repeated)
        self.assertNotEqual(first, changed_prefix)
        self.assertNotEqual(first, changed_parent)
        root = BranchIdentity(first)
        branch = root.child(2).child(4)
        self.assertEqual(branch.path, (2, 4))
        self.assertEqual(branch.depth, 2)
        self.assertEqual(branch.parent, root.child(2))
        self.assertIn("/2.4", branch.key)

    def test_tree_rejects_identity_for_another_prefix_or_parent_version(self) -> None:
        identity = WaveIdentity.for_prefix("r", 0, (1, 2), KVVersion(5))
        with self.assertRaisesRegex(ValueError, "does not match"):
            MacroWaveTree(identity, (1, 3), KVVersion(5))
        with self.assertRaisesRegex(ValueError, "does not match"):
            MacroWaveTree(identity, (1, 2), KVVersion(6))

    def test_prefix_tree_reuses_shared_prefix_and_forks_parent_kv_versions(self) -> None:
        tree = MacroWaveTree.create("chat-1", 2, (1, 2), KVVersion(40))
        first_leaf = tree.ensure_path((10, 11))
        second_leaf = tree.ensure_path((10, 12))

        branches = tree.branches()
        self.assertEqual(len(branches), 4)  # root, shared 10, and two leaves
        shared = tree.children(tree.root_identity)[0]
        self.assertEqual(shared.token_id, 10)
        self.assertEqual(first_leaf.path, (0, 0))
        self.assertEqual(second_leaf.path, (0, 1))
        self.assertEqual(tree.candidate_tokens(second_leaf), (10, 12))
        self.assertEqual(tree.full_prefix(second_leaf), (1, 2, 10, 12))

        shared_branch = tree.branch(shared.identity)
        second_branch = tree.branch(second_leaf)
        self.assertEqual(shared_branch.parent_kv_version, KVVersion(40))
        self.assertEqual(second_branch.parent_kv_version, shared_branch.kv_version)
        self.assertEqual(
            tree.ledger.resolve_tokens(second_branch.kv_version),
            (1, 2, 10, 12),
        )
        # Branch snapshots contain one-token deltas instead of copied prefixes.
        self.assertEqual(tree.ledger.snapshot(second_branch.kv_version).delta_tokens, (12,))
        self.assertEqual(tree.ledger.physical_copy_count, 0)

    def test_duplicate_sibling_token_is_rejected(self) -> None:
        tree = MacroWaveTree.create("chat", 0, (), KVVersion(0))
        tree.add_branch(tree.root_identity, 5)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            tree.add_branch(tree.root_identity, 5)


class ExactGreedyMacroWaveTests(unittest.TestCase):
    def test_exact_greedy_accepts_one_path_and_logically_rolls_back_siblings(self) -> None:
        tree = MacroWaveTree.create("chat", 1, (1, 2), KVVersion(5))
        leaf_11 = tree.ensure_path((10, 11))
        leaf_12 = tree.ensure_path((10, 12))
        leaf_21 = tree.ensure_path((20, 21))
        branch_10 = leaf_12.parent
        self.assertIsNotNone(branch_10)
        assert branch_10 is not None

        acceptance = verify_greedy_exact(
            tree,
            {
                tree.root_identity: 10,
                branch_10: 12,
                leaf_12: 99,
            },
        )
        self.assertEqual(acceptance.accepted_branches, (branch_10, leaf_12))
        self.assertEqual(acceptance.accepted_tokens, (10, 12))
        self.assertEqual(acceptance.emitted_tokens, (10, 12, 99))
        self.assertEqual(acceptance.continuation_token, 99)
        self.assertEqual(acceptance.stop_reason, "candidate_exhausted")

        committed = tree.commit_greedy(acceptance)
        self.assertEqual(tree.state, MacroWaveState.COMMITTED)
        self.assertEqual(committed.kv_prefix_tokens, (1, 2, 10, 12))
        self.assertEqual(committed.visible_prefix_tokens, (1, 2, 10, 12, 99))
        self.assertEqual(committed.pending_token, 99)
        self.assertEqual(
            tree.ledger.resolve_tokens(committed.committed_kv_version),
            (1, 2, 10, 12),
        )
        self.assertEqual(tree.branch(branch_10).state, MacroBranchState.COMMITTED)
        self.assertEqual(tree.branch(leaf_12).state, MacroBranchState.COMMITTED)
        self.assertEqual(tree.branch(leaf_11).state, MacroBranchState.ROLLED_BACK)
        self.assertEqual(tree.branch(leaf_21).state, MacroBranchState.ROLLED_BACK)
        rolled_versions = {
            snapshot.version
            for snapshot in tree.ledger.snapshots()
            if snapshot.state is KVSnapshotState.ROLLED_BACK
        }
        self.assertEqual(rolled_versions, set(committed.rolled_back_kv_versions))
        self.assertEqual(tree.ledger.physical_copy_count, 0)
        with self.assertRaisesRegex(RuntimeError, "already committed"):
            tree.ensure_path((30,))

    def test_first_mismatch_emits_exact_target_token_and_commits_no_draft(self) -> None:
        tree = MacroWaveTree.create("chat", 0, (7,), KVVersion(2))
        tree.ensure_path((10, 11))
        tree.ensure_path((20, 21))

        acceptance = verify_greedy_exact(tree, {tree.root_identity: 42})
        self.assertEqual(acceptance.accepted_branches, ())
        self.assertEqual(acceptance.accepted_tokens, ())
        self.assertEqual(acceptance.emitted_tokens, (42,))
        self.assertEqual(acceptance.stop_reason, "candidate_mismatch")
        committed = tree.commit_greedy(acceptance)
        self.assertEqual(committed.committed_kv_version, KVVersion(2))
        self.assertEqual(committed.kv_prefix_tokens, (7,))
        self.assertEqual(committed.visible_prefix_tokens, (7, 42))
        self.assertEqual(len(committed.rolled_back_branch_ids), 4)

    def test_missing_target_argmax_on_selected_path_fails_closed(self) -> None:
        tree = MacroWaveTree.create("chat", 0, (), KVVersion(0))
        leaf = tree.ensure_path((10,))
        with self.assertRaisesRegex(ValueError, "missing target argmax"):
            verify_greedy_exact(tree, {tree.root_identity: 10})
        self.assertEqual(tree.state, MacroWaveState.OPEN)
        self.assertEqual(tree.branch(leaf).state, MacroBranchState.SPECULATIVE)

    def test_explicit_wave_rollback_restores_parent_without_physical_copy(self) -> None:
        tree = MacroWaveTree.create("chat", 4, (1, 2, 3), KVVersion(100))
        tree.ensure_path((8, 9, 10))
        tree.ensure_path((8, 12))
        rolled_back = tree.rollback()

        self.assertEqual(tree.state, MacroWaveState.ROLLED_BACK)
        self.assertEqual(rolled_back.restored_kv_version, KVVersion(100))
        self.assertEqual(rolled_back.restored_prefix_tokens, (1, 2, 3))
        self.assertEqual(len(rolled_back.rolled_back_branch_ids), 4)
        self.assertEqual(tree.ledger.head_version, KVVersion(100))
        self.assertEqual(tree.ledger.physical_copy_count, 0)
        with self.assertRaisesRegex(RuntimeError, "already rolled-back"):
            tree.rollback()

    def test_acceptance_from_another_wave_cannot_be_committed(self) -> None:
        first = MacroWaveTree.create("one", 0, (), KVVersion(0))
        second = MacroWaveTree.create("two", 0, (), KVVersion(0))
        first_leaf = first.ensure_path((5,))
        acceptance = verify_greedy_exact(
            first,
            {first.root_identity: 5, first_leaf: 6},
        )
        with self.assertRaisesRegex(ValueError, "another wave"):
            second.commit_greedy(acceptance)


def _observation(**overrides: object) -> MacroWaveCostObservation:
    values: dict[str, object] = {
        "route_rtt_ms": 100.0,
        "bandwidth_bytes_per_second": 100_000_000.0,
        "acceptance_by_width": ((1, 0.90), (2, 0.97)),
        "activation_bytes_per_node": 10_000,
        "metadata_bytes_per_node": 100,
        "kv_bytes_per_node": 20_000,
        "workspace_bytes_per_node": 10_000,
        "rollback_ms_per_node": 0.02,
        "target_base_ms": 10.0,
        "target_ms_per_node": 1.0,
        "vram_budget_bytes": 100_000_000,
        "vram_reserved_bytes": 0,
        "fixed_wire_bytes": 0,
    }
    values.update(overrides)
    return MacroWaveCostObservation(**values)  # type: ignore[arg-type]


class MacroWaveCostControllerTests(unittest.TestCase):
    def test_high_rtt_and_high_acceptance_choose_a_deep_narrow_wave(self) -> None:
        controller = MacroWaveCostController(
            MacroWaveCostConfig(
                candidate_depths=(1, 2, 4),
                candidate_widths=(1, 2),
                minimum_speedup=1.01,
            )
        )
        plan = controller.choose(_observation())

        self.assertTrue(plan.enabled)
        self.assertEqual(plan.reason, "beneficial")
        self.assertEqual((plan.depth, plan.width), (4, 1))
        self.assertGreater(plan.predicted_speedup or 0.0, 3.0)
        assert plan.estimate is not None
        self.assertAlmostEqual(
            plan.estimate.expected_accepted_tokens,
            0.9 + 0.9**2 + 0.9**3 + 0.9**4,
        )

    def test_width_is_chosen_from_measured_acceptance_not_assumed_independence(self) -> None:
        controller = MacroWaveCostController(
            MacroWaveCostConfig(
                candidate_depths=(2,),
                candidate_widths=(1, 2),
                minimum_speedup=1.0,
            )
        )
        plan = controller.choose(
            _observation(acceptance_by_width=((1, 0.20), (2, 0.95)))
        )
        self.assertTrue(plan.enabled)
        self.assertEqual((plan.depth, plan.width), (2, 2))

    def test_vram_budget_prunes_deeper_candidates(self) -> None:
        controller = MacroWaveCostController(
            MacroWaveCostConfig(
                candidate_depths=(1, 2, 4),
                candidate_widths=(1,),
                minimum_speedup=1.0,
            )
        )
        observation = _observation(
            acceptance_by_width=((1, 0.95),),
            vram_reserved_bytes=10_000,
            vram_budget_bytes=80_000,
        )
        plan = controller.choose(observation)
        self.assertTrue(plan.enabled)
        self.assertEqual((plan.depth, plan.width), (2, 1))
        assert plan.estimate is not None
        self.assertLessEqual(plan.estimate.vram_required_bytes, 80_000)
        too_large = [
            estimate
            for estimate in controller.estimates(observation)
            if estimate.depth == 4
        ][0]
        self.assertFalse(too_large.fits_vram)

    def test_no_candidate_that_fits_returns_insufficient_vram(self) -> None:
        controller = MacroWaveCostController(
            MacroWaveCostConfig(candidate_depths=(1,), candidate_widths=(1,))
        )
        plan = controller.choose(
            _observation(
                acceptance_by_width=((1, 0.9),),
                vram_reserved_bytes=75_000,
                vram_budget_bytes=80_000,
            )
        )
        self.assertFalse(plan.enabled)
        self.assertEqual(plan.reason, "insufficient_vram")
        self.assertIsNone(plan.estimate)

    def test_rollback_cost_can_switch_choice_from_wide_to_narrow(self) -> None:
        controller = MacroWaveCostController(
            MacroWaveCostConfig(
                candidate_depths=(2,),
                candidate_widths=(1, 2),
                minimum_speedup=1.0,
            )
        )
        base = _observation(acceptance_by_width=((1, 0.50), (2, 0.95)))
        cheap = controller.choose(base)
        expensive = controller.choose(replace(base, rollback_ms_per_node=100.0))

        self.assertEqual(cheap.width, 2)
        self.assertEqual(expensive.width, 1)
        assert cheap.estimate is not None and expensive.estimate is not None
        self.assertGreater(cheap.estimate.expected_rollback_nodes, 4.0)
        self.assertLess(expensive.estimate.expected_rollback_nodes, 2.0)

    def test_low_bandwidth_prefers_a_shallow_wave_because_bytes_scale_with_tree(self) -> None:
        controller = MacroWaveCostController(
            MacroWaveCostConfig(
                candidate_depths=(1, 4),
                candidate_widths=(1,),
                minimum_speedup=1.0,
            )
        )
        common = {
            "acceptance_by_width": ((1, 0.90),),
            "activation_bytes_per_node": 10_000_000,
            "metadata_bytes_per_node": 0,
        }
        fast = controller.choose(
            _observation(**common, bandwidth_bytes_per_second=1_000_000_000.0)
        )
        slow = controller.choose(
            _observation(**common, bandwidth_bytes_per_second=1_000_000.0)
        )
        self.assertEqual(fast.depth, 4)
        self.assertEqual(slow.depth, 1)
        assert slow.estimate is not None
        self.assertEqual(slow.estimate.wire_bytes, 10_000_000)

    def test_poor_acceptance_respects_minimum_speedup_gate(self) -> None:
        controller = MacroWaveCostController(
            MacroWaveCostConfig(
                candidate_depths=(1,),
                candidate_widths=(1,),
                minimum_speedup=1.10,
            )
        )
        plan = controller.choose(
            _observation(
                route_rtt_ms=0.0,
                acceptance_by_width=((1, 0.05),),
                activation_bytes_per_node=0,
                metadata_bytes_per_node=0,
                target_base_ms=0.0,
                target_ms_per_node=10.0,
            )
        )
        self.assertFalse(plan.enabled)
        self.assertEqual(plan.reason, "not_beneficial")
        self.assertLess(plan.predicted_speedup or 0.0, 1.10)

    def test_invalid_observations_and_configuration_fail_fast(self) -> None:
        with self.assertRaises(ValueError):
            MacroWaveCostConfig(candidate_depths=())
        with self.assertRaises(ValueError):
            MacroWaveCostConfig(candidate_widths=(0,))
        with self.assertRaises(ValueError):
            MacroWaveCostConfig(minimum_speedup=0.9)
        with self.assertRaises(ValueError):
            _observation(acceptance_by_width=((1, 0.5), (1, 0.6)))
        with self.assertRaises(ValueError):
            _observation(acceptance_by_width=((1, 1.1),))
        with self.assertRaises(ValueError):
            _observation(bandwidth_bytes_per_second=0.0)
        with self.assertRaises(ValueError):
            _observation(vram_reserved_bytes=101, vram_budget_bytes=100)


if __name__ == "__main__":
    unittest.main()
