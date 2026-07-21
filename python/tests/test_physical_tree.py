from __future__ import annotations

import unittest

from distributed_runtime.macro_wave import KVVersion, MacroWaveState
from distributed_runtime.macro_wave_adapter import branched_candidates_to_macro_wave
from distributed_runtime.physical_tree import (
    CancelCommand,
    EndCommand,
    ForkCommand,
    PhysicalTreeCoordinator,
    PhysicalTreeError,
    PromoteCommand,
    TombstoneDrain,
    TreeAbortPlan,
    TreeCommitPlan,
    TreePhase,
    TruncateCommand,
    UnknownVirtualRequestError,
    VerifyCommand,
)


PATHS = ((10, 11), (10, 12), (20, 21))


def _proposal(*, ordinal: int = 0):
    return branched_candidates_to_macro_wave(
        PATHS,
        request_id="physical-tree-test",
        ordinal=ordinal,
        base_prefix_tokens=(1, 2, 99),
        parent_kv_version=KVVersion(2),
        strategy="fixed-test",
    )


def _prepared(
    *,
    parent: int = 7,
    ids: tuple[int, ...] = (100, 101, 102),
    deadline: float = 100.0,
    ordinal: int = 0,
) -> tuple[PhysicalTreeCoordinator, object]:
    coordinator = PhysicalTreeCoordinator()
    wave = coordinator.prepare_wave(
        parent_request_id=parent,
        proposal=_proposal(ordinal=ordinal),
        base_kv_tokens=2,
        pending_token=99,
        inherited_step=5,
        virtual_request_ids=ids,
        deadline_at=deadline,
    )
    return coordinator, wave


def _issue_all(coordinator: PhysicalTreeCoordinator, parent: int = 7) -> None:
    forks = []
    while (command := coordinator.next_fork_command(parent)) is not None:
        forks.append(command)
    assert [command.request_id for command in forks] == [100, 101, 102]
    verifies = []
    while (command := coordinator.next_verify_command(parent, now=1.0)) is not None:
        verifies.append(command)
    assert [command.request_id for command in verifies] == [100, 101, 102]


class PhysicalTreePreparationTests(unittest.TestCase):
    def test_flat_leaves_use_caller_ids_same_step_and_deterministic_order(self) -> None:
        coordinator, wave = _prepared()

        self.assertEqual(wave.phase, TreePhase.PREPARED)
        fork_commands = tuple(
            command
            for _ in range(3)
            if (command := coordinator.next_fork_command(7)) is not None
        )
        self.assertEqual(
            fork_commands,
            (
                ForkCommand(100, 7),
                ForkCommand(101, 7),
                ForkCommand(102, 7),
            ),
        )
        self.assertIsNone(coordinator.next_fork_command(7))

        verify_commands = tuple(
            command
            for _ in range(3)
            if (command := coordinator.next_verify_command(7, now=1.0)) is not None
        )
        self.assertEqual(
            verify_commands,
            (
                VerifyCommand(100, 5, (99, 10, 11)),
                VerifyCommand(101, 5, (99, 10, 12)),
                VerifyCommand(102, 5, (99, 20, 21)),
            ),
        )
        self.assertIsNone(coordinator.next_verify_command(7, now=1.0))
        self.assertEqual(wave.phase, TreePhase.VERIFYING)

    def test_virtual_ids_are_global_monotonic_and_never_reused(self) -> None:
        coordinator, wave = _prepared()
        cancellation = coordinator.request_cancel(7)
        self.assertEqual(cancellation.commands, (CancelCommand(7),))
        coordinator.retire_wave(7)

        with self.assertRaisesRegex(ValueError, "globally monotonic"):
            coordinator.prepare_wave(
                parent_request_id=8,
                proposal=_proposal(ordinal=1),
                base_kv_tokens=2,
                pending_token=99,
                inherited_step=6,
                virtual_request_ids=(100, 101, 102),
                deadline_at=100.0,
            )
        self.assertEqual(wave.phase, TreePhase.ABORTED)

    def test_pending_token_and_physical_base_are_bound_to_visible_prefix(self) -> None:
        coordinator = PhysicalTreeCoordinator()
        with self.assertRaisesRegex(ValueError, "last visible"):
            coordinator.prepare_wave(
                parent_request_id=7,
                proposal=_proposal(),
                base_kv_tokens=2,
                pending_token=98,
                inherited_step=5,
                virtual_request_ids=(100, 101, 102),
                deadline_at=100.0,
            )

    def test_same_route_fallback_exists_only_before_first_fork(self) -> None:
        coordinator, wave = _prepared()
        plan = coordinator.abort_before_wire(7, "local capacity changed")
        self.assertTrue(plan.fallback_allowed)
        self.assertFalse(plan.route_fatal)
        self.assertEqual(plan.commands, ())
        self.assertEqual(wave.phase, TreePhase.ABORTED)
        self.assertEqual(coordinator.leaf_route, {})

        coordinator, _wave = _prepared()
        coordinator.next_fork_command(7)
        with self.assertRaisesRegex(PhysicalTreeError, "before first FORK"):
            coordinator.abort_before_wire(7, "too late")


class PhysicalTreeResolutionTests(unittest.TestCase):
    def test_complete_leaf_acceptance_selects_exact_carrier_without_truncate(self) -> None:
        coordinator, wave = _prepared()
        _issue_all(coordinator)

        self.assertIsNone(
            coordinator.accept_verify_result(
                100, step=5, target_tokens=(10, 12, 55), now=2.0
            )
        )
        self.assertIsNone(
            coordinator.accept_verify_result(
                102, step=5, target_tokens=(10, 22, 23), now=2.0
            )
        )
        plan = coordinator.accept_verify_result(
            101, step=5, target_tokens=(10, 12, 77), now=2.0
        )

        self.assertIsInstance(plan, TreeCommitPlan)
        assert isinstance(plan, TreeCommitPlan)
        self.assertEqual(plan.resolution.commit_tokens, (10, 12))
        self.assertEqual(plan.resolution.emitted_tokens, (10, 12, 77))
        self.assertEqual(plan.carrier_request_id, 101)
        self.assertEqual(plan.next_step, 6)
        self.assertEqual(
            plan.commands,
            (EndCommand(100), EndCommand(102), PromoteCommand(7, 101)),
        )
        self.assertFalse(any(isinstance(item, TruncateCommand) for item in plan.commands))
        self.assertEqual(wave.phase, TreePhase.CLEANING)

    def test_internal_mismatch_uses_lowest_carrier_and_exact_keep_formula(self) -> None:
        coordinator, _wave = _prepared()
        _issue_all(coordinator)
        coordinator.accept_verify_result(
            100, step=5, target_tokens=(10, 88, 55), now=2.0
        )
        coordinator.accept_verify_result(
            101, step=5, target_tokens=(10, 88, 77), now=2.0
        )
        plan = coordinator.accept_verify_result(
            102, step=5, target_tokens=(10, 22, 23), now=2.0
        )

        self.assertIsInstance(plan, TreeCommitPlan)
        assert isinstance(plan, TreeCommitPlan)
        self.assertEqual(plan.resolution.commit_tokens, (10,))
        self.assertEqual(plan.resolution.emitted_tokens, (10, 88))
        self.assertEqual(plan.carrier_request_id, 100)
        self.assertEqual(
            plan.commands,
            (
                EndCommand(101),
                EndCommand(102),
                PromoteCommand(7, 100),
                TruncateCommand(7, 4),  # B + pending + one accepted draft
            ),
        )

    def test_root_mismatch_uses_lowest_leaf_and_keeps_only_pending_token(self) -> None:
        coordinator, _wave = _prepared()
        _issue_all(coordinator)
        coordinator.accept_verify_result(
            100, step=5, target_tokens=(88, 12, 55), now=2.0
        )
        coordinator.accept_verify_result(
            101, step=5, target_tokens=(88, 12, 77), now=2.0
        )
        plan = coordinator.accept_verify_result(
            102, step=5, target_tokens=(88, 22, 23), now=2.0
        )

        self.assertIsInstance(plan, TreeCommitPlan)
        assert isinstance(plan, TreeCommitPlan)
        self.assertEqual(plan.resolution.accepted_draft_tokens, 0)
        self.assertEqual(plan.resolution.commit_tokens, ())
        self.assertEqual(plan.resolution.emitted_tokens, (88,))
        self.assertEqual(plan.carrier_request_id, 100)
        self.assertEqual(
            plan.commands,
            (
                EndCommand(101),
                EndCommand(102),
                PromoteCommand(7, 100),
                TruncateCommand(7, 3),  # B + pending; accepted draft count is zero
            ),
        )

    def test_sixty_four_leaves_share_one_inherited_and_next_step(self) -> None:
        paths = tuple((1_000 + index,) for index in range(64))
        proposal = branched_candidates_to_macro_wave(
            paths,
            request_id="physical-tree-64",
            ordinal=0,
            base_prefix_tokens=(1, 2, 99),
            parent_kv_version=KVVersion(2),
            strategy="fixed-64-test",
        )
        coordinator = PhysicalTreeCoordinator()
        ids = tuple(range(10_000, 10_064))
        wave = coordinator.prepare_wave(
            parent_request_id=7,
            proposal=proposal,
            base_kv_tokens=2,
            pending_token=99,
            inherited_step=5,
            virtual_request_ids=ids,
            deadline_at=100.0,
        )
        fork_ids = []
        while (command := coordinator.next_fork_command(7)) is not None:
            fork_ids.append(command.request_id)
        verify_commands = []
        while (command := coordinator.next_verify_command(7, now=1.0)) is not None:
            verify_commands.append(command)

        self.assertEqual(tuple(fork_ids), ids)
        self.assertEqual(len(verify_commands), 64)
        self.assertEqual({command.step for command in verify_commands}, {5})
        self.assertEqual({leaf.inherited_step for leaf in wave.ordered_leaves}, {5})

        selected = paths[31][0]
        plan = None
        for command in verify_commands:
            candidate = command.input_tokens[1]
            plan = coordinator.accept_verify_result(
                command.request_id,
                step=5,
                target_tokens=(selected, 20_000 + candidate),
                now=2.0,
            )
        self.assertIsInstance(plan, TreeCommitPlan)
        assert isinstance(plan, TreeCommitPlan)
        self.assertEqual(plan.resolution.commit_tokens, (selected,))
        self.assertEqual(plan.next_step, 6)
        self.assertEqual(wave.inherited_step, 5)

    def test_shared_prefix_contradiction_is_fatal_and_never_votes(self) -> None:
        coordinator, wave = _prepared()
        _issue_all(coordinator)
        coordinator.accept_verify_result(
            100, step=5, target_tokens=(10, 11, 55), now=2.0
        )
        coordinator.accept_verify_result(
            101, step=5, target_tokens=(10, 12, 77), now=2.0
        )
        plan = coordinator.accept_verify_result(
            102, step=5, target_tokens=(10, 22, 23), now=2.0
        )

        self.assertIsInstance(plan, TreeAbortPlan)
        assert isinstance(plan, TreeAbortPlan)
        self.assertTrue(plan.route_fatal)
        self.assertFalse(plan.fallback_allowed)
        self.assertEqual(plan.commands, ())
        self.assertIn("contradictory", plan.reason)
        self.assertEqual(wave.phase, TreePhase.ABORTED)
        self.assertEqual(wave.proposal.tree.state, MacroWaveState.ROLLED_BACK)
        self.assertEqual(coordinator.leaf_route, {})

    def test_bad_step_and_bad_shape_quarantine_the_route(self) -> None:
        for name, kwargs in (
            ("step", {"step": 6, "target_tokens": (10, 12, 55)}),
            ("shape", {"step": 5, "target_tokens": (10, 12)}),
        ):
            with self.subTest(name=name):
                coordinator, wave = _prepared()
                _issue_all(coordinator)
                plan = coordinator.accept_verify_result(100, now=2.0, **kwargs)
                self.assertIsInstance(plan, TreeAbortPlan)
                assert isinstance(plan, TreeAbortPlan)
                self.assertTrue(plan.route_fatal)
                self.assertEqual(plan.commands, ())
                self.assertEqual(wave.phase, TreePhase.ABORTED)

    def test_duplicate_physical_result_quarantines_instead_of_reusing_value(self) -> None:
        coordinator, wave = _prepared()
        _issue_all(coordinator)
        self.assertIsNone(
            coordinator.accept_verify_result(
                100, step=5, target_tokens=(10, 12, 55), now=2.0
            )
        )
        plan = coordinator.accept_verify_result(
            100, step=5, target_tokens=(10, 12, 55), now=2.1
        )
        self.assertIsInstance(plan, TreeAbortPlan)
        assert isinstance(plan, TreeAbortPlan)
        self.assertTrue(plan.route_fatal)
        self.assertEqual(plan.commands, ())
        self.assertEqual(wave.phase, TreePhase.ABORTED)


class PhysicalTreeCleanupTests(unittest.TestCase):
    def test_cleanup_confirmation_enforces_end_promote_truncate_order(self) -> None:
        coordinator, wave = _prepared()
        _issue_all(coordinator)
        coordinator.accept_verify_result(
            100, step=5, target_tokens=(10, 88, 55), now=2.0
        )
        coordinator.accept_verify_result(
            101, step=5, target_tokens=(10, 88, 77), now=2.0
        )
        plan = coordinator.accept_verify_result(
            102, step=5, target_tokens=(10, 22, 23), now=2.0
        )
        assert isinstance(plan, TreeCommitPlan)

        with self.assertRaisesRegex(PhysicalTreeError, "order violation"):
            coordinator.confirm_cleanup_command(7, plan.commands[2])
        self.assertEqual(wave.phase, TreePhase.CLEANING)
        coordinator.confirm_cleanup_command(7, plan.commands[0])
        coordinator.confirm_cleanup_command(7, plan.commands[1])
        self.assertEqual(wave.phase, TreePhase.PROMOTING)
        coordinator.confirm_cleanup_command(7, plan.commands[2])
        self.assertEqual(wave.phase, TreePhase.TRUNCATING)
        coordinator.confirm_cleanup_command(7, plan.commands[3])
        self.assertEqual(wave.phase, TreePhase.COMMITTED)
        self.assertEqual(coordinator.leaf_route, {})

    def test_cancel_during_cleanup_is_deferred_until_transition_is_complete(self) -> None:
        coordinator, wave = _prepared()
        _issue_all(coordinator)
        coordinator.accept_verify_result(
            100, step=5, target_tokens=(10, 88, 55), now=2.0
        )
        coordinator.accept_verify_result(
            101, step=5, target_tokens=(10, 88, 77), now=2.0
        )
        plan = coordinator.accept_verify_result(
            102, step=5, target_tokens=(10, 22, 23), now=2.0
        )
        assert isinstance(plan, TreeCommitPlan)

        cancellation = coordinator.request_cancel(7)
        repeated = coordinator.request_cancel(7)
        self.assertTrue(cancellation.deferred_until_commit)
        self.assertTrue(repeated.deferred_until_commit)
        self.assertEqual(cancellation.commands, ())
        self.assertEqual(repeated.commands, ())
        emitted_cancels = [*cancellation.commands, *repeated.commands]
        for command in plan.commands:
            followup = coordinator.confirm_cleanup_command(7, command)
            if followup is not None:
                emitted_cancels.append(followup)
        after_commit = coordinator.request_cancel(7)
        emitted_cancels.extend(after_commit.commands)
        self.assertEqual(emitted_cancels, [CancelCommand(7)])
        self.assertEqual(wave.phase, TreePhase.ABORTED)

    def test_retire_requires_no_routes_and_keeps_constant_size_high_water(self) -> None:
        coordinator, wave = _prepared()
        _issue_all(coordinator)
        coordinator.accept_verify_result(
            100, step=5, target_tokens=(10, 12, 55), now=2.0
        )
        coordinator.accept_verify_result(
            102, step=5, target_tokens=(10, 22, 23), now=2.0
        )
        plan = coordinator.accept_verify_result(
            101, step=5, target_tokens=(10, 12, 77), now=2.0
        )
        assert isinstance(plan, TreeCommitPlan)

        with self.assertRaisesRegex(PhysicalTreeError, "while it is cleaning"):
            coordinator.retire_wave(7)
        for command in plan.commands:
            coordinator.confirm_cleanup_command(7, command)
        self.assertEqual(coordinator.leaf_route, {})

        # Exercise the defensive zero-route condition even if a caller corrupts
        # a terminal index: retirement must not hide a live virtual route.
        coordinator.leaf_route[100] = (7, wave.virtual_routes[100])
        with self.assertRaisesRegex(PhysicalTreeError, "live virtual routes"):
            coordinator.retire_wave(7)
        coordinator.leaf_route.pop(100)
        retired = coordinator.retire_wave(7)
        self.assertIs(retired, wave)
        self.assertEqual(coordinator._virtual_id_high_water, 102)
        self.assertNotIn("_used_virtual_ids", vars(coordinator))

        coordinator.prepare_wave(
            parent_request_id=8,
            proposal=_proposal(ordinal=1),
            base_kv_tokens=2,
            pending_token=99,
            inherited_step=6,
            virtual_request_ids=(103, 104, 105),
            deadline_at=100.0,
        )
        self.assertEqual(coordinator._virtual_id_high_water, 105)
        self.assertNotIn("_used_virtual_ids", vars(coordinator))


class PhysicalTreeCancellationAndTimeoutTests(unittest.TestCase):
    def test_cancel_inflight_keeps_tombstones_until_every_return_is_drained(self) -> None:
        coordinator, wave = _prepared()
        _issue_all(coordinator)
        coordinator.accept_verify_result(
            100, step=5, target_tokens=(10, 12, 55), now=2.0
        )

        cancellation = coordinator.request_cancel(7)
        self.assertEqual(
            cancellation.commands,
            (
                CancelCommand(100),
                CancelCommand(101),
                CancelCommand(102),
                CancelCommand(7),
            ),
        )
        self.assertEqual(cancellation.draining_virtual_request_ids, (101, 102))
        self.assertEqual(wave.phase, TreePhase.DRAINING)

        first = coordinator.accept_verify_result(
            101, step=999, target_tokens=(), now=3.0
        )
        self.assertEqual(first, TombstoneDrain(7, 101, False))
        second = coordinator.accept_verify_result(
            102, step=999, target_tokens=(), now=3.0
        )
        self.assertEqual(second, TombstoneDrain(7, 102, True))
        self.assertEqual(wave.phase, TreePhase.ABORTED)
        self.assertEqual(coordinator.leaf_route, {})
        with self.assertRaises(UnknownVirtualRequestError):
            coordinator.accept_verify_result(
                102, step=5, target_tokens=(10, 22, 23), now=3.0
            )

    def test_repeated_cancel_while_draining_emits_no_duplicate_controls(self) -> None:
        coordinator, _wave = _prepared()
        _issue_all(coordinator)
        first = coordinator.request_cancel(7)
        second = coordinator.request_cancel(7)
        self.assertTrue(first.commands)
        self.assertEqual(second.commands, ())
        self.assertEqual(second.draining_virtual_request_ids, (100, 101, 102))

    def test_return_at_deadline_fails_inside_accept_before_consuming_it(self) -> None:
        coordinator, wave = _prepared(deadline=5.0)
        _issue_all(coordinator)

        plan = coordinator.accept_verify_result(
            101, step=5, target_tokens=(10, 12, 77), now=5.0
        )
        self.assertIsInstance(plan, TreeAbortPlan)
        assert isinstance(plan, TreeAbortPlan)
        self.assertTrue(plan.route_fatal)
        self.assertEqual(plan.oldest_virtual_request_id, 100)
        self.assertEqual(plan.commands, ())
        self.assertEqual(wave.phase, TreePhase.ABORTED)
        self.assertEqual(coordinator.leaf_route, {})

    def test_timeout_reports_oldest_outstanding_and_never_retries_controls(self) -> None:
        coordinator, wave = _prepared(deadline=5.0)
        while coordinator.next_fork_command(7) is not None:
            pass
        coordinator.next_verify_command(7, now=1.0)
        coordinator.next_verify_command(7, now=2.0)
        coordinator.next_verify_command(7, now=3.0)

        self.assertEqual(coordinator.check_timeouts(4.999), ())
        plans = coordinator.check_timeouts(5.0)
        self.assertEqual(len(plans), 1)
        self.assertTrue(plans[0].route_fatal)
        self.assertEqual(plans[0].oldest_virtual_request_id, 100)
        self.assertEqual(plans[0].commands, ())
        self.assertEqual(wave.phase, TreePhase.ABORTED)
        self.assertEqual(coordinator.check_timeouts(6.0), ())


if __name__ == "__main__":
    unittest.main()
