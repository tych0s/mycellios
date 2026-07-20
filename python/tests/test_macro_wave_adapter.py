from __future__ import annotations

from dataclasses import dataclass
import unittest

from distributed_runtime.engine import _resolve_verified_tokens
from distributed_runtime.macro_wave import KVVersion, MacroBranchState
from distributed_runtime.macro_wave_adapter import (
    ContinuationKind,
    branched_candidates_to_macro_wave,
    linear_draft_to_macro_wave,
    prepare_linear_macro_wave,
    record_linear_resolution,
    resolve_linear_macro_wave,
    resolve_macro_wave,
)
from distributed_runtime.speculation import (
    AdaptiveSpeculationConfig,
    AdaptiveSpeculationController,
)


@dataclass
class _FixedDraftProvider:
    tokens: tuple[int, ...]
    max_draft_tokens: int = 8
    strategy: str = "fixed-test"

    def draft(
        self,
        token_history: object,
        max_tokens: int | None = None,
    ) -> tuple[int, ...]:
        del token_history
        limit = self.max_draft_tokens if max_tokens is None else max_tokens
        return self.tokens[:limit]


def _controller(*, verification_ready: bool) -> AdaptiveSpeculationController:
    controller = AdaptiveSpeculationController(
        AdaptiveSpeculationConfig(
            max_draft_tokens=4,
            candidate_sizes=(2,),
            min_token_history=0,
            min_classic_observations=1,
            min_verify_observations=1,
            minimum_speedup=1.0,
        )
    )
    controller.record_classic(
        latency_seconds=1.0,
        transferred_bytes=100,
    )
    if verification_ready:
        controller.record_verification(
            proposed_tokens=2,
            accepted_tokens=2,
            latency_seconds=0.1,
            transferred_bytes=100,
        )
    return controller


class LinearMacroWaveAdapterTests(unittest.TestCase):
    def test_linear_draft_becomes_exact_width_one_wave(self) -> None:
        proposal = linear_draft_to_macro_wave(
            (10, 11, 12),
            request_id="chat-7",
            ordinal=3,
            base_prefix_tokens=(1, 2),
            parent_kv_version=KVVersion(20),
            strategy="ngram",
        )

        self.assertTrue(proposal.is_linear)
        self.assertEqual(proposal.width, 1)
        self.assertEqual(proposal.max_depth, 3)
        self.assertEqual(proposal.linear_tokens, (10, 11, 12))
        self.assertEqual(proposal.strategy, "ngram")
        self.assertEqual(
            proposal.prefixes(),
            ((), (10,), (10, 11), (10, 11, 12)),
        )
        leaf = proposal.branch_for_prefix((10, 11, 12))
        self.assertEqual(
            proposal.tree.ledger.resolve_tokens(proposal.tree.branch(leaf).kv_version),
            (1, 2, 10, 11, 12),
        )

    def test_resolution_is_token_exact_with_existing_greedy_acceptance(self) -> None:
        draft = (10, 11, 12)
        cases = (
            ("first mismatch", (90, 91, 92, 93)),
            ("middle mismatch", (10, 90, 92, 93)),
            ("last mismatch", (10, 11, 90, 93)),
            ("all accepted", (10, 11, 12, 93)),
        )
        for ordinal, (name, targets) in enumerate(cases):
            with self.subTest(name=name):
                legacy_accepted, legacy_emitted = _resolve_verified_tokens(
                    draft,
                    targets,
                )
                proposal = linear_draft_to_macro_wave(
                    draft,
                    request_id="equivalence",
                    ordinal=ordinal,
                    base_prefix_tokens=(1, 2),
                    parent_kv_version=KVVersion(50 + ordinal),
                )
                result = resolve_linear_macro_wave(proposal, targets)

                self.assertEqual(result.accepted_draft_tokens, legacy_accepted)
                self.assertEqual(result.commit_tokens, draft[:legacy_accepted])
                self.assertEqual(result.truncate_draft_to, legacy_accepted)
                self.assertEqual(result.emitted_tokens, legacy_emitted)
                self.assertEqual(
                    result.truncate_required,
                    legacy_accepted < len(draft),
                )
                self.assertEqual(
                    result.commit.kv_prefix_tokens,
                    (1, 2, *draft[:legacy_accepted]),
                )
                self.assertEqual(
                    result.commit.visible_prefix_tokens,
                    (1, 2, *legacy_emitted),
                )
                if legacy_accepted == len(draft):
                    self.assertEqual(result.continuation_kind, ContinuationKind.BONUS)
                    self.assertEqual(result.bonus_token, targets[-1])
                    self.assertIsNone(result.correction_token)
                else:
                    self.assertEqual(
                        result.continuation_kind,
                        ContinuationKind.CORRECTION,
                    )
                    self.assertEqual(
                        result.correction_token,
                        targets[legacy_accepted],
                    )
                    self.assertIsNone(result.bonus_token)

    def test_provider_and_controller_select_a_legacy_linear_wave(self) -> None:
        controller = _controller(verification_ready=True)
        preparation = prepare_linear_macro_wave(
            _FixedDraftProvider((10, 11, 12, 13)),
            controller,
            (1, 2, 3),
            request_id=7,
            ordinal=2,
            parent_kv_version=KVVersion(8),
            max_tokens=4,
        )

        self.assertTrue(preparation.decision.enabled)
        self.assertTrue(preparation.enabled)
        self.assertFalse(preparation.is_probe)
        self.assertEqual(preparation.available_draft_tokens, (10, 11, 12, 13))
        self.assertEqual(preparation.selected_draft_tokens, (10, 11))
        assert preparation.proposal is not None
        self.assertEqual(preparation.proposal.width, 1)
        self.assertEqual(preparation.proposal.linear_tokens, (10, 11))
        self.assertEqual(preparation.proposal.strategy, "fixed-test")

    def test_disabled_decision_stays_disabled_and_probe_is_explicit(self) -> None:
        provider = _FixedDraftProvider((10, 11, 12, 13))

        disabled = prepare_linear_macro_wave(
            provider,
            _controller(verification_ready=False),
            (1, 2, 3),
            request_id="disabled",
            ordinal=0,
        )
        self.assertFalse(disabled.decision.enabled)
        self.assertEqual(disabled.decision.reason, "verification_warmup")
        self.assertFalse(disabled.enabled)
        self.assertFalse(disabled.is_probe)
        self.assertIsNone(disabled.proposal)

        probe = prepare_linear_macro_wave(
            provider,
            _controller(verification_ready=False),
            (1, 2, 3),
            request_id="probe",
            ordinal=0,
            allow_probe=True,
        )
        self.assertFalse(probe.decision.enabled)
        self.assertTrue(probe.enabled)
        self.assertTrue(probe.is_probe)
        self.assertEqual(probe.selected_draft_tokens, (10, 11))

    def test_linear_result_records_through_unchanged_controller_api(self) -> None:
        controller = AdaptiveSpeculationController(
            AdaptiveSpeculationConfig(
                max_draft_tokens=3,
                candidate_sizes=(3,),
                min_token_history=0,
                min_classic_observations=1,
                min_verify_observations=1,
            )
        )
        proposal = linear_draft_to_macro_wave(
            (10, 11, 12),
            request_id="record",
            ordinal=0,
            base_prefix_tokens=(1,),
        )
        result = resolve_linear_macro_wave(proposal, (10, 11, 99, 100))
        record_linear_resolution(
            controller,
            proposal,
            result,
            latency_seconds=0.25,
            transferred_bytes=400,
        )

        stats = controller.stats()
        self.assertEqual(stats.verification_observations, 1)
        self.assertEqual(stats.proposed_tokens, 3)
        self.assertEqual(stats.accepted_tokens, 2)
        self.assertEqual(stats.emitted_tokens, 3)


class BranchedMacroWaveAdapterTests(unittest.TestCase):
    def test_branched_candidates_share_prefix_and_commit_only_exact_path(self) -> None:
        proposal = branched_candidates_to_macro_wave(
            ((10, 11), (10, 12), (20, 21)),
            request_id="tree",
            ordinal=1,
            base_prefix_tokens=(1, 2),
            parent_kv_version=KVVersion(5),
            strategy="top-k-tree",
        )
        self.assertFalse(proposal.is_linear)
        self.assertEqual(proposal.width, 2)
        self.assertEqual(len(proposal.tree.branches()), 6)

        result = resolve_macro_wave(
            proposal,
            {
                (): 10,
                (10,): 12,
                (10, 12): 99,
            },
        )
        self.assertEqual(result.commit_tokens, (10, 12))
        self.assertEqual(result.emitted_tokens, (10, 12, 99))
        self.assertEqual(result.continuation_kind, ContinuationKind.BONUS)
        self.assertEqual(result.bonus_token, 99)
        self.assertFalse(result.truncate_required)
        committed = set(result.commit.committed_branch_ids)
        for branch in proposal.tree.branches():
            if branch.identity == proposal.tree.root_identity:
                continue
            expected = (
                MacroBranchState.COMMITTED
                if branch.identity in committed
                else MacroBranchState.ROLLED_BACK
            )
            self.assertEqual(proposal.tree.branch(branch.identity).state, expected)

    def test_branched_mismatch_names_correction_and_truncate_depth(self) -> None:
        proposal = branched_candidates_to_macro_wave(
            ((10, 11), (10, 12), (20, 21)),
            request_id="tree-mismatch",
            ordinal=0,
            base_prefix_tokens=(),
        )
        result = resolve_macro_wave(
            proposal,
            {
                (): 10,
                (10,): 77,
            },
        )

        self.assertEqual(result.commit_tokens, (10,))
        self.assertEqual(result.truncate_draft_to, 1)
        self.assertTrue(result.truncate_required)
        self.assertEqual(result.continuation_kind, ContinuationKind.CORRECTION)
        self.assertEqual(result.correction_token, 77)
        self.assertIsNone(result.bonus_token)
        self.assertEqual(result.emitted_tokens, (10, 77))

    def test_invalid_or_ambiguous_proposals_fail_closed(self) -> None:
        common = {
            "request_id": "invalid",
            "ordinal": 0,
            "base_prefix_tokens": (),
        }
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            linear_draft_to_macro_wave((), **common)
        with self.assertRaisesRegex(ValueError, "duplicates"):
            branched_candidates_to_macro_wave(((1, 2), (1, 2)), **common)
        with self.assertRaisesRegex(ValueError, "cannot prefix"):
            branched_candidates_to_macro_wave(((1,), (1, 2)), **common)

        proposal = linear_draft_to_macro_wave((1, 2), **common)
        with self.assertRaisesRegex(ValueError, "plus bonus"):
            resolve_linear_macro_wave(proposal, (1, 2))

        proposal = branched_candidates_to_macro_wave(((1, 2), (3, 4)), **common)
        with self.assertRaisesRegex(ValueError, "unknown candidate prefix"):
            resolve_macro_wave(proposal, {(99,): 1})

    def test_missing_selected_prefix_target_fails_without_committing(self) -> None:
        proposal = branched_candidates_to_macro_wave(
            ((1, 2), (3, 4)),
            request_id="missing",
            ordinal=0,
            base_prefix_tokens=(),
        )
        with self.assertRaisesRegex(ValueError, "missing target argmax"):
            resolve_macro_wave(proposal, {(): 1})
        for branch in proposal.tree.branches()[1:]:
            self.assertEqual(branch.state, MacroBranchState.SPECULATIVE)


if __name__ == "__main__":
    unittest.main()
