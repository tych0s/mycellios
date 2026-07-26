from __future__ import annotations

import math
import unittest

from distributed_runtime.speculation import (
    MAX_DRAFT_TOKENS,
    MAX_TREE_DRAFT_BRANCHES,
    AdaptiveSpeculationConfig,
    AdaptiveSpeculationController,
    CandidateEstimate,
    DraftProvider,
    NgramDraftProvider,
    NgramTreeDraftProvider,
    TreeDraftProvider,
)


class NgramDraftProviderTests(unittest.TestCase):
    def test_satisfies_runtime_protocol(self) -> None:
        self.assertIsInstance(NgramDraftProvider(), DraftProvider)

    def test_finds_longest_repeated_suffix_and_historical_continuation(self) -> None:
        provider = NgramDraftProvider(max_draft_tokens=4, min_match_tokens=2)
        history = [90, 1, 2, 3, 4, 70, 1, 2, 3]
        self.assertEqual(provider.draft(history), (4, 70, 1, 2))

    def test_searches_the_complete_history_not_a_recent_window(self) -> None:
        provider = NgramDraftProvider(max_draft_tokens=3, max_match_tokens=4)
        history = [11, 12, 13, 14, 15] + list(range(100, 400)) + [11, 12, 13]
        self.assertEqual(provider.draft(history), (14, 15, 100))

    def test_longer_match_wins_over_a_more_recent_short_match(self) -> None:
        provider = NgramDraftProvider(max_draft_tokens=2, min_match_tokens=2)
        history = [1, 2, 3, 8, 2, 3, 9, 1, 2, 3]
        self.assertEqual(provider.draft(history), (8, 2))

    def test_equal_width_tie_uses_most_recent_occurrence(self) -> None:
        provider = NgramDraftProvider(max_draft_tokens=2, min_match_tokens=2)
        history = [1, 2, 7, 1, 2, 8, 1, 2]
        self.assertEqual(provider.draft(history), (8, 1))

    def test_returns_empty_without_a_repeated_suffix(self) -> None:
        provider = NgramDraftProvider()
        self.assertEqual(provider.draft([]), ())
        self.assertEqual(provider.draft([1, 2]), ())
        self.assertEqual(provider.draft([1, 2, 3, 4]), ())

    def test_respects_provider_and_per_call_limits_without_mutating_history(self) -> None:
        provider = NgramDraftProvider(max_draft_tokens=3)
        history = [1, 2, 3, 4, 1, 2]
        original = list(history)
        self.assertEqual(provider.draft(history), (3, 4, 1))
        self.assertEqual(provider.draft(history, max_tokens=2), (3, 4))
        self.assertEqual(provider.draft(history, max_tokens=99), (3, 4, 1))
        self.assertEqual(provider.draft(history, max_tokens=0), ())
        self.assertEqual(history, original)

    def test_supports_every_allowed_maximum(self) -> None:
        for maximum in range(1, MAX_DRAFT_TOKENS + 1):
            with self.subTest(maximum=maximum):
                provider = NgramDraftProvider(
                    max_draft_tokens=maximum, min_match_tokens=1
                )
                self.assertLessEqual(
                    len(provider.draft([5, *range(30), 5])), maximum
                )

    def test_rejects_invalid_configuration_and_inputs(self) -> None:
        for invalid in (0, MAX_DRAFT_TOKENS + 1, True, 1.5):
            with self.subTest(max_draft_tokens=invalid):
                with self.assertRaises(ValueError):
                    NgramDraftProvider(max_draft_tokens=invalid)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            NgramDraftProvider(min_match_tokens=0)
        with self.assertRaises(ValueError):
            NgramDraftProvider(min_match_tokens=4, max_match_tokens=3)
        with self.assertRaises(ValueError):
            NgramDraftProvider(strategy="draft-model")

        provider = NgramDraftProvider()
        for invalid_history in ("123", [1, -1], [1, True], [1, 2.5]):
            with self.subTest(history=invalid_history):
                with self.assertRaises(ValueError):
                    provider.draft(invalid_history)  # type: ignore[arg-type]
        for invalid_limit in (-1, True, 1.5):
            with self.subTest(limit=invalid_limit):
                with self.assertRaises(ValueError):
                    provider.draft([1, 2, 1, 2], invalid_limit)  # type: ignore[arg-type]


class NgramTreeDraftProviderTests(unittest.TestCase):
    def test_satisfies_tree_runtime_protocol(self) -> None:
        self.assertIsInstance(NgramTreeDraftProvider(), TreeDraftProvider)

    def test_returns_several_ranked_historical_continuations(self) -> None:
        provider = NgramTreeDraftProvider(
            max_draft_tokens=3,
            max_branches=3,
            max_match_tokens=2,
        )
        history = [
            1,
            2,
            10,
            11,
            1,
            2,
            20,
            21,
            1,
            2,
            30,
            31,
            1,
            2,
        ]
        self.assertEqual(
            provider.draft_paths(history),
            ((30, 31, 1), (20, 21, 1), (10, 11, 1)),
        )

    def test_shared_prefixes_are_retained_as_distinct_leaf_paths(self) -> None:
        provider = NgramTreeDraftProvider(
            max_draft_tokens=3,
            max_branches=2,
            max_match_tokens=2,
        )
        history = [1, 2, 10, 11, 9, 1, 2, 10, 12, 8, 1, 2]
        paths = provider.draft_paths(history)
        self.assertEqual(paths, ((10, 12, 8), (10, 11, 9)))
        self.assertEqual(paths[0][0], paths[1][0])

    def test_removes_redundant_prefix_leaf(self) -> None:
        provider = NgramTreeDraftProvider(
            max_draft_tokens=2,
            max_branches=4,
            min_match_tokens=1,
            max_match_tokens=1,
        )
        self.assertEqual(provider.draft_paths([1, 1, 1]), ((1, 1),))

    def test_longer_replacement_inherits_removed_prefix_priority(self) -> None:
        provider = NgramTreeDraftProvider(
            max_draft_tokens=2,
            max_branches=1,
            min_match_tokens=1,
            max_match_tokens=1,
        )
        # The most recent match proposes the short path (1,), while an older
        # occurrence provides (1, 9).  The longer path covers the short one and
        # must keep its priority over unrelated alternatives under width one.
        history = [1, 1, 9, 2, 1, 2, 2, 1, 1]
        self.assertEqual(provider.draft_paths(history), ((1, 9),))

    def test_per_call_limits_are_closed_and_do_not_mutate_history(self) -> None:
        provider = NgramTreeDraftProvider(
            max_draft_tokens=4,
            max_branches=4,
            max_match_tokens=2,
        )
        history = [1, 2, 10, 11, 1, 2, 20, 21, 1, 2, 30, 31, 1, 2]
        original = list(history)
        self.assertEqual(provider.draft_paths(history, max_tokens=1, max_branches=2), ((30,), (20,)))
        self.assertEqual(provider.draft_paths(history, max_tokens=0), ())
        self.assertEqual(provider.draft_paths(history, max_branches=0), ())
        self.assertEqual(history, original)

    def test_rejects_invalid_configuration_and_inputs(self) -> None:
        for invalid in (0, MAX_DRAFT_TOKENS + 1, True, 1.5):
            with self.subTest(max_draft_tokens=invalid):
                with self.assertRaises(ValueError):
                    NgramTreeDraftProvider(max_draft_tokens=invalid)  # type: ignore[arg-type]
        for invalid in (0, MAX_TREE_DRAFT_BRANCHES + 1, True, 1.5):
            with self.subTest(max_branches=invalid):
                with self.assertRaises(ValueError):
                    NgramTreeDraftProvider(max_branches=invalid)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            NgramTreeDraftProvider(min_match_tokens=0)
        with self.assertRaises(ValueError):
            NgramTreeDraftProvider(min_match_tokens=3, max_match_tokens=2)
        with self.assertRaises(ValueError):
            NgramTreeDraftProvider(strategy="ngram")

        provider = NgramTreeDraftProvider()
        for invalid_history in ("123", [1, -1], [1, True], [1, 2.5]):
            with self.subTest(history=invalid_history):
                with self.assertRaises(ValueError):
                    provider.draft_paths(invalid_history)  # type: ignore[arg-type]
        for invalid_limit in (-1, True, 1.5):
            with self.subTest(limit=invalid_limit):
                with self.assertRaises(ValueError):
                    provider.draft_paths([1, 2, 1, 2], invalid_limit)  # type: ignore[arg-type]
                with self.assertRaises(ValueError):
                    provider.draft_paths(
                        [1, 2, 1, 2], max_branches=invalid_limit  # type: ignore[arg-type]
                    )


class AdaptiveSpeculationControllerTests(unittest.TestCase):
    def test_candidate_estimate_preserves_the_legacy_positional_constructor(
        self,
    ) -> None:
        estimate = CandidateEstimate(
            2,
            0,
            0,
            0,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            0.95,
            False,
        )

        self.assertEqual(estimate.candidate_size, 2)
        self.assertIsNone(estimate.observed_emitted_tokens_per_verification)
        self.assertIsNone(
            estimate.observed_emitted_tokens_per_verification_lower_bound
        )
        self.assertIsNone(
            estimate.observed_emitted_tokens_per_verification_upper_bound
        )

    @staticmethod
    def _controller(
        *,
        candidates: tuple[int, ...] = (2, 4, 8),
        byte_cost: float = 0.0,
        threshold: float = 1.0,
    ) -> AdaptiveSpeculationController:
        return AdaptiveSpeculationController(
            AdaptiveSpeculationConfig(
                max_draft_tokens=8,
                candidate_sizes=candidates,
                min_token_history=6,
                min_classic_observations=2,
                min_verify_observations=2,
                seconds_per_byte=byte_cost,
                minimum_speedup=threshold,
            )
        )

    @staticmethod
    def _record_classic(
        controller: AdaptiveSpeculationController,
        *,
        latency: float = 1.0,
        transferred_bytes: int = 100,
    ) -> None:
        controller.record_classic(
            latency_seconds=latency,
            transferred_bytes=transferred_bytes,
        )
        controller.record_classic(
            latency_seconds=latency,
            transferred_bytes=transferred_bytes,
        )

    @staticmethod
    def _record_verify(
        controller: AdaptiveSpeculationController,
        size: int,
        accepted: int,
        *,
        latency: float,
        transferred_bytes: int = 100,
    ) -> None:
        for _ in range(2):
            controller.record_verification(
                proposed_tokens=size,
                accepted_tokens=accepted,
                latency_seconds=latency,
                transferred_bytes=transferred_bytes,
            )

    def test_automatic_candidates_cover_one_to_sixteen(self) -> None:
        expected = {
            1: (1,),
            3: (1, 2, 3),
            8: (1, 2, 4, 8),
            16: (1, 2, 4, 8, 16),
        }
        for maximum, candidates in expected.items():
            with self.subTest(maximum=maximum):
                config = AdaptiveSpeculationConfig(max_draft_tokens=maximum)
                self.assertEqual(config.candidate_sizes, candidates)

    def test_custom_candidates_are_sorted_and_deduplicated(self) -> None:
        config = AdaptiveSpeculationConfig(
            max_draft_tokens=8, candidate_sizes=(8, 2, 4, 2)
        )
        self.assertEqual(config.candidate_sizes, (2, 4, 8))

    def test_rejects_invalid_configuration(self) -> None:
        invalid_kwargs = (
            {"max_draft_tokens": 0},
            {"max_draft_tokens": 17},
            {"max_draft_tokens": True},
            {"max_draft_tokens": 8, "candidate_sizes": (0, 2)},
            {"max_draft_tokens": 8, "candidate_sizes": (2, 9)},
            {"max_draft_tokens": 8, "candidate_sizes": (True,)},
            {"min_token_history": -1},
            {"min_classic_observations": 0},
            {"min_verify_observations": 0},
            {"seconds_per_byte": -1.0},
            {"seconds_per_byte": math.inf},
            {"minimum_speedup": 0.99},
            {"minimum_speedup": math.nan},
            {"confidence_level": 0.5},
            {"confidence_level": 1.0},
            {"confidence_level": math.nan},
            {"speedup_hysteresis": -0.01},
            {"minimum_speedup": 1.1, "speedup_hysteresis": 1.1},
        )
        for kwargs in invalid_kwargs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    AdaptiveSpeculationConfig(**kwargs)  # type: ignore[arg-type]

    def test_gate_remains_disabled_during_each_warmup_phase(self) -> None:
        controller = self._controller(candidates=(2,))
        self.assertEqual(
            controller.decide(history_tokens=5, available_draft_tokens=2).reason,
            "history_too_short",
        )
        self.assertEqual(
            controller.decide(history_tokens=6, available_draft_tokens=0).reason,
            "no_draft_tokens",
        )
        self.assertEqual(
            controller.decide(history_tokens=6, available_draft_tokens=2).reason,
            "classic_warmup",
        )
        self._record_classic(controller)
        decision = controller.decide(history_tokens=6, available_draft_tokens=2)
        self.assertFalse(decision.enabled)
        self.assertEqual(decision.reason, "verification_warmup")
        self.assertEqual(decision.candidate_size, 0)

    def test_probe_is_explicit_and_advances_across_candidate_sizes(self) -> None:
        controller = self._controller(candidates=(2, 4, 8))
        self.assertIsNone(
            controller.next_probe_size(history_tokens=6, available_draft_tokens=8)
        )
        self._record_classic(controller)
        self.assertEqual(
            controller.next_probe_size(history_tokens=6, available_draft_tokens=8), 2
        )
        self._record_verify(controller, 2, 1, latency=1.0)
        self.assertEqual(
            controller.next_probe_size(history_tokens=6, available_draft_tokens=8), 4
        )
        self._record_verify(controller, 4, 2, latency=1.0)
        self.assertIsNone(
            controller.next_probe_size(history_tokens=6, available_draft_tokens=4)
        )

    def test_selects_best_beneficial_candidate_and_reports_estimate(self) -> None:
        controller = self._controller(candidates=(2, 4, 8))
        self._record_classic(controller)
        self._record_verify(controller, 2, 2, latency=1.0)  # 3.0x
        self._record_verify(controller, 4, 2, latency=1.0)  # 3.0x
        self._record_verify(controller, 8, 0, latency=2.0)  # 0.5x

        decision = controller.decide(history_tokens=100, available_draft_tokens=8)
        self.assertTrue(decision.enabled)
        self.assertEqual(decision.reason, "beneficial")
        # An exact speedup tie resolves toward the smaller verification wave.
        self.assertEqual(decision.candidate_size, 2)
        self.assertEqual(decision.draft_tokens, 2)
        self.assertAlmostEqual(decision.predicted_speedup or 0, 3.0)
        self.assertAlmostEqual(decision.expected_emitted_tokens or 0, 3.0)
        self.assertAlmostEqual(decision.predicted_latency_speedup or 0, 3.0)
        self.assertAlmostEqual(
            decision.predicted_speedup_lower_bound or 0,
            3.0,
        )

    def test_noisy_point_estimate_cannot_enable_without_a_speedup_lower_bound(
        self,
    ) -> None:
        controller = AdaptiveSpeculationController(
            AdaptiveSpeculationConfig(
                max_draft_tokens=2,
                candidate_sizes=(2,),
                min_token_history=1,
                min_classic_observations=4,
                min_verify_observations=4,
                minimum_speedup=1.05,
                confidence_level=0.95,
            )
        )
        for _ in range(4):
            controller.record_classic(
                latency_seconds=1.0,
                transferred_bytes=0,
            )
        for latency in (0.1, 0.1, 5.0, 0.1):
            controller.record_verification(
                proposed_tokens=2,
                accepted_tokens=2,
                latency_seconds=latency,
                transferred_bytes=0,
            )

        estimate = controller.candidate_estimate(2)
        self.assertGreater(estimate.predicted_speedup or 0, 2.0)
        self.assertLess(estimate.predicted_speedup_lower_bound or math.inf, 1.05)
        decision = controller.decide(
            history_tokens=20,
            available_draft_tokens=2,
        )
        self.assertFalse(decision.enabled)
        self.assertEqual(decision.reason, "not_beneficial")

    def test_enabled_strategy_uses_a_lower_exit_threshold_to_avoid_flapping(
        self,
    ) -> None:
        controller = AdaptiveSpeculationController(
            AdaptiveSpeculationConfig(
                max_draft_tokens=2,
                candidate_sizes=(2,),
                min_token_history=1,
                min_classic_observations=2,
                min_verify_observations=2,
                minimum_speedup=1.2,
                speedup_hysteresis=0.1,
            )
        )
        for _ in range(40):
            controller.record_classic(
                latency_seconds=1.0,
                transferred_bytes=0,
            )
        for _ in range(20):
            controller.record_verification(
                proposed_tokens=2,
                accepted_tokens=2,
                latency_seconds=2.4,
                transferred_bytes=0,
            )
        entered = controller.decide(
            history_tokens=20,
            available_draft_tokens=2,
        )
        self.assertTrue(entered.enabled)
        self.assertGreater(
            entered.predicted_speedup_lower_bound or 0,
            1.2,
        )

        for _ in range(20):
            controller.record_verification(
                proposed_tokens=2,
                accepted_tokens=2,
                latency_seconds=2.6,
                transferred_bytes=0,
            )
        held = controller.decide(
            history_tokens=20,
            available_draft_tokens=2,
        )
        self.assertTrue(held.enabled)
        self.assertEqual(held.reason, "hysteresis_hold")
        self.assertLess(held.predicted_speedup_lower_bound or math.inf, 1.2)
        self.assertGreater(held.predicted_speedup_lower_bound or 0, 1.1)

    def test_available_draft_count_filters_larger_candidates(self) -> None:
        controller = self._controller(candidates=(2, 4))
        self._record_classic(controller)
        self._record_verify(controller, 2, 1, latency=1.0)  # 2x
        self._record_verify(controller, 4, 4, latency=1.0)  # 5x
        decision = controller.decide(history_tokens=50, available_draft_tokens=3)
        self.assertTrue(decision.enabled)
        self.assertEqual(decision.candidate_size, 2)

    def test_exact_break_even_and_configured_margin_disable_speculation(self) -> None:
        controller = self._controller(candidates=(2,))
        self._record_classic(controller)
        self._record_verify(controller, 2, 2, latency=3.0)  # exactly 1.0x
        decision = controller.decide(history_tokens=50, available_draft_tokens=2)
        self.assertFalse(decision.enabled)
        self.assertEqual(decision.reason, "not_beneficial")
        self.assertEqual(decision.candidate_size, 0)
        self.assertAlmostEqual(decision.predicted_speedup or 0, 1.0)

        margin = self._controller(candidates=(2,), threshold=1.5)
        self._record_classic(margin)
        self._record_verify(margin, 2, 1, latency=1.5)  # 1.333x
        self.assertFalse(
            margin.decide(history_tokens=50, available_draft_tokens=2).enabled
        )

    def test_byte_cost_can_reverse_a_latency_only_win(self) -> None:
        controller = self._controller(candidates=(2,), byte_cost=0.01)
        self._record_classic(controller, latency=1.0, transferred_bytes=100)
        self._record_verify(
            controller,
            2,
            2,
            latency=0.5,
            transferred_bytes=1_000,
        )
        estimate = controller.candidate_estimate(2)
        self.assertAlmostEqual(estimate.predicted_latency_speedup or 0, 6.0)
        self.assertAlmostEqual(estimate.predicted_byte_efficiency or 0, 0.3)
        self.assertLess(estimate.predicted_speedup or math.inf, 1.0)
        self.assertFalse(
            controller.decide(history_tokens=50, available_draft_tokens=2).enabled
        )

    def test_zero_byte_verification_is_not_a_bandwidth_constraint(self) -> None:
        controller = self._controller(candidates=(2,))
        self._record_classic(controller, transferred_bytes=0)
        self._record_verify(
            controller, 2, 2, latency=1.0, transferred_bytes=0
        )
        estimate = controller.candidate_estimate(2)
        self.assertTrue(math.isinf(estimate.predicted_byte_efficiency or 0))
        self.assertTrue(controller.decide(history_tokens=50, available_draft_tokens=2).enabled)

    def test_aggregates_classic_and_verification_statistics(self) -> None:
        controller = self._controller(candidates=(2, 4))
        controller.record_classic(
            latency_seconds=2.0, transferred_bytes=200, generated_tokens=2
        )
        controller.record_classic(
            latency_seconds=1.0, transferred_bytes=50, generated_tokens=1
        )
        self._record_verify(
            controller, 2, 1, latency=0.75, transferred_bytes=80
        )
        self._record_verify(
            controller, 4, 3, latency=1.25, transferred_bytes=120
        )
        stats = controller.stats()
        self.assertEqual(stats.classic_observations, 2)
        self.assertEqual(stats.classic_generated_tokens, 3)
        self.assertEqual(stats.classic_latency_seconds, 3.0)
        self.assertEqual(stats.classic_bytes, 250)
        self.assertEqual(stats.verification_observations, 4)
        self.assertEqual(stats.proposed_tokens, 12)
        self.assertEqual(stats.accepted_tokens, 8)
        self.assertEqual(stats.emitted_tokens, 12)
        self.assertEqual(stats.verification_latency_seconds, 4.0)
        self.assertEqual(stats.verification_bytes, 400)
        self.assertAlmostEqual(stats.acceptance_rate or 0, 2 / 3)
        self.assertAlmostEqual(
            stats.observed_emitted_tokens_per_verification or 0,
            3.0,
        )

        estimate = controller.candidate_estimate(4)
        self.assertTrue(estimate.ready)
        self.assertEqual(estimate.observations, 2)
        self.assertEqual(estimate.proposed_tokens, 8)
        self.assertEqual(estimate.accepted_tokens, 6)
        self.assertAlmostEqual(estimate.acceptance_rate or 0, 0.75)
        self.assertAlmostEqual(
            estimate.observed_emitted_tokens_per_verification or 0,
            4.0,
        )
        self.assertAlmostEqual(
            estimate.observed_emitted_tokens_per_verification_lower_bound or 0,
            4.0,
        )
        self.assertAlmostEqual(
            estimate.observed_emitted_tokens_per_verification_upper_bound or 0,
            4.0,
        )
        self.assertAlmostEqual(estimate.mean_verification_latency_seconds or 0, 1.25)
        self.assertAlmostEqual(estimate.mean_verification_bytes or 0, 120.0)

    def test_observed_emitted_bounds_fail_closed_until_two_verifications(
        self,
    ) -> None:
        controller = self._controller(candidates=(2,))
        empty = controller.candidate_estimate(2)
        self.assertIsNone(empty.observed_emitted_tokens_per_verification)
        self.assertIsNone(
            empty.observed_emitted_tokens_per_verification_lower_bound
        )
        self.assertIsNone(
            empty.observed_emitted_tokens_per_verification_upper_bound
        )
        self.assertIsNone(
            controller.stats().observed_emitted_tokens_per_verification
        )

        controller.record_verification(
            proposed_tokens=2,
            accepted_tokens=1,
            latency_seconds=1.0,
            transferred_bytes=10,
        )
        one = controller.candidate_estimate(2)
        self.assertEqual(one.observed_emitted_tokens_per_verification, 2.0)
        self.assertIsNone(
            one.observed_emitted_tokens_per_verification_lower_bound
        )
        self.assertIsNone(
            one.observed_emitted_tokens_per_verification_upper_bound
        )

        for accepted in (0, 2, 1):
            controller.record_verification(
                proposed_tokens=2,
                accepted_tokens=accepted,
                latency_seconds=1.0,
                transferred_bytes=10,
            )
        observed = controller.candidate_estimate(2)
        mean = observed.observed_emitted_tokens_per_verification
        lower = observed.observed_emitted_tokens_per_verification_lower_bound
        upper = observed.observed_emitted_tokens_per_verification_upper_bound
        self.assertIsNotNone(mean)
        self.assertIsNotNone(lower)
        self.assertIsNotNone(upper)
        assert mean is not None and lower is not None and upper is not None
        self.assertTrue(all(math.isfinite(value) for value in (mean, lower, upper)))
        self.assertLessEqual(1.0, lower)
        self.assertLessEqual(lower, mean)
        self.assertLessEqual(mean, upper)
        self.assertLessEqual(upper, 3.0)

    def test_reset_discards_all_observations(self) -> None:
        controller = self._controller(candidates=(2,))
        self._record_classic(controller)
        self._record_verify(controller, 2, 2, latency=1.0)
        controller.reset()
        self.assertEqual(controller.stats().classic_observations, 0)
        self.assertEqual(controller.stats().verification_observations, 0)
        self.assertEqual(
            controller.decide(history_tokens=50, available_draft_tokens=2).reason,
            "classic_warmup",
        )

    def test_rejects_invalid_observations_and_decision_inputs(self) -> None:
        controller = self._controller(candidates=(2,))
        invalid_classic = (
            {"latency_seconds": 0.0, "transferred_bytes": 1},
            {"latency_seconds": math.nan, "transferred_bytes": 1},
            {"latency_seconds": 1.0, "transferred_bytes": -1},
            {"latency_seconds": 1.0, "transferred_bytes": 1, "generated_tokens": 0},
            {"latency_seconds": 1.0, "transferred_bytes": True},
        )
        for kwargs in invalid_classic:
            with self.subTest(classic=kwargs):
                with self.assertRaises(ValueError):
                    controller.record_classic(**kwargs)  # type: ignore[arg-type]

        invalid_verify = (
            {"proposed_tokens": 0, "accepted_tokens": 0},
            {"proposed_tokens": 9, "accepted_tokens": 0},
            {"proposed_tokens": 2, "accepted_tokens": 3},
            {"proposed_tokens": 2, "accepted_tokens": -1},
            {"proposed_tokens": 2, "accepted_tokens": 1, "latency_seconds": 0.0},
            {"proposed_tokens": 2, "accepted_tokens": 1, "transferred_bytes": -1},
        )
        for partial in invalid_verify:
            kwargs = {
                "proposed_tokens": 2,
                "accepted_tokens": 1,
                "latency_seconds": 1.0,
                "transferred_bytes": 1,
                **partial,
            }
            with self.subTest(verification=kwargs):
                with self.assertRaises(ValueError):
                    controller.record_verification(**kwargs)

        for history, available in ((-1, 1), (1, -1), (True, 1), (1, 1.5)):
            with self.subTest(history=history, available=available):
                with self.assertRaises(ValueError):
                    controller.decide(
                        history_tokens=history,  # type: ignore[arg-type]
                        available_draft_tokens=available,  # type: ignore[arg-type]
                    )
        with self.assertRaises(ValueError):
            controller.candidate_estimate(9)


class DelayAdaptiveSpeculationTests(unittest.TestCase):
    _record_classic = staticmethod(AdaptiveSpeculationControllerTests._record_classic)
    _record_verify = staticmethod(AdaptiveSpeculationControllerTests._record_verify)

    @staticmethod
    def _controller(
        *,
        candidates: tuple[int, ...] = (2, 4, 8),
        delay_adaptive: bool = False,
        reference_ms: float = 25.0,
        alpha: float = 0.2,
    ) -> AdaptiveSpeculationController:
        return AdaptiveSpeculationController(
            AdaptiveSpeculationConfig(
                max_draft_tokens=8,
                candidate_sizes=candidates,
                min_token_history=6,
                min_classic_observations=2,
                min_verify_observations=2,
                delay_adaptive=delay_adaptive,
                delay_reference_ms=reference_ms,
                rtt_ewma_alpha=alpha,
            )
        )

    def test_defaults_leave_delay_policy_disabled(self) -> None:
        config = AdaptiveSpeculationConfig()
        self.assertFalse(config.delay_adaptive)
        self.assertEqual(config.delay_reference_ms, 25.0)
        self.assertEqual(config.rtt_ewma_alpha, 0.2)

    def test_flag_off_matches_baseline_decision_exactly(self) -> None:
        baseline = self._controller()
        with_rtt = self._controller()
        for controller in (baseline, with_rtt):
            self._record_classic(controller)
            self._record_verify(controller, 2, 1, latency=1.0)
            self._record_verify(controller, 8, 8, latency=1.0)
        # An RTT so low that an enabled cap would force depth 1.
        with_rtt.record_rtt(0.0)
        expected = baseline.decide(history_tokens=50, available_draft_tokens=8)
        actual = with_rtt.decide(history_tokens=50, available_draft_tokens=8)
        self.assertEqual(actual, expected)
        self.assertEqual(actual.candidate_size, 8)

    def test_cap_is_monotone_non_decreasing_in_rtt(self) -> None:
        controller = self._controller(delay_adaptive=True)
        grid = (0.0, 1.0, 5.0, 12.5, 25.0, 50.0, 100.0, 200.0, 400.0, 1000.0, 1e6)
        caps = [controller.delay_draft_cap(rtt) for rtt in grid]
        for lower, upper in zip(caps, caps[1:]):
            self.assertLessEqual(lower, upper)

    def test_cap_clamps_and_matches_formula(self) -> None:
        controller = self._controller(delay_adaptive=True)
        self.assertEqual(controller.delay_draft_cap(0.0), 1)
        self.assertEqual(controller.delay_draft_cap(24.9), 1)
        self.assertEqual(controller.delay_draft_cap(25.0), 2)
        self.assertEqual(controller.delay_draft_cap(75.0), 3)
        self.assertEqual(controller.delay_draft_cap(200.0), 4)
        self.assertEqual(controller.delay_draft_cap(1e12), 8)

    def test_cap_without_rtt_evidence_is_unrestricted(self) -> None:
        controller = self._controller(delay_adaptive=True)
        self.assertIsNone(controller.rtt_ewma_ms)
        self.assertEqual(controller.delay_draft_cap(), 8)

    def test_cap_saturates_when_ratio_overflows(self) -> None:
        # A pathological (validation-passing) sub-normal reference makes the
        # rtt/reference ratio overflow to +inf; the cap must saturate to the
        # ceiling rather than raise OverflowError from floor(inf).
        controller = self._controller(delay_adaptive=True, reference_ms=1e-300)
        controller.record_rtt(1e12)
        self.assertEqual(controller.delay_draft_cap(), 8)
        decision = controller.decide(history_tokens=50, available_draft_tokens=8)
        self.assertLessEqual(decision.candidate_size, 8)

    def test_rtt_ewma_blends_and_reset_clears(self) -> None:
        controller = self._controller(delay_adaptive=True, alpha=0.5)
        controller.record_rtt(100.0)
        self.assertEqual(controller.rtt_ewma_ms, 100.0)
        controller.record_rtt(0.0)
        self.assertEqual(controller.rtt_ewma_ms, 50.0)
        controller.reset()
        self.assertIsNone(controller.rtt_ewma_ms)

    def test_high_rtt_cap_bounds_chosen_depth(self) -> None:
        controller = self._controller(delay_adaptive=True)
        self._record_classic(controller)
        self._record_verify(controller, 2, 1, latency=1.0)  # 2.0x
        self._record_verify(controller, 4, 3, latency=1.0)  # 4.0x
        self._record_verify(controller, 8, 8, latency=1.0)  # 9.0x, best overall
        controller.record_rtt(200.0)
        self.assertEqual(controller.delay_draft_cap(), 4)
        decision = controller.decide(history_tokens=50, available_draft_tokens=8)
        self.assertTrue(decision.enabled)
        self.assertEqual(decision.candidate_size, 4)

    def test_low_rtt_cap_does_not_force_depth_on_poor_acceptance(self) -> None:
        controller = self._controller(delay_adaptive=True, candidates=(2,))
        self._record_classic(controller)
        self._record_verify(controller, 2, 0, latency=2.0)  # 0.5x
        controller.record_rtt(25.0)
        self.assertEqual(controller.delay_draft_cap(), 2)
        decision = controller.decide(history_tokens=50, available_draft_tokens=2)
        self.assertFalse(decision.enabled)
        self.assertEqual(decision.reason, "not_beneficial")
        self.assertEqual(decision.candidate_size, 0)

    def test_probe_sizes_respect_the_delay_cap(self) -> None:
        controller = self._controller(delay_adaptive=True)
        self._record_classic(controller)
        controller.record_rtt(25.0)  # cap = 2
        self.assertEqual(
            controller.next_probe_size(history_tokens=6, available_draft_tokens=8), 2
        )
        self._record_verify(controller, 2, 1, latency=1.0)
        self.assertIsNone(
            controller.next_probe_size(history_tokens=6, available_draft_tokens=8)
        )

    def test_rejects_invalid_delay_configuration_and_rtt(self) -> None:
        invalid_kwargs = (
            {"delay_adaptive": 1},
            {"delay_reference_ms": 0.0},
            {"delay_reference_ms": -1.0},
            {"delay_reference_ms": math.inf},
            {"rtt_ewma_alpha": 0.0},
            {"rtt_ewma_alpha": 1.5},
            {"rtt_ewma_alpha": math.nan},
        )
        for kwargs in invalid_kwargs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    AdaptiveSpeculationConfig(**kwargs)  # type: ignore[arg-type]

        controller = self._controller(delay_adaptive=True)
        for invalid_rtt in (-1.0, math.nan, math.inf):
            with self.subTest(rtt=invalid_rtt):
                with self.assertRaises(ValueError):
                    controller.record_rtt(invalid_rtt)
        with self.assertRaises(ValueError):
            controller.delay_draft_cap(-1.0)


if __name__ == "__main__":
    unittest.main()
