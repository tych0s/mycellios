from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
import math
import unittest

from distributed_runtime.tree_policy import (
    ClassicObservation,
    MAX_POLICY_COUNT,
    RTTAwareTreeWaveController,
    TreeWaveBudget,
    TreeWaveCandidate,
    TreeWaveObservation,
    TreeWavePolicyConfig,
    TreeWaveShape,
)


class RTTAwareTreeWavePolicyTests(unittest.TestCase):
    SHALLOW = TreeWaveShape(
        width=2,
        depth=2,
        total_candidate_tokens=4,
    )
    DEEP = TreeWaveShape(
        width=4,
        depth=4,
        total_candidate_tokens=12,
    )
    SHALLOW_CANDIDATE = TreeWaveCandidate(SHALLOW, 200)
    DEEP_CANDIDATE = TreeWaveCandidate(DEEP, 800)

    @classmethod
    def _controller(
        cls,
        *,
        shapes: tuple[TreeWaveShape, ...] | None = None,
        seconds_per_byte: float = 0.0,
        minimum_speedup: float = 1.0,
        hysteresis: float = 0.0,
        cooldown: int = 0,
        measurement_alpha: float = 1.0,
    ) -> RTTAwareTreeWaveController:
        return RTTAwareTreeWaveController(
            TreeWavePolicyConfig(
                candidate_shapes=shapes or (cls.SHALLOW, cls.DEEP),
                min_classic_observations=2,
                min_tree_observations=2,
                seconds_per_byte=seconds_per_byte,
                minimum_speedup=minimum_speedup,
                max_projected_kv_bytes=10_000,
                max_branches=16,
                max_candidate_tokens=100,
                max_depth=8,
                rtt_reference_ms=25.0,
                rtt_ewma_alpha=1.0,
                measurement_ewma_alpha=measurement_alpha,
                hysteresis_fraction=hysteresis,
                cooldown_decisions=cooldown,
            )
        )

    @staticmethod
    def _record_classic(
        controller: RTTAwareTreeWaveController,
        *,
        latency: float = 1.0,
        byte_count: int = 100,
        repetitions: int = 2,
    ) -> None:
        for _ in range(repetitions):
            controller.record_classic(
                ClassicObservation(
                    latency_seconds=latency,
                    transferred_bytes=byte_count,
                    emitted_tokens=1,
                )
            )

    @staticmethod
    def _record_tree(
        controller: RTTAwareTreeWaveController,
        shape: TreeWaveShape,
        *,
        latency: float,
        byte_count: int = 100,
        emitted: int | None = None,
        accepted: int | None = None,
        repetitions: int = 2,
    ) -> None:
        emitted_count = shape.depth + 1 if emitted is None else emitted
        accepted_count = shape.depth if accepted is None else accepted
        for _ in range(repetitions):
            controller.record_tree(
                TreeWaveObservation(
                    shape=shape,
                    latency_seconds=latency,
                    transferred_bytes=byte_count,
                    emitted_tokens=emitted_count,
                    accepted_tokens=accepted_count,
                )
            )

    @classmethod
    def _candidates(
        cls,
        shapes: tuple[TreeWaveShape, ...] | None = None,
        *,
        shallow_bytes: int = 200,
        deep_bytes: int = 800,
    ) -> tuple[TreeWaveCandidate, ...]:
        selected = shapes or (cls.SHALLOW, cls.DEEP)
        projections = {
            cls.SHALLOW: shallow_bytes,
            cls.DEEP: deep_bytes,
        }
        return tuple(
            TreeWaveCandidate(shape, projections.get(shape, 1))
            for shape in selected
        )

    @classmethod
    def _current_candidates(
        cls,
        controller: RTTAwareTreeWaveController,
        *,
        shallow_bytes: int = 200,
        deep_bytes: int = 800,
    ) -> tuple[TreeWaveCandidate, ...]:
        return cls._candidates(
            controller.config.candidate_shapes,
            shallow_bytes=shallow_bytes,
            deep_bytes=deep_bytes,
        )

    @classmethod
    def _decide(
        cls,
        controller: RTTAwareTreeWaveController,
        *,
        budget: TreeWaveBudget | None = None,
        shallow_bytes: int = 200,
        deep_bytes: int = 800,
    ):
        return controller.decide(
            candidates=cls._current_candidates(
                controller,
                shallow_bytes=shallow_bytes,
                deep_bytes=deep_bytes,
            ),
            budget=budget,
        )

    @classmethod
    def _next_probe(cls, controller: RTTAwareTreeWaveController):
        return controller.next_probe(
            candidates=cls._current_candidates(controller)
        )

    def test_shapes_and_observations_are_sealed(self) -> None:
        shape = TreeWaveShape(2, 3, 5)
        observation = TreeWaveObservation(shape, 1.0, 10, 3, 2)
        with self.assertRaises(FrozenInstanceError):
            shape.width = 9  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            observation.accepted_tokens = 0  # type: ignore[misc]

    def test_exact_observation_requires_one_bonus_or_correction_token(self) -> None:
        valid = TreeWaveObservation(self.SHALLOW, 1.0, 10, 3, 2)
        self.assertEqual(valid.emitted_tokens, valid.accepted_tokens + 1)
        for emitted, accepted in ((2, 0), (1, 1), (3, 1)):
            with self.subTest(emitted=emitted, accepted=accepted):
                with self.assertRaises(ValueError):
                    TreeWaveObservation(
                        self.SHALLOW,
                        latency_seconds=1.0,
                        transferred_bytes=10,
                        emitted_tokens=emitted,
                        accepted_tokens=accepted,
                    )

    def test_decision_stays_classic_through_both_warmup_phases(self) -> None:
        controller = self._controller(shapes=(self.SHALLOW,))
        controller.record_rtt(25.0)
        decision = self._decide(controller)
        self.assertFalse(decision.enabled)
        self.assertEqual(decision.reason, "classic_warmup")

        self._record_classic(controller)
        decision = self._decide(controller)
        self.assertFalse(decision.enabled)
        self.assertEqual(decision.reason, "tree_warmup")

        controller.record_tree(TreeWaveObservation(self.SHALLOW, 1.0, 100, 3, 2))
        decision = self._decide(controller)
        self.assertFalse(decision.enabled)
        self.assertEqual(decision.reason, "tree_warmup")

    def test_probe_is_explicit_single_flight_and_advances_after_observation(self) -> None:
        controller = self._controller(shapes=(self.SHALLOW, self.DEEP))
        controller.record_rtt(175.0)  # depth cap 4
        self.assertIsNone(self._next_probe(controller))
        self._record_classic(controller)

        first = self._next_probe(controller)
        self.assertEqual(first, self.SHALLOW_CANDIDATE)
        self.assertIsNone(self._next_probe(controller))
        controller.record_tree(TreeWaveObservation(self.SHALLOW, 0.8, 100, 3, 2))
        self.assertEqual(self._next_probe(controller), self.SHALLOW_CANDIDATE)
        self.assertTrue(controller.cancel_probe(self.SHALLOW))
        controller.record_tree(TreeWaveObservation(self.SHALLOW, 0.8, 100, 3, 2))
        self.assertEqual(self._next_probe(controller), self.DEEP_CANDIDATE)

    def test_selects_lowest_measured_cost_per_emitted_token(self) -> None:
        controller = self._controller(minimum_speedup=1.05)
        controller.record_rtt(175.0)
        self._record_classic(controller, latency=1.0)
        self._record_tree(controller, self.SHALLOW, latency=1.5)  # 0.5 TPOT
        self._record_tree(controller, self.DEEP, latency=1.0)  # 0.2 TPOT

        decision = self._decide(controller)
        self.assertTrue(decision.enabled)
        self.assertEqual(decision.shape, self.DEEP)
        self.assertAlmostEqual(decision.selected_cost_per_token or 0.0, 0.2)
        self.assertAlmostEqual(decision.predicted_speedup or 0.0, 5.0)

    def test_byte_price_can_reverse_latency_only_ranking(self) -> None:
        controller = self._controller(seconds_per_byte=0.01)
        controller.record_rtt(175.0)
        self._record_classic(controller, latency=2.0, byte_count=100)
        # Faster latency TPOT, but very expensive network volume.
        self._record_tree(
            controller, self.SHALLOW, latency=0.6, byte_count=1_500
        )
        # Slower latency TPOT, but low byte volume: total cost is lower.
        self._record_tree(controller, self.DEEP, latency=2.0, byte_count=100)

        decision = self._decide(controller)
        self.assertTrue(decision.enabled)
        self.assertEqual(decision.shape, self.DEEP)

    def test_observed_slowdown_drops_to_classic_and_arms_cooldown(self) -> None:
        controller = self._controller(
            shapes=(self.SHALLOW,), cooldown=2, measurement_alpha=1.0
        )
        controller.record_rtt(25.0)
        self._record_classic(controller, latency=1.0)
        self._record_tree(controller, self.SHALLOW, latency=0.6)
        self.assertTrue(self._decide(controller).enabled)

        controller.record_tree(
            TreeWaveObservation(self.SHALLOW, 6.0, 100, 1, 0)
        )
        fallback = self._decide(controller)
        self.assertFalse(fallback.enabled)
        self.assertEqual(fallback.reason, "selected_shape_slowdown")
        self.assertEqual(controller.stats().cooldown_remaining, 2)

        controller.record_tree(
            TreeWaveObservation(self.SHALLOW, 0.3, 100, 3, 2)
        )
        self.assertEqual(self._decide(controller).reason, "cooldown_hold")
        self.assertEqual(self._decide(controller).reason, "cooldown_hold")
        self.assertTrue(self._decide(controller).enabled)

    def test_kv_branch_and_candidate_limits_each_fail_closed(self) -> None:
        controller = self._controller(shapes=(self.DEEP,))
        controller.record_rtt(175.0)
        self._record_classic(controller)
        self._record_tree(controller, self.DEEP, latency=1.0)

        budgets = (
            TreeWaveBudget(799, 4, 12),
            TreeWaveBudget(800, 3, 12),
            TreeWaveBudget(800, 4, 11),
            TreeWaveBudget(0, 0, 0),
        )
        for budget in budgets:
            with self.subTest(budget=budget):
                decision = self._decide(controller, budget=budget)
                self.assertFalse(decision.enabled)
                self.assertEqual(decision.reason, "no_eligible_shape")

        self.assertTrue(
            self._decide(
                controller,
                budget=TreeWaveBudget(800, 4, 12),
            ).enabled
        )

    def test_kv_projection_changes_with_context_without_changing_shape_identity(self) -> None:
        controller = self._controller(shapes=(self.SHALLOW,))
        controller.record_rtt(25.0)
        self._record_classic(controller)
        self._record_tree(controller, self.SHALLOW, latency=0.6)

        short_chat = self._decide(
            controller,
            budget=TreeWaveBudget(800, 2, 4),
            shallow_bytes=200,
        )
        self.assertTrue(short_chat.enabled)
        self.assertEqual(short_chat.projected_kv_bytes, 200)

        long_chat = self._decide(
            controller,
            budget=TreeWaveBudget(800, 2, 4),
            shallow_bytes=900,
        )
        self.assertFalse(long_chat.enabled)
        self.assertEqual(long_chat.reason, "no_eligible_shape")

        fits_again = self._decide(
            controller,
            budget=TreeWaveBudget(800, 2, 4),
            shallow_bytes=500,
        )
        self.assertTrue(fits_again.enabled)
        self.assertEqual(fits_again.projected_kv_bytes, 500)
        self.assertEqual(controller.stats().shapes[0].observations, 2)

    def test_missing_current_projection_fails_closed(self) -> None:
        controller = self._controller(shapes=(self.SHALLOW,))
        controller.record_rtt(25.0)
        self._record_classic(controller)
        self._record_tree(controller, self.SHALLOW, latency=0.6)

        decision = controller.decide()
        self.assertFalse(decision.enabled)
        self.assertEqual(decision.reason, "no_eligible_shape")
        self.assertIsNone(controller.next_probe())

    def test_rtt_depth_cap_is_monotone_and_logarithmic(self) -> None:
        controller = self._controller()
        measured_rtts = (0.0, 24.9, 25.0, 75.0, 175.0, 375.0, 10_000.0)
        caps = tuple(controller.rtt_depth_cap(rtt) for rtt in measured_rtts)
        self.assertEqual(caps[:5], (1, 1, 2, 3, 4))
        self.assertEqual(tuple(sorted(caps)), caps)
        self.assertLessEqual(caps[-1], controller.config.max_depth)

        controller.record_rtt(25.0)
        first = controller.rtt_depth_cap()
        controller.record_rtt(175.0)
        self.assertGreaterEqual(controller.rtt_depth_cap(), first)

    def test_rtt_cap_prevents_deep_shape_until_delay_justifies_it(self) -> None:
        controller = self._controller(shapes=(self.DEEP,))
        self._record_classic(controller)
        self._record_tree(controller, self.DEEP, latency=1.0)
        controller.record_rtt(25.0)
        self.assertEqual(self._decide(controller).reason, "no_eligible_shape")
        controller.record_rtt(175.0)
        self.assertTrue(self._decide(controller).enabled)

    def test_hysteresis_and_cooldown_prevent_shape_oscillation(self) -> None:
        controller = self._controller(
            hysteresis=0.10, cooldown=2, measurement_alpha=1.0
        )
        controller.record_rtt(175.0)
        self._record_classic(controller, latency=1.0)
        self._record_tree(controller, self.SHALLOW, latency=1.2)  # 0.4 TPOT
        self._record_tree(controller, self.DEEP, latency=2.25)  # 0.45 TPOT
        self.assertEqual(self._decide(controller).shape, self.SHALLOW)

        # Deep improves materially, but the two-decision cooldown retains the
        # currently valid and still beneficial shallow shape.
        controller.record_tree(TreeWaveObservation(self.DEEP, 1.5, 100, 5, 4))
        self.assertEqual(self._decide(controller).shape, self.SHALLOW)
        self.assertEqual(self._decide(controller).shape, self.SHALLOW)
        switched = self._decide(controller)
        self.assertEqual(switched.shape, self.DEEP)
        self.assertEqual(switched.reason, "better_shape")

        # A <10% reversal is ignored by hysteresis after cooldown expires.
        self._decide(controller)
        self._decide(controller)
        controller.record_tree(TreeWaveObservation(self.SHALLOW, 0.87, 100, 3, 2))
        held = self._decide(controller)
        self.assertEqual(held.shape, self.DEEP)
        self.assertEqual(held.reason, "hysteresis_hold")

    def test_zero_cost_measurements_never_divide_and_never_enable_tree(self) -> None:
        controller = self._controller(shapes=(self.SHALLOW,))
        controller.record_rtt(25.0)
        self._record_classic(controller, latency=0.0, byte_count=0)
        self._record_tree(
            controller, self.SHALLOW, latency=0.0, byte_count=0
        )
        decision = self._decide(controller)
        self.assertFalse(decision.enabled)
        self.assertEqual(decision.reason, "classic_zero_cost")
        self.assertIsNone(controller.stats().shapes[0].predicted_speedup)

    def test_stats_are_strict_json_serializable(self) -> None:
        controller = self._controller(shapes=(self.SHALLOW,))
        controller.record_rtt(25.0)
        self._record_classic(controller)
        self._record_tree(controller, self.SHALLOW, latency=1.0)
        self._decide(controller)
        payload = controller.stats().to_dict()
        encoded = json.dumps(payload, allow_nan=False, sort_keys=True)
        self.assertIn('"rtt_ewma_ms": 25.0', encoded)
        self.assertEqual(payload["shapes"][0]["shape"]["width"], 2)

    def test_rejects_nan_inf_negative_and_bool_without_mutation(self) -> None:
        for invalid in (math.nan, math.inf, -1.0, True):
            with self.subTest(rtt=invalid):
                controller = self._controller(shapes=(self.SHALLOW,))
                with self.assertRaises(ValueError):
                    controller.record_rtt(invalid)  # type: ignore[arg-type]
                self.assertEqual(controller.stats().rtt_observations, 0)

        invalid_shapes = (
            (True, 2, 2),
            (1, 0, 1),
            (2, 2, 1),
            (2, 2, 5),
        )
        for arguments in invalid_shapes:
            with self.subTest(shape=arguments):
                with self.assertRaises(ValueError):
                    TreeWaveShape(*arguments)  # type: ignore[arg-type]

        for projected in (-1, True):
            with self.subTest(projected_kv_bytes=projected):
                with self.assertRaises(ValueError):
                    TreeWaveCandidate(
                        self.SHALLOW,
                        projected,  # type: ignore[arg-type]
                    )

        invalid_observations = (
            (math.nan, 0, 1),
            (math.inf, 0, 1),
            (-1.0, 0, 1),
            (True, 0, 1),
            (1.0, True, 1),
            (1.0, 0, True),
        )
        for latency, byte_count, emitted in invalid_observations:
            with self.subTest(observation=(latency, byte_count, emitted)):
                with self.assertRaises(ValueError):
                    ClassicObservation(latency, byte_count, emitted)  # type: ignore[arg-type]

    def test_rejects_invalid_config_and_foreign_shapes(self) -> None:
        invalid_kwargs = (
            {"minimum_speedup": 0.99},
            {"minimum_speedup": math.nan},
            {"seconds_per_byte": math.inf},
            {"seconds_per_byte": True},
            {"rtt_ewma_alpha": 0.0},
            {"measurement_ewma_alpha": 1.1},
            {"hysteresis_fraction": -0.1},
            {"cooldown_decisions": True},
            {"max_branches": -1},
        )
        for kwargs in invalid_kwargs:
            with self.subTest(config=kwargs):
                with self.assertRaises(ValueError):
                    TreeWavePolicyConfig(
                        candidate_shapes=(self.SHALLOW,), **kwargs  # type: ignore[arg-type]
                    )

        controller = self._controller(shapes=(self.SHALLOW,))
        foreign = TreeWaveShape(1, 1, 1)
        with self.assertRaises(ValueError):
            controller.record_tree(TreeWaveObservation(foreign, 1.0, 1, 1, 0))

    def test_aggregate_overflow_is_rejected_atomically(self) -> None:
        controller = self._controller(shapes=(self.SHALLOW,))
        controller.record_classic(ClassicObservation(1e308, 0, 1))
        before = controller.stats().classic
        with self.assertRaises(ValueError):
            controller.record_classic(ClassicObservation(1e308, 0, 1))
        self.assertEqual(controller.stats().classic, before)

        byte_controller = self._controller(shapes=(self.SHALLOW,))
        byte_controller.record_classic(
            ClassicObservation(1.0, MAX_POLICY_COUNT, 1)
        )
        before = byte_controller.stats().classic
        with self.assertRaises(ValueError):
            byte_controller.record_classic(ClassicObservation(1.0, 1, 1))
        self.assertEqual(byte_controller.stats().classic, before)

    def test_policy_counter_overflow_is_rejected_without_partial_mutation(self) -> None:
        rtt_controller = self._controller(shapes=(self.SHALLOW,))
        rtt_controller.record_rtt(25.0)
        rtt_controller._rtt_observations = MAX_POLICY_COUNT
        before_rtt = rtt_controller.rtt_ewma_ms
        with self.assertRaises(ValueError):
            rtt_controller.record_rtt(100.0)
        self.assertEqual(rtt_controller.stats().rtt_observations, MAX_POLICY_COUNT)
        self.assertEqual(rtt_controller.rtt_ewma_ms, before_rtt)

        decision_controller = self._controller(shapes=(self.SHALLOW,))
        decision_controller._decisions = MAX_POLICY_COUNT
        before_shape = decision_controller.selected_shape
        with self.assertRaises(ValueError):
            decision_controller.decide(
                candidates=self._current_candidates(decision_controller)
            )
        self.assertEqual(decision_controller.stats().decisions, MAX_POLICY_COUNT)
        self.assertEqual(decision_controller.selected_shape, before_shape)

        probe_controller = self._controller(shapes=(self.SHALLOW,))
        probe_controller.record_rtt(25.0)
        self._record_classic(probe_controller)
        probe_controller._probes_issued = MAX_POLICY_COUNT
        with self.assertRaises(ValueError):
            probe_controller.next_probe(
                candidates=self._current_candidates(probe_controller)
            )
        probe_stats = probe_controller.stats()
        self.assertEqual(probe_stats.probes_issued, MAX_POLICY_COUNT)
        self.assertIsNone(probe_stats.pending_probe)


if __name__ == "__main__":
    unittest.main()
