from __future__ import annotations

import math
import unittest

from distributed_runtime.lossless_sampling import (
    CounterSamplingRng,
    lossless_speculative_sample,
    probabilities_from_logits,
)


class LosslessSamplingTests(unittest.TestCase):
    def test_rng_checkpoint_restores_exact_stream(self) -> None:
        rng = CounterSamplingRng(bytes(range(32)))
        prefix = [rng.uniform() for _ in range(3)]
        checkpoint = rng.checkpoint()
        continuation = [rng.uniform() for _ in range(8)]
        restored = CounterSamplingRng.from_checkpoint(checkpoint)
        self.assertEqual([restored.uniform() for _ in range(8)], continuation)
        self.assertNotEqual(prefix, continuation[:3])

    def test_logits_are_stable_and_top_p_is_deterministic(self) -> None:
        full = probabilities_from_logits((1001.0, 1000.0, 999.0), temperature=1.0)
        self.assertAlmostEqual(sum(full), 1.0)
        truncated = probabilities_from_logits(
            (1001.0, 1000.0, 999.0), temperature=1.0, top_p=0.7
        )
        self.assertEqual(truncated[2], 0.0)
        self.assertAlmostEqual(sum(truncated), 1.0)

    def test_rejection_correction_preserves_target_distribution(self) -> None:
        target = (0.60, 0.30, 0.10)
        draft = (0.20, 0.50, 0.30)
        counts = [0, 0, 0]
        samples = 100_000
        for seed_index in range(samples):
            seed = seed_index.to_bytes(32, "big")
            rng = CounterSamplingRng(seed)
            proposed = rng.categorical(draft)
            decision = lossless_speculative_sample(target, draft, proposed, rng)
            counts[decision.token_id] += 1
        observed = tuple(count / samples for count in counts)
        tvd = 0.5 * math.fsum(abs(left - right) for left, right in zip(observed, target))
        self.assertLess(tvd, 0.01)

    def test_draft_never_overrides_target_authority(self) -> None:
        rng = CounterSamplingRng(b"x" * 32)
        decision = lossless_speculative_sample((1.0, 0.0), (0.0, 1.0), 1, rng)
        self.assertFalse(decision.accepted_draft)
        self.assertEqual(decision.token_id, 0)
        self.assertEqual(decision.acceptance_probability, 0.0)

    def test_invalid_inputs_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "temperature"):
            probabilities_from_logits((1.0, 2.0), temperature=0.0)
        with self.assertRaisesRegex(ValueError, "finite"):
            probabilities_from_logits((1.0, math.nan), temperature=1.0)
        with self.assertRaisesRegex(ValueError, "vocabulary sizes"):
            lossless_speculative_sample(
                (0.5, 0.5), (1.0,), 0, CounterSamplingRng(b"y" * 32)
            )
        with self.assertRaisesRegex(ValueError, "zero draft probability"):
            lossless_speculative_sample(
                (0.5, 0.5), (1.0, 0.0), 1, CounterSamplingRng(b"z" * 32)
            )


if __name__ == "__main__":
    unittest.main()
