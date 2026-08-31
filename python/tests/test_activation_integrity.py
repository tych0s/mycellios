from __future__ import annotations

import copy
import math
import unittest

from distributed_runtime.activation_integrity import (
    SCHEMA, ActivationSketchError, compare_activation_sketches, generate_seed,
    projection_indices, seed_commitment, verify_seed_commitment,
)


class ActivationIntegrityTests(unittest.TestCase):
    def sketch(self, values: list[float], *, seed: str = "01" * 16,
               norm: float = 10.0) -> dict:
        return {"schema": SCHEMA, "seed": seed,
                "seedCommitment": seed_commitment(seed), "elementCount": 4096,
                "sampleCount": len(values), "norm": norm, "projection": values}

    def test_seed_commitment_is_fresh_and_binding(self) -> None:
        first, second = generate_seed(), generate_seed()
        self.assertNotEqual(first, second)
        commitment = seed_commitment(first)
        self.assertTrue(verify_seed_commitment(first, commitment))
        self.assertFalse(verify_seed_commitment(second, commitment))

    def test_projection_indices_are_deterministic_unique_and_seeded(self) -> None:
        first = projection_indices("02" * 16, 10_000, 256)
        self.assertEqual(first, projection_indices("02" * 16, 10_000, 256))
        self.assertEqual(len(first), len(set(first)))
        self.assertNotEqual(first, projection_indices("03" * 16, 10_000, 256))

    def test_projection_bounds_fail_closed(self) -> None:
        with self.assertRaisesRegex(ActivationSketchError, "sample_count"):
            projection_indices("04" * 16, 8, 9)
        with self.assertRaisesRegex(ActivationSketchError, "seed"):
            projection_indices("not-a-seed", 8, 4)

    def test_honest_numeric_drift_passes(self) -> None:
        trusted = self.sketch([float(index + 1) for index in range(32)])
        suspect = copy.deepcopy(trusted)
        suspect["projection"] = [value * 1.00001 for value in trusted["projection"]]
        suspect["norm"] = trusted["norm"] * 1.00001
        verdict = compare_activation_sketches(suspect, trusted)
        self.assertTrue(verdict["passed"])
        self.assertGreater(verdict["cosine"], 0.9999)

    def test_wrong_activation_fails(self) -> None:
        trusted = self.sketch([float(index + 1) for index in range(32)])
        suspect = self.sketch([float((-1) ** index * (index + 1)) for index in range(32)])
        verdict = compare_activation_sketches(suspect, trusted)
        self.assertFalse(verdict["passed"])
        self.assertEqual(verdict["error"], "activation_sketch_diverged")

    def test_malformed_or_mismatched_sketches_fail(self) -> None:
        trusted = self.sketch([1.0, 2.0, 3.0, 4.0])
        wrong_seed = self.sketch([1.0, 2.0, 3.0, 4.0], seed="05" * 16)
        wrong_commitment = copy.deepcopy(trusted)
        wrong_commitment["seedCommitment"] = seed_commitment("06" * 16)
        wrong_shape = copy.deepcopy(trusted)
        wrong_shape["elementCount"] += 1
        non_finite = copy.deepcopy(trusted)
        non_finite["projection"][0] = math.nan
        oversized = copy.deepcopy(trusted)
        oversized["sampleCount"] = 1025
        oversized["projection"] = [1.0] * 1025
        for suspect in (wrong_seed, wrong_commitment, wrong_shape, non_finite, oversized):
            with self.subTest(suspect=suspect):
                self.assertFalse(compare_activation_sketches(suspect, trusted)["passed"])


if __name__ == "__main__":
    unittest.main()
