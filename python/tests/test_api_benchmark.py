from __future__ import annotations

import unittest

from distributed_runtime.api_benchmark import percentile, positive_csv


class ApiBenchmarkUtilityTests(unittest.TestCase):
    def test_concurrency_list_is_strictly_positive(self) -> None:
        self.assertEqual(positive_csv("1, 2,8"), (1, 2, 8))
        for raw in ("", "0,1", "1,nope"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                positive_csv(raw)

    def test_percentile_interpolates(self) -> None:
        self.assertEqual(percentile([7], 0.95), 7)
        self.assertAlmostEqual(percentile([0, 10], 0.95), 9.5)
        with self.assertRaises(ValueError):
            percentile([], 0.95)


if __name__ == "__main__":
    unittest.main()
