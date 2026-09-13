import unittest

import torch

from tests.cell_parity import assert_fp32_cell_close


class CellParityTests(unittest.TestCase):
    def test_accepts_bounded_cancellation_roundoff(self):
        expected = torch.tensor([[[65.0, 2.36, 0.0]]])
        actual = expected.clone()
        actual[0, 0, 1] += 4.2e-5
        assert_fp32_cell_close(actual, expected)

    def test_rejects_corruption_of_one_coordinate(self):
        expected = torch.full((1, 1, 16), 100.0)
        actual = expected.clone()
        actual[0, 0, 3] += 0.001
        with self.assertRaises(AssertionError):
            assert_fp32_cell_close(actual, expected)

    def test_larger_tokens_cannot_hide_another_tokens_error(self):
        expected = torch.tensor([[[1e6, 1e6], [1.0, 1.0]]])
        actual = expected.clone()
        actual[0, 1, 0] += 1e-4
        with self.assertRaises(AssertionError):
            assert_fp32_cell_close(actual, expected)

    def test_zero_reference_has_a_fixed_absolute_budget(self):
        expected = torch.zeros((1, 1, 16))
        with self.assertRaises(AssertionError):
            assert_fp32_cell_close(expected + 1e-5, expected)

    def test_rejects_broadcasting_dtype_changes_and_nonfinite_values(self):
        expected = torch.ones((1, 2, 16))
        for actual in (
            expected[:, :1, :],
            expected.double(),
            torch.full_like(expected, float("nan")),
            torch.full_like(expected, float("inf")),
        ):
            with self.subTest(shape=actual.shape, dtype=actual.dtype):
                with self.assertRaises(AssertionError):
                    assert_fp32_cell_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
