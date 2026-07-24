"""Grouped-prefix decode attention: same function, nothing materialised.

The eligibility gate is the part that matters for correctness.  It must take the
fast path exactly when the decode step is a single query position over a dense,
fully visible prefix, and must hand everything else back to the stock
implementation -- silently mishandling a mask would change tokens.
"""

from __future__ import annotations

import unittest

import torch
from transformers.integrations.sdpa_attention import sdpa_attention_forward

from distributed_runtime.decode_attention import (
    GROUPED_PREFIX_ATTENTION,
    STATS,
    grouped_prefix_attention_forward,
    register_grouped_prefix_attention,
    reset_stats,
)

HEADS = 9
KV_HEADS = 3
HEAD_DIM = 64
SCALING = HEAD_DIM**-0.5


class _Module(torch.nn.Module):
    num_key_value_groups = HEADS // KV_HEADS


def _inputs(batch: int, length: int, query_tokens: int = 1, dtype=torch.float32):
    torch.manual_seed(31)
    query = torch.randn(batch, HEADS, query_tokens, HEAD_DIM, dtype=dtype)
    key = torch.randn(batch, KV_HEADS, length, HEAD_DIM, dtype=dtype)
    value = torch.randn(batch, KV_HEADS, length, HEAD_DIM, dtype=dtype)
    return query, key, value


def _stock(module, query, key, value, mask):
    out, _ = sdpa_attention_forward(
        module, query, key, value, mask, dropout=0.0, scaling=SCALING, is_causal=False
    )
    return out.reshape(query.shape[0], query.shape[2], -1)


def _grouped(module, query, key, value, mask):
    out, _ = grouped_prefix_attention_forward(
        module, query, key, value, mask, dropout=0.0, scaling=SCALING, is_causal=False
    )
    return out.reshape(query.shape[0], query.shape[2], -1)


class GroupedPrefixEquivalenceTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_stats()
        self.module = _Module()

    def _assert_matches_stock(self, batch: int, length: int, tolerance: float) -> None:
        query, key, value = _inputs(batch, length)
        stock = _stock(self.module, query, key, value, None)
        grouped = _grouped(self.module, query, key, value, None)
        self.assertEqual(stock.shape, grouped.shape)
        deviation = (stock - grouped).abs().max().item()
        self.assertLess(deviation, tolerance, f"deviation {deviation:.3e} at B={batch} L={length}")

    def test_matches_stock_attention_for_single_stream(self) -> None:
        self._assert_matches_stock(1, 777, 1e-5)

    def test_matches_stock_attention_for_a_batch(self) -> None:
        self._assert_matches_stock(4, 512, 1e-5)

    def test_matches_stock_attention_for_a_short_prefix(self) -> None:
        self._assert_matches_stock(1, 1, 1e-5)

    def test_deviation_stays_at_float32_rounding(self) -> None:
        """The claim being defended is 'reference-equivalent', not 'bit-identical'.

        A different GEMM shape means a different reduction order. The deviation must
        stay at FP32 rounding -- orders of magnitude below the FP16 activation codec
        the pipeline already uses by default -- or the claim is not honest.
        """

        query, key, value = _inputs(1, 2048)
        deviation = (
            (_stock(self.module, query, key, value, None) - _grouped(self.module, query, key, value, None))
            .abs()
            .max()
            .item()
        )
        self.assertLess(deviation, 1e-6)

    def test_reads_a_strided_cache_view_without_copying(self) -> None:
        """The cache hands over a view into an arena; it must not be made contiguous."""

        torch.manual_seed(32)
        arena_keys = torch.randn(1, KV_HEADS, 4096, HEAD_DIM)
        arena_values = torch.randn(1, KV_HEADS, 4096, HEAD_DIM)
        key, value = arena_keys[..., :1000, :], arena_values[..., :1000, :]
        self.assertFalse(key.is_contiguous())
        query = torch.randn(1, HEADS, 1, HEAD_DIM)
        grouped = _grouped(self.module, query, key, value, None)
        stock = _stock(self.module, query, key, value, None)
        self.assertLess((stock - grouped).abs().max().item(), 1e-5)
        self.assertEqual(STATS["grouped"], 1)


class GroupedPrefixEligibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_stats()
        self.module = _Module()

    def test_single_decode_position_takes_the_fast_path(self) -> None:
        query, key, value = _inputs(1, 128)
        _grouped(self.module, query, key, value, None)
        self.assertEqual((STATS["grouped"], STATS["fallback"]), (1, 0))

    def test_prefill_falls_back(self) -> None:
        query, key, value = _inputs(1, 64, query_tokens=8)
        _grouped(self.module, query, key, value, None)
        self.assertEqual((STATS["grouped"], STATS["fallback"]), (0, 1))

    def test_all_visible_boolean_mask_takes_the_fast_path(self) -> None:
        query, key, value = _inputs(1, 96)
        mask = torch.ones(1, 1, 1, 96, dtype=torch.bool)
        _grouped(self.module, query, key, value, mask)
        self.assertEqual((STATS["grouped"], STATS["fallback"]), (1, 0))

    def test_zero_additive_mask_takes_the_fast_path(self) -> None:
        query, key, value = _inputs(1, 96)
        mask = torch.zeros(1, 1, 1, 96)
        _grouped(self.module, query, key, value, mask)
        self.assertEqual((STATS["grouped"], STATS["fallback"]), (1, 0))

    def test_padding_mask_falls_back(self) -> None:
        query, key, value = _inputs(1, 96)
        mask = torch.ones(1, 1, 1, 96, dtype=torch.bool)
        mask[..., :5] = False
        _grouped(self.module, query, key, value, mask)
        self.assertEqual((STATS["grouped"], STATS["fallback"]), (0, 1))

    def test_masked_positions_still_produce_stock_results(self) -> None:
        query, key, value = _inputs(1, 96)
        mask = torch.zeros(1, 1, 1, 96)
        mask[..., :5] = float("-inf")
        stock = _stock(self.module, query, key, value, mask)
        grouped = _grouped(self.module, query, key, value, mask)
        self.assertTrue(torch.equal(stock, grouped))
        self.assertEqual(STATS["fallback"], 1)

    def test_short_mask_falls_back_and_lets_the_stock_error_surface(self) -> None:
        """A mask shorter than the cache cannot be proven a no-op.

        The fast path must not quietly decide the missing entries are visible; it
        hands the call to the stock implementation, which rejects it. Failing loudly
        on malformed input is the wanted behaviour.
        """

        query, key, value = _inputs(1, 96)
        mask = torch.ones(1, 1, 1, 32, dtype=torch.bool)
        with self.assertRaises(RuntimeError):
            _grouped(self.module, query, key, value, mask)
        self.assertEqual((STATS["grouped"], STATS["fallback"]), (0, 1))

    def test_output_attentions_falls_back(self) -> None:
        query, key, value = _inputs(1, 96)
        grouped_prefix_attention_forward(
            self.module,
            query,
            key,
            value,
            None,
            dropout=0.0,
            scaling=SCALING,
            output_attentions=False,
        )
        self.assertEqual(STATS["grouped"], 1)

    def test_dropout_falls_back(self) -> None:
        query, key, value = _inputs(1, 96)
        grouped_prefix_attention_forward(
            self.module, query, key, value, None, dropout=0.1, scaling=SCALING
        )
        self.assertEqual((STATS["grouped"], STATS["fallback"]), (0, 1))

    def test_head_counts_that_do_not_divide_fall_back(self) -> None:
        """7 query heads over 3 KV heads is not a grouping; the regroup would be wrong."""

        query = torch.randn(1, 7, 1, HEAD_DIM)
        key = torch.randn(1, KV_HEADS, 16, HEAD_DIM)
        value = torch.randn(1, KV_HEADS, 16, HEAD_DIM)
        with self.assertRaises(RuntimeError):
            grouped_prefix_attention_forward(
                self.module, query, key, value, None, dropout=0.0, scaling=SCALING
            )
        self.assertEqual((STATS["grouped"], STATS["fallback"]), (0, 1))

    def test_multi_query_grouping_is_handled(self) -> None:
        """Hq == Hkv (no grouping) must still be exact, with G collapsing to 1."""

        torch.manual_seed(33)
        query = torch.randn(1, KV_HEADS, 1, HEAD_DIM)
        key = torch.randn(1, KV_HEADS, 200, HEAD_DIM)
        value = torch.randn(1, KV_HEADS, 200, HEAD_DIM)
        stock = _stock(self.module, query, key, value, None)
        grouped = _grouped(self.module, query, key, value, None)
        self.assertLess((stock - grouped).abs().max().item(), 1e-5)
        self.assertEqual(STATS["grouped"], 1)

    def test_default_scaling_is_applied_when_absent(self) -> None:
        query, key, value = _inputs(1, 64)
        out, _ = grouped_prefix_attention_forward(
            self.module, query, key, value, None, dropout=0.0, scaling=None
        )
        reference = _stock(self.module, query, key, value, None)
        self.assertLess(
            (out.reshape(1, 1, -1) - reference).abs().max().item(), 1e-5
        )


class RegistrationTests(unittest.TestCase):
    def test_registers_under_a_selectable_name(self) -> None:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        name = register_grouped_prefix_attention()
        self.assertEqual(name, GROUPED_PREFIX_ATTENTION)
        self.assertIn(GROUPED_PREFIX_ATTENTION, ALL_ATTENTION_FUNCTIONS.keys())


if __name__ == "__main__":
    unittest.main()
