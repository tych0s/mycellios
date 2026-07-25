"""The arena cache must be indistinguishable from the concatenating cache.

Every assertion here compares against ``DynamicCache`` directly rather than against
a recorded golden value: the claim being defended is *equivalence*, so the reference
has to be the thing it replaces.
"""

from __future__ import annotations

import copy
import unittest

import torch
from transformers.cache_utils import DynamicCache, DynamicLayer

from distributed_runtime.kv_arena import (
    MINIMUM_ARENA_TOKENS,
    ArenaCache,
    ArenaLayer,
    arena_reserved_bytes,
    is_arena_layer,
    stack_into_arena,
)


HEADS = 3
HEAD_DIM = 4


def _states(tokens: int, batch: int = 1, seed: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    if seed is not None:
        torch.manual_seed(seed)
    keys = torch.randn(batch, HEADS, tokens, HEAD_DIM)
    values = torch.randn(batch, HEADS, tokens, HEAD_DIM)
    return keys, values


class ArenaLayerEquivalenceTests(unittest.TestCase):
    def test_append_sequence_matches_concatenating_layer_exactly(self) -> None:
        arena = ArenaLayer()
        dynamic = DynamicLayer()
        torch.manual_seed(11)
        for step in range(40):
            keys, values = _states(1)
            arena_keys, arena_values = arena.update(keys, values)
            dynamic_keys, dynamic_values = dynamic.update(keys, values)
            self.assertTrue(torch.equal(arena_keys, dynamic_keys), f"keys diverged at step {step}")
            self.assertTrue(torch.equal(arena_values, dynamic_values), f"values diverged at step {step}")
            self.assertEqual(arena.get_seq_length(), dynamic.get_seq_length())

    def test_prefill_then_decode_matches_concatenating_layer(self) -> None:
        arena = ArenaLayer()
        dynamic = DynamicLayer()
        torch.manual_seed(12)
        prefill_keys, prefill_values = _states(37)
        arena.update(prefill_keys, prefill_values)
        dynamic.update(prefill_keys, prefill_values)
        for _ in range(9):
            keys, values = _states(1)
            arena.update(keys, values)
            dynamic.update(keys, values)
        self.assertTrue(torch.equal(arena.keys, dynamic.keys))
        self.assertTrue(torch.equal(arena.values, dynamic.values))

    def test_growth_is_amortised_and_capacity_covers_length(self) -> None:
        arena = ArenaLayer()
        torch.manual_seed(13)
        for _ in range(MINIMUM_ARENA_TOKENS + 5):
            arena.update(*_states(1))
        self.assertGreaterEqual(arena.arena_capacity_tokens, arena.get_seq_length())
        self.assertLessEqual(arena.arena_capacity_tokens, 2 * MINIMUM_ARENA_TOKENS)

    def test_crop_then_append_matches_concatenating_layer(self) -> None:
        """Speculative rollback: rejecting drafts must land on the same bytes."""

        arena = ArenaLayer()
        dynamic = DynamicLayer()
        torch.manual_seed(14)
        for _ in range(20):
            keys, values = _states(1)
            arena.update(keys, values)
            dynamic.update(keys, values)
        arena.crop(12)
        dynamic.crop(12)
        self.assertEqual(arena.get_seq_length(), 12)
        self.assertTrue(torch.equal(arena.keys, dynamic.keys))
        for _ in range(6):
            keys, values = _states(1)
            arena_keys, _ = arena.update(keys, values)
            dynamic_keys, _ = dynamic.update(keys, values)
            self.assertTrue(torch.equal(arena_keys, dynamic_keys))

    def test_crop_reuses_capacity_instead_of_reallocating(self) -> None:
        arena = ArenaLayer()
        torch.manual_seed(15)
        for _ in range(30):
            arena.update(*_states(1))
        capacity = arena.arena_capacity_tokens
        storage = arena.keys.untyped_storage().data_ptr()
        arena.crop(10)
        arena.update(*_states(1))
        self.assertEqual(arena.arena_capacity_tokens, capacity)
        self.assertEqual(arena.keys.untyped_storage().data_ptr(), storage)

    def test_reset_empties_the_layer(self) -> None:
        arena = ArenaLayer()
        torch.manual_seed(16)
        arena.update(*_states(5))
        arena.reset()
        self.assertEqual(arena.get_seq_length(), 0)
        keys, values = _states(1)
        arena_keys, _ = arena.update(keys, values)
        self.assertTrue(torch.equal(arena_keys, keys))

    def test_deepcopy_is_exact_and_shares_no_storage(self) -> None:
        arena = ArenaLayer()
        torch.manual_seed(17)
        for _ in range(15):
            arena.update(*_states(1))
        clone = copy.deepcopy(arena)
        self.assertTrue(torch.equal(clone.keys, arena.keys))
        self.assertNotEqual(
            clone.keys.untyped_storage().data_ptr(),
            arena.keys.untyped_storage().data_ptr(),
        )
        # Diverging the clone must not touch the parent: this is the invariant the
        # exact speculative fork depends on.
        before = arena.keys.clone()
        clone.update(*_states(1))
        self.assertTrue(torch.equal(arena.keys, before))

    def test_deepcopy_of_untouched_layer_is_safe(self) -> None:
        cache = ArenaCache()
        self.assertEqual(len(copy.deepcopy(cache).layers), 0)

    def test_batch_select_indices_matches_concatenating_layer(self) -> None:
        arena = ArenaLayer()
        dynamic = DynamicLayer()
        torch.manual_seed(18)
        keys, values = _states(6, batch=4)
        arena.update(keys, values)
        dynamic.update(keys, values)
        indices = torch.tensor([2, 0])
        arena.batch_select_indices(indices)
        dynamic.batch_select_indices(indices)
        self.assertTrue(torch.equal(arena.keys, dynamic.keys))
        follow_keys, follow_values = _states(1, batch=2)
        arena_keys, _ = arena.update(follow_keys, follow_values)
        dynamic_keys, _ = dynamic.update(follow_keys, follow_values)
        self.assertTrue(torch.equal(arena_keys, dynamic_keys))

    def test_batch_repeat_interleave_matches_concatenating_layer(self) -> None:
        arena = ArenaLayer()
        dynamic = DynamicLayer()
        torch.manual_seed(19)
        keys, values = _states(3, batch=2)
        arena.update(keys, values)
        dynamic.update(keys, values)
        arena.batch_repeat_interleave(2)
        dynamic.batch_repeat_interleave(2)
        self.assertTrue(torch.equal(arena.keys, dynamic.keys))


class ArenaCacheTests(unittest.TestCase):
    def test_cache_reports_arena_layers(self) -> None:
        cache = ArenaCache()
        cache.update(*_states(1, seed=20), 0)
        self.assertTrue(is_arena_layer(cache.layers[0]))
        self.assertGreater(arena_reserved_bytes(cache), 0)

    def test_cache_matches_dynamic_cache_across_layers(self) -> None:
        arena = ArenaCache()
        dynamic = DynamicCache()
        torch.manual_seed(21)
        for step in range(8):
            for layer_index in range(3):
                keys, values = _states(1)
                arena_keys, _ = arena.update(keys, values, layer_index)
                dynamic_keys, _ = dynamic.update(keys, values, layer_index)
                self.assertTrue(
                    torch.equal(arena_keys, dynamic_keys),
                    f"layer {layer_index} diverged at step {step}",
                )

    def test_reserved_bytes_is_zero_for_an_untouched_cache(self) -> None:
        self.assertEqual(arena_reserved_bytes(ArenaCache()), 0)


class StackIntoArenaTests(unittest.TestCase):
    def test_stack_matches_torch_cat(self) -> None:
        torch.manual_seed(22)
        key_slices = [_states(7)[0] for _ in range(4)]
        value_slices = [_states(7)[1] for _ in range(4)]
        layer = ArenaLayer()
        stack_into_arena(layer, key_slices, value_slices)
        self.assertTrue(torch.equal(layer.keys, torch.cat(key_slices, dim=0)))
        self.assertTrue(torch.equal(layer.values, torch.cat(value_slices, dim=0)))

    def test_stacked_batch_keeps_appending_correctly(self) -> None:
        torch.manual_seed(23)
        key_slices = [_states(5)[0] for _ in range(3)]
        value_slices = [_states(5)[1] for _ in range(3)]
        layer = ArenaLayer()
        stack_into_arena(layer, key_slices, value_slices)
        expected = torch.cat(key_slices, dim=0)
        tail_keys, tail_values = _states(1, batch=3)
        got, _ = layer.update(tail_keys, tail_values)
        self.assertTrue(torch.equal(got, torch.cat([expected, tail_keys], dim=-2)))

    def test_headroom_absorbs_the_forward_append_without_reallocating(self) -> None:
        """The regression this guards: an arena sized exactly to the history grows
        on the model's first append, and that growth copies the entire batch."""

        torch.manual_seed(24)
        key_slices = [_states(64)[0] for _ in range(4)]
        value_slices = [_states(64)[1] for _ in range(4)]
        layer = ArenaLayer()
        stack_into_arena(layer, key_slices, value_slices, headroom=1)
        storage = layer.keys.untyped_storage().data_ptr()
        capacity = layer.arena_capacity_tokens
        layer.update(*_states(1, batch=4))
        self.assertEqual(layer.arena_capacity_tokens, capacity)
        self.assertEqual(layer.keys.untyped_storage().data_ptr(), storage)

    def test_without_headroom_the_arena_is_sized_to_the_history(self) -> None:
        torch.manual_seed(25)
        key_slices = [_states(48)[0] for _ in range(2)]
        value_slices = [_states(48)[1] for _ in range(2)]
        layer = ArenaLayer()
        stack_into_arena(layer, key_slices, value_slices)
        self.assertEqual(layer.arena_capacity_tokens, 48)

    def test_negative_headroom_is_refused(self) -> None:
        torch.manual_seed(26)
        with self.assertRaises(ValueError):
            stack_into_arena(ArenaLayer(), [_states(2)[0]], [_states(2)[1]], headroom=-1)

    def test_empty_batch_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            stack_into_arena(ArenaLayer(), [], [])


if __name__ == "__main__":
    unittest.main()
