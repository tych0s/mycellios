"""The arena must be invisible in the runtime, and must actually take the fast path.

Two separate claims are checked here, because a silent fallback would keep every
equivalence test green while giving back none of the performance:

1. Equivalence -- a stage running the arena produces bit-identical hidden states to
   the same stage running Transformers' concatenating cache, single and batched.
2. Behaviour -- the batched commit really appends only the new positions instead of
   rebuilding each request's cache, which is observable as the request keeping the
   same storage across decode steps.
"""

from __future__ import annotations

import unittest

import torch
from torch import nn
from transformers import LlamaConfig, LlamaModel

from distributed_runtime.kv_arena import is_arena_layer
from distributed_runtime.model import StageModelSpec, StageRunner


def _tiny_runner(kv_cache: str) -> StageRunner:
    torch.manual_seed(67)
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
    )
    runner = StageRunner.__new__(StageRunner)
    runner.base = LlamaModel(config).eval()
    runner.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False).eval()
    runner.hidden_size = config.hidden_size
    runner.spec = StageModelSpec("tiny", 0, 2, 2, 1, kv_cache=kv_cache)
    runner.caches = {}
    runner.tokens_seen = {}
    runner.active_requests = set()
    runner._physical_batch_cache_supported = True
    runner.model_forward_calls = 0
    runner.physical_batch_calls = 0
    runner.physical_batch_items = 0
    runner.max_observed_physical_batch_size = 1
    return runner


def _prompt(tokens: int, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randint(0, 32, (1, tokens), dtype=torch.long)


class ArenaRuntimeEquivalenceTests(unittest.TestCase):
    def test_single_request_decode_matches_dynamic_cache(self) -> None:
        prompt = _prompt(24, seed=3)
        outputs = {}
        for mode in ("dynamic", "arena"):
            runner = _tiny_runner(mode)
            runner.begin(1)
            runner.forward_ids(1, prompt)
            for _ in range(12):
                hidden = runner.forward_ids(1, torch.tensor([[7]], dtype=torch.long))
            outputs[mode] = hidden
            self.assertEqual(runner.sequence_length(1), 24 + 12)
        self.assertTrue(torch.equal(outputs["dynamic"], outputs["arena"]))

    def test_batched_decode_matches_dynamic_cache(self) -> None:
        prompt = _prompt(20, seed=4)
        request_ids = (1, 2, 3)
        outputs = {}
        for mode in ("dynamic", "arena"):
            runner = _tiny_runner(mode)
            for request_id in request_ids:
                runner.begin(request_id)
                runner.forward_ids(request_id, prompt)
            for _ in range(10):
                hidden = runner.forward_ids_batch(
                    request_ids,
                    [torch.tensor([[5]], dtype=torch.long) for _ in request_ids],
                )
            outputs[mode] = torch.cat(hidden, dim=0)
            for request_id in request_ids:
                self.assertEqual(runner.sequence_length(request_id), 30)
        self.assertTrue(torch.equal(outputs["dynamic"], outputs["arena"]))

    def test_speculative_rollback_matches_dynamic_cache(self) -> None:
        """Truncate is the rollback primitive; the arena moves a cursor instead of slicing."""

        prompt = _prompt(16, seed=5)
        outputs = {}
        for mode in ("dynamic", "arena"):
            runner = _tiny_runner(mode)
            runner.begin(1)
            runner.forward_ids(1, prompt)
            for _ in range(6):
                runner.forward_ids(1, torch.tensor([[9]], dtype=torch.long))
            runner.truncate(1, 18)
            self.assertEqual(runner.sequence_length(1), 18)
            for _ in range(4):
                hidden = runner.forward_ids(1, torch.tensor([[11]], dtype=torch.long))
            outputs[mode] = hidden
        self.assertTrue(torch.equal(outputs["dynamic"], outputs["arena"]))

    def test_fork_then_diverge_matches_dynamic_cache(self) -> None:
        prompt = _prompt(16, seed=6)
        outputs = {}
        for mode in ("dynamic", "arena"):
            runner = _tiny_runner(mode)
            runner.begin(1)
            runner.forward_ids(1, prompt)
            runner.fork_request(2, 1, max_cache_bytes=1 << 30)
            for _ in range(3):
                runner.forward_ids(2, torch.tensor([[13]], dtype=torch.long))
            parent = runner.forward_ids(1, torch.tensor([[13]], dtype=torch.long))
            outputs[mode] = parent
            # The branch must not have moved the parent forward.
            self.assertEqual(runner.sequence_length(1), 17)
            self.assertEqual(runner.sequence_length(2), 19)
        self.assertTrue(torch.equal(outputs["dynamic"], outputs["arena"]))


class ArenaRuntimeBehaviourTests(unittest.TestCase):
    def test_stage_uses_arena_layers_by_default(self) -> None:
        runner = _tiny_runner("arena")
        runner.begin(1)
        runner.forward_ids(1, _prompt(8, seed=7))
        self.assertTrue(all(is_arena_layer(layer) for layer in runner.caches[1].layers))

    def test_batched_commit_appends_instead_of_rebuilding(self) -> None:
        """A full split would hand each request a brand-new cache every token.

        Keeping the same storage across steps is the observable signature of the
        append-only path; if this regresses, the batched decode silently goes back
        to copying the whole history per token.
        """

        runner = _tiny_runner("arena")
        request_ids = (1, 2)
        for request_id in request_ids:
            runner.begin(request_id)
            runner.forward_ids(request_id, _prompt(12, seed=8))
        caches = {request_id: runner.caches[request_id] for request_id in request_ids}
        pointers = {
            request_id: caches[request_id].layers[0].keys.untyped_storage().data_ptr()
            for request_id in request_ids
        }
        for _ in range(5):
            runner.forward_ids_batch(
                request_ids,
                [torch.tensor([[3]], dtype=torch.long) for _ in request_ids],
            )
        for request_id in request_ids:
            self.assertIs(runner.caches[request_id], caches[request_id])
            self.assertEqual(
                runner.caches[request_id].layers[0].keys.untyped_storage().data_ptr(),
                pointers[request_id],
            )

    def test_batched_requests_never_share_storage(self) -> None:
        runner = _tiny_runner("arena")
        request_ids = (1, 2, 3)
        for request_id in request_ids:
            runner.begin(request_id)
            runner.forward_ids(request_id, _prompt(10, seed=9))
        for _ in range(3):
            runner.forward_ids_batch(
                request_ids,
                [torch.tensor([[4]], dtype=torch.long) for _ in request_ids],
            )
        pointers = [
            runner.caches[request_id].layers[0].keys.untyped_storage().data_ptr()
            for request_id in request_ids
        ]
        self.assertEqual(len(set(pointers)), len(pointers))

    def test_dynamic_mode_still_available_for_the_ab(self) -> None:
        runner = _tiny_runner("dynamic")
        runner.begin(1)
        runner.forward_ids(1, _prompt(8, seed=10))
        self.assertFalse(any(is_arena_layer(layer) for layer in runner.caches[1].layers))

    def test_unknown_kv_cache_mode_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            StageModelSpec("tiny", 0, 2, 2, 1, kv_cache="paged")


if __name__ == "__main__":
    unittest.main()
