from __future__ import annotations

import unittest

import torch
from torch import nn
from transformers import LlamaConfig, LlamaModel

from distributed_runtime.model import StageModelSpec, StageRunner


def _runner(mode: str) -> StageRunner:
    torch.manual_seed(811)
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
    runner.spec = StageModelSpec("tiny", 0, 2, 2, 1, kv_cache=mode)
    runner.compute_device = torch.device("cpu")
    runner.caches = {}
    runner.tokens_seen = {}
    runner.active_requests = set()
    runner._physical_batch_cache_supported = True
    runner.model_forward_calls = 0
    runner.physical_batch_calls = 0
    runner.physical_batch_items = 0
    runner.max_observed_physical_batch_size = 1
    return runner


class ActivationCheckpointTests(unittest.TestCase):
    def test_restored_suffix_is_bit_exact_for_dynamic_and_arena_kv(self) -> None:
        prompt = torch.tensor([[1, 4, 7, 2, 9, 3, 5, 8]], dtype=torch.long)
        suffix = torch.tensor([[6]], dtype=torch.long)
        for mode in ("dynamic", "arena"):
            with self.subTest(mode=mode):
                original = _runner(mode)
                original.begin(1)
                original.forward_ids(1, prompt)
                payload = original.activation_checkpoint_payload(1)
                position = original.sequence_length(1)
                expected = original.forward_ids(1, suffix)

                restored = _runner(mode)
                restored.restore_activation_checkpoint_payload(1, payload, position)
                actual = restored.forward_ids(1, suffix)
                self.assertTrue(torch.equal(actual, expected))
                self.assertEqual(restored.sequence_length(1), position + 1)

    def test_corruption_and_limits_fail_before_publishing_request_state(self) -> None:
        runner = _runner("arena")
        runner.begin(1)
        runner.forward_ids(1, torch.tensor([[1, 2, 3, 4]], dtype=torch.long))
        payload = runner.activation_checkpoint_payload(1)

        restored = _runner("arena")
        with self.assertRaisesRegex(ValueError, "Safetensors"):
            restored.restore_activation_checkpoint_payload(1, payload[:-1], 4)
        self.assertNotIn(1, restored.active_requests)
        self.assertNotIn(1, restored.caches)

        with self.assertRaisesRegex(ValueError, "exceeds limit"):
            runner.activation_checkpoint_payload(1, max_bytes=len(payload) - 1)

    def test_restore_rejects_wrong_position_and_active_request(self) -> None:
        runner = _runner("dynamic")
        runner.begin(1)
        runner.forward_ids(1, torch.tensor([[1, 2, 3]], dtype=torch.long))
        payload = runner.activation_checkpoint_payload(1)

        restored = _runner("dynamic")
        with self.assertRaisesRegex(ValueError, "geometry"):
            restored.restore_activation_checkpoint_payload(1, payload, 2)
        self.assertNotIn(1, restored.active_requests)
        restored.begin(1)
        with self.assertRaisesRegex(ValueError, "already active"):
            restored.restore_activation_checkpoint_payload(1, payload, 3)


if __name__ == "__main__":
    unittest.main()
