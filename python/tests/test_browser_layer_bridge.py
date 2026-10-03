"""A browser-owned Qwen3 layer must preserve whole-model output and native KV fallback."""

from __future__ import annotations

import base64
import copy
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.cache_utils import DynamicCache

from distributed_runtime.browser_expert_owner import BrowserExpertOwnerError
from distributed_runtime.browser_layer_bridge import (
    BrowserQwen3Layer, attach_browser_layer_bridge, browser_layer_request,
)


def _decode(value: str, shape: tuple[int, ...]) -> torch.Tensor:
    data = base64.b64decode(value)
    return torch.frombuffer(bytearray(data), dtype=torch.float32).reshape(shape)


def _encode(value: torch.Tensor) -> str:
    return base64.b64encode(value.detach().contiguous().numpy().tobytes()).decode("ascii")


class LocalBrowser:
    def __init__(self, layer):
        self.layer = layer
        self.caches = {}
        self.calls = 0
        self.fail_on = 3

    def _request_json(self, _method, path, payload):
        if path.endswith("/prepare"):
            return {"resident": True}
        if path.endswith("/reset"):
            self.caches.pop(payload["requestId"], None)
            return {"reset": True}
        self.calls += 1
        if self.calls == self.fail_on:
            raise BrowserExpertOwnerError("browser disappeared")
        tokens = payload["tokens"]
        position = payload["position"]
        width = self.layer.self_attn.config.hidden_size
        dim = self.layer.self_attn.head_dim
        cache = self.caches.setdefault(payload["requestId"], DynamicCache())
        hidden = _decode(payload["hiddenBase64"], (1, tokens, width))
        auxiliary = payload["auxiliaryBase64"]
        cos = _decode(auxiliary["cos"], (1, tokens, dim))
        sin = _decode(auxiliary["sin"], (1, tokens, dim))
        mask = _decode(auxiliary["attention_mask"], (1, 1, tokens, position + tokens))
        output = self.layer(hidden, attention_mask=mask,
                            position_embeddings=(cos, sin), past_key_values=cache,
                            use_cache=True)
        key = cache.layers[0].keys[:, :, -tokens:, :]
        value = cache.layers[0].values[:, :, -tokens:, :]
        return {"artifactId": payload["artifactId"], "requestId": payload["requestId"],
                "position": position, "tokens": tokens,
                "outputBase64": _encode(output),
                "stateBase64": {"key": _encode(key), "value": _encode(value)}}


def test_browser_layer_continues_full_model_and_falls_back_with_exact_kv():
    torch.manual_seed(480)
    torch.set_num_threads(2)
    config = Qwen3Config(vocab_size=32, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=8,
                         max_position_embeddings=64, _attn_implementation="eager")
    reference = Qwen3ForCausalLM(config).eval()
    mixed = copy.deepcopy(reference).eval()
    browser = LocalBrowser(copy.deepcopy(reference.model.layers[0]).eval())
    wrapper = BrowserQwen3Layer(mixed.model.layers[0], browser,
                                "a" * 64, 0, 64)
    mixed.model.layers[0] = wrapper
    reference_cache = DynamicCache(config=config)
    mixed_cache = DynamicCache(config=config)
    for index, ids in enumerate((torch.tensor([[1, 2, 3]]),
                                 torch.tensor([[4]]), torch.tensor([[5]]),
                                 torch.tensor([[6]]))):
        with torch.inference_mode():
            expected = reference(input_ids=ids, past_key_values=reference_cache,
                                 use_cache=True).logits
            with browser_layer_request(7):
                actual = mixed(input_ids=ids, past_key_values=mixed_cache,
                               use_cache=True).logits
        assert torch.max(torch.abs(expected - actual)).item() < 2e-5
        for layer_index in range(2):
            torch.testing.assert_close(reference_cache.layers[layer_index].keys,
                                       mixed_cache.layers[layer_index].keys,
                                       atol=2e-5, rtol=2e-5)
        if index == 2:
            assert 7 in wrapper.disabled_requests
    assert wrapper.browser_forwards == 2
    assert wrapper.native_forwards == 2
    wrapper.end(7)
    assert 7 not in wrapper.disabled_requests


def test_llama_layer_uses_the_same_browser_bridge_and_native_fallback():
    torch.manual_seed(481)
    torch.set_num_threads(2)
    config = LlamaConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, max_position_embeddings=64,
                         _attn_implementation="eager")
    reference = LlamaForCausalLM(config).eval()
    mixed = copy.deepcopy(reference).eval()
    browser = LocalBrowser(copy.deepcopy(reference.model.layers[0]).eval())
    wrapper = BrowserQwen3Layer(mixed.model.layers[0], browser, "b" * 64, 0, 64)
    mixed.model.layers[0] = wrapper
    expected_cache = DynamicCache(config=config)
    actual_cache = DynamicCache(config=config)
    for ids in (torch.tensor([[1, 2, 3]]), torch.tensor([[4]]),
                torch.tensor([[5]]), torch.tensor([[6]])):
        with torch.inference_mode():
            expected = reference(input_ids=ids, past_key_values=expected_cache,
                                 use_cache=True).logits
            with browser_layer_request(8):
                actual = mixed(input_ids=ids, past_key_values=actual_cache,
                               use_cache=True).logits
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(actual_cache.layers[0].keys,
                                   expected_cache.layers[0].keys,
                                   atol=2e-5, rtol=2e-5)
    assert wrapper.browser_forwards == 2
    assert wrapper.native_forwards == 2


class BrowserLayerBridgeTests(unittest.TestCase):
    def test_qwen3_full_model_and_recovery(self):
        test_browser_layer_continues_full_model_and_falls_back_with_exact_kv()

    def test_llama_full_model_and_recovery(self):
        test_llama_layer_uses_the_same_browser_bridge_and_native_fallback()

    def test_published_model_layer_is_discovered_without_a_per_model_bridge_file(self):
        config = Qwen3Config(vocab_size=32, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=2, num_attention_heads=4,
                             num_key_value_heads=2, head_dim=8,
                             max_position_embeddings=64, _attn_implementation="eager")
        runner = SimpleNamespace(
            base=Qwen3ForCausalLM(config).model,
            spec=SimpleNamespace(layer_start=0, layer_end=2,
                                 quantize=None, compile_mode=None),
            executor_manifest=SimpleNamespace(model_identity="sha256:" + "a" * 64),
            compute_dtype=torch.float32,
        )
        registry = {"data": [{"artifactId": "b" * 64,
                              "modelDigest": "sha256:" + "a" * 64,
                              "layer": 1, "maxContextTokens": 64}]}
        with patch.dict(os.environ, {
                "MYCELLIOS_BROWSER_LAYER_BRIDGE_FILE": "",
                "MYCELLIOS_BROWSER_LAYER_COORDINATOR_URL": "http://127.0.0.1:8772"}):
            with patch("distributed_runtime.browser_layer_bridge.BrowserExpertOwner") as owner:
                owner.return_value._request_json.return_value = registry
                self.assertTrue(attach_browser_layer_bridge(runner))
        self.assertIsInstance(runner.base.layers[1], BrowserQwen3Layer)
        self.assertEqual(runner.base.layers[1].artifact_id, "b" * 64)
        self.assertEqual(runner.base.layers[1].local_index, 1)

    def test_registry_cannot_assign_another_model_digest(self):
        config = LlamaConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=1, num_attention_heads=4,
                             num_key_value_heads=2, max_position_embeddings=64,
                             _attn_implementation="eager")
        runner = SimpleNamespace(
            base=LlamaForCausalLM(config).model,
            spec=SimpleNamespace(layer_start=0, layer_end=1,
                                 quantize=None, compile_mode=None),
            executor_manifest=SimpleNamespace(model_identity="sha256:" + "a" * 64),
            compute_dtype=torch.float32,
        )
        with patch.dict(os.environ, {
                "MYCELLIOS_BROWSER_LAYER_BRIDGE_FILE": "",
                "MYCELLIOS_BROWSER_LAYER_COORDINATOR_URL": "http://127.0.0.1:8772"}):
            with patch("distributed_runtime.browser_layer_bridge.BrowserExpertOwner") as owner:
                owner.return_value._request_json.return_value = {"data": [{
                    "artifactId": "b" * 64, "modelDigest": "sha256:" + "c" * 64,
                    "layer": 0, "maxContextTokens": 64}]}
                with self.assertRaisesRegex(ValueError, "wrong model range"):
                    attach_browser_layer_bridge(runner)
