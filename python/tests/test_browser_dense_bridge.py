from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM

from distributed_runtime.browser_dense_bridge import BrowserDenseMlp
from distributed_runtime.model import StageModelSpec, StageRunner, model_artifact_reference
from distributed_runtime.ram_expert_cache import ExpertKey
from distributed_runtime.resident_expert_mesh import OwnerExpertBatchResult


class _Owner:
    resident = True

    def __init__(self, node_id: str, coordinator_url: str) -> None:
        self.node_id = node_id
        self.coordinator_url = coordinator_url

    def publish_swiglu_expert(self, **values) -> str:
        self.values = values
        return "a" * 64

    def is_expert_resident(self, key, content_id) -> bool:
        return self.resident and key == self.values["key"] and content_id == self.values["content_id"]

    def execute_batch(self, items):
        values = self.values
        return tuple(OwnerExpertBatchResult(
            item.key,
            F.linear(
                F.silu(F.linear(item.activations, values["gate_projection"]))
                * F.linear(item.activations, values["up_projection"]),
                values["down_projection"],
            ),
        ) for item in items)


class BrowserDenseBridgeTests(unittest.TestCase):
    def test_browser_result_returns_to_stage_device_and_precision(self) -> None:
        class Float32Owner:
            def is_expert_resident(self, key, content_id) -> bool:
                return True

            def execute_batch(self, items):
                return tuple(OwnerExpertBatchResult(
                    item.key, item.activations.float().cpu().clone(),
                ) for item in items)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hidden = torch.tensor([[[0.25, -0.5]]], dtype=torch.float16, device=device)
        wrapper = BrowserDenseMlp(torch.nn.Identity(), Float32Owner(), ExpertKey(0, 0), "exact")
        actual = wrapper(hidden)
        self.assertEqual(actual.device, hidden.device)
        self.assertEqual(actual.dtype, hidden.dtype)
        torch.testing.assert_close(actual, hidden, rtol=0, atol=0)
        self.assertEqual(wrapper.browser_forwards, 1)

    def test_stage_runner_routes_and_falls_back_without_changing_mlp_output(self) -> None:
        torch.manual_seed(491)
        config = Qwen3Config(
            vocab_size=32, hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
            head_dim=8, max_position_embeddings=64,
            architectures=["Qwen3ForCausalLM"],
        )
        model = Qwen3ForCausalLM(config).eval()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model.save_pretrained(root, safe_serialization=True)
            artifact = model_artifact_reference(str(root))
            bridge_path = root / "bridge.json"
            bridge_path.write_text(json.dumps({
                "schema": "mycellios-browser-dense-bridge/1",
                "artifactIdentity": artifact.identity,
                "coordinatorUrl": "http://127.0.0.1:8770",
                "layer": 0,
            }), encoding="utf-8")
            with patch.dict(os.environ, {"MYCELLIOS_BROWSER_DENSE_BRIDGE_FILE": str(bridge_path)}), patch(
                "distributed_runtime.browser_dense_bridge.BrowserExpertOwner", _Owner
            ):
                runner = StageRunner(StageModelSpec(
                    str(root), 0, 1, 1, 1, artifact_identity=artifact.identity,
                ), device="cpu")
            try:
                wrapper = runner.base.layers[0].mlp
                self.assertIsInstance(wrapper, BrowserDenseMlp)
                hidden = torch.randn((2, 3, 32)) * 0.1
                expected = wrapper.native(hidden)
                actual = wrapper(hidden)
                torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
                self.assertEqual((wrapper.browser_forwards, wrapper.native_forwards), (1, 0))
                wrapper.owner.resident = False
                actual = wrapper(hidden)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual((wrapper.browser_forwards, wrapper.native_forwards), (1, 1))
            finally:
                runner.close()


if __name__ == "__main__":
    unittest.main()
