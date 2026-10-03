from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from distributed_runtime.browser_moe_bridge import (
    BrowserMoeBridgeConfig,
    attach_browser_moe_bridge,
)
from distributed_runtime.model import StageModelSpec, model_artifact_reference
from distributed_runtime.ram_backed_moe_stage import RamBackedMoeStageRunner
from distributed_runtime.ram_expert_cache import PredictiveCacheConfig
from distributed_runtime.resident_expert_mesh import OwnerExpertBatchResult
from tests.test_ram_backed_moe_mesh_integration import _bundles, _module
from tests.test_ram_backed_moe_stage import _pipeline_tokens, _reference_tokens


_IDENTITY = "sha256:" + "a" * 64


def _config(url: str = "http://127.0.0.1:8770") -> dict[str, object]:
    return {
        "schema": "mycellios-browser-moe-bridge/1",
        "artifactIdentity": _IDENTITY,
        "coordinatorUrl": url,
        "layer": 0,
        "expert": 1,
        "localComputeMsPerToken": 100.0,
        "browserComputeMsPerToken": 0.01,
        "ramToDeviceGbytesPerSecond": 0.001,
        "roundTripMs": 0.01,
        "bandwidthMbps": 10_000,
        "localVramBudgetBytes": 1_024,
        "browserVramBudgetBytes": 1_024,
    }


class _BrowserOwner:
    resident = True
    execution_calls = 0

    def __init__(self, node_id: str, coordinator_url: str) -> None:
        self.node_id = node_id
        self.coordinator_url = coordinator_url
        self.published = None

    def publish_swiglu_expert(self, **values) -> str:
        self.published = values
        return "b" * 64

    def has_expert(self, key, content_id) -> bool:
        return (self.published is not None
                and key == self.published["key"]
                and content_id == self.published["content_id"])

    def is_expert_resident(self, key, content_id) -> bool:
        return self.resident and self.has_expert(key, content_id)

    def execute_batch(self, items):
        type(self).execution_calls += len(items)
        values = self.published
        assert values is not None
        return tuple(
            OwnerExpertBatchResult(
                item.key,
                F.linear(
                    F.silu(F.linear(item.activations, values["gate_projection"]))
                    * F.linear(item.activations, values["up_projection"]),
                    values["down_projection"],
                ),
            ) for item in items
        )


class BrowserMoeBridgeTests(unittest.TestCase):
    def test_rejects_untrusted_url_and_unknown_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "bridge.json")
            path.write_text(json.dumps(_config("http://example.com")), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "requires HTTPS"):
                BrowserMoeBridgeConfig.from_file(path)
            bad = _config()
            bad["internalToken"] = "should-never-be-in-a-config-file"
            path.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fields do not match"):
                BrowserMoeBridgeConfig.from_file(path)

    def test_real_layer_routes_to_browser_and_falls_back_to_native(self) -> None:
        bundles, records = _bundles()
        module = _module(bundles, records)
        runner = object.__new__(RamBackedMoeStageRunner)
        runner.spec = StageModelSpec("tiny-model", 0, 1, 1, 1, artifact_identity=_IDENTITY)
        runner.sparse_layers = (0,)
        runner._expert_modules = (module,)
        runner.expert_scheduler = module.scheduler
        runner.expert_store = module.store
        runner._resident_mesh_owned_owners = {}
        runner._browser_expert_meshes = []
        hidden = torch.tensor([[0.25, -0.5, 1.0]], dtype=torch.float32)
        selected = torch.tensor([[1]], dtype=torch.long)
        weights = torch.ones((1, 1), dtype=torch.float32)
        expected = F.linear(
            F.silu(F.linear(hidden, bundles[records[1].key].tensor("gate_proj.weight")))
            * F.linear(hidden, bundles[records[1].key].tensor("up_proj.weight")),
            bundles[records[1].key].tensor("down_proj.weight"),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "bridge.json")
            path.write_text(json.dumps(_config()), encoding="utf-8")
            with patch.dict(os.environ, {"MYCELLIOS_BROWSER_MOE_BRIDGE_FILE": str(path)}), patch(
                "distributed_runtime.browser_moe_bridge.BrowserExpertOwner", _BrowserOwner
            ):
                self.assertTrue(attach_browser_moe_bridge(runner))
        try:
            output = module(hidden, selected, weights)
            torch.testing.assert_close(output, expected, rtol=0, atol=1e-6)
            self.assertEqual(_BrowserOwner.execution_calls, 1, module.last_resident_mesh_plan)
            self.assertEqual(
                {dispatch.path for dispatch in module.last_resident_mesh_plan.dispatches},
                {"remote-resident"},
            )
            _BrowserOwner.resident = False
            output = module(hidden, selected, weights)
            torch.testing.assert_close(output, expected, rtol=0, atol=1e-6)
            self.assertEqual(_BrowserOwner.execution_calls, 1)
            self.assertEqual(
                {dispatch.path for dispatch in module.last_resident_mesh_plan.dispatches},
                {"local-ram"},
            )
        finally:
            _BrowserOwner.resident = True
            _BrowserOwner.execution_calls = 0
            module.detach_resident_expert_mesh()
            for mesh in runner._browser_expert_meshes:
                mesh.close()

    def test_complete_tiny_moe_generation_uses_browser_expert(self) -> None:
        torch.manual_seed(871)
        config = Qwen3MoeConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            moe_intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            num_experts=1,
            num_experts_per_tok=1,
            max_position_embeddings=64,
            architectures=["Qwen3MoeForCausalLM"],
        )
        model = Qwen3MoeForCausalLM(config).eval()
        prompt = torch.tensor([[1, 5, 9, 3]], dtype=torch.long)
        expected = _reference_tokens(model, prompt, 3)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model.save_pretrained(root, safe_serialization=True)
            del model
            artifact = model_artifact_reference(str(root))
            expert_bytes = 16 * 32 * 3 * 4
            cache = PredictiveCacheConfig(
                capacity_bytes=expert_bytes * 2,
                prefetch_reserve_bytes=expert_bytes,
                pcie_bandwidth_gbytes_per_second=8,
            )
            first = RamBackedMoeStageRunner(
                StageModelSpec(str(root), 0, 1, 2, 1, artifact_identity=artifact.identity),
                cache,
                device="cpu",
            )
            last = RamBackedMoeStageRunner(
                StageModelSpec(str(root), 1, 2, 2, 1, artifact_identity=artifact.identity),
                cache,
                device="cpu",
            )
            bridge_config = _config()
            bridge_config["artifactIdentity"] = artifact.identity
            bridge_config["expert"] = 0
            bridge_config["localVramBudgetBytes"] = 100_000
            bridge_config["browserVramBudgetBytes"] = 100_000
            bridge_path = root / "browser-bridge.json"
            bridge_path.write_text(json.dumps(bridge_config), encoding="utf-8")
            try:
                with patch.dict(os.environ, {"MYCELLIOS_BROWSER_MOE_BRIDGE_FILE": str(bridge_path)}), patch(
                    "distributed_runtime.browser_moe_bridge.BrowserExpertOwner", _BrowserOwner
                ):
                    self.assertTrue(attach_browser_moe_bridge(first))
                actual = _pipeline_tokens(first, last, prompt, 3, request_id=871)
                self.assertEqual(actual, expected)
                self.assertGreater(_BrowserOwner.execution_calls, 0)
                self.assertEqual(
                    {dispatch.path for dispatch in first._expert_modules[0].last_resident_mesh_plan.dispatches},
                    {"remote-resident"},
                )
            finally:
                first.close()
                last.close()
                _BrowserOwner.execution_calls = 0


if __name__ == "__main__":
    unittest.main()
