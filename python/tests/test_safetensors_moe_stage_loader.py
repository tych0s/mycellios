from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from safetensors.torch import save_file
import torch
from torch import nn
from torch.nn import functional as F

from distributed_runtime.ram_expert_cache import ExpertKey, PredictiveCacheConfig
from distributed_runtime.safetensors_moe_stage_loader import (
    LOCAL_MOE_STAGE_SCHEMA,
    UnsupportedLocalMoeStageError,
    _CheckpointReader,
    execute_swiglu_expert_cpu,
    load_local_safetensors_moe_stage,
)


_EP_PLAN = {
    "layers.*.mlp.gate": "ep_router",
    "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
    "layers.*.mlp.experts.down_proj": "grouped_gemm",
    "layers.*.mlp.experts": "moe_tp_experts",
}


class _CanonicalExpert(nn.Module):
    def __init__(self, hidden: int, intermediate: int, offset: float) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)
        with torch.no_grad():
            cursor = offset
            for parameter in self.parameters():
                values = torch.arange(parameter.numel(), dtype=torch.float32)
                parameter.copy_((values.reshape(parameter.shape) + cursor) / 37.0)
                cursor += parameter.numel()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


def _base_config(*, family: str, layers: int) -> dict[str, object]:
    config: dict[str, object] = {
        "num_hidden_layers": layers,
        "hidden_size": 4,
        "intermediate_size": 6,
        "moe_intermediate_size": 3,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 2,
        "num_experts_per_tok": 1,
        "num_local_experts": 2,
        "layer_types": ["full_attention"] * layers,
        "use_sliding_window": False,
        "base_model_ep_plan": dict(_EP_PLAN),
    }
    if family == "qwen":
        config.update(
            {
                "model_type": "qwen3_moe",
                "architectures": ["Qwen3MoeForCausalLM"],
                "num_experts": 2,
                "decoder_sparse_step": 1,
                "mlp_only_layers": [],
            }
        )
    elif family == "glm":
        config.update(
            {
                "model_type": "glm4_moe",
                "architectures": ["Glm4MoeForCausalLM"],
                "n_routed_experts": 2,
                "n_shared_experts": 1,
                "first_k_dense_replace": 1,
            }
        )
    else:
        raise AssertionError(f"unknown test family {family}")
    return config


def _qwen_fixture(
    root: Path,
    *,
    missing_up: bool = False,
    extra_bias: bool = False,
) -> tuple[_CanonicalExpert, _CanonicalExpert]:
    config = _base_config(family="qwen", layers=1)
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    experts = (_CanonicalExpert(4, 3, 1), _CanonicalExpert(4, 3, 101))
    gate_up = torch.stack(
        [
            torch.cat((expert.gate_proj.weight, expert.up_proj.weight), dim=0)
            for expert in experts
        ]
    )
    tensors = {
        "model.layers.0.mlp.experts.gate_up_proj": gate_up,
        "model.layers.0.mlp.experts.down_proj": torch.stack(
            [expert.down_proj.weight for expert in experts]
        ),
        "model.layers.0.mlp.gate.weight": torch.arange(8, dtype=torch.float32).reshape(2, 4),
    }
    if missing_up:
        tensors["model.layers.0.mlp.experts.gate_up_proj"] = gate_up[:, :3, :].contiguous()
    if extra_bias:
        tensors["model.layers.0.mlp.gate.bias"] = torch.zeros(2)
    save_file(tensors, root / "model.safetensors")
    return experts


def _glm_fixture(root: Path) -> None:
    config = _base_config(family="glm", layers=2)
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    dense = {
        "model.layers.0.mlp.gate_proj.weight": torch.zeros(6, 4),
        "model.layers.0.mlp.up_proj.weight": torch.zeros(6, 4),
        "model.layers.0.mlp.down_proj.weight": torch.zeros(4, 6),
    }
    sparse = {
        "model.layers.1.mlp.gate.weight": torch.zeros(2, 4),
        "model.layers.1.mlp.gate.e_score_correction_bias": torch.zeros(2),
        "model.layers.1.mlp.shared_experts.gate_proj.weight": torch.zeros(3, 4),
        "model.layers.1.mlp.shared_experts.up_proj.weight": torch.ones(3, 4),
        "model.layers.1.mlp.shared_experts.down_proj.weight": torch.zeros(4, 3),
    }
    for expert in range(2):
        prefix = f"model.layers.1.mlp.experts.{expert}"
        sparse[f"{prefix}.gate_proj.weight"] = torch.full((3, 4), expert + 1.0)
        sparse[f"{prefix}.up_proj.weight"] = torch.full((3, 4), expert + 2.0)
        sparse[f"{prefix}.down_proj.weight"] = torch.full((4, 3), expert + 3.0)
    first = "model-00001-of-00002.safetensors"
    second = "model-00002-of-00002.safetensors"
    save_file(dense, root / first)
    save_file(sparse, root / second)
    weight_map = {name: first for name in dense}
    weight_map.update({name: second for name in sparse})
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}),
        encoding="utf-8",
    )


class LocalSafetensorsMoeStageLoaderTests(unittest.TestCase):
    def test_runner_mode_retains_only_routed_experts_and_exposes_header_peak(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _qwen_fixture(root)
            loaded_names: list[str] = []
            original_load = _CheckpointReader.load

            def recording_load(reader: _CheckpointReader, name: str) -> torch.Tensor:
                loaded_names.append(name)
                return original_load(reader, name)

            with patch.object(
                _CheckpointReader,
                "load",
                autospec=True,
                side_effect=recording_load,
            ):
                loaded = load_local_safetensors_moe_stage(
                    root,
                    layer_start=0,
                    layer_end=1,
                    preload_resident_artifacts=False,
                )

        self.assertFalse(loaded.preloaded_resident_artifacts)
        self.assertEqual(loaded.routers, {})
        self.assertEqual(loaded.shared_experts, {})
        self.assertEqual(
            set(loaded_names),
            {
                "model.layers.0.mlp.experts.gate_up_proj",
                "model.layers.0.mlp.experts.down_proj",
            },
        )
        self.assertEqual(loaded.loaded_backing_bytes, 288)
        self.assertEqual(loaded.resident_streaming_transient_bytes, 32)
        self.assertEqual(loaded.bounded_pinned_staging_reserve_bytes, 0)
        self.assertEqual(loaded.host_ram_steady_state_bytes, 288)
        self.assertEqual(loaded.host_ram_peak_upper_bound_bytes, 320)

    def test_host_peak_budget_fails_from_headers_before_loading_any_tensor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _qwen_fixture(root)
            with patch(
                "distributed_runtime.safetensors_moe_stage_loader._CheckpointReader.load",
                side_effect=AssertionError("tensor body must remain unread"),
            ) as load:
                with self.assertRaisesRegex(MemoryError, "host RAM peak budget"):
                    load_local_safetensors_moe_stage(
                        root,
                        layer_start=0,
                        layer_end=1,
                        preload_resident_artifacts=False,
                        reserve_bounded_pinned_staging=True,
                        expected_resident_streaming_transient_bytes=32,
                        expected_bounded_pinned_staging_reserve_bytes=288,
                        # routed 288 + two 144-byte slots + transient 32 = 608
                        expected_host_ram_peak_bytes=607,
                    )
            load.assert_not_called()

    def test_routed_byte_contract_fails_from_headers_before_loading_any_tensor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _qwen_fixture(root)
            with patch(
                "distributed_runtime.safetensors_moe_stage_loader._CheckpointReader.load",
                side_effect=AssertionError("tensor body must remain unread"),
            ) as load:
                with self.assertRaisesRegex(
                    UnsupportedLocalMoeStageError,
                    "routed expert byte budget mismatch",
                ):
                    load_local_safetensors_moe_stage(
                        root,
                        layer_start=0,
                        layer_end=1,
                        expected_total_routed_expert_bytes=289,
                        expected_largest_expert_bytes=144,
                    )
            load.assert_not_called()

    def test_qwen_packed_stage_is_adopted_without_second_ram_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _qwen_fixture(root)
            loaded = load_local_safetensors_moe_stage(
                root,
                layer_start=0,
                layer_end=1,
            )

        self.assertEqual(loaded.schema, LOCAL_MOE_STAGE_SCHEMA)
        self.assertEqual(loaded.adapter_id, "transformers-qwen3-moe-v1")
        self.assertEqual(loaded.storage_layout, "grouped-3d-swiglu")
        self.assertEqual(loaded.sparse_layers, (0,))
        self.assertEqual(len(loaded.inventory.records), 2)
        self.assertEqual(loaded.routed_expert_bytes, 288)
        self.assertEqual(loaded.resident_router_bytes, 32)
        self.assertEqual(loaded.loaded_backing_bytes, 320)
        self.assertTrue(loaded.preloaded_resident_artifacts)
        self.assertEqual(loaded.resident_streaming_transient_bytes, 0)
        self.assertEqual(loaded.host_ram_steady_state_bytes, 320)
        self.assertEqual(loaded.host_ram_peak_upper_bound_bytes, 320)
        key = ExpertKey(0, 1)
        source = loaded.expert_bundles[key].tensor("up_proj.weight")

        config = PredictiveCacheConfig(
            capacity_bytes=576,
            prefetch_reserve_bytes=144,
            pcie_bandwidth_gbytes_per_second=8,
        )
        store = loaded.create_store(
            config,
            device="cpu",
            bounded_pinned_staging=True,
        )
        adopted = store.ram_bundle(key).tensor("up_proj.weight")

        self.assertIs(adopted, source)
        self.assertEqual(adopted.data_ptr(), source.data_ptr())
        self.assertTrue(store.snapshot().ram_adopted_without_clone)
        self.assertEqual(store.snapshot().ram_bytes, loaded.routed_expert_bytes)
        self.assertTrue(store.snapshot().bounded_pinned_staging_requested)
        self.assertFalse(store.snapshot().bounded_pinned_staging_enabled)

    def test_extracted_qwen_bundle_executes_with_numeric_module_parity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experts = _qwen_fixture(root)
            loaded = load_local_safetensors_moe_stage(
                root,
                layer_start=0,
                layer_end=1,
            )

        hidden = torch.tensor(
            [[[-0.5, 0.25, 1.0, 0.75], [0.1, -0.2, 0.3, -0.4]]],
            dtype=torch.float32,
        )
        with torch.inference_mode():
            expected = experts[1](hidden)
            actual = execute_swiglu_expert_cpu(
                loaded.expert_bundles[ExpertKey(0, 1)],
                hidden,
            )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_glm_unpacked_stage_keeps_dense_router_and_shared_weights_separate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _glm_fixture(root)
            loaded = load_local_safetensors_moe_stage(
                root,
                layer_start=0,
                layer_end=2,
            )

        self.assertEqual(loaded.adapter_id, "transformers-glm4-moe-v1")
        self.assertEqual(loaded.storage_layout, "unpacked-per-expert-swiglu")
        self.assertEqual(loaded.dense_layers, (0,))
        self.assertEqual(loaded.sparse_layers, (1,))
        self.assertEqual(set(loaded.routers), {1})
        self.assertEqual(
            tuple(name for name, _ in loaded.routers[1].tensors),
            ("weight", "e_score_correction_bias"),
        )
        self.assertEqual(set(loaded.shared_experts), {1})
        self.assertEqual(loaded.shared_experts[1].byte_size, 144)
        self.assertEqual(loaded.routed_expert_bytes, 288)
        self.assertTrue(
            all(
                "shared" not in name
                for bundle in loaded.expert_bundles.values()
                for name, _ in bundle.tensors
            )
        )

    def test_unknown_geometry_extra_key_identity_and_stale_index_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _qwen_fixture(root, missing_up=True)
            with self.assertRaisesRegex(UnsupportedLocalMoeStageError, "shape"):
                load_local_safetensors_moe_stage(root, layer_start=0, layer_end=1)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _qwen_fixture(root, extra_bias=True)
            with self.assertRaisesRegex(UnsupportedLocalMoeStageError, "not certified"):
                load_local_safetensors_moe_stage(root, layer_start=0, layer_end=1)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _qwen_fixture(root)
            with self.assertRaisesRegex(UnsupportedLocalMoeStageError, "identity mismatch"):
                load_local_safetensors_moe_stage(
                    root,
                    layer_start=0,
                    layer_end=1,
                    expected_artifact_identity="sha256:" + "0" * 64,
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _glm_fixture(root)
            index_path = root / "model.safetensors.index.json"
            index = json.loads(index_path.read_text(encoding="utf-8"))
            index["weight_map"].pop("model.layers.1.mlp.gate.weight")
            index_path.write_text(json.dumps(index), encoding="utf-8")
            with self.assertRaisesRegex(UnsupportedLocalMoeStageError, "index"):
                load_local_safetensors_moe_stage(root, layer_start=1, layer_end=2)

    def test_reference_executor_rejects_dtype_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _qwen_fixture(root)
            loaded = load_local_safetensors_moe_stage(
                root,
                layer_start=0,
                layer_end=1,
            )
        with self.assertRaisesRegex(ValueError, "same dtype"):
            execute_swiglu_expert_cpu(
                loaded.expert_bundles[ExpertKey(0, 0)],
                torch.ones(1, 4, dtype=torch.float64),
            )


if __name__ == "__main__":
    unittest.main()
