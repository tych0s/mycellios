from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from safetensors.torch import save_file
import torch

from distributed_runtime.profile import (
    ModelProfileOptions,
    PROFILE_SCHEMA,
    compile_model_profile,
    main,
    read_checkpoint_metadata,
)


_EP_PLAN = {
    "layers.*.mlp.gate": "ep_router",
    "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
    "layers.*.mlp.experts.down_proj": "grouped_gemm",
    "layers.*.mlp.experts": "moe_tp_experts",
}


class ModelProfileTests(unittest.TestCase):
    def test_compiles_exact_layer_and_endpoint_bytes_without_loading_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_config(root, tied=False)
            tensors = {
                "model.embed_tokens.weight": torch.zeros((16, 8), dtype=torch.float16),
                "model.layers.0.self_attn.q_proj.weight": torch.zeros((8, 8), dtype=torch.float16),
                "model.layers.0.mlp.down_proj.weight": torch.zeros((8, 16), dtype=torch.float16),
                "model.layers.1.self_attn.q_proj.weight": torch.zeros((8, 8), dtype=torch.float16),
                "model.layers.1.mlp.down_proj.weight": torch.zeros((8, 16), dtype=torch.float16),
                "model.norm.weight": torch.zeros(8, dtype=torch.float16),
                "lm_head.weight": torch.zeros((16, 8), dtype=torch.float16),
            }
            save_file(tensors, root / "model.safetensors")

            result = compile_model_profile(
                str(root),
                options=ModelProfileOptions(runtime_overhead_bytes_per_stage=1234),
            )

            self.assertEqual(result["schema"], PROFILE_SCHEMA)
            self.assertNotIn("snapshot", result["source"])
            self.assertRegex(
                result["source"]["snapshotIdentityUint64Hex"], r"^[0-9a-f]{16}$"
            )
            self.assertEqual(result["inspection"]["layerPrefix"], "model.layers")
            self.assertEqual(result["model"]["layers"][0]["weightBytes"], 384)
            self.assertEqual(result["model"]["layers"][1]["weightBytes"], 384)
            self.assertEqual(
                result["model"]["layers"][0]["largestResidentTensorBytes"],
                256,
            )
            self.assertEqual(result["model"]["layers"][0]["activationElements"], 8)
            # 2 (K+V) * 1 KV head * 4 head dim * FP16.
            self.assertEqual(result["model"]["layers"][0]["kvBytesPerToken"], 16)
            self.assertEqual(result["model"]["embeddingBytes"], 256)
            self.assertEqual(result["model"]["lmHeadBytes"], 272)
            self.assertEqual(result["model"]["largestEmbeddingTensorBytes"], 256)
            self.assertEqual(result["model"]["largestLmHeadTensorBytes"], 256)
            self.assertEqual(result["model"]["runtimeOverheadBytesPerStage"], 1234)
            self.assertNotIn("expertParallel", result["model"]["layers"][0])
            self.assertNotIn("macroWave", result["model"]["layers"][0])
            self.assertNotIn("certifiedMoe", result["inspection"])
            self.assertTrue(result["compatibility"]["selectiveSafetensors"])
            self.assertTrue(result["inspection"]["calibrationRequired"])
            self.assertEqual(
                result["accounting"]["checkpointStorageBytes"],
                sum(tensor.numel() * tensor.element_size() for tensor in tensors.values()),
            )

    def test_budgets_a_tied_head_on_the_last_split_stage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_config(root, tied=True)
            save_file(
                {
                    "model.embed_tokens.weight": torch.zeros((16, 8), dtype=torch.float16),
                    "model.layers.0.weight": torch.zeros((8, 8), dtype=torch.float16),
                    "model.layers.1.weight": torch.zeros((8, 8), dtype=torch.float16),
                    "model.norm.weight": torch.zeros(8, dtype=torch.float16),
                },
                root / "model.safetensors",
            )

            result = compile_model_profile(str(root))

            self.assertEqual(result["model"]["embeddingBytes"], 256)
            self.assertEqual(result["model"]["lmHeadBytes"], 272)
            self.assertEqual(result["model"]["largestEmbeddingTensorBytes"], 256)
            self.assertEqual(result["model"]["largestLmHeadTensorBytes"], 256)
            self.assertEqual(result["inspection"]["duplicatedTiedHeadBytes"], 256)
            self.assertTrue(result["compatibility"]["selectiveSafetensors"])
            self.assertGreater(
                result["accounting"]["plannedResidentWeightBytesAcrossSplitEndpoints"],
                result["accounting"]["checkpointStorageBytes"],
            )

    def test_detects_an_adaptable_non_model_layers_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_config(root, tied=False)
            save_file(
                {
                    "transformer.wte.weight": torch.zeros((16, 8), dtype=torch.float16),
                    "transformer.h.0.weight": torch.zeros((8, 8), dtype=torch.float16),
                    "transformer.h.1.weight": torch.zeros((8, 8), dtype=torch.float16),
                    "lm_head.weight": torch.zeros((16, 8), dtype=torch.float16),
                },
                root / "model.safetensors",
            )

            result = compile_model_profile(str(root))

            self.assertEqual(result["inspection"]["layerPrefix"], "transformer.h")
            self.assertFalse(result["compatibility"]["selectiveSafetensors"])
            self.assertTrue(result["compatibility"]["requiresAdapter"])
            self.assertIn("transformer.h", result["compatibility"]["reasons"][0])

    def test_rejects_an_incomplete_layer_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_config(root, tied=True)
            save_file(
                {
                    "model.embed_tokens.weight": torch.zeros((16, 8), dtype=torch.float16),
                    "model.layers.0.weight": torch.zeros((8, 8), dtype=torch.float16),
                },
                root / "model.safetensors",
            )
            with self.assertRaisesRegex(ValueError, "could not infer a complete"):
                compile_model_profile(str(root))

    def test_reads_a_sharded_index_and_exact_offsets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_file({"model.layers.0.weight": torch.zeros(3)}, root / "one.safetensors")
            save_file({"model.layers.1.weight": torch.zeros(5)}, root / "two.safetensors")
            (root / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "model.layers.0.weight": "one.safetensors",
                            "model.layers.1.weight": "two.safetensors",
                        }
                    }
                ),
                encoding="utf-8",
            )
            metadata = read_checkpoint_metadata(root)
            self.assertEqual([tensor.bytes for tensor in metadata], [12, 20])

    def test_profiles_certified_packed_qwen_moe_from_headers_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_qwen_moe(root, layers=2, experts_per_token=2)

            with patch(
                "distributed_runtime.safetensors_moe_stage_loader._CheckpointReader.load",
                side_effect=AssertionError("profile must not load tensor bodies"),
            ) as load:
                result = compile_model_profile(str(root))

            load.assert_not_called()
            for layer in result["model"]["layers"]:
                # The packed expert tensors are 96 and 48 bytes; neither is a
                # resident loader transient. Attention is the largest resident
                # tensor in this certified sparse layer.
                self.assertEqual(layer["largestResidentTensorBytes"], 32)
                self.assertEqual(
                    layer["expertParallel"],
                    {
                        "expertWeightBytes": 144,
                        "expertCount": 2,
                        "expertsPerToken": 2,
                    },
                )
                self.assertEqual(
                    layer["macroWave"],
                    {
                        "activeWeightBytesPerWave": 144,
                        "largestTransferUnitBytes": 72,
                        "expertWorkspaceBytesPerPosition": 256,
                    },
                )
            self.assertEqual(
                result["inspection"]["certifiedMoe"],
                {
                    "adapterId": "transformers-qwen3-moe-v1",
                    "storageLayout": "grouped-3d-swiglu",
                    "sourceDtype": "F16",
                    "expertCount": 2,
                    "expertsPerToken": 2,
                    "expertWorkspace": {
                        "bytesPerPosition": 256,
                        "executionMode": "serial-exact-swiglu",
                        "includesCublasWorkspace": False,
                    },
                },
            )
            self.assertEqual(result["accounting"]["moeRoutedExpertWeightBytes"], 288)
            self.assertEqual(
                result["accounting"]["moeNonRoutedLayerWeightBytes"],
                sum(layer["weightBytes"] for layer in result["model"]["layers"]) - 288,
            )

    def test_profiles_unpacked_glm_and_marks_its_dense_prefix_as_zero_routed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_glm_moe(root)

            result = compile_model_profile(str(root))
            dense, sparse = result["model"]["layers"]

            self.assertEqual(dense["expertParallel"], {"expertWeightBytes": 0})
            self.assertEqual(dense["largestResidentTensorBytes"], 48)
            self.assertNotIn("macroWave", dense)
            self.assertEqual(
                sparse["expertParallel"],
                {
                    "expertWeightBytes": 144,
                    "expertCount": 2,
                    "expertsPerToken": 1,
                },
            )
            self.assertEqual(sparse["macroWave"]["activeWeightBytesPerWave"], 72)
            self.assertEqual(sparse["macroWave"]["largestTransferUnitBytes"], 72)
            self.assertEqual(
                sparse["macroWave"]["expertWorkspaceBytesPerPosition"],
                256,
            )
            # The 48-byte shared-expert matrices are resident and therefore
            # dominate 32-byte attention, while routed experts stay excluded.
            self.assertEqual(sparse["largestResidentTensorBytes"], 48)
            self.assertEqual(
                result["inspection"]["certifiedMoe"]["storageLayout"],
                "unpacked-per-expert-swiglu",
            )

    def test_moe_profile_fails_closed_on_shape_layout_and_dtype_ambiguity(self) -> None:
        mutations = ("shape", "extra", "dtype")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self._write_qwen_moe(root, mutation=mutation)
                with self.assertRaisesRegex(
                    ValueError,
                    "shape|not certified|mixed dtypes",
                ):
                    compile_model_profile(str(root))

    def test_cli_json_and_accounting_preserve_exact_moe_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "profile.json"
            self._write_qwen_moe(root)
            stdout = io.StringIO()
            with patch.object(
                sys,
                "argv",
                ["profile", str(root), "--json-out", str(output)],
            ), redirect_stdout(stdout):
                main()

            printed = json.loads(stdout.getvalue())
            written = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(printed, written)
            layer = written["model"]["layers"][0]
            self.assertEqual(layer["expertParallel"]["expertCount"], 2)
            self.assertEqual(layer["expertParallel"]["expertsPerToken"], 1)
            self.assertEqual(
                written["accounting"]["moeRoutedExpertWeightBytes"],
                layer["expertParallel"]["expertWeightBytes"],
            )

    @staticmethod
    def _write_config(root: Path, *, tied: bool) -> None:
        (root / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "llama",
                    "architectures": ["LlamaForCausalLM"],
                    "vocab_size": 16,
                    "hidden_size": 8,
                    "intermediate_size": 16,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "max_position_embeddings": 128,
                    "tie_word_embeddings": tied,
                    "torch_dtype": "float16",
                }
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _moe_config(*, family: str, layers: int, experts_per_token: int) -> dict[str, object]:
        config: dict[str, object] = {
            "vocab_size": 16,
            "num_hidden_layers": layers,
            "hidden_size": 4,
            "intermediate_size": 6,
            "moe_intermediate_size": 3,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 2,
            "num_experts_per_tok": experts_per_token,
            "num_local_experts": 2,
            "layer_types": ["full_attention"] * layers,
            "use_sliding_window": False,
            "base_model_ep_plan": dict(_EP_PLAN),
            "tie_word_embeddings": False,
            "torch_dtype": "float16",
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
            raise AssertionError(f"unknown MoE fixture family {family}")
        return config

    @classmethod
    def _write_qwen_moe(
        cls,
        root: Path,
        *,
        layers: int = 1,
        experts_per_token: int = 1,
        mutation: str | None = None,
    ) -> None:
        (root / "config.json").write_text(
            json.dumps(
                cls._moe_config(
                    family="qwen",
                    layers=layers,
                    experts_per_token=experts_per_token,
                )
            ),
            encoding="utf-8",
        )
        tensors: dict[str, torch.Tensor] = {
            "model.embed_tokens.weight": torch.zeros((16, 4), dtype=torch.float16),
            "model.norm.weight": torch.zeros(4, dtype=torch.float16),
            "lm_head.weight": torch.zeros((16, 4), dtype=torch.float16),
        }
        for layer in range(layers):
            prefix = f"model.layers.{layer}"
            tensors[f"{prefix}.self_attn.q_proj.weight"] = torch.zeros(
                (4, 4), dtype=torch.float16
            )
            tensors[f"{prefix}.mlp.gate.weight"] = torch.zeros(
                (2, 4), dtype=torch.float16
            )
            tensors[f"{prefix}.mlp.experts.gate_up_proj"] = torch.zeros(
                (2, 6, 4), dtype=torch.float16
            )
            tensors[f"{prefix}.mlp.experts.down_proj"] = torch.zeros(
                (2, 4, 3),
                dtype=(torch.float32 if mutation == "dtype" and layer == 0 else torch.float16),
            )
        if mutation == "shape":
            tensors["model.layers.0.mlp.experts.gate_up_proj"] = torch.zeros(
                (2, 5, 4), dtype=torch.float16
            )
        if mutation == "extra":
            tensors["model.layers.0.mlp.gate.bias"] = torch.zeros(2, dtype=torch.float16)
        save_file(tensors, root / "model.safetensors")

    @classmethod
    def _write_glm_moe(cls, root: Path) -> None:
        config = cls._moe_config(family="glm", layers=2, experts_per_token=1)
        config["n_shared_experts"] = 2
        (root / "config.json").write_text(
            json.dumps(config),
            encoding="utf-8",
        )
        tensors: dict[str, torch.Tensor] = {
            "model.embed_tokens.weight": torch.zeros((16, 4), dtype=torch.float16),
            "model.norm.weight": torch.zeros(4, dtype=torch.float16),
            "lm_head.weight": torch.zeros((16, 4), dtype=torch.float16),
            "model.layers.0.self_attn.q_proj.weight": torch.zeros((4, 4), dtype=torch.float16),
            "model.layers.0.mlp.gate_proj.weight": torch.zeros((6, 4), dtype=torch.float16),
            "model.layers.0.mlp.up_proj.weight": torch.zeros((6, 4), dtype=torch.float16),
            "model.layers.0.mlp.down_proj.weight": torch.zeros((4, 6), dtype=torch.float16),
            "model.layers.1.self_attn.q_proj.weight": torch.zeros((4, 4), dtype=torch.float16),
            "model.layers.1.mlp.gate.weight": torch.zeros((2, 4), dtype=torch.float16),
            "model.layers.1.mlp.gate.e_score_correction_bias": torch.zeros(
                2, dtype=torch.float16
            ),
            "model.layers.1.mlp.shared_experts.gate_proj.weight": torch.zeros(
                (6, 4), dtype=torch.float16
            ),
            "model.layers.1.mlp.shared_experts.up_proj.weight": torch.zeros(
                (6, 4), dtype=torch.float16
            ),
            "model.layers.1.mlp.shared_experts.down_proj.weight": torch.zeros(
                (4, 6), dtype=torch.float16
            ),
        }
        for expert in range(2):
            prefix = f"model.layers.1.mlp.experts.{expert}"
            tensors[f"{prefix}.gate_proj.weight"] = torch.zeros(
                (3, 4), dtype=torch.float16
            )
            tensors[f"{prefix}.up_proj.weight"] = torch.zeros(
                (3, 4), dtype=torch.float16
            )
            tensors[f"{prefix}.down_proj.weight"] = torch.zeros(
                (4, 3), dtype=torch.float16
            )
        save_file(tensors, root / "model.safetensors")


if __name__ == "__main__":
    unittest.main()
