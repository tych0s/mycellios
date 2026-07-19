from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from safetensors.torch import save_file
import torch

from distributed_runtime.profile import (
    ModelProfileOptions,
    PROFILE_SCHEMA,
    compile_model_profile,
    read_checkpoint_metadata,
)


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
            self.assertEqual(result["model"]["layers"][0]["activationElements"], 8)
            # 2 (K+V) * 1 KV head * 4 head dim * FP16.
            self.assertEqual(result["model"]["layers"][0]["kvBytesPerToken"], 16)
            self.assertEqual(result["model"]["embeddingBytes"], 256)
            self.assertEqual(result["model"]["lmHeadBytes"], 272)
            self.assertEqual(result["model"]["runtimeOverheadBytesPerStage"], 1234)
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


if __name__ == "__main__":
    unittest.main()
