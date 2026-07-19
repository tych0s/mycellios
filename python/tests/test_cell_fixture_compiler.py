from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch

import distributed_runtime.cell_fixture_compiler as compiler
from distributed_runtime.cell_fixture_compiler import (
    CELL_FIXTURE_SCHEMA,
    CELL_FIXTURE_SOURCE_SCHEMA,
    compile_hf_llama_cell_fixture,
)
from distributed_runtime.cell_stage import write_llama_stage_cell_fixture
from distributed_runtime.model import _checkpoint_key_map, resolve_model_snapshot


class HuggingFaceCellFixtureCompilerTests(unittest.TestCase):
    def test_compiles_sealed_unequal_rank_slices_from_capacity_weights(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "checkpoint"
            output = root / "compiled-weighted"
            self._write_checkpoint(
                snapshot,
                source_dtype=torch.float32,
                num_key_value_heads=4,
            )

            result = compile_hf_llama_cell_fixture(
                str(snapshot),
                output,
                layer_start=1,
                layer_end=3,
                world_size=2,
                rank_weights=(3.0, 1.0),
            )

            manifest = json.loads((output / "cell.json").read_text(encoding="utf-8"))
            self.assertEqual(result.rank_weights, (3.0, 1.0))
            self.assertEqual(manifest["rankWeights"], [3.0, 1.0])
            rank_zero = load_file(str(output / "rank-000.safetensors"))
            rank_one = load_file(str(output / "rank-001.safetensors"))
            self.assertEqual(rank_zero["layers.0.query"].shape, (6, 8))
            self.assertEqual(rank_one["layers.0.query"].shape, (2, 8))
            self.assertEqual(rank_zero["layers.0.gate"].shape, (7, 8))
            self.assertEqual(rank_one["layers.0.gate"].shape, (3, 8))
            self.assertGreater(result.rank_fixed_bytes[0], result.rank_fixed_bytes[1])
            provenance = json.loads((output / "source.json").read_text(encoding="utf-8"))
            self.assertEqual(provenance["rankWeights"], [3.0, 1.0])

    def test_streams_bfloat16_bits_exactly_and_reports_two_byte_weights(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "checkpoint"
            output = root / "compiled-bf16"
            dense = self._write_checkpoint(snapshot, source_dtype=torch.float32)

            result = compile_hf_llama_cell_fixture(
                str(snapshot),
                output,
                layer_start=1,
                layer_end=3,
                world_size=2,
                output_dtype="bfloat16",
            )

            manifest = json.loads((output / "cell.json").read_text(encoding="utf-8"))
            self.assertEqual(result.output_dtype, "bfloat16")
            self.assertEqual(manifest["dtype"], "bfloat16")
            self.assertEqual(result.output_tensor_bytes, sum(result.rank_fixed_bytes))
            rank_zero = load_file(str(output / "rank-000.safetensors"))
            self.assertTrue(all(value.dtype == torch.bfloat16 for value in rank_zero.values()))

            expected_query = dense[1]["query"][:4, :].to(torch.bfloat16).contiguous()
            actual_query = rank_zero["layers.0.query"].contiguous()
            self.assertTrue(
                torch.equal(actual_query.view(torch.uint16), expected_query.view(torch.uint16))
            )
            self.assertEqual(
                result.output_tensor_bytes,
                sum(value.numel() * 2 for rank in range(2) for value in load_file(
                    str(output / f"rank-{rank:03d}.safetensors")
                ).values()),
            )

    def test_preserves_explicit_float16_fixture_and_memory_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "checkpoint"
            output = root / "compiled-f16"
            self._write_checkpoint(snapshot, source_dtype=torch.bfloat16)

            result = compile_hf_llama_cell_fixture(
                str(snapshot),
                output,
                layer_start=1,
                layer_end=3,
                world_size=2,
                output_dtype="float16",
            )

            manifest = json.loads((output / "cell.json").read_text(encoding="utf-8"))
            self.assertEqual(result.output_dtype, "float16")
            self.assertEqual(manifest["dtype"], "float16")
            self.assertEqual(list(result.rank_fixed_bytes), manifest["rankFixedBytes"])
            self.assertEqual(
                list(result.rank_kv_bytes_per_token),
                manifest["rankKvBytesPerToken"],
            )
            self.assertEqual(result.output_tensor_bytes, sum(result.rank_fixed_bytes))
            for rank in range(2):
                tensors = load_file(str(output / f"rank-{rank:03d}.safetensors"))
                self.assertTrue(tensors)
                self.assertTrue(all(value.dtype == torch.float16 for value in tensors.values()))
            provenance = json.loads((output / "source.json").read_text(encoding="utf-8"))
            self.assertEqual(provenance["outputDtype"], "float16")
            self.assertIn("output-dtype-conversion", provenance["memoryPolicy"])

    def test_rejects_unknown_output_dtype_before_publishing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "checkpoint"
            output = root / "compiled"
            self._write_checkpoint(snapshot, source_dtype=torch.float32)
            with self.assertRaisesRegex(ValueError, "output_dtype"):
                compile_hf_llama_cell_fixture(
                    str(snapshot),
                    output,
                    layer_start=1,
                    layer_end=3,
                    world_size=2,
                    output_dtype="int8",
                )
            self.assertFalse(output.exists())

    def test_compiles_only_selected_sharded_layers_to_exact_v2_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "checkpoint"
            output = root / "compiled"
            reference = root / "reference"
            dense = self._write_checkpoint(snapshot, source_dtype=torch.float16)
            requested: list[str] = []
            original_get = compiler._CheckpointTensorReader.get_tensor

            def tracking_get(reader, name: str):
                requested.append(name)
                return original_get(reader, name)

            with patch.object(
                compiler._CheckpointTensorReader,
                "get_tensor",
                tracking_get,
            ):
                result = compile_hf_llama_cell_fixture(
                    str(snapshot),
                    output,
                    layer_start=1,
                    layer_end=3,
                    world_size=2,
                )

            write_llama_stage_cell_fixture(
                reference,
                [
                    {name: value.float() for name, value in dense[layer].items()}
                    for layer in (1, 2)
                ],
                world_size=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=2,
                rms_norm_epsilon=1e-5,
                rope_theta=20_000.0,
            )

            self.assertEqual(result.schema, CELL_FIXTURE_SCHEMA)
            self.assertEqual(result.layer_start, 1)
            self.assertEqual(result.layer_end, 3)
            self.assertEqual(result.source_tensor_count, 18)
            self.assertEqual(result.output_tensor_count, 36)
            self.assertEqual(len(requested), 18)
            self.assertTrue(all("model.layers.1." in name or "model.layers.2." in name for name in requested))
            self.assertFalse(any("model.layers.0." in name for name in requested))
            actual_manifest = json.loads(
                (output / "cell.json").read_text(encoding="utf-8")
            )
            expected_manifest = json.loads(
                (reference / "cell.json").read_text(encoding="utf-8")
            )
            actual_digests = actual_manifest.pop("shardSha256")
            expected_manifest.pop("shardSha256")
            self.assertEqual(actual_manifest, expected_manifest)
            self.assertEqual(
                actual_digests,
                [
                    hashlib.sha256(
                        (output / f"rank-{rank:03d}.safetensors").read_bytes()
                    ).hexdigest()
                    for rank in range(2)
                ],
            )
            for rank in range(2):
                actual = load_file(str(output / f"rank-{rank:03d}.safetensors"))
                expected = load_file(str(reference / f"rank-{rank:03d}.safetensors"))
                self.assertEqual(set(actual), set(expected))
                for name in expected:
                    torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
                    self.assertEqual(actual[name].dtype, torch.float32)

            provenance = json.loads((output / "source.json").read_text(encoding="utf-8"))
            self.assertEqual(provenance["schema"], CELL_FIXTURE_SOURCE_SCHEMA)
            self.assertEqual(provenance["targetSchema"], CELL_FIXTURE_SCHEMA)
            self.assertEqual(provenance["layerRange"], {"start": 1, "end": 3, "count": 2})
            self.assertEqual(provenance["source"]["dtypes"], ["float16"])
            self.assertNotIn("snapshot", provenance["source"])
            self.assertEqual(
                result.source_tensor_bytes,
                sum(value.numel() * value.element_size() for layer in dense[1:3] for value in layer.values()),
            )
            self.assertEqual(
                result.output_tensor_bytes,
                sum(
                    value.numel() * value.element_size()
                    for rank in range(2)
                    for value in load_file(str(output / f"rank-{rank:03d}.safetensors")).values()
                ),
            )

    def test_compiles_mqa_for_more_members_than_kv_heads_with_minimal_replication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "checkpoint"
            output = root / "compiled"
            dense = self._write_checkpoint(
                snapshot,
                source_dtype=torch.float32,
                num_key_value_heads=1,
            )

            result = compile_hf_llama_cell_fixture(
                str(snapshot),
                output,
                layer_start=1,
                layer_end=3,
                world_size=2,
            )

            self.assertEqual(result.output_tensor_count, 36)
            manifest = json.loads((output / "cell.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["worldSize"], 2)
            self.assertTrue(
                all(layer["numKeyValueHeads"] == 1 for layer in manifest["layers"])
            )
            rank_zero = load_file(str(output / "rank-000.safetensors"))
            rank_one = load_file(str(output / "rank-001.safetensors"))
            for layer_offset, source_layer in enumerate((1, 2)):
                for logical_name in ("key", "value"):
                    name = f"layers.{layer_offset}.{logical_name}"
                    # The one MQA KV head is referenced by both Q partitions,
                    # therefore and only therefore it is replicated verbatim.
                    torch.testing.assert_close(
                        rank_zero[name], rank_one[name], rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        rank_zero[name],
                        dense[source_layer][logical_name].float(),
                        rtol=0,
                        atol=0,
                    )
                self.assertEqual(rank_zero[f"layers.{layer_offset}.query"].shape, (4, 8))
                self.assertEqual(rank_one[f"layers.{layer_offset}.query"].shape, (4, 8))

    def test_rejects_unsupported_bias_without_publishing_partial_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "checkpoint"
            output = root / "compiled"
            self._write_checkpoint(snapshot, source_dtype=torch.float32, attention_bias=True)

            with self.assertRaisesRegex(ValueError, "attention projection bias"):
                compile_hf_llama_cell_fixture(
                    str(snapshot),
                    output,
                    layer_start=1,
                    layer_end=3,
                    world_size=2,
                )

            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".compiled.compile-*")), [])

    def test_shape_failure_is_atomic_and_does_not_replace_an_existing_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "checkpoint"
            output = root / "compiled"
            self._write_checkpoint(snapshot, source_dtype=torch.float32, bad_layer=2)

            with self.assertRaisesRegex(ValueError, "has shape"):
                compile_hf_llama_cell_fixture(
                    str(snapshot),
                    output,
                    layer_start=1,
                    layer_end=3,
                    world_size=2,
                )
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".compiled.compile-*")), [])

            output.mkdir()
            marker = output / "belongs-to-user.txt"
            marker.write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                compile_hf_llama_cell_fixture(
                    str(snapshot),
                    output,
                    layer_start=1,
                    layer_end=3,
                    world_size=2,
                )
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_rejects_ranges_and_topologies_the_v2_runtime_cannot_execute(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "checkpoint"
            self._write_checkpoint(snapshot, source_dtype=torch.float32)

            with self.assertRaisesRegex(ValueError, "at least two layers"):
                compile_hf_llama_cell_fixture(
                    str(snapshot), root / "one", layer_start=1, layer_end=2, world_size=2
                )
            with self.assertRaisesRegex(ValueError, "world_size cannot exceed"):
                compile_hf_llama_cell_fixture(
                    str(snapshot), root / "wide", layer_start=1, layer_end=3, world_size=5
                )

    @unittest.skipUnless(
        os.getenv("RUN_DISTRIBUTED_MODEL_TESTS") == "1",
        "set RUN_DISTRIBUTED_MODEL_TESTS=1 for the cached/downloaded SmolLM2 fixture check",
    )
    def test_compiles_real_smollm2_intermediate_layers_without_retaining_fixture(self) -> None:
        model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "cell"
            result = compile_hf_llama_cell_fixture(
                model_name,
                output,
                layer_start=15,
                layer_end=17,
                world_size=2,
            )
            self.assertEqual(result.source_tensor_count, 18)
            manifest = json.loads((output / "cell.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema"], CELL_FIXTURE_SCHEMA)
            self.assertEqual(manifest["layerCount"], 2)
            self.assertEqual(manifest["worldSize"], 2)

            snapshot = Path(resolve_model_snapshot(model_name))
            mapping = _checkpoint_key_map(snapshot)
            source_name = "model.layers.15.self_attn.q_proj.weight"
            with safe_open(
                snapshot / mapping[source_name], framework="pt", device="cpu"
            ) as source_file:
                source = source_file.get_tensor(source_name)
            with safe_open(
                output / "rank-000.safetensors", framework="pt", device="cpu"
            ) as local_file:
                local = local_file.get_tensor("layers.0.query")
            torch.testing.assert_close(
                local,
                source[: local.shape[0], :].float(),
                rtol=0,
                atol=0,
            )

    @staticmethod
    def _write_checkpoint(
        root: Path,
        *,
        source_dtype: torch.dtype,
        attention_bias: bool = False,
        bad_layer: int | None = None,
        num_key_value_heads: int = 2,
    ) -> list[dict[str, torch.Tensor]]:
        root.mkdir(parents=True)
        (root / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "llama",
                    "architectures": ["LlamaForCausalLM"],
                    "vocab_size": 32,
                    "hidden_size": 8,
                    "intermediate_size": 10,
                    "num_hidden_layers": 4,
                    "num_attention_heads": 4,
                    "num_key_value_heads": num_key_value_heads,
                    "head_dim": 2,
                    "rms_norm_eps": 1e-5,
                    "rope_theta": 20_000.0,
                    "hidden_act": "silu",
                    "attention_bias": attention_bias,
                    "mlp_bias": False,
                    "tie_word_embeddings": True,
                }
            ),
            encoding="utf-8",
        )
        dense: list[dict[str, torch.Tensor]] = []
        first: dict[str, torch.Tensor] = {
            "model.layers.0.unused.weight": torch.arange(3, dtype=source_dtype)
        }
        second: dict[str, torch.Tensor] = {}
        weight_map: dict[str, str] = {
            "model.layers.0.unused.weight": "model-00001-of-00002.safetensors"
        }
        counter = 1
        key_value_features = num_key_value_heads * 2
        for layer in range(4):
            logical = {
                "input_norm": torch.arange(8, dtype=source_dtype) + counter,
                "post_attention_norm": torch.arange(8, dtype=source_dtype) + counter + 1,
                "query": torch.arange(64, dtype=source_dtype).reshape(8, 8) + counter,
                "key": torch.arange(
                    key_value_features * 8, dtype=source_dtype
                ).reshape(key_value_features, 8) + counter,
                "value": torch.arange(
                    key_value_features * 8, dtype=source_dtype
                ).reshape(key_value_features, 8) + counter + 1,
                "output": torch.arange(64, dtype=source_dtype).reshape(8, 8) + counter + 2,
                "gate": torch.arange(80, dtype=source_dtype).reshape(10, 8) + counter,
                "up": torch.arange(80, dtype=source_dtype).reshape(10, 8) + counter + 1,
                "down": torch.arange(80, dtype=source_dtype).reshape(8, 10) + counter + 2,
            }
            if bad_layer == layer:
                logical["down"] = torch.zeros((8, 9), dtype=source_dtype)
            dense.append(logical)
            for position, (name, suffix) in enumerate(compiler._HF_SUFFIXES.items()):
                checkpoint_name = f"model.layers.{layer}.{suffix}"
                target = first if position % 2 == 0 else second
                target[checkpoint_name] = logical[name]
                filename = (
                    "model-00001-of-00002.safetensors"
                    if position % 2 == 0
                    else "model-00002-of-00002.safetensors"
                )
                weight_map[checkpoint_name] = filename
            if attention_bias and layer in (1, 2):
                name = f"model.layers.{layer}.self_attn.q_proj.bias"
                second[name] = torch.zeros(8, dtype=source_dtype)
                weight_map[name] = "model-00002-of-00002.safetensors"
            counter += 10
        save_file(first, root / "model-00001-of-00002.safetensors")
        save_file(second, root / "model-00002-of-00002.safetensors")
        (root / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {}, "weight_map": weight_map}),
            encoding="utf-8",
        )
        return dense


if __name__ == "__main__":
    unittest.main()
