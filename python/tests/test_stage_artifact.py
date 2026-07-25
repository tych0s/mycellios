from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from safetensors.torch import load_file, save_file
import torch
from transformers import (
    AutoModelForCausalLM,
    Glm4MoeConfig,
    LlamaConfig,
    Qwen3Config,
    Qwen3MoeConfig,
)

import distributed_runtime.stage_artifact as stage_artifact
from distributed_runtime.model import StageModelSpec, StageRunner
from distributed_runtime.stage_artifact import (
    STAGE_ARTIFACT_CONFIG,
    STAGE_ARTIFACT_MANIFEST,
    STAGE_ARTIFACT_SCHEMA,
    STAGE_ARTIFACT_WEIGHTS,
    STAGE_TENSOR_ABI,
    compile_safetensors_stage_artifact,
    parse_stage_artifact_manifest,
    verify_stage_artifact,
)


class StageArtifactTests(unittest.TestCase):
    def test_compiles_only_the_requested_indexed_layers_without_materializing_tensors(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            source = self._write_checkpoint(checkpoint, family="llama")
            destination = root / "stage-1-3"
            real_safe_open = stage_artifact.safe_open

            def guarded_safe_open(*args, **kwargs):
                return _NoMaterializeSafeOpen(real_safe_open(*args, **kwargs))

            with patch.object(stage_artifact, "safe_open", guarded_safe_open):
                result = compile_safetensors_stage_artifact(
                    str(checkpoint),
                    destination,
                    layer_start=1,
                    layer_end=3,
                )

            self.assertEqual(result.schema, STAGE_ARTIFACT_SCHEMA)
            self.assertEqual(result.artifact_identity, f"sha256:{result.package_id}")
            self.assertEqual((result.layer_start, result.layer_end), (1, 3))
            self.assertEqual(
                set(path.name for path in destination.iterdir()),
                {
                    STAGE_ARTIFACT_CONFIG,
                    STAGE_ARTIFACT_MANIFEST,
                    STAGE_ARTIFACT_WEIGHTS,
                },
            )
            weights = load_file(str(destination / STAGE_ARTIFACT_WEIGHTS))
            self.assertTrue(weights)
            self.assertTrue(
                all(
                    name.startswith("model.layers.1.")
                    or name.startswith("model.layers.2.")
                    for name in weights
                )
            )
            self.assertFalse(any("model.layers.0." in name for name in weights))
            self.assertFalse(any("model.layers.3." in name for name in weights))
            for name, value in weights.items():
                torch.testing.assert_close(value, source[name], rtol=0, atol=0)

            manifest = json.loads(
                (destination / STAGE_ARTIFACT_MANIFEST).read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["model"]["family"], "llama")
            self.assertEqual(manifest["model"]["architecture"], "LlamaForCausalLM")
            self.assertEqual(manifest["model"]["adapter"], "transformers-llama-v1")
            self.assertEqual(manifest["tensorAbi"]["id"], STAGE_TENSOR_ABI)
            self.assertEqual(manifest["tensorAbi"]["weightDtypes"], ["F16"])
            self.assertEqual(
                manifest["model"]["quantization"],
                {
                    "scheme": "native-f16",
                    "configSha256": None,
                    "weightDtypes": ["F16"],
                },
            )
            self.assertEqual(
                manifest["artifact"]["tensorBytes"],
                sum(value.numel() * value.element_size() for value in weights.values()),
            )
            self.assertEqual(
                hashlib.sha256(
                    (destination / STAGE_ARTIFACT_WEIGHTS).read_bytes()
                ).hexdigest(),
                result.weights_sha256,
            )
            verified = verify_stage_artifact(destination)
            self.assertEqual(verified.manifest.package_id, result.package_id)

    def test_first_and_last_packages_receive_only_their_required_endpoints(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            self._write_checkpoint(checkpoint, family="llama")
            first = root / "first"
            last = root / "last"

            compile_safetensors_stage_artifact(
                str(checkpoint), first, layer_start=0, layer_end=2
            )
            compile_safetensors_stage_artifact(
                str(checkpoint), last, layer_start=2, layer_end=4
            )

            first_names = set(load_file(str(first / STAGE_ARTIFACT_WEIGHTS)))
            last_names = set(load_file(str(last / STAGE_ARTIFACT_WEIGHTS)))
            self.assertIn("model.embed_tokens.weight", first_names)
            self.assertNotIn("model.norm.weight", first_names)
            self.assertNotIn("lm_head.weight", first_names)
            self.assertNotIn("model.embed_tokens.weight", last_names)
            self.assertIn("model.norm.weight", last_names)
            self.assertIn("lm_head.weight", last_names)
            self.assertFalse(first_names.intersection(last_names))
            self.assertTrue(
                all(
                    "model.layers.0." in name
                    or "model.layers.1." in name
                    or name == "model.embed_tokens.weight"
                    for name in first_names
                )
            )
            self.assertTrue(
                all(
                    "model.layers.2." in name
                    or "model.layers.3." in name
                    or name in {"model.norm.weight", "lm_head.weight"}
                    for name in last_names
                )
            )

    def test_qwen3_family_contract_keeps_qk_norm_and_rejects_aliasing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "qwen3"
            self._write_checkpoint(checkpoint, family="qwen3")
            destination = root / "stage"

            result = compile_safetensors_stage_artifact(
                str(checkpoint), destination, layer_start=1, layer_end=2
            )

            self.assertEqual(result.family, "qwen3")
            self.assertEqual(result.adapter, "transformers-qwen3-v1")
            names = set(load_file(str(destination / STAGE_ARTIFACT_WEIGHTS)))
            self.assertIn("model.layers.1.self_attn.q_norm.weight", names)
            self.assertIn("model.layers.1.self_attn.k_norm.weight", names)
            self.assertTrue(all("model.layers.1." in name for name in names))
            manifest = verify_stage_artifact(destination).manifest.to_document()
            self.assertEqual(manifest["model"]["architecture"], "Qwen3ForCausalLM")

    def test_every_certified_moe_family_compiles_its_exact_global_range(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for family, adapter_id in (
                ("qwen3_moe", "transformers-qwen3-moe-v1"),
                ("glm4_moe", "transformers-glm4-moe-v1"),
            ):
                checkpoint = root / family
                self._write_checkpoint(checkpoint, family=family)
                first = root / f"{family}-first"
                last = root / f"{family}-last"
                compile_safetensors_stage_artifact(
                    str(checkpoint), first, layer_start=0, layer_end=1
                )
                result = compile_safetensors_stage_artifact(
                    str(checkpoint), last, layer_start=1, layer_end=2
                )

                self.assertEqual(result.family, family)
                self.assertEqual(result.adapter, adapter_id)
                first_names = set(load_file(str(first / STAGE_ARTIFACT_WEIGHTS)))
                last_names = set(load_file(str(last / STAGE_ARTIFACT_WEIGHTS)))
                self.assertTrue(
                    all(
                        "model.layers.0." in name or name == "model.embed_tokens.weight"
                        for name in first_names
                    )
                )
                self.assertTrue(
                    all(
                        "model.layers.1." in name
                        or name in {"model.norm.weight", "lm_head.weight"}
                        for name in last_names
                    )
                )
                verify_stage_artifact(first)
                verify_stage_artifact(last)

    def test_runtime_accepts_a_verified_package_and_rejects_one_tampered_byte(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            self._write_checkpoint(checkpoint, family="llama")
            destination = root / "stage"
            result = compile_safetensors_stage_artifact(
                str(checkpoint), destination, layer_start=1, layer_end=3
            )
            spec = StageModelSpec(
                model_name=str(destination),
                layer_start=1,
                layer_end=3,
                total_layers=4,
                threads=1,
                artifact_identity=result.artifact_identity,
            )

            with self.assertRaisesRegex(
                ValueError, "package identity does not match the request"
            ):
                StageRunner(
                    StageModelSpec(
                        model_name=str(destination),
                        layer_start=1,
                        layer_end=3,
                        total_layers=4,
                        threads=1,
                        artifact_identity="sha256:" + "0" * 64,
                    ),
                    device="cpu",
                )
            runner = StageRunner(spec, device="cpu")
            runner.close()
            weights_path = destination / STAGE_ARTIFACT_WEIGHTS
            contents = bytearray(weights_path.read_bytes())
            contents[-1] ^= 0x01
            weights_path.write_bytes(contents)

            with self.assertRaisesRegex(ValueError, "stage artifact digest mismatch"):
                verify_stage_artifact(destination)
            with self.assertRaisesRegex(ValueError, "stage artifact digest mismatch"):
                StageRunner(spec, device="cpu")

            missing_manifest = root / "missing-manifest"
            missing_result = compile_safetensors_stage_artifact(
                str(checkpoint),
                missing_manifest,
                layer_start=1,
                layer_end=3,
            )
            (missing_manifest / STAGE_ARTIFACT_MANIFEST).unlink()
            with self.assertRaisesRegex(FileNotFoundError, "manifest is missing"):
                StageRunner(
                    StageModelSpec(
                        model_name=str(missing_manifest),
                        layer_start=1,
                        layer_end=3,
                        total_layers=4,
                        threads=1,
                        artifact_identity=missing_result.artifact_identity,
                    ),
                    device="cpu",
                )

    def test_manifest_tampering_and_unsealed_files_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            self._write_checkpoint(checkpoint, family="llama")
            destination = root / "stage"
            compile_safetensors_stage_artifact(
                str(checkpoint), destination, layer_start=1, layer_end=3
            )
            manifest_path = destination / STAGE_ARTIFACT_MANIFEST
            document = json.loads(manifest_path.read_text(encoding="utf-8"))
            document["model"]["source"] = "tampered/model"
            manifest_path.write_text(json.dumps(document), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "package identity does not match"):
                verify_stage_artifact(destination)

            canonical = root / "canonical"
            compile_safetensors_stage_artifact(
                str(checkpoint), canonical, layer_start=1, layer_end=3
            )
            canonical_manifest = canonical / STAGE_ARTIFACT_MANIFEST
            canonical_manifest.write_bytes(canonical_manifest.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "encoding is not canonical"):
                verify_stage_artifact(canonical)

            # Recompile to isolate the exact-directory gate.
            other = root / "other"
            compile_safetensors_stage_artifact(
                str(checkpoint), other, layer_start=1, layer_end=3
            )
            (other / "unsealed.bin").write_bytes(b"not part of the package")
            with self.assertRaisesRegex(ValueError, "unsealed files"):
                verify_stage_artifact(other)

    def test_unknown_selected_tensor_and_unknown_family_publish_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            self._write_checkpoint(
                checkpoint,
                family="llama",
                extra_tensor="model.layers.1.not_certified.weight",
            )
            destination = root / "stage"

            with self.assertRaisesRegex(KeyError, "unconsumed required tensors"):
                compile_safetensors_stage_artifact(
                    str(checkpoint),
                    destination,
                    layer_start=1,
                    layer_end=3,
                )
            self.assertFalse(destination.exists())
            self.assertEqual(list(root.glob(".stage.compile-*")), [])

            unsupported = root / "unsupported"
            self._write_checkpoint(unsupported, family="llama")
            config_path = unsupported / "config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["model_type"] = "bert"
            config["architectures"] = ["BertForMaskedLM"]
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "no certified selective-stage adapter"
            ):
                compile_safetensors_stage_artifact(
                    str(unsupported),
                    root / "unsupported-output",
                    layer_start=1,
                    layer_end=3,
                )
            self.assertFalse((root / "unsupported-output").exists())

    def test_existing_destination_is_never_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            self._write_checkpoint(checkpoint, family="llama")
            destination = root / "stage"
            destination.mkdir()
            marker = destination / "user-owned.txt"
            marker.write_text("keep", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "already exists"):
                compile_safetensors_stage_artifact(
                    str(checkpoint),
                    destination,
                    layer_start=1,
                    layer_end=3,
                )
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_writer_failure_removes_the_private_partial_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            self._write_checkpoint(checkpoint, family="llama")
            destination = root / "stage"

            def fail_after_partial_write(path, _selected):
                path.write_bytes(b"partial")
                raise RuntimeError("injected writer failure")

            with (
                patch.object(
                    stage_artifact,
                    "_write_stage_safetensors",
                    fail_after_partial_write,
                ),
                self.assertRaisesRegex(RuntimeError, "injected writer failure"),
            ):
                compile_safetensors_stage_artifact(
                    str(checkpoint),
                    destination,
                    layer_start=1,
                    layer_end=3,
                )
            self.assertFalse(destination.exists())
            self.assertEqual(list(root.glob(".stage.compile-*")), [])

    def test_parser_rejects_unknown_fields_before_any_file_access(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            self._write_checkpoint(checkpoint, family="llama")
            destination = root / "stage"
            compile_safetensors_stage_artifact(
                str(checkpoint), destination, layer_start=1, layer_end=3
            )
            document = json.loads(
                (destination / STAGE_ARTIFACT_MANIFEST).read_text(encoding="utf-8")
            )
            document["surprise"] = True
            with self.assertRaisesRegex(ValueError, "unknown or missing fields"):
                parse_stage_artifact_manifest(document)

    @staticmethod
    def _write_checkpoint(
        root: Path,
        *,
        family: str,
        extra_tensor: str | None = None,
    ) -> dict[str, torch.Tensor]:
        root.mkdir(parents=True)
        if family == "llama":
            config = LlamaConfig(
                vocab_size=32,
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=4,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=2,
                tie_word_embeddings=False,
                attention_bias=False,
                mlp_bias=False,
            )
            config.architectures = ["LlamaForCausalLM"]
        elif family == "qwen3":
            config = Qwen3Config(
                vocab_size=32,
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=3,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=2,
                tie_word_embeddings=False,
            )
            config.architectures = ["Qwen3ForCausalLM"]
        elif family == "qwen3_moe":
            config = Qwen3MoeConfig(
                vocab_size=32,
                hidden_size=32,
                intermediate_size=64,
                moe_intermediate_size=16,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
                num_experts=4,
                num_experts_per_tok=2,
                max_position_embeddings=64,
                tie_word_embeddings=False,
                architectures=["Qwen3MoeForCausalLM"],
            )
        elif family == "glm4_moe":
            config = Glm4MoeConfig(
                vocab_size=32,
                hidden_size=32,
                intermediate_size=64,
                moe_intermediate_size=16,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
                n_routed_experts=4,
                num_experts_per_tok=2,
                n_shared_experts=1,
                first_k_dense_replace=1,
                max_position_embeddings=64,
                tie_word_embeddings=False,
                architectures=["Glm4MoeForCausalLM"],
            )
        else:
            raise ValueError(f"unsupported test family: {family}")
        config.save_pretrained(root)
        torch.manual_seed(7)
        model = AutoModelForCausalLM.from_config(config, dtype=torch.float16)
        state = {
            name: value.detach().cpu().contiguous().clone()
            for name, value in model.state_dict().items()
        }
        if extra_tensor is not None:
            state[extra_tensor] = torch.arange(8, dtype=torch.float16)
        first: dict[str, torch.Tensor] = {}
        second: dict[str, torch.Tensor] = {}
        weight_map: dict[str, str] = {}
        for index, (name, value) in enumerate(sorted(state.items())):
            target = first if index % 2 == 0 else second
            target[name] = value
            weight_map[name] = (
                "model-00001-of-00002.safetensors"
                if index % 2 == 0
                else "model-00002-of-00002.safetensors"
            )
        save_file(first, root / "model-00001-of-00002.safetensors")
        save_file(second, root / "model-00002-of-00002.safetensors")
        (root / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {}, "weight_map": weight_map}),
            encoding="utf-8",
        )
        del model
        return state


class _NoMaterializeSafeOpen:
    def __init__(self, inner) -> None:
        self._inner = inner
        self._reader = None

    def __enter__(self):
        self._reader = self._inner.__enter__()
        return self

    def __exit__(self, *args):
        return self._inner.__exit__(*args)

    def keys(self):
        return self._reader.keys()

    def get_tensor(self, name):
        raise AssertionError(f"compiler materialized source tensor {name!r}")


if __name__ == "__main__":
    unittest.main()
