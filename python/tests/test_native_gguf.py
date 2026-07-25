from __future__ import annotations

import json
from pathlib import Path
import struct
import tempfile
import unittest
from argparse import ArgumentParser
from unittest.mock import patch

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM

from distributed_runtime.dense_tiering import DenseTieringConfig, DenseTieringError
from distributed_runtime.native_gguf import (
    GgufTensor,
    NativeGgufError,
    NativeGgufStagePackage,
    build_native_gguf_fleet,
    build_native_gguf_stage,
    dequantize_gguf_tensor,
    materialize_native_gguf_stage,
    parse_gguf,
    verify_native_gguf_fleet,
    verify_native_gguf_stage,
)
from distributed_runtime.native_gguf_runtime import (
    NativeGgufRuntimeConfig,
    NativeGgufStageRunner,
    add_native_gguf_arguments,
    native_gguf_launch_document,
    native_gguf_runtime_from_args,
)
from distributed_runtime.model import StageModelSpec
from distributed_runtime.model import StageRunner
from distributed_runtime.executor_abi import validate_executor_chain
from distributed_runtime.engine import (
    DistributedPipelineEngine,
    GenerationInput,
    PipelineEngineConfig,
)
from distributed_runtime.protocol import TensorCodec
from distributed_runtime.stage import build_stage_runner, validate_stage_config
from distributed_runtime.stage_cli import (
    build_config as build_stage_config,
    parse_args as parse_stage_args,
)


class NativeGgufTests(unittest.TestCase):
    def test_builds_exact_first_and_last_layer_packages(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "model.gguf"
            config = _write_config(root, layers=2, hidden=4)
            _write_fixture_gguf(source)

            first = build_native_gguf_stage(
                source,
                root / "first",
                config_source=config,
                layer_start=0,
                layer_end=1,
                model_source="hf://fixture/model",
                model_revision="a" * 40,
            )
            last = build_native_gguf_stage(
                source,
                root / "last",
                config_source=config,
                layer_start=1,
                layer_end=2,
                model_source="hf://fixture/model",
                model_revision="a" * 40,
            )

            self.assertIn("token_embd.weight", first.tensor_names)
            self.assertTrue(any(name.startswith("blk.0.") for name in first.tensor_names))
            self.assertFalse(any(name.startswith("blk.1.") for name in first.tensor_names))
            self.assertNotIn("output.weight", first.tensor_names)

            self.assertIn("output.weight", last.tensor_names)
            self.assertIn("output_norm.weight", last.tensor_names)
            self.assertTrue(any(name.startswith("blk.1.") for name in last.tensor_names))
            self.assertFalse(any(name.startswith("blk.0.") for name in last.tensor_names))
            self.assertNotIn("token_embd.weight", last.tensor_names)
            self.assertNotEqual(first.package_id, last.package_id)

    def test_materializes_only_mapped_stage_tensors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "model.gguf"
            config = _write_config(root, layers=2, hidden=4)
            fixture = _write_fixture_gguf(source)
            package = build_native_gguf_stage(
                source,
                root / "stage",
                config_source=config,
                layer_start=1,
                layer_end=2,
                model_source="hf://fixture/model",
                model_revision=None,
            )

            materialized = materialize_native_gguf_stage(
                package.root, root / "materialized"
            )
            tensors = load_file(materialized / "model.safetensors")
            self.assertEqual(
                set(tensors),
                {
                    "model.layers.1.input_layernorm.weight",
                    "model.layers.1.self_attn.q_proj.weight",
                    "model.norm.weight",
                    "lm_head.weight",
                },
            )
            self.assertTrue(
                torch.equal(
                    tensors["model.layers.1.self_attn.q_proj.weight"],
                    torch.from_numpy(fixture["blk.1.attn_q.weight"]).reshape(4, 4),
                )
            )
            self.assertTrue(
                torch.equal(
                    tensors["lm_head.weight"],
                    torch.from_numpy(fixture["output.weight"]).reshape(8, 4),
                )
            )

    def test_manifest_and_payload_tampering_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "model.gguf"
            config = _write_config(root, layers=2, hidden=4)
            _write_fixture_gguf(source)
            package = build_native_gguf_stage(
                source,
                root / "stage",
                config_source=config,
                layer_start=0,
                layer_end=1,
                model_source="hf://fixture/model",
                model_revision=None,
            )
            stage_file = package.root / "native-stage.gguf"
            payload = bytearray(stage_file.read_bytes())
            payload[-1] ^= 0x01
            stage_file.write_bytes(payload)
            with self.assertRaisesRegex(NativeGgufError, "digest differs"):
                verify_native_gguf_stage(package.root)

    def test_existing_destination_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "model.gguf"
            config = _write_config(root, layers=2, hidden=4)
            _write_fixture_gguf(source)
            destination = root / "stage"
            destination.mkdir()
            marker = destination / "keep.txt"
            marker.write_text("owned", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                build_native_gguf_stage(
                    source,
                    destination,
                    config_source=config,
                    layer_start=0,
                    layer_end=1,
                    model_source="hf://fixture/model",
                    model_revision=None,
                )
            self.assertEqual(marker.read_text(encoding="utf-8"), "owned")

    def test_builds_and_verifies_one_atomic_complete_stage_fleet(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "model.gguf"
            config = _write_config(root, layers=2, hidden=4)
            _write_fixture_gguf(source)
            fleet = build_native_gguf_fleet(
                source,
                root / "fleet",
                config_source=config,
                model_source="hf://fixture/model",
                model_revision="a" * 40,
                layers_per_stage=1,
            )
            self.assertEqual(len(fleet.stages), 2)
            self.assertEqual(
                [item.layer_start for item in fleet.stages],
                [0, 1],
            )
            self.assertEqual(
                {item.artifact_identity for item in fleet.stages},
                {fleet.artifact_identity},
            )
            verified = verify_native_gguf_fleet(
                fleet.root,
                expected_fleet_id=fleet.fleet_id,
            )
            self.assertEqual(verified.fleet_id, fleet.fleet_id)
            self.assertEqual(verified.total_layers, 2)

            (fleet.root / "unsealed.bin").write_bytes(b"unexpected")
            with self.assertRaisesRegex(NativeGgufError, "unsealed entries"):
                verify_native_gguf_fleet(fleet.root)

    def test_fleet_refuses_gaps_before_writing_any_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "model.gguf"
            config = _write_config(root, layers=2, hidden=4)
            _write_fixture_gguf(source)
            destination = root / "fleet"
            with self.assertRaisesRegex(
                NativeGgufError, "ordered, contiguous and complete"
            ):
                build_native_gguf_fleet(
                    source,
                    destination,
                    config_source=config,
                    model_source="hf://fixture/model",
                    model_revision=None,
                    ranges=((0, 1),),
                )
            self.assertFalse(destination.exists())

    def test_unimplemented_quant_is_preserved_but_execution_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "model.gguf"
            config = _write_config(root, layers=1, hidden=4)
            tensors = _basic_tensors(layers=1)
            tensors["blk.0.attn_q.weight"] = (9, (32,), bytes(40))
            _write_gguf(source, layers=1, hidden=4, tensors=tensors)
            package = build_native_gguf_stage(
                source,
                root / "stage",
                config_source=config,
                layer_start=0,
                layer_end=1,
                model_source="hf://fixture/model",
                model_revision=None,
            )
            self.assertIn("blk.0.attn_q.weight", package.tensor_names)
            with self.assertRaisesRegex(NativeGgufError, "unsupported executable"):
                materialize_native_gguf_stage(package.root, root / "materialized")

    def test_overlapping_tensor_ranges_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.gguf"
            tensors = _basic_tensors(layers=1)
            _write_gguf(path, layers=1, hidden=4, tensors=tensors, overlap=True)
            with self.assertRaisesRegex(NativeGgufError, "overlap"):
                parse_gguf(path)

    def test_quantized_row_width_must_match_ggml_block_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad-row.gguf"
            tensors = _basic_tensors(layers=1)
            tensors["blk.0.attn_q.weight"] = (2, (16, 2), bytes(18))
            _write_gguf(path, layers=1, hidden=4, tensors=tensors)
            with self.assertRaisesRegex(NativeGgufError, "row width"):
                parse_gguf(path)

    def test_tensor_rank_above_ggml_limit_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad-rank.gguf"
            tensors = _basic_tensors(layers=1)
            tensors["blk.0.attn_q.weight"] = (
                0,
                (1, 1, 1, 4, 4),
                bytes(4 * 4 * 4),
            )
            _write_gguf(path, layers=1, hidden=4, tensors=tensors)
            with self.assertRaisesRegex(NativeGgufError, "invalid rank"):
                parse_gguf(path)

    def test_empty_metadata_strings_round_trip_in_native_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "empty-string.gguf"
            config = _write_config(root, layers=1, hidden=4)
            _write_gguf(
                source,
                layers=1,
                hidden=4,
                tensors=_basic_tensors(layers=1),
                extra_metadata=(("llama.test_empty", 8, ""),),
            )
            package = build_native_gguf_stage(
                source,
                root / "stage",
                config_source=config,
                layer_start=0,
                layer_end=1,
                model_source="hf://fixture/model",
                model_revision=None,
            )
            parsed = parse_gguf(package.root / "native-stage.gguf")
            self.assertEqual(
                parsed.metadata_map()["llama.test_empty"],
                "",
            )

    def test_model_identity_binds_the_exact_runtime_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "model.gguf"
            config = _write_config(root, layers=2, hidden=4)
            _write_fixture_gguf(source)
            changed = root / "config-changed.json"
            document = json.loads(config.read_text(encoding="utf-8"))
            document["rope_theta"] = 20_000
            changed.write_text(json.dumps(document), encoding="utf-8")
            first = build_native_gguf_stage(
                source,
                root / "first",
                config_source=config,
                layer_start=0,
                layer_end=1,
                model_source="hf://fixture/model",
                model_revision=None,
            )
            different = build_native_gguf_stage(
                source,
                root / "different",
                config_source=changed,
                layer_start=1,
                layer_end=2,
                model_source="hf://fixture/model",
                model_revision=None,
            )
            self.assertNotEqual(first.config_sha256, different.config_sha256)
            self.assertNotEqual(first.artifact_identity, different.artifact_identity)

    def test_stage_verifier_rejects_every_unsealed_extra_entry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "model.gguf"
            config = _write_config(root, layers=1, hidden=4)
            _write_gguf(
                source,
                layers=1,
                hidden=4,
                tensors=_basic_tensors(layers=1),
            )
            package = build_native_gguf_stage(
                source,
                root / "stage",
                config_source=config,
                layer_start=0,
                layer_end=1,
                model_source="hf://fixture/model",
                model_revision=None,
            )
            (package.root / "unsealed.bin").write_bytes(b"unexpected")
            with self.assertRaisesRegex(NativeGgufError, "unsealed entries"):
                verify_native_gguf_stage(package.root)

    def test_llama3_rope_factors_are_proven_from_config_and_not_loaded_as_state(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _write_config(root, layers=1, hidden=4)
            config_document = json.loads(config.read_text(encoding="utf-8"))
            config_document["rope_scaling"] = {
                "rope_type": "llama3",
                "factor": 8.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8192,
            }
            config.write_text(json.dumps(config_document), encoding="utf-8")
            tensors = _basic_tensors(layers=1)
            tensors["rope_freqs.weight"] = (
                0,
                (2,),
                np.ones(2, dtype=np.float32).tobytes(),
            )
            source = root / "rope.gguf"
            _write_gguf(source, layers=1, hidden=4, tensors=tensors)
            package = build_native_gguf_stage(
                source,
                root / "stage",
                config_source=config,
                layer_start=0,
                layer_end=1,
                model_source="hf://fixture/llama3",
                model_revision=None,
            )
            materialized = materialize_native_gguf_stage(
                package.root,
                root / "materialized",
            )
            self.assertNotIn(
                "model.rope_freqs.weight",
                load_file(materialized / "model.safetensors"),
            )

            tensors["rope_freqs.weight"] = (
                0,
                (2,),
                np.array([1.0, 2.0], dtype=np.float32).tobytes(),
            )
            bad_source = root / "bad-rope.gguf"
            _write_gguf(bad_source, layers=1, hidden=4, tensors=tensors)
            bad = build_native_gguf_stage(
                bad_source,
                root / "bad-stage",
                config_source=config,
                layer_start=0,
                layer_end=1,
                model_source="hf://fixture/llama3",
                model_revision=None,
            )
            with self.assertRaisesRegex(
                NativeGgufError, "differs from the sealed"
            ):
                materialize_native_gguf_stage(
                    bad.root,
                    root / "bad-materialized",
                )

    def test_q4_q5_q8_native_dequantizers_match_constructed_values(self) -> None:
        cases = (
            (2, _q4_0_block(), np.arange(32, dtype=np.float32) % 16 - 8),
            (3, _q4_1_block(), (np.arange(32, dtype=np.float32) % 16) * 0.5 + 2),
            (6, _q5_0_block(), np.arange(32, dtype=np.float32) - 16),
            (7, _q5_1_block(), np.arange(32, dtype=np.float32) * 0.25 + 1),
            (8, _q8_0_block(), (np.arange(32, dtype=np.float32) - 16) * 0.5),
        )
        block_sizes = {2: 18, 3: 20, 6: 22, 7: 24, 8: 34}
        for ggml_type, raw, expected in cases:
            with self.subTest(ggml_type=ggml_type):
                tensor = GgufTensor(
                    name="fixture",
                    dimensions=(32,),
                    ggml_type=ggml_type,
                    relative_offset=0,
                    size_bytes=block_sizes[ggml_type],
                    data_offset=0,
                )
                actual = dequantize_gguf_tensor(tensor, raw)
                self.assertTrue(
                    torch.allclose(actual, torch.from_numpy(expected), atol=1e-3)
                )

    def test_native_k_quant_dequantizers_match_primary_layout_equations(self) -> None:
        cases = (
            (10, _q2_k_block()),
            (11, _q3_k_block()),
            (12, _q4_k_block()),
            (13, _q5_k_block()),
            (14, _q6_k_block()),
            (15, _q8_k_block()),
        )
        sizes = {10: 84, 11: 110, 12: 144, 13: 176, 14: 210, 15: 292}
        for ggml_type, (raw, expected) in cases:
            with self.subTest(ggml_type=ggml_type):
                repeated_raw = raw * 3
                tensor = GgufTensor(
                    name="fixture",
                    dimensions=(768,),
                    ggml_type=ggml_type,
                    relative_offset=0,
                    size_bytes=sizes[ggml_type] * 3,
                    data_offset=0,
                )
                actual = dequantize_gguf_tensor(tensor, repeated_raw)
                torch.testing.assert_close(
                    actual,
                    torch.from_numpy(np.tile(expected, 3)),
                    rtol=0,
                    atol=1e-6,
                )

    def test_runtime_arguments_require_a_sealed_package_pair(self) -> None:
        parser = ArgumentParser()
        add_native_gguf_arguments(parser)
        self.assertIsNone(native_gguf_runtime_from_args(parser.parse_args([])))
        with self.assertRaisesRegex(ValueError, "supplied together"):
            native_gguf_runtime_from_args(
                parser.parse_args(["--native-gguf-package", "stage"])
            )
        runtime = native_gguf_runtime_from_args(
            parser.parse_args(
                [
                    "--native-gguf-package",
                    "stage",
                    "--native-gguf-package-id",
                    "a" * 64,
                ]
            )
        )
        self.assertEqual(
            runtime,
            NativeGgufRuntimeConfig(package="stage", package_id="a" * 64),
        )

    def test_runtime_is_bound_to_package_range_and_canonical_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "model.gguf"
            config = _write_config(root, layers=2, hidden=4)
            _write_fixture_gguf(source)
            package = build_native_gguf_stage(
                source,
                root / "stage",
                config_source=config,
                layer_start=0,
                layer_end=1,
                model_source="hf://fixture/model",
                model_revision="a" * 40,
            )
            runtime = NativeGgufRuntimeConfig(
                package=str(package.root),
                package_id=package.package_id,
            )
            launch = native_gguf_launch_document(runtime)
            self.assertEqual(launch["modelIdentity"], package.artifact_identity)
            self.assertEqual(launch["packageIdentity"], package.package_identity)
            self.assertFalse(launch["externalRuntimeRequired"])

            wrong_range = StageModelSpec(
                model_name="sealed-native-gguf",
                layer_start=1,
                layer_end=2,
                total_layers=2,
                threads=1,
                artifact_identity=package.artifact_identity,
                canonical_model_source=package.model_source,
                canonical_model_revision=package.model_revision,
                stage_package_identity=package.package_identity,
            )
            with self.assertRaisesRegex(
                NativeGgufError, "layer_start differs from launch contract"
            ):
                NativeGgufStageRunner(wrong_range, runtime)

            wrong_identity = StageModelSpec(
                model_name="sealed-native-gguf",
                layer_start=0,
                layer_end=1,
                total_layers=2,
                threads=1,
                artifact_identity="sha256:" + "b" * 64,
                canonical_model_source=package.model_source,
                canonical_model_revision=package.model_revision,
                stage_package_identity=package.package_identity,
            )
            with self.assertRaisesRegex(
                NativeGgufError, "artifact identity differs"
            ):
                NativeGgufStageRunner(wrong_identity, runtime)

            valid = StageModelSpec(
                model_name="sealed-native-gguf",
                layer_start=0,
                layer_end=1,
                total_layers=2,
                threads=1,
                artifact_identity=package.artifact_identity,
                canonical_model_source=package.model_source,
                canonical_model_revision=package.model_revision,
                stage_package_identity=package.package_identity,
            )
            with (
                patch(
                    "distributed_runtime.native_gguf_runtime.AutoConfig.from_pretrained",
                    side_effect=AssertionError(
                        "model construction must happen after memory preflight"
                    ),
                ),
                self.assertRaisesRegex(
                    DenseTieringError,
                    "authenticated_gguf_stage_exceeds_host_ram_budget",
                ),
            ):
                NativeGgufStageRunner(
                    valid,
                    runtime,
                    device="cpu",
                    dense_tiering=DenseTieringConfig(
                        host_ram_budget_bytes=1,
                    ),
                )

    def test_two_native_gguf_stages_match_monolithic_greedy_tokens(self) -> None:
        """Physical F32 parity for the complete native package/runner path."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            config = LlamaConfig(
                vocab_size=32,
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=2,
                max_position_embeddings=64,
                tie_word_embeddings=False,
                attention_bias=False,
                mlp_bias=False,
            )
            config.architectures = ["LlamaForCausalLM"]
            torch.manual_seed(829)
            monolithic = LlamaForCausalLM(config).eval()
            monolithic.save_pretrained(checkpoint, safe_serialization=True)
            source = root / "model-f32.gguf"
            _write_complete_llama_gguf(
                source,
                state=monolithic.state_dict(),
                config=config,
            )
            packages = tuple(
                build_native_gguf_stage(
                    source,
                    root / f"stage-{start}-{end}",
                    config_source=checkpoint / "config.json",
                    layer_start=start,
                    layer_end=end,
                    model_source="hf://fixture/tiny-llama",
                    model_revision="c" * 40,
                )
                for start, end in ((0, 1), (1, 2))
            )
            self.assertEqual(
                packages[0].artifact_identity,
                packages[1].artifact_identity,
            )
            self.assertNotEqual(
                packages[0].package_identity,
                packages[1].package_identity,
            )

            classic_specs = (
                StageModelSpec(str(checkpoint), 0, 1, 2, 1),
                StageModelSpec(str(checkpoint), 1, 2, 2, 1),
            )
            classic = tuple(StageRunner(spec, device="cpu") for spec in classic_specs)
            with patch(
                "distributed_runtime.native_gguf.materialize_native_gguf_stage",
                side_effect=AssertionError(
                    "NativeGgufStageRunner must never materialize SafeTensors"
                ),
            ) as materialize:
                native = tuple(
                    _native_runner_from_stage_cli(package)
                    for package in packages
                )
            materialize.assert_not_called()
            try:
                validate_executor_chain(tuple(item.executor_manifest for item in native))
                self.assertEqual(
                    native[0].executor_manifest.model_identity,
                    packages[0].artifact_identity,
                )
                self.assertEqual(native[0].loader, "native-gdlp-gguf-stage")
                execution = native[0].execution_snapshot()
                startup = execution["nativeGgufStartup"]
                self.assertEqual(startup["loadMode"], "one-tensor-at-a-time")
                self.assertEqual(startup["temporarySafetensorsBytesWritten"], 0)
                self.assertGreater(startup["tensorCount"], 0)
                self.assertGreater(startup["maxExplicitLiveTensorBytes"], 0)
                self.assertGreaterEqual(
                    startup["avoidedFullStageTensorMapBytes"],
                    startup["largestDecodedTensorBytes"],
                )
                dense = execution["denseTiering"]
                self.assertEqual(dense["mode"], "cpu-resident")
                self.assertEqual(
                    dense["storageToRam"]["artifactBytesMaterialized"],
                    startup["encodedBytesRead"],
                )
                self.assertEqual(
                    dense["storageToRam"]["tensorReadOperations"],
                    startup["tensorReadOperations"],
                )
                input_ids = torch.tensor([[1, 7, 11, 3]], dtype=torch.long)
                expected = _monolithic_greedy(monolithic, input_ids, steps=5)
                classic_tokens = _two_stage_greedy(classic, input_ids, steps=5)
                native_tokens = _two_stage_greedy(native, input_ids, steps=5)
                self.assertEqual(classic_tokens, expected)
                self.assertEqual(native_tokens, expected)
            finally:
                for runner in (*classic, *native):
                    runner.close()

            engine = DistributedPipelineEngine(
                PipelineEngineConfig(
                    model_name=str(checkpoint),
                    boundaries=(0, 1, 2),
                    codec=TensorCodec.FP32,
                    threads_per_stage=1,
                    device="cpu",
                    startup_timeout_seconds=60,
                    socket_timeout_seconds=60,
                    native_gguf_stages=tuple(
                        NativeGgufRuntimeConfig(
                            package=str(package.root),
                            package_id=package.package_id,
                        )
                        for package in packages
                    ),
                    route_probe_interval_seconds=0,
                    root_batch_window_ms=0,
                )
            )
            try:
                outputs = engine.generate(
                    [GenerationInput(9501, input_ids, 5)]
                )
                self.assertEqual(list(outputs[0].token_ids), expected)
                self.assertEqual(
                    engine._require_runner().loader,
                    "native-gdlp-gguf-stage",
                )
            finally:
                engine.close()

    def test_qwen3_native_stages_match_monolithic_greedy_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            config = Qwen3Config(
                vocab_size=32,
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=2,
                max_position_embeddings=64,
                tie_word_embeddings=False,
            )
            config.architectures = ["Qwen3ForCausalLM"]
            torch.manual_seed(839)
            monolithic = Qwen3ForCausalLM(config).eval()
            monolithic.save_pretrained(checkpoint, safe_serialization=True)
            source = root / "qwen3-f32.gguf"
            _write_complete_qwen3_gguf(
                source,
                state=monolithic.state_dict(),
                config=config,
            )
            packages = tuple(
                build_native_gguf_stage(
                    source,
                    root / f"stage-{start}-{end}",
                    config_source=checkpoint / "config.json",
                    layer_start=start,
                    layer_end=end,
                    model_source="hf://fixture/tiny-qwen3",
                    model_revision="d" * 40,
                )
                for start, end in ((0, 1), (1, 2))
            )
            classic = tuple(
                StageRunner(
                    StageModelSpec(str(checkpoint), start, end, 2, 1),
                    device="cpu",
                )
                for start, end in ((0, 1), (1, 2))
            )
            native = tuple(
                _native_runner_from_stage_cli(package)
                for package in packages
            )
            try:
                input_ids = torch.tensor([[2, 5, 13, 8]], dtype=torch.long)
                expected = _monolithic_greedy(monolithic, input_ids, steps=5)
                self.assertEqual(
                    _two_stage_greedy(classic, input_ids, steps=5),
                    expected,
                )
                self.assertEqual(
                    _two_stage_greedy(native, input_ids, steps=5),
                    expected,
                )
            finally:
                for runner in (*classic, *native):
                    runner.close()


def _write_config(root: Path, *, layers: int, hidden: int) -> Path:
    path = root / f"config-{layers}.json"
    path.write_text(
        json.dumps(
            {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "num_hidden_layers": layers,
                "hidden_size": hidden,
                "intermediate_size": hidden * 2,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "vocab_size": 8,
                "max_position_embeddings": 128,
                "rms_norm_eps": 1e-5,
                "rope_theta": 10_000,
                "tie_word_embeddings": False,
            }
        ),
        encoding="utf-8",
    )
    return path


def _native_runner_from_stage_cli(
    package: NativeGgufStagePackage,
) -> NativeGgufStageRunner:
    pipeline_id = int(
        package.artifact_identity.removeprefix("sha256:")[:16],
        16,
    )
    argv = [
        "--model",
        "sealed-native-gguf",
        "--model-artifact-identity",
        package.artifact_identity,
        "--stage-package-identity",
        package.package_identity,
        "--model-canonical-source",
        package.model_source,
        "--pipeline-snapshot-identity",
        str(pipeline_id),
        "--native-gguf-package",
        str(package.root),
        "--native-gguf-package-id",
        package.package_id,
        "--layer-start",
        str(package.layer_start),
        "--layer-end",
        str(package.layer_end),
        "--total-layers",
        str(package.total_layers),
        "--threads",
        "1",
        "--device",
        "cpu",
        "--listen-host",
        "127.0.0.1",
        "--listen-port",
        str(31_000 + package.layer_start),
        "--return-host",
        "127.0.0.1",
        "--return-port",
        "31999",
        "--codec",
        "fp32",
    ]
    if package.model_revision is not None:
        argv.extend(
            (
                "--revision",
                package.model_revision,
                "--model-canonical-revision",
                package.model_revision,
            )
        )
    if package.layer_end < package.total_layers:
        argv.extend(
            (
                "--next-host",
                "127.0.0.1",
                "--next-port",
                str(31_001 + package.layer_start),
                "--next-layer-end",
                str(package.layer_end + 1),
            )
        )
    config = build_stage_config(parse_stage_args(argv))
    validate_stage_config(config)
    runner = build_stage_runner(config)
    assert isinstance(runner, NativeGgufStageRunner)
    return runner


def _write_fixture_gguf(path: Path) -> dict[str, np.ndarray]:
    values = {
        "token_embd.weight": np.arange(32, dtype=np.float32),
        "blk.0.attn_norm.weight": np.arange(4, dtype=np.float32) + 100,
        "blk.0.attn_q.weight": np.arange(16, dtype=np.float32) + 200,
        "blk.1.attn_norm.weight": np.arange(4, dtype=np.float32) + 300,
        "blk.1.attn_q.weight": np.arange(16, dtype=np.float32) + 400,
        "output_norm.weight": np.arange(4, dtype=np.float32) + 500,
        "output.weight": np.arange(32, dtype=np.float32) + 600,
    }
    tensors = {
        "token_embd.weight": (0, (4, 8), values["token_embd.weight"].tobytes()),
        "blk.0.attn_norm.weight": (
            0,
            (4,),
            values["blk.0.attn_norm.weight"].tobytes(),
        ),
        "blk.0.attn_q.weight": (
            0,
            (4, 4),
            _llama_gguf_permute(
                values["blk.0.attn_q.weight"].reshape(4, 4),
                heads=1,
            ).tobytes(),
        ),
        "blk.1.attn_norm.weight": (
            0,
            (4,),
            values["blk.1.attn_norm.weight"].tobytes(),
        ),
        "blk.1.attn_q.weight": (
            0,
            (4, 4),
            _llama_gguf_permute(
                values["blk.1.attn_q.weight"].reshape(4, 4),
                heads=1,
            ).tobytes(),
        ),
        "output_norm.weight": (
            0,
            (4,),
            values["output_norm.weight"].tobytes(),
        ),
        "output.weight": (0, (4, 8), values["output.weight"].tobytes()),
    }
    _write_gguf(path, layers=2, hidden=4, tensors=tensors)
    return values


def _write_complete_llama_gguf(
    path: Path,
    *,
    state: dict[str, torch.Tensor],
    config: LlamaConfig,
) -> None:
    direct = {
        "model.embed_tokens.weight": "token_embd.weight",
        "model.norm.weight": "output_norm.weight",
        "lm_head.weight": "output.weight",
    }
    suffixes = {
        "input_layernorm.weight": "attn_norm.weight",
        "self_attn.q_proj.weight": "attn_q.weight",
        "self_attn.k_proj.weight": "attn_k.weight",
        "self_attn.v_proj.weight": "attn_v.weight",
        "self_attn.o_proj.weight": "attn_output.weight",
        "post_attention_layernorm.weight": "ffn_norm.weight",
        "mlp.gate_proj.weight": "ffn_gate.weight",
        "mlp.up_proj.weight": "ffn_up.weight",
        "mlp.down_proj.weight": "ffn_down.weight",
    }
    tensors: dict[str, tuple[int, tuple[int, ...], bytes]] = {}
    for name, source_value in state.items():
        gguf_name = direct.get(name)
        layer = None
        suffix = None
        if gguf_name is None and name.startswith("model.layers."):
            remainder = name.removeprefix("model.layers.")
            layer_text, separator, suffix = remainder.partition(".")
            if not separator or suffix not in suffixes:
                raise AssertionError(f"unmapped tiny Llama tensor {name}")
            layer = int(layer_text)
            gguf_name = f"blk.{layer}.{suffixes[suffix]}"
        if gguf_name is None:
            raise AssertionError(f"unmapped tiny Llama tensor {name}")
        value = source_value.detach().cpu().to(torch.float32).contiguous().numpy()
        if suffix == "self_attn.q_proj.weight":
            value = _llama_gguf_permute(
                value,
                heads=int(config.num_attention_heads),
            )
        elif suffix == "self_attn.k_proj.weight":
            value = _llama_gguf_permute(
                value,
                heads=int(config.num_key_value_heads),
            )
        tensors[gguf_name] = (
            0,
            tuple(reversed(value.shape)),
            value.tobytes(order="C"),
        )
    _write_gguf(
        path,
        layers=int(config.num_hidden_layers),
        hidden=int(config.hidden_size),
        tensors=tensors,
    )


def _write_complete_qwen3_gguf(
    path: Path,
    *,
    state: dict[str, torch.Tensor],
    config: Qwen3Config,
) -> None:
    direct = {
        "model.embed_tokens.weight": "token_embd.weight",
        "model.norm.weight": "output_norm.weight",
        "lm_head.weight": "output.weight",
    }
    suffixes = {
        "input_layernorm.weight": "attn_norm.weight",
        "self_attn.q_proj.weight": "attn_q.weight",
        "self_attn.k_proj.weight": "attn_k.weight",
        "self_attn.v_proj.weight": "attn_v.weight",
        "self_attn.o_proj.weight": "attn_output.weight",
        "self_attn.q_norm.weight": "attn_q_norm.weight",
        "self_attn.k_norm.weight": "attn_k_norm.weight",
        "post_attention_layernorm.weight": "ffn_norm.weight",
        "mlp.gate_proj.weight": "ffn_gate.weight",
        "mlp.up_proj.weight": "ffn_up.weight",
        "mlp.down_proj.weight": "ffn_down.weight",
    }
    tensors: dict[str, tuple[int, tuple[int, ...], bytes]] = {}
    for name, source_value in state.items():
        gguf_name = direct.get(name)
        if gguf_name is None and name.startswith("model.layers."):
            remainder = name.removeprefix("model.layers.")
            layer_text, separator, suffix = remainder.partition(".")
            if not separator or suffix not in suffixes:
                raise AssertionError(f"unmapped tiny Qwen3 tensor {name}")
            gguf_name = f"blk.{int(layer_text)}.{suffixes[suffix]}"
        if gguf_name is None:
            raise AssertionError(f"unmapped tiny Qwen3 tensor {name}")
        value = source_value.detach().cpu().to(torch.float32).contiguous().numpy()
        tensors[gguf_name] = (
            0,
            tuple(reversed(value.shape)),
            value.tobytes(order="C"),
        )
    _write_gguf(
        path,
        layers=int(config.num_hidden_layers),
        hidden=int(config.hidden_size),
        tensors=tensors,
        architecture="qwen3",
    )


def _monolithic_greedy(
    model: LlamaForCausalLM,
    input_ids: torch.Tensor,
    *,
    steps: int,
) -> list[int]:
    sequence = input_ids.clone()
    generated: list[int] = []
    with torch.inference_mode():
        for _ in range(steps):
            logits = model(input_ids=sequence).logits[:, -1, :]
            token = int(torch.argmax(logits, dim=-1).item())
            generated.append(token)
            sequence = torch.cat(
                (sequence, torch.tensor([[token]], dtype=torch.long)),
                dim=1,
            )
    return generated


def _two_stage_greedy(
    runners: tuple[StageRunner, StageRunner],
    input_ids: torch.Tensor,
    *,
    steps: int,
) -> list[int]:
    request_id = 9001
    first, last = runners
    first.begin(request_id)
    last.begin(request_id)
    generated: list[int] = []
    try:
        hidden = first.forward_ids(request_id, input_ids)
        for index in range(steps):
            _hidden, token = last.forward_hidden(request_id, hidden)
            assert isinstance(token, int)
            generated.append(token)
            if index + 1 < steps:
                hidden = first.forward_ids(
                    request_id,
                    torch.tensor([[token]], dtype=torch.long),
                )
        return generated
    finally:
        first.end(request_id)
        last.end(request_id)


def _llama_gguf_permute(values: np.ndarray, *, heads: int) -> np.ndarray:
    return (
        values.reshape(heads, 2, values.shape[0] // heads // 2, *values.shape[1:])
        .swapaxes(1, 2)
        .reshape(values.shape)
        .copy()
    )


def _basic_tensors(*, layers: int) -> dict[str, tuple[int, tuple[int, ...], bytes]]:
    tensors: dict[str, tuple[int, tuple[int, ...], bytes]] = {
        "token_embd.weight": (0, (4, 8), bytes(4 * 8 * 4)),
        "output_norm.weight": (0, (4,), bytes(4 * 4)),
        "output.weight": (0, (4, 8), bytes(4 * 8 * 4)),
    }
    for layer in range(layers):
        tensors[f"blk.{layer}.attn_norm.weight"] = (0, (4,), bytes(4 * 4))
        tensors[f"blk.{layer}.attn_q.weight"] = (0, (4, 4), bytes(4 * 4 * 4))
    return tensors


def _write_gguf(
    path: Path,
    *,
    layers: int,
    hidden: int,
    tensors: dict[str, tuple[int, tuple[int, ...], bytes]],
    overlap: bool = False,
    architecture: str = "llama",
    extra_metadata: tuple[tuple[str, int, object], ...] = (),
) -> None:
    metadata = (
        ("general.architecture", 8, architecture),
        ("general.alignment", 4, 32),
        (f"{architecture}.block_count", 4, layers),
        (f"{architecture}.embedding_length", 4, hidden),
        *extra_metadata,
    )
    offsets: list[int] = []
    cursor = 0
    for _name, (_ggml_type, _dimensions, payload) in tensors.items():
        cursor = (cursor + 31) // 32 * 32
        offsets.append(0 if overlap else cursor)
        cursor += len(payload)
    with path.open("wb") as output:
        output.write(b"GGUF")
        output.write(struct.pack("<IQQ", 3, len(tensors), len(metadata)))
        for key, value_type, value in metadata:
            _string(output, key)
            output.write(struct.pack("<I", value_type))
            if value_type == 8:
                _string(output, value)
            else:
                output.write(struct.pack("<I", value))
        for (name, (ggml_type, dimensions, _payload)), offset in zip(
            tensors.items(), offsets, strict=True
        ):
            _string(output, name)
            output.write(struct.pack("<I", len(dimensions)))
            for dimension in dimensions:
                output.write(struct.pack("<Q", dimension))
            output.write(struct.pack("<IQ", ggml_type, offset))
        output.write(bytes((-output.tell()) % 32))
        data_start = output.tell()
        for (_name, (_ggml_type, _dimensions, payload)), offset in zip(
            tensors.items(), offsets, strict=True
        ):
            desired = data_start + offset
            if output.tell() < desired:
                output.write(bytes(desired - output.tell()))
            output.write(payload)


def _string(output, value: str) -> None:
    encoded = value.encode("utf-8")
    output.write(struct.pack("<Q", len(encoded)))
    output.write(encoded)


def _pack_nibbles(values: np.ndarray) -> bytes:
    return bytes(
        int(values[index] & 0x0F) | (int(values[index + 16] & 0x0F) << 4)
        for index in range(16)
    )


def _q4_0_block() -> bytes:
    values = np.arange(32, dtype=np.int16) % 16 - 8
    encoded = (values + 8).astype(np.uint8)
    return struct.pack("<e", 1.0) + _pack_nibbles(encoded)


def _q4_1_block() -> bytes:
    values = np.arange(32, dtype=np.uint8) % 16
    return struct.pack("<ee", 0.5, 2.0) + _pack_nibbles(values)


def _q5_payload(values: np.ndarray) -> tuple[bytes, bytes]:
    unsigned = values.astype(np.uint8)
    qh = 0
    for index, value in enumerate(unsigned):
        qh |= int((value >> 4) & 1) << index
    packed = _pack_nibbles(unsigned)
    return struct.pack("<I", qh), packed


def _q5_0_block() -> bytes:
    high, packed = _q5_payload(np.arange(32, dtype=np.uint8))
    return struct.pack("<e", 1.0) + high + packed


def _q5_1_block() -> bytes:
    high, packed = _q5_payload(np.arange(32, dtype=np.uint8))
    return struct.pack("<ee", 0.25, 1.0) + high + packed


def _q8_0_block() -> bytes:
    values = np.arange(32, dtype=np.int16) - 16
    return struct.pack("<e", 0.5) + values.astype(np.int8).tobytes()


def _q2_k_block() -> tuple[bytes, np.ndarray]:
    scales = bytes([(3 << 4) | 2] * 16)
    quants = bytes([0xE4] * 64)
    raw = scales + quants + struct.pack("<ee", 0.5, 0.25)
    groups = np.repeat(np.array([0, 0, 1, 1, 2, 2, 3, 3]), 16)
    expected = np.tile(groups, 2).astype(np.float32) - 0.75
    return raw, expected


def _pack_q3_k_scales(encoded: list[int]) -> bytes:
    packed = np.zeros(12, dtype=np.uint8)
    for index, value in enumerate(encoded):
        if index < 8:
            packed[index] = value & 0x0F
        else:
            packed[index - 8] |= (value & 0x0F) << 4
        packed[index % 4 + 8] |= (value >> 4) << (2 * (index // 4))
    return packed.tobytes()


def _q3_k_block() -> tuple[bytes, np.ndarray]:
    raw = (
        bytes([0xFF] * 32)
        + bytes([0xE4] * 64)
        + _pack_q3_k_scales([34] * 16)
        + struct.pack("<e", 0.5)
    )
    groups = np.repeat(np.array([0, 0, 1, 1, 2, 2, 3, 3]), 16)
    expected = np.tile(groups, 2).astype(np.float32)
    return raw, expected


def _pack_k_scales(scales: list[int], minimums: list[int]) -> bytes:
    packed = np.zeros(12, dtype=np.uint8)
    for index, (scale, minimum) in enumerate(
        zip(scales, minimums, strict=True)
    ):
        if index < 4:
            packed[index] = scale
            packed[index + 4] = minimum
        else:
            packed[index + 4] = (scale & 0x0F) | ((minimum & 0x0F) << 4)
            packed[index - 4] |= (scale >> 4) << 6
            packed[index] |= (minimum >> 4) << 6
    return packed.tobytes()


def _q4_k_block() -> tuple[bytes, np.ndarray]:
    raw = (
        struct.pack("<ee", 0.5, 0.25)
        + _pack_k_scales([2] * 8, [3] * 8)
        + bytes([0x21] * 128)
    )
    expected = np.tile(
        np.concatenate(
            (
                np.full(32, 0.25, dtype=np.float32),
                np.full(32, 1.25, dtype=np.float32),
            )
        ),
        4,
    )
    return raw, expected


def _q5_k_block() -> tuple[bytes, np.ndarray]:
    raw = (
        struct.pack("<ee", 0.5, 0.25)
        + _pack_k_scales([2] * 8, [3] * 8)
        + bytes([0xFF] * 32)
        + bytes([0x21] * 128)
    )
    expected = np.tile(
        np.concatenate(
            (
                np.full(32, 16.25, dtype=np.float32),
                np.full(32, 17.25, dtype=np.float32),
            )
        ),
        4,
    )
    return raw, expected


def _q6_k_block() -> tuple[bytes, np.ndarray]:
    scales = np.arange(1, 17, dtype=np.int8)
    raw = (
        bytes([0x11] * 128)
        + bytes([0xAA] * 64)
        + scales.tobytes()
        + struct.pack("<e", 0.5)
    )
    expected = np.repeat(scales.astype(np.float32) * 0.5, 16)
    return raw, expected


def _q8_k_block() -> tuple[bytes, np.ndarray]:
    quants = np.arange(-128, 128, dtype=np.int16).astype(np.int8)
    raw = struct.pack("<f", 0.5) + quants.tobytes() + bytes(32)
    return raw, quants.astype(np.float32) * 0.5


if __name__ == "__main__":
    unittest.main()
