from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from distributed_runtime.dense_tiering import DenseTieringConfig
from distributed_runtime.model import StageModelSpec
from distributed_runtime.native_gguf import (
    NATIVE_GGUF_STAGE_WEIGHTS,
    NativeGgufError,
    build_native_gguf_stage,
    parse_gguf,
    verify_native_gguf_stage,
)
from distributed_runtime.native_gguf_disk_tiering import (
    decide_native_gguf_disk_tiering,
)
from distributed_runtime.native_gguf_runtime import (
    NativeGgufRuntimeConfig,
    NativeGgufStageRunner,
)
from tests.test_native_gguf import _write_complete_llama_gguf


class NativeGgufDiskTieringTests(unittest.TestCase):
    def test_paged_layers_match_resident_runner_with_one_layer_peak(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            monolithic, package, spec = _three_layer_fixture(Path(temporary))
            parsed = parse_gguf(package.root / NATIVE_GGUF_STAGE_WEIGHTS)
            unconstrained = decide_native_gguf_disk_tiering(
                parsed,
                architecture=package.architecture,
                resident_dtype=torch.float32,
                config=DenseTieringConfig(),
                measured_available_bytes=2**60,
                derived_static_bytes=8,
            )
            self.assertGreater(
                unconstrained.full_resident_weight_bytes,
                unconstrained.explicit_host_working_set_upper_bound_bytes,
            )
            budget = unconstrained.explicit_host_working_set_upper_bound_bytes
            decision = decide_native_gguf_disk_tiering(
                parsed,
                architecture=package.architecture,
                resident_dtype=torch.float32,
                config=DenseTieringConfig(host_ram_budget_bytes=budget),
                measured_available_bytes=2**60,
                derived_static_bytes=8,
            )
            self.assertTrue(decision.enabled)

            runtime = NativeGgufRuntimeConfig(
                package=str(package.root),
                package_id=package.package_id,
            )
            resident = NativeGgufStageRunner(spec, runtime, device="cpu")
            with patch(
                "distributed_runtime.native_gguf_disk_tiering."
                "_AUTHENTICATION_CHUNK_BYTES",
                64,
            ):
                paged = NativeGgufStageRunner(
                    spec,
                    runtime,
                    device="cpu",
                    dense_tiering=DenseTieringConfig(
                        host_ram_budget_bytes=budget,
                    ),
                )
            try:
                input_ids = torch.tensor([[1, 7, 11, 3]], dtype=torch.long)
                expected = _greedy_monolithic(monolithic, input_ids, steps=4)
                self.assertEqual(
                    _greedy_runner(resident, input_ids, steps=4),
                    expected,
                )
                self.assertEqual(
                    _greedy_runner(paged, input_ids, steps=4),
                    expected,
                )
                snapshot = paged.execution_snapshot()
                startup = snapshot["nativeGgufStartup"]
                tier = snapshot["denseTiering"]
                self.assertEqual(
                    startup["loadMode"],
                    "authenticated-offset-layer-cache",
                )
                self.assertEqual(tier["mode"], "storage-backed-layer-cache")
                self.assertFalse(tier["prefetchEnabled"])
                self.assertFalse(tier["overlapVerified"])
                self.assertEqual(tier["maxConcurrentResidentLayers"], 1)
                self.assertEqual(tier["currentResidentLayers"], 0)
                self.assertEqual(tier["layerLoads"], 3 * 4)
                self.assertEqual(tier["layerEvictions"], 3 * 4)
                self.assertLessEqual(
                    tier["explicitHostWorkingSetUpperBoundBytes"],
                    budget,
                )
                self.assertLessEqual(tier["peakRamWeightBytes"], budget)
                self.assertEqual(tier["storageToRam"]["writes"], 0)
                self.assertIsNone(
                    tier["storageToRam"]["physicalDiskBytes"]
                )
                self.assertGreater(
                    tier["storageToRam"]["fileBytesAuthenticated"],
                    tier["storageToRam"]["artifactBytesAuthenticated"],
                )
                self.assertTrue(_all_layer_parameters_are_meta(paged))
            finally:
                resident.close()
                paged.close()

    def test_modified_authenticated_range_fails_before_layer_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _monolithic, package, spec = _three_layer_fixture(Path(temporary))
            parsed = parse_gguf(package.root / NATIVE_GGUF_STAGE_WEIGHTS)
            decision = decide_native_gguf_disk_tiering(
                parsed,
                architecture=package.architecture,
                resident_dtype=torch.float32,
                config=DenseTieringConfig(),
                measured_available_bytes=2**60,
                derived_static_bytes=8,
            )
            budget = decision.explicit_host_working_set_upper_bound_bytes
            runner = NativeGgufStageRunner(
                spec,
                NativeGgufRuntimeConfig(
                    package=str(package.root),
                    package_id=package.package_id,
                ),
                device="cpu",
                dense_tiering=DenseTieringConfig(
                    host_ram_budget_bytes=budget,
                ),
            )
            try:
                layer_tensor = next(
                    tensor
                    for tensor in parsed.tensors
                    if tensor.name.startswith("blk.0.")
                )
                path = package.root / NATIVE_GGUF_STAGE_WEIGHTS
                original_stat = path.stat()
                with path.open("r+b") as stream:
                    stream.seek(layer_tensor.data_offset)
                    original = stream.read(1)
                    stream.seek(layer_tensor.data_offset)
                    stream.write(bytes((original[0] ^ 0x01,)))
                    stream.flush()
                    os.fsync(stream.fileno())
                os.utime(
                    path,
                    ns=(
                        original_stat.st_atime_ns,
                        original_stat.st_mtime_ns,
                    ),
                )
                runner.begin(501)
                with self.assertRaisesRegex(
                    NativeGgufError,
                    "range .* changed",
                ):
                    runner.forward_ids(
                        501,
                        torch.tensor([[1, 2]], dtype=torch.long),
                    )
                self.assertTrue(_all_layer_parameters_are_meta(runner))
                tier = runner.execution_snapshot()["denseTiering"]
                self.assertEqual(tier["currentResidentLayers"], 0)
            finally:
                runner.close()

    @unittest.skipUnless(
        os.environ.get("MYCELLIOS_PHYSICAL_GGUF_DISK_TIER") == "1",
        "requires an operator-provided physical GGUF campaign",
    )
    def test_physical_campaign_is_an_explicit_external_gate(self) -> None:
        package_path = os.environ.get("MYCELLIOS_PHYSICAL_GGUF_PACKAGE")
        package_id = os.environ.get("MYCELLIOS_PHYSICAL_GGUF_PACKAGE_ID")
        budget = int(
            os.environ.get(
                "MYCELLIOS_PHYSICAL_GGUF_RAM_BUDGET_BYTES",
                "0",
            )
        )
        self.assertTrue(
            package_path and Path(package_path).is_dir(),
            "MYCELLIOS_PHYSICAL_GGUF_PACKAGE must name a sealed package",
        )
        self.assertTrue(package_id, "sealed package ID is required")
        self.assertGreater(budget, 0, "a sealed RAM budget is required")
        package = verify_native_gguf_stage(
            package_path,
            expected_package_id=package_id,
        )
        spec = StageModelSpec(
            model_name="sealed-native-gguf",
            layer_start=package.layer_start,
            layer_end=package.layer_end,
            total_layers=package.total_layers,
            threads=1,
            artifact_identity=package.artifact_identity,
            canonical_model_source=package.model_source,
            canonical_model_revision=package.model_revision,
            stage_package_identity=package.package_identity,
        )
        runner = NativeGgufStageRunner(
            spec,
            NativeGgufRuntimeConfig(
                package=str(package.root),
                package_id=package.package_id,
            ),
            device=os.environ.get(
                "MYCELLIOS_PHYSICAL_GGUF_DEVICE",
                "auto",
            ),
            dense_tiering=DenseTieringConfig(
                host_ram_budget_bytes=budget,
            ),
        )
        try:
            self.assertEqual(
                runner.execution_snapshot()["denseTiering"]["mode"],
                "storage-backed-layer-cache",
            )
            runner.begin(99001)
            if spec.first:
                runner.forward_ids(
                    99001,
                    torch.tensor([[0]], dtype=torch.long),
                )
            else:
                runner.forward_hidden(
                    99001,
                    torch.zeros((1, 1, runner.hidden_size)),
                    token_mode="none",
                )
            self.assertTrue(_all_layer_parameters_are_meta(runner))
        finally:
            runner.close()


def _three_layer_fixture(root: Path):
    checkpoint = root / "checkpoint"
    checkpoint.mkdir()
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=2,
        max_position_embeddings=64,
        tie_word_embeddings=False,
        attention_bias=False,
        mlp_bias=False,
    )
    config.architectures = ["LlamaForCausalLM"]
    torch.manual_seed(1849)
    monolithic = LlamaForCausalLM(config).eval()
    monolithic.save_pretrained(checkpoint, safe_serialization=True)
    source = root / "model.gguf"
    _write_complete_llama_gguf(
        source,
        state=monolithic.state_dict(),
        config=config,
    )
    package = build_native_gguf_stage(
        source,
        root / "stage",
        config_source=checkpoint / "config.json",
        layer_start=0,
        layer_end=3,
        model_source="hf://fixture/tiny-llama-disk-tier",
        model_revision="e" * 40,
    )
    spec = StageModelSpec(
        model_name="sealed-native-gguf",
        layer_start=0,
        layer_end=3,
        total_layers=3,
        threads=1,
        artifact_identity=package.artifact_identity,
        canonical_model_source=package.model_source,
        canonical_model_revision=package.model_revision,
        stage_package_identity=package.package_identity,
    )
    return monolithic, package, spec


def _greedy_monolithic(
    model: LlamaForCausalLM,
    input_ids: torch.Tensor,
    *,
    steps: int,
) -> list[int]:
    generated: list[int] = []
    current = input_ids
    cache = None
    with torch.inference_mode():
        for _ in range(steps):
            output = model(
                input_ids=current,
                past_key_values=cache,
                use_cache=True,
            )
            cache = output.past_key_values
            token = int(
                torch.argmax(output.logits[:, -1:, :], dim=-1).item()
            )
            generated.append(token)
            current = torch.tensor([[token]], dtype=torch.long)
    return generated


def _greedy_runner(
    runner: NativeGgufStageRunner,
    input_ids: torch.Tensor,
    *,
    steps: int,
) -> list[int]:
    request_id = 7100 + id(runner) % 1000
    runner.begin(request_id)
    generated: list[int] = []
    current = input_ids
    try:
        for _ in range(steps):
            hidden = runner.forward_ids(request_id, current)
            assert runner.head is not None
            with torch.inference_mode():
                token = int(
                    torch.argmax(
                        runner.head(hidden[:, -1:, :]),
                        dim=-1,
                    ).item()
                )
            generated.append(token)
            current = torch.tensor([[token]], dtype=torch.long)
    finally:
        runner.end(request_id)
    return generated


def _all_layer_parameters_are_meta(
    runner: NativeGgufStageRunner,
) -> bool:
    return all(
        parameter.device.type == "meta"
        for layer in runner.base.layers
        for parameter in layer.parameters()
    )
