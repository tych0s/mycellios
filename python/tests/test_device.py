from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from distributed_runtime.device import (
    TorchExecutionDevice,
    normalize_torch_device_request,
    resolve_torch_execution_device,
)
from distributed_runtime.engine import PipelineEngineConfig
from distributed_runtime.model import ModelArtifactReference, StageModelSpec, StageRunner
from distributed_runtime.model_adapters import SelectiveStageAdapter
from distributed_runtime.protocol import TensorCodec
from distributed_runtime.stage import (
    StageProcessConfig,
    build_stage_runner,
    execution_metric_snapshot,
)
from distributed_runtime.stage_cli import parse_args as parse_stage_args
from distributed_runtime.server import parse_args as parse_server_args


class TorchExecutionDeviceTests(unittest.TestCase):
    def test_auto_falls_back_truthfully_to_cpu(self) -> None:
        with (
            patch("distributed_runtime.device.torch.cuda.is_available", return_value=False),
            patch("distributed_runtime.device._xpu_is_usable", return_value=False),
            patch("distributed_runtime.device._mps_is_usable", return_value=False),
            patch("distributed_runtime.device.platform.processor", return_value="Test CPU"),
        ):
            selected = resolve_torch_execution_device("auto")

        self.assertEqual(selected.device, torch.device("cpu"))
        self.assertEqual(selected.kind, "cpu")
        self.assertEqual(selected.backend, "cpu")
        self.assertFalse(selected.accelerated)
        self.assertEqual(
            selected.snapshot(weight_bytes=123, precision="float32"),
            {
                "requested_device": "auto",
                "device": "cpu",
                "device_kind": "cpu",
                "backend": "cpu",
                "device_name": "Test CPU",
                "accelerated": False,
                "precision": "float32",
                "weight_bytes": 123,
                "total_memory_bytes": None,
                "allocated_bytes": None,
                "reserved_bytes": None,
                "peak_allocated_bytes": None,
            },
        )

    def test_explicit_cuda_fails_instead_of_silently_using_cpu(self) -> None:
        with patch(
            "distributed_runtime.device.torch.cuda.is_available", return_value=False
        ):
            with self.assertRaisesRegex(RuntimeError, "cannot use"):
                resolve_torch_execution_device("cuda")

        spec = StageModelSpec("never-downloaded", 0, 1, 1, 1)
        with patch(
            "distributed_runtime.device.torch.cuda.is_available", return_value=False
        ):
            with self.assertRaisesRegex(RuntimeError, "cannot use"):
                StageRunner(spec, device="cuda")

    def test_real_cuda_metadata_and_allocator_evidence_are_reported(self) -> None:
        properties = SimpleNamespace(name="GeForce Test", total_memory=8_000)
        with (
            patch("distributed_runtime.device.torch.cuda.is_available", return_value=True),
            patch("distributed_runtime.device.torch.cuda.device_count", return_value=1),
            patch(
                "distributed_runtime.device.torch.cuda.get_device_properties",
                return_value=properties,
            ),
            patch("distributed_runtime.device.torch.cuda.memory_allocated", return_value=101),
            patch("distributed_runtime.device.torch.cuda.memory_reserved", return_value=202),
            patch(
                "distributed_runtime.device.torch.cuda.max_memory_allocated",
                return_value=303,
            ),
            patch.object(torch.version, "hip", None),
        ):
            selected = resolve_torch_execution_device("cuda:0")
            snapshot = selected.snapshot(weight_bytes=77, precision="float16")

        self.assertEqual(selected.backend, "cuda")
        self.assertEqual(selected.kind, "gpu")
        self.assertTrue(selected.accelerated)
        self.assertEqual(snapshot["precision"], "float16")
        self.assertEqual(snapshot["device"], "cuda:0")
        self.assertEqual(snapshot["device_name"], "GeForce Test")
        self.assertEqual(snapshot["total_memory_bytes"], 8_000)
        self.assertEqual(snapshot["allocated_bytes"], 101)
        self.assertEqual(snapshot["reserved_bytes"], 202)
        self.assertEqual(snapshot["peak_allocated_bytes"], 303)

    def test_rocm_is_only_reported_when_torch_exposes_a_real_hip_backend(self) -> None:
        properties = SimpleNamespace(name="Radeon Test", total_memory=16_000)
        with (
            patch("distributed_runtime.device.torch.cuda.is_available", return_value=True),
            patch("distributed_runtime.device.torch.cuda.device_count", return_value=1),
            patch(
                "distributed_runtime.device.torch.cuda.get_device_properties",
                return_value=properties,
            ),
            patch.object(torch.version, "hip", "6.4"),
        ):
            selected = resolve_torch_execution_device("auto")

        self.assertEqual(selected.backend, "rocm")
        self.assertEqual(selected.device, torch.device("cuda:0"))
        self.assertTrue(selected.accelerated)

    def test_mps_reports_real_device_and_allocator_evidence(self) -> None:
        with (
            patch("distributed_runtime.device._mps_is_usable", return_value=True),
            patch("distributed_runtime.device._mps_device_name", return_value="Apple M4 GPU"),
            patch("distributed_runtime.device._mps_total_memory", return_value=12_000),
            patch("distributed_runtime.device.torch.mps.current_allocated_memory", return_value=101),
            patch("distributed_runtime.device.torch.mps.driver_allocated_memory", return_value=202),
        ):
            selected = resolve_torch_execution_device("mps")
            snapshot = selected.snapshot(weight_bytes=77, precision="float16")

        self.assertEqual(selected.backend, "mps")
        self.assertEqual(selected.device, torch.device("mps"))
        self.assertEqual(selected.name, "Apple M4 GPU")
        self.assertTrue(selected.accelerated)
        self.assertEqual(snapshot["allocated_bytes"], 101)
        self.assertEqual(snapshot["reserved_bytes"], 202)
        self.assertIsNone(snapshot["peak_allocated_bytes"])

    def test_xpu_reports_real_intel_device(self) -> None:
        fake_xpu = SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            get_device_properties=lambda _index: SimpleNamespace(name="Intel Arc Test", total_memory=9_000),
            memory_allocated=lambda _device: 111,
            memory_reserved=lambda _device: 222,
            max_memory_allocated=lambda _device: 333,
        )
        with patch.object(torch, "xpu", fake_xpu, create=True):
            selected = resolve_torch_execution_device("xpu:0")
            snapshot = selected.snapshot(weight_bytes=88, precision="float16")

        self.assertEqual(selected.backend, "xpu")
        self.assertEqual(selected.device, torch.device("xpu:0"))
        self.assertEqual(selected.name, "Intel Arc Test")
        self.assertEqual(snapshot["allocated_bytes"], 111)
        self.assertEqual(snapshot["reserved_bytes"], 222)
        self.assertEqual(snapshot["peak_allocated_bytes"], 333)

    def test_unknown_backends_are_rejected(self) -> None:
        for requested in ("", "directml", "vulkan", "cuda:-1", "cuda:x"):
            with self.subTest(requested=requested), self.assertRaises(ValueError):
                normalize_torch_device_request(requested)
        with self.assertRaises(ValueError):
            PipelineEngineConfig("model", (0, 1, 2), device="directml")


class StageDeviceWiringTests(unittest.TestCase):
    def test_accelerated_dense_runner_uses_fp16_and_rocm_batch_one(self) -> None:
        class Layer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.self_attn = SimpleNamespace(layer_idx=99)
                self.projection = nn.Linear(4, 4, bias=False)

        class Decoder(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = nn.ModuleList((Layer(),))
                self.config = SimpleNamespace()

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = Decoder()
                self.lm_head = nn.Linear(4, 8, bias=False)
                self.config = SimpleNamespace(hidden_size=4)
                self._gdlp_selective_stage_adapter = SelectiveStageAdapter(
                    adapter_id="test-adapter-v1",
                    model_type="test",
                    architectures=("TestForCausalLM",),
                    required_layer_modules=(),
                    semantic_features=(),
                )
                self._gdlp_resolved_snapshot = "test-snapshot"

        rocm = TorchExecutionDevice(
            requested="auto",
            device=torch.device("cpu"),
            kind="gpu",
            backend="rocm",
            name="Radeon Test",
            accelerated=True,
            total_memory_bytes=16_000,
        )
        artifact = ModelArtifactReference(
            identity="a" * 64,
            canonical_source="hf://test/model",
            canonical_revision=None,
            snapshot_identity=1,
        )
        with (
            patch(
                "distributed_runtime.model.resolve_torch_execution_device",
                return_value=rocm,
            ),
            patch(
                "distributed_runtime.model._load_selective_stage_model",
                return_value=Model(),
            ),
            patch(
                "distributed_runtime.model.model_artifact_reference",
                return_value=artifact,
            ),
        ):
            runner = StageRunner(StageModelSpec("test", 0, 1, 1, 1))

        self.assertEqual(runner.compute_dtype, torch.float16)
        self.assertTrue(
            all(parameter.dtype == torch.float16 for parameter in runner.base.parameters())
        )
        self.assertEqual(runner.executor_manifest.activation_dtype, "float16")
        self.assertEqual(runner.executor_manifest.max_batch_size, 1)
        self.assertNotIn(
            "physical-tensor-batching", runner.executor_manifest.features
        )
        execution = runner.execution_snapshot()
        self.assertEqual(execution["backend"], "rocm")
        self.assertEqual(execution["precision"], "float16")

    def test_dense_stage_passes_requested_device_to_runner(self) -> None:
        spec = StageModelSpec("fake", 1, 2, 2, 1)
        config = StageProcessConfig(
            spec=spec,
            pipeline_id=1,
            listen_host="127.0.0.1",
            listen_port=20_001,
            next_host=None,
            next_port=None,
            next_layer_end=None,
            return_host="127.0.0.1",
            return_port=20_002,
            codec=TensorCodec.FP16,
            one_way_delay_ms=0,
            bandwidth_mbps=0,
            device="cuda:1",
        )
        sentinel = object()
        with patch("distributed_runtime.stage.StageRunner", return_value=sentinel) as runner:
            self.assertIs(build_stage_runner(config), sentinel)
        runner.assert_called_once_with(spec, device="cuda:1")

    def test_cli_defaults_to_auto_and_accepts_an_indexed_cuda_device(self) -> None:
        base = [
            "--model",
            "fake",
            "--layer-start",
            "1",
            "--layer-end",
            "2",
            "--total-layers",
            "2",
            "--listen-port",
            "20001",
            "--return-host",
            "127.0.0.1",
            "--return-port",
            "20002",
        ]
        self.assertEqual(parse_stage_args(base).device, "auto")
        self.assertEqual(parse_stage_args([*base, "--device", "cuda:2"]).device, "cuda:2")
        self.assertEqual(parse_stage_args([*base, "--device", "mps"]).device, "mps")
        self.assertEqual(parse_stage_args([*base, "--device", "xpu:0"]).device, "xpu:0")
        self.assertEqual(parse_server_args([]).device, "auto")
        self.assertEqual(parse_server_args(["--device", "cuda:1"]).device, "cuda:1")

    def test_execution_metric_uses_effective_runtime_snapshot(self) -> None:
        runner = SimpleNamespace(
            execution_snapshot=lambda: {
                "device_kind": "gpu",
                "backend": "cuda",
                "accelerated": True,
            }
        )
        self.assertEqual(
            execution_metric_snapshot(runner),
            {
                "device_kind": "gpu",
                "backend": "cuda",
                "accelerated": True,
            },
        )


if __name__ == "__main__":
    unittest.main()
