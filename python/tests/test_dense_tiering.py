from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

import torch
from torch import nn

from distributed_runtime.dense_tiering import (
    BoundedLayerResidency,
    DenseMemoryUnit,
    DenseTieringConfig,
    DenseTieringError,
    configure_dense_tiering,
    unique_tensor_bytes,
)
from distributed_runtime.device import TorchExecutionDevice
from distributed_runtime.device import resolve_torch_execution_device


class DenseTieringPolicyTests(unittest.TestCase):
    def test_lru_is_byte_bounded_deterministic_and_counts_real_moves(self) -> None:
        modules = (nn.Linear(1, 1), nn.Linear(1, 1), nn.Linear(1, 1))
        units = tuple(
            DenseMemoryUnit(key, module, size)
            for key, module, size in zip(
                ("layer-0", "layer-1", "layer-2"),
                modules,
                (4, 5, 6),
                strict=True,
            )
        )
        moves: list[tuple[str, bool]] = []
        ticks = iter(range(0, 10_000, 10))

        def move(unit: DenseMemoryUnit, resident: bool) -> int:
            moves.append((unit.key, resident))
            return unit.bytes if resident else 0

        cache = BoundedLayerResidency(
            units,
            capacity_bytes=10,
            move=move,
            clock_ns=lambda: next(ticks),
        )

        cache.admit_next("layer-0")
        cache.acquire("layer-0")
        cache.admit_next("layer-1")
        cache.acquire("layer-1")
        cache.admit_next("layer-2")
        cache.acquire("layer-2")

        snapshot = cache.snapshot()
        self.assertEqual(snapshot["cacheResidentUnits"], ["layer-2"])
        self.assertEqual(snapshot["cacheResidentBytes"], 6)
        self.assertLessEqual(
            snapshot["cachePeakResidentBytes"],
            snapshot["cacheCapacityBytes"],
        )
        self.assertEqual(snapshot["cacheHits"], 3)
        self.assertEqual(snapshot["cacheMisses"], 0)
        self.assertEqual(snapshot["nextLayerAdmissionHits"], 3)
        self.assertEqual(snapshot["evictions"], 2)
        self.assertEqual(snapshot["ramToVramBytes"], 15)
        self.assertEqual(snapshot["vramToRamBytes"], 0)
        self.assertEqual(snapshot["vramTensorBytesReleased"], 9)
        self.assertEqual(
            moves,
            [
                ("layer-0", True),
                ("layer-1", True),
                ("layer-0", False),
                ("layer-1", False),
                ("layer-2", True),
            ],
        )

    def test_cache_rejects_a_layer_before_attempting_any_move(self) -> None:
        moves: list[tuple[str, bool]] = []
        with self.assertRaisesRegex(
            DenseTieringError,
            "largest_layer_exceeds",
        ):
            BoundedLayerResidency(
                (DenseMemoryUnit("layer-0", nn.Linear(1, 1), 11),),
                capacity_bytes=10,
                move=lambda unit, resident: moves.append((unit.key, resident)),
            )
        self.assertEqual(moves, [])

    def test_cpu_residency_reports_exact_live_tensor_bytes(self) -> None:
        model = _TinyDenseModel()
        expected = unique_tensor_bytes(model)
        cpu = TorchExecutionDevice(
            requested="cpu",
            device=torch.device("cpu"),
            kind="cpu",
            backend="cpu",
            name="Test CPU",
            accelerated=False,
            total_memory_bytes=None,
        )
        with patch(
            "distributed_runtime.dense_tiering.available_host_memory_bytes",
            return_value=expected * 2,
        ):
            runtime = configure_dense_tiering(
                model,
                execution_device=cpu,
                compute_dtype=torch.float32,
                config=DenseTieringConfig(host_ram_budget_bytes=expected),
                storage_evidence={
                    "schema": "mycellios-storage-to-ram/1",
                    "format": "safetensors",
                    "artifactBytesMaterialized": expected,
                },
            )

        snapshot = runtime.snapshot()
        self.assertEqual(snapshot["mode"], "cpu-resident")
        self.assertEqual(
            snapshot["hostBudgetSource"],
            "sealed-clamped-to-os-available-plus-loaded-stage",
        )
        self.assertEqual(snapshot["totalWeightBytes"], expected)
        self.assertEqual(snapshot["hostResidentWeightBytes"], expected)
        self.assertEqual(
            snapshot["storageToRam"]["artifactBytesMaterialized"],
            expected,
        )
        self.assertIsNone(snapshot["vramBudgetBytes"])

    def test_host_budget_fails_closed_before_device_placement(self) -> None:
        model = _TinyDenseModel()
        cpu = TorchExecutionDevice(
            requested="cpu",
            device=torch.device("cpu"),
            kind="cpu",
            backend="cpu",
            name="Test CPU",
            accelerated=False,
            total_memory_bytes=None,
        )
        with self.assertRaisesRegex(
            DenseTieringError,
            "exceeds_host_ram_budget",
        ):
            configure_dense_tiering(
                model,
                execution_device=cpu,
                compute_dtype=torch.float32,
                config=DenseTieringConfig(host_ram_budget_bytes=1),
            )

    def test_sealed_vram_budget_still_requires_measured_free_memory(self) -> None:
        model = _TinyDenseModel()
        weights = unique_tensor_bytes(model, dtype=torch.float16)
        accelerator = TorchExecutionDevice(
            requested="cuda:0",
            device=torch.device("cuda:0"),
            kind="gpu",
            backend="cuda",
            name="Unmeasurable GPU",
            accelerated=True,
            total_memory_bytes=weights * 4,
        )
        with (
            patch(
                "distributed_runtime.dense_tiering.accelerator_free_memory_bytes",
                return_value=(None, "unavailable"),
            ),
            patch.object(model, "to", return_value=model) as whole_model_move,
            self.assertRaisesRegex(
                DenseTieringError,
                "cannot_measure_accelerator_free_memory",
            ),
        ):
            configure_dense_tiering(
                model,
                execution_device=accelerator,
                compute_dtype=torch.float16,
                config=DenseTieringConfig(
                    host_ram_budget_bytes=unique_tensor_bytes(model),
                    vram_budget_bytes=weights * 2,
                    activation_reserve_bytes=0,
                ),
            )

        whole_model_move.assert_not_called()

    def test_all_fit_fast_path_has_no_layer_cache_or_evictions(self) -> None:
        model = _TinyDenseModel()
        weights = unique_tensor_bytes(model, dtype=torch.float16)
        accelerator = TorchExecutionDevice(
            requested="cuda:0",
            device=torch.device("cuda:0"),
            kind="gpu",
            backend="cuda",
            name="Budgeted GPU",
            accelerated=True,
            total_memory_bytes=weights * 4,
        )
        with (
            patch(
                "distributed_runtime.dense_tiering.accelerator_free_memory_bytes",
                return_value=(weights * 2, "test-measured-free"),
            ),
            patch.object(model, "to", return_value=model) as whole_model_move,
        ):
            runtime = configure_dense_tiering(
                model,
                execution_device=accelerator,
                compute_dtype=torch.float16,
                config=DenseTieringConfig(
                    host_ram_budget_bytes=unique_tensor_bytes(model),
                    vram_budget_bytes=weights * 2,
                    activation_reserve_bytes=0,
                ),
            )

        whole_model_move.assert_called_once_with(
            device=torch.device("cuda:0"),
            dtype=torch.float16,
        )
        snapshot = runtime.snapshot()
        self.assertEqual(snapshot["mode"], "full-resident")
        self.assertEqual(snapshot["currentVramWeightBytes"], weights)
        self.assertEqual(snapshot["peakVramWeightBytes"], weights)
        self.assertEqual(snapshot["startupRamToVramBytes"], weights)
        self.assertEqual(snapshot["ramToVramBytes"], weights)
        self.assertNotIn("cacheHits", snapshot)
        self.assertNotIn("cacheMisses", snapshot)
        self.assertNotIn("evictions", snapshot)

    @unittest.skipUnless(torch.cuda.is_available(), "requires a physical CUDA GPU")
    def test_physical_cuda_forward_uses_bounded_layer_residency(self) -> None:
        model = _TinyDenseModel().eval()
        reference = copy.deepcopy(model).eval()
        layers = tuple(model.model.layers)
        layer_ids = {
            id(tensor)
            for layer in layers
            for tensor in (*layer.parameters(), *layer.buffers())
        }
        static_bytes = unique_tensor_bytes(
            model,
            dtype=torch.float16,
            excluded_tensor_ids=layer_ids,
        )
        largest_layer = max(
            unique_tensor_bytes(layer, dtype=torch.float16)
            for layer in layers
        )
        runtime = configure_dense_tiering(
            model,
            execution_device=resolve_torch_execution_device("cuda:0"),
            compute_dtype=torch.float16,
                config=DenseTieringConfig(
                    host_ram_budget_bytes=unique_tensor_bytes(model) + largest_layer,
                    vram_budget_bytes=static_bytes + largest_layer,
                    activation_reserve_bytes=0,
                    admit_next_layer=True,
                ),
        )
        hidden = torch.randn((1, 3, 4))
        expected = hidden
        with torch.no_grad():
            for layer in reference.model.layers:
                expected = layer(expected)[0]
            actual = hidden
            for layer in model.model.layers:
                actual = layer(actual)[0]
        self.assertTrue(
            torch.allclose(actual.cpu().float(), expected, atol=2e-3, rtol=2e-3)
        )
        snapshot = runtime.snapshot()
        self.assertEqual(snapshot["mode"], "bounded-layer-cache")
        self.assertEqual(snapshot["cacheHits"], len(layers))
        self.assertGreater(snapshot["ramToVramBytes"], 0)
        self.assertGreater(snapshot["layerRamToVramBytes"], 0)
        self.assertEqual(snapshot["vramToRamBytes"], 0)
        self.assertGreater(snapshot["vramTensorBytesReleased"], 0)
        self.assertLessEqual(
            snapshot["currentVramWeightBytes"],
            snapshot["vramBudgetBytes"],
        )
        runtime.close()


class _TinyLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(4, 4, bias=False)

    def forward(self, hidden_states: torch.Tensor, **_kwargs):
        return (self.projection(hidden_states),)


class _TinyDenseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        decoder = nn.Module()
        decoder.layers = nn.ModuleList((_TinyLayer(), _TinyLayer()))
        decoder.embed_tokens = nn.Embedding(8, 4)
        self.model = decoder
        self.lm_head = nn.Linear(4, 8, bias=False)


if __name__ == "__main__":
    unittest.main()
