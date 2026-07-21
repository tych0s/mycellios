from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from distributed_runtime.cell_backend import (
    CellExecutorBackend,
    MAX_CELL_ACTIVATION_BATCH_SIZE,
    rank_backend,
)


class CellExecutorBackendTests(unittest.TestCase):
    def test_reference_backend_is_cpu_float32_gloo(self) -> None:
        backend = CellExecutorBackend()
        self.assertEqual(backend.torch_device, torch.device("cpu"))
        self.assertEqual(backend.torch_dtype, torch.float32)
        backend.validate_fixture_dtype("float32")
        self.assertEqual(
            backend.memory_report(),
            {
                "device": "cpu",
                "computeDtype": "float32",
                "collectiveBackend": "gloo",
                "allocatedBytes": 0,
                "reservedBytes": 0,
                "peakAllocatedBytes": 0,
            },
        )

    def test_contract_rejects_crossed_device_backend_and_dtype(self) -> None:
        for arguments in (
            {"collective_backend": "gloo", "device": "cuda:0"},
            {"collective_backend": "gloo", "compute_dtype": "float16"},
            {"collective_backend": "nccl", "device": "cpu"},
            {"collective_backend": "nccl", "device": "cuda"},
            {"collective_backend": "mpi"},
            {"compute_dtype": "int8"},
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                CellExecutorBackend(**arguments)

    def test_nccl_plan_compiles_on_cpu_but_fails_before_ready(self) -> None:
        backend = CellExecutorBackend(
            collective_backend="nccl",
            device="cuda:0",
            compute_dtype="float16",
        )
        backend.validate_fixture_dtype("float16")
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "no CUDA/ROCm device"):
                backend.activate()

    def test_rank_backend_uses_the_ordered_device_contract(self) -> None:
        backend = rank_backend("nccl", ("cuda:3", "cuda:7"), "bfloat16", 1)
        self.assertEqual(backend.device, "cuda:7")
        self.assertEqual(backend.torch_dtype, torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "no declared device"):
            rank_backend("nccl", ("cuda:0",), "float16", 1)

    def test_collective_broadcast_moves_only_the_rank_zero_value(self) -> None:
        backend = CellExecutorBackend()
        source = torch.arange(24, dtype=torch.float16).reshape(1, 3, 8)
        with patch("torch.distributed.broadcast") as broadcast:
            result = backend.broadcast_activation(source, [1, 3, 8])
        self.assertEqual(result.dtype, torch.float32)
        torch.testing.assert_close(result, source.float(), rtol=0, atol=0)
        broadcast.assert_called_once_with(result, src=0)

    def test_collective_broadcast_accepts_one_bounded_physical_batch(self) -> None:
        backend = CellExecutorBackend()
        source = torch.arange(48, dtype=torch.float16).reshape(2, 3, 8)
        with patch("torch.distributed.broadcast") as broadcast:
            result = backend.broadcast_activation(source, [2, 3, 8])
        self.assertEqual(result.shape, (2, 3, 8))
        torch.testing.assert_close(result, source.float(), rtol=0, atol=0)
        broadcast.assert_called_once_with(result, src=0)

        with self.assertRaisesRegex(ValueError, "bounded physical batch"):
            backend.broadcast_activation(
                None,
                [MAX_CELL_ACTIVATION_BATCH_SIZE + 1, 3, 8],
            )


if __name__ == "__main__":
    unittest.main()
