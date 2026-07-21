from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from distributed_runtime.physical_probe import collect_physical_probe


class _FakeCuda:
    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return 1

    def get_device_properties(self, index: int):
        self.index = index
        return SimpleNamespace(
            name="Test GPU",
            total_memory=4 * 1024**3,
            major=8,
            minor=6,
            uuid="GPU-physical-test",
        )

    def mem_get_info(self, index: int):
        return (3 * 1024**3, 4 * 1024**3)

    class nccl:
        @staticmethod
        def version():
            return (2, 25, 1)


class _FakeDistributed:
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def is_nccl_available() -> bool:
        return True


class PhysicalProbeTests(unittest.TestCase):
    def test_probe_reports_hashed_identity_gpu_memory_and_nccl(self) -> None:
        fake_torch = SimpleNamespace(
            __version__="2.test",
            version=SimpleNamespace(cuda="13.0", hip=None),
            cuda=_FakeCuda(),
            distributed=_FakeDistributed(),
        )
        with (
            patch("distributed_runtime.physical_probe.torch", fake_torch),
            patch(
                "distributed_runtime.physical_probe._machine_material",
                return_value=("test", b"machine-secret"),
            ),
        ):
            first = collect_physical_probe("campaign-nonce-0001")
            second = collect_physical_probe("campaign-nonce-0001")

        self.assertEqual(first, second)
        self.assertRegex(first["host"]["fingerprintSha256"], r"^sha256:[0-9a-f]{64}$")
        self.assertNotIn("machine-secret", str(first))
        self.assertTrue(first["runtime"]["ncclAvailable"])
        self.assertEqual(first["runtime"]["ncclVersion"], "2.25.1")
        self.assertEqual(first["devices"][0]["totalMemoryBytes"], 4 * 1024**3)
        self.assertRegex(first["devices"][0]["uuidSha256"], r"^sha256:[0-9a-f]{64}$")

    def test_probe_rejects_replay_nonce_with_unsafe_shape(self) -> None:
        for value in ("short", "contains space 123456", "x" * 129):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "nonce"):
                collect_physical_probe(value)


if __name__ == "__main__":
    unittest.main()
