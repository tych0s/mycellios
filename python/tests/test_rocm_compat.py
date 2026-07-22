from __future__ import annotations

import types
import unittest

from distributed_runtime.rocm_compat import install_windows_rocm_transformers_compat


class WindowsRocmTransformersCompatTests(unittest.TestCase):
    def test_cpu_and_cuda_builds_are_untouched(self) -> None:
        registry: dict[str, object] = {}
        torch_module = types.SimpleNamespace(version=types.SimpleNamespace(hip=None))
        self.assertFalse(
            install_windows_rocm_transformers_compat(
                torch_module,
                distributed_c10d_available=False,
                module_registry=registry,
            )
        )
        self.assertEqual(registry, {})

    def test_missing_rocm_c10d_installs_explicit_no_fsdp_adapter(self) -> None:
        registry: dict[str, object] = {}
        torch_module = types.SimpleNamespace(
            version=types.SimpleNamespace(hip="7.2.53211-158bd99533"),
            distributed=types.SimpleNamespace(is_available=lambda: True),
        )
        self.assertTrue(
            install_windows_rocm_transformers_compat(
                torch_module,
                distributed_c10d_available=False,
                module_registry=registry,
            )
        )
        adapter = registry["transformers.distributed.fsdp"]
        sharding = registry["transformers.distributed.sharding_utils"]
        self.assertFalse(torch_module.distributed.is_available())
        self.assertFalse(adapter.is_fsdp_enabled())
        self.assertFalse(adapter.is_fsdp_managed_module(object()))
        adapter.verify_fsdp_plan([], None)
        with self.assertRaisesRegex(RuntimeError, "ROCm runtime for Windows"):
            adapter.verify_fsdp_plan(["model.layers.0"], {"model.layers.*": "shard"})
        with self.assertRaisesRegex(RuntimeError, "ROCm runtime for Windows"):
            sharding.DtensorShardOperation(object())

    def test_real_distributed_rocm_build_is_untouched(self) -> None:
        registry: dict[str, object] = {}
        torch_module = types.SimpleNamespace(version=types.SimpleNamespace(hip="7.2"))
        self.assertFalse(
            install_windows_rocm_transformers_compat(
                torch_module,
                distributed_c10d_available=True,
                module_registry=registry,
            )
        )
        self.assertEqual(registry, {})


if __name__ == "__main__":
    unittest.main()
