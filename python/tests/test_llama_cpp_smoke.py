from __future__ import annotations

import os
from pathlib import Path
import unittest

from distributed_runtime.llama_cpp import (
    benchmark_llama_cpp,
    build_whole_model_deployment,
    inspect_gguf,
    parse_llama_cpp_deployment,
    probe_llama_cpp,
)


@unittest.skipUnless(
    os.environ.get("RUN_LLAMA_CPP_SMOKE_TESTS") == "1",
    "set RUN_LLAMA_CPP_SMOKE_TESTS=1 for the local llama.cpp/Vulkan smoke",
)
class LlamaCppPhysicalSmokeTests(unittest.TestCase):
    def test_local_vulkan_runtime_qwen_gguf_and_json_benchmark(self) -> None:
        workspace = Path(__file__).resolve().parents[2]
        runtime_root = Path(
            os.environ.get(
                "LLAMA_CPP_RUNTIME_DIR",
                workspace / "runtime" / "llama-b10068-vulkan",
            )
        ).resolve()
        model_path = Path(
            os.environ.get(
                "LLAMA_CPP_GGUF",
                workspace / "runtime" / "models" / "Qwen3-0.6B-Q8_0.gguf",
            )
        ).resolve()
        if not runtime_root.is_dir() or not model_path.is_file():
            self.skipTest("local llama.cpp runtime or Qwen3 GGUF is unavailable")

        runtime = probe_llama_cpp(runtime_root)
        artifact = inspect_gguf(model_path)
        self.assertGreater(runtime.build_number, 0)
        self.assertTrue(runtime.devices)
        self.assertFalse(runtime.native_partial_stage_abi)
        self.assertEqual(artifact.storage_layout, "single-file")
        self.assertGreater(artifact.block_count, 0)
        benchmark = benchmark_llama_cpp(
            runtime_root,
            runtime,
            artifact,
            model_path,
            prompt_tokens=1,
            generation_tokens=1,
            repetitions=1,
            no_warmup=True,
            timeout_seconds=120.0,
        )
        deployment = build_whole_model_deployment(
            runtime,
            artifact,
            benchmark=benchmark,
        )
        self.assertEqual(parse_llama_cpp_deployment(deployment.to_document()), deployment)
        self.assertEqual(deployment.mode, "whole-model")
        self.assertIsNone(deployment.stage_executor)


if __name__ == "__main__":
    unittest.main()
