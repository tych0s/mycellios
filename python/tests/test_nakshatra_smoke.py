from __future__ import annotations

import os
from pathlib import Path
import unittest

import torch

from distributed_runtime.model import StageModelSpec
from distributed_runtime.nakshatra import (
    NakshatraStageRunner,
    NakshatraStageRuntimeSpec,
)
from distributed_runtime.nakshatra_package import (
    NAKSHATRA_COMMIT,
    load_nakshatra_stage_package,
)


@unittest.skipUnless(
    os.environ.get("RUN_NAKSHATRA_SMOKE_TESTS") == "1",
    "set RUN_NAKSHATRA_SMOKE_TESTS=1 for the physical Nakshatra stdio smoke",
)
class NakshatraPhysicalSmokeTests(unittest.TestCase):
    def test_real_daemon_loads_sealed_slice_decodes_and_truncates_kv(self) -> None:
        workspace = Path(__file__).resolve().parents[2]
        package_root = Path(
            os.environ.get(
                "NAKSHATRA_STAGE_PACKAGE",
                workspace
                / "runtime"
                / "packages"
                / "smollm2-135m-f16-nakshatra-l15-30",
            )
        ).resolve()
        smoke_backend = os.environ.get("NAKSHATRA_SMOKE_BACKEND", "cpu").lower()
        backend_binary = (
            workspace
            / "runtime"
            / "external"
            / "nakshatra-stage"
            / "llama.cpp"
            / f"build-gdlp-nakshatra-{smoke_backend}"
            / "bin"
            / ("llama-nakshatra-worker.exe" if os.name == "nt" else "llama-nakshatra-worker")
        )
        legacy_binary = (
            workspace
            / "runtime"
            / "external"
            / "nakshatra-stage"
            / "llama.cpp"
            / "build-gdlp-nakshatra"
            / "bin"
            / ("llama-nakshatra-worker.exe" if os.name == "nt" else "llama-nakshatra-worker")
        )
        default_binary = backend_binary if backend_binary.is_file() else legacy_binary
        daemon_binary = Path(
            os.environ.get("NAKSHATRA_DAEMON_BIN", default_binary)
        ).resolve()
        if not package_root.is_dir() or not daemon_binary.is_file():
            self.skipTest("physical Nakshatra daemon or sealed stage package is unavailable")

        package = load_nakshatra_stage_package(package_root)
        if package.layer_start == 0:
            self.skipTest("physical smoke requires a non-root Nakshatra stage package")
        runtime = NakshatraStageRuntimeSpec(
            package=str(package.root),
            daemon_command=(str(daemon_binary),),
            context_tokens=min(64, package.max_context_tokens),
            threads=int(os.environ.get("NAKSHATRA_SMOKE_THREADS", "2")),
            gpu_layers=int(os.environ.get("NAKSHATRA_SMOKE_GPU_LAYERS", "0")),
            compute_api=os.environ.get("NAKSHATRA_SMOKE_COMPUTE_API", "cpu"),
            startup_timeout_seconds=120.0,
            call_timeout_seconds=120.0,
            expected_pipeline_id=package.pipeline_id,
            expected_package_id=package.package_id,
            expected_manifest_sha256=package.manifest_sha256,
        )
        runner = NakshatraStageRunner(
            StageModelSpec(
                package.model_source,
                package.layer_start,
                package.layer_end,
                package.total_layers,
                max(runtime.threads, 1),
                package.model_revision,
            ),
            runtime,
        )
        try:
            self.assertEqual(runner.client.binary_commit, NAKSHATRA_COMMIT)
            runner.begin(0x4E4B5354)
            output, token = runner.forward_hidden(
                0x4E4B5354,
                torch.zeros((1, 1, package.hidden_size), dtype=torch.float32),
            )
            self.assertEqual(tuple(output.shape), (1, 1, package.hidden_size))
            if package.layer_end == package.total_layers:
                self.assertIsInstance(token, int)
            else:
                self.assertIsNone(token)
            self.assertEqual(runner.sequence_length(0x4E4B5354), 1)
            runner.truncate(0x4E4B5354, 0)
            self.assertEqual(runner.sequence_length(0x4E4B5354), 0)
            runner.end(0x4E4B5354)
        finally:
            runner.close()
        self.assertEqual(runner.client.process.returncode, 0)


if __name__ == "__main__":
    unittest.main()
