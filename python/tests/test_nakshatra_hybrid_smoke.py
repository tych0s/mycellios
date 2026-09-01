from __future__ import annotations

import json
import os
from pathlib import Path
import time
import unittest

import torch

from distributed_runtime.model import (
    StageModelSpec,
    StageRunner,
    load_tokenizer,
    reference_generate,
    resolve_model_snapshot,
)
from distributed_runtime.nakshatra import (
    NakshatraStageRunner,
    NakshatraStageRuntimeSpec,
)
from distributed_runtime.nakshatra_package import load_nakshatra_stage_package


@unittest.skipUnless(
    os.environ.get("RUN_NAKSHATRA_HYBRID_SMOKE_TESTS") == "1",
    "set RUN_NAKSHATRA_HYBRID_SMOKE_TESTS=1 for physical hybrid parity",
)
class NakshatraHybridPhysicalSmokeTests(unittest.TestCase):
    """Run a real PyTorch first range into a real partial-GGUF last range."""

    def test_hybrid_pipeline_matches_monolithic_greedy_tokens(self) -> None:
        workspace = Path(__file__).resolve().parents[2]
        package = load_nakshatra_stage_package(
            Path(
                os.environ.get(
                    "NAKSHATRA_STAGE_PACKAGE",
                    workspace
                    / "runtime"
                    / "packages"
                    / "smollm2-135m-f16-nakshatra-l15-30",
                )
            )
        )
        backend = os.environ.get("NAKSHATRA_SMOKE_BACKEND", "cpu").lower()
        executable = "llama-nakshatra-worker.exe" if os.name == "nt" else "llama-nakshatra-worker"
        backend_daemon = (
            workspace
            / "runtime"
            / "external"
            / "nakshatra-stage"
            / "llama.cpp"
            / f"build-gdlp-nakshatra-{backend}"
            / "bin"
            / executable
        )
        legacy_daemon = (
            workspace
            / "runtime"
            / "external"
            / "nakshatra-stage"
            / "llama.cpp"
            / "build-gdlp-nakshatra"
            / "bin"
            / executable
        )
        default_daemon = backend_daemon if backend_daemon.is_file() else legacy_daemon
        daemon = Path(
            os.environ.get(
                "NAKSHATRA_DAEMON_BIN",
                default_daemon,
            )
        ).resolve()
        if not daemon.is_file():
            self.skipTest(f"physical Nakshatra {backend} daemon is unavailable")

        model_name = os.environ.get(
            "NAKSHATRA_HYBRID_MODEL",
            "HuggingFaceTB/SmolLM2-135M-Instruct",
        )
        snapshot = resolve_model_snapshot(model_name, package.model_revision)
        tokenizer = load_tokenizer(snapshot)
        prompt = os.environ.get(
            "NAKSHATRA_HYBRID_PROMPT", "The capital of France is"
        )
        output_tokens = int(os.environ.get("NAKSHATRA_HYBRID_OUTPUT_TOKENS", "4"))
        warmups = int(os.environ.get("NAKSHATRA_HYBRID_WARMUPS", "1"))
        iterations = int(os.environ.get("NAKSHATRA_HYBRID_ITERATIONS", "3"))
        if warmups < 0 or iterations < 1:
            self.fail("hybrid warmups must be non-negative and iterations positive")
        threads = int(os.environ.get("NAKSHATRA_SMOKE_THREADS", "2"))
        gpu_layers = int(os.environ.get("NAKSHATRA_SMOKE_GPU_LAYERS", "0"))
        compute_api = os.environ.get("NAKSHATRA_SMOKE_COMPUTE_API", "cpu")
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        reference, reference_metrics = reference_generate(
            snapshot, input_ids, output_tokens, threads
        )

        first = StageRunner(
            StageModelSpec(
                snapshot,
                0,
                package.layer_start,
                package.total_layers,
                threads,
                package.model_revision,
            )
        )
        last = NakshatraStageRunner(
            StageModelSpec(
                package.model_source,
                package.layer_start,
                package.layer_end,
                package.total_layers,
                threads,
                package.model_revision,
            ),
            NakshatraStageRuntimeSpec(
                package=str(package.root),
                daemon_command=(str(daemon),),
                context_tokens=max(64, int(input_ids.shape[1]) + output_tokens),
                threads=threads,
                gpu_layers=gpu_layers,
                compute_api=compute_api,
                expected_pipeline_id=package.pipeline_id,
                expected_package_id=package.package_id,
                expected_manifest_sha256=package.manifest_sha256,
            ),
        )
        runs: list[dict[str, object]] = []
        try:
            for run_index in range(warmups + iterations):
                request_id = 0x4E4B485942 + run_index
                produced: list[int] = []
                latencies_ms: list[float] = []
                first.begin(request_id)
                last.begin(request_id)
                current = input_ids
                try:
                    for _ in range(output_tokens):
                        started = time.perf_counter()
                        hidden = first.forward_ids(request_id, current)
                        _, token = last.forward_hidden(request_id, hidden)
                        latencies_ms.append((time.perf_counter() - started) * 1_000)
                        self.assertIsInstance(token, int)
                        produced.append(int(token))
                        current = torch.tensor([[token]], dtype=torch.long)
                finally:
                    first.end(request_id)
                    last.end(request_id)
                self.assertEqual(produced, reference)
                runs.append(
                    {
                        "measured": run_index >= warmups,
                        "tokens": produced,
                        "latencyMs": latencies_ms,
                        "ttftMs": latencies_ms[0],
                        "meanDecodeMs": (
                            sum(latencies_ms[1:]) / len(latencies_ms[1:])
                            if len(latencies_ms) > 1
                            else 0.0
                        ),
                    }
                )

            if gpu_layers > 0:
                stderr = "\n".join(last.client.stderr_tail).lower()
                self.assertIn("using device vulkan", stderr)
                self.assertIn("offloaded 31/31 layers to gpu", stderr)
                self.assertIn("vulkan0 model buffer size", stderr)
        finally:
            first.close()
            last.close()

        measured = [run for run in runs if bool(run["measured"])]
        produced = list(measured[-1]["tokens"])
        ttft_values = [float(run["ttftMs"]) for run in measured]
        decode_values = [float(run["meanDecodeMs"]) for run in measured]
        result = {
            "schema": "gdlp-nakshatra-hybrid-physical-smoke/1",
            "backend": backend,
            "computeApi": compute_api,
            "gpuLayers": gpu_layers,
            "ranges": [[0, package.layer_start], [package.layer_start, package.layer_end]],
            "promptTokens": int(input_ids.shape[1]),
            "completionTokens": output_tokens,
            "warmups": warmups,
            "iterations": iterations,
            "reference": reference,
            "produced": produced,
            "exact": produced == reference,
            "runs": runs,
            "ttftMeanMs": sum(ttft_values) / len(ttft_values),
            "meanDecodeMs": sum(decode_values) / len(decode_values),
            "referenceMetrics": reference_metrics,
            "daemonCommit": last.client.binary_commit,
            "daemonStderrDeviceEvidence": [
                line
                for line in last.client.stderr_tail
                if "using device" in line.lower()
                or "offloaded" in line.lower()
                or "model buffer size" in line.lower()
            ],
        }
        print("GDLP_NAKSHATRA_HYBRID_SMOKE " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
