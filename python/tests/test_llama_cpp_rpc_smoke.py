from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import time
import unittest

from distributed_runtime.llama_cpp import inspect_gguf
from distributed_runtime.llama_cpp_rpc import (
    LlamaCppRpcWorkerSpec,
    build_rpc_benchmark_argv,
    build_rpc_cell_manifest,
    build_rpc_worker_manifest,
    certify_gguf_support,
    launch_rpc_worker,
    probe_llama_cpp_rpc,
    probe_rpc_topology,
    wait_rpc_worker_ready,
)


@unittest.skipUnless(
    os.environ.get("RUN_LLAMA_CPP_RPC_SMOKE_TESTS") == "1",
    "set RUN_LLAMA_CPP_RPC_SMOKE_TESTS=1 for the physical llama.cpp RPC gate",
)
class LlamaCppRpcPhysicalSmokeTests(unittest.TestCase):
    def test_loopback_vulkan_worker_executes_the_whole_gguf_and_matches_direct(self) -> None:
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
            self.skipTest("local llama.cpp RPC runtime or Qwen3 GGUF is unavailable")

        runtime = probe_llama_cpp_rpc(runtime_root)
        requested_device = os.environ.get("LLAMA_CPP_RPC_WORKER_DEVICE")
        available = {device.identifier: device for device in runtime.runtime.devices}
        if requested_device is None:
            requested_device = next(
                (
                    device.identifier
                    for device in runtime.runtime.devices
                    if device.backend.lower() != "cpu"
                ),
                None,
            )
        if requested_device is None or requested_device not in available:
            self.skipTest("no physical accelerator device is available to the RPC worker")

        prompt_tokens = int(os.environ.get("LLAMA_CPP_RPC_PROMPT_TOKENS", "8"))
        generation_tokens = int(os.environ.get("LLAMA_CPP_RPC_GENERATION_TOKENS", "4"))
        repetitions = int(os.environ.get("LLAMA_CPP_RPC_REPETITIONS", "1"))
        port = _reserve_loopback_port()
        worker_manifest = build_rpc_worker_manifest(
            runtime,
            LlamaCppRpcWorkerSpec(
                bind_host="127.0.0.1",
                advertise_host="127.0.0.1",
                port=port,
                devices=(requested_device,),
                threads=max(1, int(os.environ.get("LLAMA_CPP_RPC_WORKER_THREADS", "2"))),
                cache=False,
            ),
        )
        worker = launch_rpc_worker(runtime_root, worker_manifest)
        started_at = time.perf_counter()
        worker_stderr = ""
        try:
            wait_rpc_worker_ready(worker, worker_manifest.endpoint, timeout_seconds=30)
            topology = probe_rpc_topology(
                runtime_root,
                runtime,
                (worker_manifest.endpoint,),
                timeout_seconds=30,
            )
            artifact = inspect_gguf(model_path)
            direct = certify_gguf_support(
                runtime_root,
                runtime,
                artifact,
                model_path,
                prompt_tokens=prompt_tokens,
                generation_tokens=generation_tokens,
                device=requested_device,
                timeout_seconds=120,
            )
            cell = build_rpc_cell_manifest(
                runtime,
                artifact,
                direct,
                (worker_manifest,),
                topology,
                split_mode="layer",
                gpu_layers=99,
            )
            rpc_benchmark = _run(
                build_rpc_benchmark_argv(
                    runtime_root,
                    cell,
                    model_path,
                    prompt_tokens=prompt_tokens,
                    generation_tokens=generation_tokens,
                    repetitions=repetitions,
                    no_warmup=True,
                ),
                runtime_root,
                120,
            )
            rpc_records = json.loads(rpc_benchmark.stdout)
            self.assertIsInstance(rpc_records, list)
            self.assertTrue(rpc_records)

            direct_prompt = _record(direct.records, prompt=prompt_tokens)
            direct_decode = _record(direct.records, generation=generation_tokens)
            rpc_prompt = _record(rpc_records, prompt=prompt_tokens)
            rpc_decode = _record(rpc_records, generation=generation_tokens)
            result = {
                "schema": "gdlp-llama-cpp-rpc-physical-smoke/1",
                "runtime": runtime.engine_version,
                "rpcRuntimeId": runtime.rpc_runtime_id,
                "workerId": worker_manifest.worker_id,
                "cellId": cell.cell_id,
                "endpoint": worker_manifest.endpoint,
                "workerDevice": requested_device,
                "rpcDevices": list(cell.rpc_devices),
                "modelSha256": artifact.file_sha256,
                "modelBytes": artifact.size_bytes,
                "splitMode": "layer",
                "graphScope": "whole-model-graph-split",
                "nativePartialStageAbi": False,
                "promptTokens": prompt_tokens,
                "generationTokens": generation_tokens,
                "directPromptTokensPerSecond": float(direct_prompt["avg_ts"]),
                "rpcPromptTokensPerSecond": float(rpc_prompt["avg_ts"]),
                "directDecodeTokensPerSecond": float(direct_decode["avg_ts"]),
                "rpcDecodeTokensPerSecond": float(rpc_decode["avg_ts"]),
                "directAndRpcGraphsExecuted": True,
                "wallMs": (time.perf_counter() - started_at) * 1_000,
            }
            print("GDLP_LLAMA_CPP_RPC_PHYSICAL_SMOKE " + json.dumps(result, sort_keys=True))
        finally:
            worker.close(timeout_seconds=10)
            worker_stderr = worker.stderr_tail
        self.assertIsNotNone(worker.process.returncode)
        self.assertIn(available[requested_device].name.lower(), worker_stderr.lower())


def _reserve_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _run(arguments: tuple[str, ...], cwd: Path, timeout: float) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        list(arguments),
        cwd=str(cwd),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed with {result.returncode}: {arguments!r}\n{result.stderr[-4000:]}"
        )
    return result


def _record(
    records: object,
    *,
    prompt: int | None = None,
    generation: int | None = None,
) -> dict[str, object]:
    if not isinstance(records, (list, tuple)):
        raise AssertionError("llama-bench records must be a sequence")
    for value in records:
        if not isinstance(value, dict):
            continue
        if prompt is not None and value.get("n_prompt") == prompt and value.get("n_gen") == 0:
            return value
        if generation is not None and value.get("n_gen") == generation and value.get("n_prompt") == 0:
            return value
    raise AssertionError("requested llama-bench record is missing")


if __name__ == "__main__":
    unittest.main()
