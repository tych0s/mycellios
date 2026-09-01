from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

from distributed_runtime.llama_cpp import CommandResult, inspect_gguf
from distributed_runtime.llama_cpp_rpc import (
    RPC_GRAPH_SCOPE,
    RPC_TOPOLOGY_EVIDENCE_SCOPE,
    RPC_TOPOLOGY_EXECUTION_VALIDATION,
    LaunchedLlamaCppRpcProcess,
    LlamaCppRpcWorkerSpec,
    _default_command_runner,
    build_rpc_benchmark_argv,
    build_rpc_cell_manifest,
    build_rpc_completion_argv,
    build_rpc_worker_manifest,
    certify_gguf_support,
    launch_rpc_client,
    launch_rpc_worker,
    parse_rpc_cell_manifest,
    parse_rpc_worker_manifest,
    probe_llama_cpp_rpc,
    probe_rpc_topology,
)


class _QueueRunner:
    def __init__(self, *results: CommandResult) -> None:
        self.results = list(results)
        self.calls: list[tuple[tuple[str, ...], Path, float]] = []

    def __call__(
        self, arguments: tuple[str, ...], cwd: Path, timeout_seconds: float
    ) -> CommandResult:
        self.calls.append((tuple(arguments), cwd, timeout_seconds))
        if not self.results:
            raise AssertionError("unexpected command")
        return self.results.pop(0)


class _FakeProcess:
    def __init__(self) -> None:
        self.returncode = None
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


class LlamaCppRpcRuntimeTests(unittest.TestCase):
    def test_probe_hashes_exact_rpc_components_and_parses_version_devices(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_runtime(root)
            runner = _runtime_runner()

            probe = probe_llama_cpp_rpc(root, runner=runner)

            self.assertEqual(probe.engine_version, "b10068-571d0d540")
            self.assertEqual(probe.runtime.devices[0].identifier, "Vulkan0")
            self.assertEqual(probe.server_sha256, _sha256(root / "ggml-rpc-server.exe"))
            self.assertEqual(probe.library_sha256, _sha256(root / "ggml-rpc.dll"))
            self.assertEqual(len(probe.rpc_runtime_id), 32)
            self.assertEqual(runner.calls[0][0][1:], ("--version",))
            self.assertEqual(runner.calls[1][0][1:], ("--list-devices",))
            self.assertEqual(runner.calls[2][0][1:], ("--help",))
            capabilities = probe.to_document()["capabilities"]
            self.assertTrue(capabilities["wholeModelGraphSplit"])
            self.assertFalse(capabilities["nativePartialStageAbi"])
            self.assertFalse(capabilities["partialLayerStage"])

    def test_default_runner_uses_argv_and_never_a_command_shell(self) -> None:
        completed = type(
            "Completed",
            (),
            {"returncode": 0, "stdout": "ok", "stderr": ""},
        )()
        with patch(
            "distributed_runtime.llama_cpp_rpc.subprocess.run",
            return_value=completed,
        ) as run:
            result = _default_command_runner(
                ("ggml-rpc-server", "--help"), Path("."), 3.0
            )
        self.assertEqual(result.stdout, "ok")
        positional, keywords = run.call_args
        self.assertEqual(positional[0], ["ggml-rpc-server", "--help"])
        self.assertIs(keywords["shell"], False)
        self.assertIs(keywords["check"], False)

    def test_probe_and_worker_configuration_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_runtime(root, include_rpc_library=False)
            with self.assertRaisesRegex(FileNotFoundError, "RPC dynamic library"):
                probe_llama_cpp_rpc(root, runner=_runtime_runner())

        with self.assertRaisesRegex(ValueError, "private LAN"):
            LlamaCppRpcWorkerSpec(
                "8.8.8.8", "8.8.8.8", 50052, ("Vulkan0",)
            )

    def test_process_drains_large_output_and_closes_pipes_after_exit(self) -> None:
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import sys; "
                "sys.stdout.write('A' * 262144); sys.stdout.flush(); "
                "sys.stderr.write('B' * 262144); sys.stderr.flush()",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=False,
        )
        launched = LaunchedLlamaCppRpcProcess((sys.executable,), process)

        self.assertEqual(process.wait(timeout=10), 0)
        launched.close(timeout_seconds=2)

        self.assertTrue(process.stdout.closed)
        self.assertTrue(process.stderr.closed)
        self.assertLessEqual(len(launched.stdout_tail), 64 * 1024)
        self.assertLessEqual(len(launched.stderr_tail), 64 * 1024)
        self.assertTrue(launched.stdout_tail.endswith("A" * 64))
        self.assertTrue(launched.stderr_tail.endswith("B" * 64))


class LlamaCppRpcManifestAndLauncherTests(unittest.TestCase):
    def test_worker_manifest_seals_argv_devices_hashes_and_shell_false(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_runtime(root)
            runtime = probe_llama_cpp_rpc(root, runner=_runtime_runner())
            spec = LlamaCppRpcWorkerSpec(
                bind_host="127.0.0.1",
                advertise_host="127.0.0.1",
                port=51052,
                devices=("Vulkan0",),
                threads=3,
                cache=True,
            )
            manifest = build_rpc_worker_manifest(runtime, spec)
            self.assertEqual(parse_rpc_worker_manifest(manifest.to_document()), manifest)
            document = manifest.to_document()
            self.assertEqual(document["execution"]["graphScope"], RPC_GRAPH_SCOPE)
            self.assertIsNone(document.get("stageExecutor"))
            self.assertEqual(
                document["launch"]["arguments"],
                [
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "51052",
                    "--device",
                    "Vulkan0",
                    "--threads",
                    "3",
                    "--cache",
                ],
            )
            self.assertIs(document["launch"]["shell"], False)

            fake_process = _FakeProcess()
            calls = []

            def fake_popen(arguments, **keywords):
                calls.append((arguments, keywords))
                return fake_process

            launched = launch_rpc_worker(
                root,
                manifest,
                runner=_runtime_runner(),
                popen_factory=fake_popen,
            )
            self.assertEqual(launched.command[1:], tuple(document["launch"]["arguments"]))
            self.assertEqual(calls[0][0], list(launched.command))
            self.assertIs(calls[0][1]["shell"], False)

            (root / "ggml-rpc.dll").write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "differs from the sealed worker"):
                launch_rpc_worker(
                    root,
                    manifest,
                    runner=_runtime_runner(),
                    popen_factory=fake_popen,
                )

    def test_whole_model_cell_requires_live_rpc_devices_and_executed_gguf(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_runtime(root)
            model = root / "qwen3.gguf"
            other_model = root / "other.gguf"
            _write_minimal_gguf(model)
            _write_minimal_gguf(other_model, name="Other Qwen")
            runtime = probe_llama_cpp_rpc(root, runner=_runtime_runner())
            artifact = inspect_gguf(model)
            evidence_runner = _QueueRunner(
                CommandResult(
                    0,
                    json.dumps(_benchmark_records(runtime.runtime, artifact)),
                    "",
                )
            )
            support = certify_gguf_support(
                root,
                runtime,
                artifact,
                model,
                device="Vulkan0",
                runner=evidence_runner,
            )
            self.assertEqual(evidence_runner.calls[0][0][-2:], ("--device", "Vulkan0"))
            worker = build_rpc_worker_manifest(
                runtime,
                LlamaCppRpcWorkerSpec(
                    "127.0.0.1", "127.0.0.1", 52052, ("Vulkan0",)
                ),
            )
            topology_runner = _QueueRunner(
                CommandResult(
                    0,
                    "Available devices:\n"
                    "  RPC0: Remote AMD 890M (8192 MiB, 7000 MiB free)\n",
                    "",
                )
            )
            topology = probe_rpc_topology(
                root,
                runtime,
                (worker.endpoint,),
                runner=topology_runner,
            )
            self.assertEqual(topology.devices[0].identifier, "RPC0")
            topology_document = topology.to_document()
            self.assertEqual(
                topology_document["evidence"]["scope"],
                RPC_TOPOLOGY_EVIDENCE_SCOPE,
            )
            self.assertEqual(
                topology_document["evidence"]["executionValidation"],
                RPC_TOPOLOGY_EXECUTION_VALIDATION,
            )
            self.assertEqual(
                topology_document["evidence"]["freeMemoryMiB"],
                "admission-snapshot-only",
            )
            self.assertEqual(
                topology_runner.calls[0][0][1:],
                ("--rpc", "127.0.0.1:52052", "--list-devices"),
            )

            cell = build_rpc_cell_manifest(
                runtime,
                artifact,
                support,
                (worker,),
                topology,
                split_mode="layer",
                gpu_layers=99,
            )
            self.assertEqual(parse_rpc_cell_manifest(cell.to_document()), cell)
            document = cell.to_document()
            self.assertEqual(document["execution"]["graphScope"], RPC_GRAPH_SCOPE)
            self.assertIsNone(document["execution"]["stageExecutor"])
            self.assertFalse(document["execution"]["nativePartialStageAbi"])
            self.assertEqual(document["modelDeployment"]["artifact"]["gguf"]["architecture"], "qwen3")

            bench_argv = build_rpc_benchmark_argv(
                root,
                cell,
                model,
                prompt_tokens=4,
                generation_tokens=2,
                runner=_execution_runner(free_memory_mib=6000),
            )
            self.assertIn("127.0.0.1:52052", bench_argv)
            self.assertIn("RPC0", bench_argv)
            self.assertEqual(bench_argv[-1], "--no-warmup")
            completion_argv = build_rpc_completion_argv(
                root,
                cell,
                model,
                prompt="hello",
                predict=3,
                seed=7,
                runner=_execution_runner(free_memory_mib=5900),
            )
            self.assertEqual(completion_argv[completion_argv.index("--temp") + 1], "0")
            self.assertEqual(
                completion_argv[completion_argv.index("--device") + 1], "RPC0"
            )

            fake_process = _FakeProcess()
            calls = []
            launched = launch_rpc_client(
                bench_argv,
                cwd=root,
                manifest_value=cell,
                runner=_execution_runner(free_memory_mib=5800),
                popen_factory=lambda arguments, **keywords: (
                    calls.append((arguments, keywords)) or fake_process
                ),
            )
            self.assertEqual(launched.command, bench_argv)
            self.assertIs(calls[0][1]["shell"], False)
            completion_calls = []
            launched_completion = launch_rpc_client(
                completion_argv,
                cwd=root,
                manifest_value=cell,
                runner=_execution_runner(free_memory_mib=5700),
                popen_factory=lambda arguments, **keywords: (
                    completion_calls.append((arguments, keywords)) or _FakeProcess()
                ),
            )
            self.assertEqual(launched_completion.command, completion_argv)
            self.assertIs(completion_calls[0][1]["shell"], False)

            sealed_alias_groups = (
                (("-rpc", "--rpc"), "127.0.0.1:52052"),
                (("-dev", "--device"), "RPC0"),
                (("-sm", "--split-mode"), "layer"),
                (("-ngl", "--n-gpu-layers", "--gpu-layers"), "99"),
                (("-m", "--model"), str(model.resolve())),
            )
            for client_kind, command in (
                ("bench", bench_argv),
                ("cli", completion_argv),
            ):
                for aliases, value in sealed_alias_groups:
                    canonical = aliases[0] if client_kind == "bench" else (
                        "-m" if aliases == ("-m", "--model") else aliases[1]
                    )
                    for duplicate_alias in aliases:
                        with self.subTest(
                            client=client_kind,
                            option=aliases,
                            duplicate=duplicate_alias,
                        ):
                            with self.assertRaisesRegex(
                                ValueError, "canonical sealed"
                            ):
                                launch_rpc_client(
                                    (*command, duplicate_alias, value),
                                    cwd=root,
                                    manifest_value=cell,
                                    runner=_execution_runner(),
                                    popen_factory=lambda *_args, **_keywords: fake_process,
                                )
                        with self.subTest(
                            client=client_kind,
                            option=aliases,
                            inline=duplicate_alias,
                        ):
                            with self.assertRaisesRegex(
                                ValueError, "canonical sealed"
                            ):
                                launch_rpc_client(
                                    (*command, f"{duplicate_alias}={value}"),
                                    cwd=root,
                                    manifest_value=cell,
                                    runner=_execution_runner(),
                                    popen_factory=lambda *_args, **_keywords: fake_process,
                                )
                    for alternate in aliases:
                        if alternate == canonical:
                            continue
                        replaced = list(command)
                        replaced[replaced.index(canonical)] = alternate
                        with self.subTest(
                            client=client_kind,
                            option=aliases,
                            replacement=alternate,
                        ):
                            with self.assertRaisesRegex(ValueError, "non-canonical"):
                                launch_rpc_client(
                                    tuple(replaced),
                                    cwd=root,
                                    manifest_value=cell,
                                    runner=_execution_runner(),
                                    popen_factory=lambda *_args, **_keywords: fake_process,
                                )

            with self.assertRaisesRegex(ValueError, "sealed model"):
                build_rpc_benchmark_argv(
                    root,
                    cell,
                    other_model,
                    runner=_execution_runner(),
                )

            with self.assertRaisesRegex(RuntimeError, "device identity changed"):
                build_rpc_benchmark_argv(
                    root,
                    cell,
                    model,
                    runner=_execution_runner(device_name="Different GPU"),
                )

            (root / "llama-cli.exe").write_bytes(b"tampered-client")
            with self.assertRaisesRegex(RuntimeError, "sealed client"):
                launch_rpc_client(
                    bench_argv,
                    cwd=root,
                    manifest_value=cell,
                    runner=_runtime_runner(),
                    popen_factory=lambda *_args, **_keywords: fake_process,
                )

            tampered = copy.deepcopy(cell.to_document())
            tampered["execution"]["nativePartialStageAbi"] = True
            tampered["cellId"] = _short_identity(tampered, "cellId")
            with self.assertRaisesRegex(ValueError, "partial Stage ABI"):
                parse_rpc_cell_manifest(tampered)


def _write_runtime(root: Path, *, include_rpc_library: bool = True) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "llama-cli.exe").write_bytes(b"cli")
    (root / "llama-bench.exe").write_bytes(b"bench")
    (root / "ggml-rpc-server.exe").write_bytes(b"rpc-server")
    if include_rpc_library:
        (root / "ggml-rpc.dll").write_bytes(b"rpc-library")


def _runtime_runner() -> _QueueRunner:
    return _QueueRunner(
        CommandResult(
            0,
            "",
            "version: 10068 (571d0d540)\n"
            "built with Clang 20.1.8 for Windows x86_64\n",
        ),
        CommandResult(
            0,
            "Available devices:\n"
            "  Vulkan0: Fake AMD GPU (8192 MiB, 7000 MiB free)\n",
            "ggml_vulkan: 0 = Fake AMD GPU | fp16: 1 | bf16: 1\n",
        ),
        CommandResult(
            0,
            "options: --help --threads --device --host --port --cache\n",
            "",
        ),
    )


def _execution_runner(
    *,
    device_name: str = "Remote AMD 890M",
    free_memory_mib: int = 7000,
) -> _QueueRunner:
    runtime = _runtime_runner()
    return _QueueRunner(
        *runtime.results,
        CommandResult(
            0,
            "Available devices:\n"
            f"  RPC0: {device_name} (8192 MiB, {free_memory_mib} MiB free)\n",
            "",
        ),
    )


def _benchmark_records(runtime, artifact) -> list[dict[str, object]]:
    common = {
        "build_commit": runtime.build_commit,
        "build_number": runtime.build_number,
        "cpu_info": "Fake CPU",
        "gpu_info": "Fake GPU",
        "backends": "Vulkan",
        "model_filename": "ignored.gguf",
        "model_type": "qwen3 tiny Q8_0",
        "model_size": max(1, artifact.size_bytes),
        "model_n_params": 1000,
        "avg_ns": 1_000_000,
        "stddev_ns": 0,
        "avg_ts": 1000.0,
        "stddev_ts": 0.0,
        "samples_ns": [1_000_000],
        "samples_ts": [1000.0],
    }
    return [
        {**common, "n_prompt": 1, "n_gen": 0},
        {**common, "n_prompt": 0, "n_gen": 1},
    ]


def _write_minimal_gguf(path: Path, *, name: str = "Qwen3 Tiny") -> None:
    metadata: list[tuple[str, int, object]] = [
        ("general.architecture", 8, "qwen3"),
        ("general.name", 8, name),
        ("general.file_type", 4, 7),
        ("qwen3.embedding_length", 4, 1024),
        ("qwen3.context_length", 4, 40960),
        ("qwen3.block_count", 4, 28),
    ]
    output = bytearray(b"GGUF")
    output.extend(struct.pack("<IQQ", 3, 0, len(metadata)))
    for key, value_type, value in metadata:
        output.extend(_gguf_string(key))
        output.extend(struct.pack("<I", value_type))
        if value_type == 8:
            output.extend(_gguf_string(str(value)))
        elif value_type == 4:
            output.extend(struct.pack("<I", int(value)))
        else:
            raise AssertionError(value_type)
    path.write_bytes(output)


def _gguf_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _short_identity(document: dict[str, object], key: str) -> str:
    body = copy.deepcopy(document)
    body.pop(key)
    encoded = json.dumps(
        body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:32]


if __name__ == "__main__":
    unittest.main()
