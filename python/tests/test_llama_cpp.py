from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from distributed_runtime.executor_abi import build_stage_executor_manifest
from distributed_runtime.llama_cpp import (
    CommandResult,
    GGUF_SPLIT_ROLE,
    LLAMA_CPP_DEPLOYMENT_SCHEMA,
    LLAMA_LAYER_ARTIFACT_FORMAT,
    LLAMA_STAGE_ADAPTER,
    LayerPackageFile,
    _default_command_runner,
    benchmark_llama_cpp,
    build_layer_package,
    build_partial_stage_deployment,
    build_whole_model_deployment,
    compatibility_key_for_llama_partial,
    inspect_gguf,
    parse_layer_package_manifest,
    parse_llama_cpp_deployment,
    probe_llama_cpp,
    verify_layer_package_files,
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


class LlamaCppProbeTests(unittest.TestCase):
    def test_probe_parses_build_devices_and_stock_capability_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "llama-cli.exe").write_bytes(b"cli")
            (root / "llama-bench.exe").write_bytes(b"bench")
            (root / "ggml-rpc-server.exe").write_bytes(b"rpc")
            runner = _QueueRunner(
                CommandResult(
                    0,
                    "",
                    "version: 10068 (571d0d540)\n"
                    "built with Clang 20.1.8 for Windows x86_64\n",
                ),
                CommandResult(
                    0,
                    "Available devices:\n"
                    "  Vulkan0: AMD Radeon(TM) 890M Graphics "
                    "(16444 MiB, 15622 MiB free)\n",
                    "ggml_vulkan: 0 = AMD Radeon(TM) 890M Graphics "
                    "(AMD proprietary driver) | uma: 1 | fp16: 1 | "
                    "bf16: 1 | warp size: 64\n",
                ),
            )

            probe = probe_llama_cpp(root, runner=runner)

            self.assertEqual(probe.build_number, 10068)
            self.assertEqual(probe.build_commit, "571d0d540")
            self.assertEqual(probe.engine_version, "b10068-571d0d540")
            self.assertEqual(probe.devices[0].identifier, "Vulkan0")
            self.assertEqual(dict(probe.devices[0].properties)["fp16"], True)
            self.assertTrue(probe.rpc_backend_present)
            self.assertFalse(probe.native_partial_stage_abi)
            self.assertEqual(
                probe.to_document()["capabilities"]["ordinaryGgufSplitRole"],
                GGUF_SPLIT_ROLE,
            )
            self.assertEqual(runner.calls[0][0][1:], ("--version",))
            self.assertEqual(runner.calls[1][0][1:], ("--list-devices",))

    def test_default_runner_always_uses_argument_vector_and_shell_false(self) -> None:
        completed = type(
            "Completed",
            (),
            {"returncode": 0, "stdout": "ok", "stderr": ""},
        )()
        with patch("distributed_runtime.llama_cpp.subprocess.run", return_value=completed) as run:
            result = _default_command_runner(
                ("llama-bench", "--list-devices"), Path("."), 2.5
            )

        self.assertEqual(result.stdout, "ok")
        positional, keywords = run.call_args
        self.assertEqual(positional[0], ["llama-bench", "--list-devices"])
        self.assertIs(keywords["shell"], False)
        self.assertIs(keywords["check"], False)

    def test_probe_rejects_unrecognized_or_failed_process_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "llama-cli.exe").write_bytes(b"cli")
            (root / "llama-bench.exe").write_bytes(b"bench")
            bad_version = _QueueRunner(CommandResult(0, "unknown", ""))
            with self.assertRaisesRegex(ValueError, "version output"):
                probe_llama_cpp(root, runner=bad_version)

            failed = _QueueRunner(CommandResult(3, "", "load failed"))
            with self.assertRaisesRegex(RuntimeError, "exit code 3"):
                probe_llama_cpp(root, runner=failed)


class GgufInspectionTests(unittest.TestCase):
    def test_inspector_validates_metadata_and_hashes_the_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tiny-Q8_0.gguf"
            _write_minimal_gguf(path)

            artifact = inspect_gguf(path)

            self.assertEqual(artifact.gguf_version, 3)
            self.assertEqual(artifact.architecture, "qwen3")
            self.assertEqual(artifact.quantization, "q8_0")
            self.assertEqual(artifact.block_count, 28)
            self.assertEqual(artifact.embedding_length, 1024)
            self.assertEqual(artifact.context_length, 40960)
            self.assertEqual(artifact.identity, "sha256:" + _sha256(path))
            self.assertEqual(len(artifact.metadata_sha256), 64)
            self.assertEqual(artifact.execution_scope, "complete-model")

    def test_ordinary_gguf_split_is_storage_not_an_executable_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model-00001-of-00002.gguf"
            _write_minimal_gguf(path)
            artifact = inspect_gguf(path)

            self.assertEqual(artifact.storage_layout, "gguf-split-shard")
            self.assertEqual(artifact.execution_scope, "complete-model-storage-shard")
            self.assertFalse(
                artifact.to_document()["storage"]["ordinaryGgufSplitIsLayerStage"]
            )
            with self.assertRaisesRegex(ValueError, "storage shard"):
                build_whole_model_deployment(_fake_runtime(), artifact)

    def test_inspector_rejects_bad_magic_and_missing_execution_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bad_magic = root / "bad.gguf"
            bad_magic.write_bytes(b"NOPE")
            with self.assertRaisesRegex(ValueError, "not a GGUF"):
                inspect_gguf(bad_magic)

            missing = root / "missing.gguf"
            _write_minimal_gguf(missing, omit_block_count=True)
            with self.assertRaisesRegex(ValueError, "block count"):
                inspect_gguf(missing)


class LlamaCppBenchmarkAndDeploymentTests(unittest.TestCase):
    def test_benchmark_parses_json_and_seals_whole_model_deployment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "llama-bench.exe").write_bytes(b"bench")
            model = root / "model.gguf"
            _write_minimal_gguf(model)
            artifact = inspect_gguf(model)
            runtime = _fake_runtime()
            runner = _QueueRunner(
                CommandResult(
                    0,
                    json.dumps(_benchmark_records(runtime, artifact)),
                    "diagnostic",
                )
            )

            benchmark = benchmark_llama_cpp(
                root,
                runtime,
                artifact,
                model,
                prompt_tokens=1,
                generation_tokens=1,
                repetitions=1,
                no_warmup=True,
                device="Vulkan0",
                runner=runner,
            )
            deployment = build_whole_model_deployment(
                runtime, artifact, benchmark=benchmark
            )

            self.assertEqual(deployment.mode, "whole-model")
            self.assertIsNone(deployment.stage_executor)
            self.assertEqual(
                deployment.to_document()["schema"], LLAMA_CPP_DEPLOYMENT_SCHEMA
            )
            self.assertEqual(parse_llama_cpp_deployment(deployment.to_document()), deployment)
            command = runner.calls[0][0]
            self.assertIn("-o", command)
            self.assertIn("json", command)
            self.assertIn("--no-warmup", command)
            self.assertEqual(command[-2:], ("--device", "Vulkan0"))

            tampered = copy.deepcopy(deployment.to_document())
            tampered["artifact"]["gguf"]["blockCount"] = 29
            with self.assertRaisesRegex(ValueError, "identity"):
                parse_llama_cpp_deployment(tampered)

    def test_benchmark_rejects_wrong_build_and_non_json_stdout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "llama-bench.exe").write_bytes(b"bench")
            model = root / "model.gguf"
            _write_minimal_gguf(model)
            artifact = inspect_gguf(model)
            runtime = _fake_runtime()
            wrong = _benchmark_records(runtime, artifact)
            wrong[0]["build_number"] = 7
            with self.assertRaisesRegex(ValueError, "does not match"):
                benchmark_llama_cpp(
                    root,
                    runtime,
                    artifact,
                    model,
                    runner=_QueueRunner(CommandResult(0, json.dumps(wrong), "")),
                )
            with self.assertRaisesRegex(ValueError, "not valid JSON"):
                benchmark_llama_cpp(
                    root,
                    runtime,
                    artifact,
                    model,
                    runner=_QueueRunner(CommandResult(0, "not-json", "")),
                )

    def test_recomputed_outer_id_cannot_hide_tampered_benchmark_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "llama-bench.exe").write_bytes(b"bench")
            model = root / "model.gguf"
            _write_minimal_gguf(model)
            artifact = inspect_gguf(model)
            runtime = _fake_runtime()
            benchmark = benchmark_llama_cpp(
                root,
                runtime,
                artifact,
                model,
                prompt_tokens=1,
                generation_tokens=1,
                runner=_QueueRunner(
                    CommandResult(
                        0,
                        json.dumps(_benchmark_records(runtime, artifact)),
                        "",
                    )
                ),
            )
            document = build_whole_model_deployment(
                runtime, artifact, benchmark=benchmark
            ).to_document()
            document["benchmark"]["records"][0]["avg_ts"] = 9999.0
            document["deploymentId"] = _short_identity(document, "deploymentId")

            with self.assertRaisesRegex(ValueError, "benchmark identity"):
                parse_llama_cpp_deployment(document)


class LlamaLayerPackageTests(unittest.TestCase):
    def test_layer_package_and_partial_stage_binding_are_sealed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            weight_path = root / "stage-08-16.gdlpgguf"
            weight_path.write_bytes(b"explicit executable layer package")
            file = LayerPackageFile(
                path=weight_path.name,
                role="layer-weights",
                size_bytes=weight_path.stat().st_size,
                sha256=_sha256(weight_path),
            )
            package = _layer_package(file)
            runtime = _fake_runtime()
            executor = _stage_executor(runtime, package)

            self.assertEqual(parse_layer_package_manifest(package.to_document()), package)
            self.assertEqual(verify_layer_package_files(package, root), package)
            deployment = build_partial_stage_deployment(runtime, package, executor)
            self.assertEqual(deployment.mode, "partial-stage")
            self.assertEqual(deployment.stage_executor, executor)
            self.assertTrue(
                deployment.to_document()["execution"]["requiresExternalStageAdapter"]
            )
            key = compatibility_key_for_llama_partial(
                deployment,
                device_kind="gpu",
                compute_api="vulkan",
                weight_dtype="q8_0",
                activation_codec="fp16",
            )
            self.assertEqual(key.executor_id, executor.executor_id)
            self.assertEqual(key.model_architecture, "qwen3")

            tampered = copy.deepcopy(package.to_document())
            tampered["stage"]["layerEnd"] = 17
            with self.assertRaisesRegex(ValueError, "identity"):
                parse_layer_package_manifest(tampered)

    def test_partial_stage_requires_explicit_matching_adapter_not_stock_gguf(self) -> None:
        file = LayerPackageFile(
            path="stage.bin",
            role="layer-weights",
            size_bytes=1,
            sha256="a" * 64,
        )
        package = _layer_package(file)
        runtime = _fake_runtime()
        wrong_adapter = _stage_executor(runtime, package, adapter="llama.cpp-cli")
        with self.assertRaisesRegex(ValueError, "Stage adapter"):
            build_partial_stage_deployment(runtime, package, wrong_adapter)

        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "model.gguf"
            _write_minimal_gguf(model)
            whole = build_whole_model_deployment(runtime, inspect_gguf(model))
            with self.assertRaisesRegex(ValueError, "do not fit"):
                compatibility_key_for_llama_partial(
                    whole,
                    device_kind="gpu",
                    compute_api="vulkan",
                    weight_dtype="q8_0",
                    activation_codec="fp16",
                )

    def test_package_file_verifier_detects_content_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "stage.bin"
            path.write_bytes(b"a")
            package = _layer_package(
                LayerPackageFile("stage.bin", "layer-weights", 1, _sha256(path))
            )
            path.write_bytes(b"b")
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                verify_layer_package_files(package, root)

    def test_package_rejects_ordinary_gguf_split_filename(self) -> None:
        with self.assertRaisesRegex(ValueError, "complete-model storage shards"):
            _layer_package(
                LayerPackageFile(
                    "model-00001-of-00002.gguf",
                    "layer-weights",
                    1,
                    "a" * 64,
                )
            )


def _fake_runtime():
    # Use the public probe so runtime identity remains subject to the same parser.
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "llama-cli.exe").write_bytes(b"cli")
        (root / "llama-bench.exe").write_bytes(b"bench")
        return probe_llama_cpp(
            root,
            runner=_QueueRunner(
                CommandResult(
                    0,
                    "",
                    "version: 10068 (571d0d540)\n"
                    "built with Clang 20.1.8 for Windows x86_64\n",
                ),
                CommandResult(
                    0,
                    "Available devices:\n"
                    "  Vulkan0: Fake GPU (8192 MiB, 4096 MiB free)\n",
                    "ggml_vulkan: 0 = Fake GPU | fp16: 1\n",
                ),
            ),
        )


def _benchmark_records(runtime, artifact) -> list[dict[str, object]]:
    common = {
        "build_commit": runtime.build_commit,
        "build_number": runtime.build_number,
        "cpu_info": "Fake CPU",
        "gpu_info": "Fake GPU",
        "backends": "Vulkan",
        "model_filename": "ignored-host-path.gguf",
        "model_type": "qwen3 tiny Q8_0",
        "model_size": max(1, artifact.size_bytes - 1),
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


def _layer_package(file: LayerPackageFile):
    return build_layer_package(
        model_identity="sha256:" + "b" * 64,
        model_source="org/qwen3",
        model_revision="commit-a",
        architecture="qwen3",
        quantization="q8_0",
        layer_start=8,
        layer_end=16,
        total_layers=28,
        hidden_size=1024,
        weight_dtype="q8_0",
        activation_dtype="float16",
        activation_codecs=("fp16", "fp32"),
        kv_format="llama.cpp-kv/1",
        files=(file,),
    )


def _stage_executor(runtime, package, *, adapter: str = LLAMA_STAGE_ADAPTER):
    return build_stage_executor_manifest(
        engine="llama.cpp",
        engine_version=runtime.engine_version,
        adapter=adapter,
        model_identity=package.model_identity,
        model_source=package.model_source,
        model_revision=package.model_revision,
        artifact_format=LLAMA_LAYER_ARTIFACT_FORMAT,
        layer_start=package.layer_start,
        layer_end=package.layer_end,
        total_layers=package.total_layers,
        hidden_size=package.hidden_size,
        activation_dtype=package.activation_dtype,
        activation_codecs=package.activation_codecs,
        kv_format=package.kv_format,
        device_kinds=("gpu",),
        compute_apis=("vulkan",),
        weight_dtypes=(package.weight_dtype,),
        features=(
            "explicit-layer-package",
            "layer-range",
            "rank-local-kv",
            "rollback",
        ),
    )


def _write_minimal_gguf(path: Path, *, omit_block_count: bool = False) -> None:
    metadata: list[tuple[str, int, object]] = [
        ("general.architecture", 8, "qwen3"),
        ("general.name", 8, "Qwen3 Tiny"),
        ("general.file_type", 4, 7),
        ("qwen3.embedding_length", 4, 1024),
        ("qwen3.context_length", 4, 40960),
        ("tokenizer.ggml.tokens", 9, (8, ("a", "b"))),
    ]
    if not omit_block_count:
        metadata.append(("qwen3.block_count", 4, 28))
    output = bytearray(b"GGUF")
    output.extend(struct.pack("<IQQ", 3, 0, len(metadata)))
    for key, value_type, value in metadata:
        output.extend(_gguf_string(key))
        output.extend(struct.pack("<I", value_type))
        output.extend(_gguf_value(value_type, value))
    path.write_bytes(output)


def _gguf_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _gguf_value(value_type: int, value: object) -> bytes:
    if value_type == 8:
        return _gguf_string(str(value))
    if value_type == 4:
        return struct.pack("<I", int(value))
    if value_type == 9:
        element_type, entries = value
        output = bytearray(struct.pack("<IQ", element_type, len(entries)))
        for entry in entries:
            output.extend(_gguf_value(element_type, entry))
        return bytes(output)
    raise AssertionError(value_type)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _short_identity(document: dict[str, object], identity_key: str) -> str:
    body = copy.deepcopy(document)
    body.pop(identity_key)
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
