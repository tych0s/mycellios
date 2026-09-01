from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path
from unittest.mock import patch

import torch
from distributed_runtime.model import StageModelSpec
from distributed_runtime.nakshatra import (
    FLAG_ALL_LOGITS,
    FLAG_KEEP_KV,
    NakshatraStageRunner,
    NakshatraStageRuntimeSpec,
)
from distributed_runtime.nakshatra_package import (
    NAKSHATRA_COMMIT,
    NAKSHATRA_PACKAGE_MANIFEST,
    load_nakshatra_stage_package,
    seal_nakshatra_stage_package,
)
from distributed_runtime.protocol import Frame, FrameType, TensorCodec
from distributed_runtime.stage import (
    SingleRequestAdmission,
    StageProcessConfig,
    build_stage_runner,
)
from distributed_runtime.stage_cli import parse_args

FAKE_DAEMON = Path(__file__).with_name("fake_nakshatra_daemon.py")
MODEL_CONTENT_SHA256 = hashlib.sha256(b"complete-logical-model").hexdigest()


class NakshatraPackageTests(unittest.TestCase):
    def test_seals_and_verifies_sub_gguf_with_exact_upstream_pin(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = _seal(root, start=2, end=4, total=6)

            loaded = load_nakshatra_stage_package(
                package.root,
                expected_package_id=package.package_id,
                expected_manifest_sha256=package.manifest_sha256,
            )
            document = json.loads(
                (package.root / NAKSHATRA_PACKAGE_MANIFEST).read_text(encoding="utf-8")
            )
            self.assertEqual(document["upstream"]["commit"], NAKSHATRA_COMMIT)
            self.assertEqual(loaded.mode, "middle")
            self.assertEqual(loaded.artifact_bytes, package.artifact_path.stat().st_size)
            self.assertEqual(document["compatibility"]["maxConcurrentRequests"], 1)
            self.assertEqual(document["compatibility"]["maxBatchSequences"], 1)

    def test_rejects_artifact_and_manifest_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = _seal(root, start=0, end=2, total=6)
            with package.artifact_path.open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "byte length"):
                load_nakshatra_stage_package(package.root)

            package = _seal(root, start=2, end=4, total=6, name="package-2")
            manifest = package.root / NAKSHATRA_PACKAGE_MANIFEST
            document = json.loads(manifest.read_text(encoding="utf-8"))
            document["stage"]["hiddenSize"] += 1
            manifest.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "identity"):
                load_nakshatra_stage_package(package.root)

    def test_rejects_non_gguf_and_existing_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "bad.gguf"
            source.write_bytes(b"not-a-gguf")
            with self.assertRaisesRegex(ValueError, "not a GGUF"):
                _seal(root, start=0, end=1, total=1, source=source)

            source.write_bytes(b"GGUFpayload")
            destination = root / "occupied"
            destination.mkdir()
            with self.assertRaises(FileExistsError):
                _seal(
                    root,
                    start=0,
                    end=1,
                    total=1,
                    source=source,
                    destination=destination,
                )


class NakshatraRunnerTests(unittest.TestCase):
    def test_executor_identity_seals_package_artifact_and_manifest_digests(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = _seal(root, start=2, end=4, total=6, name="baseline")

            # JSON whitespace is not part of packageId, but it is part of the
            # launch-sealed manifest bytes and must therefore change the
            # executor/recovery identity.
            manifest_variant = _seal(
                root, start=2, end=4, total=6, name="manifest-variant"
            )
            manifest_path = manifest_variant.root / NAKSHATRA_PACKAGE_MANIFEST
            manifest_path.write_bytes(manifest_path.read_bytes() + b"\n")
            manifest_variant = load_nakshatra_stage_package(manifest_variant.root)

            # Keep every numerical/model coordinate identical while changing
            # the actual sub-GGUF bytes.
            artifact_source = root / "artifact-variant.source.gguf"
            artifact_source.write_bytes(b"GGUF" + bytes(range(64)) + b"different")
            artifact_variant = _seal(
                root,
                start=2,
                end=4,
                total=6,
                source=artifact_source,
                destination=root / "artifact-variant",
            )

            runners = (
                _runner(package, root / "baseline.events.jsonl"),
                _runner(manifest_variant, root / "manifest-variant.events.jsonl"),
                _runner(artifact_variant, root / "artifact-variant.events.jsonl"),
            )
            try:
                baseline, changed_manifest, changed_artifact = runners
                features = set(baseline.executor_manifest.features)
                self.assertIn(
                    f"nakshatra-package-id:{package.package_id}", features
                )
                self.assertIn(
                    f"nakshatra-manifest-sha256:{package.manifest_sha256}",
                    features,
                )
                self.assertIn(
                    f"nakshatra-artifact-sha256:{package.artifact_sha256}",
                    features,
                )
                self.assertIn(
                    f"nakshatra-model-content-sha256:{package.model_content_sha256}",
                    features,
                )

                self.assertEqual(package.package_id, manifest_variant.package_id)
                self.assertEqual(
                    package.artifact_sha256, manifest_variant.artifact_sha256
                )
                self.assertNotEqual(
                    package.manifest_sha256, manifest_variant.manifest_sha256
                )
                self.assertNotEqual(
                    baseline.executor_manifest.executor_id,
                    changed_manifest.executor_manifest.executor_id,
                )

                self.assertEqual(package.layer_start, artifact_variant.layer_start)
                self.assertEqual(package.layer_end, artifact_variant.layer_end)
                self.assertEqual(package.hidden_size, artifact_variant.hidden_size)
                self.assertEqual(package.weight_type, artifact_variant.weight_type)
                self.assertNotEqual(
                    package.artifact_sha256, artifact_variant.artifact_sha256
                )
                self.assertNotEqual(package.package_id, artifact_variant.package_id)
                self.assertNotEqual(
                    baseline.executor_manifest.executor_id,
                    changed_artifact.executor_manifest.executor_id,
                )
            finally:
                for runner in runners:
                    runner.close()

    def test_middle_stage_runs_persistent_embedding_decode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = _seal(root, start=2, end=4, total=6)
            events = root / "events.jsonl"
            runner = _runner(package, events)
            try:
                runner.begin(11)
                hidden = torch.tensor(
                    [[[1.0, -2.0, 3.0, -4.0], [0.5, 0.25, -0.5, -0.25]]],
                    dtype=torch.float16,
                )
                output, tokens = runner.forward_hidden(11, hidden, token_mode="all")
                self.assertTrue(
                    torch.equal(output, hidden.to(torch.float32) + torch.tensor(0.25))
                )
                self.assertIsNone(tokens)
                self.assertEqual(runner.sequence_length(11), 2)
                self.assertEqual(runner.executor_manifest.engine, "nakshatra-llama.cpp")
                self.assertIn("sub-gguf", runner.executor_manifest.features)
                self.assertEqual(runner.executor_manifest.max_batch_size, 1)
                self.assertEqual(runner.client.binary_commit, NAKSHATRA_COMMIT)
            finally:
                runner.close()
            self.assertEqual(runner.client.process.returncode, 0)
            observed = _events(events)
            self.assertEqual(len({event["pid"] for event in observed}), 1)
            commands = [event for event in observed if event["event"] == "command"]
            self.assertEqual([event["command"] for event in commands], [3, 2])
            self.assertEqual(commands[1]["flags"], FLAG_ALL_LOGITS)
            self.assertEqual(observed[-1]["event"], "eof")

    def test_last_stage_keeps_truncates_verifies_and_restarts_kv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = _seal(
                root,
                start=4,
                end=6,
                total=6,
                keep_token_embeddings=True,
            )
            events = root / "events.jsonl"
            runner = _runner(package, events)
            try:
                runner.begin(41)
                with self.assertRaisesRegex(RuntimeError, "one active request"):
                    runner.begin(42)
                verify_hidden = torch.tensor(
                    [[[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, -3.0, -4.0]]]
                )
                output, tokens = runner.forward_hidden(
                    41, verify_hidden, token_mode="all"
                )
                self.assertTrue(torch.equal(output, verify_hidden))
                self.assertEqual(tokens, (4, 4))
                runner.truncate(41, 1)
                self.assertEqual(runner.sequence_length(41), 1)
                _, token = runner.forward_hidden(
                    41, torch.tensor([[[0.001, 0.002, 0.003, 0.004]]])
                )
                self.assertEqual(token, 10)
                self.assertEqual(runner.sequence_length(41), 2)
                runner.end(41)

                runner.begin(42)
                _, token = runner.forward_hidden(
                    42,
                    torch.tensor([[[0.001, 0.002, 0.003, 0.004]]]),
                    token_mode="none",
                )
                self.assertIsNone(token)
                runner.end(42)
            finally:
                runner.close()
                runner.close()

            commands = [
                event for event in _events(events) if event["event"] == "command"
            ]
            self.assertEqual([event["command"] for event in commands], [3, 2, 4, 2, 2])
            self.assertEqual(commands[1]["flags"], FLAG_ALL_LOGITS)
            self.assertEqual(commands[3]["flags"], FLAG_KEEP_KV)
            self.assertEqual(commands[3]["startPos"], 1)
            self.assertEqual(commands[4]["flags"], 0)
            self.assertEqual(commands[4]["startPos"], 0)
            self.assertEqual(runner.client.process.returncode, 0)

    def test_stage_spec_and_live_dimensions_must_match_sealed_package(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = _seal(root, start=2, end=4, total=6)
            runtime = _runtime(package, root / "events.jsonl")
            with self.assertRaisesRegex(ValueError, "does not match"):
                NakshatraStageRunner(
                    StageModelSpec("example/llama", 1, 4, 6, 1, "revision-a"),
                    runtime,
                )

            wrong_live = NakshatraStageRuntimeSpec(
                **{
                    **runtime.__dict__,
                    "daemon_command": (
                        sys.executable,
                        str(FAKE_DAEMON),
                        "--fake-hidden-size",
                        "5",
                        "--fake-vocab-size",
                        "17",
                        "--fake-event-log",
                        str(root / "wrong-events.jsonl"),
                    ),
                }
            )
            with self.assertRaisesRegex(ValueError, "hidden size"):
                NakshatraStageRunner(
                    StageModelSpec("example/llama", 2, 4, 6, 1, "revision-a"),
                    wrong_live,
                )

    def test_rejects_daemon_built_from_another_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = _seal(root, start=2, end=4, total=6)
            runtime = _runtime(package, root / "events.jsonl")
            wrong_runtime = NakshatraStageRuntimeSpec(
                **{
                    **runtime.__dict__,
                    "daemon_command": (
                        *runtime.daemon_command,
                        "--fake-version-sha",
                        "1111111111111111111111111111111111111111",
                    ),
                }
            )
            with self.assertRaisesRegex(RuntimeError, "required"):
                NakshatraStageRunner(
                    StageModelSpec("example/llama", 2, 4, 6, 1, "revision-a"),
                    wrong_runtime,
                )

    def test_rejects_root_or_full_range_without_token_id_executor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = _seal(root, start=0, end=6, total=6)
            runtime = _runtime(package, root / "events.jsonl")
            with self.assertRaisesRegex(ValueError, "layer_start=0"):
                NakshatraStageRunner(
                    StageModelSpec("example/llama", 0, 6, 6, 1, "revision-a"),
                    runtime,
                )

    def test_stage_cli_rejects_known_external_backends(self) -> None:
        for argument, backend in (
            ("--nakshatra-package", "nakshatra"),
            ("--llama-cpp-server", "llama.cpp"),
            ("--ollama-url", "ollama"),
            ("--vllm-endpoint", "vllm"),
        ):
            with self.subTest(argument=argument), self.assertRaisesRegex(
                ValueError,
                f"mycellios_native_runtime_forbids_external_backend:{backend}",
            ):
                parse_args([argument, "research-only"])

    def test_canonical_stage_process_has_no_external_backend_escape_hatch(self) -> None:
        config = _canonical_stage_config()
        external_fields = {
            "nakshatra_package",
            "nakshatra_package_id",
            "nakshatra_manifest_sha256",
            "nakshatra_daemon_command",
            "nakshatra_context_tokens",
            "nakshatra_gpu_layers",
            "nakshatra_compute_api",
            "nakshatra_startup_timeout_seconds",
            "nakshatra_call_timeout_seconds",
            "nakshatra_close_timeout_seconds",
            "allow_research_external_backend",
        }
        self.assertTrue(
            external_fields.isdisjoint(field.name for field in fields(StageProcessConfig))
        )

        # StageProcessConfig is intentionally not slotted, so also prove that a
        # caller cannot resurrect the retired route by attaching the old
        # attributes dynamically.
        config.__dict__["nakshatra_package"] = "research-only"
        config.__dict__["allow_research_external_backend"] = True
        native_runner = object()
        with patch(
            "distributed_runtime.stage.StageRunner",
            return_value=native_runner,
        ) as stage_runner:
            self.assertIs(build_stage_runner(config), native_runner)
        stage_runner.assert_called_once_with(config.spec)

    def test_single_request_admission_preserves_each_deferred_stream(self) -> None:
        admission = SingleRequestAdmission()
        begin_a = _frame(FrameType.BEGIN, 10)
        begin_b = _frame(FrameType.BEGIN, 20)
        activation_b = _frame(FrameType.ACTIVATION, 20)
        end_b = _frame(FrameType.END, 20)
        begin_c = _frame(FrameType.BEGIN, 30)
        activation_a = _frame(FrameType.ACTIVATION, 10)

        self.assertTrue(admission.admit_or_defer(begin_a))
        self.assertFalse(admission.admit_or_defer(begin_b))
        self.assertFalse(admission.admit_or_defer(activation_b))
        self.assertFalse(admission.admit_or_defer(end_b))
        self.assertFalse(admission.admit_or_defer(begin_c))
        self.assertTrue(admission.admit_or_defer(activation_a))
        admission.release(10)

        for expected in (begin_b, activation_b, end_b):
            observed = admission.next_deferred()
            self.assertEqual(observed, expected)
            self.assertTrue(admission.admit_or_defer(observed))
        admission.release(20)
        observed = admission.next_deferred()
        self.assertEqual(observed, begin_c)
        self.assertTrue(admission.admit_or_defer(observed))

def _frame(frame_type: FrameType, request_id: int) -> Frame:
    return Frame(frame_type, 0, request_id, 0, 0, 0, b"")


def _canonical_stage_config() -> StageProcessConfig:
    return StageProcessConfig(
        spec=StageModelSpec("example/llama", 0, 1, 1, 1, "revision-a"),
        pipeline_id=0x0102030405060708,
        listen_host="127.0.0.1",
        listen_port=20_110,
        next_host=None,
        next_port=None,
        next_layer_end=None,
        return_host="127.0.0.1",
        return_port=20_111,
        codec=TensorCodec.FP32,
        one_way_delay_ms=0,
        bandwidth_mbps=0,
        connect_timeout_seconds=5,
    )


def _seal(
    root: Path,
    *,
    start: int,
    end: int,
    total: int,
    name: str = "package",
    source: Path | None = None,
    destination: Path | None = None,
    keep_token_embeddings: bool = False,
):
    source_path = source or (root / f"{name}.source.gguf")
    if source is None:
        source_path.write_bytes(b"GGUF" + bytes(range(64)))
    return seal_nakshatra_stage_package(
        source_path,
        destination or (root / name),
        model_source="example/llama",
        model_revision="revision-a",
        model_content_sha256=MODEL_CONTENT_SHA256,
        pipeline_id=0x0102030405060708,
        layer_start=start,
        layer_end=end,
        total_layers=total,
        hidden_size=4,
        vocab_size=17,
        weight_type="Q4_K_M",
        max_context_tokens=16,
        keep_token_embeddings=keep_token_embeddings,
    )


def _runtime(package, events: Path) -> NakshatraStageRuntimeSpec:
    return NakshatraStageRuntimeSpec(
        package=str(package.root),
        daemon_command=(
            sys.executable,
            str(FAKE_DAEMON),
            "--fake-hidden-size",
            "4",
            "--fake-vocab-size",
            "17",
            "--fake-event-log",
            str(events),
            "--fake-fragment-bytes",
            "2",
        ),
        context_tokens=16,
        threads=2,
        gpu_layers=3,
        compute_api="cuda",
        startup_timeout_seconds=5.0,
        call_timeout_seconds=5.0,
        expected_pipeline_id=package.pipeline_id,
        expected_package_id=package.package_id,
        expected_manifest_sha256=package.manifest_sha256,
    )


def _runner(package, events: Path) -> NakshatraStageRunner:
    return NakshatraStageRunner(
        StageModelSpec(
            package.model_source,
            package.layer_start,
            package.layer_end,
            package.total_layers,
            1,
            package.model_revision,
        ),
        _runtime(package, events),
    )


def _events(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


if __name__ == "__main__":
    unittest.main()
