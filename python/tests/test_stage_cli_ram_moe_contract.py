from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

from distributed_runtime.stage_cli import build_config, parse_args


class StageCliRamBackedMoeContractTests(unittest.TestCase):
    def _write_config(self, snapshot: Path) -> None:
        (snapshot / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "qwen3_moe",
                    "architectures": ["Qwen3MoeForCausalLM"],
                }
            ),
            encoding="utf-8",
        )

    def _argv(self, snapshot: Path) -> list[str]:
        return [
            "--model",
            str(snapshot),
            "--pipeline-snapshot-identity",
            "81985529216486895",
            "--layer-start",
            "1",
            "--layer-end",
            "2",
            "--total-layers",
            "2",
            "--listen-port",
            "23001",
            "--return-host",
            "127.0.0.1",
            "--return-port",
            "23002",
            "--ram-moe-artifact-schema",
            "gdlp-local-safetensors-moe-stage/1",
            "--ram-moe-artifact-identity",
            "sha256:0123456789abcdef" + "0" * 48,
            "--ram-moe-adapter-id",
            "transformers-qwen3-moe-v1",
            "--ram-moe-device",
            "cuda:0",
            "--ram-moe-pin-memory",
            "false",
            "--ram-moe-allow-cpu-fallback",
            "false",
            "--ram-moe-cache-schema",
            "gdlp-predictive-expert-cache/1",
            "--ram-moe-cache-capacity-bytes",
            "1024",
            "--ram-moe-prefetch-reserve-bytes",
            "512",
            "--ram-moe-pcie-bandwidth-gbytes-per-second",
            "12.5",
            "--ram-moe-hotness-decay",
            "0.95",
            "--ram-moe-min-prefetch-confidence",
            "0.6",
            "--ram-moe-resident-parameter-budget-bytes",
            "256",
            "--ram-moe-total-routed-expert-bytes",
            "1024",
            "--ram-moe-largest-expert-bytes",
            "512",
            "--ram-moe-resident-streaming-transient-bytes",
            "256",
            "--ram-moe-bounded-pinned-staging-reserve-bytes",
            "1024",
            "--ram-moe-host-ram-peak-upper-bound-bytes",
            "2304",
        ]

    def test_complete_contract_builds_ram_runner_config_without_standard_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, patch(
            "distributed_runtime.stage_cli.resolve_model_snapshot",
            side_effect=AssertionError("ordinary loader must remain unreachable"),
        ) as resolve:
            snapshot = Path(temporary).resolve()
            self._write_config(snapshot)
            config = build_config(parse_args(self._argv(snapshot)))
            self.assertIsNotNone(config.ram_backed_moe)
            assert config.ram_backed_moe is not None
            self.assertEqual(config.ram_backed_moe.expected_largest_expert_bytes, 512)
            self.assertEqual(
                config.ram_backed_moe.expected_bounded_pinned_staging_reserve_bytes,
                1024,
            )
            self.assertEqual(
                config.ram_backed_moe.expected_host_ram_peak_upper_bound_bytes,
                2304,
            )
            self.assertEqual(config.spec.artifact_identity, config.ram_backed_moe.artifact_identity)
        resolve.assert_not_called()

    def test_partial_contract_never_falls_through_to_the_standard_loader(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, patch(
            "distributed_runtime.stage_cli.resolve_model_snapshot",
            side_effect=AssertionError("ordinary loader must remain unreachable"),
        ) as resolve:
            snapshot = Path(temporary).resolve()
            self._write_config(snapshot)
            argv = self._argv(snapshot)
            index = argv.index("--ram-moe-cache-schema")
            del argv[index : index + 2]
            with self.assertRaisesRegex(ValueError, "must be supplied together"):
                build_config(parse_args(argv))
        resolve.assert_not_called()

    def test_partial_wave_limits_fail_before_model_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, patch(
            "distributed_runtime.stage_cli.resolve_model_snapshot",
            side_effect=AssertionError("model resolution must remain unreachable"),
        ) as resolve:
            snapshot = Path(temporary).resolve()
            self._write_config(snapshot)
            argv = [*self._argv(snapshot), "--sealed-wave-tokens", "1"]
            with self.assertRaisesRegex(ValueError, "must be supplied together"):
                build_config(parse_args(argv))
        resolve.assert_not_called()

    def test_contract_rejects_relative_path_cpu_fallback_and_small_buffers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary).resolve()
            self._write_config(snapshot)
            base = self._argv(snapshot)

            relative = list(base)
            relative[relative.index("--model") + 1] = "org/model"
            with self.assertRaisesRegex(ValueError, "absolute host-local path"):
                build_config(parse_args(relative))

            fallback = list(base)
            fallback[fallback.index("--ram-moe-allow-cpu-fallback") + 1] = "true"
            with self.assertRaisesRegex(ValueError, "CPU fallback is forbidden"):
                build_config(parse_args(fallback))

            undersized = list(base)
            undersized[
                undersized.index("--ram-moe-cache-capacity-bytes") + 1
            ] = "900"
            with self.assertRaisesRegex(ValueError, "cannot exceed half"):
                build_config(parse_args(undersized))

    def test_artifact_identity_and_pipeline_identity_are_mandatory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary).resolve()
            self._write_config(snapshot)
            base = self._argv(snapshot)
            malformed = list(base)
            malformed[
                malformed.index("--ram-moe-artifact-identity") + 1
            ] = "sha256:ABC"
            with self.assertRaisesRegex(ValueError, "lowercase SHA-256"):
                build_config(parse_args(malformed))

            no_pipeline_identity = list(base)
            index = no_pipeline_identity.index("--pipeline-snapshot-identity")
            del no_pipeline_identity[index : index + 2]
            with self.assertRaisesRegex(ValueError, "pipeline-snapshot-identity"):
                build_config(parse_args(no_pipeline_identity))

            mismatched_pipeline = list(base)
            mismatched_pipeline[
                mismatched_pipeline.index("--pipeline-snapshot-identity") + 1
            ] = "1"
            with self.assertRaisesRegex(ValueError, "identities do not match"):
                build_config(parse_args(mismatched_pipeline))


if __name__ == "__main__":
    unittest.main()
