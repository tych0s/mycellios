from __future__ import annotations

import copy
import unittest

from distributed_runtime.executor_abi import (
    STAGE_EXECUTOR_SCHEMA,
    STAGE_KV_FORK_REPORT_SCHEMA,
    StageKVForkReport,
    build_stage_executor_manifest,
    model_identity_for_source,
    parse_stage_executor_manifest,
    parse_stage_kv_fork_report,
    validate_executor_chain,
)


class StageExecutorAbiTests(unittest.TestCase):
    def test_physical_kv_fork_report_round_trip_is_strict(self) -> None:
        report = StageKVForkReport(
            logical_bytes=12_288,
            unique_physical_bytes=5_120,
            copied_bytes=1_024,
            newly_reserved_bytes=1_024,
            peak_workspace_bytes=None,
        )
        document = report.to_document()
        self.assertEqual(document["schema"], STAGE_KV_FORK_REPORT_SCHEMA)
        self.assertEqual(parse_stage_kv_fork_report(document), report)

        unknown = copy.deepcopy(document)
        unknown["backendGuess"] = 1
        with self.assertRaisesRegex(ValueError, "unknown or missing"):
            parse_stage_kv_fork_report(unknown)
        negative = copy.deepcopy(document)
        negative["copiedBytes"] = -1
        with self.assertRaisesRegex(ValueError, "non-negative"):
            parse_stage_kv_fork_report(negative)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            StageKVForkReport(
                logical_bytes=0,
                unique_physical_bytes=0,
                copied_bytes=0,
                newly_reserved_bytes=-1,
                peak_workspace_bytes=0,
            )

    def test_hub_snapshot_identity_ignores_absolute_cache_root(self) -> None:
        commit = "c1899de289a04d12100db370d81485cdf75e47ca"
        first = model_identity_for_source(
            f"C:/host-a/cache/models--Qwen--Qwen3-0.6B/snapshots/{commit}",
            None,
        )
        second = model_identity_for_source(
            f"D:\\host-b\\hf\\snapshots\\{commit}",
            None,
        )
        by_revision = model_identity_for_source("Qwen/Qwen3-0.6B", commit)
        self.assertEqual(first, second)
        self.assertEqual(first, by_revision)
        self.assertRegex(first, r"^sha256:[0-9a-f]{64}$")

    def test_manifest_round_trip_seals_every_execution_field(self) -> None:
        manifest = _manifest(
            engine="external GGUF runtime",
            adapter="gdlp-llama-stage",
            layer_start=0,
            layer_end=8,
            total_layers=24,
            device_kinds=("gpu", "cpu"),
            compute_apis=("vulkan",),
            weight_dtypes=("q4_k_m",),
        )
        document = manifest.to_document()
        self.assertEqual(document["schema"], STAGE_EXECUTOR_SCHEMA)
        self.assertTrue(manifest.first)
        self.assertFalse(manifest.last)
        self.assertEqual(parse_stage_executor_manifest(document), manifest)
        self.assertEqual(len(manifest.executor_id), 32)
        self.assertEqual(len(manifest.sha256), 64)

        tampered = copy.deepcopy(document)
        tampered["tensor"]["hiddenSize"] = 8192
        with self.assertRaisesRegex(ValueError, "identity"):
            parse_stage_executor_manifest(tampered)

    def test_heterogeneous_engines_form_one_compatible_contiguous_chain(self) -> None:
        chain = validate_executor_chain(
            (
                _manifest(
                    engine="external GGUF runtime",
                    adapter="gdlp-llama-stage",
                    layer_start=0,
                    layer_end=8,
                    total_layers=24,
                    compute_apis=("cuda",),
                ),
                _manifest(
                    engine="mlx",
                    adapter="gdlp-mlx-stage",
                    layer_start=8,
                    layer_end=16,
                    total_layers=24,
                    compute_apis=("metal",),
                ),
                _manifest(
                    engine="onnxruntime",
                    adapter="gdlp-ort-stage",
                    layer_start=16,
                    layer_end=24,
                    total_layers=24,
                    compute_apis=("directml",),
                ),
            )
        )
        self.assertEqual([item.engine for item in chain], ["external GGUF runtime", "mlx", "onnxruntime"])
        self.assertTrue(chain[-1].last)

    def test_chain_rejects_gaps_model_mismatch_and_tensor_mismatch(self) -> None:
        first = _manifest(layer_start=0, layer_end=8, total_layers=16)
        gap = _manifest(layer_start=9, layer_end=16, total_layers=16)
        with self.assertRaisesRegex(ValueError, "not contiguous"):
            validate_executor_chain((first, gap))

        other_model = _manifest(
            layer_start=8,
            layer_end=16,
            total_layers=16,
            model_identity="model-b",
        )
        with self.assertRaisesRegex(ValueError, "model identity"):
            validate_executor_chain((first, other_model))

        other_tensor = _manifest(
            layer_start=8,
            layer_end=16,
            total_layers=16,
            activation_dtype="bfloat16",
        )
        with self.assertRaisesRegex(ValueError, "tensor contract"):
            validate_executor_chain((first, other_tensor))

    def test_parser_rejects_missing_rollback_and_nonfinite_limits(self) -> None:
        document = _manifest(layer_start=0, layer_end=16, total_layers=16).to_document()
        document["execution"]["operations"].remove("truncate")
        # Recompute is intentionally impossible through the public builder;
        # the stale identity is already sufficient to fail closed.
        with self.assertRaises(ValueError):
            parse_stage_executor_manifest(document)

        with self.assertRaisesRegex(ValueError, "maxBatchSize"):
            build_stage_executor_manifest(
                **_arguments(layer_start=0, layer_end=16, total_layers=16),
                max_batch_size=0,
            )


def _arguments(
    *,
    layer_start: int,
    layer_end: int,
    total_layers: int,
    engine: str = "python-torch",
    adapter: str = "transformers-safetensors",
    model_identity: str = "model-a",
    activation_dtype: str = "float32",
    device_kinds: tuple[str, ...] = ("cpu",),
    compute_apis: tuple[str, ...] = ("torch",),
    weight_dtypes: tuple[str, ...] = ("float32",),
) -> dict[str, object]:
    return {
        "engine": engine,
        "engine_version": "1",
        "adapter": adapter,
        "model_identity": model_identity,
        "model_source": "org/model",
        "model_revision": "commit-a",
        "artifact_format": "safetensors",
        "layer_start": layer_start,
        "layer_end": layer_end,
        "total_layers": total_layers,
        "hidden_size": 4096,
        "activation_dtype": activation_dtype,
        "activation_codecs": ("fp16", "fp32"),
        "device_kinds": device_kinds,
        "compute_apis": compute_apis,
        "weight_dtypes": weight_dtypes,
    }


def _manifest(**overrides: object):
    values = _arguments(
        layer_start=int(overrides.pop("layer_start", 0)),
        layer_end=int(overrides.pop("layer_end", 16)),
        total_layers=int(overrides.pop("total_layers", 16)),
    )
    values.update(overrides)
    return build_stage_executor_manifest(**values)


if __name__ == "__main__":
    unittest.main()
