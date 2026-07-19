from __future__ import annotations

import copy
import unittest

from distributed_runtime.compatibility import (
    CompatibilityNotCertifiedError,
    ExecutorCompatibilityRegistry,
    build_executor_certification,
    compatibility_key_for_manifest,
    parse_executor_certification,
    validate_certified_executor_chain,
)
from distributed_runtime.executor_abi import build_stage_executor_manifest


class ExecutorCompatibilityRegistryTests(unittest.TestCase):
    def test_certification_round_trip_is_identity_sealed(self) -> None:
        manifest = _manifest(0, 8)
        key = _key(manifest)
        certification = build_executor_certification(
            key,
            evidence_level="hardware-physical",
            parity="exact-greedy",
            test_id="gpu-a100-smoke-001",
            artifact_sha256="a" * 64,
            device_fingerprint="nvidia-a100-pcie-40gb/driver-590",
        )

        self.assertEqual(parse_executor_certification(certification.to_document()), certification)
        self.assertEqual(len(certification.certification_id), 32)
        self.assertEqual(len(certification.sha256), 64)

        tampered = copy.deepcopy(certification.to_document())
        tampered["key"]["runtime"]["worldSize"] = 4
        with self.assertRaisesRegex(ValueError, "identity"):
            parse_executor_certification(tampered)

    def test_registry_requires_the_exact_model_backend_and_parallelism_tuple(self) -> None:
        manifest = _manifest(0, 8)
        key = _key(manifest)
        registry = ExecutorCompatibilityRegistry(
            (
                build_executor_certification(
                    key,
                    evidence_level="loopback-physical",
                    parity="exact-greedy",
                    test_id="cpu-loopback-001",
                ),
            )
        )

        self.assertIsNotNone(
            registry.resolve(
                key,
                minimum_evidence="loopback-physical",
                require_exact_greedy=True,
            )
        )
        changed = copy.copy(key)
        object.__setattr__(changed, "quantization", "q4_k_m")
        with self.assertRaises(CompatibilityNotCertifiedError):
            registry.require(changed)

        changed_api = copy.copy(key)
        object.__setattr__(changed_api, "compute_api", "vulkan")
        self.assertIsNone(registry.resolve(changed_api))

    def test_evidence_gate_does_not_promote_loopback_to_real_gpu_proof(self) -> None:
        key = _key(_manifest(0, 8))
        registry = ExecutorCompatibilityRegistry(
            (
                build_executor_certification(
                    key,
                    evidence_level="loopback-physical",
                    parity="exact-greedy",
                    test_id="loopback-only",
                ),
            )
        )

        self.assertIsNotNone(registry.resolve(key, minimum_evidence="unit"))
        self.assertIsNone(registry.resolve(key, minimum_evidence="hardware-physical"))
        with self.assertRaises(CompatibilityNotCertifiedError):
            registry.require(key, minimum_evidence="hardware-physical")

    def test_key_builder_rejects_capabilities_not_advertised_by_executor(self) -> None:
        manifest = _manifest(0, 8)
        with self.assertRaisesRegex(ValueError, "compute API"):
            _key(manifest, compute_api="rocm")
        with self.assertRaisesRegex(ValueError, "multiple members"):
            _key(manifest, world_size=1)
        with self.assertRaisesRegex(ValueError, "activation codec"):
            _key(manifest, activation_codec="int8")

    def test_complete_heterogeneous_chain_requires_every_stage_certified(self) -> None:
        manifests = (
            _manifest(0, 8, engine="external GGUF runtime", adapter="gdlp-llama-stage"),
            _manifest(8, 16, engine="python-torch", adapter="llama-tp-safetensors"),
        )
        keys = tuple(_key(manifest) for manifest in manifests)
        certifications = tuple(
            build_executor_certification(
                key,
                evidence_level="unit",
                parity="exact-greedy",
                test_id=f"stage-{index}",
            )
            for index, key in enumerate(keys)
        )
        registry = ExecutorCompatibilityRegistry(certifications)

        resolved = validate_certified_executor_chain(
            manifests,
            keys,
            registry,
            require_exact_greedy=True,
        )
        self.assertEqual(resolved, certifications)

        incomplete = ExecutorCompatibilityRegistry(certifications[:1])
        with self.assertRaises(CompatibilityNotCertifiedError):
            validate_certified_executor_chain(manifests, keys, incomplete)


def _manifest(
    layer_start: int,
    layer_end: int,
    *,
    engine: str = "python-torch",
    adapter: str = "llama-tp-safetensors",
):
    return build_stage_executor_manifest(
        engine=engine,
        engine_version="1",
        adapter=adapter,
        model_identity="sha256:" + "b" * 64,
        model_source="org/model",
        model_revision="commit-a",
        artifact_format="safetensors",
        layer_start=layer_start,
        layer_end=layer_end,
        total_layers=16,
        hidden_size=4096,
        activation_dtype="float16",
        activation_codecs=("fp16", "fp32"),
        device_kinds=("gpu",),
        compute_apis=("cuda",),
        weight_dtypes=("float16",),
        features=("layer-range", "rank-local-kv", "rollback", "unequal-tp"),
    )


def _key(manifest, **overrides):
    values = {
        "model_architecture": "LlamaForCausalLM",
        "quantization": "none",
        "device_kind": "gpu",
        "compute_api": "cuda",
        "weight_dtype": "float16",
        "activation_codec": "fp16",
        "parallelism_mode": "tensor-parallel-cell",
        "world_size": 2,
        "partitioning": "weighted-head-mlp/1",
    }
    values.update(overrides)
    return compatibility_key_for_manifest(manifest, **values)


if __name__ == "__main__":
    unittest.main()
