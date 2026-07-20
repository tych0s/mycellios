from __future__ import annotations

import copy
import gc
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from transformers import (
    Glm4MoeConfig,
    Glm4MoeForCausalLM,
    LlamaConfig,
    Qwen3Config,
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
)

from distributed_runtime.model import (
    StageModelSpec,
    StageRunner,
    model_artifact_reference,
)
from distributed_runtime.executor_abi import validate_executor_chain
from distributed_runtime.compatibility import (
    ExecutorCompatibilityRegistry,
    build_executor_certification,
    compatibility_key_for_manifest,
    validate_certified_executor_chain,
)
from distributed_runtime.stage_cli import build_config as build_stage_config
from distributed_runtime.stage_cli import parse_args as parse_stage_args
from distributed_runtime.model_adapters import (
    UnsupportedSelectiveStageArchitectureError,
    adapter_registry_document,
    resolve_selective_stage_adapter,
)


class SelectiveStageAdapterRegistryTests(unittest.TestCase):
    def test_registry_is_closed_and_separates_llama_from_qwen3(self) -> None:
        llama = LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            architectures=["LlamaForCausalLM"],
        )
        qwen = _tiny_qwen3_config()

        self.assertEqual(
            resolve_selective_stage_adapter(llama).adapter_id,
            "transformers-llama-v1",
        )
        self.assertEqual(
            resolve_selective_stage_adapter(qwen).adapter_id,
            "transformers-qwen3-v1",
        )
        with self.assertRaisesRegex(
            UnsupportedSelectiveStageArchitectureError, "no certified"
        ):
            resolve_selective_stage_adapter(
                SimpleNamespace(
                    model_type="new_decoder",
                    architectures=["NewDecoderForCausalLM"],
                )
            )
        qwen.architectures = ["Qwen3ForSequenceClassification"]
        with self.assertRaisesRegex(
            UnsupportedSelectiveStageArchitectureError, "does not certify"
        ):
            resolve_selective_stage_adapter(qwen)

    def test_qwen3_adapter_preserves_global_layer_slice_and_qk_norms(self) -> None:
        from transformers import AutoModelForCausalLM

        source = _tiny_qwen3_config(layers=4)
        adapter = resolve_selective_stage_adapter(source)
        local = copy.deepcopy(source)
        adapter.slice_config(
            local,
            layer_start=1,
            layer_end=3,
            total_layers=4,
        )
        self.assertEqual(local.num_hidden_layers, 2)
        self.assertEqual(local.layer_types, ["full_attention", "full_attention"])

        model = AutoModelForCausalLM.from_config(local, dtype="float32")
        parts = adapter.inspect_constructed_model(model, local_layers=2)
        self.assertEqual(len(parts.layers), 2)
        for layer in parts.layers:
            self.assertEqual(layer.self_attn.q_norm.weight.numel(), local.head_dim)
            self.assertEqual(layer.self_attn.k_norm.weight.numel(), local.head_dim)

    def test_qwen3_hybrid_attention_fails_closed_until_certified(self) -> None:
        config = _tiny_qwen3_config(layers=2)
        config.layer_types = ["full_attention", "sliding_attention"]
        adapter = resolve_selective_stage_adapter(config)
        with self.assertRaisesRegex(
            UnsupportedSelectiveStageArchitectureError, "full_attention"
        ):
            adapter.validate_source_config(config, 2)

    def test_registry_document_exposes_semantics_not_wildcards(self) -> None:
        document = adapter_registry_document()
        self.assertEqual(document["schema"], "gdlp-transformers-stage-adapters/1")
        self.assertEqual(
            [entry["modelType"] for entry in document["adapters"]],
            ["llama", "qwen3", "qwen3_moe", "glm4_moe"],
        )
        qwen = document["adapters"][1]
        self.assertIn("self_attn.q_norm", qwen["requiredLayerModules"])
        self.assertEqual(qwen["attentionScope"], "full-only")

    def test_tiny_qwen3_moe_checkpoint_is_exact_across_two_pipeline_ranges(self) -> None:
        torch.manual_seed(107)
        config = Qwen3MoeConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            moe_intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            num_experts=4,
            num_experts_per_tok=2,
            max_position_embeddings=64,
            architectures=["Qwen3MoeForCausalLM"],
        )
        model = Qwen3MoeForCausalLM(config).eval()
        input_ids = torch.tensor([[1, 5, 9, 3]], dtype=torch.long)
        expected: list[int] = []
        cache = None
        current = input_ids
        with torch.inference_mode():
            for _ in range(4):
                output = model(input_ids=current, past_key_values=cache, use_cache=True)
                cache = output.past_key_values
                token = int(torch.argmax(output.logits[:, -1, :], dim=-1).item())
                expected.append(token)
                current = torch.tensor([[token]], dtype=torch.long)

        with tempfile.TemporaryDirectory() as temporary:
            model.save_pretrained(temporary, safe_serialization=True)
            del model, cache, output
            gc.collect()
            artifact = model_artifact_reference(temporary)
            identity = {
                "artifact_identity": artifact.identity,
                "canonical_model_source": artifact.canonical_source,
                "canonical_model_revision": artifact.canonical_revision,
            }
            first = StageRunner(
                StageModelSpec(temporary, 0, 1, 2, 1, **identity)
            )
            last = StageRunner(
                StageModelSpec(temporary, 1, 2, 2, 1, **identity)
            )

        chain = validate_executor_chain(
            (first.executor_manifest, last.executor_manifest)
        )
        self.assertEqual(
            [manifest.adapter for manifest in chain],
            ["transformers-qwen3-moe-v1", "transformers-qwen3-moe-v1"],
        )
        for manifest in chain:
            self.assertIn("sparse-moe", manifest.features)
            self.assertIn("ep-plan-metadata-only", manifest.features)
            self.assertNotIn("expert-parallel", manifest.features)
        self.assertEqual(
            first.base.config.base_model_ep_plan,
            config.base_model_ep_plan,
        )
        self.assertEqual(
            tuple(first.base.layers[0].mlp.experts.gate_up_proj.shape),
            (4, 32, 32),
        )

        first.begin(71)
        last.begin(71)
        actual: list[int] = []
        try:
            hidden = first.forward_ids(71, input_ids)
            for index in range(len(expected)):
                _, token = last.forward_hidden(71, hidden)
                self.assertIsInstance(token, int)
                actual.append(token)
                if index + 1 < len(expected):
                    hidden = first.forward_ids(
                        71,
                        torch.tensor([[token]], dtype=torch.long),
                    )
        finally:
            first.end(71)
            last.end(71)
        self.assertEqual(actual, expected)
        certifications = _certify_tiny_pipeline(
            chain,
            architecture="Qwen3MoeForCausalLM",
            test_prefix="tiny-qwen3-moe-two-range",
        )
        self.assertTrue(
            all(item.evidence.parity == "exact-greedy" for item in certifications)
        )

    def test_tiny_glm4_moe_preserves_global_dense_prefix_across_ranges(self) -> None:
        torch.manual_seed(109)
        config = Glm4MoeConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            moe_intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            n_routed_experts=4,
            num_experts_per_tok=2,
            n_shared_experts=1,
            first_k_dense_replace=1,
            max_position_embeddings=64,
            architectures=["Glm4MoeForCausalLM"],
        )
        model = Glm4MoeForCausalLM(config).eval()
        input_ids = torch.tensor([[2, 6, 1]], dtype=torch.long)
        expected: list[int] = []
        cache = None
        current = input_ids
        with torch.inference_mode():
            for _ in range(4):
                output = model(input_ids=current, past_key_values=cache, use_cache=True)
                cache = output.past_key_values
                token = int(torch.argmax(output.logits[:, -1, :], dim=-1).item())
                expected.append(token)
                current = torch.tensor([[token]], dtype=torch.long)

        with tempfile.TemporaryDirectory() as temporary:
            model.save_pretrained(temporary, safe_serialization=True)
            del model, cache, output
            gc.collect()
            artifact = model_artifact_reference(temporary)
            identity = {
                "artifact_identity": artifact.identity,
                "canonical_model_source": artifact.canonical_source,
                "canonical_model_revision": artifact.canonical_revision,
            }
            first = StageRunner(
                StageModelSpec(temporary, 0, 1, 2, 1, **identity)
            )
            last = StageRunner(
                StageModelSpec(temporary, 1, 2, 2, 1, **identity)
            )

        chain = validate_executor_chain(
            (first.executor_manifest, last.executor_manifest)
        )
        self.assertEqual(
            [manifest.adapter for manifest in chain],
            ["transformers-glm4-moe-v1", "transformers-glm4-moe-v1"],
        )
        self.assertTrue(hasattr(first.base.layers[0].mlp, "gate_proj"))
        self.assertFalse(hasattr(first.base.layers[0].mlp, "experts"))
        self.assertTrue(hasattr(last.base.layers[0].mlp, "experts"))
        self.assertEqual(last.base.config.first_k_dense_replace, 0)
        for manifest in chain:
            self.assertIn("hybrid-dense-moe", manifest.features)
            self.assertIn("ep-plan-metadata-only", manifest.features)
            self.assertNotIn("expert-parallel", manifest.features)

        first.begin(73)
        last.begin(73)
        actual: list[int] = []
        try:
            hidden = first.forward_ids(73, input_ids)
            for index in range(len(expected)):
                _, token = last.forward_hidden(73, hidden)
                self.assertIsInstance(token, int)
                actual.append(token)
                if index + 1 < len(expected):
                    hidden = first.forward_ids(
                        73,
                        torch.tensor([[token]], dtype=torch.long),
                    )
        finally:
            first.end(73)
            last.end(73)
        self.assertEqual(actual, expected)
        certifications = _certify_tiny_pipeline(
            chain,
            architecture="Glm4MoeForCausalLM",
            test_prefix="tiny-glm4-moe-two-range",
        )
        self.assertTrue(
            all(item.evidence.parity == "exact-greedy" for item in certifications)
        )


class ModelArtifactReferenceTests(unittest.TestCase):
    def test_hub_identity_and_coordinates_ignore_host_cache_root(self) -> None:
        commit = "c1899de289a04d12100db370d81485cdf75e47ca"
        with tempfile.TemporaryDirectory() as first_temporary, tempfile.TemporaryDirectory() as second_temporary:
            snapshots = []
            for temporary in (first_temporary, second_temporary):
                snapshot = (
                    Path(temporary)
                    / "models--Qwen--Qwen3-0.6B"
                    / "snapshots"
                    / commit
                )
                snapshot.mkdir(parents=True)
                snapshots.append(snapshot)
            first = model_artifact_reference(str(snapshots[0]))
            second = model_artifact_reference(str(snapshots[1]))

        self.assertEqual(first, second)
        self.assertEqual(first.canonical_source, "hf://Qwen/Qwen3-0.6B")
        self.assertEqual(first.canonical_revision, commit)
        self.assertRegex(first.identity, r"^sha256:[0-9a-f]{64}$")

    def test_explicit_orchestrator_identity_avoids_rehashing_local_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            reference = model_artifact_reference(
                temporary,
                artifact_identity="snapshot:uint64:0011223344556677",
                canonical_source="content-addressed://snapshot:uint64:0011223344556677",
                canonical_revision="revision-a",
            )
        self.assertEqual(reference.identity, "snapshot:uint64:0011223344556677")
        self.assertEqual(reference.canonical_revision, "revision-a")

    def test_stage_spec_requires_identity_before_canonical_coordinates(self) -> None:
        with self.assertRaisesRegex(ValueError, "require an explicit"):
            StageModelSpec(
                "model",
                0,
                1,
                1,
                1,
                canonical_model_source="hf://org/model",
            )

    def test_stage_cli_propagates_precomputed_identity_without_content_rehash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, patch(
            "distributed_runtime.model._model_snapshot_digest",
            side_effect=AssertionError("orchestrated stage must not rehash weights"),
        ):
            config = build_stage_config(
                parse_stage_args(
                    [
                        "--model",
                        temporary,
                        "--model-artifact-identity",
                        "snapshot:uint64:0000000000003039",
                        "--model-canonical-source",
                        "content-addressed://snapshot:uint64:0000000000003039",
                        "--pipeline-snapshot-identity",
                        "12345",
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
                    ]
                )
            )
        self.assertEqual(config.pipeline_id, 12345)
        self.assertEqual(
            config.spec.artifact_identity,
            "snapshot:uint64:0000000000003039",
        )
        self.assertEqual(
            config.spec.canonical_model_source,
            "content-addressed://snapshot:uint64:0000000000003039",
        )


def _tiny_qwen3_config(*, layers: int = 2) -> Qwen3Config:
    return Qwen3Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        architectures=["Qwen3ForCausalLM"],
        layer_types=["full_attention"] * layers,
    )


def _certify_tiny_pipeline(chain, *, architecture: str, test_prefix: str):
    keys = tuple(
        compatibility_key_for_manifest(
            manifest,
            model_architecture=architecture,
            quantization="none",
            device_kind="cpu",
            compute_api="torch",
            weight_dtype="float32",
            activation_codec="fp32",
            parallelism_mode="pipeline-stage",
            world_size=1,
            partitioning="contiguous-layer-range/1",
        )
        for manifest in chain
    )
    certifications = tuple(
        build_executor_certification(
            key,
            evidence_level="loopback-physical",
            parity="exact-greedy",
            test_id=f"{test_prefix}-stage-{index}",
            device_fingerprint="physical-windows-x86_64-cpu/synthetic-checkpoint",
        )
        for index, key in enumerate(keys)
    )
    registry = ExecutorCompatibilityRegistry(certifications)
    return validate_certified_executor_chain(
        chain,
        keys,
        registry,
        minimum_evidence="loopback-physical",
        require_exact_greedy=True,
    )


if __name__ == "__main__":
    unittest.main()
