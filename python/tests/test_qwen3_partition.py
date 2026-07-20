from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import torch

from distributed_runtime.compatibility import (
    ExecutorCompatibilityRegistry,
    build_executor_certification,
    compatibility_key_for_manifest,
    validate_certified_executor_chain,
)
from distributed_runtime.executor_abi import validate_executor_chain
from distributed_runtime.model import (
    StageModelSpec,
    StageRunner,
    load_tokenizer,
    reference_generate,
)


@unittest.skipUnless(
    os.environ.get("RUN_QWEN3_MODEL_TESTS") == "1",
    "set RUN_QWEN3_MODEL_TESTS=1 to run the cached Qwen3-0.6B parity gate",
)
class Qwen3PartitionIntegrationTests(unittest.TestCase):
    MODEL_NAME = os.environ.get("QWEN3_TEST_MODEL", "Qwen/Qwen3-0.6B")

    def test_monolithic_and_two_selective_ranges_are_greedy_token_exact(self) -> None:
        tokenizer = load_tokenizer(self.MODEL_NAME)
        input_ids = tokenizer(
            "The capital of France is",
            return_tensors="pt",
        ).input_ids
        expected, _ = reference_generate(self.MODEL_NAME, input_ids, 3, 2)

        with patch(
            "distributed_runtime.model.AutoModelForCausalLM.from_pretrained",
            side_effect=AssertionError("selective Qwen3 stages cannot load the full model"),
        ):
            first = StageRunner(StageModelSpec(self.MODEL_NAME, 0, 14, 28, 2))
            last = StageRunner(StageModelSpec(self.MODEL_NAME, 14, 28, 28, 2))
        chain = validate_executor_chain(
            (first.executor_manifest, last.executor_manifest)
        )
        self.assertEqual(
            [manifest.adapter for manifest in chain],
            ["transformers-qwen3-v1", "transformers-qwen3-v1"],
        )
        self.assertEqual(
            first.executor_manifest.model_identity,
            last.executor_manifest.model_identity,
        )
        self.assertEqual(
            first.executor_manifest.model_source,
            "hf://Qwen/Qwen3-0.6B",
        )
        self.assertIn("qk-rms-norm", first.executor_manifest.features)
        self.assertEqual(len(first.base.layers), 14)
        self.assertEqual(len(last.base.layers), 14)
        self.assertEqual(first.base.layers[0].self_attn.q_norm.weight.numel(), 128)
        self.assertEqual(first.base.layers[0].self_attn.k_norm.weight.numel(), 128)

        request_id = 1
        generated: list[int] = []
        first.begin(request_id)
        last.begin(request_id)
        try:
            hidden = first.forward_ids(request_id, input_ids)
            for index in range(len(expected)):
                _, token = last.forward_hidden(request_id, hidden)
                self.assertIsInstance(token, int)
                generated.append(token)
                if index + 1 < len(expected):
                    hidden = first.forward_ids(
                        request_id,
                        torch.tensor([[token]], dtype=torch.long),
                    )
            self.assertEqual(generated, expected)
            seen = int(input_ids.shape[1]) + len(expected) - 1
            self.assertEqual(first.sequence_length(request_id), seen)
            self.assertEqual(last.sequence_length(request_id), seen)
        finally:
            first.end(request_id)
            last.end(request_id)

        keys = tuple(
            compatibility_key_for_manifest(
                manifest,
                model_architecture="Qwen3ForCausalLM",
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
                evidence_level="hardware-physical",
                parity="exact-greedy",
                test_id=f"qwen3-0.6b-cpu-two-range-stage-{index}",
                device_fingerprint="physical-windows-x86_64-cpu/torch-2.13.0",
            )
            for index, key in enumerate(keys)
        )
        registry = ExecutorCompatibilityRegistry(certifications)
        self.assertEqual(
            validate_certified_executor_chain(
                chain,
                keys,
                registry,
                minimum_evidence="hardware-physical",
                require_exact_greedy=True,
            ),
            certifications,
        )


if __name__ == "__main__":
    unittest.main()
