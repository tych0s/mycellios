from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from safetensors.torch import save_file
import torch
from torch import nn

from distributed_runtime.model import (
    StageModelSpec,
    StageRunner,
    _checkpoint_key_map,
    _checkpoint_name,
    _load_stage_parameters_from_safetensors,
    _slice_layer_specific_config,
    load_tokenizer,
    mean_intervals_ms,
    model_snapshot_identity,
    reference_generate,
    resolve_model_snapshot,
)
from distributed_runtime.engine import _resolve_verified_tokens
from distributed_runtime.executor_abi import validate_executor_chain
from distributed_runtime.protocol import (
    Frame,
    FrameType,
    TensorCodec,
    decode_tensor,
    encode_tensor,
)


class ModelContractTests(unittest.TestCase):
    def test_stage_spec_rejects_invalid_partitions(self) -> None:
        invalid = (
            ("", 0, 1, 1, 1),
            ("model", -1, 1, 1, 1),
            ("model", 0, 0, 1, 1),
            ("model", 1, 1, 1, 1),
            ("model", 0, 2, 1, 1),
            ("model", 0, 1, 1, 0),
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                StageModelSpec(*values)
        with self.assertRaisesRegex(ValueError, "revision"):
            StageModelSpec("model", 0, 1, 1, 1, revision=" ")

    def test_layer_specific_lists_and_tuples_keep_the_global_slice(self) -> None:
        config = SimpleNamespace(
            layer_types=["a", "b", "c", "d", "e", "f"],
            per_layer_tuple=(0, 1, 2, 3, 4, 5),
            unrelated=[10, 20],
        )
        spec = StageModelSpec("model", 2, 5, 6, 1)
        _slice_layer_specific_config(config, spec)
        self.assertEqual(config.layer_types, ["c", "d", "e"])
        self.assertEqual(config.per_layer_tuple, (2, 3, 4))
        self.assertEqual(config.unrelated, [10, 20])

    def test_tied_embedding_alias_is_never_used_for_an_untied_model(self) -> None:
        spec = StageModelSpec("model", 0, 1, 1, 1)
        both = {
            "model.embed_tokens.weight": "model.safetensors",
            "lm_head.weight": "model.safetensors",
        }
        self.assertEqual(
            _checkpoint_name(
                "lm_head.weight", spec, both, tied_embeddings=True
            ),
            "model.embed_tokens.weight",
        )
        self.assertEqual(
            _checkpoint_name(
                "lm_head.weight", spec, both, tied_embeddings=False
            ),
            "lm_head.weight",
        )
        head_only = {"lm_head.weight": "model.safetensors"}
        self.assertEqual(
            _checkpoint_name(
                "model.embed_tokens.weight",
                spec,
                head_only,
                tied_embeddings=True,
            ),
            "lm_head.weight",
        )

    def test_sharded_index_is_validated_and_local_paths_are_resolved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            save_file({"a": torch.ones(1)}, root / "part-1.safetensors")
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"a": "part-1.safetensors"}}),
                encoding="utf-8",
            )
            self.assertEqual(_checkpoint_key_map(root), {"a": "part-1.safetensors"})
            self.assertEqual(resolve_model_snapshot(str(root)), str(root.resolve()))

            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"a": "../outside.safetensors"}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "relative path"):
                _checkpoint_key_map(root)

        missing = Path(tempfile.gettempdir()).resolve() / "missing-distributed-model"
        with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
            resolve_model_snapshot(str(missing))

    def test_model_snapshot_identity_is_path_independent_and_content_sensitive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            first = parent / "first"
            second = parent / "second"
            first.mkdir()
            second.mkdir()
            for root in (first, second):
                (root / "config.json").write_text('{"hidden_size":4}', encoding="utf-8")
                save_file({"weight": torch.arange(4)}, root / "model.safetensors")
            first_id = model_snapshot_identity(str(first))
            self.assertEqual(first_id, model_snapshot_identity(str(first)))
            self.assertEqual(first_id, model_snapshot_identity(str(second)))
            self.assertGreaterEqual(first_id, 0)
            self.assertLessEqual(first_id, (1 << 64) - 1)

            save_file({"weight": torch.arange(4) + 1}, second / "model.safetensors")
            self.assertNotEqual(first_id, model_snapshot_identity(str(second)))

    def test_selective_loader_loads_persistent_buffers_and_rejects_omissions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = StageModelSpec(str(root), 1, 2, 2, 1)
            checkpoint = {
                "model.layers.1.weight": torch.full((2, 2), 7.0),
                "model.layers.1.scale": torch.tensor([3.0, 4.0]),
                "model.layers.1.omitted": torch.tensor([1.0]),
            }
            save_file(checkpoint, root / "model.safetensors")
            model = _ToyStageModel()
            with self.assertRaisesRegex(KeyError, "unconsumed required tensors"):
                _load_stage_parameters_from_safetensors(model, spec)

            save_file(
                {key: value for key, value in checkpoint.items() if not key.endswith("omitted")},
                root / "model.safetensors",
            )
            _load_stage_parameters_from_safetensors(model, spec)
            self.assertTrue(
                torch.equal(model.model.layers[0].weight, torch.full((2, 2), 7.0))
            )
            self.assertTrue(
                torch.equal(model.model.layers[0].scale, torch.tensor([3.0, 4.0]))
            )

    def test_zero_token_reference_does_not_load_a_model(self) -> None:
        ids = torch.tensor([[1]], dtype=torch.long)
        tokens, metrics = reference_generate("does-not-exist", ids, 0, 1)
        self.assertEqual(tokens, [])
        self.assertEqual(metrics, {"ttft_ms": 0.0, "total_ms": 0.0, "tpot_ms": 0.0})

    def test_mean_intervals(self) -> None:
        self.assertEqual(mean_intervals_ms([]), 0.0)
        self.assertEqual(mean_intervals_ms([1.0]), 0.0)
        self.assertAlmostEqual(mean_intervals_ms([1.0, 1.01, 1.04]), 20.0)

    def test_stage_cache_can_be_truncated_to_an_accepted_prefix(self) -> None:
        cache = _CroppableCache()
        runner = StageRunner.__new__(StageRunner)
        runner.active_requests = {77}
        runner.tokens_seen = {77: 12}
        runner.caches = {77: cache}

        runner.truncate(77, 8)
        self.assertEqual(cache.crops, [8])
        self.assertEqual(runner.sequence_length(77), 8)
        runner.truncate(77, 8)
        self.assertEqual(cache.crops, [8])
        with self.assertRaisesRegex(ValueError, "cannot truncate"):
            runner.truncate(77, 9)

    def test_reference_does_not_compute_after_the_last_requested_token(self) -> None:
        fake = _FakeCausalModel()
        with patch(
            "distributed_runtime.model.AutoModelForCausalLM.from_pretrained",
            return_value=fake,
        ):
            tokens, _ = reference_generate(
                "fake", torch.tensor([[7, 8]], dtype=torch.long), 3, 1
            )
        self.assertEqual(fake.calls, 3, "prefill plus two decode forwards are sufficient")
        self.assertEqual(tokens, [1, 2, 0])


@unittest.skipUnless(
    os.environ.get("RUN_DISTRIBUTED_MODEL_TESTS") == "1",
    "set RUN_DISTRIBUTED_MODEL_TESTS=1 to run the cached SmolLM integration test",
)
class SmolLMPartitionIntegrationTests(unittest.TestCase):
    MODEL_NAME = os.environ.get(
        "DISTRIBUTED_TEST_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct"
    )

    def test_two_stages_are_token_exact_for_interleaved_chats(self) -> None:
        tokenizer = load_tokenizer(self.MODEL_NAME)
        prompts = (
            "The capital of France is",
            "The largest planet in our solar system is",
        )
        input_ids = [
            tokenizer(prompt, return_tensors="pt").input_ids for prompt in prompts
        ]
        output_tokens = 6
        references = [
            reference_generate(self.MODEL_NAME, ids, output_tokens, 2)[0]
            for ids in input_ids
        ]

        with patch(
            "distributed_runtime.model.AutoModelForCausalLM.from_pretrained",
            side_effect=AssertionError("stage loader must not instantiate the full checkpoint"),
        ):
            first = StageRunner(StageModelSpec(self.MODEL_NAME, 0, 15, 30, 2))
            last = StageRunner(StageModelSpec(self.MODEL_NAME, 15, 30, 30, 2))
        self.assertEqual(first.loader, "selective-safetensors")
        self.assertEqual(len(first.base.layers), 15)
        self.assertEqual(len(last.base.layers), 15)
        self.assertGreater(first.parameter_bytes, 0)
        self.assertGreater(last.parameter_bytes, 0)
        validate_executor_chain((first.executor_manifest, last.executor_manifest))
        self.assertEqual(first.executor_manifest.engine, "python-torch")
        self.assertEqual(first.base.embed_tokens.num_embeddings, 49_152)
        self.assertEqual(last.base.embed_tokens.num_embeddings, 1)
        self.assertIsNotNone(last.head)
        self.assertTrue(
            torch.equal(first.base.embed_tokens.weight, last.head.weight),
            "both distributed copies of a tied vocabulary must use one canonical tensor",
        )
        request_ids = (101, 202)
        generated: dict[int, list[int]] = {request_id: [] for request_id in request_ids}
        pending_hidden: dict[int, torch.Tensor] = {}
        try:
            for request_id, ids in zip(request_ids, input_ids):
                first.begin(request_id)
                last.begin(request_id)
                pending_hidden[request_id] = self.fp32_wire_round_trip(
                    first.forward_ids(request_id, ids), request_id, 0
                )

            for step in range(output_tokens):
                for request_id in request_ids:
                    _, token = last.forward_hidden(
                        request_id, pending_hidden[request_id]
                    )
                    self.assertIsNotNone(token)
                    generated[request_id].append(token)
                if step + 1 < output_tokens:
                    for request_id in request_ids:
                        token_ids = torch.tensor(
                            [[generated[request_id][-1]]], dtype=torch.long
                        )
                        pending_hidden[request_id] = self.fp32_wire_round_trip(
                            first.forward_ids(request_id, token_ids),
                            request_id,
                            step + 1,
                        )

            for request_id, ids, reference in zip(
                request_ids, input_ids, references
            ):
                self.assertEqual(generated[request_id], reference)
                expected_seen = int(ids.shape[1]) + output_tokens - 1
                self.assertEqual(first.sequence_length(request_id), expected_seen)
                self.assertEqual(last.sequence_length(request_id), expected_seen)
        finally:
            for request_id in request_ids:
                first.end(request_id)
                last.end(request_id)

        with self.assertRaisesRegex(ValueError, "has not received BEGIN"):
            first.forward_ids(request_ids[0], input_ids[0])

        speculative_id = 303
        first.begin(speculative_id)
        last.begin(speculative_id)
        try:
            prompt_hidden = self.fp32_wire_round_trip(
                first.forward_ids(speculative_id, input_ids[0]),
                speculative_id,
                0,
            )
            _, first_token = last.forward_hidden(speculative_id, prompt_hidden)
            self.assertEqual(first_token, references[0][0])
            wrong_second_draft = (references[0][2] + 1) % 49_152
            drafts = (references[0][1], wrong_second_draft)
            verify_ids = torch.tensor(
                [[int(first_token), *drafts]],
                dtype=torch.long,
            )
            verify_hidden = self.fp32_wire_round_trip(
                first.forward_ids(speculative_id, verify_ids),
                speculative_id,
                1,
            )
            _, target_tokens = last.forward_hidden(
                speculative_id,
                verify_hidden,
                token_mode="all",
            )
            self.assertIsInstance(target_tokens, tuple)
            accepted, emitted = _resolve_verified_tokens(drafts, target_tokens)
            self.assertEqual(accepted, 1)
            self.assertEqual(emitted, tuple(references[0][1:3]))

            keep_tokens = int(input_ids[0].shape[1]) + 1 + accepted
            first.truncate(speculative_id, keep_tokens)
            last.truncate(speculative_id, keep_tokens)
            correction_hidden = self.fp32_wire_round_trip(
                first.forward_ids(
                    speculative_id,
                    torch.tensor([[emitted[-1]]], dtype=torch.long),
                ),
                speculative_id,
                2,
            )
            _, continued = last.forward_hidden(speculative_id, correction_hidden)
            self.assertEqual(continued, references[0][3])
        finally:
            first.end(speculative_id)
            last.end(speculative_id)

    @staticmethod
    def fp32_wire_round_trip(
        hidden: torch.Tensor, request_id: int, step: int
    ) -> torch.Tensor:
        payload = encode_tensor(hidden, TensorCodec.FP32)
        frame = Frame(
            frame_type=FrameType.ACTIVATION,
            flags=int(TensorCodec.FP32),
            request_id=request_id,
            step=step,
            token_count=hidden.shape[1],
            hidden_size=hidden.shape[2],
            payload=payload,
        )
        return decode_tensor(frame)


class _CroppableCache:
    def __init__(self) -> None:
        self.crops: list[int] = []

    def crop(self, token_count: int) -> None:
        self.crops.append(token_count)


class _FakeCausalModel:
    def __init__(self) -> None:
        self.calls = 0

    def eval(self) -> "_FakeCausalModel":
        return self

    def __call__(self, **_: object) -> SimpleNamespace:
        self.calls += 1
        logits = torch.zeros(1, 1, 3, dtype=torch.float32)
        logits[0, 0, self.calls % 3] = 1
        return SimpleNamespace(logits=logits, past_key_values=(self.calls,))


class _ToyLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2, 2))
        self.register_buffer("scale", torch.zeros(2), persistent=True)


class _ToyStageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(tie_word_embeddings=False)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_ToyLayer()])


if __name__ == "__main__":
    unittest.main()
