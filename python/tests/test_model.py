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
from transformers import LlamaConfig, LlamaModel

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

    def test_exact_request_fork_has_independent_kv_and_promote_moves_selected_state(self) -> None:
        runner = _tiny_stage_runner()
        parent = 71
        child = 72
        runner.begin(parent)
        runner.forward_ids(parent, torch.tensor([[1, 2, 3]], dtype=torch.long))

        cache_bytes = runner.request_cache_bytes(parent)
        self.assertGreater(cache_bytes, 0)
        self.assertGreater(
            runner.project_request_cache_bytes(parent, 1),
            cache_bytes,
        )
        copied_bytes = runner.fork_request(
            child,
            parent,
            max_cache_bytes=cache_bytes,
        )
        self.assertGreater(copied_bytes, 0)
        self.assertEqual(runner.sequence_length(child), 3)
        self.assertNotEqual(
            runner.caches[parent].layers[0].keys.data_ptr(),
            runner.caches[child].layers[0].keys.data_ptr(),
        )

        parent_output = runner.forward_ids(
            parent, torch.tensor([[4]], dtype=torch.long)
        )
        child_projected_bytes = runner.project_request_cache_bytes(child, 1)
        child_output = runner.forward_ids(
            child, torch.tensor([[4]], dtype=torch.long)
        )
        self.assertLessEqual(
            runner.request_cache_bytes(child),
            child_projected_bytes,
        )
        torch.testing.assert_close(parent_output, child_output, rtol=1e-5, atol=1e-6)
        selected_cache = runner.caches[child]
        runner.forward_ids(child, torch.tensor([[5]], dtype=torch.long))
        self.assertEqual(runner.sequence_length(parent), 4)
        self.assertEqual(runner.sequence_length(child), 5)

        runner.promote_request(parent, child)
        self.assertEqual(runner.sequence_length(parent), 5)
        self.assertIs(runner.caches[parent], selected_cache)
        self.assertNotIn(child, runner.active_requests)
        with self.assertRaisesRegex(ValueError, "has not received BEGIN"):
            runner.sequence_length(child)
        runner.forward_ids(parent, torch.tensor([[6]], dtype=torch.long))
        self.assertEqual(runner.sequence_length(parent), 6)

    def test_safe_copy_runner_exposes_optional_physical_accounting(self) -> None:
        runner = _tiny_stage_runner()
        runner.begin(1)
        runner.forward_ids(1, torch.tensor([[1, 2, 3]], dtype=torch.long))
        parent_bytes = runner.request_cache_bytes(1)

        self.assertEqual(runner.unique_physical_cache_bytes((1,)), parent_bytes)
        self.assertEqual(
            runner.project_incremental_physical_cache_bytes(
                1,
                new_leaf_count=2,
                delta_tokens=1,
            ),
            2 * runner.project_request_cache_bytes(1, 1),
        )
        self.assertEqual(
            runner.project_tree_incremental_physical_cache_bytes(
                1,
                delta_tokens_by_leaf=(0, 2),
            ),
            runner.project_request_cache_bytes(1, 0)
            + runner.project_request_cache_bytes(1, 2),
        )
        copied = runner.fork_request(2, 1, max_cache_bytes=parent_bytes)
        report = runner.last_fork_report()
        self.assertIsNotNone(report)
        assert report is not None
        self.assertEqual(copied, parent_bytes)
        self.assertEqual(report.copied_bytes, parent_bytes)
        self.assertEqual(report.newly_reserved_bytes, parent_bytes)
        self.assertEqual(report.logical_bytes, 2 * parent_bytes)
        self.assertEqual(report.unique_physical_bytes, 2 * parent_bytes)
        self.assertEqual(
            runner.unique_physical_cache_bytes((1, 2)),
            2 * parent_bytes,
        )
        with self.assertRaisesRegex(ValueError, "unique"):
            runner.unique_physical_cache_bytes((1, 1))

    def test_exact_request_fork_and_promote_fail_closed_on_invalid_lifecycle(self) -> None:
        runner = _tiny_stage_runner()
        runner.begin(1)
        with self.assertRaisesRegex(ValueError, "must differ"):
            runner.fork_request(1, 1, max_cache_bytes=0)
        with self.assertRaisesRegex(ValueError, "has not received BEGIN"):
            runner.fork_request(2, 999, max_cache_bytes=0)
        runner.forward_ids(1, torch.tensor([[1]], dtype=torch.long))
        cache_bytes = runner.request_cache_bytes(1)
        with self.assertRaisesRegex(ValueError, "preflight byte budget"):
            runner.fork_request(2, 1, max_cache_bytes=cache_bytes - 1)
        self.assertNotIn(2, runner.active_requests)
        runner.fork_request(2, 1, max_cache_bytes=cache_bytes)
        with self.assertRaisesRegex(ValueError, "already active"):
            runner.fork_request(2, 1, max_cache_bytes=cache_bytes)
        with self.assertRaisesRegex(ValueError, "must differ"):
            runner.promote_request(1, 1)
        with self.assertRaisesRegex(ValueError, "has not received BEGIN"):
            runner.promote_request(1, 999)

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


class PhysicalTensorBatchTests(unittest.TestCase):
    def test_token_batch_uses_one_forward_and_splits_independent_dynamic_caches(self) -> None:
        runner = _tiny_stage_runner()
        first = torch.tensor([[1, 2, 3]], dtype=torch.long)
        second = torch.tensor([[1, 4, 5]], dtype=torch.long)
        next_first = torch.tensor([[6]], dtype=torch.long)
        next_second = torch.tensor([[7]], dtype=torch.long)

        for request_id in (11, 22):
            runner.begin(request_id)
        sequential_first = (
            runner.forward_ids(11, first),
            runner.forward_ids(22, second),
        )
        sequential_next = (
            runner.forward_ids(11, next_first),
            runner.forward_ids(22, next_second),
        )
        sequential_caches = {
            request_id: tuple(
                (layer.keys.clone(), layer.values.clone())
                for layer in runner.caches[request_id].layers
            )
            for request_id in (11, 22)
        }
        runner.end(11)
        runner.end(22)

        for request_id in (101, 202):
            runner.begin(request_id)
        calls_before = runner.model_forward_calls
        batched_first = runner.forward_ids_batch((101, 202), (first, second))
        self.assertEqual(runner.model_forward_calls - calls_before, 1)
        batched_next = runner.forward_ids_batch(
            (101, 202), (next_first, next_second)
        )

        for actual, expected in zip(batched_first, sequential_first, strict=True):
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        for actual, expected in zip(batched_next, sequential_next, strict=True):
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        for batched_id, sequential_id in ((101, 11), (202, 22)):
            self.assertEqual(runner.sequence_length(batched_id), 4)
            for layer, (expected_keys, expected_values) in zip(
                runner.caches[batched_id].layers,
                sequential_caches[sequential_id],
                strict=True,
            ):
                torch.testing.assert_close(layer.keys, expected_keys, rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(
                    layer.values, expected_values, rtol=1e-5, atol=1e-6
                )
        self.assertNotEqual(
            runner.caches[101].layers[0].keys.data_ptr(),
            runner.caches[202].layers[0].keys.data_ptr(),
            "split request caches must not alias the shared batch allocation",
        )
        self.assertEqual(runner.physical_batch_calls, 2)
        self.assertEqual(runner.physical_batch_items, 4)
        self.assertEqual(runner.max_observed_physical_batch_size, 2)

    def test_hidden_batch_is_token_exact_and_refuses_unequal_cache_lengths(self) -> None:
        runner = _tiny_stage_runner()
        torch.manual_seed(73)
        first = torch.randn(1, 3, runner.hidden_size)
        second = torch.randn(1, 3, runner.hidden_size)

        for request_id in (1, 2):
            runner.begin(request_id)
        sequential = (
            runner.forward_hidden(1, first, token_mode="all"),
            runner.forward_hidden(2, second, token_mode="all"),
        )
        runner.end(1)
        runner.end(2)

        for request_id in (3, 4):
            runner.begin(request_id)
        calls_before = runner.model_forward_calls
        batched = runner.forward_hidden_batch(
            (3, 4), (first, second), token_mode="all"
        )
        self.assertEqual(runner.model_forward_calls - calls_before, 1)
        for (actual_hidden, actual_tokens), (expected_hidden, expected_tokens) in zip(
            batched, sequential, strict=True
        ):
            torch.testing.assert_close(
                actual_hidden, expected_hidden, rtol=1e-5, atol=1e-6
            )
            self.assertEqual(actual_tokens, expected_tokens)

        runner.forward_hidden(3, torch.randn(1, 1, runner.hidden_size))
        self.assertNotEqual(
            runner.physical_batch_key(3, token_count=1, token_mode="last"),
            runner.physical_batch_key(4, token_count=1, token_mode="last"),
        )
        with self.assertRaisesRegex(ValueError, "equal cache length"):
            runner.forward_hidden_batch(
                (3, 4),
                (
                    torch.randn(1, 1, runner.hidden_size),
                    torch.randn(1, 1, runner.hidden_size),
                ),
            )

        runner._physical_batch_cache_supported = False
        self.assertIsNone(
            runner.physical_batch_key(3, token_count=1, token_mode="last")
        )


@unittest.skipUnless(
    os.environ.get("RUN_DISTRIBUTED_MODEL_TESTS") == "1",
    "set RUN_DISTRIBUTED_MODEL_TESTS=1 to run the cached SmolLM integration test",
)
class SmolLMPartitionIntegrationTests(unittest.TestCase):
    MODEL_NAME = os.environ.get(
        "DISTRIBUTED_TEST_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct"
    )

    def test_real_stage_physically_batches_and_preserves_per_request_kv(self) -> None:
        with patch(
            "distributed_runtime.model.AutoModelForCausalLM.from_pretrained",
            side_effect=AssertionError("stage loader must remain selective"),
        ):
            runner = StageRunner(StageModelSpec(self.MODEL_NAME, 15, 30, 30, 2))
        self.assertIn("physical-tensor-batching", runner.executor_manifest.features)
        self.assertEqual(runner.executor_manifest.max_batch_size, 8)
        torch.manual_seed(79)
        first_prefill = torch.randn(1, 3, runner.hidden_size)
        second_prefill = torch.randn(1, 3, runner.hidden_size)
        first_decode = torch.randn(1, 1, runner.hidden_size)
        second_decode = torch.randn(1, 1, runner.hidden_size)

        for request_id in (11, 22):
            runner.begin(request_id)
        sequential_prefill = (
            runner.forward_hidden(11, first_prefill, token_mode="all"),
            runner.forward_hidden(22, second_prefill, token_mode="all"),
        )
        sequential_decode = (
            runner.forward_hidden(11, first_decode),
            runner.forward_hidden(22, second_decode),
        )
        sequential_caches = {
            request_id: tuple(
                (layer.keys.clone(), layer.values.clone())
                for layer in runner.caches[request_id].layers
            )
            for request_id in (11, 22)
        }
        runner.end(11)
        runner.end(22)

        for request_id in (101, 202):
            runner.begin(request_id)
        before = runner.model_forward_calls
        batched_prefill = runner.forward_hidden_batch(
            (101, 202),
            (first_prefill, second_prefill),
            token_mode="all",
        )
        self.assertEqual(runner.model_forward_calls - before, 1)
        batched_decode = runner.forward_hidden_batch(
            (101, 202),
            (first_decode, second_decode),
        )

        for actual, expected in zip(
            batched_prefill, sequential_prefill, strict=True
        ):
            torch.testing.assert_close(actual[0], expected[0], rtol=1e-3, atol=3e-5)
            self.assertEqual(actual[1], expected[1])
        for actual, expected in zip(batched_decode, sequential_decode, strict=True):
            torch.testing.assert_close(actual[0], expected[0], rtol=1e-3, atol=3e-5)
            self.assertEqual(actual[1], expected[1])
        for batched_id, sequential_id in ((101, 11), (202, 22)):
            self.assertEqual(runner.sequence_length(batched_id), 4)
            for layer, (expected_keys, expected_values) in zip(
                runner.caches[batched_id].layers,
                sequential_caches[sequential_id],
                strict=True,
            ):
                torch.testing.assert_close(
                    layer.keys, expected_keys, rtol=1e-3, atol=3e-5
                )
                torch.testing.assert_close(
                    layer.values, expected_values, rtol=1e-3, atol=3e-5
                )
        self.assertNotEqual(
            runner.caches[101].layers[0].keys.data_ptr(),
            runner.caches[202].layers[0].keys.data_ptr(),
        )
        runner.truncate(101, 3)
        self.assertNotEqual(
            runner.physical_batch_key(101, token_count=1, token_mode="last"),
            runner.physical_batch_key(202, token_count=1, token_mode="last"),
        )
        runner.truncate(202, 3)
        self.assertEqual(
            runner.physical_batch_key(101, token_count=1, token_mode="last"),
            runner.physical_batch_key(202, token_count=1, token_mode="last"),
        )
        runner.end(101)
        runner.end(202)

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


def _tiny_stage_runner() -> StageRunner:
    torch.manual_seed(67)
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    runner = StageRunner.__new__(StageRunner)
    runner.base = LlamaModel(config).eval()
    runner.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False).eval()
    runner.hidden_size = config.hidden_size
    runner.spec = StageModelSpec("tiny", 0, 2, 2, 1)
    runner.caches = {}
    runner.tokens_seen = {}
    runner.active_requests = set()
    runner._physical_batch_cache_supported = True
    runner.model_forward_calls = 0
    runner.physical_batch_calls = 0
    runner.physical_batch_items = 0
    runner.max_observed_physical_batch_size = 1
    return runner


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
