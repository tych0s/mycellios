from __future__ import annotations

import copy
import tempfile
import unittest

import torch
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM, LlamaModel

from distributed_runtime.paged_stage import (
    HFPagedStageCache,
    HFPagedStageRunner,
    PagedCacheSnapshot,
    PagedStageCorruptionError,
)
from distributed_runtime.model import StageModelSpec, StageRunner
from distributed_runtime.executor_abi import StageKVPhysicalAccounting


def _tiny_llama(seed: int) -> LlamaModel:
    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        attention_dropout=0.0,
    )
    model = LlamaModel(config).eval()
    model.set_attn_implementation("eager")
    return model


def _paged_adapter(
    model: nn.Module,
    *,
    num_blocks: int = 16,
    max_active_requests: int = 4,
) -> HFPagedStageCache:
    return HFPagedStageCache.for_model(
        model,
        attention_backend="eager",
        block_size=4,
        num_blocks=num_blocks,
        max_batch_tokens=16,
        max_active_requests=max_active_requests,
        max_sequence_tokens=min(32, num_blocks * 4),
    )


def _requests(snapshot: PagedCacheSnapshot):
    return {request.request_id: request for request in snapshot.requests}


def _write_tiny_checkpoint(directory: str, seed: int) -> None:
    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        attention_dropout=0.0,
    )
    LlamaForCausalLM(config).eval().save_pretrained(
        directory,
        safe_serialization=True,
    )


class HFPagedStageRunnerTests(unittest.TestCase):
    def test_selective_first_stage_parity_manifest_and_cow_accounting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _write_tiny_checkpoint(temporary, 751)
            spec = StageModelSpec(temporary, 0, 1, 2, 1)
            classic = StageRunner(spec)
            paged = HFPagedStageRunner(
                spec,
                block_size=4,
                num_blocks=16,
                max_batch_tokens=16,
                max_active_requests=4,
                max_sequence_tokens=32,
            )
            try:
                classic.base.set_attn_implementation("eager")
                self.assertIsInstance(paged, StageKVPhysicalAccounting)
                self.assertIsNone(paged.head)
                features = set(paged.executor_manifest.features)
                self.assertTrue(
                    {
                        "paged-kv-shared-prefix",
                        "fork-complete-block-zero-copy",
                        "fork-tail-copy",
                        "physical-kv-accounting",
                    }.issubset(features)
                )
                self.assertNotIn("exact-tree-verify-batching", features)
                self.assertEqual(paged.executor_manifest.max_batch_size, 1)
                self.assertEqual(
                    paged.executor_manifest.kv_format,
                    "transformers-paged-cow-v1",
                )

                prompt = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
                classic.begin(1)
                paged.begin(1)
                expected = classic.forward_ids(1, prompt)
                actual = paged.forward_ids(1, prompt)
                self.assertTrue(
                    torch.allclose(expected, actual, atol=1e-5, rtol=1e-5)
                )
                logical_parent = paged.request_cache_bytes(1)
                self.assertEqual(
                    logical_parent,
                    6 * paged.paged_cache.bytes_per_token,
                )
                self.assertEqual(
                    paged.unique_physical_cache_bytes((1,)),
                    2 * paged.paged_cache.bytes_per_block,
                )
                fork_reservation = paged.project_incremental_physical_cache_bytes(
                    1,
                    new_leaf_count=1,
                    delta_tokens=0,
                )
                self.assertEqual(
                    fork_reservation,
                    paged.paged_cache.bytes_per_block,
                )
                self.assertEqual(
                    paged.project_incremental_physical_cache_bytes(
                        1,
                        new_leaf_count=2,
                        delta_tokens=3,
                    ),
                    4 * paged.paged_cache.bytes_per_block,
                )
                self.assertEqual(
                    paged.project_tree_incremental_physical_cache_bytes(
                        1,
                        delta_tokens_by_leaf=(0, 3),
                    ),
                    3 * paged.paged_cache.bytes_per_block,
                )
                self.assertEqual(
                    paged.project_request_incremental_physical_cache_bytes(1, 3),
                    paged.paged_cache.bytes_per_block,
                )
                with self.assertRaisesRegex(ValueError, "physical preflight"):
                    paged.fork_request(
                        2,
                        1,
                        max_cache_bytes=fork_reservation - 1,
                    )
                with self.assertRaisesRegex(ValueError, "unknown paged request 2"):
                    paged.sequence_length(2)

                copied = paged.fork_request(
                    2,
                    1,
                    max_cache_bytes=fork_reservation,
                )
                report = paged.last_fork_report()
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(copied, paged.paged_cache.bytes_per_block)
                self.assertEqual(report.copied_bytes, copied)
                self.assertEqual(report.newly_reserved_bytes, copied)
                self.assertEqual(report.logical_bytes, 2 * logical_parent)
                self.assertEqual(
                    report.unique_physical_bytes,
                    3 * paged.paged_cache.bytes_per_block,
                )
                self.assertIsNone(report.peak_workspace_bytes)
                self.assertEqual(
                    paged.unique_physical_cache_bytes((1, 2)),
                    report.unique_physical_bytes,
                )

                paged.end(2)
                paged.end(1)
                paged.begin(3)
                paged.forward_ids(
                    3,
                    torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]),
                )
                self.assertEqual(
                    paged.project_incremental_physical_cache_bytes(
                        3,
                        new_leaf_count=1,
                        delta_tokens=0,
                    ),
                    0,
                )
                self.assertEqual(
                    paged.fork_request(4, 3, max_cache_bytes=0),
                    0,
                )
                boundary_report = paged.last_fork_report()
                assert boundary_report is not None
                self.assertEqual(boundary_report.copied_bytes, 0)
                self.assertEqual(boundary_report.newly_reserved_bytes, 0)
                self.assertEqual(
                    boundary_report.unique_physical_bytes,
                    2 * paged.paged_cache.bytes_per_block,
                )
            finally:
                classic.close()
                paged.close()

    def test_selective_last_stage_head_truncate_and_hidden_parity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _write_tiny_checkpoint(temporary, 757)
            spec = StageModelSpec(temporary, 1, 2, 2, 1)
            classic = StageRunner(spec)
            paged = HFPagedStageRunner(
                spec,
                block_size=4,
                num_blocks=16,
                max_batch_tokens=16,
                max_active_requests=4,
                max_sequence_tokens=32,
            )
            try:
                classic.base.set_attn_implementation("eager")
                self.assertIsNotNone(paged.head)
                torch.manual_seed(761)
                hidden = torch.randn(1, 6, paged.hidden_size)
                continuation = torch.randn(1, 2, paged.hidden_size)
                classic.begin(9)
                paged.begin(9)

                expected_hidden, expected_tokens = classic.forward_hidden(
                    9,
                    hidden,
                    token_mode="all",
                )
                actual_hidden, actual_tokens = paged.forward_hidden(
                    9,
                    hidden,
                    token_mode="all",
                )
                self.assertTrue(
                    torch.allclose(
                        expected_hidden,
                        actual_hidden,
                        atol=1e-5,
                        rtol=1e-5,
                    )
                )
                self.assertEqual(actual_tokens, expected_tokens)

                classic.truncate(9, 4)
                paged.truncate(9, 4)
                self.assertEqual(paged.sequence_length(9), 4)
                expected_next, expected_token = classic.forward_hidden(
                    9,
                    continuation,
                )
                actual_next, actual_token = paged.forward_hidden(
                    9,
                    continuation,
                )
                self.assertTrue(
                    torch.allclose(
                        expected_next,
                        actual_next,
                        atol=1e-5,
                        rtol=1e-5,
                    )
                )
                self.assertEqual(actual_token, expected_token)
            finally:
                classic.close()
                paged.close()


class HFPagedStageCacheTests(unittest.TestCase):
    def test_stage_local_hidden_outputs_match_dynamic_cache(self) -> None:
        reference = _tiny_llama(701)
        paged_model = copy.deepcopy(reference)
        adapter = _paged_adapter(paged_model)
        self.addCleanup(adapter.close)

        torch.manual_seed(703)
        prefill = torch.randn(1, 5, reference.config.hidden_size)
        decode = torch.randn(1, 2, reference.config.hidden_size)

        reference_prefill = reference(inputs_embeds=prefill, use_cache=True)
        reference_decode = reference(
            inputs_embeds=decode,
            past_key_values=reference_prefill.past_key_values,
            use_cache=True,
        )

        adapter.begin(1)
        paged_prefill = adapter.forward_model(
            paged_model, 1, inputs_embeds=prefill
        )
        paged_decode = adapter.forward_model(
            paged_model, 1, inputs_embeds=decode
        )

        self.assertTrue(
            torch.allclose(
                reference_prefill.last_hidden_state,
                paged_prefill.output.last_hidden_state,
                atol=1e-6,
                rtol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                reference_decode.last_hidden_state,
                paged_decode.output.last_hidden_state,
                atol=1e-6,
                rtol=1e-6,
            )
        )
        self.assertEqual(adapter.sequence_length(1), 7)
        self.assertEqual(
            paged_decode.metrics.logical_bytes,
            7 * adapter.bytes_per_token,
        )
        self.assertIsNone(paged_decode.metrics.peak_workspace)
        self.assertGreater(
            paged_decode.metrics.pool_reserved_bytes,
            paged_decode.metrics.unique_physical_bytes,
        )

    def test_fork_shares_complete_blocks_copies_tail_and_diverges_exactly(self) -> None:
        reference = _tiny_llama(709)
        paged_model = copy.deepcopy(reference)
        adapter = _paged_adapter(paged_model)
        self.addCleanup(adapter.close)

        prompt = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
        parent_token = torch.tensor([[7]], dtype=torch.long)
        child_token = torch.tensor([[9]], dtype=torch.long)

        adapter.begin(10)
        adapter.forward_model(paged_model, 10, input_ids=prompt)
        fork_metrics = adapter.fork(11, 10)
        forked = _requests(adapter.snapshot())

        self.assertEqual(forked[10].block_ids[0], forked[11].block_ids[0])
        self.assertNotEqual(forked[10].block_ids[1], forked[11].block_ids[1])
        self.assertEqual(forked[10].ref_counts, (2, 1))
        self.assertEqual(forked[11].ref_counts, (2, 1))
        self.assertEqual(fork_metrics.copied_bytes, adapter.bytes_per_block)
        self.assertEqual(
            fork_metrics.newly_reserved_bytes, adapter.bytes_per_block
        )
        self.assertEqual(
            fork_metrics.unique_physical_bytes, 3 * adapter.bytes_per_block
        )

        parent_output = adapter.forward_model(
            paged_model, 10, input_ids=parent_token
        ).output.last_hidden_state[:, -1]
        child_output = adapter.forward_model(
            paged_model, 11, input_ids=child_token
        ).output.last_hidden_state[:, -1]
        expected_parent = reference(
            input_ids=torch.cat((prompt, parent_token), dim=1)
        ).last_hidden_state[:, -1]
        expected_child = reference(
            input_ids=torch.cat((prompt, child_token), dim=1)
        ).last_hidden_state[:, -1]

        self.assertTrue(
            torch.allclose(parent_output, expected_parent, atol=1e-5, rtol=1e-5)
        )
        self.assertTrue(
            torch.allclose(child_output, expected_child, atol=1e-5, rtol=1e-5)
        )
        self.assertFalse(torch.allclose(parent_output, child_output))

    def test_boundary_fork_is_zero_copy_and_cleanup_releases_references(self) -> None:
        model = _tiny_llama(719)
        adapter = _paged_adapter(model)
        self.addCleanup(adapter.close)

        adapter.begin(1)
        adapter.forward_model(
            model,
            1,
            input_ids=torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]),
        )
        fork_metrics = adapter.fork(2, 1)
        forked = _requests(adapter.snapshot())

        self.assertEqual(forked[1].block_ids, forked[2].block_ids)
        self.assertEqual(forked[1].ref_counts, (2, 2))
        self.assertEqual(fork_metrics.copied_bytes, 0)
        self.assertEqual(fork_metrics.newly_reserved_bytes, 0)
        self.assertEqual(
            fork_metrics.logical_bytes, 16 * adapter.bytes_per_token
        )
        self.assertEqual(
            fork_metrics.unique_physical_bytes, 2 * adapter.bytes_per_block
        )

        adapter.end(2)
        parent_only = _requests(adapter.snapshot())
        self.assertEqual(parent_only[1].ref_counts, (1, 1))
        final_metrics = adapter.end(1)
        self.assertEqual(final_metrics.logical_bytes, 0)
        self.assertEqual(final_metrics.unique_physical_bytes, 0)
        self.assertEqual(final_metrics.free_blocks, adapter.num_blocks)
        self.assertEqual(adapter.snapshot().requests, ())

    def test_truncate_privatizes_complete_tail_then_promote_moves_it(self) -> None:
        reference = _tiny_llama(727)
        model = copy.deepcopy(reference)
        adapter = _paged_adapter(model)
        self.addCleanup(adapter.close)

        adapter.begin(1)
        adapter.forward_model(
            model,
            1,
            input_ids=torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]),
        )
        adapter.fork(2, 1)
        adapter.forward_model(
            model,
            2,
            input_ids=torch.tensor([[9, 10, 11, 12]]),
        )

        truncate_metrics = adapter.truncate(2, 6)
        truncated = _requests(adapter.snapshot())
        self.assertEqual(truncated[1].sequence_length, 8)
        self.assertEqual(truncated[2].sequence_length, 6)
        self.assertEqual(truncated[1].block_ids[0], truncated[2].block_ids[0])
        self.assertNotEqual(truncated[1].block_ids[1], truncated[2].block_ids[1])
        self.assertEqual(truncate_metrics.copied_bytes, adapter.bytes_per_block)
        self.assertEqual(
            truncate_metrics.newly_reserved_bytes, adapter.bytes_per_block
        )

        continued = adapter.forward_model(
            model,
            2,
            input_ids=torch.tensor([[13]]),
        ).output.last_hidden_state[:, -1]
        expected = reference(
            input_ids=torch.tensor([[1, 2, 3, 4, 5, 6, 13]])
        ).last_hidden_state[:, -1]
        self.assertTrue(torch.allclose(continued, expected, atol=1e-5, rtol=1e-5))

        promote_metrics = adapter.promote(1, 2)
        promoted = _requests(adapter.snapshot())
        self.assertEqual(set(promoted), {1})
        self.assertEqual(promoted[1].sequence_length, 7)
        self.assertEqual(promoted[1].ref_counts, (1, 1))
        self.assertEqual(promote_metrics.copied_bytes, 0)
        self.assertEqual(promote_metrics.newly_reserved_bytes, 0)
        with self.assertRaisesRegex(ValueError, "unknown paged request 2"):
            adapter.sequence_length(2)

    def test_unknown_ids_capacity_and_corruption_fail_closed(self) -> None:
        model = _tiny_llama(733)
        adapter = _paged_adapter(model, num_blocks=2, max_active_requests=2)
        self.addCleanup(adapter.close)

        with self.assertRaisesRegex(ValueError, "unknown paged request"):
            adapter.sequence_length(99)
        adapter.begin(1)
        with self.assertRaisesRegex(ValueError, "already active"):
            adapter.begin(1)
        adapter.forward_model(
            model,
            1,
            input_ids=torch.tensor([[1, 2, 3, 4, 5, 6]]),
        )
        with self.assertRaisesRegex(MemoryError, "private fork tail"):
            adapter.fork(2, 1)
        self.assertEqual(adapter.sequence_length(1), 6)
        with self.assertRaisesRegex(ValueError, "unknown paged request"):
            adapter.end(2)

        block_id = adapter.snapshot().requests[0].block_ids[0]
        block = adapter.cache._block_manager._id_to_block[block_id]
        block.ref_count += 1
        try:
            with self.assertRaises(PagedStageCorruptionError):
                adapter.metrics()
            with self.assertRaisesRegex(PagedStageCorruptionError, "poisoned"):
                adapter.sequence_length(1)
        finally:
            # Restore the deliberately damaged test fixture so close() can
            # release the real reference without leaking manager state.
            block.ref_count -= 1

    def test_failed_model_forward_discards_the_request(self) -> None:
        model = _tiny_llama(739)
        adapter = _paged_adapter(model)
        self.addCleanup(adapter.close)
        adapter.begin(1)

        class RaisingModel(nn.Module):
            def __init__(self, config) -> None:
                super().__init__()
                self.config = config

            def forward(self, **kwargs):
                raise RuntimeError("synthetic stage failure")

        raising_model = RaisingModel(model.config).eval()
        with self.assertRaisesRegex(RuntimeError, "synthetic stage failure"):
            adapter.forward_model(
                raising_model,
                1,
                input_ids=torch.tensor([[1, 2, 3, 4]]),
            )
        with self.assertRaisesRegex(ValueError, "unknown paged request 1"):
            adapter.sequence_length(1)
        self.assertEqual(adapter.metrics().active_requests, 0)
        self.assertEqual(adapter.metrics().free_blocks, adapter.num_blocks)


if __name__ == "__main__":
    unittest.main()
