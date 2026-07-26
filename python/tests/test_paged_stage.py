from __future__ import annotations

import argparse
import builtins
import copy
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM, LlamaModel

from distributed_runtime.paged_stage import (
    HFPagedStageCache,
    HFPagedStageRuntimeConfig,
    HFPagedStageRunner,
    PagedCacheSnapshot,
    PagedStageCorruptionError,
    add_paged_kv_arguments,
    paged_kv_config_from_args,
)
from distributed_runtime.model import StageModelSpec, StageRunner
from distributed_runtime.executor_abi import StageKVPhysicalAccounting
from distributed_runtime.protocol import Frame, FrameType, TreePrepareRejection
from distributed_runtime.stage import (
    StageBranchLedger,
    end_physical_stage_request,
    fork_physical_stage_request,
    promote_physical_stage_descendant,
    project_tree_capacity,
    speculative_kv_bytes,
    validate_speculative_kv_preflight,
)


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


def _tiny_llama_causal(seed: int) -> LlamaForCausalLM:
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
    model = LlamaForCausalLM(config).eval()
    model.set_attn_implementation("eager")
    return model


def _paged_adapter(
    model: nn.Module,
    *,
    num_blocks: int = 16,
    max_active_requests: int = 4,
    cpu_spill_bytes: int = 0,
) -> HFPagedStageCache:
    return HFPagedStageCache.for_model(
        model,
        attention_backend="eager",
        block_size=4,
        num_blocks=num_blocks,
        max_batch_tokens=16,
        max_active_requests=max_active_requests,
        max_sequence_tokens=min(32, num_blocks * 4),
        cpu_spill_bytes=cpu_spill_bytes,
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


class HFPagedStageImportCompatibilityTests(unittest.TestCase):
    def test_incompatible_version_rejects_cache_and_runner_before_optional_imports(
        self,
    ) -> None:
        optional_import_attempts: list[str] = []
        real_import = builtins.__import__

        def guarded_import(
            name: str,
            globals: dict[str, object] | None = None,
            locals: dict[str, object] | None = None,
            fromlist: tuple[str, ...] = (),
            level: int = 0,
        ) -> object:
            optional_configuration_import = (
                name == "transformers.generation.configuration_utils"
                and "ContinuousBatchingConfig" in fromlist
            )
            if name.startswith(
                "transformers.generation.continuous_batching"
            ) or optional_configuration_import:
                optional_import_attempts.append(name)
                raise AssertionError(
                    "optional continuous-batching import happened before version gate"
                )
            return real_import(name, globals, locals, fromlist, level)

        with (
            patch(
                "distributed_runtime.paged_stage.transformers.__version__",
                "4.57.3",
            ),
            patch("builtins.__import__", side_effect=guarded_import),
            patch(
                "distributed_runtime.paged_stage._load_selective_stage_model"
            ) as load_model,
        ):
            runtime_config = HFPagedStageRuntimeConfig()
            self.assertEqual(runtime_config.device, "cpu")
            parser = argparse.ArgumentParser()
            add_paged_kv_arguments(parser)
            self.assertIsNone(paged_kv_config_from_args(parser.parse_args([])))

            expected_error = (
                r"sealed to transformers 5\.14\.1, found 4\.57\.3; "
                r"continuous-batching symbols were not imported"
            )
            with self.assertRaisesRegex(RuntimeError, expected_error):
                HFPagedStageCache(
                    LlamaConfig(),
                    device="cpu",
                    dtype=torch.float32,
                )
            with self.assertRaisesRegex(RuntimeError, expected_error):
                HFPagedStageRunner(StageModelSpec("unused", 0, 1, 1, 1))

        load_model.assert_not_called()
        self.assertEqual(optional_import_attempts, [])


class HFPagedStageRunnerTests(unittest.TestCase):
    def test_nested_cow_carrier_can_end_then_descendant_promotes_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _write_tiny_checkpoint(temporary, 743)
            spec = StageModelSpec(temporary, 0, 1, 2, 1)
            classic = StageRunner(spec)
            paged = HFPagedStageRunner(
                spec,
                block_size=4,
                num_blocks=16,
                max_batch_tokens=16,
                max_active_requests=4,
                max_sequence_tokens=32,
                cpu_spill_bytes=8192,
            )
            try:
                classic.base.set_attn_implementation("eager")
                config = type(
                    "NestedBranchConfig",
                    (),
                    {
                        "max_speculative_branches": 3,
                        "max_speculative_branch_tokens": 32,
                        "max_speculative_kv_bytes": (
                            4 * paged.paged_cache.bytes_per_block
                        ),
                    },
                )()
                prompt = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
                carrier_token = torch.tensor([[7]], dtype=torch.long)
                descendant_token = torch.tensor([[9]], dtype=torch.long)
                continued_token = torch.tensor([[10]], dtype=torch.long)

                paged.begin(1)
                paged.forward_ids(1, prompt)
                root_before = _requests(paged.paged_cache.snapshot())[1]
                lineage = StageBranchLedger()
                request_metrics = {
                    1: {
                        "frames": 1,
                        "tokens": 6,
                        "compute_ms": 0,
                        "bytes_out": 0,
                    }
                }

                fork_physical_stage_request(
                    child_request_id=2,
                    fork_source_physical_request_id=1,
                    logical_owner_request_id=1,
                    config=config,
                    runner=paged,
                    request_metrics=request_metrics,
                    branch_lineage=lineage,
                )
                paged.forward_ids(2, carrier_token)
                request_metrics[2]["frames"] = 2
                request_metrics[2]["tokens"] = 7
                fork_physical_stage_request(
                    child_request_id=3,
                    fork_source_physical_request_id=2,
                    logical_owner_request_id=1,
                    config=config,
                    runner=paged,
                    request_metrics=request_metrics,
                    branch_lineage=lineage,
                )
                paged.forward_ids(3, descendant_token)
                request_metrics[3]["frames"] = 3
                request_metrics[3]["tokens"] = 8

                nested = _requests(paged.paged_cache.snapshot())
                self.assertEqual(nested[1].block_ids, root_before.block_ids)
                self.assertEqual(nested[1].ref_counts, (3, 1))
                self.assertEqual(nested[2].ref_counts, (3, 1))
                self.assertEqual(nested[3].ref_counts, (3, 1))
                self.assertEqual(lineage.logical_owners, {2: 1, 3: 1})
                self.assertEqual(lineage.fork_sources, {2: 1, 3: 2})
                self.assertEqual(
                    request_metrics[3]["branch_logical_owner_request_id"], 1
                )
                self.assertEqual(
                    request_metrics[3]["branch_fork_source_request_id"], 2
                )
                self.assertEqual(
                    speculative_kv_bytes(paged, lineage.logical_owners),
                    2 * paged.paged_cache.bytes_per_block,
                )
                with self.assertRaisesRegex(ValueError, "logically owns"):
                    end_physical_stage_request(
                        1,
                        operation="END",
                        runner=paged,
                        request_metrics=request_metrics,
                        branch_lineage=lineage,
                    )
                self.assertEqual(paged.sequence_length(1), 6)

                # The physical source is no longer needed. Ending it releases
                # only its references; the descendant stays live and budgeted.
                ended_metrics = end_physical_stage_request(
                    2,
                    operation="END",
                    runner=paged,
                    request_metrics=request_metrics,
                    branch_lineage=lineage,
                )
                self.assertIsNotNone(ended_metrics)
                after_carrier_end = _requests(paged.paged_cache.snapshot())
                self.assertEqual(set(after_carrier_end), {1, 3})
                self.assertEqual(after_carrier_end[1].ref_counts, (2, 1))
                self.assertEqual(after_carrier_end[3].ref_counts, (2, 1))
                self.assertEqual(paged.sequence_length(1), 6)
                self.assertEqual(paged.sequence_length(3), 8)
                self.assertEqual(lineage.logical_owners, {3: 1})
                self.assertEqual(lineage.fork_sources, {3: 2})
                self.assertEqual(
                    speculative_kv_bytes(paged, lineage.logical_owners),
                    paged.paged_cache.bytes_per_block,
                )

                promote_physical_stage_descendant(
                    logical_owner_request_id=1,
                    descendant_request_id=3,
                    config=config,
                    runner=paged,
                    request_metrics=request_metrics,
                    branch_lineage=lineage,
                )
                self.assertEqual(lineage.logical_owners, {})
                self.assertEqual(lineage.fork_sources, {})
                self.assertEqual(paged.sequence_length(1), 8)
                self.assertEqual(request_metrics[1]["tokens"], 8)
                promoted = paged.paged_cache.snapshot()
                promoted_requests = _requests(promoted)
                self.assertEqual(set(promoted_requests), {1})
                self.assertEqual(promoted_requests[1].ref_counts, (1, 1))
                self.assertEqual(promoted.free_blocks, paged.paged_cache.num_blocks - 2)

                actual = paged.forward_ids(1, continued_token)[:, -1]
                classic.begin(99)
                expected = classic.forward_ids(
                    99,
                    torch.cat(
                        (prompt, carrier_token, descendant_token, continued_token),
                        dim=1,
                    ),
                )[:, -1]
                self.assertTrue(
                    torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
                )
            finally:
                classic.close()
                paged.close()

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
                cpu_spill_bytes=8192,
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
                        "bounded-cpu-kv-spill",
                        "integrity-checked-atomic-kv-restore",
                    }.issubset(features)
                )
                self.assertNotIn("exact-tree-verify-batching", features)
                self.assertEqual(paged.executor_manifest.max_batch_size, 1)
                self.assertEqual(
                    paged.executor_manifest.kv_format,
                    "transformers-paged-cow-v1",
                )
                execution = paged.execution_snapshot()
                self.assertTrue(execution["cpuSpill"]["supported"])
                self.assertTrue(execution["cpuSpill"]["enabled"])
                self.assertEqual(execution["cpuSpill"]["maxBytes"], 8192)
                self.assertEqual(execution["cpuSpillState"]["usedBytes"], 0)

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
                capacity_config = type(
                    "CapacityConfig",
                    (),
                    {
                        "max_speculative_branches": 4,
                        "max_speculative_branch_tokens": 32,
                        "max_speculative_kv_bytes": (
                            3 * paged.paged_cache.bytes_per_block
                        ),
                    },
                )()
                physical_projection = project_tree_capacity(
                    (1, 3),
                    parent_request_id=1,
                    config=capacity_config,
                    runner=paged,
                    branch_parents={},
                )
                self.assertEqual(
                    physical_projection.projected_kv_bytes,
                    3 * paged.paged_cache.bytes_per_block,
                )
                self.assertEqual(
                    physical_projection.rejection,
                    TreePrepareRejection.NONE,
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
                self.assertEqual(
                    speculative_kv_bytes(paged, {2: 1}),
                    paged.paged_cache.bytes_per_block,
                )
                growth_config = type(
                    "GrowthConfig",
                    (),
                    {
                        "max_speculative_kv_bytes": (
                            2 * paged.paged_cache.bytes_per_block
                        ),
                    },
                )()
                validate_speculative_kv_preflight(
                    (Frame(FrameType.VERIFY, 0, 2, 0, 3, 0, b""),),
                    growth_config,
                    paged,
                    {2: 1},
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
    def test_runtime_config_seals_bounded_integrity_checked_cpu_spill(self) -> None:
        runtime = HFPagedStageRuntimeConfig(cpu_spill_bytes=4096)
        document = runtime.to_document()

        self.assertEqual(document["schema"], "mycellios-hf-paged-stage/2")
        self.assertEqual(document["cpuSpillBytes"], 4096)
        self.assertEqual(
            document["cpuSpill"],
            {
                "supported": True,
                "enabled": True,
                "maxBytes": 4096,
                "storedPayloadLimitBytes": 4096,
                "restoreWorkspaceUpperBoundBytes": 4096,
                "maxCpuPayloadUpperBoundBytes": 8192,
                "budgetScope": "stored-tensor-payload",
                "rssBounded": False,
                "allocatorOverheadIncluded": False,
                "restoreWorkspaceIncluded": False,
                "transferByteCounters": "successful-full-payloads-only",
                "policy": "integrity-checked-request-lru",
                "restore": "verify-source-allocate-copy-publish",
            },
        )
        self.assertNotEqual(
            runtime.configuration_id,
            HFPagedStageRuntimeConfig(cpu_spill_bytes=0).configuration_id,
        )

    def test_explicit_spill_restore_preserves_exact_continuation(self) -> None:
        reference = _tiny_llama_causal(691)
        model = copy.deepcopy(reference)
        adapter = _paged_adapter(model, cpu_spill_bytes=8192)
        self.addCleanup(adapter.close)
        prompt = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
        continuation = torch.tensor([[7]], dtype=torch.long)

        adapter.begin(1)
        adapter.forward_model(model, 1, input_ids=prompt)
        spill_metrics = adapter.spill(1)
        spilled = _requests(adapter.snapshot())[1]

        self.assertEqual(spilled.residency, "cpu-spill")
        self.assertEqual(spilled.block_ids, ())
        self.assertEqual(spilled.spill_bytes, 2 * adapter.bytes_per_block)
        self.assertEqual(spill_metrics.spilled_requests, 1)
        self.assertEqual(spill_metrics.free_blocks, adapter.num_blocks)

        observed = adapter.forward_model(
            model,
            1,
            input_ids=continuation,
        ).output.logits[:, -1]
        expected = reference(
            input_ids=torch.cat((prompt, continuation), dim=1)
        ).logits[:, -1]
        metrics = adapter.metrics()

        self.assertTrue(torch.allclose(observed, expected, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.equal(observed.argmax(dim=-1), expected.argmax(dim=-1)))
        self.assertEqual(_requests(adapter.snapshot())[1].residency, "device")
        self.assertEqual(metrics.spilled_requests, 0)
        self.assertEqual(metrics.spilled_bytes, 0)
        self.assertEqual(metrics.spill_count, 1)
        self.assertEqual(metrics.restore_count, 1)

    def test_spill_restore_rejoins_shared_complete_prefix(self) -> None:
        model = _tiny_llama(693)
        adapter = _paged_adapter(model, cpu_spill_bytes=8192)
        self.addCleanup(adapter.close)

        adapter.begin(1)
        adapter.forward_model(
            model,
            1,
            input_ids=torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]),
        )
        adapter.fork(2, 1)
        adapter.spill(2)
        after_spill = _requests(adapter.snapshot())

        self.assertEqual(after_spill[1].ref_counts, (1, 1))
        self.assertEqual(after_spill[2].residency, "cpu-spill")

        adapter.restore(2)
        restored = _requests(adapter.snapshot())
        self.assertEqual(restored[1].block_ids, restored[2].block_ids)
        self.assertEqual(restored[1].ref_counts, (2, 2))
        self.assertEqual(restored[2].ref_counts, (2, 2))

    def test_auto_spill_swaps_lru_requests_within_hard_limit(self) -> None:
        reference = _tiny_llama_causal(697)
        model = copy.deepcopy(reference)
        adapter = _paged_adapter(
            model,
            num_blocks=3,
            max_active_requests=2,
            cpu_spill_bytes=4096,
        )
        self.addCleanup(adapter.close)
        first_prompt = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)

        adapter.begin(1)
        adapter.forward_model(model, 1, input_ids=first_prompt)
        adapter.begin(2)
        adapter.forward_model(
            model,
            2,
            input_ids=torch.tensor([[11, 12, 13, 14]], dtype=torch.long),
        )
        adapter.forward_model(
            model,
            2,
            input_ids=torch.tensor([[15]], dtype=torch.long),
        )
        after_first_swap = _requests(adapter.snapshot())
        self.assertEqual(after_first_swap[1].residency, "cpu-spill")
        self.assertEqual(after_first_swap[2].residency, "device")

        observed = adapter.forward_model(
            model,
            1,
            input_ids=torch.tensor([[7]], dtype=torch.long),
        ).output.logits[:, -1]
        expected = reference(
            input_ids=torch.tensor([[1, 2, 3, 4, 5, 6, 7]], dtype=torch.long)
        ).logits[:, -1]
        after_second_swap = _requests(adapter.snapshot())

        self.assertTrue(torch.allclose(observed, expected, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.equal(observed.argmax(dim=-1), expected.argmax(dim=-1)))
        self.assertEqual(after_second_swap[1].residency, "device")
        self.assertEqual(after_second_swap[2].residency, "cpu-spill")
        self.assertLessEqual(
            adapter.metrics().spilled_bytes,
            adapter.cpu_spill_limit_bytes,
        )
        self.assertEqual(adapter.metrics().spill_count, 2)
        self.assertEqual(adapter.metrics().restore_count, 1)

    def test_restore_reuses_its_exact_spill_slot_without_deadlock(self) -> None:
        reference = _tiny_llama_causal(698)
        model = copy.deepcopy(reference)
        adapter = _paged_adapter(
            model,
            num_blocks=3,
            max_active_requests=2,
            cpu_spill_bytes=1024,
        )
        self.addCleanup(adapter.close)
        first_prompt = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)

        adapter.begin(1)
        adapter.forward_model(model, 1, input_ids=first_prompt)
        self.assertEqual(2 * adapter.bytes_per_block, adapter.cpu_spill_limit_bytes)
        adapter.spill(1)
        adapter.begin(2)
        adapter.forward_model(
            model,
            2,
            input_ids=torch.tensor([[11, 12, 13, 14, 15, 16]], dtype=torch.long),
        )
        before_projection = adapter.snapshot()
        self.assertEqual(
            adapter.project_request_incremental_physical_bytes(1, 1),
            0,
        )
        self.assertEqual(adapter.snapshot(), before_projection)

        observed = adapter.forward_model(
            model,
            1,
            input_ids=torch.tensor([[7]], dtype=torch.long),
        ).output.logits[:, -1]
        expected = reference(
            input_ids=torch.tensor([[1, 2, 3, 4, 5, 6, 7]], dtype=torch.long)
        ).logits[:, -1]
        requests = _requests(adapter.snapshot())
        metrics = adapter.metrics()

        self.assertTrue(torch.allclose(observed, expected, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.equal(observed.argmax(dim=-1), expected.argmax(dim=-1)))
        self.assertEqual(requests[1].residency, "device")
        self.assertEqual(requests[2].residency, "cpu-spill")
        self.assertEqual(metrics.spilled_bytes, adapter.cpu_spill_limit_bytes)
        self.assertEqual(metrics.restore_workspace_bytes, 0)
        self.assertEqual(
            metrics.peak_restore_workspace_bytes,
            adapter.cpu_spill_limit_bytes,
        )
        self.assertEqual(
            metrics.current_cpu_payload_bytes,
            adapter.cpu_spill_limit_bytes,
        )
        self.assertEqual(
            metrics.peak_cpu_payload_bytes,
            2 * adapter.cpu_spill_limit_bytes,
        )
        self.assertEqual(
            metrics.spill_bytes_transferred,
            2 * adapter.cpu_spill_limit_bytes,
        )
        self.assertEqual(
            metrics.restore_bytes_transferred,
            adapter.cpu_spill_limit_bytes,
        )
        self.assertGreater(metrics.spill_time_ns, 0)
        self.assertGreater(metrics.restore_time_ns, 0)

    def test_failed_restore_rolls_back_all_residency_changes(self) -> None:
        model = _tiny_llama(700)
        adapter = _paged_adapter(
            model,
            num_blocks=3,
            max_active_requests=2,
            cpu_spill_bytes=1024,
        )
        self.addCleanup(adapter.close)

        adapter.begin(1)
        adapter.forward_model(
            model,
            1,
            input_ids=torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long),
        )
        adapter.spill(1)
        adapter.begin(2)
        adapter.forward_model(
            model,
            2,
            input_ids=torch.tensor([[11, 12, 13, 14, 15, 16]], dtype=torch.long),
        )
        before = adapter.snapshot()
        original_copy = adapter._copy_cpu_blocks_to_device
        calls = 0

        def fail_once(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("synthetic restore copy failure")
            return original_copy(*args, **kwargs)

        with patch.object(adapter, "_copy_cpu_blocks_to_device", side_effect=fail_once):
            with self.assertRaisesRegex(RuntimeError, "synthetic restore copy failure"):
                adapter.restore(1)

        after = adapter.snapshot()
        self.assertEqual(
            {
                request.request_id: request.residency for request in after.requests
            },
            {
                request.request_id: request.residency for request in before.requests
            },
        )
        self.assertEqual(after.free_blocks, before.free_blocks)
        self.assertEqual(after.spilled_bytes, before.spilled_bytes)
        self.assertEqual(adapter.metrics().restore_workspace_bytes, 0)
        self.assertEqual(adapter.metrics().restore_failures, 1)
        self.assertFalse(adapter._poisoned)

    def test_available_capacity_counts_only_reclaimable_unprotected_blocks(
        self,
    ) -> None:
        model = _tiny_llama(701)
        adapter = _paged_adapter(
            model,
            num_blocks=3,
            max_active_requests=2,
            cpu_spill_bytes=512,
        )
        self.addCleanup(adapter.close)
        self.assertEqual(adapter.bytes_per_block, 512)

        adapter.begin(1)
        adapter.forward_model(
            model,
            1,
            input_ids=torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        )
        adapter.begin(2)
        adapter.forward_model(
            model,
            2,
            input_ids=torch.tensor([[11, 12, 13, 14]], dtype=torch.long),
        )
        self.assertEqual(adapter.cache.get_num_free_blocks(), 1)
        access_before_projection = dict(adapter._last_access)
        available_before_projection = adapter.available_physical_bytes()
        self.assertEqual(adapter.sequence_length(1), 4)
        self.assertEqual(
            adapter.project_request_incremental_physical_bytes(1, 1),
            adapter.bytes_per_block,
        )
        self.assertEqual(adapter._last_access, access_before_projection)
        self.assertEqual(
            adapter.available_physical_bytes(),
            available_before_projection,
        )
        self.assertEqual(
            available_before_projection,
            2 * adapter.bytes_per_block,
        )

        adapter.close()
        shared_model = _tiny_llama(703)
        shared = _paged_adapter(
            shared_model,
            num_blocks=3,
            max_active_requests=2,
            cpu_spill_bytes=512,
        )
        self.addCleanup(shared.close)
        shared.begin(10)
        shared.forward_model(
            shared_model,
            10,
            input_ids=torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        )
        shared.fork(11, 10)
        self.assertEqual(shared.cache.get_num_free_blocks(), 2)
        self.assertEqual(
            shared.available_physical_bytes(),
            2 * shared.bytes_per_block,
        )

    def test_promote_resident_child_drops_spilled_parent_without_restore(self) -> None:
        reference = _tiny_llama_causal(704)
        model = copy.deepcopy(reference)
        adapter = _paged_adapter(
            model,
            num_blocks=3,
            max_active_requests=2,
            cpu_spill_bytes=1024,
        )
        self.addCleanup(adapter.close)
        prompt = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
        child_token = torch.tensor([[7]], dtype=torch.long)
        continuation = torch.tensor([[8]], dtype=torch.long)

        adapter.begin(1)
        adapter.forward_model(model, 1, input_ids=prompt)
        adapter.fork(2, 1)
        adapter.forward_model(model, 2, input_ids=child_token)
        adapter.spill(1)
        before_restore_count = adapter.metrics().restore_count

        adapter.promote(1, 2)
        promoted = _requests(adapter.snapshot())
        self.assertEqual(set(promoted), {1})
        self.assertEqual(promoted[1].residency, "device")
        self.assertEqual(adapter.metrics().restore_count, before_restore_count)
        self.assertEqual(adapter.metrics().spilled_bytes, 0)

        observed = adapter.forward_model(
            model,
            1,
            input_ids=continuation,
        ).output.logits[:, -1]
        expected = reference(
            input_ids=torch.cat((prompt, child_token, continuation), dim=1)
        ).logits[:, -1]
        self.assertTrue(torch.allclose(observed, expected, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.equal(observed.argmax(dim=-1), expected.argmax(dim=-1)))

    def test_promote_spilled_child_without_device_residency_ping_pong(self) -> None:
        reference = _tiny_llama_causal(702)
        model = copy.deepcopy(reference)
        adapter = _paged_adapter(model, cpu_spill_bytes=8192)
        self.addCleanup(adapter.close)
        prompt = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
        child_token = torch.tensor([[7]], dtype=torch.long)
        continuation = torch.tensor([[8]], dtype=torch.long)

        adapter.begin(1)
        adapter.forward_model(model, 1, input_ids=prompt)
        adapter.fork(2, 1)
        adapter.forward_model(model, 2, input_ids=child_token)
        adapter.spill(2)
        restore_count = adapter.metrics().restore_count

        adapter.promote(1, 2)
        promoted = _requests(adapter.snapshot())
        self.assertEqual(set(promoted), {1})
        self.assertEqual(promoted[1].residency, "cpu-spill")
        self.assertEqual(adapter.metrics().restore_count, restore_count)

        observed = adapter.forward_model(
            model,
            1,
            input_ids=continuation,
        ).output.logits[:, -1]
        expected = reference(
            input_ids=torch.cat((prompt, child_token, continuation), dim=1)
        ).logits[:, -1]
        self.assertTrue(torch.allclose(observed, expected, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.equal(observed.argmax(dim=-1), expected.argmax(dim=-1)))

    def test_allocator_release_failure_after_mutation_poisons_immediately(
        self,
    ) -> None:
        model = _tiny_llama(706)
        adapter = _paged_adapter(model, cpu_spill_bytes=8192)
        self.addCleanup(adapter.close)
        adapter.begin(1)
        adapter.forward_model(
            model,
            1,
            input_ids=torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        )
        original_free = adapter.cache.free_blocks

        def free_then_fail(request_id):
            original_free(request_id)
            raise RuntimeError("synthetic post-release failure")

        with patch.object(adapter.cache, "free_blocks", side_effect=free_then_fail):
            with self.assertRaisesRegex(
                PagedStageCorruptionError,
                "after changing allocator state",
            ):
                adapter.spill(1)

        self.assertTrue(adapter._poisoned)
        self.assertIn(1, adapter._spilled_requests)
        self.assertNotIn(1, adapter._requests)
        with self.assertRaisesRegex(PagedStageCorruptionError, "poisoned"):
            adapter.metrics()

    def test_spill_limit_and_integrity_check_fail_closed(self) -> None:
        model = _tiny_llama(699)
        adapter = _paged_adapter(model, cpu_spill_bytes=512)
        self.addCleanup(adapter.close)
        adapter.begin(1)
        adapter.forward_model(
            model,
            1,
            input_ids=torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long),
        )

        with self.assertRaisesRegex(MemoryError, "bounded CPU spill capacity"):
            adapter.spill(1)
        unchanged = _requests(adapter.snapshot())[1]
        self.assertEqual(unchanged.residency, "device")
        self.assertEqual(adapter.metrics().spilled_bytes, 0)

        adapter.cpu_spill_limit_bytes = 8192
        adapter.spill(1)
        spilled = adapter._spilled_requests[1]
        raw = spilled.key_blocks[0].view(torch.uint8)
        raw[0] ^= 1
        with self.assertRaisesRegex(
            PagedStageCorruptionError,
            "integrity check failed",
        ):
            adapter.restore(1)
        with self.assertRaisesRegex(PagedStageCorruptionError, "poisoned"):
            adapter.metrics()

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
