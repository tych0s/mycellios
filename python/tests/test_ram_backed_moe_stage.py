from __future__ import annotations

import gc
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F
from transformers import (
    Glm4MoeConfig,
    Glm4MoeForCausalLM,
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
)

from distributed_runtime.engine import (
    DistributedPipelineEngine,
    GenerationInput,
    PipelineEngineConfig,
)
from distributed_runtime.model import (
    StageModelSpec,
    StageRunner,
    model_artifact_reference,
)
from distributed_runtime.protocol import TensorCodec
from distributed_runtime.ram_backed_moe_runtime import RamBackedMoeRuntimeConfig
from distributed_runtime.ram_backed_moe_stage import (
    RamBackedMoeExperts,
    RamBackedMoeStageRunner,
)
from distributed_runtime.ram_expert_cache import (
    ExpertKey,
    ExpertRecord,
    MacroStageExpertInventory,
    PredictiveCacheConfig,
    RamBackedExpertScheduler,
)
from distributed_runtime.torch_ram_expert_store import (
    TorchExpertBundle,
    TorchRamExpertStore,
)


def _cache_for_one_active_one_prefetch(expert_bytes: int) -> PredictiveCacheConfig:
    return PredictiveCacheConfig(
        capacity_bytes=expert_bytes * 2,
        prefetch_reserve_bytes=expert_bytes,
        pcie_bandwidth_gbytes_per_second=8,
    )


def _reference_tokens(model, input_ids: torch.Tensor, count: int) -> list[int]:
    result: list[int] = []
    cache = None
    current = input_ids
    with torch.inference_mode():
        for _ in range(count):
            output = model(input_ids=current, past_key_values=cache, use_cache=True)
            cache = output.past_key_values
            token = int(torch.argmax(output.logits[:, -1, :], dim=-1).item())
            result.append(token)
            current = torch.tensor([[token]], dtype=torch.long)
    return result


def _pipeline_tokens(
    first: RamBackedMoeStageRunner,
    last: RamBackedMoeStageRunner,
    input_ids: torch.Tensor,
    count: int,
    *,
    request_id: int,
) -> list[int]:
    first.begin(request_id)
    last.begin(request_id)
    result: list[int] = []
    try:
        hidden = first.forward_ids(request_id, input_ids)
        for index in range(count):
            _, token = last.forward_hidden(request_id, hidden)
            assert isinstance(token, int)
            result.append(token)
            if index + 1 < count:
                hidden = first.forward_ids(
                    request_id,
                    torch.tensor([[token]], dtype=torch.long),
                )
    finally:
        first.end(request_id)
        last.end(request_id)
    return result


class RamBackedMoeStageRunnerTests(unittest.TestCase):
    def test_strict_cuda_unavailable_fails_before_snapshot_or_expert_load(self) -> None:
        spec = StageModelSpec("unused", 0, 1, 1, 1)
        cache = PredictiveCacheConfig(
            capacity_bytes=2,
            prefetch_reserve_bytes=1,
            pcie_bandwidth_gbytes_per_second=8,
        )
        with patch("torch.cuda.is_available", return_value=False), patch(
            "distributed_runtime.ram_backed_moe_stage.resolve_model_snapshot",
            side_effect=AssertionError("snapshot resolution must remain unreachable"),
        ) as resolve, patch(
            "distributed_runtime.ram_backed_moe_stage.load_local_safetensors_moe_stage",
            side_effect=AssertionError("expert loading must remain unreachable"),
        ) as load:
            with self.assertRaisesRegex(RuntimeError, "CUDA device is unavailable"):
                RamBackedMoeStageRunner(
                    spec,
                    cache,
                    device="cuda:0",
                    allow_cpu_fallback=False,
                )
        resolve.assert_not_called()
        load.assert_not_called()

    def test_prediction_fp_fn_are_counted_once_per_layer_without_route_history(self) -> None:
        bundles: dict[ExpertKey, TorchExpertBundle] = {}
        records: list[ExpertRecord] = []
        for expert in range(3):
            key = ExpertKey(0, expert)
            bundle = TorchExpertBundle(
                key=key,
                content_id=f"telemetry:{expert}",
                tensors=(
                    ("gate_proj.weight", torch.eye(2) * (expert + 1)),
                    ("up_proj.weight", torch.ones(2, 2)),
                    ("down_proj.weight", torch.eye(2)),
                ),
            )
            bundles[key] = bundle
            records.append(ExpertRecord(key, bundle.byte_size, bundle.content_id))
        inventory = MacroStageExpertInventory(tuple(records))
        cache_config = _cache_for_one_active_one_prefetch(records[0].byte_size)
        store = TorchRamExpertStore(
            inventory,
            bundles,
            cache_config,
            device="cpu",
            pin_memory=False,
        )
        scheduler = RamBackedExpertScheduler(
            inventory,
            cache_config,
            weight_backend=store,
        )

        class FixedCoordinator:
            prefetch_deadline_ms = 1_000.0

            @staticmethod
            def prediction_for(layer: int) -> tuple[int, ...]:
                assert layer == 0
                return (0, 2)

            @staticmethod
            def after_authoritative_route(layer: int, experts) -> None:
                assert layer == 0

        module = RamBackedMoeExperts(
            layer=0,
            num_experts=3,
            hidden_dim=2,
            intermediate_dim=2,
            act_fn=F.silu,
            scheduler=scheduler,
            store=store,
            coordinator=FixedCoordinator(),  # type: ignore[arg-type]
        )
        hidden = torch.tensor([[0.25, -0.5]], dtype=torch.float32)
        selected = torch.tensor([[0, 1]], dtype=torch.long)
        weights = torch.tensor([[0.6, 0.4]], dtype=torch.float32)

        for _ in range(2):
            output = module(hidden, selected, weights)
            self.assertTrue(bool(torch.isfinite(output).all().item()))

        # Prediction {0, 2} versus route {0, 1}: one FP and one FN per
        # layer call, independent of the serial per-expert resolution count.
        self.assertEqual(module.routed_layer_calls, 2)
        self.assertEqual(module.prediction_false_positives, 2)
        self.assertEqual(module.prediction_false_negatives, 2)
        self.assertFalse(hasattr(module, "resolutions"))

    def test_single_stage_meta_materialization_preserves_tied_embedding_alias(self) -> None:
        torch.manual_seed(697)
        config = Qwen3MoeConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            moe_intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            num_experts=4,
            num_experts_per_tok=2,
            max_position_embeddings=64,
            tie_word_embeddings=True,
            architectures=["Qwen3MoeForCausalLM"],
        )
        model = Qwen3MoeForCausalLM(config).eval()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model.save_pretrained(root, safe_serialization=True)
            del model
            gc.collect()
            expert_bytes = (16 * 32 * 3) * 4
            runner = RamBackedMoeStageRunner(
                StageModelSpec(str(root), 0, 1, 1, 1),
                _cache_for_one_active_one_prefetch(expert_bytes),
                device="cpu",
            )
            with self.assertRaisesRegex(ValueError, "adapter identity mismatch"):
                RamBackedMoeStageRunner(
                    StageModelSpec(str(root), 0, 1, 1, 1),
                    _cache_for_one_active_one_prefetch(expert_bytes),
                    device="cpu",
                    expected_adapter_id="transformers-glm4-moe-v1",
                )

        self.assertIsNotNone(runner.head)
        self.assertEqual(
            runner.base.embed_tokens.weight.data_ptr(),
            runner.head.weight.data_ptr(),
        )

    def test_resident_budget_fails_before_meta_parameters_reach_device(self) -> None:
        torch.manual_seed(699)
        config = Qwen3MoeConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            moe_intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            num_experts=4,
            num_experts_per_tok=2,
            max_position_embeddings=64,
            architectures=["Qwen3MoeForCausalLM"],
        )
        model = Qwen3MoeForCausalLM(config).eval()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model.save_pretrained(root, safe_serialization=True)
            del model
            gc.collect()
            expert_bytes = (16 * 32 * 3) * 4
            with patch.object(
                torch.nn.Module,
                "to_empty",
                side_effect=AssertionError("device allocation must remain unreachable"),
            ) as to_empty:
                with self.assertRaisesRegex(MemoryError, "resident parameter budget"):
                    RamBackedMoeStageRunner(
                        StageModelSpec(str(root), 0, 1, 1, 1),
                        _cache_for_one_active_one_prefetch(expert_bytes),
                        device="cpu",
                        expected_resident_parameter_bytes=1,
                    )
            to_empty.assert_not_called()

    def test_tiny_qwen_pipeline_matches_reference_with_two_expert_working_set(self) -> None:
        torch.manual_seed(701)
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
        prompt = torch.tensor([[1, 5, 9, 3]], dtype=torch.long)
        expected = _reference_tokens(model, prompt, 4)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model.save_pretrained(root, safe_serialization=True)
            del model
            gc.collect()
            # 3 canonical matrices: gate/up [16,32], down [32,16].
            expert_bytes = (16 * 32 * 3) * 4
            cache = _cache_for_one_active_one_prefetch(expert_bytes)
            resident_first = StageRunner(StageModelSpec(str(root), 0, 1, 2, 1))
            resident_last = StageRunner(StageModelSpec(str(root), 1, 2, 2, 1))
            first = RamBackedMoeStageRunner(
                StageModelSpec(str(root), 0, 1, 2, 1),
                cache,
                device="cpu",
            )
            last = RamBackedMoeStageRunner(
                StageModelSpec(str(root), 1, 2, 2, 1),
                cache,
                device="cpu",
            )

            resident_first.begin(700)
            resident_last.begin(700)
            first.begin(700)
            last.begin(700)
            try:
                resident_hidden = resident_first.forward_ids(700, prompt)
                ram_hidden = first.forward_ids(700, prompt)
                torch.testing.assert_close(
                    ram_hidden,
                    resident_hidden,
                    rtol=0,
                    atol=0,
                )
                resident_output, resident_token = resident_last.forward_hidden(
                    700, resident_hidden
                )
                ram_output, ram_token = last.forward_hidden(700, ram_hidden)
                torch.testing.assert_close(
                    ram_output,
                    resident_output,
                    rtol=0,
                    atol=0,
                )
                assert resident_last.head is not None and last.head is not None
                with torch.inference_mode():
                    resident_logits = resident_last.head(resident_output[:, -1:, :])
                    ram_logits = last.head(ram_output[:, -1:, :])
                torch.testing.assert_close(
                    ram_logits,
                    resident_logits,
                    rtol=0,
                    atol=0,
                )
                self.assertEqual(ram_token, resident_token)
            finally:
                resident_first.end(700)
                resident_last.end(700)
                first.end(700)
                last.end(700)

            actual = _pipeline_tokens(first, last, prompt, 4, request_id=701)

        self.assertEqual(actual, expected)
        self.assertEqual(first.compute_device.type, "cpu")
        self.assertIn("cpu", first.executor_manifest.device_kinds)
        self.assertIn("ram-authoritative-routed-experts", first.executor_manifest.features)
        self.assertNotIn("resident-expert-set", first.executor_manifest.features)
        self.assertEqual(first.ram_parameter_bytes, expert_bytes * 4)
        self.assertEqual(first.loaded_backing_bytes, first.ram_parameter_bytes)
        self.assertEqual(first.bounded_pinned_staging_reserve_bytes, 0)
        self.assertGreater(first.resident_streaming_transient_bytes, 0)
        self.assertEqual(
            first.host_ram_steady_state_bytes,
            first.ram_parameter_bytes,
        )
        self.assertEqual(
            first.host_ram_peak_upper_bound_bytes,
            first.ram_parameter_bytes
            + first.resident_streaming_transient_bytes,
        )
        self.assertEqual(first.resident_parameter_bytes, first.parameter_bytes)
        self.assertIsInstance(first.base.layers[0].mlp.experts, RamBackedMoeExperts)
        self.assertFalse(
            any("gate_up_proj" in name for name, _ in first.base.named_parameters())
        )
        snapshot = first.expert_store.snapshot()
        self.assertLessEqual(snapshot.active_bytes, expert_bytes)
        self.assertLessEqual(snapshot.prefetch_bytes, expert_bytes)
        # A union resolve of the top-2 route cannot fit the active budget. The
        # serial exact path passes and normally consumes the second via prefetch.
        self.assertGreater(first.execution_snapshot().prefetch_hits, 0)

    def test_physical_engine_root_and_spawned_child_both_use_ram_runner(self) -> None:
        torch.manual_seed(705)
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
        prompt = torch.tensor([[1, 4, 7, 2]], dtype=torch.long)
        expected = _reference_tokens(model, prompt, 4)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model.save_pretrained(root, safe_serialization=True)
            del model
            gc.collect()
            artifact = model_artifact_reference(str(root))
            expert_bytes = (16 * 32 * 3) * 4
            cache = _cache_for_one_active_one_prefetch(expert_bytes)

            stage_runtime: list[RamBackedMoeRuntimeConfig] = []
            for layer_start in (0, 1):
                probe = RamBackedMoeStageRunner(
                    StageModelSpec(
                        str(root),
                        layer_start,
                        layer_start + 1,
                        2,
                        1,
                        artifact_identity=artifact.identity,
                    ),
                    cache,
                    device="cpu",
                )
                try:
                    stage_runtime.append(
                        RamBackedMoeRuntimeConfig.for_cpu_certification(
                            artifact_identity=artifact.identity,
                            adapter_id="transformers-qwen3-moe-v1",
                            cache_config=cache,
                            expected_resident_parameter_bytes=(
                                probe.resident_parameter_bytes
                            ),
                            expected_total_routed_expert_bytes=(
                                probe.ram_parameter_bytes
                            ),
                            expected_largest_expert_bytes=(
                                probe.largest_expert_bytes
                            ),
                            expected_resident_streaming_transient_bytes=(
                                probe.resident_streaming_transient_bytes
                            ),
                            expected_host_ram_peak_upper_bound_bytes=(
                                probe.host_ram_peak_upper_bound_bytes
                            ),
                        )
                    )
                finally:
                    probe.close()

            engine = DistributedPipelineEngine(
                PipelineEngineConfig(
                    model_name=str(root),
                    boundaries=(0, 1, 2),
                    codec=TensorCodec.FP32,
                    threads_per_stage=1,
                    startup_timeout_seconds=60,
                    socket_timeout_seconds=60,
                    pipeline_snapshot_identity=artifact.snapshot_identity,
                    ram_backed_moe_stages=tuple(stage_runtime),
                    route_probe_interval_seconds=0,
                    root_batch_window_ms=0,
                )
            )
            try:
                self.assertEqual(
                    engine._require_runner().loader,
                    "selective-safetensors-ram-backed-moe",
                )
                self.assertEqual(len(engine._processes), 1)
                outputs = engine.generate([GenerationInput(705, prompt, 4)])
                self.assertEqual(list(outputs[0].token_ids), expected)
            finally:
                engine.close()

        child_metrics = [
            metric
            for metric in engine.stage_metrics
            if metric.get("stage") == 1 and "request_id" in metric
        ]
        self.assertEqual(len(child_metrics), 1, repr(engine.stage_metrics))
        self.assertEqual(
            child_metrics[0]["loader"],
            "selective-safetensors-ram-backed-moe",
        )

    def test_glm_dense_shared_router_and_kv_rollback_remain_exact(self) -> None:
        torch.manual_seed(709)
        config = Glm4MoeConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            moe_intermediate_size=16,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            n_routed_experts=4,
            num_experts_per_tok=2,
            n_shared_experts=1,
            first_k_dense_replace=1,
            max_position_embeddings=64,
            architectures=["Glm4MoeForCausalLM"],
            num_mtp_layers=0,
        )
        model = Glm4MoeForCausalLM(config).eval()
        prompt = torch.tensor([[2, 6, 1]], dtype=torch.long)
        expected = _reference_tokens(model, prompt, 3)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model.save_pretrained(root, safe_serialization=True)
            del model
            gc.collect()
            expert_bytes = (16 * 32 * 3) * 4
            cache = _cache_for_one_active_one_prefetch(expert_bytes)
            first = RamBackedMoeStageRunner(
                StageModelSpec(str(root), 0, 2, 3, 1),
                cache,
                device="cpu",
            )
            last = RamBackedMoeStageRunner(
                StageModelSpec(str(root), 2, 3, 3, 1),
                cache,
                device="cpu",
            )
            actual = _pipeline_tokens(first, last, prompt, 3, request_id=709)

            first.begin(710)
            last.begin(710)
            hidden = first.forward_ids(710, prompt)
            last.forward_hidden(710, hidden)
            self.assertEqual(first.sequence_length(710), prompt.shape[1])
            self.assertEqual(last.sequence_length(710), prompt.shape[1])
            first.truncate(710, 1)
            last.truncate(710, 1)
            self.assertEqual(first.sequence_length(710), 1)
            self.assertEqual(last.sequence_length(710), 1)
            first.end(710)
            last.end(710)

        self.assertEqual(actual, expected)
        self.assertTrue(hasattr(first.base.layers[0].mlp, "gate_proj"))
        sparse = first.base.layers[1].mlp
        self.assertIsInstance(sparse.experts, RamBackedMoeExperts)
        self.assertTrue(hasattr(sparse, "gate"))
        self.assertTrue(hasattr(sparse, "shared_experts"))
        self.assertGreater(first.resident_parameter_bytes, 0)
        self.assertGreater(first.execution_snapshot().routed_tokens, 0)

    def test_qwen_physical_batch_preserves_request_local_kv_and_cpu_transport(self) -> None:
        torch.manual_seed(719)
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
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model.save_pretrained(root, safe_serialization=True)
            del model
            gc.collect()
            expert_bytes = (16 * 32 * 3) * 4
            cache = _cache_for_one_active_one_prefetch(expert_bytes)
            first = RamBackedMoeStageRunner(
                StageModelSpec(str(root), 0, 1, 2, 1), cache, device="cpu"
            )
            last = RamBackedMoeStageRunner(
                StageModelSpec(str(root), 1, 2, 2, 1), cache, device="cpu"
            )
            for request_id in (801, 802):
                first.begin(request_id)
                last.begin(request_id)
            prompts = (
                torch.tensor([[1, 2, 3]], dtype=torch.long),
                torch.tensor([[4, 5, 6]], dtype=torch.long),
            )
            hidden = first.forward_ids_batch((801, 802), prompts)
            self.assertTrue(all(value.device.type == "cpu" for value in hidden))
            self.assertTrue(all(value.dtype == torch.float32 for value in hidden))
            results = last.forward_hidden_batch((801, 802), hidden)
            self.assertEqual(len(results), 2)
            self.assertTrue(all(output.device.type == "cpu" for output, _ in results))
            self.assertEqual(first.sequence_length(801), 3)
            self.assertEqual(first.sequence_length(802), 3)
            self.assertEqual(last.sequence_length(801), 3)
            self.assertEqual(last.sequence_length(802), 3)
            for request_id in (801, 802):
                first.end(request_id)
                last.end(request_id)


if __name__ == "__main__":
    unittest.main()
