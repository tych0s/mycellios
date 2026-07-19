from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

from safetensors.torch import save_file
import torch
import torch.distributed as distributed

from distributed_runtime.cell_fixture_compiler import compile_hf_llama_cell_fixture
from distributed_runtime.cell_parallel import (
    llama_attention_shard_plan,
    llama_decoder_layer_tensor_parallel,
)
from distributed_runtime.cell_stage import (
    TensorParallelCellSpec,
    TensorParallelCellStageRunner,
    _validate_local_weights,
)
from distributed_runtime.model import StageModelSpec


_HF_SUFFIXES = {
    "input_norm": "input_layernorm.weight",
    "post_attention_norm": "post_attention_layernorm.weight",
    "query": "self_attn.q_proj.weight",
    "key": "self_attn.k_proj.weight",
    "value": "self_attn.v_proj.weight",
    "output": "self_attn.o_proj.weight",
    "gate": "mlp.gate_proj.weight",
    "up": "mlp.up_proj.weight",
    "down": "mlp.down_proj.weight",
}


def _gpu_test_skip_reason() -> str | None:
    if os.environ.get("RUN_DISTRIBUTED_GPU_TESTS") != "1":
        return "set RUN_DISTRIBUTED_GPU_TESTS=1 to run the physical GPU cell test"
    if not distributed.is_available() or not distributed.is_nccl_available():
        return "this PyTorch build does not provide NCCL/RCCL"
    if not torch.cuda.is_available():
        return "no CUDA/ROCm accelerator is visible"
    if torch.cuda.device_count() < 2:
        return "the physical GPU cell test requires at least two visible GPUs"
    return None


_GPU_TEST_SKIP_REASON = _gpu_test_skip_reason()


class TensorParallelGpuDtypeContractTests(unittest.TestCase):
    def test_rank_validator_accepts_the_planned_fp16_dtype(self) -> None:
        plan = llama_attention_shard_plan(8, 4, 2, 2, 0, 2)
        weights = {
            "input_norm": torch.empty(8, dtype=torch.float16),
            "post_attention_norm": torch.empty(8, dtype=torch.float16),
            "query": torch.empty((4, 8), dtype=torch.float16),
            "key": torch.empty((2, 8), dtype=torch.float16),
            "value": torch.empty((2, 8), dtype=torch.float16),
            "output": torch.empty((8, 4), dtype=torch.float16),
            "gate": torch.empty((5, 8), dtype=torch.float16),
            "up": torch.empty((5, 8), dtype=torch.float16),
            "down": torch.empty((8, 5), dtype=torch.float16),
        }
        _validate_local_weights(
            weights,
            {"hiddenSize": 8},
            plan,
            5,
            expected_dtype=torch.float16,
        )
        with self.assertRaisesRegex(ValueError, "planned dtype"):
            _validate_local_weights(weights, {"hiddenSize": 8}, plan, 5)


@unittest.skipIf(_GPU_TEST_SKIP_REASON is not None, _GPU_TEST_SKIP_REASON or "")
class TensorParallelGpuCellTests(unittest.TestCase):
    def test_fp16_two_gpu_nccl_prefill_decode_and_rollback(self) -> None:
        generator = torch.Generator().manual_seed(20260719)
        hidden_size = 16
        attention_heads = 4
        key_value_heads = 2
        head_dim = 4
        intermediate_size = 20
        dense_layers = tuple(
            _dense_layer(
                generator,
                hidden_size=hidden_size,
                key_value_heads=key_value_heads,
                head_dim=head_dim,
                intermediate_size=intermediate_size,
            )
            for _ in range(2)
        )
        prompt = torch.randn((1, 3, hidden_size), generator=generator) * 0.2
        next_hidden = torch.randn((1, 1, hidden_size), generator=generator) * 0.2
        correction = torch.randn((1, 1, hidden_size), generator=generator) * 0.2

        # The dense oracle uses the same FP16-quantized weights and accelerator
        # kernels as the cell, but a world size of one and therefore no TP
        # collectives. This isolates sharding/collective correctness from the
        # expected error introduced by compiling FP32 source weights to FP16.
        reference_device = torch.device("cuda:0")
        reference_layers = tuple(
            {
                name: value.to(device=reference_device, dtype=torch.float16)
                for name, value in layer.items()
            }
            for layer in dense_layers
        )
        expected_prompt, prompt_caches = _dense_stage_forward(
            prompt.to(device=reference_device, dtype=torch.float16),
            reference_layers,
            attention_heads=attention_heads,
            key_value_heads=key_value_heads,
            head_dim=head_dim,
        )
        expected_next, next_caches = _dense_stage_forward(
            next_hidden.to(device=reference_device, dtype=torch.float16),
            reference_layers,
            attention_heads=attention_heads,
            key_value_heads=key_value_heads,
            head_dim=head_dim,
            caches=prompt_caches,
        )
        truncated_caches = tuple(
            tuple(value[:, :, :2, :].contiguous() for value in cache)
            for cache in next_caches
        )
        expected_correction, correction_caches = _dense_stage_forward(
            correction.to(device=reference_device, dtype=torch.float16),
            reference_layers,
            attention_heads=attention_heads,
            key_value_heads=key_value_heads,
            head_dim=head_dim,
            caches=truncated_caches,
        )
        expected_prompt = expected_prompt.float().cpu()
        expected_next = expected_next.float().cpu()
        expected_correction = expected_correction.float().cpu()
        del (
            reference_layers,
            prompt_caches,
            next_caches,
            truncated_caches,
            correction_caches,
        )
        torch.cuda.synchronize(reference_device)
        torch.cuda.empty_cache()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            fixture = root / "cell-fp16"
            _write_synthetic_checkpoint(
                checkpoint,
                dense_layers,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                attention_heads=attention_heads,
                key_value_heads=key_value_heads,
                head_dim=head_dim,
            )
            compilation = compile_hf_llama_cell_fixture(
                str(checkpoint),
                fixture,
                layer_start=0,
                layer_end=2,
                world_size=2,
                output_dtype="float16",
            )
            self.assertEqual(compilation.output_dtype, "float16")

            runner = TensorParallelCellStageRunner(
                StageModelSpec("synthetic-gpu-fixture", 1, 3, 4, 1),
                TensorParallelCellSpec(
                    fixture=str(fixture),
                    world_size=2,
                    backend="nccl",
                    rank_devices=("cuda:0", "cuda:1"),
                    compute_dtype="float16",
                    operation_timeout_seconds=120.0,
                ),
            )
            try:
                self.assertEqual(
                    tuple(report.device for report in runner.member_reports),
                    ("cuda:0", "cuda:1"),
                )
                full_dense_fp16_bytes = sum(
                    value.numel() * 2
                    for layer in dense_layers
                    for value in layer.values()
                )
                for report in runner.member_reports:
                    self.assertEqual(report.collective_backend, "nccl")
                    self.assertEqual(report.compute_dtype, "float16")
                    self.assertGreater(report.parameter_bytes, 0)
                    self.assertLess(report.parameter_bytes, full_dense_fp16_bytes)
                    self.assertGreaterEqual(report.allocated_bytes, report.parameter_bytes)
                    self.assertGreaterEqual(report.reserved_bytes, report.allocated_bytes)
                    self.assertGreaterEqual(
                        report.peak_allocated_bytes, report.allocated_bytes
                    )

                request_id = 9102
                runner.begin(request_id)
                actual_prompt, _ = runner.forward_hidden(
                    request_id, prompt, token_mode="none"
                )
                _assert_fp16_close(self, actual_prompt, expected_prompt, "prefill")
                self.assertEqual(runner.sequence_length(request_id), 3)

                actual_next, _ = runner.forward_hidden(request_id, next_hidden)
                _assert_fp16_close(self, actual_next, expected_next, "decode")
                self.assertEqual(runner.sequence_length(request_id), 4)

                runner.truncate(request_id, 2)
                self.assertEqual(runner.sequence_length(request_id), 2)
                self.assertEqual(
                    runner.member_layer_cache_shapes[request_id],
                    (
                        ((1, 1, 2, head_dim), (1, 1, 2, head_dim)),
                        ((1, 1, 2, head_dim), (1, 1, 2, head_dim)),
                    ),
                )
                actual_correction, _ = runner.forward_hidden(request_id, correction)
                _assert_fp16_close(
                    self, actual_correction, expected_correction, "rollback decode"
                )
                self.assertEqual(runner.sequence_length(request_id), 3)

                for rank in range(2):
                    memory = runner.member_memory_reports[rank]
                    self.assertEqual(memory["device"], f"cuda:{rank}")
                    self.assertEqual(memory["collectiveBackend"], "nccl")
                    self.assertEqual(memory["computeDtype"], "float16")
                    allocated = int(memory["allocatedBytes"])
                    reserved = int(memory["reservedBytes"])
                    peak = int(memory["peakAllocatedBytes"])
                    self.assertGreaterEqual(
                        allocated, runner.member_reports[rank].parameter_bytes
                    )
                    self.assertGreaterEqual(reserved, allocated)
                    self.assertGreaterEqual(peak, allocated)
                runner.end(request_id)
            finally:
                runner.close()
            self.assertTrue(all(not process.is_alive() for process in runner._processes))


def _dense_layer(
    generator: torch.Generator,
    *,
    hidden_size: int,
    key_value_heads: int,
    head_dim: int,
    intermediate_size: int,
) -> dict[str, torch.Tensor]:
    def weight(*shape: int) -> torch.Tensor:
        return torch.randn(shape, generator=generator) * 0.04

    return {
        "input_norm": 1.0 + torch.randn(hidden_size, generator=generator) * 0.02,
        "post_attention_norm": 1.0
        + torch.randn(hidden_size, generator=generator) * 0.02,
        "query": weight(hidden_size, hidden_size),
        "key": weight(key_value_heads * head_dim, hidden_size),
        "value": weight(key_value_heads * head_dim, hidden_size),
        "output": weight(hidden_size, hidden_size),
        "gate": weight(intermediate_size, hidden_size),
        "up": weight(intermediate_size, hidden_size),
        "down": weight(hidden_size, intermediate_size),
    }


@torch.inference_mode()
def _dense_stage_forward(
    inputs: torch.Tensor,
    layers: tuple[dict[str, torch.Tensor], ...],
    *,
    attention_heads: int,
    key_value_heads: int,
    head_dim: int,
    caches: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], ...]]:
    output = inputs
    presents: list[tuple[torch.Tensor, torch.Tensor]] = []
    for index, layer in enumerate(layers):
        plan = llama_attention_shard_plan(
            int(inputs.shape[-1]), attention_heads, key_value_heads, head_dim, 0, 1
        )
        output, present = llama_decoder_layer_tensor_parallel(
            output,
            layer["input_norm"],
            layer["post_attention_norm"],
            layer["query"],
            layer["key"],
            layer["value"],
            layer["output"],
            plan,
            layer["gate"],
            layer["up"],
            layer["down"],
            (int(layer["gate"].shape[0]),),
            past_key_value=None if caches is None else caches[index],
            rms_norm_epsilon=1e-5,
            rope_theta=20_000.0,
        )
        presents.append(present)
    return output, tuple(presents)


def _write_synthetic_checkpoint(
    root: Path,
    layers: tuple[dict[str, torch.Tensor], ...],
    *,
    hidden_size: int,
    intermediate_size: int,
    attention_heads: int,
    key_value_heads: int,
    head_dim: int,
) -> None:
    root.mkdir(parents=True)
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "architectures": ["LlamaForCausalLM"],
                "vocab_size": 32,
                "hidden_size": hidden_size,
                "intermediate_size": intermediate_size,
                "num_hidden_layers": len(layers),
                "num_attention_heads": attention_heads,
                "num_key_value_heads": key_value_heads,
                "head_dim": head_dim,
                "rms_norm_eps": 1e-5,
                "rope_theta": 20_000.0,
                "hidden_act": "silu",
                "attention_bias": False,
                "mlp_bias": False,
                "tie_word_embeddings": True,
            }
        ),
        encoding="utf-8",
    )
    checkpoint = {
        f"model.layers.{layer_index}.{_HF_SUFFIXES[name]}": value.contiguous()
        for layer_index, layer in enumerate(layers)
        for name, value in layer.items()
    }
    save_file(checkpoint, root / "model.safetensors")


def _assert_fp16_close(
    case: unittest.TestCase,
    actual: torch.Tensor,
    expected: torch.Tensor,
    phase: str,
) -> None:
    case.assertTrue(
        torch.isfinite(actual).all().item(),
        f"{phase} produced non-finite values",
    )
    maximum_error = float((actual - expected).abs().max().item())
    case.assertTrue(
        torch.allclose(actual, expected, rtol=4e-2, atol=3e-3),
        f"{phase} max FP16 TP error: {maximum_error}",
    )


if __name__ == "__main__":
    unittest.main()
