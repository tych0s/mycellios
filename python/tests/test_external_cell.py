from __future__ import annotations

import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest

import torch

from tests.cell_parity import assert_fp32_cell_close

from distributed_runtime.cell_parallel import (
    llama_attention_shard_plan,
    llama_decoder_layer_tensor_parallel,
)
from distributed_runtime.cell_stage import write_llama_stage_cell_fixture
from distributed_runtime.external_cell import ExternalTensorParallelCellStageRunner
from distributed_runtime.model import StageModelSpec
from distributed_runtime.protocol import TensorCodec
from distributed_runtime.stage import (
    StageProcessConfig,
    build_stage_runner,
    validate_stage_config,
)


class ExternalTensorParallelCellTests(unittest.TestCase):
    def test_external_configuration_is_complete_and_disjoint(self) -> None:
        config = _external_stage_config("fixture", _ports(6))
        validate_stage_config(config)
        missing = StageProcessConfig(
            **{**config.__dict__, "cell_distributed_port": None}
        )
        with self.assertRaisesRegex(ValueError, "require fixture"):
            validate_stage_config(missing)
        overlapping = StageProcessConfig(
            **{**config.__dict__, "cell_control_port": config.listen_port}
        )
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_stage_config(overlapping)

    def test_external_rank_cli_and_anchor_rank_zero_execute_one_logical_stage(self) -> None:
        generator = torch.Generator().manual_seed(503)
        hidden_size = 16
        attention_heads = 8
        kv_heads = 4
        head_dim = 2
        rank_weights = (3.0, 1.0)
        layers = (
            _dense_layer(generator, hidden_size, kv_heads, head_dim, 9),
            _dense_layer(generator, hidden_size, kv_heads, head_dim, 11),
        )
        prompt = torch.randn((1, 3, hidden_size), generator=generator)
        next_hidden = torch.randn((1, 1, hidden_size), generator=generator)
        correction = torch.randn((1, 1, hidden_size), generator=generator)
        batch_prompt_a = torch.randn((1, 2, hidden_size), generator=generator)
        batch_prompt_b = torch.randn((1, 2, hidden_size), generator=generator)
        batch_next_a = torch.randn((1, 1, hidden_size), generator=generator)
        batch_next_b = torch.randn((1, 1, hidden_size), generator=generator)
        fork_prompt = torch.randn((1, 2, hidden_size), generator=generator)
        fork_parent_next = torch.randn((1, 1, hidden_size), generator=generator)
        fork_child_next = torch.randn((1, 1, hidden_size), generator=generator)
        fork_child_second = torch.randn((1, 1, hidden_size), generator=generator)
        promoted_next = torch.randn((1, 1, hidden_size), generator=generator)
        expected_prompt, prompt_caches = _dense_stage_forward(
            prompt, layers, attention_heads, kv_heads, head_dim
        )
        expected_next, next_caches = _dense_stage_forward(
            next_hidden,
            layers,
            attention_heads,
            kv_heads,
            head_dim,
            prompt_caches,
        )
        truncated = tuple(
            tuple(value[:, :, :2, :].contiguous() for value in cache)
            for cache in next_caches
        )
        expected_correction, _ = _dense_stage_forward(
            correction,
            layers,
            attention_heads,
            kv_heads,
            head_dim,
            truncated,
        )
        expected_batch_prompt_a, batch_caches_a = _dense_stage_forward(
            batch_prompt_a, layers, attention_heads, kv_heads, head_dim
        )
        expected_batch_prompt_b, batch_caches_b = _dense_stage_forward(
            batch_prompt_b, layers, attention_heads, kv_heads, head_dim
        )
        expected_batch_next_a, _ = _dense_stage_forward(
            batch_next_a,
            layers,
            attention_heads,
            kv_heads,
            head_dim,
            batch_caches_a,
        )
        expected_batch_next_b, _ = _dense_stage_forward(
            batch_next_b,
            layers,
            attention_heads,
            kv_heads,
            head_dim,
            batch_caches_b,
        )
        expected_fork_prompt, fork_caches = _dense_stage_forward(
            fork_prompt, layers, attention_heads, kv_heads, head_dim
        )
        expected_fork_parent_next, _ = _dense_stage_forward(
            fork_parent_next,
            layers,
            attention_heads,
            kv_heads,
            head_dim,
            fork_caches,
        )
        expected_fork_child_next, fork_child_caches = _dense_stage_forward(
            fork_child_next,
            layers,
            attention_heads,
            kv_heads,
            head_dim,
            fork_caches,
        )
        expected_fork_child_second, fork_child_caches = _dense_stage_forward(
            fork_child_second,
            layers,
            attention_heads,
            kv_heads,
            head_dim,
            fork_child_caches,
        )
        expected_promoted_next, _ = _dense_stage_forward(
            promoted_next,
            layers,
            attention_heads,
            kv_heads,
            head_dim,
            fork_child_caches,
        )

        with tempfile.TemporaryDirectory() as temporary:
            fixture = write_llama_stage_cell_fixture(
                temporary,
                layers,
                world_size=2,
                num_attention_heads=attention_heads,
                num_key_value_heads=kv_heads,
                head_dim=head_dim,
                rank_weights=rank_weights,
            )
            ports = _ports(6)
            config = _external_stage_config(str(fixture), ports)
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
            member = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "distributed_runtime.cell_member_cli",
                    "--fixture",
                    str(fixture),
                    "--rank",
                    "1",
                    "--world-size",
                    "2",
                    "--pipeline-id",
                    str(config.pipeline_id),
                    "--layer-start",
                    "1",
                    "--layer-end",
                    "3",
                    "--control-host",
                    "127.0.0.1",
                    "--control-port",
                    str(config.cell_control_port),
                    "--threads",
                    "1",
                    "--connect-timeout-seconds",
                    "30",
                    "--operation-timeout-seconds",
                    "30",
                ],
                cwd=str(Path(__file__).resolve().parents[2]),
                env=environment,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            runner: ExternalTensorParallelCellStageRunner | None = None
            try:
                validate_stage_config(config)
                built = build_stage_runner(config)
                self.assertIsInstance(built, ExternalTensorParallelCellStageRunner)
                runner = built
                self.assertEqual(
                    runner.loader,
                    "tensor-parallel-cell-external-safetensors-gloo",
                )
                self.assertIn("external-ranks", runner.executor_manifest.features)
                self.assertIn(
                    "unequal-tensor-parallel", runner.executor_manifest.features
                )
                self.assertIn(
                    "multi-request-physical-batch", runner.executor_manifest.features
                )
                self.assertIn(
                    "exact-request-fork-safe-copy", runner.executor_manifest.features
                )
                self.assertEqual(
                    [report.shard_file for report in runner.member_reports],
                    ["rank-000.safetensors", "rank-001.safetensors"],
                )
                self.assertEqual(
                    [report.tensor_count for report in runner.member_reports], [18, 18]
                )
                self.assertGreater(
                    runner.member_reports[0].parameter_bytes,
                    runner.member_reports[1].parameter_bytes,
                )
                self.assertEqual(
                    runner.member_work_reports,
                    {},
                    "READY must not be treated as evidence of executed work",
                )
                self.assertEqual(runner.member_batch_work_reports, {})
                self.assertIsNone(member.poll(), "external rank exited before BEGIN")
                with self.assertRaisesRegex(ValueError, "outside the physical batch bound"):
                    runner.forward_hidden_batch(
                        tuple(range(9)),
                        tuple(prompt[:, :1, :] for _ in range(9)),
                    )
                with self.assertRaisesRegex(ValueError, "outside the physical batch bound"):
                    runner.forward_hidden_batch((707,), (prompt[:, :1, :],))

                request_id = 707
                runner.begin(request_id)
                actual_prompt, _ = runner.forward_hidden(
                    request_id, prompt, token_mode="none"
                )
                assert_fp32_cell_close(actual_prompt, expected_prompt)
                self.assertEqual(
                    runner.member_work_reports,
                    _expected_cpu_rank_work(1, 5, 3),
                )
                self.assertEqual(
                    runner.member_batch_work_reports,
                    _expected_cpu_rank_batch_work(1, 1, 1),
                )
                self.assertEqual(
                    runner.member_layer_cache_shapes[request_id],
                    (
                        ((1, 3, 3, head_dim), (1, 3, 3, head_dim)),
                        ((1, 1, 3, head_dim), (1, 1, 3, head_dim)),
                    ),
                )

                actual_next, _ = runner.forward_hidden(request_id, next_hidden)
                assert_fp32_cell_close(actual_next, expected_next)
                self.assertEqual(
                    runner.member_work_reports,
                    _expected_cpu_rank_work(2, 10, 4),
                )
                self.assertEqual(
                    runner.member_batch_work_reports,
                    _expected_cpu_rank_batch_work(2, 2, 1),
                )
                runner.truncate(request_id, 2)
                self.assertEqual(
                    runner.member_work_reports,
                    _expected_cpu_rank_work(2, 10, 4),
                )
                self.assertEqual(
                    runner.member_layer_cache_shapes[request_id],
                    (
                        ((1, 3, 2, head_dim), (1, 3, 2, head_dim)),
                        ((1, 1, 2, head_dim), (1, 1, 2, head_dim)),
                    ),
                )
                actual_correction, _ = runner.forward_hidden(request_id, correction)
                assert_fp32_cell_close(actual_correction, expected_correction)
                self.assertEqual(
                    runner.member_work_reports,
                    _expected_cpu_rank_work(3, 15, 5),
                )
                self.assertEqual(
                    runner.member_batch_work_reports,
                    _expected_cpu_rank_batch_work(3, 3, 1),
                )
                self.assertEqual(runner.sequence_length(request_id), 3)
                runner.end(request_id)

                request_ids = (801, 802)
                for batched_request_id in request_ids:
                    runner.begin(batched_request_id)
                self.assertEqual(
                    runner.physical_batch_key(
                        request_ids[0], token_count=2, token_mode="none"
                    ),
                    runner.physical_batch_key(
                        request_ids[1], token_count=2, token_mode="none"
                    ),
                )
                with self.assertRaisesRegex(ValueError, "distinct requests"):
                    runner.forward_hidden_batch(
                        (request_ids[0], request_ids[0]),
                        (batch_prompt_a, batch_prompt_a),
                        token_mode="none",
                    )
                with self.assertRaisesRegex(ValueError, "equal rank-one"):
                    runner.forward_hidden_batch(
                        request_ids,
                        (batch_prompt_a, batch_next_b),
                        token_mode="none",
                    )

                batch_prompt_results = runner.forward_hidden_batch(
                    request_ids,
                    (batch_prompt_a, batch_prompt_b),
                    token_mode="none",
                )
                assert_fp32_cell_close(batch_prompt_results[0][0], expected_batch_prompt_a)
                assert_fp32_cell_close(batch_prompt_results[1][0], expected_batch_prompt_b)
                self.assertEqual(
                    tuple(result[1] for result in batch_prompt_results),
                    (None, None),
                )
                self.assertEqual(
                    runner.member_work_reports,
                    _expected_cpu_rank_work(4, 20, 9),
                )
                self.assertEqual(
                    runner.member_batch_work_reports,
                    _expected_cpu_rank_batch_work(4, 5, 2),
                )
                for batched_request_id in request_ids:
                    self.assertEqual(runner.sequence_length(batched_request_id), 2)
                    self.assertEqual(
                        runner.member_layer_cache_shapes[batched_request_id],
                        (
                            ((1, 3, 2, head_dim), (1, 3, 2, head_dim)),
                            ((1, 1, 2, head_dim), (1, 1, 2, head_dim)),
                        ),
                    )

                batch_next_results = runner.forward_hidden_batch(
                    request_ids,
                    (batch_next_a, batch_next_b),
                )
                assert_fp32_cell_close(batch_next_results[0][0], expected_batch_next_a)
                assert_fp32_cell_close(batch_next_results[1][0], expected_batch_next_b)
                self.assertEqual(
                    runner.member_work_reports,
                    _expected_cpu_rank_work(5, 25, 11),
                )
                self.assertEqual(
                    runner.member_batch_work_reports,
                    _expected_cpu_rank_batch_work(5, 7, 2),
                )
                for batched_request_id in request_ids:
                    self.assertEqual(runner.sequence_length(batched_request_id), 3)
                    self.assertEqual(
                        runner.member_layer_cache_shapes[batched_request_id],
                        (
                            ((1, 3, 3, head_dim), (1, 3, 3, head_dim)),
                            ((1, 1, 3, head_dim), (1, 1, 3, head_dim)),
                        ),
                    )
                    runner.end(batched_request_id)

                parent_request_id = 901
                child_request_id = 902
                runner.begin(parent_request_id)
                actual_fork_prompt, _ = runner.forward_hidden(
                    parent_request_id,
                    fork_prompt,
                    token_mode="none",
                )
                assert_fp32_cell_close(actual_fork_prompt, expected_fork_prompt)
                parent_cache_bytes = runner.request_cache_bytes(parent_request_id)
                self.assertEqual(parent_cache_bytes, 256)
                self.assertEqual(
                    runner.project_request_cache_bytes(parent_request_id, 1),
                    384,
                )
                work_before_fork = runner.member_work_reports
                batch_work_before_fork = runner.member_batch_work_reports
                with self.assertRaisesRegex(ValueError, "must differ"):
                    runner.fork_request(
                        parent_request_id,
                        parent_request_id,
                        max_cache_bytes=parent_cache_bytes,
                    )
                with self.assertRaisesRegex(ValueError, "preflight byte budget"):
                    runner.fork_request(
                        child_request_id,
                        parent_request_id,
                        max_cache_bytes=parent_cache_bytes - 1,
                    )
                with self.assertRaisesRegex(ValueError, "has not received BEGIN"):
                    runner.sequence_length(child_request_id)

                copied_bytes = runner.fork_request(
                    child_request_id,
                    parent_request_id,
                    max_cache_bytes=parent_cache_bytes,
                )
                self.assertEqual(copied_bytes, parent_cache_bytes)
                self.assertEqual(runner.sequence_length(child_request_id), 2)
                self.assertEqual(
                    runner.member_request_cache_bytes[child_request_id],
                    runner.member_request_cache_bytes[parent_request_id],
                )
                self.assertTrue(
                    all(
                        report["aliasFree"] is True
                        and report["copiedCacheBytes"] > 0
                        and report["childRequestId"] == child_request_id
                        and report["parentRequestId"] == parent_request_id
                        for report in runner.member_fork_reports.values()
                    )
                )
                self.assertEqual(runner.member_work_reports, work_before_fork)
                self.assertEqual(runner.member_batch_work_reports, batch_work_before_fork)
                with self.assertRaisesRegex(ValueError, "already active"):
                    runner.fork_request(
                        child_request_id,
                        parent_request_id,
                        max_cache_bytes=parent_cache_bytes,
                    )

                actual_fork_parent_next, _ = runner.forward_hidden(
                    parent_request_id,
                    fork_parent_next,
                )
                assert_fp32_cell_close(actual_fork_parent_next, expected_fork_parent_next)
                self.assertEqual(runner.sequence_length(child_request_id), 2)
                self.assertEqual(runner.request_cache_bytes(child_request_id), 256)

                actual_fork_child_next, _ = runner.forward_hidden(
                    child_request_id,
                    fork_child_next,
                )
                assert_fp32_cell_close(actual_fork_child_next, expected_fork_child_next)
                actual_fork_child_second, _ = runner.forward_hidden(
                    child_request_id,
                    fork_child_second,
                )
                assert_fp32_cell_close(actual_fork_child_second, expected_fork_child_second)
                self.assertEqual(runner.sequence_length(parent_request_id), 3)
                self.assertEqual(runner.sequence_length(child_request_id), 4)
                self.assertEqual(runner.request_cache_bytes(parent_request_id), 384)
                self.assertEqual(runner.request_cache_bytes(child_request_id), 512)

                selected_shapes = runner.member_layer_cache_shapes[child_request_id]
                selected_rank_bytes = runner.member_request_cache_bytes[child_request_id]
                work_before_promote = runner.member_work_reports
                batch_work_before_promote = runner.member_batch_work_reports
                with self.assertRaisesRegex(ValueError, "must differ"):
                    runner.promote_request(parent_request_id, parent_request_id)
                runner.promote_request(parent_request_id, child_request_id)
                self.assertEqual(runner.sequence_length(parent_request_id), 4)
                self.assertEqual(runner.request_cache_bytes(parent_request_id), 512)
                self.assertEqual(
                    runner.member_layer_cache_shapes[parent_request_id], selected_shapes
                )
                self.assertEqual(
                    runner.member_request_cache_bytes[parent_request_id],
                    selected_rank_bytes,
                )
                self.assertTrue(
                    all(
                        report["movedWithoutCopy"] is True
                        and report["parentRequestId"] == parent_request_id
                        and report["childRequestId"] == child_request_id
                        for report in runner.member_promotion_reports.values()
                    )
                )
                self.assertEqual(runner.member_work_reports, work_before_promote)
                self.assertEqual(
                    runner.member_batch_work_reports, batch_work_before_promote
                )
                with self.assertRaisesRegex(ValueError, "has not received BEGIN"):
                    runner.sequence_length(child_request_id)

                actual_promoted_next, _ = runner.forward_hidden(
                    parent_request_id,
                    promoted_next,
                )
                assert_fp32_cell_close(actual_promoted_next, expected_promoted_next)
                self.assertEqual(runner.sequence_length(parent_request_id), 5)
                self.assertEqual(
                    runner.member_work_reports,
                    _expected_cpu_rank_work(10, 50, 17),
                )
                self.assertEqual(
                    runner.member_batch_work_reports,
                    _expected_cpu_rank_batch_work(10, 12, 2),
                )
                runner.end(parent_request_id)
            finally:
                if runner is not None:
                    runner.close()
                if member.poll() is None:
                    try:
                        member.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        member.terminate()
                        member.wait(timeout=10)
                stderr = member.stderr.read() if member.stderr is not None else ""
                if member.stderr is not None:
                    member.stderr.close()
            self.assertEqual(member.returncode, 0, stderr)
            self.assertIn('"event": "joining_external_cell"', stderr)
            self.assertIn('"event": "external_cell_stopped"', stderr)

    def test_anchor_rejects_a_rank_with_a_modified_shard(self) -> None:
        generator = torch.Generator().manual_seed(811)
        layers = (
            _dense_layer(generator, 16, 2, 4, 9),
            _dense_layer(generator, 16, 2, 4, 11),
        )
        with tempfile.TemporaryDirectory() as temporary:
            fixture = write_llama_stage_cell_fixture(
                temporary,
                layers,
                world_size=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=4,
            )
            # Simulate a corrupted or mismatched remote copy after compilation.
            with (fixture / "rank-001.safetensors").open("ab") as handle:
                handle.write(b"corrupt")
            ports = _ports(6)
            config = _external_stage_config(str(fixture), ports)
            config = StageProcessConfig(
                # A cold Windows spawn imports Torch independently in rank zero
                # and in the external CLI. Ten seconds made this integrity test
                # race process startup instead of exercising the digest HELLO.
                **{**config.__dict__, "cell_startup_timeout_seconds": 30.0}
            )
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
            member = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "distributed_runtime.cell_member_cli",
                    "--fixture",
                    str(fixture),
                    "--rank",
                    "1",
                    "--world-size",
                    "2",
                    "--pipeline-id",
                    str(config.pipeline_id),
                    "--layer-start",
                    "1",
                    "--layer-end",
                    "3",
                    "--control-host",
                    "127.0.0.1",
                    "--control-port",
                    str(config.cell_control_port),
                    "--connect-timeout-seconds",
                    "10",
                    "--operation-timeout-seconds",
                    "10",
                ],
                cwd=str(Path(__file__).resolve().parents[2]),
                env=environment,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                with self.assertRaisesRegex(ValueError, "shard digest"):
                    build_stage_runner(config)
            finally:
                if member.poll() is None:
                    member.terminate()
                try:
                    member.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    member.kill()
                    member.wait(timeout=5)
                stderr = member.stderr.read() if member.stderr is not None else ""
                if member.stderr is not None:
                    member.stderr.close()
            self.assertNotEqual(member.returncode, 0)
            self.assertIn("external cell anchor rejected rank 1", stderr)
            self.assertIn("shard digest", stderr)


def _external_stage_config(fixture: str, ports: tuple[int, ...]) -> StageProcessConfig:
    stage_port, next_port, return_port, control_port, distributed_port, _ = ports
    return StageProcessConfig(
        spec=StageModelSpec("unused", 1, 3, 4, 1),
        pipeline_id=12_345,
        listen_host="127.0.0.1",
        listen_port=stage_port,
        next_host="127.0.0.1",
        next_port=next_port,
        next_layer_end=4,
        return_host="127.0.0.1",
        return_port=return_port,
        codec=TensorCodec.FP32,
        one_way_delay_ms=0.0,
        bandwidth_mbps=0.0,
        connect_timeout_seconds=30.0,
        cell_fixture=fixture,
        cell_world_size=2,
        cell_operation_timeout_seconds=30.0,
        cell_mode="external",
        cell_control_host="127.0.0.1",
        cell_control_port=control_port,
        cell_control_advertise_host="127.0.0.1",
        cell_distributed_advertise_host="127.0.0.1",
        cell_distributed_port=distributed_port,
        cell_startup_timeout_seconds=30.0,
    )


def _expected_cpu_rank_work(
    forward_calls: int,
    collective_calls: int,
    tokens_processed: int,
) -> dict[int, dict[str, object]]:
    return {
        rank: {
            "rank": rank,
            "device": "cpu",
            "computeDtype": "float32",
            "collectiveBackend": "gloo",
            "forwardCalls": forward_calls,
            "collectiveCalls": collective_calls,
            "tokensProcessed": tokens_processed,
            "memory": {
                "allocatedBytes": 0,
                "reservedBytes": 0,
                "peakAllocatedBytes": 0,
            },
        }
        for rank in range(2)
    }


def _expected_cpu_rank_batch_work(
    physical_forward_calls: int,
    logical_forward_items: int,
    max_physical_batch_size: int,
) -> dict[int, dict[str, int]]:
    return {
        rank: {
            "rank": rank,
            "physicalForwardCalls": physical_forward_calls,
            "logicalForwardItems": logical_forward_items,
            "maxPhysicalBatchSize": max_physical_batch_size,
        }
        for rank in range(2)
    }


def _ports(count: int) -> tuple[int, ...]:
    ports: list[int] = []
    while len(ports) < count:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            listener.bind(("127.0.0.1", 0))
            port = int(listener.getsockname()[1])
        finally:
            listener.close()
        if port not in ports:
            ports.append(port)
    return tuple(ports)


def _dense_layer(
    generator: torch.Generator,
    hidden_size: int,
    kv_heads: int,
    head_dim: int,
    intermediate_size: int,
) -> dict[str, torch.Tensor]:
    return {
        "input_norm": torch.randn(hidden_size, generator=generator),
        "post_attention_norm": torch.randn(hidden_size, generator=generator),
        "query": torch.randn((hidden_size, hidden_size), generator=generator),
        "key": torch.randn((kv_heads * head_dim, hidden_size), generator=generator),
        "value": torch.randn((kv_heads * head_dim, hidden_size), generator=generator),
        "output": torch.randn((hidden_size, hidden_size), generator=generator),
        "gate": torch.randn((intermediate_size, hidden_size), generator=generator),
        "up": torch.randn((intermediate_size, hidden_size), generator=generator),
        "down": torch.randn((hidden_size, intermediate_size), generator=generator),
    }


def _dense_stage_forward(
    inputs: torch.Tensor,
    layers: tuple[dict[str, torch.Tensor], ...],
    attention_heads: int,
    kv_heads: int,
    head_dim: int,
    caches: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], ...]]:
    output = inputs
    presents: list[tuple[torch.Tensor, torch.Tensor]] = []
    for index, layer in enumerate(layers):
        plan = llama_attention_shard_plan(
            int(inputs.shape[-1]), attention_heads, kv_heads, head_dim, 0, 1
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
        )
        presents.append(present)
    return output, tuple(presents)


if __name__ == "__main__":
    unittest.main()
