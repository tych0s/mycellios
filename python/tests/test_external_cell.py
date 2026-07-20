from __future__ import annotations

import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest

import torch

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
                self.assertIsNone(member.poll(), "external rank exited before BEGIN")

                request_id = 707
                runner.begin(request_id)
                actual_prompt, _ = runner.forward_hidden(
                    request_id, prompt, token_mode="none"
                )
                self.assertTrue(
                    torch.allclose(actual_prompt, expected_prompt, rtol=1e-5, atol=1e-5)
                )
                self.assertEqual(
                    runner.member_layer_cache_shapes[request_id],
                    (
                        ((1, 3, 3, head_dim), (1, 3, 3, head_dim)),
                        ((1, 1, 3, head_dim), (1, 1, 3, head_dim)),
                    ),
                )

                actual_next, _ = runner.forward_hidden(request_id, next_hidden)
                self.assertTrue(
                    torch.allclose(actual_next, expected_next, rtol=1e-5, atol=1e-5)
                )
                runner.truncate(request_id, 2)
                self.assertEqual(
                    runner.member_layer_cache_shapes[request_id],
                    (
                        ((1, 3, 2, head_dim), (1, 3, 2, head_dim)),
                        ((1, 1, 2, head_dim), (1, 1, 2, head_dim)),
                    ),
                )
                actual_correction, _ = runner.forward_hidden(request_id, correction)
                self.assertTrue(
                    torch.allclose(
                        actual_correction,
                        expected_correction,
                        rtol=1e-5,
                        atol=1e-5,
                    )
                )
                self.assertEqual(runner.sequence_length(request_id), 3)
                runner.end(request_id)
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
