from __future__ import annotations

import os
from pathlib import Path
import tempfile
import queue
import socket
import threading
import unittest

from safetensors import safe_open
from safetensors.torch import load_file
import torch
from transformers import AutoConfig

from distributed_runtime.cell_parallel import (
    llama_attention_shard_plan,
    llama_decoder_layer_tensor_parallel,
)
from distributed_runtime.cell_stage import (
    TensorParallelCellStageRunner,
    write_llama_layer_cell_fixture,
    write_llama_stage_cell_fixture,
)
from distributed_runtime.model import (
    StageModelSpec,
    StageRunner,
    _checkpoint_key_map,
    resolve_model_snapshot,
)
from distributed_runtime.protocol import (
    Frame,
    FrameType,
    TensorCodec,
    decode_tensor,
    encode_tensor,
    recv_frame,
    send_frame,
)
from distributed_runtime.stage import (
    StageProcessConfig,
    build_stage_runner,
    run_stage_process,
    validate_stage_config,
)


class TensorParallelCellStageTests(unittest.TestCase):
    def test_config_requires_complete_intermediate_cell(self) -> None:
        base = _stage_config()
        invalid = StageProcessConfig(
            **{
                **base.__dict__,
                "cell_fixture": "fixture",
                "cell_world_size": None,
            }
        )
        with self.assertRaisesRegex(ValueError, "supplied together"):
            validate_stage_config(invalid)

        first = StageProcessConfig(
            **{
                **base.__dict__,
                "spec": StageModelSpec("unused", 0, 1, 3, 1),
                "cell_fixture": "fixture",
                "cell_world_size": 2,
            }
        )
        with self.assertRaisesRegex(ValueError, "intermediate"):
            validate_stage_config(first)

    def test_stage_rejects_fixture_whose_manifest_does_not_match_launch_hash(self) -> None:
        generator = torch.Generator().manual_seed(17)
        layers = (
            _dense_layer(generator, 8, 1, 2, 6),
            _dense_layer(generator, 8, 1, 2, 6),
        )
        with tempfile.TemporaryDirectory() as directory:
            fixture = write_llama_stage_cell_fixture(
                Path(directory) / "cell",
                layers,
                world_size=2,
                num_attention_heads=4,
                num_key_value_heads=1,
                head_dim=2,
            )
            base = _stage_config(layer_end=3, total_layers=4)
            config = StageProcessConfig(
                **{
                    **base.__dict__,
                    "cell_fixture": str(fixture),
                    "cell_world_size": 2,
                    "cell_manifest_sha256": "0" * 64,
                }
            )
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                build_stage_runner(config)

    def test_two_member_stage_lifecycle_is_exact_and_keeps_rank_local_kv(self) -> None:
        generator = torch.Generator().manual_seed(91)
        hidden_size = 16
        attention_heads = 4
        kv_heads = 2
        head_dim = 4
        intermediate_size = 9
        dense = {
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
        prompt = torch.randn((1, 3, hidden_size), generator=generator)
        next_hidden = torch.randn((1, 1, hidden_size), generator=generator)
        correction = torch.randn((1, 1, hidden_size), generator=generator)
        dense_plan = llama_attention_shard_plan(
            hidden_size, attention_heads, kv_heads, head_dim, 0, 1
        )
        expected_prompt, prompt_cache = llama_decoder_layer_tensor_parallel(
            prompt,
            dense["input_norm"],
            dense["post_attention_norm"],
            dense["query"],
            dense["key"],
            dense["value"],
            dense["output"],
            dense_plan,
            dense["gate"],
            dense["up"],
            dense["down"],
            (intermediate_size,),
        )
        expected_next, _ = llama_decoder_layer_tensor_parallel(
            next_hidden,
            dense["input_norm"],
            dense["post_attention_norm"],
            dense["query"],
            dense["key"],
            dense["value"],
            dense["output"],
            dense_plan,
            dense["gate"],
            dense["up"],
            dense["down"],
            (intermediate_size,),
            past_key_value=prompt_cache,
        )
        truncated_cache = tuple(value[:, :, :2, :].contiguous() for value in prompt_cache)
        expected_correction, _ = llama_decoder_layer_tensor_parallel(
            correction,
            dense["input_norm"],
            dense["post_attention_norm"],
            dense["query"],
            dense["key"],
            dense["value"],
            dense["output"],
            dense_plan,
            dense["gate"],
            dense["up"],
            dense["down"],
            (intermediate_size,),
            past_key_value=truncated_cache,
        )

        with tempfile.TemporaryDirectory() as temporary:
            fixture = write_llama_layer_cell_fixture(
                temporary,
                dense,
                world_size=2,
                num_attention_heads=attention_heads,
                num_key_value_heads=kv_heads,
                head_dim=head_dim,
            )
            first_shard = load_file(str(fixture / "rank-000.safetensors"))
            second_shard = load_file(str(fixture / "rank-001.safetensors"))
            # Each member owns half Q/K/V/O and half the MLP; neither shard can
            # reconstruct the dense layer by itself.
            self.assertEqual(tuple(first_shard["query"].shape), (8, hidden_size))
            self.assertEqual(tuple(second_shard["key"].shape), (4, hidden_size))
            self.assertEqual(tuple(first_shard["down"].shape), (hidden_size, 5))
            self.assertEqual(tuple(second_shard["down"].shape), (hidden_size, 4))

            config = StageProcessConfig(
                **{
                    **_stage_config().__dict__,
                    "cell_fixture": str(fixture),
                    "cell_world_size": 2,
                }
            )
            validate_stage_config(config)
            runner = build_stage_runner(config)
            self.assertIsInstance(runner, TensorParallelCellStageRunner)
            try:
                self.assertEqual(runner.loader, "tensor-parallel-cell-safetensors-gloo")
                self.assertEqual([report.tensor_count for report in runner.member_reports], [9, 9])
                dense_bytes = sum(value.nelement() * value.element_size() for value in dense.values())
                replicated_norm_bytes = (
                    dense["input_norm"].nelement()
                    + dense["post_attention_norm"].nelement()
                ) * dense["input_norm"].element_size()
                self.assertEqual(runner.parameter_bytes, dense_bytes + replicated_norm_bytes)

                request_id = 404
                runner.begin(request_id)
                actual_prompt, token = runner.forward_hidden(
                    request_id, prompt, token_mode="none"
                )
                self.assertIsNone(token)
                self.assertTrue(
                    torch.allclose(actual_prompt, expected_prompt, rtol=1e-5, atol=1e-5)
                )
                self.assertEqual(runner.sequence_length(request_id), 3)
                self.assertEqual(
                    runner.member_cache_shapes[request_id],
                    ((1, 1, 3, head_dim), (1, 1, 3, head_dim)),
                )

                actual_next, _ = runner.forward_hidden(request_id, next_hidden)
                self.assertTrue(
                    torch.allclose(actual_next, expected_next, rtol=1e-5, atol=1e-5)
                )
                self.assertEqual(runner.sequence_length(request_id), 4)

                runner.truncate(request_id, 2)
                self.assertEqual(runner.sequence_length(request_id), 2)
                self.assertEqual(
                    runner.member_cache_shapes[request_id],
                    ((1, 1, 2, head_dim), (1, 1, 2, head_dim)),
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
                runner.end(request_id)
                with self.assertRaisesRegex(ValueError, "has not received BEGIN"):
                    runner.sequence_length(request_id)
            finally:
                runner.close()
            self.assertTrue(all(not process.is_alive() for process in runner._processes))

    def test_two_layer_stage_prefill_decode_and_rollback_match_dense_reference(self) -> None:
        generator = torch.Generator().manual_seed(211)
        hidden_size = 16
        attention_heads = 4
        kv_heads = 2
        head_dim = 4
        dense_layers = (
            _dense_layer(generator, hidden_size, kv_heads, head_dim, 9),
            _dense_layer(generator, hidden_size, kv_heads, head_dim, 11),
        )
        prompt = torch.randn((1, 3, hidden_size), generator=generator)
        next_hidden = torch.randn((1, 1, hidden_size), generator=generator)
        correction = torch.randn((1, 1, hidden_size), generator=generator)
        expected_prompt, prompt_caches = _dense_stage_forward(
            prompt,
            dense_layers,
            attention_heads=attention_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
        )
        expected_next, next_caches = _dense_stage_forward(
            next_hidden,
            dense_layers,
            attention_heads=attention_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            caches=prompt_caches,
        )
        truncated_caches = tuple(
            tuple(value[:, :, :2, :].contiguous() for value in cache)
            for cache in next_caches
        )
        expected_correction, _ = _dense_stage_forward(
            correction,
            dense_layers,
            attention_heads=attention_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            caches=truncated_caches,
        )

        with tempfile.TemporaryDirectory() as temporary:
            fixture = write_llama_stage_cell_fixture(
                temporary,
                dense_layers,
                world_size=2,
                num_attention_heads=attention_heads,
                num_key_value_heads=kv_heads,
                head_dim=head_dim,
            )
            for rank in range(2):
                local = load_file(str(fixture / f"rank-{rank:03d}.safetensors"))
                self.assertEqual(len(local), 18)
                self.assertEqual(
                    set(local),
                    {
                        f"layers.{layer}.{name}"
                        for layer in range(2)
                        for name in (
                            "input_norm",
                            "post_attention_norm",
                            "query",
                            "key",
                            "value",
                            "output",
                            "gate",
                            "up",
                            "down",
                        )
                    },
                )
                self.assertEqual(tuple(local["layers.0.query"].shape), (8, hidden_size))
                self.assertEqual(tuple(local["layers.1.key"].shape), (4, hidden_size))

            config = StageProcessConfig(
                **{
                    **_stage_config(layer_end=3, total_layers=4).__dict__,
                    "cell_fixture": str(fixture),
                    "cell_world_size": 2,
                }
            )
            validate_stage_config(config)
            runner = build_stage_runner(config)
            self.assertIsInstance(runner, TensorParallelCellStageRunner)
            try:
                self.assertEqual(runner.layer_count, 2)
                self.assertEqual([report.tensor_count for report in runner.member_reports], [18, 18])
                request_id = 505
                runner.begin(request_id)
                actual_prompt, _ = runner.forward_hidden(request_id, prompt, token_mode="none")
                self.assertTrue(
                    torch.allclose(actual_prompt, expected_prompt, rtol=1e-5, atol=1e-5)
                )
                expected_member_layers = (
                    ((1, 1, 3, head_dim), (1, 1, 3, head_dim)),
                    ((1, 1, 3, head_dim), (1, 1, 3, head_dim)),
                )
                self.assertEqual(
                    runner.member_layer_cache_shapes[request_id],
                    expected_member_layers,
                )

                actual_next, _ = runner.forward_hidden(request_id, next_hidden)
                self.assertTrue(
                    torch.allclose(actual_next, expected_next, rtol=1e-5, atol=1e-5)
                )
                self.assertEqual(runner.sequence_length(request_id), 4)

                runner.truncate(request_id, 2)
                expected_truncated_shapes = (
                    ((1, 1, 2, head_dim), (1, 1, 2, head_dim)),
                    ((1, 1, 2, head_dim), (1, 1, 2, head_dim)),
                )
                self.assertEqual(
                    runner.member_layer_cache_shapes[request_id],
                    expected_truncated_shapes,
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
                runner.close()
            self.assertTrue(all(not process.is_alive() for process in runner._processes))

    def test_weighted_three_to_one_cell_executes_exact_unequal_shards(self) -> None:
        generator = torch.Generator().manual_seed(271)
        hidden_size = 16
        attention_heads = 8
        kv_heads = 4
        head_dim = 2
        rank_weights = (3.0, 1.0)
        layers = (
            _dense_layer(generator, hidden_size, kv_heads, head_dim, 12),
            _dense_layer(generator, hidden_size, kv_heads, head_dim, 12),
        )
        prompt = torch.randn((1, 3, hidden_size), generator=generator)
        decode = torch.randn((1, 1, hidden_size), generator=generator)
        expected_prompt, caches = _dense_stage_forward(
            prompt,
            layers,
            attention_heads=attention_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
        )
        expected_decode, _ = _dense_stage_forward(
            decode,
            layers,
            attention_heads=attention_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            caches=caches,
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
            rank_zero = load_file(str(fixture / "rank-000.safetensors"))
            rank_one = load_file(str(fixture / "rank-001.safetensors"))
            self.assertEqual(rank_zero["layers.0.query"].shape, (12, 16))
            self.assertEqual(rank_one["layers.0.query"].shape, (4, 16))
            self.assertEqual(rank_zero["layers.0.gate"].shape, (9, 16))
            self.assertEqual(rank_one["layers.0.gate"].shape, (3, 16))

            config = StageProcessConfig(
                **{
                    **_stage_config(layer_end=3, total_layers=4).__dict__,
                    "cell_fixture": str(fixture),
                    "cell_world_size": 2,
                }
            )
            runner = build_stage_runner(config)
            self.assertIsInstance(runner, TensorParallelCellStageRunner)
            try:
                self.assertEqual(runner.executor_manifest.engine, "python-torch-cell")
                self.assertIn(
                    "unequal-tensor-parallel", runner.executor_manifest.features
                )
                self.assertGreater(
                    runner.member_reports[0].parameter_bytes,
                    runner.member_reports[1].parameter_bytes,
                )
                runner.begin(606)
                actual_prompt, _ = runner.forward_hidden(606, prompt, token_mode="none")
                torch.testing.assert_close(
                    actual_prompt, expected_prompt, rtol=1e-5, atol=1e-5
                )
                self.assertEqual(
                    runner.member_cache_shapes[606],
                    ((1, 3, 3, head_dim), (1, 1, 3, head_dim)),
                )
                actual_decode, _ = runner.forward_hidden(606, decode)
                torch.testing.assert_close(
                    actual_decode, expected_decode, rtol=1e-5, atol=1e-5
                )
                runner.truncate(606, 2)
                self.assertEqual(
                    runner.member_cache_shapes[606],
                    ((1, 3, 2, head_dim), (1, 1, 2, head_dim)),
                )
                runner.end(606)
            finally:
                runner.close()

    def test_wire_loop_wraps_two_member_cell_as_one_logical_gdlp_stage(self) -> None:
        generator = torch.Generator().manual_seed(307)
        hidden_size = 16
        attention_heads = 4
        kv_heads = 2
        head_dim = 4
        layers = (
            _dense_layer(generator, hidden_size, kv_heads, head_dim, 9),
            _dense_layer(generator, hidden_size, kv_heads, head_dim, 11),
        )
        prompt = torch.randn((1, 3, hidden_size), generator=generator)
        next_hidden = torch.randn((1, 1, hidden_size), generator=generator)
        correction = torch.randn((1, 1, hidden_size), generator=generator)
        expected_prompt, prompt_caches = _dense_stage_forward(
            prompt,
            layers,
            attention_heads=attention_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
        )
        expected_next, next_caches = _dense_stage_forward(
            next_hidden,
            layers,
            attention_heads=attention_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            caches=prompt_caches,
        )
        truncated_caches = tuple(
            tuple(value[:, :, :2, :].contiguous() for value in cache)
            for cache in next_caches
        )
        expected_correction, _ = _dense_stage_forward(
            correction,
            layers,
            attention_heads=attention_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            caches=truncated_caches,
        )

        with tempfile.TemporaryDirectory() as temporary:
            fixture = write_llama_stage_cell_fixture(
                temporary,
                layers,
                world_size=2,
                num_attention_heads=attention_heads,
                num_key_value_heads=kv_heads,
                head_dim=head_dim,
            )
            stage_port = _free_port()
            downstream_port = _free_port()
            while downstream_port == stage_port:
                downstream_port = _free_port()
            downstream_listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            downstream_listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            downstream_listener.bind(("127.0.0.1", downstream_port))
            downstream_listener.listen(1)
            downstream_ready = threading.Event()
            downstream_frames: queue.Queue[Frame] = queue.Queue()
            downstream_errors: list[BaseException] = []

            def run_downstream() -> None:
                connection: socket.socket | None = None
                try:
                    downstream_ready.set()
                    connection, _ = downstream_listener.accept()
                    connection.settimeout(30)
                    hello = recv_frame(connection)
                    downstream_frames.put(hello)
                    send_frame(connection, FrameType.READY, hello.request_id)
                    while True:
                        frame = recv_frame(connection)
                        downstream_frames.put(frame)
                        if frame.frame_type == FrameType.SHUTDOWN:
                            break
                except BaseException as error:
                    downstream_errors.append(error)
                finally:
                    if connection is not None:
                        connection.close()

            downstream_worker = threading.Thread(target=run_downstream, daemon=True)
            downstream_worker.start()
            self.assertTrue(downstream_ready.wait(1))

            config = StageProcessConfig(
                spec=StageModelSpec("unused", 1, 3, 4, 1),
                pipeline_id=808,
                listen_host="127.0.0.1",
                listen_port=stage_port,
                next_host="127.0.0.1",
                next_port=downstream_port,
                next_layer_end=4,
                return_host="127.0.0.1",
                return_port=_free_port(),
                codec=TensorCodec.FP32,
                one_way_delay_ms=0.0,
                bandwidth_mbps=0.0,
                connect_timeout_seconds=10.0,
                cell_fixture=str(fixture),
                cell_world_size=2,
                cell_operation_timeout_seconds=30.0,
            )
            stage_ready = threading.Event()
            metrics: queue.Queue[dict[str, object]] = queue.Queue()
            stage_errors: list[BaseException] = []

            def run_stage() -> None:
                try:
                    run_stage_process(config, stage_ready, metrics)
                except BaseException as error:
                    stage_errors.append(error)

            stage_worker = threading.Thread(target=run_stage, daemon=True)
            stage_worker.start()
            self.assertTrue(stage_ready.wait(30), "cell stage did not bind its GDLP listener")
            upstream = socket.create_connection(("127.0.0.1", stage_port), timeout=30)
            upstream.settimeout(30)
            try:
                send_frame(
                    upstream,
                    FrameType.HELLO,
                    808,
                    step=1,
                    token_count=3,
                    hidden_size=hidden_size,
                    flags=int(TensorCodec.FP32),
                )
                downstream_hello = downstream_frames.get(timeout=30)
                self.assertEqual(downstream_hello.frame_type, FrameType.HELLO)
                self.assertEqual((downstream_hello.step, downstream_hello.token_count), (3, 4))
                self.assertEqual(recv_frame(upstream).frame_type, FrameType.READY)

                request_id = 909
                send_frame(upstream, FrameType.BEGIN, request_id)
                self.assertEqual(
                    downstream_frames.get(timeout=30).frame_type, FrameType.BEGIN
                )
                send_frame(
                    upstream,
                    FrameType.PREFILL,
                    request_id,
                    step=0,
                    token_count=3,
                    hidden_size=hidden_size,
                    flags=int(TensorCodec.FP32),
                    payload=encode_tensor(prompt, TensorCodec.FP32),
                )
                prefill = downstream_frames.get(timeout=30)
                self.assertEqual(prefill.frame_type, FrameType.PREFILL)
                self.assertTrue(
                    torch.allclose(
                        decode_tensor(prefill), expected_prompt, rtol=1e-5, atol=1e-5
                    )
                )

                send_frame(
                    upstream,
                    FrameType.ACTIVATION,
                    request_id,
                    step=1,
                    token_count=1,
                    hidden_size=hidden_size,
                    flags=int(TensorCodec.FP32),
                    payload=encode_tensor(next_hidden, TensorCodec.FP32),
                )
                decode = downstream_frames.get(timeout=30)
                self.assertEqual(decode.frame_type, FrameType.ACTIVATION)
                self.assertTrue(
                    torch.allclose(
                        decode_tensor(decode), expected_next, rtol=1e-5, atol=1e-5
                    )
                )

                send_frame(
                    upstream,
                    FrameType.TRUNCATE,
                    request_id,
                    token_count=2,
                )
                truncate = downstream_frames.get(timeout=30)
                self.assertEqual(truncate.frame_type, FrameType.TRUNCATE)
                self.assertEqual(truncate.token_count, 2)
                send_frame(
                    upstream,
                    FrameType.ACTIVATION,
                    request_id,
                    step=2,
                    token_count=1,
                    hidden_size=hidden_size,
                    flags=int(TensorCodec.FP32),
                    payload=encode_tensor(correction, TensorCodec.FP32),
                )
                corrected = downstream_frames.get(timeout=30)
                self.assertEqual(corrected.frame_type, FrameType.ACTIVATION)
                self.assertTrue(
                    torch.allclose(
                        decode_tensor(corrected),
                        expected_correction,
                        rtol=1e-5,
                        atol=1e-5,
                    )
                )

                send_frame(upstream, FrameType.END, request_id)
                self.assertEqual(
                    downstream_frames.get(timeout=30).frame_type, FrameType.END
                )
                metric = metrics.get(timeout=30)
                self.assertEqual(metric["request_id"], request_id)
                self.assertEqual(metric["frames"], 3)
                self.assertEqual(metric["tokens"], 5)
                self.assertEqual(
                    metric["loader"], "tensor-parallel-cell-safetensors-gloo"
                )

                send_frame(upstream, FrameType.SHUTDOWN, 808)
                self.assertEqual(
                    downstream_frames.get(timeout=30).frame_type, FrameType.SHUTDOWN
                )
            finally:
                upstream.close()
            stage_worker.join(30)
            downstream_worker.join(30)
            downstream_listener.close()
            self.assertFalse(stage_worker.is_alive(), "logical cell stage did not stop")
            self.assertFalse(downstream_worker.is_alive(), "downstream stub did not stop")
            self.assertEqual(stage_errors, [])
            self.assertEqual(downstream_errors, [])


@unittest.skipUnless(
    os.environ.get("RUN_DISTRIBUTED_MODEL_TESTS") == "1",
    "set RUN_DISTRIBUTED_MODEL_TESTS=1 for the real SmolLM cell integration",
)
class SmolLMRealTensorParallelCellTests(unittest.TestCase):
    MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"

    def test_real_layers_15_and_16_match_selective_transformers_stage(self) -> None:
        snapshot = resolve_model_snapshot(self.MODEL_NAME)
        config = AutoConfig.from_pretrained(snapshot)
        self.assertEqual(config.model_type, "llama")
        layer_indexes = (15, 16)
        dense_layers = tuple(
            _load_checkpoint_llama_layer(Path(snapshot), layer_index)
            for layer_index in layer_indexes
        )
        rope_theta = float(config.rope_parameters["rope_theta"])
        prompt = torch.randn(
            (1, 2, int(config.hidden_size)),
            generator=torch.Generator().manual_seed(401),
        )
        next_hidden = torch.randn(
            (1, 1, int(config.hidden_size)),
            generator=torch.Generator().manual_seed(402),
        )

        with tempfile.TemporaryDirectory() as temporary:
            fixture = write_llama_stage_cell_fixture(
                temporary,
                dense_layers,
                world_size=2,
                num_attention_heads=int(config.num_attention_heads),
                num_key_value_heads=int(config.num_key_value_heads),
                head_dim=int(config.head_dim),
                rms_norm_epsilon=float(config.rms_norm_eps),
                rope_theta=rope_theta,
            )
            reference = StageRunner(
                StageModelSpec(snapshot, 15, 17, int(config.num_hidden_layers), 1)
            )
            cell_config = StageProcessConfig(
                **{
                    **_stage_config(layer_end=17, total_layers=int(config.num_hidden_layers)).__dict__,
                    "spec": StageModelSpec(
                        snapshot,
                        15,
                        17,
                        int(config.num_hidden_layers),
                        1,
                    ),
                    "next_layer_end": 18,
                    "cell_fixture": str(fixture),
                    "cell_world_size": 2,
                    "cell_operation_timeout_seconds": 60.0,
                }
            )
            cell = build_stage_runner(cell_config)
            request_id = 606
            try:
                reference.begin(request_id)
                cell.begin(request_id)
                expected_prompt, _ = reference.forward_hidden(
                    request_id, prompt, token_mode="none"
                )
                actual_prompt, _ = cell.forward_hidden(
                    request_id, prompt, token_mode="none"
                )
                self.assertTrue(
                    torch.allclose(
                        actual_prompt,
                        expected_prompt,
                        rtol=2e-4,
                        atol=2e-4,
                    ),
                    f"real prefill max error: {(actual_prompt - expected_prompt).abs().max().item()}",
                )
                expected_next, _ = reference.forward_hidden(request_id, next_hidden)
                actual_next, _ = cell.forward_hidden(request_id, next_hidden)
                self.assertTrue(
                    torch.allclose(
                        actual_next,
                        expected_next,
                        rtol=2e-4,
                        atol=2e-4,
                    ),
                    f"real decode max error: {(actual_next - expected_next).abs().max().item()}",
                )
                self.assertEqual(reference.sequence_length(request_id), 3)
                self.assertEqual(cell.sequence_length(request_id), 3)
            finally:
                reference.end(request_id)
                cell.end(request_id)
                reference.close()
                cell.close()


def _stage_config(*, layer_end: int = 2, total_layers: int = 3) -> StageProcessConfig:
    return StageProcessConfig(
        spec=StageModelSpec("unused", 1, layer_end, total_layers, 1),
        pipeline_id=5,
        listen_host="127.0.0.1",
        listen_port=24001,
        next_host="127.0.0.1",
        next_port=24002,
        next_layer_end=total_layers,
        return_host="127.0.0.1",
        return_port=24003,
        codec=TensorCodec.FP32,
        one_way_delay_ms=0.0,
        bandwidth_mbps=0.0,
    )


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
    *,
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


def _free_port() -> int:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])
    finally:
        listener.close()


def _load_checkpoint_llama_layer(
    snapshot: Path,
    layer_index: int,
) -> dict[str, torch.Tensor]:
    """Load exactly nine tensors for one real checkpoint layer."""

    checkpoint_files = _checkpoint_key_map(snapshot)
    prefix = f"model.layers.{layer_index}."
    checkpoint_names = {
        "input_norm": prefix + "input_layernorm.weight",
        "post_attention_norm": prefix + "post_attention_layernorm.weight",
        "query": prefix + "self_attn.q_proj.weight",
        "key": prefix + "self_attn.k_proj.weight",
        "value": prefix + "self_attn.v_proj.weight",
        "output": prefix + "self_attn.o_proj.weight",
        "gate": prefix + "mlp.gate_proj.weight",
        "up": prefix + "mlp.up_proj.weight",
        "down": prefix + "mlp.down_proj.weight",
    }
    missing = [value for value in checkpoint_names.values() if value not in checkpoint_files]
    if missing:
        raise KeyError(f"checkpoint is missing Llama layer tensors: {missing}")
    by_file: dict[Path, list[tuple[str, str]]] = {}
    for local_name, checkpoint_name in checkpoint_names.items():
        by_file.setdefault(snapshot / checkpoint_files[checkpoint_name], []).append(
            (local_name, checkpoint_name)
        )
    loaded: dict[str, torch.Tensor] = {}
    for file_path, names in by_file.items():
        with safe_open(file_path, framework="pt", device="cpu") as tensors:
            for local_name, checkpoint_name in names:
                # Convert only this selected tensor; embeddings, heads and every
                # other layer remain memory-mapped and unread.
                loaded[local_name] = (
                    tensors.get_tensor(checkpoint_name).to(torch.float32).contiguous()
                )
    return loaded


if __name__ == "__main__":
    unittest.main()
