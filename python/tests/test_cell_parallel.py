from __future__ import annotations

from datetime import timedelta
import multiprocessing as multiprocessing
import queue
import socket
import unittest

import torch
import torch.distributed as distributed
from torch.nn import functional as functional

from distributed_runtime.cell_parallel import (
    balanced_shard,
    column_parallel_linear,
    llama_gated_mlp_tensor_parallel,
    llama_attention_shard_plan,
    llama_decoder_layer_tensor_parallel,
    row_parallel_linear,
    shard_column_linear,
    shard_row_linear,
    shard_llama_attention,
    shard_sizes,
    weighted_shard,
)


class CellParallelUnitTests(unittest.TestCase):
    def test_balanced_ranges_cover_uneven_dimension_exactly(self) -> None:
        ranges = [balanced_shard(10, rank, 3) for rank in range(3)]
        self.assertEqual([(value.start, value.end) for value in ranges], [(0, 4), (4, 7), (7, 10)])
        self.assertEqual(shard_sizes(10, 3), (4, 3, 3))
        self.assertEqual(sum(value.size for value in ranges), 10)

    def test_weighted_ranges_are_deterministic_nonempty_and_drive_attention(self) -> None:
        weights = (3.0, 1.0)
        ranges = [weighted_shard(8, rank, weights) for rank in range(2)]
        self.assertEqual([(value.start, value.end) for value in ranges], [(0, 6), (6, 8)])
        self.assertEqual(shard_sizes(10, 2, weights), (7, 3))
        self.assertEqual(
            shard_sizes(10, 3, (1.0, 1.0, 1.0)),
            shard_sizes(10, 3),
        )

        plans = [
            llama_attention_shard_plan(16, 8, 4, 2, rank, 2, weights)
            for rank in range(2)
        ]
        self.assertEqual(
            [(plan.query_heads.start, plan.query_heads.end) for plan in plans],
            [(0, 6), (6, 8)],
        )
        self.assertEqual(
            [(plan.key_value_heads.start, plan.key_value_heads.end) for plan in plans],
            [(0, 3), (3, 4)],
        )
        self.assertEqual(plans[0].query_output_sizes, (12, 4))
        self.assertEqual(plans[1].query_output_sizes, (12, 4))

        for invalid in ((), (1.0,), (1.0, 0.0), (1.0, float("nan"))):
            with self.subTest(invalid=invalid), self.assertRaises((TypeError, ValueError)):
                shard_sizes(8, 2, invalid)

    def test_single_member_paths_equal_dense_linear(self) -> None:
        generator = torch.Generator().manual_seed(7)
        inputs = torch.randn((2, 5), generator=generator)
        weight = torch.randn((7, 5), generator=generator)
        bias = torch.randn(7, generator=generator)
        local_weight, local_bias, _ = shard_column_linear(weight, bias, 0, 1)
        self.assertTrue(
            torch.equal(
                column_parallel_linear(inputs, local_weight, local_bias, (7,)),
                functional.linear(inputs, weight, bias),
            )
        )
        row_weight, _ = shard_row_linear(weight, 0, 1)
        self.assertTrue(
            torch.equal(
                row_parallel_linear(inputs, row_weight, bias, (5,)),
                functional.linear(inputs, weight, bias),
            )
        )

    def test_contract_rejects_invalid_shards_and_uninitialized_collective(self) -> None:
        for values in ((0, 0, 1), (4, -1, 2), (4, 2, 2), (2, 0, 3)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                balanced_shard(*values)
        inputs = torch.zeros((1, 2))
        weight = torch.zeros((2, 2))
        with self.assertRaisesRegex(RuntimeError, "must be initialized"):
            column_parallel_linear(inputs, weight, None, (2, 2))

    def test_mqa_plan_splits_query_heads_and_replicates_only_referenced_kv(self) -> None:
        plans = [
            llama_attention_shard_plan(8, 4, 1, 2, rank, 2)
            for rank in range(2)
        ]
        self.assertEqual(
            [(plan.query_heads.start, plan.query_heads.end) for plan in plans],
            [(0, 2), (2, 4)],
        )
        self.assertEqual(
            [(plan.key_value_heads.start, plan.key_value_heads.end) for plan in plans],
            [(0, 1), (0, 1)],
        )
        self.assertEqual(
            [plan.query_to_local_key_value for plan in plans],
            [(0, 0), (0, 0)],
        )
        self.assertEqual([plan.query_output_sizes for plan in plans], [(4, 4), (4, 4)])

        boundary_plans = [
            llama_attention_shard_plan(16, 8, 2, 2, rank, 3)
            for rank in range(3)
        ]
        self.assertEqual(
            [
                (plan.query_heads.start, plan.query_heads.end)
                for plan in boundary_plans
            ],
            [(0, 3), (3, 6), (6, 8)],
        )
        self.assertEqual(
            [
                (plan.key_value_heads.start, plan.key_value_heads.end)
                for plan in boundary_plans
            ],
            [(0, 1), (0, 2), (1, 2)],
        )
        self.assertEqual(
            [plan.query_to_local_key_value for plan in boundary_plans],
            [(0, 0, 0), (0, 1, 1), (0, 0)],
        )

        with self.assertRaisesRegex(ValueError, "world_size cannot exceed"):
            llama_attention_shard_plan(8, 4, 1, 2, 0, 5)


@unittest.skipUnless(
    distributed.is_available() and distributed.is_gloo_available(),
    "Gloo is required for the physical multi-process cell test",
)
class CellParallelPhysicalTests(unittest.TestCase):
    def test_two_members_match_dense_column_and_row_linear(self) -> None:
        generator = torch.Generator().manual_seed(19)
        column_input = torch.randn((3, 5), generator=generator)
        column_weight = torch.randn((7, 5), generator=generator)
        column_bias = torch.randn(7, generator=generator)
        expected_column = functional.linear(column_input, column_weight, column_bias)

        row_input = torch.randn((3, 7), generator=generator)
        row_weight = torch.randn((6, 7), generator=generator)
        row_bias = torch.randn(6, generator=generator)
        expected_row = functional.linear(row_input, row_weight, row_bias)

        mlp_input = torch.randn((3, 5), generator=generator)
        gate_weight = torch.randn((9, 5), generator=generator)
        up_weight = torch.randn((9, 5), generator=generator)
        down_weight = torch.randn((5, 9), generator=generator)
        expected_mlp = functional.linear(
            functional.silu(functional.linear(mlp_input, gate_weight))
            * functional.linear(mlp_input, up_weight),
            down_weight,
        )

        hidden_size = 16
        attention_heads = 4
        kv_heads = 2
        head_dim = 4
        layer_input = torch.randn((1, 3, hidden_size), generator=generator)
        next_layer_input = torch.randn((1, 1, hidden_size), generator=generator)
        input_norm = torch.randn(hidden_size, generator=generator)
        post_norm = torch.randn(hidden_size, generator=generator)
        query_weight = torch.randn((hidden_size, hidden_size), generator=generator)
        key_weight = torch.randn((kv_heads * head_dim, hidden_size), generator=generator)
        value_weight = torch.randn((kv_heads * head_dim, hidden_size), generator=generator)
        output_weight = torch.randn((hidden_size, hidden_size), generator=generator)
        layer_intermediate = 9
        layer_gate = torch.randn((layer_intermediate, hidden_size), generator=generator)
        layer_up = torch.randn((layer_intermediate, hidden_size), generator=generator)
        layer_down = torch.randn((hidden_size, layer_intermediate), generator=generator)
        full_plan = llama_attention_shard_plan(
            hidden_size, attention_heads, kv_heads, head_dim, 0, 1
        )
        expected_layer, dense_present = llama_decoder_layer_tensor_parallel(
            layer_input,
            input_norm,
            post_norm,
            query_weight,
            key_weight,
            value_weight,
            output_weight,
            full_plan,
            layer_gate,
            layer_up,
            layer_down,
            (layer_intermediate,),
        )
        expected_next_layer, dense_next_present = llama_decoder_layer_tensor_parallel(
            next_layer_input,
            input_norm,
            post_norm,
            query_weight,
            key_weight,
            value_weight,
            output_weight,
            full_plan,
            layer_gate,
            layer_up,
            layer_down,
            (layer_intermediate,),
            past_key_value=dense_present,
        )

        world_size = 2
        port = _reserve_port()
        context = multiprocessing.get_context("spawn")
        results = context.Queue()
        processes = []
        for rank in range(world_size):
            local_column_weight, local_column_bias, _ = shard_column_linear(
                column_weight, column_bias, rank, world_size
            )
            local_row_weight, row_range = shard_row_linear(
                row_weight, rank, world_size
            )
            local_gate_weight, _, _ = shard_column_linear(
                gate_weight, None, rank, world_size
            )
            local_up_weight, _, _ = shard_column_linear(
                up_weight, None, rank, world_size
            )
            local_down_weight, _ = shard_row_linear(
                down_weight, rank, world_size
            )
            local_query, local_key, local_value, local_output, attention_plan = (
                shard_llama_attention(
                    query_weight,
                    key_weight,
                    value_weight,
                    output_weight,
                    num_attention_heads=attention_heads,
                    num_key_value_heads=kv_heads,
                    head_dim=head_dim,
                    rank=rank,
                    world_size=world_size,
                )
            )
            local_layer_gate, _, _ = shard_column_linear(
                layer_gate, None, rank, world_size
            )
            local_layer_up, _, _ = shard_column_linear(
                layer_up, None, rank, world_size
            )
            local_layer_down, _ = shard_row_linear(
                layer_down, rank, world_size
            )
            process = context.Process(
                target=_cell_worker,
                args=(
                    rank,
                    world_size,
                    port,
                    column_input,
                    local_column_weight,
                    local_column_bias,
                    shard_sizes(column_weight.shape[0], world_size),
                    row_input[:, row_range.start : row_range.end].contiguous(),
                    local_row_weight,
                    row_bias,
                    shard_sizes(row_weight.shape[1], world_size),
                    mlp_input,
                    local_gate_weight,
                    local_up_weight,
                    local_down_weight,
                    shard_sizes(gate_weight.shape[0], world_size),
                    layer_input,
                    next_layer_input,
                    input_norm,
                    post_norm,
                    local_query,
                    local_key,
                    local_value,
                    local_output,
                    attention_plan,
                    local_layer_gate,
                    local_layer_up,
                    local_layer_down,
                    shard_sizes(layer_intermediate, world_size),
                    results,
                ),
            )
            process.start()
            processes.append(process)

        reports = []
        try:
            for _ in processes:
                reports.append(results.get(timeout=30))
        except queue.Empty as error:
            raise AssertionError("cell workers did not return before timeout") from error
        finally:
            for process in processes:
                process.join(timeout=10)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)

        self.assertTrue(all(process.exitcode == 0 for process in processes))
        errors = [report[1] for report in reports if report[0] == "error"]
        self.assertEqual(errors, [])
        rank_zero = next(report for report in reports if report[0] == "ok" and report[1] == 0)
        actual_column = torch.tensor(rank_zero[2])
        actual_row = torch.tensor(rank_zero[3])
        actual_mlp = torch.tensor(rank_zero[4])
        actual_layer = torch.tensor(rank_zero[5])
        actual_next_layer = torch.tensor(rank_zero[6])
        self.assertTrue(torch.allclose(actual_column, expected_column, rtol=1e-6, atol=1e-6))
        self.assertTrue(torch.allclose(actual_row, expected_row, rtol=1e-6, atol=1e-6))
        self.assertTrue(torch.allclose(actual_mlp, expected_mlp, rtol=1e-5, atol=1e-6))
        self.assertTrue(torch.allclose(actual_layer, expected_layer, rtol=1e-5, atol=1e-5))
        self.assertTrue(
            torch.allclose(
                actual_next_layer,
                expected_next_layer,
                rtol=1e-5,
                atol=1e-5,
            )
        )
        expected_local_cache_shape = [1, kv_heads // world_size, 4, head_dim]
        for report in reports:
            if report[0] == "ok":
                self.assertEqual(report[7], expected_local_cache_shape)
                self.assertEqual(report[8], expected_local_cache_shape)
        self.assertEqual(list(dense_next_present[0].shape), [1, kv_heads, 4, head_dim])
        self.assertEqual(list(dense_next_present[1].shape), [1, kv_heads, 4, head_dim])

    def test_two_members_match_dense_mqa_when_world_exceeds_kv_heads(self) -> None:
        generator = torch.Generator().manual_seed(43)
        hidden_size = 8
        attention_heads = 4
        kv_heads = 1
        head_dim = 2
        intermediate_size = 7
        prefill = torch.randn((1, 3, hidden_size), generator=generator)
        decode = torch.randn((1, 1, hidden_size), generator=generator)
        input_norm = torch.randn(hidden_size, generator=generator)
        post_norm = torch.randn(hidden_size, generator=generator)
        query_weight = torch.randn((hidden_size, hidden_size), generator=generator)
        key_weight = torch.randn((kv_heads * head_dim, hidden_size), generator=generator)
        value_weight = torch.randn((kv_heads * head_dim, hidden_size), generator=generator)
        output_weight = torch.randn((hidden_size, hidden_size), generator=generator)
        gate_weight = torch.randn((intermediate_size, hidden_size), generator=generator)
        up_weight = torch.randn((intermediate_size, hidden_size), generator=generator)
        down_weight = torch.randn((hidden_size, intermediate_size), generator=generator)

        dense_plan = llama_attention_shard_plan(
            hidden_size, attention_heads, kv_heads, head_dim, 0, 1
        )
        expected_prefill, dense_present = llama_decoder_layer_tensor_parallel(
            prefill,
            input_norm,
            post_norm,
            query_weight,
            key_weight,
            value_weight,
            output_weight,
            dense_plan,
            gate_weight,
            up_weight,
            down_weight,
            (intermediate_size,),
        )
        expected_decode, dense_next_present = llama_decoder_layer_tensor_parallel(
            decode,
            input_norm,
            post_norm,
            query_weight,
            key_weight,
            value_weight,
            output_weight,
            dense_plan,
            gate_weight,
            up_weight,
            down_weight,
            (intermediate_size,),
            past_key_value=dense_present,
        )

        world_size = 2
        port = _reserve_port()
        context = multiprocessing.get_context("spawn")
        results = context.Queue()
        processes = []
        local_keys = []
        local_values = []
        for rank in range(world_size):
            local_query, local_key, local_value, local_output, plan = (
                shard_llama_attention(
                    query_weight,
                    key_weight,
                    value_weight,
                    output_weight,
                    num_attention_heads=attention_heads,
                    num_key_value_heads=kv_heads,
                    head_dim=head_dim,
                    rank=rank,
                    world_size=world_size,
                )
            )
            local_gate, _, _ = shard_column_linear(
                gate_weight, None, rank, world_size
            )
            local_up, _, _ = shard_column_linear(
                up_weight, None, rank, world_size
            )
            local_down, _ = shard_row_linear(down_weight, rank, world_size)
            local_keys.append(local_key)
            local_values.append(local_value)
            process = context.Process(
                target=_mqa_cell_worker,
                args=(
                    rank,
                    world_size,
                    port,
                    prefill,
                    decode,
                    input_norm,
                    post_norm,
                    local_query,
                    local_key,
                    local_value,
                    local_output,
                    plan,
                    local_gate,
                    local_up,
                    local_down,
                    shard_sizes(intermediate_size, world_size),
                    results,
                ),
            )
            process.start()
            processes.append(process)

        # MQA has one KV head, so both ranks intentionally carry the same
        # minimal K/V shard while Q/O and MLP weights remain partitioned.
        self.assertTrue(torch.equal(local_keys[0], local_keys[1]))
        self.assertTrue(torch.equal(local_values[0], local_values[1]))

        reports = []
        try:
            for _ in processes:
                reports.append(results.get(timeout=30))
        except queue.Empty as error:
            raise AssertionError("MQA cell workers did not return before timeout") from error
        finally:
            for process in processes:
                process.join(timeout=10)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)

        self.assertTrue(all(process.exitcode == 0 for process in processes))
        errors = [report[1] for report in reports if report[0] == "error"]
        self.assertEqual(errors, [])
        rank_zero = next(report for report in reports if report[0] == "ok" and report[1] == 0)
        torch.testing.assert_close(
            torch.tensor(rank_zero[2]), expected_prefill, rtol=1e-5, atol=1e-5
        )
        torch.testing.assert_close(
            torch.tensor(rank_zero[3]), expected_decode, rtol=1e-5, atol=1e-5
        )
        for report in reports:
            if report[0] == "ok":
                self.assertEqual(report[4], [1, 1, 4, head_dim])
                self.assertEqual(report[5], [1, 1, 4, head_dim])
        self.assertEqual(list(dense_next_present[0].shape), [1, 1, 4, head_dim])
        self.assertEqual(list(dense_next_present[1].shape), [1, 1, 4, head_dim])


def _cell_worker(
    rank: int,
    world_size: int,
    port: int,
    column_input: torch.Tensor,
    column_weight: torch.Tensor,
    column_bias: torch.Tensor,
    column_sizes: tuple[int, ...],
    row_input: torch.Tensor,
    row_weight: torch.Tensor,
    row_bias: torch.Tensor,
    row_sizes: tuple[int, ...],
    mlp_input: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    intermediate_sizes: tuple[int, ...],
    layer_input: torch.Tensor,
    next_layer_input: torch.Tensor,
    input_norm: torch.Tensor,
    post_norm: torch.Tensor,
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    value_weight: torch.Tensor,
    output_weight: torch.Tensor,
    attention_plan: object,
    layer_gate: torch.Tensor,
    layer_up: torch.Tensor,
    layer_down: torch.Tensor,
    layer_intermediate_sizes: tuple[int, ...],
    results: object,
) -> None:
    try:
        distributed.init_process_group(
            backend="gloo",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=20),
        )
        column = column_parallel_linear(
            column_input,
            column_weight,
            column_bias,
            column_sizes,
        )
        row = row_parallel_linear(
            row_input,
            row_weight,
            row_bias,
            row_sizes,
        )
        mlp = llama_gated_mlp_tensor_parallel(
            mlp_input,
            gate_weight,
            up_weight,
            down_weight,
            intermediate_sizes,
        )
        layer, present = llama_decoder_layer_tensor_parallel(
            layer_input,
            input_norm,
            post_norm,
            query_weight,
            key_weight,
            value_weight,
            output_weight,
            attention_plan,
            layer_gate,
            layer_up,
            layer_down,
            layer_intermediate_sizes,
        )
        next_layer, next_present = llama_decoder_layer_tensor_parallel(
            next_layer_input,
            input_norm,
            post_norm,
            query_weight,
            key_weight,
            value_weight,
            output_weight,
            attention_plan,
            layer_gate,
            layer_up,
            layer_down,
            layer_intermediate_sizes,
            past_key_value=present,
        )
        results.put(
            (
                "ok",
                rank,
                column.tolist(),
                row.tolist(),
                mlp.tolist(),
                layer.tolist(),
                next_layer.tolist(),
                list(next_present[0].shape),
                list(next_present[1].shape),
            )
        )
    except BaseException as error:
        results.put(("error", f"{type(error).__name__}: {error}"))
        raise
    finally:
        if distributed.is_initialized():
            distributed.destroy_process_group()


def _mqa_cell_worker(
    rank: int,
    world_size: int,
    port: int,
    prefill: torch.Tensor,
    decode: torch.Tensor,
    input_norm: torch.Tensor,
    post_norm: torch.Tensor,
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    value_weight: torch.Tensor,
    output_weight: torch.Tensor,
    attention_plan: object,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    intermediate_sizes: tuple[int, ...],
    results: object,
) -> None:
    try:
        distributed.init_process_group(
            backend="gloo",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=20),
        )
        actual_prefill, present = llama_decoder_layer_tensor_parallel(
            prefill,
            input_norm,
            post_norm,
            query_weight,
            key_weight,
            value_weight,
            output_weight,
            attention_plan,
            gate_weight,
            up_weight,
            down_weight,
            intermediate_sizes,
        )
        actual_decode, next_present = llama_decoder_layer_tensor_parallel(
            decode,
            input_norm,
            post_norm,
            query_weight,
            key_weight,
            value_weight,
            output_weight,
            attention_plan,
            gate_weight,
            up_weight,
            down_weight,
            intermediate_sizes,
            past_key_value=present,
        )
        results.put(
            (
                "ok",
                rank,
                actual_prefill.tolist(),
                actual_decode.tolist(),
                list(next_present[0].shape),
                list(next_present[1].shape),
            )
        )
    except BaseException as error:
        results.put(("error", f"{type(error).__name__}: {error}"))
        raise
    finally:
        if distributed.is_initialized():
            distributed.destroy_process_group()


def _reserve_port() -> int:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])
    finally:
        listener.close()


if __name__ == "__main__":
    unittest.main()
