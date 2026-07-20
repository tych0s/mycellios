from __future__ import annotations

import threading
import time
import unittest

import torch

from distributed_runtime.tiled_expert import (
    InMemoryTiledExpertOwner,
    InMemoryTiledSwiGLUExecutor,
    SwiGLUTileWeights,
    SwiGLUTilingSpec,
    SwiGLUWeights,
    TILED_EXPERT_MODE,
    TILED_EXPERT_REDUCTION_ORDER,
    TiledExpertWorkerProfile,
    canonical_swiglu_tile,
    monolithic_swiglu,
    plan_tiled_swiglu_expert,
)


_ARTIFACT_ID = "sha256:" + "a" * 64
_COORDINATOR_BUDGET = 1_000_000
_COORDINATOR_EGRESS_MBPS = 10_000.0
_COORDINATOR_INGRESS_MBPS = 10_000.0


def _worker(
    owner_id: str,
    *,
    budget: int = 1_200,
    rtt_ms: float = 1.0,
    compute_gflops: float = 0.001,
    bandwidth_mbps: float = 1_000,
) -> TiledExpertWorkerProfile:
    return TiledExpertWorkerProfile(
        owner_id=owner_id,
        usable_vram_bytes=budget,
        reserved_vram_bytes=0,
        compute_gflops=compute_gflops,
        round_trip_ms=rtt_ms,
        bandwidth_mbps=bandwidth_mbps,
    )


def _weights(
    *,
    hidden: int = 8,
    intermediate: int = 13,
    dtype: torch.dtype = torch.float32,
    seed: int = 20260720,
) -> SwiGLUWeights:
    generator = torch.Generator().manual_seed(seed)
    return SwiGLUWeights.from_tensors(
        _ARTIFACT_ID,
        gate=torch.randn(intermediate, hidden, generator=generator, dtype=dtype) * 0.2,
        up=torch.randn(intermediate, hidden, generator=generator, dtype=dtype) * 0.2,
        down=torch.randn(hidden, intermediate, generator=generator, dtype=dtype) * 0.2,
    )


def _spec(weights: SwiGLUWeights, *, positions: int = 5) -> SwiGLUTilingSpec:
    dtype_name = weights.dtype_name
    element_bytes = weights.gate.element_size()
    return SwiGLUTilingSpec(
        hidden_size=weights.hidden_size,
        intermediate_size=weights.intermediate_size,
        positions=positions,
        artifact_identity=weights.artifact_identity,
        expert_content_id=weights.expert_content_id,
        weight_dtype=dtype_name,
        activation_dtype=dtype_name,
        accumulation_dtype=dtype_name,
        weight_element_bytes=element_bytes,
        activation_element_bytes=element_bytes,
    )


class _BarrierOwner(InMemoryTiledExpertOwner):
    def __init__(self, *args, barrier: threading.Barrier, delay_s: float, **kwargs):
        super().__init__(*args, **kwargs)
        self._barrier = barrier
        self._delay_s = delay_s

    def execute_tile(self, tile, hidden):
        self._barrier.wait(timeout=5)
        time.sleep(self._delay_s)
        return super().execute_tile(tile, hidden)


class _CountingOwner(InMemoryTiledExpertOwner):
    def __init__(self, *args, started: list[str], **kwargs):
        super().__init__(*args, **kwargs)
        self._started = started

    def execute_tile(self, tile, hidden):
        self._started.append(self.owner_id)
        return super().execute_tile(tile, hidden)


class TiledExpertTests(unittest.TestCase):
    def test_canonical_tiles_cover_and_reconstruct_the_intermediate_axis(self) -> None:
        weights = _weights()
        spec = _spec(weights)
        plan = plan_tiled_swiglu_expert(
            spec,
            (_worker("owner-c"), _worker("owner-a"), _worker("owner-b")),
            weights=weights,
            coordinator_vram_budget_bytes=_COORDINATOR_BUDGET,
            coordinator_egress_mbps=_COORDINATOR_EGRESS_MBPS,
            coordinator_ingress_mbps=_COORDINATOR_INGRESS_MBPS,
        )
        slices = tuple(canonical_swiglu_tile(weights, tile) for tile in plan.tiles)

        self.assertEqual(plan.mode, TILED_EXPERT_MODE)
        self.assertEqual(plan.reduction_order, TILED_EXPERT_REDUCTION_ORDER)
        self.assertFalse(plan.bitwise_equivalent_to_monolithic)
        cursor = 0
        for tile in plan.tiles:
            self.assertEqual(tile.intermediate_start, cursor)
            cursor = tile.intermediate_end
        self.assertEqual(cursor, spec.intermediate_size)
        self.assertEqual(len({tile.owner_id for tile in plan.tiles}), len(plan.tiles))
        self.assertTrue(
            all(tile.vram_required_bytes <= tile.worker_vram_budget_bytes for tile in plan.tiles)
        )
        self.assertEqual(plan.total_weight_bytes, spec.total_expert_weight_bytes)
        torch.testing.assert_close(
            torch.cat(tuple(tile.gate for tile in slices), dim=0),
            weights.gate,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            torch.cat(tuple(tile.up for tile in slices), dim=0),
            weights.up,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            torch.cat(tuple(tile.down for tile in slices), dim=1),
            weights.down,
            rtol=0,
            atol=0,
        )

    def test_parallel_fp32_executor_is_close_deterministic_and_token_stable(self) -> None:
        weights = _weights()
        spec = _spec(weights)
        plan = plan_tiled_swiglu_expert(
            spec,
            (_worker("owner-a"), _worker("owner-b"), _worker("owner-c")),
            weights=weights,
            coordinator_vram_budget_bytes=_COORDINATOR_BUDGET,
            coordinator_egress_mbps=_COORDINATOR_EGRESS_MBPS,
            coordinator_ingress_mbps=_COORDINATOR_INGRESS_MBPS,
        )
        self.assertGreater(len(plan.tiles), 1)
        barrier = threading.Barrier(len(plan.tiles))
        owners = {}
        for tile in plan.tiles:
            owners[tile.owner_id] = _BarrierOwner(
                tile.owner_id,
                tile,
                canonical_swiglu_tile(weights, tile),
                barrier=barrier,
                # Reverse completion pressure demonstrates that reduction does
                # not depend on which future finishes first.
                delay_s=(len(plan.tiles) - tile.tile_index) * 0.002,
            )
        executor = InMemoryTiledSwiGLUExecutor(plan, owners)
        hidden = torch.randn(5, 8, generator=torch.Generator().manual_seed(99))

        expected = monolithic_swiglu(hidden, weights)
        first = executor.execute(hidden)
        second = executor.execute(hidden)

        torch.testing.assert_close(first, expected, rtol=2e-5, atol=2e-6)
        self.assertTrue(torch.equal(first, second))
        classifier = torch.tensor(
            [
                [0.5, -0.1, 0.3, 0.2, -0.4, 0.8, 0.1, -0.2],
                [-0.2, 0.7, 0.1, -0.3, 0.6, 0.2, -0.5, 0.4],
                [0.3, 0.1, -0.6, 0.9, 0.2, -0.1, 0.5, 0.7],
            ],
            dtype=torch.float32,
        )
        expected_tokens = (expected @ classifier.T).argmax(dim=-1)
        actual_tokens = (first @ classifier.T).argmax(dim=-1)
        self.assertTrue(torch.equal(actual_tokens, expected_tokens))

    def test_projection_counts_weights_activation_round_trip_and_straggler(self) -> None:
        weights = _weights()
        spec = _spec(weights)
        plan = plan_tiled_swiglu_expert(
            spec,
            (
                _worker("fast", rtt_ms=0.5),
                _worker("medium", rtt_ms=2.0),
                _worker("straggler", rtt_ms=8.0),
            ),
            weights=weights,
            coordinator_vram_budget_bytes=_COORDINATOR_BUDGET,
            coordinator_egress_mbps=_COORDINATOR_EGRESS_MBPS,
            coordinator_ingress_mbps=_COORDINATOR_INGRESS_MBPS,
            reduction_gbytes_per_second=20,
        )

        self.assertEqual(
            plan.total_weight_bytes,
            3 * spec.hidden_size * spec.intermediate_size * spec.weight_element_bytes,
        )
        self.assertEqual(
            plan.activation_round_trip_bytes,
            len(plan.tiles) * 2 * spec.positions * spec.hidden_size * 4,
        )
        self.assertAlmostEqual(
            plan.per_owner_parallel_ms,
            max(tile.exposed_ms for tile in plan.tiles),
        )
        self.assertAlmostEqual(
            plan.fanout_parallel_ms,
            plan.coordinator_rpc_setup_ms
            + max(
                plan.per_owner_parallel_ms,
                plan.coordinator_causal_path_ms,
            ),
        )
        self.assertAlmostEqual(
            plan.coordinator_causal_path_ms,
            plan.coordinator_egress_ms
            + max(tile.round_trip_ms + tile.compute_ms for tile in plan.tiles)
            + plan.coordinator_ingress_ms,
        )
        self.assertAlmostEqual(
            plan.serial_owner_sum_ms,
            sum(tile.exposed_ms for tile in plan.tiles),
        )
        self.assertLess(plan.fanout_parallel_ms, plan.serial_owner_sum_ms)
        self.assertAlmostEqual(
            plan.projected_exposed_ms,
            plan.fanout_parallel_ms + plan.reduction_ms,
        )
        self.assertEqual(
            plan.coordinator_partial_output_bytes,
            len(plan.tiles) * spec.partial_output_bytes,
        )
        self.assertEqual(
            plan.coordinator_vram_required_bytes,
            plan.coordinator_reserved_vram_bytes
            + plan.coordinator_input_activation_bytes
            + plan.coordinator_partial_output_bytes
            + plan.coordinator_reduction_output_bytes
            + plan.coordinator_workspace_bytes,
        )
        self.assertLessEqual(
            plan.coordinator_vram_required_bytes,
            plan.coordinator_vram_budget_bytes,
        )

    def test_two_fast_owner_links_are_limited_by_shared_coordinator_nic(self) -> None:
        weights = _weights(intermediate=8)
        spec = _spec(weights)
        plan = plan_tiled_swiglu_expert(
            spec,
            (
                _worker(
                    "owner-a",
                    rtt_ms=0,
                    compute_gflops=1_000_000,
                    bandwidth_mbps=1_000_000,
                ),
                _worker(
                    "owner-b",
                    rtt_ms=0,
                    compute_gflops=1_000_000,
                    bandwidth_mbps=1_000_000,
                ),
            ),
            weights=weights,
            coordinator_vram_budget_bytes=_COORDINATOR_BUDGET,
            coordinator_egress_mbps=1.0,
            coordinator_ingress_mbps=1_000.0,
            coordinator_rpc_setup_ms_per_owner=0.1,
        )

        self.assertEqual(len(plan.tiles), 2)
        self.assertEqual(
            plan.coordinator_egress_bytes,
            len(plan.tiles) * spec.input_activation_bytes,
        )
        self.assertAlmostEqual(
            plan.coordinator_egress_ms,
            plan.coordinator_egress_bytes / 125.0,
        )
        self.assertGreater(
            plan.coordinator_egress_ms,
            plan.per_owner_parallel_ms,
        )
        self.assertAlmostEqual(plan.coordinator_rpc_setup_ms, 0.2)
        self.assertAlmostEqual(
            plan.fanout_parallel_ms,
            plan.coordinator_rpc_setup_ms
            + max(
                plan.per_owner_parallel_ms,
                plan.coordinator_causal_path_ms,
            ),
        )
        self.assertGreater(plan.fanout_parallel_ms, plan.per_owner_parallel_ms * 100)

        with self.assertRaisesRegex(ValueError, "coordinator_egress_mbps"):
            plan_tiled_swiglu_expert(
                spec,
                (_worker("owner-a"), _worker("owner-b")),
                weights=weights,
                coordinator_vram_budget_bytes=_COORDINATOR_BUDGET,
                coordinator_egress_mbps=0,
                coordinator_ingress_mbps=_COORDINATOR_INGRESS_MBPS,
            )

    def test_slow_bidirectional_caps_are_causal_not_maxed_independently(self) -> None:
        weights = _weights(intermediate=8)
        spec = _spec(weights)
        plan = plan_tiled_swiglu_expert(
            spec,
            (
                _worker(
                    "owner-a",
                    rtt_ms=0,
                    compute_gflops=1_000_000,
                    bandwidth_mbps=1_000_000,
                ),
                _worker(
                    "owner-b",
                    rtt_ms=0,
                    compute_gflops=1_000_000,
                    bandwidth_mbps=1_000_000,
                ),
            ),
            weights=weights,
            coordinator_vram_budget_bytes=_COORDINATOR_BUDGET,
            coordinator_egress_mbps=1.0,
            coordinator_ingress_mbps=1.0,
        )

        old_optimistic_fanout = max(
            plan.per_owner_parallel_ms,
            plan.coordinator_egress_ms,
            plan.coordinator_ingress_ms,
        )
        self.assertAlmostEqual(
            plan.coordinator_causal_path_ms,
            plan.coordinator_egress_ms
            + max(tile.round_trip_ms + tile.compute_ms for tile in plan.tiles)
            + plan.coordinator_ingress_ms,
        )
        self.assertAlmostEqual(
            plan.fanout_parallel_ms,
            max(plan.per_owner_parallel_ms, plan.coordinator_causal_path_ms),
        )
        self.assertGreater(plan.fanout_parallel_ms, old_optimistic_fanout * 1.99)

    def test_coordinator_budget_fails_closed_during_planning(self) -> None:
        weights = _weights()
        spec = _spec(weights)

        with self.assertRaisesRegex(MemoryError, "coordinator VRAM"):
            plan_tiled_swiglu_expert(
                spec,
                (_worker("owner-a"), _worker("owner-b"), _worker("owner-c")),
                weights=weights,
                coordinator_vram_budget_bytes=1,
                coordinator_egress_mbps=_COORDINATOR_EGRESS_MBPS,
                coordinator_ingress_mbps=_COORDINATOR_INGRESS_MBPS,
            )

    def test_same_shape_wrong_tile_is_rejected_before_any_worker_starts(self) -> None:
        weights = _weights()
        spec = _spec(weights)
        plan = plan_tiled_swiglu_expert(
            spec,
            (_worker("owner-a"), _worker("owner-b"), _worker("owner-c")),
            weights=weights,
            coordinator_vram_budget_bytes=_COORDINATOR_BUDGET,
            coordinator_egress_mbps=_COORDINATOR_EGRESS_MBPS,
            coordinator_ingress_mbps=_COORDINATOR_INGRESS_MBPS,
        )
        bad_tile = plan.tiles[-1]
        canonical = canonical_swiglu_tile(weights, bad_tile)
        bad_gate = canonical.gate.clone()
        bad_gate.view(-1)[0].add_(1)
        wrong_same_shape = SwiGLUTileWeights.from_tensors(
            artifact_identity=canonical.artifact_identity,
            expert_content_id=canonical.expert_content_id,
            intermediate_start=canonical.intermediate_start,
            intermediate_end=canonical.intermediate_end,
            full_intermediate_size=canonical.full_intermediate_size,
            gate=bad_gate,
            up=canonical.up.clone(),
            down=canonical.down.clone(),
        )
        self.assertEqual(wrong_same_shape.byte_size, bad_tile.weight_bytes)
        self.assertNotEqual(wrong_same_shape.tile_content_id, bad_tile.tile_content_id)

        started: list[str] = []
        owners = {}
        for tile in plan.tiles:
            tile_weights = (
                wrong_same_shape
                if tile == bad_tile
                else canonical_swiglu_tile(weights, tile)
            )
            owners[tile.owner_id] = _CountingOwner(
                tile.owner_id,
                tile,
                tile_weights,
                started=started,
            )
        executor = InMemoryTiledSwiGLUExecutor(plan, owners)

        with self.assertRaisesRegex(ValueError, "tile content"):
            executor.execute(torch.ones(spec.positions, spec.hidden_size))
        self.assertEqual(started, [])

    def test_inference_tensor_mutation_cannot_bypass_content_identity(self) -> None:
        with torch.inference_mode():
            weights = SwiGLUWeights.from_tensors(
                _ARTIFACT_ID,
                gate=torch.ones(2, 3),
                up=torch.ones(2, 3),
                down=torch.ones(3, 2),
            )
            self.assertEqual(weights._content_versions, (None, None, None))
            weights.gate.fill_(2)
            with self.assertRaisesRegex(ValueError, "changed after content sealing"):
                weights.validate_content()

    def test_equal_width_wrong_dtype_is_rejected_before_any_worker_starts(self) -> None:
        weights = _weights(dtype=torch.float16)
        spec = _spec(weights)
        plan = plan_tiled_swiglu_expert(
            spec,
            (_worker("owner-a"), _worker("owner-b"), _worker("owner-c")),
            weights=weights,
            coordinator_vram_budget_bytes=_COORDINATOR_BUDGET,
            coordinator_egress_mbps=_COORDINATOR_EGRESS_MBPS,
            coordinator_ingress_mbps=_COORDINATOR_INGRESS_MBPS,
        )
        started: list[str] = []
        owners = {
            tile.owner_id: _CountingOwner(
                tile.owner_id,
                tile,
                canonical_swiglu_tile(weights, tile),
                started=started,
            )
            for tile in plan.tiles
        }
        hidden = torch.ones(
            spec.positions,
            spec.hidden_size,
            dtype=torch.bfloat16,
        )
        self.assertEqual(hidden.element_size(), weights.gate.element_size())

        with self.assertRaisesRegex(ValueError, "activation dtype"):
            InMemoryTiledSwiGLUExecutor(plan, owners).execute(hidden)
        self.assertEqual(started, [])

        with self.assertRaisesRegex(ValueError, "identical weight"):
            SwiGLUTilingSpec(
                hidden_size=weights.hidden_size,
                intermediate_size=weights.intermediate_size,
                positions=5,
                artifact_identity=weights.artifact_identity,
                expert_content_id=weights.expert_content_id,
                weight_dtype="float16",
                activation_dtype="float16",
                accumulation_dtype="bfloat16",
                weight_element_bytes=2,
                activation_element_bytes=2,
            )

    def test_invalid_budgets_and_oversized_owner_fail_before_execution(self) -> None:
        weights = _weights()
        spec = _spec(weights)
        with self.assertRaisesRegex(ValueError, "reserved_vram_bytes"):
            TiledExpertWorkerProfile(
                owner_id="invalid",
                usable_vram_bytes=500,
                reserved_vram_bytes=500,
                compute_gflops=1,
                round_trip_ms=1,
                bandwidth_mbps=1_000,
            )
        with self.assertRaisesRegex(MemoryError, "one canonical intermediate"):
            plan_tiled_swiglu_expert(
                spec,
                (_worker("tiny-a", budget=475), _worker("tiny-b", budget=475)),
                weights=weights,
                coordinator_vram_budget_bytes=_COORDINATOR_BUDGET,
                coordinator_egress_mbps=_COORDINATOR_EGRESS_MBPS,
                coordinator_ingress_mbps=_COORDINATOR_INGRESS_MBPS,
            )
        with self.assertRaisesRegex(MemoryError, "combined worker VRAM"):
            plan_tiled_swiglu_expert(
                spec,
                (_worker("small-a", budget=632), _worker("small-b", budget=632)),
                weights=weights,
                coordinator_vram_budget_bytes=_COORDINATOR_BUDGET,
                coordinator_egress_mbps=_COORDINATOR_EGRESS_MBPS,
                coordinator_ingress_mbps=_COORDINATOR_INGRESS_MBPS,
            )

        plan = plan_tiled_swiglu_expert(
            spec,
            (_worker("owner-a"), _worker("owner-b"), _worker("owner-c")),
            weights=weights,
            coordinator_vram_budget_bytes=_COORDINATOR_BUDGET,
            coordinator_egress_mbps=_COORDINATOR_EGRESS_MBPS,
            coordinator_ingress_mbps=_COORDINATOR_INGRESS_MBPS,
        )
        owners = {
            tile.owner_id: InMemoryTiledExpertOwner.from_monolithic(tile, weights)
            for tile in plan.tiles
        }
        oversized = plan.tiles[0]
        owners[oversized.owner_id] = InMemoryTiledExpertOwner.from_monolithic(
            oversized,
            weights,
            vram_budget_bytes=oversized.vram_required_bytes - 1,
        )
        executor = InMemoryTiledSwiGLUExecutor(plan, owners)

        with self.assertRaisesRegex(MemoryError, "requires"):
            executor.execute(torch.ones(spec.positions, spec.hidden_size))


if __name__ == "__main__":
    unittest.main()
