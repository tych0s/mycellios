from __future__ import annotations

import unittest

from distributed_runtime.ram_expert_cache import (
    ExpertKey,
    ExpertRecord,
    MacroStageExpertInventory,
    PredictiveCacheConfig,
    RamBackedExpertScheduler,
    estimate_ram_pcie_break_even,
)


def _inventory(
    *,
    layers: int = 1,
    experts: int = 4,
    expert_bytes: int = 20,
) -> MacroStageExpertInventory:
    return MacroStageExpertInventory.uniform(
        layer_start=10,
        layer_end=10 + layers,
        num_experts=experts,
        expert_bytes=expert_bytes,
        content_prefix="test-model",
    )


def _scheduler(
    inventory: MacroStageExpertInventory,
    loads: list[ExpertKey],
    *,
    capacity_bytes: int = 100,
    prefetch_reserve_bytes: int = 40,
    bandwidth_gbytes_per_second: float = 0.001,
    hotness_decay: float = 1.0,
    min_confidence: float = 0.0,
) -> RamBackedExpertScheduler:
    return RamBackedExpertScheduler(
        inventory,
        PredictiveCacheConfig(
            capacity_bytes=capacity_bytes,
            prefetch_reserve_bytes=prefetch_reserve_bytes,
            pcie_bandwidth_gbytes_per_second=bandwidth_gbytes_per_second,
            hotness_decay=hotness_decay,
            min_prefetch_confidence=min_confidence,
        ),
        ram_loader=lambda record: loads.append(record.key),
    )


class MacroStageExpertInventoryTests(unittest.TestCase):
    def test_inventory_is_contiguous_and_content_addressed(self) -> None:
        inventory = _inventory(layers=2, experts=3, expert_bytes=12)
        self.assertEqual(inventory.layers, (10, 11))
        self.assertEqual(inventory.layer_start, 10)
        self.assertEqual(inventory.layer_end, 12)
        self.assertEqual(inventory.total_bytes, 72)
        record = inventory.record(ExpertKey(11, 2))
        self.assertEqual(record.byte_size, 12)
        self.assertEqual(record.content_id, "test-model:l11:e2")

    def test_rejects_layer_gaps_and_duplicate_experts(self) -> None:
        first = ExpertRecord(ExpertKey(0, 0), 10, "a")
        with self.assertRaisesRegex(ValueError, "contiguous"):
            MacroStageExpertInventory(
                (first, ExpertRecord(ExpertKey(2, 0), 10, "b"))
            )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            MacroStageExpertInventory((first, first))


class RamBackedExpertSchedulerTests(unittest.TestCase):
    def test_hot_expert_survives_before_cold_lru_entry(self) -> None:
        loads: list[ExpertKey] = []
        scheduler = _scheduler(
            _inventory(experts=3),
            loads,
            capacity_bytes=60,
            prefetch_reserve_bytes=20,
        )

        scheduler.resolve_route(layer=10, authoritative_expert_ids=(0,))
        scheduler.resolve_route(layer=10, authoritative_expert_ids=(1,))
        scheduler.resolve_route(layer=10, authoritative_expert_ids=(0,))
        result = scheduler.resolve_route(layer=10, authoritative_expert_ids=(2,))

        snapshot = scheduler.snapshot()
        self.assertEqual(loads, [ExpertKey(10, 0), ExpertKey(10, 1), ExpertKey(10, 2)])
        self.assertIn(ExpertKey(10, 0), snapshot.active_keys)
        self.assertIn(ExpertKey(10, 2), snapshot.active_keys)
        self.assertNotIn(ExpertKey(10, 1), snapshot.active_keys)
        self.assertEqual(result.evicted, (ExpertKey(10, 1),))
        self.assertEqual(snapshot.active_bytes, 40)
        self.assertLessEqual(snapshot.active_bytes, 40)

    def test_prefetch_uses_deadline_then_confidence_and_a_double_buffer(self) -> None:
        loads: list[ExpertKey] = []
        scheduler = _scheduler(
            _inventory(experts=4),
            loads,
            capacity_bytes=80,
            prefetch_reserve_bytes=40,
        )
        scheduler.submit_prefetch(
            ExpertKey(10, 0), deadline_ms=10, confidence=0.99, now_ms=0
        )
        scheduler.submit_prefetch(
            ExpertKey(10, 1), deadline_ms=5, confidence=0.20, now_ms=0
        )
        scheduler.submit_prefetch(
            ExpertKey(10, 2), deadline_ms=5, confidence=0.90, now_ms=0
        )

        staged = scheduler.stage_prefetch(now_ms=0, byte_budget=40)
        self.assertEqual(
            staged.staged,
            (ExpertKey(10, 2), ExpertKey(10, 1)),
        )
        self.assertEqual(staged.deferred, (ExpertKey(10, 0),))
        self.assertEqual(loads, [ExpertKey(10, 2), ExpertKey(10, 1)])
        before = scheduler.snapshot()
        self.assertEqual(before.active_keys, ())
        self.assertEqual(
            before.prefetch_keys,
            (ExpertKey(10, 1), ExpertKey(10, 2)),
        )

        promotion = scheduler.promote_prefetch()
        self.assertEqual(set(promotion.promoted), {ExpertKey(10, 1), ExpertKey(10, 2)})
        after = scheduler.snapshot()
        self.assertEqual(after.prefetch_keys, ())
        self.assertEqual(
            after.active_keys,
            (ExpertKey(10, 1), ExpertKey(10, 2)),
        )
        self.assertEqual(after.active_bytes, 40)

    def test_late_or_low_confidence_prefetch_does_not_consume_vram(self) -> None:
        loads: list[ExpertKey] = []
        scheduler = _scheduler(
            _inventory(experts=3, expert_bytes=20),
            loads,
            min_confidence=0.5,
            bandwidth_gbytes_per_second=0.000001,
        )
        scheduler.submit_prefetch(
            ExpertKey(10, 0), deadline_ms=1, confidence=0.9, now_ms=0
        )
        scheduler.submit_prefetch(
            ExpertKey(10, 1), deadline_ms=100, confidence=0.1, now_ms=0
        )
        result = scheduler.stage_prefetch(now_ms=0)

        self.assertEqual(result.expired, (ExpertKey(10, 0),))
        self.assertEqual(result.below_confidence, (ExpertKey(10, 1),))
        self.assertEqual(loads, [])
        self.assertEqual(scheduler.snapshot().prefetch_bytes, 0)

    def test_authoritative_router_ignores_false_prediction_and_falls_back_exactly(self) -> None:
        loads: list[ExpertKey] = []
        scheduler = _scheduler(_inventory(experts=4), loads)
        scheduler.submit_prefetch(
            ExpertKey(10, 1), deadline_ms=10, confidence=0.9, now_ms=0
        )
        scheduler.stage_prefetch(now_ms=0)
        self.assertEqual(loads, [ExpertKey(10, 1)])

        result = scheduler.resolve_route(
            layer=10,
            authoritative_expert_ids=(1, 2),
            predicted_expert_ids=(0, 1, 99),
        )

        self.assertTrue(result.exact)
        self.assertEqual(
            result.authoritative_experts,
            (ExpertKey(10, 1), ExpertKey(10, 2)),
        )
        self.assertEqual(result.prefetch_hits, (ExpertKey(10, 1),))
        self.assertEqual(result.ram_fallbacks, (ExpertKey(10, 2),))
        self.assertEqual(loads, [ExpertKey(10, 1), ExpertKey(10, 2)])
        self.assertNotIn(ExpertKey(10, 0), result.authoritative_experts)
        self.assertEqual(result.invalid_predictions, (99,))
        self.assertEqual(
            result.prediction_false_positives,
            (ExpertKey(10, 0),),
        )
        self.assertEqual(
            result.prediction_false_negatives,
            (ExpertKey(10, 2),),
        )
        self.assertEqual(result.fallback_bytes, 20)

    def test_oversized_expert_is_executed_from_ram_without_breaking_capacity(self) -> None:
        inventory = MacroStageExpertInventory(
            (
                ExpertRecord(ExpertKey(0, 0), 50, "large"),
                ExpertRecord(ExpertKey(0, 1), 10, "small"),
            )
        )
        loads: list[ExpertKey] = []
        scheduler = _scheduler(
            inventory,
            loads,
            capacity_bytes=40,
            prefetch_reserve_bytes=10,
        )
        result = scheduler.resolve_route(layer=0, authoritative_expert_ids=(0,))

        self.assertEqual(loads, [ExpertKey(0, 0)])
        self.assertEqual(result.ram_fallbacks, (ExpertKey(0, 0),))
        self.assertEqual(result.uncached_after_use, (ExpertKey(0, 0),))
        self.assertEqual(scheduler.snapshot().active_bytes, 0)


class RamPcieBreakEvenTests(unittest.TestCase):
    def test_pcie_bandwidth_is_explicitly_decimal_gbytes_per_second(self) -> None:
        config = PredictiveCacheConfig(
            capacity_bytes=64,
            prefetch_reserve_bytes=16,
            pcie_bandwidth_gbytes_per_second=8,
        )
        self.assertEqual(config.pcie_bytes_per_ms, 8_000_000)
        with self.assertRaisesRegex(TypeError, "pcie_bandwidth_gbps"):
            PredictiveCacheConfig(
                capacity_bytes=64,
                prefetch_reserve_bytes=16,
                pcie_bandwidth_gbps=8,  # type: ignore[call-arg]
            )

    def test_wan_macro_stage_break_even_uses_one_way_streaming_hops(self) -> None:
        estimate = estimate_ram_pcie_break_even(
            layers=10,
            experts_per_token=8,
            expert_bytes=11_800_000,
            cache_hit_rate=0.70,
            pcie_bandwidth_gbytes_per_second=8,
            rtt_ms=10,
            one_way_hops_avoided=9,
        )

        self.assertEqual(estimate.active_expert_bytes, 944_000_000)
        self.assertAlmostEqual(estimate.expected_miss_bytes, 283_200_000, delta=1)
        self.assertAlmostEqual(estimate.raw_pcie_transfer_ms, 35.4)
        self.assertAlmostEqual(estimate.network_wait_saved_ms, 45.0)
        self.assertAlmostEqual(estimate.margin_ms, 9.6)
        self.assertAlmostEqual(estimate.required_cache_hit_rate, 0.618644, places=5)
        self.assertTrue(estimate.beneficial)

        insufficient = estimate_ram_pcie_break_even(
            layers=10,
            experts_per_token=8,
            expert_bytes=11_800_000,
            cache_hit_rate=0.50,
            pcie_bandwidth_gbytes_per_second=8,
            rtt_ms=10,
            one_way_hops_avoided=9,
        )
        self.assertFalse(insufficient.beneficial)

    def test_fast_lan_requires_nearly_complete_cache_hits(self) -> None:
        estimate = estimate_ram_pcie_break_even(
            layers=10,
            experts_per_token=8,
            expert_bytes=11_800_000,
            cache_hit_rate=0.95,
            pcie_bandwidth_gbytes_per_second=8,
            rtt_ms=0.5,
            one_way_hops_avoided=9,
        )
        self.assertAlmostEqual(estimate.network_wait_saved_ms, 2.25)
        self.assertAlmostEqual(estimate.required_cache_hit_rate, 0.980932, places=5)
        self.assertFalse(estimate.beneficial)

    def test_prefetch_overlap_and_expert_parallel_round_trips_are_modeled(self) -> None:
        estimate = estimate_ram_pcie_break_even(
            layers=4,
            experts_per_token=8,
            expert_bytes=12_000_000,
            cache_hit_rate=0,
            pcie_bandwidth_gbytes_per_second=8,
            rtt_ms=10,
            round_trips_avoided=4,
            one_way_hops_avoided=0,
            prefetch_overlap_ms=20,
        )
        self.assertAlmostEqual(estimate.raw_pcie_transfer_ms, 48.0)
        self.assertAlmostEqual(estimate.hidden_by_prefetch_ms, 20.0)
        self.assertAlmostEqual(estimate.effective_pcie_stall_ms, 28.0)
        self.assertAlmostEqual(estimate.network_wait_saved_ms, 40.0)
        self.assertTrue(estimate.beneficial)


if __name__ == "__main__":
    unittest.main()
