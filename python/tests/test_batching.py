from __future__ import annotations

from dataclasses import replace
import math
import unittest

from distributed_runtime.batching import (
    BatchSelection,
    BatchWork,
    FairAdaptiveBatchQueue,
    FairBatchConfig,
    InferenceBatchKey,
)


def _config(**overrides: object) -> FairBatchConfig:
    base = FairBatchConfig(
        max_batch_size=4,
        initial_batch_size=2,
        target_batch_latency_ms=10,
        phase_weights=(("decode", 2), ("verify", 1), ("prefill", 1)),
        phase_max_wait_ms=(("decode", 0), ("verify", 0), ("prefill", 2)),
        starvation_ms=20,
        ewma_alpha=0.5,
        growth_interval=2,
        growth_headroom_ratio=0.8,
    )
    return replace(base, **overrides)


def _key(*, input_tokens: int = 1, cache_tokens: int = 4) -> InferenceBatchKey:
    return InferenceBatchKey(
        model_id="model@revision",
        stage_id="0:8",
        input_tokens=input_tokens,
        cache_tokens=cache_tokens,
        hidden_size=512,
        dtype="float16",
        codec="int8-grouped",
        backend="torch-cuda",
    )


class InferenceBatchKeyTests(unittest.TestCase):
    def test_shape_and_cache_position_are_part_of_exact_compatibility(self) -> None:
        key = _key()
        self.assertNotEqual(key, _key(input_tokens=2))
        self.assertNotEqual(key, _key(cache_tokens=5))

    def test_rejects_incomplete_or_invalid_keys(self) -> None:
        with self.assertRaises(ValueError):
            replace(_key(), model_id="")
        with self.assertRaises(ValueError):
            replace(_key(), input_tokens=0)
        with self.assertRaises(ValueError):
            replace(_key(), cache_tokens=-1)
        with self.assertRaises(ValueError):
            replace(_key(), hidden_size=True)  # type: ignore[arg-type]


class FairAdaptiveBatchQueueTests(unittest.TestCase):
    def test_groups_only_equal_keys_and_distinct_requests(self) -> None:
        queue = FairAdaptiveBatchQueue[str](_config())
        common = _key()
        queue.enqueue("first", request_id=1, phase="prefill", compatibility_key=common, now=0)
        queue.enqueue("same request next", request_id=1, phase="prefill", compatibility_key=common, now=0)
        queue.enqueue("compatible", request_id=2, phase="prefill", compatibility_key=common, now=0)
        queue.enqueue(
            "different cache",
            request_id=3,
            phase="prefill",
            compatibility_key=_key(cache_tokens=0),
            now=0,
        )

        batch = queue.pop_batch(now=0, force=True)
        assert batch is not None
        self.assertEqual(batch.payloads, ("first", "compatible"))
        self.assertEqual({item.request_id for item in batch.items}, {1, 2})
        self.assertEqual(len(queue), 2)

    def test_round_robin_prevents_one_request_from_monopolizing_a_phase(self) -> None:
        queue = FairAdaptiveBatchQueue[str](
            _config(max_batch_size=1, initial_batch_size=1)
        )
        for payload in ("a1", "a2", "a3"):
            queue.enqueue(payload, request_id=1, phase="decode", compatibility_key="x", now=0)
        queue.enqueue("b1", request_id=2, phase="decode", compatibility_key="x", now=0)

        outputs = []
        while not queue.empty:
            batch = queue.pop_batch(now=0, force=True)
            assert batch is not None
            outputs.extend(batch.payloads)
        self.assertEqual(outputs, ["a1", "b1", "a2", "a3"])

    def test_weighted_phase_cycle_is_fair(self) -> None:
        queue = FairAdaptiveBatchQueue[str](
            _config(max_batch_size=1, initial_batch_size=1)
        )
        for index in range(6):
            queue.enqueue(
                f"d{index}", request_id=100 + index, phase="decode", compatibility_key="x", now=0
            )
        for index in range(3):
            queue.enqueue(
                f"p{index}", request_id=200 + index, phase="prefill", compatibility_key="x", now=0
            )

        phases = []
        for _ in range(6):
            batch = queue.pop_batch(now=0, force=True)
            assert batch is not None
            phases.append(batch.phase)
        self.assertEqual(phases, ["decode", "decode", "prefill", "decode", "decode", "prefill"])

    def test_starvation_overrides_weighted_phase_order(self) -> None:
        queue = FairAdaptiveBatchQueue[str](
            _config(max_batch_size=1, initial_batch_size=1)
        )
        queue.enqueue("old-prefill", request_id=1, phase="prefill", compatibility_key="x", now=0)
        queue.enqueue("new-decode", request_id=2, phase="decode", compatibility_key="x", now=0.019)
        batch = queue.pop_batch(now=0.021)
        assert batch is not None
        self.assertTrue(batch.starved)
        self.assertEqual(batch.payloads, ("old-prefill",))

    def test_prefill_waits_for_window_or_full_adaptive_batch(self) -> None:
        queue = FairAdaptiveBatchQueue[str](_config())
        queue.enqueue("one", request_id=1, phase="prefill", compatibility_key="x", now=1)
        self.assertIsNone(queue.pop_batch(now=1.001))
        self.assertAlmostEqual(queue.next_ready_in_ms(now=1.001) or 0, 1.0)
        batch = queue.pop_batch(now=1.002)
        assert batch is not None
        self.assertEqual(batch.payloads, ("one",))

        queue.enqueue("two", request_id=2, phase="prefill", compatibility_key="x", now=2)
        queue.enqueue("three", request_id=3, phase="prefill", compatibility_key="x", now=2)
        immediate = queue.pop_batch(now=2)
        assert immediate is not None
        self.assertEqual(len(immediate.items), 2)

    def test_decode_is_eager_but_coalesces_already_queued_work(self) -> None:
        queue = FairAdaptiveBatchQueue[str](_config())
        queue.enqueue("a", request_id=1, phase="decode", compatibility_key="x", now=0)
        queue.enqueue("b", request_id=2, phase="decode", compatibility_key="x", now=0)
        batch = queue.pop_batch(now=0)
        assert batch is not None
        self.assertEqual(batch.payloads, ("a", "b"))

    def test_adaptive_limit_grows_with_headroom_and_halves_on_violation(self) -> None:
        queue = FairAdaptiveBatchQueue[str](_config())

        def run(size: int, latency_ms: float, start: int) -> BatchSelection[str]:
            for index in range(size):
                queue.enqueue(
                    f"{start + index}",
                    request_id=start + index,
                    phase="decode",
                    compatibility_key="x",
                    now=0,
                )
            batch = queue.pop_batch(now=0)
            assert batch is not None
            queue.record_batch(batch, latency_ms=latency_ms, transferred_bytes=10)
            return batch

        self.assertEqual(run(2, 5, 10).adaptive_limit, 2)
        self.assertEqual(run(2, 5, 20).adaptive_limit, 2)
        profile = queue.stats().profiles[0]
        self.assertEqual(profile.adaptive_limit, 3)

        grown = run(3, 15, 30)
        self.assertEqual(grown.adaptive_limit, 3)
        profile = queue.stats().profiles[0]
        self.assertEqual(profile.adaptive_limit, 1)
        self.assertEqual(profile.target_violations, 1)
        self.assertAlmostEqual(profile.ewma_latency_ms or 0, 10.0)
        self.assertAlmostEqual(profile.ewma_bytes or 0, 10.0)

    def test_adaptation_is_isolated_by_phase_and_compatibility_key(self) -> None:
        queue = FairAdaptiveBatchQueue[str](_config())
        for request_id in (1, 2):
            queue.enqueue("a", request_id=request_id, phase="decode", compatibility_key="A", now=0)
        batch = queue.pop_batch(now=0)
        assert batch is not None
        queue.record_batch(batch, latency_ms=20)

        for request_id in (3, 4):
            queue.enqueue("b", request_id=request_id, phase="decode", compatibility_key="B", now=0)
        other = queue.pop_batch(now=0)
        assert other is not None
        self.assertEqual(other.adaptive_limit, 2)
        limits = {
            (profile.phase, profile.compatibility_key): profile.adaptive_limit
            for profile in queue.stats().profiles
        }
        self.assertEqual(limits[("decode", "A")], 1)
        self.assertEqual(limits[("decode", "B")], 2)

    def test_cancellation_removes_all_phases_for_one_request(self) -> None:
        queue = FairAdaptiveBatchQueue[str](_config())
        queue.enqueue("prefill", request_id=7, phase="prefill", compatibility_key="x", now=2)
        queue.enqueue("decode", request_id=7, phase="decode", compatibility_key="x", now=1)
        queue.enqueue("other", request_id=8, phase="decode", compatibility_key="x", now=1)
        removed = queue.cancel_request(7)
        self.assertEqual([item.payload for item in removed], ["decode", "prefill"])
        self.assertEqual(len(queue), 1)
        self.assertEqual(queue.stats().cancelled_items, 2)

    def test_metrics_cover_wait_fill_phases_bytes_and_depth(self) -> None:
        queue = FairAdaptiveBatchQueue[str](_config())
        queue.enqueue("a", request_id=1, phase="decode", compatibility_key="x", now=1)
        queue.enqueue("b", request_id=2, phase="decode", compatibility_key="x", now=1.001)
        batch = queue.pop_batch(now=1.003)
        assert batch is not None
        queue.record_batch(batch, latency_ms=4, transferred_bytes=123)
        stats = queue.stats()
        self.assertEqual(stats.queued_items, 0)
        self.assertEqual(stats.enqueued_items, 2)
        self.assertEqual(stats.dequeued_items, 2)
        self.assertEqual(stats.batches, 1)
        self.assertEqual(stats.max_queue_depth, 2)
        self.assertAlmostEqual(stats.average_wait_ms, 2.5)
        self.assertAlmostEqual(stats.maximum_wait_ms, 3.0)
        self.assertEqual(stats.average_batch_size, 2.0)
        self.assertEqual(stats.fill_ratio, 0.5)
        self.assertEqual(stats.transferred_bytes, 123)
        self.assertEqual(dict(stats.phase_batches)["decode"], 1)

    def test_empty_queue_and_force_semantics(self) -> None:
        queue = FairAdaptiveBatchQueue[str](_config())
        self.assertTrue(queue.empty)
        self.assertIsNone(queue.next_ready_in_ms(now=0))
        self.assertIsNone(queue.pop_batch(now=0))
        queue.enqueue("one", request_id=1, phase="prefill", compatibility_key="x", now=0)
        self.assertIsNone(queue.pop_batch(now=0))
        self.assertIsNotNone(queue.pop_batch(now=0, force=True))

    def test_rejects_invalid_configuration_and_operations(self) -> None:
        invalid_configs = (
            {"max_batch_size": 0},
            {"initial_batch_size": 0},
            {"initial_batch_size": 9},
            {"target_batch_latency_ms": 0},
            {"target_batch_latency_ms": math.inf},
            {"phase_weights": ()},
            {"phase_weights": (("decode", 1), ("decode", 2))},
            {"phase_weights": (("decode", 0),)},
            {"phase_max_wait_ms": (("decode", 0),)},
            {"starvation_ms": 0},
            {"ewma_alpha": 0},
            {"ewma_alpha": 2},
            {"growth_interval": 0},
            {"growth_headroom_ratio": 1},
        )
        for kwargs in invalid_configs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    _config(**kwargs)

        queue = FairAdaptiveBatchQueue[str](_config())
        invalid_enqueues = (
            {"request_id": -1, "phase": "decode", "compatibility_key": "x"},
            {"request_id": True, "phase": "decode", "compatibility_key": "x"},
            {"request_id": 1, "phase": "unknown", "compatibility_key": "x"},
            {"request_id": 1, "phase": "decode", "compatibility_key": []},
            {"request_id": 1, "phase": "decode", "compatibility_key": "x", "token_count": 0},
            {"request_id": 1, "phase": "decode", "compatibility_key": "x", "now": math.nan},
        )
        for kwargs in invalid_enqueues:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    queue.enqueue("payload", **kwargs)  # type: ignore[arg-type]

        work = BatchWork(1, "decode", "x", 1, "payload", 0)
        empty = BatchSelection((work,), 0, 1, False)
        for kwargs in (
            {"latency_ms": 0},
            {"latency_ms": math.inf},
            {"latency_ms": 1, "transferred_bytes": -1},
            {"latency_ms": 1, "transferred_bytes": True},
        ):
            with self.subTest(record=kwargs):
                with self.assertRaises(ValueError):
                    queue.record_batch(empty, **kwargs)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
