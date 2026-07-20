from __future__ import annotations

from contextlib import nullcontext
import unittest
from unittest.mock import patch

import torch

from distributed_runtime.ram_expert_cache import (
    ExpertKey,
    ExpertRecord,
    MacroStageExpertInventory,
    PredictiveCacheConfig,
    RamBackedExpertScheduler,
)
from distributed_runtime.torch_ram_expert_store import (
    BoundedPinnedExpertStaging,
    TorchExpertBundle,
    TorchRamExpertStore,
)


class _FakeCompletion:
    def __init__(self, *, complete: bool = False) -> None:
        self.complete = complete
        self.query_calls = 0
        self.synchronize_calls = 0
        self.recorded_stream: object | None = None

    def query(self) -> bool:
        self.query_calls += 1
        return self.complete

    def synchronize(self) -> None:
        self.synchronize_calls += 1
        self.complete = True

    def record(self, stream: object) -> None:
        self.recorded_stream = stream


def _fixture(
    values: tuple[float, ...],
    *,
    capacity_bytes: int = 64,
    reserve_bytes: int = 16,
    device: str = "cpu",
    pin_memory: bool = False,
) -> tuple[
    MacroStageExpertInventory,
    PredictiveCacheConfig,
    TorchRamExpertStore,
    RamBackedExpertScheduler,
]:
    tensors = {
        ExpertKey(4, expert): torch.full((4,), value, dtype=torch.float32)
        for expert, value in enumerate(values)
    }
    inventory = MacroStageExpertInventory(
        tuple(
            ExpertRecord(
                key=key,
                byte_size=tensor.numel() * tensor.element_size(),
                content_id=f"torch-test:{key.layer}:{key.expert}",
            )
            for key, tensor in tensors.items()
        )
    )
    config = PredictiveCacheConfig(
        capacity_bytes=capacity_bytes,
        prefetch_reserve_bytes=reserve_bytes,
        pcie_bandwidth_gbytes_per_second=1,
        hotness_decay=1,
    )
    store = TorchRamExpertStore(
        inventory,
        tensors,
        config,
        device=device,
        pin_memory=pin_memory,
        allow_cpu_fallback=True,
    )
    scheduler = RamBackedExpertScheduler(
        inventory,
        config,
        weight_backend=store,
    )
    return inventory, config, store, scheduler


class TorchRamExpertStoreTests(unittest.TestCase):
    def test_bounded_pinned_staging_has_exactly_two_largest_expert_slots(self) -> None:
        allocations: list[int] = []

        def allocate(byte_count: int) -> torch.Tensor:
            allocations.append(byte_count)
            return torch.empty(byte_count, dtype=torch.uint8)

        staging = BoundedPinnedExpertStaging(96, allocator=allocate)
        snapshot = staging.snapshot()

        self.assertEqual(allocations, [96, 96])
        self.assertEqual(snapshot.slot_bytes, 96)
        self.assertEqual(snapshot.capacity_bytes, 192)
        self.assertEqual(snapshot.occupied_bytes, 0)
        self.assertEqual(snapshot.pageable_to_pinned_bytes, 0)

    def test_bounded_staging_waits_before_overwriting_and_reuses_same_expert(self) -> None:
        ticks = iter((1_000_000, 4_500_000))
        staging = BoundedPinnedExpertStaging(
            32,
            allocator=lambda size: torch.empty(size, dtype=torch.uint8),
            clock_ns=lambda: next(ticks),
        )
        first_key = ExpertKey(1, 0)
        second_key = ExpertKey(1, 1)
        first = TorchExpertBundle(
            key=first_key,
            content_id="first",
            tensors=(("weight", torch.arange(8, dtype=torch.float32)),),
        )
        second = TorchExpertBundle(
            key=second_key,
            content_id="second",
            tensors=(("weight", torch.arange(8, dtype=torch.float32) + 20),),
        )

        staged_first = staging.stage("execution", first)
        pending = _FakeCompletion(complete=False)
        staging.mark_in_flight(
            "execution",
            key=first_key,
            completion=pending,
        )
        staged_hit = staging.stage("execution", first)
        self.assertEqual(pending.synchronize_calls, 0)
        torch.testing.assert_close(staged_hit.tensor(), first.tensor())

        newer_pending = _FakeCompletion(complete=False)
        staging.mark_in_flight(
            "execution",
            key=first_key,
            completion=newer_pending,
        )
        staged_second = staging.stage("execution", second)

        self.assertEqual(newer_pending.query_calls, 1)
        self.assertEqual(newer_pending.synchronize_calls, 1)
        torch.testing.assert_close(staged_second.tensor(), second.tensor())
        # Both bundles are views of the same reusable slot; overwrite is
        # observable and proves no third hidden allocation retained ``first``.
        self.assertEqual(
            staged_first.tensor().untyped_storage().data_ptr(),
            staged_second.tensor().untyped_storage().data_ptr(),
        )
        snapshot = staging.snapshot()
        self.assertEqual(snapshot.staging_hits, 1)
        self.assertEqual(snapshot.staging_misses, 2)
        self.assertEqual(snapshot.reuse_stalls, 1)
        self.assertEqual(snapshot.reuse_stall_ms, 3.5)
        self.assertEqual(snapshot.pageable_to_pinned_bytes, 64)
        self.assertEqual(snapshot.execution_key, second_key)

    def test_bounded_staging_keeps_execution_and_prefetch_lifecycles_independent(self) -> None:
        staging = BoundedPinnedExpertStaging(
            16,
            allocator=lambda size: torch.empty(size, dtype=torch.uint8),
        )
        execution = TorchExpertBundle(
            key=ExpertKey(2, 0),
            content_id="execution",
            tensors=(("weight", torch.full((4,), 3.0)),),
        )
        prefetch = TorchExpertBundle(
            key=ExpertKey(3, 1),
            content_id="prefetch",
            tensors=(("weight", torch.full((4,), 7.0)),),
        )
        execution_view = staging.stage("execution", execution)
        prefetch_view = staging.stage("prefetch", prefetch)

        self.assertNotEqual(
            execution_view.tensor().untyped_storage().data_ptr(),
            prefetch_view.tensor().untyped_storage().data_ptr(),
        )
        snapshot = staging.snapshot()
        self.assertEqual(snapshot.capacity_bytes, 32)
        self.assertEqual(snapshot.occupied_bytes, 32)
        self.assertEqual(snapshot.execution_key, execution.key)
        self.assertEqual(snapshot.prefetch_key, prefetch.key)

    def test_bounded_staging_packs_mixed_dtypes_without_alignment_overhead(self) -> None:
        key = ExpertKey(6, 0)
        source = TorchExpertBundle(
            key=key,
            content_id="mixed-dtype",
            tensors=(
                ("half", torch.arange(3, dtype=torch.float16)),
                ("double", torch.arange(2, dtype=torch.float64)),
                ("float", torch.arange(5, dtype=torch.float32)),
            ),
        )
        self.assertEqual(source.byte_size, 42)
        staging = BoundedPinnedExpertStaging(
            source.byte_size,
            allocator=lambda size: torch.empty(size, dtype=torch.uint8),
        )

        staged = staging.stage("prefetch", source)

        self.assertEqual(staged.byte_size, source.byte_size)
        self.assertEqual(staging.snapshot().capacity_bytes, 84)
        for name, expected in source.tensors:
            torch.testing.assert_close(staged.tensor(name), expected)

    def test_bounded_staging_preserves_canonical_gate_up_packing_contract(self) -> None:
        key = ExpertKey(6, 1)
        gate = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        up = gate + 20
        down = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        source = TorchExpertBundle(
            key=key,
            content_id="canonical-through-staging",
            tensors=(
                ("gate_proj.weight", gate),
                ("up_proj.weight", up),
                ("down_proj.weight", down),
            ),
        )
        staging = BoundedPinnedExpertStaging(
            source.byte_size,
            allocator=lambda size: torch.empty(size, dtype=torch.uint8),
        )
        staged = staging.stage("execution", source)

        copied = TorchRamExpertStore._copy_execution_bundle(
            staged,
            device=torch.device("cpu"),
            non_blocking=False,
        )

        torch.testing.assert_close(copied.packed_gate_up(), torch.cat((gate, up)))
        self.assertEqual(copied.byte_size, source.byte_size)
        self.assertEqual(
            copied.tensor("gate_proj.weight").untyped_storage().data_ptr(),
            copied.tensor("up_proj.weight").untyped_storage().data_ptr(),
        )

    def test_cpu_fallback_does_not_attempt_bounded_pinned_allocation(self) -> None:
        key = ExpertKey(0, 0)
        tensor = torch.ones(4, dtype=torch.float32)
        inventory = MacroStageExpertInventory(
            (ExpertRecord(key, 16, "cpu-no-pin"),)
        )
        config = PredictiveCacheConfig(
            capacity_bytes=32,
            prefetch_reserve_bytes=16,
            pcie_bandwidth_gbytes_per_second=1,
        )

        with patch(
            "distributed_runtime.torch_ram_expert_store.BoundedPinnedExpertStaging",
            side_effect=AssertionError("CPU certification attempted to pin"),
        ):
            store = TorchRamExpertStore(
                inventory,
                {key: tensor},
                config,
                device="cpu",
                pin_memory=False,
                bounded_pinned_staging=True,
                allow_cpu_fallback=False,
            )
        scheduler = RamBackedExpertScheduler(
            inventory,
            config,
            weight_backend=store,
        )
        scheduler.resolve_route(layer=0, authoritative_expert_ids=(0,))

        snapshot = store.snapshot()
        self.assertTrue(snapshot.bounded_pinned_staging_requested)
        self.assertFalse(snapshot.bounded_pinned_staging_enabled)
        self.assertEqual(snapshot.pinned_capacity_bytes, 0)
        self.assertEqual(snapshot.pageable_to_pinned_bytes, 0)
        self.assertEqual(snapshot.pinned_to_device_bytes, 0)
        self.assertFalse(store.ram_bundle(key).tensor().is_pinned())

    def test_bounded_staging_does_not_pin_or_copy_the_full_inventory(self) -> None:
        keys = tuple(ExpertKey(5, expert) for expert in range(12))
        bundles = {
            key: TorchExpertBundle(
                key=key,
                content_id=f"owned:{key.expert}",
                tensors=(("weight", torch.full((8,), float(key.expert))),),
            )
            for key in keys
        }
        inventory = MacroStageExpertInventory(
            tuple(
                ExpertRecord(key, bundle.byte_size, bundle.content_id)
                for key, bundle in bundles.items()
            )
        )
        config = PredictiveCacheConfig(
            capacity_bytes=64,
            prefetch_reserve_bytes=32,
            pcie_bandwidth_gbytes_per_second=1,
        )
        # CPU mode deliberately cannot instantiate the CUDA staging pool. Its
        # zero-copy ownership path still proves that opting in does not mutate
        # or globally duplicate the authoritative inventory before CUDA exists.
        store = TorchRamExpertStore.adopt_owned_cpu_bundles(
            inventory,
            bundles,
            config,
            device="cpu",
            bounded_pinned_staging=True,
            allow_cpu_fallback=False,
        )

        snapshot = store.snapshot()
        self.assertTrue(snapshot.ram_adopted_without_clone)
        self.assertEqual(snapshot.ram_bytes, 12 * 32)
        self.assertEqual(snapshot.copied_bytes, 0)
        self.assertEqual(snapshot.pinned_capacity_bytes, 0)
        self.assertTrue(
            all(
                not tensor.is_pinned()
                for bundle in bundles.values()
                for _, tensor in bundle.tensors
            )
        )
        allocations: list[int] = []

        def allocate(byte_count: int) -> torch.Tensor:
            allocations.append(byte_count)
            return torch.empty(byte_count, dtype=torch.uint8)

        staging = BoundedPinnedExpertStaging(
            32,
            allocator=allocate,
        )
        staging.stage("prefetch", bundles[keys[0]])
        staging_snapshot = staging.snapshot()
        self.assertEqual(allocations, [32, 32])
        self.assertEqual(staging_snapshot.capacity_bytes, 64)
        self.assertLess(staging_snapshot.capacity_bytes, snapshot.ram_bytes)
        self.assertEqual(staging_snapshot.pageable_to_pinned_bytes, 32)

    def test_mock_cuda_store_uses_two_slots_stream_events_and_byte_telemetry(self) -> None:
        keys = tuple(ExpertKey(10, expert) for expert in range(4))
        tensors = {
            key: torch.full((4,), float(key.expert), dtype=torch.float32)
            for key in keys
        }
        inventory = MacroStageExpertInventory(
            tuple(
                ExpertRecord(key, 16, f"mock-cuda:{key.expert}")
                for key in keys
            )
        )
        config = PredictiveCacheConfig(
            capacity_bytes=64,
            prefetch_reserve_bytes=32,
            pcie_bandwidth_gbytes_per_second=1,
        )
        allocations: list[int] = []
        events: list[_FakeCompletion] = []
        non_blocking_copies: list[bool] = []
        transfer_stream = object()

        def staging_factory(byte_count: int) -> BoundedPinnedExpertStaging:
            def allocate(size: int) -> torch.Tensor:
                allocations.append(size)
                return torch.empty(size, dtype=torch.uint8)

            return BoundedPinnedExpertStaging(byte_count, allocator=allocate)

        def event_factory(**_: object) -> _FakeCompletion:
            event = _FakeCompletion(complete=False)
            events.append(event)
            return event

        def copy_bundle(
            source: TorchExpertBundle,
            *,
            device: torch.device,
            non_blocking: bool,
        ) -> TorchExpertBundle:
            self.assertEqual(device.type, "cuda")
            non_blocking_copies.append(non_blocking)
            return TorchExpertBundle(
                key=source.key,
                content_id=source.content_id,
                tensors=tuple(
                    (name, tensor.clone()) for name, tensor in source.tensors
                ),
            )

        with (
            patch.object(
                TorchRamExpertStore,
                "_select_device",
                return_value=(torch.device("cuda"), False),
            ),
            patch(
                "distributed_runtime.torch_ram_expert_store.BoundedPinnedExpertStaging",
                side_effect=staging_factory,
            ),
            patch(
                "distributed_runtime.torch_ram_expert_store.torch.cuda.Stream",
                return_value=transfer_stream,
            ),
            patch(
                "distributed_runtime.torch_ram_expert_store.torch.cuda.stream",
                return_value=nullcontext(),
            ),
            patch(
                "distributed_runtime.torch_ram_expert_store.torch.cuda.Event",
                side_effect=event_factory,
            ),
            patch.object(
                TorchRamExpertStore,
                "_copy_execution_bundle",
                side_effect=copy_bundle,
            ),
        ):
            store = TorchRamExpertStore(
                inventory,
                tensors,
                config,
                device="cuda",
                pin_memory=False,
                bounded_pinned_staging=True,
                allow_cpu_fallback=False,
            )
            store.prefetch(inventory.record(keys[0]))
            store.prefetch(inventory.record(keys[1]))
            store.discard_prefetch((keys[0], keys[1]))
            # Same immutable prefetch expert remains in the slot: this skips
            # the pageable-to-pinned copy but still performs a fresh H2D copy.
            store.prefetch(inventory.record(keys[1]))
            store.load_for_execution(inventory.record(keys[2]))

        snapshot = store.snapshot()
        self.assertEqual(allocations, [16, 16])
        self.assertTrue(snapshot.bounded_pinned_staging_enabled)
        self.assertEqual(snapshot.pinned_capacity_bytes, 32)
        self.assertEqual(snapshot.ram_bytes, 64)
        self.assertEqual(snapshot.pageable_to_pinned_bytes, 48)
        self.assertEqual(snapshot.pinned_to_device_bytes, 64)
        self.assertEqual(snapshot.pinned_staging_hits, 1)
        self.assertEqual(snapshot.pinned_staging_misses, 3)
        self.assertEqual(snapshot.pinned_reuse_stalls, 1)
        self.assertEqual(snapshot.pinned_execution_key, keys[2])
        self.assertEqual(snapshot.pinned_prefetch_key, keys[1])
        self.assertEqual(non_blocking_copies, [True, True, True, True])
        self.assertEqual(len(events), 4)
        self.assertTrue(
            all(event.recorded_stream is transfer_stream for event in events)
        )

    def test_cuda_eviction_waits_for_recorded_compute_before_releasing_bundle(self) -> None:
        keys = (ExpertKey(11, 0), ExpertKey(11, 1))
        tensors = {
            key: torch.full((4,), float(key.expert), dtype=torch.float32)
            for key in keys
        }
        inventory = MacroStageExpertInventory(
            tuple(
                ExpertRecord(key, 16, f"mock-cuda-lifetime:{key.expert}")
                for key in keys
            )
        )
        config = PredictiveCacheConfig(
            capacity_bytes=32,
            prefetch_reserve_bytes=16,
            pcie_bandwidth_gbytes_per_second=1,
        )
        transfer_stream = object()
        timeline: list[str] = []
        active_when_synchronized: list[tuple[ExpertKey, ...]] = []
        events: list[_FakeCompletion] = []

        class FakeComputeStream:
            def __init__(self) -> None:
                self.waited_events: list[_FakeCompletion] = []
                self.synchronize_calls = 0

            def wait_event(self, event: _FakeCompletion) -> None:
                self.waited_events.append(event)

            def synchronize(self) -> None:
                self.synchronize_calls += 1
                timeline.append("compute-synchronize")
                active_when_synchronized.append(tuple(sorted(store._active)))

        compute_stream = FakeComputeStream()

        def staging_factory(byte_count: int) -> BoundedPinnedExpertStaging:
            return BoundedPinnedExpertStaging(
                byte_count,
                allocator=lambda size: torch.empty(size, dtype=torch.uint8),
            )

        def event_factory(**_: object) -> _FakeCompletion:
            event = _FakeCompletion(complete=False)
            events.append(event)
            return event

        def copy_bundle(
            source: TorchExpertBundle,
            *,
            device: torch.device,
            non_blocking: bool,
        ) -> TorchExpertBundle:
            self.assertEqual(device.type, "cuda")
            self.assertTrue(non_blocking)
            timeline.append(f"copy-{source.key.expert}")
            return TorchExpertBundle(
                key=source.key,
                content_id=source.content_id,
                tensors=tuple(
                    (name, tensor.clone()) for name, tensor in source.tensors
                ),
            )

        with (
            patch.object(
                TorchRamExpertStore,
                "_select_device",
                return_value=(torch.device("cuda"), False),
            ),
            patch(
                "distributed_runtime.torch_ram_expert_store.BoundedPinnedExpertStaging",
                side_effect=staging_factory,
            ),
            patch(
                "distributed_runtime.torch_ram_expert_store.torch.cuda.Stream",
                return_value=transfer_stream,
            ),
            patch(
                "distributed_runtime.torch_ram_expert_store.torch.cuda.stream",
                return_value=nullcontext(),
            ),
            patch(
                "distributed_runtime.torch_ram_expert_store.torch.cuda.Event",
                side_effect=event_factory,
            ),
            patch(
                "distributed_runtime.torch_ram_expert_store.torch.cuda.current_stream",
                return_value=compute_stream,
            ),
            patch.object(torch.Tensor, "record_stream", autospec=True) as record_stream,
            patch.object(
                TorchRamExpertStore,
                "_copy_execution_bundle",
                side_effect=copy_bundle,
            ),
        ):
            store = TorchRamExpertStore(
                inventory,
                tensors,
                config,
                device="cuda",
                pin_memory=False,
                bounded_pinned_staging=True,
                allow_cpu_fallback=False,
            )
            scheduler = RamBackedExpertScheduler(
                inventory,
                config,
                weight_backend=store,
            )

            scheduler.resolve_route(
                layer=keys[0].layer,
                authoritative_expert_ids=(keys[0].expert,),
            )
            self.assertEqual(compute_stream.synchronize_calls, 0)
            resolution = scheduler.resolve_route(
                layer=keys[1].layer,
                authoritative_expert_ids=(keys[1].expert,),
            )

        self.assertEqual(resolution.evicted, (keys[0],))
        self.assertEqual(compute_stream.synchronize_calls, 1)
        self.assertEqual(active_when_synchronized, [(keys[0],)])
        self.assertEqual(store.snapshot().active_keys, (keys[1],))
        self.assertEqual(store.snapshot().pinned_capacity_bytes, 32)
        self.assertGreaterEqual(record_stream.call_count, 2)
        self.assertTrue(
            all(call.args[0] is compute_stream for call in record_stream.call_args_list)
        )
        self.assertLess(
            timeline.index("copy-1"),
            timeline.index("compute-synchronize"),
        )

    def test_canonical_device_bundle_packs_gate_up_without_extra_bytes(self) -> None:
        key = ExpertKey(0, 0)
        gate = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        up = gate + 100
        down = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        bundle = TorchExpertBundle(
            key=key,
            content_id="packed-canonical",
            tensors=(
                ("gate_proj.weight", gate),
                ("up_proj.weight", up),
                ("down_proj.weight", down),
            ),
        )
        inventory = MacroStageExpertInventory(
            (ExpertRecord(key, bundle.byte_size, bundle.content_id),)
        )
        config = PredictiveCacheConfig(
            capacity_bytes=bundle.byte_size * 2,
            prefetch_reserve_bytes=bundle.byte_size,
            pcie_bandwidth_gbytes_per_second=1,
        )
        store = TorchRamExpertStore(
            inventory,
            {key: bundle},
            config,
            device="cpu",
            pin_memory=False,
        )
        scheduler = RamBackedExpertScheduler(
            inventory,
            config,
            weight_backend=store,
        )

        scheduler.resolve_route(layer=0, authoritative_expert_ids=(0,))
        copied = store.bundle(key)
        packed = copied.packed_gate_up()

        torch.testing.assert_close(packed, torch.cat((gate, up), dim=0))
        self.assertEqual(
            copied.tensor("gate_proj.weight").untyped_storage().data_ptr(),
            copied.tensor("up_proj.weight").untyped_storage().data_ptr(),
        )
        self.assertEqual(copied.byte_size, bundle.byte_size)
        unique_storages = {
            tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes()
            for _, tensor in copied.tensors
        }
        self.assertEqual(sum(unique_storages.values()), bundle.byte_size)
        self.assertEqual(store.snapshot().active_bytes, bundle.byte_size)

    def test_public_constructor_accepts_bundle_and_clones_it_defensively(self) -> None:
        key = ExpertKey(0, 0)
        source = torch.arange(4, dtype=torch.float32)
        record = ExpertRecord(key, source.numel() * source.element_size(), "bundle")
        inventory = MacroStageExpertInventory((record,))
        config = PredictiveCacheConfig(
            capacity_bytes=64,
            prefetch_reserve_bytes=16,
            pcie_bandwidth_gbytes_per_second=1,
        )
        bundle = TorchExpertBundle(
            key=key,
            content_id=record.content_id,
            tensors=(("weight", source),),
        )

        store = TorchRamExpertStore(
            inventory,
            {key: bundle},
            config,
            device="cpu",
            pin_memory=False,
        )
        source.fill_(-1)

        self.assertTrue(
            torch.equal(
                store.ram_bundle(key).tensor(),
                torch.arange(4, dtype=torch.float32),
            )
        )
        self.assertFalse(store.snapshot().ram_adopted_without_clone)

    def test_authoritative_ram_owns_contiguous_cpu_tensor_and_cuda_degrades(self) -> None:
        source = torch.arange(6, dtype=torch.float32).reshape(2, 3).t()
        self.assertFalse(source.is_contiguous())
        key = ExpertKey(0, 0)
        inventory = MacroStageExpertInventory(
            (
                ExpertRecord(
                    key,
                    source.numel() * source.element_size(),
                    "non-contiguous-source",
                ),
            )
        )
        config = PredictiveCacheConfig(
            capacity_bytes=64,
            prefetch_reserve_bytes=32,
            pcie_bandwidth_gbytes_per_second=1,
        )
        expected = source.contiguous().clone()
        store = TorchRamExpertStore(
            inventory,
            {key: source},
            config,
            device="cuda",
            pin_memory=True,
            allow_cpu_fallback=True,
        )
        source.fill_(-1)

        ram_tensor = store.ram_bundle(key).tensor()
        self.assertEqual(ram_tensor.device.type, "cpu")
        self.assertTrue(ram_tensor.is_contiguous())
        self.assertTrue(torch.equal(ram_tensor, expected))
        snapshot = store.snapshot()
        self.assertEqual(snapshot.cpu_fallback, not torch.cuda.is_available())
        if not torch.cuda.is_available():
            self.assertEqual(snapshot.effective_device, "cpu")
            self.assertFalse(snapshot.pin_memory_enabled)
            self.assertFalse(snapshot.separate_prefetch_stream)

    def test_prefetch_is_physically_staged_then_promoted_before_route(self) -> None:
        _, _, store, scheduler = _fixture(
            (2, 3, 5),
            capacity_bytes=64,
            reserve_bytes=32,
        )
        key = ExpertKey(4, 1)
        scheduler.submit_prefetch(key, deadline_ms=10, confidence=0.9, now_ms=0)
        staged = scheduler.stage_prefetch(now_ms=0)

        self.assertEqual(staged.staged, (key,))
        self.assertEqual(scheduler.snapshot().active_keys, ())
        self.assertEqual(store.snapshot().prefetch_keys, (key,))
        resolution = scheduler.resolve_route(
            layer=4,
            authoritative_expert_ids=(1,),
            predicted_expert_ids=(1,),
        )

        self.assertEqual(resolution.prefetch_hits, (key,))
        self.assertEqual(resolution.ram_fallbacks, ())
        self.assertEqual(store.snapshot().active_keys, (key,))
        self.assertEqual(store.snapshot().prefetch_keys, ())
        output = store.bundle(key).tensor() * torch.tensor(4.0)
        self.assertTrue(torch.equal(output, torch.full((4,), 12.0)))

    def test_prefetched_authoritative_overflow_is_ephemeral_without_stale_state(self) -> None:
        _, _, store, scheduler = _fixture(
            (2, 3, 5),
            capacity_bytes=64,
            reserve_bytes=32,
        )
        active = ExpertKey(4, 0)
        first_prefetch = ExpertKey(4, 1)
        overflow_prefetch = ExpertKey(4, 2)
        scheduler.resolve_route(layer=4, authoritative_expert_ids=(0,))
        for key in (first_prefetch, overflow_prefetch):
            scheduler.submit_prefetch(
                key,
                deadline_ms=10,
                confidence=0.9,
                now_ms=0,
            )
        scheduler.stage_prefetch(now_ms=0)

        resolution = scheduler.resolve_route(
            layer=4,
            authoritative_expert_ids=(0, 1, 2),
            predicted_expert_ids=(1, 2),
        )

        self.assertEqual(resolution.cache_hits, (active,))
        self.assertEqual(
            resolution.prefetch_hits,
            (first_prefetch, overflow_prefetch),
        )
        self.assertEqual(resolution.ram_fallbacks, ())
        self.assertEqual(resolution.uncached_after_use, (overflow_prefetch,))
        self.assertEqual(scheduler.snapshot().prefetch_keys, ())
        self.assertEqual(store.snapshot().prefetch_keys, ())
        self.assertEqual(store.snapshot().ephemeral_keys, (overflow_prefetch,))

        scheduler.release_route(resolution)
        second = scheduler.resolve_route(
            layer=4,
            authoritative_expert_ids=(2,),
        )
        self.assertEqual(second.ram_fallbacks, (overflow_prefetch,))
        self.assertEqual(store.snapshot().prefetch_keys, ())

    def test_wrong_prediction_cannot_replace_authoritative_expert(self) -> None:
        _, _, store, scheduler = _fixture(
            (2, 3, 5),
            capacity_bytes=80,
            reserve_bytes=32,
        )
        wrong = ExpertKey(4, 0)
        actual = ExpertKey(4, 2)
        scheduler.submit_prefetch(
            wrong,
            deadline_ms=10,
            confidence=0.99,
            now_ms=0,
        )
        scheduler.stage_prefetch(now_ms=0)

        resolution = scheduler.resolve_route(
            layer=4,
            authoritative_expert_ids=(2,),
            predicted_expert_ids=(0,),
        )
        bundles = store.bundles_for_route(resolution.authoritative_experts)
        output = bundles[0].tensor() * torch.tensor(4.0)

        self.assertTrue(resolution.exact)
        self.assertEqual(resolution.authoritative_experts, (actual,))
        self.assertEqual(resolution.ram_fallbacks, (actual,))
        self.assertEqual(resolution.prediction_false_positives, (wrong,))
        self.assertEqual(resolution.prediction_false_negatives, (actual,))
        self.assertTrue(torch.equal(output, torch.full((4,), 20.0)))
        snapshot = store.snapshot()
        self.assertEqual(snapshot.active_keys, (actual,))
        self.assertEqual(snapshot.prefetch_keys, (wrong,))
        scheduler.discard_prefetch()
        self.assertEqual(store.snapshot().prefetch_keys, ())

    def test_scheduler_eviction_is_mirrored_by_physical_byte_bounded_store(self) -> None:
        _, config, store, scheduler = _fixture(
            (2, 3, 5),
            capacity_bytes=48,
            reserve_bytes=16,
        )
        scheduler.resolve_route(layer=4, authoritative_expert_ids=(0,))
        scheduler.resolve_route(layer=4, authoritative_expert_ids=(1,))
        scheduler.resolve_route(layer=4, authoritative_expert_ids=(0,))
        resolution = scheduler.resolve_route(layer=4, authoritative_expert_ids=(2,))

        expected = (ExpertKey(4, 0), ExpertKey(4, 2))
        self.assertEqual(resolution.evicted, (ExpertKey(4, 1),))
        self.assertEqual(store.snapshot().active_keys, expected)
        self.assertEqual(scheduler.snapshot().active_keys, expected)
        self.assertEqual(store.snapshot().active_bytes, 32)
        self.assertLessEqual(
            store.snapshot().active_bytes,
            config.active_capacity_bytes,
        )

    def test_uncached_exact_fallback_has_explicit_release_lifecycle(self) -> None:
        key = ExpertKey(2, 0)
        tensor = torch.arange(8, dtype=torch.float32)
        inventory = MacroStageExpertInventory(
            (ExpertRecord(key, 32, "oversized-for-active-cache"),)
        )
        config = PredictiveCacheConfig(
            capacity_bytes=40,
            prefetch_reserve_bytes=10,
            pcie_bandwidth_gbytes_per_second=1,
        )
        store = TorchRamExpertStore(
            inventory,
            {key: tensor},
            config,
            device="cpu",
            pin_memory=False,
        )
        scheduler = RamBackedExpertScheduler(
            inventory,
            config,
            weight_backend=store,
        )

        resolution = scheduler.resolve_route(
            layer=2,
            authoritative_expert_ids=(0,),
            predicted_expert_ids=(99,),
        )
        self.assertEqual(resolution.uncached_after_use, (key,))
        self.assertEqual(store.snapshot().active_bytes, 0)
        self.assertEqual(store.snapshot().ephemeral_bytes, 32)
        self.assertTrue(torch.equal(store.bundle(key).tensor(), tensor))

        scheduler.release_route(resolution)
        self.assertEqual(store.snapshot().ephemeral_bytes, 0)
        with self.assertRaisesRegex(KeyError, "not loaded"):
            store.bundle(key)

    def test_route_reservation_prevents_active_prefetch_ephemeral_overflow(self) -> None:
        active = ExpertKey(7, 0)
        speculative = ExpertKey(7, 1)
        authoritative = ExpertKey(7, 2)
        tensors = {
            active: torch.ones(4, dtype=torch.float32),
            speculative: torch.ones(2, dtype=torch.float32),
            authoritative: torch.ones(8, dtype=torch.float32),
        }
        inventory = MacroStageExpertInventory(
            tuple(
                ExpertRecord(
                    key,
                    tensor.numel() * tensor.element_size(),
                    f"budget-regression:{key.expert}",
                )
                for key, tensor in tensors.items()
            )
        )
        config = PredictiveCacheConfig(
            capacity_bytes=40,
            prefetch_reserve_bytes=10,
            pcie_bandwidth_gbytes_per_second=1,
            hotness_decay=1,
        )
        store = TorchRamExpertStore(
            inventory,
            tensors,
            config,
            device="cpu",
            pin_memory=False,
        )
        scheduler = RamBackedExpertScheduler(
            inventory,
            config,
            weight_backend=store,
        )

        scheduler.resolve_route(layer=7, authoritative_expert_ids=(0,))
        scheduler.submit_prefetch(
            speculative,
            deadline_ms=10,
            confidence=0.9,
            now_ms=0,
        )
        scheduler.stage_prefetch(now_ms=0)
        before = store.snapshot()
        self.assertEqual(
            before.active_bytes + before.prefetch_bytes + 32,
            56,
        )

        resolution = scheduler.resolve_route(
            layer=7,
            authoritative_expert_ids=(2,),
            predicted_expert_ids=(1,),
        )

        snapshot = store.snapshot()
        self.assertEqual(resolution.authoritative_experts, (authoritative,))
        self.assertEqual(resolution.ram_fallbacks, (authoritative,))
        self.assertEqual(resolution.evicted, (active,))
        self.assertEqual(snapshot.prefetch_keys, ())
        self.assertEqual(snapshot.active_keys, ())
        self.assertEqual(snapshot.ephemeral_keys, (authoritative,))
        self.assertLessEqual(
            snapshot.active_bytes
            + snapshot.prefetch_bytes
            + snapshot.ephemeral_bytes,
            config.capacity_bytes,
        )

    def test_store_fails_before_copy_when_callers_skip_route_reservation(self) -> None:
        active = ExpertKey(8, 0)
        speculative = ExpertKey(8, 1)
        oversized_fallback = ExpertKey(8, 2)
        tensors = {
            active: torch.ones(4, dtype=torch.float32),
            speculative: torch.ones(2, dtype=torch.float32),
            oversized_fallback: torch.ones(8, dtype=torch.float32),
        }
        inventory = MacroStageExpertInventory(
            tuple(
                ExpertRecord(
                    key,
                    tensor.numel() * tensor.element_size(),
                    f"store-guard:{key.expert}",
                )
                for key, tensor in tensors.items()
            )
        )
        config = PredictiveCacheConfig(
            capacity_bytes=40,
            prefetch_reserve_bytes=10,
            pcie_bandwidth_gbytes_per_second=1,
        )
        store = TorchRamExpertStore(
            inventory,
            tensors,
            config,
            device="cpu",
            pin_memory=False,
        )
        store.load_for_execution(inventory.record(active))
        store.commit_loaded(active, cached=True)
        store.prefetch(inventory.record(speculative))
        copies_before = store.snapshot().fallback_copies

        with self.assertRaisesRegex(
            MemoryError,
            "physical execution load would require 56 bytes",
        ):
            store.load_for_execution(inventory.record(oversized_fallback))

        snapshot = store.snapshot()
        self.assertEqual(snapshot.fallback_copies, copies_before)
        self.assertEqual(snapshot.ephemeral_keys, ())
        self.assertEqual(
            snapshot.active_bytes + snapshot.prefetch_bytes,
            24,
        )

    def test_authoritative_route_too_large_fails_without_evicting_protected_expert(self) -> None:
        active = ExpertKey(9, 0)
        second = ExpertKey(9, 1)
        tensors = {
            active: torch.ones(4, dtype=torch.float32),
            second: torch.ones(8, dtype=torch.float32),
        }
        inventory = MacroStageExpertInventory(
            tuple(
                ExpertRecord(
                    key,
                    tensor.numel() * tensor.element_size(),
                    f"protected-route:{key.expert}",
                )
                for key, tensor in tensors.items()
            )
        )
        config = PredictiveCacheConfig(
            capacity_bytes=40,
            prefetch_reserve_bytes=10,
            pcie_bandwidth_gbytes_per_second=1,
        )
        store = TorchRamExpertStore(
            inventory,
            tensors,
            config,
            device="cpu",
            pin_memory=False,
        )
        scheduler = RamBackedExpertScheduler(
            inventory,
            config,
            weight_backend=store,
        )
        scheduler.resolve_route(layer=9, authoritative_expert_ids=(0,))

        with self.assertRaisesRegex(
            MemoryError,
            "authoritative expert route requires 48 bytes",
        ):
            scheduler.resolve_route(
                layer=9,
                authoritative_expert_ids=(0, 1),
            )

        self.assertEqual(scheduler.snapshot().active_keys, (active,))
        self.assertEqual(store.snapshot().active_keys, (active,))
        self.assertEqual(store.snapshot().fallback_copies, 1)

    def test_store_rejects_tensor_identity_and_scheduler_config_mismatches(self) -> None:
        key = ExpertKey(0, 0)
        inventory = MacroStageExpertInventory(
            (ExpertRecord(key, 16, "identity"),)
        )
        config = PredictiveCacheConfig(
            capacity_bytes=64,
            prefetch_reserve_bytes=16,
            pcie_bandwidth_gbytes_per_second=1,
        )
        with self.assertRaisesRegex(ValueError, "tensor bytes"):
            TorchRamExpertStore(
                inventory,
                {key: torch.ones(3, dtype=torch.float32)},
                config,
                device="cpu",
            )

        store = TorchRamExpertStore(
            inventory,
            {key: torch.ones(4, dtype=torch.float32)},
            config,
            device="cpu",
        )
        different = PredictiveCacheConfig(
            capacity_bytes=80,
            prefetch_reserve_bytes=16,
            pcie_bandwidth_gbytes_per_second=1,
        )
        with self.assertRaisesRegex(ValueError, "configs must match"):
            RamBackedExpertScheduler(
                inventory,
                different,
                weight_backend=store,
            )


if __name__ == "__main__":
    unittest.main()
