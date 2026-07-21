from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from distributed_runtime.ram_backed_moe_stage import (
    RamBackedMoeExperts,
    RamBackedMoeStageRunner,
    RamBackedSwiGluExpertOwner,
)
from distributed_runtime.ram_expert_cache import (
    ExpertKey,
    ExpertRecord,
    MacroStageExpertInventory,
    PredictiveCacheConfig,
    RamBackedExpertScheduler,
)
from distributed_runtime.resident_expert_mesh import (
    MeshLinkProfile,
    MeshNodeProfile,
    OwnerCoalescedExpertBatchItem,
    OwnerExpertBatchItem,
    ResidentExpertMesh,
    ResidentExpertReplica,
)
from distributed_runtime.torch_ram_expert_store import (
    TorchExpertBundle,
    TorchRamExpertStore,
)


class _Coordinator:
    prefetch_deadline_ms = 1_000.0

    def prediction_for(self, layer: int) -> tuple[int, ...]:
        if layer != 0:
            raise AssertionError("unexpected layer")
        return ()

    def after_authoritative_route(self, layer: int, experts) -> None:
        if layer != 0 or not tuple(experts):
            raise AssertionError("invalid authoritative route")


class _ClosableOwner:
    def __init__(self, delegate: RamBackedSwiGluExpertOwner) -> None:
        self.delegate = delegate
        self.node_id = delegate.node_id
        self.close_calls = 0

    def has_expert(self, key: ExpertKey, content_id: str) -> bool:
        return self.delegate.has_expert(key, content_id)

    def is_expert_resident(self, key: ExpertKey, content_id: str) -> bool:
        return self.delegate.is_expert_resident(key, content_id)

    def execute_batch(self, items):
        return self.delegate.execute_batch(items)

    def close(self) -> None:
        self.close_calls += 1


def _bundles() -> tuple[dict[ExpertKey, TorchExpertBundle], tuple[ExpertRecord, ...]]:
    result: dict[ExpertKey, TorchExpertBundle] = {}
    records: list[ExpertRecord] = []
    for expert in range(2):
        key = ExpertKey(0, expert)
        scale = float(expert + 1)
        bundle = TorchExpertBundle(
            key=key,
            content_id=f"sha256:test-layer-0-expert-{expert}",
            tensors=(
                (
                    "gate_proj.weight",
                    torch.tensor(
                        [
                            [0.2, -0.1, 0.4],
                            [0.5, 0.3, -0.2],
                            [-0.4, 0.6, 0.1],
                            [0.7, -0.5, 0.2],
                        ],
                        dtype=torch.float32,
                    )
                    * scale,
                ),
                (
                    "up_proj.weight",
                    torch.tensor(
                        [
                            [0.3, 0.4, -0.2],
                            [-0.1, 0.8, 0.5],
                            [0.6, -0.3, 0.7],
                            [0.2, 0.1, 0.9],
                        ],
                        dtype=torch.float32,
                    ),
                ),
                (
                    "down_proj.weight",
                    torch.tensor(
                        [
                            [0.4, -0.2, 0.6, 0.1],
                            [0.3, 0.7, -0.1, 0.5],
                            [-0.5, 0.2, 0.8, 0.4],
                        ],
                        dtype=torch.float32,
                    ),
                ),
            ),
        )
        result[key] = bundle
        records.append(ExpertRecord(key, bundle.byte_size, bundle.content_id))
    return result, tuple(records)


def _packed_bundles(
    bundles: dict[ExpertKey, TorchExpertBundle],
) -> dict[ExpertKey, TorchExpertBundle]:
    result: dict[ExpertKey, TorchExpertBundle] = {}
    for key, unpacked in bundles.items():
        rows = int(unpacked.tensor("gate_proj.weight").shape[0])
        packed_gate_up = torch.cat(
            (
                unpacked.tensor("gate_proj.weight"),
                unpacked.tensor("up_proj.weight"),
            ),
            dim=0,
        ).contiguous()
        result[key] = TorchExpertBundle(
            key=key,
            content_id=unpacked.content_id,
            tensors=(
                ("gate_proj.weight", packed_gate_up[:rows]),
                ("up_proj.weight", packed_gate_up[rows:]),
                ("down_proj.weight", unpacked.tensor("down_proj.weight")),
            ),
        )
    return result


def _cache(expert_bytes: int) -> PredictiveCacheConfig:
    return PredictiveCacheConfig(
        capacity_bytes=expert_bytes * 2,
        prefetch_reserve_bytes=expert_bytes,
        pcie_bandwidth_gbytes_per_second=8.0,
    )


def _module(
    bundles: dict[ExpertKey, TorchExpertBundle],
    records: tuple[ExpertRecord, ...],
) -> RamBackedMoeExperts:
    inventory = MacroStageExpertInventory(records)
    config = _cache(max(record.byte_size for record in records))
    store = TorchRamExpertStore(
        inventory,
        bundles,
        config,
        device="cpu",
        pin_memory=False,
    )
    scheduler = RamBackedExpertScheduler(inventory, config, weight_backend=store)
    return RamBackedMoeExperts(
        layer=0,
        num_experts=2,
        hidden_dim=3,
        intermediate_dim=4,
        act_fn=F.silu,
        scheduler=scheduler,
        store=store,
        coordinator=_Coordinator(),  # type: ignore[arg-type]
    )


def _remote_owner(
    bundle: TorchExpertBundle,
    record: ExpertRecord,
) -> _ClosableOwner:
    inventory = MacroStageExpertInventory((record,))
    config = _cache(record.byte_size)
    store = TorchRamExpertStore(
        inventory,
        {record.key: bundle},
        config,
        device="cpu",
        pin_memory=False,
    )
    scheduler = RamBackedExpertScheduler(inventory, config, weight_backend=store)
    delegate = RamBackedSwiGluExpertOwner(
        "remote",
        layer=0,
        num_experts=2,
        hidden_dim=3,
        intermediate_dim=4,
        act_fn=F.silu,
        scheduler=scheduler,
        store=store,
    )
    # A configured remote-resident replica must be physically active before it
    # can attest residency. This warm-up is outside mesh execution.
    resolution = scheduler.resolve_route(
        layer=record.key.layer,
        authoritative_expert_ids=(record.key.expert,),
    )
    scheduler.release_route(resolution)
    return _ClosableOwner(delegate)


def _mesh(records: tuple[ExpertRecord, ...]) -> ResidentExpertMesh:
    keys = tuple(record.key for record in records)
    workspace = 4 * 4 * 4
    return ResidentExpertMesh(
        coordinator_id="root",
        experts=records,
        nodes=(
            MeshNodeProfile(
                "root",
                resident_vram_budget_bytes=1024 * 1024,
                reserved_vram_bytes=0,
                expert_compute_ms_per_token=2.0,
                ram_to_device_gbytes_per_second=0.0001,
                aggregate_ingress_mbps=10_000,
                aggregate_egress_mbps=10_000,
                expert_workspace_bytes_per_token=workspace,
            ),
            MeshNodeProfile(
                "remote",
                resident_vram_budget_bytes=1024 * 1024,
                reserved_vram_bytes=0,
                expert_compute_ms_per_token=0.01,
                expert_workspace_bytes_per_token=workspace,
            ),
        ),
        links=(
            MeshLinkProfile(
                "root",
                "remote",
                round_trip_ms=0.01,
                bandwidth_mbps=10_000,
                rpc_setup_ms_per_batch=0.01,
                host_device_staging_gbytes_per_second=8.0,
            ),
        ),
        local_ram_keys=keys,
        local_gpu_keys=(ExpertKey(0, 0),),
        replicas=(
            ResidentExpertReplica(
                ExpertKey(0, 1),
                "remote",
                records[1].content_id,
            ),
        ),
        local_weight_buffer_bytes=max(record.byte_size for record in records),
        activation_bytes_per_token=3 * 4,
    )


class RamBackedMoeMeshIntegrationTests(unittest.TestCase):
    def test_coalesced_owner_matches_raw_experts_and_stages_shared_input_once(
        self,
    ) -> None:
        bundles, records = _bundles()
        executable_bundles = _packed_bundles(bundles)
        inventory = MacroStageExpertInventory(records)
        config = _cache(max(record.byte_size for record in records))

        class FakeStore:
            effective_device = torch.device("cuda:0")

            @staticmethod
            def ram_bundle(candidate: ExpertKey) -> TorchExpertBundle:
                return executable_bundles[candidate]

            @staticmethod
            def bundle(candidate: ExpertKey) -> TorchExpertBundle:
                return executable_bundles[candidate]

        class FakeScheduler:
            def __init__(self) -> None:
                self.inventory = inventory
                self.config = config
                self.resolved: list[ExpertKey] = []
                self.released = 0

            def resolve_route(self, *, layer, authoritative_expert_ids):
                self.assert_layer(layer)
                key = ExpertKey(layer, tuple(authoritative_expert_ids)[0])
                self.resolved.append(key)
                return object()

            @staticmethod
            def assert_layer(layer: int) -> None:
                if layer != 0:
                    raise AssertionError("unexpected layer")

            def release_route(self, _) -> None:
                self.released += 1

            @staticmethod
            def submit_prefetch(*_, **__) -> None:
                return None

            @staticmethod
            def stage_prefetch(*_, **__) -> None:
                return None

        scheduler = FakeScheduler()
        owner = RamBackedSwiGluExpertOwner(
            "remote",
            layer=0,
            num_experts=2,
            hidden_dim=3,
            intermediate_dim=4,
            act_fn=F.silu,
            scheduler=scheduler,  # type: ignore[arg-type]
            store=FakeStore(),  # type: ignore[arg-type]
        )
        shared = torch.tensor(
            [
                [0.25, -0.5, 1.0],
                [1.5, 0.1, -0.75],
                [-0.4, 0.8, 0.2],
            ],
            dtype=torch.float32,
        )
        items = (
            OwnerCoalescedExpertBatchItem(
                ExpertKey(0, 0),
                records[0].content_id,
                (0, 2),
            ),
            OwnerCoalescedExpertBatchItem(
                ExpertKey(0, 1),
                records[1].content_id,
                (0, 1),
            ),
        )
        requested_devices: list[torch.device] = []
        original_to = torch.Tensor.to

        def record_accelerator_transfer(tensor, *args, **kwargs):
            requested = kwargs.get("device")
            if requested is not None and torch.device(requested).type == "cuda":
                requested_devices.append(torch.device(requested))
                # CI has no CUDA.  Preserve the tensor on CPU while proving
                # that the whole shared matrix requested one accelerator copy.
                return tensor
            return original_to(tensor, *args, **kwargs)

        with patch.object(torch.Tensor, "to", new=record_accelerator_transfer):
            actual = owner.execute_coalesced_batch(shared, items)

        expected = []
        for item in items:
            current = shared[list(item.row_indices)]
            bundle = executable_bundles[item.key]
            gate_values, up_values = F.linear(
                current,
                bundle.packed_gate_up(),
            ).chunk(2, dim=-1)
            expected.append(
                F.linear(
                    F.silu(gate_values) * up_values,
                    bundle.tensor("down_proj.weight"),
                )
            )
        self.assertEqual(requested_devices, [torch.device("cuda:0")])
        self.assertEqual(scheduler.resolved, [ExpertKey(0, 0), ExpertKey(0, 1)])
        self.assertEqual(scheduler.released, 2)
        self.assertEqual(owner.batch_calls, 1)
        self.assertEqual(owner.coalesced_batch_calls, 1)
        for result, reference in zip(actual, expected, strict=True):
            torch.testing.assert_close(result.output, reference, rtol=0, atol=0)

    def test_coalesced_preflight_rejects_unused_rows_before_scheduler_mutation(
        self,
    ) -> None:
        bundles, records = _bundles()
        module = _module(bundles, records)
        owner = RamBackedSwiGluExpertOwner(
            "root",
            layer=0,
            num_experts=2,
            hidden_dim=3,
            intermediate_dim=4,
            act_fn=F.silu,
            scheduler=module.scheduler,
            store=module.store,
        )
        before = module.scheduler.snapshot()

        with self.assertRaisesRegex(ValueError, "every shared activation"):
            owner.execute_coalesced_batch(
                torch.ones((2, 3), dtype=torch.float32),
                (
                    OwnerCoalescedExpertBatchItem(
                        ExpertKey(0, 0),
                        records[0].content_id,
                        (0,),
                    ),
                ),
            )

        self.assertEqual(module.scheduler.snapshot(), before)
        self.assertEqual(owner.batch_calls, 0)
        self.assertEqual(owner.coalesced_batch_calls, 0)

    def test_coalesced_rechecks_residency_before_each_expert(self) -> None:
        bundles, records = _bundles()
        executable_bundles = _packed_bundles(bundles)
        inventory = MacroStageExpertInventory(records)
        config = _cache(max(record.byte_size for record in records))

        class FakeStore:
            effective_device = torch.device("cpu")

            @staticmethod
            def ram_bundle(candidate: ExpertKey) -> TorchExpertBundle:
                return executable_bundles[candidate]

            @staticmethod
            def bundle(candidate: ExpertKey) -> TorchExpertBundle:
                return executable_bundles[candidate]

        class Resolution:
            ram_fallbacks = ()

            def __init__(self, key: ExpertKey) -> None:
                self.cache_hits = (key,)

        class FakeScheduler:
            def __init__(self) -> None:
                self.inventory = inventory
                self.config = config
                self.resolved: list[ExpertKey] = []
                self.released = 0

            def resolve_route(self, *, layer, authoritative_expert_ids):
                key = ExpertKey(layer, tuple(authoritative_expert_ids)[0])
                self.resolved.append(key)
                return Resolution(key)

            def release_route(self, _) -> None:
                self.released += 1

            @staticmethod
            def submit_prefetch(*_, **__) -> None:
                return None

            @staticmethod
            def stage_prefetch(*_, **__) -> None:
                return None

        scheduler = FakeScheduler()
        owner = RamBackedSwiGluExpertOwner(
            "remote",
            layer=0,
            num_experts=2,
            hidden_dim=3,
            intermediate_dim=4,
            act_fn=F.silu,
            scheduler=scheduler,  # type: ignore[arg-type]
            store=FakeStore(),  # type: ignore[arg-type]
        )
        items = tuple(
            OwnerCoalescedExpertBatchItem(
                ExpertKey(0, expert),
                records[expert].content_id,
                (expert,),
                require_resident=True,
            )
            for expert in range(2)
        )

        # Two preflight proofs, then one proof per execution.  The second
        # execution proof fails after the first route has been released.
        with patch.object(
            owner,
            "is_expert_resident",
            side_effect=(True, True, True, False),
        ) as residency:
            with self.assertRaisesRegex(RuntimeError, "before execution"):
                owner.execute_coalesced_batch(
                    torch.ones((2, 3), dtype=torch.float32),
                    items,
                )

        self.assertEqual(residency.call_count, 4)
        self.assertEqual(scheduler.resolved, [ExpertKey(0, 0)])
        self.assertEqual(scheduler.released, 1)

    def test_cold_ram_backed_owner_rejects_resident_route_without_ram_miss(self) -> None:
        bundles, records = _bundles()
        module = _module(bundles, records)
        owner = RamBackedSwiGluExpertOwner(
            "root",
            layer=0,
            num_experts=2,
            hidden_dim=3,
            intermediate_dim=4,
            act_fn=F.silu,
            scheduler=module.scheduler,
            store=module.store,
        )
        key = ExpertKey(0, 0)
        before = module.scheduler.snapshot().ram_misses

        with self.assertRaisesRegex(RuntimeError, "physical residency"):
            owner.execute_batch(
                (
                    OwnerExpertBatchItem(
                        key,
                        records[0].content_id,
                        torch.ones((1, 3), dtype=torch.float32),
                        require_resident=True,
                    ),
                )
            )

        self.assertEqual(module.scheduler.snapshot().ram_misses, before)

    def test_rpc_cpu_activation_requests_h2d_for_accelerator_owner(self) -> None:
        bundles, records = _bundles()
        key = ExpertKey(0, 1)
        unpacked = bundles[key]
        packed_gate_up = torch.cat(
            (
                unpacked.tensor("gate_proj.weight"),
                unpacked.tensor("up_proj.weight"),
            ),
            dim=0,
        ).contiguous()
        bundle = TorchExpertBundle(
            key=key,
            content_id=unpacked.content_id,
            tensors=(
                ("gate_proj.weight", packed_gate_up[:4]),
                ("up_proj.weight", packed_gate_up[4:]),
                ("down_proj.weight", unpacked.tensor("down_proj.weight")),
            ),
        )
        inventory = MacroStageExpertInventory((records[1],))

        class FakeStore:
            effective_device = torch.device("cuda:0")

            @staticmethod
            def ram_bundle(candidate: ExpertKey) -> TorchExpertBundle:
                if candidate != key:
                    raise KeyError(candidate)
                return bundle

            @staticmethod
            def bundle(candidate: ExpertKey) -> TorchExpertBundle:
                if candidate != key:
                    raise KeyError(candidate)
                return bundle

        class FakeScheduler:
            config = _cache(records[1].byte_size)

            def __init__(self) -> None:
                self.inventory = inventory

            @staticmethod
            def resolve_route(**_):
                return object()

            @staticmethod
            def release_route(_) -> None:
                return None

        owner = RamBackedSwiGluExpertOwner(
            "remote",
            layer=0,
            num_experts=2,
            hidden_dim=3,
            intermediate_dim=4,
            act_fn=F.silu,
            scheduler=FakeScheduler(),  # type: ignore[arg-type]
            store=FakeStore(),  # type: ignore[arg-type]
        )
        activation = torch.tensor([[0.25, -0.5, 1.0]], dtype=torch.float32)
        requested_devices: list[torch.device] = []
        original_to = torch.Tensor.to

        def record_accelerator_transfer(tensor, *args, **kwargs):
            requested = kwargs.get("device")
            if requested is not None and torch.device(requested).type == "cuda":
                requested_devices.append(torch.device(requested))
                # CI has no CUDA. Returning the CPU tensor lets the rest of the
                # exact operator verify that the explicit transfer was issued.
                return tensor
            return original_to(tensor, *args, **kwargs)

        with patch.object(torch.Tensor, "to", new=record_accelerator_transfer):
            result = owner.execute_batch(
                (
                    OwnerExpertBatchItem(
                        key,
                        records[1].content_id,
                        activation,
                    ),
                )
            )

        self.assertEqual(requested_devices, [torch.device("cuda:0")])
        self.assertEqual(tuple(result[0].output.shape), (1, 3))

    def test_serial_and_opt_in_remote_mesh_match_output_and_tokens(self) -> None:
        bundles, records = _bundles()
        serial = _module(bundles, records)
        meshed = _module(bundles, records)
        remote = _remote_owner(bundles[ExpertKey(0, 1)], records[1])
        local_resolution = meshed.scheduler.resolve_route(
            layer=0,
            authoritative_expert_ids=(0,),
        )
        meshed.scheduler.release_route(local_resolution)
        meshed.attach_resident_expert_mesh(_mesh(records), {"remote": remote})
        remote_ram_misses_before = remote.delegate.scheduler.snapshot().ram_misses

        hidden = torch.tensor(
            [
                [0.25, -0.5, 1.0],
                [1.5, 0.1, -0.75],
                [-0.4, 0.8, 0.2],
                [0.7, -1.2, 0.3],
            ],
            dtype=torch.float32,
        )
        selected = torch.tensor(
            [[0, 1], [1, 0], [0, 1], [1, 0]],
            dtype=torch.long,
        )
        weights = torch.tensor(
            [[0.7, 0.3], [0.55, 0.45], [0.2, 0.8], [0.6, 0.4]],
            dtype=torch.float32,
        )

        expected = serial(hidden, selected, weights)
        actual = meshed(hidden, selected, weights)

        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
        lm_head = torch.tensor(
            [
                [0.3, -0.1, 0.7],
                [-0.5, 0.8, 0.2],
                [0.6, 0.4, -0.3],
                [0.1, 0.9, 0.5],
                [-0.2, 0.3, 0.4],
            ],
            dtype=torch.float32,
        )
        expected_tokens = torch.argmax(expected @ lm_head.T, dim=-1)
        actual_tokens = torch.argmax(actual @ lm_head.T, dim=-1)
        self.assertEqual(actual_tokens.tolist(), expected_tokens.tolist())

        plan = meshed.last_resident_mesh_plan
        self.assertIsNotNone(plan)
        self.assertEqual(
            {dispatch.key.expert: dispatch.owner_id for dispatch in plan.dispatches},
            {0: "root", 1: "remote"},
        )
        # The link and workspace are sealed, but this integration fixture does
        # not claim measured coordinator dispatch/serialization/reduction
        # costs.  The projection must therefore remain explicitly partial.
        self.assertTrue(plan.transport_calibration_required)
        self.assertTrue(plan.coordination_calibration_required)
        self.assertFalse(plan.workspace_calibration_required)
        self.assertEqual(remote.delegate.batch_calls, 1)
        self.assertEqual(meshed.resident_expert_local_owner.batch_calls, 1)
        self.assertEqual(meshed.ram_misses, 0)
        self.assertEqual(
            remote.delegate.scheduler.snapshot().ram_misses,
            remote_ram_misses_before,
        )

        meshed.detach_resident_expert_mesh()
        self.assertEqual(remote.close_calls, 0)

    def test_runner_attach_default_does_not_take_owner_lifecycle(self) -> None:
        bundles, records = _bundles()
        module = _module(bundles, records)
        remote = _remote_owner(bundles[ExpertKey(0, 1)], records[1])
        runner = object.__new__(RamBackedMoeStageRunner)
        runner._expert_modules = (module,)
        runner._resident_mesh_owned_owners = {}

        runner.attach_resident_expert_mesh(
            0,
            _mesh(records),
            {"remote": remote},
        )
        self.assertEqual(runner._resident_mesh_owned_owners, {})
        runner.detach_resident_expert_mesh(0)
        self.assertEqual(remote.close_calls, 0)

    def test_attach_rejects_unsealed_workspace_before_forward(self) -> None:
        bundles, records = _bundles()
        module = _module(bundles, records)
        remote = _remote_owner(bundles[ExpertKey(0, 1)], records[1])
        good = _mesh(records)
        bad = ResidentExpertMesh(
            coordinator_id="root",
            experts=records,
            nodes=tuple(
                MeshNodeProfile(
                    node.node_id,
                    node.resident_vram_budget_bytes,
                    node.reserved_vram_bytes,
                    node.expert_compute_ms_per_token,
                    node.ram_to_device_gbytes_per_second,
                    node.available,
                    node.aggregate_ingress_mbps,
                    node.aggregate_egress_mbps,
                    0,
                )
                for node in good.nodes
            ),
            links=(
                MeshLinkProfile(
                    "root",
                    "remote",
                    round_trip_ms=0.01,
                    bandwidth_mbps=10_000,
                    rpc_setup_ms_per_batch=0.01,
                    host_device_staging_gbytes_per_second=8.0,
                ),
            ),
            local_ram_keys=tuple(record.key for record in records),
            local_gpu_keys=(ExpertKey(0, 0),),
            replicas=good.replicas_for_layer(0),
            local_weight_buffer_bytes=max(record.byte_size for record in records),
            activation_bytes_per_token=12,
        )
        with self.assertRaisesRegex(ValueError, "unsealed SwiGLU workspace"):
            module.attach_resident_expert_mesh(bad, {"remote": remote})


if __name__ == "__main__":
    unittest.main()
