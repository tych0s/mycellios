from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import multiprocessing
import queue
import socket
import sys
import threading
import time
import traceback
import unittest

import torch
import torch.nn.functional as F

from distributed_runtime.ram_expert_cache import ExpertKey, ExpertRecord
from distributed_runtime.resident_expert_mesh import (
    AuthoritativeRouting,
    ExpertRouteUnavailableError,
    InMemoryExpertOwner,
    MeshLinkProfile,
    MeshNodeProfile,
    OwnerCoalescedExpertBatchItem,
    OwnerExpertBatchItem,
    OwnerExpertBatchResult,
    ResidentExpertMesh,
    ResidentExpertReplica,
)
from distributed_runtime.resident_expert_rpc import (
    RESIDENT_EXPERT_RPC_SCHEMA,
    ResidentExpertRpcClient,
    ResidentExpertRpcContractError,
    ResidentExpertRpcError,
    ResidentExpertRpcFrameTooLarge,
    ResidentExpertRpcInventoryEntry,
    ResidentExpertRpcLimits,
    ResidentExpertRpcRemoteError,
    ResidentExpertRpcResidentSlotUnavailable,
    ResidentExpertRpcServer,
    ResidentExpertRpcTruncatedFrame,
    _FRAME_PREFIX,
    _MAGIC,
    _TelemetryCounters,
    _canonical_header,
    _coverage_bitmap_bytes,
    _limits_document,
    _recv_frame,
    _send_frame,
    _tensor_payload,
)


HIDDEN_SIZE = 4
INTERMEDIATE_SIZE = 7
OWNER_NODE_ID = "owner-1"
KEYS = (ExpertKey(0, 0), ExpertKey(0, 1))
CONTENT_IDS = {
    KEYS[0]: "sha256:swiglu-layer-0-expert-0",
    KEYS[1]: "sha256:swiglu-layer-0-expert-1",
}


def _limits() -> ResidentExpertRpcLimits:
    return ResidentExpertRpcLimits(
        max_header_bytes=64 * 1024,
        max_payload_bytes=1024 * 1024,
        max_batch_items=8,
        max_inventory_items=8,
        max_tensor_elements=4096,
        io_timeout_seconds=1.0,
    )


def _inventory() -> tuple[ResidentExpertRpcInventoryEntry, ...]:
    return tuple(
        ResidentExpertRpcInventoryEntry(
            key=key,
            content_id=CONTENT_IDS[key],
            shape=(HIDDEN_SIZE, HIDDEN_SIZE),
            dtype="float32",
        )
        for key in KEYS
    )


def _swiglu_weights(
    key: ExpertKey,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    offset = float(key.expert + 1)
    gate = (
        torch.arange(
            INTERMEDIATE_SIZE * HIDDEN_SIZE,
            dtype=torch.float32,
        ).reshape(INTERMEDIATE_SIZE, HIDDEN_SIZE)
        / 31.0
        - 0.4
        + offset * 0.03
    )
    up = (
        torch.arange(
            INTERMEDIATE_SIZE * HIDDEN_SIZE,
            dtype=torch.float32,
        ).flip(0).reshape(INTERMEDIATE_SIZE, HIDDEN_SIZE)
        / 37.0
        - 0.25
        - offset * 0.02
    )
    down = (
        torch.arange(
            HIDDEN_SIZE * INTERMEDIATE_SIZE,
            dtype=torch.float32,
        ).reshape(HIDDEN_SIZE, INTERMEDIATE_SIZE)
        / 41.0
        - 0.3
        + offset * 0.01
    )
    return gate, up, down


def _run_swiglu(key: ExpertKey, hidden: torch.Tensor) -> torch.Tensor:
    gate, up, down = _swiglu_weights(key)
    return F.linear(F.silu(F.linear(hidden, gate)) * F.linear(hidden, up), down)


class _SwiGLUOwner:
    """Owner whose weights remain internal to the spawned server process."""

    def __init__(self) -> None:
        self.node_id = OWNER_NODE_ID
        self._weights = {key: _swiglu_weights(key) for key in KEYS}
        self._resident = set(KEYS)
        self.batch_calls = 0
        self.require_resident_values: list[bool] = []

    def has_expert(self, key: ExpertKey, content_id: str) -> bool:
        return key in self._weights and CONTENT_IDS[key] == content_id

    def is_expert_resident(self, key: ExpertKey, content_id: str) -> bool:
        return key in self._resident and self.has_expert(key, content_id)

    @torch.no_grad()
    def execute_batch(
        self,
        items: tuple[OwnerExpertBatchItem, ...],
    ) -> tuple[OwnerExpertBatchResult, ...]:
        if not items:
            raise ValueError("SwiGLU owner batch cannot be empty")
        self.batch_calls += 1
        results: list[OwnerExpertBatchResult] = []
        seen: set[ExpertKey] = set()
        for item in items:
            if item.key in seen:
                raise ValueError("SwiGLU owner batch repeats an expert")
            seen.add(item.key)
            if not self.has_expert(item.key, item.content_id):
                raise ExpertRouteUnavailableError("SwiGLU owner content mismatch")
            self.require_resident_values.append(item.require_resident)
            if item.require_resident and not self.is_expert_resident(
                item.key,
                item.content_id,
            ):
                raise ExpertRouteUnavailableError("SwiGLU expert is not resident")
            gate, up, down = self._weights[item.key]
            output = F.linear(
                F.silu(F.linear(item.activations, gate))
                * F.linear(item.activations, up),
                down,
            )
            results.append(OwnerExpertBatchResult(item.key, output))
        return tuple(results)


class _CoalescedSwiGLUOwner(_SwiGLUOwner):
    def __init__(self) -> None:
        super().__init__()
        self.coalesced_calls = 0

    @torch.no_grad()
    def execute_coalesced_batch(
        self,
        shared_activations: torch.Tensor,
        items: tuple[OwnerCoalescedExpertBatchItem, ...],
    ) -> tuple[OwnerExpertBatchResult, ...]:
        self.batch_calls += 1
        self.coalesced_calls += 1
        results: list[OwnerExpertBatchResult] = []
        for item in items:
            self.require_resident_values.append(item.require_resident)
            if item.require_resident and not self.is_expert_resident(
                item.key,
                item.content_id,
            ):
                raise ExpertRouteUnavailableError("SwiGLU expert is not resident")
            rows = torch.tensor(item.row_indices, dtype=torch.long)
            results.append(
                OwnerExpertBatchResult(
                    item.key,
                    _run_swiglu(
                        item.key,
                        shared_activations.index_select(0, rows),
                    ),
                )
            )
        return tuple(results)


def _server_process(
    message_queue: object,
    stop_event: object,
) -> None:
    owner = _SwiGLUOwner()
    server: ResidentExpertRpcServer | None = None
    try:
        server = ResidentExpertRpcServer(
            "127.0.0.1",
            0,
            node_id=OWNER_NODE_ID,
            owner=owner,
            inventory=_inventory(),
            limits=_limits(),
            accept_poll_seconds=0.05,
        )
        _, port = server.bind()
        message_queue.put({"kind": "ready", "port": port})
        server.serve_forever(stop_event)
        message_queue.put(
            {
                "kind": "stopped",
                "ownerBatchCalls": owner.batch_calls,
                "requireResidentValues": owner.require_resident_values,
                "telemetry": asdict(server.telemetry_snapshot()),
            }
        )
    except BaseException:
        message_queue.put({"kind": "error", "traceback": traceback.format_exc()})
        raise
    finally:
        if server is not None:
            server.close()


def _expert_weight_bytes() -> int:
    gate, up, down = _swiglu_weights(KEYS[0])
    return sum(tensor.numel() * tensor.element_size() for tensor in (gate, up, down))


def _start_threaded_server(
    owner: _SwiGLUOwner,
    *,
    limits: ResidentExpertRpcLimits | None = None,
    enable_coalesced_extension: bool = True,
) -> tuple[ResidentExpertRpcServer, threading.Event, threading.Thread, int]:
    server = ResidentExpertRpcServer(
        "127.0.0.1",
        0,
        node_id=OWNER_NODE_ID,
        owner=owner,
        inventory=_inventory(),
        limits=limits or _limits(),
        accept_poll_seconds=0.02,
        enable_coalesced_extension=enable_coalesced_extension,
    )
    _, port = server.bind()
    stop = threading.Event()
    thread = threading.Thread(
        target=server.serve_forever,
        args=(stop,),
        name="resident-expert-rpc-test-thread",
    )
    thread.start()
    return server, stop, thread, port


def _fault_peer(
    listener: socket.socket,
    mode: str,
    errors: queue.Queue[str],
) -> None:
    try:
        listener.settimeout(3.0)
        connection, _ = listener.accept()
        with connection:
            connection.settimeout(3.0)
            hello = _recv_frame(connection, _limits())
            if hello is None or hello[0].get("type") != "hello":
                raise RuntimeError("fault peer did not receive a hello")
            _send_frame(
                connection,
                {
                    "inventory": [entry.to_document() for entry in _inventory()],
                    "limits": _limits_document(_limits()),
                    "nodeId": OWNER_NODE_ID,
                    "schema": RESIDENT_EXPERT_RPC_SCHEMA,
                    "type": "hello-ok",
                },
                b"",
                _limits(),
            )
            request = _recv_frame(connection, _limits())
            if request is None or request[0].get("type") != "execute-batch":
                raise RuntimeError("fault peer did not receive a batch")
            if mode == "mismatched-error":
                _send_frame(
                    connection,
                    {
                        "code": "batch_rejected",
                        "message": "deliberately mismatched response",
                        "requestId": 999,
                        "schema": RESIDENT_EXPERT_RPC_SCHEMA,
                        "type": "error",
                    },
                    b"",
                    _limits(),
                )
            elif mode == "unrelated-resident-error":
                request_id = int(request[0]["requestId"])
                _send_frame(
                    connection,
                    {
                        "code": "resident_slot_unavailable",
                        "message": '{"expert":1,"layer":0}',
                        "requestId": request_id,
                        "schema": RESIDENT_EXPERT_RPC_SCHEMA,
                        "type": "error",
                    },
                    b"",
                    _limits(),
                )
            elif mode == "truncated":
                partial_header = b'{"requestId":1'
                connection.sendall(
                    _FRAME_PREFIX.pack(_MAGIC, len(partial_header) + 10, 0)
                    + partial_header
                )
                connection.shutdown(socket.SHUT_WR)
            else:
                raise RuntimeError(f"unknown fault mode {mode!r}")
    except BaseException:
        errors.put(traceback.format_exc())
    finally:
        listener.close()


class ResidentExpertRpcTests(unittest.TestCase):
    def test_spawned_persistent_rpc_batches_two_swiglu_experts_for_mesh(self) -> None:
        context = multiprocessing.get_context("spawn")
        message_queue = context.Queue()
        stop_event = context.Event()
        process = context.Process(
            target=_server_process,
            args=(message_queue, stop_event),
            name="resident-expert-rpc-test-server",
        )
        client: ResidentExpertRpcClient | None = None
        stopped: dict[str, object] | None = None
        process.start()
        try:
            ready = message_queue.get(timeout=20)
            self.assertEqual(ready["kind"], "ready", ready)
            port = int(ready["port"])

            with self.assertRaises(ResidentExpertRpcRemoteError):
                ResidentExpertRpcClient(
                    "127.0.0.1",
                    port,
                    client_node_id="root",
                    expected_node_id="wrong-owner",
                    limits=_limits(),
                )

            aggregate_limits = ResidentExpertRpcLimits(
                max_header_bytes=64 * 1024,
                max_payload_bytes=31,
                max_batch_items=8,
                max_inventory_items=8,
                max_tensor_elements=4096,
                io_timeout_seconds=1.0,
            )
            bounded_client = ResidentExpertRpcClient(
                "127.0.0.1",
                port,
                client_node_id="root",
                expected_node_id=OWNER_NODE_ID,
                limits=aggregate_limits,
            )
            try:
                with self.assertRaises(ResidentExpertRpcFrameTooLarge):
                    bounded_client.execute_batch(
                        tuple(
                            OwnerExpertBatchItem(
                                key,
                                CONTENT_IDS[key],
                                torch.ones((1, HIDDEN_SIZE), dtype=torch.float32),
                            )
                            for key in KEYS
                        )
                    )
                self.assertEqual(
                    bounded_client.telemetry_snapshot().batch_round_trips,
                    0,
                )
                self.assertFalse(bounded_client.closed)
            finally:
                bounded_client.close()

            client = ResidentExpertRpcClient(
                "127.0.0.1",
                port,
                client_node_id="root",
                expected_node_id=OWNER_NODE_ID,
                limits=_limits(),
            )
            after_handshake = client.telemetry_snapshot()
            for _ in range(3):
                self.assertTrue(client.has_expert(KEYS[0], CONTENT_IDS[KEYS[0]]))
                self.assertTrue(
                    client.is_expert_resident(KEYS[0], CONTENT_IDS[KEYS[0]])
                )
                self.assertFalse(client.has_expert(KEYS[0], "sha256:stale"))
                self.assertFalse(client.is_expert_resident(KEYS[0], "sha256:stale"))
            self.assertEqual(client.telemetry_snapshot(), after_handshake)

            with self.assertRaises(ExpertRouteUnavailableError):
                client.execute_batch(
                    (
                        OwnerExpertBatchItem(
                            KEYS[0],
                            "sha256:stale",
                            torch.ones((1, HIDDEN_SIZE), dtype=torch.float32),
                        ),
                    )
                )
            with self.assertRaises(ResidentExpertRpcContractError):
                client.execute_batch(
                    (
                        OwnerExpertBatchItem(
                            KEYS[0],
                            CONTENT_IDS[KEYS[0]],
                            torch.ones((1, HIDDEN_SIZE + 1), dtype=torch.float32),
                        ),
                    )
                )
            self.assertEqual(client.telemetry_snapshot(), after_handshake)

            weight_bytes = _expert_weight_bytes()
            records = tuple(
                ExpertRecord(key, weight_bytes, CONTENT_IDS[key]) for key in KEYS
            )
            mesh = ResidentExpertMesh(
                coordinator_id="root",
                experts=records,
                nodes=(
                    MeshNodeProfile(
                        node_id="root",
                        resident_vram_budget_bytes=1024 * 1024,
                        reserved_vram_bytes=0,
                        expert_compute_ms_per_token=0.1,
                    ),
                    MeshNodeProfile(
                        node_id=OWNER_NODE_ID,
                        resident_vram_budget_bytes=(
                            weight_bytes * len(KEYS) + 1024 * 1024
                        ),
                        reserved_vram_bytes=0,
                        expert_compute_ms_per_token=0.1,
                        expert_workspace_bytes_per_token=(
                            4 * INTERMEDIATE_SIZE * 4
                        ),
                    ),
                ),
                links=(
                    MeshLinkProfile(
                        from_node="root",
                        to_node=OWNER_NODE_ID,
                        round_trip_ms=0.05,
                        bandwidth_mbps=10_000,
                    ),
                ),
                local_ram_keys=(),
                local_gpu_keys=(),
                replicas=tuple(
                    ResidentExpertReplica(key, OWNER_NODE_ID, CONTENT_IDS[key])
                    for key in KEYS
                ),
                local_weight_buffer_bytes=0,
                activation_bytes_per_token=HIDDEN_SIZE * 4,
                require_local_ram_fallback=False,
            )
            hidden = torch.tensor(
                [
                    [0.5, -1.0, 0.25, 2.0],
                    [1.5, 0.0, -0.75, 0.5],
                    [-0.5, 0.4, 1.25, -1.5],
                ],
                dtype=torch.float32,
            )
            routing = AuthoritativeRouting(
                expert_ids=torch.tensor(
                    [[0, 1], [1, 0], [0, 1]],
                    dtype=torch.long,
                ),
                expert_weights=torch.tensor(
                    [[0.7, 0.3], [0.45, 0.55], [0.2, 0.8]],
                    dtype=torch.float32,
                ),
            )
            expected = torch.zeros_like(hidden)
            for token_index in range(hidden.shape[0]):
                for slot_index in range(routing.top_k):
                    key = ExpertKey(
                        0,
                        int(routing.expert_ids[token_index, slot_index].item()),
                    )
                    expected[token_index] += (
                        routing.expert_weights[token_index, slot_index]
                        * _run_swiglu(key, hidden[token_index : token_index + 1])[0]
                    )

            actual, plan = mesh.execute_layer(
                hidden,
                0,
                routing,
                {OWNER_NODE_ID: client},
            )

            torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
            self.assertEqual(
                {dispatch.path for dispatch in plan.dispatches},
                {"remote-resident"},
            )
            telemetry = client.telemetry_snapshot()
            self.assertEqual(telemetry.handshakes, 1)
            self.assertEqual(telemetry.batch_round_trips, 1)
            self.assertEqual(telemetry.batch_items, 2)
            self.assertEqual(telemetry.coalesced_round_trips, 1)
            self.assertEqual(telemetry.coalesced_input_bytes_saved, 24)
            self.assertGreater(telemetry.bytes_sent, 0)
            self.assertGreater(telemetry.bytes_received, 0)
            self.assertFalse(client.closed)

            client.close()
            self.assertTrue(client.closed)
            self.assertEqual(client.telemetry_snapshot().close_round_trips, 1)
            stop_event.set()
            process.join(timeout=10)
            self.assertFalse(process.is_alive(), "spawned RPC server did not stop")
            self.assertEqual(process.exitcode, 0)
            stopped = message_queue.get(timeout=5)
            self.assertEqual(stopped["kind"], "stopped", stopped)
            self.assertEqual(stopped["ownerBatchCalls"], 1)
            self.assertEqual(stopped["requireResidentValues"], [True, True])
            server_telemetry = stopped["telemetry"]
            self.assertEqual(server_telemetry["batch_round_trips"], 1)
            self.assertEqual(server_telemetry["batch_items"], 2)
            self.assertEqual(server_telemetry["coalesced_round_trips"], 1)
            self.assertEqual(server_telemetry["coalesced_shared_input_bytes"], 48)
        finally:
            if client is not None and not client.closed:
                client.close()
            stop_event.set()
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            message_queue.close()
            message_queue.join_thread()
        self.assertFalse(process.is_alive(), "RPC test left an orphan process")
        self.assertIsNotNone(stopped)

    def test_exact_coalesced_transport_matches_v1_and_accounts_saved_bytes(self) -> None:
        owner = _CoalescedSwiGLUOwner()
        server, stop, thread, port = _start_threaded_server(owner)
        client: ResidentExpertRpcClient | None = None
        try:
            client = ResidentExpertRpcClient(
                "127.0.0.1",
                port,
                client_node_id="root",
                expected_node_id=OWNER_NODE_ID,
                limits=_limits(),
            )
            shared = torch.tensor(
                [
                    [0.5, -1.0, 0.25, 2.0],
                    [1.5, 0.0, -0.75, 0.5],
                    [-0.5, 0.4, 1.25, -1.5],
                ],
                dtype=torch.float32,
            )
            v1 = client.execute_batch(
                tuple(
                    OwnerExpertBatchItem(
                        key,
                        CONTENT_IDS[key],
                        shared,
                        require_resident=(key == KEYS[0]),
                    )
                    for key in KEYS
                )
            )
            coalesced = client.execute_coalesced_batch(
                shared,
                (
                    OwnerCoalescedExpertBatchItem(
                        KEYS[0], CONTENT_IDS[KEYS[0]], (0, 1, 2), True
                    ),
                    OwnerCoalescedExpertBatchItem(
                        KEYS[1], CONTENT_IDS[KEYS[1]], (0, 1, 2), False
                    ),
                ),
            )
            self.assertEqual(tuple(result.key for result in coalesced), KEYS)
            for original, packed in zip(v1, coalesced, strict=True):
                torch.testing.assert_close(packed.output, original.output, rtol=0, atol=0)

            telemetry = client.telemetry_snapshot()
            self.assertTrue(client.supports_exact_input_coalescing)
            self.assertEqual(client.coalesced_row_index_bytes_per_assignment, 4)
            self.assertEqual(telemetry.capability_round_trips, 1)
            self.assertEqual(telemetry.batch_round_trips, 2)
            self.assertEqual(telemetry.coalesced_round_trips, 1)
            self.assertEqual(telemetry.coalesced_items, 2)
            self.assertEqual(telemetry.coalesced_shared_input_bytes, 48)
            self.assertEqual(telemetry.coalesced_row_index_bytes, 24)
            self.assertEqual(telemetry.coalesced_v1_equivalent_input_bytes, 96)
            self.assertEqual(telemetry.coalesced_input_bytes_saved, 24)
            # RPC inventory is resident-only. The optional coalesced flag is
            # normalized to the same fail-closed requirement as v1.
            self.assertEqual(owner.require_resident_values[-2:], [True, True])
            self.assertEqual(owner.coalesced_calls, 1)

            # Q=E: no row reuse. The public client must select v1 even after a
            # successful extension probe because shared+uint32 is not smaller.
            no_reuse = client.execute_coalesced_batch(
                shared[:2],
                (
                    OwnerCoalescedExpertBatchItem(
                        KEYS[0], CONTENT_IDS[KEYS[0]], (0,), True
                    ),
                    OwnerCoalescedExpertBatchItem(
                        KEYS[1], CONTENT_IDS[KEYS[1]], (1,), True
                    ),
                ),
            )
            torch.testing.assert_close(
                no_reuse[0].output,
                _run_swiglu(KEYS[0], shared[0:1]),
                rtol=0,
                atol=0,
            )
            after_no_reuse = client.telemetry_snapshot()
            self.assertEqual(
                after_no_reuse.batch_round_trips,
                telemetry.batch_round_trips + 1,
            )
            self.assertEqual(
                after_no_reuse.coalesced_round_trips,
                telemetry.coalesced_round_trips,
            )
            self.assertEqual(owner.coalesced_calls, 1)
            self.assertFalse(client.closed)
        finally:
            if client is not None:
                client.close()
            stop.set()
            server.close()
            thread.join(timeout=2)
        self.assertFalse(thread.is_alive())

    def test_legacy_capability_rejection_falls_back_to_v1_on_same_stream(self) -> None:
        owner = _SwiGLUOwner()
        server, stop, thread, port = _start_threaded_server(
            owner,
            enable_coalesced_extension=False,
        )
        client: ResidentExpertRpcClient | None = None
        try:
            client = ResidentExpertRpcClient(
                "127.0.0.1",
                port,
                client_node_id="root",
                expected_node_id=OWNER_NODE_ID,
                limits=_limits(),
            )
            shared = torch.arange(12, dtype=torch.float32).reshape(3, 4)
            result = client.execute_coalesced_batch(
                shared,
                (
                    OwnerCoalescedExpertBatchItem(
                        KEYS[0], CONTENT_IDS[KEYS[0]], (0, 1, 2), True
                    ),
                    OwnerCoalescedExpertBatchItem(
                        KEYS[1], CONTENT_IDS[KEYS[1]], (0, 1, 2), False
                    ),
                ),
            )
            torch.testing.assert_close(
                result[0].output,
                _run_swiglu(KEYS[0], shared),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                result[1].output,
                _run_swiglu(KEYS[1], shared),
                rtol=0,
                atol=0,
            )
            telemetry = client.telemetry_snapshot()
            self.assertFalse(client.supports_exact_input_coalescing)
            self.assertEqual(telemetry.capability_round_trips, 1)
            self.assertEqual(telemetry.batch_round_trips, 1)
            self.assertEqual(telemetry.coalesced_round_trips, 0)
            self.assertFalse(client.closed)
        finally:
            if client is not None:
                client.close()
            stop.set()
            server.close()
            thread.join(timeout=2)
        self.assertFalse(thread.is_alive())

    def test_coalesced_rejects_malformed_rows_before_owner_compute(self) -> None:
        owner = _SwiGLUOwner()
        server = ResidentExpertRpcServer(
            "127.0.0.1",
            0,
            node_id=OWNER_NODE_ID,
            owner=owner,
            inventory=_inventory(),
            limits=_limits(),
        )
        shared = torch.ones((1, HIDDEN_SIZE), dtype=torch.float32)
        _cpu, shared_raw = _tensor_payload(shared)
        payload = bytearray(shared_raw)
        payload.extend((99).to_bytes(4, "little"))
        header = {
            "activation": {
                "dtype": "float32",
                "nbytes": 16,
                "offset": 0,
                "shape": [1, HIDDEN_SIZE],
            },
            "items": [
                {
                    "contentId": CONTENT_IDS[KEYS[0]],
                    "expert": KEYS[0].expert,
                    "layer": KEYS[0].layer,
                    "requireResident": True,
                    "rowCount": 1,
                    "rowNbytes": 4,
                    "rowOffset": 16,
                }
            ],
            "layer": 0,
            "requestId": 1,
            "schema": RESIDENT_EXPERT_RPC_SCHEMA,
            "type": "execute-coalesced",
        }
        with self.assertRaisesRegex(
            ResidentExpertRpcContractError,
            "row reference exceeds",
        ):
            server._execute_coalesced_batch(header, payload, _limits())
        self.assertEqual(owner.batch_calls, 0)
        server.close()

    def test_coalesced_limit_counts_decoded_row_reference_memory(self) -> None:
        owner = _SwiGLUOwner()
        server = ResidentExpertRpcServer(
            "127.0.0.1",
            0,
            node_id=OWNER_NODE_ID,
            owner=owner,
            inventory=_inventory(),
            limits=_limits(),
        )
        shared = torch.ones((1, HIDDEN_SIZE), dtype=torch.float32)
        _cpu, shared_raw = _tensor_payload(shared)
        payload = bytearray(shared_raw)
        payload.extend((0).to_bytes(4, "little") * 100)
        header = {
            "activation": {
                "dtype": "float32",
                "nbytes": 16,
                "offset": 0,
                "shape": [1, HIDDEN_SIZE],
            },
            "items": [
                {
                    "contentId": CONTENT_IDS[KEYS[0]],
                    "expert": KEYS[0].expert,
                    "layer": KEYS[0].layer,
                    "requireResident": True,
                    "rowCount": 100,
                    "rowNbytes": 400,
                    "rowOffset": 16,
                }
            ],
            "layer": 0,
            "requestId": 1,
            "schema": RESIDENT_EXPERT_RPC_SCHEMA,
            "type": "execute-coalesced",
        }
        decoded_limits = ResidentExpertRpcLimits(
            max_header_bytes=64 * 1024,
            max_payload_bytes=1024 * 1024,
            max_batch_items=8,
            max_inventory_items=8,
            max_tensor_elements=4096,
            max_host_transient_bytes=2_000,
            io_timeout_seconds=1.0,
        )
        self.assertLess(
            len(_canonical_header(header)) + 2 * len(payload),
            decoded_limits.max_host_transient_bytes,
        )
        with self.assertRaisesRegex(
            ResidentExpertRpcFrameTooLarge,
            "decoded row references",
        ):
            server._execute_coalesced_batch(header, payload, decoded_limits)
        self.assertEqual(owner.batch_calls, 0)
        self.assertEqual(
            server.telemetry_snapshot().host_transient_preflight_rejections,
            1,
        )
        server.close()

        remote_owner = _SwiGLUOwner()
        remote_server, stop, thread, port = _start_threaded_server(
            remote_owner,
            limits=decoded_limits,
        )
        client: ResidentExpertRpcClient | None = None
        try:
            client = ResidentExpertRpcClient(
                "127.0.0.1",
                port,
                client_node_id="root",
                expected_node_id=OWNER_NODE_ID,
                limits=decoded_limits,
            )
            with self.assertRaisesRegex(
                ResidentExpertRpcFrameTooLarge,
                "peer would require",
            ):
                client.execute_coalesced_batch(
                    shared,
                    (
                        OwnerCoalescedExpertBatchItem(
                            KEYS[0],
                            CONTENT_IDS[KEYS[0]],
                            (0,) * 100,
                            True,
                        ),
                    ),
                )
            self.assertEqual(client.telemetry_snapshot().batch_round_trips, 0)
            self.assertEqual(remote_owner.batch_calls, 0)
            self.assertFalse(client.closed)
        finally:
            if client is not None:
                client.close()
            stop.set()
            remote_server.close()
            thread.join(timeout=2)
        self.assertFalse(thread.is_alive())

    def test_coalesced_releases_request_staging_before_response_peak(self) -> None:
        # Request tensor+indices = 12,288 bytes and response tensors = 16,384.
        # 72,000 covers the sealed request peak including decoded Python row
        # refs (~66 KiB), but is below their old overlapping request staging +
        # decoded refs + 2*response: 12,288 + 40,960 + 32,768 = 86,016.
        causal_limits = ResidentExpertRpcLimits(
            max_header_bytes=64 * 1024,
            max_payload_bytes=1024 * 1024,
            max_batch_items=8,
            max_inventory_items=8,
            max_tensor_elements=4096,
            max_host_transient_bytes=72_000,
            io_timeout_seconds=1.0,
        )
        owner = _SwiGLUOwner()
        server, stop, thread, port = _start_threaded_server(
            owner,
            limits=causal_limits,
        )
        client: ResidentExpertRpcClient | None = None
        try:
            client = ResidentExpertRpcClient(
                "127.0.0.1",
                port,
                client_node_id="root",
                expected_node_id=OWNER_NODE_ID,
                limits=causal_limits,
            )
            shared = torch.arange(512 * HIDDEN_SIZE, dtype=torch.float32).reshape(
                512,
                HIDDEN_SIZE,
            )
            rows = tuple(range(512))
            result = client.execute_coalesced_batch(
                shared,
                tuple(
                    OwnerCoalescedExpertBatchItem(
                        key,
                        CONTENT_IDS[key],
                        rows,
                        True,
                    )
                    for key in KEYS
                ),
            )
            self.assertEqual(tuple(value.output.shape for value in result), ((512, 4),) * 2)
            self.assertFalse(client.closed)
            self.assertLess(causal_limits.max_host_transient_bytes, 86_016)
        finally:
            if client is not None:
                client.close()
            stop.set()
            server.close()
            thread.join(timeout=2)
        self.assertFalse(thread.is_alive())

    def test_response_identity_or_truncation_closes_ambiguous_stream(self) -> None:
        cases = (
            ("mismatched-error", ResidentExpertRpcContractError),
            ("unrelated-resident-error", ResidentExpertRpcContractError),
            ("truncated", ResidentExpertRpcTruncatedFrame),
        )
        for mode, expected_error in cases:
            with self.subTest(mode=mode):
                listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                listener.bind(("127.0.0.1", 0))
                listener.listen(1)
                port = int(listener.getsockname()[1])
                errors: queue.Queue[str] = queue.Queue()
                peer = threading.Thread(
                    target=_fault_peer,
                    args=(listener, mode, errors),
                    name=f"resident-expert-rpc-{mode}",
                )
                peer.start()
                client: ResidentExpertRpcClient | None = None
                try:
                    client = ResidentExpertRpcClient(
                        "127.0.0.1",
                        port,
                        client_node_id="root",
                        expected_node_id=OWNER_NODE_ID,
                        limits=_limits(),
                    )
                    with self.assertRaises(expected_error):
                        client.execute_batch(
                            (
                                OwnerExpertBatchItem(
                                    KEYS[0],
                                    CONTENT_IDS[KEYS[0]],
                                    torch.ones(
                                        (1, HIDDEN_SIZE),
                                        dtype=torch.float32,
                                    ),
                                ),
                            )
                        )
                    if mode == "unrelated-resident-error":
                        self.assertTrue(
                            client.is_expert_resident(
                                KEYS[1],
                                CONTENT_IDS[KEYS[1]],
                            )
                        )
                    self.assertTrue(client.closed)
                    with self.assertRaises(ResidentExpertRpcError):
                        client.execute_batch(
                            (
                                OwnerExpertBatchItem(
                                    KEYS[0],
                                    CONTENT_IDS[KEYS[0]],
                                    torch.ones(
                                        (1, HIDDEN_SIZE),
                                        dtype=torch.float32,
                                    ),
                                ),
                            )
                        )
                finally:
                    if client is not None and not client.closed:
                        client.close()
                    listener.close()
                    peer.join(timeout=5)
                self.assertFalse(peer.is_alive(), "fault peer thread did not stop")
                if not errors.empty():
                    self.fail(errors.get_nowait())

    def test_frame_prefix_rejects_oversized_payload_before_body_read(self) -> None:
        self.assertGreaterEqual(
            _coverage_bitmap_bytes(1),
            sys.getsizeof(bytearray(1)),
        )
        sender, receiver = socket.socketpair()
        try:
            limits = ResidentExpertRpcLimits(
                max_header_bytes=64,
                max_payload_bytes=64,
                max_batch_items=1,
                max_inventory_items=1,
                max_tensor_elements=16,
                io_timeout_seconds=0.2,
            )
            sender.sendall(_FRAME_PREFIX.pack(_MAGIC, 2, 65))
            with self.assertRaises(ResidentExpertRpcFrameTooLarge):
                _recv_frame(receiver, limits)
        finally:
            sender.close()
            receiver.close()

    def test_host_transient_budget_rejects_before_body_and_tracks_peak(self) -> None:
        sender, receiver = socket.socketpair()
        try:
            limits = ResidentExpertRpcLimits(
                max_header_bytes=64,
                max_payload_bytes=64,
                max_batch_items=1,
                max_inventory_items=1,
                max_tensor_elements=16,
                max_host_transient_bytes=100,
                io_timeout_seconds=0.2,
            )
            telemetry = _TelemetryCounters()
            # 2 header bytes + wire(50) + materialized tensor(50) = 102.
            # No body is sent: rejection must happen from the prefix alone.
            sender.sendall(_FRAME_PREFIX.pack(_MAGIC, 2, 50))
            with self.assertRaisesRegex(
                ResidentExpertRpcFrameTooLarge,
                "host-transient",
            ):
                _recv_frame(receiver, limits, telemetry=telemetry)
            rejected = telemetry.snapshot()
            self.assertEqual(rejected.host_transient_preflight_rejections, 1)
            self.assertEqual(rejected.peak_host_transient_bytes, 0)
        finally:
            sender.close()
            receiver.close()

        sender, receiver = socket.socketpair()
        try:
            limits = ResidentExpertRpcLimits(
                max_header_bytes=64,
                max_payload_bytes=64,
                max_batch_items=1,
                max_inventory_items=1,
                max_tensor_elements=16,
                max_host_transient_bytes=128,
                io_timeout_seconds=0.2,
            )
            telemetry = _TelemetryCounters()
            _send_frame(sender, {"kind": "probe"}, (b"1234", b"5678"), limits)
            frame = _recv_frame(receiver, limits, telemetry=telemetry)
            self.assertIsNotNone(frame)
            assert frame is not None
            header, payload, _ = frame
            self.assertEqual(header, {"kind": "probe"})
            self.assertIsInstance(payload, bytearray)
            self.assertEqual(payload, b"12345678")
            accepted = telemetry.snapshot()
            self.assertEqual(
                accepted.peak_host_transient_bytes,
                len(b'{"kind":"probe"}') + 2 * 8,
            )
            self.assertEqual(accepted.host_transient_preflight_rejections, 0)
        finally:
            sender.close()
            receiver.close()

    def test_server_close_unblocks_idle_active_connection_idempotently(self) -> None:
        owner = _SwiGLUOwner()
        server = ResidentExpertRpcServer(
            "127.0.0.1",
            0,
            node_id=OWNER_NODE_ID,
            owner=owner,
            inventory=_inventory(),
            limits=_limits(),
            accept_poll_seconds=0.05,
        )
        _, port = server.bind()
        stop = threading.Event()
        errors: queue.Queue[str] = queue.Queue()

        def serve() -> None:
            try:
                server.serve_forever(stop)
            except BaseException:
                errors.put(traceback.format_exc())

        thread = threading.Thread(target=serve, name="resident-rpc-idle-close")
        thread.start()
        idle = socket.create_connection(("127.0.0.1", port), timeout=1.0)
        try:
            idle.settimeout(1.0)
            _send_frame(
                idle,
                {
                    "clientNodeId": "idle-root",
                    "expectedNodeId": OWNER_NODE_ID,
                    "limits": _limits_document(_limits()),
                    "schema": RESIDENT_EXPERT_RPC_SCHEMA,
                    "type": "hello",
                },
                b"",
                _limits(),
            )
            hello = _recv_frame(idle, _limits())
            self.assertIsNotNone(hello)

            closers = [threading.Thread(target=server.close) for _ in range(4)]
            started = time.monotonic()
            for closer in closers:
                closer.start()
            for closer in closers:
                closer.join(timeout=1)
                self.assertFalse(closer.is_alive())
            thread.join(timeout=1)
            self.assertFalse(thread.is_alive(), "server close left idle recv blocked")
            self.assertLess(time.monotonic() - started, 1.0)
            self.assertTrue(errors.empty(), errors.queue)
            self.assertEqual(server.telemetry_snapshot().errors, 0)
        finally:
            stop.set()
            server.close()
            idle.close()
            thread.join(timeout=2)

    def test_residence_loss_after_handshake_rejects_before_owner_execution(self) -> None:
        nonresident = _SwiGLUOwner()
        nonresident._resident.clear()
        with self.assertRaisesRegex(ValueError, "not physically resident"):
            ResidentExpertRpcServer(
                "127.0.0.1",
                0,
                node_id=OWNER_NODE_ID,
                owner=nonresident,
                inventory=_inventory(),
                limits=_limits(),
            )

        owner = _SwiGLUOwner()
        server = ResidentExpertRpcServer(
            "127.0.0.1",
            0,
            node_id=OWNER_NODE_ID,
            owner=owner,
            inventory=_inventory(),
            limits=_limits(),
            accept_poll_seconds=0.05,
        )
        _, port = server.bind()
        stop = threading.Event()
        thread = threading.Thread(
            target=server.serve_forever,
            args=(stop,),
            name="resident-rpc-stale-residence",
        )
        thread.start()
        client: ResidentExpertRpcClient | None = None
        try:
            client = ResidentExpertRpcClient(
                "127.0.0.1",
                port,
                client_node_id="root",
                expected_node_id=OWNER_NODE_ID,
                limits=_limits(),
            )
            self.assertTrue(
                client.is_expert_resident(KEYS[0], CONTENT_IDS[KEYS[0]])
            )
            owner._resident.remove(KEYS[0])
            with self.assertRaises(ResidentExpertRpcResidentSlotUnavailable):
                client.execute_batch(
                    (
                        OwnerExpertBatchItem(
                            KEYS[0],
                            CONTENT_IDS[KEYS[0]],
                            torch.ones((1, HIDDEN_SIZE), dtype=torch.float32),
                            require_resident=True,
                        ),
                    )
                )
            self.assertFalse(
                client.is_expert_resident(KEYS[0], CONTENT_IDS[KEYS[0]])
            )
            rejected_round_trips = client.telemetry_snapshot().batch_round_trips
            with self.assertRaises(ResidentExpertRpcResidentSlotUnavailable):
                client.execute_coalesced_batch(
                    torch.ones((1, HIDDEN_SIZE), dtype=torch.float32),
                    (
                        OwnerCoalescedExpertBatchItem(
                            KEYS[0],
                            CONTENT_IDS[KEYS[0]],
                            (0,),
                            require_resident=True,
                        ),
                    ),
                )
            self.assertEqual(
                client.telemetry_snapshot().batch_round_trips,
                rejected_round_trips,
            )
            survivor = client.execute_batch(
                (
                    OwnerExpertBatchItem(
                        KEYS[1],
                        CONTENT_IDS[KEYS[1]],
                        torch.ones((1, HIDDEN_SIZE), dtype=torch.float32),
                        require_resident=True,
                    ),
                )
            )
            self.assertEqual(tuple(result.key for result in survivor), (KEYS[1],))
            self.assertEqual(owner.batch_calls, 1)
            self.assertEqual(owner.require_resident_values, [True])
            self.assertFalse(client.closed)
        finally:
            if client is not None:
                client.close()
            stop.set()
            server.close()
            thread.join(timeout=2)
        self.assertFalse(thread.is_alive())

    def test_concurrent_batches_share_one_residency_failure_round_trip(self) -> None:
        owner = _SwiGLUOwner()
        server, stop, thread, port = _start_threaded_server(owner)
        client: ResidentExpertRpcClient | None = None
        try:
            client = ResidentExpertRpcClient(
                "127.0.0.1",
                port,
                client_node_id="root",
                expected_node_id=OWNER_NODE_ID,
                limits=_limits(),
            )
            self.assertTrue(
                client.is_expert_resident(KEYS[0], CONTENT_IDS[KEYS[0]])
            )
            owner._resident.remove(KEYS[0])
            rendezvous = threading.Barrier(3)

            def execute_stale_batch() -> BaseException | None:
                rendezvous.wait(timeout=5.0)
                try:
                    client.execute_batch(
                        (
                            OwnerExpertBatchItem(
                                KEYS[0],
                                CONTENT_IDS[KEYS[0]],
                                torch.ones(
                                    (1, HIDDEN_SIZE),
                                    dtype=torch.float32,
                                ),
                                require_resident=True,
                            ),
                        )
                    )
                except BaseException as error:
                    return error
                return None

            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = tuple(
                    executor.submit(execute_stale_batch) for _ in range(2)
                )
                rendezvous.wait(timeout=5.0)
                outcomes = tuple(future.result(timeout=5.0) for future in futures)

            self.assertTrue(
                all(
                    isinstance(
                        outcome,
                        ResidentExpertRpcResidentSlotUnavailable,
                    )
                    for outcome in outcomes
                ),
                outcomes,
            )
            self.assertEqual(client.telemetry_snapshot().batch_round_trips, 1)
            self.assertFalse(
                client.is_expert_resident(KEYS[0], CONTENT_IDS[KEYS[0]])
            )
            self.assertEqual(owner.batch_calls, 0)
            self.assertFalse(client.closed)

            survivor = client.execute_batch(
                (
                    OwnerExpertBatchItem(
                        KEYS[1],
                        CONTENT_IDS[KEYS[1]],
                        torch.ones((1, HIDDEN_SIZE), dtype=torch.float32),
                        require_resident=True,
                    ),
                )
            )
            self.assertEqual(tuple(result.key for result in survivor), (KEYS[1],))
            self.assertEqual(client.telemetry_snapshot().batch_round_trips, 2)
            self.assertEqual(owner.batch_calls, 1)
            self.assertFalse(client.closed)
        finally:
            if client is not None:
                client.close()
            stop.set()
            server.close()
            thread.join(timeout=2)
        self.assertFalse(thread.is_alive())

    def test_rpc_residency_loss_replans_atomic_layer_to_local_ram(self) -> None:
        remote_owner = _SwiGLUOwner()
        server, stop, thread, port = _start_threaded_server(remote_owner)
        client: ResidentExpertRpcClient | None = None
        try:
            client = ResidentExpertRpcClient(
                "127.0.0.1",
                port,
                client_node_id="root",
                expected_node_id=OWNER_NODE_ID,
                limits=_limits(),
            )
            remote_owner._resident.remove(KEYS[0])
            weight_bytes = _expert_weight_bytes()
            records = tuple(
                ExpertRecord(key, weight_bytes, CONTENT_IDS[key]) for key in KEYS
            )
            mesh = ResidentExpertMesh(
                coordinator_id="root",
                experts=records,
                nodes=(
                    MeshNodeProfile(
                        node_id="root",
                        resident_vram_budget_bytes=4 * 1024 * 1024,
                        reserved_vram_bytes=0,
                        expert_compute_ms_per_token=5.0,
                        ram_to_device_gbytes_per_second=0.000001,
                        expert_workspace_bytes_per_token=128,
                    ),
                    MeshNodeProfile(
                        node_id=OWNER_NODE_ID,
                        resident_vram_budget_bytes=(
                            weight_bytes * len(KEYS) + 1024 * 1024
                        ),
                        reserved_vram_bytes=0,
                        expert_compute_ms_per_token=0.01,
                        expert_workspace_bytes_per_token=128,
                    ),
                ),
                links=(
                    MeshLinkProfile(
                        "root",
                        OWNER_NODE_ID,
                        0.01,
                        10_000.0,
                    ),
                ),
                local_ram_keys=KEYS,
                local_gpu_keys=(),
                replicas=tuple(
                    ResidentExpertReplica(key, OWNER_NODE_ID, CONTENT_IDS[key])
                    for key in KEYS
                ),
                local_weight_buffer_bytes=weight_bytes,
                activation_bytes_per_token=HIDDEN_SIZE * 4,
            )
            local_owner = InMemoryExpertOwner(
                "root",
                {
                    key: (CONTENT_IDS[key], torch.eye(HIDDEN_SIZE))
                    for key in KEYS
                },
            )
            hidden = torch.tensor(
                [[0.5, -1.0, 0.25, 2.0]],
                dtype=torch.float32,
            )
            routing = AuthoritativeRouting(
                torch.tensor([[0, 1]], dtype=torch.long),
                torch.tensor([[0.6, 0.4]], dtype=torch.float32),
            )

            actual, plan = mesh.execute_layer(
                hidden,
                0,
                routing,
                {"root": local_owner, OWNER_NODE_ID: client},
            )

            expected = 0.6 * hidden + 0.4 * _run_swiglu(KEYS[1], hidden)
            torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
            self.assertEqual(
                {dispatch.key: dispatch.path for dispatch in plan.dispatches},
                {KEYS[0]: "local-ram", KEYS[1]: "remote-resident"},
            )
            self.assertFalse(
                client.is_expert_resident(KEYS[0], CONTENT_IDS[KEYS[0]])
            )
            self.assertEqual(client.telemetry_snapshot().batch_round_trips, 2)
            self.assertEqual(remote_owner.batch_calls, 1)
            self.assertFalse(client.closed)
            mesh.close()
        finally:
            if client is not None:
                client.close()
            stop.set()
            server.close()
            thread.join(timeout=2)
        self.assertFalse(thread.is_alive())

    def test_frame_reports_truncated_canonical_header(self) -> None:
        sender, receiver = socket.socketpair()
        try:
            limits = ResidentExpertRpcLimits(io_timeout_seconds=0.2)
            sender.sendall(_FRAME_PREFIX.pack(_MAGIC, 12, 0) + b'{"schema"')
            sender.shutdown(socket.SHUT_WR)
            with self.assertRaises(ResidentExpertRpcTruncatedFrame):
                _recv_frame(receiver, limits)
        finally:
            sender.close()
            receiver.close()


if __name__ == "__main__":
    unittest.main()
