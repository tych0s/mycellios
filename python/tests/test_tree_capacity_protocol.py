from __future__ import annotations

from collections import deque
import queue
import socket
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from distributed_runtime.model import StageModelSpec
from distributed_runtime.protocol import (
    HEADER,
    MAGIC,
    VERSION,
    Frame,
    FrameType,
    TensorCodec,
    TreePrepareQuote,
    TreePrepareRejection,
    TreePrepareStatus,
    branch_request_payload,
    decode_tree_reservation_nonce,
    decode_tree_prepare,
    encode_tensor,
    encode_tree_prepare_quote,
    recv_frame,
    send_frame,
    tree_prepare_payload,
    tree_reservation_payload,
)
from distributed_runtime.stage import (
    StageProcessConfig,
    TreeReservationBook,
    cancel_tree_reservation,
    collect_compatible_activation_frames,
    commit_tree_reservation,
    consume_tree_reservation_verifies,
    expire_tree_reservation,
    fork_stage_request,
    prepare_tree_capacity,
    project_tree_capacity,
    run_stage_process,
    tree_reservation_defers_frame,
    validate_activation,
    validate_tree_reservation_frame_order,
)


class _QuoteRunner:
    hidden_size = 4
    parameter_bytes = 16
    loader = "fake-tree-quote"
    instances: dict[int, "_QuoteRunner"] = {}

    def __init__(self, spec: StageModelSpec) -> None:
        self.spec = spec
        self.active: dict[int, int] = {}
        self.fork_calls: list[tuple[int, int]] = []
        self.bytes_per_token = {0: 8, 1: 128, 2: 16}.get(spec.layer_start, 8)
        type(self).instances[spec.layer_start] = self

    def begin(self, request_id: int) -> None:
        if request_id in self.active:
            raise ValueError("duplicate BEGIN")
        self.active[request_id] = 0

    def end(self, request_id: int) -> None:
        self.active.pop(request_id, None)

    def sequence_length(self, request_id: int) -> int:
        if request_id not in self.active:
            raise ValueError("inactive request")
        return self.active[request_id]

    def truncate(self, request_id: int, token_count: int) -> None:
        if request_id not in self.active or not 0 <= token_count <= self.active[request_id]:
            raise ValueError("invalid truncate")
        self.active[request_id] = token_count

    def request_cache_bytes(self, request_id: int) -> int:
        return self.sequence_length(request_id) * self.bytes_per_token

    def project_request_cache_bytes(
        self, request_id: int, additional_tokens: int
    ) -> int:
        return (
            self.sequence_length(request_id) + additional_tokens
        ) * self.bytes_per_token

    def fork_request(
        self,
        child_request_id: int,
        parent_request_id: int,
        *,
        max_cache_bytes: int,
    ) -> int:
        copied = self.request_cache_bytes(parent_request_id)
        if copied > max_cache_bytes:
            raise ValueError("fork exceeds preflight")
        self.active[child_request_id] = self.active[parent_request_id]
        self.fork_calls.append((child_request_id, parent_request_id))
        return copied

    def promote_request(self, parent_request_id: int, child_request_id: int) -> None:
        self.active[parent_request_id] = self.active.pop(child_request_id)

    def forward_hidden(
        self,
        request_id: int,
        hidden: torch.Tensor,
        *,
        token_mode: str = "last",
    ) -> tuple[torch.Tensor, int | tuple[int, ...] | None]:
        self.active[request_id] += int(hidden.shape[1])
        if token_mode == "none":
            token = None
        elif token_mode == "all":
            token = tuple(42 for _ in range(int(hidden.shape[1])))
        else:
            token = 42
        return hidden, token


def _last_stage_config(
    *,
    max_kv_bytes: int = 1_024,
    max_branches: int = 4,
    max_tokens: int = 64,
) -> StageProcessConfig:
    return StageProcessConfig(
        spec=StageModelSpec("fake", 0, 1, 1, 1),
        pipeline_id=900,
        listen_host="127.0.0.1",
        listen_port=20_801,
        next_host=None,
        next_port=None,
        next_layer_end=None,
        return_host="127.0.0.1",
        return_port=20_802,
        codec=TensorCodec.FP32,
        one_way_delay_ms=0,
        bandwidth_mbps=0,
        max_speculative_branches=max_branches,
        max_speculative_branch_tokens=max_tokens,
        max_speculative_kv_bytes=max_kv_bytes,
    )


def _prepare_frame(
    parent: int,
    step: int,
    nonce: int,
    paths: tuple[int, ...],
) -> Frame:
    return Frame(
        FrameType.TREE_PREPARE,
        0,
        parent,
        step,
        len(paths),
        0,
        tree_prepare_payload(nonce, paths),
    )


class TreeCapacityPayloadTests(unittest.TestCase):
    def test_commit_and_route_ack_metadata_are_canonical(self) -> None:
        sender, receiver = socket.socketpair()
        try:
            nonce = 2**63 + 11
            payload = tree_reservation_payload(nonce)
            for predecessor_count in (0, (2**16) - 2):
                send_frame(
                    sender,
                    FrameType.TREE_RESERVATION_COMMIT,
                    41,
                    step=5,
                    token_count=predecessor_count,
                    payload=payload,
                )
                commit = recv_frame(receiver)
                self.assertEqual(commit.token_count, predecessor_count)
                self.assertEqual(decode_tree_reservation_nonce(commit), nonce)

            for stage_count in (1, (2**16) - 1):
                send_frame(
                    sender,
                    FrameType.TREE_RESERVATION_COMMIT_RESULT,
                    41,
                    step=5,
                    token_count=stage_count,
                    payload=payload,
                )
                result = recv_frame(receiver)
                self.assertEqual(result.token_count, stage_count)
                self.assertEqual(decode_tree_reservation_nonce(result), nonce)

            with self.assertRaisesRegex(ValueError, "predecessor count"):
                send_frame(
                    sender,
                    FrameType.TREE_RESERVATION_COMMIT,
                    41,
                    token_count=(2**16) - 1,
                    payload=payload,
                )
            for stage_count in (0, 2**16):
                with self.subTest(stage_count=stage_count):
                    with self.assertRaisesRegex(ValueError, "positive uint16"):
                        send_frame(
                            sender,
                            FrameType.TREE_RESERVATION_COMMIT_RESULT,
                            41,
                            token_count=stage_count,
                            payload=payload,
                        )

            for frame_type, token_count in (
                (FrameType.TREE_RESERVATION_COMMIT, 0),
                (FrameType.TREE_RESERVATION_COMMIT_RESULT, 1),
            ):
                with self.subTest(frame_type=frame_type.name, field="flags"):
                    with self.assertRaisesRegex(ValueError, "flags=hidden_size=0"):
                        send_frame(
                            sender,
                            frame_type,
                            41,
                            token_count=token_count,
                            flags=1,
                            payload=payload,
                        )
                with self.subTest(frame_type=frame_type.name, field="hidden_size"):
                    with self.assertRaisesRegex(ValueError, "flags=hidden_size=0"):
                        send_frame(
                            sender,
                            frame_type,
                            41,
                            token_count=token_count,
                            hidden_size=1,
                            payload=payload,
                        )
                with self.subTest(frame_type=frame_type.name, field="payload"):
                    with self.assertRaisesRegex(ValueError, "exactly one uint64"):
                        send_frame(
                            sender,
                            frame_type,
                            41,
                            token_count=token_count,
                            payload=b"short",
                        )
        finally:
            sender.close()
            receiver.close()

    def test_commit_and_route_ack_reject_noncanonical_wire_headers(self) -> None:
        payload = tree_reservation_payload(99)
        cases = (
            (FrameType.TREE_RESERVATION_COMMIT, 1, 0, 0, 8, "flags"),
            (FrameType.TREE_RESERVATION_COMMIT, 0, 0, 1, 8, "hidden_size"),
            (
                FrameType.TREE_RESERVATION_COMMIT,
                0,
                (2**16) - 1,
                0,
                8,
                "predecessor count",
            ),
            (FrameType.TREE_RESERVATION_COMMIT, 0, 0, 0, 7, "one uint64"),
            (
                FrameType.TREE_RESERVATION_COMMIT_RESULT,
                0,
                0,
                0,
                8,
                "positive uint16",
            ),
            (
                FrameType.TREE_RESERVATION_COMMIT_RESULT,
                0,
                2**16,
                0,
                8,
                "positive uint16",
            ),
            (
                FrameType.TREE_RESERVATION_COMMIT_RESULT,
                0,
                1,
                0,
                9,
                "one uint64",
            ),
        )
        for frame_type, flags, token_count, hidden_size, size, pattern in cases:
            with self.subTest(frame_type=frame_type.name, pattern=pattern):
                sender, receiver = socket.socketpair()
                try:
                    sender.sendall(
                        HEADER.pack(
                            MAGIC,
                            VERSION,
                            int(frame_type),
                            flags,
                            41,
                            5,
                            token_count,
                            hidden_size,
                            size,
                        )
                        + payload[:size]
                    )
                    with self.assertRaisesRegex(ValueError, pattern):
                        recv_frame(receiver)
                finally:
                    sender.close()
                    receiver.close()

    def test_quote_and_structured_result_are_canonical_and_bounded(self) -> None:
        sender, receiver = socket.socketpair()
        try:
            payload = tree_prepare_payload(2**63 + 9, (1, 3, 7))
            send_frame(
                sender,
                FrameType.TREE_PREPARE,
                41,
                step=5,
                token_count=3,
                payload=payload,
            )
            quote = decode_tree_prepare(recv_frame(receiver))
            self.assertEqual(quote.nonce, 2**63 + 9)
            self.assertEqual(quote.path_lengths, (1, 3, 7))
            self.assertEqual(quote.status, TreePrepareStatus.READY)

            rejected = TreePrepareQuote(
                nonce=quote.nonce,
                path_lengths=quote.path_lengths,
                status=TreePrepareStatus.REJECT,
                rejection=TreePrepareRejection.KV_BYTES,
                stage_count=3,
                rejecting_layer_start=10,
                required=8_192,
                limit=4_096,
                total_projected_bytes=9_000,
            )
            send_frame(
                sender,
                FrameType.TREE_PREPARE_RESULT,
                41,
                step=5,
                token_count=3,
                payload=encode_tree_prepare_quote(rejected),
            )
            self.assertEqual(decode_tree_prepare(recv_frame(receiver)), rejected)
        finally:
            sender.close()
            receiver.close()

    def test_malformed_quote_fails_closed_before_stage_logic(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            tree_prepare_payload(1, (0,))
        with self.assertRaisesRegex(ValueError, "path count"):
            tree_prepare_payload(1, tuple())
        sender, receiver = socket.socketpair()
        try:
            with self.assertRaisesRegex(ValueError, "expected"):
                send_frame(
                    sender,
                    FrameType.TREE_PREPARE,
                    1,
                    token_count=2,
                    payload=tree_prepare_payload(1, (1,)),
                )
        finally:
            sender.close()
            receiver.close()
        malformed = bytearray(tree_prepare_payload(1, (1,)))
        malformed[8] = 99  # status byte after uint64 nonce
        with self.assertRaisesRegex(ValueError, "unknown tree prepare status"):
            decode_tree_prepare(
                Frame(FrameType.TREE_PREPARE, 0, 1, 0, 1, 0, malformed)
            )


class TreeCapacityStageTests(unittest.TestCase):
    def setUp(self) -> None:
        _QuoteRunner.instances = {}

    def test_projection_counts_live_children_and_every_full_leaf(self) -> None:
        config = _last_stage_config(max_kv_bytes=10_000)
        runner = _QuoteRunner(config.spec)
        runner.begin(1)
        runner.active[1] = 2
        runner.begin(2)
        runner.active[2] = 3
        projection = project_tree_capacity(
            (1, 3),
            parent_request_id=1,
            config=config,
            runner=runner,
            branch_parents={2: 1},
        )
        # Existing child: 3*8. New full leaves: (2+1+1)*8 and (2+1+3)*8.
        self.assertEqual(projection.current_branch_bytes, 24)
        self.assertEqual(projection.projected_kv_bytes, 24 + 32 + 48)
        self.assertEqual(projection.current_branch_count, 1)

    def test_ready_quote_commit_and_fork_are_bound_to_shape(self) -> None:
        config = _last_stage_config(max_kv_bytes=10_000)
        runner = _QuoteRunner(config.spec)
        runner.begin(11)
        runner.active[11] = 2
        metrics = {11: {"frames": 3, "compute_ms": 0, "bytes_out": 0, "tokens": 2}}
        book = TreeReservationBook()
        stage_sock, peer = socket.socketpair()
        try:
            accumulated = prepare_tree_capacity(
                _prepare_frame(11, 3, 700, (2,)),
                config=config,
                runner=runner,
                request_metrics=metrics,
                branch_parents={},
                reservations=book,
                downstream=stage_sock,
                return_socket=None,
            )
            self.assertEqual(accumulated.status, TreePrepareStatus.READY)
            self.assertIsNotNone(book.active)
            self.assertEqual(recv_frame(peer).frame_type, FrameType.TREE_PREPARE)

            commit = Frame(
                FrameType.TREE_RESERVATION_COMMIT,
                0,
                11,
                3,
                0,
                0,
                tree_reservation_payload(700),
            )
            commit_tree_reservation(
                commit,
                config=config,
                runner=runner,
                request_metrics=metrics,
                branch_parents={},
                reservations=book,
                downstream=stage_sock,
            )
            self.assertTrue(book.active and book.active.committed)
            self.assertEqual(
                recv_frame(peer).frame_type, FrameType.TREE_RESERVATION_COMMIT
            )

            branches: dict[int, int] = {}
            shapes = {}
            fork_stage_request(
                Frame(
                    FrameType.FORK,
                    0,
                    12,
                    0,
                    0,
                    0,
                    branch_request_payload(11),
                ),
                config,
                runner,
                metrics,
                branches,
                stage_sock,
                book,
                shapes,
            )
            self.assertIsNotNone(book.active)
            self.assertTrue(book.active and book.active.committed)
            self.assertEqual(shapes[12].expected_tokens, 3)
            self.assertEqual(recv_frame(peer).frame_type, FrameType.FORK)
            valid = Frame(FrameType.VERIFY, 0, 12, 3, 3, 4, b"")
            self.assertFalse(tree_reservation_defers_frame(valid, book, branches))
            validate_tree_reservation_frame_order(valid, book, branches)
            self.assertTrue(
                tree_reservation_defers_frame(
                    Frame(FrameType.ACTIVATION, 0, 99, 0, 1, 4, b""),
                    book,
                    branches,
                )
            )
            # Payload decoding belongs to process_activation_frames; this helper
            # proves the committed quote is bound to the VERIFY shape first.
            validate_activation(valid, config, runner, metrics, branches, shapes)
            with self.assertRaisesRegex(ValueError, "shape mismatch"):
                validate_activation(
                    Frame(FrameType.VERIFY, 0, 12, 3, 2, 4, b""),
                    config,
                    runner,
                    metrics,
                    branches,
                    shapes,
                )
            self.assertTrue(consume_tree_reservation_verifies((valid,), book))
            self.assertIsNone(book.active)
            with self.assertRaisesRegex(ValueError, "after FORK consumption"):
                cancel_tree_reservation(
                    Frame(
                        FrameType.TREE_RESERVATION_CANCEL,
                        0,
                        11,
                        3,
                        0,
                        0,
                        tree_reservation_payload(700),
                    ),
                    reservations=book,
                    downstream=None,
                )
        finally:
            stage_sock.close()
            peer.close()

    def test_unquoted_fork_is_rejected_on_the_v6_stage_path(self) -> None:
        config = _last_stage_config(max_kv_bytes=10_000)
        runner = _QuoteRunner(config.spec)
        runner.begin(11)
        runner.active[11] = 2
        with self.assertRaisesRegex(ValueError, "active committed"):
            fork_stage_request(
                Frame(
                    FrameType.FORK,
                    0,
                    12,
                    0,
                    0,
                    0,
                    branch_request_payload(11),
                ),
                config,
                runner,
                {11: {"frames": 3}},
                {},
                None,
                TreeReservationBook(),
                {},
            )
        self.assertEqual(runner.active, {11: 2})
        self.assertEqual(runner.fork_calls, [])

    def test_commit_drift_is_fatal_not_a_capacity_reject(self) -> None:
        config = _last_stage_config(max_kv_bytes=10_000)
        runner = _QuoteRunner(config.spec)
        runner.begin(11)
        runner.active[11] = 2
        metrics = {11: {"frames": 3}}
        book = TreeReservationBook()
        stage_sock, peer = socket.socketpair()
        try:
            prepare_tree_capacity(
                _prepare_frame(11, 3, 701, (1,)),
                config=config,
                runner=runner,
                request_metrics=metrics,
                branch_parents={},
                reservations=book,
                downstream=stage_sock,
                return_socket=None,
            )
            recv_frame(peer)
            runner.active[11] += 1
            with self.assertRaisesRegex(RuntimeError, "snapshot changed"):
                commit_tree_reservation(
                    Frame(
                        FrameType.TREE_RESERVATION_COMMIT,
                        0,
                        11,
                        3,
                        0,
                        0,
                        tree_reservation_payload(701),
                    ),
                    config=config,
                    runner=runner,
                    request_metrics=metrics,
                    branch_parents={},
                    reservations=book,
                    downstream=None,
                )
        finally:
            stage_sock.close()
            peer.close()

    def test_quote_defers_every_runtime_kv_mutation_until_consumed(self) -> None:
        config = _last_stage_config(max_kv_bytes=10_000)
        runner = _QuoteRunner(config.spec)
        runner.begin(11)
        runner.active[11] = 2
        book = TreeReservationBook()
        stage_sock, peer = socket.socketpair()
        try:
            prepare_tree_capacity(
                _prepare_frame(11, 3, 705, (1,)),
                config=config,
                runner=runner,
                request_metrics={11: {"frames": 3}},
                branch_parents={},
                reservations=book,
                downstream=stage_sock,
                return_socket=None,
            )
            recv_frame(peer)
            for frame_type in (
                FrameType.BEGIN,
                FrameType.ACTIVATION,
                FrameType.PREFILL,
                FrameType.VERIFY,
                FrameType.TRUNCATE,
                FrameType.END,
                FrameType.CANCEL,
            ):
                candidate = Frame(frame_type, 0, 99, 0, 0, 0, b"")
                self.assertTrue(
                    tree_reservation_defers_frame(candidate, book, {})
                )
                with self.assertRaisesRegex(ValueError, "cannot overtake"):
                    validate_tree_reservation_frame_order(candidate, book, {})
            with self.assertRaisesRegex(ValueError, "cannot overtake"):
                validate_tree_reservation_frame_order(
                    Frame(FrameType.ACTIVATION, 0, 11, 3, 1, 4, b""),
                    book,
                    {},
                )
            with self.assertRaisesRegex(ValueError, "cannot overtake"):
                validate_tree_reservation_frame_order(
                    Frame(FrameType.ACTIVATION, 0, 22, 3, 1, 4, b""),
                    book,
                    {22: 11},
                )
            with self.assertRaisesRegex(ValueError, "cannot overtake"):
                validate_tree_reservation_frame_order(
                    _prepare_frame(99, 0, 706, (1,)), book, {}
                )
        finally:
            stage_sock.close()
            peer.close()

    def test_batch_read_ahead_cannot_bypass_reserved_parent_barrier(self) -> None:
        config = _last_stage_config(max_kv_bytes=10_000)
        runner = _QuoteRunner(config.spec)
        runner.max_active_requests = 8
        runner.physical_batch_key = lambda *_args, **_kwargs: ("same",)
        runner.forward_hidden_batch = lambda *_args, **_kwargs: ()
        runner.begin(11)
        runner.begin(99)
        runner.active[11] = 2
        runner.active[99] = 2
        metrics = {11: {"frames": 3}, 99: {"frames": 3}}
        book = TreeReservationBook()
        quote_sock, quote_peer = socket.socketpair()
        sender, upstream = socket.socketpair()
        hidden = torch.arange(4, dtype=torch.float32).reshape(1, 1, 4)
        payload = encode_tensor(hidden, TensorCodec.FP32)
        first = Frame(
            FrameType.ACTIVATION,
            int(TensorCodec.FP32),
            99,
            3,
            1,
            4,
            payload,
        )
        try:
            prepare_tree_capacity(
                _prepare_frame(11, 3, 709, (1,)),
                config=config,
                runner=runner,
                request_metrics=metrics,
                branch_parents={},
                reservations=book,
                downstream=quote_sock,
                return_socket=None,
            )
            recv_frame(quote_peer)
            send_frame(
                sender,
                FrameType.ACTIVATION,
                11,
                step=3,
                token_count=1,
                hidden_size=4,
                flags=int(TensorCodec.FP32),
                payload=payload,
            )
            with self.assertRaisesRegex(ValueError, "cannot overtake"):
                collect_compatible_activation_frames(
                    first,
                    upstream=upstream,
                    pending_frames=deque(),
                    config=config,
                    runner=runner,
                    request_metrics=metrics,
                    branch_parents={},
                    tree_leaf_shapes={},
                    tree_reservations=book,
                    downstream=None,
                )
            self.assertEqual(runner.active, {11: 2, 99: 2})
        finally:
            quote_sock.close()
            quote_peer.close()
            sender.close()
            upstream.close()

    def test_tree_batch_read_ahead_queues_unrelated_mutation_until_verify_release(
        self,
    ) -> None:
        config = _last_stage_config(max_kv_bytes=10_000)
        runner = _QuoteRunner(config.spec)
        runner.max_active_requests = 8
        runner.executor_manifest = SimpleNamespace(
            features=(
                "exact-tree-verify-batching",
                "bounded-tree-verify-workspace",
            )
        )
        runner.physical_batch_key = lambda *_args, **_kwargs: ("same",)
        runner.forward_hidden_batch = lambda *_args, **_kwargs: ()
        runner.begin(11)
        runner.begin(99)
        runner.active[11] = 2
        runner.active[99] = 2
        metrics = {
            11: {"frames": 3, "compute_ms": 0, "bytes_out": 0, "tokens": 2},
            99: {"frames": 3, "compute_ms": 0, "bytes_out": 0, "tokens": 2},
        }
        book = TreeReservationBook()
        quote_sock, quote_peer = socket.socketpair()
        sender, upstream = socket.socketpair()
        try:
            prepare_tree_capacity(
                _prepare_frame(11, 3, 711, (1,)),
                config=config,
                runner=runner,
                request_metrics=metrics,
                branch_parents={},
                reservations=book,
                downstream=quote_sock,
                return_socket=None,
            )
            recv_frame(quote_peer)
            commit_tree_reservation(
                Frame(
                    FrameType.TREE_RESERVATION_COMMIT,
                    0,
                    11,
                    3,
                    0,
                    0,
                    tree_reservation_payload(711),
                ),
                config=config,
                runner=runner,
                request_metrics=metrics,
                branch_parents={},
                reservations=book,
                downstream=quote_sock,
            )
            recv_frame(quote_peer)
            branches: dict[int, int] = {}
            shapes = {}
            fork_stage_request(
                Frame(
                    FrameType.FORK,
                    0,
                    12,
                    0,
                    0,
                    0,
                    branch_request_payload(11),
                ),
                config,
                runner,
                metrics,
                branches,
                quote_sock,
                book,
                shapes,
            )
            recv_frame(quote_peer)
            tree_hidden = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)
            tree_payload = encode_tensor(tree_hidden, TensorCodec.FP32)
            first = Frame(
                FrameType.VERIFY,
                int(TensorCodec.FP32),
                12,
                3,
                2,
                4,
                tree_payload,
            )
            validate_activation(first, config, runner, metrics, branches, shapes)
            ordinary_hidden = torch.arange(4, dtype=torch.float32).reshape(1, 1, 4)
            send_frame(
                sender,
                FrameType.ACTIVATION,
                99,
                step=3,
                token_count=1,
                hidden_size=4,
                flags=int(TensorCodec.FP32),
                payload=encode_tensor(ordinary_hidden, TensorCodec.FP32),
            )
            pending = deque()
            self.assertEqual(
                collect_compatible_activation_frames(
                    first,
                    upstream=upstream,
                    pending_frames=pending,
                    config=config,
                    runner=runner,
                    request_metrics=metrics,
                    branch_parents=branches,
                    tree_leaf_shapes=shapes,
                    tree_reservations=book,
                    downstream=None,
                ),
                (first,),
            )
            self.assertEqual(len(pending), 1)
            self.assertEqual(pending[0].request_id, 99)
            self.assertIsNotNone(book.active)
        finally:
            quote_sock.close()
            quote_peer.close()
            sender.close()
            upstream.close()

    def test_expired_quote_is_released_but_late_commit_is_fatal(self) -> None:
        config = _last_stage_config(max_kv_bytes=10_000)
        runner = _QuoteRunner(config.spec)
        runner.begin(11)
        runner.active[11] = 2
        book = TreeReservationBook()
        stage_sock, peer = socket.socketpair()
        try:
            prepare_tree_capacity(
                _prepare_frame(11, 3, 707, (1,)),
                config=config,
                runner=runner,
                request_metrics={11: {"frames": 3}},
                branch_parents={},
                reservations=book,
                downstream=stage_sock,
                return_socket=None,
            )
            recv_frame(peer)
            self.assertIsNotNone(book.active)
            assert book.active is not None
            self.assertTrue(
                expire_tree_reservation(book, now=book.active.deadline_at + 0.001)
            )
            with self.assertRaisesRegex(ValueError, "no active quote"):
                commit_tree_reservation(
                    Frame(
                        FrameType.TREE_RESERVATION_COMMIT,
                        0,
                        11,
                        3,
                        0,
                        0,
                        tree_reservation_payload(707),
                    ),
                    config=config,
                    runner=runner,
                    request_metrics={11: {"frames": 3}},
                    branch_parents={},
                    reservations=book,
                    downstream=None,
                )
        finally:
            stage_sock.close()
            peer.close()

    def test_expired_reservation_after_fork_is_route_fatal_and_stays_barred(self) -> None:
        config = _last_stage_config(max_kv_bytes=10_000)
        runner = _QuoteRunner(config.spec)
        runner.begin(11)
        runner.active[11] = 2
        metrics = {
            11: {
                "frames": 3,
                "compute_ms": 0,
                "bytes_out": 0,
                "tokens": 2,
            }
        }
        book = TreeReservationBook()
        stage_sock, peer = socket.socketpair()
        try:
            prepare_tree_capacity(
                _prepare_frame(11, 3, 710, (1,)),
                config=config,
                runner=runner,
                request_metrics=metrics,
                branch_parents={},
                reservations=book,
                downstream=stage_sock,
                return_socket=None,
            )
            recv_frame(peer)
            commit_tree_reservation(
                Frame(
                    FrameType.TREE_RESERVATION_COMMIT,
                    0,
                    11,
                    3,
                    0,
                    0,
                    tree_reservation_payload(710),
                ),
                config=config,
                runner=runner,
                request_metrics=metrics,
                branch_parents={},
                reservations=book,
                downstream=stage_sock,
            )
            recv_frame(peer)
            branches: dict[int, int] = {}
            fork_stage_request(
                Frame(
                    FrameType.FORK,
                    0,
                    12,
                    0,
                    0,
                    0,
                    branch_request_payload(11),
                ),
                config,
                runner,
                metrics,
                branches,
                stage_sock,
                book,
                {},
            )
            recv_frame(peer)
            assert book.active is not None
            with self.assertRaisesRegex(TimeoutError, "after speculative KV mutation"):
                expire_tree_reservation(book, now=book.active.deadline_at + 0.001)
            self.assertIsNotNone(book.active)
        finally:
            stage_sock.close()
            peer.close()

    def test_rejected_quote_cancel_is_idempotent_but_other_nonce_is_fatal(self) -> None:
        config = _last_stage_config(max_kv_bytes=8)
        runner = _QuoteRunner(config.spec)
        runner.begin(11)
        runner.active[11] = 2
        metrics = {11: {"frames": 3}}
        book = TreeReservationBook()
        stage_sock, peer = socket.socketpair()
        try:
            result = prepare_tree_capacity(
                _prepare_frame(11, 3, 702, (1,)),
                config=config,
                runner=runner,
                request_metrics=metrics,
                branch_parents={},
                reservations=book,
                downstream=stage_sock,
                return_socket=None,
            )
            recv_frame(peer)
            self.assertEqual(result.status, TreePrepareStatus.REJECT)
            self.assertEqual(result.rejection, TreePrepareRejection.KV_BYTES)
            self.assertIsNone(book.active)
            cancel = Frame(
                FrameType.TREE_RESERVATION_CANCEL,
                0,
                11,
                3,
                0,
                0,
                tree_reservation_payload(702),
            )
            cancel_tree_reservation(cancel, reservations=book, downstream=None)
            cancel_tree_reservation(cancel, reservations=book, downstream=None)
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                cancel_tree_reservation(
                    Frame(
                        FrameType.TREE_RESERVATION_CANCEL,
                        0,
                        11,
                        3,
                        0,
                        0,
                        tree_reservation_payload(999),
                    ),
                    reservations=book,
                    downstream=None,
                )
        finally:
            stage_sock.close()
            peer.close()

    def test_unknown_parent_and_false_step_are_protocol_errors(self) -> None:
        config = _last_stage_config(max_kv_bytes=10_000)
        runner = _QuoteRunner(config.spec)
        runner.begin(11)
        runner.active[11] = 2
        with self.assertRaisesRegex(ValueError, "not active"):
            prepare_tree_capacity(
                _prepare_frame(99, 0, 800, (1,)),
                config=config,
                runner=runner,
                request_metrics={11: {"frames": 3}},
                branch_parents={},
                reservations=TreeReservationBook(),
                downstream=None,
                return_socket=None,
            )
        with self.assertRaisesRegex(ValueError, "step mismatch"):
            prepare_tree_capacity(
                _prepare_frame(11, 2, 801, (1,)),
                config=config,
                runner=runner,
                request_metrics={11: {"frames": 3}},
                branch_parents={},
                reservations=TreeReservationBook(),
                downstream=None,
                return_socket=None,
            )

    def test_stage_emits_error_not_reject_for_invalid_prepare_contracts(self) -> None:
        for case in ("unknown-parent", "false-step", "malformed"):
            with self.subTest(case=case):
                self._assert_fatal_prepare_case(case)

    def test_three_stage_route_rejects_heavy_stage_before_any_fork_then_recovers(self) -> None:
        return_listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        return_listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return_listener.bind(("127.0.0.1", 0))
        return_listener.listen(1)
        return_listener.settimeout(5)
        return_port = int(return_listener.getsockname()[1])
        ports = tuple(self._free_port() for _ in range(3))
        while len(set((*ports, return_port))) != 4:
            ports = tuple(self._free_port() for _ in range(3))
        configs = tuple(
            StageProcessConfig(
                spec=StageModelSpec("fake", index, index + 1, 3, 1),
                pipeline_id=901,
                listen_host="127.0.0.1",
                listen_port=ports[index],
                next_host="127.0.0.1" if index < 2 else None,
                next_port=ports[index + 1] if index < 2 else None,
                next_layer_end=index + 2 if index < 2 else None,
                return_host="127.0.0.1",
                return_port=return_port,
                codec=TensorCodec.FP32,
                one_way_delay_ms=0,
                bandwidth_mbps=0,
                connect_timeout_seconds=5,
                max_speculative_branches=4,
                max_speculative_branch_tokens=64,
                max_speculative_kv_bytes=512,
            )
            for index in range(3)
        )
        ready = tuple(threading.Event() for _ in configs)
        errors: list[tuple[int, BaseException]] = []
        workers = []

        def run(index: int) -> None:
            try:
                run_stage_process(configs[index], ready[index], queue.Queue())
            except BaseException as error:
                errors.append((index, error))

        _QuoteRunner.instances = {}
        upstream: socket.socket | None = None
        direct_return: socket.socket | None = None
        try:
            with patch("distributed_runtime.stage.StageRunner", _QuoteRunner):
                for index in range(3):
                    worker = threading.Thread(target=run, args=(index,), daemon=True)
                    worker.start()
                    workers.append(worker)
                self.assertTrue(all(event.wait(2) for event in ready))
                upstream = socket.create_connection(("127.0.0.1", ports[0]), timeout=5)
                upstream.settimeout(5)
                send_frame(
                    upstream,
                    FrameType.HELLO,
                    901,
                    step=0,
                    token_count=1,
                    hidden_size=4,
                    flags=int(TensorCodec.FP32),
                )
                direct_return, _ = return_listener.accept()
                direct_return.settimeout(5)
                self.assertEqual(recv_frame(upstream).frame_type, FrameType.READY)

                parent = 77
                send_frame(upstream, FrameType.BEGIN, parent)
                hidden = torch.arange(4, dtype=torch.float32).reshape(1, 1, 4)
                send_frame(
                    upstream,
                    FrameType.ACTIVATION,
                    parent,
                    step=0,
                    token_count=1,
                    hidden_size=4,
                    flags=int(TensorCodec.FP32),
                    payload=encode_tensor(hidden, TensorCodec.FP32),
                )
                self.assertEqual(recv_frame(direct_return).frame_type, FrameType.TOKEN)

                send_frame(
                    upstream,
                    FrameType.TREE_PREPARE,
                    parent,
                    step=1,
                    token_count=2,
                    payload=tree_prepare_payload(1_001, (1, 2)),
                )
                rejected = decode_tree_prepare(recv_frame(direct_return))
                self.assertEqual(rejected.status, TreePrepareStatus.REJECT)
                self.assertEqual(rejected.rejection, TreePrepareRejection.KV_BYTES)
                self.assertEqual(rejected.stage_count, 3)
                self.assertEqual(rejected.rejecting_layer_start, 1)
                self.assertEqual((rejected.required, rejected.limit), (896, 512))
                self.assertEqual(rejected.total_projected_bytes, 1_064)
                self.assertEqual(
                    sum(len(instance.fork_calls) for instance in _QuoteRunner.instances.values()),
                    0,
                )

                # Ordered CANCEL clears the READY reservation held by stage 0.
                # A smaller quote immediately behind it proves all three stages
                # were cleaned without an ACK or a hidden KV allocation.
                send_frame(
                    upstream,
                    FrameType.TREE_RESERVATION_CANCEL,
                    parent,
                    step=1,
                    payload=tree_reservation_payload(1_001),
                )
                send_frame(
                    upstream,
                    FrameType.TREE_PREPARE,
                    parent,
                    step=1,
                    token_count=1,
                    payload=tree_prepare_payload(1_002, (1,)),
                )
                accepted = decode_tree_prepare(recv_frame(direct_return))
                self.assertEqual(accepted.status, TreePrepareStatus.READY)
                self.assertEqual(accepted.stage_count, 3)
                self.assertEqual(accepted.total_projected_bytes, 456)

                # Every cache mutation is held behind the quote: allowing an
                # unrelated chat to grow could consume the very allocator
                # capacity that the route just promised to this tree.
                other = 88
                send_frame(upstream, FrameType.BEGIN, other)
                send_frame(
                    upstream,
                    FrameType.ACTIVATION,
                    other,
                    step=0,
                    token_count=1,
                    hidden_size=4,
                    flags=int(TensorCodec.FP32),
                    payload=encode_tensor(hidden, TensorCodec.FP32),
                )
                direct_return.settimeout(0.05)
                with self.assertRaises(socket.timeout):
                    recv_frame(direct_return)
                direct_return.settimeout(5)

                # COMMIT alone still cannot expose a partial transaction. The
                # last stage returns one ACK containing the number of stages
                # that revalidated it, and no FORK has happened yet.
                send_frame(
                    upstream,
                    FrameType.TREE_RESERVATION_COMMIT,
                    parent,
                    step=1,
                    token_count=0,
                    payload=tree_reservation_payload(1_002),
                )
                commit_result = recv_frame(direct_return)
                self.assertEqual(
                    (
                        commit_result.frame_type,
                        commit_result.request_id,
                        commit_result.token_count,
                    ),
                    (FrameType.TREE_RESERVATION_COMMIT_RESULT, parent, 3),
                )
                self.assertEqual(
                    sum(
                        len(instance.fork_calls)
                        for instance in _QuoteRunner.instances.values()
                    ),
                    0,
                )

                # FORK alone cannot release the physical capacity guarantee:
                # the unrelated chat remains held until the quoted VERIFY has
                # materialised all of its promised KV growth on every stage.
                child = 78
                send_frame(
                    upstream,
                    FrameType.FORK,
                    child,
                    payload=branch_request_payload(parent),
                )
                direct_return.settimeout(0.05)
                with self.assertRaises(socket.timeout):
                    recv_frame(direct_return)
                direct_return.settimeout(5)
                tree_hidden = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)
                send_frame(
                    upstream,
                    FrameType.VERIFY,
                    child,
                    step=1,
                    token_count=2,
                    hidden_size=4,
                    flags=int(TensorCodec.FP32),
                    payload=encode_tensor(tree_hidden, TensorCodec.FP32),
                )
                tree_result = recv_frame(direct_return)
                self.assertEqual(
                    (
                        tree_result.frame_type,
                        tree_result.request_id,
                        tree_result.token_count,
                    ),
                    (FrameType.VERIFY_RESULT, child, 2),
                )
                other_result = recv_frame(direct_return)
                self.assertEqual(
                    (other_result.frame_type, other_result.request_id),
                    (FrameType.TOKEN, other),
                )
                send_frame(upstream, FrameType.END, other)
                send_frame(upstream, FrameType.END, child)
                send_frame(upstream, FrameType.END, parent)
                send_frame(upstream, FrameType.SHUTDOWN, 901)
                for worker in workers:
                    worker.join(5)
                self.assertTrue(all(not worker.is_alive() for worker in workers))
                self.assertEqual(errors, [])
        finally:
            if upstream is not None:
                upstream.close()
            if direct_return is not None:
                direct_return.close()
            return_listener.close()

    @staticmethod
    def _free_port() -> int:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])
        finally:
            sock.close()

    def _assert_fatal_prepare_case(self, case: str) -> None:
        return_listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        return_listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return_listener.bind(("127.0.0.1", 0))
        return_listener.listen(1)
        return_listener.settimeout(5)
        return_port = int(return_listener.getsockname()[1])
        stage_port = self._free_port()
        while stage_port == return_port:
            stage_port = self._free_port()
        base = _last_stage_config(max_kv_bytes=10_000)
        config = StageProcessConfig(
            **{
                **base.__dict__,
                "pipeline_id": 902,
                "listen_port": stage_port,
                "return_port": return_port,
                "connect_timeout_seconds": 5,
            }
        )
        ready = threading.Event()
        errors: list[BaseException] = []

        def run() -> None:
            try:
                run_stage_process(config, ready, queue.Queue())
            except BaseException as error:
                errors.append(error)

        upstream: socket.socket | None = None
        direct_return: socket.socket | None = None
        try:
            with patch("distributed_runtime.stage.StageRunner", _QuoteRunner):
                worker = threading.Thread(target=run, daemon=True)
                worker.start()
                self.assertTrue(ready.wait(2))
                upstream = socket.create_connection(
                    ("127.0.0.1", stage_port), timeout=5
                )
                upstream.settimeout(5)
                send_frame(
                    upstream,
                    FrameType.HELLO,
                    902,
                    step=0,
                    token_count=1,
                    hidden_size=4,
                    flags=int(TensorCodec.FP32),
                )
                direct_return, _ = return_listener.accept()
                direct_return.settimeout(5)
                self.assertEqual(recv_frame(upstream).frame_type, FrameType.READY)

                parent = 11
                if case != "unknown-parent":
                    send_frame(upstream, FrameType.BEGIN, parent)
                if case == "false-step":
                    hidden = torch.zeros((1, 1, 4), dtype=torch.float32)
                    send_frame(
                        upstream,
                        FrameType.ACTIVATION,
                        parent,
                        step=0,
                        token_count=1,
                        hidden_size=4,
                        flags=int(TensorCodec.FP32),
                        payload=encode_tensor(hidden, TensorCodec.FP32),
                    )
                    self.assertEqual(
                        recv_frame(direct_return).frame_type, FrameType.TOKEN
                    )

                request_id = 99 if case == "unknown-parent" else parent
                step = 0 if case != "false-step" else 0
                payload = bytearray(tree_prepare_payload(2_000, (1,)))
                if case == "malformed":
                    payload[8] = 99
                send_frame(
                    upstream,
                    FrameType.TREE_PREPARE,
                    request_id,
                    step=step,
                    token_count=1,
                    payload=payload,
                )
                self.assertEqual(recv_frame(upstream).frame_type, FrameType.ERROR)
                self.assertEqual(
                    recv_frame(direct_return).frame_type, FrameType.ERROR
                )
                worker.join(5)
                self.assertFalse(worker.is_alive())
                self.assertEqual(len(errors), 1)
                self.assertIsInstance(errors[0], ValueError)
        finally:
            if upstream is not None:
                upstream.close()
            if direct_return is not None:
                direct_return.close()
            return_listener.close()


if __name__ == "__main__":
    unittest.main()
