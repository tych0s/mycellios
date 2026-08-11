from __future__ import annotations

from collections import deque
import json
import math
import socket
import struct
import queue
import random
import threading
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import zlib

import torch

from distributed_runtime.protocol import (
    HEADER,
    HEADER_BYTES,
    MAGIC,
    MAX_PAYLOAD_BYTES,
    STATIC_WAVE_ARTIFACT_DIGEST,
    STATIC_WAVE_STRATEGY_DIGEST,
    VERSION,
    Frame,
    FrameType,
    LinkEmulator,
    TensorCodec,
    bind_socket_deployment_generation,
    branch_request_payload,
    decode_branch_request_id,
    decode_receipt_ack,
    decode_receipt_envelope,
    decode_sampling_logits,
    decode_sampling_rng_checkpoint,
    decode_tensor,
    decode_token,
    decode_verify_result,
    encode_tensor,
    encode_tensor_payload,
    recv_exact,
    recv_frame,
    receipt_ack_payload,
    receipt_envelope_payload,
    route_identity_digest,
    send_frame,
    sampling_logits_payload,
    sampling_rng_checkpoint_payload,
    token_payload,
    wave_identity_digest,
    verify_result_payload,
)
from distributed_runtime.model import StageModelSpec
from distributed_runtime.stage_cli import JsonMetricSink
from distributed_runtime.stage import (
    StageProcessConfig,
    collect_compatible_activation_frames,
    fork_stage_request,
    forward_shutdown_and_wait,
    monitor_downstream_control,
    promote_stage_request,
    route_stage_result,
    run_stage_process,
    validate_activation,
    validate_hello,
    validate_speculative_kv_preflight,
    validate_speculative_runner,
    validate_stage_config,
)


class ProtocolFramingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sender, self.receiver = socket.socketpair()

    def tearDown(self) -> None:
        self.sender.close()
        self.receiver.close()

    def test_header_is_fixed_112_bytes(self) -> None:
        self.assertEqual(HEADER_BYTES, 112)

    def test_control_frame_round_trip(self) -> None:
        sent = send_frame(
            self.sender,
            FrameType.HELLO,
            2**63 + 7,
            step=10,
            token_count=20,
            hidden_size=576,
            flags=int(TensorCodec.FP16),
        )
        frame = recv_frame(self.receiver)
        self.assertEqual(sent, HEADER_BYTES)
        self.assertEqual(frame.frame_type, FrameType.HELLO)
        self.assertEqual(frame.request_id, 2**63 + 7)
        self.assertEqual(frame.step, 10)
        self.assertEqual(frame.token_count, 20)
        self.assertEqual(frame.hidden_size, 576)
        self.assertEqual(frame.flags, int(TensorCodec.FP16))
        self.assertEqual(frame.deployment_generation, 0)
        self.assertEqual(frame.payload, b"")

    def test_signed_receipt_envelope_and_ack_round_trip(self) -> None:
        envelope = json.dumps(
            {
                "deploymentGeneration": 7,
                "keyId": "receipt-key-1",
                "receipt": {"receiptId": "sha256:" + "a" * 64},
                "routeDigest": "1" * 32,
                "schema": "mycellios-execution-receipt-envelope/1",
                "signature": "signed-upstream",
                "waveArtifactDigest": "3" * 32,
                "waveStrategyDigest": "2" * 32,
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        send_frame(
            self.sender,
            FrameType.RECEIPT_ENVELOPE,
            51,
            payload=receipt_envelope_payload(envelope),
        )
        received = recv_frame(self.receiver)
        self.assertEqual(decode_receipt_envelope(received), envelope)

        acknowledgement = receipt_ack_payload(envelope)
        send_frame(
            self.receiver,
            FrameType.RECEIPT_ACK,
            51,
            payload=acknowledgement,
        )
        self.assertEqual(
            decode_receipt_ack(recv_frame(self.sender)), acknowledgement
        )

    def test_sampling_logits_and_rng_checkpoint_round_trip(self) -> None:
        payload, token_count, vocabulary_size = sampling_logits_payload(
            ((1.25, -0.5, 3.0), (0.0, 2.0, -4.0))
        )
        send_frame(
            self.sender,
            FrameType.SAMPLING_VERIFY_RESULT,
            61,
            step=7,
            token_count=token_count,
            hidden_size=vocabulary_size,
            payload=payload,
        )
        frame = recv_frame(self.receiver)
        self.assertEqual(
            decode_sampling_logits(frame),
            ((1.25, -0.5, 3.0), (0.0, 2.0, -4.0)),
        )

        checkpoint = sampling_rng_checkpoint_payload(bytes(range(32)), 2**63 + 9)
        send_frame(
            self.receiver,
            FrameType.SAMPLING_RNG_CHECKPOINT,
            61,
            payload=checkpoint,
        )
        self.assertEqual(
            decode_sampling_rng_checkpoint(recv_frame(self.sender)),
            (bytes(range(32)), 2**63 + 9),
        )

    def test_sampling_frames_reject_malformed_or_nonfinite_payloads(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            sampling_logits_payload(((0.0, math.nan),))
        with self.assertRaisesRegex(ValueError, "different vocabulary"):
            sampling_logits_payload(((0.0, 1.0), (2.0,)))
        with self.assertRaisesRegex(ValueError, "one float32"):
            send_frame(
                self.sender,
                FrameType.SAMPLING_VERIFY_RESULT,
                62,
                token_count=2,
                hidden_size=3,
                payload=b"\0" * 4,
            )
        with self.assertRaisesRegex(ValueError, "exactly 32"):
            sampling_rng_checkpoint_payload(b"short", 0)

    def test_receipt_envelope_rejects_noncanonical_and_oversized_payloads(self) -> None:
        with self.assertRaisesRegex(ValueError, "not canonical"):
            receipt_envelope_payload(
                '{"schema": "mycellios-execution-receipt-envelope/1"}'
            )
        with self.assertRaisesRegex(ValueError, "duplicate key"):
            receipt_envelope_payload(
                '{"schema":"mycellios-execution-receipt-envelope/1",'
                '"schema":"mycellios-execution-receipt-envelope/1"}'
            )
        with self.assertRaisesRegex(ValueError, "between 1 and"):
            send_frame(
                self.sender,
                FrameType.RECEIPT_ENVELOPE,
                51,
                payload=b"x" * (1024 * 1024 + 1),
            )

    def test_stateful_mixed_wave_control_tail_and_receipt_fuzz(self) -> None:
        bind_args = {
            "wave_strategy_id": "ngram-certified-v1",
            "wave_artifact_identity": "sha256:" + "a" * 64,
        }
        bind_socket_deployment_generation(
            self.sender, 31, "route-fuzz", **bind_args
        )
        bind_socket_deployment_generation(
            self.receiver, 31, "route-fuzz", **bind_args
        )
        envelope = receipt_envelope_payload(
            json.dumps(
                {
                    "deploymentGeneration": 31,
                    "keyId": "receipt-key-1",
                    "receipt": {"receiptId": "sha256:" + "b" * 64},
                    "routeDigest": "1" * 32,
                    "schema": "mycellios-execution-receipt-envelope/1",
                    "signature": "signed-upstream",
                    "waveArtifactDigest": "3" * 32,
                    "waveStrategyDigest": "2" * 32,
                },
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        activation = encode_tensor(torch.tensor([[0.25, -0.5]]), TensorCodec.FP32)
        sampling_result, sampling_rows, sampling_vocab = sampling_logits_payload(
            ((0.25, -0.5),)
        )
        rng = random.Random(0x6D7963656C6C696F73)
        observed: set[FrameType] = set()
        transitions = (
            FrameType.BEGIN,
            FrameType.ACTIVATION,
            FrameType.VERIFY,
            FrameType.SAMPLING_VERIFY,
            FrameType.TRUNCATE,
            FrameType.CANCEL,
            FrameType.TOKEN,
            FrameType.SAMPLING_VERIFY_RESULT,
            FrameType.SAMPLING_RNG_CHECKPOINT,
            FrameType.RECEIPT_ENVELOPE,
            FrameType.RECEIPT_ACK,
        )
        for ordinal in range(512):
            frame_type = transitions[rng.randrange(len(transitions))]
            observed.add(frame_type)
            kwargs: dict[str, object] = {}
            reverse = frame_type in (
                FrameType.TOKEN,
                FrameType.SAMPLING_VERIFY_RESULT,
                FrameType.RECEIPT_ACK,
            )
            if frame_type in (
                FrameType.ACTIVATION,
                FrameType.VERIFY,
                FrameType.SAMPLING_VERIFY,
            ):
                kwargs.update(
                    flags=int(TensorCodec.FP32),
                    token_count=1,
                    hidden_size=2,
                    payload=activation,
                )
            elif frame_type == FrameType.TRUNCATE:
                kwargs["token_count"] = rng.randrange(4096)
            elif frame_type == FrameType.TOKEN:
                kwargs["payload"] = token_payload(rng.randrange(2**32))
            elif frame_type == FrameType.SAMPLING_VERIFY_RESULT:
                kwargs.update(
                    token_count=sampling_rows,
                    hidden_size=sampling_vocab,
                    payload=sampling_result,
                )
            elif frame_type == FrameType.SAMPLING_RNG_CHECKPOINT:
                kwargs["payload"] = sampling_rng_checkpoint_payload(
                    bytes(rng.randrange(256) for _ in range(32)),
                    rng.randrange(2**64),
                )
            elif frame_type == FrameType.RECEIPT_ENVELOPE:
                kwargs["payload"] = envelope
            elif frame_type == FrameType.RECEIPT_ACK:
                kwargs["payload"] = receipt_ack_payload(envelope)
            outbound, inbound = (
                (self.receiver, self.sender) if reverse else (self.sender, self.receiver)
            )
            send_frame(outbound, frame_type, ordinal + 1, **kwargs)
            frame = recv_frame(inbound)
            self.assertEqual(frame.frame_type, frame_type)
            if frame_type == FrameType.RECEIPT_ENVELOPE:
                self.assertEqual(decode_receipt_envelope(frame), envelope)
            elif frame_type == FrameType.RECEIPT_ACK:
                self.assertEqual(decode_receipt_ack(frame), receipt_ack_payload(envelope))
            elif frame_type == FrameType.SAMPLING_VERIFY_RESULT:
                self.assertEqual(
                    decode_sampling_logits(frame), ((0.25, -0.5),)
                )
            elif frame_type == FrameType.SAMPLING_RNG_CHECKPOINT:
                seed, counter = decode_sampling_rng_checkpoint(frame)
                self.assertEqual(len(seed), 32)
                self.assertGreaterEqual(counter, 0)

        self.assertEqual(observed, set(transitions))
        for frame_type in (
            FrameType.BEGIN,
            FrameType.END,
            FrameType.CANCEL,
            FrameType.SHUTDOWN,
            FrameType.TRUNCATE,
        ):
            with self.subTest(frame_type=frame_type.name):
                with self.assertRaisesRegex(ValueError, "require"):
                    send_frame(self.sender, frame_type, 999, flags=1)

    def test_bound_socket_seals_every_frame_to_one_deployment_generation(self) -> None:
        bind_socket_deployment_generation(self.sender, 17)
        bind_socket_deployment_generation(self.receiver, 17)
        send_frame(self.sender, FrameType.BEGIN, 41)
        self.assertEqual(recv_frame(self.receiver).deployment_generation, 17)
        with self.assertRaisesRegex(ValueError, "differs from socket binding"):
            send_frame(
                self.sender,
                FrameType.END,
                41,
                deployment_generation=18,
            )

    def test_bound_receiver_rejects_a_superseded_generation(self) -> None:
        bind_socket_deployment_generation(self.receiver, 18)
        send_frame(
            self.sender,
            FrameType.BEGIN,
            41,
            deployment_generation=17,
        )
        with self.assertRaisesRegex(ValueError, "another deployment generation"):
            recv_frame(self.receiver)

    def test_bound_receiver_rejects_cross_route_and_replayed_frames(self) -> None:
        bind_socket_deployment_generation(self.receiver, 17, "route-current")
        current = route_identity_digest("route-current")
        send_frame(
            self.sender,
            FrameType.BEGIN,
            41,
            deployment_generation=17,
            route_digest=current,
            sequence=0,
        )
        self.assertEqual(recv_frame(self.receiver).sequence, 0)
        send_frame(
            self.sender,
            FrameType.END,
            41,
            deployment_generation=17,
            route_digest=current,
            sequence=0,
        )
        with self.assertRaisesRegex(ValueError, "replayed or out of order"):
            recv_frame(self.receiver)

        other_sender, other_receiver = socket.socketpair()
        try:
            bind_socket_deployment_generation(other_receiver, 17, "route-current")
            send_frame(
                other_sender,
                FrameType.BEGIN,
                42,
                deployment_generation=17,
                route_digest=route_identity_digest("route-old"),
            )
            with self.assertRaisesRegex(ValueError, "another route"):
                recv_frame(other_receiver)
        finally:
            other_sender.close()
            other_receiver.close()

    def test_bound_receiver_rejects_cross_strategy_and_artifact_frames(self) -> None:
        cases = (
            (
                "wave strategy",
                wave_identity_digest("strategy", "strategy-old"),
                wave_identity_digest("artifact", "sha256:current"),
            ),
            (
                "wave artifact",
                wave_identity_digest("strategy", "strategy-current"),
                wave_identity_digest("artifact", "sha256:old"),
            ),
        )
        for message, strategy_digest, artifact_digest in cases:
            with self.subTest(message=message):
                sender, receiver = socket.socketpair()
                try:
                    bind_socket_deployment_generation(
                        receiver,
                        17,
                        "route-current",
                        wave_strategy_id="strategy-current",
                        wave_artifact_identity="sha256:current",
                    )
                    send_frame(
                        sender,
                        FrameType.BEGIN,
                        42,
                        deployment_generation=17,
                        route_digest=route_identity_digest("route-current"),
                        wave_strategy_digest=strategy_digest,
                        wave_artifact_digest=artifact_digest,
                    )
                    with self.assertRaisesRegex(ValueError, message):
                        recv_frame(receiver)
                finally:
                    sender.close()
                    receiver.close()

    def test_bound_route_sequences_mixed_frames_monotonically(self) -> None:
        bind_socket_deployment_generation(self.sender, 23, "route-stateful")
        bind_socket_deployment_generation(self.receiver, 23, "route-stateful")
        frame_types = (
            FrameType.BEGIN,
            FrameType.END,
            FrameType.CANCEL,
            FrameType.PING,
        )
        for sequence in range(256):
            frame_type = frame_types[sequence % len(frame_types)]
            send_frame(
                self.sender,
                frame_type,
                sequence + 1,
                step=sequence if frame_type == FrameType.PING else 0,
            )
            frame = recv_frame(self.receiver)
            self.assertEqual(frame.sequence, sequence)
            self.assertEqual(frame.deployment_generation, 23)
            self.assertEqual(frame.route_digest, route_identity_digest("route-stateful"))

    def test_route_probe_frames_are_payload_free_control_frames(self) -> None:
        self.assertEqual(
            send_frame(self.sender, FrameType.PING, 1234, step=41),
            HEADER_BYTES,
        )
        ping = recv_frame(self.receiver)
        self.assertEqual(ping.frame_type, FrameType.PING)
        self.assertEqual(ping.request_id, 1234)
        self.assertEqual(ping.step, 41)
        self.assertEqual((ping.flags, ping.token_count, ping.hidden_size), (0, 0, 0))
        send_frame(self.sender, FrameType.PONG, ping.request_id, step=ping.step)
        pong = recv_frame(self.receiver)
        self.assertEqual(pong.frame_type, FrameType.PONG)
        self.assertEqual((pong.request_id, pong.step), (1234, 41))
        self.assertEqual((pong.flags, pong.token_count, pong.hidden_size), (0, 0, 0))
        with self.assertRaisesRegex(ValueError, "cannot carry"):
            send_frame(self.sender, FrameType.PING, 1234, payload=b"x")

    def test_route_probe_send_rejects_noncanonical_metadata(self) -> None:
        for frame_type in (FrameType.PING, FrameType.PONG):
            for field, value in (
                ("flags", 1),
                ("token_count", 1),
                ("hidden_size", 1),
            ):
                with self.subTest(frame_type=frame_type.name, field=field):
                    with self.assertRaisesRegex(
                        ValueError, "flags=token_count=hidden_size=0"
                    ):
                        send_frame(
                            self.sender,
                            frame_type,
                            1234,
                            step=41,
                            **{field: value},
                        )

    def test_exact_branch_control_frames_are_canonical_and_round_trip_uint64_ids(self) -> None:
        parent = 2**63 + 19
        child = 2**63 + 23
        send_frame(
            self.sender,
            FrameType.FORK,
            child,
            payload=branch_request_payload(parent),
        )
        fork = recv_frame(self.receiver)
        self.assertEqual((fork.frame_type, fork.request_id), (FrameType.FORK, child))
        self.assertEqual(decode_branch_request_id(fork), parent)

        send_frame(
            self.sender,
            FrameType.PROMOTE,
            parent,
            payload=branch_request_payload(child),
        )
        promote = recv_frame(self.receiver)
        self.assertEqual(
            (promote.frame_type, promote.request_id),
            (FrameType.PROMOTE, parent),
        )
        self.assertEqual(decode_branch_request_id(promote), child)

        for frame_type in (FrameType.FORK, FrameType.PROMOTE):
            with self.subTest(frame_type=frame_type.name):
                with self.assertRaisesRegex(ValueError, "exactly one uint64"):
                    send_frame(self.sender, frame_type, child, payload=b"short")
                with self.assertRaisesRegex(ValueError, "flags=step=token_count"):
                    send_frame(
                        self.sender,
                        frame_type,
                        child,
                        step=1,
                        payload=branch_request_payload(parent),
                    )
        with self.assertRaisesRegex(ValueError, "branch lifecycle"):
            decode_branch_request_id(
                Frame(FrameType.BEGIN, 0, child, 0, 0, 0, b"")
            )

    def test_route_probe_recv_rejects_noncanonical_wire_metadata(self) -> None:
        malformed_fields = (
            ("flags", 1, 0, 0, 0),
            ("token_count", 0, 1, 0, 0),
            ("hidden_size", 0, 0, 1, 0),
            ("payload", 0, 0, 0, 1),
        )
        for frame_type in (FrameType.PING, FrameType.PONG):
            for field, flags, token_count, hidden_size, payload_size in malformed_fields:
                with self.subTest(frame_type=frame_type.name, field=field):
                    sender, receiver = socket.socketpair()
                    try:
                        sender.sendall(
                            HEADER.pack(
                                MAGIC,
                                VERSION,
                                int(frame_type),
                                flags,
                                1234,
                                0,
                                route_identity_digest("static"),
                                STATIC_WAVE_STRATEGY_DIGEST,
                                STATIC_WAVE_ARTIFACT_DIGEST,
                                0,
                                41,
                                token_count,
                                hidden_size,
                                payload_size,
                                b"\0" * 16,
                            )
                            + (b"x" if payload_size else b"")
                        )
                        with self.assertRaisesRegex(
                            ValueError, "flags=token_count=hidden_size=0"
                        ):
                            recv_frame(receiver)
                    finally:
                        sender.close()
                        receiver.close()

    def test_activation_frame_survives_fragmented_transport(self) -> None:
        tensor = torch.arange(21, dtype=torch.float32).reshape(1, 3, 7) / 5
        payload = encode_tensor(tensor, TensorCodec.FP32)
        raw_sender, raw_receiver = socket.socketpair()
        try:
            send_frame(
                raw_sender,
                FrameType.ACTIVATION,
                99,
                step=4,
                token_count=3,
                hidden_size=7,
                flags=int(TensorCodec.FP32),
                payload=payload,
            )
            raw = bytes(recv_exact(raw_receiver, HEADER_BYTES + len(payload)))
        finally:
            raw_sender.close()
            raw_receiver.close()
        for offset in range(0, len(raw), 3):
            self.sender.sendall(raw[offset : offset + 3])

        frame = recv_frame(self.receiver)
        decoded = decode_tensor(frame)
        self.assertEqual(frame.request_id, 99)
        self.assertEqual(frame.step, 4)
        self.assertTrue(torch.equal(decoded, tensor))

    def test_wave_digest_rejects_payload_tampering_before_decode(self) -> None:
        raw_sender, raw_receiver = socket.socketpair()
        try:
            send_frame(raw_sender, FrameType.ERROR, 9, payload=b"original")
            raw = bytearray(recv_exact(raw_receiver, HEADER_BYTES + len(b"original")))
        finally:
            raw_sender.close()
            raw_receiver.close()
        raw[-1] ^= 0x01
        self.sender.sendall(raw)
        with self.assertRaisesRegex(ValueError, "wave digest is invalid"):
            recv_frame(self.receiver)

    def test_error_frame_can_carry_a_message(self) -> None:
        send_frame(self.sender, FrameType.ERROR, 8, payload=b"stage failed")
        frame = recv_frame(self.receiver)
        self.assertEqual(frame.frame_type, FrameType.ERROR)
        self.assertEqual(frame.payload, b"stage failed")

    def test_semantically_invalid_payload_is_rejected_before_send(self) -> None:
        with self.assertRaisesRegex(ValueError, "activation payload"):
            send_frame(
                self.sender,
                FrameType.ACTIVATION,
                1,
                token_count=2,
                hidden_size=3,
                flags=int(TensorCodec.FP32),
                payload=b"short",
            )
        with self.assertRaisesRegex(ValueError, "cannot carry"):
            send_frame(self.sender, FrameType.BEGIN, 1, payload=b"unexpected")
        with self.assertRaisesRegex(ValueError, "four bytes"):
            send_frame(self.sender, FrameType.TOKEN, 1, payload=b"x")

    def test_invalid_wire_header_is_rejected_without_allocating_payload(self) -> None:
        raw = HEADER.pack(
            MAGIC,
            VERSION,
            int(FrameType.ERROR),
            0,
            0,
            0,
            route_identity_digest("static"),
            STATIC_WAVE_STRATEGY_DIGEST,
            STATIC_WAVE_ARTIFACT_DIGEST,
            0,
            0,
            0,
            0,
            MAX_PAYLOAD_BYTES + 1,
            b"\0" * 16,
        )
        self.sender.sendall(raw)
        with self.assertRaisesRegex(ValueError, "payload exceeds"):
            recv_frame(self.receiver)

    def test_truncated_frame_raises_eof(self) -> None:
        self.sender.sendall(b"partial")
        self.sender.close()
        with self.assertRaises(EOFError):
            recv_frame(self.receiver)

    def test_numeric_fields_are_range_checked(self) -> None:
        with self.assertRaisesRegex(ValueError, "request_id"):
            send_frame(self.sender, FrameType.BEGIN, -1)
        with self.assertRaisesRegex(ValueError, "step"):
            send_frame(self.sender, FrameType.BEGIN, 1, step=2**32)
        with self.assertRaisesRegex(TypeError, "flags"):
            send_frame(self.sender, FrameType.BEGIN, 1, flags=True)

    def test_recv_exact_zero_and_negative(self) -> None:
        self.assertEqual(recv_exact(self.receiver, 0), b"")
        with self.assertRaisesRegex(ValueError, "negative"):
            recv_exact(self.receiver, -1)


class TensorCodecTests(unittest.TestCase):
    def round_trip(self, tensor: torch.Tensor, codec: TensorCodec) -> torch.Tensor:
        payload = encode_tensor(tensor, codec)
        frame = Frame(
            frame_type=FrameType.ACTIVATION,
            flags=int(codec),
            request_id=1,
            step=0,
            token_count=tensor.shape[1],
            hidden_size=tensor.shape[2],
            payload=payload,
        )
        return decode_tensor(frame)

    def test_fp32_is_bit_exact(self) -> None:
        source = torch.tensor(
            [[[-3.25, -0.0, 0.125], [1.0, 17.75, 0.000_003]]],
            dtype=torch.float32,
        )
        self.assertTrue(torch.equal(self.round_trip(source, TensorCodec.FP32), source))

    def test_fp16_has_expected_precision(self) -> None:
        torch.manual_seed(7)
        source = torch.randn(1, 11, 23, dtype=torch.float32)
        decoded = self.round_trip(source, TensorCodec.FP16)
        self.assertTrue(torch.allclose(decoded, source, atol=0.002, rtol=0.001))

    def test_owner_backed_fp16_payload_stages_directly_in_wire_dtype(self) -> None:
        source = torch.randn(1, 3, 17, dtype=torch.float32)
        encoded = encode_tensor_payload(source, TensorCodec.FP16)
        self.assertEqual(encoded.staging_dtype, torch.float16)
        self.assertEqual(encoded.source_device, "cpu")
        self.assertEqual(encoded.nbytes, source.numel() * 2)
        self.assertIsInstance(encoded.owner, torch.Tensor)
        self.assertEqual(encoded.view.tobytes(), encode_tensor(source, TensorCodec.FP16))

    def test_owner_backed_payload_can_be_sent_without_materialising_bytes(self) -> None:
        source = torch.randn(1, 2, 9, dtype=torch.float32)
        encoded = encode_tensor_payload(source, TensorCodec.FP32)
        sender, receiver = socket.socketpair()
        try:
            sent = send_frame(
                sender,
                FrameType.ACTIVATION,
                33,
                token_count=2,
                hidden_size=9,
                flags=int(TensorCodec.FP32),
                payload=encoded.view,
            )
            frame = recv_frame(receiver)
            self.assertEqual(sent, HEADER_BYTES + encoded.nbytes)
            self.assertTrue(torch.equal(decode_tensor(frame), source))
        finally:
            sender.close()
            receiver.close()

    def test_owner_backed_payload_is_an_immutable_snapshot(self) -> None:
        for codec, dtype in (
            (TensorCodec.FP32, torch.float32),
            (TensorCodec.FP16, torch.float16),
        ):
            with self.subTest(codec=codec.name):
                source = torch.tensor([[[1.0, 2.0]]], dtype=dtype)
                encoded = encode_tensor_payload(source, codec)
                snapshot = encoded.view.tobytes()
                source.zero_()
                self.assertEqual(encoded.view.tobytes(), snapshot)

    def test_activation_encoder_rejects_non_floating_tensors(self) -> None:
        source = torch.tensor([[[-(1 << 63)]]], dtype=torch.int64)
        for codec in TensorCodec:
            with self.subTest(codec=codec.name), self.assertRaisesRegex(
                ValueError,
                "floating dtype",
            ):
                encode_tensor_payload(source, codec)

    def test_int8_error_is_bounded_by_one_quantization_step(self) -> None:
        torch.manual_seed(11)
        source = torch.randn(1, 13, 19, dtype=torch.float32) * 3
        decoded = self.round_trip(source, TensorCodec.INT8)
        step = float(source.abs().max().item()) / 127
        self.assertLessEqual(float((decoded - source).abs().max().item()), step + 1e-6)

    def test_grouped_and_hadamard_int8_support_arbitrary_hidden_sizes(self) -> None:
        torch.manual_seed(29)
        source = torch.randn(1, 3, 70, dtype=torch.float32)
        for codec in (TensorCodec.INT8_GROUPED, TensorCodec.INT8_HADAMARD):
            with self.subTest(codec=codec.name):
                payload = encode_tensor(source, codec)
                decoded = self.round_trip(source, codec)
                self.assertEqual(tuple(decoded.shape), (1, 3, 70))
                self.assertLess(float((decoded - source).abs().mean().item()), 0.02)
                self.assertLess(len(payload), source.numel() * 2)

    def test_hadamard_rotation_reduces_outlier_quantization_error(self) -> None:
        torch.manual_seed(31)
        source = torch.randn(1, 1, 64, dtype=torch.float32) * 0.2
        source[0, 0, 0] = 100.0
        global_int8 = self.round_trip(source, TensorCodec.INT8)
        rotated_int8 = self.round_trip(source, TensorCodec.INT8_HADAMARD)
        global_mse = float(torch.mean((global_int8 - source) ** 2).item())
        rotated_mse = float(torch.mean((rotated_int8 - source) ** 2).item())
        self.assertLess(rotated_mse, global_mse * 0.2)

    def test_zero_int8_tensor_round_trips(self) -> None:
        source = torch.zeros(1, 2, 5, dtype=torch.float32)
        self.assertTrue(torch.equal(self.round_trip(source, TensorCodec.INT8), source))

    def test_int8_rejects_non_finite_values_and_scale(self) -> None:
        source = torch.tensor([[[float("nan")]]], dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            encode_tensor(source, TensorCodec.INT8)

        frame = Frame(
            frame_type=FrameType.ACTIVATION,
            flags=int(TensorCodec.INT8),
            request_id=1,
            step=0,
            token_count=1,
            hidden_size=1,
            payload=struct.pack("<f", float("nan")) + b"\x00",
        )
        with self.assertRaisesRegex(ValueError, "quantization scale"):
            decode_tensor(frame)

    def test_shape_and_payload_must_match(self) -> None:
        frame = Frame(
            frame_type=FrameType.ACTIVATION,
            flags=int(TensorCodec.FP32),
            request_id=1,
            step=0,
            token_count=3,
            hidden_size=2,
            payload=b"\x00" * 4,
        )
        with self.assertRaisesRegex(ValueError, "expected 24"):
            decode_tensor(frame)

    def test_token_payload_round_trip_and_range(self) -> None:
        frame = Frame(FrameType.TOKEN, 0, 9, 3, 0, 0, token_payload(49_152))
        self.assertEqual(decode_token(frame), 49_152)
        with self.assertRaisesRegex(ValueError, "token_id"):
            token_payload(-1)
        with self.assertRaisesRegex(ValueError, "token_id"):
            token_payload(2**32)

    def test_prefill_verify_and_verification_result_frames(self) -> None:
        source = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
        for frame_type in (FrameType.PREFILL, FrameType.VERIFY):
            frame = Frame(
                frame_type,
                int(TensorCodec.INT8_GROUPED),
                3,
                7,
                3,
                4,
                encode_tensor(source, TensorCodec.INT8_GROUPED),
            )
            self.assertEqual(tuple(decode_tensor(frame).shape), (1, 3, 4))

        result = Frame(
            FrameType.VERIFY_RESULT,
            0,
            3,
            7,
            3,
            0,
            verify_result_payload((11, 12, 13)),
        )
        self.assertEqual(decode_verify_result(result), (11, 12, 13))


# Frozen copy of the historical scalar INT8_GROUPED / INT8_HADAMARD codec.
# The vectorized implementation in protocol.py must stay bit-identical to this
# reference for every input, so old frames keep decoding to the same values.
_REFERENCE_GROUP_SIZE = 64


def _reference_quantization_blocks(
    hidden_size: int, *, hadamard: bool
) -> tuple[tuple[int, int], ...]:
    blocks: list[tuple[int, int]] = []
    start = 0
    while start < hidden_size:
        remaining = hidden_size - start
        if hadamard:
            size = 1 << int(math.floor(math.log2(min(_REFERENCE_GROUP_SIZE, remaining))))
        else:
            size = min(_REFERENCE_GROUP_SIZE, remaining)
        blocks.append((start, start + size))
        start += size
    return tuple(blocks)


def _reference_normalized_fwht(values: torch.Tensor) -> torch.Tensor:
    size = int(values.numel())
    output = values.to(torch.float32).clone()
    stride = 1
    while stride < size:
        view = output.reshape(-1, stride * 2)
        left = view[:, :stride].clone()
        right = view[:, stride:].clone()
        view[:, :stride] = left + right
        view[:, stride:] = left - right
        stride *= 2
    return output / math.sqrt(size)


def _reference_encode_grouped(tensor: torch.Tensor, codec: TensorCodec) -> bytes:
    contiguous = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    hidden_size = int(contiguous.shape[-1])
    rows = contiguous.reshape(-1, hidden_size)
    blocks = _reference_quantization_blocks(
        hidden_size,
        hadamard=codec == TensorCodec.INT8_HADAMARD,
    )
    scales: list[float] = []
    quantized_rows: list[torch.Tensor] = []
    for row in rows:
        row_chunks: list[torch.Tensor] = []
        for start, end in blocks:
            values = row[start:end]
            if codec == TensorCodec.INT8_HADAMARD:
                values = _reference_normalized_fwht(values)
            max_value = float(values.abs().max().item())
            scale = max_value / 127.0 if max_value > 0 else 1.0
            scales.append(scale)
            row_chunks.append(
                torch.clamp(torch.round(values / scale), -127, 127).to(torch.int8)
            )
        quantized_rows.append(torch.cat(row_chunks))
    scale_payload = struct.pack(f"<{len(scales)}f", *scales)
    quantized = torch.stack(quantized_rows).contiguous()
    return scale_payload + quantized.numpy().tobytes(order="C")


def _reference_decode_grouped(frame: Frame) -> torch.Tensor:
    codec = TensorCodec(frame.flags)
    blocks = _reference_quantization_blocks(
        frame.hidden_size,
        hadamard=codec == TensorCodec.INT8_HADAMARD,
    )
    scale_count = frame.token_count * len(blocks)
    scale_bytes = scale_count * 4
    payload = bytearray(frame.payload)
    scales = struct.unpack(f"<{scale_count}f", payload[:scale_bytes])
    quantized = torch.frombuffer(
        memoryview(payload)[scale_bytes:], dtype=torch.int8
    ).to(torch.float32)
    rows = quantized.reshape(frame.token_count, frame.hidden_size)
    decoded_rows: list[torch.Tensor] = []
    scale_index = 0
    for row in rows:
        chunks: list[torch.Tensor] = []
        for start, end in blocks:
            values = row[start:end] * scales[scale_index]
            scale_index += 1
            if codec == TensorCodec.INT8_HADAMARD:
                values = _reference_normalized_fwht(values)
            chunks.append(values)
        decoded_rows.append(torch.cat(chunks))
    return torch.stack(decoded_rows).reshape(1, frame.token_count, frame.hidden_size)


class VectorizedGroupedCodecEquivalenceTests(unittest.TestCase):
    HIDDEN_SIZES = (8, 64, 96, 128, 100, 4096, 5120)
    TOKEN_COUNTS = (1, 3, 17)
    CODECS = (TensorCodec.INT8_GROUPED, TensorCodec.INT8_HADAMARD)

    @staticmethod
    def _grid_tensor(token_count: int, hidden_size: int) -> torch.Tensor:
        torch.manual_seed(token_count * 10_007 + hidden_size)
        source = torch.randn(1, token_count, hidden_size, dtype=torch.float32)
        source[0, :, 0] = -source[0, :, 0].abs() - 1.0
        source[0, 0, hidden_size // 2] = 1_000.0
        if hidden_size >= 3:
            source[0, :, 1:3] = 0.0
        if token_count > 1:
            # A whole zero row exercises the scale = 1.0 path per block.
            source[0, token_count - 1, :] = 0.0
        return source

    @staticmethod
    def _frame(codec: TensorCodec, tensor_shape: tuple[int, int], payload: bytes) -> Frame:
        token_count, hidden_size = tensor_shape
        return Frame(
            frame_type=FrameType.ACTIVATION,
            flags=int(codec),
            request_id=1,
            step=0,
            token_count=token_count,
            hidden_size=hidden_size,
            payload=payload,
        )

    def test_grid_is_bit_identical_to_frozen_reference(self) -> None:
        for codec in self.CODECS:
            for hidden_size in self.HIDDEN_SIZES:
                for token_count in self.TOKEN_COUNTS:
                    with self.subTest(
                        codec=codec.name, hidden=hidden_size, tokens=token_count
                    ):
                        source = self._grid_tensor(token_count, hidden_size)
                        payload = encode_tensor(source, codec)
                        self.assertEqual(
                            payload, _reference_encode_grouped(source, codec)
                        )
                        frame = self._frame(codec, (token_count, hidden_size), payload)
                        decoded = decode_tensor(frame)
                        self.assertEqual(
                            tuple(decoded.shape), (1, token_count, hidden_size)
                        )
                        self.assertTrue(
                            torch.equal(decoded, _reference_decode_grouped(frame))
                        )

    def test_all_zero_tensor_matches_reference_and_round_trips(self) -> None:
        source = torch.zeros(1, 2, 100, dtype=torch.float32)
        for codec in self.CODECS:
            with self.subTest(codec=codec.name):
                payload = encode_tensor(source, codec)
                self.assertEqual(payload, _reference_encode_grouped(source, codec))
                decoded = decode_tensor(self._frame(codec, (2, 100), payload))
                self.assertTrue(torch.equal(decoded, source))

    def test_round_trip_error_stays_within_one_quantization_step(self) -> None:
        torch.manual_seed(97)
        source = torch.randn(1, 3, 100, dtype=torch.float32) * 2
        for codec in self.CODECS:
            with self.subTest(codec=codec.name):
                payload = encode_tensor(source, codec)
                decoded = decode_tensor(self._frame(codec, (3, 100), payload))
                self.assertLess(float((decoded - source).abs().mean().item()), 0.05)


class DeflateCodecTests(unittest.TestCase):
    PAIRS = (
        (TensorCodec.INT8_GROUPED_DEFLATE, TensorCodec.INT8_GROUPED),
        (TensorCodec.INT8_HADAMARD_DEFLATE, TensorCodec.INT8_HADAMARD),
    )

    @staticmethod
    def _frame(
        codec: TensorCodec, token_count: int, hidden_size: int, payload: bytes
    ) -> Frame:
        return Frame(
            frame_type=FrameType.ACTIVATION,
            flags=int(codec),
            request_id=1,
            step=0,
            token_count=token_count,
            hidden_size=hidden_size,
            payload=payload,
        )

    def test_deflate_payload_is_zlib_of_base_and_decodes_identically(self) -> None:
        torch.manual_seed(41)
        source = torch.randn(1, 5, 100, dtype=torch.float32)
        for deflate_codec, base_codec in self.PAIRS:
            with self.subTest(codec=deflate_codec.name):
                base_payload = encode_tensor(source, base_codec)
                deflate_payload = encode_tensor(source, deflate_codec)
                self.assertEqual(zlib.decompress(deflate_payload), base_payload)
                decoded = decode_tensor(
                    self._frame(deflate_codec, 5, 100, deflate_payload)
                )
                base_decoded = decode_tensor(self._frame(base_codec, 5, 100, base_payload))
                self.assertTrue(torch.equal(decoded, base_decoded))

    def test_deflate_frame_survives_the_wire(self) -> None:
        sender, receiver = socket.socketpair()
        try:
            torch.manual_seed(43)
            source = torch.randn(1, 3, 96, dtype=torch.float32)
            for deflate_codec, base_codec in self.PAIRS:
                with self.subTest(codec=deflate_codec.name):
                    send_frame(
                        sender,
                        FrameType.ACTIVATION,
                        5,
                        token_count=3,
                        hidden_size=96,
                        flags=int(deflate_codec),
                        payload=encode_tensor(source, deflate_codec),
                    )
                    frame = recv_frame(receiver)
                    base_decoded = decode_tensor(
                        self._frame(base_codec, 3, 96, encode_tensor(source, base_codec))
                    )
                    self.assertTrue(torch.equal(decode_tensor(frame), base_decoded))
        finally:
            sender.close()
            receiver.close()

    def test_corrupt_deflate_payload_is_rejected(self) -> None:
        torch.manual_seed(47)
        source = torch.randn(1, 2, 64, dtype=torch.float32)
        for deflate_codec, _ in self.PAIRS:
            with self.subTest(codec=deflate_codec.name):
                payload = bytearray(encode_tensor(source, deflate_codec))
                payload[len(payload) // 2] ^= 0xFF
                with self.assertRaises(ValueError):
                    decode_tensor(self._frame(deflate_codec, 2, 64, bytes(payload)))

    def test_wrong_inflated_size_is_rejected(self) -> None:
        torch.manual_seed(53)
        source = torch.randn(1, 2, 64, dtype=torch.float32)
        for deflate_codec, base_codec in self.PAIRS:
            with self.subTest(codec=deflate_codec.name):
                base_payload = encode_tensor(source, base_codec)
                oversized = zlib.compress(base_payload + b"\x00", 1)
                with self.assertRaisesRegex(ValueError, "inflate"):
                    decode_tensor(self._frame(deflate_codec, 2, 64, oversized))
                undersized = zlib.compress(base_payload[:-1], 1)
                with self.assertRaisesRegex(ValueError, "inflate"):
                    decode_tensor(self._frame(deflate_codec, 2, 64, undersized))
                trailing = zlib.compress(base_payload, 1) + b"junk"
                with self.assertRaisesRegex(ValueError, "inflate"):
                    decode_tensor(self._frame(deflate_codec, 2, 64, trailing))

    def test_hostile_token_count_is_rejected_before_inflating(self) -> None:
        # token_count comes from the untrusted header, so the decompression
        # budget must be capped by MAX_PAYLOAD_BYTES, not by the header shape.
        # The garbage payload proves ordering: inflating first would raise
        # "not valid zlib data" instead of the size rejection.
        token_count = MAX_PAYLOAD_BYTES // 4096 + 1
        for deflate_codec, _ in self.PAIRS:
            with self.subTest(codec=deflate_codec.name):
                with self.assertRaisesRegex(ValueError, "would inflate"):
                    decode_tensor(
                        self._frame(deflate_codec, token_count, 4096, b"junk")
                    )

    def test_recv_rejects_header_whose_inflated_size_exceeds_max_payload(self) -> None:
        sender, receiver = socket.socketpair()
        try:
            payload = zlib.compress(b"\x00" * (1024 * 1024), 1)
            raw = HEADER.pack(
                MAGIC,
                VERSION,
                int(FrameType.ACTIVATION),
                int(TensorCodec.INT8_GROUPED_DEFLATE),
                1,
                0,
                route_identity_digest("static"),
                STATIC_WAVE_STRATEGY_DIGEST,
                STATIC_WAVE_ARTIFACT_DIGEST,
                0,
                0,
                MAX_PAYLOAD_BYTES // 4096 + 1,
                4096,
                len(payload),
                b"\0" * 16,
            ) + payload
            sender.sendall(raw)
            with self.assertRaisesRegex(ValueError, "would inflate"):
                recv_frame(receiver)
        finally:
            sender.close()
            receiver.close()

    def test_oversized_wire_payload_is_rejected_before_send(self) -> None:
        sender, receiver = socket.socketpair()
        try:
            with self.assertRaisesRegex(ValueError, "activation payload"):
                send_frame(
                    sender,
                    FrameType.ACTIVATION,
                    1,
                    token_count=1,
                    hidden_size=8,
                    flags=int(TensorCodec.INT8_GROUPED_DEFLATE),
                    payload=b"\x00" * 10_000,
                )
        finally:
            sender.close()
            receiver.close()

    def test_gaussian_activations_compress_below_unity(self) -> None:
        torch.manual_seed(59)
        # Realistic decode wave: gaussian activations over a large hidden size.
        source = torch.randn(1, 17, 4096, dtype=torch.float32)
        for deflate_codec, base_codec in self.PAIRS:
            with self.subTest(codec=deflate_codec.name):
                base_payload = encode_tensor(source, base_codec)
                deflate_payload = encode_tensor(source, deflate_codec)
                self.assertLess(len(deflate_payload) / len(base_payload), 1.0)


class PersistentStageDataPlaneTests(unittest.TestCase):
    def test_last_stage_returns_full_sampling_distribution(self) -> None:
        config = StageProcessConfig(
            spec=StageModelSpec("fake", 0, 1, 1, 1),
            pipeline_id=778,
            listen_host="127.0.0.1",
            listen_port=20_099,
            next_host=None,
            next_port=None,
            next_layer_end=None,
            return_host="127.0.0.1",
            return_port=20_100,
            codec=TensorCodec.FP32,
            one_way_delay_ms=0,
            bandwidth_mbps=0,
        )
        runner = _FakeLastStageRunner(config.spec)
        frame = Frame(
            FrameType.SAMPLING_VERIFY,
            int(TensorCodec.FP32),
            91,
            3,
            2,
            runner.hidden_size,
            encode_tensor(torch.ones(1, 2, runner.hidden_size), TensorCodec.FP32),
        )
        metrics: dict[str, object] = {"bytes_out": 0}
        stage_return, root_return = socket.socketpair()
        try:
            route_stage_result(
                frame,
                torch.ones(1, 2, runner.hidden_size),
                (42, 42),
                runner=runner,
                config=config,
                metrics=metrics,
                downstream=None,
                return_socket=stage_return,
                emulator=LinkEmulator(),
            )
            returned = recv_frame(root_return)
            self.assertEqual(returned.frame_type, FrameType.SAMPLING_VERIFY_RESULT)
            self.assertEqual(returned.token_count, 2)
            self.assertEqual(returned.hidden_size, 3)
            self.assertEqual(
                decode_sampling_logits(returned),
                ((1.0, 2.0, 3.0), (1.0, 2.0, 3.0)),
            )
        finally:
            stage_return.close()
            root_return.close()

    def test_stage_fork_and_promote_enforce_sealed_exact_leaf_lifecycle(self) -> None:
        config = StageProcessConfig(
            spec=StageModelSpec("fake", 15, 30, 30, 1),
            pipeline_id=779,
            listen_host="127.0.0.1",
            listen_port=20_101,
            next_host=None,
            next_port=None,
            next_layer_end=None,
            return_host="127.0.0.1",
            return_port=20_102,
            codec=TensorCodec.FP32,
            one_way_delay_ms=0,
            bandwidth_mbps=0,
            max_speculative_branches=2,
            max_speculative_branch_tokens=4,
            max_speculative_kv_bytes=31,
        )
        validate_stage_config(config)
        runner = _FakeLastStageRunner(config.spec)
        validate_speculative_runner(config, runner)
        runner.begin(11)
        runner.active[11] = 3
        request_metrics: dict[int, dict[str, object]] = {
            11: {
                "frames": 1,
                "compute_ms": 7,
                "bytes_out": 99,
                "tokens": 3,
                "model_forward_calls": 1,
                "physical_batch_calls": 0,
                "physical_batch_items": 1,
                "max_physical_batch_size": 1,
            }
        }
        branches: dict[int, int] = {}
        stage_downstream, child_downstream = socket.socketpair()
        try:
            fork = Frame(
                FrameType.FORK,
                0,
                22,
                0,
                0,
                0,
                branch_request_payload(11),
            )
            fork_stage_request(
                fork,
                config,
                runner,
                request_metrics,
                branches,
                stage_downstream,
            )
            relayed = recv_frame(child_downstream)
            self.assertEqual((relayed.frame_type, relayed.request_id), (FrameType.FORK, 22))
            self.assertEqual(decode_branch_request_id(relayed), 11)
            self.assertEqual(branches, {22: 11})
            self.assertEqual(runner.sequence_length(22), 3)
            self.assertEqual(request_metrics[22]["frames"], 1)
            self.assertEqual(request_metrics[22]["branch_inherited_tokens"], 3)

            oversized = Frame(
                FrameType.ACTIVATION,
                int(TensorCodec.FP32),
                22,
                1,
                2,
                4,
                b"\x00" * 32,
            )
            with self.assertRaisesRegex(ValueError, "branch_tokens"):
                validate_activation(
                    oversized,
                    config,
                    runner,
                    request_metrics,
                    branches,
                )

            byte_oversized = Frame(
                FrameType.ACTIVATION,
                int(TensorCodec.FP32),
                22,
                1,
                1,
                4,
                b"\x00" * 16,
            )
            with self.assertRaisesRegex(ValueError, "KV bytes"):
                validate_activation(
                    byte_oversized,
                    config,
                    runner,
                    request_metrics,
                    branches,
                )

            runner.active[22] = 4
            promote = Frame(
                FrameType.PROMOTE,
                0,
                11,
                0,
                0,
                0,
                branch_request_payload(22),
            )
            promote_stage_request(
                promote,
                config,
                runner,
                request_metrics,
                branches,
                stage_downstream,
            )
            relayed = recv_frame(child_downstream)
            self.assertEqual(
                (relayed.frame_type, relayed.request_id),
                (FrameType.PROMOTE, 11),
            )
            self.assertEqual(decode_branch_request_id(relayed), 22)
            self.assertEqual(branches, {})
            self.assertEqual(runner.sequence_length(11), 4)
            self.assertNotIn(22, runner.active)

            with self.assertRaisesRegex(ValueError, "before cloning"):
                fork_stage_request(
                    Frame(
                        FrameType.FORK,
                        0,
                        33,
                        0,
                        0,
                        0,
                        branch_request_payload(11),
                    ),
                    config,
                    runner,
                    request_metrics,
                    branches,
                    None,
                )
            self.assertNotIn(33, runner.active)

            with self.assertRaisesRegex(ValueError, "not active"):
                promote_stage_request(
                    promote,
                    config,
                    runner,
                    request_metrics,
                    branches,
                    None,
                )
        finally:
            stage_downstream.close()
            child_downstream.close()

    def test_stage_tree_contract_is_disabled_by_default_and_pair_sealed(self) -> None:
        base = StageProcessConfig(
            spec=StageModelSpec("fake", 15, 30, 30, 1),
            pipeline_id=780,
            listen_host="127.0.0.1",
            listen_port=20_103,
            next_host=None,
            next_port=None,
            next_layer_end=None,
            return_host="127.0.0.1",
            return_port=20_104,
            codec=TensorCodec.FP32,
            one_way_delay_ms=0,
            bandwidth_mbps=0,
        )
        validate_stage_config(base)
        for updates in (
            {"max_speculative_branches": 1},
            {"max_speculative_branch_tokens": 8},
            {"max_speculative_kv_bytes": 1024},
            {
                "max_speculative_branches": 65,
                "max_speculative_branch_tokens": 8,
                "max_speculative_kv_bytes": 1024,
            },
        ):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                validate_stage_config(
                    StageProcessConfig(**{**base.__dict__, **updates})
                )

        runner = _FakeLastStageRunner(base.spec)
        runner.begin(1)
        with self.assertRaisesRegex(ValueError, "disabled"):
            fork_stage_request(
                Frame(
                    FrameType.FORK,
                    0,
                    2,
                    0,
                    0,
                    0,
                    branch_request_payload(1),
                ),
                base,
                runner,
                {1: {"frames": 0}},
                {},
                None,
            )

    def test_speculative_physical_batch_preflights_combined_kv_growth(self) -> None:
        config = StageProcessConfig(
            spec=StageModelSpec("fake", 15, 30, 30, 1),
            pipeline_id=781,
            listen_host="127.0.0.1",
            listen_port=20_105,
            next_host=None,
            next_port=None,
            next_layer_end=None,
            return_host="127.0.0.1",
            return_port=20_106,
            codec=TensorCodec.FP32,
            one_way_delay_ms=0,
            bandwidth_mbps=0,
            max_speculative_branches=2,
            max_speculative_branch_tokens=4,
            max_speculative_kv_bytes=60,
        )
        runner = _FakeLastStageRunner(config.spec)
        runner.begin(1)
        runner.active[1] = 3
        metrics: dict[int, dict[str, object]] = {
            1: {
                "frames": 1,
                "compute_ms": 0,
                "bytes_out": 0,
                "tokens": 3,
                "model_forward_calls": 1,
                "physical_batch_calls": 0,
                "physical_batch_items": 1,
                "max_physical_batch_size": 1,
            }
        }
        branches: dict[int, int] = {}
        for child in (2, 3):
            fork_stage_request(
                Frame(
                    FrameType.FORK,
                    0,
                    child,
                    0,
                    0,
                    0,
                    branch_request_payload(1),
                ),
                config,
                runner,
                metrics,
                branches,
                None,
            )
        frames = tuple(
            Frame(
                FrameType.ACTIVATION,
                int(TensorCodec.FP32),
                child,
                1,
                1,
                4,
                b"\x00" * 16,
            )
            for child in (2, 3)
        )
        for frame in frames:
            validate_speculative_kv_preflight(
                (frame,), config, runner, branches
            )
        with self.assertRaisesRegex(ValueError, "KV bytes"):
            validate_speculative_kv_preflight(frames, config, runner, branches)

    def test_same_request_prefill_chunks_are_deferred_before_next_step_validation(self) -> None:
        sender, upstream = socket.socketpair()
        runner = _FakePhysicalBatchLastStageRunner(
            StageModelSpec("fake", 15, 30, 30, 1)
        )
        runner.begin(11)
        config = SimpleNamespace(
            max_physical_batch_size=4,
            physical_batch_window_ms=50,
            codec=TensorCodec.FP32,
            sealed_wave_tokens=None,
            max_prefill_chunk_tokens=2,
        )
        request_metrics: dict[int, dict[str, object]] = {11: {"frames": 0}}
        pending: deque[Frame] = deque()
        hidden = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)
        payload = encode_tensor(hidden, TensorCodec.FP32)
        try:
            for step in (0, 1):
                send_frame(
                    sender,
                    FrameType.PREFILL,
                    11,
                    step=step,
                    token_count=2,
                    hidden_size=4,
                    flags=int(TensorCodec.FP32),
                    payload=payload,
                )
            first = recv_frame(upstream)
            collected = collect_compatible_activation_frames(
                first,
                upstream=upstream,
                pending_frames=pending,
                config=config,
                runner=runner,
                request_metrics=request_metrics,
                downstream=None,
            )
            self.assertEqual([frame.step for frame in collected], [0])
            self.assertEqual([frame.step for frame in pending], [1])

            # Once step zero commits, the deferred frame validates normally.
            request_metrics[11]["frames"] = 1
            runner.active[11] = 2
            validate_activation(pending[0], config, runner, request_metrics)
        finally:
            sender.close()
            upstream.close()

    def test_only_tree_verify_batches_require_exact_and_workspace_certification(self) -> None:
        feature_cases = (
            (False, (), 2),
            (True, (), 1),
            (True, ("exact-tree-verify-batching",), 1),
            (True, ("bounded-tree-verify-workspace",), 1),
            (
                True,
                (
                    "exact-tree-verify-batching",
                    "bounded-tree-verify-workspace",
                ),
                2,
            ),
        )
        for tree_leaves, features, expected_count in feature_cases:
            with self.subTest(tree_leaves=tree_leaves, features=features):
                sender, upstream = socket.socketpair()
                try:
                    runner = _FakePhysicalBatchLastStageRunner(
                        StageModelSpec("fake", 15, 30, 30, 1)
                    )
                    runner.executor_manifest = SimpleNamespace(features=features)
                    for request_id in (11, 22):
                        runner.begin(request_id)
                    config = SimpleNamespace(
                        max_physical_batch_size=4,
                        physical_batch_window_ms=50,
                        codec=TensorCodec.FP32,
                        sealed_wave_tokens=4,
                        max_prefill_chunk_tokens=4,
                        max_speculative_branch_tokens=16,
                        max_speculative_kv_bytes=1_024,
                    )
                    request_metrics: dict[int, dict[str, object]] = {
                        11: {"frames": 0},
                        22: {"frames": 0},
                    }
                    payload = encode_tensor(
                        torch.arange(8, dtype=torch.float32).reshape(1, 2, 4),
                        TensorCodec.FP32,
                    )
                    for request_id in (11, 22):
                        send_frame(
                            sender,
                            FrameType.VERIFY,
                            request_id,
                            step=0,
                            token_count=2,
                            hidden_size=4,
                            flags=int(TensorCodec.FP32),
                            payload=payload,
                        )
                    first = recv_frame(upstream)
                    branch_parents = {11: 1, 22: 1} if tree_leaves else {}
                    collected = collect_compatible_activation_frames(
                        first,
                        upstream=upstream,
                        pending_frames=deque(),
                        config=config,
                        runner=runner,
                        request_metrics=request_metrics,
                        branch_parents=branch_parents,
                        downstream=None,
                    )
                    self.assertEqual(len(collected), expected_count)
                    if expected_count == 1:
                        self.assertEqual(recv_frame(upstream).request_id, 22)
                    else:
                        self.assertEqual(
                            tuple(frame.request_id for frame in collected),
                            (11, 22),
                        )
                finally:
                    sender.close()
                    upstream.close()

    def test_stage_ingress_executes_compatible_requests_in_one_tensor_batch(self) -> None:
        return_listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        return_listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return_listener.bind(("127.0.0.1", 0))
        return_listener.listen(1)
        return_port = int(return_listener.getsockname()[1])
        stage_port = self.free_port()
        while stage_port == return_port:
            stage_port = self.free_port()
        config = StageProcessConfig(
            spec=StageModelSpec("fake", 15, 30, 30, 1),
            pipeline_id=778,
            listen_host="127.0.0.1",
            listen_port=stage_port,
            next_host=None,
            next_port=None,
            next_layer_end=None,
            return_host="127.0.0.1",
            return_port=return_port,
            codec=TensorCodec.FP32,
            one_way_delay_ms=0,
            bandwidth_mbps=0,
            connect_timeout_seconds=2,
            max_physical_batch_size=4,
            physical_batch_window_ms=50,
        )
        ready = threading.Event()
        metrics: queue.Queue[dict[str, object]] = queue.Queue()
        errors: list[BaseException] = []
        _FakePhysicalBatchLastStageRunner.batch_calls = []

        def run() -> None:
            try:
                run_stage_process(config, ready, metrics)
            except BaseException as error:
                errors.append(error)

        with patch(
            "distributed_runtime.stage.StageRunner",
            _FakePhysicalBatchLastStageRunner,
        ):
            worker = threading.Thread(target=run, daemon=True)
            worker.start()
            self.assertTrue(ready.wait(2))
            upstream = socket.create_connection(("127.0.0.1", stage_port), timeout=2)
            upstream.settimeout(2)
            bind_socket_deployment_generation(upstream, 0)
            send_frame(
                upstream,
                FrameType.HELLO,
                778,
                step=15,
                token_count=30,
                hidden_size=4,
                flags=int(TensorCodec.FP32),
            )
            direct_return, _ = return_listener.accept()
            direct_return.settimeout(2)
            self.assertEqual(recv_frame(upstream).frame_type, FrameType.READY)

            hidden = torch.arange(4, dtype=torch.float32).reshape(1, 1, 4)
            payload = encode_tensor(hidden, TensorCodec.FP32)
            for request_id in (11, 22):
                send_frame(upstream, FrameType.BEGIN, request_id)
                send_frame(
                    upstream,
                    FrameType.ACTIVATION,
                    request_id,
                    step=0,
                    token_count=1,
                    hidden_size=4,
                    flags=int(TensorCodec.FP32),
                    payload=payload,
                )

            returned = (recv_frame(direct_return), recv_frame(direct_return))
            self.assertEqual([frame.request_id for frame in returned], [11, 22])
            self.assertEqual(
                [decode_token(frame) for frame in returned],
                [111, 122],
            )
            self.assertEqual(
                _FakePhysicalBatchLastStageRunner.batch_calls,
                [(11, 22)],
            )

            for request_id in (11, 22):
                send_frame(upstream, FrameType.END, request_id)
            observations = {
                int(observation["request_id"]): observation
                for observation in (metrics.get(timeout=2), metrics.get(timeout=2))
            }
            for request_id in (11, 22):
                observation = observations[request_id]
                self.assertEqual(observation["model_forward_calls"], 1)
                self.assertEqual(observation["physical_batch_calls"], 1)
                self.assertEqual(observation["physical_batch_items"], 2)
                self.assertEqual(observation["max_physical_batch_size"], 2)

            send_frame(upstream, FrameType.SHUTDOWN, 778)
            worker.join(2)
            self.assertFalse(worker.is_alive())
            self.assertEqual(errors, [])
            direct_return.close()
            upstream.close()
        return_listener.close()

    def test_last_stage_handshake_cache_lifecycle_and_direct_token_return(self) -> None:
        return_listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        return_listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return_listener.bind(("127.0.0.1", 0))
        return_listener.listen(1)
        return_port = return_listener.getsockname()[1]
        stage_port = self.free_port()
        while stage_port == return_port:
            stage_port = self.free_port()

        config = StageProcessConfig(
            spec=StageModelSpec("fake", 15, 30, 30, 1),
            pipeline_id=777,
            listen_host="127.0.0.1",
            listen_port=stage_port,
            next_host=None,
            next_port=None,
            next_layer_end=None,
            return_host="127.0.0.1",
            return_port=return_port,
            codec=TensorCodec.FP32,
            one_way_delay_ms=0,
            bandwidth_mbps=0,
            connect_timeout_seconds=2,
        )
        ready = threading.Event()
        metrics: queue.Queue[dict[str, object]] = queue.Queue()
        errors: list[BaseException] = []

        def run() -> None:
            try:
                run_stage_process(config, ready, metrics)
            except BaseException as error:
                errors.append(error)

        with patch(
            "distributed_runtime.stage.StageRunner", _FakeCellWorkLastStageRunner
        ):
            worker = threading.Thread(target=run, daemon=True)
            worker.start()
            self.assertTrue(ready.wait(2), "stage did not bind its persistent listener")

            upstream = socket.create_connection(("127.0.0.1", stage_port), timeout=2)
            upstream.settimeout(2)
            bind_socket_deployment_generation(upstream, 0)
            send_frame(
                upstream,
                FrameType.HELLO,
                777,
                step=15,
                token_count=30,
                hidden_size=4,
                flags=int(TensorCodec.FP32),
            )
            direct_return, _ = return_listener.accept()
            direct_return.settimeout(2)
            self.assertEqual(recv_frame(upstream).frame_type, FrameType.READY)

            send_frame(upstream, FrameType.PING, 777, step=41)
            pong = recv_frame(direct_return)
            self.assertEqual(pong.frame_type, FrameType.PONG)
            self.assertEqual(pong.request_id, 777)
            self.assertEqual(pong.step, 41)

            request_id = 91
            send_frame(upstream, FrameType.BEGIN, request_id)
            hidden = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)
            payload = encode_tensor(hidden, TensorCodec.FP32)
            send_frame(
                upstream,
                FrameType.PREFILL,
                request_id,
                step=0,
                token_count=2,
                hidden_size=4,
                flags=int(TensorCodec.FP32),
                payload=payload,
            )
            prefill_ack = recv_frame(direct_return)
            self.assertEqual(prefill_ack.frame_type, FrameType.PREFILL_ACK)
            self.assertEqual(prefill_ack.request_id, request_id)
            self.assertEqual(prefill_ack.step, 0)

            verify_hidden = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
            send_frame(
                upstream,
                FrameType.VERIFY,
                request_id,
                step=1,
                token_count=3,
                hidden_size=4,
                flags=int(TensorCodec.FP32),
                payload=encode_tensor(verify_hidden, TensorCodec.FP32),
            )
            verify_hidden_2 = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)
            send_frame(
                upstream,
                FrameType.VERIFY,
                request_id,
                step=2,
                token_count=2,
                hidden_size=4,
                flags=int(TensorCodec.FP32),
                payload=encode_tensor(verify_hidden_2, TensorCodec.FP32),
            )
            verify_frame = recv_frame(direct_return)
            self.assertEqual(verify_frame.frame_type, FrameType.VERIFY_RESULT)
            self.assertEqual(verify_frame.step, 1)
            self.assertEqual(decode_verify_result(verify_frame), (42, 42, 42))
            verify_frame_2 = recv_frame(direct_return)
            self.assertEqual(verify_frame_2.frame_type, FrameType.VERIFY_RESULT)
            self.assertEqual(verify_frame_2.step, 2)
            self.assertEqual(decode_verify_result(verify_frame_2), (42, 42))

            send_frame(
                upstream,
                FrameType.TRUNCATE,
                request_id,
                token_count=3,
            )
            final_hidden = torch.arange(4, dtype=torch.float32).reshape(1, 1, 4)
            send_frame(
                upstream,
                FrameType.ACTIVATION,
                request_id,
                step=3,
                token_count=1,
                hidden_size=4,
                flags=int(TensorCodec.FP32),
                payload=encode_tensor(final_hidden, TensorCodec.FP32),
            )
            token_frame = recv_frame(direct_return)
            self.assertEqual(token_frame.request_id, request_id)
            self.assertEqual(token_frame.step, 3)
            self.assertEqual(decode_token(token_frame), 42)

            send_frame(upstream, FrameType.END, request_id)
            result = metrics.get(timeout=2)
            self.assertEqual(result["request_id"], request_id)
            self.assertEqual(result["frames"], 4)
            self.assertEqual(result["tokens"], 8)
            self.assertEqual(result["bytes_out"], HEADER_BYTES * 4 + 24)
            self.assertEqual(result["loader"], "fake-selective")
            self.assertEqual(result["parameter_bytes"], 16)
            self.assertEqual(
                result["cell_rank_work"],
                [
                    {
                        "rank": rank,
                        "device": "cpu",
                        "forwardCalls": 4,
                        "collectiveCalls": 20,
                        "tokensProcessed": 8,
                        "memory": {
                            "allocatedBytes": 0,
                            "reservedBytes": 0,
                            "peakAllocatedBytes": 0,
                        },
                    }
                    for rank in range(2)
                ],
            )

            send_frame(upstream, FrameType.SHUTDOWN, 777)
            worker.join(2)
            self.assertFalse(worker.is_alive(), "stage did not shut down")
            self.assertEqual(errors, [])

            direct_return.close()
            upstream.close()
        return_listener.close()

    def test_stage_contract_rejects_non_contiguous_plan_and_bad_sequence(self) -> None:
        config = StageProcessConfig(
            spec=StageModelSpec("fake", 10, 20, 30, 1),
            pipeline_id=123,
            listen_host="127.0.0.1",
            listen_port=20_001,
            next_host="127.0.0.1",
            next_port=20_002,
            next_layer_end=30,
            return_host="127.0.0.1",
            return_port=20_003,
            codec=TensorCodec.FP32,
            one_way_delay_ms=0,
            bandwidth_mbps=0,
        )
        validate_stage_config(config)
        runner = _FakeLastStageRunner(config.spec)
        runner.begin(5)
        metrics = {
            5: {
                "stage": 10.0,
                "frames": 1,
                "compute_ms": 0,
                "bytes_out": 0,
                "tokens": 2,
            }
        }
        frame = Frame(
            FrameType.ACTIVATION,
            int(TensorCodec.FP32),
            5,
            0,
            1,
            4,
            b"\x00" * 16,
        )
        with self.assertRaisesRegex(ValueError, "step mismatch"):
            validate_activation(frame, config, runner, metrics)

        sealed = StageProcessConfig(
            **{
                **config.__dict__,
                "sealed_wave_tokens": 4,
                "max_prefill_chunk_tokens": 8,
            }
        )
        oversized_verify = Frame(
            FrameType.VERIFY,
            int(TensorCodec.FP32),
            5,
            1,
            5,
            4,
            b"",
        )
        with self.assertRaisesRegex(ValueError, "exceeds sealed_wave_tokens"):
            validate_activation(oversized_verify, sealed, runner, metrics)

        hello = Frame(
            FrameType.HELLO,
            int(TensorCodec.FP32),
            123,
            11,
            20,
            4,
            b"",
        )
        with self.assertRaisesRegex(ValueError, "layer range mismatch"):
            validate_hello(hello, config, 4)

        wrong_pipeline = Frame(
            FrameType.HELLO,
            int(TensorCodec.FP32),
            124,
            10,
            20,
            4,
            b"",
        )
        with self.assertRaisesRegex(ValueError, "pipeline identity mismatch"):
            validate_hello(wrong_pipeline, config, 4)

        wrong_generation = Frame(
            FrameType.HELLO,
            int(TensorCodec.FP32),
            123,
            10,
            20,
            4,
            b"",
            deployment_generation=1,
        )
        with self.assertRaisesRegex(ValueError, "deployment generation mismatch"):
            validate_hello(wrong_generation, config, 4)

        wrong_route = Frame(
            FrameType.HELLO,
            int(TensorCodec.FP32),
            123,
            10,
            20,
            4,
            b"",
            route_digest=route_identity_digest("other-route"),
        )
        with self.assertRaisesRegex(ValueError, "route identity mismatch"):
            validate_hello(wrong_route, config, 4)

        replayed_hello = Frame(
            FrameType.HELLO,
            int(TensorCodec.FP32),
            123,
            10,
            20,
            4,
            b"",
            route_digest=route_identity_digest("static"),
            sequence=1,
        )
        with self.assertRaisesRegex(ValueError, "HELLO sequence must be zero"):
            validate_hello(replayed_hello, config, 4)

        invalid = StageProcessConfig(
            **{
                **config.__dict__,
                "next_layer_end": 20,
            }
        )
        with self.assertRaisesRegex(ValueError, "extend the contiguous"):
            validate_stage_config(invalid)
        for invalid_id in (-1, 1 << 64, True):
            with self.subTest(pipeline_id=invalid_id):
                invalid_identity = StageProcessConfig(
                    **{**config.__dict__, "pipeline_id": invalid_id}
                )
                with self.assertRaisesRegex(ValueError, "pipeline_id"):
                    validate_stage_config(invalid_identity)

    def test_downstream_error_and_eof_are_relayed_but_planned_shutdown_is_quiet(self) -> None:
        for mode in ("error", "eof"):
            with self.subTest(mode=mode):
                stage_upstream, root = socket.socketpair()
                stage_downstream, child = socket.socketpair()
                stopping = threading.Event()
                failed = threading.Event()
                failures: list[BaseException] = []
                monitor = threading.Thread(
                    target=monitor_downstream_control,
                    args=(
                        stage_downstream,
                        stage_upstream,
                        444,
                        stopping,
                        failed,
                        failures,
                        threading.Lock(),
                    ),
                    daemon=True,
                )
                monitor.start()
                if mode == "error":
                    send_frame(child, FrameType.ERROR, 987, payload=b"child exploded")
                else:
                    child.close()
                relayed = recv_frame(root)
                self.assertEqual(relayed.frame_type, FrameType.ERROR)
                self.assertEqual(relayed.request_id, 987 if mode == "error" else 444)
                if mode == "error":
                    self.assertEqual(relayed.payload, b"child exploded")
                else:
                    self.assertIn(b"control channel failed", relayed.payload)
                self.assertTrue(failed.wait(1))
                self.assertEqual(len(failures), 1)
                monitor.join(1)
                child.close()
                stage_downstream.close()
                stage_upstream.close()
                root.close()

        stage_upstream, root = socket.socketpair()
        stage_downstream, child = socket.socketpair()
        stopping = threading.Event()
        stopping.set()
        monitor = threading.Thread(
            target=monitor_downstream_control,
            args=(
                stage_downstream,
                stage_upstream,
                444,
                stopping,
                threading.Event(),
                [],
                threading.Lock(),
            ),
            daemon=True,
        )
        monitor.start()
        child.close()
        monitor.join(1)
        root.settimeout(0.05)
        with self.assertRaises(TimeoutError):
            root.recv(1)
        stage_downstream.close()
        stage_upstream.close()
        root.close()

    def test_forward_shutdown_half_closes_then_waits_for_downstream_eof(self) -> None:
        stage_upstream, root = socket.socketpair()
        stage_downstream, child = socket.socketpair()
        stopping = threading.Event()
        stopping.set()
        monitor = threading.Thread(
            target=monitor_downstream_control,
            args=(
                stage_downstream,
                stage_upstream,
                444,
                stopping,
                threading.Event(),
                [],
                threading.Lock(),
            ),
            daemon=True,
        )
        events: list[str] = []

        def consume_shutdown() -> None:
            frame = recv_frame(child)
            self.assertEqual(frame.frame_type, FrameType.SHUTDOWN)
            self.assertEqual(frame.request_id, 444)
            events.append("shutdown-received")
            self.assertEqual(child.recv(1), b"")
            events.append("write-eof-received")
            child.close()

        consumer = threading.Thread(target=consume_shutdown, daemon=True)
        monitor.start()
        consumer.start()
        try:
            forward_shutdown_and_wait(
                stage_downstream,
                444,
                monitor,
                timeout_seconds=1.0,
            )
            events.append("forward-returned")
            consumer.join(1)
            self.assertFalse(consumer.is_alive())
            self.assertFalse(monitor.is_alive())
            self.assertEqual(
                events,
                [
                    "shutdown-received",
                    "write-eof-received",
                    "forward-returned",
                ],
            )
            root.settimeout(0.05)
            with self.assertRaises(TimeoutError):
                root.recv(1)
        finally:
            child.close()
            stage_downstream.close()
            stage_upstream.close()
            root.close()

    def test_forward_shutdown_timeout_is_bounded_and_fail_closed(self) -> None:
        stage_downstream, child = socket.socketpair()
        control = _StubbornControlThread()
        try:
            with self.assertRaisesRegex(TimeoutError, "acknowledge SHUTDOWN"):
                forward_shutdown_and_wait(
                    stage_downstream,
                    555,
                    control,
                    timeout_seconds=0.0,
                )
            self.assertEqual(control.join_timeouts, [0.0])
            frame = recv_frame(child)
            self.assertEqual(frame.frame_type, FrameType.SHUTDOWN)
            self.assertEqual(frame.request_id, 555)
            self.assertEqual(child.recv(1), b"")
        finally:
            child.close()
            stage_downstream.close()

    def test_metrics_failure_cannot_suppress_stage_error(self) -> None:
        return_listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        return_listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return_listener.bind(("127.0.0.1", 0))
        return_listener.listen(1)
        return_port = int(return_listener.getsockname()[1])
        stage_port = self.free_port()
        config = StageProcessConfig(
            spec=StageModelSpec("fake", 15, 30, 30, 1),
            pipeline_id=999,
            listen_host="127.0.0.1",
            listen_port=stage_port,
            next_host=None,
            next_port=None,
            next_layer_end=None,
            return_host="127.0.0.1",
            return_port=return_port,
            codec=TensorCodec.FP32,
            one_way_delay_ms=0,
            bandwidth_mbps=0,
            connect_timeout_seconds=2,
        )
        ready = threading.Event()
        errors: list[BaseException] = []

        def run() -> None:
            try:
                run_stage_process(config, ready, _ExplodingMetricSink())
            except BaseException as error:
                errors.append(error)

        with patch("distributed_runtime.stage.StageRunner", _FailingLastStageRunner):
            worker = threading.Thread(target=run, daemon=True)
            worker.start()
            self.assertTrue(ready.wait(2))
            upstream = socket.create_connection(("127.0.0.1", stage_port), timeout=2)
            upstream.settimeout(2)
            bind_socket_deployment_generation(upstream, 0)
            send_frame(
                upstream,
                FrameType.HELLO,
                999,
                step=15,
                token_count=30,
                hidden_size=4,
                flags=int(TensorCodec.FP32),
            )
            direct_return, _ = return_listener.accept()
            direct_return.settimeout(2)
            self.assertEqual(recv_frame(upstream).frame_type, FrameType.READY)
            send_frame(upstream, FrameType.BEGIN, 7)
            hidden = torch.zeros((1, 1, 4), dtype=torch.float32)
            send_frame(
                upstream,
                FrameType.ACTIVATION,
                7,
                token_count=1,
                hidden_size=4,
                flags=int(TensorCodec.FP32),
                payload=encode_tensor(hidden, TensorCodec.FP32),
            )
            reported = recv_frame(upstream)
            self.assertEqual(reported.frame_type, FrameType.ERROR)
            self.assertEqual(reported.request_id, 999)
            self.assertIn(b"synthetic compute failure", reported.payload)
            worker.join(2)
            self.assertFalse(worker.is_alive())
            self.assertEqual(len(errors), 1)
            direct_return.close()
            upstream.close()
        return_listener.close()

    def test_json_metric_sink_is_best_effort(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            # Serializing object() fails before this directory path can be opened.
            JsonMetricSink(Path(temporary)).put({"bad": object()})

    @staticmethod
    def free_port() -> int:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])
        finally:
            sock.close()


class _FakeLastStageRunner:
    hidden_size = 4
    parameter_bytes = 16
    loader = "fake-selective"

    def __init__(self, spec: StageModelSpec) -> None:
        self.spec = spec
        self.active: dict[int, int] = {}

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
        if request_id not in self.active:
            raise ValueError("inactive request")
        if not 0 <= token_count <= self.active[request_id]:
            raise ValueError("invalid truncate")
        self.active[request_id] = token_count

    def request_cache_bytes(self, request_id: int) -> int:
        return self.sequence_length(request_id) * 8

    def project_request_cache_bytes(
        self, request_id: int, additional_tokens: int
    ) -> int:
        return (self.sequence_length(request_id) + additional_tokens) * 8

    def fork_request(
        self,
        child_request_id: int,
        parent_request_id: int,
        *,
        max_cache_bytes: int,
    ) -> int:
        if parent_request_id not in self.active:
            raise ValueError("inactive parent request")
        if child_request_id in self.active:
            raise ValueError("active child request")
        copied_bytes = self.active[parent_request_id] * 8
        if copied_bytes > max_cache_bytes:
            raise ValueError("preflight byte budget")
        self.active[child_request_id] = self.active[parent_request_id]
        return copied_bytes

    def promote_request(self, parent_request_id: int, child_request_id: int) -> None:
        if parent_request_id not in self.active or child_request_id not in self.active:
            raise ValueError("inactive request")
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
            return hidden, None
        if token_mode == "all":
            return hidden, tuple(42 for _ in range(int(hidden.shape[1])))
        return hidden, 42

    def project_sampling_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        rows = int(hidden.shape[1])
        return torch.tensor([[[1.0, 2.0, 3.0]] * rows], dtype=torch.float32)


class _FakeCellWorkLastStageRunner(_FakeLastStageRunner):
    def __init__(self, spec: StageModelSpec) -> None:
        super().__init__(spec)
        self.member_work_reports: dict[int, dict[str, object]] = {}

    def forward_hidden(
        self,
        request_id: int,
        hidden: torch.Tensor,
        *,
        token_mode: str = "last",
    ) -> tuple[torch.Tensor, int | tuple[int, ...] | None]:
        result = super().forward_hidden(request_id, hidden, token_mode=token_mode)
        input_tokens = int(hidden.shape[1])
        previous = self.member_work_reports.get(0)
        forward_calls = 1 if previous is None else int(previous["forwardCalls"]) + 1
        collective_calls = (
            5 if previous is None else int(previous["collectiveCalls"]) + 5
        )
        tokens_processed = (
            input_tokens
            if previous is None
            else int(previous["tokensProcessed"]) + input_tokens
        )
        self.member_work_reports = {
            rank: {
                "rank": rank,
                "device": "cpu",
                "computeDtype": "float32",
                "collectiveBackend": "gloo",
                "forwardCalls": forward_calls,
                "collectiveCalls": collective_calls,
                "tokensProcessed": tokens_processed,
                "memory": {
                    "allocatedBytes": 0,
                    "reservedBytes": 0,
                    "peakAllocatedBytes": 0,
                },
            }
            for rank in range(2)
        }
        return result


class _FailingLastStageRunner(_FakeLastStageRunner):
    def forward_hidden(
        self,
        request_id: int,
        hidden: torch.Tensor,
        *,
        token_mode: str = "last",
    ) -> tuple[torch.Tensor, int | None]:
        raise RuntimeError("synthetic compute failure")


class _FakePhysicalBatchLastStageRunner(_FakeLastStageRunner):
    batch_calls: list[tuple[int, ...]] = []

    def physical_batch_key(
        self,
        request_id: int,
        *,
        token_count: int,
        token_mode: str,
    ) -> tuple[int, int, str]:
        return self.sequence_length(request_id), token_count, token_mode

    def forward_hidden_batch(
        self,
        request_ids: tuple[int, ...],
        hidden_states: tuple[torch.Tensor, ...],
        *,
        token_mode: str = "last",
    ) -> tuple[tuple[torch.Tensor, int], ...]:
        self.batch_calls.append(request_ids)
        results = []
        for request_id, hidden in zip(request_ids, hidden_states, strict=True):
            self.active[request_id] += int(hidden.shape[1])
            results.append((hidden, 100 + request_id))
        return tuple(results)


class _ExplodingMetricSink:
    def put(self, value: dict[str, object]) -> None:
        raise OSError("synthetic metrics failure")


class _StubbornControlThread:
    def __init__(self) -> None:
        self.join_timeouts: list[float | None] = []

    def join(self, timeout: float | None = None) -> None:
        self.join_timeouts.append(timeout)

    def is_alive(self) -> bool:
        return True


if __name__ == "__main__":
    unittest.main()
