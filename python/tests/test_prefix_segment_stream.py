from __future__ import annotations

from dataclasses import replace
import random
import unittest

import distributed_runtime.prefix_segment_stream as wire
from distributed_runtime.prefix_segment_schedule import (
    PrefixSegmentSchedule,
    plan_prefix_segment_schedule,
)
from distributed_runtime.prefix_segment_stream import (
    PrefixSegmentCodecError,
    PrefixSegmentStreamEncoder,
    PrefixSegmentStreamError,
    PrefixSegmentStreamGuard,
    PrefixSegmentStreamLimits,
    PrefixSegmentStreamState,
    build_prefix_segment_stream_manifest,
    build_prefix_segment_stream_manifest_from_records,
    decode_prefix_segment_frame,
    encode_prefix_segment_frame,
    validate_prefix_segment_stream_manifest,
)
from distributed_runtime.prefix_token_arena import (
    PrefixTokenArena,
    PrefixWaveChunking,
    build_prefix_token_arena,
)
from distributed_runtime.prefix_wave_topology import build_prefix_wave_topology


IDENTITY = {
    "nonce": 0x0102_0304_0506_0708,
    "parent_request_id": 700_000,
    "step": 19,
}

PATHS = (
    (10, 20, 30, 40),
    (10, 20, 30, 41),
    (10, 20, 31, 50),
    (10, 21, 60, 61),
    (11, 70, 71, 72),
)


def _arena(
    paths=PATHS,
    *,
    shared=(900, 901),
    identity: dict[str, int] = IDENTITY,
    chunking: PrefixWaveChunking = PrefixWaveChunking(1, 1, 1),
    first_lane_request_id: int = 800_000,
) -> tuple[PrefixSegmentSchedule, PrefixTokenArena]:
    schedule = plan_prefix_segment_schedule(
        paths,
        shared_prefix_tokens=shared,
        max_paths=64,
        max_depth=32,
        max_nodes=4_096,
    )
    topology = build_prefix_wave_topology(
        schedule,
        lane_request_ids=tuple(
            range(first_lane_request_id, first_lane_request_id + len(schedule.leaves))
        ),
        **identity,
    )
    arena = build_prefix_token_arena(
        schedule,
        topology,
        chunking=chunking,
        **identity,
    )
    return schedule, arena


def _manifest(
    arena: PrefixTokenArena | None = None,
    *,
    codec_id: int = 1,
    hidden_size: int = 16,
):
    if arena is None:
        _, arena = _arena()
    return build_prefix_segment_stream_manifest(
        arena,
        expected_nonce=arena.nonce,
        expected_wave_digest=arena.wave_digest,
        codec_id=codec_id,
        hidden_size=hidden_size,
    )


def _payload(record_id: int, size: int = 37) -> bytes:
    return bytes((record_id * 31 + offset * 17) & 0xFF for offset in range(size))


def _packets(manifest):
    encoder = PrefixSegmentStreamEncoder(manifest)
    return tuple(
        encoder.encode(record, _payload(record.record_id))
        for record in manifest.records
    )


def _replace_header_field(packet: bytes, index: int, value: object) -> bytes:
    fields = list(wire._FIXED_HEADER.unpack_from(packet))
    fields[index] = value
    return wire._FIXED_HEADER.pack(*fields) + packet[wire._FIXED_HEADER.size :]


def _xor_at(packet: bytes, offset: int, mask: int = 1) -> bytes:
    changed = bytearray(packet)
    changed[offset] ^= mask
    return bytes(changed)


class PrefixSegmentCodecTests(unittest.TestCase):
    def test_round_trip_binds_every_field_and_owns_payload_snapshot(self) -> None:
        manifest = _manifest()
        record = manifest.records[0]
        source = bytearray(_payload(0))
        expected = bytes(source)
        packet = encode_prefix_segment_frame(manifest, record, source)
        source[:] = b"x" * len(source)

        frame = decode_prefix_segment_frame(packet, manifest)
        self.assertEqual(frame.nonce, manifest.nonce)
        self.assertEqual(frame.wave_digest, manifest.wave_digest)
        self.assertEqual(frame.contract_digest, manifest.contract_digest)
        self.assertEqual(frame.record_index, 0)
        self.assertEqual(frame.record_total, len(manifest.records))
        self.assertEqual(frame.record, record)
        self.assertEqual(frame.codec_id, manifest.codec_id)
        self.assertEqual(
            frame.shape,
            (record.segment_count, record.slice_tokens, manifest.hidden_size),
        )
        self.assertEqual(frame.payload, expected)
        self.assertEqual(len(frame.payload_digest), 32)
        self.assertEqual(len(frame.binding_digest), 32)
        self.assertFalse(frame.descriptor_contains_tokens)

    def test_reusable_encoder_and_guard_complete_exact_contiguous_stream(self) -> None:
        manifest = _manifest()
        packets = _packets(manifest)
        guard = PrefixSegmentStreamGuard(manifest)
        observed: list[tuple[int, bytes]] = []

        for index, packet in enumerate(packets):
            result = guard.accept(
                packet,
                lambda frame: observed.append((frame.record_index, frame.payload)),
            )
            self.assertEqual(result.frame.record_index, index)
            self.assertEqual(result.completed_record_count, index + 1)
            self.assertEqual(result.complete, index + 1 == len(packets))

        self.assertEqual(
            observed,
            [(index, _payload(index)) for index in range(len(packets))],
        )
        snapshot = guard.snapshot()
        self.assertEqual(snapshot.state, PrefixSegmentStreamState.COMPLETE)
        self.assertEqual(snapshot.next_record_index, len(packets))
        self.assertEqual(snapshot.completed_record_count, len(packets))
        self.assertTrue(snapshot.mutation_started)
        self.assertFalse(snapshot.fallback_allowed)

    def test_wire_descriptor_is_stable_under_token_relabeling(self) -> None:
        left_paths = ((1, 2, 3), (1, 2, 4), (1, 5, 6))
        right_paths = (
            (1_001, 2_002, 3_003),
            (1_001, 2_002, 4_004),
            (1_001, 5_005, 6_006),
        )
        chunking = PrefixWaveChunking(1, 2, 4)
        _, left_arena = _arena(
            left_paths,
            shared=(99, 98),
            chunking=chunking,
        )
        _, right_arena = _arena(
            right_paths,
            shared=(9_999, 9_998),
            chunking=chunking,
        )
        left = _manifest(left_arena)
        right = _manifest(right_arena)

        self.assertNotEqual(
            left_arena.token_ids_by_canonical_node,
            right_arena.token_ids_by_canonical_node,
        )
        self.assertEqual(left, right)
        self.assertNotIn("token", wire.PrefixSegmentFrame.__dataclass_fields__)
        self.assertNotIn("token", wire.PrefixSegmentStreamManifest.__dataclass_fields__)
        for left_record, right_record in zip(left.records, right.records):
            payload = _payload(left_record.record_id)
            self.assertEqual(
                encode_prefix_segment_frame(left, left_record, payload),
                encode_prefix_segment_frame(right, right_record, payload),
            )

    def test_codec_shape_and_wave_each_create_a_distinct_contract(self) -> None:
        _, arena = _arena()
        base = _manifest(arena, codec_id=1, hidden_size=16)
        other_codec = _manifest(arena, codec_id=2, hidden_size=16)
        other_shape = _manifest(arena, codec_id=1, hidden_size=32)
        other_identity = {**IDENTITY, "nonce": IDENTITY["nonce"] + 1}
        _, other_arena = _arena(identity=other_identity)
        other_wave = _manifest(other_arena, codec_id=1, hidden_size=16)

        self.assertEqual(
            len(
                {
                    base.contract_digest,
                    other_codec.contract_digest,
                    other_shape.contract_digest,
                    other_wave.contract_digest,
                }
            ),
            4,
        )
        packet = encode_prefix_segment_frame(
            base, base.records[0], _payload(0)
        )
        for hostile in (other_codec, other_shape, other_wave):
            with self.subTest(contract=hostile.contract_digest.hex()[:8]):
                with self.assertRaises(PrefixSegmentCodecError):
                    decode_prefix_segment_frame(packet, hostile)

    def test_stale_arena_manifest_and_record_are_rejected(self) -> None:
        _, arena = _arena()
        with self.assertRaisesRegex(PrefixSegmentCodecError, "wave digest"):
            build_prefix_segment_stream_manifest(
                replace(arena, wave_digest=b"w" * 32),
                expected_nonce=arena.nonce,
                expected_wave_digest=arena.wave_digest,
                codec_id=1,
                hidden_size=16,
            )

        with self.assertRaisesRegex(PrefixSegmentCodecError, "independently supplied"):
            build_prefix_segment_stream_manifest(
                replace(arena, nonce=arena.nonce + 1),
                expected_nonce=arena.nonce,
                expected_wave_digest=arena.wave_digest,
                codec_id=1,
                hidden_size=16,
            )
        with self.assertRaisesRegex(PrefixSegmentCodecError, "independently supplied"):
            build_prefix_segment_stream_manifest(
                arena,
                expected_nonce=arena.nonce + 1,
                expected_wave_digest=arena.wave_digest,
                codec_id=1,
                hidden_size=16,
            )

        changed_record = replace(arena.records[0], group_id=999)
        with self.assertRaisesRegex(
            PrefixSegmentCodecError, "group ids|structural descriptor"
        ):
            build_prefix_segment_stream_manifest(
                replace(arena, records=(changed_record, *arena.records[1:])),
                expected_nonce=arena.nonce,
                expected_wave_digest=arena.wave_digest,
                codec_id=1,
                hidden_size=16,
            )

        manifest = _manifest(arena)
        with self.assertRaisesRegex(PrefixSegmentCodecError, "stale"):
            validate_prefix_segment_stream_manifest(
                replace(manifest, contract_digest=b"c" * 32)
            )
        with self.assertRaisesRegex(PrefixSegmentCodecError, "canonical"):
            validate_prefix_segment_stream_manifest(
                replace(
                    manifest,
                    shapes=((1, 1, manifest.hidden_size + 1), *manifest.shapes[1:]),
                )
            )
        with self.assertRaisesRegex(PrefixSegmentCodecError, "sealed manifest"):
            encode_prefix_segment_frame(
                manifest,
                replace(manifest.records[0], group_id=999),
                _payload(0),
            )

    def test_remote_token_free_builder_is_byte_exact_with_root_builder(self) -> None:
        _, arena = _arena(
            chunking=PrefixWaveChunking(
                max_slice_tokens=2,
                max_segments_per_record=3,
                max_nodes_per_record=6,
            )
        )
        root = _manifest(arena, codec_id=4, hidden_size=96)
        remote = build_prefix_segment_stream_manifest_from_records(
            tuple(arena.records),
            nonce=arena.nonce,
            wave_digest=arena.wave_digest,
            topology_digest=arena.topology_digest,
            chunking=arena.chunking,
            codec_id=4,
            hidden_size=96,
        )

        self.assertEqual(remote, root)
        self.assertEqual(remote.topology_digest, arena.topology_digest)
        self.assertEqual(remote.chunking, arena.chunking)
        self.assertNotIn("token", remote.__dataclass_fields__)
        self.assertNotIn(
            "token",
            build_prefix_segment_stream_manifest_from_records.__annotations__,
        )

    def test_remote_builder_and_validator_reject_cross_object_swaps(self) -> None:
        _, base_arena = _arena(
            chunking=PrefixWaveChunking(2, 3, 6)
        )
        base = _manifest(base_arena)

        other_paths = (
            (100, 200, 300, 400),
            (100, 201, 301, 401),
            (101, 202, 302, 402),
        )
        _, other_records_arena = _arena(
            other_paths,
            chunking=base_arena.chunking,
        )
        other_records = _manifest(other_records_arena)

        other_chunking = PrefixWaveChunking(1, 1, 1)
        _, other_chunk_arena = _arena(chunking=other_chunking)
        other_chunk = _manifest(other_chunk_arena)

        other_identity = {**IDENTITY, "nonce": IDENTITY["nonce"] + 9}
        _, other_topology_arena = _arena(
            identity=other_identity,
            chunking=base_arena.chunking,
        )
        other_topology = _manifest(other_topology_arena)

        remote_swaps = (
            {
                "records": other_records.records,
                "topology_digest": base.topology_digest,
                "chunking": base.chunking,
            },
            {
                "records": base.records,
                "topology_digest": base.topology_digest,
                "chunking": other_chunk.chunking,
            },
            {
                "records": base.records,
                "topology_digest": other_topology.topology_digest,
                "chunking": base.chunking,
            },
        )
        for swap in remote_swaps:
            with self.subTest(swap=swap):
                with self.assertRaises(PrefixSegmentCodecError):
                    build_prefix_segment_stream_manifest_from_records(
                        swap["records"],
                        nonce=base.nonce,
                        wave_digest=base.wave_digest,
                        topology_digest=swap["topology_digest"],
                        chunking=swap["chunking"],
                        codec_id=base.codec_id,
                        hidden_size=base.hidden_size,
                    )

        # A contract digest copied from the object that owns the substituted
        # field cannot rescue a hybrid manifest: wave recomputation happens
        # independently before the contract digest is trusted.
        hostile_manifests = (
            replace(
                base,
                records=other_records.records,
                shapes=other_records.shapes,
                contract_digest=other_records.contract_digest,
            ),
            replace(
                base,
                chunking=other_chunk.chunking,
                records=other_chunk.records,
                shapes=other_chunk.shapes,
                contract_digest=other_chunk.contract_digest,
            ),
            replace(
                base,
                topology_digest=other_topology.topology_digest,
                contract_digest=other_topology.contract_digest,
            ),
        )
        for hostile in hostile_manifests:
            with self.subTest(contract=hostile.contract_digest.hex()[:8]):
                with self.assertRaises(PrefixSegmentCodecError):
                    validate_prefix_segment_stream_manifest(hostile)

    def test_limits_and_noncontiguous_buffers_fail_before_copy_or_mutation(self) -> None:
        manifest = _manifest()
        packet = encode_prefix_segment_frame(
            manifest, manifest.records[0], b"12345"
        )
        limits = PrefixSegmentStreamLimits(
            max_payload_bytes=4,
            max_packet_bytes=1_024,
            max_records=64,
            max_hidden_size=64,
        )
        with self.assertRaisesRegex(PrefixSegmentCodecError, "payload length"):
            decode_prefix_segment_frame(packet, manifest, limits=limits)
        with self.assertRaisesRegex(PrefixSegmentCodecError, "at most"):
            encode_prefix_segment_frame(
                manifest,
                manifest.records[0],
                b"12345",
                limits=limits,
            )
        with self.assertRaisesRegex(PrefixSegmentCodecError, "contiguous"):
            decode_prefix_segment_frame(memoryview(bytearray(packet))[::2], manifest)


class PrefixSegmentHostileWireTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = _manifest()
        self.record = self.manifest.records[0]
        self.packet = encode_prefix_segment_frame(
            self.manifest, self.record, _payload(0)
        )

    def assertRejected(self, packet: bytes, pattern: str | None = None) -> None:
        context = (
            self.assertRaisesRegex(PrefixSegmentCodecError, pattern)
            if pattern is not None
            else self.assertRaises(PrefixSegmentCodecError)
        )
        with context:
            decode_prefix_segment_frame(packet, self.manifest)

    def test_identity_codec_total_and_record_metadata_substitutions_fail(self) -> None:
        hostile = (
            (_replace_header_field(self.packet, 0, b"EVIL"), "magic"),
            (_replace_header_field(self.packet, 1, 2), "version"),
            (_replace_header_field(self.packet, 2, 1), "reserved"),
            (_replace_header_field(self.packet, 3, 6), "codec"),
            (_replace_header_field(self.packet, 4, self.manifest.nonce + 1), "nonce"),
            (_replace_header_field(self.packet, 5, b"w" * 32), "wave digest"),
            (_replace_header_field(self.packet, 6, b"c" * 32), "contract digest"),
            (_replace_header_field(self.packet, 8, len(self.manifest.records) + 1), "total"),
            (_replace_header_field(self.packet, 9, self.record.frontier_index + 1), "metadata"),
            (_replace_header_field(self.packet, 10, self.record.group_id + 1), "metadata"),
            (_replace_header_field(self.packet, 11, self.record.slice_offset + 1), "metadata"),
            (_replace_header_field(self.packet, 16, 1), "reserved"),
        )
        for packet, pattern in hostile:
            with self.subTest(pattern=pattern):
                self.assertRejected(packet, pattern)

    def test_shape_segment_node_payload_and_digests_are_all_sealed(self) -> None:
        shape_offset = wire._FIXED_HEADER.size
        segment_offset = shape_offset + len(self.manifest.shapes[0]) * 4
        node_offset = segment_offset + self.record.segment_count * 4
        payload_offset = node_offset + self.record.node_count * 4
        hostile = (
            (_xor_at(self.packet, shape_offset), "shape"),
            (_xor_at(self.packet, segment_offset), "segment ids"),
            (_xor_at(self.packet, node_offset), "node ids"),
            (_xor_at(self.packet, payload_offset), "payload SHA-256"),
            (_replace_header_field(self.packet, 18, b"p" * 32), "payload SHA-256"),
            (_replace_header_field(self.packet, 19, b"b" * 32), "binding SHA-256"),
        )
        for packet, pattern in hostile:
            with self.subTest(pattern=pattern):
                self.assertRejected(packet, pattern)

    def test_truncation_trailing_bytes_and_hostile_lengths_fail_closed(self) -> None:
        self.assertRejected(self.packet[:-1], "truncated")
        self.assertRejected(self.packet + b"\x00", "trailing bytes")
        self.assertRejected(
            _replace_header_field(self.packet, 17, len(_payload(0)) + 1),
            "truncated",
        )
        self.assertRejected(
            _replace_header_field(self.packet, 17, len(_payload(0)) - 1),
            "trailing bytes",
        )
        self.assertRejected(
            _replace_header_field(self.packet, 13, 0xFFFF_FFFF),
            "metadata",
        )
        self.assertRejected(self.packet[: wire._FIXED_HEADER.size - 1], "truncated")

    def test_every_single_bit_position_is_detected(self) -> None:
        # Every wire byte is either a checked canonical field, covered by the
        # frame-binding digest, or covered by the payload digest.  There is no
        # unbound padding/trailer where a one-bit mutation can hide.
        for offset in range(len(self.packet)):
            with self.subTest(offset=offset):
                self.assertRejected(_xor_at(self.packet, offset))


class PrefixSegmentStreamStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = _manifest()
        self.packets = _packets(self.manifest)
        self.assertGreaterEqual(len(self.packets), 3)

    def test_invalid_first_frame_and_initial_gap_allow_fallback_without_callback(self) -> None:
        cases = (
            _xor_at(self.packets[0], len(self.packets[0]) - 1),
            self.packets[1],
        )
        for packet in cases:
            with self.subTest(kind="corrupt" if packet != self.packets[1] else "gap"):
                guard = PrefixSegmentStreamGuard(self.manifest)
                calls: list[int] = []
                with self.assertRaises(PrefixSegmentStreamError) as caught:
                    guard.accept(packet, lambda frame: calls.append(frame.record_index))
                self.assertTrue(caught.exception.fallback_allowed)
                self.assertFalse(caught.exception.route_fatal)
                self.assertEqual(calls, [])
                snapshot = guard.snapshot()
                self.assertEqual(snapshot.state, PrefixSegmentStreamState.ABORTED)
                self.assertFalse(snapshot.mutation_started)
                self.assertTrue(snapshot.fallback_allowed)
                with self.assertRaises(PrefixSegmentStreamError) as retry:
                    guard.accept(self.packets[0], lambda frame: None)
                self.assertTrue(retry.exception.fallback_allowed)

    def test_duplicate_gap_modified_payload_and_wave_swap_after_mutation_are_fatal(self) -> None:
        other_identity = {**IDENTITY, "nonce": IDENTITY["nonce"] + 1}
        _, other_arena = _arena(identity=other_identity)
        other_manifest = _manifest(other_arena)
        other_packet = encode_prefix_segment_frame(
            other_manifest,
            other_manifest.records[1],
            _payload(1),
        )
        cases = (
            ("duplicate", self.packets[0]),
            ("gap", self.packets[2]),
            ("modified", _xor_at(self.packets[1], len(self.packets[1]) - 1)),
            ("wave-swap", other_packet),
        )
        for name, hostile in cases:
            with self.subTest(name=name):
                guard = PrefixSegmentStreamGuard(self.manifest)
                guard.accept(self.packets[0], lambda frame: None)
                with self.assertRaises(PrefixSegmentStreamError) as caught:
                    guard.accept(hostile, lambda frame: None)
                self.assertTrue(caught.exception.route_fatal)
                self.assertFalse(caught.exception.fallback_allowed)
                snapshot = guard.snapshot()
                self.assertEqual(snapshot.state, PrefixSegmentStreamState.FATAL)
                self.assertTrue(snapshot.mutation_started)
                self.assertEqual(snapshot.completed_record_count, 1)

    def test_callback_baseexception_is_fatal_at_the_first_mutation_boundary(self) -> None:
        guard = PrefixSegmentStreamGuard(self.manifest)

        def interrupted(_frame):
            raise KeyboardInterrupt("backend may already have mutated")

        with self.assertRaises(PrefixSegmentStreamError) as caught:
            guard.accept(self.packets[0], interrupted)
        self.assertTrue(caught.exception.route_fatal)
        self.assertEqual(caught.exception.phase, "mutation")
        self.assertEqual(caught.exception.completed_record_count, 0)
        snapshot = guard.snapshot()
        self.assertEqual(snapshot.state, PrefixSegmentStreamState.FATAL)
        self.assertTrue(snapshot.mutation_started)
        self.assertEqual(snapshot.completed_record_count, 0)

    def test_invalid_callback_is_safe_before_mutation_and_fatal_after_one_record(self) -> None:
        fresh = PrefixSegmentStreamGuard(self.manifest)
        with self.assertRaises(PrefixSegmentStreamError) as safe:
            fresh.accept(self.packets[0], None)  # type: ignore[arg-type]
        self.assertTrue(safe.exception.fallback_allowed)
        self.assertEqual(fresh.snapshot().state, PrefixSegmentStreamState.ABORTED)

        mutated = PrefixSegmentStreamGuard(self.manifest)
        mutated.accept(self.packets[0], lambda frame: None)
        with self.assertRaises(PrefixSegmentStreamError) as fatal:
            mutated.accept(self.packets[1], None)  # type: ignore[arg-type]
        self.assertTrue(fatal.exception.route_fatal)
        self.assertEqual(mutated.snapshot().state, PrefixSegmentStreamState.FATAL)

    def test_reentrant_accept_during_callback_poison_routes_both_calls(self) -> None:
        guard = PrefixSegmentStreamGuard(self.manifest)

        def reentrant(_frame):
            guard.accept(self.packets[0], lambda frame: None)

        with self.assertRaises(PrefixSegmentStreamError) as caught:
            guard.accept(self.packets[0], reentrant)
        self.assertTrue(caught.exception.route_fatal)
        self.assertEqual(guard.snapshot().state, PrefixSegmentStreamState.FATAL)

    def test_replay_after_complete_is_route_fatal(self) -> None:
        guard = PrefixSegmentStreamGuard(self.manifest)
        for packet in self.packets:
            guard.accept(packet, lambda frame: None)
        with self.assertRaises(PrefixSegmentStreamError) as caught:
            guard.accept(self.packets[-1], lambda frame: None)
        self.assertTrue(caught.exception.route_fatal)
        self.assertEqual(guard.snapshot().state, PrefixSegmentStreamState.FATAL)


class PrefixSegmentRandomizedTests(unittest.TestCase):
    def test_random_trees_round_trip_and_random_corruption_never_escapes(self) -> None:
        generator = random.Random(0x51E6_6A7D)
        for case in range(120):
            leaf_count = generator.randint(1, 8)
            depth = generator.randint(1, 7)
            paths: set[tuple[int, ...]] = set()
            while len(paths) < leaf_count:
                paths.add(tuple(generator.randrange(20) for _ in range(depth)))
            shuffled = list(paths)
            generator.shuffle(shuffled)
            identity = {
                "nonce": case + 1,
                "parent_request_id": 2_000_000 + case,
                "step": case % 257,
            }
            chunking = PrefixWaveChunking(
                max_slice_tokens=generator.randint(1, 7),
                max_segments_per_record=generator.randint(1, 6),
                max_nodes_per_record=generator.randint(1, 32),
            )
            _, arena = _arena(
                tuple(shuffled),
                shared=tuple(
                    generator.randrange(10_000, 20_000)
                    for _ in range(generator.randint(0, 3))
                ),
                identity=identity,
                chunking=chunking,
                first_lane_request_id=3_000_000 + case * 16,
            )
            manifest = _manifest(
                arena,
                codec_id=generator.randrange(7),
                hidden_size=generator.randint(1, 128),
            )
            encoder = PrefixSegmentStreamEncoder(manifest)
            packets = []
            expected_payloads = []
            for record in manifest.records:
                payload = bytes(
                    generator.randrange(256)
                    for _ in range(generator.randint(1, 96))
                )
                expected_payloads.append(payload)
                packets.append(encoder.encode(record, payload))

            guard = PrefixSegmentStreamGuard(manifest)
            observed: list[bytes] = []
            for packet in packets:
                guard.accept(packet, lambda frame: observed.append(frame.payload))
            self.assertEqual(observed, expected_payloads, f"case {case}")
            self.assertEqual(
                guard.snapshot().state,
                PrefixSegmentStreamState.COMPLETE,
                f"case {case}",
            )

            first = packets[0]
            bit_offset = generator.randrange(len(first))
            bit_mask = 1 << generator.randrange(8)
            corrupted = _xor_at(first, bit_offset, bit_mask)
            fresh = PrefixSegmentStreamGuard(manifest)
            with self.assertRaises(PrefixSegmentStreamError) as caught:
                fresh.accept(corrupted, lambda frame: self.fail("mutation escaped"))
            self.assertTrue(caught.exception.fallback_allowed, f"case {case}")

            if len(packets) > 1:
                mutated = PrefixSegmentStreamGuard(manifest)
                mutated.accept(packets[0], lambda frame: None)
                hostile = (
                    packets[0]
                    if generator.randrange(2) == 0 or len(packets) == 2
                    else packets[2]
                )
                with self.assertRaises(PrefixSegmentStreamError) as fatal:
                    mutated.accept(hostile, lambda frame: None)
                self.assertTrue(fatal.exception.route_fatal, f"case {case}")


if __name__ == "__main__":
    unittest.main()
