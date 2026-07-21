from __future__ import annotations

import math
import struct
import unittest

import distributed_runtime.packed_tree_wave as ptw
from distributed_runtime.packed_tree_wave import (
    CompressionMode,
    CompressionPolicy,
    PackedTreeLeaf,
    PackedTreeWaveError,
    PackedTreeWaveLimits,
    compress_lossless,
    compression_break_even,
    decode_packed_tree_wave,
    decompress_lossless,
    encode_packed_tree_wave,
    pack_packed_tree_wave,
)
from distributed_runtime.packed_tree_wave_benchmark import (
    EVIDENCE_CLASS,
    SCHEMA,
    SWEEP_SCHEMA,
    build_benchmark_report,
    build_bandwidth_sweep_report,
    representative_tensor_bytes,
)


def _leaves() -> tuple[PackedTreeLeaf, ...]:
    # Deliberately unordered.  All leaves encode four bytes per token.
    return (
        PackedTreeLeaf(8, 4, 2, bytearray(b"abcdefgh")),
        PackedTreeLeaf(3, 4, 1, memoryview(b"WXYZ")),
        PackedTreeLeaf(12, 5, 3, b"abcdefghijkl"),
    )


class PackedTreeWaveRoundtripTests(unittest.TestCase):
    def test_canonical_roundtrip_is_deterministic_and_sorted_by_request_id(self) -> None:
        packet = encode_packed_tree_wave(_leaves())
        reordered = encode_packed_tree_wave(tuple(reversed(_leaves())))
        self.assertEqual(packet, reordered)

        decoded = decode_packed_tree_wave(packet)
        self.assertEqual([leaf.request_id for leaf in decoded.leaves], [3, 8, 12])
        self.assertEqual([leaf.step for leaf in decoded.leaves], [4, 4, 5])
        self.assertEqual([leaf.token_count for leaf in decoded.leaves], [1, 2, 3])
        self.assertEqual(
            [bytes(leaf.slab) for leaf in decoded.leaves],
            [b"WXYZ", b"abcdefgh", b"abcdefghijkl"],
        )
        self.assertEqual(
            [leaf.offset for leaf in decoded.leaves],
            [0, 4, 12],
        )
        self.assertEqual(decoded.compression, CompressionMode.NONE)
        self.assertTrue(decoded.zero_copy_slabs)
        self.assertFalse(decoded.copied_input)
        self.assertEqual(len(decoded.digest), 32)

    def test_readonly_bytes_memoryview_is_zero_copy(self) -> None:
        packet = encode_packed_tree_wave(_leaves())
        decoded = decode_packed_tree_wave(memoryview(packet))
        self.assertTrue(decoded.zero_copy_slabs)
        self.assertFalse(decoded.copied_input)

    def test_mutable_or_ambiguously_backed_input_is_copied_before_validation(self) -> None:
        mutable = bytearray(encode_packed_tree_wave(_leaves()))
        decoded = decode_packed_tree_wave(memoryview(mutable).toreadonly())
        before = bytes(decoded.leaves[-1].slab)
        mutable[-1] ^= 0xFF
        self.assertEqual(bytes(decoded.leaves[-1].slab), before)
        self.assertFalse(decoded.zero_copy_slabs)
        self.assertTrue(decoded.copied_input)

    def test_duplicate_ids_noncontiguous_slabs_and_shape_mismatch_are_rejected(self) -> None:
        with self.assertRaisesRegex(PackedTreeWaveError, "duplicate request id"):
            encode_packed_tree_wave(
                (
                    PackedTreeLeaf(1, 0, 1, b"abcd"),
                    PackedTreeLeaf(1, 0, 1, b"efgh"),
                )
            )
        with self.assertRaisesRegex(PackedTreeWaveError, "C-contiguous"):
            encode_packed_tree_wave(
                (PackedTreeLeaf(1, 0, 1, memoryview(b"abcdef")[::2]),)
            )
        with self.assertRaisesRegex(PackedTreeWaveError, "same bytes per token"):
            encode_packed_tree_wave(
                (
                    PackedTreeLeaf(1, 0, 1, b"abcd"),
                    PackedTreeLeaf(2, 0, 2, b"abcdefghijkl"),
                )
            )
        with self.assertRaisesRegex(PackedTreeWaveError, "divisible"):
            encode_packed_tree_wave((PackedTreeLeaf(1, 0, 2, b"abc"),))

    def test_sealed_encode_and_decode_limits_fail_closed(self) -> None:
        small = PackedTreeWaveLimits(
            max_entries=2,
            max_slab_bytes=8,
            max_logical_bytes=16,
            max_token_count=3,
        )
        with self.assertRaisesRegex(PackedTreeWaveError, "entry count"):
            encode_packed_tree_wave(_leaves(), limits=small)
        with self.assertRaisesRegex(PackedTreeWaveError, "slab exceeds"):
            encode_packed_tree_wave(
                (PackedTreeLeaf(1, 0, 3, b"abcdefghijkl"),), limits=small
            )
        packet = encode_packed_tree_wave(_leaves())
        with self.assertRaisesRegex(PackedTreeWaveError, "limits"):
            decode_packed_tree_wave(packet, limits=small)
        with self.assertRaises(ValueError):
            PackedTreeWaveLimits(max_entries=65)


class PackedTreeWaveCorruptionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.packet = encode_packed_tree_wave(_leaves())

    def test_magic_version_reserved_digest_and_payload_corruption_are_rejected(self) -> None:
        invalid_magic = bytearray(self.packet)
        invalid_magic[0] ^= 0xFF
        with self.assertRaisesRegex(PackedTreeWaveError, "magic"):
            decode_packed_tree_wave(invalid_magic)

        invalid_version = bytearray(self.packet)
        invalid_version[4] = 99
        with self.assertRaisesRegex(PackedTreeWaveError, "version"):
            decode_packed_tree_wave(invalid_version)

        invalid_reserved = bytearray(self.packet)
        fields = list(ptw._HEADER.unpack_from(invalid_reserved))
        fields[7] = 1
        invalid_reserved[: ptw.HEADER_BYTES] = ptw._HEADER.pack(*fields)
        with self.assertRaisesRegex(PackedTreeWaveError, "reserved"):
            decode_packed_tree_wave(invalid_reserved)

        invalid_digest = bytearray(self.packet)
        invalid_digest[ptw.HEADER_BYTES - 1] ^= 0x01
        with self.assertRaisesRegex(PackedTreeWaveError, "SHA-256"):
            decode_packed_tree_wave(invalid_digest)

        invalid_payload = bytearray(self.packet)
        invalid_payload[-1] ^= 0x01
        with self.assertRaisesRegex(PackedTreeWaveError, "stored slab arena CRC"):
            decode_packed_tree_wave(invalid_payload)

    def test_truncation_trailing_bytes_and_table_crc_are_rejected(self) -> None:
        with self.assertRaisesRegex(PackedTreeWaveError, "truncated"):
            decode_packed_tree_wave(self.packet[:-1])
        with self.assertRaisesRegex(PackedTreeWaveError, "trailing"):
            decode_packed_tree_wave(self.packet + b"x")

        invalid_step = bytearray(self.packet)
        # First descriptor: request id occupies bytes [0:8], step [8:12].
        invalid_step[ptw.HEADER_BYTES + 8] ^= 0x01
        with self.assertRaisesRegex(PackedTreeWaveError, "table CRC"):
            decode_packed_tree_wave(invalid_step)

    def test_duplicate_overlap_gap_and_uint32_overflow_descriptors_are_rejected(self) -> None:
        duplicate = bytearray(self.packet)
        first_id = struct.unpack_from("<Q", duplicate, ptw.HEADER_BYTES)[0]
        struct.pack_into(
            "<Q", duplicate, ptw.HEADER_BYTES + ptw.DESCRIPTOR_BYTES, first_id
        )
        with self.assertRaisesRegex(PackedTreeWaveError, "duplicate request id"):
            decode_packed_tree_wave(duplicate)

        overlap = bytearray(self.packet)
        second_offset = ptw.HEADER_BYTES + ptw.DESCRIPTOR_BYTES + 16
        struct.pack_into("<I", overlap, second_offset, 0)
        with self.assertRaisesRegex(PackedTreeWaveError, "overlaps"):
            decode_packed_tree_wave(overlap)

        gap = bytearray(self.packet)
        struct.pack_into("<I", gap, second_offset, 5)
        with self.assertRaisesRegex(PackedTreeWaveError, "gap"):
            decode_packed_tree_wave(gap)

        overflow = bytearray(self.packet)
        struct.pack_into("<I", overflow, second_offset, ptw.UINT32_MAX)
        with self.assertRaisesRegex(PackedTreeWaveError, "overflows"):
            decode_packed_tree_wave(overflow)


class LosslessCompressionTests(unittest.TestCase):
    def test_raw_and_byte_plane_deflate_are_deterministic_and_byte_exact(self) -> None:
        raw16 = representative_tensor_bytes("fp16", 2048, seed=17)
        raw32 = representative_tensor_bytes("fp32", 2048, seed=17)
        for mode, raw, width in (
            (CompressionMode.RAW_DEFLATE, raw16, 0),
            (CompressionMode.BYTE_PLANE_DEFLATE, raw16, 2),
            (CompressionMode.BYTE_PLANE_DEFLATE, raw32, 4),
        ):
            with self.subTest(mode=mode.name, width=width):
                compressed = compress_lossless(raw, mode, element_width=width)
                self.assertEqual(
                    compressed,
                    compress_lossless(raw, mode, element_width=width),
                )
                self.assertEqual(
                    decompress_lossless(
                        compressed,
                        mode,
                        expected_bytes=len(raw),
                        element_width=width,
                        maximum_bytes=len(raw),
                    ),
                    raw,
                )

    def test_compression_corruption_truncation_and_declared_size_are_rejected(self) -> None:
        raw = b"\0\x3c" * 2048
        for mode, width in (
            (CompressionMode.RAW_DEFLATE, 0),
            (CompressionMode.BYTE_PLANE_DEFLATE, 2),
        ):
            compressed = compress_lossless(raw, mode, element_width=width)
            with self.subTest(mode=mode.name):
                with self.assertRaises(PackedTreeWaveError):
                    decompress_lossless(
                        compressed[:-1],
                        mode,
                        expected_bytes=len(raw),
                        element_width=width,
                        maximum_bytes=len(raw),
                    )
                with self.assertRaises(PackedTreeWaveError):
                    decompress_lossless(
                        compressed + b"x",
                        mode,
                        expected_bytes=len(raw),
                        element_width=width,
                        maximum_bytes=len(raw),
                    )
                with self.assertRaises(PackedTreeWaveError):
                    decompress_lossless(
                        compressed,
                        mode,
                        expected_bytes=len(raw) - 2,
                        element_width=width,
                        maximum_bytes=len(raw),
                    )

    def test_packet_compression_can_only_activate_after_positive_break_even(self) -> None:
        leaves = tuple(
            PackedTreeLeaf(index + 1, 0, 2, b"\0\x3c" * 1024)
            for index in range(4)
        )
        slow_link = pack_packed_tree_wave(
            leaves,
            compression_policy=CompressionPolicy(
                bandwidth_mbps=0.001, repeats=1
            ),
            element_width=2,
        )
        self.assertNotEqual(slow_link.compression.mode, CompressionMode.NONE)
        selected_assessment = slow_link.compression.selected_assessment
        self.assertIsNotNone(selected_assessment)
        assert selected_assessment is not None
        self.assertTrue(selected_assessment.break_even.beneficial)
        decoded = decode_packed_tree_wave(slow_link.packet)
        self.assertEqual(
            [bytes(leaf.slab) for leaf in decoded.leaves],
            [b"\0\x3c" * 1024] * 4,
        )
        self.assertFalse(decoded.zero_copy_slabs)

        effectively_infinite_link = pack_packed_tree_wave(
            leaves,
            compression_policy=CompressionPolicy(
                bandwidth_mbps=1_000_000_000.0, repeats=1
            ),
            element_width=2,
        )
        self.assertEqual(
            effectively_infinite_link.compression.mode, CompressionMode.NONE
        )
        self.assertTrue(
            all(
                not item.break_even.beneficial
                for item in effectively_infinite_link.compression.assessments
            )
        )


class CompressionBreakEvenTests(unittest.TestCase):
    def test_formula_includes_compression_and_decompression_time(self) -> None:
        beneficial = compression_break_even(
            raw_bytes=1_000_000,
            compressed_bytes=500_000,
            compression_seconds=0.05,
            decompression_seconds=0.10,
            bandwidth_mbps=10.0,
        )
        self.assertTrue(beneficial.beneficial)
        self.assertAlmostEqual(beneficial.raw_transfer_seconds, 0.8)
        self.assertAlmostEqual(beneficial.compressed_total_seconds, 0.55)

        # Compression + transfer alone would be 0.45 s and look profitable.
        # Including decompression makes it 0.85 s, so it must remain disabled.
        too_slow_to_decode = compression_break_even(
            raw_bytes=1_000_000,
            compressed_bytes=500_000,
            compression_seconds=0.05,
            decompression_seconds=0.40,
            bandwidth_mbps=10.0,
        )
        self.assertFalse(too_slow_to_decode.beneficial)
        self.assertAlmostEqual(too_slow_to_decode.compressed_total_seconds, 0.85)
        self.assertLess(too_slow_to_decode.net_savings_seconds, 0)

    def test_invalid_or_nonfinite_break_even_inputs_fail(self) -> None:
        for value in (0.0, -1.0, math.inf, math.nan):
            with self.subTest(bandwidth=value):
                with self.assertRaises(ValueError):
                    compression_break_even(
                        raw_bytes=10,
                        compressed_bytes=5,
                        compression_seconds=0,
                        decompression_seconds=0,
                        bandwidth_mbps=value,
                    )


class PackedTreeWaveBenchmarkTests(unittest.TestCase):
    def test_representative_tensor_bytes_are_deterministic_and_width_exact(self) -> None:
        for dtype, width in (("fp16", 2), ("bf16", 2), ("fp32", 4)):
            with self.subTest(dtype=dtype):
                first = representative_tensor_bytes(dtype, 33, seed=91)
                self.assertEqual(len(first), 33 * width)
                self.assertEqual(
                    first, representative_tensor_bytes(dtype, 33, seed=91)
                )
                self.assertNotEqual(
                    first, representative_tensor_bytes(dtype, 33, seed=92)
                )

    def test_pure_benchmark_covers_all_dtypes_and_states_evidence_boundary(self) -> None:
        report = build_benchmark_report(
            leaves=4,
            tokens_per_leaf=2,
            hidden_size=64,
            repeats=1,
            bandwidth_mbps=50.0,
        )
        self.assertEqual(report["schema"], SCHEMA)
        self.assertEqual(report["evidence_class"], EVIDENCE_CLASS)
        self.assertTrue(report["success"])
        self.assertEqual(
            [case["dtype"] for case in report["cases"]],
            ["fp16", "bf16", "fp32"],
        )
        self.assertFalse(report["evidence_boundary"]["network_io_executed"])
        self.assertFalse(report["evidence_boundary"]["gpu_executed"])
        self.assertTrue(
            report["evidence_boundary"]["bandwidth_is_an_input_not_a_measurement"]
        )
        for case in report["cases"]:
            self.assertEqual(case["framing"]["per_leaf_record_count"], 4)
            self.assertEqual(case["framing"]["packed_record_count"], 1)
            self.assertEqual(len(case["compression"]["candidates"]), 2)
            self.assertTrue(
                all(
                    candidate["byte_exact_roundtrip"]
                    for candidate in case["compression"]["candidates"]
                )
            )

    def test_bandwidth_sweep_reuses_one_codec_measurement_and_never_claims_tps(
        self,
    ) -> None:
        report = build_bandwidth_sweep_report(
            bandwidths_mbps=(0.001, 100_000_000.0),
            leaves=4,
            tokens_per_leaf=2,
            hidden_size=64,
            repeats=2,
            seed=0x12345678,
        )
        self.assertEqual(report["schema"], SWEEP_SCHEMA)
        self.assertTrue(report["success"])
        self.assertTrue(
            report["inputs"][
                "same_local_codec_measurement_reused_across_bandwidths"
            ]
        )
        self.assertFalse(report["evidence_boundary"]["tokens_per_second_claimed"])
        for case in report["cases"]:
            rows = case["bandwidth_sweep"]
            self.assertEqual(
                [row["bandwidth_mbps"] for row in rows],
                [0.001, 100_000_000.0],
            )
            self.assertNotEqual(rows[0]["selected_mode"], "NONE")
            self.assertEqual(rows[1]["selected_mode"], "NONE")
            self.assertEqual(len(case["local_codec_measurements"]), 2)
            for measurement in case["local_codec_measurements"]:
                self.assertTrue(measurement["byte_exact_roundtrip"])

    def test_bandwidth_sweep_rejects_unsorted_duplicate_or_nonfinite_values(
        self,
    ) -> None:
        for bandwidths in ((10.0, 10.0), (100.0, 10.0), (10.0, math.inf)):
            with self.subTest(bandwidths=bandwidths):
                with self.assertRaises(ValueError):
                    build_bandwidth_sweep_report(
                        bandwidths_mbps=bandwidths,
                        leaves=2,
                        tokens_per_leaf=1,
                        hidden_size=8,
                        repeats=1,
                        dtypes=("fp16",),
                    )


if __name__ == "__main__":
    unittest.main()
