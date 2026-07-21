"""Pure CPU benchmark for the not-yet-wired packed tree-wave codec.

No socket, GPU, model or remote host is involved.  The report compares only:

* one 32-byte framing header per leaf versus one canonical packed envelope;
* raw DEFLATE versus byte-plane split DEFLATE on deterministic activation-like
  FP16, BF16 and FP32 byte streams;
* measured local compression + decompression time plus transfer time at the
  explicitly supplied bandwidth.

Compression ratios and timings are reported as local observations.  They are
never extrapolated to model throughput or a physical WAN route.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import struct
import sys
from typing import Any, Sequence

from .packed_tree_wave import (
    CompressionAssessment,
    CompressionMode,
    CompressionPolicy,
    DESCRIPTOR_BYTES,
    HEADER_BYTES,
    PACKET_DEFLATE_LEVEL,
    PackedTreeLeaf,
    compression_break_even,
    decode_packed_tree_wave,
    pack_packed_tree_wave,
    select_compression_for_bandwidth,
)


SCHEMA = "gdlp-packed-tree-wave-codec-benchmark/1"
SWEEP_SCHEMA = "gdlp-packed-tree-wave-codec-bandwidth-sweep/1"
EVIDENCE_CLASS = "MEDIDO_CPU_LOCAL_CODEC_PURO"
DEFAULT_PER_LEAF_HEADER_BYTES = 32
DEFAULT_TENSOR_SEED = 0xC0DEC0DE
DTYPE_WIDTHS = {"fp16": 2, "bf16": 2, "fp32": 4}


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def _positive_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return value


def _seed(raw: str) -> int:
    try:
        value = int(raw, 0)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a decimal or 0x-prefixed integer") from error
    if value < 0 or value > 0xFFFFFFFF:
        raise argparse.ArgumentTypeError("must be in [0, 2^32-1]")
    return value


def _bandwidth_sweep(raw: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("bandwidth sweep must contain numbers") from error
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise argparse.ArgumentTypeError(
            "bandwidth sweep values must be finite and greater than zero"
        )
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("bandwidth sweep cannot contain duplicates")
    if tuple(sorted(values)) != values:
        raise argparse.ArgumentTypeError("bandwidth sweep must be strictly increasing")
    return values


def _xorshift32(state: int) -> int:
    state ^= (state << 13) & 0xFFFFFFFF
    state ^= state >> 17
    state ^= (state << 5) & 0xFFFFFFFF
    return state & 0xFFFFFFFF


def _activation_values(count: int, *, seed: int) -> list[float]:
    """Deterministic bounded activation-like values without NumPy or Torch."""

    if count < 1:
        raise ValueError("count must be positive")
    state = seed & 0xFFFFFFFF or 0x6D2B79F5
    values: list[float] = []
    for index in range(count):
        # Six uniforms approximate a centered bell distribution.  A small
        # deterministic residual avoids manufacturing long identical runs.
        total = 0
        for _ in range(6):
            state = _xorshift32(state)
            total += state & 0xFFFF
        centered = (total - 196_605.0) / 92_680.0
        residual = math.sin((index + 1) * 0.017) * 0.03125
        values.append(centered + residual)
    return values


def _float_to_bf16_bits(value: float) -> int:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    # Round-to-nearest-even before retaining the high 16 bits.
    rounding_bias = 0x7FFF + ((bits >> 16) & 1)
    return ((bits + rounding_bias) >> 16) & 0xFFFF


def representative_tensor_bytes(dtype: str, elements: int, *, seed: int) -> bytes:
    """Encode one deterministic finite tensor byte-for-byte in little endian."""

    normalized = dtype.lower()
    if normalized not in DTYPE_WIDTHS:
        raise ValueError(f"unsupported dtype {dtype!r}")
    values = _activation_values(elements, seed=seed)
    if normalized == "fp16":
        return b"".join(struct.pack("<e", value) for value in values)
    if normalized == "bf16":
        return b"".join(
            struct.pack("<H", _float_to_bf16_bits(value)) for value in values
        )
    return b"".join(struct.pack("<f", value) for value in values)


def _tensor_case(
    dtype: str,
    *,
    leaves: int,
    tokens_per_leaf: int,
    hidden_size: int,
    seed: int,
) -> tuple[int, bytes, tuple[PackedTreeLeaf, ...]]:
    width = DTYPE_WIDTHS[dtype]
    elements_per_leaf = tokens_per_leaf * hidden_size
    dtype_seed = (seed ^ (width << 16) ^ len(dtype)) & 0xFFFFFFFF
    raw = representative_tensor_bytes(
        dtype,
        leaves * elements_per_leaf,
        seed=dtype_seed,
    )
    slab_bytes = elements_per_leaf * width
    slabs = tuple(
        raw[index * slab_bytes : (index + 1) * slab_bytes]
        for index in range(leaves)
    )
    entries = tuple(
        PackedTreeLeaf(
            request_id=10_000 + index,
            step=7,
            token_count=tokens_per_leaf,
            slab=slab,
        )
        for index, slab in enumerate(slabs)
    )
    return width, raw, entries


def _assessment_report(
    assessment: CompressionAssessment,
    *,
    packed_metadata_bytes: int,
) -> dict[str, Any]:
    evidence = assessment.break_even
    codec_seconds = evidence.compression_seconds + evidence.decompression_seconds
    maximum_beneficial_bandwidth_mbps = (
        (evidence.raw_bytes - evidence.compressed_bytes)
        * 8.0
        / (codec_seconds * 1_000_000.0)
        if evidence.compressed_bytes < evidence.raw_bytes and codec_seconds > 0
        else None
    )
    return {
        "mode": assessment.mode.name,
        "byte_exact_roundtrip": assessment.byte_exact,
        "element_width": assessment.element_width,
        "raw_arena_bytes": evidence.raw_bytes,
        "compressed_arena_bytes": evidence.compressed_bytes,
        "compressed_over_raw_ratio": evidence.compressed_bytes / evidence.raw_bytes,
        "packed_packet_bytes_if_used": packed_metadata_bytes
        + evidence.compressed_bytes,
        "compression_ms_p50_local": evidence.compression_seconds * 1_000.0,
        "decompression_ms_p50_local": evidence.decompression_seconds * 1_000.0,
        "raw_transfer_ms_at_supplied_bandwidth": evidence.raw_transfer_seconds
        * 1_000.0,
        "compressed_transfer_ms_at_supplied_bandwidth": (
            evidence.compressed_transfer_seconds * 1_000.0
        ),
        "compressed_total_ms_at_supplied_bandwidth": (
            evidence.compressed_total_seconds * 1_000.0
        ),
        "net_savings_ms_at_supplied_bandwidth": evidence.net_savings_seconds
        * 1_000.0,
        "maximum_bandwidth_mbps_for_positive_break_even_local": (
            maximum_beneficial_bandwidth_mbps
        ),
        "passes_break_even": evidence.beneficial,
    }


def _benchmark_dtype(
    dtype: str,
    *,
    leaves: int,
    tokens_per_leaf: int,
    hidden_size: int,
    repeats: int,
    bandwidth_mbps: float,
    per_leaf_header_bytes: int,
    seed: int,
) -> dict[str, Any]:
    width, raw, entries = _tensor_case(
        dtype,
        leaves=leaves,
        tokens_per_leaf=tokens_per_leaf,
        hidden_size=hidden_size,
        seed=seed,
    )
    uncompressed = pack_packed_tree_wave(entries)
    decoded_uncompressed = decode_packed_tree_wave(uncompressed.packet)
    if b"".join(bytes(leaf.slab) for leaf in decoded_uncompressed.leaves) != raw:
        raise AssertionError("uncompressed packed wave failed byte-exact roundtrip")
    policy = CompressionPolicy(
        bandwidth_mbps=bandwidth_mbps,
        repeats=repeats,
    )
    selected = pack_packed_tree_wave(
        entries,
        compression_policy=policy,
        element_width=width,
    )
    decoded_selected = decode_packed_tree_wave(selected.packet)
    if b"".join(bytes(leaf.slab) for leaf in decoded_selected.leaves) != raw:
        raise AssertionError("selected packed wave failed byte-exact roundtrip")
    per_leaf_bytes = len(raw) + leaves * per_leaf_header_bytes
    packed_metadata_bytes = HEADER_BYTES + leaves * DESCRIPTOR_BYTES
    return {
        "dtype": dtype,
        "element_width": width,
        "shape": [leaves, tokens_per_leaf, hidden_size],
        "raw_slab_bytes": len(raw),
        "framing": {
            "per_leaf_header_bytes_assumption": per_leaf_header_bytes,
            "per_leaf_record_count": leaves,
            "per_leaf_total_bytes": per_leaf_bytes,
            "packed_record_count": 1,
            "packed_uncompressed_total_bytes": len(uncompressed.packet),
            "packed_minus_per_leaf_bytes": len(uncompressed.packet)
            - per_leaf_bytes,
            "packed_metadata_bytes": packed_metadata_bytes,
            "comparison_scope": "bytes and logical record count only; no socket calls",
        },
        "compression": {
            "selected_mode": selected.compression.mode.name,
            "selected_packet_bytes": len(selected.packet),
            "selection_required_positive_break_even": True,
            "candidates": [
                _assessment_report(
                    assessment, packed_metadata_bytes=packed_metadata_bytes
                )
                for assessment in selected.compression.assessments
            ],
        },
        "roundtrip": {
            "uncompressed_byte_exact": True,
            "selected_byte_exact": True,
            "uncompressed_zero_copy_decode": decoded_uncompressed.zero_copy_slabs,
            "selected_zero_copy_decode": decoded_selected.zero_copy_slabs,
        },
    }


def _validate_case_inputs(
    *,
    leaves: int,
    tokens_per_leaf: int,
    hidden_size: int,
    repeats: int,
    per_leaf_header_bytes: int,
    dtypes: Sequence[str],
    seed: int,
) -> tuple[str, ...]:
    for name, value in (
        ("leaves", leaves),
        ("tokens_per_leaf", tokens_per_leaf),
        ("hidden_size", hidden_size),
        ("repeats", repeats),
        ("per_leaf_header_bytes", per_leaf_header_bytes),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if leaves > 64:
        raise ValueError("leaves cannot exceed the packed codec limit of 64")
    if tokens_per_leaf > 4096:
        raise ValueError("tokens_per_leaf exceeds the packed codec limit")
    if repeats > 100:
        raise ValueError("repeats cannot exceed 100")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 0xFFFFFFFF:
        raise ValueError("seed must be an integer in [0, 2^32-1]")
    normalized_dtypes = tuple(dtype.lower() for dtype in dtypes)
    if not normalized_dtypes or len(set(normalized_dtypes)) != len(normalized_dtypes):
        raise ValueError("dtypes must be a non-empty sequence without duplicates")
    if any(dtype not in DTYPE_WIDTHS for dtype in normalized_dtypes):
        raise ValueError("dtypes can only contain fp16, bf16 and fp32")
    return normalized_dtypes


def build_benchmark_report(
    *,
    leaves: int = 16,
    tokens_per_leaf: int = 4,
    hidden_size: int = 4096,
    repeats: int = 5,
    bandwidth_mbps: float = 100.0,
    per_leaf_header_bytes: int = DEFAULT_PER_LEAF_HEADER_BYTES,
    dtypes: Sequence[str] = ("fp16", "bf16", "fp32"),
    seed: int = DEFAULT_TENSOR_SEED,
) -> dict[str, Any]:
    normalized_dtypes = _validate_case_inputs(
        leaves=leaves,
        tokens_per_leaf=tokens_per_leaf,
        hidden_size=hidden_size,
        repeats=repeats,
        per_leaf_header_bytes=per_leaf_header_bytes,
        dtypes=dtypes,
        seed=seed,
    )
    if not math.isfinite(bandwidth_mbps) or bandwidth_mbps <= 0:
        raise ValueError("bandwidth_mbps must be finite and positive")
    cases = [
        _benchmark_dtype(
            dtype,
            leaves=leaves,
            tokens_per_leaf=tokens_per_leaf,
            hidden_size=hidden_size,
            repeats=repeats,
            bandwidth_mbps=float(bandwidth_mbps),
            per_leaf_header_bytes=per_leaf_header_bytes,
            seed=seed,
        )
        for dtype in normalized_dtypes
    ]
    return {
        "schema": SCHEMA,
        "evidence_class": EVIDENCE_CLASS,
        "success": all(
            case["roundtrip"]["uncompressed_byte_exact"]
            and case["roundtrip"]["selected_byte_exact"]
            for case in cases
        ),
        "inputs": {
            "leaves": leaves,
            "tokens_per_leaf": tokens_per_leaf,
            "hidden_size": hidden_size,
            "repeats": repeats,
            "bandwidth_mbps": float(bandwidth_mbps),
            "per_leaf_header_bytes": per_leaf_header_bytes,
            "dtypes": list(normalized_dtypes),
            "tensor_seed": seed,
            "tensor_seed_hex": f"0x{seed:08x}",
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "evidence_boundary": {
            "physical_hosts": 1,
            "network_io_executed": False,
            "gpu_executed": False,
            "model_inference_executed": False,
            "ratios_extrapolated_to_tokens_per_second": False,
            "bandwidth_is_an_input_not_a_measurement": True,
            "break_even_includes": [
                "measured_local_compression_time",
                "measured_local_decompression_time",
                "transfer_time_at_supplied_bandwidth",
            ],
            "excluded": [
                "RTT",
                "socket_syscalls",
                "TCP_TLS_overhead",
                "GPU_codec_cost",
                "model_compute",
            ],
        },
        "cases": cases,
    }


def _local_measurement_report(
    assessment: CompressionAssessment,
) -> dict[str, Any]:
    evidence = assessment.break_even
    codec_seconds = evidence.compression_seconds + evidence.decompression_seconds
    maximum_bandwidth = (
        (evidence.raw_bytes - evidence.compressed_bytes)
        * 8.0
        / (codec_seconds * 1_000_000.0)
        if evidence.compressed_bytes < evidence.raw_bytes and codec_seconds > 0
        else None
    )
    return {
        "mode": assessment.mode.name,
        "byte_exact_roundtrip": assessment.byte_exact,
        "element_width": assessment.element_width,
        "raw_arena_bytes": evidence.raw_bytes,
        "compressed_arena_bytes": evidence.compressed_bytes,
        "compressed_over_raw_ratio": evidence.compressed_bytes / evidence.raw_bytes,
        "compression_ms_p50_local": evidence.compression_seconds * 1_000.0,
        "decompression_ms_p50_local": evidence.decompression_seconds * 1_000.0,
        "codec_total_ms_p50_local": codec_seconds * 1_000.0,
        "maximum_bandwidth_mbps_for_positive_break_even_local": maximum_bandwidth,
    }


def _sweep_row(
    assessments: Sequence[CompressionAssessment],
    *,
    bandwidth_mbps: float,
    raw_bytes: int,
    packed_metadata_bytes: int,
) -> dict[str, Any]:
    evaluated: list[tuple[CompressionAssessment, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    metadata_transfer_seconds = (
        packed_metadata_bytes * 8.0 / (bandwidth_mbps * 1_000_000.0)
    )
    for assessment in assessments:
        measured = assessment.break_even
        evidence = compression_break_even(
            raw_bytes=measured.raw_bytes,
            compressed_bytes=measured.compressed_bytes,
            compression_seconds=measured.compression_seconds,
            decompression_seconds=measured.decompression_seconds,
            bandwidth_mbps=bandwidth_mbps,
        )
        evaluated.append((assessment, evidence))
        candidate_rows.append(
            {
                "mode": assessment.mode.name,
                "compressed_arena_bytes": evidence.compressed_bytes,
                "arena_transfer_ms_at_bandwidth": (
                    evidence.compressed_transfer_seconds * 1_000.0
                ),
                "modeled_full_packet_total_ms": (
                    evidence.compressed_total_seconds + metadata_transfer_seconds
                )
                * 1_000.0,
                "net_savings_ms_vs_packed_raw": evidence.net_savings_seconds
                * 1_000.0,
                "passes_break_even": evidence.beneficial,
            }
        )
    beneficial = [item for item in evaluated if item[1].beneficial]
    selected = min(
        beneficial,
        key=lambda item: (item[1].compressed_total_seconds, int(item[0].mode)),
        default=None,
    )
    raw_packet_bytes = packed_metadata_bytes + raw_bytes
    raw_packet_transfer_seconds = raw_packet_bytes * 8.0 / (
        bandwidth_mbps * 1_000_000.0
    )
    if selected is None:
        selected_mode = CompressionMode.NONE
        selected_arena_bytes = raw_bytes
        selected_packet_bytes = raw_packet_bytes
        selected_total_seconds = raw_packet_transfer_seconds
        selected_net_savings_seconds = 0.0
    else:
        selected_assessment, selected_evidence = selected
        selected_mode = selected_assessment.mode
        selected_arena_bytes = selected_evidence.compressed_bytes
        selected_packet_bytes = packed_metadata_bytes + selected_arena_bytes
        selected_total_seconds = (
            selected_evidence.compressed_total_seconds + metadata_transfer_seconds
        )
        selected_net_savings_seconds = selected_evidence.net_savings_seconds
    return {
        "bandwidth_mbps": bandwidth_mbps,
        "packed_raw_packet_bytes": raw_packet_bytes,
        "packed_raw_transfer_ms_at_bandwidth": raw_packet_transfer_seconds * 1_000.0,
        "selected_mode": selected_mode.name,
        "selected_arena_bytes": selected_arena_bytes,
        "selected_packet_bytes": selected_packet_bytes,
        "selected_modeled_full_packet_total_ms": selected_total_seconds * 1_000.0,
        "selected_net_savings_ms_vs_packed_raw": selected_net_savings_seconds
        * 1_000.0,
        "candidates": candidate_rows,
    }


def build_bandwidth_sweep_report(
    *,
    bandwidths_mbps: Sequence[float],
    leaves: int = 16,
    tokens_per_leaf: int = 4,
    hidden_size: int = 4096,
    repeats: int = 7,
    per_leaf_header_bytes: int = DEFAULT_PER_LEAF_HEADER_BYTES,
    dtypes: Sequence[str] = ("fp16", "bf16", "fp32"),
    seed: int = DEFAULT_TENSOR_SEED,
) -> dict[str, Any]:
    bandwidths = tuple(float(value) for value in bandwidths_mbps)
    if (
        not bandwidths
        or any(not math.isfinite(value) or value <= 0 for value in bandwidths)
        or tuple(sorted(set(bandwidths))) != bandwidths
    ):
        raise ValueError("bandwidths_mbps must be unique, finite, positive and increasing")
    normalized_dtypes = _validate_case_inputs(
        leaves=leaves,
        tokens_per_leaf=tokens_per_leaf,
        hidden_size=hidden_size,
        repeats=repeats,
        per_leaf_header_bytes=per_leaf_header_bytes,
        dtypes=dtypes,
        seed=seed,
    )
    cases: list[dict[str, Any]] = []
    for dtype in normalized_dtypes:
        width, raw, entries = _tensor_case(
            dtype,
            leaves=leaves,
            tokens_per_leaf=tokens_per_leaf,
            hidden_size=hidden_size,
            seed=seed,
        )
        uncompressed = pack_packed_tree_wave(entries)
        decoded = decode_packed_tree_wave(uncompressed.packet)
        if b"".join(bytes(leaf.slab) for leaf in decoded.leaves) != raw:
            raise AssertionError("uncompressed packed wave failed byte-exact roundtrip")
        selection = select_compression_for_bandwidth(
            raw,
            element_width=width,
            policy=CompressionPolicy(
                bandwidth_mbps=bandwidths[0],
                repeats=repeats,
            ),
        )
        if not all(item.byte_exact for item in selection.assessments):
            raise AssertionError("a compression candidate failed byte-exact roundtrip")
        metadata_bytes = HEADER_BYTES + leaves * DESCRIPTOR_BYTES
        per_leaf_total_bytes = len(raw) + leaves * per_leaf_header_bytes
        cases.append(
            {
                "dtype": dtype,
                "element_width": width,
                "shape": [leaves, tokens_per_leaf, hidden_size],
                "raw_slab_bytes": len(raw),
                "framing": {
                    "per_leaf_header_bytes_assumption": per_leaf_header_bytes,
                    "per_leaf_record_count": leaves,
                    "per_leaf_total_bytes": per_leaf_total_bytes,
                    "packed_record_count": 1,
                    "packed_metadata_bytes": metadata_bytes,
                    "packed_uncompressed_total_bytes": len(uncompressed.packet),
                    "packed_minus_per_leaf_bytes": len(uncompressed.packet)
                    - per_leaf_total_bytes,
                    "comparison_scope": (
                        "bytes and logical record count only; no socket calls"
                    ),
                },
                "local_codec_measurements": [
                    _local_measurement_report(item)
                    for item in selection.assessments
                ],
                "bandwidth_sweep": [
                    _sweep_row(
                        selection.assessments,
                        bandwidth_mbps=bandwidth,
                        raw_bytes=len(raw),
                        packed_metadata_bytes=metadata_bytes,
                    )
                    for bandwidth in bandwidths
                ],
            }
        )
    return {
        "schema": SWEEP_SCHEMA,
        "evidence_class": EVIDENCE_CLASS,
        "success": all(
            measurement["byte_exact_roundtrip"]
            for case in cases
            for measurement in case["local_codec_measurements"]
        ),
        "inputs": {
            "bandwidths_mbps": list(bandwidths),
            "leaves": leaves,
            "tokens_per_leaf": tokens_per_leaf,
            "hidden_size": hidden_size,
            "repeats": repeats,
            "compression_level": PACKET_DEFLATE_LEVEL,
            "per_leaf_header_bytes": per_leaf_header_bytes,
            "dtypes": list(normalized_dtypes),
            "tensor_seed": seed,
            "tensor_seed_hex": f"0x{seed:08x}",
            "same_local_codec_measurement_reused_across_bandwidths": True,
        },
        "environment": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "processor": platform.processor() or "unknown",
        },
        "evidence_boundary": {
            "physical_hosts": 1,
            "network_io_executed": False,
            "gpu_executed": False,
            "model_inference_executed": False,
            "tokens_per_second_claimed": False,
            "bandwidth_values_are_inputs_not_measurements": True,
            "codec_timings_are_local_cpu_measurements": True,
            "bandwidth_rows_reuse_one_measurement_per_dtype": True,
            "break_even_includes": [
                "measured_local_compression_time",
                "measured_local_decompression_time",
                "calculated_transfer_time_at_supplied_bandwidth",
            ],
            "excluded": [
                "RTT",
                "socket_syscalls",
                "TCP_TLS_overhead",
                "GPU_codec_cost",
                "model_compute",
            ],
        },
        "cases": cases,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leaves", type=_positive_int, default=16)
    parser.add_argument("--tokens-per-leaf", type=_positive_int, default=4)
    parser.add_argument("--hidden-size", type=_positive_int, default=4096)
    parser.add_argument("--repeats", type=_positive_int, default=5)
    parser.add_argument("--bandwidth-mbps", type=_positive_float, default=100.0)
    parser.add_argument(
        "--bandwidth-sweep-mbps",
        type=_bandwidth_sweep,
        help="comma-separated increasing Mbps values; measures each dtype once",
    )
    parser.add_argument("--seed", type=_seed, default=DEFAULT_TENSOR_SEED)
    parser.add_argument(
        "--per-leaf-header-bytes",
        type=_positive_int,
        default=DEFAULT_PER_LEAF_HEADER_BYTES,
    )
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.bandwidth_sweep_mbps is None:
        report = build_benchmark_report(
            leaves=args.leaves,
            tokens_per_leaf=args.tokens_per_leaf,
            hidden_size=args.hidden_size,
            repeats=args.repeats,
            bandwidth_mbps=args.bandwidth_mbps,
            per_leaf_header_bytes=args.per_leaf_header_bytes,
            seed=args.seed,
        )
    else:
        report = build_bandwidth_sweep_report(
            bandwidths_mbps=args.bandwidth_sweep_mbps,
            leaves=args.leaves,
            tokens_per_leaf=args.tokens_per_leaf,
            hidden_size=args.hidden_size,
            repeats=args.repeats,
            per_leaf_header_bytes=args.per_leaf_header_bytes,
            seed=args.seed,
        )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_PER_LEAF_HEADER_BYTES",
    "DEFAULT_TENSOR_SEED",
    "DTYPE_WIDTHS",
    "EVIDENCE_CLASS",
    "SCHEMA",
    "SWEEP_SCHEMA",
    "build_bandwidth_sweep_report",
    "build_benchmark_report",
    "main",
    "representative_tensor_bytes",
]
