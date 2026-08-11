"""Physical Qwen3-dense stage calibration for coordinator challenges.

The probe executes real PyTorch kernels on the selected production device. It
does not fabricate a result when the requested KV shape cannot be allocated or
when the backend is unavailable. Artifact identity and loaded-stage parity are
bound separately by the coordinator's current deployment canary.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import statistics
import time
from typing import Callable, Sequence

import torch

from .runtime_profile import resolve_runtime_target


MINIMUM_SAMPLES = 7
MAXIMUM_KV_PROBE_BYTES = 4 * 1024 * 1024 * 1024


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Measure a Qwen3 dense stage profile.")
    result.add_argument("--probe-kind", choices=("qwen3-dense-v1",), required=True)
    result.add_argument("--backend", required=True)
    result.add_argument("--device", required=True)
    result.add_argument("--precision", choices=("float16", "float32"), required=True)
    result.add_argument("--hidden-size", type=int, required=True)
    result.add_argument("--attention-heads", type=int, required=True)
    result.add_argument("--kv-heads", type=int, required=True)
    result.add_argument("--head-dim", type=int, required=True)
    result.add_argument("--layer-count", type=int, required=True)
    result.add_argument("--context-tokens", type=int, required=True)
    result.add_argument("--kv-bytes-per-token", type=int, required=True)
    result.add_argument("--layer-weight-bytes", type=int, required=True)
    result.add_argument("--reference-decode-ms-per-token", type=float, required=True)
    result.add_argument("--reference-prefill-ms-per-token", type=float, required=True)
    result.add_argument("--warmup-samples", type=int, default=2)
    result.add_argument("--samples", type=int, default=21)
    result.add_argument("--threads", type=int, default=1)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    _validate(args)
    torch.set_num_threads(args.threads)
    target = resolve_runtime_target(args.backend, args.device, args.precision)
    measurement = measure_qwen3_dense(target.device, target.dtype, args)
    print(json.dumps(measurement, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


def measure_qwen3_dense(
    device: torch.device,
    dtype: torch.dtype,
    args: argparse.Namespace,
) -> dict[str, object]:
    synchronize = _synchronizer(device)
    hidden = args.hidden_size
    layer_count = args.layer_count
    projection = torch.randn((hidden, hidden), dtype=dtype, device=device)
    decode_input = torch.randn((1, hidden), dtype=dtype, device=device)
    prefill_input = torch.randn((32, hidden), dtype=dtype, device=device)
    verify_input = torch.randn((4, hidden), dtype=dtype, device=device)

    # Allocate the full challenged KV budget. Successful allocation is the
    # mechanical evidence behind maxKvTokens; a formula alone is not enough.
    kv_probe_bytes = args.kv_bytes_per_token * args.context_tokens
    if kv_probe_bytes > MAXIMUM_KV_PROBE_BYTES:
        raise RuntimeError("challenged KV probe exceeds the bounded physical probe limit")
    kv_elements = kv_probe_bytes // torch.empty((), dtype=dtype).element_size()
    kv_probe = torch.empty(kv_elements, dtype=dtype, device=device)
    kv_probe.zero_()
    if kv_probe.numel() * kv_probe.element_size() != kv_probe_bytes:
        raise RuntimeError("physical KV allocation does not match the challenged shape")

    decode = _latency_series(
        lambda: torch.mm(decode_input, projection),
        synchronize,
        args.warmup_samples,
        args.samples,
        layer_count,
        1,
    )
    prefill = _latency_series(
        lambda: torch.mm(prefill_input, projection),
        synchronize,
        args.warmup_samples,
        args.samples,
        layer_count,
        prefill_input.shape[0],
    )
    verify = _latency_series(
        lambda: torch.mm(verify_input, projection),
        synchronize,
        args.warmup_samples,
        args.samples,
        layer_count,
        verify_input.shape[0],
    )
    fast_kernel = _probe_attention_kernel(
        device,
        dtype,
        args.attention_heads,
        args.kv_heads,
        args.head_dim,
        synchronize,
    )
    usable_memory = _usable_memory_bytes(device)
    del kv_probe, projection, decode_input, prefill_input, verify_input

    return {
        "measuredAt": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
            "+00:00", "Z"
        ),
        "samples": args.samples,
        "confidenceHalfWidthPct": max(
            decode["confidence"], prefill["confidence"], verify["confidence"]
        ),
        "capacity": {
            "contextTokens": args.context_tokens,
            "maxLayerCount": layer_count,
            "kvBytesPerToken": args.kv_bytes_per_token,
            "maxKvTokens": args.context_tokens,
            "usableMemoryBytes": max(1, usable_memory),
        },
        "costs": {
            "decodeMsPerTokenP50": decode["p50"],
            "decodeMsPerTokenP95": decode["p95"],
            "prefillMsPerTokenP50": prefill["p50"],
            "prefillMsPerTokenP95": prefill["p95"],
            "verifyMsPerTokenP50": verify["p50"],
            "verifyMsPerTokenP95": verify["p95"],
            "decodeScale": max(
                0.01,
                min(100.0, decode["p50"] / args.reference_decode_ms_per_token),
            ),
            "prefillScale": max(
                0.01,
                min(100.0, prefill["p50"] / args.reference_prefill_ms_per_token),
            ),
        },
        "features": {
            "fastKernel": fast_kernel,
            "graphMode": "available" if _graph_available(device) else "unavailable",
            "roles": ["head", "middle", "tail"],
        },
    }


def _validate(args: argparse.Namespace) -> None:
    integers = (
        args.hidden_size,
        args.attention_heads,
        args.kv_heads,
        args.head_dim,
        args.layer_count,
        args.context_tokens,
        args.kv_bytes_per_token,
        args.layer_weight_bytes,
        args.warmup_samples,
        args.samples,
        args.threads,
    )
    if any(not isinstance(value, int) or value <= 0 for value in integers):
        raise ValueError("engine runtime probe integers must be positive")
    if args.samples < MINIMUM_SAMPLES or args.samples > 100_000:
        raise ValueError("engine runtime probe sample count is invalid")
    if args.warmup_samples > 1_000 or args.threads > 256:
        raise ValueError("engine runtime probe bounds are invalid")
    if args.hidden_size != args.attention_heads * args.head_dim:
        raise ValueError("Qwen3 hidden size does not match attention shape")
    if args.attention_heads % args.kv_heads:
        raise ValueError("Qwen3 query heads must be divisible by KV heads")
    expected = 2 * args.kv_heads * args.head_dim * 2 * args.layer_count
    if args.kv_bytes_per_token != expected:
        raise ValueError("Qwen3 KV bytes do not match the challenged layer range")
    if (
        not math.isfinite(args.reference_decode_ms_per_token)
        or args.reference_decode_ms_per_token <= 0
        or not math.isfinite(args.reference_prefill_ms_per_token)
        or args.reference_prefill_ms_per_token <= 0
    ):
        raise ValueError("Qwen3 reference costs must be finite and positive")


def _latency_series(
    operation: Callable[[], torch.Tensor],
    synchronize: Callable[[], None],
    warmups: int,
    samples: int,
    layer_count: int,
    tokens: int,
) -> dict[str, float]:
    for _ in range(warmups):
        result = operation()
        synchronize()
    values: list[float] = []
    for _ in range(samples):
        synchronize()
        started = time.perf_counter()
        result = operation()
        synchronize()
        if not torch.isfinite(result).all().item():
            raise RuntimeError("engine runtime probe produced non-finite output")
        elapsed_ms = (time.perf_counter() - started) * 1_000
        values.append(elapsed_ms * layer_count / tokens)
    ordered = sorted(values)
    p50 = statistics.median(ordered)
    p95 = ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)]
    deviation = statistics.stdev(ordered) if len(ordered) > 1 else 0.0
    confidence = 0.0 if p50 == 0 else min(100.0, 1.96 * deviation / math.sqrt(len(ordered)) / p50 * 100)
    return {"p50": max(p50, 1e-9), "p95": max(p95, p50), "confidence": confidence}


def _probe_attention_kernel(
    device: torch.device,
    dtype: torch.dtype,
    heads: int,
    kv_heads: int,
    head_dim: int,
    synchronize: Callable[[], None],
) -> bool:
    functional = getattr(torch.nn.functional, "scaled_dot_product_attention", None)
    if not callable(functional):
        return False
    probe_heads = min(heads, 8)
    query = torch.randn((1, probe_heads, 1, head_dim), dtype=dtype, device=device)
    probe_kv_heads = min(kv_heads, probe_heads)
    key = torch.randn((1, probe_kv_heads, 16, head_dim), dtype=dtype, device=device)
    value = torch.randn_like(key)
    try:
        output = functional(
            query,
            key,
            value,
            is_causal=True,
            enable_gqa=probe_heads != probe_kv_heads,
        )
    except TypeError:
        return False
    synchronize()
    return bool(torch.isfinite(output).all().item())


def _synchronizer(device: torch.device) -> Callable[[], None]:
    if device.type == "cuda":
        return lambda: torch.cuda.synchronize(device)
    if device.type == "xpu" and hasattr(torch, "xpu"):
        return lambda: torch.xpu.synchronize(device)
    if device.type == "mps":
        return torch.mps.synchronize
    return lambda: None


def _usable_memory_bytes(device: torch.device) -> int:
    if device.type == "cuda":
        free, _total = torch.cuda.mem_get_info(device)
        return int(free)
    return 1


def _graph_available(device: torch.device) -> bool:
    return device.type == "cuda" and callable(getattr(torch.cuda, "CUDAGraph", None))


if __name__ == "__main__":
    raise SystemExit(main())
