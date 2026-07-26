"""Physical calibration for the native Mycellios distributed runtime.

This module deliberately benchmarks the same PyTorch device and the exact GDLP
FP16 activation codec used by automatic distribution.  It has no simulated
mode: an unavailable or mismatched backend exits non-zero and no profile is
published.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import statistics
import time
from typing import Callable, Sequence

import torch

from .protocol import Frame, FrameType, TensorCodec, decode_tensor, encode_tensor_payload


MINIMUM_SAMPLES = 7
ACTIVATION_CODEC_ID = "fp16"
_T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


@dataclass(frozen=True)
class RuntimeTarget:
    backend: str
    device: torch.device
    device_name: str
    dtype: torch.dtype
    precision: str


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Measure a sealed-input Mycellios runtime performance profile."
    )
    result.add_argument(
        "--backend", required=True, choices=("cuda", "rocm", "mps", "xpu", "cpu")
    )
    result.add_argument("--device", required=True)
    result.add_argument(
        "--precision", required=True, choices=("float16", "float32")
    )
    result.add_argument("--warmup-samples", type=int, default=2)
    result.add_argument("--samples", type=int, default=9)
    result.add_argument("--threads", type=int, default=1)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not 1 <= args.warmup_samples <= 1_000:
        raise ValueError("warmup samples must be between 1 and 1000")
    if not MINIMUM_SAMPLES <= args.samples <= 10_000:
        raise ValueError(
            f"measured samples must be between {MINIMUM_SAMPLES} and 10000"
        )
    if not 1 <= args.threads <= 256:
        raise ValueError("threads must be between 1 and 256")
    torch.set_num_threads(args.threads)
    target = resolve_runtime_target(args.backend, args.device, args.precision)
    profile = measure_runtime_profile(
        target,
        warmup_samples=args.warmup_samples,
        samples=args.samples,
    )
    print(json.dumps(profile, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


def resolve_runtime_target(backend: str, device: str, precision: str) -> RuntimeTarget:
    requested = torch.device(device)
    dtype = torch.float16 if precision == "float16" else torch.float32
    if backend == "cpu":
        if requested.type != "cpu" or precision != "float32":
            raise RuntimeError("CPU calibration requires device=cpu and float32")
        device_name = _cpu_device_name()
    elif backend in ("cuda", "rocm"):
        if requested.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError(f"{backend} calibration requires an available CUDA device")
        hip_version = getattr(torch.version, "hip", None)
        if backend == "rocm" and not hip_version:
            raise RuntimeError("ROCm calibration requires a HIP-enabled PyTorch runtime")
        if backend == "cuda" and hip_version:
            raise RuntimeError("CUDA calibration cannot use a ROCm PyTorch runtime")
        device_name = torch.cuda.get_device_name(requested)
    elif backend == "mps":
        mps = getattr(torch.backends, "mps", None)
        if requested.type != "mps" or mps is None or not mps.is_available():
            raise RuntimeError("MPS calibration requires an available MPS device")
        device_name = "Apple MPS"
    elif backend == "xpu":
        xpu = getattr(torch, "xpu", None)
        if (
            requested.type != "xpu"
            or xpu is None
            or not callable(getattr(xpu, "is_available", None))
            or not xpu.is_available()
        ):
            raise RuntimeError("XPU calibration requires an available XPU device")
        get_name = getattr(xpu, "get_device_name", None)
        device_name = get_name(requested) if callable(get_name) else str(requested)
    else:
        raise RuntimeError(f"unsupported physical backend {backend}")
    if backend != "cpu" and precision != "float16":
        raise RuntimeError("accelerator calibration requires the FP16 production precision")
    return RuntimeTarget(
        backend=backend,
        device=requested,
        device_name=str(device_name).strip(),
        dtype=dtype,
        precision=precision,
    )


def measure_runtime_profile(
    target: RuntimeTarget,
    *,
    warmup_samples: int,
    samples: int,
) -> dict[str, object]:
    synchronize = _synchronizer(target)
    decode = _measure_series(
        lambda: _decode_memory_sample(target, synchronize),
        unit="GB/s",
        warmup_samples=warmup_samples,
        samples=samples,
    )
    prefill = _measure_series(
        lambda: _prefill_compute_sample(target, synchronize),
        unit="TFLOP/s",
        warmup_samples=warmup_samples,
        samples=samples,
    )
    codec = _measure_series(
        lambda: _activation_codec_sample(target, synchronize),
        unit="GB/s",
        warmup_samples=warmup_samples,
        samples=samples,
    )
    return {
        "measuredAt": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
            "+00:00", "Z"
        ),
        "backend": target.backend,
        "deviceName": target.device_name,
        "precision": target.precision,
        "source": "physical-microbenchmark",
        "activationCodecId": ACTIVATION_CODEC_ID,
        "decodeMemory": decode,
        "prefillCompute": prefill,
        "activationCodec": codec,
    }


def _decode_memory_sample(
    target: RuntimeTarget, synchronize: Callable[[], None]
) -> float:
    # A 16 MiB tensor repeated 64 times is large enough to be memory-bound
    # without reserving a material fraction of a contributor's VRAM.
    tensor_bytes = 16 * 1024 * 1024
    elements = tensor_bytes // torch.empty((), dtype=target.dtype).element_size()
    source = torch.empty(elements, dtype=target.dtype, device=target.device)
    destination = torch.empty_like(source)
    source.fill_(0.125)
    repetitions = 64
    synchronize()
    started = time.perf_counter()
    for _ in range(repetitions):
        destination.copy_(source)
    synchronize()
    elapsed = time.perf_counter() - started
    if float(destination[0].item()) != 0.125:
        raise RuntimeError("physical memory calibration produced an invalid copy")
    # One read and one write per copied byte.
    moved_bytes = tensor_bytes * 2 * repetitions
    return moved_bytes / elapsed / 1_000_000_000


def _prefill_compute_sample(
    target: RuntimeTarget, synchronize: Callable[[], None]
) -> float:
    # The same shape and operation are used on every backend so the resulting
    # effective throughput is directly comparable by the planner.
    size = 1_024
    repetitions = 4
    left = torch.randn((size, size), dtype=target.dtype, device=target.device)
    right = torch.randn((size, size), dtype=target.dtype, device=target.device)
    output = torch.empty((size, size), dtype=target.dtype, device=target.device)
    synchronize()
    started = time.perf_counter()
    for _ in range(repetitions):
        torch.mm(left, right, out=output)
    synchronize()
    elapsed = time.perf_counter() - started
    checksum = float(output[0, 0].item())
    if not math.isfinite(checksum):
        raise RuntimeError("physical prefill calibration produced a non-finite result")
    operations = 2 * size * size * size * repetitions
    return operations / elapsed / 1_000_000_000_000


def _activation_codec_sample(
    target: RuntimeTarget, synchronize: Callable[[], None]
) -> float:
    # 256 tokens x 4096 hidden matches a real prefill activation. FP16 is the
    # only codec selected by the exact automatic planner today.
    token_count = 256
    hidden_size = 4_096
    repetitions = 4
    activation = torch.randn(
        (1, token_count, hidden_size),
        dtype=target.dtype,
        device=target.device,
    )
    wire_bytes = token_count * hidden_size * 2
    elapsed = 0.0
    decoded: torch.Tensor | None = None
    for index in range(repetitions):
        synchronize()
        started = time.perf_counter()
        encoded = encode_tensor_payload(activation, TensorCodec.FP16)
        synchronize()
        elapsed += time.perf_counter() - started

        # recv_frame fills a mutable bytearray directly. Creating that network
        # buffer is intentionally outside codec timing.
        received = bytearray(encoded.view)
        frame = Frame(
            frame_type=FrameType.PREFILL,
            flags=int(TensorCodec.FP16),
            request_id=index + 1,
            step=0,
            token_count=token_count,
            hidden_size=hidden_size,
            payload=received,
        )
        started = time.perf_counter()
        decoded = decode_tensor(frame)
        elapsed += time.perf_counter() - started
    if decoded is None or decoded.shape != activation.shape:
        raise RuntimeError("physical codec calibration returned an invalid tensor")
    if not bool(torch.isfinite(decoded).all().item()):
        raise RuntimeError("physical codec calibration returned non-finite values")
    # Count one logical activation for encode and one for decode. This yields
    # the effective throughput of the complete native codec path.
    processed_bytes = wire_bytes * 2 * repetitions
    return processed_bytes / elapsed / 1_000_000_000


def _measure_series(
    operation: Callable[[], float],
    *,
    unit: str,
    warmup_samples: int,
    samples: int,
) -> dict[str, float | int | str]:
    for _ in range(warmup_samples):
        _require_positive_finite(operation())
    values = [_require_positive_finite(operation()) for _ in range(samples)]
    mean = statistics.fmean(values)
    deviation = statistics.stdev(values)
    t_value = _T_CRITICAL_95.get(samples - 1, 1.96)
    half_width_pct = (t_value * deviation / math.sqrt(samples)) / mean * 100
    ordered = sorted(values)
    return {
        "unit": unit,
        "warmupSamples": warmup_samples,
        "samples": samples,
        "p5": _rounded(_percentile(ordered, 0.05)),
        "p50": _rounded(_percentile(ordered, 0.50)),
        "p95": _rounded(_percentile(ordered, 0.95)),
        "confidenceHalfWidthPct": _rounded(half_width_pct),
    }


def _percentile(ordered: Sequence[float], quantile: float) -> float:
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _require_positive_finite(value: float) -> float:
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError("physical calibration produced a non-positive measurement")
    return value


def _rounded(value: float) -> float:
    return float(f"{value:.9g}")


def _synchronizer(target: RuntimeTarget) -> Callable[[], None]:
    if target.device.type == "cuda":
        return lambda: torch.cuda.synchronize(target.device)
    if target.device.type == "mps":
        return torch.mps.synchronize
    if target.device.type == "xpu":
        return lambda: torch.xpu.synchronize(target.device)
    return lambda: None


def _cpu_device_name() -> str:
    try:
        import platform

        return platform.processor().strip() or "CPU"
    except Exception:
        return "CPU"


if __name__ == "__main__":
    raise SystemExit(main())
