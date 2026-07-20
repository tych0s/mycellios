"""Reproducible loopback microbenchmark for Resident Expert RPC.

This measures framing, CPU tensor serialization, a persistent TCP round trip
and response reconstruction.  The injected owner only clones activations, so
the result deliberately excludes expert compute, CUDA staging and real-network
latency.  It is a transport floor, not a model throughput prediction.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import threading
import time
from typing import Sequence

import torch

from .ram_expert_cache import ExpertKey
from .resident_expert_mesh import (
    OwnerCoalescedExpertBatchItem,
    OwnerExpertBatchItem,
    OwnerExpertBatchResult,
)
from .resident_expert_rpc import (
    ResidentExpertRpcClient,
    ResidentExpertRpcInventoryEntry,
    ResidentExpertRpcLimits,
    ResidentExpertRpcServer,
)


_DTYPES: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


class _EchoOwner:
    node_id = "loopback-owner"

    def __init__(self, keys: Sequence[ExpertKey]) -> None:
        self._keys = frozenset(keys)
        self.calls = 0

    @staticmethod
    def content_id(key: ExpertKey) -> str:
        return f"benchmark-layer-{key.layer}-expert-{key.expert}"

    def has_expert(self, key: ExpertKey, content_id: str) -> bool:
        return key in self._keys and content_id == self.content_id(key)

    def is_expert_resident(self, key: ExpertKey, content_id: str) -> bool:
        return self.has_expert(key, content_id)

    @torch.no_grad()
    def execute_batch(
        self,
        items: Sequence[OwnerExpertBatchItem],
    ) -> tuple[OwnerExpertBatchResult, ...]:
        self.calls += 1
        return tuple(
            OwnerExpertBatchResult(item.key, item.activations.clone())
            for item in items
        )

    @torch.no_grad()
    def execute_coalesced_batch(
        self,
        shared_activations: torch.Tensor,
        items: Sequence[OwnerCoalescedExpertBatchItem],
    ) -> tuple[OwnerExpertBatchResult, ...]:
        self.calls += 1
        return tuple(
            OwnerExpertBatchResult(
                item.key,
                shared_activations.index_select(
                    0,
                    torch.tensor(item.row_indices, dtype=torch.long),
                ),
            )
            for item in items
        )


def _percentile(sorted_values: Sequence[float], percentile: float) -> float:
    index = int(percentile * (len(sorted_values) - 1))
    return float(sorted_values[index])


def _latency_document(elapsed_ms: Sequence[float]) -> dict[str, float]:
    ordered = sorted(elapsed_ms)
    return {
        "min": min(elapsed_ms),
        "p50": statistics.median(elapsed_ms),
        "p95": _percentile(ordered, 0.95),
        "max": max(elapsed_ms),
        "mean": statistics.fmean(elapsed_ms),
    }


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_value(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ("git", *args),
            cwd=Path(__file__).resolve().parents[2],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _environment_document() -> dict[str, object]:
    source = Path(__file__).resolve()
    root = source.parents[2]
    dirty = _git_value("status", "--short")
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logicalCpuCount": os.cpu_count(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchThreads": torch.get_num_threads(),
        "torchInteropThreads": torch.get_num_interop_threads(),
        "cudaAvailable": torch.cuda.is_available(),
        "cudaRuntime": torch.version.cuda,
        "gitHead": _git_value("rev-parse", "HEAD"),
        "gitWorktreeDirty": bool(dirty) if dirty is not None else None,
        "sourceSha256": {
            "benchmark": _sha256_file(source),
            "rpc": _sha256_file(root / "python/distributed_runtime/resident_expert_rpc.py"),
            "meshAbi": _sha256_file(root / "python/distributed_runtime/resident_expert_mesh.py"),
        },
    }


def run_benchmark(
    *,
    iterations: int,
    warmup: int,
    positions: int,
    hidden_size: int,
    experts: int,
    dtype_name: str,
    transport: str = "v1",
) -> dict[str, object]:
    started_at_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    for name, value in (
        ("iterations", iterations),
        ("positions", positions),
        ("hidden_size", hidden_size),
        ("experts", experts),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(warmup, int) or isinstance(warmup, bool) or warmup < 0:
        raise ValueError("warmup must be a non-negative integer")
    try:
        dtype = _DTYPES[dtype_name]
    except KeyError as error:
        raise ValueError(f"unsupported dtype {dtype_name!r}") from error
    if transport not in {"v1", "coalesced", "compare"}:
        raise ValueError("transport must be 'v1', 'coalesced' or 'compare'")

    keys = tuple(ExpertKey(0, expert) for expert in range(experts))
    owner = _EchoOwner(keys)
    tensor_elements = positions * hidden_size
    tensor_bytes = tensor_elements * torch.empty((), dtype=dtype).element_size()
    batch_payload_bytes = experts * tensor_bytes
    payload_limit = max(1024 * 1024, batch_payload_bytes * 2)
    limits = ResidentExpertRpcLimits(
        max_header_bytes=256 * 1024,
        max_payload_bytes=payload_limit,
        max_batch_items=max(8, experts),
        max_inventory_items=max(8, experts),
        max_tensor_elements=max(
            16_777_216,
            tensor_elements,
            experts * positions,
        ),
        max_host_transient_bytes=256 * 1024 + 2 * payload_limit,
        io_timeout_seconds=10.0,
    )
    inventory = tuple(
        ResidentExpertRpcInventoryEntry(
            key=key,
            content_id=owner.content_id(key),
            shape=(hidden_size, hidden_size),
            dtype=dtype_name,
        )
        for key in keys
    )
    server = ResidentExpertRpcServer(
        "127.0.0.1",
        0,
        node_id=owner.node_id,
        owner=owner,
        inventory=inventory,
        limits=limits,
        accept_poll_seconds=0.01,
    )
    _, port = server.bind()
    stop = threading.Event()
    server_thread = threading.Thread(
        target=server.serve_forever,
        args=(stop,),
        name="resident-expert-rpc-loopback-benchmark",
        daemon=True,
    )
    server_thread.start()
    client: ResidentExpertRpcClient | None = None
    try:
        client = ResidentExpertRpcClient(
            "127.0.0.1",
            port,
            client_node_id="loopback-root",
            expected_node_id=owner.node_id,
            limits=limits,
        )
        base = (
            (torch.arange(tensor_elements, dtype=torch.float32) % 1024) - 512
        ).div(512).reshape(positions, hidden_size).to(dtype=dtype)
        if not bool(torch.isfinite(base).all().item()):
            raise RuntimeError("loopback benchmark generated non-finite activations")
        input_sha256 = hashlib.sha256(
            base.contiguous().view(torch.uint8).cpu().numpy().tobytes()
        ).hexdigest()
        items = tuple(
            OwnerExpertBatchItem(key, owner.content_id(key), base)
            for key in keys
        )
        coalesced_items = tuple(
            OwnerCoalescedExpertBatchItem(
                key,
                owner.content_id(key),
                tuple(range(positions)),
                require_resident=True,
            )
            for key in keys
        )
        if transport in {"coalesced", "compare"}:
            client.probe_exact_input_coalescing()

        def execute(mode: str) -> tuple[OwnerExpertBatchResult, ...]:
            if mode == "coalesced":
                return client.execute_coalesced_batch(base, coalesced_items)
            return client.execute_batch(items)

        schedule = (
            ("v1", "coalesced", "coalesced", "v1")
            if transport == "compare"
            else (transport,)
        )
        for _ in range(warmup):
            for mode in schedule:
                execute(mode)

        measurements: list[dict[str, object]] = []
        reference_by_mode: dict[str, tuple[OwnerExpertBatchResult, ...]] = {}
        for cycle in range(iterations):
            for slot, mode in enumerate(schedule):
                before_call = client.telemetry_snapshot()
                started = time.perf_counter_ns()
                result = execute(mode)
                elapsed_ns = time.perf_counter_ns() - started
                after_call = client.telemetry_snapshot()
                sent_bytes = after_call.bytes_sent - before_call.bytes_sent
                received_bytes = (
                    after_call.bytes_received - before_call.bytes_received
                )
                measurements.append(
                    {
                        "cycle": cycle,
                        "slot": slot,
                        "mode": mode,
                        "elapsedNs": elapsed_ns,
                        "applicationBytesSent": sent_bytes,
                        "applicationBytesReceived": received_bytes,
                    }
                )
                if len(result) != experts or any(
                    not torch.equal(value.output, base) for value in result
                ):
                    raise RuntimeError("loopback benchmark response failed parity")
                reference = reference_by_mode.setdefault(mode, result)
                if any(
                    not torch.equal(current.output, expected.output)
                    for current, expected in zip(result, reference, strict=True)
                ):
                    raise RuntimeError("loopback benchmark is not deterministic")

        if transport == "compare":
            v1_reference = reference_by_mode["v1"]
            coalesced_reference = reference_by_mode["coalesced"]
            if any(
                not torch.equal(v1.output, packed.output)
                for v1, packed in zip(
                    v1_reference,
                    coalesced_reference,
                    strict=True,
                )
            ):
                raise RuntimeError("v1/coalesced benchmark parity failed")

        ended_at_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        def mode_document(mode: str) -> dict[str, object]:
            mode_measurements = [
                measurement
                for measurement in measurements
                if measurement["mode"] == mode
            ]
            elapsed_ms = [
                int(measurement["elapsedNs"]) / 1_000_000
                for measurement in mode_measurements
            ]
            measured_application_bytes = sum(
                int(measurement["applicationBytesSent"])
                + int(measurement["applicationBytesReceived"])
                for measurement in mode_measurements
            )
            request_payload_bytes = (
                batch_payload_bytes
                if mode == "v1"
                else tensor_bytes + experts * positions * 4
            )
            row_index_bytes = 0 if mode == "v1" else experts * positions * 4
            return {
                "measuredCalls": len(elapsed_ms),
                "requestPayloadBytes": request_payload_bytes,
                "responsePayloadBytes": batch_payload_bytes,
                "nominalRoundTripTensorBytes": (
                    batch_payload_bytes * 2
                    if mode == "v1"
                    else tensor_bytes + batch_payload_bytes
                ),
                "nominalRoundTripPayloadBytes": (
                    request_payload_bytes + batch_payload_bytes
                ),
                "rowIndexBytes": row_index_bytes,
                "latencyMs": _latency_document(elapsed_ms),
                "effectiveApplicationFrameMiBPerSecond": (
                    measured_application_bytes / (1024 * 1024)
                ) / (sum(elapsed_ms) / 1000),
                "measuredApplicationFrameBytes": measured_application_bytes,
                "meanMeasuredApplicationFrameBytesPerCall": (
                    measured_application_bytes / len(elapsed_ms)
                ),
            }

        configuration: dict[str, object] = {
            "transport": transport,
            "iterations": iterations,
            "warmupIterations": warmup,
            "positions": positions,
            "hiddenSize": hidden_size,
            "expertsPerOwnerBatch": experts,
            "dtype": dtype_name,
            "persistentTcp": True,
        }
        if transport == "compare":
            configuration["abbaOrder"] = list(schedule)

        common: dict[str, object] = {
            "schema": "gdlp-resident-expert-rpc-loopback-benchmark/3",
            "kind": (
                "abba-transport-compare"
                if transport == "compare"
                else "single-transport"
            ),
            "startedAtUtc": started_at_utc,
            "endedAtUtc": ended_at_utc,
            "scope": (
                "persistent TCP loopback plus CPU framing; excludes expert compute, "
                "CUDA staging, TCP/IP overhead and external network"
            ),
            "positions": positions,
            "hiddenSize": hidden_size,
            "expertsPerOwnerBatch": experts,
            "dtype": dtype_name,
            "transport": transport,
            "warmupIterations": warmup,
            "measuredIterations": iterations,
            "ownerBatchCallsIncludingWarmup": owner.calls,
            "parity": "bit-exact identity transport",
            "configuration": configuration,
            "method": {
                "clock": "time.perf_counter_ns",
                "p50": "statistics.median",
                "p95": "sorted[floor(0.95 * (n - 1))]",
                "applicationFrameBytes": (
                    "20-byte prefix plus canonical JSON plus payload; "
                    "excludes TCP/IP"
                ),
                "excluded": [
                    "connection",
                    "handshake",
                    "capability-probe",
                    "warmup",
                    "close",
                ],
                "ownerOperator": "identity clone/index_select; no expert compute",
                "parityCheckedEveryCall": True,
            },
            "measurements": measurements,
            "correctness": {
                "checkedCalls": len(measurements),
                "parityFailures": 0,
                "result": "bit-exact identity transport",
            },
            "input": {
                "generator": "bounded-arange-modulo-1024/1",
                "shape": [positions, hidden_size],
                "finite": True,
                "sha256": input_sha256,
            },
            "environment": _environment_document(),
        }
        if transport != "compare":
            common.update(mode_document(transport))
            return common

        v1_document = mode_document("v1")
        coalesced_document = mode_document("coalesced")
        v1_mean = float(v1_document["latencyMs"]["mean"])  # type: ignore[index]
        coalesced_mean = float(
            coalesced_document["latencyMs"]["mean"]  # type: ignore[index]
        )
        v1_application_bytes = float(
            v1_document["meanMeasuredApplicationFrameBytesPerCall"]
        )
        coalesced_application_bytes = float(
            coalesced_document["meanMeasuredApplicationFrameBytesPerCall"]
        )
        common.update(
            {
                "abbaOrder": list(schedule),
                "results": {
                    "v1": v1_document,
                    "coalesced": coalesced_document,
                },
                "comparison": {
                    "meanLatencyReductionPercent": (
                        100.0 * (1.0 - coalesced_mean / v1_mean)
                    ),
                    "meanApplicationFrameReductionPercent": (
                        100.0
                        * (
                            1.0
                            - coalesced_application_bytes
                            / v1_application_bytes
                        )
                    ),
                },
            }
        )
        return common
    finally:
        if client is not None:
            client.close()
        stop.set()
        server_thread.join(timeout=5)
        server.close()
        if server_thread.is_alive():
            raise RuntimeError("loopback benchmark server did not stop")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--positions", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=8192)
    parser.add_argument("--experts", type=int, default=2)
    parser.add_argument("--dtype", choices=tuple(_DTYPES), default="float16")
    parser.add_argument(
        "--transport",
        choices=("v1", "coalesced", "compare"),
        default="v1",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument(
        "--format",
        choices=("json", "markdown"),
        default="json",
    )
    return parser


def _markdown_report(result: dict[str, object]) -> str:
    if result.get("kind") != "abba-transport-compare":
        return "```json\n" + json.dumps(result, indent=2, sort_keys=True) + "\n```"
    results = result["results"]
    assert isinstance(results, dict)
    lines = [
        "| Modo | Calls | Frame aplicacion/call | Media ms | P50 ms | P95 ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode in ("v1", "coalesced"):
        row = results[mode]
        latency = row["latencyMs"]
        lines.append(
            f"| {mode} | {row['measuredCalls']} | "
            f"{row['meanMeasuredApplicationFrameBytesPerCall']:.0f} | "
            f"{latency['mean']:.3f} | {latency['p50']:.3f} | "
            f"{latency['p95']:.3f} |"
        )
    comparison = result["comparison"]
    lines.extend(
        [
            "",
            f"Reduccion media de latencia: "
            f"{comparison['meanLatencyReductionPercent']:.2f}%.",
            f"Reduccion de bytes de frame de aplicacion: "
            f"{comparison['meanApplicationFrameReductionPercent']:.2f}%.",
            "",
            "No incluye TCP/IP, computo de experto, CUDA ni LAN/WAN.",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_benchmark(
        iterations=args.iterations,
        warmup=args.warmup,
        positions=args.positions,
        hidden_size=args.hidden_size,
        experts=args.experts,
        dtype_name=args.dtype,
        transport=args.transport,
    )
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(serialized, encoding="utf-8")
    if args.format == "markdown":
        print(_markdown_report(result))
    else:
        print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
