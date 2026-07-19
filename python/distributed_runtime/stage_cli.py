from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from .model import StageModelSpec, model_snapshot_identity, resolve_model_snapshot
from .protocol import TensorCodec
from .stage import StageProcessConfig, run_stage_process


class ReadyPrinter:
    def set(self) -> None:
        print("stage_ready", file=sys.stderr, flush=True)


class JsonMetricSink:
    def __init__(self, path: Path | None) -> None:
        self.path = path

    def put(self, value: dict[str, Any]) -> None:
        try:
            rendered = json.dumps(value, sort_keys=True)
            if self.path is None:
                print(rendered, file=sys.stderr, flush=True)
                return
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(rendered + "\n")
        except BaseException:
            # Metrics are deliberately best-effort: a full disk, broken stderr,
            # or non-serializable observation must never stop token inference.
            pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one persistent remote layer stage for a GDLP pipeline."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--layer-start", type=int, required=True)
    parser.add_argument("--layer-end", type=int, required=True)
    parser.add_argument("--total-layers", type=int, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--listen-port", type=int, required=True)
    parser.add_argument("--next-host")
    parser.add_argument("--next-port", type=int)
    parser.add_argument("--next-layer-end", type=int)
    parser.add_argument("--return-host", required=True)
    parser.add_argument("--return-port", type=int, required=True)
    parser.add_argument(
        "--codec",
        choices=(
            "fp32",
            "fp16",
            "int8",
            "int8-grouped",
            "int8-hadamard",
            "int8-grouped-deflate",
            "int8-hadamard-deflate",
        ),
        default="fp16",
    )
    parser.add_argument("--one-way-delay-ms", type=float, default=0.0)
    parser.add_argument("--bandwidth-mbps", type=float, default=0.0)
    parser.add_argument("--connect-timeout-seconds", type=float, default=180.0)
    parser.add_argument(
        "--cell-fixture",
        help=(
            "rank-sharded cell fixture (/1 single-layer or /2 multi-layer; "
            "external mode requires /2)"
        ),
    )
    parser.add_argument("--cell-world-size", type=int)
    parser.add_argument("--cell-manifest-sha256")
    parser.add_argument(
        "--cell-collective-backend",
        choices=("gloo", "nccl"),
        default="gloo",
    )
    parser.add_argument(
        "--cell-compute-dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument(
        "--cell-device",
        action="append",
        default=[],
        help="Rank-local device in rank order; repeat once per cell member.",
    )
    parser.add_argument("--cell-operation-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--cell-mode", choices=("local", "external"), default="local")
    parser.add_argument("--cell-control-host")
    parser.add_argument("--cell-control-port", type=int)
    parser.add_argument("--cell-control-advertise-host")
    parser.add_argument("--cell-distributed-advertise-host")
    parser.add_argument("--cell-distributed-port", type=int)
    parser.add_argument("--cell-startup-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--metrics-jsonl", type=Path)
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> StageProcessConfig:
    downstream_values = (args.next_host, args.next_port, args.next_layer_end)
    if any(value is not None for value in downstream_values) and not all(
        value is not None for value in downstream_values
    ):
        raise ValueError("next-host, next-port and next-layer-end must be supplied together")
    cell_values = (args.cell_fixture, args.cell_world_size)
    if any(value is not None for value in cell_values) and not all(
        value is not None for value in cell_values
    ):
        raise ValueError("cell-fixture and cell-world-size must be supplied together")
    codec = {
        "fp32": TensorCodec.FP32,
        "fp16": TensorCodec.FP16,
        "int8": TensorCodec.INT8,
        "int8-grouped": TensorCodec.INT8_GROUPED,
        "int8-hadamard": TensorCodec.INT8_HADAMARD,
        "int8-grouped-deflate": TensorCodec.INT8_GROUPED_DEFLATE,
        "int8-hadamard-deflate": TensorCodec.INT8_HADAMARD_DEFLATE,
    }[args.codec]
    snapshot = resolve_model_snapshot(args.model, args.revision)
    return StageProcessConfig(
        spec=StageModelSpec(
            snapshot,
            args.layer_start,
            args.layer_end,
            args.total_layers,
            args.threads,
        ),
        pipeline_id=model_snapshot_identity(snapshot),
        listen_host=args.listen_host,
        listen_port=args.listen_port,
        next_host=args.next_host,
        next_port=args.next_port,
        next_layer_end=args.next_layer_end,
        return_host=args.return_host,
        return_port=args.return_port,
        codec=codec,
        one_way_delay_ms=args.one_way_delay_ms,
        bandwidth_mbps=args.bandwidth_mbps,
        connect_timeout_seconds=args.connect_timeout_seconds,
        cell_fixture=args.cell_fixture,
        cell_manifest_sha256=args.cell_manifest_sha256,
        cell_world_size=args.cell_world_size,
        cell_collective_backend=args.cell_collective_backend,
        cell_devices=tuple(args.cell_device),
        cell_compute_dtype=args.cell_compute_dtype,
        cell_operation_timeout_seconds=args.cell_operation_timeout_seconds,
        cell_mode=args.cell_mode,
        cell_control_host=args.cell_control_host,
        cell_control_port=args.cell_control_port,
        cell_control_advertise_host=args.cell_control_advertise_host,
        cell_distributed_advertise_host=args.cell_distributed_advertise_host,
        cell_distributed_port=args.cell_distributed_port,
        cell_startup_timeout_seconds=args.cell_startup_timeout_seconds,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = build_config(args)
    print(
        json.dumps(
            {
                "event": "loading_stage",
                "layers": [config.spec.layer_start, config.spec.layer_end],
                "pipeline_id": config.pipeline_id,
                "listen": [config.listen_host, config.listen_port],
                "next": [config.next_host, config.next_port],
                "return": [config.return_host, config.return_port],
                "codec": config.codec.name.lower(),
                "cell": (
                    None
                    if config.cell_fixture is None
                    else {
                        "fixture": config.cell_fixture,
                        "manifest_sha256": config.cell_manifest_sha256,
                        "world_size": config.cell_world_size,
                        "collective_backend": config.cell_collective_backend,
                        "compute_dtype": config.cell_compute_dtype,
                        "devices": list(config.cell_devices),
                        "mode": config.cell_mode,
                        "control": [
                            config.cell_control_advertise_host,
                            config.cell_control_port,
                        ],
                        "distributed": [
                            config.cell_distributed_advertise_host,
                            config.cell_distributed_port,
                        ],
                    }
                ),
            },
            sort_keys=True,
        ),
        file=sys.stderr,
        flush=True,
    )
    run_stage_process(config, ReadyPrinter(), JsonMetricSink(args.metrics_jsonl))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
