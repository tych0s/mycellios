from __future__ import annotations

import argparse
import json
import sys

from .external_cell import ExternalCellMemberConfig, run_external_cell_member


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Join one external rank to a GDLP tensor-parallel LAN cell. "
            "Rank zero is always owned and launched by the anchor stage."
        )
    )
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--pipeline-id", type=int, required=True)
    parser.add_argument("--layer-start", type=int, required=True)
    parser.add_argument("--layer-end", type=int, required=True)
    parser.add_argument("--control-host", required=True)
    parser.add_argument("--control-port", type=int, required=True)
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
    parser.add_argument("--cell-device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--connect-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--operation-timeout-seconds", type=float, default=30.0)
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> ExternalCellMemberConfig:
    if args.rank == 0:
        raise ValueError("rank zero is launched internally by the anchor stage")
    return ExternalCellMemberConfig(
        fixture=args.fixture,
        rank=args.rank,
        world_size=args.world_size,
        pipeline_id=args.pipeline_id,
        layer_start=args.layer_start,
        layer_end=args.layer_end,
        control_host=args.control_host,
        control_port=args.control_port,
        collective_backend=args.cell_collective_backend,
        device=args.cell_device,
        compute_dtype=args.cell_compute_dtype,
        threads=args.threads,
        connect_timeout_seconds=args.connect_timeout_seconds,
        operation_timeout_seconds=args.operation_timeout_seconds,
    )


def main(argv: list[str] | None = None) -> int:
    config = build_config(parse_args(argv))
    print(
        json.dumps(
            {
                "event": "joining_external_cell",
                "rank": config.rank,
                "world_size": config.world_size,
                "pipeline_id": config.pipeline_id,
                "layers": [config.layer_start, config.layer_end],
                "control": [config.control_host, config.control_port],
                "fixture": config.fixture,
                "collective_backend": config.collective_backend,
                "compute_dtype": config.compute_dtype,
                "device": config.device,
            },
            sort_keys=True,
        ),
        file=sys.stderr,
        flush=True,
    )
    run_external_cell_member(config)
    print(
        json.dumps(
            {
                "event": "external_cell_stopped",
                "rank": config.rank,
                "pipeline_id": config.pipeline_id,
            },
            sort_keys=True,
        ),
        file=sys.stderr,
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
