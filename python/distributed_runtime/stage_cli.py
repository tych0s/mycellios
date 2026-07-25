from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from .dense_tiering import (
    add_dense_tiering_arguments,
    dense_tiering_config_from_args,
)
from .model import StageModelSpec, model_artifact_reference, resolve_model_snapshot
from .native_gguf import verify_native_gguf_stage
from .native_gguf_runtime import (
    add_native_gguf_arguments,
    native_gguf_runtime_from_args,
)
from .paged_stage import add_paged_kv_arguments, paged_kv_config_from_args
from .protocol import TensorCodec
from .ram_backed_moe_runtime import (
    add_ram_backed_moe_arguments,
    ram_backed_moe_config_from_args,
)
from .runtime_policy import reject_external_backend_arguments
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

    def put_startup(self, value: dict[str, Any]) -> None:
        self.put(value)


def _uint64_argument(value: str) -> int:
    try:
        parsed = int(value, 0)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be an integer") from error
    if not 0 <= parsed <= (1 << 64) - 1:
        raise argparse.ArgumentTypeError("value must fit uint64")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    arguments = list(sys.argv[1:] if argv is None else argv)
    reject_external_backend_arguments(arguments)
    parser = argparse.ArgumentParser(
        description="Run one persistent remote layer stage for a GDLP pipeline."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--model-artifact-identity")
    parser.add_argument("--stage-package-identity")
    parser.add_argument("--model-canonical-source")
    parser.add_argument("--model-canonical-revision")
    parser.add_argument("--pipeline-snapshot-identity", type=_uint64_argument)
    add_ram_backed_moe_arguments(parser)
    add_paged_kv_arguments(parser)
    add_native_gguf_arguments(parser)
    parser.add_argument("--layer-start", type=int, required=True)
    parser.add_argument("--layer-end", type=int, required=True)
    parser.add_argument("--total-layers", type=int, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--device",
        default="auto",
        help=(
            "Dense Torch execution device: auto, cpu, cuda[:index], mps or xpu[:index]. "
            "An explicit accelerator request fails if it is unavailable."
        ),
    )
    add_dense_tiering_arguments(parser)
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
    parser.add_argument("--sealed-wave-tokens", type=int)
    parser.add_argument("--max-prefill-chunk-tokens", type=int)
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
    parser.add_argument("--max-physical-batch-size", type=int, default=8)
    parser.add_argument("--physical-batch-window-ms", type=float, default=0.5)
    parser.add_argument(
        "--max-speculative-branches",
        type=int,
        default=0,
        help="Sealed concurrent exact KV children; zero disables FORK/PROMOTE.",
    )
    parser.add_argument(
        "--max-speculative-branch-tokens",
        type=int,
        default=0,
        help="Sealed context-token ceiling for every speculative KV child.",
    )
    parser.add_argument(
        "--max-speculative-kv-bytes",
        type=int,
        default=0,
        help="Sealed total live speculative KV bytes, checked before allocation.",
    )
    parser.add_argument("--metrics-jsonl", type=Path)
    return parser.parse_args(arguments)


def build_config(args: argparse.Namespace) -> StageProcessConfig:
    if (args.sealed_wave_tokens is None) != (
        args.max_prefill_chunk_tokens is None
    ):
        raise ValueError(
            "sealed-wave-tokens and max-prefill-chunk-tokens must be supplied together"
        )
    ram_backed_moe = ram_backed_moe_config_from_args(args)
    paged_kv = paged_kv_config_from_args(args)
    native_gguf = native_gguf_runtime_from_args(args)
    dense_tiering = dense_tiering_config_from_args(args)
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
    if native_gguf is not None:
        if (
            ram_backed_moe is not None
            or paged_kv is not None
            or args.cell_fixture is not None
        ):
            raise ValueError(
                "native GGUF cannot be combined with another model, package or "
                "tensor-parallel backend"
            )
        package = verify_native_gguf_stage(
            native_gguf.package,
            expected_package_id=native_gguf.package_id,
            expected_layer_start=args.layer_start,
            expected_layer_end=args.layer_end,
            expected_total_layers=args.total_layers,
        )
        supplied_coordinates = (
            ("model-artifact-identity", args.model_artifact_identity, package.artifact_identity),
            ("stage-package-identity", args.stage_package_identity, package.package_identity),
            ("model-canonical-source", args.model_canonical_source, package.model_source),
            (
                "model-canonical-revision",
                args.model_canonical_revision,
                package.model_revision,
            ),
            ("revision", args.revision, package.model_revision),
        )
        for name, supplied, sealed in supplied_coordinates:
            if supplied is not None and supplied != sealed:
                raise ValueError(
                    f"{name} differs from the authenticated native GGUF package"
                )
        snapshot = "sealed-native-gguf"
        pipeline_id = (
            args.pipeline_snapshot_identity
            if args.pipeline_snapshot_identity is not None
            else int(package.artifact_identity.removeprefix("sha256:")[:16], 16)
        )
        spec = StageModelSpec(
            snapshot,
            args.layer_start,
            args.layer_end,
            args.total_layers,
            args.threads,
            artifact_identity=package.artifact_identity,
            canonical_model_source=package.model_source,
            canonical_model_revision=package.model_revision,
            stage_package_identity=package.package_identity,
        )
    elif ram_backed_moe is not None:
        if paged_kv is not None or any(
            value is not None
            for value in (
                args.cell_fixture,
                args.cell_world_size,
                args.stage_package_identity,
            )
        ):
            raise ValueError(
                "paged KV, RAM-backed MoE and tensor-parallel cell backends "
                "are mutually exclusive"
            )
        snapshot = str(Path(args.model).expanduser().resolve())
        pipeline_id = args.pipeline_snapshot_identity
        spec = StageModelSpec(
            snapshot,
            args.layer_start,
            args.layer_end,
            args.total_layers,
            args.threads,
            artifact_identity=ram_backed_moe.artifact_identity,
            canonical_model_source=(
                f"content-addressed://{ram_backed_moe.artifact_identity}"
            ),
        )
    else:
        snapshot = resolve_model_snapshot(args.model, args.revision)
        artifact = model_artifact_reference(
            snapshot,
            artifact_identity=args.model_artifact_identity,
            canonical_source=args.model_canonical_source,
            canonical_revision=args.model_canonical_revision,
        )
        pipeline_id = (
            args.pipeline_snapshot_identity
            if args.pipeline_snapshot_identity is not None
            else artifact.snapshot_identity
        )
        spec = StageModelSpec(
            snapshot,
            args.layer_start,
            args.layer_end,
            args.total_layers,
            args.threads,
            artifact_identity=artifact.identity,
            canonical_model_source=artifact.canonical_source,
            canonical_model_revision=artifact.canonical_revision,
            stage_package_identity=args.stage_package_identity,
        )
    if paged_kv is not None and any(
        value is not None for value in (args.cell_fixture, args.cell_world_size)
    ):
        raise ValueError(
            "paged KV, RAM-backed MoE and tensor-parallel cell backends "
            "are mutually exclusive"
        )
    return StageProcessConfig(
        spec=spec,
        pipeline_id=pipeline_id,
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
        device=args.device,
        dense_tiering=dense_tiering,
        connect_timeout_seconds=args.connect_timeout_seconds,
        sealed_wave_tokens=args.sealed_wave_tokens,
        max_prefill_chunk_tokens=args.max_prefill_chunk_tokens,
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
        max_physical_batch_size=args.max_physical_batch_size,
        physical_batch_window_ms=args.physical_batch_window_ms,
        max_speculative_branches=args.max_speculative_branches,
        max_speculative_branch_tokens=args.max_speculative_branch_tokens,
        max_speculative_kv_bytes=args.max_speculative_kv_bytes,
        ram_backed_moe=ram_backed_moe,
        paged_kv=paged_kv,
        native_gguf=native_gguf,
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
                "requested_device": config.device,
                "pagedKv": (
                    None
                    if config.paged_kv is None
                    else {
                        **config.paged_kv.to_document(),
                        "configurationId": config.paged_kv.configuration_id,
                    }
                ),
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
                "nativeGguf": (
                    None
                    if config.native_gguf is None
                    else {
                        "package_id": config.native_gguf.package_id,
                        "model_identity": config.spec.artifact_identity,
                        "stage_identity": config.spec.stage_package_identity,
                        "external_runtime_required": False,
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
