"""Compile and inspect Mycellios-owned native GGUF stage fleets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .native_gguf import (
    NATIVE_GGUF_FLEET_MANIFEST,
    build_native_gguf_fleet,
    materialize_native_gguf_stage,
    verify_native_gguf_fleet,
    verify_native_gguf_stage,
)


def _range(value: str) -> tuple[int, int]:
    start_text, separator, end_text = value.partition(":")
    if not separator:
        raise argparse.ArgumentTypeError("range must use START:END")
    try:
        start = int(start_text)
        end = int(end_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("range bounds must be integers") from error
    if not 0 <= start < end:
        raise argparse.ArgumentTypeError("range must satisfy 0 <= START < END")
    return start, end


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mycellios native GGUF stage compiler and verifier"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser(
        "build",
        help="atomically compile one complete contiguous fleet",
    )
    build.add_argument("--source", type=Path, required=True)
    build.add_argument("--config", type=Path, required=True)
    build.add_argument("--destination", type=Path, required=True)
    build.add_argument("--model-source", required=True)
    build.add_argument("--model-revision")
    ranges = build.add_mutually_exclusive_group(required=True)
    ranges.add_argument("--layers-per-stage", type=int)
    ranges.add_argument(
        "--range",
        type=_range,
        action="append",
        dest="ranges",
        help="repeat ordered START:END ranges covering the complete model",
    )

    verify = commands.add_parser(
        "verify",
        help="verify a native stage package or complete fleet",
    )
    verify.add_argument("path", type=Path)
    verify.add_argument("--expected-id")

    materialize = commands.add_parser(
        "materialize",
        help="debug-only stage-local SafeTensors materialization",
    )
    materialize.add_argument("package", type=Path)
    materialize.add_argument("destination", type=Path)
    return parser.parse_args(argv)


def _stage_document(stage: Any) -> dict[str, Any]:
    return {
        "packageId": stage.package_id,
        "packageIdentity": stage.package_identity,
        "modelIdentity": stage.artifact_identity,
        "modelSource": stage.model_source,
        "modelRevision": stage.model_revision,
        "layerStart": stage.layer_start,
        "layerEnd": stage.layer_end,
        "totalLayers": stage.total_layers,
        "architecture": stage.architecture,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "build":
        fleet = build_native_gguf_fleet(
            args.source,
            args.destination,
            config_source=args.config,
            model_source=args.model_source,
            model_revision=args.model_revision,
            ranges=args.ranges,
            layers_per_stage=args.layers_per_stage,
        )
        result = {
            "kind": "native-gguf-fleet",
            "path": str(fleet.root),
            "fleetId": fleet.fleet_id,
            "modelIdentity": fleet.artifact_identity,
            "totalLayers": fleet.total_layers,
            "stages": [_stage_document(stage) for stage in fleet.stages],
        }
    elif args.command == "verify":
        if (args.path / NATIVE_GGUF_FLEET_MANIFEST).is_file():
            fleet = verify_native_gguf_fleet(
                args.path,
                expected_fleet_id=args.expected_id,
            )
            result = {
                "kind": "native-gguf-fleet",
                "path": str(fleet.root),
                "fleetId": fleet.fleet_id,
                "modelIdentity": fleet.artifact_identity,
                "totalLayers": fleet.total_layers,
                "stages": [_stage_document(stage) for stage in fleet.stages],
            }
        else:
            stage = verify_native_gguf_stage(
                args.path,
                expected_package_id=args.expected_id,
            )
            result = {
                "kind": "native-gguf-stage",
                "path": str(stage.root),
                **_stage_document(stage),
            }
    else:
        destination = materialize_native_gguf_stage(
            args.package,
            args.destination,
        )
        result = {
            "kind": "native-gguf-materialization",
            "path": str(destination),
        }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
