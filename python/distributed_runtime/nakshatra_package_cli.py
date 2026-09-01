from __future__ import annotations

import argparse
import json
from pathlib import Path

from .nakshatra_package import (
    load_nakshatra_stage_package,
    seal_nakshatra_stage_package,
)


def _uint64(value: str) -> int:
    try:
        parsed = int(value, 0)
    except ValueError as error:
        raise argparse.ArgumentTypeError("pipeline id must be an integer") from error
    if not 0 <= parsed <= (1 << 64) - 1:
        raise argparse.ArgumentTypeError("pipeline id must fit uint64")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seal or verify a pinned Nakshatra sub-GGUF stage package."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    seal = commands.add_parser("seal")
    seal.add_argument("source_sub_gguf", type=Path)
    seal.add_argument("destination", type=Path)
    seal.add_argument("--model-source", required=True)
    seal.add_argument("--model-revision")
    seal.add_argument("--model-content-sha256", required=True)
    seal.add_argument("--pipeline-id", required=True, type=_uint64)
    seal.add_argument("--layer-start", required=True, type=int)
    seal.add_argument("--layer-end", required=True, type=int)
    seal.add_argument("--total-layers", required=True, type=int)
    seal.add_argument("--hidden-size", required=True, type=int)
    seal.add_argument("--vocab-size", required=True, type=int)
    seal.add_argument("--weight-type", required=True)
    seal.add_argument("--max-context-tokens", required=True, type=int)
    seal.add_argument("--keep-token-embeddings", action="store_true")

    verify = commands.add_parser("verify")
    verify.add_argument("package", type=Path)
    verify.add_argument("--expected-package-id")
    verify.add_argument("--expected-manifest-sha256")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "seal":
        package = seal_nakshatra_stage_package(
            args.source_sub_gguf,
            args.destination,
            model_source=args.model_source,
            model_revision=args.model_revision,
            model_content_sha256=args.model_content_sha256,
            pipeline_id=args.pipeline_id,
            layer_start=args.layer_start,
            layer_end=args.layer_end,
            total_layers=args.total_layers,
            hidden_size=args.hidden_size,
            vocab_size=args.vocab_size,
            weight_type=args.weight_type,
            max_context_tokens=args.max_context_tokens,
            keep_token_embeddings=args.keep_token_embeddings,
        )
    else:
        package = load_nakshatra_stage_package(
            args.package,
            expected_package_id=args.expected_package_id,
            expected_manifest_sha256=args.expected_manifest_sha256,
        )
    print(
        json.dumps(
            {
                "package": str(package.root),
                "packageId": package.package_id,
                "manifestSha256": package.manifest_sha256,
                "artifactSha256": package.artifact_sha256,
                "artifactBytes": package.artifact_bytes,
                "pipelineId": package.pipeline_id,
                "layers": [package.layer_start, package.layer_end],
                "mode": package.mode,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
