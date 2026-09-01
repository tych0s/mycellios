from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .llama_cpp import (
    benchmark_llama_cpp,
    build_partial_stage_deployment,
    build_whole_model_deployment,
    inspect_gguf,
    parse_layer_package_manifest,
    probe_llama_cpp,
    verify_layer_package_files,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe llama.cpp and emit an identity-sealed GDLP deployment manifest."
    )
    parser.add_argument("--runtime-dir", required=True)
    artifact = parser.add_mutually_exclusive_group(required=True)
    artifact.add_argument("--model", help="complete single-file GGUF")
    artifact.add_argument(
        "--layer-package", help="explicit gdlp-llama-layer-package/1 JSON"
    )
    parser.add_argument(
        "--stage-executor",
        help="gdlp-stage-executor/1 JSON implemented by an external llama.cpp adapter",
    )
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--generation-tokens", type=int, default=32)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--gpu-layers", type=int, default=-1)
    parser.add_argument("--device")
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args()

    runtime_root = Path(args.runtime_dir).resolve()
    runtime = probe_llama_cpp(runtime_root)
    if args.model is not None:
        if args.stage_executor is not None:
            parser.error("--stage-executor is only valid with --layer-package")
        model_path = Path(args.model).resolve()
        gguf = inspect_gguf(model_path)
        benchmark = None
        if args.benchmark:
            benchmark = benchmark_llama_cpp(
                runtime_root,
                runtime,
                gguf,
                model_path,
                prompt_tokens=args.prompt_tokens,
                generation_tokens=args.generation_tokens,
                repetitions=args.repetitions,
                gpu_layers=args.gpu_layers,
                device=args.device,
                no_warmup=args.no_warmup,
            )
        deployment = build_whole_model_deployment(
            runtime,
            gguf,
            benchmark=benchmark,
        )
    else:
        if args.benchmark:
            parser.error("a stock whole-model benchmark cannot certify a partial stage")
        if args.stage_executor is None:
            parser.error("--layer-package requires --stage-executor")
        package_path = Path(args.layer_package).resolve()
        package = parse_layer_package_manifest(_read_json(package_path))
        verify_layer_package_files(package, package_path.parent)
        executor = _read_json(Path(args.stage_executor).resolve())
        deployment = build_partial_stage_deployment(runtime, package, executor)

    print(json.dumps(deployment.to_document(), indent=2, ensure_ascii=False))


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


if __name__ == "__main__":
    main()
