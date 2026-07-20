from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import queue as queue_module
import socket
import tempfile
import time
import traceback
from typing import Any, Sequence

import torch
import torch.distributed as distributed

from .expert_parallel import (
    EXPERT_PARALLEL_GATE_SCHEMA,
    HfNativeExpertParallelCellExecutor,
    inspect_expert_parallel_checkpoint,
    runtime_versions,
)


def create_tiny_qwen3_moe_checkpoint(directory: str | Path) -> Path:
    """Create the deterministic physical certification checkpoint."""

    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    target = Path(directory).resolve()
    target.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(7)
    config = Qwen3MoeConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=64,
        moe_intermediate_size=8,
        num_experts_per_tok=2,
        num_experts=4,
        use_cache=True,
        tie_word_embeddings=False,
        bos_token_id=1,
        eos_token_id=2,
    )
    Qwen3MoeForCausalLM(config).eval().save_pretrained(
        target,
        safe_serialization=True,
    )
    return target


def run_physical_expert_parallel_gate(
    checkpoint: str | Path | None = None,
    *,
    world_size: int = 2,
    timeout_seconds: float = 120.0,
    model_source: str | None = None,
) -> dict[str, Any]:
    if world_size != 2:
        raise ValueError("the v1 physical certification fixture requires world_size=2")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    with tempfile.TemporaryDirectory(prefix="gdlp-ep-gate-") as temporary:
        temporary_path = Path(temporary)
        checkpoint_path = (
            create_tiny_qwen3_moe_checkpoint(temporary_path / "checkpoint")
            if checkpoint is None
            else Path(checkpoint).expanduser().resolve()
        )
        stable_source = (
            model_source
            or (
                "gdlp/tiny-qwen3-moe-ep-gate"
                if checkpoint is None
                else None
            )
        )
        inspection = inspect_expert_parallel_checkpoint(
            checkpoint_path,
            world_size=world_size,
            backend="gloo",
            device_kind="cpu",
            model_source=stable_source,
        )
        (
            dense_model,
            expected_prefill_logits,
            expected_decode_logits,
            dense_parameter_bytes,
            dense_expert_bytes,
        ) = _dense_reference(checkpoint_path)
        del dense_model

        context = multiprocessing.get_context("spawn")
        results = context.Queue()
        port = _reserve_loopback_port()
        processes = [
            context.Process(
                target=_physical_rank_worker,
                args=(
                    rank,
                    world_size,
                    port,
                    str(checkpoint_path),
                    inspection.model_source,
                    expected_prefill_logits,
                    expected_decode_logits,
                    results,
                ),
                name=f"gdlp-ep-rank-{rank}",
            )
            for rank in range(world_size)
        ]
        gate_started = time.perf_counter()
        for process in processes:
            process.start()

        rank_results: list[dict[str, Any]] = []
        deadline = time.monotonic() + timeout_seconds
        try:
            while len(rank_results) < world_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("physical EP ranks did not finish before the gate timeout")
                try:
                    rank_results.append(results.get(timeout=min(remaining, 2.0)))
                except queue_module.Empty:
                    if any(
                        process.exitcode not in (None, 0)
                        for process in processes
                    ):
                        break
        finally:
            for process in processes:
                process.join(timeout=5)
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)

        exit_codes = [process.exitcode for process in processes]
        if len(rank_results) != world_size:
            raise RuntimeError(
                f"physical EP gate received {len(rank_results)}/{world_size} rank results; "
                f"exit_codes={exit_codes}"
            )
        ordered = sorted(rank_results, key=lambda item: int(item["rank"]))
        failures = [item for item in ordered if not item.get("ok")]
        if failures:
            raise RuntimeError(
                "physical EP rank failed: "
                + json.dumps(failures, ensure_ascii=False, sort_keys=True)
            )
        if exit_codes != [0] * world_size:
            raise RuntimeError(f"physical EP ranks exited abnormally: {exit_codes}")

        max_abs = max(float(item["maxAbsLogitError"]) for item in ordered)
        logits_digests = {str(item["logitsSha256"]) for item in ordered}
        next_tokens = {tuple(item["nextTokenIds"]) for item in ordered}
        contract_ids = {str(item["cellContractId"]) for item in ordered}
        expert_ranges = [tuple(item["localExpertRange"]) for item in ordered]
        passed = (
            max_abs <= 1e-6
            and all(bool(item["logitsEqual"]) for item in ordered)
            and all(bool(item["tokensEqual"]) for item in ordered)
            and len(logits_digests) == 1
            and len(next_tokens) == 1
            and len(contract_ids) == 1
            and expert_ranges
            == [
                (
                    rank * inspection.local_experts,
                    (rank + 1) * inspection.local_experts,
                )
                for rank in range(world_size)
            ]
        )
        if not passed:
            raise RuntimeError(
                "physical EP gate failed parity or rank-layout certification: "
                + json.dumps(ordered, ensure_ascii=False, sort_keys=True)
            )
        elapsed_ms = (time.perf_counter() - gate_started) * 1_000
        return {
            "schema": EXPERT_PARALLEL_GATE_SCHEMA,
            "status": "pass",
            "evidence": "loopback-physical-multiprocess",
            "runtime": runtime_versions(),
            "model": {
                "identity": inspection.model_identity,
                "source": inspection.model_source,
                "modelType": inspection.model_type,
                "layers": inspection.total_layers,
                "hiddenSize": inspection.hidden_size,
            },
            "experts": {
                "globalCount": inspection.num_experts,
                "localCountPerRank": inspection.local_experts,
                "topK": inspection.top_k,
                "ranges": [list(value) for value in expert_ranges],
                "plan": dict(inspection.ep_plan),
            },
            "distributed": {
                "processes": world_size,
                "backend": "gloo",
                "deviceKind": "cpu",
                "scope": "low-latency-cell-only",
                "wanAllowed": False,
                "elapsedMs": elapsed_ms,
            },
            "parity": {
                "maxAbsLogitError": max_abs,
                "prefillMaxAbsLogitError": max(
                    float(item["prefillMaxAbsLogitError"]) for item in ordered
                ),
                "decodeMaxAbsLogitError": max(
                    float(item["decodeMaxAbsLogitError"]) for item in ordered
                ),
                "logitsExactlyEqual": all(bool(item["logitsEqual"]) for item in ordered),
                "tokensEqual": all(bool(item["tokensEqual"]) for item in ordered),
                "ranksExactlyEqual": len(logits_digests) == 1,
                "nextTokenIds": list(next(iter(next_tokens))),
            },
            "memory": {
                "denseParameterBytes": dense_parameter_bytes,
                "denseExpertParameterBytes": dense_expert_bytes,
                "rankLocalParameterBytes": [
                    int(item["rankLocalParameterBytes"]) for item in ordered
                ],
                "rankLocalExpertParameterBytes": [
                    int(item["rankLocalExpertParameterBytes"]) for item in ordered
                ],
            },
            "ranks": ordered,
            "limitations": {
                "fullModelCellOnly": True,
                "partialStage": False,
                "stageRunner": False,
                "nonExpertWeightsReplicated": True,
                "expertParallelAcrossWan": False,
            },
        }


def _dense_reference(
    checkpoint: Path,
) -> tuple[Any, torch.Tensor, torch.Tensor, int, int]:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.float32,
    ).eval()
    input_ids = _gate_input_ids()
    with torch.no_grad():
        prefill = model(input_ids=input_ids, use_cache=True)
        prefill_logits = prefill.logits.cpu()
        first_token = prefill_logits[:, -1, :].argmax(dim=-1, keepdim=True)
        decode = model(
            input_ids=first_token,
            past_key_values=prefill.past_key_values,
            use_cache=True,
        )
        decode_logits = decode.logits.cpu()
    total = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    expert = sum(
        parameter.numel() * parameter.element_size()
        for name, parameter in model.named_parameters()
        if ".mlp.experts." in name
    )
    return model, prefill_logits, decode_logits, total, expert


def _physical_rank_worker(
    rank: int,
    world_size: int,
    port: int,
    checkpoint: str,
    model_source: str,
    expected_prefill_logits: torch.Tensor,
    expected_decode_logits: torch.Tensor,
    results: Any,
) -> None:
    os.environ.update(
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world_size),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        TOKENIZERS_PARALLELISM="false",
    )
    try:
        distributed.init_process_group(
            "gloo",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=world_size,
        )
        executor = HfNativeExpertParallelCellExecutor.load(
            checkpoint,
            model_source=model_source,
            backend="gloo",
            device_kind="cpu",
            dtype=torch.float32,
            local_files_only=True,
        )
        input_ids = _gate_input_ids()
        started = time.perf_counter()
        with torch.no_grad():
            prefill = executor.forward(input_ids, use_cache=True)
            prefill_logits = prefill.logits.detach().cpu()
            first_token = prefill_logits[:, -1, :].argmax(dim=-1, keepdim=True)
            decode = executor.forward(
                first_token,
                past_key_values=prefill.past_key_values,
                use_cache=True,
            )
            decode_logits = decode.logits.detach().cpu()
        forward_ms = (time.perf_counter() - started) * 1_000
        second_token = decode_logits[:, -1, :].argmax(dim=-1, keepdim=True)
        next_tokens = torch.cat((first_token.cpu(), second_token), dim=1)
        expected_first = expected_prefill_logits[:, -1, :].argmax(
            dim=-1, keepdim=True
        )
        expected_second = expected_decode_logits[:, -1, :].argmax(
            dim=-1, keepdim=True
        )
        expected_tokens = torch.cat((expected_first, expected_second), dim=1)
        prefill_error = float(
            (prefill_logits - expected_prefill_logits).abs().max()
        )
        decode_error = float((decode_logits - expected_decode_logits).abs().max())
        digest = hashlib.sha256()
        digest.update(prefill_logits.contiguous().numpy().tobytes())
        digest.update(decode_logits.contiguous().numpy().tobytes())
        manifest = executor.manifest.to_document()
        local_range = manifest["rank"]["localExpertRange"]
        results.put(
            {
                "rank": rank,
                "ok": True,
                "loadMs": executor.load_ms,
                "forwardMs": forward_ms,
                "maxAbsLogitError": max(prefill_error, decode_error),
                "prefillMaxAbsLogitError": prefill_error,
                "decodeMaxAbsLogitError": decode_error,
                "logitsEqual": bool(
                    torch.equal(prefill_logits, expected_prefill_logits)
                    and torch.equal(decode_logits, expected_decode_logits)
                ),
                "tokensEqual": bool(torch.equal(next_tokens, expected_tokens)),
                "nextTokenIds": next_tokens.reshape(-1).tolist(),
                "logitsSha256": digest.hexdigest(),
                "expertTensorShape": list(
                    executor.model.model.layers[0].mlp.experts.gate_up_proj.shape
                ),
                "localExpertRange": [local_range["start"], local_range["end"]],
                "rankLocalParameterBytes": manifest["memory"]["rankLocalParameterBytes"],
                "rankLocalExpertParameterBytes": manifest["memory"][
                    "rankLocalExpertParameterBytes"
                ],
                "cellContractId": manifest["cellContractId"],
                "executorId": manifest["executorId"],
            }
        )
    except BaseException as error:
        results.put(
            {
                "rank": rank,
                "ok": False,
                "errorType": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(limit=12),
            }
        )
    finally:
        if distributed.is_initialized():
            distributed.destroy_process_group()


def _gate_input_ids() -> torch.Tensor:
    return torch.tensor([[1, 5, 9, 3]], dtype=torch.long)


def _reserve_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _write_json(path: str | Path, document: Any) -> None:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect and physically certify HF native expert-parallel cells."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect = subparsers.add_parser("inspect", help="Inspect a local Qwen3Moe checkpoint.")
    inspect.add_argument("--model", required=True)
    inspect.add_argument("--model-source")
    inspect.add_argument("--world-size", type=int, default=2)
    inspect.add_argument("--backend", default="gloo")
    inspect.add_argument("--device-kind", default="cpu")
    inspect.add_argument("--json-out")

    gate = subparsers.add_parser(
        "physical-gate",
        help="Run dense-vs-EP parity with two real CPU/Gloo processes.",
    )
    gate.add_argument("--model")
    gate.add_argument("--model-source")
    gate.add_argument("--world-size", type=int, default=2)
    gate.add_argument("--timeout-seconds", type=float, default=120.0)
    gate.add_argument("--json-out")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "inspect":
        document = inspect_expert_parallel_checkpoint(
            args.model,
            world_size=args.world_size,
            backend=args.backend,
            device_kind=args.device_kind,
            model_source=args.model_source,
        ).to_document()
    else:
        document = run_physical_expert_parallel_gate(
            args.model,
            world_size=args.world_size,
            timeout_seconds=args.timeout_seconds,
            model_source=args.model_source,
        )
    if args.json_out:
        _write_json(args.json_out, document)
    print(json.dumps(document, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "create_tiny_qwen3_moe_checkpoint",
    "main",
    "run_physical_expert_parallel_gate",
]
