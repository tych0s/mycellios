"""Small, offline canary for the installed Mycellios stage runtime.

This is deliberately not a model-quality test.  It builds a deterministic,
single-layer Llama checkpoint in a temporary directory and then loads it
through the same selective SafeTensors path used by a real Mycellios stage.
The canary proves that the packaged Python, Torch, Transformers and Mycellios
sources can execute the StageRunner ABI, physical batching and request-local
KV lifecycle together without contacting a model registry.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Any, Iterator

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from .batching import FairAdaptiveBatchQueue, FairBatchConfig, InferenceBatchKey
from .executor_abi import parse_stage_executor_manifest
from .model import StageModelSpec, StageRunner
from .model_adapters import ADAPTER_REGISTRY_ID, resolve_selective_stage_adapter


SCHEMA = "mycellios-installed-stage-canary/1"
OUTPUT_MARKER = "MYCELLIOS_INSTALLED_STAGE_CANARY="


def run_installed_stage_canary() -> dict[str, Any]:
    """Execute the deterministic installed-runtime proof and return evidence."""

    # A verifier must never succeed by filling a missing local artifact from a
    # registry.  The checkpoint path below is local, but these flags also make
    # accidental future Auto* changes fail closed.
    with _offline_environment(), TemporaryDirectory(
        prefix="mycellios-installed-stage-canary-"
    ) as temporary:
        checkpoint = Path(temporary) / "tiny-stage"
        _write_deterministic_checkpoint(checkpoint)

        reference = _load_runner(checkpoint)
        batched = _load_runner(checkpoint)
        try:
            manifest = parse_stage_executor_manifest(
                batched.executor_manifest.to_document()
            )
            if manifest.engine != "python-torch":
                raise RuntimeError("installed canary did not load the Python Torch executor")
            if manifest.adapter != "transformers-llama-v1":
                raise RuntimeError("installed canary did not load the certified Llama adapter")
            adapter_contract = resolve_selective_stage_adapter(batched.base.config)
            if adapter_contract.adapter_id != manifest.adapter:
                raise RuntimeError("installed canary adapter registry drifted")
            if manifest.max_batch_size < 2:
                raise RuntimeError("installed canary executor cannot form a physical batch")

            hidden_rows = (
                _hidden_row(0.125),
                _hidden_row(-0.375),
            )
            reference_rows, reference_tokens = _run_sequential(
                reference,
                (101, 102),
                hidden_rows,
            )

            for request_id in (201, 202):
                batched.begin(request_id)
            key = InferenceBatchKey(
                model_id=manifest.model_identity,
                stage_id=f"{manifest.layer_start}:{manifest.layer_end}",
                input_tokens=2,
                cache_tokens=0,
                hidden_size=manifest.hidden_size,
                dtype=manifest.activation_dtype,
                codec="fp32",
                backend=manifest.engine,
            )
            queue = FairAdaptiveBatchQueue[torch.Tensor](
                FairBatchConfig(
                    max_batch_size=2,
                    initial_batch_size=2,
                    target_batch_latency_ms=1_000,
                )
            )
            for request_id, hidden in zip((201, 202), hidden_rows):
                queue.enqueue(
                    hidden,
                    request_id=request_id,
                    phase="prefill",
                    compatibility_key=key,
                    token_count=2,
                    now=1.0,
                )
            selection = queue.pop_batch(now=1.0, force=True)
            if selection is None or tuple(
                item.request_id for item in selection.items
            ) != (201, 202):
                raise RuntimeError("installed canary batching queue did not coalesce work")
            batch_results = batched.forward_hidden_batch(
                tuple(item.request_id for item in selection.items),
                selection.payloads,
                token_mode="last",
            )
            queue.record_batch(selection, latency_ms=1.0)
            batch_rows = tuple(result[0] for result in batch_results)
            batch_tokens = tuple(_token(result[1]) for result in batch_results)
            sequential_batch_parity = all(
                torch.allclose(expected, actual, rtol=1e-5, atol=1e-6)
                for expected, actual in zip(reference_rows, batch_rows)
            ) and reference_tokens == batch_tokens
            if not sequential_batch_parity:
                raise RuntimeError("installed canary sequential/batch parity failed")
            if any(batched.sequence_length(request_id) != 2 for request_id in (201, 202)):
                raise RuntimeError("installed canary batch did not commit two KV positions")

            parent_bytes = batched.request_cache_bytes(201)
            if parent_bytes < 1:
                raise RuntimeError("installed canary did not materialize stage-local KV")
            copied_bytes = batched.fork_request(
                203,
                201,
                max_cache_bytes=parent_bytes,
            )
            report = batched.last_fork_report()
            if (
                copied_bytes != parent_bytes
                or report is None
                or report.copied_bytes != copied_bytes
                or report.unique_physical_bytes < parent_bytes * 2
            ):
                raise RuntimeError("installed canary KV fork accounting is inconsistent")

            suffix = _hidden_row(0.625, tokens=1)
            fork_results = batched.forward_hidden_batch(
                (201, 203),
                (suffix, suffix.clone()),
                token_mode="last",
            )
            fork_parity = (
                torch.allclose(
                    fork_results[0][0],
                    fork_results[1][0],
                    rtol=1e-5,
                    atol=1e-6,
                )
                and _token(fork_results[0][1]) == _token(fork_results[1][1])
            )
            if not fork_parity:
                raise RuntimeError("installed canary forked KV parity failed")

            expected_after_suffix = fork_results[1][0].detach().cpu().clone()
            expected_token = _token(fork_results[1][1])
            batched.truncate(203, 2)
            if batched.sequence_length(203) != 2:
                raise RuntimeError("installed canary KV rollback did not restore the prefix")
            replay_row, replay_token = batched.forward_hidden(
                203,
                suffix.clone(),
                token_mode="last",
            )
            rollback_parity = torch.allclose(
                expected_after_suffix,
                replay_row.detach().cpu(),
                rtol=1e-5,
                atol=1e-6,
            ) and expected_token == _token(replay_token)
            if not rollback_parity:
                raise RuntimeError("installed canary KV rollback parity failed")

            stats = queue.stats()
            if stats.batches != 1 or stats.dequeued_items != 2:
                raise RuntimeError("installed canary batching statistics are inconsistent")
            tokens = (*batch_tokens, _token(fork_results[0][1]), expected_token)
            evidence = {
                "schema": SCHEMA,
                "ok": True,
                "pythonVersion": ".".join(map(str, sys.version_info[:3])),
                "pythonPrefix": sys.prefix,
                "torchVersion": torch.__version__,
                "transformersVersion": importlib.metadata.version("transformers"),
                "engine": manifest.engine,
                "adapter": manifest.adapter,
                "adapterContractId": adapter_contract.adapter_contract_id,
                "adapterRegistryId": ADAPTER_REGISTRY_ID,
                "loader": batched.loader,
                "batchSize": len(selection.items),
                "physicalBatchCalls": batched.physical_batch_calls,
                "physicalBatchItems": batched.physical_batch_items,
                "sequenceTokens": batched.sequence_length(203),
                "kvBytes": parent_bytes,
                "copiedKvBytes": copied_bytes,
                "batchQueueBatches": stats.batches,
                "outputTokenSha256": hashlib.sha256(
                    json.dumps(tokens, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
                "parity": {
                    "sequentialVsBatch": sequential_batch_parity,
                    "fork": fork_parity,
                    "rollback": rollback_parity,
                },
            }
            for request_id in (201, 202, 203):
                batched.end(request_id)
            if batched.active_requests or batched.caches or batched.tokens_seen:
                raise RuntimeError("installed canary END did not release request KV state")
            return evidence
        finally:
            reference.close()
            batched.close()


def _write_deterministic_checkpoint(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        rms_norm_eps=1e-6,
        attention_bias=False,
        mlp_bias=False,
        tie_word_embeddings=False,
        use_cache=True,
    )
    config.architectures = ["LlamaForCausalLM"]
    torch.manual_seed(7)
    model = LlamaForCausalLM(config).eval()
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            values = torch.arange(
                parameter.numel(),
                dtype=torch.float32,
            ).reshape(parameter.shape)
            values = ((values + index * 17) % 101 - 50) / 2_500
            parameter.copy_(values.to(dtype=parameter.dtype))
    model.save_pretrained(destination, safe_serialization=True)
    del model


def _load_runner(checkpoint: Path) -> StageRunner:
    return StageRunner(
        StageModelSpec(
            model_name=str(checkpoint),
            layer_start=0,
            layer_end=1,
            total_layers=1,
            threads=1,
            kv_cache="arena",
            decode_attention="stock",
        ),
        device="cpu",
    )


def _run_sequential(
    runner: StageRunner,
    request_ids: tuple[int, int],
    hidden_rows: tuple[torch.Tensor, torch.Tensor],
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[int, int]]:
    rows: list[torch.Tensor] = []
    tokens: list[int] = []
    try:
        for request_id, hidden in zip(request_ids, hidden_rows):
            runner.begin(request_id)
            row, token = runner.forward_hidden(
                request_id,
                hidden.clone(),
                token_mode="last",
            )
            rows.append(row.detach().cpu())
            tokens.append(_token(token))
    finally:
        for request_id in request_ids:
            runner.end(request_id)
    return (rows[0], rows[1]), (tokens[0], tokens[1])


def _hidden_row(offset: float, *, tokens: int = 2) -> torch.Tensor:
    values = torch.arange(tokens * 16, dtype=torch.float32).reshape(1, tokens, 16)
    return (values / 64 + offset).contiguous()


def _token(value: int | tuple[int, ...] | None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise RuntimeError("installed canary stage did not return one greedy token")
    return value


@contextmanager
def _offline_environment() -> Iterator[None]:
    names = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    previous = {name: os.environ.get(name) for name in names}
    os.environ.update(names)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def main() -> int:
    evidence = run_installed_stage_canary()
    print(
        OUTPUT_MARKER
        + json.dumps(
            evidence,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
