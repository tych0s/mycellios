from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM

from .model import load_tokenizer, model_artifact_reference, resolve_model_snapshot
from .server import OUTPUT_TOKEN_HASH_SCHEME, output_token_ids_sha256


SCHEMA = "gdlp-greedy-canary-reference/1"
PROMPT_TOKEN_HASH_SCHEME = "gdlp-prompt-token-ids-v1"
PROMPT_TOKEN_DIGEST_DOMAIN = PROMPT_TOKEN_HASH_SCHEME.encode("ascii") + b"\0"


def prompt_token_ids_sha256(token_ids: list[int] | tuple[int, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(PROMPT_TOKEN_DIGEST_DOMAIN)
    digest.update(struct.pack(">Q", len(token_ids)))
    for token_id in token_ids:
        if type(token_id) is not int or not 0 <= token_id <= 0xFFFFFFFF:
            raise ValueError("prompt token id must be a uint32")
        digest.update(struct.pack(">I", token_id))
    return "sha256:" + digest.hexdigest()


def normalize_messages(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value or len(value) > 1_024:
        raise ValueError("messages must be a non-empty bounded array")
    normalized: list[dict[str, str]] = []
    for index, message in enumerate(value):
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise ValueError(f"messages[{index}] must contain only role and content")
        role = message["role"]
        content = message["content"]
        if role not in ("system", "developer", "user", "assistant"):
            raise ValueError(f"messages[{index}] has an unsupported role")
        if not isinstance(content, str):
            raise ValueError(f"messages[{index}] content must be a string")
        normalized.append(
            {"role": "system" if role == "developer" else role, "content": content}
        )
    return normalized


def parse_dtype(value: str) -> torch.dtype:
    try:
        return {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[value]
    except KeyError as error:
        raise ValueError("dtype must be float32, float16 or bfloat16") from error


@torch.inference_mode()
def greedy_reference(
    model: Any,
    input_ids: torch.Tensor,
    *,
    max_tokens: int,
    eos_token_ids: frozenset[int],
) -> tuple[list[int], str, dict[str, float]]:
    if not 1 <= max_tokens <= 1_000_000:
        raise ValueError("max_tokens must be positive and bounded")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
        raise ValueError("input_ids must have shape [1,tokens]")
    started = time.perf_counter()
    output = model(input_ids=input_ids, use_cache=True)
    cache = output.past_key_values
    tokens: list[int] = []
    arrivals: list[float] = []
    finish_reason = "length"
    for index in range(max_tokens):
        token_tensor = torch.argmax(output.logits[:, -1, :], dim=-1)
        token = int(token_tensor.item())
        tokens.append(token)
        arrivals.append(time.perf_counter())
        if token in eos_token_ids:
            finish_reason = "stop"
            break
        if index + 1 < max_tokens:
            output = model(
                input_ids=token_tensor[:, None],
                past_key_values=cache,
                use_cache=True,
            )
            cache = output.past_key_values
    intervals = [
        (right - left) * 1_000
        for left, right in zip(arrivals, arrivals[1:])
    ]
    return (
        tokens,
        finish_reason,
        {
            "ttftMs": (arrivals[0] - started) * 1_000,
            "tpotMs": sum(intervals) / len(intervals) if intervals else 0.0,
            "pipelineMs": (arrivals[-1] - started) * 1_000,
        },
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an exact greedy chat canary reference for a sealed campaign."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--messages-json", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument("--tokenizer-id")
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args(argv)


def build_reference(args: argparse.Namespace) -> dict[str, Any]:
    if not 1 <= args.max_tokens <= 1_000_000:
        raise ValueError("max-tokens must be positive and bounded")
    if not 1 <= args.threads <= 1_024:
        raise ValueError("threads must be positive and bounded")
    messages = normalize_messages(
        json.loads(args.messages_json.read_text(encoding="utf-8"))
    )
    snapshot = resolve_model_snapshot(args.model, args.revision)
    artifact = model_artifact_reference(snapshot)
    tokenizer = load_tokenizer(snapshot)
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    input_ids = encoded if isinstance(encoded, torch.Tensor) else encoded["input_ids"]
    input_ids = input_ids.to(dtype=torch.long, device=args.device)
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    eos = tokenizer.eos_token_id
    if eos is None:
        eos_ids = frozenset()
    elif isinstance(eos, int):
        eos_ids = frozenset((eos,))
    else:
        eos_ids = frozenset(int(value) for value in eos)
    torch.set_num_threads(args.threads)
    dtype = parse_dtype(args.dtype)
    model = AutoModelForCausalLM.from_pretrained(snapshot, dtype=dtype).to(args.device).eval()
    tokens, finish_reason, metrics = greedy_reference(
        model,
        input_ids,
        max_tokens=args.max_tokens,
        eos_token_ids=eos_ids,
    )
    prompt_ids = [int(value) for value in input_ids[0].detach().to(device="cpu").tolist()]
    return {
        "schema": SCHEMA,
        "model": {
            "requestedId": args.model,
            "requestedRevision": args.revision,
            "artifactIdentity": artifact.identity,
            "canonicalSource": artifact.canonical_source,
            "canonicalRevision": artifact.canonical_revision,
            "snapshotIdentity": str(artifact.snapshot_identity),
            "tokenizerId": args.tokenizer_id or args.model,
            "device": args.device,
            "dtype": args.dtype,
        },
        "messages": messages,
        "maxTokens": args.max_tokens,
        "promptTokens": len(prompt_ids),
        "promptTokenIdsHashScheme": PROMPT_TOKEN_HASH_SCHEME,
        "promptTokenIdsSha256": prompt_token_ids_sha256(prompt_ids),
        "completionTokens": len(tokens),
        "finishReason": finish_reason,
        "outputTokenIds": tokens,
        "outputTokenIdsHashScheme": OUTPUT_TOKEN_HASH_SCHEME,
        "outputTokenIdsSha256": output_token_ids_sha256(tokens),
        "metrics": metrics,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    reference = build_reference(args)
    rendered = json.dumps(reference, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
