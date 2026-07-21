"""Multi-turn growing-chat benchmark over the real HTTP/SSE serving path.

Conversations are fully decoupled: each concurrent conversation runs as its
own asyncio task and advances to its next turn as soon as its previous turn
finished, with no barrier between conversations. (The historical version
synchronized all conversations at every turn, which idled the server at each
turn tail and hid the interleaving the continuous scheduler actually
produces.) Turns within one conversation remain sequential because each turn
resends the grown history.

Token accounting: SSE content chunks are still reported, but chunks
undercount tokens — empty deltas (special tokens) and withheld UTF-8
suffixes produce no chunk. Real completion tokens are taken from the
``usage`` field when the server streams it, otherwise the assistant text is
retokenized with ``--tokenizer`` after the measurement window; without
either source token figures are reported as null.

Relationship to ``distributed_runtime.benchmark``: that raw-TCP harness
drives the pipeline in per-step lockstep (``receive_token_round`` waits for
all requests to return step k before any request sends step k+1), a policy
the real engine does not use. Keep it exclusively as an exact-parity
verification against the monolithic greedy reference; do not quote its
TTFT/TPOT numbers as serving latency. This module and ``api_benchmark`` are
the sources of latency and throughput metrics.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any
import uuid

import aiohttp

from .api_benchmark import (
    TokenCounter,
    build_token_counter,
    count_completion_tokens,
    parse_metadata,
    positive_csv,
    stream_request,
    summary_stats,
)


DEFAULT_TURNS = (
    "Explain in one short sentence what distributed inference is.",
    "What is its main bottleneck over a home Internet connection?",
    "Give one practical way to reduce that bottleneck.",
    "Summarize the conversation so far in one sentence.",
)


@dataclass(frozen=True)
class TurnMeasurement:
    iteration: int
    conversation: int
    turn: int
    history_messages: int
    request_characters: int
    ttft_ms: float
    response_ms: float
    content_chunks: int
    inter_chunk_ms: tuple[float, ...]
    response_characters: int
    assistant_text: str
    usage_completion_tokens: int | None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure a typical growing multi-turn chat on the distributed API"
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8081")
    parser.add_argument("--model", default="distributed-small")
    parser.add_argument("--conversations", default="1,2,4")
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--turns", type=int, default=len(DEFAULT_TURNS))
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument(
        "--tokenizer",
        help=(
            "HF tokenizer name/path used to retokenize assistant texts into real "
            "token counts when the server does not stream a usage block."
        ),
    )
    parser.add_argument(
        "--metadata",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra configuration recorded verbatim in the JSON output (repeatable).",
    )
    parser.add_argument(
        "--session-mode",
        choices=("none", "header"),
        default="none",
        help=(
            "'header' sends a stable X-Session-Id per conversation so a server "
            "with --max-retained-sessions can reuse the chat's KV between turns."
        ),
    )
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> tuple[int, ...]:
    conversations = positive_csv(args.conversations)
    if args.iterations < 1:
        raise ValueError("iterations must be positive")
    if not 1 <= args.turns <= len(DEFAULT_TURNS):
        raise ValueError(f"turns must be between 1 and {len(DEFAULT_TURNS)}")
    if args.output_tokens < 1:
        raise ValueError("output-tokens must be positive")
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        raise ValueError("timeout-seconds must be finite and positive")
    return conversations


async def stream_chat_turn(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    messages: list[dict[str, str]],
    output_tokens: int,
    *,
    iteration: int,
    conversation: int,
    turn: int,
    headers: dict[str, str] | None = None,
) -> TurnMeasurement:
    request_characters = sum(len(message["content"]) for message in messages)
    result = await stream_request(
        session, url, model, list(messages), output_tokens, headers=headers
    )
    return TurnMeasurement(
        iteration=iteration,
        conversation=conversation,
        turn=turn,
        history_messages=len(messages),
        request_characters=request_characters,
        ttft_ms=result.ttft_ms,
        response_ms=result.response_ms,
        content_chunks=len(result.chunk_times),
        inter_chunk_ms=result.inter_chunk_ms,
        response_characters=len(result.text),
        assistant_text=result.text,
        usage_completion_tokens=result.usage_completion_tokens,
    )


async def run_conversation(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    iteration: int,
    conversation: int,
) -> list[TurnMeasurement]:
    """Run one conversation's turns sequentially, no cross-conversation barrier."""

    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    # Each conversation run is a distinct chat: a fresh session id per run so
    # KV retention on the server never crosses conversations or iterations.
    headers = (
        {"X-Session-Id": f"bench-{iteration}-{conversation}-{uuid.uuid4().hex}"}
        if getattr(args, "session_mode", "none") == "header"
        else None
    )
    history: list[dict[str, str]] = [
        {
            "role": "system",
            "content": "Answer concisely. This is a latency benchmark.",
        }
    ]
    measurements: list[TurnMeasurement] = []
    for turn, prompt in enumerate(DEFAULT_TURNS[: args.turns], start=1):
        history.append(
            {
                "role": "user",
                "content": f"{prompt} [chat {iteration}-{conversation}]",
            }
        )
        value = await stream_chat_turn(
            session,
            url,
            args.model,
            history,
            args.output_tokens,
            iteration=iteration,
            conversation=conversation,
            turn=turn,
            headers=headers,
        )
        measurements.append(value)
        history.append({"role": "assistant", "content": value.assistant_text})
    return measurements


async def run_scenario(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    concurrency: int,
    iteration: int,
) -> tuple[list[TurnMeasurement], float]:
    started = time.perf_counter()
    per_conversation = await asyncio.gather(
        *(
            run_conversation(session, args, iteration, conversation)
            for conversation in range(concurrency)
        )
    )
    wall_ms = (time.perf_counter() - started) * 1_000
    return [value for values in per_conversation for value in values], wall_ms


def summarize_scenario(
    concurrency: int,
    args: argparse.Namespace,
    values: list[TurnMeasurement],
    walls_ms: list[float],
    counter: TokenCounter | None,
) -> dict[str, Any]:
    tokens_by_id: dict[int, int | None] = {}
    sources: set[str] = set()
    for value in values:
        tokens, source = count_completion_tokens(
            value.assistant_text, value.usage_completion_tokens, counter
        )
        tokens_by_id[id(value)] = tokens
        sources.add(source)

    turns: list[dict[str, Any]] = []
    for turn in range(1, args.turns + 1):
        row = [value for value in values if value.turn == turn]
        total_response_seconds = sum(value.response_ms for value in row) / 1_000
        observed_chunks = sum(value.content_chunks for value in row)
        row_tokens = [tokens_by_id[id(value)] for value in row]
        tokens_total = (
            sum(tokens for tokens in row_tokens if tokens is not None)
            if all(tokens is not None for tokens in row_tokens)
            else None
        )
        pooled_intervals = [
            interval for value in row for interval in value.inter_chunk_ms
        ]
        per_request_tpot = [
            statistics.fmean(value.inter_chunk_ms)
            for value in row
            if value.inter_chunk_ms
        ]
        turns.append(
            {
                "turn": turn,
                "requests": len(row),
                "historyMessages": row[0].history_messages,
                "requestCharactersMean": statistics.fmean(
                    value.request_characters for value in row
                ),
                "ttftMs": summary_stats(value.ttft_ms for value in row),
                "tpotMs": summary_stats(per_request_tpot),
                "interChunkMs": summary_stats(pooled_intervals),
                "responseMs": summary_stats(value.response_ms for value in row),
                "observedContentChunks": observed_chunks,
                "completionTokens": tokens_total,
                "perUserContentChunksPerSecond": (
                    observed_chunks / total_response_seconds
                    if total_response_seconds > 0
                    else 0.0
                ),
                "perUserCompletionTokensPerSecond": (
                    tokens_total / total_response_seconds
                    if tokens_total is not None and total_response_seconds > 0
                    else None
                ),
            }
        )

    total_wall_seconds = sum(walls_ms) / 1_000
    total_chunks = sum(value.content_chunks for value in values)
    all_tokens = [tokens_by_id[id(value)] for value in values]
    total_tokens = (
        sum(tokens for tokens in all_tokens if tokens is not None)
        if all(tokens is not None for tokens in all_tokens)
        else None
    )
    return {
        "concurrentConversations": concurrency,
        "completedConversationRuns": concurrency * args.iterations,
        "scenarioWallMs": walls_ms,
        "observedContentChunks": total_chunks,
        "completionTokens": total_tokens,
        "tokenSources": sorted(sources),
        "aggregateContentChunksPerSecond": (
            total_chunks / total_wall_seconds if total_wall_seconds > 0 else 0.0
        ),
        "aggregateCompletionTokensPerSecond": (
            total_tokens / total_wall_seconds
            if total_tokens is not None and total_wall_seconds > 0
            else None
        ),
        "turns": turns,
    }


async def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    concurrencies = validate_args(args)
    metadata = parse_metadata(args.metadata)
    counter = build_token_counter(getattr(args, "tokenizer", None))
    timeout = aiohttp.ClientTimeout(total=args.timeout_seconds)
    connector = aiohttp.TCPConnector(
        limit=max(32, max(concurrencies) * 2),
        limit_per_host=max(32, max(concurrencies) * 2),
    )
    scenarios: list[dict[str, Any]] = []
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        health_url = args.base_url.rstrip("/") + "/health"
        async with session.get(health_url) as response:
            response.raise_for_status()
            initial_health = await response.json()
        for concurrency in concurrencies:
            values: list[TurnMeasurement] = []
            walls_ms: list[float] = []
            for iteration in range(args.iterations):
                scenario_values, wall_ms = await run_scenario(
                    session, args, concurrency, iteration
                )
                values.extend(scenario_values)
                walls_ms.append(wall_ms)
            # Token counting is deferred to after the scenario so tokenizer CPU
            # work cannot perturb the measured latencies.
            scenarios.append(
                summarize_scenario(concurrency, args, values, walls_ms, counter)
            )
        async with session.get(health_url) as response:
            final_health = await response.json()
    return {
        "schemaVersion": 2,
        "kind": "gdlp_physical_growing_chat_benchmark",
        "provenance": (
            "physical HTTP/SSE run; conversations decoupled (no turn barrier); "
            "content chunks undercount tokens, completionTokens carries the real count "
            "when a usage block or --tokenizer is available"
        ),
        "configuration": {
            "baseUrl": args.base_url,
            "model": args.model,
            "concurrencies": list(concurrencies),
            "iterations": args.iterations,
            "turns": args.turns,
            "maxOutputTokensPerTurn": args.output_tokens,
            "historyIsResentEachTurn": True,
            "sessionMode": getattr(args, "session_mode", "none"),
            "conversationsDecoupled": True,
            "tokenizer": getattr(args, "tokenizer", None),
            "metadata": metadata,
        },
        "initialHealth": initial_health,
        "scenarios": scenarios,
        "finalHealth": final_health,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = asyncio.run(run_benchmark(args))
    rendered = json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True)
    print(rendered)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
