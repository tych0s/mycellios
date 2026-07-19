from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any

import aiohttp

from .api_benchmark import percentile, positive_csv


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
    response_characters: int
    assistant_text: str
    batch_wall_ms: float = 0.0


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
) -> TurnMeasurement:
    request_characters = sum(len(message["content"]) for message in messages)
    started = time.perf_counter()
    first_content: float | None = None
    text_parts: list[str] = []
    content_chunks = 0
    async with session.post(
        url,
        json={
            "model": model,
            "messages": messages,
            "max_tokens": output_tokens,
            "temperature": 0,
            "stream": True,
        },
    ) as response:
        response.raise_for_status()
        buffer = b""
        async for block in response.content.iter_chunked(4096):
            buffer += block
            while b"\n\n" in buffer:
                event, buffer = buffer.split(b"\n\n", 1)
                for line in event.splitlines():
                    if not line.startswith(b"data: "):
                        continue
                    raw = line[6:]
                    if raw == b"[DONE]":
                        continue
                    document = json.loads(raw)
                    if "error" in document:
                        raise RuntimeError(str(document["error"]))
                    choice = document.get("choices", [{}])[0]
                    text = choice.get("delta", {}).get("content", "")
                    if text:
                        now = time.perf_counter()
                        if first_content is None:
                            first_content = now
                        text_parts.append(text)
                        content_chunks += 1
    finished = time.perf_counter()
    if first_content is None:
        first_content = finished
    assistant_text = "".join(text_parts)
    return TurnMeasurement(
        iteration=iteration,
        conversation=conversation,
        turn=turn,
        history_messages=len(messages),
        request_characters=request_characters,
        ttft_ms=(first_content - started) * 1_000,
        response_ms=(finished - started) * 1_000,
        content_chunks=content_chunks,
        response_characters=len(assistant_text),
        assistant_text=assistant_text,
    )


async def run_scenario(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    concurrency: int,
    iteration: int,
) -> list[TurnMeasurement]:
    histories: list[list[dict[str, str]]] = [
        [
            {
                "role": "system",
                "content": "Answer concisely. This is a latency benchmark.",
            }
        ]
        for _ in range(concurrency)
    ]
    measurements: list[TurnMeasurement] = []
    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    for turn, prompt in enumerate(DEFAULT_TURNS[: args.turns], start=1):
        for conversation, history in enumerate(histories):
            history.append(
                {
                    "role": "user",
                    "content": f"{prompt} [chat {iteration}-{conversation}]",
                }
            )
        batch_started = time.perf_counter()
        values = await asyncio.gather(
            *(
                stream_chat_turn(
                    session,
                    url,
                    args.model,
                    history,
                    args.output_tokens,
                    iteration=iteration,
                    conversation=conversation,
                    turn=turn,
                )
                for conversation, history in enumerate(histories)
            )
        )
        batch_wall_ms = (time.perf_counter() - batch_started) * 1_000
        values = [replace(value, batch_wall_ms=batch_wall_ms) for value in values]
        measurements.extend(values)
        for value, history in zip(values, histories):
            history.append({"role": "assistant", "content": value.assistant_text})
    return measurements


async def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    concurrencies = validate_args(args)
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
            for iteration in range(args.iterations):
                values.extend(await run_scenario(session, args, concurrency, iteration))
            turns: list[dict[str, Any]] = []
            for turn in range(1, args.turns + 1):
                row = [value for value in values if value.turn == turn]
                total_response_seconds = sum(value.response_ms for value in row) / 1_000
                total_batch_wall_seconds = (
                    sum(
                        next(
                            value.batch_wall_ms
                            for value in row
                            if value.iteration == iteration
                        )
                        for iteration in sorted({value.iteration for value in row})
                    )
                    / 1_000
                )
                observed_chunks = sum(value.content_chunks for value in row)
                turns.append(
                    {
                        "turn": turn,
                        "requests": len(row),
                        "historyMessages": row[0].history_messages,
                        "requestCharactersMean": statistics.fmean(
                            value.request_characters for value in row
                        ),
                        "ttftMeanMs": statistics.fmean(value.ttft_ms for value in row),
                        "ttftP95Ms": percentile((value.ttft_ms for value in row), 0.95),
                        "responseMeanMs": statistics.fmean(
                            value.response_ms for value in row
                        ),
                        "responseP95Ms": percentile(
                            (value.response_ms for value in row), 0.95
                        ),
                        "observedContentChunks": observed_chunks,
                        "perUserContentChunksPerSecond": (
                            observed_chunks / total_response_seconds
                            if total_response_seconds > 0
                            else 0.0
                        ),
                        "aggregateContentChunksPerSecond": (
                            observed_chunks / total_batch_wall_seconds
                            if total_batch_wall_seconds > 0
                            else 0.0
                        ),
                        "batchWallMeanMs": statistics.fmean(
                            next(
                                value.batch_wall_ms
                                for value in row
                                if value.iteration == iteration
                            )
                            for iteration in sorted({value.iteration for value in row})
                        ),
                    }
                )
            scenarios.append(
                {
                    "concurrentConversations": concurrency,
                    "completedConversationRuns": concurrency * args.iterations,
                    "turns": turns,
                }
            )
        async with session.get(health_url) as response:
            final_health = await response.json()
    return {
        "schemaVersion": 1,
        "kind": "gdlp_physical_growing_chat_benchmark",
        "provenance": "physical HTTP/SSE run; content chunks normally correspond to emitted tokenizer pieces",
        "configuration": {
            "baseUrl": args.base_url,
            "model": args.model,
            "concurrencies": list(concurrencies),
            "iterations": args.iterations,
            "turns": args.turns,
            "maxOutputTokensPerTurn": args.output_tokens,
            "historyIsResentEachTurn": True,
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
