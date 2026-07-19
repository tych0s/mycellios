from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Iterable

import aiohttp


@dataclass(frozen=True)
class RequestMeasurement:
    ttft_ms: float
    response_ms: float
    finish_reason: str | None
    text_nonempty: bool


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the complete OpenAI-compatible distributed API path."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8081")
    parser.add_argument("--model", default="distributed-small")
    parser.add_argument("--concurrencies", default="1,2,4,8")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--output-tokens", type=int, default=8)
    parser.add_argument(
        "--prompt",
        default="Explain briefly what a GPU is. Request {request}.",
        help="Prompt template; {request} is replaced by a unique label.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args(argv)


def positive_csv(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in raw.split(","))
    except ValueError as error:
        raise ValueError("concurrencies must contain comma-separated integers") from error
    if not values or any(value < 1 for value in values):
        raise ValueError("concurrencies must contain positive integers")
    return values


def percentile(values: Iterable[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot calculate a percentile of no values")
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


async def stream_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    prompt: str,
    output_tokens: int,
) -> RequestMeasurement:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": output_tokens,
        "temperature": 0,
        "stream": True,
    }
    started = time.perf_counter()
    first_content: float | None = None
    text_parts: list[str] = []
    finish_reason: str | None = None
    async with session.post(url, json=body) as response:
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
                    delta = choice.get("delta", {}).get("content", "")
                    if delta:
                        if first_content is None:
                            first_content = time.perf_counter()
                        text_parts.append(delta)
                    if choice.get("finish_reason") is not None:
                        finish_reason = str(choice["finish_reason"])
    finished = time.perf_counter()
    if first_content is None:
        first_content = finished
    return RequestMeasurement(
        ttft_ms=(first_content - started) * 1_000,
        response_ms=(finished - started) * 1_000,
        finish_reason=finish_reason,
        text_nonempty=bool(text_parts),
    )


async def run_batch(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    concurrency: int,
    label: str,
) -> tuple[list[RequestMeasurement], float]:
    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    started = time.perf_counter()
    values = await asyncio.gather(
        *(
            stream_request(
                session,
                url,
                args.model,
                args.prompt.format(request=f"{label}-{index}"),
                args.output_tokens,
            )
            for index in range(concurrency)
        )
    )
    return values, (time.perf_counter() - started) * 1_000


async def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    concurrencies = positive_csv(args.concurrencies)
    if args.iterations < 1 or args.warmups < 0 or args.output_tokens < 1:
        raise ValueError("iterations/output-tokens must be positive and warmups non-negative")
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        raise ValueError("timeout-seconds must be finite and positive")
    maximum = max(concurrencies)
    connector = aiohttp.TCPConnector(limit=max(32, maximum * 2), limit_per_host=max(32, maximum * 2))
    timeout = aiohttp.ClientTimeout(total=args.timeout_seconds)
    rows: list[dict[str, Any]] = []
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        health_url = args.base_url.rstrip("/") + "/health"
        # Establish the same number of persistent connections used by the largest
        # scenario so connection setup does not masquerade as model TTFT.
        health_responses = await asyncio.gather(
            *(session.get(health_url) for _ in range(maximum))
        )
        for response in health_responses:
            response.release()

        for concurrency in concurrencies:
            for warmup in range(args.warmups):
                await run_batch(session, args, concurrency, f"warm-{concurrency}-{warmup}")
            requests: list[RequestMeasurement] = []
            walls: list[float] = []
            for iteration in range(args.iterations):
                values, wall = await run_batch(
                    session,
                    args,
                    concurrency,
                    f"measure-{concurrency}-{iteration}",
                )
                requests.extend(values)
                walls.append(wall)
            mean_wall = statistics.fmean(walls)
            rows.append(
                {
                    "concurrency": concurrency,
                    "measured_requests": len(requests),
                    "ttft_mean_ms": statistics.fmean(value.ttft_ms for value in requests),
                    "ttft_p95_ms": percentile((value.ttft_ms for value in requests), 0.95),
                    "response_mean_ms": statistics.fmean(
                        value.response_ms for value in requests
                    ),
                    "response_p95_ms": percentile(
                        (value.response_ms for value in requests), 0.95
                    ),
                    "batch_wall_mean_ms": mean_wall,
                    "aggregate_nominal_tok_s": (
                        concurrency * args.output_tokens / (mean_wall / 1_000)
                    ),
                    "length_finishes": sum(
                        value.finish_reason == "length" for value in requests
                    ),
                    "nonempty": sum(value.text_nonempty for value in requests),
                }
            )
        async with session.get(health_url) as response:
            health = await response.json()
    return {
        "schema_version": 1,
        "kind": "openai_api_continuous_scheduler",
        "configuration": {
            "base_url": args.base_url,
            "model": args.model,
            "concurrencies": list(concurrencies),
            "output_tokens": args.output_tokens,
            "warm_batches_per_scenario": args.warmups,
            "measured_batches_per_scenario": args.iterations,
            "persistent_http_connections_prewarmed": True,
        },
        "rows": rows,
        "server_after_measurement": health,
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
