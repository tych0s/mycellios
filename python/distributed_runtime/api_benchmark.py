"""Benchmarks for the complete OpenAI-compatible HTTP/SSE serving path.

Modes
-----
``closed`` (default)
    The historical batch mode: C simultaneous requests per batch with a
    barrier between batches (``asyncio.gather``). Useful for repeatable A/B
    comparisons at a fixed concurrency, but the barrier makes it a
    closed-loop harness: the server idles at every batch tail and queueing
    delay that an open user population would experience is hidden.

``open``
    Open-loop mode: request arrivals follow a Poisson process with a
    configurable rate ``--lambda-rps`` sustained for ``--duration-seconds``.
    Every SSE content chunk is timestamped per request and the report derives
    TTFT p50/p95, TPOT p50/p95 (intervals between content-bearing chunks) and
    the aggregate steady-state token rate after discarding the configurable
    ``--warmup-seconds`` prefix. Real token counts come from the ``usage``
    field whenever the server includes it on the stream; otherwise the final
    text is retokenized with ``--tokenizer``. Without either source only SSE
    chunk counts are reported, and chunks undercount tokens: empty deltas
    (special tokens) and withheld UTF-8 suffixes produce no chunk.

Relationship to ``distributed_runtime.benchmark``
-------------------------------------------------
The raw-TCP harness in ``benchmark.py`` drives the pipeline in per-step
lockstep: its ``receive_token_round`` waits for *all* concurrent requests to
return step k before any request advances to step k+1, a scheduling policy
the real engine does not use. That harness remains useful exclusively as an
exact-parity verification against the monolithic greedy reference; do not
quote its TTFT/TPOT/throughput numbers as serving latency. This module,
measured over HTTP/SSE against the real server and continuous scheduler, is
the source of truth for latency and throughput metrics.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import statistics
import time
from typing import Any, Callable, Iterable

import aiohttp


TokenCounter = Callable[[str], int]


@dataclass(frozen=True)
class StreamResult:
    """Raw timing record of one streamed chat completion."""

    started: float
    finished: float
    chunk_times: tuple[float, ...]
    text: str
    finish_reason: str | None
    usage_completion_tokens: int | None

    @property
    def ttft_ms(self) -> float:
        first = self.chunk_times[0] if self.chunk_times else self.finished
        return (first - self.started) * 1_000

    @property
    def response_ms(self) -> float:
        return (self.finished - self.started) * 1_000

    @property
    def inter_chunk_ms(self) -> tuple[float, ...]:
        return tuple(
            (right - left) * 1_000
            for left, right in zip(self.chunk_times, self.chunk_times[1:])
        )


@dataclass(frozen=True)
class RequestMeasurement:
    ttft_ms: float
    response_ms: float
    inter_chunk_ms: tuple[float, ...]
    content_chunks: int
    completion_tokens: int | None
    token_source: str
    finish_reason: str | None
    text_nonempty: bool


@dataclass(frozen=True)
class OpenLoopSample:
    """One request of an open-loop scenario; offsets are relative to scenario start."""

    index: int
    scheduled_offset_s: float
    started_offset_s: float
    finished_offset_s: float
    ttft_ms: float | None
    response_ms: float
    chunk_offsets_s: tuple[float, ...]
    inter_chunk_ms: tuple[float, ...]
    content_chunks: int
    text: str
    usage_completion_tokens: int | None
    finish_reason: str | None
    error: str | None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the complete OpenAI-compatible distributed API path."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8081")
    parser.add_argument("--model", default="distributed-small")
    parser.add_argument(
        "--mode",
        choices=("closed", "open"),
        default="closed",
        help=(
            "closed: barrier batches at fixed concurrencies (legacy). "
            "open: Poisson arrivals at --lambda-rps for --duration-seconds."
        ),
    )
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
    parser.add_argument(
        "--lambda-rps",
        default="1.0",
        help="Open mode: comma-separated Poisson arrival rates (requests/second); one scenario per rate.",
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=30.0,
        help="Open mode: arrival window length per scenario.",
    )
    parser.add_argument(
        "--warmup-seconds",
        type=float,
        default=5.0,
        help="Open mode: steady-state statistics discard this initial prefix.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Open mode: arrival process seed.")
    parser.add_argument(
        "--prewarm-connections",
        type=int,
        default=16,
        help="Persistent HTTP connections established before measuring.",
    )
    parser.add_argument(
        "--tokenizer",
        help=(
            "HF tokenizer name/path used to retokenize final texts into real token "
            "counts when the server does not stream a usage block."
        ),
    )
    parser.add_argument(
        "--metadata",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra configuration recorded verbatim in the JSON output (repeatable).",
    )
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


def positive_float_csv(raw: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in raw.split(","))
    except ValueError as error:
        raise ValueError("rates must contain comma-separated numbers") from error
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("rates must contain positive finite numbers")
    return values


def parse_metadata(items: Iterable[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for item in items:
        key, separator, value = item.partition("=")
        if not separator or not key.strip():
            raise ValueError(f"metadata entries must look like key=value, got {item!r}")
        metadata[key.strip()] = value.strip()
    return metadata


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


def summary_stats(values: Iterable[float]) -> dict[str, float | int] | None:
    materialized = [float(value) for value in values]
    if not materialized:
        return None
    return {
        "count": len(materialized),
        "mean": statistics.fmean(materialized),
        "p50": percentile(materialized, 0.50),
        "p95": percentile(materialized, 0.95),
        "min": min(materialized),
        "max": max(materialized),
    }


def build_token_counter(tokenizer_name: str | None) -> TokenCounter | None:
    """Load a HF tokenizer lazily; retokenized counts skip special tokens.

    Retokenizing the final text counts only text-producing tokens, so it is a
    close lower bound of the server-side ``usage.completion_tokens`` (EOS and
    other special tokens emit no text).
    """

    if not tokenizer_name:
        return None
    from transformers import AutoTokenizer  # deferred: heavy optional dependency

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def count_tokens(text: str) -> int:
        if not text:
            return 0
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    return count_tokens


def count_completion_tokens(
    text: str,
    usage_completion_tokens: int | None,
    counter: TokenCounter | None,
) -> tuple[int | None, str]:
    if usage_completion_tokens is not None:
        return usage_completion_tokens, "usage"
    if counter is not None:
        return counter(text), "retokenized"
    return None, "unavailable"


async def stream_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    messages: list[dict[str, str]],
    output_tokens: int,
    headers: dict[str, str] | None = None,
) -> StreamResult:
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": output_tokens,
        "temperature": 0,
        "stream": True,
    }
    started = time.perf_counter()
    chunk_times: list[float] = []
    text_parts: list[str] = []
    finish_reason: str | None = None
    usage_completion: int | None = None
    async with session.post(url, json=body, headers=headers) as response:
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
                    usage = document.get("usage")
                    if isinstance(usage, dict) and isinstance(
                        usage.get("completion_tokens"), int
                    ):
                        usage_completion = usage["completion_tokens"]
                    choices = document.get("choices") or [{}]
                    choice = choices[0]
                    delta = choice.get("delta", {}).get("content", "")
                    if delta:
                        chunk_times.append(time.perf_counter())
                        text_parts.append(delta)
                    if choice.get("finish_reason") is not None:
                        finish_reason = str(choice["finish_reason"])
    finished = time.perf_counter()
    return StreamResult(
        started=started,
        finished=finished,
        chunk_times=tuple(chunk_times),
        text="".join(text_parts),
        finish_reason=finish_reason,
        usage_completion_tokens=usage_completion,
    )


def to_measurement(result: StreamResult, counter: TokenCounter | None) -> RequestMeasurement:
    tokens, source = count_completion_tokens(
        result.text, result.usage_completion_tokens, counter
    )
    return RequestMeasurement(
        ttft_ms=result.ttft_ms,
        response_ms=result.response_ms,
        inter_chunk_ms=result.inter_chunk_ms,
        content_chunks=len(result.chunk_times),
        completion_tokens=tokens,
        token_source=source,
        finish_reason=result.finish_reason,
        text_nonempty=bool(result.text),
    )


async def run_batch(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    concurrency: int,
    label: str,
) -> tuple[list[StreamResult], float]:
    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    started = time.perf_counter()
    values = await asyncio.gather(
        *(
            stream_request(
                session,
                url,
                args.model,
                [
                    {
                        "role": "user",
                        "content": args.prompt.format(request=f"{label}-{index}"),
                    }
                ],
                args.output_tokens,
            )
            for index in range(concurrency)
        )
    )
    return list(values), (time.perf_counter() - started) * 1_000


def summarize_closed_scenario(
    concurrency: int,
    requests: list[RequestMeasurement],
    walls: list[float],
    output_tokens: int,
) -> dict[str, Any]:
    mean_wall = statistics.fmean(walls)
    total_wall_seconds = sum(walls) / 1_000
    pooled_intervals = [interval for value in requests for interval in value.inter_chunk_ms]
    per_request_tpot = [
        statistics.fmean(value.inter_chunk_ms)
        for value in requests
        if value.inter_chunk_ms
    ]
    known_tokens = [
        value.completion_tokens
        for value in requests
        if value.completion_tokens is not None
    ]
    tokens_total = sum(known_tokens) if len(known_tokens) == len(requests) else None
    total_chunks = sum(value.content_chunks for value in requests)
    return {
        "concurrency": concurrency,
        "measured_requests": len(requests),
        "ttft_ms": summary_stats(value.ttft_ms for value in requests),
        "tpot_ms": summary_stats(per_request_tpot),
        "inter_chunk_ms": summary_stats(pooled_intervals),
        "response_ms": summary_stats(value.response_ms for value in requests),
        "batch_wall_mean_ms": mean_wall,
        "aggregate_nominal_tok_s": (
            concurrency * output_tokens / (mean_wall / 1_000)
        ),
        "aggregate_content_chunks_per_second": (
            total_chunks / total_wall_seconds if total_wall_seconds > 0 else 0.0
        ),
        "aggregate_completion_tokens_per_second": (
            tokens_total / total_wall_seconds
            if tokens_total is not None and total_wall_seconds > 0
            else None
        ),
        "completion_tokens_total": tokens_total,
        "token_sources": sorted({value.token_source for value in requests}),
        "length_finishes": sum(
            value.finish_reason == "length" for value in requests
        ),
        "nonempty": sum(value.text_nonempty for value in requests),
    }


async def run_closed_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    concurrencies = positive_csv(args.concurrencies)
    if args.iterations < 1 or args.warmups < 0 or args.output_tokens < 1:
        raise ValueError("iterations/output-tokens must be positive and warmups non-negative")
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        raise ValueError("timeout-seconds must be finite and positive")
    metadata = parse_metadata(args.metadata)
    counter = build_token_counter(args.tokenizer)
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
            results: list[StreamResult] = []
            walls: list[float] = []
            for iteration in range(args.iterations):
                values, wall = await run_batch(
                    session,
                    args,
                    concurrency,
                    f"measure-{concurrency}-{iteration}",
                )
                results.extend(values)
                walls.append(wall)
            requests = [to_measurement(result, counter) for result in results]
            rows.append(
                summarize_closed_scenario(concurrency, requests, walls, args.output_tokens)
            )
        async with session.get(health_url) as response:
            health = await response.json()
    return {
        "schema_version": 2,
        "kind": "openai_api_continuous_scheduler",
        "configuration": {
            "base_url": args.base_url,
            "mode": "closed",
            "model": args.model,
            "concurrencies": list(concurrencies),
            "output_tokens": args.output_tokens,
            "warm_batches_per_scenario": args.warmups,
            "measured_batches_per_scenario": args.iterations,
            "persistent_http_connections_prewarmed": True,
            "tokenizer": args.tokenizer,
            "metadata": metadata,
        },
        "rows": rows,
        "server_after_measurement": health,
    }


async def open_loop_request(
    session: aiohttp.ClientSession,
    url: str,
    args: argparse.Namespace,
    origin: float,
    scheduled_offset_s: float,
    index: int,
) -> OpenLoopSample:
    messages = [
        {"role": "user", "content": args.prompt.format(request=f"open-{index}")}
    ]
    started = time.perf_counter()
    try:
        result = await stream_request(session, url, args.model, messages, args.output_tokens)
    except Exception as error:  # noqa: BLE001 - a failed request is data, not fatal
        finished = time.perf_counter()
        return OpenLoopSample(
            index=index,
            scheduled_offset_s=scheduled_offset_s,
            started_offset_s=started - origin,
            finished_offset_s=finished - origin,
            ttft_ms=None,
            response_ms=(finished - started) * 1_000,
            chunk_offsets_s=(),
            inter_chunk_ms=(),
            content_chunks=0,
            text="",
            usage_completion_tokens=None,
            finish_reason=None,
            error=f"{type(error).__name__}: {error}",
        )
    return OpenLoopSample(
        index=index,
        scheduled_offset_s=scheduled_offset_s,
        started_offset_s=result.started - origin,
        finished_offset_s=result.finished - origin,
        ttft_ms=result.ttft_ms if result.chunk_times else None,
        response_ms=result.response_ms,
        chunk_offsets_s=tuple(stamp - origin for stamp in result.chunk_times),
        inter_chunk_ms=result.inter_chunk_ms,
        content_chunks=len(result.chunk_times),
        text=result.text,
        usage_completion_tokens=result.usage_completion_tokens,
        finish_reason=result.finish_reason,
        error=None,
    )


def mean_inflight(
    samples: list[OpenLoopSample], window_start: float, window_end: float
) -> float:
    window = window_end - window_start
    if window <= 0:
        return 0.0
    overlap = sum(
        max(
            0.0,
            min(sample.finished_offset_s, window_end)
            - max(sample.started_offset_s, window_start),
        )
        for sample in samples
    )
    return overlap / window


def summarize_open_loop(
    samples: list[OpenLoopSample],
    rate: float,
    args: argparse.Namespace,
    counter: TokenCounter | None,
) -> dict[str, Any]:
    window_start = args.warmup_seconds
    window_end = args.duration_seconds
    window = window_end - window_start
    completed = [sample for sample in samples if sample.error is None]
    failed = [sample for sample in samples if sample.error is not None]
    # Token counting happens after the scenario finished so tokenizer CPU work
    # cannot perturb the measurement.
    tokens_by_index: dict[int, int | None] = {}
    sources: set[str] = set()
    for sample in completed:
        tokens, source = count_completion_tokens(
            sample.text, sample.usage_completion_tokens, counter
        )
        tokens_by_index[sample.index] = tokens
        sources.add(source)

    steady = [
        sample
        for sample in completed
        if window_start <= sample.started_offset_s <= window_end
    ]
    pooled_intervals = [
        interval for sample in steady for interval in sample.inter_chunk_ms
    ]
    per_request_tpot = [
        statistics.fmean(sample.inter_chunk_ms)
        for sample in steady
        if sample.inter_chunk_ms
    ]
    steady_tokens = [
        tokens_by_index[sample.index]
        for sample in steady
        if tokens_by_index.get(sample.index) is not None
    ]

    chunks_in_window = 0
    token_weight_in_window = 0.0
    token_rate_complete = True
    for sample in samples:
        in_window = sum(
            1
            for offset in sample.chunk_offsets_s
            if window_start <= offset <= window_end
        )
        if in_window == 0:
            continue
        chunks_in_window += in_window
        tokens = tokens_by_index.get(sample.index)
        if tokens is None or sample.content_chunks == 0:
            token_rate_complete = False
            continue
        token_weight_in_window += in_window * (tokens / sample.content_chunks)

    half = window_start + window / 2
    return {
        "mode": "open",
        "arrival_process": "poisson",
        "lambda_rps": rate,
        "duration_seconds": args.duration_seconds,
        "warmup_seconds": args.warmup_seconds,
        "seed": args.seed,
        "offered_requests": len(samples),
        "achieved_arrival_rate_rps": len(samples) / args.duration_seconds,
        "completed_requests": len(completed),
        "errors": len(failed),
        "error_examples": [sample.error for sample in failed[:5]],
        "steady_state": {
            "window_seconds": window,
            "requests": len(steady),
            "average_inflight_requests": mean_inflight(samples, window_start, window_end),
            "average_inflight_first_half": mean_inflight(samples, window_start, half),
            "average_inflight_second_half": mean_inflight(samples, half, window_end),
            "ttft_ms": summary_stats(
                sample.ttft_ms for sample in steady if sample.ttft_ms is not None
            ),
            "tpot_ms": summary_stats(per_request_tpot),
            "inter_chunk_ms": summary_stats(pooled_intervals),
            "response_ms": summary_stats(sample.response_ms for sample in steady),
            "completion_tokens_per_request": summary_stats(steady_tokens),
            "aggregate_content_chunks_per_second": (
                chunks_in_window / window if window > 0 else 0.0
            ),
            "aggregate_completion_tokens_per_second": (
                token_weight_in_window / window
                if window > 0 and (token_rate_complete or token_weight_in_window > 0)
                else None
            ),
            "token_rate_is_complete": token_rate_complete,
        },
        "token_sources": sorted(sources),
    }


async def run_open_loop_scenario(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    rate: float,
    counter: TokenCounter | None,
) -> dict[str, Any]:
    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    rng = random.Random(args.seed)
    tasks: list[asyncio.Task[OpenLoopSample]] = []
    origin = time.perf_counter()
    arrival = rng.expovariate(rate)
    index = 0
    while arrival <= args.duration_seconds:
        delay = origin + arrival - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        tasks.append(
            asyncio.create_task(
                open_loop_request(session, url, args, origin, arrival, index)
            )
        )
        index += 1
        arrival += rng.expovariate(rate)
    samples = list(await asyncio.gather(*tasks)) if tasks else []
    return summarize_open_loop(samples, rate, args, counter)


async def run_open_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    rates = positive_float_csv(args.lambda_rps)
    if not math.isfinite(args.duration_seconds) or args.duration_seconds <= 0:
        raise ValueError("duration-seconds must be finite and positive")
    if (
        not math.isfinite(args.warmup_seconds)
        or args.warmup_seconds < 0
        or args.warmup_seconds >= args.duration_seconds
    ):
        raise ValueError("warmup-seconds must be non-negative and below duration-seconds")
    if args.output_tokens < 1:
        raise ValueError("output-tokens must be positive")
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        raise ValueError("timeout-seconds must be finite and positive")
    if args.prewarm_connections < 1:
        raise ValueError("prewarm-connections must be positive")
    metadata = parse_metadata(args.metadata)
    counter = build_token_counter(args.tokenizer)
    connector = aiohttp.TCPConnector(limit=512, limit_per_host=512)
    timeout = aiohttp.ClientTimeout(total=args.timeout_seconds)
    rows: list[dict[str, Any]] = []
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        health_url = args.base_url.rstrip("/") + "/health"
        health_responses = await asyncio.gather(
            *(session.get(health_url) for _ in range(args.prewarm_connections))
        )
        for response in health_responses:
            response.release()
        async with session.get(health_url) as response:
            response.raise_for_status()
            initial_health = await response.json()
        for rate in rates:
            rows.append(await run_open_loop_scenario(session, args, rate, counter))
        async with session.get(health_url) as response:
            final_health = await response.json()
    return {
        "schema_version": 2,
        "kind": "openai_api_open_loop",
        "configuration": {
            "base_url": args.base_url,
            "mode": "open",
            "model": args.model,
            "lambda_rps": list(rates),
            "duration_seconds": args.duration_seconds,
            "warmup_seconds": args.warmup_seconds,
            "output_tokens": args.output_tokens,
            "prompt_template": args.prompt,
            "timeout_seconds": args.timeout_seconds,
            "seed": args.seed,
            "prewarm_connections": args.prewarm_connections,
            "tokenizer": args.tokenizer,
            "token_source_priority": ["usage", "retokenized"],
            "metadata": metadata,
        },
        "rows": rows,
        "server_health_before": initial_health,
        "server_after_measurement": final_health,
    }


async def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    if args.mode == "open":
        return await run_open_benchmark(args)
    return await run_closed_benchmark(args)


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
