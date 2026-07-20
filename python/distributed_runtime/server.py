from __future__ import annotations

import argparse
import asyncio
import codecs
from dataclasses import dataclass, field
import json
import math
import multiprocessing as mp
import sys
import time
from typing import Any
import uuid

from aiohttp import web
import torch
from transformers import AutoConfig

from .engine import (
    DistributedPipelineEngine,
    GenerationCancelledError,
    GenerationInput,
    GenerationOutput,
    PipelineEngineConfig,
    balanced_boundaries,
    parse_boundaries,
)
from .model import load_tokenizer, resolve_model_snapshot
from .protocol import TensorCodec


DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


@dataclass
class PendingGeneration:
    client_id: int
    input_ids: torch.Tensor
    max_new_tokens: int
    prompt_tokens: int
    events: asyncio.Queue[tuple[str, Any]] = field(default_factory=asyncio.Queue)
    abandoned: bool = False
    session_key: str | None = None


class ContinuousMicroBatcher:
    """Admission window in front of the engine's continuous request scheduler."""

    def __init__(
        self,
        engine: DistributedPipelineEngine,
        *,
        max_batch_size: int,
        batch_window_ms: float,
        eos_token_ids: frozenset[int],
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        if batch_window_ms < 0:
            raise ValueError("batch_window_ms cannot be negative")
        self.engine = engine
        self.max_batch_size = max_batch_size
        self.batch_window_ms = batch_window_ms
        self.eos_token_ids = eos_token_ids
        self.queue: asyncio.Queue[PendingGeneration | None] = asyncio.Queue()
        self.task: asyncio.Task[None] | None = None
        self.inflight: set[asyncio.Task[None]] = set()
        self.active: dict[int, PendingGeneration] = {}
        self.closing = False
        self.batches = 0
        self.requests = 0
        self.last_batch_size = 0

    async def start(self) -> None:
        if self.task is None:
            self.task = asyncio.create_task(self._run(), name="distributed-microbatcher")

    async def submit(self, pending: PendingGeneration) -> None:
        if self.closing or self.task is None or self.task.done():
            raise RuntimeError("microbatcher is not running")
        self.queue.put_nowait(pending)

    def cancel(self, pending: PendingGeneration) -> None:
        pending.abandoned = True
        self.engine.cancel(pending.client_id)

    async def close(self) -> None:
        if self.closing:
            return
        self.closing = True
        if self.task is not None and not self.task.done():
            self.queue.put_nowait(None)
            await self.task
        for pending in list(self.active.values()):
            self.cancel(pending)
        await asyncio.to_thread(self.engine.close)
        if self.inflight:
            await asyncio.gather(*self.inflight, return_exceptions=True)

    async def _run(self) -> None:
        while True:
            first = await self.queue.get()
            if first is None:
                self._fail_queued(RuntimeError("microbatcher is closing"))
                return
            batch = [first]
            if self.batch_window_ms > 0:
                await asyncio.sleep(self.batch_window_ms / 1_000)
            while len(batch) < self.max_batch_size:
                try:
                    item = self.queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if item is None:
                    self.queue.put_nowait(None)
                    break
                batch.append(item)
            task = asyncio.create_task(self._dispatch(batch))
            self.inflight.add(task)
            task.add_done_callback(self.inflight.discard)

    async def _dispatch(self, batch: list[PendingGeneration]) -> None:
        batch = [pending for pending in batch if not pending.abandoned]
        if not batch:
            return
        by_id = {pending.client_id: pending for pending in batch}
        self.active.update(by_id)
        loop = asyncio.get_running_loop()

        def on_token(
            client_id: int,
            token_id: int,
            step: int,
            arrived: float,
        ) -> None:
            pending = by_id.get(client_id)
            if pending is None or pending.abandoned:
                return
            loop.call_soon_threadsafe(
                pending.events.put_nowait,
                ("token", (token_id, step, arrived)),
            )

        inputs = [
            GenerationInput(
                client_id=pending.client_id,
                input_ids=pending.input_ids,
                max_new_tokens=pending.max_new_tokens,
                eos_token_ids=self.eos_token_ids,
                session_key=pending.session_key,
            )
            for pending in batch
        ]
        try:
            futures = self.engine.submit(inputs, on_token)
        except BaseException as error:
            for pending in batch:
                if not pending.abandoned:
                    pending.events.put_nowait(("error", error))
                self.active.pop(pending.client_id, None)
            return

        self.batches += 1
        self.requests += len(batch)
        self.last_batch_size = len(batch)

        async def bridge(pending: PendingGeneration, future: Any) -> None:
            try:
                output = await asyncio.wrap_future(future)
            except asyncio.CancelledError:
                if not pending.abandoned:
                    pending.events.put_nowait(
                        ("error", GenerationCancelledError("generation cancelled"))
                    )
            except BaseException as error:
                if not pending.abandoned:
                    pending.events.put_nowait(("error", error))
            else:
                if not pending.abandoned:
                    pending.events.put_nowait(("done", output))
            finally:
                self.active.pop(pending.client_id, None)

        await asyncio.gather(
            *(bridge(pending, future) for pending, future in zip(batch, futures))
        )

    def _fail_queued(self, error: BaseException) -> None:
        while True:
            try:
                pending = self.queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            if pending is not None and not pending.abandoned:
                pending.events.put_nowait(("error", error))


class IncrementalTokenDecoder:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.token_ids: list[int] = []
        self.emitted = ""
        self.byte_decoder = getattr(tokenizer, "byte_decoder", None)
        self.special_ids = frozenset(int(value) for value in getattr(tokenizer, "all_special_ids", ()))
        self.utf8_decoder = (
            codecs.getincrementaldecoder("utf-8")(errors="replace")
            if isinstance(self.byte_decoder, dict)
            else None
        )

    def push(self, token_id: int) -> str:
        self.token_ids.append(token_id)
        if self.utf8_decoder is not None:
            if token_id in self.special_ids:
                return ""
            token = self.tokenizer.convert_ids_to_tokens(token_id)
            try:
                raw = bytes(self.byte_decoder[character] for character in token)
            except (KeyError, TypeError):
                # Added non-special tokens may not use the GPT-2 byte alphabet.
                # Fall back to the generic cumulative path for this tokenizer.
                self.utf8_decoder = None
            else:
                delta = self.utf8_decoder.decode(raw, final=False)
                self.emitted += delta
                return delta
        decoded = self._decode()
        # Byte-level tokenizers use U+FFFD for an incomplete final UTF-8 sequence.
        # Withhold that unstable suffix so a later token can complete it exactly.
        stable = decoded.rstrip("\ufffd")
        if not stable.startswith(self.emitted):
            return ""
        delta = stable[len(self.emitted) :]
        self.emitted = stable
        return delta

    def finish(self) -> str:
        if self.utf8_decoder is not None:
            delta = self.utf8_decoder.decode(b"", final=True)
            self.emitted += delta
            return delta
        decoded = self._decode()
        if decoded.startswith(self.emitted):
            delta = decoded[len(self.emitted) :]
            self.emitted = decoded
            return delta
        return ""

    def _decode(self) -> str:
        return self.tokenizer.decode(
            self.token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )


class DistributedOpenAIServer:
    def __init__(
        self,
        engine: DistributedPipelineEngine,
        tokenizer: Any,
        *,
        public_model_name: str,
        max_batch_size: int,
        batch_window_ms: float,
        max_output_tokens: int,
    ) -> None:
        self.engine = engine
        self.tokenizer = tokenizer
        self.public_model_name = public_model_name
        self.max_output_tokens = max_output_tokens
        eos = tokenizer.eos_token_id
        if eos is None:
            eos_ids = frozenset()
        elif isinstance(eos, int):
            eos_ids = frozenset((eos,))
        else:
            eos_ids = frozenset(int(value) for value in eos)
        self.batcher = ContinuousMicroBatcher(
            engine,
            max_batch_size=max_batch_size,
            batch_window_ms=batch_window_ms,
            eos_token_ids=eos_ids,
        )
        self._client_counter = 0
        self.started_at = time.time()

    def create_app(self) -> web.Application:
        app = web.Application(client_max_size=1 * 1024 * 1024)
        app.router.add_get("/health", self.health)
        app.router.add_get("/v1/models", self.models)
        app.router.add_post("/v1/chat/completions", self.chat_completions)
        app.on_startup.append(self._on_startup)
        app.on_cleanup.append(self._on_cleanup)
        return app

    async def _on_startup(self, _: web.Application) -> None:
        await self.batcher.start()

    async def _on_cleanup(self, _: web.Application) -> None:
        await self.batcher.close()

    async def health(self, _: web.Request) -> web.Response:
        healthy = self.engine.healthy
        return web.json_response(
            {
                "status": "ready" if healthy else "degraded",
                "error": self.engine.fatal_error,
                "model": self.public_model_name,
                "stages": self.engine.stages,
                "boundaries": list(self.engine.config.boundaries),
                "codec": self.engine.config.codec.name.lower(),
                "prefill_chunk_tokens": self.engine.config.prefill_chunk_tokens,
                "speculation": self.engine.speculation_stats,
                "sessions": self.engine.session_stats,
                "root_parameter_bytes": self.engine.root_parameter_bytes,
                "batcher": {
                    "queued": self.batcher.queue.qsize(),
                    "batches": self.batcher.batches,
                    "requests": self.batcher.requests,
                    "last_batch_size": self.batcher.last_batch_size,
                    "active": len(self.batcher.active),
                    "dispatches": len(self.batcher.inflight),
                },
                "uptime_seconds": max(0.0, time.time() - self.started_at),
            }
        )

    async def models(self, _: web.Request) -> web.Response:
        return web.json_response(
            {
                "object": "list",
                "data": [
                    {
                        "id": self.public_model_name,
                        "object": "model",
                        "created": int(self.started_at),
                        "owned_by": "distributed-runtime",
                    }
                ],
            }
        )

    async def chat_completions(self, request: web.Request) -> web.StreamResponse:
        try:
            body = await request.json()
            pending, stream = self._prepare_request(
                body,
                header_session_id=request.headers.get("X-Session-Id"),
            )
        except (json.JSONDecodeError, TypeError, ValueError, KeyError) as error:
            return error_response(str(error), "invalid_request", 400)
        try:
            await self.batcher.submit(pending)
        except RuntimeError as error:
            return error_response(str(error), "service_unavailable", 503)
        if stream:
            return await self._stream_response(request, pending)
        return await self._complete_response(pending)

    def _prepare_request(
        self,
        body: Any,
        header_session_id: str | None = None,
    ) -> tuple[PendingGeneration, bool]:
        if not isinstance(body, dict):
            raise ValueError("request body must be a JSON object")
        supported_fields = {
            "model",
            "messages",
            "max_tokens",
            "max_completion_tokens",
            "temperature",
            "top_p",
            "n",
            "stream",
            "seed",
            "user",
        }
        unsupported = sorted(set(body) - supported_fields)
        if unsupported:
            raise ValueError(f"unsupported request field {unsupported[0]!r}")
        if "model" not in body or not isinstance(body["model"], str) or not body["model"]:
            raise ValueError("model is required and must be a non-empty string")
        requested_model = body["model"]
        if requested_model != self.public_model_name:
            raise ValueError(f"unknown model {requested_model!r}")
        temperature = exact_number(body.get("temperature", 0.0), "temperature")
        if temperature != 0.0:
            raise ValueError("this low-latency runtime currently supports greedy temperature=0 only")
        top_p = exact_number(body.get("top_p", 1.0), "top_p")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be greater than 0 and at most 1")
        if exact_integer(body.get("n", 1), "n") != 1:
            raise ValueError("n must be 1")
        if "seed" in body:
            exact_integer(body["seed"], "seed")
        if "stream" in body and type(body["stream"]) is not bool:
            raise ValueError("stream must be a boolean")
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty array")
        normalized: list[dict[str, str]] = []
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ValueError(f"messages[{index}] must be an object")
            role = message.get("role")
            content = message.get("content")
            if role not in ("system", "developer", "user", "assistant") or not isinstance(content, str):
                raise ValueError(f"messages[{index}] requires a supported role and string content")
            normalized.append(
                {"role": "system" if role == "developer" else role, "content": content}
            )
        if "user" in body and (
            not isinstance(body["user"], str) or not body["user"].strip()
        ):
            raise ValueError("user must be a non-empty string")
        # An explicit transport header wins; the OpenAI ``user`` field is the
        # compatible fallback so unmodified clients can still pin their chat.
        session_key = normalize_session_key(
            header_session_id if header_session_id is not None else body.get("user")
        )
        if "max_tokens" in body and "max_completion_tokens" in body:
            raise ValueError("use max_tokens or max_completion_tokens, not both")
        raw_max = body.get("max_tokens", body.get("max_completion_tokens", 64))
        max_new_tokens = exact_integer(raw_max, "max_tokens")
        if not 1 <= max_new_tokens <= self.max_output_tokens:
            raise ValueError(f"max_tokens must be between 1 and {self.max_output_tokens}")

        encoded = self.tokenizer.apply_chat_template(
            normalized,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        input_ids = encoded if isinstance(encoded, torch.Tensor) else encoded["input_ids"]
        input_ids = input_ids.to(dtype=torch.long, device="cpu")
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        prompt_tokens = int(input_ids.shape[1])
        if self.engine.maximum_context and prompt_tokens + max_new_tokens > self.engine.maximum_context:
            raise ValueError(
                f"prompt plus output exceeds context window {self.engine.maximum_context}"
            )
        self._client_counter += 1
        return (
            PendingGeneration(
                client_id=self._client_counter,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                prompt_tokens=prompt_tokens,
                session_key=session_key,
            ),
            body.get("stream", False),
        )

    async def _stream_response(
        self,
        request: web.Request,
        pending: PendingGeneration,
    ) -> web.StreamResponse:
        response = web.StreamResponse(
            status=200,
            headers={
                "content-type": "text/event-stream; charset=utf-8",
                "cache-control": "no-cache",
                "x-accel-buffering": "no",
            },
        )
        await response.prepare(request)
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        decoder = IncrementalTokenDecoder(self.tokenizer)
        sent_role = False
        completed = False
        try:
            while True:
                kind, payload = await pending.events.get()
                if kind == "token":
                    token_id, _, _ = payload
                    delta = decoder.push(token_id)
                    if delta:
                        await write_sse(
                            response,
                            chunk_payload(
                                completion_id,
                                created,
                                self.public_model_name,
                                delta,
                                role="assistant" if not sent_role else None,
                            ),
                        )
                        sent_role = True
                elif kind == "done":
                    output: GenerationOutput = payload
                    tail = decoder.finish()
                    if tail:
                        await write_sse(
                            response,
                            chunk_payload(
                                completion_id,
                                created,
                                self.public_model_name,
                                tail,
                                role="assistant" if not sent_role else None,
                            ),
                        )
                        sent_role = True
                    if not sent_role:
                        await write_sse(
                            response,
                            chunk_payload(
                                completion_id,
                                created,
                                self.public_model_name,
                                "",
                                role="assistant",
                            ),
                        )
                        sent_role = True
                    await write_sse(
                        response,
                        chunk_payload(
                            completion_id,
                            created,
                            self.public_model_name,
                            "",
                            finish_reason=output.finish_reason,
                        ),
                    )
                    await response.write(b"data: [DONE]\n\n")
                    completed = True
                    break
                elif kind == "error":
                    await write_sse(
                        response,
                        {"error": {"message": str(payload), "type": "pipeline_error"}},
                    )
                    await response.write(b"data: [DONE]\n\n")
                    completed = True
                    break
        except (ConnectionResetError, BrokenPipeError):
            self.batcher.cancel(pending)
        except asyncio.CancelledError:
            self.batcher.cancel(pending)
            raise
        finally:
            if not completed:
                self.batcher.cancel(pending)
        return response

    async def _complete_response(self, pending: PendingGeneration) -> web.Response:
        token_ids: list[int] = []
        try:
            while True:
                kind, payload = await pending.events.get()
                if kind == "token":
                    token_ids.append(int(payload[0]))
                elif kind == "error":
                    return error_response(str(payload), "pipeline_error", 500)
                elif kind == "done":
                    output: GenerationOutput = payload
                    text = self.tokenizer.decode(
                        token_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    return web.json_response(
                        {
                            "id": f"chatcmpl-{uuid.uuid4().hex}",
                            "object": "chat.completion",
                            "created": int(time.time()),
                            "model": self.public_model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {"role": "assistant", "content": text},
                                    "finish_reason": output.finish_reason,
                                }
                            ],
                            "usage": {
                                "prompt_tokens": pending.prompt_tokens,
                                "completion_tokens": len(token_ids),
                                "total_tokens": pending.prompt_tokens + len(token_ids),
                            },
                            "distribution_metrics": {
                                "ttft_ms": output.ttft_ms,
                                "tpot_ms": output.tpot_ms,
                                "pipeline_ms": output.total_ms,
                                "reused_kv_tokens": output.reused_kv_tokens,
                            },
                        }
                    )
        except asyncio.CancelledError:
            self.batcher.cancel(pending)
            raise


def normalize_session_key(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("session identifier must be a string")
    normalized = value.strip()
    if not normalized:
        return None
    if len(normalized) > 128:
        raise ValueError("session identifier must be at most 128 characters")
    return normalized


def exact_integer(value: Any, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    return value


def exact_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def chunk_payload(
    completion_id: str,
    created: int,
    model: str,
    text: str,
    *,
    role: str | None = None,
    finish_reason: str | None = None,
) -> dict[str, Any]:
    delta: dict[str, str] = {}
    if role is not None:
        delta["role"] = role
    if text:
        delta["content"] = text
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


async def write_sse(response: web.StreamResponse, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    await response.write(b"data: " + encoded + b"\n\n")


def error_response(message: str, kind: str, status: int) -> web.Response:
    return web.json_response(
        {"error": {"message": message, "type": kind}},
        status=status,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OpenAI-compatible API backed by the persistent GDLP layer pipeline."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--public-model-name", default="distributed-small")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--stages", type=int, default=2)
    parser.add_argument("--boundaries")
    parser.add_argument(
        "--codec",
        choices=(
            "fp32",
            "fp16",
            "int8",
            "int8-grouped",
            "int8-hadamard",
            "int8-grouped-deflate",
            "int8-hadamard-deflate",
        ),
        default="fp16",
    )
    parser.add_argument("--threads-per-stage", type=int, default=1)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--max-active-sequences", type=int, default=8)
    parser.add_argument("--max-pending-requests", type=int, default=128)
    parser.add_argument("--batch-window-ms", type=float, default=2.0)
    parser.add_argument(
        "--prefill-chunk-tokens",
        type=int,
        default=0,
        help="Split long prompts into bounded pipeline waves; zero disables chunking.",
    )
    parser.add_argument(
        "--speculation",
        choices=("off", "ngram"),
        default="off",
        help="Enable exact adaptive speculative verification with the selected drafter.",
    )
    parser.add_argument("--speculative-max-draft-tokens", type=int, default=4)
    parser.add_argument("--speculation-minimum-speedup", type=float, default=1.05)
    parser.add_argument("--no-speculation-probes", action="store_true")
    parser.add_argument(
        "--max-retained-sessions",
        type=int,
        default=0,
        help=(
            "Keep the KV of up to this many finished chats alive on every stage "
            "so the next turn (X-Session-Id header or OpenAI user field) only "
            "prefills the new suffix; zero disables session retention."
        ),
    )
    parser.add_argument(
        "--max-retained-session-tokens",
        type=int,
        default=0,
        help="Total idle KV tokens across retained sessions; zero is unbounded.",
    )
    parser.add_argument("--retained-session-ttl-seconds", type=float, default=600.0)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--first-stage-host")
    parser.add_argument("--first-stage-port", type=int)
    parser.add_argument("--return-bind-host", default="127.0.0.1")
    parser.add_argument("--return-advertise-host", default="127.0.0.1")
    parser.add_argument("--return-port", type=int, default=0)
    parser.add_argument("--startup-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--socket-timeout-seconds", type=float, default=180.0)
    return parser.parse_args(argv)


def build_server(args: argparse.Namespace) -> DistributedOpenAIServer:
    if not 1 <= args.port <= 65_535:
        raise ValueError("port must be between 1 and 65535")
    if not args.public_model_name.strip():
        raise ValueError("public-model-name cannot be empty")
    if args.threads_per_stage < 1:
        raise ValueError("threads-per-stage must be positive")
    if args.max_batch_size < 1:
        raise ValueError("max-batch-size must be positive")
    if args.max_active_sequences < 1:
        raise ValueError("max-active-sequences must be positive")
    if args.max_batch_size > args.max_active_sequences:
        raise ValueError("max-batch-size cannot exceed max-active-sequences")
    if args.max_pending_requests < args.max_active_sequences:
        raise ValueError("max-pending-requests must be at least max-active-sequences")
    if not math.isfinite(args.batch_window_ms) or args.batch_window_ms < 0:
        raise ValueError("batch-window-ms must be finite and non-negative")
    if args.prefill_chunk_tokens < 0:
        raise ValueError("prefill-chunk-tokens must be non-negative")
    if not 1 <= args.speculative_max_draft_tokens <= 16:
        raise ValueError("speculative-max-draft-tokens must be between 1 and 16")
    if (
        not math.isfinite(args.speculation_minimum_speedup)
        or args.speculation_minimum_speedup < 1
    ):
        raise ValueError("speculation-minimum-speedup must be finite and at least 1")
    if args.max_output_tokens < 1:
        raise ValueError("max-output-tokens must be positive")
    if args.max_retained_sessions < 0 or args.max_retained_session_tokens < 0:
        raise ValueError("retained session limits must be non-negative")
    if (
        not math.isfinite(args.retained_session_ttl_seconds)
        or args.retained_session_ttl_seconds <= 0
    ):
        raise ValueError("retained-session-ttl-seconds must be finite and positive")
    snapshot = resolve_model_snapshot(args.model, args.revision)
    model_config = AutoConfig.from_pretrained(snapshot)
    total_layers = int(model_config.num_hidden_layers)
    boundaries = (
        parse_boundaries(args.boundaries, total_layers)
        if args.boundaries
        else balanced_boundaries(total_layers, args.stages)
    )
    remote = args.first_stage_host is not None or args.first_stage_port is not None
    if remote and (args.first_stage_host is None or args.first_stage_port is None):
        raise ValueError("remote mode requires both first-stage-host and first-stage-port")
    codec = {
        "fp32": TensorCodec.FP32,
        "fp16": TensorCodec.FP16,
        "int8": TensorCodec.INT8,
        "int8-grouped": TensorCodec.INT8_GROUPED,
        "int8-hadamard": TensorCodec.INT8_HADAMARD,
        "int8-grouped-deflate": TensorCodec.INT8_GROUPED_DEFLATE,
        "int8-hadamard-deflate": TensorCodec.INT8_HADAMARD_DEFLATE,
    }[args.codec]
    engine = DistributedPipelineEngine(
        PipelineEngineConfig(
            model_name=snapshot,
            boundaries=boundaries,
            codec=codec,
            threads_per_stage=args.threads_per_stage,
            startup_timeout_seconds=args.startup_timeout_seconds,
            socket_timeout_seconds=args.socket_timeout_seconds,
            spawn_local_stages=not remote,
            first_stage_host=args.first_stage_host or "127.0.0.1",
            first_stage_port=args.first_stage_port,
            return_bind_host=args.return_bind_host,
            return_advertise_host=args.return_advertise_host,
            return_port=args.return_port,
            max_active_sequences=args.max_active_sequences,
            max_pending_requests=args.max_pending_requests,
            prefill_chunk_tokens=args.prefill_chunk_tokens,
            speculative_max_draft_tokens=(
                args.speculative_max_draft_tokens
                if args.speculation == "ngram"
                else 0
            ),
            speculation_minimum_speedup=args.speculation_minimum_speedup,
            speculation_probe=not args.no_speculation_probes,
            max_retained_sessions=args.max_retained_sessions,
            max_retained_session_tokens=args.max_retained_session_tokens,
            retained_session_ttl_seconds=args.retained_session_ttl_seconds,
        )
    )
    try:
        tokenizer = load_tokenizer(engine.model_snapshot)
        return DistributedOpenAIServer(
            engine,
            tokenizer,
            public_model_name=args.public_model_name,
            max_batch_size=args.max_batch_size,
            batch_window_ms=args.batch_window_ms,
            max_output_tokens=args.max_output_tokens,
        )
    except BaseException:
        engine.close()
        raise


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(
        json.dumps(
            {
                "event": "root_loading",
                "model": args.model,
                "boundaries": args.boundaries,
                "remote": args.first_stage_host is not None,
            },
            sort_keys=True,
        ),
        file=sys.stderr,
        flush=True,
    )
    server = build_server(args)
    print(
        json.dumps(
            {
                "event": "root_engine_ready",
                "model": args.public_model_name,
                "stages": server.engine.stages,
                "boundaries": list(server.engine.config.boundaries),
            },
            sort_keys=True,
        ),
        file=sys.stderr,
        flush=True,
    )
    web.run_app(
        server.create_app(),
        host=args.host,
        port=args.port,
        access_log=None,
        handler_cancellation=True,
    )
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
