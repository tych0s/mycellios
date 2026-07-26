from __future__ import annotations

import argparse
import asyncio
import codecs
from dataclasses import dataclass, field, replace
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import struct
import sys
import time
from typing import Any
import uuid

from aiohttp import web
import torch
from transformers import AutoConfig

from .dense_tiering import (
    add_dense_tiering_arguments,
    dense_tiering_config_from_args,
)
from .draft_model import (
    add_local_draft_model_arguments,
    load_local_draft_model_provider,
    local_draft_model_config_from_args,
)
from .engine import (
    DistributedPipelineEngine,
    GenerationCancelledError,
    GenerationInput,
    GenerationOutput,
    MAX_SPECULATIVE_INFLIGHT_BYTES,
    MAX_SPECULATIVE_INFLIGHT_WAVES,
    PipelineEngineConfig,
    QueueFullError,
    balanced_boundaries,
    parse_boundaries,
)
from .model import load_tokenizer, resolve_model_snapshot
from .native_gguf import verify_native_gguf_stage
from .native_gguf_runtime import (
    add_native_gguf_arguments,
    native_gguf_runtime_from_args,
)
from .paged_stage import add_paged_kv_arguments, paged_kv_config_from_args
from .protocol import TensorCodec
from .ram_backed_moe_runtime import (
    add_ram_backed_moe_arguments,
    ram_backed_moe_config_from_args,
)
from .runtime_policy import reject_external_backend_arguments
from .recovery import (
    RecoveringPipelineEngine,
    RemoteRecoveryStandbyEngineFactory,
    RemoteRecoveryStandbyRoute,
)
from .stage import (
    MAX_SPECULATIVE_BRANCHES,
    MAX_SPECULATIVE_BRANCH_TOKENS,
    MAX_SPECULATIVE_KV_BYTES,
)
from .speculation import NgramTreeDraftProvider


DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"

# Backlog depth per decode slot. Four keeps the pipeline fed across a wave boundary
# without letting the queue grow into work that will only ever time out; measured
# overload with an unrelated flat queue drained to zero completions.
PENDING_PER_ACTIVE_SLOT = 4
OUTPUT_TOKEN_HASH_SCHEME = "gdlp-output-token-ids-v1"
OUTPUT_TOKEN_DIGEST_DOMAIN = OUTPUT_TOKEN_HASH_SCHEME.encode("ascii") + b"\0"


def tree_draft_provider_from_args(
    args: argparse.Namespace,
) -> NgramTreeDraftProvider | None:
    """Materialize only the native provider explicitly selected by the CLI."""

    if args.speculation != "draft-tree":
        return None
    return NgramTreeDraftProvider(
        max_draft_tokens=args.speculative_max_draft_tokens,
        max_branches=args.max_speculative_branches,
    )


def output_token_ids_sha256(token_ids: list[int] | tuple[int, ...]) -> str:
    """Seal an exact token sequence without publishing the token ids themselves."""

    digest = hashlib.sha256()
    digest.update(OUTPUT_TOKEN_DIGEST_DOMAIN)
    digest.update(struct.pack(">Q", len(token_ids)))
    for token_id in token_ids:
        if type(token_id) is not int or token_id < 0 or token_id > 0xFFFFFFFF:
            raise ValueError("output token id must be a uint32")
        digest.update(struct.pack(">I", token_id))
    return f"sha256:{digest.hexdigest()}"


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
        engine: DistributedPipelineEngine | RecoveringPipelineEngine,
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


class DistributedMycelliosServer:
    def __init__(
        self,
        engine: DistributedPipelineEngine | RecoveringPipelineEngine,
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
        artifact = self.engine.model_artifact
        recovery = getattr(
            self.engine,
            "recovery_stats",
            {"configured": False, "state": "disabled"},
        )
        recovering = recovery.get("state") == "recovering"
        status = "recovering" if recovering else ("ready" if healthy else "degraded")
        # `degraded` significa que el motor tiene un error fatal y NO va a servir
        # ni una petición más. Devolver 200 con esa palabra dentro convierte una
        # caída en un outage silencioso: todo supervisor, balanceador y sonda
        # mira el código, no el cuerpo. Y en el despliegue real (GpuCloud) la
        # recuperación automática está prohibida, así que nadie se entera nunca.
        # `recovering` sí es 200: el motor está trabajando en volver.
        return web.json_response(
            {
                "status": status,
                "error": self.engine.fatal_error,
                "model": self.public_model_name,
                "artifact_identity": artifact.identity,
                "canonical_model_source": artifact.canonical_source,
                "canonical_model_revision": artifact.canonical_revision,
                # JSON numbers cannot represent every uint64 exactly.  Keep the
                # pipeline identity lossless across Python and JS consumers.
                "pipeline_snapshot_identity": str(self.engine.pipeline_id),
                "stages": self.engine.stages,
                "boundaries": list(self.engine.config.boundaries),
                "codec": self.engine.config.codec.name.lower(),
                "prefill_chunk_tokens": self.engine.config.prefill_chunk_tokens,
                "prefill_inflight_chunks": (
                    self.engine.config.prefill_inflight_chunks
                ),
                "prefill_inflight_bytes": self.engine.config.prefill_inflight_bytes,
                "max_speculative_branches": (
                    self.engine.config.max_speculative_branches
                ),
                "max_speculative_branch_tokens": (
                    self.engine.config.max_speculative_branch_tokens
                ),
                "max_speculative_kv_bytes": self.engine.config.max_speculative_kv_bytes,
                "speculative_inflight_waves": getattr(
                    self.engine.config,
                    "speculative_inflight_waves",
                    1,
                ),
                "speculative_inflight_bytes": getattr(
                    self.engine.config,
                    "speculative_inflight_bytes",
                    0,
                ),
                "sealed_wave_tokens": self.engine.config.sealed_wave_token_limit,
                "max_prefill_chunk_tokens": (
                    self.engine.config.prefill_token_limit or 0
                ),
                "speculation": self.engine.speculation_stats,
                "sessions": getattr(
                    self.engine,
                    "session_stats",
                    {"configured": False, "retained_sessions": 0},
                ),
                "prefill_window": self.engine.prefill_window_stats,
                "speculative_window": getattr(
                    self.engine,
                    "speculative_window_stats",
                    {
                        "configured": False,
                        "configured_waves_per_request": 1,
                        "configured_bytes_per_request": 0,
                    },
                ),
                "root_batching": self.engine.root_batch_stats,
                "recovery": recovery,
                "root_parameter_bytes": self.engine.root_parameter_bytes,
                "execution": getattr(
                    self.engine,
                    "execution_topology",
                    {
                        "requested_device": "unknown",
                        "observed_stage_count": 0,
                        "total_stage_count": self.engine.stages,
                        "stages": [],
                    },
                ),
                "batcher": {
                    "queued": self.batcher.queue.qsize(),
                    "batches": self.batcher.batches,
                    "requests": self.batcher.requests,
                    "last_batch_size": self.batcher.last_batch_size,
                    "active": len(self.batcher.active),
                    "dispatches": len(self.batcher.inflight),
                },
                "uptime_seconds": max(0.0, time.time() - self.started_at),
            },
            status=200 if (healthy or recovering) else 503,
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
        except QueueFullError as error:
            # Backpressure, not failure. A 503 tells a client (and any load
            # balancer in front) that this node is broken and should be taken
            # out; a 429 with Retry-After tells it to slow down and come back,
            # which is what a saturated but healthy pipeline actually wants.
            # Measured under overload with a flat queue, every admitted request
            # timed out and goodput fell to ZERO
            # (docs/benchmarks/gpu_cloud-exp10-salida-larga-2026-07-24): refusing
            # work we cannot serve is what keeps the served work served.
            return overloaded_response(error)
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
        # An explicit transport header wins; the legacy ``user`` field is the
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
                    if tuple(decoder.token_ids) != output.token_ids:
                        await write_sse(
                            response,
                            {
                                "error": {
                                    "message": "token event stream does not match engine output",
                                    "type": "pipeline_evidence_error",
                                }
                            },
                        )
                        await response.write(b"data: [DONE]\n\n")
                        completed = True
                        break
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
                            usage={
                                "prompt_tokens": pending.prompt_tokens,
                                "completion_tokens": len(decoder.token_ids),
                                "total_tokens": pending.prompt_tokens
                                + len(decoder.token_ids),
                            },
                            distribution_metrics={
                                "ttft_ms": output.ttft_ms,
                                "tpot_ms": output.tpot_ms,
                                "pipeline_ms": output.total_ms,
                                "reused_kv_tokens": output.reused_kv_tokens,
                                "output_token_ids_sha256": output_token_ids_sha256(
                                    output.token_ids
                                ),
                                "output_token_ids_hash_scheme": OUTPUT_TOKEN_HASH_SCHEME,
                            },
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
                    if tuple(token_ids) != output.token_ids:
                        return error_response(
                            "token event stream does not match engine output",
                            "pipeline_evidence_error",
                            500,
                        )
                    text = self.tokenizer.decode(
                        output.token_ids,
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
                                "completion_tokens": len(output.token_ids),
                                "total_tokens": pending.prompt_tokens
                                + len(output.token_ids),
                            },
                            "distribution_metrics": {
                                "ttft_ms": output.ttft_ms,
                                "tpot_ms": output.tpot_ms,
                                "pipeline_ms": output.total_ms,
                                "reused_kv_tokens": output.reused_kv_tokens,
                                "output_token_ids_sha256": output_token_ids_sha256(
                                    output.token_ids
                                ),
                                "output_token_ids_hash_scheme": OUTPUT_TOKEN_HASH_SCHEME,
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
    usage: dict[str, int] | None = None,
    distribution_metrics: dict[str, float | str] | None = None,
) -> dict[str, Any]:
    delta: dict[str, str] = {}
    if role is not None:
        delta["role"] = role
    if text:
        delta["content"] = text
    payload: dict[str, Any] = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    if usage is not None:
        payload["usage"] = usage
    if distribution_metrics is not None:
        payload["distribution_metrics"] = distribution_metrics
    return payload


async def write_sse(response: web.StreamResponse, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    await response.write(b"data: " + encoded + b"\n\n")


def error_response(message: str, kind: str, status: int) -> web.Response:
    return web.json_response(
        {"error": {"message": message, "type": kind}},
        status=status,
    )


# A refused request costs the client one round trip; retrying into a queue that
# is still full costs everyone. One second is long enough for a decode slot to
# free at fleet speeds and short enough that a caller does not give up.
OVERLOAD_RETRY_AFTER_SECONDS = 1


def overloaded_response(error: QueueFullError) -> web.Response:
    response = web.json_response(
        {
            "error": {
                "message": str(error),
                "type": "overloaded",
                "pending": error.pending,
                "capacity": error.capacity,
            }
        },
        status=429,
    )
    response.headers["Retry-After"] = str(OVERLOAD_RETRY_AFTER_SECONDS)
    return response


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    arguments = list(sys.argv[1:] if argv is None else argv)
    reject_external_backend_arguments(arguments)
    parser = argparse.ArgumentParser(
        description="Mycellios API backed by the persistent native layer pipeline."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--model-artifact-identity")
    parser.add_argument("--stage-package-identity")
    parser.add_argument("--model-canonical-source")
    parser.add_argument("--model-canonical-revision")
    parser.add_argument("--pipeline-snapshot-identity", type=int)
    add_ram_backed_moe_arguments(parser)
    add_paged_kv_arguments(parser)
    add_native_gguf_arguments(parser)
    add_local_draft_model_arguments(parser)
    parser.add_argument(
        "--stage-executor-id",
        action="append",
        default=[],
        help=(
            "Ordered sealed executor id for root and each child; repeat once per "
            "stage when constructing a remote recovery contract."
        ),
    )
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
    parser.add_argument(
        "--device",
        default="auto",
        help=(
            "Dense Torch execution device: auto, cpu, cuda[:index], mps or xpu[:index]. "
            "An explicit accelerator request fails if it is unavailable."
        ),
    )
    add_dense_tiering_arguments(parser)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument(
        "--max-active-sequences",
        type=int,
        default=32,
        help=(
            "Sequences admitted into decode at once. Measured on separate GPUs over "
            "WAN: 8 caps aggregate "
            "throughput at roughly half of what the hardware sustains (52,7 vs 102,6 "
            "tok/s under load) and collapses under overload (18,8 tok/s, 177 errors); "
            "32 fixes both and also drains the queue faster at low load. 64 measured "
            "no better than 32, so 32 is the knee, not a ceiling to raise blindly."
        ),
    )
    parser.add_argument(
        "--max-pending-requests",
        type=int,
        default=None,
        help=(
            "Requests allowed to wait for a decode slot. Defaults to "
            f"{PENDING_PER_ACTIVE_SLOT}x --max-active-sequences, because a backlog "
            "must be sized against what the pipeline can actually drain, not set as "
            "a flat number: measured under overload with 8 active slots and a flat "
            "128-deep queue, every admitted request timed out and goodput fell to "
            "ZERO. Accepting "
            "work that provably cannot be served turns a slowdown into an outage."
        ),
    )
    parser.add_argument("--batch-window-ms", type=float, default=2.0)
    parser.add_argument(
        "--root-batch-window-ms",
        type=float,
        default=0.5,
        help="Coalesce ready root continuations into physical tensor batches.",
    )
    parser.add_argument(
        "--route-probe-interval-seconds",
        type=float,
        default=5.0,
        help="Measure full-route latency periodically; zero disables probes.",
    )
    parser.add_argument(
        "--route-probe-timeout-seconds",
        type=float,
        default=10.0,
        help="Fail a route whose no-compute probe does not return in time.",
    )
    parser.add_argument(
        "--prefill-chunk-tokens",
        type=int,
        default=0,
        help="Split long prompts into bounded pipeline waves; zero disables chunking.",
    )
    parser.add_argument(
        "--prefill-inflight-chunks",
        type=int,
        default=1,
        help="Maximum ordered prefill chunks allowed on the route at once.",
    )
    parser.add_argument(
        "--prefill-inflight-bytes",
        type=int,
        default=0,
        help=(
            "Per-request maximum encoded prefill wire bytes in flight; "
            "zero disables the byte cap."
        ),
    )
    parser.add_argument("--sealed-wave-tokens", type=int)
    parser.add_argument("--max-prefill-chunk-tokens", type=int)
    parser.add_argument(
        "--max-speculative-branches",
        type=int,
        default=0,
        help="Sealed concurrent exact KV children; zero disables physical trees.",
    )
    parser.add_argument(
        "--max-speculative-branch-tokens",
        type=int,
        default=0,
        help="Sealed absolute context-token ceiling for every physical KV child.",
    )
    parser.add_argument(
        "--max-speculative-kv-bytes",
        type=int,
        default=0,
        help="Sealed aggregate stage-local byte ceiling for child KV caches.",
    )
    parser.add_argument(
        "--speculation",
        choices=("off", "ngram", "draft-tree", "draft-model"),
        default="off",
        help=(
            "Enable exact adaptive speculative verification with the selected "
            "native Mycellios drafter."
        ),
    )
    parser.add_argument("--speculative-max-draft-tokens", type=int, default=4)
    parser.add_argument(
        "--speculative-inflight-waves",
        type=int,
        default=1,
        help=(
            "Maximum ordered linear VERIFY waves allowed on the route at once; "
            "one preserves the historical exact path."
        ),
    )
    parser.add_argument(
        "--speculative-inflight-bytes",
        type=int,
        default=0,
        help=(
            "Per-request maximum encoded VERIFY bytes in flight; zero is valid "
            "only with one historical in-flight wave."
        ),
    )
    parser.add_argument("--speculation-minimum-speedup", type=float, default=1.05)
    parser.add_argument("--no-speculation-probes", action="store_true")
    parser.add_argument(
        "--max-retained-sessions",
        type=int,
        default=4,
        help=(
            "Keep the KV of up to this many finished chats alive on every stage "
            "so the next turn (X-Session-Id header or legacy user field) only "
            "prefills the new suffix; zero disables session retention."
        ),
    )
    parser.add_argument(
        "--max-retained-session-tokens",
        type=int,
        default=8_192,
        help="Total idle KV tokens across retained sessions; zero is unbounded.",
    )
    parser.add_argument("--retained-session-ttl-seconds", type=float, default=600.0)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument(
        "--recovery-max-retries",
        type=int,
        default=0,
        help=(
            "Recreate a failed route and exactly recompute the visible greedy "
            "token prefix; zero disables recovery. This is a lightweight token "
            "checkpoint, not KV transfer."
        ),
    )
    parser.add_argument(
        "--recovery-standby-route",
        action="append",
        default=[],
        help=(
            "Repeatable sealed remote standby JSON using schema "
            "gdlp-recovery-standby-route/1 with routeId, firstStage "
            "{host,port}, and the complete ordered stageExecutorIds contract."
        ),
    )
    parser.add_argument("--first-stage-host")
    parser.add_argument("--first-stage-port", type=int)
    parser.add_argument("--return-bind-host", default="127.0.0.1")
    parser.add_argument("--return-advertise-host", default="127.0.0.1")
    parser.add_argument("--return-port", type=int, default=0)
    parser.add_argument("--startup-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--socket-timeout-seconds", type=float, default=180.0)
    parsed = parser.parse_args(arguments)
    if parsed.max_pending_requests is None:
        # Resolved here rather than at build time so every caller sees a complete
        # namespace; the backlog is tied to serving capacity, not to a constant.
        parsed.max_pending_requests = (
            PENDING_PER_ACTIVE_SLOT * parsed.max_active_sequences
        )
    return parsed


def build_server(args: argparse.Namespace) -> DistributedMycelliosServer:
    if (args.sealed_wave_tokens is None) != (
        args.max_prefill_chunk_tokens is None
    ):
        raise ValueError(
            "sealed-wave-tokens and max-prefill-chunk-tokens must be supplied together"
        )
    ram_backed_moe = ram_backed_moe_config_from_args(args)
    if (
        getattr(args, "paged_kv", False)
        and getattr(args, "paged_max_active_requests", None) is None
        and args.max_active_sequences > 0
        and args.max_speculative_branches >= 0
        and args.max_retained_sessions >= 0
    ):
        # The product CLI is automatic: when the operator does not seal a
        # stricter value, reserve every live, speculative and retained logical
        # request slot instead of inheriting the lower standalone-runner
        # default.
        args.paged_max_active_requests = (
            args.max_active_sequences
            + args.max_speculative_branches
            + args.max_retained_sessions
        )
    paged_kv = paged_kv_config_from_args(args)
    native_gguf = native_gguf_runtime_from_args(args)
    local_draft_model = local_draft_model_config_from_args(args)
    if sum(
        backend is not None
        for backend in (ram_backed_moe, paged_kv, native_gguf)
    ) > 1:
        raise ValueError(
            "native GGUF, paged KV and RAM-backed MoE backends are mutually exclusive"
        )
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
    if args.max_pending_requests is None:
        # Callers that build a namespace by hand (tests, embedders) never went
        # through parse_args, so the same default is applied here too.
        args.max_pending_requests = PENDING_PER_ACTIVE_SLOT * args.max_active_sequences
    if args.max_pending_requests < args.max_active_sequences:
        raise ValueError("max-pending-requests must be at least max-active-sequences")
    if not math.isfinite(args.batch_window_ms) or args.batch_window_ms < 0:
        raise ValueError("batch-window-ms must be finite and non-negative")
    if (
        not math.isfinite(args.root_batch_window_ms)
        or not 0 <= args.root_batch_window_ms <= 100
    ):
        raise ValueError("root-batch-window-ms must be between 0 and 100")
    if (
        not math.isfinite(args.route_probe_interval_seconds)
        or args.route_probe_interval_seconds < 0
    ):
        raise ValueError(
            "route-probe-interval-seconds must be finite and non-negative"
        )
    if (
        not math.isfinite(args.route_probe_timeout_seconds)
        or args.route_probe_timeout_seconds <= 0
    ):
        raise ValueError("route-probe-timeout-seconds must be finite and positive")
    if args.prefill_chunk_tokens < 0:
        raise ValueError("prefill-chunk-tokens must be non-negative")
    if not 1 <= args.prefill_inflight_chunks <= 64:
        raise ValueError("prefill-inflight-chunks must be between 1 and 64")
    if not 0 <= args.prefill_inflight_bytes <= 1024 * 1024 * 1024:
        raise ValueError("prefill-inflight-bytes must be between 0 and 1 GiB")
    for name, value, maximum in (
        (
            "max-speculative-branches",
            args.max_speculative_branches,
            MAX_SPECULATIVE_BRANCHES,
        ),
        (
            "max-speculative-branch-tokens",
            args.max_speculative_branch_tokens,
            MAX_SPECULATIVE_BRANCH_TOKENS,
        ),
        (
            "max-speculative-kv-bytes",
            args.max_speculative_kv_bytes,
            MAX_SPECULATIVE_KV_BYTES,
        ),
    ):
        if not 0 <= value <= maximum:
            raise ValueError(f"{name} must be between 0 and {maximum}")
    speculative_tree_limits_enabled = (
        args.max_speculative_branches > 0,
        args.max_speculative_branch_tokens > 0,
        args.max_speculative_kv_bytes > 0,
    )
    if any(speculative_tree_limits_enabled) and not all(
        speculative_tree_limits_enabled
    ):
        raise ValueError(
            "speculative branch count, tokens and KV bytes must all be zero "
            "or all be positive"
        )
    if args.speculation == "draft-tree" and not all(
        speculative_tree_limits_enabled
    ):
        raise ValueError(
            "draft-tree requires sealed positive branch count, branch-token "
            "and KV-byte limits"
        )
    if (
        args.speculation in ("draft-tree", "draft-model")
        and args.sealed_wave_tokens is None
    ):
        raise ValueError(
            f"{args.speculation} requires an explicit sealed-wave-tokens limit"
        )
    if args.sealed_wave_tokens is not None and not 1 <= args.sealed_wave_tokens <= 17:
        raise ValueError("sealed-wave-tokens must be between 1 and 17")
    if (
        args.max_prefill_chunk_tokens is not None
        and args.max_prefill_chunk_tokens < 1
    ):
        raise ValueError("max-prefill-chunk-tokens must be positive")
    if (
        args.max_prefill_chunk_tokens is not None
        and args.prefill_chunk_tokens > args.max_prefill_chunk_tokens
    ):
        raise ValueError(
            "prefill-chunk-tokens cannot exceed max-prefill-chunk-tokens"
        )
    if not 1 <= args.speculative_max_draft_tokens <= 16:
        raise ValueError("speculative-max-draft-tokens must be between 1 and 16")
    if not 1 <= args.speculative_inflight_waves <= MAX_SPECULATIVE_INFLIGHT_WAVES:
        raise ValueError(
            "speculative-inflight-waves must be between 1 and "
            f"{MAX_SPECULATIVE_INFLIGHT_WAVES}"
        )
    if not (
        0
        <= args.speculative_inflight_bytes
        <= MAX_SPECULATIVE_INFLIGHT_BYTES
    ):
        raise ValueError(
            "speculative-inflight-bytes must be between 0 and 1 GiB"
        )
    speculative_conveyor_enabled = (
        args.speculative_inflight_waves > 1
        or args.speculative_inflight_bytes > 0
    )
    if speculative_conveyor_enabled and (
        args.speculative_inflight_waves <= 1
        or args.speculative_inflight_bytes <= 0
    ):
        raise ValueError(
            "speculative conveyor requires more than one in-flight wave and "
            "a positive byte ceiling"
        )
    if speculative_conveyor_enabled and args.speculation not in (
        "ngram",
        "draft-model",
    ):
        raise ValueError(
            "speculative conveyor requires linear ngram or draft-model speculation"
        )
    if speculative_conveyor_enabled and any(speculative_tree_limits_enabled):
        raise ValueError(
            "speculative conveyor cannot be combined with physical "
            "speculative-tree limits"
        )
    if speculative_conveyor_enabled and args.max_active_sequences != 1:
        raise ValueError(
            "speculative conveyor currently requires max-active-sequences=1"
        )
    if args.sealed_wave_tokens is not None:
        required_wave_tokens = (
            args.speculative_max_draft_tokens + 1
            if args.speculation in ("ngram", "draft-tree", "draft-model")
            else 1
        )
        if args.sealed_wave_tokens < required_wave_tokens:
            raise ValueError(
                "sealed-wave-tokens cannot be smaller than the VERIFY input"
            )
        if (
            args.speculation in ("draft-tree", "draft-model")
            and args.sealed_wave_tokens != required_wave_tokens
        ):
            raise ValueError(
                f"{args.speculation} sealed-wave-tokens must equal draft depth plus one"
            )
        if args.speculation == "off" and args.sealed_wave_tokens != 1:
            raise ValueError(
                "sealed-wave-tokens greater than one require speculation"
            )
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
    if paged_kv is not None:
        required_request_slots = (
            args.max_active_sequences
            + args.max_speculative_branches
            + args.max_retained_sessions
        )
        if paged_kv.max_active_requests < required_request_slots:
            raise ValueError(
                "paged max-active-requests cannot hold all active sequences, "
                "sealed speculative branches and retained sessions"
            )
        if (
            args.max_speculative_branch_tokens > 0
            and paged_kv.max_sequence_tokens
            < args.max_speculative_branch_tokens
        ):
            raise ValueError(
                "paged max-sequence-tokens is smaller than the sealed "
                "speculative branch token ceiling"
            )
    if args.recovery_max_retries < 0:
        raise ValueError("recovery-max-retries must be non-negative")
    standby_routes = parse_remote_recovery_standby_routes(
        getattr(args, "recovery_standby_route", ())
    )
    remote_requested = (
        args.first_stage_host is not None or args.first_stage_port is not None
    )
    if standby_routes and args.recovery_max_retries == 0:
        raise ValueError(
            "recovery standby routes require a positive recovery-max-retries"
        )
    if standby_routes and not remote_requested:
        raise ValueError("recovery standby routes require a remote primary route")
    if remote_requested and args.recovery_max_retries > 0:
        if not getattr(args, "stage_executor_id", ()):
            raise ValueError(
                "remote recovery requires the complete ordered stage-executor-id contract"
            )
        if not standby_routes:
            raise ValueError(
                "remote recovery requires at least one sealed recovery-standby-route"
            )
        primary_executor_ids = tuple(args.stage_executor_id)
        for route in standby_routes:
            if route.stage_executor_ids != primary_executor_ids:
                raise ValueError(
                    f"recovery standby {route.route_id!r} executor contract "
                    "does not match the configured active route"
                )
            if (
                args.first_stage_host is not None
                and args.first_stage_port is not None
                and route.first_stage_host.casefold()
                == args.first_stage_host.casefold()
                and route.first_stage_port == args.first_stage_port
            ):
                raise ValueError(
                    f"recovery standby {route.route_id!r} reuses the primary endpoint"
                )
    remote = remote_requested
    if remote and (args.first_stage_host is None or args.first_stage_port is None):
        raise ValueError("remote mode requires both first-stage-host and first-stage-port")
    if ram_backed_moe is not None and not remote:
        raise ValueError(
            "server CLI RAM-backed MoE requires remote child stages with their "
            "own sealed bindings"
        )
    if native_gguf is not None and not remote:
        raise ValueError(
            "server CLI native GGUF requires remote child stages with their "
            "own authenticated Mycellios packages"
        )
    native_package = (
        None
        if native_gguf is None
        else verify_native_gguf_stage(
            native_gguf.package,
            expected_package_id=native_gguf.package_id,
        )
    )
    if native_package is not None:
        if (
            args.revision is not None
            and args.revision != native_package.model_revision
        ):
            raise ValueError(
                "revision differs from the authenticated native GGUF package"
            )
        # `model` is retained as the tokenizer coordinate only. Executable
        # config and weights are loaded from the authenticated Mycellios package.
        snapshot = args.model
        model_config = AutoConfig.from_pretrained(
            native_package.root,
            local_files_only=True,
            trust_remote_code=False,
        )
    elif ram_backed_moe is not None:
        snapshot = str(Path(args.model).expanduser().resolve())
        model_config = AutoConfig.from_pretrained(
            snapshot,
            local_files_only=True,
            trust_remote_code=False,
        )
    else:
        snapshot = resolve_model_snapshot(args.model, args.revision)
        model_config = AutoConfig.from_pretrained(snapshot)
    total_layers = int(model_config.num_hidden_layers)
    boundaries = (
        parse_boundaries(args.boundaries, total_layers)
        if args.boundaries
        else balanced_boundaries(total_layers, args.stages)
    )
    tokenizer = load_tokenizer(snapshot)
    if native_package is not None:
        verify_native_gguf_stage(
            native_package.root,
            expected_package_id=native_package.package_id,
            expected_layer_start=boundaries[0],
            expected_layer_end=boundaries[1],
            expected_total_layers=total_layers,
        )
    codec = {
        "fp32": TensorCodec.FP32,
        "fp16": TensorCodec.FP16,
        "int8": TensorCodec.INT8,
        "int8-grouped": TensorCodec.INT8_GROUPED,
        "int8-hadamard": TensorCodec.INT8_HADAMARD,
        "int8-grouped-deflate": TensorCodec.INT8_GROUPED_DEFLATE,
        "int8-hadamard-deflate": TensorCodec.INT8_HADAMARD_DEFLATE,
    }[args.codec]
    engine_config = PipelineEngineConfig(
            model_name=snapshot,
            boundaries=boundaries,
            codec=codec,
            threads_per_stage=args.threads_per_stage,
            device=args.device,
            dense_tiering=dense_tiering_config_from_args(args),
            startup_timeout_seconds=args.startup_timeout_seconds,
            socket_timeout_seconds=args.socket_timeout_seconds,
            spawn_local_stages=not remote,
            first_stage_host=args.first_stage_host or "127.0.0.1",
            first_stage_port=args.first_stage_port,
            return_bind_host=args.return_bind_host,
            return_advertise_host=args.return_advertise_host,
            return_port=args.return_port,
            artifact_identity=args.model_artifact_identity,
            stage_package_identity=args.stage_package_identity,
            canonical_model_source=args.model_canonical_source,
            canonical_model_revision=args.model_canonical_revision,
            pipeline_snapshot_identity=args.pipeline_snapshot_identity,
            ram_backed_moe_stages=(
                None
                if ram_backed_moe is None
                else (ram_backed_moe,) + (None,) * (len(boundaries) - 2)
            ),
            paged_kv_stages=(
                None
                if paged_kv is None
                else (
                    ((paged_kv,) * (len(boundaries) - 1))
                    if not remote
                    else (paged_kv,) + (None,) * (len(boundaries) - 2)
                )
            ),
            native_gguf_stages=(
                None
                if native_gguf is None
                else (native_gguf,) + (None,) * (len(boundaries) - 2)
            ),
            stage_executor_ids=(
                tuple(args.stage_executor_id) if args.stage_executor_id else None
            ),
            max_active_sequences=args.max_active_sequences,
            max_pending_requests=args.max_pending_requests,
            prefill_chunk_tokens=args.prefill_chunk_tokens,
            prefill_inflight_chunks=args.prefill_inflight_chunks,
            prefill_inflight_bytes=args.prefill_inflight_bytes,
            max_speculative_branches=args.max_speculative_branches,
            max_speculative_branch_tokens=args.max_speculative_branch_tokens,
            max_speculative_kv_bytes=args.max_speculative_kv_bytes,
            sealed_wave_tokens=args.sealed_wave_tokens,
            max_prefill_chunk_tokens=args.max_prefill_chunk_tokens,
            speculative_max_draft_tokens=(
                args.speculative_max_draft_tokens
                if args.speculation in ("ngram", "draft-tree", "draft-model")
                else 0
            ),
            speculative_inflight_waves=args.speculative_inflight_waves,
            speculative_inflight_bytes=args.speculative_inflight_bytes,
            speculation_minimum_speedup=args.speculation_minimum_speedup,
            speculation_probe=not args.no_speculation_probes,
            root_batch_window_ms=args.root_batch_window_ms,
            route_probe_interval_seconds=args.route_probe_interval_seconds,
            route_probe_timeout_seconds=args.route_probe_timeout_seconds,
            max_retained_sessions=args.max_retained_sessions,
            max_retained_session_tokens=args.max_retained_session_tokens,
            retained_session_ttl_seconds=args.retained_session_ttl_seconds,
    )
    draft_provider = (
        None
        if local_draft_model is None
        else load_local_draft_model_provider(
            local_draft_model,
            target_tokenizer=tokenizer,
            max_cached_requests=args.max_active_sequences,
        )
    )

    def make_engine(config: PipelineEngineConfig = engine_config) -> DistributedPipelineEngine:
        tree_provider = tree_draft_provider_from_args(args)
        return DistributedPipelineEngine(
            config,
            draft_provider=draft_provider,
            tree_draft_provider=tree_provider,
        )

    engine_factory = make_engine
    initial_engine = engine_factory()
    engine: DistributedPipelineEngine | RecoveringPipelineEngine
    if args.recovery_max_retries > 0:
        expected_executor_ids = initial_engine.recovery_identity.stage_executor_ids
        standby_factories: list[RemoteRecoveryStandbyEngineFactory] = []
        for route in standby_routes:
            if len(route.stage_executor_ids) != len(boundaries) - 1:
                initial_engine.close()
                raise ValueError(
                    f"recovery standby {route.route_id!r} must contain one "
                    "stageExecutorId per stage"
                )
            if route.stage_executor_ids != expected_executor_ids:
                initial_engine.close()
                raise ValueError(
                    f"recovery standby {route.route_id!r} executor contract "
                    "does not match the active route"
                )
            standby_config = replace(
                engine_config,
                first_stage_host=route.first_stage_host,
                first_stage_port=route.first_stage_port,
                stage_executor_ids=route.stage_executor_ids,
            )
            standby_factories.append(
                RemoteRecoveryStandbyEngineFactory(
                    route=route,
                    engine_factory=(
                        lambda config=standby_config: make_engine(config)
                    ),
                )
            )
        engine = RecoveringPipelineEngine(
            engine_factory,
            max_retries=args.recovery_max_retries,
            initial_engine=initial_engine,
            standby_factories=standby_factories,
        )
    else:
        engine = initial_engine
    try:
        return DistributedMycelliosServer(
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


def parse_remote_recovery_standby_routes(
    values: Any,
) -> tuple[RemoteRecoveryStandbyRoute, ...]:
    """Parse and deduplicate repeatable CLI standby route contracts."""

    if values is None:
        return ()
    if not isinstance(values, (list, tuple)):
        raise ValueError("recovery standby routes must be a list")
    routes = tuple(RemoteRecoveryStandbyRoute.parse_json(value) for value in values)
    route_ids = [route.route_id for route in routes]
    if len(route_ids) != len(set(route_ids)):
        raise ValueError("recovery standby routeId values must be unique")
    endpoints = [
        (route.first_stage_host.casefold(), route.first_stage_port)
        for route in routes
    ]
    if len(endpoints) != len(set(endpoints)):
        raise ValueError("recovery standby first-stage endpoints must be unique")
    return routes


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
                "root_batch_window_ms": server.engine.config.root_batch_window_ms,
                "execution": server.engine.execution_topology,
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
