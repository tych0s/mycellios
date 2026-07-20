from __future__ import annotations

from concurrent.futures import CancelledError as FutureCancelledError, Future
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
import math
import multiprocessing as mp
import queue
import socket
import threading
import time
from typing import Any, Callable

import torch
from transformers import AutoConfig

from .model import (
    StageModelSpec,
    StageRunner,
    model_snapshot_identity,
    resolve_model_snapshot,
)
from .protocol import (
    HEADER_BYTES,
    FrameType,
    LinkEmulator,
    TensorCodec,
    configure_socket,
    decode_token,
    decode_verify_result,
    encode_tensor,
    recv_frame,
    send_frame,
)
from .speculation import (
    MAX_DRAFT_TOKENS,
    AdaptiveSpeculationConfig,
    AdaptiveSpeculationController,
    DraftProvider,
    NgramDraftProvider,
)
from .stage import StageProcessConfig, connect_with_retry, drain_metrics, run_stage_process


TokenCallback = Callable[[int, int, int, float], None]


@dataclass(frozen=True)
class PipelineEngineConfig:
    model_name: str
    boundaries: tuple[int, ...]
    codec: TensorCodec = TensorCodec.FP16
    threads_per_stage: int = 1
    startup_timeout_seconds: float = 180.0
    socket_timeout_seconds: float = 180.0
    one_way_delay_ms: float = 0.0
    bandwidth_mbps: float = 0.0
    spawn_local_stages: bool = True
    first_stage_host: str = "127.0.0.1"
    first_stage_port: int | None = None
    return_bind_host: str = "127.0.0.1"
    return_advertise_host: str = "127.0.0.1"
    return_port: int = 0
    revision: str | None = None
    max_active_sequences: int = 8
    max_pending_requests: int = 128
    # Zero keeps the whole prompt in one physical prefill wave.  A positive
    # value sends bounded chunks and lets decode work enter between their ACKs.
    prefill_chunk_tokens: int = 0
    # Zero is exact autoregressive decode. Positive values enable exact target
    # verification with an adaptive n-gram draft source by default.
    speculative_max_draft_tokens: int = 0
    speculation_minimum_speedup: float = 1.05
    speculation_probe: bool = True
    # Zero keeps today's behavior: every finished generation sends END and the
    # per-request KV cache is dropped on every stage. A positive value keeps up
    # to that many finished sessions alive (same wire_id, KV intact on every
    # stage) so the next turn of the same chat only prefills the new suffix.
    # Retained sessions hold KV memory on every stage but never occupy one of
    # the max_active_sequences decode slots while idle.
    max_retained_sessions: int = 0
    # Total idle KV tokens allowed across retained sessions; zero is unbounded.
    max_retained_session_tokens: int = 0
    retained_session_ttl_seconds: float = 600.0

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError("model_name cannot be empty")
        if len(self.boundaries) < 3:
            raise ValueError("a distributed pipeline requires at least two stages")
        if self.boundaries[0] != 0:
            raise ValueError("boundaries must start at zero")
        if any(right <= left for left, right in zip(self.boundaries, self.boundaries[1:])):
            raise ValueError("boundaries must be strictly increasing")
        if self.threads_per_stage < 1:
            raise ValueError("threads_per_stage must be positive")
        if self.max_active_sequences < 1:
            raise ValueError("max_active_sequences must be positive")
        if self.max_pending_requests < self.max_active_sequences:
            raise ValueError("max_pending_requests must be at least max_active_sequences")
        if (
            not isinstance(self.prefill_chunk_tokens, int)
            or isinstance(self.prefill_chunk_tokens, bool)
            or self.prefill_chunk_tokens < 0
        ):
            raise ValueError("prefill_chunk_tokens must be a non-negative integer")
        if (
            not isinstance(self.speculative_max_draft_tokens, int)
            or isinstance(self.speculative_max_draft_tokens, bool)
            or not 0 <= self.speculative_max_draft_tokens <= MAX_DRAFT_TOKENS
        ):
            raise ValueError(
                f"speculative_max_draft_tokens must be between 0 and {MAX_DRAFT_TOKENS}"
            )
        if (
            not math.isfinite(self.speculation_minimum_speedup)
            or self.speculation_minimum_speedup < 1.0
        ):
            raise ValueError("speculation_minimum_speedup must be finite and at least 1")
        if not isinstance(self.speculation_probe, bool):
            raise TypeError("speculation_probe must be boolean")
        for name, value in (
            ("max_retained_sessions", self.max_retained_sessions),
            ("max_retained_session_tokens", self.max_retained_session_tokens),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if (
            not math.isfinite(self.retained_session_ttl_seconds)
            or self.retained_session_ttl_seconds <= 0
        ):
            raise ValueError("retained_session_ttl_seconds must be finite and positive")
        for name, value in (
            ("startup_timeout_seconds", self.startup_timeout_seconds),
            ("socket_timeout_seconds", self.socket_timeout_seconds),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name, value in (
            ("one_way_delay_ms", self.one_way_delay_ms),
            ("bandwidth_mbps", self.bandwidth_mbps),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not self.first_stage_host.strip():
            raise ValueError("first_stage_host cannot be empty")
        if not self.return_bind_host.strip() or not self.return_advertise_host.strip():
            raise ValueError("return hosts cannot be empty")
        if self.spawn_local_stages:
            if self.first_stage_port is not None:
                raise ValueError("first_stage_port is allocated automatically for local stages")
        elif self.first_stage_port is None:
            raise ValueError("first_stage_port is required for remote stages")
        for name, port in (
            ("first_stage_port", self.first_stage_port),
            ("return_port", self.return_port),
        ):
            if port is not None and (not isinstance(port, int) or isinstance(port, bool) or not 0 <= port <= 65_535):
                raise ValueError(f"{name} must be between 0 and 65535")
        if not self.spawn_local_stages and self.first_stage_port == 0:
            raise ValueError("remote stages require a non-zero first_stage_port")
        if not self.spawn_local_stages and self.return_port == 0:
            raise ValueError("remote stages require a fixed, advertised return_port")


@dataclass(frozen=True)
class GenerationInput:
    client_id: int
    input_ids: torch.Tensor
    max_new_tokens: int
    eos_token_ids: frozenset[int] = frozenset()
    # Opaque chat/session identity. With session retention enabled, input_ids
    # must contain the COMPLETE tokenized conversation; the engine reuses the
    # live KV prefix it already served for this key and prefills only the rest.
    session_key: str | None = None

    def __post_init__(self) -> None:
        if self.client_id < 0:
            raise ValueError("client_id must be non-negative")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if (
            self.input_ids.ndim != 2
            or self.input_ids.shape[0] != 1
            or self.input_ids.shape[1] < 1
            or self.input_ids.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("input_ids must be an integer tensor with shape [1, tokens]")
        if self.session_key is not None and (
            not isinstance(self.session_key, str)
            or not self.session_key.strip()
            or len(self.session_key) > 256
        ):
            raise ValueError(
                "session_key must be a non-empty string of at most 256 characters"
            )


@dataclass(frozen=True)
class GenerationOutput:
    client_id: int
    token_ids: tuple[int, ...]
    finish_reason: str
    ttft_ms: float
    tpot_ms: float
    total_ms: float
    # KV tokens served from a retained session instead of being re-prefilled.
    reused_kv_tokens: int = 0


class GenerationCancelledError(RuntimeError):
    pass


@dataclass
class _GenerationJob:
    request: GenerationInput
    callback: TokenCallback | None
    future: Future[GenerationOutput] = field(default_factory=Future)
    cancel_requested: threading.Event = field(default_factory=threading.Event)
    wire_id: int | None = None
    step: int = 0
    started_at: float = 0.0
    last_sent_at: float = 0.0
    token_ids: list[int] = field(default_factory=list)
    arrivals: list[float] = field(default_factory=list)
    cancel_sent: bool = False
    prefill_offset: int = 0
    wave_started_at: float = 0.0
    last_outbound_bytes: int = 0
    verify_drafts: tuple[int, ...] = ()
    verify_base_tokens: int = 0
    speculation_profile: str = "load-1"
    # Retained session this job checked out at admission (None for fresh wires).
    session: "_RetainedSession | None" = None
    reused_tokens: int = 0
    # Number of leading KV positions on every stage that are known to equal the
    # exact served sequence (prompt + emitted tokens). The physical KV may be
    # longer when a turn ends right after a partially rejected VERIFY wave.
    kv_valid: int = 0


@dataclass
class _RetainedSession:
    """A finished chat whose wire and KV stay alive on every stage.

    The wire was opened with BEGIN once and never received END. ``served_ids``
    is the exact token sequence served for the last turn (full prompt plus the
    emitted completion); ``kv_valid_tokens`` bounds the prefix of it that the
    physical KV provably encodes. Only the scheduler thread touches instances.
    """

    key: str
    wire_id: int
    step: int
    served_ids: tuple[int, ...]
    kv_tokens: int
    kv_valid_tokens: int
    last_used: float
    busy: bool = False


class DistributedPipelineEngine:
    """Long-lived root stage for a contiguous layer pipeline.

    One scheduler thread owns the root model and the pipeline writer. Calls to
    ``generate`` may overlap: new requests are admitted between decode rounds and
    every request advances with its own wire step and KV cache.
    """

    def __init__(
        self,
        config: PipelineEngineConfig,
        *,
        draft_provider: DraftProvider | None = None,
        speculation_controller: AdaptiveSpeculationController | None = None,
    ) -> None:
        self.config = config
        self._state_lock = threading.Lock()
        self._closed = False
        self._fatal_error: str | None = None
        self._request_counter = 0
        self._processes: list[Any] = []
        self._metrics_queue: Any | None = None
        self._runner: StageRunner | None = None
        self._downstream: socket.socket | None = None
        self._return_socket: socket.socket | None = None
        self._return_listener: socket.socket | None = None
        self._shutdown_sent = False
        self._received_frames: queue.Queue[tuple[Any, float] | BaseException] = queue.Queue()
        self._callback_routes: dict[int, _GenerationJob] = {}
        self._callback_lock = threading.Lock()
        self._receiver_thread: threading.Thread | None = None
        self._control_thread: threading.Thread | None = None
        self._submission_queue: queue.Queue[list[_GenerationJob] | None] = queue.Queue()
        self._deferred_batches: deque[list[_GenerationJob]] = deque()
        self._jobs_by_client: dict[int, _GenerationJob] = {}
        self._scheduler_stop = threading.Event()
        self._scheduler_thread: threading.Thread | None = None
        self._speculation_lock = threading.Lock()
        self._speculation_enabled_decisions = 0
        self._speculation_disabled_decisions = 0
        self._speculation_probe_waves = 0
        self._speculation_decision_reasons: dict[str, int] = {}
        self._speculation_selected_sizes: dict[int, int] = {}
        # Session retention state. The map is owned by the scheduler thread;
        # insertion order doubles as the LRU order because every retention
        # refresh re-inserts the entry. The integer counters are read without a
        # lock by observability endpoints, which is safe for CPython int reads.
        self._retained_sessions: dict[str, _RetainedSession] = {}
        self._session_retained_tokens = 0
        self._session_reuse_hits = 0
        self._session_reuse_misses = 0
        self._session_busy_fallbacks = 0
        self._session_divergence_fallbacks = 0
        self._session_ttl_evictions = 0
        self._session_budget_evictions = 0
        self._session_cancel_releases = 0
        self._session_reused_token_total = 0
        self.stage_metrics: list[dict[str, Any]] = []
        if config.speculative_max_draft_tokens > 0:
            self.draft_provider: DraftProvider | None = draft_provider or NgramDraftProvider(
                max_draft_tokens=config.speculative_max_draft_tokens
            )
            self.speculation_controller: AdaptiveSpeculationController | None = (
                speculation_controller
                or AdaptiveSpeculationController(
                    AdaptiveSpeculationConfig(
                        max_draft_tokens=config.speculative_max_draft_tokens,
                        minimum_speedup=config.speculation_minimum_speedup,
                    )
                )
            )
            self._speculation_controllers: dict[
                str, AdaptiveSpeculationController
            ] = {}
        else:
            if draft_provider is not None or speculation_controller is not None:
                raise ValueError(
                    "draft provider/controller require speculative_max_draft_tokens > 0"
                )
            self.draft_provider = None
            self.speculation_controller = None
            self._speculation_controllers = {}

        model_config = AutoConfig.from_pretrained(config.model_name, revision=config.revision)
        self.total_layers = int(model_config.num_hidden_layers)
        self.hidden_size = int(model_config.hidden_size)
        if config.boundaries[-1] != self.total_layers:
            raise ValueError(
                f"boundaries end at {config.boundaries[-1]}, model has {self.total_layers} layers"
            )
        self.maximum_context = int(getattr(model_config, "max_position_embeddings", 0) or 0)
        self.model_snapshot = resolve_model_snapshot(config.model_name, config.revision)
        self.pipeline_id = model_snapshot_identity(self.model_snapshot)
        try:
            self._start()
        except BaseException:
            self.close()
            raise

    @property
    def stages(self) -> int:
        return len(self.config.boundaries) - 1

    @property
    def root_parameter_bytes(self) -> int:
        return self._require_runner().parameter_bytes

    @property
    def healthy(self) -> bool:
        with self._state_lock:
            return not self._closed and self._fatal_error is None

    @property
    def fatal_error(self) -> str | None:
        with self._state_lock:
            return self._fatal_error

    @property
    def session_stats(self) -> dict[str, Any]:
        return {
            "configured": self.config.max_retained_sessions > 0,
            "max_retained_sessions": self.config.max_retained_sessions,
            "max_retained_session_tokens": self.config.max_retained_session_tokens,
            "ttl_seconds": self.config.retained_session_ttl_seconds,
            "retained_sessions": len(self._retained_sessions),
            "retained_tokens": self._session_retained_tokens,
            "reuse_hits": self._session_reuse_hits,
            "reuse_misses": self._session_reuse_misses,
            "busy_fallbacks": self._session_busy_fallbacks,
            "divergence_fallbacks": self._session_divergence_fallbacks,
            "ttl_evictions": self._session_ttl_evictions,
            "budget_evictions": self._session_budget_evictions,
            "cancel_releases": self._session_cancel_releases,
            "reused_tokens_total": self._session_reused_token_total,
        }

    @property
    def speculation_stats(self) -> dict[str, Any]:
        controller = self.speculation_controller
        if controller is None:
            return {"configured": False, "enabled": False}
        with self._speculation_lock:
            controllers = (
                tuple(sorted(self._speculation_controllers.items()))
                if self._speculation_controllers
                else (("unobserved", controller),)
            )
            observations = [current.stats() for _, current in controllers]
            classic_observations = sum(
                value.classic_observations for value in observations
            )
            verification_observations = sum(
                value.verification_observations for value in observations
            )
            proposed_tokens = sum(value.proposed_tokens for value in observations)
            accepted_tokens = sum(value.accepted_tokens for value in observations)
            acceptance_rate = (
                accepted_tokens / proposed_tokens if proposed_tokens else None
            )
            return {
                # Keep enabled as a compatibility alias for older health clients;
                # actual use is exposed separately by enabled_decisions.
                "configured": True,
                "enabled": True,
                "enabled_decisions": self._speculation_enabled_decisions,
                "disabled_decisions": self._speculation_disabled_decisions,
                "probe_waves": self._speculation_probe_waves,
                "decision_reasons": dict(self._speculation_decision_reasons),
                "selected_candidate_sizes": {
                    str(size): count
                    for size, count in sorted(self._speculation_selected_sizes.items())
                },
                "classic_observations": classic_observations,
                "verification_observations": verification_observations,
                "proposed_tokens": proposed_tokens,
                "accepted_tokens": accepted_tokens,
                "acceptance_rate": acceptance_rate,
                "verification_bytes": sum(
                    value.verification_bytes for value in observations
                ),
                "profiles": {
                    profile: {
                        "classic_observations": current.stats().classic_observations,
                        "verification_observations": current.stats().verification_observations,
                        "candidates": [
                            {
                                "size": estimate.candidate_size,
                                "ready": estimate.ready,
                                "observations": estimate.observations,
                                "acceptance_rate": estimate.acceptance_rate,
                                "predicted_speedup": _finite_or_none(
                                    estimate.predicted_speedup
                                ),
                                "predicted_latency_speedup": _finite_or_none(
                                    estimate.predicted_latency_speedup
                                ),
                                "predicted_byte_efficiency": _finite_or_none(
                                    estimate.predicted_byte_efficiency
                                ),
                            }
                            for estimate in current.candidate_estimates()
                        ],
                    }
                    for profile, current in controllers
                },
            }

    def generate(
        self,
        requests: list[GenerationInput],
        on_token: TokenCallback | None = None,
    ) -> list[GenerationOutput]:
        futures = self.submit(requests, on_token)
        outputs: list[GenerationOutput] = []
        first_cancelled: GenerationCancelledError | None = None
        for request, future in zip(requests, futures):
            try:
                outputs.append(future.result())
            except FutureCancelledError:
                first_cancelled = first_cancelled or GenerationCancelledError(
                    f"generation {request.client_id} was cancelled"
                )
        if first_cancelled is not None:
            raise first_cancelled
        return outputs

    def submit(
        self,
        requests: list[GenerationInput],
        on_token: TokenCallback | None = None,
    ) -> list[Future[GenerationOutput]]:
        if not requests:
            return []
        if len({request.client_id for request in requests}) != len(requests):
            raise ValueError("client_id values must be unique within a batch")
        for request in requests:
            if self.maximum_context and request.input_ids.shape[1] + request.max_new_tokens > self.maximum_context:
                raise ValueError(
                    f"request {request.client_id} exceeds model context {self.maximum_context}"
                )
        jobs = [_GenerationJob(request=request, callback=on_token) for request in requests]
        with self._state_lock:
            if self._closed:
                raise RuntimeError("pipeline engine is closed")
            if self._fatal_error is not None:
                raise RuntimeError(f"pipeline engine is degraded: {self._fatal_error}")
            duplicate_active = [
                job.request.client_id
                for job in jobs
                if job.request.client_id in self._jobs_by_client
            ]
            if duplicate_active:
                raise ValueError(f"client_id already active: {duplicate_active[0]}")
            if len(self._jobs_by_client) + len(jobs) > self.config.max_pending_requests:
                raise RuntimeError("pipeline request queue is full")
            for job in jobs:
                self._jobs_by_client[job.request.client_id] = job
        self._submission_queue.put(jobs)
        return [job.future for job in jobs]

    def cancel(self, client_id: int) -> bool:
        with self._state_lock:
            job = self._jobs_by_client.get(client_id)
            if job is None:
                return False
            job.cancel_requested.set()
        return True

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
        self._scheduler_stop.set()
        self._submission_queue.put(None)
        if self._scheduler_thread is not None:
            self._scheduler_thread.join(timeout=1.0)
        if self._scheduler_thread is not None and self._scheduler_thread.is_alive():
            # Network shutdown is deliberately outside the state lock so it can
            # interrupt a scheduler waiting on a failed stage.
            self._interrupt_transports()
            self._received_frames.put(RuntimeError("pipeline closed"))
            self._scheduler_thread.join(timeout=5.0)
        self._close_transports()
        for thread in (self._receiver_thread, self._control_thread):
            if thread is not None:
                thread.join(timeout=5.0)
        for process in self._processes:
            if process.is_alive():
                process.join(timeout=2.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
        if self._metrics_queue is not None:
            self.stage_metrics.extend(drain_metrics(self._metrics_queue))
        if self._scheduler_thread is None or not self._scheduler_thread.is_alive():
            self._runner = None

    def __enter__(self) -> "DistributedPipelineEngine":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _start(self) -> None:
        config = self.config
        boundaries = config.boundaries
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((config.return_bind_host, config.return_port))
        listener.listen(1)
        listener.settimeout(config.socket_timeout_seconds)
        self._return_listener = listener
        actual_return_port = int(listener.getsockname()[1])

        first_stage_host = config.first_stage_host
        first_stage_port = config.first_stage_port
        if config.spawn_local_stages:
            first_stage_host = "127.0.0.1"
            listen_ports = [_reserve_port() for _ in range(self.stages - 1)]
            first_stage_port = listen_ports[0]
            context = mp.get_context("spawn")
            self._metrics_queue = context.Queue()
            child_configs: list[StageProcessConfig] = []
            for child_index in range(self.stages - 1):
                stage_index = child_index + 1
                has_next = stage_index + 1 < self.stages
                child_configs.append(
                    StageProcessConfig(
                        spec=StageModelSpec(
                            self.model_snapshot,
                            boundaries[stage_index],
                            boundaries[stage_index + 1],
                            self.total_layers,
                            config.threads_per_stage,
                        ),
                        pipeline_id=self.pipeline_id,
                        listen_host="127.0.0.1",
                        listen_port=listen_ports[child_index],
                        next_host="127.0.0.1" if has_next else None,
                        next_port=listen_ports[child_index + 1] if has_next else None,
                        next_layer_end=boundaries[stage_index + 2] if has_next else None,
                        return_host=config.return_advertise_host,
                        return_port=actual_return_port,
                        codec=config.codec,
                        one_way_delay_ms=config.one_way_delay_ms,
                        bandwidth_mbps=config.bandwidth_mbps,
                        connect_timeout_seconds=config.startup_timeout_seconds,
                    )
                )
            for child_config in reversed(child_configs):
                ready_event = context.Event()
                process = context.Process(
                    target=run_stage_process,
                    args=(child_config, ready_event, self._metrics_queue),
                    name=f"distribution-stage-{child_config.spec.layer_start}",
                )
                process.start()
                self._processes.append(process)
                _wait_until_listening(process, ready_event, config.startup_timeout_seconds)

        if first_stage_port is None:
            raise RuntimeError("first stage port was not resolved")
        self._runner = StageRunner(
            StageModelSpec(
                self.model_snapshot,
                0,
                boundaries[1],
                self.total_layers,
                config.threads_per_stage,
            )
        )
        if self._runner.hidden_size != self.hidden_size:
            raise RuntimeError("root stage hidden size differs from the model configuration")

        downstream = connect_with_retry(
            first_stage_host,
            first_stage_port,
            config.startup_timeout_seconds,
        )
        downstream.settimeout(config.socket_timeout_seconds)
        self._downstream = downstream
        send_frame(
            downstream,
            FrameType.HELLO,
            self.pipeline_id,
            step=boundaries[1],
            token_count=boundaries[2],
            hidden_size=self.hidden_size,
            flags=int(config.codec),
        )
        ready = recv_frame(downstream)
        if ready.frame_type == FrameType.ERROR:
            raise RuntimeError(ready.payload.decode("utf-8", errors="replace"))
        if ready.frame_type != FrameType.READY:
            raise RuntimeError("pipeline did not return READY")
        if ready.request_id != self.pipeline_id:
            raise RuntimeError("pipeline READY has a different model identity")
        downstream.settimeout(None)
        return_socket, _ = listener.accept()
        configure_socket(return_socket)
        # A model server may sit idle indefinitely. A dedicated reader keeps the
        # direct-return path drained and exposes first tokens while the root stage
        # is still preparing later requests in the same microbatch.
        return_socket.settimeout(None)
        self._return_socket = return_socket
        self._receiver_thread = threading.Thread(
            target=self._receive_loop,
            name="distribution-token-return",
            daemon=True,
        )
        self._receiver_thread.start()
        self._control_thread = threading.Thread(
            target=self._control_receive_loop,
            name="distribution-upstream-control",
            daemon=True,
        )
        self._control_thread.start()
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            name="distribution-continuous-scheduler",
            daemon=True,
        )
        self._scheduler_thread.start()

    def _scheduler_loop(self) -> None:
        runner = self._require_runner()
        downstream = self._require_socket(self._downstream, "downstream")
        emulator = LinkEmulator(
            self.config.one_way_delay_ms,
            self.config.bandwidth_mbps,
        )
        active: dict[int, _GenerationJob] = {}
        decode_since_admission = 0
        fatal: BaseException | None = None
        try:
            while not self._scheduler_stop.is_set():
                self._expire_retained_sessions(runner, downstream)
                if not active:
                    try:
                        idle_value = self._received_frames.get_nowait()
                    except queue.Empty:
                        idle_value = None
                    if idle_value is not None:
                        self._raise_idle_return(idle_value)
                    batch = self._next_submission(timeout=0.1)
                    if batch is None:
                        continue
                    self._admit_batch(batch, active, runner, downstream, emulator)
                    decode_since_admission = 0
                    continue

                self._send_requested_cancellations(active, downstream)
                if (
                    len(active) < self.config.max_active_sequences
                    and decode_since_admission >= max(1, len(active))
                ):
                    batch = self._next_submission(timeout=0.0)
                    if batch is not None:
                        self._admit_batch(batch, active, runner, downstream, emulator)
                        decode_since_admission = 0
                        continue

                self._check_pipeline_timeouts(active)
                try:
                    value = self._received_frames.get(timeout=0.05)
                except queue.Empty:
                    # An idle return channel is also an admission point. This lets a
                    # new chat enter while another request is between network hops.
                    batch = (
                        self._next_submission(timeout=0.0)
                        if len(active) < self.config.max_active_sequences
                        else None
                    )
                    if batch is not None:
                        self._admit_batch(batch, active, runner, downstream, emulator)
                        decode_since_admission = 0
                    continue
                self._handle_return_value(
                    value,
                    active,
                    runner,
                    downstream,
                    emulator,
                )
                decode_since_admission += 1
        except BaseException as error:
            fatal = error
            self._set_fatal(error)
        finally:
            terminal_error: BaseException = fatal or GenerationCancelledError(
                "pipeline engine closed"
            )
            for job in list(active.values()):
                self._retire_job(job, runner, exception=terminal_error)
            # Retained sessions die with the pipeline: the SHUTDOWN below (or
            # the stage processes exiting) frees their KV remotely; release the
            # root's copies here. runner.end is a no-op for already-ended ids.
            for session in list(self._retained_sessions.values()):
                try:
                    runner.end(session.wire_id)
                except BaseException:
                    pass
            self._retained_sessions.clear()
            self._session_retained_tokens = 0
            self._fail_pending_submissions(terminal_error)
            if downstream is not None and not self._shutdown_sent:
                try:
                    send_frame(downstream, FrameType.SHUTDOWN, 0)
                    self._shutdown_sent = True
                except BaseException:
                    pass

    def _next_submission(self, timeout: float) -> list[_GenerationJob] | None:
        if self._deferred_batches:
            return self._deferred_batches.popleft()
        try:
            batch = self._submission_queue.get(timeout=timeout) if timeout > 0 else self._submission_queue.get_nowait()
        except queue.Empty:
            return None
        if batch is None:
            self._scheduler_stop.set()
            return None
        return batch

    def _admit_batch(
        self,
        batch: list[_GenerationJob],
        active: dict[int, _GenerationJob],
        runner: StageRunner,
        downstream: socket.socket,
        emulator: LinkEmulator,
    ) -> None:
        available = self.config.max_active_sequences - len(active)
        if available <= 0:
            self._deferred_batches.appendleft(batch)
            return
        selected = batch[:available]
        if len(selected) < len(batch):
            self._deferred_batches.appendleft(batch[available:])
        for index, job in enumerate(selected):
            if job.cancel_requested.is_set():
                self._retire_job(
                    job,
                    runner,
                    exception=GenerationCancelledError("generation cancelled before admission"),
                    began=False,
                )
                continue
            try:
                session = self._checkout_session(job, runner, downstream)
                if session is not None:
                    job.wire_id = session.wire_id
                    job.step = session.step
                    job.started_at = time.perf_counter()
                    active[session.wire_id] = job
                    with self._callback_lock:
                        self._callback_routes[session.wire_id] = job
                else:
                    wire_id = self._next_request_id()
                    job.wire_id = wire_id
                    job.started_at = time.perf_counter()
                    active[wire_id] = job
                    with self._callback_lock:
                        self._callback_routes[wire_id] = job
                    runner.begin(wire_id)
                    send_frame(downstream, FrameType.BEGIN, wire_id)
                self._send_next_prefill_chunk(
                    job,
                    runner,
                    downstream,
                    emulator,
                )
            except BaseException as error:
                for remaining in selected[index + 1 :]:
                    self._retire_job(remaining, runner, exception=error, began=False)
                raise

    def _checkout_session(
        self,
        job: _GenerationJob,
        runner: StageRunner,
        downstream: socket.socket,
    ) -> _RetainedSession | None:
        """Reuse the retained KV of this chat if an exact prefix is still live.

        Returns the checked-out session with the wire already truncated to the
        reusable prefix, or None when the job must open a fresh wire. Runs on
        the scheduler thread only.
        """

        key = job.request.session_key
        if key is None or self.config.max_retained_sessions < 1:
            return None
        session = self._retained_sessions.get(key)
        if session is None:
            self._session_reuse_misses += 1
            return None
        if session.busy:
            # A concurrent turn of the same chat still owns the wire. Serve
            # this request exactly with a fresh full prefill instead.
            self._session_busy_fallbacks += 1
            return None
        new_ids = [int(token) for token in job.request.input_ids.reshape(-1).tolist()]
        # The longest common prefix between the retokenized history and the
        # exactly-served sequence is a safe reuse bound even when BPE merges
        # differ at turn boundaries. Keep at least one token to forward so the
        # pipeline can produce the first token of this turn.
        prefix = _longest_common_prefix(session.served_ids, new_ids)
        reuse = min(prefix, session.kv_valid_tokens, len(new_ids) - 1)
        if reuse <= 0:
            # The histories diverge from the first token (for example an edited
            # system prompt): a full prefill costs the same as rebuilding, so
            # free the stale KV everywhere and start a fresh wire.
            self._release_retained_session(session, runner, downstream)
            self._session_divergence_fallbacks += 1
            return None
        if session.kv_tokens != reuse:
            # TRUNCATE is ordered on the stage stream, so every stage crops its
            # KV before the suffix prefill below can be processed.
            runner.truncate(session.wire_id, reuse)
            send_frame(
                downstream,
                FrameType.TRUNCATE,
                session.wire_id,
                token_count=reuse,
            )
            self._session_retained_tokens += reuse - session.kv_tokens
            session.kv_tokens = reuse
        session.kv_valid_tokens = reuse
        session.busy = True
        session.last_used = time.monotonic()
        job.session = session
        job.prefill_offset = reuse
        job.kv_valid = reuse
        job.reused_tokens = reuse
        self._session_reuse_hits += 1
        self._session_reused_token_total += reuse
        return session

    def _try_retain_session(
        self,
        job: _GenerationJob,
        runner: StageRunner,
        downstream: socket.socket,
    ) -> bool:
        """Keep this finished turn's wire and KV alive for the next turn.

        Returns True when the session machinery now owns the wire lifecycle
        (no END must be sent and the root cache must stay); False when the
        caller should retire the wire normally.
        """

        key = job.request.session_key
        if (
            key is None
            or self.config.max_retained_sessions < 1
            or job.wire_id is None
            or job.cancel_requested.is_set()
        ):
            return False
        session = job.session
        if session is None and key in self._retained_sessions:
            # Another turn of the same chat owns the retained slot; this job
            # ran on an independent fresh wire and retires normally.
            return False
        prompt_ids = tuple(
            int(token) for token in job.request.input_ids.reshape(-1).tolist()
        )
        served = prompt_ids + tuple(job.token_ids)
        kv_tokens = runner.sequence_length(job.wire_id)
        now = time.monotonic()
        if session is None:
            session = _RetainedSession(
                key=key,
                wire_id=job.wire_id,
                step=job.step + 1,
                served_ids=served,
                kv_tokens=kv_tokens,
                kv_valid_tokens=job.kv_valid,
                last_used=now,
            )
            self._session_retained_tokens += kv_tokens
        else:
            self._session_retained_tokens += kv_tokens - session.kv_tokens
            session.step = job.step + 1
            session.served_ids = served
            session.kv_tokens = kv_tokens
            session.kv_valid_tokens = job.kv_valid
            session.busy = False
            session.last_used = now
            self._retained_sessions.pop(key, None)
        # (Re-)insert last so dict order stays least-recently-used first.
        self._retained_sessions[key] = session
        self._enforce_session_budget(runner, downstream)
        return True

    def _release_retained_session(
        self,
        session: _RetainedSession,
        runner: StageRunner,
        downstream: socket.socket,
    ) -> None:
        """Retire an idle retained wire: END frees the KV on every stage."""

        self._retained_sessions.pop(session.key, None)
        self._session_retained_tokens -= session.kv_tokens
        runner.end(session.wire_id)
        send_frame(downstream, FrameType.END, session.wire_id)

    def _expire_retained_sessions(
        self,
        runner: StageRunner,
        downstream: socket.socket,
    ) -> None:
        if not self._retained_sessions:
            return
        now = time.monotonic()
        ttl = self.config.retained_session_ttl_seconds
        expired = [
            session
            for session in self._retained_sessions.values()
            if not session.busy and now - session.last_used > ttl
        ]
        for session in expired:
            self._release_retained_session(session, runner, downstream)
            self._session_ttl_evictions += 1

    def _enforce_session_budget(
        self,
        runner: StageRunner,
        downstream: socket.socket,
    ) -> None:
        max_sessions = self.config.max_retained_sessions
        max_tokens = self.config.max_retained_session_tokens
        while (max_sessions and len(self._retained_sessions) > max_sessions) or (
            max_tokens and self._session_retained_tokens > max_tokens
        ):
            victim = next(
                (
                    session
                    for session in self._retained_sessions.values()
                    if not session.busy
                ),
                None,
            )
            if victim is None:
                # Only busy sessions remain; their turns retire them later.
                return
            self._release_retained_session(victim, runner, downstream)
            self._session_budget_evictions += 1

    def _drop_job_session(self, job: _GenerationJob) -> None:
        """Forget the retained state of a cancelled turn.

        The CANCEL frame already frees the KV on every stage and _retire_job
        frees the root cache, so only the bookkeeping is removed here.
        """

        session = job.session
        if session is None:
            return
        job.session = None
        if self._retained_sessions.get(session.key) is session:
            self._retained_sessions.pop(session.key, None)
            self._session_retained_tokens -= session.kv_tokens
            self._session_cancel_releases += 1

    def _finish_turn(
        self,
        job: _GenerationJob,
        active: dict[int, _GenerationJob],
        runner: StageRunner,
        downstream: socket.socket,
        reason: str,
    ) -> None:
        if job.wire_id is not None:
            active.pop(job.wire_id, None)
        retained = self._try_retain_session(job, runner, downstream)
        if not retained and job.wire_id is not None:
            send_frame(downstream, FrameType.END, job.wire_id)
        self._retire_job(
            job,
            runner,
            result=self._generation_output(job, reason),
            retained=retained,
        )

    def _send_requested_cancellations(
        self,
        active: dict[int, _GenerationJob],
        downstream: socket.socket,
    ) -> None:
        for wire_id, job in active.items():
            if job.cancel_requested.is_set() and not job.cancel_sent:
                # The cancellation follows the already-sent activation on the same
                # TCP stream. Its token may still return and is consumed as a
                # tombstone before the job is retired.
                send_frame(downstream, FrameType.CANCEL, wire_id)
                job.cancel_sent = True

    def _handle_return_value(
        self,
        value: tuple[Any, float] | BaseException,
        active: dict[int, _GenerationJob],
        runner: StageRunner,
        downstream: socket.socket,
        emulator: LinkEmulator,
    ) -> None:
        if isinstance(value, BaseException):
            raise RuntimeError(f"pipeline connection failed: {value}") from value
        frame, arrived = value
        if frame.frame_type == FrameType.ERROR:
            raise RuntimeError(frame.payload.decode("utf-8", errors="replace"))
        if frame.frame_type not in (
            FrameType.PREFILL_ACK,
            FrameType.TOKEN,
            FrameType.VERIFY_RESULT,
        ):
            raise RuntimeError(f"unexpected return frame {frame.frame_type.name}")
        job = active.get(frame.request_id)
        if job is None:
            raise RuntimeError(f"token for unknown request {frame.request_id}")
        if frame.step != job.step:
            raise RuntimeError(
                f"request {frame.request_id} returned step {frame.step}, expected {job.step}"
            )
        if frame.frame_type == FrameType.PREFILL_ACK:
            if job.prefill_offset >= int(job.request.input_ids.shape[1]):
                raise RuntimeError(
                    f"request {frame.request_id} returned an unexpected prefill ACK"
                )
            job.step += 1
            self._send_next_prefill_chunk(
                job,
                runner,
                downstream,
                emulator,
            )
            return

        if job.cancel_requested.is_set():
            if not job.cancel_sent:
                send_frame(downstream, FrameType.CANCEL, frame.request_id)
                job.cancel_sent = True
            active.pop(frame.request_id)
            # A cancelled turn never retains: CANCEL already freed the KV on
            # every stage, so the session bookkeeping must forget this chat.
            self._drop_job_session(job)
            self._retire_job(
                job,
                runner,
                exception=GenerationCancelledError("generation cancelled"),
            )
            return

        if frame.frame_type == FrameType.VERIFY_RESULT:
            drafts = job.verify_drafts
            if not drafts:
                raise RuntimeError(
                    f"request {frame.request_id} returned verification without a draft"
                )
            targets = decode_verify_result(frame)
            accepted, emitted = _resolve_verified_tokens(drafts, targets)
            if self.speculation_controller is None:
                raise RuntimeError("verification returned while speculation is disabled")
            with self._speculation_lock:
                controller = self._speculation_controller_for_profile_locked(
                    job.speculation_profile
                )
                controller.record_verification(
                    proposed_tokens=len(drafts),
                    accepted_tokens=accepted,
                    latency_seconds=max(1e-9, arrived - job.wave_started_at),
                    transferred_bytes=(
                        job.last_outbound_bytes + HEADER_BYTES + len(frame.payload)
                    ),
                )
            job.verify_drafts = ()
            appended_before = len(job.token_ids)
            reason = self._append_verified_tokens(job, emitted, arrived)
            appended = len(job.token_ids) - appended_before
            # Draft positions in the KV are valid only up to the accepted
            # prefix AND only as far as tokens were actually emitted.
            job.kv_valid += min(accepted, appended)
            if reason is not None:
                self._finish_turn(job, active, runner, downstream, reason)
                return

            if accepted < len(drafts):
                keep_tokens = job.verify_base_tokens + 1 + accepted
                runner.truncate(frame.request_id, keep_tokens)
                send_frame(
                    downstream,
                    FrameType.TRUNCATE,
                    frame.request_id,
                    token_count=keep_tokens,
                )
            job.step += 1
            self._send_decode_wave(
                job,
                runner,
                downstream,
                emulator,
                active_sequences=len(active),
            )
            return

        if job.verify_drafts:
            raise RuntimeError(
                f"request {frame.request_id} returned TOKEN for a verification wave"
            )
        prior_output_tokens = len(job.token_ids)
        token = decode_token(frame)
        job.token_ids.append(token)
        job.arrivals.append(arrived)
        if self.speculation_controller is not None and prior_output_tokens > 0:
            with self._speculation_lock:
                controller = self._speculation_controller_for_profile_locked(
                    job.speculation_profile
                )
                controller.record_classic(
                    latency_seconds=max(1e-9, arrived - job.wave_started_at),
                    transferred_bytes=(
                        job.last_outbound_bytes + HEADER_BYTES + len(frame.payload)
                    ),
                )

        reached_eos = token in job.request.eos_token_ids
        reached_limit = len(job.token_ids) >= job.request.max_new_tokens
        if reached_eos or reached_limit:
            self._finish_turn(
                job,
                active,
                runner,
                downstream,
                "stop" if reached_eos else "length",
            )
            return

        job.step += 1
        self._send_decode_wave(
            job,
            runner,
            downstream,
            emulator,
            active_sequences=len(active),
        )

    def _append_verified_tokens(
        self,
        job: _GenerationJob,
        tokens: tuple[int, ...],
        arrived: float,
    ) -> str | None:
        for token in tokens:
            if len(job.token_ids) >= job.request.max_new_tokens:
                return "length"
            output_index = len(job.token_ids)
            job.token_ids.append(token)
            job.arrivals.append(arrived)
            if job.callback is not None:
                try:
                    job.callback(
                        job.request.client_id,
                        token,
                        output_index,
                        arrived,
                    )
                except BaseException:
                    pass
            if token in job.request.eos_token_ids:
                return "stop"
        if len(job.token_ids) >= job.request.max_new_tokens:
            return "length"
        return None

    def _send_decode_wave(
        self,
        job: _GenerationJob,
        runner: StageRunner,
        downstream: socket.socket,
        emulator: LinkEmulator,
        *,
        active_sequences: int,
    ) -> None:
        if job.wire_id is None or not job.token_ids:
            raise RuntimeError("decode job has no active token history")
        remaining = job.request.max_new_tokens - len(job.token_ids)
        provider = self.draft_provider
        controller = self.speculation_controller
        drafts: tuple[int, ...] = ()
        if provider is not None and controller is not None and remaining > 1:
            prompt_history = [
                int(token)
                for token in job.request.input_ids.reshape(-1).tolist()
            ]
            history = (*prompt_history, *job.token_ids)
            available = provider.draft(
                history,
                max_tokens=min(
                    remaining - 1,
                    self.config.speculative_max_draft_tokens,
                ),
            )
            profile = _speculation_load_profile(active_sequences)
            job.speculation_profile = profile
            with self._speculation_lock:
                controller = self._speculation_controller_for_profile_locked(profile)
                decision = controller.decide(
                    history_tokens=len(history),
                    available_draft_tokens=len(available),
                )
                reason = decision.reason
                self._speculation_decision_reasons[reason] = (
                    self._speculation_decision_reasons.get(reason, 0) + 1
                )
                if decision.enabled:
                    self._speculation_enabled_decisions += 1
                else:
                    self._speculation_disabled_decisions += 1
                selected = decision.candidate_size if decision.enabled else 0
                if selected == 0 and self.config.speculation_probe:
                    selected = controller.next_probe_size(
                        history_tokens=len(history),
                        available_draft_tokens=len(available),
                    ) or 0
                    if selected > 0:
                        self._speculation_probe_waves += 1
                if selected > 0:
                    self._speculation_selected_sizes[selected] = (
                        self._speculation_selected_sizes.get(selected, 0) + 1
                    )
            if selected > 0:
                drafts = tuple(available[:selected])

        job.verify_drafts = drafts
        if drafts:
            job.verify_base_tokens = runner.sequence_length(job.wire_id)
            input_tokens = (job.token_ids[-1], *drafts)
            frame_type = FrameType.VERIFY
        else:
            input_tokens = (job.token_ids[-1],)
            frame_type = FrameType.ACTIVATION
        next_ids = torch.tensor([input_tokens], dtype=torch.long)
        # The adaptive gate must price the complete root-to-result wave.  Starting
        # after root forward made multi-position verification look artificially
        # cheap and could enable speculation that slowed the actual conversation.
        job.wave_started_at = time.perf_counter()
        hidden = runner.forward_ids(job.wire_id, next_ids)
        # The re-forwarded last emitted token always matches the served
        # sequence; draft positions become valid only after verification.
        job.kv_valid += 1
        job.last_outbound_bytes = self._send_activation(
            downstream,
            emulator,
            job.wire_id,
            job.step,
            hidden,
            frame_type=frame_type,
        )
        job.last_sent_at = time.monotonic()

    def _speculation_controller_for_profile_locked(
        self,
        profile: str,
    ) -> AdaptiveSpeculationController:
        current = self._speculation_controllers.get(profile)
        if current is not None:
            return current
        base = self.speculation_controller
        if base is None:
            raise RuntimeError("speculation profile requested while disabled")
        # Preserve an explicitly injected controller for the first observed load
        # profile; later profiles use independent measurements with the same policy.
        current = (
            base
            if not self._speculation_controllers
            else AdaptiveSpeculationController(base.config)
        )
        self._speculation_controllers[profile] = current
        return current

    @staticmethod
    def _raise_idle_return(value: tuple[Any, float] | BaseException) -> None:
        if isinstance(value, BaseException):
            raise RuntimeError(f"pipeline connection failed while idle: {value}") from value
        frame, _ = value
        if frame.frame_type == FrameType.ERROR:
            raise RuntimeError(frame.payload.decode("utf-8", errors="replace"))
        raise RuntimeError(f"unexpected {frame.frame_type.name} frame while pipeline is idle")

    def _generation_output(self, job: _GenerationJob, reason: str) -> GenerationOutput:
        intervals = [
            (right - left) * 1_000
            for left, right in zip(job.arrivals, job.arrivals[1:])
        ]
        return GenerationOutput(
            client_id=job.request.client_id,
            token_ids=tuple(job.token_ids),
            finish_reason=reason,
            ttft_ms=(job.arrivals[0] - job.started_at) * 1_000,
            tpot_ms=sum(intervals) / len(intervals) if intervals else 0.0,
            total_ms=(job.arrivals[-1] - job.started_at) * 1_000,
            reused_kv_tokens=job.reused_tokens,
        )

    def _retire_job(
        self,
        job: _GenerationJob,
        runner: StageRunner,
        *,
        result: GenerationOutput | None = None,
        exception: BaseException | None = None,
        began: bool = True,
        retained: bool = False,
    ) -> None:
        if began and job.wire_id is not None:
            if not retained:
                # A retained wire keeps its root KV; the session machinery owns
                # its lifecycle (TTL, budget eviction or the next turn).
                runner.end(job.wire_id)
            with self._callback_lock:
                self._callback_routes.pop(job.wire_id, None)
        with self._state_lock:
            self._jobs_by_client.pop(job.request.client_id, None)
        if job.future.done():
            return
        if result is not None:
            job.future.set_result(result)
        elif isinstance(exception, GenerationCancelledError):
            job.future.cancel()
        else:
            job.future.set_exception(exception or RuntimeError("generation failed"))

    def _check_pipeline_timeouts(self, active: dict[int, _GenerationJob]) -> None:
        now = time.monotonic()
        expired = [
            job.request.client_id
            for job in active.values()
            if job.last_sent_at and now - job.last_sent_at > self.config.socket_timeout_seconds
        ]
        if expired:
            raise TimeoutError(f"timed out waiting for request {expired[0]}")

    def _set_fatal(self, error: BaseException) -> None:
        with self._state_lock:
            if self._fatal_error is None:
                self._fatal_error = f"{type(error).__name__}: {error}"

    def _fail_pending_submissions(self, error: BaseException) -> None:
        while self._deferred_batches:
            for job in self._deferred_batches.popleft():
                self._retire_job(
                    job,
                    self._require_runner(),
                    exception=error,
                    began=False,
                )
        while True:
            try:
                batch = self._submission_queue.get_nowait()
            except queue.Empty:
                return
            if batch is None:
                continue
            for job in batch:
                self._retire_job(job, self._require_runner(), exception=error, began=False)

    def _send_activation(
        self,
        downstream: socket.socket,
        emulator: LinkEmulator,
        request_id: int,
        step: int,
        hidden: torch.Tensor,
        *,
        frame_type: FrameType = FrameType.ACTIVATION,
    ) -> int:
        payload = encode_tensor(hidden, self.config.codec)
        return send_frame(
            downstream,
            frame_type,
            request_id,
            step=step,
            token_count=int(hidden.shape[1]),
            hidden_size=int(hidden.shape[2]),
            flags=int(self.config.codec),
            payload=payload,
            emulator=emulator,
        )

    def _send_next_prefill_chunk(
        self,
        job: _GenerationJob,
        runner: StageRunner,
        downstream: socket.socket,
        emulator: LinkEmulator,
    ) -> None:
        if job.wire_id is None:
            raise RuntimeError("prefill job has no wire identity")
        total = int(job.request.input_ids.shape[1])
        start = job.prefill_offset
        if not 0 <= start < total:
            raise RuntimeError(
                f"request {job.wire_id} has invalid prefill offset {start}/{total}"
            )
        configured = self.config.prefill_chunk_tokens
        chunk_size = total if configured == 0 else configured
        end = min(total, start + chunk_size)
        input_chunk = job.request.input_ids[:, start:end].to(
            dtype=torch.long,
            device="cpu",
        )
        hidden = runner.forward_ids(job.wire_id, input_chunk)
        job.prefill_offset = end
        # Prompt tokens are the served sequence by definition.
        job.kv_valid += end - start
        job.wave_started_at = time.perf_counter()
        job.last_outbound_bytes = self._send_activation(
            downstream,
            emulator,
            job.wire_id,
            job.step,
            hidden,
            frame_type=(
                FrameType.ACTIVATION if end == total else FrameType.PREFILL
            ),
        )
        job.last_sent_at = time.monotonic()

    def _receive_loop(self) -> None:
        return_socket = self._return_socket
        if return_socket is None:
            return
        try:
            while True:
                frame = recv_frame(return_socket)
                arrived = time.perf_counter()
                if frame.frame_type == FrameType.TOKEN:
                    with self._callback_lock:
                        job = self._callback_routes.get(frame.request_id)
                    if (
                        job is not None
                        and job.callback is not None
                        and frame.step == job.step
                        and not job.cancel_requested.is_set()
                    ):
                        try:
                            job.callback(
                                job.request.client_id,
                                decode_token(frame),
                                len(job.token_ids),
                                arrived,
                            )
                        except BaseException:
                            # Streaming is observational; a disconnected client must
                            # not poison the shared model pipeline for other users.
                            pass
                self._received_frames.put((frame, arrived))
        except (EOFError, OSError) as error:
            if not self._closed:
                self._received_frames.put(error)
        except BaseException as error:
            self._received_frames.put(error)

    def _control_receive_loop(self) -> None:
        downstream = self._downstream
        if downstream is None:
            return
        try:
            frame = recv_frame(downstream)
            arrived = time.perf_counter()
            if frame.frame_type != FrameType.ERROR:
                raise RuntimeError(
                    f"unexpected upstream control frame {frame.frame_type.name}"
                )
            self._received_frames.put((frame, arrived))
        except (EOFError, OSError) as error:
            if not self._closed and self._fatal_error is None:
                self._received_frames.put(error)
        except BaseException as error:
            self._received_frames.put(error)

    def _interrupt_transports(self) -> None:
        for sock in (self._return_socket, self._downstream, self._return_listener):
            if sock is not None:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

    def _close_transports(self) -> None:
        self._interrupt_transports()
        for sock in (self._return_socket, self._downstream, self._return_listener):
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass

    def _next_request_id(self) -> int:
        self._request_counter += 1
        return self._request_counter

    def _require_runner(self) -> StageRunner:
        if self._runner is None:
            raise RuntimeError("pipeline root stage is unavailable")
        return self._runner

    @staticmethod
    def _require_socket(value: socket.socket | None, name: str) -> socket.socket:
        if value is None:
            raise RuntimeError(f"pipeline {name} socket is unavailable")
        return value


def _resolve_verified_tokens(
    drafts: tuple[int, ...],
    target_tokens: tuple[int, ...],
) -> tuple[int, tuple[int, ...]]:
    """Return accepted draft count and the exact target-equivalent output burst."""

    if not drafts:
        raise ValueError("verification requires at least one draft token")
    if len(target_tokens) != len(drafts) + 1:
        raise ValueError(
            "verification target vector must contain one prediction per draft plus bonus"
        )
    accepted = 0
    while accepted < len(drafts) and drafts[accepted] == target_tokens[accepted]:
        accepted += 1
    if accepted == len(drafts):
        return accepted, (*drafts, target_tokens[-1])
    return accepted, (*drafts[:accepted], target_tokens[accepted])


def _longest_common_prefix(left: Sequence[int], right: Sequence[int]) -> int:
    """Length of the shared exact token prefix of two sequences.

    Session reuse depends on this being a LOWER bound of KV equivalence:
    tokenizing a conversation again may merge bytes differently exactly at the
    first position where the sequences stop matching, never before it.
    """

    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def _finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _speculation_load_profile(active_sequences: int) -> str:
    if active_sequences < 1:
        raise ValueError("active_sequences must be positive")
    if active_sequences == 1:
        return "load-1"
    if active_sequences == 2:
        return "load-2"
    if active_sequences <= 4:
        return "load-3-4"
    return "load-5-plus"


def balanced_boundaries(total_layers: int, stages: int) -> tuple[int, ...]:
    if total_layers < 2:
        raise ValueError("model must have at least two layers")
    if not 2 <= stages <= total_layers:
        raise ValueError("stages must be between 2 and the number of model layers")
    return tuple(round(index * total_layers / stages) for index in range(stages + 1))


def parse_boundaries(raw: str, total_layers: int) -> tuple[int, ...]:
    try:
        boundaries = tuple(int(item.strip()) for item in raw.split(","))
    except ValueError as error:
        raise ValueError("boundaries must be comma-separated integers") from error
    if len(boundaries) < 3 or boundaries[0] != 0 or boundaries[-1] != total_layers:
        raise ValueError(f"boundaries must describe at least two stages from 0 to {total_layers}")
    if any(right <= left for left, right in zip(boundaries, boundaries[1:])):
        raise ValueError("boundaries must be strictly increasing")
    return boundaries


def _reserve_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
    finally:
        sock.close()


def _wait_until_listening(process: Any, ready_event: Any, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if ready_event.wait(0.1):
            return
        if process.exitcode is not None:
            raise RuntimeError(
                f"stage process {process.pid} exited during startup with code {process.exitcode}"
            )
    raise TimeoutError(f"stage process {process.pid} did not listen within {timeout_seconds}s")
