from __future__ import annotations

from concurrent.futures import CancelledError as FutureCancelledError, Future
from collections import deque
from collections.abc import Sequence
import copy
from dataclasses import dataclass, field
import math
import multiprocessing as mp
import queue
import select
import socket
import threading
import time
from typing import Any, Callable

import torch
from transformers import AutoConfig

from .device import normalize_torch_device_request
from .macro_wave import KVVersion, MacroWaveState
from .macro_wave_adapter import (
    MacroWaveProposal,
    prepare_linear_macro_wave,
    prepare_tree_macro_wave,
    record_linear_resolution,
    resolve_linear_macro_wave,
)
from .model import (
    StageModelSpec,
    StageRunner,
    StageRunnerContract,
    model_artifact_reference,
    resolve_model_snapshot,
)
from .protocol import (
    HEADER_BYTES,
    FrameType,
    LinkEmulator,
    TensorCodec,
    TreePrepareStatus,
    branch_request_payload,
    configure_socket,
    decode_tree_prepare,
    decode_tree_reservation_nonce,
    decode_token,
    decode_verify_result,
    encode_tensor_payload,
    recv_frame,
    send_frame,
    tree_prepare_payload,
    tree_reservation_payload,
)
from .physical_tree import (
    CancelCommand,
    EndCommand,
    PhysicalTreeCoordinator,
    PhysicalTreeError,
    PromoteCommand,
    TombstoneDrain,
    TreeAbortPlan,
    TreeCommitPlan,
    TreePhase,
    TruncateCommand,
    VerifyCommand,
)
from .ram_backed_moe_runtime import (
    RamBackedMoeRuntimeConfig,
    build_ram_backed_moe_stage_runner,
    validate_ram_backed_moe_binding,
)
from .speculation import (
    MAX_DRAFT_TOKENS,
    AdaptiveSpeculationConfig,
    AdaptiveSpeculationController,
    DraftProvider,
    NgramDraftProvider,
    NgramTreeDraftProvider,
    TreeDraftProvider,
)
from .stage import (
    MAX_SPECULATIVE_BRANCHES,
    MAX_SPECULATIVE_BRANCH_TOKENS,
    MAX_SPECULATIVE_KV_BYTES,
    StageProcessConfig,
    connect_with_retry,
    drain_metrics,
    run_stage_process,
    validate_speculative_runner,
)


TokenCallback = Callable[[int, int, int, float], None]

MAX_PREFILL_INFLIGHT_CHUNKS = 64
MAX_PREFILL_INFLIGHT_BYTES = 1 << 30
SCHEDULER_SHUTDOWN_GRACE_SECONDS = 2.0
CHILD_SHUTDOWN_GRACE_SECONDS = 5.0
IO_THREAD_SHUTDOWN_GRACE_SECONDS = 2.0
HARD_SHUTDOWN_GRACE_SECONDS = 5.0


def _sealed_wave_token_limit(config: Any) -> int:
    explicit = getattr(config, "sealed_wave_tokens", None)
    if explicit is not None:
        return int(explicit)
    draft_tokens = int(getattr(config, "speculative_max_draft_tokens", 0))
    return draft_tokens + 1 if draft_tokens > 0 else 1


def _prefill_token_limit(config: Any) -> int | None:
    explicit = getattr(config, "max_prefill_chunk_tokens", None)
    if explicit is not None:
        return int(explicit)
    configured = int(getattr(config, "prefill_chunk_tokens", 0))
    return configured if configured > 0 else None


def _prefill_inflight_chunk_limit(config: Any) -> int:
    return int(getattr(config, "prefill_inflight_chunks", 1))


def _prefill_inflight_byte_limit(config: Any) -> int:
    return int(getattr(config, "prefill_inflight_bytes", 0))


def _hadamard_quantization_block_count(hidden_size: int) -> int:
    remaining = hidden_size
    blocks = 0
    while remaining > 0:
        size = 1 << int(math.floor(math.log2(min(64, remaining))))
        remaining -= size
        blocks += 1
    return blocks


def _prefill_frame_byte_reservation(
    codec: TensorCodec,
    token_count: int,
    hidden_size: int,
) -> int:
    """Conservative, data-independent bytes reserved by one prefill frame."""

    if token_count < 1 or hidden_size < 1:
        raise ValueError("prefill frame shape must be positive")
    elements = token_count * hidden_size
    base_codec = {
        TensorCodec.INT8_GROUPED_DEFLATE: TensorCodec.INT8_GROUPED,
        TensorCodec.INT8_HADAMARD_DEFLATE: TensorCodec.INT8_HADAMARD,
    }.get(codec, codec)
    if base_codec == TensorCodec.FP32:
        payload_bytes = elements * 4
    elif base_codec == TensorCodec.FP16:
        payload_bytes = elements * 2
    elif base_codec == TensorCodec.INT8:
        payload_bytes = elements + 4
    elif base_codec == TensorCodec.INT8_GROUPED:
        blocks = (hidden_size + 63) // 64
        payload_bytes = elements + token_count * blocks * 4
    elif base_codec == TensorCodec.INT8_HADAMARD:
        blocks = _hadamard_quantization_block_count(hidden_size)
        payload_bytes = elements + token_count * blocks * 4
    else:
        raise ValueError(f"unsupported tensor codec {codec}")
    if base_codec != codec:
        # Same zlib worst-case bound used by protocol._deflate_bound. Reserving
        # the bound, rather than an observed compression ratio, keeps credit
        # admission deterministic before root KV is advanced.
        payload_bytes = (
            payload_bytes
            + (payload_bytes >> 12)
            + (payload_bytes >> 14)
            + (payload_bytes >> 25)
            + 19
        )
    return HEADER_BYTES + payload_bytes


@dataclass(frozen=True)
class PipelineEngineConfig:
    model_name: str
    boundaries: tuple[int, ...]
    codec: TensorCodec = TensorCodec.FP16
    threads_per_stage: int = 1
    device: str = "auto"
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
    artifact_identity: str | None = None
    canonical_model_source: str | None = None
    canonical_model_revision: str | None = None
    pipeline_snapshot_identity: int | None = None
    # Root + every child stage for local physical tests. In remote mode only
    # the root slot may be configured here; each child process receives its own
    # independently sealed host-local binding from stage_cli.
    ram_backed_moe_stages: tuple[RamBackedMoeRuntimeConfig | None, ...] | None = None
    # Ordered executor ids for root + every child stage. Remote recovery must
    # receive this sealed route contract from its launcher because the root
    # process cannot infer a remote cell/GGUF/quantized backend from boundaries.
    stage_executor_ids: tuple[str, ...] | None = None
    max_active_sequences: int = 8
    max_pending_requests: int = 128
    # Zero keeps the whole prompt in one physical prefill wave.  A positive
    # value sends bounded chunks and lets decode work enter between their ACKs.
    prefill_chunk_tokens: int = 0
    # Exact prefill may pipeline several ordered chunks from one request. Both
    # ceilings are per request; the global maximum is therefore bounded by
    # max_active_sequences times each value. One chunk preserves historical
    # stop-and-wait. The byte ceiling reserves a conservative payload bound
    # before advancing root KV; zero disables only this secondary ceiling.
    prefill_inflight_chunks: int = 1
    prefill_inflight_bytes: int = 0
    # Physical sparse-tree execution remains off unless all three limits are
    # positive. They are part of the immutable execution/recovery contract even
    # before a branch scheduler is enabled at the root.
    max_speculative_branches: int = 0
    max_speculative_branch_tokens: int = 0
    max_speculative_kv_bytes: int = 0
    # Hard frame limits sealed by the launcher/planner. None preserves direct
    # programmatic compatibility while deriving the smallest safe decode wave.
    sealed_wave_tokens: int | None = None
    max_prefill_chunk_tokens: int | None = None
    # Zero is exact autoregressive decode. Positive values enable exact target
    # verification with an adaptive n-gram draft source by default.
    speculative_max_draft_tokens: int = 0
    speculation_minimum_speedup: float = 1.05
    speculation_probe: bool = True
    # Return frames from one downstream physical batch arrive individually.
    # Hold the first ready continuation very briefly so the root can reconstruct
    # the same tensor wave instead of serializing it into batch-one forwards.
    root_batch_window_ms: float = 0.5
    # A tiny no-compute frame periodically traverses the full forward route and
    # returns over the token socket. Zero disables periodic/startup probes.
    route_probe_interval_seconds: float = 5.0
    route_probe_timeout_seconds: float = 10.0
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
        normalize_torch_device_request(self.device)
        if not self.model_name.strip():
            raise ValueError("model_name cannot be empty")
        for name, value in (
            ("artifact_identity", self.artifact_identity),
            ("canonical_model_source", self.canonical_model_source),
            ("canonical_model_revision", self.canonical_model_revision),
        ):
            if value is not None and not value.strip():
                raise ValueError(f"{name} cannot be blank")
        if self.artifact_identity is None and (
            self.canonical_model_source is not None
            or self.canonical_model_revision is not None
        ):
            raise ValueError(
                "canonical model coordinates require an explicit artifact identity"
            )
        if self.pipeline_snapshot_identity is not None and (
            not isinstance(self.pipeline_snapshot_identity, int)
            or isinstance(self.pipeline_snapshot_identity, bool)
            or not 0 <= self.pipeline_snapshot_identity <= (1 << 64) - 1
        ):
            raise ValueError("pipeline_snapshot_identity must fit uint64")
        if len(self.boundaries) < 3:
            raise ValueError("a distributed pipeline requires at least two stages")
        if self.boundaries[0] != 0:
            raise ValueError("boundaries must start at zero")
        if any(right <= left for left, right in zip(self.boundaries, self.boundaries[1:])):
            raise ValueError("boundaries must be strictly increasing")
        if self.ram_backed_moe_stages is not None:
            stage_count = len(self.boundaries) - 1
            if len(self.ram_backed_moe_stages) != stage_count:
                raise ValueError(
                    "ram_backed_moe_stages must contain one entry per stage"
                )
            configured_ram = tuple(
                entry for entry in self.ram_backed_moe_stages if entry is not None
            )
            if configured_ram:
                if self.pipeline_snapshot_identity is None:
                    raise ValueError(
                        "RAM-backed MoE stages require pipeline_snapshot_identity"
                    )
                if len({entry.artifact_identity for entry in configured_ram}) != 1:
                    raise ValueError(
                        "RAM-backed MoE stages must bind one common model artifact"
                    )
                if self.spawn_local_stages and len(configured_ram) != stage_count:
                    raise ValueError(
                        "local RAM-backed MoE execution requires a binding for every stage"
                    )
                if not self.spawn_local_stages and any(
                    entry is not None for entry in self.ram_backed_moe_stages[1:]
                ):
                    raise ValueError(
                        "remote child RAM-backed MoE bindings belong to stage_cli"
                    )
                if any(
                    value is not None
                    for value in (
                        self.artifact_identity,
                        self.canonical_model_source,
                        self.canonical_model_revision,
                    )
                ):
                    raise ValueError(
                        "RAM-backed MoE identity cannot be combined with standard "
                        "model identity fields"
                    )
        if self.stage_executor_ids is not None:
            if len(self.stage_executor_ids) != len(self.boundaries) - 1:
                raise ValueError("stage_executor_ids must contain one id per stage")
            for executor_id in self.stage_executor_ids:
                if (
                    len(executor_id) != 32
                    or executor_id != executor_id.lower()
                    or any(character not in "0123456789abcdef" for character in executor_id)
                ):
                    raise ValueError("stage executor ids must be 32 lowercase hex characters")
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
            not isinstance(self.prefill_inflight_chunks, int)
            or isinstance(self.prefill_inflight_chunks, bool)
            or not 1
            <= self.prefill_inflight_chunks
            <= MAX_PREFILL_INFLIGHT_CHUNKS
        ):
            raise ValueError(
                "prefill_inflight_chunks must be between 1 and "
                f"{MAX_PREFILL_INFLIGHT_CHUNKS}"
            )
        if (
            not isinstance(self.prefill_inflight_bytes, int)
            or isinstance(self.prefill_inflight_bytes, bool)
            or not 0 <= self.prefill_inflight_bytes <= MAX_PREFILL_INFLIGHT_BYTES
        ):
            raise ValueError(
                "prefill_inflight_bytes must be zero or between 1 and "
                f"{MAX_PREFILL_INFLIGHT_BYTES}"
            )
        for name, value, maximum in (
            (
                "max_speculative_branches",
                self.max_speculative_branches,
                MAX_SPECULATIVE_BRANCHES,
            ),
            (
                "max_speculative_branch_tokens",
                self.max_speculative_branch_tokens,
                MAX_SPECULATIVE_BRANCH_TOKENS,
            ),
            (
                "max_speculative_kv_bytes",
                self.max_speculative_kv_bytes,
                MAX_SPECULATIVE_KV_BYTES,
            ),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or not 0 <= value <= maximum
            ):
                raise ValueError(f"{name} must be between 0 and {maximum}")
        speculative_tree_limits_enabled = (
            self.max_speculative_branches > 0,
            self.max_speculative_branch_tokens > 0,
            self.max_speculative_kv_bytes > 0,
        )
        if any(speculative_tree_limits_enabled) and not all(
            speculative_tree_limits_enabled
        ):
            raise ValueError(
                "speculative branch count, tokens and KV bytes must all be zero "
                "or all be positive"
            )
        if (self.sealed_wave_tokens is None) != (
            self.max_prefill_chunk_tokens is None
        ):
            raise ValueError(
                "sealed_wave_tokens and max_prefill_chunk_tokens must be supplied together"
            )
        if (
            not isinstance(self.speculative_max_draft_tokens, int)
            or isinstance(self.speculative_max_draft_tokens, bool)
            or not 0 <= self.speculative_max_draft_tokens <= MAX_DRAFT_TOKENS
        ):
            raise ValueError(
                f"speculative_max_draft_tokens must be between 0 and {MAX_DRAFT_TOKENS}"
            )
        if self.sealed_wave_tokens is not None:
            if (
                not isinstance(self.sealed_wave_tokens, int)
                or isinstance(self.sealed_wave_tokens, bool)
                or not 1 <= self.sealed_wave_tokens <= MAX_DRAFT_TOKENS + 1
            ):
                raise ValueError(
                    f"sealed_wave_tokens must be between 1 and {MAX_DRAFT_TOKENS + 1}"
                )
            required_wave_tokens = (
                self.speculative_max_draft_tokens + 1
                if self.speculative_max_draft_tokens > 0
                else 1
            )
            if self.sealed_wave_tokens < required_wave_tokens:
                raise ValueError(
                    "sealed_wave_tokens cannot be smaller than the VERIFY input"
                )
            if (
                self.speculative_max_draft_tokens == 0
                and self.sealed_wave_tokens != 1
            ):
                raise ValueError(
                    "sealed_wave_tokens greater than one require a draft provider"
                )
        if self.max_prefill_chunk_tokens is not None:
            if (
                not isinstance(self.max_prefill_chunk_tokens, int)
                or isinstance(self.max_prefill_chunk_tokens, bool)
                or self.max_prefill_chunk_tokens < 1
            ):
                raise ValueError("max_prefill_chunk_tokens must be positive")
            if (
                self.prefill_chunk_tokens > 0
                and self.prefill_chunk_tokens > self.max_prefill_chunk_tokens
            ):
                raise ValueError(
                    "prefill_chunk_tokens cannot exceed max_prefill_chunk_tokens"
                )
        if (
            not math.isfinite(self.speculation_minimum_speedup)
            or self.speculation_minimum_speedup < 1.0
        ):
            raise ValueError("speculation_minimum_speedup must be finite and at least 1")
        if not isinstance(self.speculation_probe, bool):
            raise TypeError("speculation_probe must be boolean")
        if (
            not math.isfinite(self.root_batch_window_ms)
            or not 0 <= self.root_batch_window_ms <= 100
        ):
            raise ValueError("root_batch_window_ms must be between 0 and 100")
        if (
            not math.isfinite(self.route_probe_interval_seconds)
            or self.route_probe_interval_seconds < 0
        ):
            raise ValueError(
                "route_probe_interval_seconds must be finite and non-negative"
            )
        if (
            not math.isfinite(self.route_probe_timeout_seconds)
            or self.route_probe_timeout_seconds <= 0
        ):
            raise ValueError("route_probe_timeout_seconds must be finite and positive")
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

    @property
    def sealed_wave_token_limit(self) -> int:
        return _sealed_wave_token_limit(self)

    @property
    def prefill_token_limit(self) -> int | None:
        return _prefill_token_limit(self)


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


@dataclass(frozen=True)
class PipelineRecoveryIdentity:
    """Route-independent identity required for token-exact replay recovery.

    Hosts and ports are deliberately excluded so a standby route can live on a
    different set of machines.  Every model- or execution-affecting field is
    included; a replacement that differs in any of them must not continue an
    already visible token stream.
    """

    schema_version: int
    artifact_identity: str
    canonical_model_source: str
    canonical_model_revision: str | None
    pipeline_snapshot_identity: int
    boundaries: tuple[int, ...]
    codec: int
    total_layers: int
    hidden_size: int
    maximum_context: int
    root_stage_loader: str
    stage_executor_ids: tuple[str, ...]
    threads_per_stage: int
    prefill_chunk_tokens: int
    prefill_inflight_chunks: int
    prefill_inflight_bytes: int
    max_speculative_branches: int
    max_speculative_branch_tokens: int
    max_speculative_kv_bytes: int
    sealed_wave_tokens: int
    max_prefill_chunk_tokens: int
    speculative_max_draft_tokens: int
    speculation_minimum_speedup: float
    speculation_probe: bool


class GenerationCancelledError(RuntimeError):
    pass


class PipelineShutdownError(RuntimeError):
    """The bounded shutdown finished without a clean local lifecycle."""

    def __init__(self, report: dict[str, Any]) -> None:
        self.report = copy.deepcopy(report)
        failed_children = [
            child
            for child in report.get("child_processes", ())
            if child.get("alive") or child.get("exit_code") != 0
        ]
        alive_threads = [
            name
            for name, alive in report.get("threads_alive", {}).items()
            if alive
        ]
        fatal_count = len(report.get("fatal_stage_metrics", ()))
        reasons: list[str] = []
        if not report.get("shutdown_sent", False):
            reasons.append("SHUTDOWN was not sent")
        if report.get("hard_fallback_used", False):
            reasons.append("hard fallback was required")
        if failed_children:
            rendered = ", ".join(
                f"{child.get('name')}={child.get('exit_code')}"
                + ("(alive)" if child.get("alive") else "")
                for child in failed_children
            )
            reasons.append(f"child exit failure: {rendered}")
        if alive_threads:
            reasons.append(f"threads still alive: {', '.join(alive_threads)}")
        if fatal_count:
            reasons.append(f"fatal stage metrics: {fatal_count}")
        for error in report.get("cleanup_errors", ()):
            reasons.append(str(error))
        super().__init__(
            "pipeline shutdown was not clean"
            + (f": {'; '.join(reasons)}" if reasons else "")
        )


@dataclass(frozen=True)
class _InflightWave:
    step: int
    frame_type: FrameType
    prefill_end: int | None
    started_at: float
    sent_at: float
    outbound_bytes: int
    reserved_bytes: int

    @property
    def is_prefill(self) -> bool:
        return self.prefill_end is not None


@dataclass
class _GenerationJob:
    request: GenerationInput
    callback: TokenCallback | None
    future: Future[GenerationOutput] = field(default_factory=Future)
    cancel_requested: threading.Event = field(default_factory=threading.Event)
    wire_id: int | None = None
    # ``step`` is the oldest return expected from the route. ``next_step`` is
    # assigned to the next immutable outbound wave. They differ only while a
    # credit-window prefill has more than one chunk in flight.
    step: int = 0
    next_step: int | None = None
    started_at: float = 0.0
    last_sent_at: float = 0.0
    token_ids: list[int] = field(default_factory=list)
    arrivals: list[float] = field(default_factory=list)
    cancel_sent: bool = False
    prefill_offset: int = 0
    prefill_acked_offset: int = 0
    prefill_inflight_bytes: int = 0
    prefill_reserved_bytes: int = 0
    inflight_waves: deque[_InflightWave] = field(default_factory=deque)
    wave_started_at: float = 0.0
    last_outbound_bytes: int = 0
    verify_proposal: MacroWaveProposal | None = None
    verify_base_tokens: int = 0
    speculation_profile: str = "load-1"
    # Retained session this job checked out at admission (None for fresh wires).
    session: "_RetainedSession | None" = None
    reused_tokens: int = 0
    # Number of leading KV positions on every stage that are known to equal the
    # exact served sequence (prompt + emitted tokens). The physical KV may be
    # longer when a turn ends right after a partially rejected VERIFY wave.
    kv_valid: int = 0
    # Physical leaves are never jobs and never receive a future/callback.  The
    # parent can nevertheless be closed before a cancelled leaf return has
    # drained, so retirement of its user future must be deferred explicitly.
    physical_tree_parent_closed: bool = False
    physical_tree_pending_result: GenerationOutput | None = None
    physical_tree_pending_exception: BaseException | None = None

    def __post_init__(self) -> None:
        if self.next_step is None:
            self.next_step = self.step


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


@dataclass(frozen=True)
class _PreparedRootWave:
    """One request continuation whose root KV has not been advanced yet."""

    job: _GenerationJob
    input_ids: torch.Tensor
    frame_type: FrameType
    step: int
    prefill_end: int | None = None
    reserved_bytes: int = 0


@dataclass(frozen=True)
class _PreparedPhysicalTreeWave:
    """One validated logical tree which has not mutated root or wire KV yet."""

    job: _GenerationJob
    proposal: MacroWaveProposal
    base_kv_tokens: int
    pending_token: int
    step: int
    active_sequences: int


@dataclass(frozen=True)
class _RootTreeCapacitySnapshot:
    parent_tokens: int
    parent_cache_bytes: int
    live_child_ids: tuple[int, ...]
    live_child_bytes: int
    projected_kv_bytes: int
    available_physical_bytes: int | None = None


@dataclass
class _PendingTreeReservation:
    prepared: _PreparedPhysicalTreeWave
    nonce: int
    path_lengths: tuple[int, ...]
    root_snapshot: _RootTreeCapacitySnapshot
    sent_at: float
    deadline_at: float
    downstream: socket.socket
    commit_sent: bool = False
    committed: bool = False


@dataclass(frozen=True)
class _CommittedPhysicalTreeWave:
    pending: _PendingTreeReservation


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
        tree_draft_provider: NgramTreeDraftProvider | TreeDraftProvider | None = None,
    ) -> None:
        self.config = config
        self._state_lock = threading.Lock()
        self._closed = False
        self._close_error: PipelineShutdownError | None = None
        self._shutdown_report: dict[str, Any] | None = None
        self._fatal_error: str | None = None
        self._request_counter = 0
        self._processes: list[Any] = []
        self._metrics_queue: Any | None = None
        self._runner: StageRunnerContract | None = None
        self._downstream: socket.socket | None = None
        self._return_socket: socket.socket | None = None
        self._return_listener: socket.socket | None = None
        self._shutdown_sent = False
        self._received_frames: queue.Queue[tuple[Any, float] | BaseException] = queue.Queue()
        # PONGs are timestamped by the socket reader but consumed by the
        # scheduler.  Keeping the probe state machine on one thread prevents a
        # timely PONG from racing the scheduler's deadline check.
        self._route_probe_returns: queue.Queue[tuple[Any, float]] = queue.Queue()
        self._callback_routes: dict[int, _GenerationJob] = {}
        # Virtual physical leaves deliberately live outside callback/chat
        # routing. Values are their real parent request IDs, never user IDs.
        self._leaf_routes: dict[int, int] = {}
        self._tree_return_lock = threading.Lock()
        self._queued_tree_returns: dict[int, deque[float]] = {}
        self._physical_tree_live_children: set[int] = set()
        self._physical_tree = PhysicalTreeCoordinator()
        self._physical_tree_prepared_waves = 0
        self._physical_tree_prepared_leaves = 0
        self._physical_tree_committed_waves = 0
        self._physical_tree_aborted_waves = 0
        self._physical_tree_batch_calls = 0
        self._physical_tree_batch_items = 0
        self._pending_tree_reservation: _PendingTreeReservation | None = None
        self._tree_quote_nonce_counter = 0
        self._queued_tree_prepare_results: deque[tuple[int, int, float]] = deque()
        self._queued_tree_commit_results: deque[tuple[int, int, float]] = deque()
        self._tree_barrier_deferred_waves: deque[
            _PreparedRootWave | _PreparedPhysicalTreeWave
        ] = deque()
        self._physical_tree_quote_requests = 0
        self._physical_tree_quote_ready = 0
        self._physical_tree_quote_rejected = 0
        self._physical_tree_quote_cancelled = 0
        self._physical_tree_quote_timeouts = 0
        self._physical_tree_quote_singleflight_fallbacks = 0
        self._physical_tree_quote_protocol_failures = 0
        self._physical_tree_quote_rtt_seconds = 0.0
        self._physical_tree_quote_rejections: dict[str, int] = {}
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
        self._root_ready_items = 0
        self._root_model_forward_calls = 0
        self._root_physical_batch_calls = 0
        self._root_physical_batch_items = 0
        self._root_sequential_items = 0
        self._root_max_physical_batch_size = 1
        self._prefill_current_chunks = 0
        self._prefill_current_bytes = 0
        self._prefill_current_reserved_bytes = 0
        self._prefill_high_water_chunks = 0
        self._prefill_high_water_bytes = 0
        self._prefill_high_water_reserved_bytes = 0
        self._prefill_max_request_chunks = 0
        self._prefill_max_request_bytes = 0
        self._prefill_max_request_reserved_bytes = 0
        self._prefill_dispatched_chunks = 0
        self._prefill_completed_chunks = 0
        self._prefill_acknowledged_chunks = 0
        self._route_probe_lock = threading.Lock()
        self._route_probe_sent_at: float | None = None
        self._route_probe_deadline_at: float | None = None
        self._route_probe_pending_step: int | None = None
        self._route_probe_sequence = 0
        self._route_probe_next_at = 0.0
        self._route_rtt_ms: float | None = None
        self._route_probe_count = 0
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

        tree_limits_enabled = all(
            value > 0
            for value in (
                config.max_speculative_branches,
                config.max_speculative_branch_tokens,
                config.max_speculative_kv_bytes,
            )
        )
        if tree_draft_provider is not None:
            if not isinstance(tree_draft_provider, TreeDraftProvider):
                raise ValueError("tree_draft_provider must implement TreeDraftProvider")
            if not tree_limits_enabled:
                raise ValueError(
                    "tree_draft_provider requires all three sealed speculative limits"
                )
            if config.speculative_max_draft_tokens < 1:
                raise ValueError(
                    "tree_draft_provider requires speculative_max_draft_tokens > 0"
                )
        self.tree_draft_provider: TreeDraftProvider | None = tree_draft_provider

        ram_stage_configs = config.ram_backed_moe_stages or tuple(
            None for _ in range(len(config.boundaries) - 1)
        )
        root_ram_config = ram_stage_configs[0]
        if root_ram_config is not None:
            snapshot_path = validate_ram_backed_moe_binding(
                root_ram_config,
                model_name=config.model_name,
                revision=config.revision,
                pipeline_snapshot_identity=config.pipeline_snapshot_identity,
            )
            self.model_snapshot = str(snapshot_path)
            model_config = AutoConfig.from_pretrained(
                self.model_snapshot,
                local_files_only=True,
                trust_remote_code=False,
            )
        else:
            model_config = AutoConfig.from_pretrained(
                config.model_name,
                revision=config.revision,
            )
        self.total_layers = int(model_config.num_hidden_layers)
        self.hidden_size = int(model_config.hidden_size)
        if config.boundaries[-1] != self.total_layers:
            raise ValueError(
                f"boundaries end at {config.boundaries[-1]}, model has {self.total_layers} layers"
            )
        self.maximum_context = int(getattr(model_config, "max_position_embeddings", 0) or 0)
        if root_ram_config is not None:
            self.model_artifact = model_artifact_reference(
                self.model_snapshot,
                artifact_identity=root_ram_config.artifact_identity,
                canonical_source=(
                    f"content-addressed://{root_ram_config.artifact_identity}"
                ),
                canonical_revision=None,
            )
        else:
            self.model_snapshot = resolve_model_snapshot(
                config.model_name,
                config.revision,
            )
            self.model_artifact = model_artifact_reference(
                self.model_snapshot,
                artifact_identity=config.artifact_identity,
                canonical_source=config.canonical_model_source,
                canonical_revision=config.canonical_model_revision,
            )
        self.pipeline_id = (
            config.pipeline_snapshot_identity
            if config.pipeline_snapshot_identity is not None
            else self.model_artifact.snapshot_identity
        )
        try:
            self._start()
        except BaseException as error:
            self._close_preserving_exception(error)
            raise

    @property
    def stages(self) -> int:
        return len(self.config.boundaries) - 1

    @property
    def root_parameter_bytes(self) -> int:
        return self._require_runner().parameter_bytes

    @property
    def execution_topology(self) -> dict[str, Any]:
        """Return effective device evidence observed by this root process.

        Remote workers publish their startup snapshot in their own process log.
        This root reports only snapshots it has directly observed, so absent
        remote or child evidence remains unobserved rather than being guessed
        from the launch plan.
        """

        if self._metrics_queue is not None:
            self.stage_metrics.extend(drain_metrics(self._metrics_queue))
        runner = self._require_runner()
        snapshot = getattr(runner, "execution_snapshot", None)
        root_execution = snapshot() if callable(snapshot) else {}
        observed: dict[int, dict[str, Any]] = {
            int(self.config.boundaries[0]): {
                "stage": int(self.config.boundaries[0]),
                "layer_end": int(self.config.boundaries[1]),
                "execution": root_execution,
            }
        }
        for metric in self.stage_metrics:
            if (
                not isinstance(metric, dict)
                or metric.get("event") != "stage_runtime_ready"
                or not isinstance(metric.get("stage"), int)
                or not isinstance(metric.get("execution"), dict)
            ):
                continue
            observed[int(metric["stage"])] = copy.deepcopy(metric)
        stages = [observed[index] for index in sorted(observed)]
        return {
            "requested_device": self.config.device,
            "observed_stage_count": len(stages),
            "total_stage_count": self.stages,
            "stages": stages,
        }

    @property
    def healthy(self) -> bool:
        with self._state_lock:
            return not self._closed and self._fatal_error is None

    @property
    def shutdown_status(self) -> dict[str, Any] | None:
        """Return immutable evidence from the last completed close attempt."""

        report = getattr(self, "_shutdown_report", None)
        return copy.deepcopy(report) if report is not None else None

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
    def recovery_identity(self) -> PipelineRecoveryIdentity:
        if getattr(self, "tree_draft_provider", None) is not None:
            raise RuntimeError(
                "physical sparse-tree execution is not replay-recovery eligible: "
                "the injected tree draft policy is not sealed in PipelineEngineConfig"
            )
        stage_executor_ids = self._recovery_stage_executor_ids()
        return PipelineRecoveryIdentity(
            schema_version=4,
            artifact_identity=self.model_artifact.identity,
            canonical_model_source=self.model_artifact.canonical_source,
            canonical_model_revision=self.model_artifact.canonical_revision,
            pipeline_snapshot_identity=self.pipeline_id,
            boundaries=self.config.boundaries,
            codec=int(self.config.codec),
            total_layers=self.total_layers,
            hidden_size=self.hidden_size,
            maximum_context=self.maximum_context,
            root_stage_loader=self._require_runner().loader,
            stage_executor_ids=stage_executor_ids,
            threads_per_stage=self.config.threads_per_stage,
            prefill_chunk_tokens=self.config.prefill_chunk_tokens,
            prefill_inflight_chunks=self.config.prefill_inflight_chunks,
            prefill_inflight_bytes=self.config.prefill_inflight_bytes,
            max_speculative_branches=self.config.max_speculative_branches,
            max_speculative_branch_tokens=self.config.max_speculative_branch_tokens,
            max_speculative_kv_bytes=self.config.max_speculative_kv_bytes,
            sealed_wave_tokens=self.config.sealed_wave_token_limit,
            max_prefill_chunk_tokens=self.config.prefill_token_limit or 0,
            speculative_max_draft_tokens=self.config.speculative_max_draft_tokens,
            speculation_minimum_speedup=self.config.speculation_minimum_speedup,
            speculation_probe=self.config.speculation_probe,
        )

    def _recovery_stage_executor_ids(self) -> tuple[str, ...]:
        runner = self._require_runner()
        manifest = getattr(runner, "executor_manifest", None)
        if manifest is None:
            raise RuntimeError("root stage does not expose a sealed executor manifest")

        configured = self.config.stage_executor_ids
        if not self.config.spawn_local_stages:
            if configured is None:
                raise RuntimeError(
                    "remote recovery requires explicit sealed stage_executor_ids"
                )
            if configured[0] != manifest.executor_id:
                raise RuntimeError(
                    "remote recovery root stage executor id does not match its manifest"
                )
            return configured

        from .executor_abi import build_stage_executor_manifest

        derived: list[str] = []
        for layer_start, layer_end in zip(
            self.config.boundaries,
            self.config.boundaries[1:],
        ):
            stage_manifest = build_stage_executor_manifest(
                engine=manifest.engine,
                engine_version=manifest.engine_version,
                adapter=manifest.adapter,
                model_identity=manifest.model_identity,
                model_source=manifest.model_source,
                model_revision=manifest.model_revision,
                artifact_format=manifest.artifact_format,
                layer_start=layer_start,
                layer_end=layer_end,
                total_layers=manifest.total_layers,
                hidden_size=manifest.hidden_size,
                activation_dtype=manifest.activation_dtype,
                activation_codecs=manifest.activation_codecs,
                kv_format=manifest.kv_format,
                supports_truncate=manifest.supports_truncate,
                phases=manifest.phases,
                operations=manifest.operations,
                max_batch_size=manifest.max_batch_size,
                max_context_tokens=manifest.max_context_tokens,
                device_kinds=manifest.device_kinds,
                compute_apis=manifest.compute_apis,
                weight_dtypes=manifest.weight_dtypes,
                features=manifest.features,
            )
            derived.append(stage_manifest.executor_id)
        result = tuple(derived)
        if configured is not None and configured != result:
            raise RuntimeError(
                "configured stage_executor_ids do not match local executor manifests"
            )
        return result

    @property
    def speculation_stats(self) -> dict[str, Any]:
        controller = self.speculation_controller
        route_rtt_ms, route_probe_count = self._route_probe_snapshot()
        tree_stats = self._physical_tree_stats()
        if controller is None:
            return {
                "configured": False,
                "enabled": False,
                "route_rtt_ms": route_rtt_ms,
                "route_probe_count": route_probe_count,
                "physical_tree": tree_stats,
            }
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
                "route_rtt_ms": route_rtt_ms,
                "route_probe_count": route_probe_count,
                "physical_tree": tree_stats,
                "verification_bytes": sum(
                    value.verification_bytes for value in observations
                ),
                "profiles": {
                    profile: {
                        "rtt_ewma_ms": _finite_or_none(current.rtt_ewma_ms),
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

    def _physical_tree_stats(self) -> dict[str, Any]:
        provider = getattr(self, "tree_draft_provider", None)
        return {
            "configured": provider is not None,
            "strategy": getattr(provider, "strategy", None),
            "provider_max_draft_tokens": int(
                getattr(provider, "max_draft_tokens", 0)
            ),
            "provider_max_branches": int(getattr(provider, "max_branches", 0)),
            "prepared_waves": int(
                getattr(self, "_physical_tree_prepared_waves", 0)
            ),
            "prepared_leaves": int(
                getattr(self, "_physical_tree_prepared_leaves", 0)
            ),
            "committed_waves": int(
                getattr(self, "_physical_tree_committed_waves", 0)
            ),
            "aborted_waves": int(
                getattr(self, "_physical_tree_aborted_waves", 0)
            ),
            "batch_calls": int(getattr(self, "_physical_tree_batch_calls", 0)),
            "batch_items": int(getattr(self, "_physical_tree_batch_items", 0)),
            "live_leaves": len(
                getattr(self, "_physical_tree_live_children", ())
            ),
            "virtual_routes": len(getattr(self, "_leaf_routes", {})),
            "capacity_quote": {
                "pending": getattr(self, "_pending_tree_reservation", None)
                is not None,
                "requests": int(
                    getattr(self, "_physical_tree_quote_requests", 0)
                ),
                "ready": int(getattr(self, "_physical_tree_quote_ready", 0)),
                "rejected": int(
                    getattr(self, "_physical_tree_quote_rejected", 0)
                ),
                "cancelled": int(
                    getattr(self, "_physical_tree_quote_cancelled", 0)
                ),
                "timeouts": int(
                    getattr(self, "_physical_tree_quote_timeouts", 0)
                ),
                "singleflight_fallbacks": int(
                    getattr(
                        self,
                        "_physical_tree_quote_singleflight_fallbacks",
                        0,
                    )
                ),
                "protocol_failures": int(
                    getattr(self, "_physical_tree_quote_protocol_failures", 0)
                ),
                "average_rtt_ms": (
                    1_000
                    * float(getattr(self, "_physical_tree_quote_rtt_seconds", 0.0))
                    / int(getattr(self, "_physical_tree_quote_ready", 0)
                          + getattr(self, "_physical_tree_quote_rejected", 0))
                    if int(getattr(self, "_physical_tree_quote_ready", 0)
                           + getattr(self, "_physical_tree_quote_rejected", 0))
                    else None
                ),
                "rejections": dict(
                    getattr(self, "_physical_tree_quote_rejections", {})
                ),
            },
        }

    @property
    def root_batch_stats(self) -> dict[str, int | float]:
        return {
            "window_ms": self.config.root_batch_window_ms,
            "ready_items": self._root_ready_items,
            "model_forward_calls": self._root_model_forward_calls,
            "physical_batch_calls": self._root_physical_batch_calls,
            "physical_batch_items": self._root_physical_batch_items,
            "sequential_items": self._root_sequential_items,
            "max_physical_batch_size": self._root_max_physical_batch_size,
        }

    @property
    def prefill_window_stats(self) -> dict[str, int]:
        configured_chunks = _prefill_inflight_chunk_limit(self.config)
        configured_bytes = _prefill_inflight_byte_limit(self.config)
        return {
            "configured_chunks_per_request": configured_chunks,
            "configured_bytes_per_request": configured_bytes,
            "global_chunk_ceiling": configured_chunks
            * int(self.config.max_active_sequences),
            "global_byte_ceiling": configured_bytes
            * int(self.config.max_active_sequences),
            "current_chunks": int(getattr(self, "_prefill_current_chunks", 0)),
            "current_bytes": int(getattr(self, "_prefill_current_bytes", 0)),
            "current_reserved_bytes": int(
                getattr(self, "_prefill_current_reserved_bytes", 0)
            ),
            "high_water_chunks": int(
                getattr(self, "_prefill_high_water_chunks", 0)
            ),
            "high_water_bytes": int(
                getattr(self, "_prefill_high_water_bytes", 0)
            ),
            "high_water_reserved_bytes": int(
                getattr(self, "_prefill_high_water_reserved_bytes", 0)
            ),
            "max_request_chunks": int(
                getattr(self, "_prefill_max_request_chunks", 0)
            ),
            "max_request_bytes": int(
                getattr(self, "_prefill_max_request_bytes", 0)
            ),
            "max_request_reserved_bytes": int(
                getattr(self, "_prefill_max_request_reserved_bytes", 0)
            ),
            "dispatched_chunks": int(
                getattr(self, "_prefill_dispatched_chunks", 0)
            ),
            "completed_chunks": int(
                getattr(self, "_prefill_completed_chunks", 0)
            ),
            "acknowledged_chunks": int(
                getattr(self, "_prefill_acknowledged_chunks", 0)
            ),
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
            self._validate_prefill_credit_capacity(request)
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

    def _validate_prefill_credit_capacity(self, request: GenerationInput) -> None:
        byte_limit = _prefill_inflight_byte_limit(self.config)
        if byte_limit == 0:
            return
        total = int(request.input_ids.shape[1])
        configured = self.config.prefill_chunk_tokens
        token_count = total if configured == 0 else min(total, configured)
        prefill_limit = _prefill_token_limit(self.config)
        if prefill_limit is not None:
            token_count = min(token_count, prefill_limit)
        reservation = _prefill_frame_byte_reservation(
            TensorCodec(self.config.codec),
            token_count,
            self.hidden_size,
        )
        if reservation > byte_limit:
            raise ValueError(
                f"request {request.client_id} prefill chunk reserves {reservation} bytes, "
                f"exceeding prefill_inflight_bytes={byte_limit}"
            )

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
                prior_error = getattr(self, "_close_error", None)
                if prior_error is not None:
                    raise prior_error
                return
            self._closed = True

        cleanup_errors: list[str] = []
        hard_fallback_used = False
        downstream_was_present = self._downstream is not None
        self._scheduler_stop.set()
        self._submission_queue.put(None)

        scheduler = self._scheduler_thread
        cleanup_errors.extend(
            self._join_components(
                (scheduler,),
                SCHEDULER_SHUTDOWN_GRACE_SECONDS,
            )
        )
        scheduler_alive = self._component_is_alive(scheduler)
        if not scheduler_alive and downstream_was_present and not self._shutdown_sent:
            # A partially constructed engine may own a connected route without
            # ever starting the scheduler. Preserve the same protocol ordering.
            try:
                send_frame(
                    self._require_socket(self._downstream, "downstream"),
                    FrameType.SHUTDOWN,
                    0,
                )
                self._shutdown_sent = True
            except BaseException as error:
                cleanup_errors.append(
                    f"SHUTDOWN send failed: {type(error).__name__}: {error}"
                )

        protocol_shutdown_ready = (
            not downstream_was_present or self._shutdown_sent
        )
        if not scheduler_alive and protocol_shutdown_ready:
            # sendall(SHUTDOWN) completed. A write half-close preserves TCP
            # ordering while making it impossible for later cleanup to overtake
            # the protocol frame with a reset.
            self._half_close_downstream_write()
            cleanup_errors.extend(
                self._join_components(
                    tuple(self._processes),
                    CHILD_SHUTDOWN_GRACE_SECONDS,
                )
            )
            cleanup_errors.extend(
                self._join_components(
                    (self._receiver_thread, self._control_thread),
                    IO_THREAD_SHUTDOWN_GRACE_SECONDS,
                )
            )

        live_processes = [
            process for process in self._processes if self._component_is_alive(process)
        ]
        live_threads = [
            thread
            for thread in (
                self._scheduler_thread,
                self._receiver_thread,
                self._control_thread,
            )
            if self._component_is_alive(thread)
        ]
        if live_processes or live_threads or not protocol_shutdown_ready:
            hard_fallback_used = True
            # Only the bounded fallback may issue SHUT_RDWR. It interrupts a
            # scheduler or reader that did not honor the graceful deadline.
            self._interrupt_transports()
            self._received_frames.put(RuntimeError("pipeline hard shutdown"))
            cleanup_errors.extend(
                self._join_components(
                    tuple(live_threads),
                    HARD_SHUTDOWN_GRACE_SECONDS,
                )
            )
            for process in live_processes:
                if not self._component_is_alive(process):
                    continue
                try:
                    process.terminate()
                except BaseException as error:
                    cleanup_errors.append(
                        f"terminate {getattr(process, 'name', 'child')} failed: "
                        f"{type(error).__name__}: {error}"
                    )
            cleanup_errors.extend(
                self._join_components(
                    tuple(live_processes),
                    HARD_SHUTDOWN_GRACE_SECONDS,
                )
            )

        self._close_transport_handles(cleanup_errors)
        # Closing handles is the final bounded unblock for readers. It is safe
        # here because the graceful child deadline has already elapsed.
        cleanup_errors.extend(
            self._join_components(
                (
                    self._scheduler_thread,
                    self._receiver_thread,
                    self._control_thread,
                ),
                HARD_SHUTDOWN_GRACE_SECONDS,
            )
        )
        if self._metrics_queue is not None:
            try:
                self.stage_metrics.extend(drain_metrics(self._metrics_queue))
            except BaseException as error:
                cleanup_errors.append(
                    f"metric drain failed: {type(error).__name__}: {error}"
                )
        if not self._component_is_alive(self._scheduler_thread):
            runner = self._runner
            self._runner = None
            if runner is not None:
                close = getattr(runner, "close", None)
                if callable(close):
                    try:
                        close()
                    except BaseException as error:
                        cleanup_errors.append(
                            f"root runner close failed: {type(error).__name__}: {error}"
                        )

        child_processes = [
            {
                "name": str(getattr(process, "name", "unknown")),
                "pid": getattr(process, "pid", None),
                "exit_code": getattr(process, "exitcode", None),
                "alive": self._component_is_alive(process),
            }
            for process in self._processes
        ]
        threads_alive = {
            "scheduler": self._component_is_alive(self._scheduler_thread),
            "token_return": self._component_is_alive(self._receiver_thread),
            "upstream_control": self._component_is_alive(self._control_thread),
        }
        fatal_stage_metrics = [
            copy.deepcopy(metric)
            for metric in self.stage_metrics
            if isinstance(metric, dict) and "fatal_error" in metric
        ]
        children_clean = all(
            not child["alive"] and child["exit_code"] == 0
            for child in child_processes
        )
        clean = (
            protocol_shutdown_ready
            and not hard_fallback_used
            and children_clean
            and not any(threads_alive.values())
            and not fatal_stage_metrics
            and not cleanup_errors
        )
        report = {
            "schema": "gdlp-pipeline-shutdown/1",
            "clean": clean,
            "shutdown_required": downstream_was_present,
            "shutdown_sent": bool(self._shutdown_sent) or not downstream_was_present,
            "hard_fallback_used": hard_fallback_used,
            "child_processes": child_processes,
            "threads_alive": threads_alive,
            "fatal_stage_metrics": fatal_stage_metrics,
            "cleanup_errors": cleanup_errors,
        }
        self._shutdown_report = report
        if not clean:
            error = PipelineShutdownError(report)
            self._close_error = error
            raise error

    def __enter__(self) -> "DistributedPipelineEngine":
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        _traceback: object,
    ) -> None:
        if exc_value is None:
            self.close()
        else:
            self._close_preserving_exception(exc_value)

    def _close_preserving_exception(self, original: BaseException) -> None:
        """Run cleanup without replacing an exception already in flight."""

        try:
            self.close()
        except BaseException as cleanup_error:
            add_note = getattr(original, "add_note", None)
            if callable(add_note):
                add_note(
                    "Secondary pipeline cleanup failure: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )

    def _start(self) -> None:
        config = self.config
        boundaries = config.boundaries
        ram_stage_configs = config.ram_backed_moe_stages or tuple(
            None for _ in range(self.stages)
        )
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
                ram_stage_config = ram_stage_configs[stage_index]
                stage_artifact_identity = (
                    ram_stage_config.artifact_identity
                    if ram_stage_config is not None
                    else self.model_artifact.identity
                )
                stage_canonical_source = (
                    f"content-addressed://{stage_artifact_identity}"
                    if ram_stage_config is not None
                    else self.model_artifact.canonical_source
                )
                child_configs.append(
                    StageProcessConfig(
                        spec=StageModelSpec(
                            self.model_snapshot,
                            boundaries[stage_index],
                            boundaries[stage_index + 1],
                            self.total_layers,
                            config.threads_per_stage,
                            artifact_identity=stage_artifact_identity,
                            canonical_model_source=stage_canonical_source,
                            canonical_model_revision=(
                                None
                                if ram_stage_config is not None
                                else self.model_artifact.canonical_revision
                            ),
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
                        device=config.device,
                        connect_timeout_seconds=config.startup_timeout_seconds,
                        sealed_wave_tokens=config.sealed_wave_tokens,
                        max_prefill_chunk_tokens=config.max_prefill_chunk_tokens,
                        max_speculative_branches=config.max_speculative_branches,
                        max_speculative_branch_tokens=(
                            config.max_speculative_branch_tokens
                        ),
                        max_speculative_kv_bytes=config.max_speculative_kv_bytes,
                        ram_backed_moe=ram_stage_config,
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
        root_ram_config = ram_stage_configs[0]
        root_spec = StageModelSpec(
            self.model_snapshot,
            0,
            boundaries[1],
            self.total_layers,
            config.threads_per_stage,
            artifact_identity=self.model_artifact.identity,
            canonical_model_source=self.model_artifact.canonical_source,
            canonical_model_revision=self.model_artifact.canonical_revision,
        )
        self._runner = (
            StageRunner(root_spec, device=config.device)
            if root_ram_config is None
            else build_ram_backed_moe_stage_runner(
                root_spec,
                root_ram_config,
                pipeline_snapshot_identity=self.pipeline_id,
            )
        )
        if self._runner.hidden_size != self.hidden_size:
            raise RuntimeError("root stage hidden size differs from the model configuration")
        validate_speculative_runner(config, self._runner)

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
        if config.route_probe_interval_seconds > 0:
            started = time.perf_counter()
            probe_timeout = min(
                config.startup_timeout_seconds,
                config.route_probe_timeout_seconds,
            )
            return_socket.settimeout(probe_timeout)
            downstream.settimeout(probe_timeout)
            try:
                send_frame(
                    downstream,
                    FrameType.PING,
                    self.pipeline_id,
                    step=0,
                    emulator=LinkEmulator(
                        config.one_way_delay_ms,
                        config.bandwidth_mbps,
                    ),
                )
                readable, _, _ = select.select(
                    (return_socket, downstream),
                    (),
                    (),
                    probe_timeout,
                )
                if not readable:
                    raise TimeoutError("pipeline route probe timed out during startup")
                if downstream in readable:
                    control = recv_frame(downstream)
                    if control.frame_type == FrameType.ERROR:
                        raise RuntimeError(
                            control.payload.decode("utf-8", errors="replace")
                        )
                    raise RuntimeError(
                        "pipeline route probe received unexpected upstream control "
                        f"{control.frame_type.name}"
                    )
                pong = recv_frame(return_socket)
            except socket.timeout as error:
                raise TimeoutError("pipeline route probe timed out during startup") from error
            finally:
                return_socket.settimeout(None)
                downstream.settimeout(None)
            arrived = time.perf_counter()
            if pong.frame_type != FrameType.PONG:
                raise RuntimeError(
                    f"pipeline route probe returned {pong.frame_type.name}, expected PONG"
                )
            if pong.request_id != self.pipeline_id:
                raise RuntimeError("pipeline route PONG has a different model identity")
            if pong.step != 0:
                raise RuntimeError("pipeline route PONG has a different probe identity")
            self._record_route_rtt((arrived - started) * 1_000)
            with self._route_probe_lock:
                self._route_probe_next_at = (
                    arrived + config.route_probe_interval_seconds
                )
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
                self._check_route_probe_timeout()
                if not active:
                    try:
                        idle_value = self._received_frames.get_nowait()
                    except queue.Empty:
                        idle_value = None
                    if idle_value is not None:
                        self._raise_idle_return(idle_value)
                    batch = self._next_submission(timeout=0.1)
                    if batch is None:
                        if self._scheduler_stop.is_set():
                            break
                        # Probe only a drained route. Live wave latency already
                        # captures compute/queue pressure; feeding that delay to
                        # the network RTT depth cap would use the wrong signal.
                        self._send_route_probe_if_due(downstream, emulator)
                        continue
                    self._admit_batch(batch, active, runner, downstream, emulator)
                    decode_since_admission = 0
                    continue

                self._send_requested_cancellations(active, runner, downstream)
                if not active:
                    continue
                if (
                    getattr(self, "_pending_tree_reservation", None) is None
                    and len(active) < self.config.max_active_sequences
                    and (
                        decode_since_admission >= max(1, len(active))
                        or any(
                            not job.cancel_requested.is_set()
                            and job.prefill_offset
                            < int(job.request.input_ids.shape[1])
                            for job in active.values()
                        )
                    )
                ):
                    batch = self._next_submission(timeout=0.0)
                    if batch is not None:
                        self._admit_batch(batch, active, runner, downstream, emulator)
                        decode_since_admission = 0
                        continue

                self._check_pipeline_timeouts(active)
                try:
                    value = self._received_frames.get_nowait()
                except queue.Empty:
                    # Fill at most one credit per request and scheduler turn.
                    # Re-entering the loop between rounds lets returns,
                    # cancellation and newly admissible work win over a long
                    # prompt before its entire window is filled.
                    if self._dispatch_prefill_credit_round(
                        active,
                        runner,
                        downstream,
                        emulator,
                    ):
                        continue
                    try:
                        value = self._received_frames.get(timeout=0.05)
                    except queue.Empty:
                        value = None
                if value is None:
                    # An idle return channel is also an admission point. This lets a
                    # new chat enter while another request is between network hops.
                    batch = (
                        self._next_submission(timeout=0.0)
                        if (
                            getattr(self, "_pending_tree_reservation", None) is None
                            and len(active) < self.config.max_active_sequences
                        )
                        else None
                    )
                    if batch is not None:
                        self._admit_batch(batch, active, runner, downstream, emulator)
                        decode_since_admission = 0
                    continue
                ready_values = self._collect_ready_return_values(value)
                prepared: list[
                    _PreparedRootWave
                    | _PreparedPhysicalTreeWave
                    | _CommittedPhysicalTreeWave
                ] = []
                for ready_value in ready_values:
                    wave = self._handle_return_value(
                        ready_value,
                        active,
                        runner,
                        downstream,
                    )
                    if wave is not None:
                        prepared.append(wave)
                self._dispatch_root_waves(
                    prepared,
                    runner,
                    downstream,
                    emulator,
                )
                decode_since_admission += len(ready_values)
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
            # Some focused unit tests construct an engine through ``__new__``;
            # keep teardown safe for those partial objects.
            retained = getattr(self, "_retained_sessions", None)
            if retained:
                for session in list(retained.values()):
                    try:
                        runner.end(session.wire_id)
                    except BaseException:
                        pass
            if retained is not None:
                retained.clear()
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

    def _collect_ready_return_values(
        self,
        first: tuple[Any, float] | BaseException,
    ) -> tuple[tuple[Any, float] | BaseException, ...]:
        """Reconstruct one downstream completion wave without blocking fairness."""

        values: list[tuple[Any, float] | BaseException] = [first]
        limit = max(1, self.config.max_active_sequences)
        deadline = time.monotonic() + self.config.root_batch_window_ms / 1_000
        while len(values) < limit:
            try:
                values.append(self._received_frames.get_nowait())
                continue
            except queue.Empty:
                pass
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                values.append(self._received_frames.get(timeout=remaining))
            except queue.Empty:
                break
        return tuple(values)

    def _admit_batch(
        self,
        batch: list[_GenerationJob],
        active: dict[int, _GenerationJob],
        runner: StageRunner,
        downstream: socket.socket,
        emulator: LinkEmulator,
    ) -> None:
        if getattr(self, "_pending_tree_reservation", None) is not None:
            self._deferred_batches.appendleft(batch)
            return
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
                # A retained session keeps its wire id, step counter and KV alive
                # on every stage, so it must skip BEGIN and resume from the
                # already-computed prefix. A miss falls back to a fresh request.
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
            except BaseException as error:
                for remaining in selected[index + 1 :]:
                    self._retire_job(remaining, runner, exception=error, began=False)
                raise
        self._dispatch_prefill_credit_round(active, runner, downstream, emulator)

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
        # Partial engines built through ``__new__`` in focused unit tests never
        # own this map; treat their absence as "nothing retained".
        if not getattr(self, "_retained_sessions", None):
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
        runner: StageRunner,
        downstream: socket.socket,
    ) -> None:
        for wire_id, job in list(active.items()):
            if job.cancel_requested.is_set() and not job.cancel_sent:
                pending_quote = getattr(self, "_pending_tree_reservation", None)
                if (
                    pending_quote is not None
                    and pending_quote.prepared.job is not job
                ):
                    # Root and remote cache mutations remain frozen until the
                    # quoted transaction is acknowledged or cancelled.
                    continue
                if (
                    pending_quote is not None
                    and pending_quote.prepared.job is job
                ):
                    self._cancel_pending_tree_reservation(
                        job,
                        downstream,
                        reason="generation cancelled while tree capacity quote was pending",
                    )
                coordinator = getattr(self, "_physical_tree", None)
                if (
                    coordinator is not None
                    and wire_id in coordinator.tree_by_parent
                ):
                    cancellation = coordinator.request_cancel(wire_id)
                    if cancellation.deferred_until_commit:
                        # The return handler owns the already-started ordered
                        # transition and will emit the parent CANCEL last.
                        job.cancel_sent = True
                        job.physical_tree_pending_exception = (
                            GenerationCancelledError("generation cancelled")
                        )
                        continue
                    self._apply_physical_tree_cancel_commands(
                        job,
                        cancellation.commands,
                        runner,
                        downstream,
                    )
                    job.physical_tree_pending_exception = (
                        GenerationCancelledError("generation cancelled")
                    )
                    self._sync_physical_leaf_routes()
                    if not cancellation.draining_virtual_request_ids:
                        self._retire_terminal_physical_tree(wire_id)
                        active.pop(wire_id, None)
                        pending = job.physical_tree_pending_exception
                        job.physical_tree_pending_exception = None
                        self._retire_job(
                            job,
                            runner,
                            exception=pending,
                            began=not job.physical_tree_parent_closed,
                        )
                    continue
                # CANCEL follows every already-sent activation on the same TCP
                # stream. Each outstanding return remains an authenticated FIFO
                # tombstone and must be drained before retiring local request state.
                send_frame(downstream, FrameType.CANCEL, wire_id)
                job.cancel_sent = True
                if not job.inflight_waves:
                    active.pop(wire_id)
                    self._retire_job(
                        job,
                        runner,
                        exception=GenerationCancelledError("generation cancelled"),
                    )

    def _cancel_pending_tree_reservation(
        self,
        job: _GenerationJob,
        downstream: socket.socket,
        *,
        reason: str,
        count_metric: bool = True,
    ) -> _PreparedPhysicalTreeWave | None:
        """Release a quote before parent CANCEL; no root/wire KV has forked yet."""

        pending = getattr(self, "_pending_tree_reservation", None)
        if pending is None or pending.prepared.job is not job:
            return None
        parent_request_id = pending.prepared.job.wire_id
        if parent_request_id is None:
            raise RuntimeError("pending tree reservation lost its parent identity")
        send_frame(
            downstream,
            FrameType.TREE_RESERVATION_CANCEL,
            parent_request_id,
            step=pending.prepared.step,
            payload=tree_reservation_payload(pending.nonce),
        )
        proposal = pending.prepared.proposal
        if proposal.tree.state is MacroWaveState.OPEN:
            proposal.tree.rollback()
        self._pending_tree_reservation = None
        if count_metric:
            self._physical_tree_quote_cancelled = int(
                getattr(self, "_physical_tree_quote_cancelled", 0)
            ) + 1
        del reason  # retained at call sites for audit-readable cancellation intent
        return pending.prepared

    def _dispatch_prefill_credit_round(
        self,
        active: dict[int, _GenerationJob],
        runner: StageRunner,
        downstream: socket.socket,
        emulator: LinkEmulator,
    ) -> int:
        """Dispatch at most one ordered prefill chunk per active request."""

        if getattr(self, "_pending_tree_reservation", None) is not None:
            return 0

        prepared: list[_PreparedRootWave] = []
        for job in active.values():
            if job.cancel_requested.is_set():
                continue
            total = int(job.request.input_ids.shape[1])
            if job.prefill_offset >= total:
                continue
            inflight_prefill = sum(
                1 for wave in job.inflight_waves if wave.is_prefill
            )
            if inflight_prefill >= _prefill_inflight_chunk_limit(self.config):
                continue
            wave = self._prepare_next_prefill_chunk(job)
            byte_limit = _prefill_inflight_byte_limit(self.config)
            if (
                byte_limit > 0
                and job.prefill_reserved_bytes + wave.reserved_bytes > byte_limit
            ):
                continue
            prepared.append(wave)
        self._dispatch_root_waves(prepared, runner, downstream, emulator)
        return len(prepared)

    def _consume_inflight_return(
        self,
        job: _GenerationJob,
        frame: Any,
    ) -> _InflightWave:
        if not job.inflight_waves:
            raise RuntimeError(
                f"request {frame.request_id} returned step {frame.step} with no wave in flight"
            )
        flight = job.inflight_waves[0]
        if frame.step != flight.step:
            raise RuntimeError(
                f"request {frame.request_id} returned step {frame.step}, "
                f"expected FIFO step {flight.step}"
            )
        expected_type = {
            FrameType.PREFILL: FrameType.PREFILL_ACK,
            FrameType.ACTIVATION: FrameType.TOKEN,
            FrameType.VERIFY: FrameType.VERIFY_RESULT,
        }.get(flight.frame_type)
        if frame.frame_type != expected_type:
            expected_name = expected_type.name if expected_type is not None else "none"
            raise RuntimeError(
                f"request {frame.request_id} returned {frame.frame_type.name} for "
                f"{flight.frame_type.name}, expected {expected_name}"
            )
        job.inflight_waves.popleft()
        next_step = job.next_step
        if next_step is None:
            raise RuntimeError("request lost its next outbound step")
        job.step = (
            job.inflight_waves[0].step if job.inflight_waves else next_step
        )
        if flight.is_prefill:
            if flight.prefill_end is None or flight.prefill_end <= job.prefill_acked_offset:
                raise RuntimeError("prefill completion offsets are not strictly increasing")
            job.prefill_acked_offset = flight.prefill_end
            current_chunks = int(getattr(self, "_prefill_current_chunks", 0))
            current_bytes = int(getattr(self, "_prefill_current_bytes", 0))
            current_reserved = int(
                getattr(self, "_prefill_current_reserved_bytes", 0)
            )
            if (
                job.prefill_inflight_bytes < flight.outbound_bytes
                or job.prefill_reserved_bytes < flight.reserved_bytes
                or current_chunks < 1
                or current_bytes < flight.outbound_bytes
                or current_reserved < flight.reserved_bytes
            ):
                raise RuntimeError("prefill in-flight telemetry underflow")
            job.prefill_inflight_bytes -= flight.outbound_bytes
            job.prefill_reserved_bytes -= flight.reserved_bytes
            self._prefill_current_chunks = current_chunks - 1
            self._prefill_current_bytes = current_bytes - flight.outbound_bytes
            self._prefill_current_reserved_bytes = (
                current_reserved - flight.reserved_bytes
            )
            self._prefill_completed_chunks = int(
                getattr(self, "_prefill_completed_chunks", 0)
            ) + 1
            if frame.frame_type == FrameType.PREFILL_ACK:
                self._prefill_acknowledged_chunks = int(
                    getattr(self, "_prefill_acknowledged_chunks", 0)
                ) + 1
        return flight

    def _handle_return_value(
        self,
        value: tuple[Any, float] | BaseException,
        active: dict[int, _GenerationJob],
        runner: StageRunner,
        downstream: socket.socket,
    ) -> (
        _PreparedRootWave
        | _PreparedPhysicalTreeWave
        | _CommittedPhysicalTreeWave
        | None
    ):
        if isinstance(value, BaseException):
            raise RuntimeError(f"pipeline connection failed: {value}") from value
        frame, arrived = value
        if frame.frame_type == FrameType.ERROR:
            raise RuntimeError(frame.payload.decode("utf-8", errors="replace"))
        if frame.frame_type not in (
            FrameType.PREFILL_ACK,
            FrameType.TOKEN,
            FrameType.VERIFY_RESULT,
            FrameType.TREE_PREPARE_RESULT,
            FrameType.TREE_RESERVATION_COMMIT_RESULT,
        ):
            raise RuntimeError(f"unexpected return frame {frame.frame_type.name}")
        if frame.frame_type == FrameType.TREE_PREPARE_RESULT:
            return self._handle_tree_prepare_result(
                frame,
                arrived,
                active,
                runner,
                downstream,
            )
        if frame.frame_type == FrameType.TREE_RESERVATION_COMMIT_RESULT:
            return self._handle_tree_commit_result(
                frame,
                arrived,
                active,
                runner,
                downstream,
            )
        if frame.request_id in getattr(self, "_leaf_routes", {}):
            if frame.frame_type != FrameType.VERIFY_RESULT:
                raise RuntimeError(
                    f"virtual leaf {frame.request_id} returned {frame.frame_type.name}"
                )
            return self._handle_physical_tree_return(
                frame,
                arrived,
                active,
                runner,
                downstream,
            )
        job = active.get(frame.request_id)
        if job is None:
            raise RuntimeError(f"token for unknown request {frame.request_id}")
        flight = self._consume_inflight_return(job, frame)
        if job.cancel_requested.is_set():
            if not job.cancel_sent:
                send_frame(downstream, FrameType.CANCEL, frame.request_id)
                job.cancel_sent = True
            if not job.inflight_waves:
                active.pop(frame.request_id)
                # A cancelled turn never retains: CANCEL already freed the KV on
                # every stage, so the session bookkeeping must forget this chat.
                self._drop_job_session(job)
                self._retire_job(
                    job,
                    runner,
                    exception=GenerationCancelledError("generation cancelled"),
                )
            return None

        if frame.frame_type == FrameType.PREFILL_ACK:
            if (
                flight.prefill_end is None
                or flight.prefill_end >= int(job.request.input_ids.shape[1])
            ):
                raise RuntimeError(
                    f"request {frame.request_id} returned an unexpected prefill ACK"
                )
            return None

        if frame.frame_type == FrameType.VERIFY_RESULT:
            proposal = job.verify_proposal
            if proposal is None:
                raise RuntimeError(
                    f"request {frame.request_id} returned verification without a draft"
                )
            targets = decode_verify_result(frame)
            resolution = resolve_linear_macro_wave(proposal, targets)
            if self.speculation_controller is None:
                raise RuntimeError("verification returned while speculation is disabled")
            with self._speculation_lock:
                controller = self._speculation_controller_for_profile_locked(
                    job.speculation_profile
                )
                record_linear_resolution(
                    controller,
                    proposal,
                    resolution,
                    latency_seconds=max(1e-9, arrived - flight.started_at),
                    transferred_bytes=(
                        flight.outbound_bytes + HEADER_BYTES + len(frame.payload)
                    ),
                )
            verify_base_tokens = job.verify_base_tokens
            job.verify_proposal = None
            job.verify_base_tokens = 0
            appended_before = len(job.token_ids)
            reason = self._append_verified_tokens(
                job,
                resolution.emitted_tokens,
                arrived,
            )
            appended = len(job.token_ids) - appended_before
            # Draft positions in the KV are valid only up to the accepted
            # prefix AND only as far as tokens were actually emitted.
            # truncate_draft_to is the committed draft count by construction.
            job.kv_valid += min(resolution.truncate_draft_to, appended)
            if reason is not None:
                self._finish_turn(job, active, runner, downstream, reason)
                return None

            if resolution.truncate_required:
                keep_tokens = (
                    verify_base_tokens
                    + 1
                    + resolution.truncate_draft_to
                )
                runner.truncate(frame.request_id, keep_tokens)
                send_frame(
                    downstream,
                    FrameType.TRUNCATE,
                    frame.request_id,
                    token_count=keep_tokens,
                )
            return self._prepare_decode_wave(
                job,
                runner,
                active_sequences=len(active),
            )

        if job.verify_proposal is not None:
            raise RuntimeError(
                f"request {frame.request_id} returned TOKEN for a verification wave"
            )
        prior_output_tokens = len(job.token_ids)
        token = decode_token(frame)
        job.token_ids.append(token)
        job.arrivals.append(arrived)
        if job.callback is not None:
            try:
                job.callback(
                    job.request.client_id,
                    token,
                    prior_output_tokens,
                    arrived,
                )
            except BaseException:
                # Streaming is observational; a disconnected client must not
                # poison the shared model pipeline for other users.
                pass
        if self.speculation_controller is not None and prior_output_tokens > 0:
            with self._speculation_lock:
                controller = self._speculation_controller_for_profile_locked(
                    job.speculation_profile
                )
                controller.record_classic(
                    latency_seconds=max(1e-9, arrived - flight.started_at),
                    transferred_bytes=(
                        flight.outbound_bytes + HEADER_BYTES + len(frame.payload)
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
            return None

        return self._prepare_decode_wave(
            job,
            runner,
            active_sequences=len(active),
        )

    def _handle_tree_prepare_result(
        self,
        frame: Any,
        arrived: float,
        active: dict[int, _GenerationJob],
        runner: StageRunnerContract,
        downstream: socket.socket,
    ) -> _PreparedRootWave | _CommittedPhysicalTreeWave | None:
        pending = getattr(self, "_pending_tree_reservation", None)
        try:
            if pending is None:
                raise RuntimeError("late or unsolicited TREE_PREPARE_RESULT")
            prepared = pending.prepared
            job = prepared.job
            parent_request_id = job.wire_id
            if parent_request_id is None:
                raise RuntimeError("pending tree quote lost its parent identity")
            if pending.commit_sent:
                raise RuntimeError("duplicate TREE_PREPARE_RESULT after COMMIT send")
            if arrived > pending.deadline_at:
                raise TimeoutError("TREE_PREPARE_RESULT arrived after its deadline")
            quote = decode_tree_prepare(frame)
            if frame.request_id != parent_request_id:
                raise RuntimeError("TREE_PREPARE_RESULT parent identity mismatch")
            if frame.step != prepared.step:
                raise RuntimeError("TREE_PREPARE_RESULT step mismatch")
            if quote.nonce != pending.nonce:
                raise RuntimeError("TREE_PREPARE_RESULT nonce mismatch")
            if quote.path_lengths != pending.path_lengths:
                raise RuntimeError("TREE_PREPARE_RESULT path shape mismatch")
            expected_remote_stages = self.stages - 1
            if quote.stage_count != expected_remote_stages:
                raise RuntimeError(
                    "TREE_PREPARE_RESULT stage count mismatch: "
                    f"got {quote.stage_count}, expected {expected_remote_stages}"
                )
            self._consume_queued_tree_prepare_result(frame, arrived)
        except BaseException:
            self._physical_tree_quote_protocol_failures = int(
                getattr(self, "_physical_tree_quote_protocol_failures", 0)
            ) + 1
            raise

        self._physical_tree_quote_rtt_seconds = float(
            getattr(self, "_physical_tree_quote_rtt_seconds", 0.0)
        ) + max(0.0, arrived - pending.sent_at)
        if job.cancel_requested.is_set():
            self._cancel_pending_tree_reservation(
                job,
                downstream,
                reason="generation cancelled before tree quote consumption",
            )
            send_frame(downstream, FrameType.CANCEL, parent_request_id)
            job.cancel_sent = True
            active.pop(parent_request_id, None)
            self._retire_job(
                job,
                runner,
                exception=GenerationCancelledError("generation cancelled"),
            )
            return None

        if quote.status is TreePrepareStatus.REJECT:
            self._physical_tree_quote_rejected = int(
                getattr(self, "_physical_tree_quote_rejected", 0)
            ) + 1
            reason_name = quote.rejection.name.lower()
            reasons = getattr(self, "_physical_tree_quote_rejections", {})
            reasons[reason_name] = int(reasons.get(reason_name, 0)) + 1
            self._cancel_pending_tree_reservation(
                job,
                downstream,
                reason=f"remote tree capacity rejected: {reason_name}",
                count_metric=False,
            )
            return self._prepare_linear_or_classic_decode_wave(
                job,
                runner,
                active_sequences=len(active),
            )
        if quote.status is not TreePrepareStatus.READY:
            self._physical_tree_quote_protocol_failures = int(
                getattr(self, "_physical_tree_quote_protocol_failures", 0)
            ) + 1
            raise RuntimeError("TREE_PREPARE_RESULT has an unknown status")

        observed = self._root_tree_capacity_snapshot(
            job,
            prepared.proposal,
            runner,
            base_kv_tokens=prepared.base_kv_tokens,
        )
        if observed is None or observed != pending.root_snapshot:
            self._physical_tree_quote_protocol_failures = int(
                getattr(self, "_physical_tree_quote_protocol_failures", 0)
            ) + 1
            raise RuntimeError("root tree capacity changed before remote COMMIT")
        send_frame(
            downstream,
            FrameType.TREE_RESERVATION_COMMIT,
            parent_request_id,
            step=prepared.step,
            token_count=0,
            payload=tree_reservation_payload(pending.nonce),
        )
        pending.commit_sent = True
        pending.deadline_at = time.perf_counter() + self.config.socket_timeout_seconds
        return None

    def _handle_tree_commit_result(
        self,
        frame: Any,
        arrived: float,
        active: dict[int, _GenerationJob],
        runner: StageRunnerContract,
        downstream: socket.socket,
    ) -> _CommittedPhysicalTreeWave | None:
        """Arm root FORKs only after the final stage acknowledges COMMIT."""

        del active, runner, downstream
        pending = getattr(self, "_pending_tree_reservation", None)
        try:
            if pending is None:
                raise RuntimeError(
                    "late or unsolicited TREE_RESERVATION_COMMIT_RESULT"
                )
            if not pending.commit_sent:
                raise RuntimeError("tree COMMIT result arrived before COMMIT send")
            if pending.committed:
                raise RuntimeError("duplicate TREE_RESERVATION_COMMIT_RESULT")
            if arrived > pending.deadline_at:
                raise TimeoutError(
                    "TREE_RESERVATION_COMMIT_RESULT arrived after its deadline"
                )
            prepared = pending.prepared
            parent_request_id = prepared.job.wire_id
            if parent_request_id is None:
                raise RuntimeError("pending tree commit lost its parent identity")
            if frame.request_id != parent_request_id:
                raise RuntimeError("tree COMMIT result parent identity mismatch")
            if frame.step != prepared.step:
                raise RuntimeError("tree COMMIT result step mismatch")
            if decode_tree_reservation_nonce(frame) != pending.nonce:
                raise RuntimeError("tree COMMIT result nonce mismatch")
            expected_remote_stages = self.stages - 1
            if frame.token_count != expected_remote_stages:
                raise RuntimeError(
                    "tree COMMIT result stage count mismatch: "
                    f"got {frame.token_count}, expected {expected_remote_stages}"
                )
            self._consume_queued_tree_commit_result(frame, arrived)
        except BaseException:
            self._physical_tree_quote_protocol_failures = int(
                getattr(self, "_physical_tree_quote_protocol_failures", 0)
            ) + 1
            raise

        pending.committed = True
        self._physical_tree_quote_ready = int(
            getattr(self, "_physical_tree_quote_ready", 0)
        ) + 1
        return _CommittedPhysicalTreeWave(pending)

    def _handle_physical_tree_return(
        self,
        frame: Any,
        arrived: float,
        active: dict[int, _GenerationJob],
        runner: StageRunnerContract,
        downstream: socket.socket,
    ) -> _PreparedRootWave | _PreparedPhysicalTreeWave | None:
        self._consume_queued_tree_return(frame.request_id, arrived)
        parent_request_id = self._leaf_routes.get(frame.request_id)
        if parent_request_id is None:
            raise RuntimeError(f"return for unknown virtual leaf {frame.request_id}")
        job = active.get(parent_request_id)
        if job is None:
            raise RuntimeError(
                f"virtual leaf {frame.request_id} lost parent {parent_request_id}"
            )
        targets = decode_verify_result(frame)
        outcome = self._physical_tree.accept_verify_result(
            frame.request_id,
            step=frame.step,
            target_tokens=targets,
            now=arrived,
        )
        self._sync_physical_leaf_routes()
        if outcome is None:
            return None
        if isinstance(outcome, TombstoneDrain):
            if outcome.wave_drained:
                self._retire_terminal_physical_tree(outcome.parent_request_id)
                self._finish_deferred_tree_retirement(
                    job,
                    active,
                    runner,
                )
            return None
        if isinstance(outcome, TreeAbortPlan):
            self._physical_tree_aborted_waves = int(
                getattr(self, "_physical_tree_aborted_waves", 0)
            ) + 1
            if outcome.route_fatal:
                raise RuntimeError(
                    "physical tree route is inconsistent and must be replayed: "
                    f"{outcome.reason}"
                )
            for command in outcome.commands:
                deferred_cancel = self._apply_physical_tree_cleanup_command(
                    outcome.parent_request_id,
                    command,
                    runner,
                    downstream,
                )
                if deferred_cancel is not None:
                    self._apply_physical_tree_cancel_commands(
                        job,
                        (deferred_cancel,),
                        runner,
                        downstream,
                    )
            self._retire_terminal_physical_tree(outcome.parent_request_id)
            return self._prepare_classic_decode_wave(job)
        if not isinstance(outcome, TreeCommitPlan):
            raise TypeError("physical tree coordinator returned an unknown plan")

        if outcome.parent_request_id != parent_request_id:
            raise RuntimeError("physical tree commit belongs to another parent")
        if job.inflight_waves:
            raise RuntimeError("physical tree parent unexpectedly has a wave in flight")

        # Cancellation during an ordered commit is deferred by the state
        # machine until END losers -> PROMOTE -> TRUNCATE is fully enqueued.
        if job.cancel_requested.is_set():
            self._physical_tree.request_cancel(parent_request_id)
        deferred_parent_cancel: CancelCommand | None = None
        for command in outcome.commands:
            if job.cancel_requested.is_set():
                self._physical_tree.request_cancel(parent_request_id)
            planned_cancel = self._apply_physical_tree_cleanup_command(
                parent_request_id,
                command,
                runner,
                downstream,
            )
            if planned_cancel is not None:
                if deferred_parent_cancel is not None:
                    raise RuntimeError("physical tree emitted duplicate parent cancellation")
                deferred_parent_cancel = planned_cancel

        job.step = outcome.next_step
        job.next_step = outcome.next_step
        expected_parent_tokens = (
            self._physical_tree.wave(parent_request_id).base_kv_tokens
            + 1
            + outcome.resolution.accepted_draft_tokens
        )
        if runner.sequence_length(parent_request_id) != expected_parent_tokens:
            raise RuntimeError("physical tree commit produced the wrong parent KV length")

        if job.cancel_requested.is_set() and deferred_parent_cancel is None:
            cancellation = self._physical_tree.request_cancel(parent_request_id)
            if cancellation.commands:
                if len(cancellation.commands) != 1:
                    raise RuntimeError("committed tree cancellation must target only parent")
                deferred_parent_cancel = cancellation.commands[0]
        if deferred_parent_cancel is not None:
            self._apply_physical_tree_cancel_commands(
                job,
                (deferred_parent_cancel,),
                runner,
                downstream,
            )

        self._sync_physical_leaf_routes()
        self._physical_tree_committed_waves = int(
            getattr(self, "_physical_tree_committed_waves", 0)
        ) + 1
        self._retire_terminal_physical_tree(parent_request_id)
        if job.cancel_requested.is_set():
            active.pop(parent_request_id, None)
            self._retire_job(
                job,
                runner,
                exception=GenerationCancelledError("generation cancelled"),
                began=not job.physical_tree_parent_closed,
            )
            return None

        # Publishing happens strictly after every physical mutation above.
        reason = self._append_verified_tokens(
            job,
            outcome.resolution.emitted_tokens,
            arrived,
        )
        if reason is not None:
            send_frame(downstream, FrameType.END, parent_request_id)
            active.pop(parent_request_id)
            self._retire_job(
                job,
                runner,
                result=self._generation_output(job, reason),
            )
            return None
        return self._prepare_decode_wave(
            job,
            runner,
            active_sequences=len(active),
        )

    def _apply_physical_tree_cleanup_command(
        self,
        parent_request_id: int,
        command: EndCommand | PromoteCommand | TruncateCommand,
        runner: StageRunnerContract,
        downstream: socket.socket,
    ) -> CancelCommand | None:
        if isinstance(command, EndCommand):
            if command.request_id not in self._physical_tree_live_children:
                raise RuntimeError("physical tree END targets a non-live leaf")
            runner.end(command.request_id)
            self._physical_tree_live_children.remove(command.request_id)
            send_frame(downstream, FrameType.END, command.request_id)
        elif isinstance(command, PromoteCommand):
            if command.parent_request_id != parent_request_id:
                raise RuntimeError("physical tree PROMOTE targets another parent")
            if command.child_request_id not in self._physical_tree_live_children:
                raise RuntimeError("physical tree PROMOTE targets a non-live carrier")
            promote_request = getattr(runner, "promote_request", None)
            if not callable(promote_request):
                raise TypeError("root runner lost its exact PROMOTE capability")
            promote_request(command.parent_request_id, command.child_request_id)
            self._physical_tree_live_children.remove(command.child_request_id)
            send_frame(
                downstream,
                FrameType.PROMOTE,
                command.parent_request_id,
                payload=branch_request_payload(command.child_request_id),
            )
        elif isinstance(command, TruncateCommand):
            if command.request_id != parent_request_id:
                raise RuntimeError("physical tree TRUNCATE targets another parent")
            runner.truncate(command.request_id, command.keep_tokens)
            send_frame(
                downstream,
                FrameType.TRUNCATE,
                command.request_id,
                token_count=command.keep_tokens,
            )
        else:
            raise TypeError("unknown physical tree cleanup command")
        deferred = self._physical_tree.confirm_cleanup_command(
            parent_request_id,
            command,
        )
        self._sync_physical_leaf_routes()
        return deferred

    def _apply_physical_tree_cancel_commands(
        self,
        job: _GenerationJob,
        commands: tuple[CancelCommand, ...],
        runner: StageRunnerContract,
        downstream: socket.socket,
    ) -> None:
        parent_request_id = job.wire_id
        if parent_request_id is None:
            raise RuntimeError("physical tree cancellation lost its parent")
        for command in commands:
            request_id = command.request_id
            if request_id in self._physical_tree_live_children:
                runner.end(request_id)
                self._physical_tree_live_children.remove(request_id)
            elif request_id == parent_request_id:
                if not job.physical_tree_parent_closed:
                    runner.end(parent_request_id)
                    job.physical_tree_parent_closed = True
                    with self._callback_lock:
                        self._callback_routes.pop(parent_request_id, None)
                job.cancel_sent = True
            else:
                raise RuntimeError(
                    f"physical tree CANCEL targets inactive request {request_id}"
                )
            send_frame(downstream, FrameType.CANCEL, request_id)
        self._sync_physical_leaf_routes()

    def _retire_terminal_physical_tree(self, parent_request_id: int) -> bool:
        coordinator = getattr(self, "_physical_tree", None)
        if coordinator is None or parent_request_id not in coordinator.tree_by_parent:
            return False
        wave = coordinator.wave(parent_request_id)
        self._sync_physical_leaf_routes()
        if wave.phase not in (TreePhase.COMMITTED, TreePhase.ABORTED):
            return False
        if any(
            virtual_id in self._leaf_routes
            for virtual_id in wave.virtual_routes
        ):
            return False
        coordinator.retire_wave(parent_request_id)
        return True

    def _finish_deferred_tree_retirement(
        self,
        job: _GenerationJob,
        active: dict[int, _GenerationJob],
        runner: StageRunnerContract,
    ) -> None:
        result = job.physical_tree_pending_result
        exception = job.physical_tree_pending_exception
        if result is None and exception is None:
            return
        if job.wire_id is not None:
            active.pop(job.wire_id, None)
        job.physical_tree_pending_result = None
        job.physical_tree_pending_exception = None
        self._retire_job(
            job,
            runner,
            result=result,
            exception=exception,
            began=not job.physical_tree_parent_closed,
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

    def _prepare_decode_wave(
        self,
        job: _GenerationJob,
        runner: StageRunner,
        *,
        active_sequences: int,
    ) -> _PreparedRootWave | _PreparedPhysicalTreeWave:
        if job.wire_id is None or not job.token_ids:
            raise RuntimeError("decode job has no active token history")
        remaining = job.request.max_new_tokens - len(job.token_ids)
        tree_provider = getattr(self, "tree_draft_provider", None)
        coordinator = getattr(self, "_physical_tree", None)
        if (
            tree_provider is not None
            and remaining > 1
            and (
                getattr(self, "_pending_tree_reservation", None) is not None
                or bool(getattr(coordinator, "tree_by_parent", {}))
                or bool(getattr(self, "_physical_tree_live_children", set()))
            )
        ):
            self._physical_tree_quote_singleflight_fallbacks = int(
                getattr(self, "_physical_tree_quote_singleflight_fallbacks", 0)
            ) + 1
            return self._prepare_linear_or_classic_decode_wave(
                job,
                runner,
                active_sequences=active_sequences,
            )
        if (
            tree_provider is not None
            and coordinator is not None
            and remaining > 1
            and job.wire_id not in coordinator.tree_by_parent
        ):
            prompt_history = tuple(
                int(token) for token in job.request.input_ids.reshape(-1).tolist()
            )
            history = (*prompt_history, *job.token_ids)
            base_kv_tokens = runner.sequence_length(job.wire_id)
            if base_kv_tokens != len(history) - 1:
                raise RuntimeError(
                    "physical tree parent KV does not match its visible pending token"
                )
            maximum_context = int(getattr(self, "maximum_context", 0))
            maximum_depth = min(
                remaining - 1,
                self.config.speculative_max_draft_tokens,
                _sealed_wave_token_limit(self.config) - 1,
                self.config.max_speculative_branch_tokens - base_kv_tokens - 1,
                maximum_context - base_kv_tokens - 1,
            )
            available_branches = (
                self.config.max_speculative_branches
                - len(getattr(self, "_physical_tree_live_children", ()))
            )
            if maximum_depth > 0 and available_branches > 0:
                proposal = prepare_tree_macro_wave(
                    tree_provider,
                    history,
                    request_id=job.wire_id,
                    ordinal=job.step,
                    parent_kv_version=KVVersion(base_kv_tokens),
                    max_tokens=maximum_depth,
                    max_branches=available_branches,
                )
                if proposal is not None:
                    if self._physical_tree_preflight(
                        job,
                        proposal,
                        runner,
                        base_kv_tokens=base_kv_tokens,
                    ):
                        return _PreparedPhysicalTreeWave(
                            job=job,
                            proposal=proposal,
                            base_kv_tokens=base_kv_tokens,
                            pending_token=job.token_ids[-1],
                            step=job.next_step if job.next_step is not None else job.step,
                            active_sequences=active_sequences,
                        )
                    if proposal.tree.state is MacroWaveState.OPEN:
                        proposal.tree.rollback()
        return self._prepare_linear_or_classic_decode_wave(
            job,
            runner,
            active_sequences=active_sequences,
        )

    def _prepare_linear_or_classic_decode_wave(
        self,
        job: _GenerationJob,
        runner: StageRunner,
        *,
        active_sequences: int,
    ) -> _PreparedRootWave:
        """Preserve the production linear controller as the exact fallback."""

        if job.wire_id is None or not job.token_ids:
            raise RuntimeError("decode job has no active token history")
        remaining = job.request.max_new_tokens - len(job.token_ids)
        provider = self.draft_provider
        controller = self.speculation_controller
        proposal: MacroWaveProposal | None = None
        verify_base_tokens = 0
        if provider is not None and controller is not None and remaining > 1:
            prompt_history = [
                int(token)
                for token in job.request.input_ids.reshape(-1).tolist()
            ]
            history = (*prompt_history, *job.token_ids)
            verify_base_tokens = runner.sequence_length(job.wire_id)
            profile = _speculation_load_profile(active_sequences)
            job.speculation_profile = profile
            with self._speculation_lock:
                controller = self._speculation_controller_for_profile_locked(profile)
                preparation = prepare_linear_macro_wave(
                    provider,
                    controller,
                    history,
                    request_id=job.wire_id,
                    ordinal=job.step,
                    parent_kv_version=KVVersion(verify_base_tokens),
                    max_tokens=min(
                        remaining - 1,
                        self.config.speculative_max_draft_tokens,
                    ),
                    allow_probe=self.config.speculation_probe,
                )
                decision = preparation.decision
                reason = decision.reason
                self._speculation_decision_reasons[reason] = (
                    self._speculation_decision_reasons.get(reason, 0) + 1
                )
                if decision.enabled:
                    self._speculation_enabled_decisions += 1
                else:
                    self._speculation_disabled_decisions += 1
                selected = len(preparation.selected_draft_tokens)
                if preparation.is_probe:
                    self._speculation_probe_waves += 1
                if selected > 0:
                    self._speculation_selected_sizes[selected] = (
                        self._speculation_selected_sizes.get(selected, 0) + 1
                    )
                proposal = preparation.proposal

        job.verify_proposal = proposal
        if proposal is not None:
            job.verify_base_tokens = verify_base_tokens
            input_tokens = (job.token_ids[-1], *proposal.linear_tokens)
            frame_type = FrameType.VERIFY
        else:
            job.verify_base_tokens = 0
            input_tokens = (job.token_ids[-1],)
            frame_type = FrameType.ACTIVATION
        sealed_wave_token_limit = _sealed_wave_token_limit(self.config)
        if len(input_tokens) > sealed_wave_token_limit:
            raise RuntimeError(
                "prepared decode wave exceeds sealed_wave_tokens: "
                f"{len(input_tokens)} > {sealed_wave_token_limit}"
            )
        next_ids = torch.tensor([input_tokens], dtype=torch.long)
        next_step = job.next_step
        if next_step is None:
            raise RuntimeError("decode job lost its next outbound step")
        if job.inflight_waves:
            raise RuntimeError("decode cannot prepare while another wave is in flight")
        return _PreparedRootWave(
            job=job,
            input_ids=next_ids,
            frame_type=frame_type,
            step=next_step,
        )

    def _prepare_classic_decode_wave(self, job: _GenerationJob) -> _PreparedRootWave:
        """Prepare one greedy token without re-entering a rejected tree."""

        if job.wire_id is None or not job.token_ids:
            raise RuntimeError("decode job has no active token history")
        if job.inflight_waves:
            raise RuntimeError("decode cannot prepare while another wave is in flight")
        next_step = job.next_step
        if next_step is None:
            raise RuntimeError("decode job lost its next outbound step")
        job.verify_proposal = None
        job.verify_base_tokens = 0
        return _PreparedRootWave(
            job=job,
            input_ids=torch.tensor([[job.token_ids[-1]]], dtype=torch.long),
            frame_type=FrameType.ACTIVATION,
            step=next_step,
        )

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
        if current is not base and base.rtt_ewma_ms is not None:
            current.record_rtt(base.rtt_ewma_ms)
        self._speculation_controllers[profile] = current
        return current

    def _route_probe_snapshot(self) -> tuple[float | None, int]:
        # Some focused unit tests construct an engine through ``__new__``. Keep
        # introspection safe for those partial objects while normal engines own
        # the lock and counters from __init__.
        lock = getattr(self, "_route_probe_lock", None)
        if lock is None:
            return getattr(self, "_route_rtt_ms", None), int(
                getattr(self, "_route_probe_count", 0)
            )
        with lock:
            return self._route_rtt_ms, self._route_probe_count

    def _record_route_rtt(self, rtt_ms: float) -> None:
        rtt = float(rtt_ms)
        if not math.isfinite(rtt) or rtt < 0:
            raise ValueError("route RTT must be finite and non-negative")
        with self._route_probe_lock:
            self._route_rtt_ms = rtt
            self._route_probe_count += 1
        base = self.speculation_controller
        if base is None:
            return
        with self._speculation_lock:
            seen: set[int] = set()
            for controller in (base, *self._speculation_controllers.values()):
                identity = id(controller)
                if identity in seen:
                    continue
                seen.add(identity)
                controller.record_rtt(rtt)

    def _send_route_probe_if_due(
        self,
        downstream: socket.socket,
        emulator: LinkEmulator,
    ) -> None:
        interval = self.config.route_probe_interval_seconds
        stop_event = getattr(self, "_scheduler_stop", None)
        if (
            interval <= 0
            or getattr(self, "_closed", False)
            or (stop_event is not None and stop_event.is_set())
        ):
            return
        now = time.perf_counter()
        with self._route_probe_lock:
            if self._route_probe_sent_at is not None or now < self._route_probe_next_at:
                return
            self._route_probe_sequence = (self._route_probe_sequence + 1) % (1 << 32)
            probe_step = self._route_probe_sequence
            self._route_probe_sent_at = now
            self._route_probe_deadline_at = (
                now + self.config.route_probe_timeout_seconds
            )
            self._route_probe_pending_step = probe_step
        try:
            send_frame(
                downstream,
                FrameType.PING,
                self.pipeline_id,
                step=probe_step,
                emulator=emulator,
            )
        except BaseException:
            with self._route_probe_lock:
                self._route_probe_sent_at = None
                self._route_probe_deadline_at = None
                self._route_probe_pending_step = None
            raise

    def _consume_route_pong_locked(self, frame: Any, arrived: float) -> float:
        """Consume a PONG while ``_route_probe_lock`` is held and return RTT."""

        if frame.request_id != self.pipeline_id:
            raise RuntimeError("pipeline route PONG has a different model identity")
        started = self._route_probe_sent_at
        deadline = self._route_probe_deadline_at
        expected_step = self._route_probe_pending_step
        if started is None:
            raise RuntimeError("pipeline returned an unsolicited route PONG")
        if expected_step is not None and frame.step != expected_step:
            raise RuntimeError("pipeline route PONG has a different probe identity")
        self._route_probe_sent_at = None
        self._route_probe_deadline_at = None
        self._route_probe_pending_step = None
        self._route_probe_next_at = (
            arrived + self.config.route_probe_interval_seconds
        )
        if deadline is not None and arrived > deadline:
            raise TimeoutError("pipeline route probe returned after its deadline")
        return (arrived - started) * 1_000

    def _consume_route_pong(self, frame: Any, arrived: float) -> None:
        """Direct PONG path used by startup/focused tests.

        Runtime socket readers enqueue PONGs instead; the scheduler drains
        them atomically with its timeout decision in
        :meth:`_check_route_probe_timeout`.
        """

        with self._route_probe_lock:
            rtt_ms = self._consume_route_pong_locked(frame, arrived)
        self._record_route_rtt(rtt_ms)

    def _check_route_probe_timeout(self) -> None:
        measurements: list[float] = []
        pending_error: BaseException | None = None
        now = time.perf_counter()
        with self._route_probe_lock:
            returns = getattr(self, "_route_probe_returns", None)
            if returns is not None:
                while True:
                    try:
                        frame, arrived = returns.get_nowait()
                    except queue.Empty:
                        break
                    try:
                        measurements.append(
                            self._consume_route_pong_locked(frame, arrived)
                        )
                    except BaseException as error:
                        pending_error = error
                        break

            if pending_error is None and self._route_probe_sent_at is not None:
                deadline = self._route_probe_deadline_at
                if deadline is not None and now > deadline:
                    step = self._route_probe_pending_step
                    self._route_probe_sent_at = None
                    self._route_probe_deadline_at = None
                    self._route_probe_pending_step = None
                    pending_error = TimeoutError(
                        f"pipeline route probe {step} timed out"
                    )

        for rtt_ms in measurements:
            self._record_route_rtt(rtt_ms)
        if pending_error is not None:
            raise pending_error

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
        self._discard_pending_tree_reservation(job)
        self._discard_physical_tree_state(job, runner)
        proposal = job.verify_proposal
        if proposal is not None:
            if proposal.tree.state is MacroWaveState.OPEN:
                proposal.tree.rollback()
            job.verify_proposal = None
            job.verify_base_tokens = 0
        self._discard_inflight_waves(job)
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

    def _discard_pending_tree_reservation(self, job: _GenerationJob) -> None:
        pending = getattr(self, "_pending_tree_reservation", None)
        if pending is None or pending.prepared.job is not job:
            return
        proposal = pending.prepared.proposal
        if proposal.tree.state is MacroWaveState.OPEN:
            proposal.tree.rollback()
        self._pending_tree_reservation = None

    def _discard_physical_tree_state(
        self,
        job: _GenerationJob,
        runner: StageRunnerContract,
    ) -> None:
        """Release root-only leaf state while a fatal route is being torn down."""

        parent_request_id = job.wire_id
        coordinator = getattr(self, "_physical_tree", None)
        if (
            parent_request_id is None
            or coordinator is None
            or parent_request_id not in coordinator.tree_by_parent
        ):
            return
        wave = coordinator.tree_by_parent.pop(parent_request_id)
        if wave.proposal.tree.state is MacroWaveState.OPEN:
            wave.proposal.tree.rollback()
        live_children = getattr(self, "_physical_tree_live_children", set())
        virtual_ids = tuple(wave.virtual_routes)
        lock = getattr(self, "_tree_return_lock", None)
        if lock is None:
            for virtual_id in virtual_ids:
                self._leaf_routes.pop(virtual_id, None)
                getattr(self, "_queued_tree_returns", {}).pop(virtual_id, None)
        else:
            with lock:
                for virtual_id in virtual_ids:
                    self._leaf_routes.pop(virtual_id, None)
                    self._queued_tree_returns.pop(virtual_id, None)
        for virtual_id in virtual_ids:
            coordinator.leaf_route.pop(virtual_id, None)
            if virtual_id in live_children:
                runner.end(virtual_id)
                live_children.remove(virtual_id)

    def _discard_inflight_waves(self, job: _GenerationJob) -> None:
        prefill = tuple(wave for wave in job.inflight_waves if wave.is_prefill)
        if prefill:
            released_bytes = sum(wave.outbound_bytes for wave in prefill)
            released_reserved = sum(wave.reserved_bytes for wave in prefill)
            current_chunks = int(getattr(self, "_prefill_current_chunks", 0))
            current_bytes = int(getattr(self, "_prefill_current_bytes", 0))
            current_reserved = int(
                getattr(self, "_prefill_current_reserved_bytes", 0)
            )
            if (
                current_chunks < len(prefill)
                or current_bytes < released_bytes
                or current_reserved < released_reserved
            ):
                raise RuntimeError("prefill retirement telemetry underflow")
            self._prefill_current_chunks = current_chunks - len(prefill)
            self._prefill_current_bytes = current_bytes - released_bytes
            self._prefill_current_reserved_bytes = current_reserved - released_reserved
        job.inflight_waves.clear()
        job.prefill_inflight_bytes = 0
        job.prefill_reserved_bytes = 0

    def _check_pipeline_timeouts(self, active: dict[int, _GenerationJob]) -> None:
        pending_quote = getattr(self, "_pending_tree_reservation", None)
        if pending_quote is not None:
            observed_at = time.perf_counter()
            waiting_for_commit = pending_quote.commit_sent
            queue_name = (
                "_queued_tree_commit_results"
                if waiting_for_commit
                else "_queued_tree_prepare_results"
            )
            lock = getattr(self, "_tree_return_lock", None)
            if lock is None:
                queued_quote_result = False
            else:
                with lock:
                    queued_quote_result = any(
                        request_id == pending_quote.prepared.job.wire_id
                        and step == pending_quote.prepared.step
                        for request_id, step, _arrived in getattr(
                            self, queue_name, ()
                        )
                    )
            if observed_at > pending_quote.deadline_at and not queued_quote_result:
                self._physical_tree_quote_timeouts = int(
                    getattr(self, "_physical_tree_quote_timeouts", 0)
                ) + 1
                self._cancel_pending_tree_reservation(
                    pending_quote.prepared.job,
                    pending_quote.downstream,
                    reason=(
                        "TREE_RESERVATION_COMMIT_RESULT timeout"
                        if waiting_for_commit
                        else "TREE_PREPARE_RESULT timeout"
                    ),
                    count_metric=False,
                )
                raise TimeoutError(
                    (
                        "TREE_RESERVATION_COMMIT_RESULT"
                        if waiting_for_commit
                        else "TREE_PREPARE_RESULT"
                    )
                    + " timeout for parent request "
                    f"{pending_quote.prepared.job.wire_id}"
                )
        coordinator = getattr(self, "_physical_tree", None)
        if coordinator is not None:
            lock = getattr(self, "_tree_return_lock", None)
            if lock is None:
                queued_request_ids: set[int] = set()
            else:
                with lock:
                    queued_request_ids = {
                        request_id
                        for request_id, arrivals in self._queued_tree_returns.items()
                        if arrivals
                    }
            observed_at = time.perf_counter()
            protected_wave = False
            expired_unprotected: list[tuple[int, int]] = []
            for wave in tuple(coordinator.tree_by_parent.values()):
                outstanding = tuple(
                    leaf
                    for leaf in wave.ordered_leaves
                    if leaf.verify_sent
                    and leaf.result is None
                    and not leaf.return_drained
                )
                if not outstanding or observed_at < wave.deadline_at:
                    continue
                # A timestamped return already owned by the scheduler must be
                # accepted (and judged by arrival time) before expiring this
                # wave. It does not shield an unrelated expired wave.
                if any(
                    leaf.virtual_request_id in queued_request_ids
                    for leaf in outstanding
                ):
                    protected_wave = True
                    continue
                oldest = min(
                    outstanding,
                    key=lambda leaf: (
                        float("inf")
                        if leaf.verify_sent_at is None
                        else leaf.verify_sent_at,
                        leaf.ordinal,
                    ),
                )
                expired_unprotected.append(
                    (wave.parent_request_id, oldest.virtual_request_id)
                )
            if expired_unprotected and protected_wave:
                parent_request_id, oldest_virtual_id = expired_unprotected[0]
                self._physical_tree_aborted_waves = int(
                    getattr(self, "_physical_tree_aborted_waves", 0)
                ) + 1
                raise TimeoutError(
                    "physical tree VERIFY_RESULT timeout for oldest virtual request "
                    f"{oldest_virtual_id} (parent {parent_request_id})"
                )
            expired_trees = (
                coordinator.check_timeouts(observed_at)
                if expired_unprotected and not protected_wave
                else ()
            )
            if expired_trees:
                self._sync_physical_leaf_routes()
                self._physical_tree_aborted_waves = int(
                    getattr(self, "_physical_tree_aborted_waves", 0)
                ) + len(expired_trees)
                oldest = expired_trees[0]
                raise TimeoutError(oldest.reason)
        now = time.monotonic()
        expired = [
            job.request.client_id
            for job in active.values()
            if job.inflight_waves
            and now - job.inflight_waves[0].sent_at
            > self.config.socket_timeout_seconds
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

    def _physical_tree_preflight(
        self,
        job: _GenerationJob,
        proposal: MacroWaveProposal,
        runner: StageRunnerContract,
        *,
        base_kv_tokens: int,
    ) -> bool:
        """Atomically price a flat leaf set before the first physical FORK.

        A capacity miss is a clean opt-out because neither root nor wire state
        has changed. Contract/type mismatches raise: treating an unverifiable
        byte estimate as spare capacity would defeat the sealed limits.
        """

        return self._root_tree_capacity_snapshot(
            job,
            proposal,
            runner,
            base_kv_tokens=base_kv_tokens,
        ) is not None

    def _root_tree_capacity_snapshot(
        self,
        job: _GenerationJob,
        proposal: MacroWaveProposal,
        runner: StageRunnerContract,
        *,
        base_kv_tokens: int,
    ) -> _RootTreeCapacitySnapshot | None:
        """Price and seal root state without mutating any request KV."""

        parent_request_id = job.wire_id
        if parent_request_id is None:
            raise RuntimeError("physical tree parent has no wire identity")
        limits = (
            self.config.max_speculative_branches,
            self.config.max_speculative_branch_tokens,
            self.config.max_speculative_kv_bytes,
        )
        if not all(value > 0 for value in limits):
            return None
        if job.inflight_waves:
            return None
        if job.next_step is None or job.next_step != job.step:
            return None
        prompt_tokens = int(job.request.input_ids.shape[1])
        if (
            job.prefill_offset != prompt_tokens
            or job.prefill_acked_offset != prompt_tokens
        ):
            return None
        parent_tokens = runner.sequence_length(parent_request_id)
        if base_kv_tokens < 1 or parent_tokens != base_kv_tokens:
            return None
        if proposal.tree.ledger.base_version != KVVersion(base_kv_tokens):
            raise RuntimeError("physical tree proposal is bound to another KV version")

        paths = proposal.candidate_paths
        live_child_ids = tuple(
            sorted(getattr(self, "_physical_tree_live_children", set()))
        )
        if len(live_child_ids) + len(paths) > self.config.max_speculative_branches:
            return None
        maximum_depth = proposal.max_depth
        if 1 + maximum_depth > _sealed_wave_token_limit(self.config):
            return None
        if (
            base_kv_tokens + 1 + maximum_depth
            > self.config.max_speculative_branch_tokens
        ):
            return None
        maximum_context = int(getattr(self, "maximum_context", 0))
        if (
            maximum_context < 1
            or base_kv_tokens + 1 + maximum_depth > maximum_context
        ):
            return None

        request_cache_bytes = getattr(runner, "request_cache_bytes", None)
        project_cache_bytes = getattr(runner, "project_request_cache_bytes", None)
        fork_request = getattr(runner, "fork_request", None)
        promote_request = getattr(runner, "promote_request", None)
        if not all(
            callable(method)
            for method in (
                request_cache_bytes,
                project_cache_bytes,
                fork_request,
                promote_request,
            )
        ):
            return None
        if getattr(runner, "max_active_requests", None) == 1:
            return None

        parent_bytes = request_cache_bytes(parent_request_id)
        if (
            not isinstance(parent_bytes, int)
            or isinstance(parent_bytes, bool)
            or parent_bytes < 1
        ):
            raise TypeError("root request_cache_bytes must return a positive integer")
        current_branch_bytes = self._physical_tree_live_kv_bytes(runner)
        available_physical = getattr(runner, "available_physical_cache_bytes", None)
        available_physical_bytes: int | None = None
        if callable(available_physical):
            available_physical_bytes = available_physical()
            if (
                not isinstance(available_physical_bytes, int)
                or isinstance(available_physical_bytes, bool)
                or available_physical_bytes < 0
            ):
                raise TypeError("root runner returned invalid physical KV availability")
        project_tree_physical = getattr(
            runner, "project_tree_incremental_physical_cache_bytes", None
        )
        if callable(project_tree_physical):
            projected_branch_bytes = project_tree_physical(
                parent_request_id,
                delta_tokens_by_leaf=tuple(1 + len(path) for path in paths),
            )
            if (
                not isinstance(projected_branch_bytes, int)
                or isinstance(projected_branch_bytes, bool)
                or projected_branch_bytes < 0
            ):
                raise TypeError("root runner returned invalid physical tree projection")
        else:
            projected_branch_bytes = 0
            for path in paths:
                projected = project_cache_bytes(parent_request_id, 1 + len(path))
                if (
                    not isinstance(projected, int)
                    or isinstance(projected, bool)
                    or projected < parent_bytes
                ):
                    raise TypeError(
                        "root project_request_cache_bytes returned an invalid projection"
                    )
                projected_branch_bytes += projected
        projected_total = current_branch_bytes + projected_branch_bytes
        if projected_total > self.config.max_speculative_kv_bytes:
            return None
        if (
            available_physical_bytes is not None
            and projected_branch_bytes > available_physical_bytes
        ):
            return None
        if runner.sequence_length(parent_request_id) != parent_tokens:
            raise RuntimeError("root tree projection mutated parent sequence length")
        if request_cache_bytes(parent_request_id) != parent_bytes:
            raise RuntimeError("root tree projection mutated parent cache bytes")
        if tuple(
            sorted(getattr(self, "_physical_tree_live_children", set()))
        ) != live_child_ids:
            raise RuntimeError("root tree projection mutated live child identities")
        if self._physical_tree_live_kv_bytes(runner) != current_branch_bytes:
            raise RuntimeError("root tree projection mutated live child KV bytes")
        if (
            callable(available_physical)
            and available_physical() != available_physical_bytes
        ):
            raise RuntimeError("root tree projection mutated physical KV availability")
        return _RootTreeCapacitySnapshot(
            parent_tokens=parent_tokens,
            parent_cache_bytes=parent_bytes,
            live_child_ids=live_child_ids,
            live_child_bytes=current_branch_bytes,
            projected_kv_bytes=projected_total,
            available_physical_bytes=available_physical_bytes,
        )

    def _physical_tree_live_kv_bytes(self, runner: StageRunnerContract) -> int:
        live_children = tuple(
            sorted(getattr(self, "_physical_tree_live_children", set()))
        )
        if not live_children:
            return 0
        unique_physical_bytes = getattr(runner, "unique_physical_cache_bytes", None)
        if callable(unique_physical_bytes):
            leaf_routes = getattr(self, "_leaf_routes", {})
            try:
                parent_ids = tuple(
                    sorted({leaf_routes[request_id] for request_id in live_children})
                )
            except KeyError as error:
                raise RuntimeError(
                    "root physical KV accounting lost a live child's parent route"
                ) from error
            all_ids = tuple(sorted(set(live_children) | set(parent_ids)))
            all_bytes = unique_physical_bytes(all_ids)
            parent_bytes = unique_physical_bytes(parent_ids)
            for name, value in (("all", all_bytes), ("parent", parent_bytes)):
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    raise TypeError(
                        f"root runner returned invalid {name} unique physical KV bytes"
                    )
            if all_bytes < parent_bytes:
                raise RuntimeError("root physical KV accounting moved backwards")
            return all_bytes - parent_bytes

        request_cache_bytes = getattr(runner, "request_cache_bytes", None)
        if not callable(request_cache_bytes):
            raise TypeError("root runner cannot measure physical leaf KV")
        total = 0
        for request_id in live_children:
            measured = request_cache_bytes(request_id)
            if (
                not isinstance(measured, int)
                or isinstance(measured, bool)
                or measured < 0
            ):
                raise TypeError("root request_cache_bytes returned invalid leaf KV bytes")
            total += measured
        return total

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
        encoded = encode_tensor_payload(hidden, self.config.codec)
        return send_frame(
            downstream,
            frame_type,
            request_id,
            step=step,
            token_count=int(hidden.shape[1]),
            hidden_size=int(hidden.shape[2]),
            flags=int(self.config.codec),
            payload=encoded.view,
            emulator=emulator,
        )

    def _record_dispatched_wave(
        self,
        wave: _PreparedRootWave,
        *,
        started_at: float,
        sent_at: float,
        outbound_bytes: int,
    ) -> None:
        job = wave.job
        next_step = job.next_step
        if next_step is None or wave.step != next_step:
            raise RuntimeError(
                f"request {job.wire_id} dispatched step {wave.step}, "
                f"expected immutable step {next_step}"
            )
        if job.inflight_waves and wave.step != job.inflight_waves[-1].step + 1:
            raise RuntimeError("request outbound steps are not contiguous")
        if not job.inflight_waves and job.step != wave.step:
            raise RuntimeError(
                f"request {job.wire_id} dispatched step {wave.step}, "
                f"but expects return step {job.step}"
            )
        flight = _InflightWave(
            step=wave.step,
            frame_type=wave.frame_type,
            prefill_end=wave.prefill_end,
            started_at=started_at,
            sent_at=sent_at,
            outbound_bytes=outbound_bytes,
            reserved_bytes=wave.reserved_bytes,
        )
        job.inflight_waves.append(flight)
        job.next_step = wave.step + 1
        job.last_sent_at = sent_at
        job.last_outbound_bytes = outbound_bytes
        if not flight.is_prefill:
            return
        if flight.prefill_end is None or flight.prefill_end <= job.prefill_offset:
            raise RuntimeError("prefill dispatch offsets are not strictly increasing")
        job.prefill_offset = flight.prefill_end
        job.prefill_inflight_bytes += outbound_bytes
        job.prefill_reserved_bytes += wave.reserved_bytes
        self._prefill_current_chunks = int(
            getattr(self, "_prefill_current_chunks", 0)
        ) + 1
        self._prefill_current_bytes = int(
            getattr(self, "_prefill_current_bytes", 0)
        ) + outbound_bytes
        self._prefill_current_reserved_bytes = int(
            getattr(self, "_prefill_current_reserved_bytes", 0)
        ) + wave.reserved_bytes
        self._prefill_high_water_chunks = max(
            int(getattr(self, "_prefill_high_water_chunks", 0)),
            self._prefill_current_chunks,
        )
        self._prefill_high_water_bytes = max(
            int(getattr(self, "_prefill_high_water_bytes", 0)),
            self._prefill_current_bytes,
        )
        self._prefill_high_water_reserved_bytes = max(
            int(getattr(self, "_prefill_high_water_reserved_bytes", 0)),
            self._prefill_current_reserved_bytes,
        )
        request_chunks = sum(1 for current in job.inflight_waves if current.is_prefill)
        self._prefill_max_request_chunks = max(
            int(getattr(self, "_prefill_max_request_chunks", 0)),
            request_chunks,
        )
        self._prefill_max_request_bytes = max(
            int(getattr(self, "_prefill_max_request_bytes", 0)),
            job.prefill_inflight_bytes,
        )
        self._prefill_max_request_reserved_bytes = max(
            int(getattr(self, "_prefill_max_request_reserved_bytes", 0)),
            job.prefill_reserved_bytes,
        )
        self._prefill_dispatched_chunks = int(
            getattr(self, "_prefill_dispatched_chunks", 0)
        ) + 1

    def _dispatch_physical_tree_wave(
        self,
        prepared: _PreparedPhysicalTreeWave,
        runner: StageRunnerContract,
        downstream: socket.socket,
        emulator: LinkEmulator,
    ) -> _PreparedRootWave | None:
        """Request a route-wide quote without mutating root or remote KV."""

        job = prepared.job
        parent_request_id = job.wire_id
        if parent_request_id is None:
            raise RuntimeError("physical tree parent has no wire identity")
        if prepared.step != job.next_step or prepared.step != job.step:
            raise RuntimeError("prepared physical tree has a stale inherited step")
        if job.cancel_requested.is_set():
            if prepared.proposal.tree.state is MacroWaveState.OPEN:
                prepared.proposal.tree.rollback()
            return None
        coordinator = getattr(self, "_physical_tree", None)
        if (
            getattr(self, "_pending_tree_reservation", None) is not None
            or bool(getattr(coordinator, "tree_by_parent", {}))
            or bool(getattr(self, "_physical_tree_live_children", set()))
        ):
            if prepared.proposal.tree.state is MacroWaveState.OPEN:
                prepared.proposal.tree.rollback()
            self._physical_tree_quote_singleflight_fallbacks = int(
                getattr(self, "_physical_tree_quote_singleflight_fallbacks", 0)
            ) + 1
            return self._prepare_linear_or_classic_decode_wave(
                job,
                runner,
                active_sequences=prepared.active_sequences,
            )
        root_snapshot = self._root_tree_capacity_snapshot(
            job,
            prepared.proposal,
            runner,
            base_kv_tokens=prepared.base_kv_tokens,
        )
        if root_snapshot is None:
            if prepared.proposal.tree.state is MacroWaveState.OPEN:
                prepared.proposal.tree.rollback()
            return self._prepare_linear_or_classic_decode_wave(
                job,
                runner,
                active_sequences=prepared.active_sequences,
            )

        ordered_paths = tuple(
            sorted(
                prepared.proposal.candidate_paths,
                key=lambda path: (len(path), path),
            )
        )
        path_lengths = tuple(len(path) for path in ordered_paths)
        nonce = self._next_tree_quote_nonce()
        sent_at = time.perf_counter()
        pending = _PendingTreeReservation(
            prepared=prepared,
            nonce=nonce,
            path_lengths=path_lengths,
            root_snapshot=root_snapshot,
            sent_at=sent_at,
            deadline_at=sent_at + self.config.socket_timeout_seconds,
            downstream=downstream,
        )
        self._pending_tree_reservation = pending
        self._physical_tree_quote_requests = int(
            getattr(self, "_physical_tree_quote_requests", 0)
        ) + 1
        send_frame(
            downstream,
            FrameType.TREE_PREPARE,
            parent_request_id,
            step=prepared.step,
            token_count=len(path_lengths),
            payload=tree_prepare_payload(nonce, path_lengths),
            emulator=emulator,
        )
        return None

    def _next_tree_quote_nonce(self) -> int:
        nonce = int(getattr(self, "_tree_quote_nonce_counter", 0)) + 1
        if nonce > (1 << 64) - 1:
            raise RuntimeError("tree capacity quote nonce space is exhausted")
        self._tree_quote_nonce_counter = nonce
        return nonce

    def _dispatch_committed_physical_tree_wave(
        self,
        committed: _CommittedPhysicalTreeWave,
        runner: StageRunnerContract,
        downstream: socket.socket,
        emulator: LinkEmulator,
    ) -> _PreparedRootWave | None:
        """Consume a committed route quote exactly once, then issue FORK*/VERIFY*."""

        pending = committed.pending
        if getattr(self, "_pending_tree_reservation", None) is not pending:
            raise RuntimeError("committed physical tree lost its pending reservation")
        if not pending.committed:
            raise RuntimeError(
                "physical tree cannot FORK before route-wide reservation COMMIT result"
            )
        prepared = pending.prepared
        job = prepared.job
        parent_request_id = job.wire_id
        if parent_request_id is None:
            raise RuntimeError("physical tree parent has no wire identity")
        if job.cancel_requested.is_set():
            self._cancel_pending_tree_reservation(
                job,
                downstream,
                reason="generation cancelled after tree COMMIT and before FORK",
            )
            return None
        observed = self._root_tree_capacity_snapshot(
            job,
            prepared.proposal,
            runner,
            base_kv_tokens=prepared.base_kv_tokens,
        )
        if observed is None or observed != pending.root_snapshot:
            self._cancel_pending_tree_reservation(
                job,
                downstream,
                reason="root capacity changed after tree COMMIT",
                count_metric=False,
            )
            self._physical_tree_quote_protocol_failures = int(
                getattr(self, "_physical_tree_quote_protocol_failures", 0)
            ) + 1
            raise RuntimeError("root tree capacity changed after remote COMMIT")

        virtual_ids = tuple(
            self._next_request_id()
            for _path in prepared.proposal.candidate_paths
        )
        coordinator = self._physical_tree
        wave = coordinator.prepare_wave(
            parent_request_id=parent_request_id,
            proposal=prepared.proposal,
            base_kv_tokens=prepared.base_kv_tokens,
            pending_token=prepared.pending_token,
            inherited_step=prepared.step,
            virtual_request_ids=virtual_ids,
            deadline_at=(
                time.perf_counter() + self.config.socket_timeout_seconds
            ),
        )
        self._sync_physical_leaf_routes()
        self._physical_tree_prepared_waves = int(
            getattr(self, "_physical_tree_prepared_waves", 0)
        ) + 1
        self._physical_tree_prepared_leaves = int(
            getattr(self, "_physical_tree_prepared_leaves", 0)
        ) + len(wave.leaves)
        if job.cancel_requested.is_set():
            self._cancel_pending_tree_reservation(
                job,
                downstream,
                reason="generation cancelled immediately before first physical FORK",
            )
            coordinator.abort_before_wire(
                parent_request_id,
                "generation cancelled before first physical FORK",
            )
            self._sync_physical_leaf_routes()
            coordinator.retire_wave(parent_request_id)
            return None

        # Every command is consumed once by the coordinator. A failure after
        # this point is fatal to the route; no KV mutation is retried.
        while True:
            command = coordinator.next_fork_command(parent_request_id)
            if command is None:
                break
            request_cache_bytes = getattr(runner, "request_cache_bytes", None)
            fork_request = getattr(runner, "fork_request", None)
            if not callable(request_cache_bytes) or not callable(fork_request):
                raise TypeError("root runner lost its exact FORK capability")
            parent_bytes = request_cache_bytes(parent_request_id)
            current_bytes = self._physical_tree_live_kv_bytes(runner)
            remaining_bytes = self.config.max_speculative_kv_bytes - current_bytes
            project_tree_physical = getattr(
                runner, "project_tree_incremental_physical_cache_bytes", None
            )
            projected_fork_bytes = (
                project_tree_physical(
                    parent_request_id,
                    delta_tokens_by_leaf=(0,),
                )
                if callable(project_tree_physical)
                else parent_bytes
            )
            if (
                not isinstance(parent_bytes, int)
                or isinstance(parent_bytes, bool)
                or parent_bytes < 1
            ):
                raise RuntimeError("root FORK parent has invalid logical KV bytes")
            if (
                not isinstance(projected_fork_bytes, int)
                or isinstance(projected_fork_bytes, bool)
                or projected_fork_bytes < 0
                or projected_fork_bytes > remaining_bytes
            ):
                raise RuntimeError("root FORK no longer fits its sealed KV budget")
            available_physical = getattr(
                runner, "available_physical_cache_bytes", None
            )
            if callable(available_physical):
                available_bytes = available_physical()
                if (
                    not isinstance(available_bytes, int)
                    or isinstance(available_bytes, bool)
                    or available_bytes < projected_fork_bytes
                ):
                    raise RuntimeError("root FORK no longer fits physical KV pool")
            copied_bytes = fork_request(
                command.child_request_id,
                command.parent_request_id,
                max_cache_bytes=remaining_bytes,
            )
            if (
                not isinstance(copied_bytes, int)
                or isinstance(copied_bytes, bool)
                or copied_bytes < 0
            ):
                runner.end(command.child_request_id)
                raise RuntimeError("root FORK returned invalid copied KV bytes")
            self._physical_tree_live_children.add(command.child_request_id)
            try:
                observed_bytes = self._physical_tree_live_kv_bytes(runner)
            except BaseException:
                self._physical_tree_live_children.remove(command.child_request_id)
                runner.end(command.child_request_id)
                raise
            observed_increment = observed_bytes - current_bytes
            if observed_increment != projected_fork_bytes:
                self._physical_tree_live_children.remove(command.child_request_id)
                runner.end(command.child_request_id)
                raise RuntimeError(
                    "root FORK physical allocation changed after preflight: "
                    f"{observed_increment} != {projected_fork_bytes}"
                )
            last_fork_report = getattr(runner, "last_fork_report", None)
            if callable(last_fork_report):
                report = last_fork_report()
                if report is not None and (
                    getattr(report, "copied_bytes", None) != copied_bytes
                    or getattr(report, "newly_reserved_bytes", None)
                    != observed_increment
                ):
                    self._physical_tree_live_children.remove(command.child_request_id)
                    runner.end(command.child_request_id)
                    raise RuntimeError("root FORK physical accounting report mismatch")
            send_frame(
                downstream,
                FrameType.FORK,
                command.child_request_id,
                payload=branch_request_payload(command.parent_request_id),
            )

        # Every remote stage consumes its logical reservation on the final
        # ordered FORK. Clearing root state here admits the next global quote;
        # no quote is considered consumed merely because COMMIT was written.
        self._pending_tree_reservation = None

        verify_commands: list[VerifyCommand] = []
        while True:
            command = coordinator.next_verify_command(
                parent_request_id,
                now=time.perf_counter(),
            )
            if command is None:
                break
            verify_commands.append(command)
        job.wave_started_at = time.perf_counter()
        self._dispatch_physical_tree_verifies(
            verify_commands,
            runner,
            downstream,
            emulator,
        )
        return None

    def _dispatch_physical_tree_verifies(
        self,
        commands: list[VerifyCommand],
        runner: StageRunnerContract,
        downstream: socket.socket,
        emulator: LinkEmulator,
    ) -> None:
        """Batch virtual leaves only when their exact physical key matches."""

        if not commands:
            raise RuntimeError("physical tree has no VERIFY commands")
        self._root_ready_items += len(commands)
        manifest = getattr(runner, "executor_manifest", None)
        features = tuple(getattr(manifest, "features", ()))
        # Equal shapes and cache lengths are necessary, but not sufficient, for
        # token-exact batch=K vs K*batch=1 parity on every GPU/dtype/kernel.
        # The capability must therefore be sealed into the executor manifest;
        # current production runners do not advertise it and remain sequential.
        batch_exact = all(
            feature in features
            for feature in (
                "exact-tree-verify-batching",
                "bounded-tree-verify-workspace",
            )
        )
        batch_forward = (
            getattr(runner, "forward_ids_batch", None) if batch_exact else None
        )
        batch_key = getattr(runner, "physical_batch_key", None) if batch_exact else None
        maximum = getattr(runner, "MAX_PHYSICAL_BATCH_SIZE", 1)
        if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum < 2:
            maximum = 1

        prepared_inputs = {
            command.request_id: torch.tensor(
                [command.input_tokens], dtype=torch.long
            )
            for command in commands
        }
        groups: dict[object, list[VerifyCommand]] = {}
        for index, command in enumerate(commands):
            key: object | None = None
            value = prepared_inputs[command.request_id]
            if callable(batch_forward) and callable(batch_key) and maximum >= 2:
                runner_key = batch_key(
                    command.request_id,
                    token_count=command.token_count,
                    token_mode=_root_token_mode(FrameType.VERIFY),
                )
                if runner_key is not None:
                    candidate = (
                        FrameType.VERIFY,
                        tuple(value.shape),
                        value.dtype,
                        value.device,
                        runner_key,
                    )
                    try:
                        hash(candidate)
                    except TypeError:
                        candidate = None
                    key = candidate
            groups.setdefault(
                key if key is not None else ("tree-sequential", index), []
            ).append(command)

        last_verify_sent_at: float | None = None
        for group in groups.values():
            for offset in range(0, len(group), max(1, maximum)):
                chunk = group[offset : offset + max(1, maximum)]
                if len(chunk) > 1:
                    if not callable(batch_forward):
                        raise TypeError("root tree batch key exists without batch forward")
                    outputs = tuple(
                        batch_forward(
                            tuple(command.request_id for command in chunk),
                            tuple(
                                prepared_inputs[command.request_id]
                                for command in chunk
                            ),
                        )
                    )
                    if len(outputs) != len(chunk):
                        raise RuntimeError(
                            "root physical tree batch returned the wrong output count"
                        )
                    self._root_physical_batch_calls += 1
                    self._root_physical_batch_items += len(chunk)
                    self._root_max_physical_batch_size = max(
                        self._root_max_physical_batch_size, len(chunk)
                    )
                    self._root_model_forward_calls += 1
                    self._physical_tree_batch_calls = int(
                        getattr(self, "_physical_tree_batch_calls", 0)
                    ) + 1
                    self._physical_tree_batch_items = int(
                        getattr(self, "_physical_tree_batch_items", 0)
                    ) + len(chunk)
                else:
                    command = chunk[0]
                    outputs = (
                        runner.forward_ids(
                            command.request_id,
                            prepared_inputs[command.request_id],
                        ),
                    )
                    self._root_sequential_items += 1
                    self._root_model_forward_calls += 1

                for command, hidden in zip(chunk, outputs, strict=True):
                    if not isinstance(hidden, torch.Tensor):
                        raise TypeError("root physical tree output must be a tensor")
                    if tuple(hidden.shape) != (
                        1,
                        command.token_count,
                        self.hidden_size,
                    ):
                        raise RuntimeError(
                            "root physical tree output has an incompatible shape"
                        )
                    self._send_activation(
                        downstream,
                        emulator,
                        command.request_id,
                        command.step,
                        hidden,
                        frame_type=FrameType.VERIFY,
                    )
                    sent_at = time.perf_counter()
                    try:
                        _parent, leaf = self._physical_tree.leaf_route[
                            command.request_id
                        ]
                    except KeyError as exc:
                        raise RuntimeError(
                            "physical tree VERIFY lost its virtual route"
                        ) from exc
                    leaf.verify_sent_at = sent_at
                    last_verify_sent_at = sent_at

        if last_verify_sent_at is None:
            raise RuntimeError("physical tree sent no VERIFY frame")
        parent_request_id = self._leaf_routes.get(commands[0].request_id)
        if parent_request_id is None:
            raise RuntimeError("physical tree VERIFY lost its parent route")
        # FORK cloning and root compute have their own fatal execution path;
        # they must not consume a leaf's return budget. Protocol v4 exposes one
        # wave deadline, so arm it after the last VERIFY is physically written.
        self._physical_tree.wave(parent_request_id).deadline_at = (
            last_verify_sent_at + self.config.socket_timeout_seconds
        )

    def _sync_physical_leaf_routes(self) -> None:
        coordinator = getattr(self, "_physical_tree", None)
        routes = (
            {}
            if coordinator is None
            else {
                virtual_id: parent_request_id
                for virtual_id, (parent_request_id, _leaf) in coordinator.leaf_route.items()
            }
        )
        lock = getattr(self, "_tree_return_lock", None)
        if lock is None:
            self._leaf_routes = routes
            return
        with lock:
            self._leaf_routes = routes

    def _record_queued_tree_return(self, request_id: int, arrived: float) -> None:
        lock = getattr(self, "_tree_return_lock", None)
        if lock is None:
            return
        with lock:
            if request_id not in self._leaf_routes:
                return
            self._queued_tree_returns.setdefault(request_id, deque()).append(arrived)

    def _consume_queued_tree_return(self, request_id: int, arrived: float) -> None:
        lock = getattr(self, "_tree_return_lock", None)
        if lock is None:
            return
        with lock:
            arrivals = self._queued_tree_returns.get(request_id)
            if not arrivals:
                return
            observed = arrivals.popleft()
            if observed != arrived:
                raise RuntimeError("physical tree return queue lost arrival FIFO")
            if not arrivals:
                self._queued_tree_returns.pop(request_id, None)

    def _record_queued_tree_prepare_result(
        self, request_id: int, step: int, arrived: float
    ) -> None:
        lock = getattr(self, "_tree_return_lock", None)
        if lock is None:
            return
        with lock:
            queued = getattr(self, "_queued_tree_prepare_results", None)
            if queued is None:
                queued = deque()
                self._queued_tree_prepare_results = queued
            queued.append((request_id, step, arrived))

    def _consume_queued_tree_prepare_result(self, frame: Any, arrived: float) -> None:
        lock = getattr(self, "_tree_return_lock", None)
        if lock is None:
            return
        with lock:
            queued = getattr(self, "_queued_tree_prepare_results", None)
            if not queued:
                # Direct unit callers do not pass through the socket reader.
                return
            observed = queued.popleft()
        if observed != (frame.request_id, frame.step, arrived):
            raise RuntimeError("tree prepare result queue lost arrival FIFO")

    def _record_queued_tree_commit_result(
        self, request_id: int, step: int, arrived: float
    ) -> None:
        lock = getattr(self, "_tree_return_lock", None)
        if lock is None:
            return
        with lock:
            queued = getattr(self, "_queued_tree_commit_results", None)
            if queued is None:
                queued = deque()
                self._queued_tree_commit_results = queued
            queued.append((request_id, step, arrived))

    def _consume_queued_tree_commit_result(self, frame: Any, arrived: float) -> None:
        lock = getattr(self, "_tree_return_lock", None)
        if lock is None:
            return
        with lock:
            queued = getattr(self, "_queued_tree_commit_results", None)
            if not queued:
                # Direct unit callers do not pass through the socket reader.
                return
            observed = queued.popleft()
        if observed != (frame.request_id, frame.step, arrived):
            raise RuntimeError("tree commit result queue lost arrival FIFO")

    def _dispatch_root_waves(
        self,
        waves: list[
            _PreparedRootWave
            | _PreparedPhysicalTreeWave
            | _CommittedPhysicalTreeWave
        ],
        runner: StageRunner,
        downstream: socket.socket,
        emulator: LinkEmulator,
    ) -> None:
        """Execute compatible root continuations in physical tensor batches.

        Compatibility is determined before any cache is advanced. Unsupported
        cache layouts and executors remain sequential. Once a physical forward
        starts, errors are fatal rather than retried, because retrying could
        advance backend-owned KV twice.
        """

        pending_transaction = getattr(self, "_pending_tree_reservation", None)
        deferred = getattr(self, "_tree_barrier_deferred_waves", None)
        if deferred is None:
            deferred = deque()
            self._tree_barrier_deferred_waves = deferred
        if pending_transaction is not None:
            commit_waves = [
                wave for wave in waves if isinstance(wave, _CommittedPhysicalTreeWave)
            ]
            if len(commit_waves) > 1:
                raise RuntimeError("one tree transaction received duplicate COMMIT results")
            for wave in waves:
                if not isinstance(wave, _CommittedPhysicalTreeWave):
                    deferred.append(wave)
            if not commit_waves:
                return
            # COMMIT acknowledgement is a global memory barrier. It must be
            # consumed before an earlier ordinary return can advance root KV.
            waves = commit_waves
        elif deferred:
            waves = [*deferred, *waves]
            deferred.clear()

        if not waves:
            return
        if any(
            isinstance(
                wave, (_PreparedPhysicalTreeWave, _CommittedPhysicalTreeWave)
            )
            for wave in waves
        ):
            # A tree is an ordered control transaction. Flush ordinary work on
            # either side as its original physical batches, but never let a
            # BEGIN/activation split FORK* -> VERIFY* or cleanup controls.
            ordinary: list[_PreparedRootWave] = []
            for wave in waves:
                if isinstance(wave, _PreparedRootWave):
                    ordinary.append(wave)
                    continue
                if ordinary:
                    self._dispatch_root_waves(ordinary, runner, downstream, emulator)
                    ordinary = []
                if isinstance(wave, _CommittedPhysicalTreeWave):
                    fallback = self._dispatch_committed_physical_tree_wave(
                        wave,
                        runner,
                        downstream,
                        emulator,
                    )
                else:
                    fallback = self._dispatch_physical_tree_wave(
                        wave,
                        runner,
                        downstream,
                        emulator,
                    )
                if fallback is not None:
                    self._dispatch_root_waves(
                        [fallback], runner, downstream, emulator
                    )
            if ordinary:
                self._dispatch_root_waves(ordinary, runner, downstream, emulator)
            if (
                getattr(self, "_pending_tree_reservation", None) is None
                and deferred
            ):
                resumed = list(deferred)
                deferred.clear()
                self._dispatch_root_waves(resumed, runner, downstream, emulator)
            return
        request_ids = [wave.job.wire_id for wave in waves]
        if any(request_id is None for request_id in request_ids):
            raise RuntimeError("prepared root wave has no wire identity")
        if len(set(request_ids)) != len(request_ids):
            raise RuntimeError("one root dispatch cannot repeat a request")
        self._root_ready_items += len(waves)

        batch_forward = getattr(runner, "forward_ids_batch", None)
        batch_key = getattr(runner, "physical_batch_key", None)
        maximum = getattr(runner, "MAX_PHYSICAL_BATCH_SIZE", 1)
        if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum < 2:
            maximum = 1

        groups: dict[object, list[_PreparedRootWave]] = {}
        for index, wave in enumerate(waves):
            key: object | None = None
            if callable(batch_forward) and callable(batch_key) and maximum >= 2:
                token_count = int(wave.input_ids.shape[1])
                runner_key = batch_key(
                    wave.job.wire_id,
                    token_count=token_count,
                    token_mode=_root_token_mode(wave.frame_type),
                )
                if runner_key is not None:
                    candidate = (
                        wave.frame_type,
                        tuple(wave.input_ids.shape),
                        wave.input_ids.dtype,
                        wave.input_ids.device,
                        runner_key,
                    )
                    try:
                        hash(candidate)
                    except TypeError:
                        candidate = None
                    key = candidate
            # A unique key makes unsupported work explicitly sequential while
            # preserving its position relative to the first compatible group.
            group_key = key if key is not None else ("sequential", index)
            groups.setdefault(group_key, []).append(wave)

        for group in groups.values():
            for offset in range(0, len(group), max(1, maximum)):
                chunk = group[offset : offset + max(1, maximum)]
                started = time.perf_counter()
                for wave in chunk:
                    # Speculation pricing includes root compute and all network
                    # work, matching the former sequential timing boundary.
                    wave.job.wave_started_at = started
                    next_step = wave.job.next_step
                    if next_step is None or wave.step != next_step:
                        raise RuntimeError("prepared root wave has a stale outbound step")
                if len(chunk) > 1:
                    if not callable(batch_forward):
                        raise TypeError("root batch key exists without forward_ids_batch")
                    outputs = tuple(
                        batch_forward(
                            tuple(wave.job.wire_id for wave in chunk),
                            tuple(wave.input_ids for wave in chunk),
                        )
                    )
                    if len(outputs) != len(chunk):
                        raise RuntimeError(
                            "root physical batch returned the wrong number of outputs"
                        )
                    self._root_physical_batch_calls += 1
                    self._root_physical_batch_items += len(chunk)
                    self._root_max_physical_batch_size = max(
                        self._root_max_physical_batch_size, len(chunk)
                    )
                    self._root_model_forward_calls += 1
                else:
                    outputs = (
                        runner.forward_ids(
                            chunk[0].job.wire_id,
                            chunk[0].input_ids,
                        ),
                    )
                    self._root_sequential_items += 1
                    self._root_model_forward_calls += 1

                for wave, hidden in zip(chunk, outputs, strict=True):
                    if not isinstance(hidden, torch.Tensor):
                        raise TypeError("root stage output must be a tensor")
                    job = wave.job
                    if wave.prefill_end is not None:
                        # Prompt tokens are the served sequence by definition.
                        job.kv_valid += wave.prefill_end - job.prefill_offset
                    else:
                        # A decode wave re-forwards the last emitted token, which
                        # always matches the served sequence; any draft position
                        # after it becomes valid only once verification accepts it.
                        job.kv_valid += 1
                    outbound_bytes = self._send_activation(
                        downstream,
                        emulator,
                        job.wire_id,
                        wave.step,
                        hidden,
                        frame_type=wave.frame_type,
                    )
                    job.last_outbound_bytes = outbound_bytes
                    sent_at = time.monotonic()
                    self._record_dispatched_wave(
                        wave,
                        started_at=started,
                        sent_at=sent_at,
                        outbound_bytes=outbound_bytes,
                    )

    def _prepare_next_prefill_chunk(
        self,
        job: _GenerationJob,
    ) -> _PreparedRootWave:
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
        prefill_limit = _prefill_token_limit(self.config)
        if prefill_limit is not None:
            chunk_size = min(chunk_size, prefill_limit)
        end = min(total, start + chunk_size)
        input_chunk = job.request.input_ids[:, start:end].to(
            dtype=torch.long,
            device="cpu",
        )
        if prefill_limit is not None and input_chunk.shape[1] > prefill_limit:
            raise RuntimeError(
                "prepared prefill chunk exceeds max_prefill_chunk_tokens: "
                f"{input_chunk.shape[1]} > {prefill_limit}"
            )
        next_step = job.next_step
        if next_step is None:
            raise RuntimeError("prefill job lost its next outbound step")
        reservation = _prefill_frame_byte_reservation(
            TensorCodec(self.config.codec),
            int(input_chunk.shape[1]),
            self.hidden_size,
        )
        byte_limit = _prefill_inflight_byte_limit(self.config)
        if byte_limit > 0 and reservation > byte_limit:
            raise RuntimeError(
                f"request {job.wire_id} prefill chunk reserves {reservation} bytes, "
                f"exceeding prefill_inflight_bytes={byte_limit}"
            )
        return _PreparedRootWave(
            job=job,
            input_ids=input_chunk,
            frame_type=(
                FrameType.ACTIVATION if end == total else FrameType.PREFILL
            ),
            step=next_step,
            prefill_end=end,
            reserved_bytes=reservation,
        )

    def _receive_loop(self) -> None:
        return_socket = self._return_socket
        if return_socket is None:
            return
        try:
            while True:
                frame = recv_frame(return_socket)
                arrived = time.perf_counter()
                if frame.frame_type == FrameType.PONG:
                    # Publish receipt and timestamp under the same lock used by
                    # the scheduler's deadline check.  Only the scheduler
                    # mutates the probe state after startup.
                    with self._route_probe_lock:
                        self._route_probe_returns.put((frame, arrived))
                    continue
                if frame.frame_type == FrameType.VERIFY_RESULT:
                    self._record_queued_tree_return(frame.request_id, arrived)
                elif frame.frame_type == FrameType.TREE_PREPARE_RESULT:
                    self._record_queued_tree_prepare_result(
                        frame.request_id, frame.step, arrived
                    )
                elif frame.frame_type == FrameType.TREE_RESERVATION_COMMIT_RESULT:
                    self._record_queued_tree_commit_result(
                        frame.request_id, frame.step, arrived
                    )
                # TOKEN callbacks run in the scheduler after the immutable FIFO
                # flight record is consumed. With multiple prefill chunks in
                # flight the reader can observe the final TOKEN before the
                # scheduler has advanced through preceding ACKs; invoking here
                # would either drop or mis-order that first streamed token.
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

    def _half_close_downstream_write(self) -> None:
        downstream = self._downstream
        if downstream is None:
            return
        try:
            downstream.shutdown(socket.SHUT_WR)
        except OSError:
            # A peer that consumed SHUTDOWN and closed immediately has already
            # satisfied the ordering guarantee.
            pass

    def _close_transport_handles(self, cleanup_errors: list[str]) -> None:
        for sock in (self._return_socket, self._downstream, self._return_listener):
            if sock is not None:
                try:
                    sock.close()
                except OSError as error:
                    cleanup_errors.append(
                        f"socket close failed: {type(error).__name__}: {error}"
                    )

    @staticmethod
    def _component_is_alive(component: Any | None) -> bool:
        if component is None:
            return False
        try:
            return bool(component.is_alive())
        except BaseException:
            # If lifecycle state cannot be certified, fail closed.
            return True

    @classmethod
    def _join_components(
        cls,
        components: tuple[Any | None, ...],
        timeout_seconds: float,
    ) -> list[str]:
        errors: list[str] = []
        deadline = time.monotonic() + max(0.0, timeout_seconds)
        for component in components:
            if component is None or not cls._component_is_alive(component):
                continue
            remaining = max(0.0, deadline - time.monotonic())
            try:
                component.join(timeout=remaining)
            except BaseException as error:
                # The final report calls is_alive/exitcode and converts an
                # uncertifiable lifecycle into PipelineShutdownError.
                errors.append(
                    f"join {getattr(component, 'name', 'component')} failed: "
                    f"{type(error).__name__}: {error}"
                )
        return errors

    def _next_request_id(self) -> int:
        if self._request_counter >= (1 << 64) - 1:
            raise OverflowError("pipeline request ID space is exhausted")
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
    """Legacy reference oracle for exact linear speculative acceptance.

    The physical engine resolves through ``resolve_linear_macro_wave``.  This
    compact oracle remains independent so adapter equivalence tests can detect
    a semantic drift in correction or bonus handling.
    """

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


def _root_token_mode(frame_type: FrameType) -> str:
    if frame_type == FrameType.PREFILL:
        return "none"
    if frame_type == FrameType.VERIFY:
        return "all"
    if frame_type == FrameType.ACTIVATION:
        return "last"
    raise ValueError(f"{frame_type.name} is not a root activation frame")


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
