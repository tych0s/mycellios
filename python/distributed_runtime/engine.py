from __future__ import annotations

from concurrent.futures import CancelledError as FutureCancelledError, Future
from collections import deque
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

from .macro_wave import KVVersion, MacroWaveState
from .macro_wave_adapter import (
    MacroWaveProposal,
    prepare_linear_macro_wave,
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
    configure_socket,
    decode_token,
    decode_verify_result,
    encode_tensor_payload,
    recv_frame,
    send_frame,
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
)
from .stage import StageProcessConfig, connect_with_retry, drain_metrics, run_stage_process


TokenCallback = Callable[[int, int, int, float], None]


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

    def __post_init__(self) -> None:
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


@dataclass(frozen=True)
class GenerationOutput:
    client_id: int
    token_ids: tuple[int, ...]
    finish_reason: str
    ttft_ms: float
    tpot_ms: float
    total_ms: float


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
    sealed_wave_tokens: int
    max_prefill_chunk_tokens: int
    speculative_max_draft_tokens: int
    speculation_minimum_speedup: float
    speculation_probe: bool


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
    verify_proposal: MacroWaveProposal | None = None
    verify_base_tokens: int = 0
    speculation_profile: str = "load-1"


@dataclass(frozen=True)
class _PreparedRootWave:
    """One request continuation whose root KV has not been advanced yet."""

    job: _GenerationJob
    input_ids: torch.Tensor
    frame_type: FrameType
    prefill_end: int | None = None


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
        self._root_ready_items = 0
        self._root_model_forward_calls = 0
        self._root_physical_batch_calls = 0
        self._root_physical_batch_items = 0
        self._root_sequential_items = 0
        self._root_max_physical_batch_size = 1
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
    def recovery_identity(self) -> PipelineRecoveryIdentity:
        stage_executor_ids = self._recovery_stage_executor_ids()
        return PipelineRecoveryIdentity(
            schema_version=2,
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
        if controller is None:
            return {
                "configured": False,
                "enabled": False,
                "route_rtt_ms": route_rtt_ms,
                "route_probe_count": route_probe_count,
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
            runner = self._runner
            self._runner = None
            if runner is not None:
                close = getattr(runner, "close", None)
                if callable(close):
                    close()

    def __enter__(self) -> "DistributedPipelineEngine":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

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
                        connect_timeout_seconds=config.startup_timeout_seconds,
                        sealed_wave_tokens=config.sealed_wave_tokens,
                        max_prefill_chunk_tokens=config.max_prefill_chunk_tokens,
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
            StageRunner(root_spec)
            if root_ram_config is None
            else build_ram_backed_moe_stage_runner(
                root_spec,
                root_ram_config,
                pipeline_snapshot_identity=self.pipeline_id,
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
                ready_values = self._collect_ready_return_values(value)
                prepared: list[_PreparedRootWave] = []
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
        available = self.config.max_active_sequences - len(active)
        if available <= 0:
            self._deferred_batches.appendleft(batch)
            return
        selected = batch[:available]
        if len(selected) < len(batch):
            self._deferred_batches.appendleft(batch[available:])
        prepared: list[_PreparedRootWave] = []
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
                wire_id = self._next_request_id()
                job.wire_id = wire_id
                job.started_at = time.perf_counter()
                active[wire_id] = job
                with self._callback_lock:
                    self._callback_routes[wire_id] = job
                runner.begin(wire_id)
                send_frame(downstream, FrameType.BEGIN, wire_id)
                prepared.append(self._prepare_next_prefill_chunk(job))
            except BaseException as error:
                for remaining in selected[index + 1 :]:
                    self._retire_job(remaining, runner, exception=error, began=False)
                raise
        self._dispatch_root_waves(prepared, runner, downstream, emulator)

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
    ) -> _PreparedRootWave | None:
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
        if job.cancel_requested.is_set():
            if not job.cancel_sent:
                send_frame(downstream, FrameType.CANCEL, frame.request_id)
                job.cancel_sent = True
            active.pop(frame.request_id)
            self._retire_job(
                job,
                runner,
                exception=GenerationCancelledError("generation cancelled"),
            )
            return None

        if frame.frame_type == FrameType.PREFILL_ACK:
            if job.prefill_offset >= int(job.request.input_ids.shape[1]):
                raise RuntimeError(
                    f"request {frame.request_id} returned an unexpected prefill ACK"
                )
            job.step += 1
            return self._prepare_next_prefill_chunk(job)

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
                    latency_seconds=max(1e-9, arrived - job.wave_started_at),
                    transferred_bytes=(
                        job.last_outbound_bytes + HEADER_BYTES + len(frame.payload)
                    ),
                )
            verify_base_tokens = job.verify_base_tokens
            job.verify_proposal = None
            job.verify_base_tokens = 0
            reason = self._append_verified_tokens(
                job,
                resolution.emitted_tokens,
                arrived,
            )
            if reason is not None:
                send_frame(downstream, FrameType.END, frame.request_id)
                active.pop(frame.request_id)
                self._retire_job(
                    job,
                    runner,
                    result=self._generation_output(job, reason),
                )
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
            job.step += 1
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
            send_frame(downstream, FrameType.END, frame.request_id)
            active.pop(frame.request_id)
            self._retire_job(
                job,
                runner,
                result=self._generation_output(job, "stop" if reached_eos else "length"),
            )
            return None

        job.step += 1
        return self._prepare_decode_wave(
            job,
            runner,
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

    def _prepare_decode_wave(
        self,
        job: _GenerationJob,
        runner: StageRunner,
        *,
        active_sequences: int,
    ) -> _PreparedRootWave:
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
        return _PreparedRootWave(
            job=job,
            input_ids=next_ids,
            frame_type=frame_type,
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
        )

    def _retire_job(
        self,
        job: _GenerationJob,
        runner: StageRunner,
        *,
        result: GenerationOutput | None = None,
        exception: BaseException | None = None,
        began: bool = True,
    ) -> None:
        proposal = job.verify_proposal
        if proposal is not None:
            if proposal.tree.state is MacroWaveState.OPEN:
                proposal.tree.rollback()
            job.verify_proposal = None
            job.verify_base_tokens = 0
        if began and job.wire_id is not None:
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

    def _dispatch_root_waves(
        self,
        waves: list[_PreparedRootWave],
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

        if not waves:
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
                        job.prefill_offset = wave.prefill_end
                    job.last_outbound_bytes = self._send_activation(
                        downstream,
                        emulator,
                        job.wire_id,
                        job.step,
                        hidden,
                        frame_type=wave.frame_type,
                    )
                    job.last_sent_at = time.monotonic()

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
        return _PreparedRootWave(
            job=job,
            input_ids=input_chunk,
            frame_type=(
                FrameType.ACTIVATION if end == total else FrameType.PREFILL
            ),
            prefill_end=end,
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
