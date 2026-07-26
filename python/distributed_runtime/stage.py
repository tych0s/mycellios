from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
import hashlib
import math
from pathlib import Path
import queue
import select
import socket
import threading
import time
from typing import TYPE_CHECKING, Any

import torch

from .compute_timing import ComputeTimer
from .device import normalize_torch_device_request
from .dense_tiering import DenseTieringConfig
from .model import (
    MAX_PHYSICAL_STAGE_BATCH_SIZE,
    StageModelSpec,
    StageRunner,
    StageRunnerContract,
    ragged_grouping_enabled,
)
from .native_gguf_runtime import NativeGgufRuntimeConfig
from .protocol import (
    Frame,
    FrameType,
    LinkEmulator,
    TensorCodec,
    TreePrepareQuote,
    TreePrepareRejection,
    TreePrepareStatus,
    configure_socket,
    decode_branch_request_id,
    decode_tree_prepare,
    decode_tree_reservation_nonce,
    decode_tensor,
    encode_tree_prepare_quote,
    encode_tensor_payload,
    recv_frame,
    send_frame,
    token_payload,
    verify_result_payload,
)
from .ram_backed_moe_runtime import (
    RamBackedMoeRuntimeConfig,
    build_ram_backed_moe_stage_runner,
    validate_ram_backed_moe_binding,
)

if TYPE_CHECKING:
    from .paged_stage import HFPagedStageRuntimeConfig

MAX_SPECULATIVE_BRANCHES = 64
MAX_SPECULATIVE_BRANCH_TOKENS = 1_048_576
STAGE_SHUTDOWN_GRACE_SECONDS = 5.0
MAX_SPECULATIVE_KV_BYTES = 1 << 40
TREE_RESERVATION_TTL_SECONDS = 120.0


@dataclass(frozen=True)
class TreeCapacityProjection:
    projected_kv_bytes: int
    current_branch_count: int
    current_branch_bytes: int
    parent_tokens: int
    parent_cache_bytes: int
    branch_ids: tuple[int, ...]
    available_physical_bytes: int | None = None
    rejection: TreePrepareRejection = TreePrepareRejection.NONE
    required: int = 0
    limit: int = 0


@dataclass
class TreeCapacityReservation:
    nonce: int
    parent_request_id: int
    step: int
    path_lengths: tuple[int, ...]
    projection: TreeCapacityProjection
    expected_commit_predecessors: int
    deadline_at: float
    committed: bool = False
    consumed_forks: int = 0
    quoted_children: dict[int, int] = field(default_factory=dict)
    consumed_verifies: set[int] = field(default_factory=set)


@dataclass(frozen=True)
class TreeLeafShape:
    parent_request_id: int
    nonce: int
    expected_tokens: int


@dataclass
class TreeReservationBook:
    """One globally serialized capacity quote for a physical stage.

    The quote allocates no KV. Its count/byte guarantee is made transactional by
    deferring every unrelated frame that can mutate runtime-visible KV, blocking
    another PREPARE/FORK, and revalidating the exact snapshot at COMMIT. This
    seals both the configured budget and, when the runner exposes it, its
    runner-owned physical KV pool.  It is not a reservation of arbitrary CUDA
    workspace or process-wide VRAM outside that pool.  The engine must serialize
    one global tree wave from PREPARE until RESULT+CANCEL or the end-to-end
    COMMIT result plus every quoted VERIFY has materialised its KV growth.
    """

    active: TreeCapacityReservation | None = None
    last_nonce: int | None = None
    last_parent_request_id: int | None = None
    last_step: int | None = None
    last_mutated: bool = False


@dataclass
class StageBranchLedger:
    """Separate logical request ownership from physical COW ancestry.

    ``logical_owners`` is the live accounting/lifecycle relation: every
    speculative request in one tree belongs directly to its ordinary root
    request.  ``fork_sources`` records which *physical* request supplied the KV
    snapshot for each fork.  The source may itself be speculative and may be
    ended after its descendants exist; a correct COW backend keeps those
    descendants alive through block refcounts (or independent safe copies).

    Keeping these relations separate is what makes a carrier -> nested fork ->
    carrier END -> descendant PROMOTE lifecycle possible without pretending
    that the carrier owns the user request or subtracting it twice from the
    speculative-KV budget.
    """

    logical_owners: dict[int, int] = field(default_factory=dict)
    fork_sources: dict[int, int] = field(default_factory=dict)

    def register(
        self,
        child_request_id: int,
        *,
        logical_owner_request_id: int,
        fork_source_physical_request_id: int,
    ) -> None:
        if (
            child_request_id in self.logical_owners
            or child_request_id in self.fork_sources
        ):
            raise ValueError(
                f"speculative request {child_request_id} is already registered"
            )
        if logical_owner_request_id in self.logical_owners:
            raise ValueError("logical owner must be an ordinary root request")
        if fork_source_physical_request_id != logical_owner_request_id:
            source_owner = self.logical_owners.get(fork_source_physical_request_id)
            if source_owner != logical_owner_request_id:
                raise ValueError(
                    "physical fork source does not belong to the logical owner"
                )
        self.logical_owners[child_request_id] = logical_owner_request_id
        self.fork_sources[child_request_id] = fork_source_physical_request_id

    def release(self, request_id: int) -> None:
        self.logical_owners.pop(request_id, None)
        self.fork_sources.pop(request_id, None)

    def owned_requests(self, logical_owner_request_id: int) -> set[int]:
        return {
            request_id
            for request_id, owner_request_id in self.logical_owners.items()
            if owner_request_id == logical_owner_request_id
        }


@dataclass(frozen=True)
class StageProcessConfig:
    spec: StageModelSpec
    pipeline_id: int
    listen_host: str
    listen_port: int
    next_host: str | None
    next_port: int | None
    next_layer_end: int | None
    return_host: str
    return_port: int
    codec: TensorCodec
    one_way_delay_ms: float
    bandwidth_mbps: float
    device: str = "auto"
    dense_tiering: DenseTieringConfig = DenseTieringConfig()
    sealed_wave_tokens: int | None = None
    max_prefill_chunk_tokens: int | None = None
    connect_timeout_seconds: float = 120.0
    cell_fixture: str | None = None
    cell_manifest_sha256: str | None = None
    cell_world_size: int | None = None
    cell_collective_backend: str = "gloo"
    cell_devices: tuple[str, ...] = ()
    cell_compute_dtype: str = "float32"
    cell_operation_timeout_seconds: float = 30.0
    cell_mode: str = "local"
    cell_control_host: str | None = None
    cell_control_port: int | None = None
    cell_control_advertise_host: str | None = None
    cell_distributed_advertise_host: str | None = None
    cell_distributed_port: int | None = None
    cell_startup_timeout_seconds: float = 120.0
    max_physical_batch_size: int = 8
    physical_batch_window_ms: float = 0.5
    # Exact tree controls are disabled unless all three sealed limits are
    # non-zero. Count, tokens and preflight KV bytes bound concurrent leaves
    # before any clone/forward allocation is attempted.
    max_speculative_branches: int = 0
    max_speculative_branch_tokens: int = 0
    max_speculative_kv_bytes: int = 0
    ram_backed_moe: RamBackedMoeRuntimeConfig | None = None
    paged_kv: HFPagedStageRuntimeConfig | None = None
    native_gguf: NativeGgufRuntimeConfig | None = None


_REQUEST_SCOPED_FRAMES = frozenset(
    (
        FrameType.BEGIN,
        FrameType.ACTIVATION,
        FrameType.PREFILL,
        FrameType.VERIFY,
        FrameType.TRUNCATE,
        FrameType.END,
        FrameType.CANCEL,
    )
)


class SingleRequestAdmission:
    """Serialize a multiplexed stage stream onto one native KV sequence.

    Native runners may advertise a single active request. Other request frames
    are retained per request while the admitted request continues to consume
    the upstream stream. This preserves each request's frame order without
    head-of-line blocking the active request behind a queued BEGIN.
    """

    def __init__(self) -> None:
        self.active_request: int | None = None
        self._order: deque[int] = deque()
        self._frames: dict[int, deque[Frame]] = {}

    def admit_or_defer(self, frame: Frame) -> bool:
        if frame.frame_type not in _REQUEST_SCOPED_FRAMES:
            return True
        if self.active_request is None:
            if frame.frame_type == FrameType.BEGIN:
                self.active_request = frame.request_id
            return True
        if frame.request_id == self.active_request:
            return True
        frames = self._frames.get(frame.request_id)
        if frames is None:
            frames = deque()
            self._frames[frame.request_id] = frames
            self._order.append(frame.request_id)
        frames.append(frame)
        return False

    def next_deferred(self) -> Frame | None:
        request_id = self.active_request
        if request_id is None:
            if not self._order:
                return None
            request_id = self._order.popleft()
        frames = self._frames.get(request_id)
        if not frames:
            return None
        frame = frames.popleft()
        if not frames:
            self._frames.pop(request_id, None)
        return frame

    def release(self, request_id: int) -> None:
        if self.active_request != request_id:
            raise RuntimeError("single-request admission released the wrong request")
        self.active_request = None


def request_admission_for_runner(
    runner: StageRunnerContract,
) -> SingleRequestAdmission | None:
    maximum = getattr(runner, "max_active_requests", None)
    if maximum is None:
        return None
    if maximum != 1:
        raise ValueError("stage runner max_active_requests is unsupported")
    return SingleRequestAdmission()


def validate_speculative_runner(
    config: StageProcessConfig,
    runner: StageRunnerContract,
) -> None:
    """Fail before READY when sealed tree controls exceed backend capability."""

    if config.max_speculative_branches == 0:
        return
    if getattr(runner, "max_active_requests", None) == 1:
        raise ValueError("speculative branches require a multi-request stage runner")
    for method_name in (
        "request_cache_bytes",
        "project_request_cache_bytes",
        "fork_request",
        "promote_request",
    ):
        if not callable(getattr(runner, method_name, None)):
            raise ValueError(
                f"stage runner does not support exact speculative {method_name}"
            )


def run_stage_process(
    config: StageProcessConfig,
    ready_event: Any,
    metrics_queue: Any,
) -> None:
    runner: StageRunnerContract | None = None
    listener: socket.socket | None = None
    upstream: socket.socket | None = None
    downstream: socket.socket | None = None
    return_socket: socket.socket | None = None
    request_metrics: dict[int, dict[str, Any]] = {}
    stopping = threading.Event()
    downstream_failed = threading.Event()
    downstream_errors: list[BaseException] = []
    upstream_send_lock = threading.Lock()
    control_thread: threading.Thread | None = None
    pending_frames: deque[Frame] = deque()
    reservation_deferred_frames: deque[Frame] = deque()
    request_admission: SingleRequestAdmission | None = None
    branch_lineage = StageBranchLedger()
    # Kept as a local alias because the v6 capacity/validation helpers consume
    # a Mapping. Its values are logical roots, never nested physical sources.
    branch_parents = branch_lineage.logical_owners
    tree_reservations = TreeReservationBook()
    tree_leaf_shapes: dict[int, TreeLeafShape] = {}
    emulator: LinkEmulator | None = None
    stage_failed = False
    try:
        validate_stage_config(config)
        runner = build_stage_runner(config)
        validate_speculative_runner(config, runner)
        request_admission = request_admission_for_runner(runner)
        put_startup_metric_best_effort(
            metrics_queue,
            {
                "event": "stage_runtime_ready",
                "stage": config.spec.layer_start,
                "layer_end": config.spec.layer_end,
                "execution": execution_metric_snapshot(runner),
            },
        )
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((config.listen_host, config.listen_port))
        listener.listen(1)
        ready_event.set()
        emulator = LinkEmulator(config.one_way_delay_ms, config.bandwidth_mbps)

        upstream, _ = listener.accept()
        configure_socket(upstream)
        listener.close()
        listener = None
        hello = recv_frame(upstream)
        validate_hello(hello, config, runner.hidden_size)

        if config.next_host is not None and config.next_port is not None:
            downstream = connect_with_retry(
                config.next_host,
                config.next_port,
                config.connect_timeout_seconds,
            )
            send_frame(
                downstream,
                FrameType.HELLO,
                config.pipeline_id,
                step=config.spec.layer_end,
                token_count=config.next_layer_end or 0,
                hidden_size=runner.hidden_size,
                flags=int(config.codec),
            )
            ready = recv_frame(downstream)
            if ready.frame_type == FrameType.ERROR:
                message = ready.payload.decode("utf-8", errors="replace")
                raise RuntimeError(f"downstream stage failed during startup: {message}")
            if ready.frame_type != FrameType.READY:
                raise RuntimeError("downstream stage did not become ready")
            if ready.request_id != hello.request_id:
                raise RuntimeError("downstream READY belongs to another pipeline session")
            control_thread = threading.Thread(
                target=monitor_downstream_control,
                args=(
                    downstream,
                    upstream,
                    config.pipeline_id,
                    stopping,
                    downstream_failed,
                    downstream_errors,
                    upstream_send_lock,
                ),
                name=f"stage-{config.spec.layer_start}-downstream-control",
                daemon=True,
            )
            control_thread.start()
        else:
            return_socket = connect_with_retry(
                config.return_host,
                config.return_port,
                config.connect_timeout_seconds,
            )

        # The downstream monitor is already active and may need to relay an
        # immediate ERROR. Serialize both writes so their headers/payloads can
        # never interleave on the shared upstream TCP stream.
        with upstream_send_lock:
            send_frame(upstream, FrameType.READY, config.pipeline_id)
        while True:
            expire_tree_reservation(tree_reservations)
            frame = (
                reservation_deferred_frames.popleft()
                if tree_reservations.active is None
                and reservation_deferred_frames
                else None
            )
            if frame is None:
                frame = pending_frames.popleft() if pending_frames else None
            if frame is None and request_admission is not None:
                frame = request_admission.next_deferred()
            if frame is None:
                frame = recv_frame(upstream)
            if tree_reservation_defers_frame(
                frame, tree_reservations, branch_parents
            ):
                reservation_deferred_frames.append(frame)
                continue
            validate_tree_reservation_frame_order(
                frame, tree_reservations, branch_parents
            )
            if (
                request_admission is not None
                and not request_admission.admit_or_defer(frame)
            ):
                continue
            if frame.frame_type == FrameType.BEGIN:
                begin_stage_request(frame, config, runner, request_metrics, downstream)
            elif frame.frame_type == FrameType.TREE_PREPARE:
                prepare_tree_capacity(
                    frame,
                    config=config,
                    runner=runner,
                    request_metrics=request_metrics,
                    branch_parents=branch_parents,
                    reservations=tree_reservations,
                    downstream=downstream,
                    return_socket=return_socket,
                    emulator=emulator,
                )
            elif frame.frame_type == FrameType.TREE_RESERVATION_COMMIT:
                commit_tree_reservation(
                    frame,
                    config=config,
                    runner=runner,
                    request_metrics=request_metrics,
                    branch_parents=branch_parents,
                    reservations=tree_reservations,
                    downstream=downstream,
                    return_socket=return_socket,
                    emulator=emulator,
                )
            elif frame.frame_type == FrameType.TREE_RESERVATION_CANCEL:
                cancel_tree_reservation(
                    frame,
                    reservations=tree_reservations,
                    downstream=downstream,
                )
            elif frame.frame_type == FrameType.FORK:
                fork_stage_request(
                    frame,
                    config,
                    runner,
                    request_metrics,
                    branch_parents,
                    downstream,
                    tree_reservations,
                    tree_leaf_shapes,
                    branch_lineage,
                )
            elif frame.frame_type == FrameType.PROMOTE:
                promote_stage_request(
                    frame,
                    config,
                    runner,
                    request_metrics,
                    branch_parents,
                    downstream,
                    tree_leaf_shapes,
                    branch_lineage,
                )
            elif frame.frame_type in (
                FrameType.ACTIVATION,
                FrameType.PREFILL,
                FrameType.VERIFY,
            ):
                validate_activation(
                    frame,
                    config,
                    runner,
                    request_metrics,
                    branch_parents,
                    tree_leaf_shapes,
                )
                frames = collect_compatible_activation_frames(
                    frame,
                    upstream=upstream,
                    pending_frames=pending_frames,
                    config=config,
                    runner=runner,
                    request_metrics=request_metrics,
                    branch_parents=branch_parents,
                    tree_leaf_shapes=tree_leaf_shapes,
                    tree_reservations=tree_reservations,
                    downstream=downstream,
                )
                process_activation_frames(
                    frames,
                    config=config,
                    runner=runner,
                    request_metrics=request_metrics,
                    branch_parents=branch_parents,
                    downstream=downstream,
                    return_socket=return_socket,
                    emulator=emulator,
                )
                consume_tree_reservation_verifies(frames, tree_reservations)
            elif frame.frame_type == FrameType.TRUNCATE:
                runner.truncate(frame.request_id, frame.token_count)
                if downstream is not None:
                    send_frame(
                        downstream,
                        FrameType.TRUNCATE,
                        frame.request_id,
                        token_count=frame.token_count,
                    )
            elif frame.frame_type in (FrameType.END, FrameType.CANCEL):
                metrics = end_physical_stage_request(
                    frame.request_id,
                    operation=frame.frame_type.name,
                    runner=runner,
                    request_metrics=request_metrics,
                    branch_lineage=branch_lineage,
                )
                tree_leaf_shapes.pop(frame.request_id, None)
                abandon_tree_reservation_for_request(
                    tree_reservations, frame.request_id
                )
                if downstream is not None:
                    send_frame(downstream, frame.frame_type, frame.request_id)
                if metrics is not None:
                    metrics["cell_rank_work"] = cell_rank_work_metric(runner)
                    put_metric_best_effort(
                        metrics_queue, {"request_id": frame.request_id, **metrics}
                    )
                if request_admission is not None:
                    request_admission.release(frame.request_id)
            elif frame.frame_type == FrameType.PING:
                if frame.request_id != config.pipeline_id:
                    raise ValueError("route PING belongs to another pipeline session")
                if downstream is not None:
                    send_frame(
                        downstream,
                        FrameType.PING,
                        frame.request_id,
                        step=frame.step,
                        emulator=emulator,
                    )
                else:
                    if return_socket is None:
                        raise RuntimeError("last stage has no route-probe return socket")
                    send_frame(
                        return_socket,
                        FrameType.PONG,
                        frame.request_id,
                        step=frame.step,
                        emulator=emulator,
                    )
            elif frame.frame_type == FrameType.SHUTDOWN:
                # Mark the lifecycle transition before forwarding SHUTDOWN. The
                # downstream monitor may observe the peer's ensuing EOF first.
                stopping.set()
                tree_reservations.active = None
                tree_leaf_shapes.clear()
                branch_lineage.logical_owners.clear()
                branch_lineage.fork_sources.clear()
                if downstream is not None:
                    forward_shutdown_and_wait(
                        downstream,
                        frame.request_id,
                        control_thread,
                        emulator=emulator,
                    )
                break
            elif frame.frame_type == FrameType.ERROR:
                message = frame.payload.decode("utf-8", errors="replace")
                raise RuntimeError(f"upstream stage reported an error: {message}")
            else:
                raise ValueError(f"unexpected frame {frame.frame_type.name}")
    except BaseException as error:
        stage_failed = True
        stopping.set()
        original_error = error
        if downstream_failed.is_set() and downstream_errors:
            error = downstream_errors[0]
        # The downstream monitor already relayed its original ERROR/EOF before
        # interrupting our blocking receive. Avoid replacing it with a second,
        # less useful local EOF. Other failures are reported on every route the
        # root may currently be reading.
        if not downstream_failed.is_set():
            payload = str(error).encode("utf-8")[:4_096]
            send_error_best_effort(
                upstream,
                config.pipeline_id,
                payload,
                upstream_send_lock,
            )
            if return_socket is not None:
                send_error_best_effort(return_socket, config.pipeline_id, payload)
        put_metric_best_effort(
            metrics_queue,
            {
                "fatal_error": f"{type(error).__name__}: {error}",
                "stage": config.spec.layer_start,
                "config": asdict(config),
            }
        )
        if error is not original_error:
            raise error from original_error
        raise
    finally:
        stopping.set()
        emulator_close_error: BaseException | None = None
        if emulator is not None:
            try:
                emulator.close(
                    timeout_seconds=config.connect_timeout_seconds,
                )
            except BaseException as error:
                emulator_close_error = error
        if downstream is not None:
            try:
                downstream.shutdown(socket.SHUT_RD)
            except OSError:
                pass
        for sock in (return_socket, downstream, upstream, listener):
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
        if control_thread is not None:
            control_thread.join(timeout=1.0)
        if runner is not None:
            close = getattr(runner, "close", None)
            if callable(close):
                close()
        if emulator_close_error is not None and not stage_failed:
            raise emulator_close_error


def begin_stage_request(
    frame: Frame,
    config: StageProcessConfig,
    runner: StageRunnerContract,
    request_metrics: dict[int, dict[str, Any]],
    downstream: socket.socket | None,
) -> None:
    """Apply BEGIN immediately, including while filling a tensor batch."""

    runner.begin(frame.request_id)
    request_metrics[frame.request_id] = {
        "stage": config.spec.layer_start,
        "layer_end": config.spec.layer_end,
        "parameter_bytes": runner.parameter_bytes,
        "loader": runner.loader,
        **executor_metric_fields(runner),
        "execution": execution_metric_snapshot(runner),
        "frames": 0,
        # Los dos acumuladores existen siempre y solo uno se llena, según el
        # régimen de medida. Un `compute_ms` a cero con
        # `compute_timing_mode="dispatch"` se lee como "no medido en kernel
        # time", que es visiblemente distinto de un número plausible pero mal
        # medido. El fallo ruidoso es el que queremos.
        "compute_ms": 0,
        "compute_dispatch_ms": 0,
        "compute_timing_mode": None,
        "bytes_out": 0,
        "tokens": 0,
        "model_forward_calls": 0,
        "physical_batch_calls": 0,
        "physical_batch_items": 0,
        "max_physical_batch_size": 0,
    }
    if downstream is not None:
        request_metrics[frame.request_id]["bytes_out"] += send_frame(
            downstream,
            FrameType.BEGIN,
            frame.request_id,
        )


def expire_tree_reservation(
    reservations: TreeReservationBook,
    *,
    now: float | None = None,
) -> bool:
    """Drop an abandoned logical quote without touching model/KV state."""

    active = reservations.active
    if active is None:
        return False
    observed = time.monotonic() if now is None else now
    if observed < active.deadline_at:
        return False
    if active.consumed_forks:
        raise TimeoutError(
            "committed tree reservation expired after speculative KV mutation"
        )
    reservations.active = None
    return True


def tree_reservation_defers_frame(
    frame: Frame,
    reservations: TreeReservationBook,
    branch_parents: Mapping[int, int],
) -> bool:
    """Hold every unrelated KV mutation behind one active tree transaction.

    Serializing runtime-visible cache mutations from PREPARE until every quoted
    VERIFY has materialised its KV prevents another request from stealing the
    runner-owned blocks promised by the quote.  Control frames and mutations of
    the quoted parent/children are never hidden here: malformed ordering must
    reach the validator and fail the route immediately.
    """

    active = reservations.active
    if active is None:
        return False
    if (
        frame.request_id == active.parent_request_id
        or frame.request_id in active.quoted_children
    ):
        return False
    del branch_parents  # all ordinary KV owners share the same allocator budget
    return frame.frame_type in {
        FrameType.BEGIN,
        FrameType.ACTIVATION,
        FrameType.PREFILL,
        FrameType.VERIFY,
        FrameType.TRUNCATE,
        FrameType.PROMOTE,
        FrameType.END,
        FrameType.CANCEL,
    }


def validate_tree_reservation_frame_order(
    frame: Frame,
    reservations: TreeReservationBook,
    branch_parents: Mapping[int, int],
) -> None:
    """Allow only the canonical tree mutation at each reservation phase."""

    active = reservations.active
    if active is None:
        return
    always_safe = {
        FrameType.PING,
        FrameType.SHUTDOWN,
        FrameType.ERROR,
        FrameType.TREE_RESERVATION_CANCEL,
    }
    if frame.frame_type in always_safe:
        return
    if frame.frame_type == FrameType.TREE_RESERVATION_COMMIT:
        if active.committed:
            raise ValueError("tree reservation COMMIT cannot be replayed")
        return
    if frame.frame_type == FrameType.FORK and active.committed:
        return
    if frame.frame_type == FrameType.VERIFY and active.committed:
        if active.consumed_forks != len(active.path_lengths):
            raise ValueError("tree VERIFY cannot overtake quoted FORKs")
        expected_tokens = active.quoted_children.get(frame.request_id)
        if expected_tokens is None:
            raise ValueError("tree VERIFY is not bound to a quoted child")
        if branch_parents.get(frame.request_id) != active.parent_request_id:
            raise ValueError("quoted tree VERIFY lost its parent binding")
        if frame.request_id in active.consumed_verifies:
            raise ValueError("quoted tree VERIFY cannot be replayed")
        if frame.token_count != expected_tokens:
            raise ValueError(
                "quoted tree VERIFY token count mismatch: "
                f"got {frame.token_count}, expected {expected_tokens}"
            )
        return
    del branch_parents
    phase = "committed" if active.committed else "prepared"
    raise ValueError(
        f"{frame.frame_type.name} cannot overtake a {phase} tree reservation"
    )


def project_tree_capacity(
    path_lengths: tuple[int, ...],
    *,
    parent_request_id: int,
    config: StageProcessConfig,
    runner: StageRunnerContract,
    branch_parents: Mapping[int, int],
) -> TreeCapacityProjection:
    """Compute the exact local quote without cloning or extending any KV cache."""

    if not path_lengths or len(path_lengths) > MAX_SPECULATIVE_BRANCHES:
        raise ValueError("tree capacity quote has an invalid path count")
    if any(
        not isinstance(length, int) or isinstance(length, bool) or length < 1
        for length in path_lengths
    ):
        raise ValueError("tree capacity path lengths must be positive integers")
    parent_tokens = runner.sequence_length(parent_request_id)
    if not isinstance(parent_tokens, int) or isinstance(parent_tokens, bool):
        raise TypeError("stage runner sequence_length must return an integer")
    if parent_tokens < 1:
        raise ValueError("TREE_PREPARE requires a non-empty parent KV cache")

    branch_ids = tuple(sorted(branch_parents))
    branch_count = len(branch_ids)
    if config.max_speculative_branches == 0:
        return TreeCapacityProjection(
            projected_kv_bytes=0,
            current_branch_count=branch_count,
            current_branch_bytes=0,
            parent_tokens=parent_tokens,
            parent_cache_bytes=0,
            branch_ids=branch_ids,
            rejection=TreePrepareRejection.TREE_DISABLED,
            required=branch_count + len(path_lengths),
            limit=0,
        )

    branch_bytes = speculative_kv_bytes(runner, branch_parents)

    available_physical = getattr(runner, "available_physical_cache_bytes", None)
    available_physical_bytes: int | None = None
    if callable(available_physical):
        available_physical_bytes = available_physical()
        if (
            not isinstance(available_physical_bytes, int)
            or isinstance(available_physical_bytes, bool)
            or available_physical_bytes < 0
        ):
            raise TypeError("stage runner returned invalid available physical KV bytes")

    request_cache_bytes = getattr(runner, "request_cache_bytes", None)
    project_cache_bytes = getattr(runner, "project_request_cache_bytes", None)
    if not callable(request_cache_bytes) or not callable(project_cache_bytes):
        raise TypeError("stage runner cannot quote speculative KV capacity")
    parent_cache_bytes = request_cache_bytes(parent_request_id)
    if (
        not isinstance(parent_cache_bytes, int)
        or isinstance(parent_cache_bytes, bool)
        or parent_cache_bytes < 1
    ):
        raise TypeError("stage runner returned invalid parent cache bytes")

    delta_tokens_by_leaf = tuple(1 + length for length in path_lengths)
    project_tree_physical = getattr(
        runner, "project_tree_incremental_physical_cache_bytes", None
    )
    if callable(project_tree_physical):
        projected_new_bytes = project_tree_physical(
            parent_request_id,
            delta_tokens_by_leaf=delta_tokens_by_leaf,
        )
        if (
            not isinstance(projected_new_bytes, int)
            or isinstance(projected_new_bytes, bool)
            or projected_new_bytes < 0
        ):
            raise TypeError("stage runner returned invalid physical tree projection")
    else:
        projected_new_bytes = 0
        for delta_tokens in delta_tokens_by_leaf:
            projected = project_cache_bytes(parent_request_id, delta_tokens)
            if (
                not isinstance(projected, int)
                or isinstance(projected, bool)
                or projected < parent_cache_bytes
            ):
                raise TypeError("stage runner returned invalid projected tree KV bytes")
            projected_new_bytes += projected
    projected_total = branch_bytes + projected_new_bytes
    if projected_total > (1 << 64) - 1:
        raise ValueError("tree capacity projection exceeds uint64")

    # A projection is contractually read-only. Detect an executor that changed
    # any observable sequence/cache state while pricing the quote.
    if runner.sequence_length(parent_request_id) != parent_tokens:
        raise RuntimeError("tree capacity projection mutated parent sequence length")
    if request_cache_bytes(parent_request_id) != parent_cache_bytes:
        raise RuntimeError("tree capacity projection mutated parent cache bytes")
    if tuple(sorted(branch_parents)) != branch_ids:
        raise RuntimeError("tree capacity projection mutated speculative children")
    if speculative_kv_bytes(runner, branch_parents) != branch_bytes:
        raise RuntimeError("tree capacity projection mutated speculative KV bytes")
    if (
        callable(available_physical)
        and available_physical() != available_physical_bytes
    ):
        raise RuntimeError("tree capacity projection mutated physical KV availability")

    required_count = branch_count + len(path_lengths)
    required_tokens = parent_tokens + 1 + max(path_lengths)
    rejection = TreePrepareRejection.NONE
    required = 0
    limit = 0
    if required_count > config.max_speculative_branches:
        rejection = TreePrepareRejection.BRANCH_COUNT
        required = required_count
        limit = config.max_speculative_branches
    elif required_tokens > config.max_speculative_branch_tokens:
        rejection = TreePrepareRejection.BRANCH_TOKENS
        required = required_tokens
        limit = config.max_speculative_branch_tokens
    elif projected_total > config.max_speculative_kv_bytes:
        rejection = TreePrepareRejection.KV_BYTES
        required = projected_total
        limit = config.max_speculative_kv_bytes
    elif (
        available_physical_bytes is not None
        and projected_new_bytes > available_physical_bytes
    ):
        rejection = TreePrepareRejection.KV_BYTES
        required = projected_total
        limit = branch_bytes + available_physical_bytes
    return TreeCapacityProjection(
        projected_kv_bytes=projected_total,
        current_branch_count=branch_count,
        current_branch_bytes=branch_bytes,
        parent_tokens=parent_tokens,
        parent_cache_bytes=parent_cache_bytes,
        branch_ids=branch_ids,
        available_physical_bytes=available_physical_bytes,
        rejection=rejection,
        required=required,
        limit=limit,
    )


def prepare_tree_capacity(
    frame: Frame,
    *,
    config: StageProcessConfig,
    runner: StageRunnerContract,
    request_metrics: Mapping[int, Mapping[str, Any]],
    branch_parents: Mapping[int, int],
    reservations: TreeReservationBook,
    downstream: socket.socket | None,
    return_socket: socket.socket | None,
    emulator: LinkEmulator | None = None,
) -> TreePrepareQuote:
    """Accumulate one mutation-free local quote and forward or return it."""

    if frame.frame_type != FrameType.TREE_PREPARE:
        raise ValueError("expected TREE_PREPARE")
    if reservations.active is not None:
        raise ValueError("another tree reservation is already active")
    quote = decode_tree_prepare(frame)
    if quote.nonce == reservations.last_nonce:
        raise ValueError("tree reservation nonce cannot be replayed")
    if frame.request_id not in request_metrics:
        raise ValueError(
            f"TREE_PREPARE parent request {frame.request_id} is not active"
        )
    expected_step = int(request_metrics[frame.request_id]["frames"])
    if frame.step != expected_step:
        raise ValueError(
            f"TREE_PREPARE step mismatch for request {frame.request_id}: "
            f"got {frame.step}, expected {expected_step}"
        )
    if quote.stage_count >= (1 << 16) - 1:
        raise ValueError("tree prepare stage count overflow")

    reservations.last_nonce = quote.nonce
    reservations.last_parent_request_id = frame.request_id
    reservations.last_step = frame.step
    reservations.last_mutated = False
    projection = project_tree_capacity(
        quote.path_lengths,
        parent_request_id=frame.request_id,
        config=config,
        runner=runner,
        branch_parents=branch_parents,
    )
    aggregate_total = quote.total_projected_bytes + projection.projected_kv_bytes
    if aggregate_total > (1 << 64) - 1:
        raise ValueError("tree prepare aggregate projection exceeds uint64")

    status = quote.status
    rejection = quote.rejection
    rejecting_layer_start = quote.rejecting_layer_start
    required = quote.required
    limit = quote.limit
    local_ready = projection.rejection is TreePrepareRejection.NONE
    if status is TreePrepareStatus.READY and not local_ready:
        status = TreePrepareStatus.REJECT
        rejection = projection.rejection
        rejecting_layer_start = config.spec.layer_start
        required = projection.required
        limit = projection.limit
    accumulated = replace(
        quote,
        status=status,
        rejection=rejection,
        stage_count=quote.stage_count + 1,
        rejecting_layer_start=rejecting_layer_start,
        required=required,
        limit=limit,
        total_projected_bytes=aggregate_total,
    )
    payload = encode_tree_prepare_quote(accumulated)

    # READY means this stage has logically reserved the exact count/byte quote.
    # The reservation allocates no KV; serialization plus COMMIT revalidation
    # makes another PREPARE/FORK unable to consume its quoted capacity.
    if quote.status is TreePrepareStatus.READY and local_ready:
        reservations.active = TreeCapacityReservation(
            nonce=quote.nonce,
            parent_request_id=frame.request_id,
            step=frame.step,
            path_lengths=quote.path_lengths,
            projection=projection,
            expected_commit_predecessors=quote.stage_count,
            deadline_at=time.monotonic() + TREE_RESERVATION_TTL_SECONDS,
        )

    if downstream is not None:
        send_frame(
            downstream,
            FrameType.TREE_PREPARE,
            frame.request_id,
            step=frame.step,
            token_count=len(quote.path_lengths),
            payload=payload,
            emulator=emulator,
        )
    else:
        if return_socket is None:
            raise RuntimeError("last stage has no tree-quote return socket")
        send_frame(
            return_socket,
            FrameType.TREE_PREPARE_RESULT,
            frame.request_id,
            step=frame.step,
            token_count=len(quote.path_lengths),
            payload=payload,
            emulator=emulator,
        )
    return accumulated


def commit_tree_reservation(
    frame: Frame,
    *,
    config: StageProcessConfig,
    runner: StageRunnerContract,
    request_metrics: Mapping[int, Mapping[str, Any]],
    branch_parents: Mapping[int, int],
    reservations: TreeReservationBook,
    downstream: socket.socket | None,
    return_socket: socket.socket | None = None,
    emulator: LinkEmulator | None = None,
) -> None:
    """Revalidate and propagate COMMIT, returning a route-wide ACK at the end."""

    nonce = decode_tree_reservation_nonce(frame)
    active = reservations.active
    if active is None:
        raise ValueError("TREE_RESERVATION_COMMIT has no active quote")
    if (
        nonce != active.nonce
        or frame.request_id != active.parent_request_id
        or frame.step != active.step
    ):
        raise ValueError("TREE_RESERVATION_COMMIT identity mismatch")
    if active.committed:
        raise ValueError("TREE_RESERVATION_COMMIT cannot be replayed")
    if frame.token_count != active.expected_commit_predecessors:
        raise ValueError(
            "TREE_RESERVATION_COMMIT predecessor count mismatch: "
            f"got {frame.token_count}, expected {active.expected_commit_predecessors}"
        )
    if frame.request_id not in request_metrics:
        raise ValueError("tree reservation parent is no longer active")
    if int(request_metrics[frame.request_id]["frames"]) != active.step:
        raise RuntimeError("tree reservation parent step changed before COMMIT")
    observed = project_tree_capacity(
        active.path_lengths,
        parent_request_id=active.parent_request_id,
        config=config,
        runner=runner,
        branch_parents=branch_parents,
    )
    if observed.rejection is not TreePrepareRejection.NONE:
        raise RuntimeError("reserved tree capacity disappeared before COMMIT")
    if observed != active.projection:
        raise RuntimeError("tree reservation snapshot changed before COMMIT")
    active.committed = True
    active.deadline_at = time.monotonic() + TREE_RESERVATION_TTL_SECONDS
    committed_stage_count = frame.token_count + 1
    if downstream is not None:
        send_frame(
            downstream,
            FrameType.TREE_RESERVATION_COMMIT,
            frame.request_id,
            step=frame.step,
            token_count=committed_stage_count,
            payload=frame.payload,
            emulator=emulator,
        )
    else:
        if return_socket is None:
            raise RuntimeError("last stage has no tree-commit return socket")
        send_frame(
            return_socket,
            FrameType.TREE_RESERVATION_COMMIT_RESULT,
            frame.request_id,
            step=frame.step,
            token_count=committed_stage_count,
            payload=frame.payload,
            emulator=emulator,
        )


def cancel_tree_reservation(
    frame: Frame,
    *,
    reservations: TreeReservationBook,
    downstream: socket.socket | None,
) -> None:
    """Idempotently release this quote, never a different outstanding nonce."""

    nonce = decode_tree_reservation_nonce(frame)
    identity = (nonce, frame.request_id, frame.step)
    expected = (
        reservations.last_nonce,
        reservations.last_parent_request_id,
        reservations.last_step,
    )
    if identity != expected:
        raise ValueError("TREE_RESERVATION_CANCEL identity mismatch")
    active = reservations.active
    if active is not None:
        if identity != (active.nonce, active.parent_request_id, active.step):
            raise ValueError("TREE_RESERVATION_CANCEL targets another nonce")
        if active.consumed_forks:
            raise ValueError("cannot cancel a reservation after FORK consumption")
        reservations.active = None
    elif reservations.last_mutated:
        raise ValueError("cannot cancel a reservation after FORK consumption")
    if downstream is not None:
        send_frame(
            downstream,
            FrameType.TREE_RESERVATION_CANCEL,
            frame.request_id,
            step=frame.step,
            payload=frame.payload,
        )


def abandon_tree_reservation_for_request(
    reservations: TreeReservationBook,
    request_id: int,
) -> None:
    active = reservations.active
    if active is not None and active.parent_request_id == request_id:
        reservations.active = None


def _resolve_branch_lineage(
    branch_parents: dict[int, int],
    branch_lineage: StageBranchLedger | None,
) -> StageBranchLedger:
    if branch_lineage is None:
        # Compatibility for direct v6 helper callers. The live stage owns one
        # persistent ledger, while legacy tests can continue passing the map.
        return StageBranchLedger(logical_owners=branch_parents)
    if branch_lineage.logical_owners is not branch_parents:
        raise ValueError("branch lineage does not own the supplied branch map")
    return branch_lineage


def _discard_failed_physical_fork(
    child_request_id: int,
    runner: StageRunnerContract,
    request_metrics: dict[int, dict[str, Any]],
    branch_lineage: StageBranchLedger,
) -> None:
    request_metrics.pop(child_request_id, None)
    branch_lineage.release(child_request_id)
    runner.end(child_request_id)


def fork_physical_stage_request(
    *,
    child_request_id: int,
    fork_source_physical_request_id: int,
    logical_owner_request_id: int,
    config: StageProcessConfig,
    runner: StageRunnerContract,
    request_metrics: dict[int, dict[str, Any]],
    branch_lineage: StageBranchLedger,
) -> dict[str, Any]:
    """Create one local KV lane while keeping ownership and COW source distinct.

    This is deliberately wire-agnostic. Protocol v6 calls it with source ==
    owner and therefore retains its flat-tree contract. PrefixWave can call the
    same primitive with a speculative carrier as ``fork_source`` while every
    created lane remains logically owned and budgeted by the ordinary root.
    """

    if config.max_speculative_branches == 0:
        raise ValueError("FORK is disabled by the sealed stage contract")
    if child_request_id == fork_source_physical_request_id:
        raise ValueError("FORK child and physical source request IDs must differ")
    if child_request_id == logical_owner_request_id:
        raise ValueError("FORK child and logical owner request IDs must differ")
    if logical_owner_request_id in branch_lineage.logical_owners:
        raise ValueError("FORK logical owner must be an ordinary root request")
    if logical_owner_request_id not in request_metrics:
        raise ValueError(
            f"FORK logical owner request {logical_owner_request_id} is not active"
        )
    if fork_source_physical_request_id not in request_metrics:
        raise ValueError(
            "FORK physical source request "
            f"{fork_source_physical_request_id} is not active on this stage"
        )
    if fork_source_physical_request_id != logical_owner_request_id:
        source_owner = branch_lineage.logical_owners.get(
            fork_source_physical_request_id
        )
        if source_owner != logical_owner_request_id:
            raise ValueError("FORK physical source is outside the logical tree")
    source_metrics = request_metrics[fork_source_physical_request_id]
    source_frames = source_metrics.get("frames")
    if not isinstance(source_frames, int) or isinstance(source_frames, bool):
        raise TypeError("FORK physical source has invalid frame metrics")
    if (
        child_request_id in request_metrics
        or child_request_id in branch_lineage.logical_owners
        or child_request_id in branch_lineage.fork_sources
    ):
        raise ValueError(f"FORK child request {child_request_id} is already active")
    if len(branch_lineage.logical_owners) >= config.max_speculative_branches:
        raise ValueError(
            "FORK exceeds max_speculative_branches: "
            f"{len(branch_lineage.logical_owners) + 1} > "
            f"{config.max_speculative_branches}"
        )

    source_tokens = runner.sequence_length(fork_source_physical_request_id)
    if source_tokens < 1:
        raise ValueError(
            "FORK requires a non-empty prefilled physical source KV cache"
        )
    if source_tokens > config.max_speculative_branch_tokens:
        raise ValueError(
            "FORK physical source cache exceeds max_speculative_branch_tokens: "
            f"{source_tokens} > {config.max_speculative_branch_tokens}"
        )
    request_cache_bytes = getattr(runner, "request_cache_bytes", None)
    fork_request = getattr(runner, "fork_request", None)
    if not callable(request_cache_bytes) or not callable(fork_request):
        raise TypeError("stage runner cannot preflight and fork request KV state")
    current_branch_bytes = speculative_kv_bytes(
        runner, branch_lineage.logical_owners
    )
    source_cache_bytes = request_cache_bytes(fork_source_physical_request_id)
    if not isinstance(source_cache_bytes, int) or isinstance(
        source_cache_bytes, bool
    ):
        raise TypeError("stage runner request_cache_bytes must return an integer")
    remaining_bytes = config.max_speculative_kv_bytes - current_branch_bytes
    if source_cache_bytes < 1:
        raise ValueError("FORK physical source has no measurable KV cache")
    project_tree_physical = getattr(
        runner, "project_tree_incremental_physical_cache_bytes", None
    )
    if callable(project_tree_physical):
        projected_fork_bytes = project_tree_physical(
            fork_source_physical_request_id,
            delta_tokens_by_leaf=(0,),
        )
    else:
        projected_fork_bytes = source_cache_bytes
    if (
        not isinstance(projected_fork_bytes, int)
        or isinstance(projected_fork_bytes, bool)
        or projected_fork_bytes < 0
    ):
        raise TypeError("stage runner returned invalid physical FORK projection")
    if projected_fork_bytes > remaining_bytes:
        raise ValueError(
            "FORK exceeds max_speculative_kv_bytes before cloning: "
            f"{current_branch_bytes + projected_fork_bytes} > "
            f"{config.max_speculative_kv_bytes}"
        )
    available_physical = getattr(runner, "available_physical_cache_bytes", None)
    if callable(available_physical):
        available_bytes = available_physical()
        if (
            not isinstance(available_bytes, int)
            or isinstance(available_bytes, bool)
            or available_bytes < projected_fork_bytes
        ):
            raise ValueError("FORK exceeds available physical KV pool capacity")

    copied_kv_bytes = fork_request(
        child_request_id,
        fork_source_physical_request_id,
        max_cache_bytes=remaining_bytes,
    )
    if not isinstance(copied_kv_bytes, int) or isinstance(copied_kv_bytes, bool):
        runner.end(child_request_id)
        raise TypeError("stage runner fork_request must return copied KV bytes")
    if copied_kv_bytes < 0:
        runner.end(child_request_id)
        raise ValueError("stage runner returned negative copied KV bytes")
    try:
        branch_lineage.register(
            child_request_id,
            logical_owner_request_id=logical_owner_request_id,
            fork_source_physical_request_id=fork_source_physical_request_id,
        )
    except BaseException:
        runner.end(child_request_id)
        raise
    try:
        observed_branch_bytes = speculative_kv_bytes(
            runner, branch_lineage.logical_owners
        )
    except BaseException:
        _discard_failed_physical_fork(
            child_request_id, runner, request_metrics, branch_lineage
        )
        raise
    observed_increment = observed_branch_bytes - current_branch_bytes
    if observed_increment != projected_fork_bytes:
        _discard_failed_physical_fork(
            child_request_id, runner, request_metrics, branch_lineage
        )
        raise RuntimeError(
            "stage runner physical FORK allocation changed after preflight: "
            f"{observed_increment} != {projected_fork_bytes}"
        )
    last_fork_report = getattr(runner, "last_fork_report", None)
    if callable(last_fork_report):
        report = last_fork_report()
        if report is not None:
            report_copied = getattr(report, "copied_bytes", None)
            report_reserved = getattr(report, "newly_reserved_bytes", None)
            if report_copied != copied_kv_bytes:
                _discard_failed_physical_fork(
                    child_request_id, runner, request_metrics, branch_lineage
                )
                raise RuntimeError("stage runner FORK report copied-byte mismatch")
            if report_reserved != observed_increment:
                _discard_failed_physical_fork(
                    child_request_id, runner, request_metrics, branch_lineage
                )
                raise RuntimeError("stage runner FORK report physical-byte mismatch")

    try:
        child_metrics = dict(source_metrics)
        child_metrics.update(
            {
                "compute_ms": 0,
                "compute_dispatch_ms": 0,
                "compute_timing_mode": None,
                "bytes_out": 0,
                "tokens": source_tokens,
                "model_forward_calls": 0,
                "physical_batch_calls": 0,
                "physical_batch_items": 0,
                "max_physical_batch_size": 0,
                # Legacy metric retained as logical ownership, not COW ancestry.
                "branch_parent_request_id": logical_owner_request_id,
                "branch_logical_owner_request_id": logical_owner_request_id,
                "branch_fork_source_request_id": fork_source_physical_request_id,
                "branch_inherited_frames": source_frames,
                "branch_inherited_tokens": source_tokens,
                "fork_copied_kv_bytes": copied_kv_bytes,
                "fork_new_physical_kv_bytes": observed_increment,
            }
        )
        request_metrics[child_request_id] = child_metrics
    except BaseException:
        _discard_failed_physical_fork(
            child_request_id, runner, request_metrics, branch_lineage
        )
        raise
    return child_metrics


def fork_stage_request(
    frame: Frame,
    config: StageProcessConfig,
    runner: StageRunnerContract,
    request_metrics: dict[int, dict[str, Any]],
    branch_parents: dict[int, int],
    downstream: socket.socket | None,
    tree_reservations: TreeReservationBook | None = None,
    tree_leaf_shapes: dict[int, TreeLeafShape] | None = None,
    branch_lineage: StageBranchLedger | None = None,
) -> None:
    """Apply one ordered protocol-v6 flat FORK(child,parent)."""

    child_request_id = frame.request_id
    parent_request_id = decode_branch_request_id(frame)
    reservation = None if tree_reservations is None else tree_reservations.active
    quoted_path_length: int | None = None
    if tree_reservations is not None and reservation is None:
        raise ValueError("FORK requires an active committed tree reservation")
    if reservation is not None:
        if not reservation.committed:
            raise ValueError("FORK cannot consume an uncommitted tree reservation")
        if tree_leaf_shapes is None:
            raise ValueError("committed tree FORK requires a leaf-shape ledger")
        if parent_request_id != reservation.parent_request_id:
            raise ValueError("FORK parent does not match committed tree reservation")
        if reservation.consumed_forks >= len(reservation.path_lengths):
            raise ValueError("FORK exceeds committed tree reservation count")
        quoted_path_length = reservation.path_lengths[reservation.consumed_forks]

    lineage = _resolve_branch_lineage(branch_parents, branch_lineage)
    child_metrics = fork_physical_stage_request(
        child_request_id=child_request_id,
        fork_source_physical_request_id=parent_request_id,
        logical_owner_request_id=parent_request_id,
        config=config,
        runner=runner,
        request_metrics=request_metrics,
        branch_lineage=lineage,
    )
    if downstream is not None:
        child_metrics["bytes_out"] += send_frame(
            downstream,
            FrameType.FORK,
            child_request_id,
            payload=frame.payload,
        )
    if reservation is not None:
        if quoted_path_length is None:
            raise RuntimeError("committed tree reservation has no leaf-shape ledger")
        tree_leaf_shapes[child_request_id] = TreeLeafShape(
            parent_request_id=parent_request_id,
            nonce=reservation.nonce,
            expected_tokens=1 + quoted_path_length,
        )
        reservation.consumed_forks += 1
        reservation.quoted_children[child_request_id] = 1 + quoted_path_length
        reservation.deadline_at = time.monotonic() + TREE_RESERVATION_TTL_SECONDS
        tree_reservations.last_mutated = True


def consume_tree_reservation_verifies(
    frames: tuple[Frame, ...],
    reservations: TreeReservationBook,
) -> bool:
    """Commit successful quoted VERIFY growth and release the mutation barrier.

    This must run only after every frame in ``frames`` has completed locally and
    its output has been forwarded.  A failed model call or socket write therefore
    leaves the reservation active and makes the route fail closed.
    """

    active = reservations.active
    if active is None:
        return False
    quoted = tuple(
        frame for frame in frames if frame.request_id in active.quoted_children
    )
    if not quoted:
        return False
    if active.consumed_forks != len(active.path_lengths):
        raise RuntimeError("quoted VERIFY completed before every FORK")
    for frame in quoted:
        if frame.frame_type != FrameType.VERIFY:
            raise RuntimeError("non-VERIFY frame consumed quoted tree capacity")
        expected_tokens = active.quoted_children[frame.request_id]
        if frame.token_count != expected_tokens:
            raise RuntimeError("quoted VERIFY changed shape after validation")
        if frame.request_id in active.consumed_verifies:
            raise RuntimeError("quoted VERIFY was consumed twice")
        active.consumed_verifies.add(frame.request_id)
    active.deadline_at = time.monotonic() + TREE_RESERVATION_TTL_SECONDS
    if len(active.consumed_verifies) > len(active.path_lengths):
        raise RuntimeError("tree reservation consumed too many VERIFY frames")
    if len(active.consumed_verifies) == len(active.path_lengths):
        reservations.active = None
        return True
    return False


def end_physical_stage_request(
    request_id: int,
    *,
    operation: str,
    runner: StageRunnerContract,
    request_metrics: dict[int, dict[str, Any]],
    branch_lineage: StageBranchLedger,
) -> dict[str, Any] | None:
    """End one lane without treating historical physical ancestry as ownership."""

    if request_id in branch_lineage.logical_owners.values():
        raise ValueError(
            f"cannot {operation} request {request_id} while it still logically "
            "owns speculative branches"
        )
    runner.end(request_id)
    metrics = request_metrics.pop(request_id, None)
    if metrics is not None:
        metrics["execution"] = execution_metric_snapshot(runner)
    branch_lineage.release(request_id)
    return metrics


def promote_physical_stage_descendant(
    *,
    logical_owner_request_id: int,
    descendant_request_id: int,
    config: StageProcessConfig,
    runner: StageRunnerContract,
    request_metrics: dict[int, dict[str, Any]],
    branch_lineage: StageBranchLedger,
) -> dict[str, Any]:
    """Move the sole surviving descendant into its ordinary root in O(1)."""

    if config.max_speculative_branches == 0:
        raise ValueError("PROMOTE is disabled by the sealed stage contract")
    if descendant_request_id == logical_owner_request_id:
        raise ValueError("PROMOTE owner and descendant request IDs must differ")
    if logical_owner_request_id in branch_lineage.logical_owners:
        raise ValueError("PROMOTE logical owner must be an ordinary root request")
    if logical_owner_request_id not in request_metrics:
        raise ValueError(
            f"PROMOTE owner request {logical_owner_request_id} is not active"
        )
    if descendant_request_id not in request_metrics:
        raise ValueError(
            f"PROMOTE descendant request {descendant_request_id} is not active"
        )
    if (
        branch_lineage.logical_owners.get(descendant_request_id)
        != logical_owner_request_id
    ):
        raise ValueError("PROMOTE descendant is outside the logical tree")
    live_owned = branch_lineage.owned_requests(logical_owner_request_id)
    if live_owned != {descendant_request_id}:
        raise ValueError(
            "PROMOTE requires the selected descendant to be the sole live branch"
        )

    promote_request = getattr(runner, "promote_request", None)
    if not callable(promote_request):
        raise TypeError("stage runner cannot promote request KV state")
    promote_request(logical_owner_request_id, descendant_request_id)
    selected_metrics = request_metrics.pop(descendant_request_id)
    branch_lineage.release(descendant_request_id)
    selected_metrics["branch_promotions"] = int(
        selected_metrics.get("branch_promotions", 0)
    ) + 1
    selected_metrics.pop("branch_parent_request_id", None)
    selected_metrics.pop("branch_logical_owner_request_id", None)
    selected_metrics.pop("branch_fork_source_request_id", None)
    request_metrics[logical_owner_request_id] = selected_metrics
    return selected_metrics


def promote_stage_request(
    frame: Frame,
    config: StageProcessConfig,
    runner: StageRunnerContract,
    request_metrics: dict[int, dict[str, Any]],
    branch_parents: dict[int, int],
    downstream: socket.socket | None,
    tree_leaf_shapes: dict[int, TreeLeafShape] | None = None,
    branch_lineage: StageBranchLedger | None = None,
) -> None:
    """Apply one ordered protocol-v6 PROMOTE(parent,child)."""

    parent_request_id = frame.request_id
    child_request_id = decode_branch_request_id(frame)
    if tree_leaf_shapes is not None and child_request_id not in tree_leaf_shapes:
        raise ValueError("PROMOTE child is not bound to a committed tree quote")
    lineage = _resolve_branch_lineage(branch_parents, branch_lineage)
    selected_metrics = promote_physical_stage_descendant(
        logical_owner_request_id=parent_request_id,
        descendant_request_id=child_request_id,
        config=config,
        runner=runner,
        request_metrics=request_metrics,
        branch_lineage=lineage,
    )
    if tree_leaf_shapes is not None:
        tree_leaf_shapes.pop(child_request_id, None)
    if downstream is not None:
        selected_metrics["bytes_out"] += send_frame(
            downstream,
            FrameType.PROMOTE,
            parent_request_id,
            payload=frame.payload,
        )


def speculative_kv_bytes(
    runner: StageRunnerContract,
    branch_parents: Mapping[int, int],
) -> int:
    """Measure physical KV beyond the ordinary logical-owner roots.

    Mapping values must be logical roots, not immediate fork sources. Thus a
    nested tree is deduplicated as ``union(root + live lanes) - union(root)``;
    ending an intermediate carrier cannot make its descendants disappear from
    the budget.
    """

    if not branch_parents:
        return 0
    unique_physical_bytes = getattr(runner, "unique_physical_cache_bytes", None)
    if callable(unique_physical_bytes):
        parent_ids = tuple(sorted(set(branch_parents.values())))
        all_ids = tuple(sorted(set(branch_parents) | set(parent_ids)))
        all_bytes = unique_physical_bytes(all_ids)
        parent_bytes = unique_physical_bytes(parent_ids)
        for name, value in (("all", all_bytes), ("parent", parent_bytes)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise TypeError(
                    f"stage runner returned invalid {name} unique physical KV bytes"
                )
        if all_bytes < parent_bytes:
            raise RuntimeError("speculative physical KV accounting moved backwards")
        return all_bytes - parent_bytes

    request_cache_bytes = getattr(runner, "request_cache_bytes", None)
    if not callable(request_cache_bytes):
        raise TypeError("stage runner cannot measure speculative KV bytes")
    total = 0
    for request_id in branch_parents:
        value = request_cache_bytes(request_id)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise TypeError("stage runner returned invalid request cache bytes")
        total += value
    return total


def validate_speculative_kv_preflight(
    frames: tuple[Frame, ...],
    config: StageProcessConfig,
    runner: StageRunnerContract,
    branch_parents: Mapping[int, int],
) -> None:
    """Seal the combined KV growth of one sequential or physical batch."""

    branch_frames = tuple(
        frame for frame in frames if frame.request_id in branch_parents
    )
    if not branch_frames:
        return
    if len({frame.request_id for frame in branch_frames}) != len(branch_frames):
        raise ValueError("one speculative request cannot appear twice in a physical batch")
    request_cache_bytes = getattr(runner, "request_cache_bytes", None)
    project_cache_bytes = getattr(runner, "project_request_cache_bytes", None)
    project_growth_physical = getattr(
        runner, "project_request_incremental_physical_cache_bytes", None
    )
    if not callable(request_cache_bytes) or not callable(project_cache_bytes):
        raise TypeError("stage runner cannot preflight speculative KV growth")

    projected_total_bytes = speculative_kv_bytes(runner, branch_parents)
    total_incremental_bytes = 0
    for frame in branch_frames:
        if callable(project_growth_physical):
            incremental = project_growth_physical(
                frame.request_id,
                frame.token_count,
            )
            if (
                not isinstance(incremental, int)
                or isinstance(incremental, bool)
                or incremental < 0
            ):
                raise TypeError(
                    "stage runner returned invalid incremental physical KV bytes"
                )
        else:
            current = request_cache_bytes(frame.request_id)
            projected = project_cache_bytes(frame.request_id, frame.token_count)
            for name, value in (("current", current), ("projected", projected)):
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    raise TypeError(f"stage runner returned invalid {name} KV bytes")
            if projected < current:
                raise ValueError("stage runner projected speculative KV bytes backwards")
            incremental = projected - current
        projected_total_bytes += incremental
        total_incremental_bytes += incremental
    available_physical = getattr(runner, "available_physical_cache_bytes", None)
    if callable(available_physical):
        available_bytes = available_physical()
        if (
            not isinstance(available_bytes, int)
            or isinstance(available_bytes, bool)
            or available_bytes < total_incremental_bytes
        ):
            raise ValueError(
                "speculative KV growth exceeds available physical cache pool"
            )
    if projected_total_bytes > config.max_speculative_kv_bytes:
        raise ValueError(
            "speculative KV bytes exceed max_speculative_kv_bytes before "
            "child forward: "
            f"{projected_total_bytes} > {config.max_speculative_kv_bytes}"
        )


def tree_verify_batch_is_certified(runner: StageRunnerContract) -> bool:
    """Require both numerical and peak-memory evidence for tree batching.

    Equal tensor shapes and cache lengths do not prove that batch=K chooses the
    same greedy tokens as K independent forwards on every device/dtype/kernel.
    Some cache implementations also retain the original rows while building
    and splitting a temporary batch, so persistent KV accounting alone can
    understate the real peak by several times.  Executors must explicitly seal
    both properties before VERIFY leaves may share one physical forward.
    """

    manifest = getattr(runner, "executor_manifest", None)
    features = frozenset(getattr(manifest, "features", ()))
    return {
        "exact-tree-verify-batching",
        "bounded-tree-verify-workspace",
    }.issubset(features)


def collect_compatible_activation_frames(
    first: Frame,
    *,
    upstream: socket.socket,
    pending_frames: deque[Frame],
    config: StageProcessConfig,
    runner: StageRunnerContract,
    request_metrics: dict[int, dict[str, Any]],
    branch_parents: Mapping[int, int] | None = None,
    tree_leaf_shapes: Mapping[int, TreeLeafShape] | None = None,
    tree_reservations: TreeReservationBook | None = None,
    downstream: socket.socket | None,
) -> tuple[Frame, ...]:
    """Read a bounded run of activations that can share one model forward.

    BEGIN is request-local and may be applied while the batch window is open.
    Every other control frame is left in order for the main loop.  Cell runners
    expose no physical batch operation and therefore never enter this collector.
    """

    batch_forward = getattr(runner, "forward_hidden_batch", None)
    # With ragged grouping on, requests are keyed WITHOUT their cache length, so two
    # sequences a token out of phase still fuse. Without it the exact-length key is
    # used, which is what keeps the default byte-identical to sequential decode.
    batch_key = getattr(
        runner,
        "physical_batch_group_key" if ragged_grouping_enabled() else "physical_batch_key",
        None,
    )
    first_is_tree_leaf = (
        branch_parents is not None and first.request_id in branch_parents
    )
    if (
        getattr(runner, "max_active_requests", None) == 1
        or config.max_physical_batch_size < 2
        or not callable(batch_forward)
        or not callable(batch_key)
        or (
            first_is_tree_leaf
            and not tree_verify_batch_is_certified(runner)
        )
    ):
        return (first,)
    token_mode = activation_token_mode(first.frame_type)
    first_key = batch_key(
        first.request_id,
        token_count=first.token_count,
        token_mode=token_mode,
    )
    if first_key is None:
        return (first,)

    collected = [first]
    request_ids = {first.request_id}
    deadline = time.monotonic() + config.physical_batch_window_ms / 1_000
    while len(collected) < config.max_physical_batch_size:
        remaining = max(0.0, deadline - time.monotonic())
        readable, _, _ = select.select([upstream], [], [], remaining)
        if not readable:
            break
        candidate = recv_frame(upstream)
        # The batch collector reads ahead of the main stage loop.  It must
        # enforce the same reservation barrier before applying an inline BEGIN
        # or accepting a compatible tensor; otherwise a parent activation could
        # mutate the snapshot while TREE_PREPARE is awaiting its route result.
        if tree_reservations is not None:
            if tree_reservation_defers_frame(
                candidate,
                tree_reservations,
                {} if branch_parents is None else branch_parents,
            ):
                pending_frames.appendleft(candidate)
                break
            validate_tree_reservation_frame_order(
                candidate,
                tree_reservations,
                {} if branch_parents is None else branch_parents,
            )
        if candidate.frame_type == FrameType.BEGIN:
            begin_stage_request(
                candidate,
                config,
                runner,
                request_metrics,
                downstream,
            )
            continue
        if candidate.frame_type not in (
            FrameType.ACTIVATION,
            FrameType.PREFILL,
            FrameType.VERIFY,
        ):
            pending_frames.appendleft(candidate)
            break
        # Consecutive credit-window chunks from one request are deliberately
        # not one physical tensor batch: chunk N+1 depends on chunk N's KV.
        # Defer it before step validation, because the first frame's forward
        # has not yet incremented request_metrics["frames"]. The main loop will
        # validate it immediately after the predecessor commits.
        if candidate.request_id in request_ids:
            pending_frames.appendleft(candidate)
            break
        candidate_is_tree_leaf = (
            branch_parents is not None
            and candidate.request_id in branch_parents
        )
        # Never merge a real request with a virtual tree leaf.  Apart from
        # making peak-memory accounting harder to certify, their lifecycle and
        # cancellation semantics are deliberately different.
        if candidate_is_tree_leaf != first_is_tree_leaf:
            pending_frames.appendleft(candidate)
            break
        if candidate_is_tree_leaf and not tree_verify_batch_is_certified(runner):
            pending_frames.appendleft(candidate)
            break
        validate_activation(
            candidate,
            config,
            runner,
            request_metrics,
            branch_parents,
            tree_leaf_shapes,
        )
        candidate_mode = activation_token_mode(candidate.frame_type)
        candidate_key = batch_key(
            candidate.request_id,
            token_count=candidate.token_count,
            token_mode=candidate_mode,
        )
        compatible = (
            candidate.frame_type == first.frame_type
            and candidate.flags == first.flags
            and candidate.token_count == first.token_count
            and candidate.hidden_size == first.hidden_size
            and candidate_key == first_key
        )
        if not compatible:
            pending_frames.appendleft(candidate)
            break
        collected.append(candidate)
        request_ids.add(candidate.request_id)
    return tuple(collected)


def process_activation_frames(
    frames: tuple[Frame, ...],
    *,
    config: StageProcessConfig,
    runner: StageRunnerContract,
    request_metrics: dict[int, dict[str, Any]],
    branch_parents: Mapping[int, int],
    downstream: socket.socket | None,
    return_socket: socket.socket | None,
    emulator: LinkEmulator,
) -> None:
    """Execute and route one sequential or genuinely batched model operation."""

    if not frames:
        raise ValueError("activation frame batch cannot be empty")
    if len(frames) > 1:
        tree_membership = tuple(
            frame.request_id in branch_parents for frame in frames
        )
        if any(tree_membership) and not all(tree_membership):
            raise ValueError("physical batches cannot mix real requests and tree leaves")
        if any(tree_membership) and not tree_verify_batch_is_certified(runner):
            raise ValueError(
                "physical tree VERIFY batching requires exact-token and "
                "bounded-workspace executor certification"
            )
    # Individual ingress validation is insufficient for a physical batch: two
    # children may each fit alone but exceed the total KV budget together.
    validate_speculative_kv_preflight(frames, config, runner, branch_parents)
    token_mode = activation_token_mode(frames[0].frame_type)
    hidden_states = tuple(decode_tensor(frame) for frame in frames)
    timings_ms: tuple[float, ...]
    # El régimen de medida viaja con el dato. Sin `GDLP_COMPUTE_TIMING=sync`
    # esto cronometra el DESPACHO del forward, no su ejecución en el
    # dispositivo, y por eso se publica bajo `compute_dispatch_ms`. Ver
    # `compute_timing.py`: publicar un tiempo sin sincronizar bajo el nombre
    # `compute_ms` es lo que convertiría el término C en otro número fantasma.
    timer = ComputeTimer()
    if len(frames) > 1:
        batch_forward = getattr(runner, "forward_hidden_batch", None)
        if not callable(batch_forward):
            raise TypeError("runner advertised a physical batch key without a batch forward")
        with timer.measure() as measurement:
            results = tuple(
                batch_forward(
                    tuple(frame.request_id for frame in frames),
                    hidden_states,
                    token_mode=token_mode,
                )
            )
        if len(results) != len(frames):
            raise RuntimeError("physical batch returned the wrong number of results")
        timings_ms = (measurement.elapsed_ms,) * len(frames)
    else:
        with timer.measure() as measurement:
            results = (
                runner.forward_hidden(
                    frames[0].request_id,
                    hidden_states[0],
                    token_mode=token_mode,
                ),
            )
        timings_ms = (measurement.elapsed_ms,)
    compute_metric_key = measurement.metric_key()

    physical_size = len(frames)
    for frame, result, compute_ms in zip(frames, results, timings_ms, strict=True):
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError("stage runner result must be an (output, token) pair")
        output, token = result
        if not isinstance(output, torch.Tensor):
            raise TypeError("stage runner output must be a tensor")
        metrics = request_metrics[frame.request_id]
        metrics[compute_metric_key] = metrics.get(compute_metric_key, 0) + compute_ms
        metrics["compute_timing_mode"] = timer.mode
        metrics["frames"] += 1
        metrics["tokens"] += frame.token_count
        metrics["model_forward_calls"] += 1
        metrics["physical_batch_items"] += physical_size
        metrics["max_physical_batch_size"] = max(
            int(metrics["max_physical_batch_size"]), physical_size
        )
        if physical_size > 1:
            metrics["physical_batch_calls"] += 1
        route_stage_result(
            frame,
            output,
            token,
            config=config,
            metrics=metrics,
            downstream=downstream,
            return_socket=return_socket,
            emulator=emulator,
        )


def route_stage_result(
    frame: Frame,
    output: torch.Tensor,
    token: int | tuple[int, ...] | None,
    *,
    config: StageProcessConfig,
    metrics: dict[str, Any],
    downstream: socket.socket | None,
    return_socket: socket.socket | None,
    emulator: LinkEmulator,
) -> None:
    if config.spec.last:
        if return_socket is None:
            raise RuntimeError("last stage has no token return connection")
        if frame.frame_type == FrameType.PREFILL:
            metrics["bytes_out"] += send_frame(
                return_socket,
                FrameType.PREFILL_ACK,
                frame.request_id,
                step=frame.step,
                emulator=emulator,
            )
        elif frame.frame_type == FrameType.VERIFY:
            if not isinstance(token, tuple) or len(token) != frame.token_count:
                raise RuntimeError("last stage returned an invalid verification vector")
            payload = verify_result_payload(token)
            metrics["bytes_out"] += send_frame(
                return_socket,
                FrameType.VERIFY_RESULT,
                frame.request_id,
                step=frame.step,
                token_count=len(token),
                payload=payload,
                emulator=emulator,
            )
        else:
            if not isinstance(token, int):
                raise RuntimeError("last stage did not return a token")
            metrics["bytes_out"] += send_frame(
                return_socket,
                FrameType.TOKEN,
                frame.request_id,
                step=frame.step,
                payload=token_payload(token),
                emulator=emulator,
            )
        return

    if downstream is None:
        raise RuntimeError("intermediate stage has no downstream connection")
    encoded = encode_tensor_payload(output, config.codec)
    metrics["bytes_out"] += send_frame(
        downstream,
        frame.frame_type,
        frame.request_id,
        step=frame.step,
        token_count=output.shape[1],
        hidden_size=output.shape[2],
        flags=int(config.codec),
        payload=encoded.view,
        emulator=emulator,
    )


def activation_token_mode(frame_type: FrameType) -> str:
    if frame_type == FrameType.PREFILL:
        return "none"
    if frame_type == FrameType.VERIFY:
        return "all"
    if frame_type == FrameType.ACTIVATION:
        return "last"
    raise ValueError(f"{frame_type.name} is not an activation frame")


def build_stage_runner(config: StageProcessConfig) -> StageRunnerContract:
    """Construct the single-member runner or a logical local TP cell."""

    if config.native_gguf is not None:
        from .native_gguf_runtime import build_native_gguf_stage_runner

        if config.dense_tiering == DenseTieringConfig():
            return build_native_gguf_stage_runner(
                config.spec,
                config.native_gguf,
                device=config.device,
            )
        return build_native_gguf_stage_runner(
            config.spec,
            config.native_gguf,
            device=config.device,
            dense_tiering=config.dense_tiering,
        )
    if config.paged_kv is not None:
        from .paged_stage import HFPagedStageRunner

        return HFPagedStageRunner.from_runtime_config(config.spec, config.paged_kv)
    if config.ram_backed_moe is not None:
        return build_ram_backed_moe_stage_runner(
            config.spec,
            config.ram_backed_moe,
            pipeline_snapshot_identity=config.pipeline_id,
        )
    if config.cell_fixture is None:
        if normalize_torch_device_request(config.device) == "auto":
            return StageRunner(config.spec)
        if config.dense_tiering == DenseTieringConfig():
            return StageRunner(config.spec, device=config.device)
        return StageRunner(
            config.spec,
            device=config.device,
            dense_tiering=config.dense_tiering,
        )
    if config.cell_manifest_sha256 is not None:
        actual = _sha256_file(Path(config.cell_fixture) / "cell.json")
        if actual != config.cell_manifest_sha256:
            raise ValueError("cell manifest SHA-256 does not match the launch contract")
    if config.cell_mode == "external":
        from .external_cell import (
            ExternalTensorParallelCellSpec,
            ExternalTensorParallelCellStageRunner,
        )

        required = (
            config.cell_world_size,
            config.cell_control_host,
            config.cell_control_port,
            config.cell_control_advertise_host,
            config.cell_distributed_advertise_host,
            config.cell_distributed_port,
        )
        if any(value is None for value in required):
            raise ValueError("external cell configuration is incomplete")
        return ExternalTensorParallelCellStageRunner(
            config.spec,
            ExternalTensorParallelCellSpec(
                fixture=config.cell_fixture,
                world_size=config.cell_world_size,
                pipeline_id=config.pipeline_id,
                control_host=config.cell_control_host,
                control_port=config.cell_control_port,
                control_advertise_host=config.cell_control_advertise_host,
                distributed_advertise_host=config.cell_distributed_advertise_host,
                distributed_port=config.cell_distributed_port,
                collective_backend=config.cell_collective_backend,
                rank_devices=(
                    config.cell_devices
                    or tuple("cpu" for _ in range(config.cell_world_size))
                ),
                compute_dtype=config.cell_compute_dtype,
                startup_timeout_seconds=config.cell_startup_timeout_seconds,
                operation_timeout_seconds=config.cell_operation_timeout_seconds,
            ),
        )
    from .cell_stage import TensorParallelCellSpec, TensorParallelCellStageRunner

    if config.cell_world_size is None:
        raise ValueError("cell_world_size is required with cell_fixture")
    return TensorParallelCellStageRunner(
        config.spec,
        TensorParallelCellSpec(
            fixture=config.cell_fixture,
            world_size=config.cell_world_size,
            backend=config.cell_collective_backend,
            rank_devices=(
                config.cell_devices
                or tuple("cpu" for _ in range(config.cell_world_size))
            ),
            compute_dtype=config.cell_compute_dtype,
            operation_timeout_seconds=config.cell_operation_timeout_seconds,
        ),
    )


def validate_hello(frame: Frame, config: StageProcessConfig, hidden_size: int) -> None:
    if frame.frame_type != FrameType.HELLO:
        raise ValueError("first frame must be HELLO")
    if frame.request_id != config.pipeline_id:
        raise ValueError(
            f"pipeline identity mismatch: got {frame.request_id}, "
            f"expected {config.pipeline_id}"
        )
    if frame.step != config.spec.layer_start or frame.token_count != config.spec.layer_end:
        raise ValueError(
            f"layer range mismatch: got [{frame.step},{frame.token_count}), "
            f"expected [{config.spec.layer_start},{config.spec.layer_end})"
        )
    if frame.hidden_size != hidden_size:
        raise ValueError(f"hidden size mismatch: got {frame.hidden_size}, expected {hidden_size}")
    if frame.flags != int(config.codec):
        raise ValueError("activation codec mismatch")


def validate_stage_config(config: StageProcessConfig) -> None:
    normalized_device = normalize_torch_device_request(config.device)
    if not isinstance(config.dense_tiering, DenseTieringConfig):
        raise TypeError("dense_tiering must be DenseTieringConfig")
    if (
        not isinstance(config.pipeline_id, int)
        or isinstance(config.pipeline_id, bool)
        or not 0 <= config.pipeline_id <= (1 << 64) - 1
    ):
        raise ValueError("pipeline_id must be an unsigned 64-bit integer")
    if not config.listen_host.strip() or not config.return_host.strip():
        raise ValueError("listen_host and return_host cannot be empty")
    for name, port in (("listen_port", config.listen_port), ("return_port", config.return_port)):
        if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65_535:
            raise ValueError(f"{name} must be between 1 and 65535")
    has_next_host = config.next_host is not None
    has_next_port = config.next_port is not None
    if has_next_host != has_next_port:
        raise ValueError("next_host and next_port must either both be set or both be absent")
    if config.spec.last:
        if has_next_host or config.next_layer_end is not None:
            raise ValueError("the final stage cannot declare a downstream stage")
    else:
        if not has_next_host or config.next_layer_end is None:
            raise ValueError("a non-final stage must declare its downstream stage")
        if not config.next_host or not config.next_host.strip():
            raise ValueError("next_host cannot be empty")
        if (
            not isinstance(config.next_port, int)
            or isinstance(config.next_port, bool)
            or not 1 <= config.next_port <= 65_535
        ):
            raise ValueError("next_port must be between 1 and 65535")
        if not config.spec.layer_end < config.next_layer_end <= config.spec.total_layers:
            raise ValueError("next_layer_end must extend the contiguous layer range")
    try:
        TensorCodec(config.codec)
    except (TypeError, ValueError) as error:
        raise ValueError(f"unsupported activation codec {config.codec}") from error
    for name, value in (
        ("one_way_delay_ms", config.one_way_delay_ms),
        ("bandwidth_mbps", config.bandwidth_mbps),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if not math.isfinite(config.connect_timeout_seconds) or config.connect_timeout_seconds <= 0:
        raise ValueError("connect_timeout_seconds must be finite and positive")
    if config.sealed_wave_tokens is not None and (
        not isinstance(config.sealed_wave_tokens, int)
        or isinstance(config.sealed_wave_tokens, bool)
        or not 1 <= config.sealed_wave_tokens <= 17
    ):
        raise ValueError("sealed_wave_tokens must be between 1 and 17")
    if config.max_prefill_chunk_tokens is not None and (
        not isinstance(config.max_prefill_chunk_tokens, int)
        or isinstance(config.max_prefill_chunk_tokens, bool)
        or config.max_prefill_chunk_tokens < 1
    ):
        raise ValueError("max_prefill_chunk_tokens must be positive")
    if (
        not isinstance(config.max_physical_batch_size, int)
        or isinstance(config.max_physical_batch_size, bool)
        or not 1 <= config.max_physical_batch_size <= MAX_PHYSICAL_STAGE_BATCH_SIZE
    ):
        raise ValueError(
            "max_physical_batch_size must be between 1 and "
            f"{MAX_PHYSICAL_STAGE_BATCH_SIZE}"
        )
    if (
        not math.isfinite(config.physical_batch_window_ms)
        or config.physical_batch_window_ms < 0
        or config.physical_batch_window_ms > 100
    ):
        raise ValueError("physical_batch_window_ms must be between 0 and 100")
    if (
        not isinstance(config.max_speculative_branches, int)
        or isinstance(config.max_speculative_branches, bool)
        or not 0 <= config.max_speculative_branches <= MAX_SPECULATIVE_BRANCHES
    ):
        raise ValueError(
            "max_speculative_branches must be between 0 and "
            f"{MAX_SPECULATIVE_BRANCHES}"
        )
    if (
        not isinstance(config.max_speculative_branch_tokens, int)
        or isinstance(config.max_speculative_branch_tokens, bool)
        or not 0
        <= config.max_speculative_branch_tokens
        <= MAX_SPECULATIVE_BRANCH_TOKENS
    ):
        raise ValueError(
            "max_speculative_branch_tokens must be between 0 and "
            f"{MAX_SPECULATIVE_BRANCH_TOKENS}"
        )
    if (
        not isinstance(config.max_speculative_kv_bytes, int)
        or isinstance(config.max_speculative_kv_bytes, bool)
        or not 0 <= config.max_speculative_kv_bytes <= MAX_SPECULATIVE_KV_BYTES
    ):
        raise ValueError(
            "max_speculative_kv_bytes must be between 0 and "
            f"{MAX_SPECULATIVE_KV_BYTES}"
        )
    enabled_limits = (
        config.max_speculative_branches > 0,
        config.max_speculative_branch_tokens > 0,
        config.max_speculative_kv_bytes > 0,
    )
    if any(enabled_limits) and not all(enabled_limits):
        raise ValueError(
            "speculative branch count, tokens and KV bytes must all be zero "
            "or all be positive"
        )
    has_ram_backed_moe = config.ram_backed_moe is not None
    has_paged_kv = config.paged_kv is not None
    has_native_gguf = config.native_gguf is not None
    if (
        config.dense_tiering != DenseTieringConfig()
        and (
            has_ram_backed_moe
            or has_paged_kv
            or config.cell_fixture is not None
        )
    ):
        raise ValueError(
            "dense tiering budgets apply only to dense SafeTensors or native "
            "GGUF stages"
        )
    if normalized_device != "auto" and (
        has_ram_backed_moe
        or has_paged_kv
        or config.cell_fixture is not None
    ):
        raise ValueError(
            "device applies only to the dense Torch stage backend; specialised "
            "backends have their own sealed device settings"
        )
    if has_ram_backed_moe:
        if (
            has_paged_kv
            or has_native_gguf
            or config.cell_fixture is not None
        ):
            raise ValueError(
                "native GGUF, paged KV, RAM-backed MoE and "
                "tensor-parallel cell backends are mutually exclusive"
            )
        validate_ram_backed_moe_binding(
            config.ram_backed_moe,
            model_name=config.spec.model_name,
            revision=config.spec.revision,
            pipeline_snapshot_identity=config.pipeline_id,
        )
        if config.spec.artifact_identity != config.ram_backed_moe.artifact_identity:
            raise ValueError(
                "stage spec and RAM-backed MoE artifact identities do not match"
            )
    if has_paged_kv:
        from .paged_stage import HFPagedStageRuntimeConfig

        if not isinstance(config.paged_kv, HFPagedStageRuntimeConfig):
            raise TypeError("paged_kv must be HFPagedStageRuntimeConfig")
        if has_native_gguf or config.cell_fixture is not None:
            raise ValueError(
                "native GGUF, paged KV, RAM-backed MoE and "
                "tensor-parallel cell backends are mutually exclusive"
            )
        required_request_slots = 1 + config.max_speculative_branches
        if config.paged_kv.max_active_requests < required_request_slots:
            raise ValueError(
                "paged max_active_requests cannot hold the root request and all "
                "sealed speculative branches"
            )
        if (
            config.max_speculative_branch_tokens > 0
            and config.paged_kv.max_sequence_tokens
            < config.max_speculative_branch_tokens
        ):
            raise ValueError(
                "paged max_sequence_tokens is smaller than the sealed "
                "speculative branch token ceiling"
            )
    if has_native_gguf:
        if not isinstance(config.native_gguf, NativeGgufRuntimeConfig):
            raise TypeError("native_gguf must be NativeGgufRuntimeConfig")
        if config.cell_fixture is not None:
            raise ValueError(
                "native GGUF and tensor-parallel cell backends are "
                "mutually exclusive"
            )
        expected_stage_identity = f"sha256:{config.native_gguf.package_id}"
        if config.spec.stage_package_identity != expected_stage_identity:
            raise ValueError(
                "native GGUF stage package identity does not match the launch spec"
            )
    has_cell_fixture = config.cell_fixture is not None
    has_cell_size = config.cell_world_size is not None
    if config.cell_mode not in ("local", "external"):
        raise ValueError("cell_mode must be local or external")
    if has_cell_fixture != has_cell_size:
        raise ValueError("cell_fixture and cell_world_size must be supplied together")
    if not has_cell_fixture and config.cell_manifest_sha256 is not None:
        raise ValueError("cell_manifest_sha256 requires cell_fixture")
    if has_cell_fixture:
        if not config.cell_fixture or not config.cell_fixture.strip():
            raise ValueError("cell_fixture cannot be empty")
        if (
            not isinstance(config.cell_world_size, int)
            or isinstance(config.cell_world_size, bool)
            or config.cell_world_size < 2
        ):
            raise ValueError("cell_world_size must be an integer of at least two")
        if config.spec.first or config.spec.last:
            raise ValueError("the cell prototype currently supports intermediate stages only")
        if config.cell_manifest_sha256 is not None and (
            not isinstance(config.cell_manifest_sha256, str)
            or len(config.cell_manifest_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in config.cell_manifest_sha256.lower()
            )
        ):
            raise ValueError("cell_manifest_sha256 must be a SHA-256 hex digest")
        devices = config.cell_devices or tuple(
            "cpu" for _ in range(config.cell_world_size)
        )
        if (
            not isinstance(devices, tuple)
            or len(devices) != config.cell_world_size
            or any(not isinstance(device, str) or not device for device in devices)
        ):
            raise ValueError("cell_devices must declare one device per rank")
        from .cell_backend import CellExecutorBackend

        for device in devices:
            CellExecutorBackend(
                collective_backend=config.cell_collective_backend,
                device=device,
                compute_dtype=config.cell_compute_dtype,
            )
    elif (
        config.cell_devices
        or config.cell_collective_backend != "gloo"
        or config.cell_compute_dtype != "float32"
    ):
        raise ValueError("cell backend settings require cell_fixture")
    external_values = (
        config.cell_control_host,
        config.cell_control_port,
        config.cell_control_advertise_host,
        config.cell_distributed_advertise_host,
        config.cell_distributed_port,
    )
    if config.cell_mode == "external":
        if not has_cell_fixture or any(value is None for value in external_values):
            raise ValueError(
                "external cells require fixture, world size, control and distributed endpoints"
            )
        for name, host in (
            ("cell_control_host", config.cell_control_host),
            ("cell_control_advertise_host", config.cell_control_advertise_host),
            ("cell_distributed_advertise_host", config.cell_distributed_advertise_host),
        ):
            if not isinstance(host, str) or not host.strip():
                raise ValueError(f"{name} cannot be empty")
        for name, host in (
            ("cell_control_advertise_host", config.cell_control_advertise_host),
            (
                "cell_distributed_advertise_host",
                config.cell_distributed_advertise_host,
            ),
        ):
            if host in ("0.0.0.0", "::", "[::]"):
                raise ValueError(f"{name} must be a connectable LAN address")
        for name, port in (
            ("cell_control_port", config.cell_control_port),
            ("cell_distributed_port", config.cell_distributed_port),
        ):
            if (
                not isinstance(port, int)
                or isinstance(port, bool)
                or not 1 <= port <= 65_535
            ):
                raise ValueError(f"{name} must be between 1 and 65535")
        reserved_ports = {config.listen_port, config.return_port}
        if config.next_port is not None:
            reserved_ports.add(config.next_port)
        if config.cell_control_port in reserved_ports or config.cell_distributed_port in reserved_ports:
            raise ValueError("external cell ports must not overlap pipeline ports")
        if config.cell_control_port == config.cell_distributed_port:
            raise ValueError("external cell control and distributed ports must differ")
    elif any(value is not None for value in external_values):
        raise ValueError("external cell endpoints require cell_mode=external")
    if (
        not math.isfinite(config.cell_operation_timeout_seconds)
        or config.cell_operation_timeout_seconds <= 0
    ):
        raise ValueError("cell_operation_timeout_seconds must be finite and positive")
    if (
        not math.isfinite(config.cell_startup_timeout_seconds)
        or config.cell_startup_timeout_seconds <= 0
    ):
        raise ValueError("cell_startup_timeout_seconds must be finite and positive")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_activation(
    frame: Frame,
    config: StageProcessConfig,
    runner: StageRunnerContract,
    request_metrics: dict[int, dict[str, Any]],
    branch_parents: Mapping[int, int] | None = None,
    tree_leaf_shapes: Mapping[int, TreeLeafShape] | None = None,
) -> None:
    if frame.request_id not in request_metrics:
        raise ValueError(f"activation for request {frame.request_id} arrived before BEGIN")
    leaf_shape = (
        None if tree_leaf_shapes is None else tree_leaf_shapes.get(frame.request_id)
    )
    if leaf_shape is not None:
        if frame.frame_type != FrameType.VERIFY:
            raise ValueError("quoted physical tree leaves require a VERIFY frame")
        if frame.token_count != leaf_shape.expected_tokens:
            raise ValueError(
                "quoted physical tree leaf shape mismatch: "
                f"got {frame.token_count}, expected {leaf_shape.expected_tokens}"
            )
        if (
            branch_parents is None
            or branch_parents.get(frame.request_id) != leaf_shape.parent_request_id
        ):
            raise RuntimeError("quoted physical tree leaf lost its parent binding")
    if (
        frame.frame_type == FrameType.VERIFY
        and config.sealed_wave_tokens is not None
        and frame.token_count > config.sealed_wave_tokens
    ):
        raise ValueError(
            "VERIFY token_count exceeds sealed_wave_tokens: "
            f"{frame.token_count} > {config.sealed_wave_tokens}"
        )
    if (
        frame.frame_type == FrameType.PREFILL
        and config.max_prefill_chunk_tokens is not None
        and frame.token_count > config.max_prefill_chunk_tokens
    ):
        raise ValueError(
            "PREFILL token_count exceeds max_prefill_chunk_tokens: "
            f"{frame.token_count} > {config.max_prefill_chunk_tokens}"
        )
    if frame.frame_type == FrameType.ACTIVATION:
        activation_limit = max(
            config.sealed_wave_tokens or 1,
            config.max_prefill_chunk_tokens or 1,
        )
        if (
            config.sealed_wave_tokens is not None
            or config.max_prefill_chunk_tokens is not None
        ) and frame.token_count > activation_limit:
            raise ValueError(
                "ACTIVATION token_count exceeds sealed activation capacity: "
                f"{frame.token_count} > {activation_limit}"
            )
    expected_step = int(request_metrics[frame.request_id]["frames"])
    if frame.step != expected_step:
        raise ValueError(
            f"activation step mismatch for request {frame.request_id}: "
            f"got {frame.step}, expected {expected_step}"
        )
    if frame.flags != int(config.codec):
        raise ValueError(
            f"activation codec mismatch: got {frame.flags}, expected {int(config.codec)}"
        )
    if frame.hidden_size != runner.hidden_size:
        raise ValueError(
            f"activation hidden size mismatch: got {frame.hidden_size}, "
            f"expected {runner.hidden_size}"
        )
    # This also proves that BEGIN reached the model runner and its local cache is active.
    current_tokens = runner.sequence_length(frame.request_id)
    if branch_parents is not None and frame.request_id in branch_parents:
        next_tokens = current_tokens + frame.token_count
        if next_tokens > config.max_speculative_branch_tokens:
            raise ValueError(
                "speculative child activation exceeds "
                "max_speculative_branch_tokens: "
                f"{next_tokens} > {config.max_speculative_branch_tokens}"
            )
        validate_speculative_kv_preflight(
            (frame,),
            config,
            runner,
            branch_parents,
        )


def connect_with_retry(host: str, port: int, timeout_seconds: float) -> socket.socket:
    deadline = time.monotonic() + timeout_seconds
    last_error: OSError | None = None
    while time.monotonic() < deadline:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        configure_socket(sock)
        try:
            remaining = max(0.001, deadline - time.monotonic())
            sock.settimeout(min(1.0, remaining))
            sock.connect((host, port))
            sock.settimeout(None)
            return sock
        except OSError as error:
            last_error = error
            sock.close()
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(min(0.05, remaining))
    raise TimeoutError(f"could not connect to {host}:{port}: {last_error}")


def monitor_downstream_control(
    downstream: socket.socket,
    upstream: socket.socket,
    pipeline_id: int,
    stopping: threading.Event,
    failed: threading.Event,
    failures: list[BaseException],
    upstream_send_lock: threading.Lock,
) -> None:
    """Relay an asynchronous downstream failure towards the pipeline root."""

    request_id = pipeline_id
    try:
        frame = recv_frame(downstream)
        if stopping.is_set():
            return
        if frame.frame_type != FrameType.ERROR:
            raise RuntimeError(
                f"unexpected downstream control frame {frame.frame_type.name}"
            )
        request_id = frame.request_id
        payload = bytes(frame.payload)
        error: BaseException = RuntimeError(
            "downstream stage failed: "
            + frame.payload.decode("utf-8", errors="replace")
        )
    except BaseException as received_error:
        if stopping.is_set():
            return
        error = received_error
        payload = (
            f"downstream control channel failed: {type(error).__name__}: {error}"
        ).encode("utf-8")[:4_096]

    # Preserve a received ERROR's pipeline/request identity verbatim. For a
    # transport EOF there is no frame identity, so use this pipeline's uint64.
    send_error_best_effort(upstream, request_id, payload, upstream_send_lock)
    failures.append(error)
    failed.set()
    try:
        upstream.shutdown(socket.SHUT_RD)
    except OSError:
        pass


def forward_shutdown_and_wait(
    downstream: socket.socket,
    request_id: int,
    control_thread: threading.Thread | None,
    *,
    emulator: LinkEmulator | None = None,
    timeout_seconds: float = STAGE_SHUTDOWN_GRACE_SECONDS,
) -> None:
    """Forward planned shutdown and wait for downstream EOF as its ACK.

    The control reader is already blocked on this socket. With ``stopping`` set
    by the caller, a downstream EOF is quiet and proves that the next stage
    consumed SHUTDOWN and closed. Closing this socket before that observation
    can turn the ordered frame into a Windows TCP reset at the next hop.
    """

    if not math.isfinite(timeout_seconds) or timeout_seconds < 0:
        raise ValueError("shutdown timeout must be finite and non-negative")
    send_frame(
        downstream,
        FrameType.SHUTDOWN,
        request_id,
        emulator=emulator,
    )
    if emulator is not None and emulator.enabled:
        emulator.flush(timeout_seconds=timeout_seconds)
    try:
        downstream.shutdown(socket.SHUT_WR)
    except OSError:
        # sendall completed; an immediate peer close also proves consumption.
        pass
    if control_thread is None:
        raise RuntimeError("intermediate stage has no downstream control reader")
    control_thread.join(timeout=timeout_seconds)
    if control_thread.is_alive():
        raise TimeoutError(
            "downstream stage did not acknowledge SHUTDOWN with EOF within "
            f"{timeout_seconds}s"
        )


def send_error_best_effort(
    sock: socket.socket | None,
    request_id: int,
    payload: bytes,
    send_lock: threading.Lock | None = None,
) -> None:
    if sock is None:
        return
    try:
        if send_lock is None:
            send_frame(sock, FrameType.ERROR, request_id, payload=payload)
        else:
            with send_lock:
                send_frame(sock, FrameType.ERROR, request_id, payload=payload)
    except BaseException:
        pass


def put_metric_best_effort(metrics_sink: Any, value: dict[str, Any]) -> None:
    """Observability is lossy by design and can never fail the data plane."""

    try:
        put_nowait = getattr(metrics_sink, "put_nowait", None)
        if callable(put_nowait):
            put_nowait(value)
        else:
            metrics_sink.put(value)
    except BaseException:
        pass


def put_startup_metric_best_effort(
    metrics_sink: Any, value: dict[str, Any]
) -> None:
    """Publish startup evidence only to sinks with a separate startup channel.

    Request metric queues are a stable FIFO ABI consumed by the engine. A
    process logger may opt into startup events without inserting a different
    document shape into that request stream.
    """

    try:
        put_startup = getattr(metrics_sink, "put_startup", None)
        if callable(put_startup):
            put_startup(value)
    except BaseException:
        pass


def executor_metric_fields(runner: StageRunnerContract) -> dict[str, str]:
    manifest = getattr(runner, "executor_manifest", None)
    to_document = getattr(manifest, "to_document", None)
    if not callable(to_document):
        return {}
    document = to_document()
    if not isinstance(document, dict) or not isinstance(document.get("schema"), str):
        return {}
    return {
        "executor_schema": document["schema"],
        "executor_id": str(manifest.executor_id),
        "executor_engine": str(manifest.engine),
        "executor_adapter": str(manifest.adapter),
    }


def execution_metric_snapshot(runner: StageRunnerContract) -> dict[str, Any]:
    """Return JSON-safe evidence for the device that really executes the stage."""

    try:
        snapshot = getattr(runner, "execution_snapshot", None)
        if not callable(snapshot):
            return {}
        value = snapshot()
        if not isinstance(value, dict):
            return {}
        return dict(value)
    except BaseException:
        # Execution telemetry follows the same best-effort contract as the
        # metric sink. It must not disrupt a valid inference path.
        return {}


def cell_rank_work_metric(runner: StageRunnerContract) -> list[dict[str, Any]]:
    """Return the closed, JSON-safe physical work evidence for one stage.

    This is intentionally best-effort like the metric sink itself. A runner
    without physical-rank evidence, or one exposing a malformed snapshot,
    produces an empty list and cannot fail an otherwise valid request.
    """

    try:
        reports = getattr(runner, "member_work_reports", None)
        if not isinstance(reports, Mapping):
            return []
        ranks = list(reports)
        if any(
            not isinstance(rank, int) or isinstance(rank, bool) or rank < 0
            for rank in ranks
        ) or sorted(ranks) != list(range(len(ranks))):
            return []

        result: list[dict[str, Any]] = []
        expected_report_keys = {
            "rank",
            "device",
            "computeDtype",
            "collectiveBackend",
            "forwardCalls",
            "collectiveCalls",
            "tokensProcessed",
            "memory",
        }
        expected_memory_keys = {
            "allocatedBytes",
            "reservedBytes",
            "peakAllocatedBytes",
        }
        for rank in sorted(ranks):
            report = reports[rank]
            if not isinstance(report, Mapping) or set(report) != expected_report_keys:
                return []
            if report.get("rank") != rank:
                return []
            device = report.get("device")
            if not isinstance(device, str) or not device:
                return []
            if not isinstance(report.get("computeDtype"), str) or not isinstance(
                report.get("collectiveBackend"), str
            ):
                return []
            counters: dict[str, int] = {}
            for name in ("forwardCalls", "collectiveCalls", "tokensProcessed"):
                value = report.get(name)
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    return []
                counters[name] = value

            memory = report.get("memory")
            if not isinstance(memory, Mapping) or set(memory) != expected_memory_keys:
                return []
            memory_values: dict[str, int] = {}
            for name in (
                "allocatedBytes",
                "reservedBytes",
                "peakAllocatedBytes",
            ):
                value = memory.get(name)
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    return []
                memory_values[name] = value
            if (
                memory_values["allocatedBytes"] > memory_values["reservedBytes"]
                or memory_values["allocatedBytes"]
                > memory_values["peakAllocatedBytes"]
            ):
                return []
            result.append(
                {
                    "rank": rank,
                    "device": device,
                    **counters,
                    "memory": memory_values,
                }
            )
        return result
    except BaseException:
        return []


def drain_metrics(metrics_queue: Any) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    while True:
        try:
            values.append(metrics_queue.get_nowait())
        except queue.Empty:
            return values
