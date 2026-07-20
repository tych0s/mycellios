from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import queue
import select
import socket
import threading
import time
from typing import Any

import torch

from .model import (
    MAX_PHYSICAL_STAGE_BATCH_SIZE,
    StageModelSpec,
    StageRunner,
    StageRunnerContract,
)
from .protocol import (
    Frame,
    FrameType,
    LinkEmulator,
    TensorCodec,
    configure_socket,
    decode_tensor,
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
    ram_backed_moe: RamBackedMoeRuntimeConfig | None = None
    native_stage_package: str | None = None
    native_stage_package_id: str | None = None
    native_stage_manifest_sha256: str | None = None
    native_stage_daemon_command: tuple[str, ...] = ()
    native_stage_context_tokens: int | None = None
    native_stage_gpu_layers: int = 0
    native_stage_compute_api: str = "cpu"
    native_stage_startup_timeout_seconds: float = 120.0
    native_stage_call_timeout_seconds: float = 120.0
    native_stage_close_timeout_seconds: float = 5.0


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
    """Serialize a multiplexed stage stream onto one backend KV sequence.

    NativeStage's pinned daemon has only seq0. Other request frames are retained
    per request while the admitted request continues to consume the upstream
    stream. This preserves each request's frame order without head-of-line
    blocking the active request behind a queued BEGIN from another request.
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
    request_admission: SingleRequestAdmission | None = None
    try:
        validate_stage_config(config)
        runner = build_stage_runner(config)
        request_admission = request_admission_for_runner(runner)
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
            frame = pending_frames.popleft() if pending_frames else None
            if frame is None and request_admission is not None:
                frame = request_admission.next_deferred()
            if frame is None:
                frame = recv_frame(upstream)
            if (
                request_admission is not None
                and not request_admission.admit_or_defer(frame)
            ):
                continue
            if frame.frame_type == FrameType.BEGIN:
                begin_stage_request(frame, config, runner, request_metrics, downstream)
            elif frame.frame_type in (
                FrameType.ACTIVATION,
                FrameType.PREFILL,
                FrameType.VERIFY,
            ):
                validate_activation(frame, config, runner, request_metrics)
                frames = collect_compatible_activation_frames(
                    frame,
                    upstream=upstream,
                    pending_frames=pending_frames,
                    config=config,
                    runner=runner,
                    request_metrics=request_metrics,
                    downstream=downstream,
                )
                process_activation_frames(
                    frames,
                    config=config,
                    runner=runner,
                    request_metrics=request_metrics,
                    downstream=downstream,
                    return_socket=return_socket,
                    emulator=emulator,
                )
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
                runner.end(frame.request_id)
                if downstream is not None:
                    send_frame(downstream, frame.frame_type, frame.request_id)
                metrics = request_metrics.pop(frame.request_id, None)
                if metrics is not None:
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
                if downstream is not None:
                    send_frame(downstream, FrameType.SHUTDOWN, frame.request_id)
                break
            elif frame.frame_type == FrameType.ERROR:
                message = frame.payload.decode("utf-8", errors="replace")
                raise RuntimeError(f"upstream stage reported an error: {message}")
            else:
                raise ValueError(f"unexpected frame {frame.frame_type.name}")
    except BaseException as error:
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
        "frames": 0,
        "compute_ms": 0,
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


def collect_compatible_activation_frames(
    first: Frame,
    *,
    upstream: socket.socket,
    pending_frames: deque[Frame],
    config: StageProcessConfig,
    runner: StageRunnerContract,
    request_metrics: dict[int, dict[str, Any]],
    downstream: socket.socket | None,
) -> tuple[Frame, ...]:
    """Read a bounded run of activations that can share one model forward.

    BEGIN is request-local and may be applied while the batch window is open.
    Every other control frame is left in order for the main loop.  Cell runners
    expose no physical batch operation and therefore never enter this collector.
    """

    batch_forward = getattr(runner, "forward_hidden_batch", None)
    batch_key = getattr(runner, "physical_batch_key", None)
    if (
        getattr(runner, "max_active_requests", None) == 1
        or config.max_physical_batch_size < 2
        or not callable(batch_forward)
        or not callable(batch_key)
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
        validate_activation(candidate, config, runner, request_metrics)
        candidate_mode = activation_token_mode(candidate.frame_type)
        candidate_key = batch_key(
            candidate.request_id,
            token_count=candidate.token_count,
            token_mode=candidate_mode,
        )
        compatible = (
            candidate.request_id not in request_ids
            and candidate.frame_type == first.frame_type
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
    downstream: socket.socket | None,
    return_socket: socket.socket | None,
    emulator: LinkEmulator,
) -> None:
    """Execute and route one sequential or genuinely batched model operation."""

    if not frames:
        raise ValueError("activation frame batch cannot be empty")
    token_mode = activation_token_mode(frames[0].frame_type)
    hidden_states = tuple(decode_tensor(frame) for frame in frames)
    timings_ms: tuple[float, ...]
    if len(frames) > 1:
        batch_forward = getattr(runner, "forward_hidden_batch", None)
        if not callable(batch_forward):
            raise TypeError("runner advertised a physical batch key without a batch forward")
        started = time.perf_counter()
        results = tuple(
            batch_forward(
                tuple(frame.request_id for frame in frames),
                hidden_states,
                token_mode=token_mode,
            )
        )
        elapsed_ms = (time.perf_counter() - started) * 1_000
        if len(results) != len(frames):
            raise RuntimeError("physical batch returned the wrong number of results")
        timings_ms = (elapsed_ms,) * len(frames)
    else:
        started = time.perf_counter()
        results = (
            runner.forward_hidden(
                frames[0].request_id,
                hidden_states[0],
                token_mode=token_mode,
            ),
        )
        timings_ms = ((time.perf_counter() - started) * 1_000,)

    physical_size = len(frames)
    for frame, result, compute_ms in zip(frames, results, timings_ms, strict=True):
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError("stage runner result must be an (output, token) pair")
        output, token = result
        if not isinstance(output, torch.Tensor):
            raise TypeError("stage runner output must be a tensor")
        metrics = request_metrics[frame.request_id]
        metrics["compute_ms"] += compute_ms
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

    if config.ram_backed_moe is not None:
        return build_ram_backed_moe_stage_runner(
            config.spec,
            config.ram_backed_moe,
            pipeline_snapshot_identity=config.pipeline_id,
        )
    if config.native_stage_package is not None:
        from .native_stage import NativeStageStageRunner, NativeStageStageRuntimeSpec

        if not config.native_stage_daemon_command:
            raise ValueError("NativeStage daemon command is missing")
        if config.native_stage_context_tokens is None:
            raise ValueError("NativeStage context-token limit is missing")
        return NativeStageStageRunner(
            config.spec,
            NativeStageStageRuntimeSpec(
                package=config.native_stage_package,
                daemon_command=config.native_stage_daemon_command,
                context_tokens=config.native_stage_context_tokens,
                threads=config.spec.threads,
                gpu_layers=config.native_stage_gpu_layers,
                compute_api=config.native_stage_compute_api,
                startup_timeout_seconds=config.native_stage_startup_timeout_seconds,
                call_timeout_seconds=config.native_stage_call_timeout_seconds,
                close_timeout_seconds=config.native_stage_close_timeout_seconds,
                expected_pipeline_id=config.pipeline_id,
                expected_package_id=config.native_stage_package_id,
                expected_manifest_sha256=config.native_stage_manifest_sha256,
            ),
        )
    if config.cell_fixture is None:
        return StageRunner(config.spec)
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
    has_native_stage = config.native_stage_package is not None
    has_ram_backed_moe = config.ram_backed_moe is not None
    if has_ram_backed_moe:
        if has_native_stage or config.cell_fixture is not None:
            raise ValueError(
                "RAM-backed MoE, NativeStage and tensor-parallel cell backends "
                "are mutually exclusive"
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
    if has_native_stage:
        if not isinstance(config.native_stage_package, str) or not config.native_stage_package.strip():
            raise ValueError("native_stage_package cannot be empty")
        if config.spec.first:
            raise ValueError(
                "NativeStage child-stage adapter cannot execute layer_start=0"
            )
        if (
            not isinstance(config.native_stage_daemon_command, tuple)
            or not config.native_stage_daemon_command
            or any(
                not isinstance(argument, str) or not argument
                for argument in config.native_stage_daemon_command
            )
        ):
            raise ValueError("NativeStage daemon command must be a non-empty argv tuple")
        if (
            not isinstance(config.native_stage_context_tokens, int)
            or isinstance(config.native_stage_context_tokens, bool)
            or config.native_stage_context_tokens < 1
        ):
            raise ValueError("native_stage_context_tokens must be a positive integer")
        for name, digest in (
            ("native_stage_package_id", config.native_stage_package_id),
            ("native_stage_manifest_sha256", config.native_stage_manifest_sha256),
        ):
            if digest is not None and (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    elif (
        config.native_stage_package_id is not None
        or config.native_stage_manifest_sha256 is not None
        or config.native_stage_daemon_command
        or config.native_stage_context_tokens is not None
        or config.native_stage_gpu_layers != 0
        or config.native_stage_compute_api != "cpu"
    ):
        raise ValueError("NativeStage runtime settings require native_stage_package")
    if (
        not isinstance(config.native_stage_gpu_layers, int)
        or isinstance(config.native_stage_gpu_layers, bool)
        or config.native_stage_gpu_layers < 0
    ):
        raise ValueError("native_stage_gpu_layers must be a non-negative integer")
    if config.native_stage_compute_api not in ("cpu", "cuda", "rocm", "metal", "vulkan"):
        raise ValueError("native_stage_compute_api is unsupported")
    for name, value in (
        ("native_stage_startup_timeout_seconds", config.native_stage_startup_timeout_seconds),
        ("native_stage_call_timeout_seconds", config.native_stage_call_timeout_seconds),
        ("native_stage_close_timeout_seconds", config.native_stage_close_timeout_seconds),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    has_cell_fixture = config.cell_fixture is not None
    has_cell_size = config.cell_world_size is not None
    if has_native_stage and has_cell_fixture:
        raise ValueError("NativeStage and tensor-parallel cell backends are mutually exclusive")
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
) -> None:
    if frame.request_id not in request_metrics:
        raise ValueError(f"activation for request {frame.request_id} arrived before BEGIN")
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
    runner.sequence_length(frame.request_id)


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


def drain_metrics(metrics_queue: Any) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    while True:
        try:
            values.append(metrics_queue.get_nowait())
        except queue.Empty:
            return values
