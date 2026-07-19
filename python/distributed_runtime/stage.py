from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import queue
import socket
import threading
import time
from typing import Any

from .model import StageModelSpec, StageRunner, StageRunnerContract
from .protocol import (
    Frame,
    FrameType,
    LinkEmulator,
    TensorCodec,
    configure_socket,
    decode_tensor,
    encode_tensor,
    recv_frame,
    send_frame,
    token_payload,
    verify_result_payload,
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
    try:
        validate_stage_config(config)
        runner = build_stage_runner(config)
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
            frame = recv_frame(upstream)
            if frame.frame_type == FrameType.BEGIN:
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
                }
                if downstream is not None:
                    request_metrics[frame.request_id]["bytes_out"] += send_frame(
                        downstream,
                        FrameType.BEGIN,
                        frame.request_id,
                    )
            elif frame.frame_type in (
                FrameType.ACTIVATION,
                FrameType.PREFILL,
                FrameType.VERIFY,
            ):
                validate_activation(frame, config, runner, request_metrics)
                metrics = request_metrics[frame.request_id]
                hidden = decode_tensor(frame)
                started = time.perf_counter()
                token_mode = (
                    "none"
                    if frame.frame_type == FrameType.PREFILL
                    else "all"
                    if frame.frame_type == FrameType.VERIFY
                    else "last"
                )
                output, token = runner.forward_hidden(
                    frame.request_id,
                    hidden,
                    token_mode=token_mode,
                )
                metrics["compute_ms"] += (time.perf_counter() - started) * 1_000
                metrics["frames"] += 1
                metrics["tokens"] += frame.token_count
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
                else:
                    if downstream is None:
                        raise RuntimeError("intermediate stage has no downstream connection")
                    payload = encode_tensor(output, config.codec)
                    metrics["bytes_out"] += send_frame(
                        downstream,
                        frame.frame_type,
                        frame.request_id,
                        step=frame.step,
                        token_count=output.shape[1],
                        hidden_size=output.shape[2],
                        flags=int(config.codec),
                        payload=payload,
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


def build_stage_runner(config: StageProcessConfig) -> StageRunnerContract:
    """Construct the single-member runner or a logical local TP cell."""

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
) -> None:
    if frame.request_id not in request_metrics:
        raise ValueError(f"activation for request {frame.request_id} arrived before BEGIN")
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
