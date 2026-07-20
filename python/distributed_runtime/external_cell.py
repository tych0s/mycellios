from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import json
import math
import multiprocessing
from pathlib import Path
import queue
import socket
import struct
import threading
import time
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as distributed

from .cell_parallel import (
    llama_attention_shard_plan,
    llama_decoder_layer_tensor_parallel,
    shard_sizes,
)
from .cell_backend import CellExecutorBackend
from .cell_stage import (
    CellMemberReport,
    _TENSOR_NAMES,
    _cache_shapes,
    _read_manifest,
    _safe_shard_path,
    _sha256_file,
    _tensor_key,
    _validate_local_weights,
)
from .model import StageModelSpec


_CONTROL_SCHEMA = "gdlp-cell-control/1"
_CONTROL_HEADER = struct.Struct("!IQ")
_MAX_DOCUMENT_BYTES = 64 * 1024
_MAX_TENSOR_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True)
class ExternalTensorParallelCellSpec:
    """Anchor configuration for a cell whose nonzero ranks are external."""

    fixture: str
    world_size: int
    pipeline_id: int
    control_host: str
    control_port: int
    control_advertise_host: str
    distributed_advertise_host: str
    distributed_port: int
    collective_backend: str = "gloo"
    rank_devices: tuple[str, ...] | None = None
    compute_dtype: str = "float32"
    startup_timeout_seconds: float = 120.0
    operation_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if not self.fixture.strip():
            raise ValueError("external cell fixture cannot be empty")
        if (
            not isinstance(self.world_size, int)
            or isinstance(self.world_size, bool)
            or self.world_size < 2
        ):
            raise ValueError("external cell world_size must be at least two")
        if (
            not isinstance(self.pipeline_id, int)
            or isinstance(self.pipeline_id, bool)
            or not 0 <= self.pipeline_id <= (1 << 64) - 1
        ):
            raise ValueError("external cell pipeline_id must be uint64")
        for name, value in (
            ("control_host", self.control_host),
            ("control_advertise_host", self.control_advertise_host),
            ("distributed_advertise_host", self.distributed_advertise_host),
        ):
            if not value.strip():
                raise ValueError(f"{name} cannot be empty")
        for name, value in (
            ("control_advertise_host", self.control_advertise_host),
            ("distributed_advertise_host", self.distributed_advertise_host),
        ):
            if value in ("0.0.0.0", "::", "[::]"):
                raise ValueError(f"{name} must be a connectable LAN address")
        for name, value in (
            ("control_port", self.control_port),
            ("distributed_port", self.distributed_port),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or not 1 <= value <= 65_535
            ):
                raise ValueError(f"{name} must be between 1 and 65535")
        if self.control_port == self.distributed_port:
            raise ValueError("control and distributed ports must be different")
        devices = self.rank_devices
        if devices is None:
            devices = tuple("cpu" for _ in range(self.world_size))
            object.__setattr__(self, "rank_devices", devices)
        if (
            not isinstance(devices, tuple)
            or len(devices) != self.world_size
            or any(not isinstance(device, str) or not device for device in devices)
        ):
            raise ValueError("external cell rank_devices must declare every rank")
        for device in devices:
            CellExecutorBackend(
                collective_backend=self.collective_backend,
                device=device,
                compute_dtype=self.compute_dtype,
            )
        for name, value in (
            ("startup_timeout_seconds", self.startup_timeout_seconds),
            ("operation_timeout_seconds", self.operation_timeout_seconds),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class ExternalCellMemberConfig:
    """Configuration consumed by ``cell_member_cli`` on one contributor."""

    fixture: str
    rank: int
    world_size: int
    pipeline_id: int
    layer_start: int
    layer_end: int
    control_host: str
    control_port: int
    collective_backend: str = "gloo"
    device: str = "cpu"
    compute_dtype: str = "float32"
    threads: int = 1
    connect_timeout_seconds: float = 120.0
    operation_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if not self.fixture.strip() or not self.control_host.strip():
            raise ValueError("member fixture and control_host cannot be empty")
        if (
            not isinstance(self.world_size, int)
            or isinstance(self.world_size, bool)
            or self.world_size < 2
        ):
            raise ValueError("member world_size must be at least two")
        if (
            not isinstance(self.rank, int)
            or isinstance(self.rank, bool)
            or not 0 <= self.rank < self.world_size
        ):
            raise ValueError("member rank is outside the process group")
        if (
            not isinstance(self.pipeline_id, int)
            or isinstance(self.pipeline_id, bool)
            or not 0 <= self.pipeline_id <= (1 << 64) - 1
        ):
            raise ValueError("member pipeline_id must be uint64")
        if (
            not isinstance(self.layer_start, int)
            or isinstance(self.layer_start, bool)
            or not isinstance(self.layer_end, int)
            or isinstance(self.layer_end, bool)
            or not 0 <= self.layer_start < self.layer_end
        ):
            raise ValueError("member layer range is invalid")
        if (
            not isinstance(self.control_port, int)
            or isinstance(self.control_port, bool)
            or not 1 <= self.control_port <= 65_535
        ):
            raise ValueError("member control_port must be between 1 and 65535")
        if not isinstance(self.threads, int) or isinstance(self.threads, bool) or self.threads < 1:
            raise ValueError("member threads must be positive")
        CellExecutorBackend(
            collective_backend=self.collective_backend,
            device=self.device,
            compute_dtype=self.compute_dtype,
        )
        for name, value in (
            ("connect_timeout_seconds", self.connect_timeout_seconds),
            ("operation_timeout_seconds", self.operation_timeout_seconds),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


class ExternalTensorParallelCellStageRunner:
    """Logical stage with rank zero on the anchor and ranks 1..N-1 external.

    All members, including the locally spawned rank zero, use the same TCP
    control protocol. Tensor-parallel collectives use a separate process
    group advertised by the anchor. The anchor itself loads only rank zero's
    shard; external members open only their own rank-specific file.
    """

    def __init__(self, spec: StageModelSpec, cell: ExternalTensorParallelCellSpec) -> None:
        if spec.first or spec.last:
            raise ValueError("external cells currently support intermediate stages only")
        root = Path(cell.fixture).resolve()
        manifest = _read_manifest(root)
        if manifest["schema"] != "gdlp-llama-cell-stage/2":
            raise ValueError("external ranks require a version-2 cell fixture")
        if manifest["worldSize"] != cell.world_size:
            raise ValueError("external cell world_size does not match its fixture")
        if spec.layer_end - spec.layer_start != len(manifest["layers"]):
            raise ValueError("external cell layer range does not match its fixture")

        self.spec = spec
        self.cell = cell
        self.hidden_size = int(manifest["layers"][0]["hiddenSize"])
        self.layer_count = len(manifest["layers"])
        self.parameter_bytes = 0
        rank_zero_backend = CellExecutorBackend(
            collective_backend=cell.collective_backend,
            device=(cell.rank_devices or ())[0],
            compute_dtype=cell.compute_dtype,
        )
        rank_zero_backend.validate_fixture_dtype(manifest["dtype"])
        self.loader = (
            "tensor-parallel-cell-external-safetensors-gloo"
            if cell.collective_backend == "gloo" and cell.compute_dtype == "float32"
            else (
                "tensor-parallel-cell-external-safetensors-"
                f"{cell.collective_backend}-{cell.compute_dtype}"
            )
        )
        from .executor_abi import (
            build_stage_executor_manifest,
            model_identity_for_source,
        )

        unequal = len(set(manifest["rankWeights"])) > 1
        model_identity = spec.artifact_identity or model_identity_for_source(
            spec.model_name, spec.revision
        )
        model_source = spec.canonical_model_source or (
            f"content-addressed://{spec.artifact_identity}"
            if spec.artifact_identity is not None
            else spec.model_name
        )
        model_revision = (
            spec.canonical_model_revision
            if spec.artifact_identity is not None
            else spec.revision
        )
        self.executor_manifest = build_stage_executor_manifest(
            engine="python-torch-cell",
            engine_version=torch.__version__,
            adapter="llama-tensor-parallel-safetensors-external",
            model_identity=model_identity,
            model_source=model_source,
            model_revision=model_revision,
            artifact_format="safetensors-tp-cell",
            layer_start=spec.layer_start,
            layer_end=spec.layer_end,
            total_layers=spec.total_layers,
            hidden_size=self.hidden_size,
            activation_dtype="float32",
            activation_codecs=(
                "fp32",
                "fp16",
                "int8",
                "int8-grouped",
                "int8-hadamard",
            ),
            device_kinds=("cpu",) if cell.collective_backend == "gloo" else ("gpu",),
            compute_apis=("torch", cell.collective_backend),
            weight_dtypes=(cell.compute_dtype,),
            features=(
                "layer-range",
                "rank-local-kv",
                "rollback",
                "tensor-parallel-cell",
                "external-ranks",
                "unequal-tensor-parallel" if unequal else "equal-tensor-parallel",
            ),
        )
        self.member_reports: tuple[CellMemberReport, ...] = ()
        self.member_memory_reports: dict[int, dict[str, int | str]] = {}
        self.active_requests: set[int] = set()
        self.tokens_seen: dict[int, int] = {}
        self.member_cache_shapes: dict[int, tuple[tuple[int, ...], ...]] = {}
        self.member_layer_cache_shapes: dict[
            int, tuple[tuple[tuple[int, ...], ...], ...]
        ] = {}
        self._closed = False
        self._ready = False
        self._failed = False
        self._operation = 0
        self._manifest_digest = _sha256_file(root / "cell.json")
        self._manifest = manifest
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind((cell.control_host, cell.control_port))
        self._listener.listen(cell.world_size)
        self._listener.settimeout(cell.startup_timeout_seconds)
        self._members: dict[int, socket.socket] = {}
        self._context = multiprocessing.get_context("spawn")
        rank_zero_config = ExternalCellMemberConfig(
            fixture=str(root),
            rank=0,
            world_size=cell.world_size,
            pipeline_id=cell.pipeline_id,
            layer_start=spec.layer_start,
            layer_end=spec.layer_end,
            control_host=cell.control_advertise_host,
            control_port=cell.control_port,
            collective_backend=cell.collective_backend,
            device=(cell.rank_devices or ())[0],
            compute_dtype=cell.compute_dtype,
            threads=spec.threads,
            connect_timeout_seconds=cell.startup_timeout_seconds,
            operation_timeout_seconds=cell.operation_timeout_seconds,
        )
        self._rank_zero = self._context.Process(
            target=run_external_cell_member,
            args=(rank_zero_config,),
            name=f"gdlp-external-cell-{spec.layer_start}-rank-0",
        )
        self._rank_zero.start()
        try:
            self._accept_members()
            ready = self._receive_all("ready", operation=0)
            reports = tuple(
                CellMemberReport(
                    rank=rank,
                    shard_file=str(document["shardFile"]),
                    parameter_bytes=int(document["parameterBytes"]),
                    tensor_count=int(document["tensorCount"]),
                    device=str(document["device"]),
                    compute_dtype=str(document["computeDtype"]),
                    collective_backend=str(document["collectiveBackend"]),
                    allocated_bytes=int(document["allocatedBytes"]),
                    reserved_bytes=int(document["reservedBytes"]),
                    peak_allocated_bytes=int(document["peakAllocatedBytes"]),
                )
                for rank, document, tensor in ready
            )
            if any(tensor is not None for _, _, tensor in ready):
                raise RuntimeError("READY packets cannot contain tensors")
            expected_fixed = self._manifest.get("rankFixedBytes")
            if expected_fixed is not None and any(
                report.parameter_bytes != int(expected_fixed[report.rank])
                for report in reports
            ):
                raise RuntimeError("cell rank parameter bytes do not match its manifest")
            self.member_reports = reports
            self.member_memory_reports = {
                report.rank: {
                    "device": report.device,
                    "computeDtype": report.compute_dtype,
                    "collectiveBackend": report.collective_backend,
                    "allocatedBytes": report.allocated_bytes,
                    "reservedBytes": report.reserved_bytes,
                    "peakAllocatedBytes": report.peak_allocated_bytes,
                }
                for report in reports
            }
            self.parameter_bytes = sum(report.parameter_bytes for report in reports)
            for connection in self._members.values():
                connection.settimeout(self.cell.operation_timeout_seconds)
            self._ready = True
        except BaseException:
            self._failed = True
            self._abort()
            raise

    def begin(self, request_id: int) -> None:
        self._require_open()
        if request_id in self.active_requests:
            raise ValueError(f"request {request_id} is already active")
        self._request("begin", {"requestId": _request_id(request_id)})
        self.active_requests.add(request_id)
        self.tokens_seen[request_id] = 0

    def end(self, request_id: int) -> None:
        self._require_open()
        if request_id not in self.active_requests:
            return
        self._request("end", {"requestId": request_id})
        self.active_requests.discard(request_id)
        self.tokens_seen.pop(request_id, None)
        self.member_cache_shapes.pop(request_id, None)
        self.member_layer_cache_shapes.pop(request_id, None)

    def truncate(self, request_id: int, token_count: int) -> None:
        self._require_active(request_id)
        if not isinstance(token_count, int) or isinstance(token_count, bool):
            raise TypeError("token_count must be an integer")
        current = self.tokens_seen[request_id]
        if not 0 <= token_count <= current:
            raise ValueError(
                f"cannot truncate request {request_id} from {current} to {token_count}"
            )
        responses = self._request(
            "truncate", {"requestId": request_id, "tokenCount": token_count}
        )
        self.tokens_seen[request_id] = token_count
        self._record_cache_shapes(request_id, responses)

    def sequence_length(self, request_id: int) -> int:
        self._require_active(request_id)
        return self.tokens_seen[request_id]

    @torch.inference_mode()
    def forward_hidden(
        self,
        request_id: int,
        hidden: torch.Tensor,
        *,
        token_mode: str = "last",
    ) -> tuple[torch.Tensor, int | tuple[int, ...] | None]:
        self._require_active(request_id)
        if token_mode not in ("none", "last", "all"):
            raise ValueError("token_mode must be none, last or all")
        if (
            not isinstance(hidden, torch.Tensor)
            or hidden.ndim != 3
            or hidden.shape[0] != 1
            or hidden.shape[1] < 1
            or hidden.shape[2] != self.hidden_size
        ):
            raise ValueError(
                f"hidden state must have shape [1, tokens, {self.hidden_size}]"
            )
        if not hidden.is_floating_point() or hidden.device.type != "cpu":
            raise ValueError("external cells accept CPU floating-point ingress activations")
        responses = self._request(
            "forward",
            {"requestId": request_id, "activationShape": list(hidden.shape)},
            hidden.detach().to(dtype=torch.float32).contiguous(),
        )
        self.tokens_seen[request_id] += int(hidden.shape[1])
        self._record_cache_shapes(request_id, responses)
        rank_zero = next(
            ((document, tensor) for rank, document, tensor in responses if rank == 0),
            None,
        )
        if rank_zero is None or rank_zero[1] is None:
            raise RuntimeError("external rank zero did not return the stage output")
        return rank_zero[1], None

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._ready and not self._failed:
                self._request("shutdown", {})
        except BaseException:
            self._failed = True
        finally:
            self._abort()
            self.active_requests.clear()
            self.tokens_seen.clear()
            self.member_cache_shapes.clear()
            self.member_layer_cache_shapes.clear()
            self._closed = True

    def _accept_members(self) -> None:
        deadline = time.monotonic() + self.cell.startup_timeout_seconds
        while len(self._members) < self.cell.world_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("external cell members did not connect before timeout")
            self._listener.settimeout(remaining)
            connection, _ = self._listener.accept()
            connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            connection.settimeout(remaining)
            try:
                document, tensor = _recv_packet(connection)
                rank = self._validate_hello(document, tensor)
                if rank in self._members:
                    raise ValueError(f"duplicate external cell rank {rank}")
                _send_packet(
                    connection,
                    {
                        "schema": _CONTROL_SCHEMA,
                        "type": "accepted",
                        "rank": rank,
                        "distributedHost": self.cell.distributed_advertise_host,
                        "distributedPort": self.cell.distributed_port,
                    },
                )
                connection.settimeout(self.cell.startup_timeout_seconds)
                self._members[rank] = connection
            except BaseException as error:
                try:
                    _send_packet(
                        connection,
                        {
                            "schema": _CONTROL_SCHEMA,
                            "type": "rejected",
                            "message": f"{type(error).__name__}: {error}",
                        },
                    )
                except BaseException:
                    pass
                connection.close()
                raise
        self._listener.close()

    def _validate_hello(
        self,
        document: Mapping[str, Any],
        tensor: torch.Tensor | None,
    ) -> int:
        if tensor is not None:
            raise ValueError("member HELLO cannot contain a tensor")
        expected = {
            "schema": _CONTROL_SCHEMA,
            "type": "hello",
            "pipelineId": self.cell.pipeline_id,
            "worldSize": self.cell.world_size,
            "layerStart": self.spec.layer_start,
            "layerEnd": self.spec.layer_end,
            "manifestSha256": self._manifest_digest,
            "collectiveBackend": self.cell.collective_backend,
            "computeDtype": self.cell.compute_dtype,
        }
        for name, value in expected.items():
            if document.get(name) != value:
                raise ValueError(f"external member HELLO has mismatched {name}")
        rank = document.get("rank")
        if (
            not isinstance(rank, int)
            or isinstance(rank, bool)
            or not 0 <= rank < self.cell.world_size
        ):
            raise ValueError("external member HELLO has invalid rank")
        if document.get("shardSha256") != self._manifest["shardSha256"][rank]:
            raise ValueError("external member shard digest does not match the manifest")
        if document.get("device") != (self.cell.rank_devices or ())[rank]:
            raise ValueError("external member HELLO has mismatched device")
        return rank

    def _request(
        self,
        kind: str,
        payload: dict[str, Any],
        tensor: torch.Tensor | None = None,
    ) -> list[tuple[int, dict[str, Any], torch.Tensor | None]]:
        self._require_open()
        self._operation += 1
        operation = self._operation
        document = {
            "schema": _CONTROL_SCHEMA,
            "type": "command",
            "operation": operation,
            "kind": kind,
            **payload,
        }
        try:
            for rank in range(self.cell.world_size):
                _send_packet(self._members[rank], document, tensor if rank == 0 else None)
            return self._receive_all(kind, operation)
        except BaseException:
            self._failed = True
            raise

    def _receive_all(
        self,
        kind: str,
        operation: int,
    ) -> list[tuple[int, dict[str, Any], torch.Tensor | None]]:
        values: queue.Queue[tuple[int, dict[str, Any] | None, torch.Tensor | None, BaseException | None]] = queue.Queue()

        def receive(rank: int, connection: socket.socket) -> None:
            try:
                document, tensor = _recv_packet(connection)
                values.put((rank, document, tensor, None))
            except BaseException as error:
                values.put((rank, None, None, error))

        threads = [
            threading.Thread(
                target=receive,
                args=(rank, self._members[rank]),
                name=f"external-cell-recv-{rank}",
                daemon=True,
            )
            for rank in range(self.cell.world_size)
        ]
        for thread in threads:
            thread.start()
        deadline = time.monotonic() + (
            self.cell.startup_timeout_seconds if operation == 0 else self.cell.operation_timeout_seconds
        )
        responses: list[tuple[int, dict[str, Any], torch.Tensor | None]] = []
        seen: set[int] = set()
        while len(responses) < self.cell.world_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"external cell {kind} timed out")
            try:
                rank, document, tensor, error = values.get(timeout=remaining)
            except queue.Empty as error:
                raise TimeoutError(f"external cell {kind} timed out") from error
            if error is not None:
                raise RuntimeError(f"external cell rank {rank} disconnected: {error}") from error
            if rank in seen or document is None:
                raise RuntimeError("external cell returned duplicate or empty response")
            seen.add(rank)
            if document.get("schema") != _CONTROL_SCHEMA:
                raise RuntimeError("external cell response schema mismatch")
            if operation == 0:
                if (
                    document.get("type") != "ready"
                    or document.get("status") != "ok"
                    or document.get("rank") != rank
                ):
                    raise RuntimeError(
                        f"external rank {rank} failed startup: {document.get('message')}"
                    )
            else:
                if (
                    document.get("type") != "response"
                    or document.get("operation") != operation
                    or document.get("kind") != kind
                    or document.get("rank") != rank
                ):
                    raise RuntimeError("external cell response identity mismatch")
                if document.get("status") != "ok":
                    raise RuntimeError(
                        f"external rank {rank} failed {kind}: {document.get('message')}"
                    )
                if kind == "forward":
                    if (rank == 0) != (tensor is not None):
                        raise RuntimeError("external FORWARD tensor ownership is invalid")
                elif tensor is not None:
                    raise RuntimeError(f"external {kind} response cannot contain a tensor")
            responses.append((rank, dict(document), tensor))
        for thread in threads:
            thread.join(timeout=0.1)
        return sorted(responses, key=lambda value: value[0])

    def _record_cache_shapes(
        self,
        request_id: int,
        responses: Sequence[tuple[int, Mapping[str, Any], torch.Tensor | None]],
    ) -> None:
        member_layers = tuple(
            tuple(
                tuple(int(value) for value in shape)
                for shape in document["cacheShapes"]
            )
            for _, document, _ in responses
        )
        self.member_layer_cache_shapes[request_id] = member_layers
        self.member_cache_shapes[request_id] = tuple(layers[-1] for layers in member_layers)
        for rank, document, _ in responses:
            memory = document.get("memory")
            if isinstance(memory, dict):
                self.member_memory_reports[rank] = dict(memory)

    def _require_active(self, request_id: int) -> None:
        self._require_open()
        if request_id not in self.active_requests:
            raise ValueError(f"request {request_id} has not received BEGIN")

    def _require_open(self) -> None:
        if self._closed or self._failed:
            raise RuntimeError("external tensor-parallel cell is not usable")

    def _abort(self) -> None:
        try:
            self._listener.close()
        except OSError:
            pass
        for connection in self._members.values():
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            connection.close()
        self._rank_zero.join(timeout=2.0)
        if self._rank_zero.is_alive():
            self._rank_zero.terminate()
            self._rank_zero.join(timeout=2.0)

    def __enter__(self) -> ExternalTensorParallelCellStageRunner:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class _MemberExecutor:
    def __init__(
        self,
        fixture: str,
        rank: int,
        world_size: int,
        backend: CellExecutorBackend,
    ) -> None:
        root = Path(fixture).resolve()
        self.manifest = _read_manifest(root)
        if self.manifest["schema"] != "gdlp-llama-cell-stage/2":
            raise ValueError("external members require a version-2 fixture")
        if self.manifest["worldSize"] != world_size:
            raise ValueError("member world_size does not match fixture")
        backend.validate_fixture_dtype(self.manifest["dtype"])
        shard_file = _safe_shard_path(root, self.manifest["shards"][rank])
        actual_digest = _sha256_file(shard_file)
        if actual_digest != self.manifest["shardSha256"][rank]:
            raise ValueError("member shard digest does not match fixture manifest")
        self.shard_file = shard_file
        self.shard_digest = actual_digest
        weights = backend.load_weights(str(shard_file))
        expected_keys = {
            _tensor_key(self.manifest, layer_index, name)
            for layer_index in range(len(self.manifest["layers"]))
            for name in _TENSOR_NAMES
        }
        if set(weights) != expected_keys:
            raise ValueError("member shard has an invalid tensor set")
        self.layer_weights: list[dict[str, torch.Tensor]] = []
        self.plans: list[Any] = []
        self.intermediate_sizes: list[tuple[int, ...]] = []
        for layer_index, layer in enumerate(self.manifest["layers"]):
            local = {
                name: weights[_tensor_key(self.manifest, layer_index, name)]
                for name in _TENSOR_NAMES
            }
            plan = llama_attention_shard_plan(
                int(layer["hiddenSize"]),
                int(layer["numAttentionHeads"]),
                int(layer["numKeyValueHeads"]),
                int(layer["headDim"]),
                rank,
                world_size,
                self.manifest["rankWeights"],
            )
            sizes = shard_sizes(
                int(layer["intermediateSize"]),
                world_size,
                self.manifest["rankWeights"],
            )
            _validate_local_weights(
                local,
                layer,
                plan,
                sizes[rank],
                expected_dtype=backend.torch_dtype,
            )
            self.layer_weights.append(local)
            self.plans.append(plan)
            self.intermediate_sizes.append(sizes)
        self.parameter_bytes = sum(
            tensor.nelement() * tensor.element_size() for tensor in weights.values()
        )
        self.tensor_count = len(weights)
        self.backend = backend
        self.caches: dict[int, list[tuple[torch.Tensor, torch.Tensor] | None]] = {}
        self.tokens_seen: dict[int, int] = {}

    def execute(
        self,
        kind: str,
        document: Mapping[str, Any],
        tensor: torch.Tensor | None,
        *,
        rank: int,
    ) -> tuple[dict[str, Any], torch.Tensor | None, bool]:
        request_id = document.get("requestId")
        result: dict[str, Any] = {}
        if kind == "begin":
            request_id = _request_id(request_id)
            if request_id in self.tokens_seen or tensor is not None:
                raise ValueError("invalid or duplicate BEGIN")
            self.tokens_seen[request_id] = 0
            self.caches[request_id] = [None] * len(self.layer_weights)
        elif kind == "end":
            request_id = _request_id(request_id)
            if tensor is not None:
                raise ValueError("END cannot contain a tensor")
            self.caches.pop(request_id, None)
            self.tokens_seen.pop(request_id, None)
        elif kind == "truncate":
            request_id = _request_id(request_id)
            self._require_request(request_id)
            if tensor is not None:
                raise ValueError("TRUNCATE cannot contain a tensor")
            token_count = document.get("tokenCount")
            if (
                not isinstance(token_count, int)
                or isinstance(token_count, bool)
                or not 0 <= token_count <= self.tokens_seen[request_id]
            ):
                raise ValueError("invalid TRUNCATE token count")
            for index, cache in enumerate(self.caches[request_id]):
                if cache is not None:
                    key, value = cache
                    self.caches[request_id][index] = (
                        key[:, :, :token_count, :].contiguous(),
                        value[:, :, :token_count, :].contiguous(),
                    )
            self.tokens_seen[request_id] = token_count
            result["cacheShapes"] = _cache_shapes(
                self.caches[request_id], self.plans
            )
        elif kind == "forward":
            request_id = _request_id(request_id)
            self._require_request(request_id)
            output = self.backend.broadcast_activation(
                tensor, document.get("activationShape")
            )
            input_tokens = int(output.shape[1])
            request_caches = self.caches[request_id]
            for index, (weights, plan, sizes, layer) in enumerate(
                zip(
                    self.layer_weights,
                    self.plans,
                    self.intermediate_sizes,
                    self.manifest["layers"],
                )
            ):
                output, present = llama_decoder_layer_tensor_parallel(
                    output,
                    weights["input_norm"],
                    weights["post_attention_norm"],
                    weights["query"],
                    weights["key"],
                    weights["value"],
                    weights["output"],
                    plan,
                    weights["gate"],
                    weights["up"],
                    weights["down"],
                    sizes,
                    past_key_value=request_caches[index],
                    rms_norm_epsilon=float(layer["rmsNormEpsilon"]),
                    rope_theta=float(layer["ropeTheta"]),
                )
                request_caches[index] = present
            self.tokens_seen[request_id] += input_tokens
            result["cacheShapes"] = _cache_shapes(request_caches, self.plans)
            result["memory"] = self.backend.memory_report()
            return (
                result,
                output.to(device="cpu", dtype=torch.float32) if rank == 0 else None,
                False,
            )
        elif kind == "shutdown":
            if tensor is not None or len(document) != 4:
                # schema/type/operation/kind plus no command payload.
                raise ValueError("invalid SHUTDOWN command")
            return result, None, True
        else:
            raise ValueError(f"unsupported external cell operation {kind!r}")
        return result, None, False

    def _require_request(self, request_id: int) -> None:
        if request_id not in self.tokens_seen:
            raise ValueError(f"request {request_id} has not received BEGIN")


def run_external_cell_member(config: ExternalCellMemberConfig) -> None:
    """Join an anchor, execute commands, and exit after ordered SHUTDOWN."""

    torch.set_num_threads(config.threads)
    backend = CellExecutorBackend(
        collective_backend=config.collective_backend,
        device=config.device,
        compute_dtype=config.compute_dtype,
    )
    backend.activate()
    root = Path(config.fixture).resolve()
    manifest = _read_manifest(root)
    if manifest["schema"] != "gdlp-llama-cell-stage/2":
        raise ValueError("external members require a version-2 fixture")
    if manifest["worldSize"] != config.world_size:
        raise ValueError("member world_size does not match fixture")
    if config.layer_end - config.layer_start != len(manifest["layers"]):
        raise ValueError("member layer range does not match fixture")
    shard_file = _safe_shard_path(root, manifest["shards"][config.rank])
    shard_digest = _sha256_file(shard_file)
    manifest_digest = _sha256_file(root / "cell.json")
    connection = _connect_with_retry(
        config.control_host,
        config.control_port,
        config.connect_timeout_seconds,
    )
    connection.settimeout(config.operation_timeout_seconds)
    operation = 0
    kind = "ready"
    try:
        _send_packet(
            connection,
            {
                "schema": _CONTROL_SCHEMA,
                "type": "hello",
                "pipelineId": config.pipeline_id,
                "rank": config.rank,
                "worldSize": config.world_size,
                "layerStart": config.layer_start,
                "layerEnd": config.layer_end,
                "manifestSha256": manifest_digest,
                "shardSha256": shard_digest,
                "collectiveBackend": config.collective_backend,
                "computeDtype": config.compute_dtype,
                "device": config.device,
            },
        )
        accepted, tensor = _recv_packet(connection)
        if tensor is not None or accepted.get("type") != "accepted":
            raise RuntimeError(
                f"external cell anchor rejected rank {config.rank}: "
                f"{accepted.get('message')}"
            )
        executor = _MemberExecutor(
            config.fixture, config.rank, config.world_size, backend
        )
        distributed_host = accepted.get("distributedHost")
        distributed_port = accepted.get("distributedPort")
        if not isinstance(distributed_host, str) or not distributed_host.strip():
            raise ValueError("anchor returned an invalid distributed host")
        if (
            not isinstance(distributed_port, int)
            or isinstance(distributed_port, bool)
            or not 1 <= distributed_port <= 65_535
        ):
            raise ValueError("anchor returned an invalid distributed port")
        distributed.init_process_group(
            backend=config.collective_backend,
            init_method=f"tcp://{distributed_host}:{distributed_port}",
            rank=config.rank,
            world_size=config.world_size,
            timeout=timedelta(seconds=config.operation_timeout_seconds),
        )
        memory = backend.memory_report()
        _send_packet(
            connection,
            {
                "schema": _CONTROL_SCHEMA,
                "type": "ready",
                "status": "ok",
                "rank": config.rank,
                "shardFile": executor.shard_file.name,
                "parameterBytes": executor.parameter_bytes,
                "tensorCount": executor.tensor_count,
                **memory,
            },
        )
        # Long pauses between chat turns are normal. Collective execution is
        # bounded by the process-group timeout; the ordered command stream may
        # remain idle indefinitely until the anchor sends work or closes it.
        connection.settimeout(None)
        expected_operation = 1
        while True:
            document, tensor = _recv_packet(connection)
            operation = document.get("operation")
            kind = document.get("kind")
            if (
                document.get("schema") != _CONTROL_SCHEMA
                or document.get("type") != "command"
                or operation != expected_operation
                or not isinstance(kind, str)
            ):
                raise ValueError("external cell command identity or order is invalid")
            result, output, stopping = executor.execute(
                kind, document, tensor, rank=config.rank
            )
            _send_packet(
                connection,
                {
                    "schema": _CONTROL_SCHEMA,
                    "type": "response",
                    "status": "ok",
                    "rank": config.rank,
                    "operation": operation,
                    "kind": kind,
                    **result,
                },
                output,
            )
            expected_operation += 1
            if stopping:
                break
    except BaseException as error:
        try:
            _send_packet(
                connection,
                {
                    "schema": _CONTROL_SCHEMA,
                    "type": "ready" if operation == 0 else "response",
                    "status": "error",
                    "rank": config.rank,
                    "operation": operation,
                    "kind": kind,
                    "message": f"{type(error).__name__}: {error}",
                },
            )
        except BaseException:
            pass
        raise
    finally:
        if distributed.is_initialized():
            distributed.destroy_process_group()
        connection.close()


def _request_id(value: Any) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not 0 <= value <= (1 << 64) - 1
    ):
        raise ValueError("requestId must be uint64")
    return value


def _send_packet(
    connection: socket.socket,
    document: Mapping[str, Any],
    tensor: torch.Tensor | None = None,
) -> None:
    rendered = json.dumps(document, separators=(",", ":"), sort_keys=True).encode("utf-8")
    if not 1 <= len(rendered) <= _MAX_DOCUMENT_BYTES:
        raise ValueError("cell control document exceeds its size bound")
    tensor_payload = b""
    if tensor is not None:
        if (
            tensor.ndim != 3
            or tensor.shape[0] != 1
            or tensor.shape[1] < 1
            or tensor.shape[2] < 1
        ):
            raise ValueError("cell control tensors must have shape [1,tokens,hidden]")
        contiguous = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
        tensor_payload = contiguous.numpy().tobytes(order="C")
        if len(tensor_payload) > _MAX_TENSOR_BYTES:
            raise ValueError("cell control tensor exceeds its size bound")
        document_with_tensor = dict(document)
        document_with_tensor["tensorShape"] = list(contiguous.shape)
        rendered = json.dumps(
            document_with_tensor, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
    connection.sendall(_CONTROL_HEADER.pack(len(rendered), len(tensor_payload)))
    connection.sendall(rendered)
    if tensor_payload:
        connection.sendall(tensor_payload)


def _recv_packet(
    connection: socket.socket,
) -> tuple[dict[str, Any], torch.Tensor | None]:
    header = _recv_exact(connection, _CONTROL_HEADER.size)
    document_size, tensor_size = _CONTROL_HEADER.unpack(header)
    if not 1 <= document_size <= _MAX_DOCUMENT_BYTES:
        raise ValueError("invalid cell control document size")
    if tensor_size > _MAX_TENSOR_BYTES or tensor_size % 4 != 0:
        raise ValueError("invalid cell control tensor size")
    document_value = json.loads(_recv_exact(connection, document_size).decode("utf-8"))
    if not isinstance(document_value, dict):
        raise ValueError("cell control document must be an object")
    tensor: torch.Tensor | None = None
    shape = document_value.pop("tensorShape", None)
    if tensor_size:
        if (
            not isinstance(shape, list)
            or len(shape) != 3
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value < 1
                for value in shape
            )
            or shape[0] != 1
            or math.prod(shape) * 4 != tensor_size
        ):
            raise ValueError("cell control tensor shape does not match its payload")
        payload = _recv_exact(connection, tensor_size)
        tensor = torch.frombuffer(payload, dtype=torch.float32).clone().reshape(shape)
    elif shape is not None:
        raise ValueError("cell control document declares a missing tensor")
    return document_value, tensor


def _recv_exact(connection: socket.socket, size: int) -> bytearray:
    payload = bytearray(size)
    view = memoryview(payload)
    offset = 0
    while offset < size:
        received = connection.recv_into(view[offset:])
        if received == 0:
            raise EOFError("external cell control socket closed")
        offset += received
    return payload


def _connect_with_retry(host: str, port: int, timeout_seconds: float) -> socket.socket:
    deadline = time.monotonic() + timeout_seconds
    last_error: OSError | None = None
    while time.monotonic() < deadline:
        connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            connection.settimeout(min(1.0, max(0.001, deadline - time.monotonic())))
            connection.connect((host, port))
            connection.settimeout(timeout_seconds)
            return connection
        except OSError as error:
            last_error = error
            connection.close()
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(min(0.05, remaining))
    raise TimeoutError(f"could not connect to external cell anchor: {last_error}")


__all__ = [
    "ExternalCellMemberConfig",
    "ExternalTensorParallelCellSpec",
    "ExternalTensorParallelCellStageRunner",
    "run_external_cell_member",
]
