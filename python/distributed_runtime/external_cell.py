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
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import torch
import torch.distributed as distributed

from .cell_parallel import (
    llama_attention_shard_plan,
    llama_decoder_layer_tensor_parallel,
    shard_sizes,
)
from .cell_backend import CellExecutorBackend, MAX_CELL_ACTIVATION_BATCH_SIZE
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
if TYPE_CHECKING:
    from .model import StageModelSpec

_CONTROL_SCHEMA = "gdlp-cell-control/2"
_CONTROL_HEADER = struct.Struct("!IQ")
_MAX_DOCUMENT_BYTES = 64 * 1024
_MAX_TENSOR_BYTES = 512 * 1024 * 1024
_DTYPE_ELEMENT_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2}
_WORK_REPORT_KEYS = frozenset(
    (
        "rank",
        "device",
        "computeDtype",
        "collectiveBackend",
        "forwardCalls",
        "collectiveCalls",
        "tokensProcessed",
        "memory",
    )
)
_WORK_MEMORY_KEYS = frozenset(
    ("allocatedBytes", "reservedBytes", "peakAllocatedBytes")
)
_MEMORY_REPORT_KEYS = frozenset(
    (
        "device",
        "computeDtype",
        "collectiveBackend",
        "allocatedBytes",
        "reservedBytes",
        "peakAllocatedBytes",
    )
)
_BATCH_WORK_REPORT_KEYS = frozenset(
    (
        "rank",
        "physicalForwardCalls",
        "logicalForwardItems",
        "maxPhysicalBatchSize",
    )
)


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

    MAX_PHYSICAL_BATCH_SIZE = MAX_CELL_ACTIVATION_BATCH_SIZE

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
                "exact-request-fork-safe-copy",
                "tensor-parallel-cell",
                "external-ranks",
                "multi-request-physical-batch",
                "unequal-tensor-parallel" if unequal else "equal-tensor-parallel",
            ),
        )
        self.member_reports: tuple[CellMemberReport, ...] = ()
        self.member_memory_reports: dict[int, dict[str, int | str]] = {}
        # READY proves only that a rank loaded its shard. Physical work appears
        # here exclusively after a complete, validated FORWARD response from
        # every member in the process group.
        self.member_work_reports: dict[int, dict[str, Any]] = {}
        # Additional counters keep the legacy rank-work evidence shape stable
        # while making one physical collective operation distinguishable from
        # the number of logical request rows completed by that operation.
        self.member_batch_work_reports: dict[int, dict[str, int]] = {}
        self.active_requests: set[int] = set()
        self.tokens_seen: dict[int, int] = {}
        self.member_cache_shapes: dict[int, tuple[tuple[int, ...], ...]] = {}
        self.member_layer_cache_shapes: dict[
            int, tuple[tuple[tuple[int, ...], ...], ...]
        ] = {}
        self.member_request_cache_bytes: dict[int, tuple[int, ...]] = {}
        self.member_fork_reports: dict[int, dict[str, Any]] = {}
        self.member_promotion_reports: dict[int, dict[str, Any]] = {}
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
        self.member_cache_shapes.pop(request_id, None)
        self.member_layer_cache_shapes.pop(request_id, None)
        self.member_request_cache_bytes[request_id] = (0,) * self.cell.world_size

    def end(self, request_id: int) -> None:
        self._require_open()
        if request_id not in self.active_requests:
            return
        self._request("end", {"requestId": request_id})
        self.active_requests.discard(request_id)
        self.tokens_seen.pop(request_id, None)
        self.member_cache_shapes.pop(request_id, None)
        self.member_layer_cache_shapes.pop(request_id, None)
        self.member_request_cache_bytes.pop(request_id, None)

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

    def request_cache_bytes(self, request_id: int) -> int:
        """Return exact unique KV storage bytes summed across cell ranks."""

        self._require_active(request_id)
        return sum(self._request_rank_cache_bytes(request_id))

    def project_request_cache_bytes(
        self,
        request_id: int,
        additional_tokens: int,
    ) -> int:
        """Conservatively project rank-local KV storage before another wave."""

        self._require_active(request_id)
        if not isinstance(additional_tokens, int) or isinstance(additional_tokens, bool):
            raise TypeError("additional_tokens must be an integer")
        if additional_tokens < 0:
            raise ValueError("additional_tokens cannot be negative")
        current_tokens = self.tokens_seen[request_id]
        current_bytes = self.request_cache_bytes(request_id)
        if additional_tokens == 0:
            return current_bytes
        if current_tokens < 1 or current_bytes < 1:
            raise ValueError("cannot project speculative KV bytes from an empty cache")
        bytes_per_token = (current_bytes + current_tokens - 1) // current_tokens
        return bytes_per_token * (current_tokens + additional_tokens)

    def fork_request(
        self,
        child_request_id: int,
        parent_request_id: int,
        *,
        max_cache_bytes: int,
    ) -> int:
        """Clone every rank's parent KV before publishing an independent child."""

        child_request_id = _request_id(child_request_id)
        parent_request_id = _request_id(parent_request_id)
        if child_request_id == parent_request_id:
            raise ValueError("fork child and parent request IDs must differ")
        self._require_active(parent_request_id)
        if child_request_id in self.active_requests:
            raise ValueError(f"fork child request {child_request_id} is already active")
        if not isinstance(max_cache_bytes, int) or isinstance(max_cache_bytes, bool):
            raise TypeError("max_cache_bytes must be an integer")
        if max_cache_bytes < 0:
            raise ValueError("max_cache_bytes cannot be negative")

        expected_rank_bytes = self._request_rank_cache_bytes(parent_request_id)
        copied_bytes = sum(expected_rank_bytes)
        if copied_bytes > max_cache_bytes:
            raise ValueError(
                "fork cache exceeds the preflight byte budget: "
                f"{copied_bytes} > {max_cache_bytes}"
            )
        parent_tokens = self.tokens_seen[parent_request_id]
        responses = self._request(
            "fork",
            {
                "childRequestId": child_request_id,
                "parentRequestId": parent_request_id,
                "expectedTokens": parent_tokens,
                "expectedCacheBytes": copied_bytes,
                "expectedRankCacheBytes": list(expected_rank_bytes),
                "maxCacheBytes": max_cache_bytes,
            },
        )
        try:
            member_layers, reports = self._validated_fork_responses(
                responses,
                child_request_id=child_request_id,
                parent_request_id=parent_request_id,
                expected_tokens=parent_tokens,
                expected_rank_bytes=expected_rank_bytes,
            )
        except BaseException:
            # Some ranks may already have published the clone. Never continue
            # with a cell whose cross-rank request identity is unproven.
            self._failed = True
            raise

        self.active_requests.add(child_request_id)
        self.tokens_seen[child_request_id] = parent_tokens
        self.member_layer_cache_shapes[child_request_id] = member_layers
        self.member_cache_shapes[child_request_id] = tuple(
            layers[-1] for layers in member_layers
        )
        self.member_request_cache_bytes[child_request_id] = expected_rank_bytes
        self.member_fork_reports = reports
        return copied_bytes

    def promote_request(self, parent_request_id: int, child_request_id: int) -> None:
        """Move a selected child's rank-local KV back to its parent without copying."""

        parent_request_id = _request_id(parent_request_id)
        child_request_id = _request_id(child_request_id)
        if child_request_id == parent_request_id:
            raise ValueError("promote parent and child request IDs must differ")
        self._require_active(parent_request_id)
        self._require_active(child_request_id)
        parent_tokens = self.tokens_seen[parent_request_id]
        child_tokens = self.tokens_seen[child_request_id]
        parent_rank_bytes = self._request_rank_cache_bytes(parent_request_id)
        child_rank_bytes = self._request_rank_cache_bytes(child_request_id)
        responses = self._request(
            "promote",
            {
                "parentRequestId": parent_request_id,
                "childRequestId": child_request_id,
                "expectedParentTokens": parent_tokens,
                "expectedChildTokens": child_tokens,
                "expectedParentRankCacheBytes": list(parent_rank_bytes),
                "expectedChildRankCacheBytes": list(child_rank_bytes),
            },
        )
        try:
            member_layers, reports = self._validated_promotion_responses(
                responses,
                parent_request_id=parent_request_id,
                child_request_id=child_request_id,
                expected_tokens=child_tokens,
                expected_rank_bytes=child_rank_bytes,
            )
        except BaseException:
            self._failed = True
            raise

        self.tokens_seen[parent_request_id] = child_tokens
        self.member_layer_cache_shapes[parent_request_id] = member_layers
        self.member_cache_shapes[parent_request_id] = tuple(
            layers[-1] for layers in member_layers
        )
        self.member_request_cache_bytes[parent_request_id] = child_rank_bytes
        self.active_requests.remove(child_request_id)
        self.tokens_seen.pop(child_request_id, None)
        self.member_layer_cache_shapes.pop(child_request_id, None)
        self.member_cache_shapes.pop(child_request_id, None)
        self.member_request_cache_bytes.pop(child_request_id, None)
        self.member_promotion_reports = reports

    def physical_batch_key(
        self,
        request_id: int,
        *,
        token_count: int,
        token_mode: str,
    ) -> tuple[int, int, str]:
        """Return the exact compatibility key for a physical cell batch.

        External cell members own simple rank-local Llama KV tensors, so unlike
        a generic Transformers cache there is no unsupported hybrid layout.
        Equal cache length, token count and token mode are sufficient.  The
        stage collector compares this key before it sends a batched operation.
        """

        self._require_active(request_id)
        if (
            not isinstance(token_count, int)
            or isinstance(token_count, bool)
            or token_count < 1
        ):
            raise ValueError("token_count must be a positive integer")
        if token_mode not in ("none", "last", "all"):
            raise ValueError("token_mode must be none, last or all")
        return self.tokens_seen[request_id], token_count, token_mode

    @torch.inference_mode()
    def forward_hidden_batch(
        self,
        request_ids: Sequence[int],
        hidden_states: Sequence[torch.Tensor],
        *,
        token_mode: str = "last",
    ) -> tuple[tuple[torch.Tensor, int | tuple[int, ...] | None], ...]:
        """Execute independent requests in one broadcast and TP operation.

        KV remains request-owned.  Every rank merges equal-length cache rows for
        the physical forward and splits the returned KV back into rank-one
        tensors before acknowledging the operation.
        """

        ids = _request_ids(request_ids, minimum=2)
        if len(ids) > self.MAX_PHYSICAL_BATCH_SIZE:
            raise ValueError("external cell physical batch exceeds its bounded maximum")
        if isinstance(hidden_states, (str, bytes, bytearray)):
            raise ValueError("hidden_states must be an ordered tensor sequence")
        values = tuple(hidden_states)
        if len(values) != len(ids):
            raise ValueError("request_ids and hidden_states must have equal length")
        if token_mode not in ("none", "last", "all"):
            raise ValueError("token_mode must be none, last or all")

        for request_id in ids:
            self._require_active(request_id)
        if len({self.tokens_seen[request_id] for request_id in ids}) != 1:
            raise ValueError("physical batching requires equal request cache lengths")

        first = values[0]
        if (
            not isinstance(first, torch.Tensor)
            or first.ndim != 3
            or first.shape[0] != 1
            or first.shape[1] < 1
            or first.shape[2] != self.hidden_size
        ):
            raise ValueError(
                f"hidden state must have shape [1, tokens, {self.hidden_size}]"
            )
        if not first.is_floating_point() or first.device.type != "cpu":
            raise ValueError("external cells accept CPU floating-point ingress activations")
        expected_shape = tuple(first.shape)
        for value in values[1:]:
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected_shape:
                raise ValueError(
                    "physical batching requires equal rank-one hidden-state shapes"
                )
            if not value.is_floating_point() or value.device.type != "cpu":
                raise ValueError(
                    "external cells accept CPU floating-point ingress activations"
                )

        physical = torch.cat(
            tuple(value.detach().to(dtype=torch.float32) for value in values),
            dim=0,
        ).contiguous()
        responses = self._request(
            "forward-batch",
            {"requestIds": list(ids), "activationShape": list(physical.shape)},
            physical,
        )
        input_tokens = int(first.shape[1])
        try:
            self._record_member_work(
                responses,
                input_tokens=len(ids) * input_tokens,
                logical_items=len(ids),
            )
            self._record_batch_cache_shapes(
                ids,
                responses,
                input_tokens=input_tokens,
            )
            rank_zero = next(
                (
                    (document, tensor)
                    for rank, document, tensor in responses
                    if rank == 0
                ),
                None,
            )
            if (
                rank_zero is None
                or rank_zero[1] is None
                or tuple(rank_zero[1].shape) != tuple(physical.shape)
                or not rank_zero[1].is_floating_point()
            ):
                raise RuntimeError("external rank zero returned an invalid batched output")
        except BaseException:
            # Every rank has already advanced its KV.  A malformed response can
            # no longer be retried without risking a double forward.
            self._failed = True
            raise

        for request_id in ids:
            self.tokens_seen[request_id] += input_tokens
        output = rank_zero[1]
        return tuple(
            (output[index : index + 1].contiguous(), None)
            for index in range(len(ids))
        )

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
        input_tokens = int(hidden.shape[1])
        try:
            self._record_member_work(
                responses,
                input_tokens=input_tokens,
                logical_items=1,
            )
        except BaseException:
            # Rank execution has advanced but its evidence is untrustworthy;
            # continuing would make later cumulative reports ambiguous.
            self._failed = True
            raise
        self.tokens_seen[request_id] += input_tokens
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
            self.member_request_cache_bytes.clear()
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
                if kind in ("forward", "forward-batch"):
                    if (rank == 0) != (tensor is not None):
                        raise RuntimeError(
                            "external forward tensor ownership is invalid"
                        )
                elif tensor is not None:
                    raise RuntimeError(f"external {kind} response cannot contain a tensor")
            responses.append((rank, dict(document), tensor))
        for thread in threads:
            thread.join(timeout=0.1)
        return sorted(responses, key=lambda value: value[0])

    def _request_rank_cache_bytes(self, request_id: int) -> tuple[int, ...]:
        values = self.member_request_cache_bytes.get(request_id)
        if (
            not isinstance(values, tuple)
            or len(values) != self.cell.world_size
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value < 0
                for value in values
            )
        ):
            raise RuntimeError(
                f"external request {request_id} has no sealed rank-cache byte inventory"
            )
        if self.tokens_seen[request_id] > 0 and sum(values) < 1:
            raise RuntimeError(
                f"external request {request_id} has tokens without measurable KV storage"
            )
        return values

    def _validated_fork_responses(
        self,
        responses: Sequence[tuple[int, Mapping[str, Any], torch.Tensor | None]],
        *,
        child_request_id: int,
        parent_request_id: int,
        expected_tokens: int,
        expected_rank_bytes: tuple[int, ...],
    ) -> tuple[
        tuple[tuple[tuple[int, ...], ...], ...],
        dict[int, dict[str, Any]],
    ]:
        member_layers: list[tuple[tuple[int, ...], ...]] = []
        reports: dict[int, dict[str, Any]] = {}
        pending_memory: dict[int, dict[str, int | str]] = {}
        for rank, document, _ in responses:
            copied_bytes = _nonnegative_int(
                document.get("copiedCacheBytes"), "fork copiedCacheBytes"
            )
            cache_bytes = _nonnegative_int(
                document.get("cacheBytes"), "fork cacheBytes"
            )
            if (
                document.get("childRequestId") != child_request_id
                or document.get("parentRequestId") != parent_request_id
                or document.get("tokensSeen") != expected_tokens
                or document.get("aliasFree") is not True
                or copied_bytes != expected_rank_bytes[rank]
                or cache_bytes != expected_rank_bytes[rank]
            ):
                raise RuntimeError("external fork response identity is inconsistent")
            shapes = _validated_cache_shapes(
                document.get("cacheShapes"),
                layer_count=self.layer_count,
                expected_sequence_length=expected_tokens,
            )
            if cache_bytes != _cache_shapes_storage_bytes(
                shapes,
                element_bytes=_DTYPE_ELEMENT_BYTES[self.cell.compute_dtype],
            ):
                raise RuntimeError("external fork cache bytes disagree with its shapes")
            memory = _validated_memory_report(document.get("memory"), rank=rank, cell=self.cell)
            pending_memory[rank] = memory
            member_layers.append(shapes)
            reports[rank] = {
                "rank": rank,
                "childRequestId": child_request_id,
                "parentRequestId": parent_request_id,
                "tokensSeen": expected_tokens,
                "copiedCacheBytes": copied_bytes,
                "aliasFree": True,
            }
        if set(reports) != set(range(self.cell.world_size)):
            raise RuntimeError("external fork responses do not cover every rank")
        sealed_layers = tuple(member_layers)
        parent_layers = self.member_layer_cache_shapes.get(parent_request_id)
        if parent_layers is not None and sealed_layers != parent_layers:
            raise RuntimeError("external fork changed parent cache tensor shapes")
        self.member_memory_reports.update(pending_memory)
        return sealed_layers, dict(sorted(reports.items()))

    def _validated_promotion_responses(
        self,
        responses: Sequence[tuple[int, Mapping[str, Any], torch.Tensor | None]],
        *,
        parent_request_id: int,
        child_request_id: int,
        expected_tokens: int,
        expected_rank_bytes: tuple[int, ...],
    ) -> tuple[
        tuple[tuple[tuple[int, ...], ...], ...],
        dict[int, dict[str, Any]],
    ]:
        member_layers: list[tuple[tuple[int, ...], ...]] = []
        reports: dict[int, dict[str, Any]] = {}
        pending_memory: dict[int, dict[str, int | str]] = {}
        for rank, document, _ in responses:
            cache_bytes = _nonnegative_int(
                document.get("cacheBytes"), "promote cacheBytes"
            )
            if (
                document.get("parentRequestId") != parent_request_id
                or document.get("childRequestId") != child_request_id
                or document.get("tokensSeen") != expected_tokens
                or document.get("movedWithoutCopy") is not True
                or cache_bytes != expected_rank_bytes[rank]
            ):
                raise RuntimeError("external promotion response identity is inconsistent")
            shapes = _validated_cache_shapes(
                document.get("cacheShapes"),
                layer_count=self.layer_count,
                expected_sequence_length=expected_tokens,
            )
            if cache_bytes != _cache_shapes_storage_bytes(
                shapes,
                element_bytes=_DTYPE_ELEMENT_BYTES[self.cell.compute_dtype],
            ):
                raise RuntimeError(
                    "external promotion cache bytes disagree with its shapes"
                )
            memory = _validated_memory_report(document.get("memory"), rank=rank, cell=self.cell)
            pending_memory[rank] = memory
            member_layers.append(shapes)
            reports[rank] = {
                "rank": rank,
                "parentRequestId": parent_request_id,
                "childRequestId": child_request_id,
                "tokensSeen": expected_tokens,
                "cacheBytes": cache_bytes,
                "movedWithoutCopy": True,
            }
        if set(reports) != set(range(self.cell.world_size)):
            raise RuntimeError("external promotion responses do not cover every rank")
        sealed_layers = tuple(member_layers)
        child_layers = self.member_layer_cache_shapes.get(child_request_id)
        if child_layers is not None and sealed_layers != child_layers:
            raise RuntimeError("external promotion changed selected cache tensor shapes")
        self.member_memory_reports.update(pending_memory)
        return sealed_layers, dict(sorted(reports.items()))

    def _record_cache_shapes(
        self,
        request_id: int,
        responses: Sequence[tuple[int, Mapping[str, Any], torch.Tensor | None]],
    ) -> None:
        member_layers_list: list[tuple[tuple[int, ...], ...]] = []
        rank_cache_bytes: list[int] = []
        for rank, document, _ in responses:
            shapes = _validated_cache_shapes(
                document.get("cacheShapes"),
                layer_count=self.layer_count,
                expected_sequence_length=self.tokens_seen[request_id],
            )
            cache_bytes = _nonnegative_int(
                document.get("cacheBytes"), "cacheBytes"
            )
            if cache_bytes != _cache_shapes_storage_bytes(
                shapes,
                element_bytes=_DTYPE_ELEMENT_BYTES[self.cell.compute_dtype],
            ):
                raise RuntimeError(
                    f"external rank {rank} cache bytes disagree with its shapes"
                )
            member_layers_list.append(shapes)
            rank_cache_bytes.append(cache_bytes)
        member_layers = tuple(member_layers_list)
        self.member_layer_cache_shapes[request_id] = member_layers
        self.member_cache_shapes[request_id] = tuple(layers[-1] for layers in member_layers)
        self.member_request_cache_bytes[request_id] = tuple(rank_cache_bytes)
        for rank, document, _ in responses:
            memory = document.get("memory")
            if isinstance(memory, dict):
                self.member_memory_reports[rank] = dict(memory)

    def _record_batch_cache_shapes(
        self,
        request_ids: tuple[int, ...],
        responses: Sequence[tuple[int, Mapping[str, Any], torch.Tensor | None]],
        *,
        input_tokens: int,
    ) -> None:
        expected_ids = list(request_ids)
        shapes_by_rank: list[tuple[tuple[tuple[int, ...], ...], ...]] = []
        bytes_by_rank: list[tuple[int, ...]] = []
        for rank, document, _ in responses:
            if document.get("requestIds") != expected_ids:
                raise RuntimeError("external batch response request order is invalid")
            entries = document.get("cacheShapesByRequest")
            if not isinstance(entries, list) or len(entries) != len(request_ids):
                raise RuntimeError("external batch response cache inventory is invalid")
            rank_shapes: list[tuple[tuple[int, ...], ...]] = []
            rank_cache_bytes: list[int] = []
            for index, entry in enumerate(entries):
                if (
                    not isinstance(entry, dict)
                    or set(entry) != {"requestId", "cacheShapes", "cacheBytes"}
                    or entry.get("requestId") != request_ids[index]
                ):
                    raise RuntimeError("external batch response cache identity is invalid")
                shapes = _validated_cache_shapes(
                    entry.get("cacheShapes"),
                    layer_count=self.layer_count,
                    expected_sequence_length=(
                        self.tokens_seen[request_ids[index]] + input_tokens
                    ),
                )
                cache_bytes = _nonnegative_int(
                    entry.get("cacheBytes"), "batch cacheBytes"
                )
                if cache_bytes != _cache_shapes_storage_bytes(
                    shapes,
                    element_bytes=_DTYPE_ELEMENT_BYTES[self.cell.compute_dtype],
                ):
                    raise RuntimeError(
                        f"external rank {rank} batch cache bytes disagree with its shapes"
                    )
                rank_shapes.append(shapes)
                rank_cache_bytes.append(cache_bytes)
            shapes_by_rank.append(tuple(rank_shapes))
            bytes_by_rank.append(tuple(rank_cache_bytes))
            memory = document.get("memory")
            if isinstance(memory, dict):
                self.member_memory_reports[rank] = dict(memory)

        if len(shapes_by_rank) != self.cell.world_size:
            raise RuntimeError("external batch cache reports do not cover every rank")
        for request_index, request_id in enumerate(request_ids):
            member_layers = tuple(
                rank_shapes[request_index] for rank_shapes in shapes_by_rank
            )
            self.member_layer_cache_shapes[request_id] = member_layers
            self.member_cache_shapes[request_id] = tuple(
                layers[-1] for layers in member_layers
            )
            self.member_request_cache_bytes[request_id] = tuple(
                rank_bytes[request_index] for rank_bytes in bytes_by_rank
            )

    def _record_member_work(
        self,
        responses: Sequence[tuple[int, Mapping[str, Any], torch.Tensor | None]],
        *,
        input_tokens: int,
        logical_items: int,
    ) -> None:
        if (
            not isinstance(input_tokens, int)
            or isinstance(input_tokens, bool)
            or input_tokens < 1
            or not isinstance(logical_items, int)
            or isinstance(logical_items, bool)
            or not 1 <= logical_items <= self.MAX_PHYSICAL_BATCH_SIZE
            or input_tokens % logical_items != 0
        ):
            raise RuntimeError("external cell work increment is invalid")
        collectives_per_forward = 1 + 2 * self.layer_count
        validated: dict[int, dict[str, Any]] = {}
        validated_batch: dict[int, dict[str, int]] = {}
        for response_rank, document, _ in responses:
            work = document.get("work")
            if not isinstance(work, dict) or set(work) != _WORK_REPORT_KEYS:
                raise RuntimeError(
                    f"external rank {response_rank} returned an invalid work report"
                )
            rank = _nonnegative_int(work.get("rank"), "work rank")
            if rank != response_rank or rank >= self.cell.world_size:
                raise RuntimeError("external cell work report rank mismatch")
            device = work.get("device")
            compute_dtype = work.get("computeDtype")
            collective_backend = work.get("collectiveBackend")
            if device != (self.cell.rank_devices or ())[rank]:
                raise RuntimeError("external cell work report device mismatch")
            if compute_dtype != self.cell.compute_dtype:
                raise RuntimeError("external cell work report dtype mismatch")
            if collective_backend != self.cell.collective_backend:
                raise RuntimeError("external cell work report backend mismatch")

            forward_calls = _nonnegative_int(
                work.get("forwardCalls"), "forwardCalls"
            )
            collective_calls = _nonnegative_int(
                work.get("collectiveCalls"), "collectiveCalls"
            )
            tokens_processed = _nonnegative_int(
                work.get("tokensProcessed"), "tokensProcessed"
            )
            if collective_calls != forward_calls * collectives_per_forward:
                raise RuntimeError("external cell collective work count is invalid")

            batch_work = document.get("batchWork")
            if (
                not isinstance(batch_work, dict)
                or set(batch_work) != _BATCH_WORK_REPORT_KEYS
            ):
                raise RuntimeError(
                    f"external rank {response_rank} returned invalid batch work"
                )
            batch_rank = _nonnegative_int(batch_work.get("rank"), "batch work rank")
            physical_forward_calls = _nonnegative_int(
                batch_work.get("physicalForwardCalls"), "physicalForwardCalls"
            )
            logical_forward_items = _nonnegative_int(
                batch_work.get("logicalForwardItems"), "logicalForwardItems"
            )
            max_physical_batch_size = _nonnegative_int(
                batch_work.get("maxPhysicalBatchSize"), "maxPhysicalBatchSize"
            )
            if (
                batch_rank != rank
                or physical_forward_calls != forward_calls
                or not 1 <= max_physical_batch_size <= self.MAX_PHYSICAL_BATCH_SIZE
            ):
                raise RuntimeError("external cell batch work counters are inconsistent")

            previous_batch = self.member_batch_work_reports.get(rank)
            expected_logical_items = logical_items
            expected_maximum = logical_items
            if previous_batch is not None:
                expected_logical_items += int(previous_batch["logicalForwardItems"])
                expected_maximum = max(
                    int(previous_batch["maxPhysicalBatchSize"]), logical_items
                )
            if (
                logical_forward_items != expected_logical_items
                or max_physical_batch_size != expected_maximum
            ):
                raise RuntimeError("external cell cumulative batch work is invalid")

            memory = work.get("memory")
            if not isinstance(memory, dict) or set(memory) != _WORK_MEMORY_KEYS:
                raise RuntimeError("external cell work memory report is invalid")
            allocated_bytes = _nonnegative_int(
                memory.get("allocatedBytes"), "allocatedBytes"
            )
            reserved_bytes = _nonnegative_int(
                memory.get("reservedBytes"), "reservedBytes"
            )
            peak_allocated_bytes = _nonnegative_int(
                memory.get("peakAllocatedBytes"), "peakAllocatedBytes"
            )
            if (
                allocated_bytes > reserved_bytes
                or allocated_bytes > peak_allocated_bytes
            ):
                raise RuntimeError("external cell work memory counters are inconsistent")

            previous = self.member_work_reports.get(rank)
            if previous is None:
                expected_forward_calls = 1
                expected_collective_calls = collectives_per_forward
                expected_tokens_processed = input_tokens
            else:
                expected_forward_calls = int(previous["forwardCalls"]) + 1
                expected_collective_calls = (
                    int(previous["collectiveCalls"]) + collectives_per_forward
                )
                expected_tokens_processed = (
                    int(previous["tokensProcessed"]) + input_tokens
                )
                previous_memory = previous["memory"]
                if peak_allocated_bytes < int(previous_memory["peakAllocatedBytes"]):
                    raise RuntimeError("external cell peak allocation moved backwards")
            if (
                forward_calls != expected_forward_calls
                or collective_calls != expected_collective_calls
                or tokens_processed != expected_tokens_processed
            ):
                raise RuntimeError("external cell cumulative work counters are invalid")

            reported_memory = document.get("memory")
            expected_memory = {
                "device": device,
                "computeDtype": compute_dtype,
                "collectiveBackend": collective_backend,
                "allocatedBytes": allocated_bytes,
                "reservedBytes": reserved_bytes,
                "peakAllocatedBytes": peak_allocated_bytes,
            }
            if (
                not isinstance(reported_memory, dict)
                or set(reported_memory) != _MEMORY_REPORT_KEYS
                or reported_memory != expected_memory
            ):
                raise RuntimeError("external cell forward memory reports disagree")

            validated[rank] = {
                "rank": rank,
                "device": device,
                "computeDtype": compute_dtype,
                "collectiveBackend": collective_backend,
                "forwardCalls": forward_calls,
                "collectiveCalls": collective_calls,
                "tokensProcessed": tokens_processed,
                "memory": {
                    "allocatedBytes": allocated_bytes,
                    "reservedBytes": reserved_bytes,
                    "peakAllocatedBytes": peak_allocated_bytes,
                },
            }
            validated_batch[rank] = {
                "rank": rank,
                "physicalForwardCalls": physical_forward_calls,
                "logicalForwardItems": logical_forward_items,
                "maxPhysicalBatchSize": max_physical_batch_size,
            }

        if set(validated) != set(range(self.cell.world_size)):
            raise RuntimeError("external cell work reports do not cover every rank")
        counter_sets = {
            (
                report["forwardCalls"],
                report["collectiveCalls"],
                report["tokensProcessed"],
            )
            for report in validated.values()
        }
        if len(counter_sets) != 1:
            raise RuntimeError("external cell ranks disagree about completed work")
        batch_counter_sets = {
            (
                report["physicalForwardCalls"],
                report["logicalForwardItems"],
                report["maxPhysicalBatchSize"],
            )
            for report in validated_batch.values()
        }
        if set(validated_batch) != set(range(self.cell.world_size)) or len(
            batch_counter_sets
        ) != 1:
            raise RuntimeError("external cell ranks disagree about batched work")
        self.member_work_reports = dict(sorted(validated.items()))
        self.member_batch_work_reports = dict(sorted(validated_batch.items()))

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
        self.rank = rank
        self.backend = backend
        self.caches: dict[int, list[tuple[torch.Tensor, torch.Tensor] | None]] = {}
        self.tokens_seen: dict[int, int] = {}
        self.forward_calls = 0
        self.logical_forward_items = 0
        self.max_physical_batch_size = 0
        self.collective_calls = 0
        self.tokens_processed = 0
        self.peak_allocated_bytes = 0

    def execute(
        self,
        kind: str,
        document: Mapping[str, Any],
        tensor: torch.Tensor | None,
        *,
        rank: int,
    ) -> tuple[dict[str, Any], torch.Tensor | None, bool]:
        if rank != self.rank:
            raise ValueError("member executor rank is inconsistent")
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
            if token_count < self.tokens_seen[request_id]:
                for index, cache in enumerate(self.caches[request_id]):
                    if cache is not None:
                        key, value = cache
                        # clone() compacts a slice that may report contiguous
                        # while still retaining the pre-truncate storage.
                        self.caches[request_id][index] = (
                            key[:, :, :token_count, :].contiguous().clone(),
                            value[:, :, :token_count, :].contiguous().clone(),
                        )
            self.tokens_seen[request_id] = token_count
            result["cacheShapes"] = _cache_shapes(
                self.caches[request_id], self.plans
            )
            result["cacheBytes"] = _request_cache_storage_bytes(
                self.caches[request_id]
            )
        elif kind == "fork":
            if tensor is not None:
                raise ValueError("FORK cannot contain a tensor")
            child_request_id = _request_id(document.get("childRequestId"))
            parent_request_id = _request_id(document.get("parentRequestId"))
            if child_request_id == parent_request_id:
                raise ValueError("fork child and parent request IDs must differ")
            self._require_request(parent_request_id)
            if (
                child_request_id in self.tokens_seen
                or child_request_id in self.caches
            ):
                raise ValueError("fork child request is already active")
            expected_tokens = _protocol_nonnegative_int(
                document.get("expectedTokens"), "expectedTokens"
            )
            expected_cache_bytes = _protocol_nonnegative_int(
                document.get("expectedCacheBytes"), "expectedCacheBytes"
            )
            max_cache_bytes = _protocol_nonnegative_int(
                document.get("maxCacheBytes"), "maxCacheBytes"
            )
            expected_rank_bytes = _protocol_rank_cache_bytes(
                document.get("expectedRankCacheBytes"),
                world_size=int(self.manifest["worldSize"]),
            )
            if (
                self.tokens_seen[parent_request_id] != expected_tokens
                or sum(expected_rank_bytes) != expected_cache_bytes
                or expected_cache_bytes > max_cache_bytes
            ):
                raise ValueError("fork preflight identity or byte budget is inconsistent")
            parent_cache = self.caches[parent_request_id]
            parent_cache_bytes = _request_cache_storage_bytes(parent_cache)
            if parent_cache_bytes != expected_rank_bytes[self.rank]:
                raise ValueError("fork rank cache bytes changed before cloning")

            # Do not publish the child until every rank-local tensor has been
            # copied and the storage identity sets prove there is no alias.
            child_cache = _clone_request_cache(parent_cache)
            parent_storage = _request_cache_storage_keys(parent_cache)
            child_storage = _request_cache_storage_keys(child_cache)
            if parent_storage & child_storage:
                raise RuntimeError("forked external cache aliases parent tensor storage")
            child_cache_bytes = _request_cache_storage_bytes(child_cache)
            if child_cache_bytes != parent_cache_bytes:
                raise RuntimeError("forked external cache changed storage byte size")
            self.caches[child_request_id] = child_cache
            self.tokens_seen[child_request_id] = expected_tokens
            result.update(
                {
                    "childRequestId": child_request_id,
                    "parentRequestId": parent_request_id,
                    "tokensSeen": expected_tokens,
                    "copiedCacheBytes": parent_cache_bytes,
                    "cacheBytes": child_cache_bytes,
                    "cacheShapes": _cache_shapes(child_cache, self.plans),
                    "aliasFree": True,
                    "memory": self._memory_report(),
                }
            )
        elif kind == "promote":
            if tensor is not None:
                raise ValueError("PROMOTE cannot contain a tensor")
            parent_request_id = _request_id(document.get("parentRequestId"))
            child_request_id = _request_id(document.get("childRequestId"))
            if child_request_id == parent_request_id:
                raise ValueError("promote parent and child request IDs must differ")
            self._require_request(parent_request_id)
            self._require_request(child_request_id)
            expected_parent_tokens = _protocol_nonnegative_int(
                document.get("expectedParentTokens"), "expectedParentTokens"
            )
            expected_child_tokens = _protocol_nonnegative_int(
                document.get("expectedChildTokens"), "expectedChildTokens"
            )
            expected_parent_rank_bytes = _protocol_rank_cache_bytes(
                document.get("expectedParentRankCacheBytes"),
                world_size=int(self.manifest["worldSize"]),
            )
            expected_rank_bytes = _protocol_rank_cache_bytes(
                document.get("expectedChildRankCacheBytes"),
                world_size=int(self.manifest["worldSize"]),
            )
            parent_cache_bytes = _request_cache_storage_bytes(
                self.caches[parent_request_id]
            )
            child_cache = self.caches[child_request_id]
            child_cache_bytes = _request_cache_storage_bytes(child_cache)
            if (
                self.tokens_seen[parent_request_id] != expected_parent_tokens
                or self.tokens_seen[child_request_id] != expected_child_tokens
                or parent_cache_bytes != expected_parent_rank_bytes[self.rank]
                or child_cache_bytes != expected_rank_bytes[self.rank]
            ):
                raise ValueError("promotion preflight identity is inconsistent")

            child_storage = _request_cache_storage_keys(child_cache)
            moved_cache = self.caches.pop(child_request_id)
            self.caches[parent_request_id] = moved_cache
            self.tokens_seen[parent_request_id] = self.tokens_seen.pop(child_request_id)
            if _request_cache_storage_keys(self.caches[parent_request_id]) != child_storage:
                raise RuntimeError("promotion copied or changed selected KV storage")
            result.update(
                {
                    "parentRequestId": parent_request_id,
                    "childRequestId": child_request_id,
                    "tokensSeen": expected_child_tokens,
                    "cacheBytes": child_cache_bytes,
                    "cacheShapes": _cache_shapes(
                        self.caches[parent_request_id], self.plans
                    ),
                    "movedWithoutCopy": True,
                    "memory": self._memory_report(),
                }
            )
        elif kind in ("forward", "forward-batch"):
            request_ids = (
                (_request_id(request_id),)
                if kind == "forward"
                else _request_ids(document.get("requestIds"), minimum=2)
            )
            if len(request_ids) > MAX_CELL_ACTIVATION_BATCH_SIZE:
                raise ValueError("external cell physical batch exceeds its maximum")
            for current_request_id in request_ids:
                self._require_request(current_request_id)
            current_lengths = {
                self.tokens_seen[current_request_id]
                for current_request_id in request_ids
            }
            if len(current_lengths) != 1:
                raise ValueError("physical batch request cache lengths differ")
            current_length = next(iter(current_lengths))

            output = self.backend.broadcast_activation(
                tensor, document.get("activationShape")
            )
            if int(output.shape[0]) != len(request_ids):
                raise ValueError("activation batch does not match requestIds")
            input_tokens = int(output.shape[1])
            for index, (weights, plan, sizes, layer) in enumerate(
                zip(
                    self.layer_weights,
                    self.plans,
                    self.intermediate_sizes,
                    self.manifest["layers"],
                )
            ):
                merged_cache = self._merge_request_layer_caches(
                    request_ids,
                    layer_index=index,
                    current_length=current_length,
                )
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
                    past_key_value=merged_cache,
                    rms_norm_epsilon=float(layer["rmsNormEpsilon"]),
                    rope_theta=float(layer["ropeTheta"]),
                )
                self._split_request_layer_cache(
                    request_ids,
                    layer_index=index,
                    present=present,
                    expected_length=current_length + input_tokens,
                )

            for current_request_id in request_ids:
                self.tokens_seen[current_request_id] += input_tokens
            physical_batch_size = len(request_ids)
            memory = self._memory_report()
            self.forward_calls += 1
            self.logical_forward_items += physical_batch_size
            self.max_physical_batch_size = max(
                self.max_physical_batch_size, physical_batch_size
            )
            self.collective_calls += 1 + 2 * len(self.layer_weights)
            self.tokens_processed += physical_batch_size * input_tokens
            if kind == "forward":
                result["cacheShapes"] = _cache_shapes(
                    self.caches[request_ids[0]], self.plans
                )
                result["cacheBytes"] = _request_cache_storage_bytes(
                    self.caches[request_ids[0]]
                )
            else:
                result["requestIds"] = list(request_ids)
                result["cacheShapesByRequest"] = [
                    {
                        "requestId": current_request_id,
                        "cacheShapes": _cache_shapes(
                            self.caches[current_request_id], self.plans
                        ),
                        "cacheBytes": _request_cache_storage_bytes(
                            self.caches[current_request_id]
                        ),
                    }
                    for current_request_id in request_ids
                ]
            result["memory"] = memory
            result["work"] = {
                "rank": self.rank,
                "device": memory["device"],
                "computeDtype": memory["computeDtype"],
                "collectiveBackend": memory["collectiveBackend"],
                # Legacy evidence keeps this physical-call counter and shape.
                "forwardCalls": self.forward_calls,
                "collectiveCalls": self.collective_calls,
                "tokensProcessed": self.tokens_processed,
                "memory": {
                    "allocatedBytes": int(memory["allocatedBytes"]),
                    "reservedBytes": int(memory["reservedBytes"]),
                    "peakAllocatedBytes": self.peak_allocated_bytes,
                },
            }
            result["batchWork"] = {
                "rank": self.rank,
                "physicalForwardCalls": self.forward_calls,
                "logicalForwardItems": self.logical_forward_items,
                "maxPhysicalBatchSize": self.max_physical_batch_size,
            }
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

    def _memory_report(self) -> dict[str, int | str]:
        memory = dict(self.backend.memory_report())
        self.peak_allocated_bytes = max(
            self.peak_allocated_bytes,
            int(memory["peakAllocatedBytes"]),
        )
        memory["peakAllocatedBytes"] = self.peak_allocated_bytes
        return memory

    def _merge_request_layer_caches(
        self,
        request_ids: tuple[int, ...],
        *,
        layer_index: int,
        current_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        caches = tuple(self.caches[request_id][layer_index] for request_id in request_ids)
        if current_length == 0:
            if any(cache is not None for cache in caches):
                raise RuntimeError("empty physical batch has unexpected KV state")
            return None
        if any(cache is None for cache in caches):
            raise RuntimeError("physical batch has incomplete KV state")
        concrete = tuple(cache for cache in caches if cache is not None)
        for key, value in concrete:
            if (
                key.ndim != 4
                or value.ndim != 4
                or key.shape[0] != 1
                or value.shape[0] != 1
                or key.shape[2] != current_length
                or value.shape[2] != current_length
                or key.shape != value.shape
                or key.dtype != self.backend.torch_dtype
                or value.dtype != self.backend.torch_dtype
                or key.device != self.backend.torch_device
                or value.device != self.backend.torch_device
            ):
                raise RuntimeError("physical batch KV tensors are incompatible")
        key_shapes = {tuple(key.shape[1:]) for key, _ in concrete}
        value_shapes = {tuple(value.shape[1:]) for _, value in concrete}
        if len(key_shapes) != 1 or len(value_shapes) != 1:
            raise RuntimeError("physical batch KV tensor shapes differ")
        return (
            torch.cat(tuple(key for key, _ in concrete), dim=0).contiguous(),
            torch.cat(tuple(value for _, value in concrete), dim=0).contiguous(),
        )

    def _split_request_layer_cache(
        self,
        request_ids: tuple[int, ...],
        *,
        layer_index: int,
        present: tuple[torch.Tensor, torch.Tensor],
        expected_length: int,
    ) -> None:
        if not isinstance(present, tuple) or len(present) != 2:
            raise RuntimeError("physical batch returned invalid KV state")
        key, value = present
        plan = self.plans[layer_index]
        expected_shape = (
            len(request_ids),
            int(plan.local_key_value_heads),
            expected_length,
            int(plan.head_dim),
        )
        if (
            key.ndim != 4
            or value.ndim != 4
            or key.shape != value.shape
            or tuple(key.shape) != expected_shape
            or key.dtype != self.backend.torch_dtype
            or value.dtype != self.backend.torch_dtype
            or key.device != self.backend.torch_device
            or value.device != self.backend.torch_device
        ):
            raise RuntimeError("physical batch returned an invalid KV shape")
        for batch_index, current_request_id in enumerate(request_ids):
            # clone() prevents one small request cache from retaining storage for
            # every other row in the physical batch.
            self.caches[current_request_id][layer_index] = (
                key[batch_index : batch_index + 1].contiguous().clone(),
                value[batch_index : batch_index + 1].contiguous().clone(),
            )


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


def _request_ids(value: Any, *, minimum: int) -> tuple[int, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
    ):
        raise ValueError("requestIds must be an ordered sequence")
    request_ids = tuple(_request_id(item) for item in value)
    if not minimum <= len(request_ids) <= MAX_CELL_ACTIVATION_BATCH_SIZE:
        raise ValueError("requestIds count is outside the physical batch bound")
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("requestIds must identify distinct requests")
    return request_ids


def _protocol_nonnegative_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _protocol_rank_cache_bytes(value: Any, *, world_size: int) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) != world_size:
        raise ValueError("rank cache byte inventory does not cover the process group")
    return tuple(
        _protocol_nonnegative_int(item, "rank cache bytes") for item in value
    )


def _request_cache_storage_keys(
    caches: Sequence[tuple[torch.Tensor, torch.Tensor] | None],
) -> frozenset[tuple[str, int | None, int, int]]:
    keys: set[tuple[str, int | None, int, int]] = set()
    for cache in caches:
        if cache is None:
            continue
        if not isinstance(cache, tuple) or len(cache) != 2:
            raise RuntimeError("external request cache entry is invalid")
        for tensor in cache:
            if not isinstance(tensor, torch.Tensor):
                raise RuntimeError("external request cache contains a non-tensor")
            storage = tensor.untyped_storage()
            storage_bytes = int(storage.nbytes())
            if storage_bytes == 0:
                continue
            keys.add(
                (
                    tensor.device.type,
                    tensor.device.index,
                    int(storage.data_ptr()),
                    storage_bytes,
                )
            )
    return frozenset(keys)


def _request_cache_storage_bytes(
    caches: Sequence[tuple[torch.Tensor, torch.Tensor] | None],
) -> int:
    return sum(key[3] for key in _request_cache_storage_keys(caches))


def _clone_request_cache(
    caches: Sequence[tuple[torch.Tensor, torch.Tensor] | None],
) -> list[tuple[torch.Tensor, torch.Tensor] | None]:
    cloned: list[tuple[torch.Tensor, torch.Tensor] | None] = []
    for cache in caches:
        if cache is None:
            cloned.append(None)
            continue
        if not isinstance(cache, tuple) or len(cache) != 2:
            raise RuntimeError("external request cache entry cannot be cloned safely")
        key, value = cache
        if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
            raise RuntimeError("external request cache tensors cannot be cloned safely")
        cloned.append(
            (
                key.clone(memory_format=torch.contiguous_format),
                value.clone(memory_format=torch.contiguous_format),
            )
        )
    return cloned


def _cache_shapes_storage_bytes(
    shapes: Sequence[Sequence[int]],
    *,
    element_bytes: int,
) -> int:
    return sum(math.prod(shape) * element_bytes * 2 for shape in shapes)


def _validated_cache_shapes(
    value: Any,
    *,
    layer_count: int,
    expected_sequence_length: int,
) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, list) or len(value) != layer_count:
        raise RuntimeError("external batch cache layer count is invalid")
    result: list[tuple[int, ...]] = []
    for shape in value:
        if (
            not isinstance(shape, list)
            or len(shape) != 4
            or any(
                not isinstance(item, int) or isinstance(item, bool) or item < 0
                for item in shape
            )
            or shape[0] != 1
            or shape[1] < 1
            or shape[2] != expected_sequence_length
            or shape[3] < 1
        ):
            raise RuntimeError("external batch cache shape is invalid")
        result.append(tuple(shape))
    return tuple(result)


def _validated_memory_report(
    value: Any,
    *,
    rank: int,
    cell: ExternalTensorParallelCellSpec,
) -> dict[str, int | str]:
    if not isinstance(value, dict) or set(value) != _MEMORY_REPORT_KEYS:
        raise RuntimeError("external control memory report is invalid")
    if (
        value.get("device") != (cell.rank_devices or ())[rank]
        or value.get("computeDtype") != cell.compute_dtype
        or value.get("collectiveBackend") != cell.collective_backend
    ):
        raise RuntimeError("external control memory identity is inconsistent")
    allocated = _nonnegative_int(value.get("allocatedBytes"), "allocatedBytes")
    reserved = _nonnegative_int(value.get("reservedBytes"), "reservedBytes")
    peak = _nonnegative_int(value.get("peakAllocatedBytes"), "peakAllocatedBytes")
    if allocated > reserved or allocated > peak:
        raise RuntimeError("external control memory counters are inconsistent")
    return {
        "device": str(value["device"]),
        "computeDtype": str(value["computeDtype"]),
        "collectiveBackend": str(value["collectiveBackend"]),
        "allocatedBytes": allocated,
        "reservedBytes": reserved,
        "peakAllocatedBytes": peak,
    }


def _nonnegative_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise RuntimeError(f"external cell {name} must be a nonnegative integer")
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
            or not 1 <= tensor.shape[0] <= MAX_CELL_ACTIVATION_BATCH_SIZE
            or tensor.shape[1] < 1
            or tensor.shape[2] < 1
        ):
            raise ValueError(
                "cell control tensors must have bounded shape [batch,tokens,hidden]"
            )
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
            or shape[0] > MAX_CELL_ACTIVATION_BATCH_SIZE
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
