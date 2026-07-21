from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import queue
import socket
import time
from typing import Any, Mapping, Sequence

from safetensors.torch import save_file
import torch
import torch.distributed as distributed

from .cell_parallel import (
    llama_attention_shard_plan,
    llama_decoder_layer_tensor_parallel,
    shard_column_linear,
    shard_llama_attention,
    shard_row_linear,
    shard_sizes,
)
from .cell_backend import rank_backend
from .model import StageModelSpec


_SCHEMA_V1 = "gdlp-llama-cell-layer/1"
_SCHEMA_V2 = "gdlp-llama-cell-stage/2"
_TENSOR_NAMES = (
    "input_norm",
    "post_attention_norm",
    "query",
    "key",
    "value",
    "output",
    "gate",
    "up",
    "down",
)


@dataclass(frozen=True)
class TensorParallelCellSpec:
    """Local-process topology for one logical tensor-parallel stage."""

    fixture: str
    world_size: int
    backend: str = "gloo"
    rank_devices: tuple[str, ...] | None = None
    compute_dtype: str = "float32"
    operation_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if not self.fixture.strip():
            raise ValueError("cell fixture cannot be empty")
        if not isinstance(self.world_size, int) or isinstance(self.world_size, bool):
            raise TypeError("cell world_size must be an integer")
        if self.world_size < 2:
            raise ValueError("a tensor-parallel cell requires at least two members")
        devices = self.rank_devices
        if devices is None:
            devices = tuple("cpu" for _ in range(self.world_size))
            object.__setattr__(self, "rank_devices", devices)
        if (
            not isinstance(devices, tuple)
            or len(devices) != self.world_size
            or any(not isinstance(device, str) or not device for device in devices)
        ):
            raise ValueError("cell rank_devices must declare one device per member")
        for rank in range(self.world_size):
            rank_backend(self.backend, devices, self.compute_dtype, rank)
        if (
            not math.isfinite(self.operation_timeout_seconds)
            or self.operation_timeout_seconds <= 0
        ):
            raise ValueError("cell operation timeout must be finite and positive")


@dataclass(frozen=True)
class CellMemberReport:
    rank: int
    shard_file: str
    parameter_bytes: int
    tensor_count: int
    device: str
    compute_dtype: str
    collective_backend: str
    allocated_bytes: int
    reserved_bytes: int
    peak_allocated_bytes: int


class TensorParallelCellStageRunner:
    """One logical pipeline stage backed by local tensor-parallel members.

    The coordinator reads only the small JSON manifest.  Each child opens its
    own rank-specific SafeTensors shard after joining the process group.  The
    children execute collectives in lockstep and retain request KV locally;
    only rank zero sends the replicated layer output back to the coordinator.

    The executable cell represents a contiguous range of *intermediate*
    Llama-style decoder layers. Embeddings and final norm/logit heads remain
    outside the cell. CPU/Gloo is the reference path; CUDA or ROCm PyTorch
    builds can execute the same ABI through NCCL/RCCL.
    """

    def __init__(self, spec: StageModelSpec, cell: TensorParallelCellSpec) -> None:
        if spec.first or spec.last:
            raise ValueError("the cell prototype currently supports intermediate stages only")
        root = Path(cell.fixture).resolve()
        manifest = _read_manifest(root)
        if manifest["worldSize"] != cell.world_size:
            raise ValueError(
                f"fixture worldSize {manifest['worldSize']} does not match requested "
                f"cell size {cell.world_size}"
            )
        layer_count = len(manifest["layers"])
        if spec.layer_end - spec.layer_start != layer_count:
            raise ValueError(
                f"stage range contains {spec.layer_end - spec.layer_start} layers, "
                f"but the fixture contains {layer_count}"
            )

        self.spec = spec
        self.cell = cell
        self.layer_count = layer_count
        self.hidden_size = int(manifest["layers"][0]["hiddenSize"])
        rank_zero_backend = rank_backend(
            cell.backend, cell.rank_devices or (), cell.compute_dtype, 0
        )
        rank_zero_backend.validate_fixture_dtype(manifest["dtype"])
        self.dtype = rank_zero_backend.torch_dtype
        self.loader = (
            "tensor-parallel-cell-safetensors-gloo"
            if cell.backend == "gloo" and cell.compute_dtype == "float32"
            else (
                "tensor-parallel-cell-safetensors-"
                f"{cell.backend}-{cell.compute_dtype}"
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
            adapter="llama-tensor-parallel-safetensors",
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
            device_kinds=("cpu",) if cell.backend == "gloo" else ("gpu",),
            compute_apis=("torch", cell.backend),
            weight_dtypes=(cell.compute_dtype,),
            features=(
                "layer-range",
                "rank-local-kv",
                "rollback",
                "tensor-parallel-cell",
                "exact-request-fork-safe-copy",
                "unequal-tensor-parallel" if unequal else "equal-tensor-parallel",
            ),
        )
        self.parameter_bytes = 0
        self.member_reports: tuple[CellMemberReport, ...] = ()
        self.member_memory_reports: dict[int, dict[str, int | str]] = {}
        self.active_requests: set[int] = set()
        self.tokens_seen: dict[int, int] = {}
        self.member_cache_shapes: dict[int, tuple[tuple[int, ...], ...]] = {}
        self.member_layer_cache_shapes: dict[
            int, tuple[tuple[tuple[int, ...], ...], ...]
        ] = {}
        self.member_fork_reports: dict[int, dict[str, Any]] = {}
        self.member_promotion_reports: dict[int, dict[str, Any]] = {}
        self._cache_element_bytes = torch.empty((), dtype=self.dtype).element_size()
        self._closed = False
        self._failed = False
        self._operation = 0
        self._context = multiprocessing.get_context("spawn")
        self._commands = [self._context.Queue() for _ in range(cell.world_size)]
        self._responses = self._context.Queue()
        port = _reserve_loopback_port()
        self._processes = [
            self._context.Process(
                target=_cell_member_main,
                args=(
                    rank,
                    cell.world_size,
                    port,
                    str(root),
                    cell.backend,
                    cell.rank_devices,
                    cell.compute_dtype,
                    spec.threads,
                    cell.operation_timeout_seconds,
                    self._commands[rank],
                    self._responses,
                ),
                name=f"gdlp-cell-layer-{spec.layer_start}-rank-{rank}",
            )
            for rank in range(cell.world_size)
        ]
        for process in self._processes:
            process.start()
        try:
            ready = self._collect(0, "ready", cell.world_size)
            reports = tuple(
                CellMemberReport(
                    rank=int(payload["rank"]),
                    shard_file=str(payload["shardFile"]),
                    parameter_bytes=int(payload["parameterBytes"]),
                    tensor_count=int(payload["tensorCount"]),
                    device=str(payload["device"]),
                    compute_dtype=str(payload["computeDtype"]),
                    collective_backend=str(payload["collectiveBackend"]),
                    allocated_bytes=int(payload["allocatedBytes"]),
                    reserved_bytes=int(payload["reservedBytes"]),
                    peak_allocated_bytes=int(payload["peakAllocatedBytes"]),
                )
                for _, payload in sorted(ready, key=lambda item: item[0])
            )
            if tuple(report.rank for report in reports) != tuple(range(cell.world_size)):
                raise RuntimeError("cell startup returned duplicate or missing ranks")
            expected_fixed = manifest.get("rankFixedBytes")
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
        except BaseException:
            self._terminate()
            raise

    def begin(self, request_id: int) -> None:
        self._require_open()
        if request_id in self.active_requests:
            raise ValueError(f"request {request_id} is already active")
        self._request("begin", {"requestId": request_id})
        self.active_requests.add(request_id)
        self.tokens_seen[request_id] = 0
        self.member_cache_shapes.pop(request_id, None)
        self.member_layer_cache_shapes.pop(request_id, None)

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
                f"cannot truncate request {request_id} from {current} to {token_count} tokens"
            )
        reports = self._request(
            "truncate", {"requestId": request_id, "tokenCount": token_count}
        )
        self.tokens_seen[request_id] = token_count
        self._record_cache_shapes(request_id, reports)

    def sequence_length(self, request_id: int) -> int:
        self._require_active(request_id)
        return self.tokens_seen[request_id]

    def request_cache_bytes(self, request_id: int) -> int:
        """Return exact rank-local KV storage bytes summed across the cell."""

        self._require_active(request_id)
        return sum(self._request_rank_cache_bytes(request_id))

    def project_request_cache_bytes(
        self,
        request_id: int,
        additional_tokens: int,
    ) -> int:
        """Conservatively project cell-wide KV storage before another wave."""

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
        """Copy every member's parent KV before publishing an independent child."""

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
        try:
            reports = self._request(
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
            member_layers, validated = self._validated_fork_reports(
                reports,
                child_request_id=child_request_id,
                parent_request_id=parent_request_id,
                expected_tokens=parent_tokens,
                expected_rank_bytes=expected_rank_bytes,
            )
        except BaseException:
            self._failed = True
            raise

        self.active_requests.add(child_request_id)
        self.tokens_seen[child_request_id] = parent_tokens
        self.member_layer_cache_shapes[child_request_id] = member_layers
        self.member_cache_shapes[child_request_id] = tuple(
            layers[-1] for layers in member_layers
        )
        self.member_fork_reports = validated
        return copied_bytes

    def promote_request(self, parent_request_id: int, child_request_id: int) -> None:
        """Move the selected child's member-local KV references onto the parent."""

        if child_request_id == parent_request_id:
            raise ValueError("promote parent and child request IDs must differ")
        self._require_active(parent_request_id)
        self._require_active(child_request_id)
        parent_tokens = self.tokens_seen[parent_request_id]
        child_tokens = self.tokens_seen[child_request_id]
        child_rank_bytes = self._request_rank_cache_bytes(child_request_id)
        try:
            reports = self._request(
                "promote",
                {
                    "parentRequestId": parent_request_id,
                    "childRequestId": child_request_id,
                    "expectedParentTokens": parent_tokens,
                    "expectedChildTokens": child_tokens,
                    "expectedChildRankCacheBytes": list(child_rank_bytes),
                },
            )
            member_layers, validated = self._validated_promotion_reports(
                reports,
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
        self.active_requests.remove(child_request_id)
        self.tokens_seen.pop(child_request_id, None)
        self.member_layer_cache_shapes.pop(child_request_id, None)
        self.member_cache_shapes.pop(child_request_id, None)
        self.member_promotion_reports = validated

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
        if not hidden.is_floating_point():
            raise TypeError("hidden state must be floating point")
        if hidden.device.type != "cpu":
            raise ValueError("the Gloo cell prototype currently accepts CPU tensors only")
        reports = self._request(
            "forward",
            {
                "requestId": request_id,
                "hidden": hidden.detach().to(dtype=self.dtype).contiguous(),
                "activationShape": list(hidden.shape),
            },
        )
        self.tokens_seen[request_id] += int(hidden.shape[1])
        self._record_cache_shapes(request_id, reports)
        rank_zero = next((payload for rank, payload in reports if rank == 0), None)
        if rank_zero is None or not isinstance(rank_zero.get("output"), torch.Tensor):
            raise RuntimeError("rank zero did not return the logical stage output")
        # This is an intermediate stage, so it never owns a token head.
        return rank_zero["output"], None

    def close(self) -> None:
        if self._closed:
            return
        try:
            if all(process.is_alive() for process in self._processes):
                self._request("shutdown", {})
        except BaseException:
            pass
        finally:
            for process in self._processes:
                process.join(timeout=2.0)
            self._terminate()
            self.active_requests.clear()
            self.tokens_seen.clear()
            self.member_cache_shapes.clear()
            self.member_layer_cache_shapes.clear()
            self._closed = True

    def _request(self, kind: str, payload: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
        self._require_open()
        self._operation += 1
        operation = self._operation
        for rank, commands in enumerate(self._commands):
            rank_payload = payload
            if kind == "forward" and rank != 0:
                rank_payload = {name: value for name, value in payload.items() if name != "hidden"}
            commands.put((operation, kind, rank_payload))
        return self._collect(operation, kind, self.cell.world_size)

    def _collect(
        self,
        operation: int,
        kind: str,
        count: int,
    ) -> list[tuple[int, dict[str, Any]]]:
        deadline = time.monotonic() + self.cell.operation_timeout_seconds
        values: list[tuple[int, dict[str, Any]]] = []
        while len(values) < count:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                dead = [process.name for process in self._processes if not process.is_alive()]
                raise TimeoutError(
                    f"cell operation {kind!r} timed out; dead members={dead}"
                )
            try:
                response = self._responses.get(timeout=remaining)
            except queue.Empty as error:
                raise TimeoutError(f"cell operation {kind!r} timed out") from error
            response_operation, response_kind, rank, status, payload = response
            if response_operation != operation or response_kind != kind:
                raise RuntimeError(
                    f"out-of-order cell response for operation {response_operation} "
                    f"({response_kind}), expected {operation} ({kind})"
                )
            if status == "error":
                raise RuntimeError(f"cell member failure on rank {rank}: {payload}")
            values.append((int(rank), payload))
        return values

    def _request_rank_cache_bytes(self, request_id: int) -> tuple[int, ...]:
        member_layers = self.member_layer_cache_shapes.get(request_id)
        if member_layers is None:
            return (0,) * self.cell.world_size
        if len(member_layers) != self.cell.world_size:
            raise RuntimeError("cell cache inventory does not cover every rank")
        return tuple(
            _cache_shapes_storage_bytes(
                layers,
                element_bytes=self._cache_element_bytes,
            )
            for layers in member_layers
        )

    def _validated_fork_reports(
        self,
        reports: Sequence[tuple[int, Mapping[str, Any]]],
        *,
        child_request_id: int,
        parent_request_id: int,
        expected_tokens: int,
        expected_rank_bytes: tuple[int, ...],
    ) -> tuple[
        tuple[tuple[tuple[int, ...], ...], ...],
        dict[int, dict[str, Any]],
    ]:
        ordered = _ordered_member_reports(reports, self.cell.world_size)
        member_layers: list[tuple[tuple[int, ...], ...]] = []
        validated: dict[int, dict[str, Any]] = {}
        for rank, payload in ordered:
            cache_bytes = _nonnegative_integer(payload.get("cacheBytes"), "cacheBytes")
            copied_bytes = _nonnegative_integer(
                payload.get("copiedCacheBytes"), "copiedCacheBytes"
            )
            if (
                payload.get("childRequestId") != child_request_id
                or payload.get("parentRequestId") != parent_request_id
                or payload.get("tokensSeen") != expected_tokens
                or payload.get("aliasFree") is not True
                or cache_bytes != expected_rank_bytes[rank]
                or copied_bytes != expected_rank_bytes[rank]
            ):
                raise RuntimeError("cell fork response identity is inconsistent")
            shapes = _validated_member_cache_shapes(
                payload.get("cacheShapes"),
                layer_count=self.layer_count,
                expected_sequence_length=expected_tokens,
            )
            if cache_bytes != _cache_shapes_storage_bytes(
                shapes,
                element_bytes=self._cache_element_bytes,
            ):
                raise RuntimeError("cell fork cache bytes disagree with its shapes")
            memory = payload.get("memory")
            if isinstance(memory, dict):
                self.member_memory_reports[rank] = dict(memory)
            member_layers.append(shapes)
            validated[rank] = {
                "rank": rank,
                "childRequestId": child_request_id,
                "parentRequestId": parent_request_id,
                "tokensSeen": expected_tokens,
                "copiedCacheBytes": copied_bytes,
                "cacheBytes": cache_bytes,
                "aliasFree": True,
            }
        return tuple(member_layers), validated

    def _validated_promotion_reports(
        self,
        reports: Sequence[tuple[int, Mapping[str, Any]]],
        *,
        parent_request_id: int,
        child_request_id: int,
        expected_tokens: int,
        expected_rank_bytes: tuple[int, ...],
    ) -> tuple[
        tuple[tuple[tuple[int, ...], ...], ...],
        dict[int, dict[str, Any]],
    ]:
        ordered = _ordered_member_reports(reports, self.cell.world_size)
        member_layers: list[tuple[tuple[int, ...], ...]] = []
        validated: dict[int, dict[str, Any]] = {}
        for rank, payload in ordered:
            cache_bytes = _nonnegative_integer(payload.get("cacheBytes"), "cacheBytes")
            if (
                payload.get("parentRequestId") != parent_request_id
                or payload.get("childRequestId") != child_request_id
                or payload.get("tokensSeen") != expected_tokens
                or payload.get("movedWithoutCopy") is not True
                or cache_bytes != expected_rank_bytes[rank]
            ):
                raise RuntimeError("cell promotion response identity is inconsistent")
            shapes = _validated_member_cache_shapes(
                payload.get("cacheShapes"),
                layer_count=self.layer_count,
                expected_sequence_length=expected_tokens,
            )
            if cache_bytes != _cache_shapes_storage_bytes(
                shapes,
                element_bytes=self._cache_element_bytes,
            ):
                raise RuntimeError("cell promotion cache bytes disagree with its shapes")
            memory = payload.get("memory")
            if isinstance(memory, dict):
                self.member_memory_reports[rank] = dict(memory)
            member_layers.append(shapes)
            validated[rank] = {
                "rank": rank,
                "parentRequestId": parent_request_id,
                "childRequestId": child_request_id,
                "tokensSeen": expected_tokens,
                "cacheBytes": cache_bytes,
                "movedWithoutCopy": True,
            }
        return tuple(member_layers), validated

    def _record_cache_shapes(
        self,
        request_id: int,
        reports: list[tuple[int, dict[str, Any]]],
    ) -> None:
        ordered = _ordered_member_reports(reports, self.cell.world_size)
        member_layers_list: list[tuple[tuple[int, ...], ...]] = []
        for rank, payload in ordered:
            shapes = _validated_member_cache_shapes(
                payload.get("cacheShapes"),
                layer_count=self.layer_count,
                expected_sequence_length=self.tokens_seen[request_id],
            )
            cache_bytes = _nonnegative_integer(payload.get("cacheBytes"), "cacheBytes")
            if cache_bytes != _cache_shapes_storage_bytes(
                shapes,
                element_bytes=self._cache_element_bytes,
            ):
                raise RuntimeError(
                    f"cell rank {rank} cache bytes disagree with its shapes"
                )
            member_layers_list.append(shapes)
        member_layers = tuple(member_layers_list)
        self.member_layer_cache_shapes[request_id] = member_layers
        # Preserve the /1 observation surface for existing callers. For a
        # multi-layer stage this reports the final local layer, while the
        # lossless nested view above reports every layer.
        self.member_cache_shapes[request_id] = tuple(
            layers[-1] for layers in member_layers
        )
        for rank, payload in ordered:
            memory = payload.get("memory")
            if isinstance(memory, dict):
                self.member_memory_reports[rank] = dict(memory)

    def _require_active(self, request_id: int) -> None:
        self._require_open()
        if request_id not in self.active_requests:
            raise ValueError(f"request {request_id} has not received BEGIN")

    def _require_open(self) -> None:
        if self._closed or self._failed:
            raise RuntimeError("tensor-parallel cell is not usable")

    def _terminate(self) -> None:
        for process in self._processes:
            if process.is_alive():
                process.terminate()
        for process in self._processes:
            process.join(timeout=2.0)

    def __enter__(self) -> TensorParallelCellStageRunner:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def write_llama_layer_cell_fixture(
    root: str | Path,
    dense: Mapping[str, torch.Tensor],
    *,
    world_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    rms_norm_epsilon: float = 1e-6,
    rope_theta: float = 10_000.0,
    rank_weights: Sequence[float] | None = None,
) -> Path:
    """Offline compiler for one dense Llama layer into rank-local fixtures.

    This helper is intentionally architecture-specific.  It is useful both for
    executable fixtures and as the seam where a future Hugging Face/GGUF
    adapter can stream checkpoint slices directly without constructing the
    dense mapping in memory.
    """

    if set(dense) != set(_TENSOR_NAMES):
        missing = sorted(set(_TENSOR_NAMES) - set(dense))
        extra = sorted(set(dense) - set(_TENSOR_NAMES))
        raise ValueError(f"invalid dense layer tensors; missing={missing}, extra={extra}")
    if not isinstance(world_size, int) or isinstance(world_size, bool) or world_size < 2:
        raise ValueError("fixture world_size must be at least two")
    normalized_rank_weights = _normalize_rank_weights(world_size, rank_weights)
    for name, value in (
        ("rms_norm_epsilon", rms_norm_epsilon),
        ("rope_theta", rope_theta),
    ):
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or value <= 0
        ):
            raise ValueError(f"{name} must be finite and positive")
    query = dense["query"]
    if query.ndim != 2:
        raise ValueError("query weight must be two-dimensional")
    if query.dtype != torch.float32 or any(
        value.dtype != torch.float32 for value in dense.values()
    ):
        raise ValueError("the Gloo cell fixture currently requires FP32 tensors")
    hidden_size = int(query.shape[1])
    intermediate_size = int(dense["gate"].shape[0])
    if dense["input_norm"].shape != (hidden_size,) or dense[
        "post_attention_norm"
    ].shape != (hidden_size,):
        raise ValueError("norm weights must match hidden_size")
    expected_shapes = {
        "query": (num_attention_heads * head_dim, hidden_size),
        "key": (num_key_value_heads * head_dim, hidden_size),
        "value": (num_key_value_heads * head_dim, hidden_size),
        "output": (hidden_size, num_attention_heads * head_dim),
        "gate": (intermediate_size, hidden_size),
        "up": (intermediate_size, hidden_size),
        "down": (hidden_size, intermediate_size),
    }
    actual_shapes = {name: tuple(dense[name].shape) for name in expected_shapes}
    if actual_shapes != expected_shapes:
        raise ValueError(
            f"dense layer shapes {actual_shapes} do not match {expected_shapes}"
        )

    destination = Path(root).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    shard_files: list[str] = []
    for rank in range(world_size):
        local_query, local_key, local_value, local_output, _ = shard_llama_attention(
            dense["query"],
            dense["key"],
            dense["value"],
            dense["output"],
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            rank=rank,
            world_size=world_size,
            rank_weights=normalized_rank_weights,
        )
        local_gate, _, _ = shard_column_linear(
            dense["gate"], None, rank, world_size, normalized_rank_weights
        )
        local_up, _, _ = shard_column_linear(
            dense["up"], None, rank, world_size, normalized_rank_weights
        )
        local_down, _ = shard_row_linear(
            dense["down"], rank, world_size, normalized_rank_weights
        )
        shard_file = f"rank-{rank:03d}.safetensors"
        save_file(
            {
                "input_norm": dense["input_norm"].detach().cpu().contiguous(),
                "post_attention_norm": dense["post_attention_norm"].detach()
                .cpu()
                .contiguous(),
                "query": local_query.detach().cpu().contiguous(),
                "key": local_key.detach().cpu().contiguous(),
                "value": local_value.detach().cpu().contiguous(),
                "output": local_output.detach().cpu().contiguous(),
                "gate": local_gate.detach().cpu().contiguous(),
                "up": local_up.detach().cpu().contiguous(),
                "down": local_down.detach().cpu().contiguous(),
            },
            destination / shard_file,
        )
        shard_files.append(shard_file)

    dtype = str(query.dtype).removeprefix("torch.")
    manifest = {
        "schema": _SCHEMA_V1,
        "worldSize": world_size,
        "hiddenSize": hidden_size,
        "intermediateSize": intermediate_size,
        "numAttentionHeads": num_attention_heads,
        "numKeyValueHeads": num_key_value_heads,
        "headDim": head_dim,
        "rmsNormEpsilon": rms_norm_epsilon,
        "ropeTheta": rope_theta,
        "dtype": dtype,
        "shards": shard_files,
    }
    if normalized_rank_weights is not None:
        manifest["rankWeights"] = list(normalized_rank_weights)
    (destination / "cell.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def write_llama_stage_cell_fixture(
    root: str | Path,
    layers: Sequence[Mapping[str, torch.Tensor]],
    *,
    world_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    rms_norm_epsilon: float = 1e-6,
    rope_theta: float = 10_000.0,
    rank_weights: Sequence[float] | None = None,
) -> Path:
    """Compile two or more dense Llama layers into a version-2 cell stage.

    The resulting fixture has one file per rank, not one file per layer. Each
    member can therefore start by opening a single SafeTensors file containing
    only its local slices for the entire contiguous stage.
    """

    if isinstance(layers, (str, bytes, bytearray)):
        raise TypeError("layers must be a sequence of dense tensor mappings")
    dense_layers = tuple(layers)
    if len(dense_layers) < 2:
        raise ValueError("a version-2 cell stage fixture requires at least two layers")
    if not isinstance(world_size, int) or isinstance(world_size, bool) or world_size < 2:
        raise ValueError("fixture world_size must be at least two")
    normalized_rank_weights = _normalize_rank_weights(world_size, rank_weights)
    for name, value in (
        ("rms_norm_epsilon", rms_norm_epsilon),
        ("rope_theta", rope_theta),
    ):
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or value <= 0
        ):
            raise ValueError(f"{name} must be finite and positive")

    layer_manifests: list[dict[str, Any]] = []
    hidden_size: int | None = None
    for index, dense in enumerate(dense_layers):
        metadata = _validate_dense_layer(
            dense,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
        )
        if hidden_size is None:
            hidden_size = metadata["hiddenSize"]
        elif metadata["hiddenSize"] != hidden_size:
            raise ValueError("all layers in one logical stage must share hidden_size")
        layer_manifests.append(
            {
                **metadata,
                "index": index,
                "rmsNormEpsilon": rms_norm_epsilon,
                "ropeTheta": rope_theta,
            }
        )

    destination = Path(root).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    shard_files: list[str] = []
    shard_digests: list[str] = []
    rank_fixed_bytes: list[int] = []
    rank_kv_bytes_per_token: list[int] = []
    for rank in range(world_size):
        rank_tensors: dict[str, torch.Tensor] = {}
        for layer_index, dense in enumerate(dense_layers):
            local = _shard_dense_layer(
                dense,
                rank=rank,
                world_size=world_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                rank_weights=normalized_rank_weights,
            )
            for name, tensor in local.items():
                rank_tensors[f"layers.{layer_index}.{name}"] = tensor
        shard_file = f"rank-{rank:03d}.safetensors"
        save_file(rank_tensors, destination / shard_file)
        shard_files.append(shard_file)
        shard_digests.append(_sha256_file(destination / shard_file))
        rank_fixed_bytes.append(
            sum(tensor.numel() * tensor.element_size() for tensor in rank_tensors.values())
        )
        rank_kv_bytes_per_token.append(
            sum(
                2
                * llama_attention_shard_plan(
                    int(layer["hiddenSize"]),
                    int(layer["numAttentionHeads"]),
                    int(layer["numKeyValueHeads"]),
                    int(layer["headDim"]),
                    rank,
                    world_size,
                    normalized_rank_weights,
                ).key_value_heads.size
                * int(layer["headDim"])
                * 4
                for layer in layer_manifests
            )
        )

    manifest = {
        "schema": _SCHEMA_V2,
        "worldSize": world_size,
        "layerCount": len(dense_layers),
        "dtype": "float32",
        "layers": layer_manifests,
        "shards": shard_files,
        "shardSha256": shard_digests,
        "rankFixedBytes": rank_fixed_bytes,
        "rankKvBytesPerToken": rank_kv_bytes_per_token,
    }
    if normalized_rank_weights is not None:
        manifest["rankWeights"] = list(normalized_rank_weights)
    (destination / "cell.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def _validate_dense_layer(
    dense: Mapping[str, torch.Tensor],
    *,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
) -> dict[str, int]:
    if set(dense) != set(_TENSOR_NAMES):
        missing = sorted(set(_TENSOR_NAMES) - set(dense))
        extra = sorted(set(dense) - set(_TENSOR_NAMES))
        raise ValueError(f"invalid dense layer tensors; missing={missing}, extra={extra}")
    if any(
        not isinstance(value, torch.Tensor) or value.dtype != torch.float32
        for value in dense.values()
    ):
        raise ValueError("the Gloo cell fixture currently requires FP32 tensors")
    query = dense["query"]
    if query.ndim != 2:
        raise ValueError("query weight must be two-dimensional")
    hidden_size = int(query.shape[1])
    intermediate_size = int(dense["gate"].shape[0])
    expected = {
        "input_norm": (hidden_size,),
        "post_attention_norm": (hidden_size,),
        "query": (num_attention_heads * head_dim, hidden_size),
        "key": (num_key_value_heads * head_dim, hidden_size),
        "value": (num_key_value_heads * head_dim, hidden_size),
        "output": (hidden_size, num_attention_heads * head_dim),
        "gate": (intermediate_size, hidden_size),
        "up": (intermediate_size, hidden_size),
        "down": (hidden_size, intermediate_size),
    }
    actual = {name: tuple(dense[name].shape) for name in expected}
    if actual != expected:
        raise ValueError(f"dense layer shapes {actual} do not match {expected}")
    # Validate GQA topology and the minimum KV head count for all members later
    # when this metadata is checked with its concrete world size.
    if num_attention_heads % num_key_value_heads != 0:
        raise ValueError("attention heads must be divisible by key/value heads")
    if num_attention_heads * head_dim != hidden_size:
        raise ValueError("attention heads times head_dim must equal hidden_size")
    return {
        "hiddenSize": hidden_size,
        "intermediateSize": intermediate_size,
        "numAttentionHeads": num_attention_heads,
        "numKeyValueHeads": num_key_value_heads,
        "headDim": head_dim,
    }


def _shard_dense_layer(
    dense: Mapping[str, torch.Tensor],
    *,
    rank: int,
    world_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    rank_weights: Sequence[float] | None = None,
) -> dict[str, torch.Tensor]:
    local_query, local_key, local_value, local_output, _ = shard_llama_attention(
        dense["query"],
        dense["key"],
        dense["value"],
        dense["output"],
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        rank=rank,
        world_size=world_size,
        rank_weights=rank_weights,
    )
    local_gate, _, _ = shard_column_linear(
        dense["gate"], None, rank, world_size, rank_weights
    )
    local_up, _, _ = shard_column_linear(
        dense["up"], None, rank, world_size, rank_weights
    )
    local_down, _ = shard_row_linear(
        dense["down"], rank, world_size, rank_weights
    )
    return {
        "input_norm": dense["input_norm"].detach().cpu().contiguous().clone(),
        "post_attention_norm": dense["post_attention_norm"].detach()
        .cpu()
        .contiguous()
        .clone(),
        "query": local_query.detach().cpu().contiguous(),
        "key": local_key.detach().cpu().contiguous(),
        "value": local_value.detach().cpu().contiguous(),
        "output": local_output.detach().cpu().contiguous(),
        "gate": local_gate.detach().cpu().contiguous(),
        "up": local_up.detach().cpu().contiguous(),
        "down": local_down.detach().cpu().contiguous(),
    }


def _cell_member_main(
    rank: int,
    world_size: int,
    port: int,
    fixture: str,
    backend: str,
    rank_devices: Sequence[str],
    compute_dtype: str,
    threads: int,
    operation_timeout_seconds: float,
    commands: Any,
    responses: Any,
) -> None:
    operation = 0
    kind = "ready"
    try:
        torch.set_num_threads(threads)
        execution = rank_backend(backend, rank_devices, compute_dtype, rank)
        execution.activate()
        distributed.init_process_group(
            backend=backend,
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=operation_timeout_seconds),
        )
        root = Path(fixture).resolve()
        manifest = _read_manifest(root)
        execution.validate_fixture_dtype(manifest["dtype"])
        shard_file = _safe_shard_path(root, manifest["shards"][rank])
        weights = execution.load_weights(str(shard_file))
        expected_keys = {
            _tensor_key(manifest, layer_index, name)
            for layer_index in range(len(manifest["layers"]))
            for name in _TENSOR_NAMES
        }
        if set(weights) != expected_keys:
            raise ValueError("rank shard does not contain the exact cell tensor set")
        layer_weights: list[dict[str, torch.Tensor]] = []
        plans: list[Any] = []
        layer_intermediate_sizes: list[tuple[int, ...]] = []
        for layer_index, layer in enumerate(manifest["layers"]):
            local_weights = {
                name: weights[_tensor_key(manifest, layer_index, name)]
                for name in _TENSOR_NAMES
            }
            plan = llama_attention_shard_plan(
                int(layer["hiddenSize"]),
                int(layer["numAttentionHeads"]),
                int(layer["numKeyValueHeads"]),
                int(layer["headDim"]),
                rank,
                world_size,
                manifest["rankWeights"],
            )
            intermediate_sizes = shard_sizes(
                int(layer["intermediateSize"]),
                world_size,
                manifest["rankWeights"],
            )
            _validate_local_weights(
                local_weights,
                layer,
                plan,
                intermediate_sizes[rank],
                expected_dtype=execution.torch_dtype,
            )
            layer_weights.append(local_weights)
            plans.append(plan)
            layer_intermediate_sizes.append(intermediate_sizes)
        parameter_bytes = sum(
            value.nelement() * value.element_size() for value in weights.values()
        )
        memory = execution.memory_report()
        responses.put(
            (
                0,
                "ready",
                rank,
                "ok",
                {
                    "rank": rank,
                    "shardFile": shard_file.name,
                    "parameterBytes": parameter_bytes,
                    "tensorCount": len(weights),
                    **memory,
                },
            )
        )
        caches: dict[int, list[tuple[torch.Tensor, torch.Tensor] | None]] = {}
        tokens_seen: dict[int, int] = {}

        while True:
            operation, kind, payload = commands.get()
            request_id = payload.get("requestId")
            response: dict[str, Any] = {}
            if kind == "begin":
                if request_id in tokens_seen:
                    raise ValueError(f"request {request_id} is already active")
                tokens_seen[request_id] = 0
                caches[request_id] = [None] * len(layer_weights)
            elif kind == "end":
                caches.pop(request_id, None)
                tokens_seen.pop(request_id, None)
            elif kind == "truncate":
                _require_member_request(tokens_seen, request_id)
                token_count = int(payload["tokenCount"])
                if not 0 <= token_count <= tokens_seen[request_id]:
                    raise ValueError("invalid member cache truncation")
                request_caches = caches[request_id]
                for layer_index, cache in enumerate(request_caches):
                    if cache is None:
                        continue
                    key, value = cache
                    request_caches[layer_index] = (
                        key[:, :, :token_count, :].clone(
                            memory_format=torch.contiguous_format
                        ),
                        value[:, :, :token_count, :].clone(
                            memory_format=torch.contiguous_format
                        ),
                    )
                tokens_seen[request_id] = token_count
                response["cacheShapes"] = _cache_shapes(
                    request_caches, plans
                )
                response["cacheBytes"] = _request_cache_storage_bytes(
                    request_caches
                )
            elif kind == "fork":
                child_request_id = _request_identifier(
                    payload.get("childRequestId"), "childRequestId"
                )
                parent_request_id = _request_identifier(
                    payload.get("parentRequestId"), "parentRequestId"
                )
                if child_request_id == parent_request_id:
                    raise ValueError("fork child and parent request IDs must differ")
                _require_member_request(tokens_seen, parent_request_id)
                if child_request_id in tokens_seen or child_request_id in caches:
                    raise ValueError("fork child request is already active")
                expected_tokens = _nonnegative_integer(
                    payload.get("expectedTokens"), "expectedTokens"
                )
                expected_cache_bytes = _nonnegative_integer(
                    payload.get("expectedCacheBytes"), "expectedCacheBytes"
                )
                max_cache_bytes = _nonnegative_integer(
                    payload.get("maxCacheBytes"), "maxCacheBytes"
                )
                expected_rank_bytes = _rank_cache_byte_inventory(
                    payload.get("expectedRankCacheBytes"),
                    world_size=world_size,
                )
                if (
                    tokens_seen[parent_request_id] != expected_tokens
                    or sum(expected_rank_bytes) != expected_cache_bytes
                    or expected_cache_bytes > max_cache_bytes
                ):
                    raise ValueError(
                        "fork preflight identity or byte budget is inconsistent"
                    )
                parent_cache = caches[parent_request_id]
                parent_cache_bytes = _request_cache_storage_bytes(parent_cache)
                if parent_cache_bytes != expected_rank_bytes[rank]:
                    raise ValueError("fork rank cache bytes changed before copying")

                child_cache = _clone_request_cache(parent_cache)
                parent_storage = _request_cache_storage_keys(parent_cache)
                child_storage = _request_cache_storage_keys(child_cache)
                if parent_storage & child_storage:
                    raise RuntimeError("forked cell cache aliases parent tensor storage")
                child_cache_bytes = _request_cache_storage_bytes(child_cache)
                if child_cache_bytes != parent_cache_bytes:
                    raise RuntimeError("forked cell cache changed storage byte size")
                caches[child_request_id] = child_cache
                tokens_seen[child_request_id] = expected_tokens
                response.update(
                    {
                        "childRequestId": child_request_id,
                        "parentRequestId": parent_request_id,
                        "tokensSeen": expected_tokens,
                        "copiedCacheBytes": parent_cache_bytes,
                        "cacheBytes": child_cache_bytes,
                        "cacheShapes": _cache_shapes(child_cache, plans),
                        "aliasFree": True,
                        "memory": execution.memory_report(),
                    }
                )
            elif kind == "promote":
                parent_request_id = _request_identifier(
                    payload.get("parentRequestId"), "parentRequestId"
                )
                child_request_id = _request_identifier(
                    payload.get("childRequestId"), "childRequestId"
                )
                if child_request_id == parent_request_id:
                    raise ValueError("promote parent and child request IDs must differ")
                _require_member_request(tokens_seen, parent_request_id)
                _require_member_request(tokens_seen, child_request_id)
                expected_parent_tokens = _nonnegative_integer(
                    payload.get("expectedParentTokens"), "expectedParentTokens"
                )
                expected_child_tokens = _nonnegative_integer(
                    payload.get("expectedChildTokens"), "expectedChildTokens"
                )
                expected_rank_bytes = _rank_cache_byte_inventory(
                    payload.get("expectedChildRankCacheBytes"),
                    world_size=world_size,
                )
                child_cache = caches[child_request_id]
                child_cache_bytes = _request_cache_storage_bytes(child_cache)
                if (
                    tokens_seen[parent_request_id] != expected_parent_tokens
                    or tokens_seen[child_request_id] != expected_child_tokens
                    or child_cache_bytes != expected_rank_bytes[rank]
                ):
                    raise ValueError("promotion preflight identity is inconsistent")

                child_storage = _request_cache_storage_keys(child_cache)
                moved_cache = caches.pop(child_request_id)
                caches[parent_request_id] = moved_cache
                tokens_seen[parent_request_id] = tokens_seen.pop(child_request_id)
                if _request_cache_storage_keys(caches[parent_request_id]) != child_storage:
                    raise RuntimeError("promotion copied or changed selected KV storage")
                response.update(
                    {
                        "parentRequestId": parent_request_id,
                        "childRequestId": child_request_id,
                        "tokensSeen": expected_child_tokens,
                        "cacheBytes": child_cache_bytes,
                        "cacheShapes": _cache_shapes(
                            caches[parent_request_id], plans
                        ),
                        "movedWithoutCopy": True,
                        "memory": execution.memory_report(),
                    }
                )
            elif kind == "forward":
                _require_member_request(tokens_seen, request_id)
                output = execution.broadcast_activation(
                    payload.get("hidden"), payload.get("activationShape")
                )
                input_tokens = int(output.shape[1])
                request_caches = caches[request_id]
                for layer_index, (local_weights, plan, intermediate_sizes, layer) in enumerate(
                    zip(
                        layer_weights,
                        plans,
                        layer_intermediate_sizes,
                        manifest["layers"],
                    )
                ):
                    output, present = llama_decoder_layer_tensor_parallel(
                        output,
                        local_weights["input_norm"],
                        local_weights["post_attention_norm"],
                        local_weights["query"],
                        local_weights["key"],
                        local_weights["value"],
                        local_weights["output"],
                        plan,
                        local_weights["gate"],
                        local_weights["up"],
                        local_weights["down"],
                        intermediate_sizes,
                        past_key_value=request_caches[layer_index],
                        rms_norm_epsilon=float(layer["rmsNormEpsilon"]),
                        rope_theta=float(layer["ropeTheta"]),
                    )
                    request_caches[layer_index] = present
                tokens_seen[request_id] += input_tokens
                response["cacheShapes"] = _cache_shapes(request_caches, plans)
                response["cacheBytes"] = _request_cache_storage_bytes(
                    request_caches
                )
                response["memory"] = execution.memory_report()
                if rank == 0:
                    response["output"] = output.to(device="cpu", dtype=torch.float32)
            elif kind == "shutdown":
                responses.put((operation, kind, rank, "ok", response))
                break
            else:
                raise ValueError(f"unsupported cell operation {kind!r}")
            responses.put((operation, kind, rank, "ok", response))
    except BaseException as error:
        try:
            responses.put(
                (
                    operation,
                    kind,
                    rank,
                    "error",
                    f"{type(error).__name__}: {error}",
                )
            )
        except BaseException:
            pass
        raise
    finally:
        if distributed.is_initialized():
            distributed.destroy_process_group()


def _read_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / "cell.json"
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"cell manifest does not exist: {manifest_path}") from error
    if not isinstance(value, dict) or value.get("schema") not in (
        _SCHEMA_V1,
        _SCHEMA_V2,
    ):
        raise ValueError(
            f"cell manifest must use schema {_SCHEMA_V1} or {_SCHEMA_V2}"
        )
    world_size = _positive_manifest_integer(value, "worldSize")
    rank_weights = _normalize_rank_weights(world_size, value.get("rankWeights"))
    effective_rank_weights = rank_weights or tuple(
        1.0 for _ in range(world_size)
    )
    if value["schema"] == _SCHEMA_V1:
        layers = [
            {
                name: value.get(name)
                for name in (
                    "hiddenSize",
                    "intermediateSize",
                    "numAttentionHeads",
                    "numKeyValueHeads",
                    "headDim",
                    "rmsNormEpsilon",
                    "ropeTheta",
                )
            }
        ]
        tensorLayout = "legacy-unprefixed"
    else:
        layer_count = _positive_manifest_integer(value, "layerCount")
        if layer_count < 2:
            raise ValueError("version-2 cell manifest layerCount must be at least two")
        raw_layers = value.get("layers")
        if not isinstance(raw_layers, list) or len(raw_layers) != layer_count:
            raise ValueError("version-2 cell manifest layers must match layerCount")
        layers = raw_layers
        tensorLayout = "layer-prefixed"
    normalized_layers: list[dict[str, Any]] = []
    for index, layer in enumerate(layers):
        if not isinstance(layer, dict):
            raise ValueError(f"cell manifest layer {index} must be an object")
        if value["schema"] == _SCHEMA_V2 and layer.get("index") != index:
            raise ValueError(
                f"version-2 cell manifest layer index must be {index}"
            )
        normalized_layers.append(
            _validate_manifest_layer(
                layer, world_size, index, effective_rank_weights
            )
        )
    hidden_sizes = {layer["hiddenSize"] for layer in normalized_layers}
    if len(hidden_sizes) != 1:
        raise ValueError("all layers in one logical cell stage must share hiddenSize")
    if value["worldSize"] < 2:
        raise ValueError("cell manifest worldSize must be at least two")
    shards = value.get("shards")
    if (
        not isinstance(shards, list)
        or len(shards) != value["worldSize"]
        or any(not isinstance(item, str) or not item for item in shards)
        or len(set(shards)) != len(shards)
    ):
        raise ValueError("cell manifest must declare one shard file per member")
    if value.get("dtype") not in ("float32", "float16", "bfloat16"):
        raise ValueError("cell manifest dtype must be float32, float16 or bfloat16")
    shard_digests = value.get("shardSha256")
    if value["schema"] == _SCHEMA_V2 and (
        not isinstance(shard_digests, list)
        or len(shard_digests) != world_size
        or any(
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for digest in shard_digests
        )
    ):
        raise ValueError("version-2 cell manifest must contain one SHA-256 per shard")
    rank_fixed_bytes = value.get("rankFixedBytes")
    rank_kv_bytes_per_token = value.get("rankKvBytesPerToken")
    if value["schema"] == _SCHEMA_V2:
        for name, profile in (
            ("rankFixedBytes", rank_fixed_bytes),
            ("rankKvBytesPerToken", rank_kv_bytes_per_token),
        ):
            if (
                not isinstance(profile, list)
                or len(profile) != world_size
                or any(
                    not isinstance(item, int)
                    or isinstance(item, bool)
                    or item < 0
                    for item in profile
                )
            ):
                raise ValueError(
                    f"version-2 cell manifest must contain one {name} value per shard"
                )
    return {
        "schema": value["schema"],
        "tensorLayout": tensorLayout,
        "worldSize": world_size,
        "dtype": value["dtype"],
        "layers": normalized_layers,
        "shards": shards,
        "shardSha256": shard_digests,
        "rankFixedBytes": rank_fixed_bytes,
        "rankKvBytesPerToken": rank_kv_bytes_per_token,
        "rankWeights": effective_rank_weights,
    }


def _validate_manifest_layer(
    layer: Mapping[str, Any],
    world_size: int,
    index: int,
    rank_weights: Sequence[float],
) -> dict[str, Any]:
    normalized = {
        name: _positive_manifest_integer(layer, name, prefix=f"layer {index} ")
        for name in (
            "hiddenSize",
            "intermediateSize",
            "numAttentionHeads",
            "numKeyValueHeads",
            "headDim",
        )
    }
    for name in ("rmsNormEpsilon", "ropeTheta"):
        field = layer.get(name)
        if not isinstance(field, (int, float)) or isinstance(field, bool):
            raise ValueError(f"cell manifest layer {index} {name} must be numeric")
        if not math.isfinite(float(field)) or field <= 0:
            raise ValueError(
                f"cell manifest layer {index} {name} must be finite and positive"
            )
        normalized[name] = float(field)
    # This validates GQA divisibility and that every member receives KV heads.
    llama_attention_shard_plan(
        normalized["hiddenSize"],
        normalized["numAttentionHeads"],
        normalized["numKeyValueHeads"],
        normalized["headDim"],
        0,
        world_size,
        rank_weights,
    )
    return normalized


def _normalize_rank_weights(
    world_size: int,
    rank_weights: object,
) -> tuple[float, ...] | None:
    if rank_weights is None:
        return None
    if (
        not isinstance(rank_weights, Sequence)
        or isinstance(rank_weights, (str, bytes, bytearray))
        or len(rank_weights) != world_size
    ):
        raise ValueError("cell rankWeights must contain one value per member")
    normalized: list[float] = []
    for value in rank_weights:
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or value <= 0
        ):
            raise ValueError("cell rankWeights must be finite positive numbers")
        normalized.append(float(value))
    return tuple(normalized)


def _positive_manifest_integer(
    value: Mapping[str, Any],
    name: str,
    *,
    prefix: str = "",
) -> int:
    field = value.get(name)
    if not isinstance(field, int) or isinstance(field, bool) or field < 1:
        raise ValueError(
            f"cell manifest {prefix}{name} must be a positive integer"
        )
    return field


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_shard_path(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if candidate.parent != root.resolve():
        raise ValueError("cell shard path must be a direct child of the fixture")
    if candidate.suffix != ".safetensors":
        raise ValueError("cell shard must be a SafeTensors file")
    return candidate


def _tensor_key(
    manifest: Mapping[str, Any],
    layer_index: int,
    name: str,
) -> str:
    if manifest["tensorLayout"] == "legacy-unprefixed":
        if layer_index != 0:
            raise ValueError("legacy fixtures can only contain one layer")
        return name
    return f"layers.{layer_index}.{name}"


def _cache_shapes(
    caches: Sequence[tuple[torch.Tensor, torch.Tensor] | None],
    plans: Sequence[Any],
) -> list[list[int]]:
    if len(caches) != len(plans):
        raise ValueError("cache and layer plan counts do not match")
    shapes: list[list[int]] = []
    for cache, plan in zip(caches, plans):
        if cache is None:
            shapes.append(
                [1, int(plan.local_key_value_heads), 0, int(plan.head_dim)]
            )
        else:
            shapes.append(list(cache[0].shape))
    return shapes


def _request_identifier(value: Any, name: str) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not 0 <= value <= (1 << 64) - 1
    ):
        raise ValueError(f"{name} must be an unsigned 64-bit integer")
    return value


def _nonnegative_integer(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _rank_cache_byte_inventory(value: Any, *, world_size: int) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) != world_size:
        raise ValueError("rank cache byte inventory does not cover the process group")
    return tuple(
        _nonnegative_integer(item, "rank cache bytes") for item in value
    )


def _request_cache_storage_keys(
    caches: Sequence[tuple[torch.Tensor, torch.Tensor] | None],
) -> frozenset[tuple[str, int | None, int, int]]:
    keys: set[tuple[str, int | None, int, int]] = set()
    for cache in caches:
        if cache is None:
            continue
        if not isinstance(cache, tuple) or len(cache) != 2:
            raise RuntimeError("cell request cache entry is invalid")
        for tensor in cache:
            if not isinstance(tensor, torch.Tensor):
                raise RuntimeError("cell request cache contains a non-tensor")
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
            raise RuntimeError("cell request cache entry cannot be copied safely")
        key, value = cache
        if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
            raise RuntimeError("cell request cache tensors cannot be copied safely")
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


def _validated_member_cache_shapes(
    value: Any,
    *,
    layer_count: int,
    expected_sequence_length: int,
) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, (list, tuple)) or len(value) != layer_count:
        raise RuntimeError("cell cache shapes do not cover every layer")
    shapes: list[tuple[int, ...]] = []
    for raw_shape in value:
        if not isinstance(raw_shape, (list, tuple)) or len(raw_shape) != 4:
            raise RuntimeError("cell cache shape is invalid")
        if any(
            not isinstance(dimension, int)
            or isinstance(dimension, bool)
            or dimension < 0
            for dimension in raw_shape
        ):
            raise RuntimeError("cell cache shape dimensions are invalid")
        shape = tuple(raw_shape)
        if (
            shape[0] != 1
            or shape[1] < 1
            or shape[2] != expected_sequence_length
            or shape[3] < 1
        ):
            raise RuntimeError("cell cache shape disagrees with request length")
        shapes.append(shape)
    return tuple(shapes)


def _ordered_member_reports(
    reports: Sequence[tuple[int, Mapping[str, Any]]],
    world_size: int,
) -> tuple[tuple[int, Mapping[str, Any]], ...]:
    ordered = tuple(sorted(reports, key=lambda item: item[0]))
    if tuple(rank for rank, _ in ordered) != tuple(range(world_size)):
        raise RuntimeError("cell responses do not cover every rank exactly once")
    if any(not isinstance(payload, Mapping) for _, payload in ordered):
        raise RuntimeError("cell response payload is invalid")
    return ordered


def _validate_local_weights(
    weights: Mapping[str, torch.Tensor],
    manifest: Mapping[str, Any],
    plan: Any,
    local_intermediate_size: int,
    *,
    expected_dtype: torch.dtype = torch.float32,
) -> None:
    hidden_size = int(manifest["hiddenSize"])
    query_features = int(plan.local_query_heads * plan.head_dim)
    kv_features = int(plan.local_key_value_heads * plan.head_dim)
    expected = {
        "input_norm": (hidden_size,),
        "post_attention_norm": (hidden_size,),
        "query": (query_features, hidden_size),
        "key": (kv_features, hidden_size),
        "value": (kv_features, hidden_size),
        "output": (hidden_size, query_features),
        "gate": (local_intermediate_size, hidden_size),
        "up": (local_intermediate_size, hidden_size),
        "down": (hidden_size, local_intermediate_size),
    }
    actual = {name: tuple(weights[name].shape) for name in expected}
    if actual != expected:
        raise ValueError(f"rank-local tensor shapes {actual} do not match {expected}")
    if any(value.dtype != expected_dtype for value in weights.values()):
        raise ValueError(
            f"rank-local cell tensors must all use planned dtype {expected_dtype}"
        )


def _require_member_request(tokens_seen: dict[int, int], request_id: int) -> None:
    if request_id not in tokens_seen:
        raise ValueError(f"request {request_id} has not received BEGIN")


def _reserve_loopback_port() -> int:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])
    finally:
        listener.close()


__all__ = [
    "CellMemberReport",
    "TensorParallelCellSpec",
    "TensorParallelCellStageRunner",
    "write_llama_layer_cell_fixture",
    "write_llama_stage_cell_fixture",
]
