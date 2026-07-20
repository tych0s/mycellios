"""Mathematical intermediate-dimension tiling for one bias-free SwiGLU expert.

This module is deliberately isolated from the production StageRunner.  It
models and executes an adapted expert artifact whose canonical intermediate
dimension is split across independent owners.  Owners fan out in parallel and
their partial hidden-size outputs are reduced in ascending tile-index order.

The result is mathematically equivalent to the original SwiGLU expression,
but floating-point GEMM accumulation changes.  The contract is therefore
``tiled-mathematical`` and never claims bitwise equivalence to a monolithic
checkpoint execution.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import hashlib
import heapq
import math
import re
from typing import Any, Mapping, Sequence

import torch
from torch.nn import functional as F


TILED_EXPERT_SCHEMA = "gdlp-tiled-swiglu-expert/2"
TILED_EXPERT_MODE = "tiled-mathematical"
TILED_EXPERT_REDUCTION_ORDER = "ascending-tile-index"

_TORCH_DTYPE_BY_NAME = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}
_DTYPE_NAME_BY_TORCH = {value: name for name, value in _TORCH_DTYPE_BY_NAME.items()}


def _positive_int(name: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(name: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _finite(name: str, value: object, *, positive: bool) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be finite and {qualifier}") from error
    invalid = number <= 0 if positive else number < 0
    if not math.isfinite(number) or invalid:
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return number


def _identifier(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} cannot be empty")
    return value.strip()


def _sha256_identity(name: str, value: object) -> str:
    identity = _identifier(name, value)
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", identity):
        raise ValueError(f"{name} must be a lowercase SHA-256 identity")
    return identity


def _dtype(name: str, value: object) -> str:
    dtype_name = _identifier(name, value)
    if dtype_name not in _TORCH_DTYPE_BY_NAME:
        raise ValueError(
            f"{name} must be one of {tuple(sorted(_TORCH_DTYPE_BY_NAME))}"
        )
    return dtype_name


def _tensor_dtype_name(tensor: torch.Tensor) -> str:
    try:
        return _DTYPE_NAME_BY_TORCH[tensor.dtype]
    except KeyError as error:
        raise TypeError(f"unsupported tiled expert tensor dtype {tensor.dtype}") from error


def _tensor_version(tensor: torch.Tensor) -> int | None:
    try:
        return int(tensor._version)
    except RuntimeError:
        # Inference tensors deliberately have no version counter.  They remain
        # mutable *inside* inference mode, so ``None`` can never be used as a
        # stable unchanged marker; callers must rehash those tensors.
        return None


def _digest_field(digest: Any, value: str | bytes) -> None:
    payload = value.encode("utf-8") if isinstance(value, str) else value
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _digest_tensor(
    digest: Any,
    name: str,
    tensor: torch.Tensor,
) -> None:
    cpu = tensor.detach().to(device="cpu").contiguous()
    _digest_field(digest, name)
    _digest_field(digest, _tensor_dtype_name(cpu))
    digest.update(cpu.ndim.to_bytes(4, "big"))
    for dimension in cpu.shape:
        digest.update(int(dimension).to_bytes(8, "big"))
    _digest_field(digest, cpu.view(torch.uint8).numpy().tobytes(order="C"))


def _expert_content_identity(
    artifact_identity: str,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"gdlp-tiled-swiglu-expert-content-v1\0")
    _digest_field(digest, artifact_identity)
    for name, tensor in (("gate", gate), ("up", up), ("down", down)):
        _digest_tensor(digest, name, tensor)
    return f"sha256:{digest.hexdigest()}"


def _tile_content_identity(
    *,
    artifact_identity: str,
    expert_content_id: str,
    intermediate_start: int,
    intermediate_end: int,
    full_intermediate_size: int,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"gdlp-tiled-swiglu-tile-content-v1\0")
    _digest_field(digest, artifact_identity)
    _digest_field(digest, expert_content_id)
    for value in (
        intermediate_start,
        intermediate_end,
        full_intermediate_size,
    ):
        digest.update(int(value).to_bytes(8, "big"))
    for name, tensor in (("gate", gate), ("up", up), ("down", down)):
        _digest_tensor(digest, name, tensor)
    return f"sha256:{digest.hexdigest()}"


@dataclass(frozen=True)
class SwiGLUTilingSpec:
    """Sealed tensor geometry used by both the planner and executor."""

    hidden_size: int
    intermediate_size: int
    positions: int
    artifact_identity: str
    expert_content_id: str
    weight_dtype: str
    activation_dtype: str
    accumulation_dtype: str
    weight_element_bytes: int = 4
    activation_element_bytes: int = 4
    intermediate_workspace_copies: int = 3

    def __post_init__(self) -> None:
        for name in ("hidden_size", "intermediate_size", "positions"):
            _positive_int(name, getattr(self, name))
        object.__setattr__(
            self,
            "artifact_identity",
            _sha256_identity("artifact_identity", self.artifact_identity),
        )
        object.__setattr__(
            self,
            "expert_content_id",
            _sha256_identity("expert_content_id", self.expert_content_id),
        )
        for name in ("weight_dtype", "activation_dtype", "accumulation_dtype"):
            object.__setattr__(self, name, _dtype(name, getattr(self, name)))
        if len(
            {
                self.weight_dtype,
                self.activation_dtype,
                self.accumulation_dtype,
            }
        ) != 1:
            raise ValueError(
                "the current tiled executor requires identical weight, activation "
                "and accumulation dtypes"
            )
        for name in ("weight_element_bytes", "activation_element_bytes"):
            _positive_int(name, getattr(self, name))
        expected_weight_bytes = torch.empty(
            (), dtype=_TORCH_DTYPE_BY_NAME[self.weight_dtype]
        ).element_size()
        expected_activation_bytes = torch.empty(
            (), dtype=_TORCH_DTYPE_BY_NAME[self.activation_dtype]
        ).element_size()
        if self.weight_element_bytes != expected_weight_bytes:
            raise ValueError("weight_element_bytes does not match weight_dtype")
        if self.activation_element_bytes != expected_activation_bytes:
            raise ValueError("activation_element_bytes does not match activation_dtype")
        _positive_int(
            "intermediate_workspace_copies",
            self.intermediate_workspace_copies,
        )

    @property
    def accumulation_element_bytes(self) -> int:
        return torch.empty(
            (), dtype=_TORCH_DTYPE_BY_NAME[self.accumulation_dtype]
        ).element_size()

    @property
    def input_activation_bytes(self) -> int:
        return self.positions * self.hidden_size * self.activation_element_bytes

    @property
    def partial_output_bytes(self) -> int:
        return self.input_activation_bytes

    @property
    def activation_round_trip_bytes_per_tile(self) -> int:
        return self.input_activation_bytes + self.partial_output_bytes

    @property
    def weight_bytes_per_intermediate(self) -> int:
        # gate[:, hidden] + up[:, hidden] + down[hidden, :]
        return 3 * self.hidden_size * self.weight_element_bytes

    @property
    def workspace_bytes_per_intermediate(self) -> int:
        return (
            self.intermediate_workspace_copies
            * self.positions
            * self.activation_element_bytes
        )

    @property
    def total_expert_weight_bytes(self) -> int:
        return self.intermediate_size * self.weight_bytes_per_intermediate

    @property
    def estimated_operations_per_intermediate(self) -> int:
        # Three matrix products (gate, up and down), counted as multiply-adds,
        # plus a small deterministic allowance for SiLU and elementwise mul.
        return (
            6 * self.positions * self.hidden_size
            + 6 * self.positions
        )


@dataclass(frozen=True)
class TiledExpertWorkerProfile:
    owner_id: str
    usable_vram_bytes: int
    reserved_vram_bytes: int
    compute_gflops: float
    round_trip_ms: float
    bandwidth_mbps: float
    available: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner_id", _identifier("owner_id", self.owner_id))
        usable = _positive_int("usable_vram_bytes", self.usable_vram_bytes)
        reserved = _nonnegative_int("reserved_vram_bytes", self.reserved_vram_bytes)
        if reserved >= usable:
            raise ValueError("reserved_vram_bytes must be below usable_vram_bytes")
        object.__setattr__(
            self,
            "compute_gflops",
            _finite("compute_gflops", self.compute_gflops, positive=True),
        )
        object.__setattr__(
            self,
            "round_trip_ms",
            _finite("round_trip_ms", self.round_trip_ms, positive=False),
        )
        object.__setattr__(
            self,
            "bandwidth_mbps",
            _finite("bandwidth_mbps", self.bandwidth_mbps, positive=True),
        )
        if not isinstance(self.available, bool):
            raise TypeError("available must be boolean")

    @property
    def bytes_per_ms(self) -> float:
        # Decimal Mbit/s -> bytes/ms.
        return self.bandwidth_mbps * 125.0


@dataclass(frozen=True)
class TiledExpertTile:
    tile_index: int
    owner_id: str
    artifact_identity: str
    expert_content_id: str
    tile_content_id: str
    weight_dtype: str
    intermediate_start: int
    intermediate_end: int
    worker_vram_budget_bytes: int
    worker_reserved_vram_bytes: int
    weight_bytes: int
    input_activation_bytes: int
    partial_output_bytes: int
    workspace_bytes: int
    vram_required_bytes: int
    round_trip_ms: float
    transfer_ms: float
    compute_ms: float
    exposed_ms: float

    def __post_init__(self) -> None:
        _nonnegative_int("tile_index", self.tile_index)
        object.__setattr__(self, "owner_id", _identifier("owner_id", self.owner_id))
        for name in ("artifact_identity", "expert_content_id", "tile_content_id"):
            object.__setattr__(
                self,
                name,
                _sha256_identity(name, getattr(self, name)),
            )
        object.__setattr__(
            self,
            "weight_dtype",
            _dtype("weight_dtype", self.weight_dtype),
        )
        start = _nonnegative_int("intermediate_start", self.intermediate_start)
        end = _positive_int("intermediate_end", self.intermediate_end)
        if end <= start:
            raise ValueError("a tiled expert slice cannot be empty")
        for name in (
            "worker_vram_budget_bytes",
            "weight_bytes",
            "input_activation_bytes",
            "partial_output_bytes",
            "vram_required_bytes",
        ):
            _positive_int(name, getattr(self, name))
        for name in ("worker_reserved_vram_bytes", "workspace_bytes"):
            _nonnegative_int(name, getattr(self, name))
        if self.vram_required_bytes > self.worker_vram_budget_bytes:
            raise MemoryError(
                f"tile {self.tile_index} exceeds owner {self.owner_id!r} VRAM budget"
            )
        for name in ("round_trip_ms", "transfer_ms", "compute_ms", "exposed_ms"):
            _finite(name, getattr(self, name), positive=False)
        if self.transfer_ms + 1e-12 < self.round_trip_ms:
            raise ValueError("tile transfer_ms cannot be below its round-trip latency")
        if not math.isclose(
            self.exposed_ms,
            self.transfer_ms + self.compute_ms,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("tile exposed_ms must equal transfer_ms plus compute_ms")

    @property
    def width(self) -> int:
        return self.intermediate_end - self.intermediate_start

    @property
    def activation_round_trip_bytes(self) -> int:
        return self.input_activation_bytes + self.partial_output_bytes


@dataclass(frozen=True)
class TiledExpertPlan:
    schema: str
    mode: str
    reduction_order: str
    bitwise_equivalent_to_monolithic: bool
    spec: SwiGLUTilingSpec
    tiles: tuple[TiledExpertTile, ...]
    total_weight_bytes: int
    activation_round_trip_bytes: int
    coordinator_vram_budget_bytes: int
    coordinator_reserved_vram_bytes: int
    coordinator_input_activation_bytes: int
    coordinator_partial_output_bytes: int
    coordinator_reduction_output_bytes: int
    coordinator_workspace_bytes: int
    coordinator_vram_required_bytes: int
    coordinator_egress_mbps: float
    coordinator_ingress_mbps: float
    coordinator_egress_bytes: int
    coordinator_ingress_bytes: int
    coordinator_egress_ms: float
    coordinator_ingress_ms: float
    coordinator_rpc_setup_ms_per_owner: float
    coordinator_rpc_setup_ms: float
    coordinator_causal_path_ms: float
    per_owner_parallel_ms: float
    fanout_parallel_ms: float
    serial_owner_sum_ms: float
    reduction_bytes: int
    reduction_gbytes_per_second: float
    reduction_ms: float
    projected_exposed_ms: float

    def __post_init__(self) -> None:
        if self.schema != TILED_EXPERT_SCHEMA:
            raise ValueError("unsupported tiled expert plan schema")
        if self.mode != TILED_EXPERT_MODE:
            raise ValueError("tiled expert mode must be tiled-mathematical")
        if self.reduction_order != TILED_EXPERT_REDUCTION_ORDER:
            raise ValueError("tiled expert reduction order must be canonical")
        if self.bitwise_equivalent_to_monolithic is not False:
            raise ValueError("tiled-mathematical cannot claim monolithic bitwise parity")
        if not isinstance(self.spec, SwiGLUTilingSpec):
            raise TypeError("spec must be SwiGLUTilingSpec")
        if not self.tiles:
            raise ValueError("a tiled expert plan must contain at least one tile")

        cursor = 0
        owners: set[str] = set()
        for expected_index, tile in enumerate(self.tiles):
            if not isinstance(tile, TiledExpertTile):
                raise TypeError("tiles must contain TiledExpertTile values")
            if tile.tile_index != expected_index:
                raise ValueError("tile indices must be contiguous and canonical")
            if tile.intermediate_start != cursor:
                raise ValueError("tile slices must cover the intermediate axis without gaps")
            if tile.owner_id in owners:
                raise ValueError("each owner may hold at most one parallel tile")
            owners.add(tile.owner_id)
            if tile.artifact_identity != self.spec.artifact_identity:
                raise ValueError("tile artifact identity does not match the sealed spec")
            if tile.expert_content_id != self.spec.expert_content_id:
                raise ValueError("tile expert content identity does not match the sealed spec")
            if tile.weight_dtype != self.spec.weight_dtype:
                raise ValueError("tile weight dtype does not match the sealed spec")
            if tile.weight_bytes != tile.width * self.spec.weight_bytes_per_intermediate:
                raise ValueError("tile weight bytes do not match its canonical width")
            if tile.input_activation_bytes != self.spec.input_activation_bytes:
                raise ValueError("tile input activation bytes changed")
            if tile.partial_output_bytes != self.spec.partial_output_bytes:
                raise ValueError("tile partial output bytes changed")
            if (
                tile.workspace_bytes
                != tile.width * self.spec.workspace_bytes_per_intermediate
            ):
                raise ValueError("tile workspace bytes do not match its canonical width")
            expected_vram = (
                tile.worker_reserved_vram_bytes
                + tile.weight_bytes
                + tile.input_activation_bytes
                + tile.partial_output_bytes
                + tile.workspace_bytes
            )
            if tile.vram_required_bytes != expected_vram:
                raise ValueError("tile VRAM byte accounting changed")
            cursor = tile.intermediate_end
        if cursor != self.spec.intermediate_size:
            raise ValueError("tile slices do not cover the full intermediate axis")

        if self.total_weight_bytes != sum(tile.weight_bytes for tile in self.tiles):
            raise ValueError("tiled expert weight byte accounting changed")
        if self.total_weight_bytes != self.spec.total_expert_weight_bytes:
            raise ValueError("tiles do not contain the full adapted expert weights")
        if self.activation_round_trip_bytes != sum(
            tile.activation_round_trip_bytes for tile in self.tiles
        ):
            raise ValueError("activation round-trip byte accounting changed")

        coordinator_budget = _positive_int(
            "coordinator_vram_budget_bytes",
            self.coordinator_vram_budget_bytes,
        )
        coordinator_reserved = _nonnegative_int(
            "coordinator_reserved_vram_bytes",
            self.coordinator_reserved_vram_bytes,
        )
        if self.coordinator_input_activation_bytes != self.spec.input_activation_bytes:
            raise ValueError("coordinator input activation byte accounting changed")
        expected_partials = len(self.tiles) * self.spec.partial_output_bytes
        if self.coordinator_partial_output_bytes != expected_partials:
            raise ValueError("coordinator partial output byte accounting changed")
        expected_reduction_output = (
            self.spec.positions
            * self.spec.hidden_size
            * self.spec.accumulation_element_bytes
        )
        if self.coordinator_reduction_output_bytes != expected_reduction_output:
            raise ValueError("coordinator reduction output byte accounting changed")
        if self.coordinator_workspace_bytes != 0:
            raise ValueError("the current in-place reducer has zero extra workspace")
        expected_coordinator_required = (
            coordinator_reserved
            + self.coordinator_input_activation_bytes
            + self.coordinator_partial_output_bytes
            + self.coordinator_reduction_output_bytes
            + self.coordinator_workspace_bytes
        )
        if self.coordinator_vram_required_bytes != expected_coordinator_required:
            raise ValueError("coordinator VRAM byte accounting changed")
        if self.coordinator_vram_required_bytes > coordinator_budget:
            raise MemoryError("tiled expert exceeds coordinator VRAM budget")

        egress_mbps = _finite(
            "coordinator_egress_mbps",
            self.coordinator_egress_mbps,
            positive=True,
        )
        ingress_mbps = _finite(
            "coordinator_ingress_mbps",
            self.coordinator_ingress_mbps,
            positive=True,
        )
        expected_egress_bytes = len(self.tiles) * self.spec.input_activation_bytes
        expected_ingress_bytes = len(self.tiles) * self.spec.partial_output_bytes
        if self.coordinator_egress_bytes != expected_egress_bytes:
            raise ValueError("coordinator aggregate egress byte accounting changed")
        if self.coordinator_ingress_bytes != expected_ingress_bytes:
            raise ValueError("coordinator aggregate ingress byte accounting changed")
        expected_egress_ms = expected_egress_bytes / (egress_mbps * 125.0)
        expected_ingress_ms = expected_ingress_bytes / (ingress_mbps * 125.0)
        if not math.isclose(
            self.coordinator_egress_ms,
            expected_egress_ms,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("coordinator aggregate egress time changed")
        if not math.isclose(
            self.coordinator_ingress_ms,
            expected_ingress_ms,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("coordinator aggregate ingress time changed")
        setup_per_owner = _finite(
            "coordinator_rpc_setup_ms_per_owner",
            self.coordinator_rpc_setup_ms_per_owner,
            positive=False,
        )
        expected_setup_ms = len(self.tiles) * setup_per_owner
        if not math.isclose(
            self.coordinator_rpc_setup_ms,
            expected_setup_ms,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("coordinator RPC setup time changed")

        expected_per_owner_parallel = max(tile.exposed_ms for tile in self.tiles)
        # The response cannot arrive before its request has left the
        # coordinator and an owner has crossed RTT plus compute.  Model the
        # two aggregate NIC directions as causal phases.  The outer max below
        # treats this shared-NIC path and the individual-link path as alternate
        # bottlenecks, avoiding a second charge for the same wire bytes.
        expected_causal_path = (
            expected_egress_ms
            + max(tile.round_trip_ms + tile.compute_ms for tile in self.tiles)
            + expected_ingress_ms
        )
        if not math.isclose(
            self.coordinator_causal_path_ms,
            expected_causal_path,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("coordinator causal fanout path changed")
        expected_serial = sum(tile.exposed_ms for tile in self.tiles)
        if not math.isclose(
            self.per_owner_parallel_ms,
            expected_per_owner_parallel,
            rel_tol=1e-12,
        ):
            raise ValueError("per-owner parallel time must equal the slowest owner")
        if not math.isclose(self.serial_owner_sum_ms, expected_serial, rel_tol=1e-12):
            raise ValueError("serial owner diagnostic time changed")
        expected_fanout = expected_setup_ms + max(
            expected_per_owner_parallel,
            expected_causal_path,
        )
        if not math.isclose(
            self.fanout_parallel_ms,
            expected_fanout,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "parallel fanout must include aggregate coordinator NIC contention"
            )
        expected_reduction_bytes = (
            max(0, len(self.tiles) - 1) * self.spec.partial_output_bytes * 2
        )
        if self.reduction_bytes != expected_reduction_bytes:
            raise ValueError("ordered reduction byte accounting changed")
        reduction_bandwidth = _finite(
            "reduction_gbytes_per_second",
            self.reduction_gbytes_per_second,
            positive=True,
        )
        expected_reduction_ms = self.reduction_bytes / (
            reduction_bandwidth * 1_000_000.0
        )
        if not math.isclose(
            self.reduction_ms,
            expected_reduction_ms,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("ordered reduction time changed")
        if not math.isclose(
            self.projected_exposed_ms,
            self.fanout_parallel_ms + self.reduction_ms,
            rel_tol=1e-12,
        ):
            raise ValueError("projected time must equal fanout plus ordered reduction")


def plan_tiled_swiglu_expert(
    spec: SwiGLUTilingSpec,
    workers: Sequence[TiledExpertWorkerProfile],
    *,
    weights: SwiGLUWeights,
    coordinator_vram_budget_bytes: int,
    coordinator_egress_mbps: float,
    coordinator_ingress_mbps: float,
    coordinator_reserved_vram_bytes: int = 0,
    coordinator_rpc_setup_ms_per_owner: float = 0.0,
    reduction_gbytes_per_second: float = 100.0,
) -> TiledExpertPlan:
    """Choose deterministic canonical tiles that fit every selected worker.

    Weight slices are assumed resident as an adapted artifact.  Network cost
    includes one full hidden activation sent to, and one partial hidden output
    returned from, each selected owner.  All selected owners execute in
    parallel, while transformer layers remain outside this plan and sequential.
    """

    if not isinstance(spec, SwiGLUTilingSpec):
        raise TypeError("spec must be SwiGLUTilingSpec")
    if not workers:
        raise ValueError("workers cannot be empty")
    if not isinstance(weights, SwiGLUWeights):
        raise TypeError("weights must be SwiGLUWeights")
    weights.validate_content()
    if weights.artifact_identity != spec.artifact_identity:
        raise ValueError("weights artifact identity does not match the sealed spec")
    if weights.expert_content_id != spec.expert_content_id:
        raise ValueError("weights expert content identity does not match the sealed spec")
    if (
        weights.hidden_size != spec.hidden_size
        or weights.intermediate_size != spec.intermediate_size
    ):
        raise ValueError("weights geometry does not match the sealed spec")
    if weights.dtype_name != spec.weight_dtype:
        raise ValueError("weights dtype does not match the sealed spec")
    actual_weight_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in (weights.gate, weights.up, weights.down)
    )
    if actual_weight_bytes != spec.total_expert_weight_bytes:
        raise ValueError("weights byte size does not match the sealed spec")
    coordinator_budget = _positive_int(
        "coordinator_vram_budget_bytes",
        coordinator_vram_budget_bytes,
    )
    coordinator_reserved = _nonnegative_int(
        "coordinator_reserved_vram_bytes",
        coordinator_reserved_vram_bytes,
    )
    egress_mbps = _finite(
        "coordinator_egress_mbps",
        coordinator_egress_mbps,
        positive=True,
    )
    ingress_mbps = _finite(
        "coordinator_ingress_mbps",
        coordinator_ingress_mbps,
        positive=True,
    )
    setup_ms_per_owner = _finite(
        "coordinator_rpc_setup_ms_per_owner",
        coordinator_rpc_setup_ms_per_owner,
        positive=False,
    )
    reduction_bandwidth = _finite(
        "reduction_gbytes_per_second",
        reduction_gbytes_per_second,
        positive=True,
    )

    indexed: dict[str, TiledExpertWorkerProfile] = {}
    for worker in workers:
        if not isinstance(worker, TiledExpertWorkerProfile):
            raise TypeError("workers must contain TiledExpertWorkerProfile values")
        if worker.owner_id in indexed:
            raise ValueError(f"duplicate tiled expert owner {worker.owner_id!r}")
        indexed[worker.owner_id] = worker

    variable_bytes_per_intermediate = (
        spec.weight_bytes_per_intermediate
        + spec.workspace_bytes_per_intermediate
    )
    capacities: dict[str, int] = {}
    for owner_id, worker in sorted(indexed.items()):
        if not worker.available:
            continue
        fixed_bytes = (
            worker.reserved_vram_bytes
            + spec.input_activation_bytes
            + spec.partial_output_bytes
        )
        remaining = worker.usable_vram_bytes - fixed_bytes
        capacity = max(0, remaining // variable_bytes_per_intermediate)
        if capacity > 0:
            capacities[owner_id] = min(capacity, spec.intermediate_size)

    if not capacities:
        raise MemoryError(
            "no available worker can fit one canonical intermediate channel"
        )
    if sum(capacities.values()) < spec.intermediate_size:
        raise MemoryError(
            "combined worker VRAM cannot hold the complete tiled expert"
        )

    # Greedy list scheduling by next projected completion.  The heap makes the
    # choice deterministic and accounts for each owner's fixed RTT/activation
    # transfer before assigning it a first channel.
    assigned = {owner_id: 0 for owner_id in capacities}
    heap: list[tuple[float, str]] = []
    compute_ms_per_channel: dict[str, float] = {}
    transfer_ms: dict[str, float] = {}
    for owner_id in sorted(capacities):
        worker = indexed[owner_id]
        transfer = (
            worker.round_trip_ms
            + spec.activation_round_trip_bytes_per_tile / worker.bytes_per_ms
        )
        compute = (
            spec.estimated_operations_per_intermediate
            / (worker.compute_gflops * 1_000_000.0)
        )
        transfer_ms[owner_id] = transfer
        compute_ms_per_channel[owner_id] = compute
        heapq.heappush(heap, (transfer + compute, owner_id))

    for _ in range(spec.intermediate_size):
        if not heap:
            raise MemoryError("worker capacities were exhausted while assigning tiles")
        _, owner_id = heapq.heappop(heap)
        assigned[owner_id] += 1
        if assigned[owner_id] < capacities[owner_id]:
            next_finish = (
                transfer_ms[owner_id]
                + compute_ms_per_channel[owner_id] * (assigned[owner_id] + 1)
            )
            heapq.heappush(heap, (next_finish, owner_id))

    tiles: list[TiledExpertTile] = []
    cursor = 0
    for owner_id in sorted(owner for owner, width in assigned.items() if width > 0):
        worker = indexed[owner_id]
        width = assigned[owner_id]
        weight_bytes = width * spec.weight_bytes_per_intermediate
        workspace_bytes = width * spec.workspace_bytes_per_intermediate
        required = (
            worker.reserved_vram_bytes
            + weight_bytes
            + spec.input_activation_bytes
            + spec.partial_output_bytes
            + workspace_bytes
        )
        compute_ms = compute_ms_per_channel[owner_id] * width
        tile = TiledExpertTile(
            tile_index=len(tiles),
            owner_id=owner_id,
            artifact_identity=spec.artifact_identity,
            expert_content_id=spec.expert_content_id,
            tile_content_id=_tile_content_identity(
                artifact_identity=spec.artifact_identity,
                expert_content_id=spec.expert_content_id,
                intermediate_start=cursor,
                intermediate_end=cursor + width,
                full_intermediate_size=spec.intermediate_size,
                gate=weights.gate[cursor : cursor + width, :],
                up=weights.up[cursor : cursor + width, :],
                down=weights.down[:, cursor : cursor + width],
            ),
            weight_dtype=spec.weight_dtype,
            intermediate_start=cursor,
            intermediate_end=cursor + width,
            worker_vram_budget_bytes=worker.usable_vram_bytes,
            worker_reserved_vram_bytes=worker.reserved_vram_bytes,
            weight_bytes=weight_bytes,
            input_activation_bytes=spec.input_activation_bytes,
            partial_output_bytes=spec.partial_output_bytes,
            workspace_bytes=workspace_bytes,
            vram_required_bytes=required,
            round_trip_ms=worker.round_trip_ms,
            transfer_ms=transfer_ms[owner_id],
            compute_ms=compute_ms,
            exposed_ms=transfer_ms[owner_id] + compute_ms,
        )
        tiles.append(tile)
        cursor += width

    output_bytes = spec.partial_output_bytes
    reduction_bytes = max(0, len(tiles) - 1) * output_bytes * 2
    reduction_ms = reduction_bytes / (reduction_bandwidth * 1_000_000.0)
    coordinator_input_bytes = spec.input_activation_bytes
    coordinator_partial_bytes = len(tiles) * spec.partial_output_bytes
    coordinator_output_bytes = (
        spec.positions * spec.hidden_size * spec.accumulation_element_bytes
    )
    coordinator_workspace_bytes = 0
    coordinator_required = (
        coordinator_reserved
        + coordinator_input_bytes
        + coordinator_partial_bytes
        + coordinator_output_bytes
        + coordinator_workspace_bytes
    )
    coordinator_egress_bytes = len(tiles) * spec.input_activation_bytes
    coordinator_ingress_bytes = len(tiles) * spec.partial_output_bytes
    coordinator_egress_ms = coordinator_egress_bytes / (egress_mbps * 125.0)
    coordinator_ingress_ms = coordinator_ingress_bytes / (ingress_mbps * 125.0)
    coordinator_rpc_setup_ms = len(tiles) * setup_ms_per_owner
    per_owner_parallel_ms = max(tile.exposed_ms for tile in tiles)
    coordinator_causal_path_ms = (
        coordinator_egress_ms
        + max(tile.round_trip_ms + tile.compute_ms for tile in tiles)
        + coordinator_ingress_ms
    )
    fanout_ms = coordinator_rpc_setup_ms + max(
        per_owner_parallel_ms,
        coordinator_causal_path_ms,
    )
    return TiledExpertPlan(
        schema=TILED_EXPERT_SCHEMA,
        mode=TILED_EXPERT_MODE,
        reduction_order=TILED_EXPERT_REDUCTION_ORDER,
        bitwise_equivalent_to_monolithic=False,
        spec=spec,
        tiles=tuple(tiles),
        total_weight_bytes=sum(tile.weight_bytes for tile in tiles),
        activation_round_trip_bytes=sum(
            tile.activation_round_trip_bytes for tile in tiles
        ),
        coordinator_vram_budget_bytes=coordinator_budget,
        coordinator_reserved_vram_bytes=coordinator_reserved,
        coordinator_input_activation_bytes=coordinator_input_bytes,
        coordinator_partial_output_bytes=coordinator_partial_bytes,
        coordinator_reduction_output_bytes=coordinator_output_bytes,
        coordinator_workspace_bytes=coordinator_workspace_bytes,
        coordinator_vram_required_bytes=coordinator_required,
        coordinator_egress_mbps=egress_mbps,
        coordinator_ingress_mbps=ingress_mbps,
        coordinator_egress_bytes=coordinator_egress_bytes,
        coordinator_ingress_bytes=coordinator_ingress_bytes,
        coordinator_egress_ms=coordinator_egress_ms,
        coordinator_ingress_ms=coordinator_ingress_ms,
        coordinator_rpc_setup_ms_per_owner=setup_ms_per_owner,
        coordinator_rpc_setup_ms=coordinator_rpc_setup_ms,
        coordinator_causal_path_ms=coordinator_causal_path_ms,
        per_owner_parallel_ms=per_owner_parallel_ms,
        fanout_parallel_ms=fanout_ms,
        serial_owner_sum_ms=sum(tile.exposed_ms for tile in tiles),
        reduction_bytes=reduction_bytes,
        reduction_gbytes_per_second=reduction_bandwidth,
        reduction_ms=reduction_ms,
        projected_exposed_ms=fanout_ms + reduction_ms,
    )


@dataclass(frozen=True)
class SwiGLUWeights:
    artifact_identity: str
    gate: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor
    expert_content_id: str = field(init=False)
    _content_versions: tuple[int | None, int | None, int | None] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        artifact_identity = _sha256_identity(
            "artifact_identity",
            self.artifact_identity,
        )
        object.__setattr__(self, "artifact_identity", artifact_identity)
        _validate_swiglu_geometry(self.gate, self.up, self.down)
        object.__setattr__(
            self,
            "expert_content_id",
            _expert_content_identity(
                artifact_identity,
                self.gate,
                self.up,
                self.down,
            ),
        )
        object.__setattr__(
            self,
            "_content_versions",
            tuple(_tensor_version(tensor) for tensor in (self.gate, self.up, self.down)),
        )

    @classmethod
    def from_tensors(
        cls,
        artifact_identity: str,
        *,
        gate: torch.Tensor,
        up: torch.Tensor,
        down: torch.Tensor,
    ) -> "SwiGLUWeights":
        return cls(
            artifact_identity=artifact_identity,
            gate=gate,
            up=up,
            down=down,
        )

    def validate_content(self) -> None:
        versions = tuple(
            _tensor_version(tensor) for tensor in (self.gate, self.up, self.down)
        )
        if None not in versions and versions == self._content_versions:
            return
        actual = _expert_content_identity(
            self.artifact_identity,
            self.gate,
            self.up,
            self.down,
        )
        if actual != self.expert_content_id:
            raise ValueError("SwiGLU expert tensors changed after content sealing")
        object.__setattr__(self, "_content_versions", versions)

    @property
    def hidden_size(self) -> int:
        return int(self.gate.shape[1])

    @property
    def intermediate_size(self) -> int:
        return int(self.gate.shape[0])

    @property
    def dtype_name(self) -> str:
        return _tensor_dtype_name(self.gate)


@dataclass(frozen=True)
class SwiGLUTileWeights:
    artifact_identity: str
    expert_content_id: str
    intermediate_start: int
    intermediate_end: int
    full_intermediate_size: int
    gate: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor
    tile_content_id: str = field(init=False)
    _content_versions: tuple[int | None, int | None, int | None] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        artifact_identity = _sha256_identity(
            "artifact_identity",
            self.artifact_identity,
        )
        expert_content_id = _sha256_identity(
            "expert_content_id",
            self.expert_content_id,
        )
        object.__setattr__(self, "artifact_identity", artifact_identity)
        object.__setattr__(self, "expert_content_id", expert_content_id)
        start = _nonnegative_int("intermediate_start", self.intermediate_start)
        end = _positive_int("intermediate_end", self.intermediate_end)
        full = _positive_int("full_intermediate_size", self.full_intermediate_size)
        if not 0 <= start < end <= full:
            raise ValueError("tile weight slice is outside the intermediate axis")
        _validate_swiglu_geometry(self.gate, self.up, self.down)
        if int(self.gate.shape[0]) != end - start:
            raise ValueError("tile weight width does not match its canonical slice")
        object.__setattr__(
            self,
            "tile_content_id",
            _tile_content_identity(
                artifact_identity=artifact_identity,
                expert_content_id=expert_content_id,
                intermediate_start=start,
                intermediate_end=end,
                full_intermediate_size=full,
                gate=self.gate,
                up=self.up,
                down=self.down,
            ),
        )
        object.__setattr__(
            self,
            "_content_versions",
            tuple(_tensor_version(tensor) for tensor in (self.gate, self.up, self.down)),
        )

    @classmethod
    def from_tensors(
        cls,
        *,
        artifact_identity: str,
        expert_content_id: str,
        intermediate_start: int,
        intermediate_end: int,
        full_intermediate_size: int,
        gate: torch.Tensor,
        up: torch.Tensor,
        down: torch.Tensor,
    ) -> "SwiGLUTileWeights":
        return cls(
            artifact_identity=artifact_identity,
            expert_content_id=expert_content_id,
            intermediate_start=intermediate_start,
            intermediate_end=intermediate_end,
            full_intermediate_size=full_intermediate_size,
            gate=gate,
            up=up,
            down=down,
        )

    def validate_content(self) -> None:
        versions = tuple(
            _tensor_version(tensor) for tensor in (self.gate, self.up, self.down)
        )
        if None not in versions and versions == self._content_versions:
            return
        actual = _tile_content_identity(
            artifact_identity=self.artifact_identity,
            expert_content_id=self.expert_content_id,
            intermediate_start=self.intermediate_start,
            intermediate_end=self.intermediate_end,
            full_intermediate_size=self.full_intermediate_size,
            gate=self.gate,
            up=self.up,
            down=self.down,
        )
        if actual != self.tile_content_id:
            raise ValueError("SwiGLU tile tensors changed after content sealing")
        object.__setattr__(self, "_content_versions", versions)

    @property
    def hidden_size(self) -> int:
        return int(self.gate.shape[1])

    @property
    def width(self) -> int:
        return self.intermediate_end - self.intermediate_start

    @property
    def byte_size(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self.gate, self.up, self.down)
        )

    @property
    def dtype_name(self) -> str:
        return _tensor_dtype_name(self.gate)


def canonical_swiglu_tile(
    weights: SwiGLUWeights,
    tile: TiledExpertTile,
) -> SwiGLUTileWeights:
    """Materialize the adapted artifact slices for one canonical tile."""

    if not isinstance(weights, SwiGLUWeights):
        raise TypeError("weights must be SwiGLUWeights")
    if not isinstance(tile, TiledExpertTile):
        raise TypeError("tile must be TiledExpertTile")
    weights.validate_content()
    if weights.artifact_identity != tile.artifact_identity:
        raise ValueError("monolithic artifact identity does not match the sealed tile")
    if weights.expert_content_id != tile.expert_content_id:
        raise ValueError("monolithic expert content does not match the sealed tile")
    if weights.dtype_name != tile.weight_dtype:
        raise ValueError("monolithic weight dtype does not match the sealed tile")
    start = tile.intermediate_start
    end = tile.intermediate_end
    if end > weights.intermediate_size:
        raise ValueError("tile exceeds monolithic expert intermediate size")
    adapted = SwiGLUTileWeights.from_tensors(
        artifact_identity=weights.artifact_identity,
        expert_content_id=weights.expert_content_id,
        intermediate_start=start,
        intermediate_end=end,
        full_intermediate_size=weights.intermediate_size,
        gate=weights.gate[start:end, :].detach().contiguous().clone(),
        up=weights.up[start:end, :].detach().contiguous().clone(),
        down=weights.down[:, start:end].detach().contiguous().clone(),
    )
    if adapted.tile_content_id != tile.tile_content_id:
        raise ValueError("adapted tile content does not match the sealed plan")
    return adapted


class InMemoryTiledExpertOwner:
    """One exact adapted tile owner used by deterministic CPU tests."""

    def __init__(
        self,
        owner_id: str,
        tile: TiledExpertTile,
        weights: SwiGLUTileWeights,
        *,
        vram_budget_bytes: int | None = None,
    ) -> None:
        self.owner_id = _identifier("owner_id", owner_id)
        if not isinstance(tile, TiledExpertTile):
            raise TypeError("tile must be TiledExpertTile")
        if not isinstance(weights, SwiGLUTileWeights):
            raise TypeError("weights must be SwiGLUTileWeights")
        self.tile = tile
        self.weights = weights
        self.vram_budget_bytes = _positive_int(
            "vram_budget_bytes",
            tile.worker_vram_budget_bytes
            if vram_budget_bytes is None
            else vram_budget_bytes,
        )

    @classmethod
    def from_monolithic(
        cls,
        tile: TiledExpertTile,
        weights: SwiGLUWeights,
        *,
        vram_budget_bytes: int | None = None,
    ) -> "InMemoryTiledExpertOwner":
        return cls(
            tile.owner_id,
            tile,
            canonical_swiglu_tile(weights, tile),
            vram_budget_bytes=vram_budget_bytes,
        )

    def validate_assignment(
        self,
        tile: TiledExpertTile,
        spec: SwiGLUTilingSpec,
        hidden: torch.Tensor,
    ) -> None:
        if tile != self.tile or tile.owner_id != self.owner_id:
            raise ValueError(f"owner {self.owner_id!r} received a non-canonical tile")
        if tile.vram_required_bytes > self.vram_budget_bytes:
            raise MemoryError(
                f"tile {tile.tile_index} requires {tile.vram_required_bytes} bytes "
                f"but owner {self.owner_id!r} exposes {self.vram_budget_bytes}"
            )
        weights = self.weights
        weights.validate_content()
        if (
            weights.artifact_identity != tile.artifact_identity
            or weights.artifact_identity != spec.artifact_identity
        ):
            raise ValueError("owner artifact identity does not match the sealed plan")
        if (
            weights.expert_content_id != tile.expert_content_id
            or weights.expert_content_id != spec.expert_content_id
        ):
            raise ValueError("owner expert content does not match the sealed plan")
        if weights.tile_content_id != tile.tile_content_id:
            raise ValueError("owner tile content does not match the sealed plan")
        if (
            weights.intermediate_start != tile.intermediate_start
            or weights.intermediate_end != tile.intermediate_end
            or weights.full_intermediate_size != spec.intermediate_size
            or weights.hidden_size != spec.hidden_size
        ):
            raise ValueError("owner tile weights do not match the sealed plan geometry")
        if weights.byte_size != tile.weight_bytes:
            raise ValueError("owner tile weight bytes do not match the sealed plan")
        if weights.dtype_name != tile.weight_dtype or weights.dtype_name != spec.weight_dtype:
            raise ValueError("owner tile dtype does not match the sealed weight dtype")
        if _tensor_dtype_name(hidden) != spec.activation_dtype:
            raise ValueError("hidden dtype does not match the sealed activation dtype")
        if hidden.device != weights.gate.device:
            raise ValueError("hidden and tile weights must share a device")

    @torch.no_grad()
    def execute_tile(
        self,
        tile: TiledExpertTile,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        if tile != self.tile:
            raise ValueError(f"owner {self.owner_id!r} received an unknown tile")
        if tile.vram_required_bytes > self.vram_budget_bytes:
            raise MemoryError(
                f"tile {tile.tile_index} exceeds owner {self.owner_id!r} VRAM budget"
            )
        weights = self.weights
        if hidden.device != weights.gate.device or hidden.dtype != weights.gate.dtype:
            raise ValueError("hidden and tile weights must share device and dtype")
        gate = F.linear(hidden, weights.gate)
        up = F.linear(hidden, weights.up)
        return F.linear(F.silu(gate) * up, weights.down)


class InMemoryTiledSwiGLUExecutor:
    """Parallel owner fanout with deterministic canonical-index reduction."""

    def __init__(
        self,
        plan: TiledExpertPlan,
        owners: Mapping[str, InMemoryTiledExpertOwner],
    ) -> None:
        if not isinstance(plan, TiledExpertPlan):
            raise TypeError("plan must be TiledExpertPlan")
        self.plan = plan
        self.owners = dict(owners)
        for owner_id, owner in self.owners.items():
            if not isinstance(owner, InMemoryTiledExpertOwner):
                raise TypeError("owners must contain InMemoryTiledExpertOwner values")
            if owner_id != owner.owner_id:
                raise ValueError("owner mapping key does not match owner identity")

    @torch.no_grad()
    def execute(self, hidden: torch.Tensor) -> torch.Tensor:
        spec = self.plan.spec
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 2:
            raise ValueError("hidden must have shape [positions, hidden_size]")
        if tuple(hidden.shape) != (spec.positions, spec.hidden_size):
            raise ValueError("hidden shape does not match the sealed tiled expert plan")
        if not hidden.is_floating_point() or _tensor_dtype_name(hidden) != spec.activation_dtype:
            raise ValueError("hidden dtype does not match the sealed activation dtype")

        # Validate all owners and budgets before any thread executes.  A stale
        # artifact or oversized tile cannot leave a partially reduced result.
        selected: list[tuple[TiledExpertTile, InMemoryTiledExpertOwner]] = []
        for tile in self.plan.tiles:
            owner = self.owners.get(tile.owner_id)
            if owner is None:
                raise KeyError(f"missing in-memory tile owner {tile.owner_id!r}")
            owner.validate_assignment(tile, spec, hidden)
            selected.append((tile, owner))

        with ThreadPoolExecutor(
            max_workers=len(selected),
            thread_name_prefix="gdlp-tiled-expert",
        ) as pool:
            futures = {
                tile.tile_index: pool.submit(owner.execute_tile, tile, hidden)
                for tile, owner in selected
            }
            # Futures may complete in any order.  Reading and summing strictly
            # by canonical tile index makes the floating reduction repeatable.
            partials = tuple(
                futures[tile.tile_index].result()
                for tile in self.plan.tiles
            )

        result = torch.zeros_like(hidden)
        for partial in partials:
            if (
                partial.shape != hidden.shape
                or partial.dtype != hidden.dtype
                or partial.device != hidden.device
            ):
                raise RuntimeError("tile owner returned an incompatible partial output")
            result.add_(partial)
        return result


@torch.no_grad()
def monolithic_swiglu(hidden: torch.Tensor, weights: SwiGLUWeights) -> torch.Tensor:
    """Reference bias-free monolithic SwiGLU evaluation."""

    if not isinstance(weights, SwiGLUWeights):
        raise TypeError("weights must be SwiGLUWeights")
    weights.validate_content()
    if hidden.ndim != 2 or int(hidden.shape[1]) != weights.hidden_size:
        raise ValueError("hidden shape does not match monolithic SwiGLU weights")
    if hidden.device != weights.gate.device or hidden.dtype != weights.gate.dtype:
        raise ValueError("hidden and monolithic weights must share device and dtype")
    gate = F.linear(hidden, weights.gate)
    up = F.linear(hidden, weights.up)
    return F.linear(F.silu(gate) * up, weights.down)


def _validate_swiglu_geometry(
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
) -> None:
    if any(not isinstance(tensor, torch.Tensor) for tensor in (gate, up, down)):
        raise TypeError("SwiGLU weights must be torch.Tensor values")
    if gate.ndim != 2 or up.ndim != 2 or down.ndim != 2:
        raise ValueError("SwiGLU weights must be rank-two matrices")
    if gate.shape != up.shape:
        raise ValueError("SwiGLU gate and up weights must have identical geometry")
    intermediate, hidden = (int(gate.shape[0]), int(gate.shape[1]))
    if intermediate < 1 or hidden < 1 or tuple(down.shape) != (hidden, intermediate):
        raise ValueError("SwiGLU down weight must have shape [hidden, intermediate]")
    if any(not tensor.is_floating_point() for tensor in (gate, up, down)):
        raise TypeError("SwiGLU weights must use floating dtypes")
    if gate.dtype != up.dtype or gate.dtype != down.dtype:
        raise ValueError("SwiGLU weights must share one dtype")
    if gate.device != up.device or gate.device != down.device:
        raise ValueError("SwiGLU weights must share one device")


__all__ = [
    "InMemoryTiledExpertOwner",
    "InMemoryTiledSwiGLUExecutor",
    "SwiGLUTileWeights",
    "SwiGLUTilingSpec",
    "SwiGLUWeights",
    "TILED_EXPERT_MODE",
    "TILED_EXPERT_REDUCTION_ORDER",
    "TILED_EXPERT_SCHEMA",
    "TiledExpertPlan",
    "TiledExpertTile",
    "TiledExpertWorkerProfile",
    "canonical_swiglu_tile",
    "monolithic_swiglu",
    "plan_tiled_swiglu_expert",
]
