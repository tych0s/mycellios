"""Exact Hugging Face MoE stage with RAM-owned and VRAM-cached experts.

This is the vertical execution path that joins the direct safetensors loader,
``TorchRamExpertStore`` and the existing pipeline ``StageRunner`` lifecycle.
The Hugging Face decoder is created on ``meta``; routed expert parameters are
replaced before any resident allocation, so attention, router, dense/shared
MLPs, norms, embeddings/head and KV cache can live on the compute device while
the complete routed expert set remains authoritative in host RAM.

The target model's own router still produces ``top_k_index`` and
``top_k_weights``.  Prediction is used only to prefetch a later layer.  A miss
must resolve the router-selected immutable expert from RAM before its GEMMs,
which preserves target-model semantics.
"""

from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import threading
import time
from typing import Callable, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModelForCausalLM

from .device import describe_torch_execution_device
from .model import (
    StageModelSpec,
    StageRunner,
    _checkpoint_key_map,
    _load_stage_parameters_from_safetensors,
    resolve_model_snapshot,
)
from .model_adapters import SelectiveStageAdapter, resolve_selective_stage_adapter
from .ram_expert_cache import (
    ExpertKey,
    PredictiveCacheConfig,
    RamBackedExpertScheduler,
    RouteResolution,
    UnknownExpertError,
)
from .resident_expert_mesh import (
    ExpertResidentSlotUnavailableError,
    AuthoritativeRouting,
    ExpertOwner,
    OwnerCoalescedExpertBatchItem,
    OwnerExpertBatchItem,
    OwnerExpertBatchResult,
    ResidentExpertMesh,
)
from .safetensors_moe_stage_loader import (
    LocalSafetensorsMoeStage,
    load_local_safetensors_moe_stage,
)
from .torch_ram_expert_store import TorchExpertBundle, TorchRamExpertStore


@dataclass(frozen=True)
class RamBackedMoeExecutionSnapshot:
    forwards: int
    routed_layers: int
    routed_tokens: int
    authoritative_expert_uses: int
    cache_hits: int
    prefetch_hits: int
    ram_misses: int
    fallback_bytes: int
    prediction_false_positives: int
    prediction_false_negatives: int


class OnlineExpertTransitionPredictor:
    """Learn exact adjacent-layer expert transitions without changing routing.

    Counts are updated only after both authoritative routes have executed.
    Predictions therefore start empty and become useful on later forwards.
    They can waste a prefetch when routing changes, but they can never select
    the expert used for computation.
    """

    def __init__(self) -> None:
        self._counts: dict[tuple[int, int, int, int], int] = defaultdict(int)

    def observe(
        self,
        source_layer: int,
        target_layer: int,
        source_experts: Sequence[int],
        target_experts: Sequence[int],
    ) -> None:
        for source in dict.fromkeys(source_experts):
            for target in dict.fromkeys(target_experts):
                self._counts[(source_layer, target_layer, source, target)] += 1

    def predict(
        self,
        source_layer: int,
        target_layer: int,
        source_experts: Sequence[int],
        *,
        limit: int,
    ) -> tuple[tuple[int, float], ...]:
        scores: dict[int, int] = defaultdict(int)
        for source in dict.fromkeys(source_experts):
            for (observed_source_layer, observed_target_layer, observed_source, target), count in self._counts.items():
                if (
                    observed_source_layer == source_layer
                    and observed_target_layer == target_layer
                    and observed_source == source
                ):
                    scores[target] += count
        if not scores:
            return ()
        maximum = max(scores.values())
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]
        return tuple((expert, score / maximum) for expert, score in ranked)


class _PrefetchCoordinator:
    def __init__(
        self,
        sparse_layers: Sequence[int],
        scheduler: RamBackedExpertScheduler,
        *,
        prediction_slots: int,
        predictor: OnlineExpertTransitionPredictor | None,
        prefetch_deadline_ms: float,
    ) -> None:
        if prefetch_deadline_ms <= 0:
            raise ValueError("prefetch_deadline_ms must be positive")
        self.sparse_layers = tuple(sparse_layers)
        self.scheduler = scheduler
        if prediction_slots < 1:
            raise ValueError("prediction_slots must be positive")
        self.prediction_slots = prediction_slots
        self.predictor = predictor
        self.prefetch_deadline_ms = float(prefetch_deadline_ms)
        self._current_routes: dict[int, tuple[int, ...]] = {}
        self._predicted: dict[int, tuple[int, ...]] = {}
        self.forwards = 0

    def begin_forward(self) -> None:
        self._current_routes.clear()
        self._predicted.clear()
        self.forwards += 1

    def end_forward(self) -> None:
        self._current_routes.clear()
        self._predicted.clear()

    def prediction_for(self, layer: int) -> tuple[int, ...]:
        return self._predicted.pop(layer, ())

    def after_authoritative_route(
        self,
        layer: int,
        authoritative_experts: Sequence[int],
    ) -> None:
        actual = tuple(dict.fromkeys(int(expert) for expert in authoritative_experts))
        index = self.sparse_layers.index(layer)
        if index > 0 and self.predictor is not None:
            previous_layer = self.sparse_layers[index - 1]
            previous = self._current_routes.get(previous_layer)
            if previous is not None:
                self.predictor.observe(previous_layer, layer, previous, actual)
        self._current_routes[layer] = actual

        if self.predictor is None or index + 1 >= len(self.sparse_layers):
            return
        next_layer = self.sparse_layers[index + 1]
        predictions = self.predictor.predict(
            layer,
            next_layer,
            actual,
            limit=self.prediction_slots,
        )
        if not predictions:
            return
        now_ms = time.monotonic() * 1_000
        for expert, confidence in predictions:
            record = self.scheduler.inventory.record(ExpertKey(next_layer, expert))
            if record.byte_size > self.scheduler.config.prefetch_reserve_bytes:
                continue
            self.scheduler.submit_prefetch(
                ExpertKey(next_layer, expert),
                now_ms=now_ms,
                deadline_ms=now_ms + self.prefetch_deadline_ms,
                confidence=confidence,
            )
        staged = self.scheduler.stage_prefetch(now_ms=now_ms)
        staged_keys = set(staged.staged) | set(staged.already_resident)
        self._predicted[next_layer] = tuple(
            expert
            for expert, _ in predictions
            if ExpertKey(next_layer, expert) in staged_keys
        )


class RamBackedSwiGluExpertOwner:
    """Exact local owner backed by the existing RAM scheduler and store.

    The mesh sends unweighted activations selected by the target router.  This
    owner executes the checkpoint's packed gate+up SwiGLU followed by down;
    top-k gates remain at the coordinator and are applied exactly once during
    canonical reduction.
    """

    supports_exact_input_coalescing = True
    # Local calls pass row indices as Python metadata; only physical RPC
    # transports account for a serialized row-map payload.
    coalesced_row_index_bytes_per_assignment = 0

    def __init__(
        self,
        node_id: str,
        *,
        layer: int,
        num_experts: int,
        hidden_dim: int,
        intermediate_dim: int,
        act_fn,
        scheduler: RamBackedExpertScheduler,
        store: TorchRamExpertStore,
        prefetch_deadline_ms: float = 1_000.0,
        resolution_observer: Callable[[RouteResolution], None] | None = None,
    ) -> None:
        if not isinstance(node_id, str) or not node_id.strip():
            raise ValueError("owner node_id cannot be empty")
        if not isinstance(layer, int) or isinstance(layer, bool) or layer < 0:
            raise ValueError("owner layer must be a non-negative integer")
        if not isinstance(num_experts, int) or num_experts < 1:
            raise ValueError("owner num_experts must be positive")
        if not isinstance(hidden_dim, int) or hidden_dim < 1:
            raise ValueError("owner hidden_dim must be positive")
        if not isinstance(intermediate_dim, int) or intermediate_dim < 1:
            raise ValueError("owner intermediate_dim must be positive")
        if not callable(act_fn):
            raise TypeError("owner act_fn must be callable")
        if prefetch_deadline_ms <= 0:
            raise ValueError("owner prefetch_deadline_ms must be positive")
        self.node_id = node_id.strip()
        self.layer = layer
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.act_fn = act_fn
        self.scheduler = scheduler
        self.store = store
        self.prefetch_deadline_ms = float(prefetch_deadline_ms)
        self.resolution_observer = resolution_observer
        self.batch_calls = 0
        self.coalesced_batch_calls = 0
        self._lock = threading.RLock()

    def has_expert(self, key: ExpertKey, content_id: str) -> bool:
        if key.layer != self.layer or not 0 <= key.expert < self.num_experts:
            return False
        try:
            record = self.scheduler.inventory.record(key)
        except UnknownExpertError:
            return False
        return record.content_id == content_id

    def is_expert_resident(self, key: ExpertKey, content_id: str) -> bool:
        """Prove the exact artifact is active in both scheduler and device store."""

        if not self.has_expert(key, content_id):
            return False
        with self._lock:
            scheduler_active = set(self.scheduler.snapshot().active_keys)
            store_active = set(self.store.snapshot().active_keys)
            return key in scheduler_active and key in store_active

    def _validate_bundle_contract(
        self,
        bundle: TorchExpertBundle,
        *,
        activation_dtype: torch.dtype,
        activation_device: torch.device | None,
    ) -> None:
        gate = bundle.tensor("gate_proj.weight")
        up = bundle.tensor("up_proj.weight")
        down = bundle.tensor("down_proj.weight")
        if tuple(gate.shape) != (self.intermediate_dim, self.hidden_dim):
            raise RuntimeError("gate projection violates the sealed expert shape")
        if tuple(up.shape) != (self.intermediate_dim, self.hidden_dim):
            raise RuntimeError("up projection violates the sealed expert shape")
        if tuple(down.shape) != (self.hidden_dim, self.intermediate_dim):
            raise RuntimeError("down projection violates the sealed expert shape")
        if not (gate.dtype == up.dtype == down.dtype == activation_dtype):
            raise RuntimeError("resolved expert dtype does not match activation dtype")
        if activation_device is not None and not (
            gate.device == up.device == down.device == activation_device
        ):
            raise RuntimeError("resolved expert is not on the activation device")

    def _preflight_batch(
        self,
        items: Sequence[OwnerExpertBatchItem],
    ) -> tuple[OwnerExpertBatchItem, ...]:
        batch = tuple(items)
        if not batch:
            raise ValueError("owner expert batch cannot be empty")
        if any(not isinstance(item, OwnerExpertBatchItem) for item in batch):
            raise TypeError("owner batch must contain OwnerExpertBatchItem values")
        keys = tuple(item.key for item in batch)
        if len(set(keys)) != len(keys):
            raise ValueError("owner expert batch repeats an expert")
        if keys != tuple(sorted(keys)):
            raise ValueError("owner expert batch order is not canonical")
        for item in batch:
            if not self.has_expert(item.key, item.content_id):
                raise RuntimeError(
                    f"owner {self.node_id!r} lacks exact content for {item.key}"
                )
            if (
                item.activations.ndim != 2
                or item.activations.shape[0] < 1
                or item.activations.shape[1] != self.hidden_dim
            ):
                raise ValueError("owner activation violates the sealed expert shape")
            if not torch.is_floating_point(item.activations):
                raise TypeError("owner activation must use a floating dtype")
            if (
                item.activations.device.type != "cpu"
                and item.activations.device != self.store.effective_device
            ):
                raise RuntimeError(
                    "owner activation must be on CPU or the sealed compute device"
                )
            ram_bundle = self.store.ram_bundle(item.key)
            if ram_bundle.content_id != item.content_id:
                raise RuntimeError(
                    f"RAM bundle identity mismatch for {item.key}"
                )
            # Check every bundle before the first scheduler mutation.  This
            # keeps malformed content from producing a partially executed
            # owner batch.
            self._validate_bundle_contract(
                ram_bundle,
                activation_dtype=item.activations.dtype,
                activation_device=None,
            )
            if item.require_resident and not self.is_expert_resident(
                item.key,
                item.content_id,
            ):
                raise ExpertResidentSlotUnavailableError(
                    item.key,
                    f"owner {self.node_id!r} lost physical residency "
                    f"for {item.key}"
                )
        return batch

    def _preflight_coalesced_batch(
        self,
        shared_activations: torch.Tensor,
        items: Sequence[OwnerCoalescedExpertBatchItem],
    ) -> tuple[OwnerCoalescedExpertBatchItem, ...]:
        """Seal the complete shared-input contract before scheduler mutation."""

        if not isinstance(shared_activations, torch.Tensor):
            raise TypeError("shared owner activations must be a tensor")
        if (
            shared_activations.ndim != 2
            or shared_activations.shape[0] < 1
            or shared_activations.shape[1] != self.hidden_dim
        ):
            raise ValueError("shared owner activation violates the sealed expert shape")
        if not torch.is_floating_point(shared_activations):
            raise TypeError("shared owner activation must use a floating dtype")
        if (
            shared_activations.device.type != "cpu"
            and shared_activations.device != self.store.effective_device
        ):
            raise RuntimeError(
                "shared owner activation must be on CPU or the sealed compute device"
            )

        batch = tuple(items)
        if not batch:
            raise ValueError("coalesced owner expert batch cannot be empty")
        if any(
            not isinstance(item, OwnerCoalescedExpertBatchItem) for item in batch
        ):
            raise TypeError(
                "coalesced owner batch must contain "
                "OwnerCoalescedExpertBatchItem values"
            )
        keys = tuple(item.key for item in batch)
        if len(set(keys)) != len(keys):
            raise ValueError("coalesced owner expert batch repeats an expert")
        if keys != tuple(sorted(keys)):
            raise ValueError("coalesced owner expert batch order is not canonical")

        shared_rows = int(shared_activations.shape[0])
        referenced_rows = bytearray(shared_rows)
        referenced_count = 0
        for item in batch:
            if not self.has_expert(item.key, item.content_id):
                raise RuntimeError(
                    f"owner {self.node_id!r} lacks exact content for {item.key}"
                )
            for row_index in item.row_indices:
                if row_index >= shared_rows:
                    raise ValueError(
                        "coalesced owner row index exceeds shared activations"
                    )
                if not referenced_rows[row_index]:
                    referenced_rows[row_index] = 1
                    referenced_count += 1
            ram_bundle = self.store.ram_bundle(item.key)
            if ram_bundle.content_id != item.content_id:
                raise RuntimeError(f"RAM bundle identity mismatch for {item.key}")
            self._validate_bundle_contract(
                ram_bundle,
                activation_dtype=shared_activations.dtype,
                activation_device=None,
            )
            if item.require_resident and not self.is_expert_resident(
                item.key,
                item.content_id,
            ):
                raise ExpertResidentSlotUnavailableError(
                    item.key,
                    f"owner {self.node_id!r} lost physical residency "
                    f"for {item.key}"
                )
        if referenced_count != shared_rows:
            raise ValueError(
                "coalesced owner row map must reference every shared activation"
            )
        return batch

    @torch.no_grad()
    def execute_batch(
        self,
        items: Sequence[OwnerExpertBatchItem],
    ) -> tuple[OwnerExpertBatchResult, ...]:
        with self._lock:
            batch = self._preflight_batch(items)
            self.batch_calls += 1
            outputs: list[OwnerExpertBatchResult] = []
            for index, item in enumerate(batch):
                if item.require_resident and not self.is_expert_resident(
                    item.key,
                    item.content_id,
                ):
                    raise ExpertResidentSlotUnavailableError(
                        item.key,
                        f"owner {self.node_id!r} lost physical residency "
                        f"for {item.key} before execution"
                    )
                resolution = self.scheduler.resolve_route(
                    layer=self.layer,
                    authoritative_expert_ids=(item.key.expert,),
                )
                try:
                    if self.resolution_observer is not None:
                        self.resolution_observer(resolution)
                    if item.require_resident and (
                        resolution.ram_fallbacks
                        or item.key not in resolution.cache_hits
                    ):
                        raise ExpertResidentSlotUnavailableError(
                            item.key,
                            f"resident route for {item.key} attempted a RAM load"
                        )
                    # RPC tensors are reconstructed on CPU.  A CUDA owner must
                    # perform this explicit H2D stage; for the coordinator's
                    # already-local activations this is a no-op.
                    current = item.activations.to(
                        device=self.store.effective_device,
                        dtype=item.activations.dtype,
                    ).contiguous()
                    bundle = self.store.bundle(item.key)
                    self._validate_bundle_contract(
                        bundle,
                        activation_dtype=current.dtype,
                        activation_device=current.device,
                    )
                    if index + 1 < len(batch):
                        next_key = batch[index + 1].key
                        next_record = self.scheduler.inventory.record(next_key)
                        if (
                            next_record.byte_size
                            <= self.scheduler.config.prefetch_reserve_bytes
                        ):
                            now_ms = time.monotonic() * 1_000
                            self.scheduler.submit_prefetch(
                                next_key,
                                now_ms=now_ms,
                                deadline_ms=now_ms + self.prefetch_deadline_ms,
                                confidence=1.0,
                            )
                            self.scheduler.stage_prefetch(now_ms=now_ms)
                    gate_values, up_values = F.linear(
                        current,
                        bundle.packed_gate_up(),
                    ).chunk(2, dim=-1)
                    activated = self.act_fn(gate_values) * up_values
                    output = F.linear(
                        activated,
                        bundle.tensor("down_proj.weight"),
                    )
                    if tuple(output.shape) != (
                        current.shape[0],
                        self.hidden_dim,
                    ):
                        raise RuntimeError("expert produced an invalid output shape")
                    outputs.append(OwnerExpertBatchResult(item.key, output))
                finally:
                    self.scheduler.release_route(resolution)
            return tuple(outputs)

    @torch.no_grad()
    def execute_coalesced_batch(
        self,
        shared_activations: torch.Tensor,
        items: Sequence[OwnerCoalescedExpertBatchItem],
    ) -> tuple[OwnerExpertBatchResult, ...]:
        """Execute exact expert views after staging their shared input once.

        Outputs remain raw and separated by expert.  Router gates and the
        canonical reduction therefore stay at the coordinator exactly as in
        ``execute_batch``.
        """

        with self._lock:
            batch = self._preflight_coalesced_batch(shared_activations, items)
            self.batch_calls += 1
            self.coalesced_batch_calls += 1

            # RPC reconstructs the shared matrix on CPU.  Transfer that matrix
            # once, then gather each expert's rows on the compute device.
            shared_current = shared_activations.to(
                device=self.store.effective_device,
                dtype=shared_activations.dtype,
            ).contiguous()
            outputs: list[OwnerExpertBatchResult] = []
            for index, item in enumerate(batch):
                # Residency is dynamic state, so prove it again immediately
                # before each authoritative resolve rather than trusting the
                # all-batch preflight snapshot.
                if item.require_resident and not self.is_expert_resident(
                    item.key,
                    item.content_id,
                ):
                    raise ExpertResidentSlotUnavailableError(
                        item.key,
                        f"owner {self.node_id!r} lost physical residency "
                        f"for {item.key} before execution"
                    )
                resolution = self.scheduler.resolve_route(
                    layer=self.layer,
                    authoritative_expert_ids=(item.key.expert,),
                )
                try:
                    if self.resolution_observer is not None:
                        self.resolution_observer(resolution)
                    if item.require_resident and (
                        resolution.ram_fallbacks
                        or item.key not in resolution.cache_hits
                    ):
                        raise ExpertResidentSlotUnavailableError(
                            item.key,
                            f"resident route for {item.key} attempted a RAM load"
                        )
                    bundle = self.store.bundle(item.key)
                    self._validate_bundle_contract(
                        bundle,
                        activation_dtype=shared_current.dtype,
                        activation_device=shared_current.device,
                    )
                    if index + 1 < len(batch):
                        next_key = batch[index + 1].key
                        next_record = self.scheduler.inventory.record(next_key)
                        if (
                            next_record.byte_size
                            <= self.scheduler.config.prefetch_reserve_bytes
                        ):
                            now_ms = time.monotonic() * 1_000
                            self.scheduler.submit_prefetch(
                                next_key,
                                now_ms=now_ms,
                                deadline_ms=(
                                    now_ms + self.prefetch_deadline_ms
                                ),
                                confidence=1.0,
                            )
                            self.scheduler.stage_prefetch(now_ms=now_ms)
                    row_indices = torch.tensor(
                        item.row_indices,
                        dtype=torch.long,
                        device=shared_current.device,
                    )
                    current = shared_current.index_select(0, row_indices)
                    gate_values, up_values = F.linear(
                        current,
                        bundle.packed_gate_up(),
                    ).chunk(2, dim=-1)
                    activated = self.act_fn(gate_values) * up_values
                    output = F.linear(
                        activated,
                        bundle.tensor("down_proj.weight"),
                    )
                    if tuple(output.shape) != (
                        len(item.row_indices),
                        self.hidden_dim,
                    ):
                        raise RuntimeError("expert produced an invalid output shape")
                    outputs.append(OwnerExpertBatchResult(item.key, output))
                finally:
                    self.scheduler.release_route(resolution)
            return tuple(outputs)


@dataclass(frozen=True)
class _ResidentExpertMeshAttachment:
    mesh: ResidentExpertMesh
    owners: Mapping[str, ExpertOwner]
    local_owner: RamBackedSwiGluExpertOwner


class RamBackedMoeExperts(nn.Module):
    """Drop-in replacement for Transformers grouped Qwen/GLM experts."""

    def __init__(
        self,
        *,
        layer: int,
        num_experts: int,
        hidden_dim: int,
        intermediate_dim: int,
        act_fn,
        scheduler: RamBackedExpertScheduler,
        store: TorchRamExpertStore,
        coordinator: _PrefetchCoordinator,
    ) -> None:
        super().__init__()
        self.layer = layer
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.act_fn = act_fn
        self.scheduler = scheduler
        self.store = store
        self.coordinator = coordinator
        self.routed_layer_calls = 0
        self.routed_tokens = 0
        self.authoritative_expert_uses = 0
        self.cache_hits = 0
        self.prefetch_hits = 0
        self.ram_misses = 0
        self.fallback_bytes = 0
        self.prediction_false_positives = 0
        self.prediction_false_negatives = 0
        self._resident_mesh_attachment: _ResidentExpertMeshAttachment | None = None
        self.last_resident_mesh_plan = None

    @property
    def resident_expert_mesh_attached(self) -> bool:
        return self._resident_mesh_attachment is not None

    @property
    def resident_expert_local_owner(self) -> RamBackedSwiGluExpertOwner | None:
        attachment = self._resident_mesh_attachment
        return attachment.local_owner if attachment is not None else None

    def attach_resident_expert_mesh(
        self,
        mesh: ResidentExpertMesh,
        owners: Mapping[str, ExpertOwner],
    ) -> None:
        """Opt into mesh execution for this layer after an exact preflight.

        ``owners`` contains injected remote executors, including a
        ``ResidentExpertRpcClient`` directly.  The coordinator owner is always
        built from this module's scheduler/store and is never supplied by the
        caller.  Attachment transfers no lifecycle ownership.
        """

        if self._resident_mesh_attachment is not None:
            raise RuntimeError(f"layer {self.layer} already has a resident mesh")
        if not isinstance(mesh, ResidentExpertMesh):
            raise TypeError("mesh must be ResidentExpertMesh")
        if mesh.closed:
            raise ValueError("cannot attach a closed resident expert mesh")
        if not isinstance(owners, Mapping):
            raise TypeError("owners must be a mapping")

        inventory_records = tuple(
            self.scheduler.inventory.record(ExpertKey(self.layer, expert))
            for expert in range(self.num_experts)
        )
        mesh_records = mesh.expert_records_for_layer(self.layer)
        if mesh_records != inventory_records:
            raise ValueError(
                f"mesh inventory does not exactly match RAM layer {self.layer}"
            )
        first_bundle = self.store.ram_bundle(inventory_records[0].key)
        source_dtype = first_bundle.tensor("gate_proj.weight").dtype
        expected_activation_bytes = (
            self.hidden_dim
            * torch.empty((), dtype=source_dtype).element_size()
        )
        if mesh.activation_bytes_per_token != expected_activation_bytes:
            raise ValueError(
                "mesh activation bytes do not match the RAM-backed layer dtype"
            )

        # Packed gate+up materializes 2I values; SiLU(gate) and the multiplied
        # activation can coexist for another 2I. Three H-sized activation /
        # reduction buffers are accounted separately by the
        # mesh.  Require every possible layer owner to seal at least this
        # operator-specific workspace instead of assuming weights are the only
        # VRAM consumer.
        minimum_workspace_bytes = (
            4
            * self.intermediate_dim
            * torch.empty((), dtype=source_dtype).element_size()
        )
        execution_node_ids = {mesh.coordinator_id} | {
            replica.node_id for replica in mesh.replicas_for_layer(self.layer)
        }
        for node_id in sorted(execution_node_ids):
            if (
                mesh.node_profile(node_id).expert_workspace_bytes_per_token
                < minimum_workspace_bytes
            ):
                raise ValueError(
                    f"mesh node {node_id!r} has unsealed SwiGLU workspace; "
                    f"requires at least {minimum_workspace_bytes} bytes/token"
                )

        mesh_node_ids = {node.node_id for node in mesh.nodes}
        normalized: dict[str, ExpertOwner] = {}
        for owner_id, owner in owners.items():
            if not isinstance(owner_id, str) or not owner_id.strip():
                raise ValueError("owner mapping keys cannot be empty")
            if owner_id != owner_id.strip():
                raise ValueError("owner mapping keys must be canonical")
            if owner_id == mesh.coordinator_id:
                raise ValueError("the coordinator owner is created by the RAM stage")
            if owner_id not in mesh_node_ids:
                raise ValueError(f"owner {owner_id!r} is absent from the mesh")
            if getattr(owner, "node_id", None) != owner_id:
                raise ValueError(f"owner mapping identity mismatch for {owner_id!r}")
            if (
                not callable(getattr(owner, "has_expert", None))
                or not callable(getattr(owner, "is_expert_resident", None))
                or not callable(getattr(owner, "execute_batch", None))
            ):
                raise TypeError(f"owner {owner_id!r} violates the ExpertOwner ABI")
            normalized[owner_id] = owner

        for replica in mesh.replicas_for_layer(self.layer):
            owner = normalized.get(replica.node_id)
            if owner is None:
                raise ValueError(
                    f"mesh owner {replica.node_id!r} is missing for {replica.key}"
                )
            if not owner.has_expert(replica.key, replica.content_id):
                raise ValueError(
                    f"mesh owner {replica.node_id!r} lacks exact content "
                    f"for {replica.key}"
                )

        local_owner = RamBackedSwiGluExpertOwner(
            mesh.coordinator_id,
            layer=self.layer,
            num_experts=self.num_experts,
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
            act_fn=self.act_fn,
            scheduler=self.scheduler,
            store=self.store,
            prefetch_deadline_ms=self.coordinator.prefetch_deadline_ms,
            resolution_observer=self._record_route_resolution,
        )
        normalized[mesh.coordinator_id] = local_owner
        self._resident_mesh_attachment = _ResidentExpertMeshAttachment(
            mesh=mesh,
            owners=dict(normalized),
            local_owner=local_owner,
        )

    def detach_resident_expert_mesh(self) -> None:
        """Return to serial RAM execution without closing injected owners."""

        self._resident_mesh_attachment = None
        self.last_resident_mesh_plan = None

    def _record_route_resolution(self, resolution: RouteResolution) -> None:
        self.cache_hits += len(resolution.cache_hits)
        self.prefetch_hits += len(resolution.prefetch_hits)
        self.ram_misses += len(resolution.ram_fallbacks)
        self.fallback_bytes += resolution.fallback_bytes

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.ndim != 2 or hidden_states.shape[1] != self.hidden_dim:
            raise ValueError(
                f"routed hidden states must have shape [tokens, {self.hidden_dim}]"
            )
        if (
            top_k_index.ndim != 2
            or top_k_index.shape[0] != hidden_states.shape[0]
            or tuple(top_k_weights.shape) != tuple(top_k_index.shape)
        ):
            raise ValueError("router indexes and weights must have shape [tokens, top_k]")
        if top_k_index.dtype not in (torch.int32, torch.int64):
            raise TypeError("router indexes must be integer tensors")
        attachment = self._resident_mesh_attachment
        routing: AuthoritativeRouting | None = None
        if attachment is not None:
            # Preserve the target router tensors verbatim.  The routing object
            # makes one grouped ID transfer for planning; weights stay on the
            # activation device and are gathered only during final reduction.
            routing = AuthoritativeRouting(top_k_index, top_k_weights)
            authoritative = routing.authoritative_experts
        else:
            authoritative = tuple(
                int(value)
                for value in torch.unique(
                    top_k_index,
                    sorted=True,
                ).detach().cpu().tolist()
            )
        if not authoritative or any(
            expert < 0 or expert >= self.num_experts for expert in authoritative
        ):
            raise ValueError("authoritative router selected an invalid expert")

        predicted = self.coordinator.prediction_for(self.layer)
        predicted_set = set(predicted)
        authoritative_set = set(authoritative)
        self.routed_layer_calls += 1
        self.routed_tokens += int(hidden_states.shape[0])
        self.authoritative_expert_uses += len(authoritative)
        # Prediction quality is a property of the complete layer route.  The
        # physical executor resolves that route one expert at a time to keep a
        # two-buffer working set, so summing per-resolution FP/FN would either
        # lose false positives or count misses more than once.
        self.prediction_false_positives += len(predicted_set - authoritative_set)
        self.prediction_false_negatives += len(authoritative_set - predicted_set)

        if attachment is not None:
            assert routing is not None
            self.last_resident_mesh_plan = None
            final_hidden_states, plan = attachment.mesh.execute_layer(
                hidden_states,
                self.layer,
                routing,
                attachment.owners,
            )
            self.last_resident_mesh_plan = plan
            self.coordinator.after_authoritative_route(self.layer, authoritative)
            return final_hidden_states

        final_hidden_states = torch.zeros_like(hidden_states)
        # Execute in the same ascending expert order as the Transformers
        # grouped implementation.  Resolving one expert at a time is crucial:
        # top-k=8 must not require eight complete expert bundles in VRAM.  Once
        # the authoritative router has exposed the complete route, the next
        # exact expert can be copied on the prefetch stream while the current
        # expert GEMMs execute.  This is look-ahead, not predicted routing.
        for index, expert in enumerate(authoritative):
            resolution = self.scheduler.resolve_route(
                layer=self.layer,
                authoritative_expert_ids=(expert,),
                predicted_expert_ids=((expert,) if expert in predicted else ()),
            )
            self._record_route_resolution(resolution)
            try:
                bundle = self.store.bundle(ExpertKey(self.layer, expert))
                if index + 1 < len(authoritative):
                    next_expert = authoritative[index + 1]
                    next_key = ExpertKey(self.layer, next_expert)
                    next_record = self.scheduler.inventory.record(next_key)
                    if (
                        next_record.byte_size
                        <= self.scheduler.config.prefetch_reserve_bytes
                    ):
                        now_ms = time.monotonic() * 1_000
                        self.scheduler.submit_prefetch(
                            next_key,
                            now_ms=now_ms,
                            deadline_ms=now_ms + self.coordinator.prefetch_deadline_ms,
                            confidence=1.0,
                        )
                        self.scheduler.stage_prefetch(now_ms=now_ms)
                token_idx, top_k_pos = torch.where(top_k_index == expert)
                current = hidden_states[token_idx]
                gate = bundle.tensor("gate_proj.weight")
                up = bundle.tensor("up_proj.weight")
                down = bundle.tensor("down_proj.weight")
                if (
                    gate.device != hidden_states.device
                    or up.device != hidden_states.device
                    or down.device != hidden_states.device
                ):
                    raise RuntimeError("resolved expert is not on the activation device")
                if not (gate.dtype == up.dtype == down.dtype == hidden_states.dtype):
                    raise RuntimeError("resolved expert dtype does not match activation dtype")
                # Transformers' grouped expert implementation performs gate+up
                # in one GEMM. Reconstituting that packed matrix preserves its
                # accumulation order for both packed and unpacked checkpoints.
                gate_values, up_values = F.linear(
                    current,
                    bundle.packed_gate_up(),
                ).chunk(2, dim=-1)
                current_hidden = self.act_fn(gate_values) * up_values
                current_hidden = F.linear(current_hidden, down)
                current_hidden = current_hidden * top_k_weights[
                    token_idx, top_k_pos, None
                ]
                final_hidden_states.index_add_(
                    0,
                    token_idx,
                    current_hidden.to(final_hidden_states.dtype),
                )
            finally:
                self.scheduler.release_route(resolution)
        self.coordinator.after_authoritative_route(self.layer, authoritative)
        return final_hidden_states


def _preflight_requested_compute_device(
    device: str | torch.device,
    *,
    allow_cpu_fallback: bool,
) -> torch.device:
    """Reject an unavailable strict CUDA target before opening model weights."""

    try:
        requested = torch.device(device)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"invalid RAM-backed MoE compute device {device!r}") from error
    if requested.type != "cuda" or allow_cpu_fallback:
        return requested
    if not torch.cuda.is_available():
        raise RuntimeError(
            "RAM-backed MoE CUDA device is unavailable and CPU fallback is forbidden"
        )
    device_count = torch.cuda.device_count()
    index = requested.index
    if index is not None and not 0 <= index < device_count:
        raise RuntimeError(
            f"RAM-backed MoE CUDA device index {index} is unavailable; "
            f"detected {device_count} CUDA device(s)"
        )
    return requested


class RamBackedMoeStageRunner(StageRunner):
    """Pipeline StageRunner whose routed experts are never fully device-resident."""

    def __init__(
        self,
        spec: StageModelSpec,
        cache_config: PredictiveCacheConfig,
        *,
        device: str | torch.device = "cuda",
        pin_memory: bool = False,
        bounded_pinned_staging: bool = False,
        allow_cpu_fallback: bool = True,
        enable_online_prefetch: bool = True,
        prefetch_deadline_ms: float = 1_000.0,
        expected_adapter_id: str | None = None,
        expected_resident_parameter_bytes: int | None = None,
        expected_total_routed_expert_bytes: int | None = None,
        expected_largest_expert_bytes: int | None = None,
        expected_resident_streaming_transient_bytes: int | None = None,
        expected_bounded_pinned_staging_reserve_bytes: int | None = None,
        expected_host_ram_peak_upper_bound_bytes: int | None = None,
    ) -> None:
        torch.set_num_threads(spec.threads)
        requested_device = _preflight_requested_compute_device(
            device,
            allow_cpu_fallback=allow_cpu_fallback,
        )
        reserve_bounded_pinned_staging = bool(
            bounded_pinned_staging
            and requested_device.type == "cuda"
            and torch.cuda.is_available()
        )
        snapshot = resolve_model_snapshot(spec.model_name, spec.revision)
        expected_identity = (
            spec.artifact_identity
            if spec.artifact_identity is not None
            and spec.artifact_identity.startswith("sha256:")
            else None
        )
        loaded = load_local_safetensors_moe_stage(
            snapshot,
            layer_start=spec.layer_start,
            layer_end=spec.layer_end,
            expected_artifact_identity=expected_identity,
            expected_adapter_id=expected_adapter_id,
            expected_total_routed_expert_bytes=expected_total_routed_expert_bytes,
            expected_largest_expert_bytes=expected_largest_expert_bytes,
            preload_resident_artifacts=False,
            reserve_bounded_pinned_staging=reserve_bounded_pinned_staging,
            expected_resident_streaming_transient_bytes=(
                expected_resident_streaming_transient_bytes
            ),
            expected_bounded_pinned_staging_reserve_bytes=(
                expected_bounded_pinned_staging_reserve_bytes
            ),
            expected_host_ram_peak_bytes=(
                expected_host_ram_peak_upper_bound_bytes
            ),
        )
        store = loaded.create_store(
            cache_config,
            device=device,
            pin_memory=pin_memory,
            bounded_pinned_staging=bounded_pinned_staging,
            allow_cpu_fallback=allow_cpu_fallback,
        )
        scheduler = RamBackedExpertScheduler(
            loaded.inventory,
            cache_config,
            weight_backend=store,
        )
        config = AutoConfig.from_pretrained(
            snapshot,
            local_files_only=True,
            trust_remote_code=False,
        )
        top_k = int(config.num_experts_per_tok)
        largest_expert_bytes = max(
            record.byte_size for record in loaded.inventory.records
        )
        prediction_slots = max(
            1,
            min(
                top_k,
                cache_config.prefetch_reserve_bytes // largest_expert_bytes,
            ),
        )
        coordinator = _PrefetchCoordinator(
            loaded.sparse_layers,
            scheduler,
            prediction_slots=prediction_slots,
            predictor=(OnlineExpertTransitionPredictor() if enable_online_prefetch else None),
            prefetch_deadline_ms=prefetch_deadline_ms,
        )
        model, adapter = _build_meta_resident_model(
            spec,
            snapshot=Path(snapshot),
            loaded=loaded,
            store=store,
            scheduler=scheduler,
            coordinator=coordinator,
            expected_resident_parameter_bytes=expected_resident_parameter_bytes,
        )
        semantic_features = tuple(
            feature
            for feature in adapter.semantic_features
            if feature not in {"resident-expert-set", "ep-plan-metadata-only"}
        ) + (
            "ram-authoritative-routed-experts",
            "device-expert-cache",
            "authoritative-router",
            "predictive-prefetch-only",
            "meta-model-construction",
            "activation-cast-at-stage-boundary",
        )
        self._initialize_from_loaded_model(
            spec,
            model,
            loader="selective-safetensors-ram-backed-moe",
            device_kinds=(store.effective_device.type,),
            execution_device=describe_torch_execution_device(
                store.effective_device
            ),
            compute_dtype=loaded.source_dtype,
            move_model=False,
            semantic_features=semantic_features,
        )
        self.expert_store = store
        self.expert_scheduler = scheduler
        self.prefetch_coordinator = coordinator
        self.compute_device = store.effective_device
        self.compute_dtype = loaded.source_dtype
        self.resident_parameter_bytes = self.parameter_bytes
        self.ram_parameter_bytes = loaded.routed_expert_bytes
        self.largest_expert_bytes = largest_expert_bytes
        self.loaded_backing_bytes = loaded.loaded_backing_bytes
        self.resident_streaming_transient_bytes = (
            loaded.resident_streaming_transient_bytes
        )
        store_snapshot = store.snapshot()
        self.bounded_pinned_staging_reserve_bytes = (
            store_snapshot.pinned_capacity_bytes
        )
        if (
            self.bounded_pinned_staging_reserve_bytes
            != loaded.bounded_pinned_staging_reserve_bytes
        ):
            raise RuntimeError(
                "bounded pinned staging allocation changed after host RAM preflight"
            )
        self.host_ram_steady_state_bytes = (
            self.ram_parameter_bytes
            + self.bounded_pinned_staging_reserve_bytes
        )
        self.host_ram_peak_upper_bound_bytes = (
            self.host_ram_steady_state_bytes
            + self.resident_streaming_transient_bytes
        )
        if self.host_ram_peak_upper_bound_bytes != loaded.host_ram_peak_upper_bound_bytes:
            raise RuntimeError("host RAM peak accounting changed after model load")
        self.sparse_layers = loaded.sparse_layers
        self.dense_layers = loaded.dense_layers
        self._expert_modules = tuple(
            layer.mlp.experts
            for layer in self.base.layers
            if isinstance(getattr(getattr(layer, "mlp", None), "experts", None), RamBackedMoeExperts)
        )
        self._resident_mesh_owned_owners: dict[int, ExpertOwner] = {}
        # The store adopted only routed bundles. Router/shared tensors were
        # streamed directly from safetensors into the resident device model;
        # they never joined the persistent host-RAM bundle set.
        del loaded

    def _expert_module_for_layer(self, layer: int) -> RamBackedMoeExperts:
        if not isinstance(layer, int) or isinstance(layer, bool) or layer < 0:
            raise ValueError("mesh layer must be a non-negative integer")
        matches = tuple(module for module in self._expert_modules if module.layer == layer)
        if len(matches) != 1:
            raise ValueError(f"RAM-backed sparse layer {layer} is not owned by this stage")
        return matches[0]

    def attach_resident_expert_mesh(
        self,
        layer: int,
        mesh: ResidentExpertMesh,
        owners: Mapping[str, ExpertOwner],
        *,
        close_owners_on_close: bool = False,
    ) -> None:
        """Attach one layer; injected owner lifecycle remains caller-owned by default."""

        if not isinstance(close_owners_on_close, bool):
            raise TypeError("close_owners_on_close must be boolean")
        if not isinstance(owners, Mapping):
            raise TypeError("owners must be a mapping")
        owner_values = tuple(owners.values())
        if close_owners_on_close:
            for owner in owner_values:
                if not callable(getattr(owner, "close", None)):
                    raise TypeError(
                        "explicitly owned mesh executors must provide close()"
                    )
        module = self._expert_module_for_layer(layer)
        module.attach_resident_expert_mesh(mesh, owners)
        if close_owners_on_close:
            for owner in owner_values:
                self._resident_mesh_owned_owners[id(owner)] = owner

    def detach_resident_expert_mesh(self, layer: int) -> None:
        """Detach a layer without closing or otherwise mutating injected owners."""

        self._expert_module_for_layer(layer).detach_resident_expert_mesh()

    def _forward_scope(self, operation):
        self.prefetch_coordinator.begin_forward()
        try:
            return operation()
        finally:
            self.prefetch_coordinator.end_forward()

    def forward_ids(self, request_id: int, input_ids: torch.Tensor) -> torch.Tensor:
        prepared = input_ids.to(device=self.compute_device)
        output = self._forward_scope(
            lambda: super(RamBackedMoeStageRunner, self).forward_ids(request_id, prepared)
        )
        return output.to(device="cpu", dtype=torch.float32).contiguous()

    def forward_ids_batch(
        self,
        request_ids: Sequence[int],
        input_ids: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, ...]:
        prepared = tuple(value.to(device=self.compute_device) for value in input_ids)
        outputs = self._forward_scope(
            lambda: super(RamBackedMoeStageRunner, self).forward_ids_batch(request_ids, prepared)
        )
        return tuple(
            output.to(device="cpu", dtype=torch.float32).contiguous()
            for output in outputs
        )

    def forward_hidden(
        self,
        request_id: int,
        hidden: torch.Tensor,
        *,
        token_mode: str = "last",
    ) -> tuple[torch.Tensor, int | tuple[int, ...] | None]:
        prepared = hidden.to(device=self.compute_device, dtype=self.compute_dtype)
        output, token = self._forward_scope(
            lambda: super(RamBackedMoeStageRunner, self).forward_hidden(
                request_id,
                prepared,
                token_mode=token_mode,
            )
        )
        return output.to(device="cpu", dtype=torch.float32).contiguous(), token

    def forward_hidden_batch(
        self,
        request_ids: Sequence[int],
        hidden_states: Sequence[torch.Tensor],
        *,
        token_mode: str = "last",
    ):
        prepared = tuple(
            value.to(device=self.compute_device, dtype=self.compute_dtype)
            for value in hidden_states
        )
        outputs = self._forward_scope(
            lambda: super(RamBackedMoeStageRunner, self).forward_hidden_batch(
                request_ids,
                prepared,
                token_mode=token_mode,
            )
        )
        return tuple(
            (output.to(device="cpu", dtype=torch.float32).contiguous(), token)
            for output, token in outputs
        )

    def execution_snapshot(self) -> RamBackedMoeExecutionSnapshot:
        return RamBackedMoeExecutionSnapshot(
            forwards=self.prefetch_coordinator.forwards,
            routed_layers=sum(
                module.routed_layer_calls for module in self._expert_modules
            ),
            routed_tokens=sum(module.routed_tokens for module in self._expert_modules),
            authoritative_expert_uses=sum(
                module.authoritative_expert_uses for module in self._expert_modules
            ),
            cache_hits=sum(module.cache_hits for module in self._expert_modules),
            prefetch_hits=sum(module.prefetch_hits for module in self._expert_modules),
            ram_misses=sum(module.ram_misses for module in self._expert_modules),
            fallback_bytes=sum(
                module.fallback_bytes for module in self._expert_modules
            ),
            prediction_false_positives=sum(
                module.prediction_false_positives for module in self._expert_modules
            ),
            prediction_false_negatives=sum(
                module.prediction_false_negatives for module in self._expert_modules
            ),
        )

    def close(self) -> None:
        self.expert_scheduler.discard_prefetch()
        try:
            super().close()
        finally:
            # Only executors accepted with the explicit lifecycle-transfer flag
            # are closed.  Normal injected owners, including RPC clients, stay
            # entirely caller-owned.
            owned = tuple(self._resident_mesh_owned_owners.values())
            self._resident_mesh_owned_owners.clear()
            for owner in owned:
                owner.close()  # type: ignore[attr-defined]


def _build_meta_resident_model(
    spec: StageModelSpec,
    *,
    snapshot: Path,
    loaded: LocalSafetensorsMoeStage,
    store: TorchRamExpertStore,
    scheduler: RamBackedExpertScheduler,
    coordinator: _PrefetchCoordinator,
    expected_resident_parameter_bytes: int | None = None,
) -> tuple[nn.Module, SelectiveStageAdapter]:
    config = AutoConfig.from_pretrained(
        str(snapshot),
        local_files_only=True,
        trust_remote_code=False,
    )
    adapter = resolve_selective_stage_adapter(config)
    adapter.validate_source_config(config, spec.total_layers)
    local_config = copy.deepcopy(config)
    adapter.slice_config(
        local_config,
        layer_start=spec.layer_start,
        layer_end=spec.layer_end,
        total_layers=spec.total_layers,
    )
    local_layers = spec.layer_end - spec.layer_start
    original_vocab_size = int(local_config.vocab_size)
    original_pad_token_id = getattr(local_config, "pad_token_id", None)
    if not spec.first and not spec.last:
        local_config.vocab_size = 1
        local_config.pad_token_id = 0

    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(
            local_config,
            dtype=loaded.source_dtype,
        )
        adapter.inspect_constructed_model(model, local_layers=local_layers)
        if not spec.last:
            model.model.norm = nn.Identity()
            model.lm_head = nn.Identity()
        if not spec.first:
            model.model.embed_tokens = nn.Embedding(1, model.config.hidden_size)

    if not spec.first and not spec.last:
        model.config.vocab_size = original_vocab_size
        model.config.pad_token_id = original_pad_token_id

    for global_layer in loaded.sparse_layers:
        local_layer = global_layer - spec.layer_start
        mlp = model.model.layers[local_layer].mlp
        original = getattr(mlp, "experts", None)
        if not isinstance(original, nn.Module):
            raise TypeError(f"sparse layer {global_layer} has no grouped expert module")
        num_experts = int(getattr(original, "num_experts"))
        hidden_dim = int(getattr(original, "hidden_dim"))
        intermediate_dim = int(getattr(original, "intermediate_dim"))
        act_fn = getattr(original, "act_fn", None)
        if not callable(act_fn):
            raise TypeError(f"sparse layer {global_layer} has no expert activation")
        mlp.experts = RamBackedMoeExperts(
            layer=global_layer,
            num_experts=num_experts,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            act_fn=act_fn,
            scheduler=scheduler,
            store=store,
            coordinator=coordinator,
        )

    persistent_names = set(model.state_dict())
    unsupported_nonpersistent = tuple(
        name
        for name, _ in model.named_buffers()
        if name not in persistent_names and not name.startswith("model.rotary_emb.")
    )
    if unsupported_nonpersistent:
        raise TypeError(
            "meta construction found unsupported derived buffers: "
            f"{unsupported_nonpersistent}"
        )

    resident_parameter_bytes = _unique_meta_parameter_bytes(model)
    if expected_resident_parameter_bytes is not None:
        if (
            not isinstance(expected_resident_parameter_bytes, int)
            or isinstance(expected_resident_parameter_bytes, bool)
            or expected_resident_parameter_bytes < 1
        ):
            raise ValueError("expected_resident_parameter_bytes must be positive")
        if resident_parameter_bytes > expected_resident_parameter_bytes:
            raise MemoryError(
                "resident parameter budget would be exceeded before device allocation: "
                f"required {resident_parameter_bytes}, budget "
                f"{expected_resident_parameter_bytes}"
            )

    model.to_empty(device=store.effective_device)
    if bool(getattr(model.config, "tie_word_embeddings", False)) and spec.first and spec.last:
        model.tie_weights()
    rotary_type = type(model.model.rotary_emb)
    try:
        model.model.rotary_emb = rotary_type(
            config=local_config,
            device=store.effective_device,
        )
    except TypeError as error:
        raise TypeError("certified rotary embedding cannot be rebuilt after meta allocation") from error

    preloaded: dict[str, torch.Tensor] = {}
    for layer, artifact in loaded.routers.items():
        for name, tensor in artifact.tensors:
            preloaded[f"model.layers.{layer}.mlp.gate.{name}"] = tensor
    for layer, artifact in loaded.shared_experts.items():
        for name, tensor in artifact.tensors:
            preloaded[f"model.layers.{layer}.mlp.shared_experts.{name}"] = tensor

    checkpoint_names = set(_checkpoint_key_map(snapshot))
    ignored = {
        name
        for name in checkpoint_names
        if any(
            name.startswith(f"model.layers.{layer}.mlp.experts.")
            for layer in loaded.sparse_layers
        )
    }
    if not ignored:
        raise KeyError("direct MoE loader found no routed checkpoint tensors to externalize")
    resolved_spec = StageModelSpec(
        model_name=str(snapshot),
        layer_start=spec.layer_start,
        layer_end=spec.layer_end,
        total_layers=spec.total_layers,
        threads=spec.threads,
        artifact_identity=spec.artifact_identity,
        canonical_model_source=spec.canonical_model_source,
        canonical_model_revision=spec.canonical_model_revision,
    )
    _load_stage_parameters_from_safetensors(
        model,
        resolved_spec,
        adapter=adapter,
        preloaded_checkpoint_tensors=preloaded,
        ignored_checkpoint_names=ignored,
    )
    model.model.config.num_hidden_layers = local_layers
    model.config.num_hidden_layers = local_layers
    model._gdlp_selective_stage_adapter = adapter
    model._gdlp_resolved_snapshot = str(snapshot)
    remaining_meta = tuple(
        name
        for name, tensor in (*model.named_parameters(), *model.named_buffers())
        if tensor.is_meta
    )
    if remaining_meta:
        raise RuntimeError(f"resident model still contains meta tensors: {remaining_meta}")
    return model.eval(), adapter


def _unique_meta_parameter_bytes(model: nn.Module) -> int:
    """Count unique parameter objects before any device storage is allocated."""

    seen: set[int] = set()
    total = 0
    for parameter in model.parameters():
        identity = id(parameter)
        if identity in seen:
            continue
        seen.add(identity)
        total += parameter.numel() * parameter.element_size()
    return total


__all__ = [
    "OnlineExpertTransitionPredictor",
    "RamBackedMoeExecutionSnapshot",
    "RamBackedMoeExperts",
    "RamBackedMoeStageRunner",
]
