"""Direct, fail-closed loading of local MoE experts from safetensors.

The regular selective model loader constructs a Hugging Face model and then
copies checkpoint tensors into it.  A RAM-backed expert cache does not need a
second resident copy of those routed weights.  This module therefore reads the
certified Qwen3-MoE or GLM4-MoE checkpoint layout directly, loads only the
selected sparse-stage tensors on CPU, and returns bundles that
``TorchRamExpertStore`` can adopt without cloning.

This is intentionally not a generic name-based loader.  Configuration
identity, stage geometry, every MLP key in the selected range, safetensors
index consistency, and the full local artifact identity are checked before a
bundle is exposed.  It never downloads a model or executes remote code.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from safetensors import safe_open
import torch
from torch.nn import functional as F
from transformers import AutoConfig

from .model import ModelArtifactReference, model_artifact_reference
from .model_adapters import (
    UnsupportedSelectiveStageArchitectureError,
    resolve_selective_stage_adapter,
)
from .ram_expert_cache import (
    ExpertKey,
    ExpertRecord,
    MacroStageExpertInventory,
    PredictiveCacheConfig,
)
from .torch_ram_expert_store import (
    TorchExpertBundle,
    TorchRamExpertStore,
)


LOCAL_MOE_STAGE_SCHEMA = "gdlp-local-safetensors-moe-stage/1"
_CANONICAL_EXPERT_NAMES = (
    "gate_proj.weight",
    "up_proj.weight",
    "down_proj.weight",
)
_SAFE_FLOAT_DTYPES = {"F16", "BF16", "F32"}


class UnsupportedLocalMoeStageError(ValueError):
    """The local artifact does not exactly match a certified stage layout."""


@dataclass(frozen=True)
class _ArchitectureSpec:
    adapter_id: str
    model_type: str
    architecture: str
    expert_count_field: str
    dense_prefix_field: str | None
    shared_count_field: str | None


_ARCHITECTURES = {
    "transformers-qwen3-moe-v1": _ArchitectureSpec(
        adapter_id="transformers-qwen3-moe-v1",
        model_type="qwen3_moe",
        architecture="Qwen3MoeForCausalLM",
        expert_count_field="num_experts",
        dense_prefix_field=None,
        shared_count_field=None,
    ),
    "transformers-glm4-moe-v1": _ArchitectureSpec(
        adapter_id="transformers-glm4-moe-v1",
        model_type="glm4_moe",
        architecture="Glm4MoeForCausalLM",
        expert_count_field="n_routed_experts",
        dense_prefix_field="first_k_dense_replace",
        shared_count_field="n_shared_experts",
    ),
}


@dataclass(frozen=True)
class CpuTensorArtifact:
    """Resident non-routed weights kept separate from the VRAM expert cache."""

    content_id: str
    layer: int
    kind: str
    tensors: tuple[tuple[str, torch.Tensor], ...]

    @property
    def byte_size(self) -> int:
        return sum(tensor.numel() * tensor.element_size() for _, tensor in self.tensors)

    def tensor(self, name: str) -> torch.Tensor:
        for tensor_name, tensor in self.tensors:
            if tensor_name == name:
                return tensor
        raise KeyError(f"tensor {name!r} is absent from {self.kind} layer {self.layer}")


@dataclass(frozen=True)
class LocalSafetensorsMoeStage:
    """Certified CPU tensors for one contiguous local macro-stage."""

    schema: str
    adapter_id: str
    model_type: str
    architecture: str
    artifact: ModelArtifactReference
    layer_start: int
    layer_end: int
    sparse_layers: tuple[int, ...]
    dense_layers: tuple[int, ...]
    storage_layout: str
    inventory: MacroStageExpertInventory
    expert_bundles: Mapping[ExpertKey, TorchExpertBundle]
    routers: Mapping[int, CpuTensorArtifact]
    shared_experts: Mapping[int, CpuTensorArtifact]
    source_dtype: torch.dtype
    loaded_backing_bytes: int
    preloaded_resident_artifacts: bool
    resident_streaming_transient_bytes: int
    bounded_pinned_staging_reserve_bytes: int
    host_ram_steady_state_bytes: int
    host_ram_peak_upper_bound_bytes: int

    @property
    def routed_expert_bytes(self) -> int:
        return self.inventory.total_bytes

    @property
    def resident_router_bytes(self) -> int:
        return sum(artifact.byte_size for artifact in self.routers.values())

    @property
    def resident_shared_expert_bytes(self) -> int:
        return sum(artifact.byte_size for artifact in self.shared_experts.values())

    def create_store(
        self,
        cache_config: PredictiveCacheConfig,
        *,
        device: str | torch.device = "cuda",
        pin_memory: bool = False,
        bounded_pinned_staging: bool = False,
        allow_cpu_fallback: bool = True,
    ) -> TorchRamExpertStore:
        """Transfer the loaded CPU buffers to a store without a RAM clone.

        After calling this method, consumers must treat ``expert_bundles`` as
        immutable. ``pin_memory=True`` still means a full pinned copy and is
        retained only for compatibility. ``bounded_pinned_staging=True`` keeps
        those authoritative buffers pageable and creates two reusable slots.
        """

        return TorchRamExpertStore.adopt_owned_cpu_bundles(
            self.inventory,
            self.expert_bundles,
            cache_config,
            device=device,
            pin_memory=pin_memory,
            bounded_pinned_staging=bounded_pinned_staging,
            allow_cpu_fallback=allow_cpu_fallback,
        )


@dataclass(frozen=True)
class _TensorMetadata:
    file_name: str
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class LocalSafetensorsMoeLayerMetadata:
    """Certified routed-expert geometry obtained without reading tensor bodies."""

    layer: int
    storage_layout: str | None
    expert_sizes_bytes: tuple[int, ...]

    @property
    def sparse(self) -> bool:
        return bool(self.expert_sizes_bytes)

    @property
    def routed_expert_bytes(self) -> int:
        return sum(self.expert_sizes_bytes)

    @property
    def largest_expert_bytes(self) -> int:
        return max(self.expert_sizes_bytes, default=0)


@dataclass(frozen=True)
class LocalSafetensorsMoeMetadata:
    """Metadata-only description of one fully certified local MoE checkpoint."""

    adapter_id: str
    model_type: str
    architecture: str
    total_layers: int
    expert_count: int
    experts_per_token: int
    source_dtype: str
    storage_layout: str
    layers: tuple[LocalSafetensorsMoeLayerMetadata, ...]

    @property
    def total_routed_expert_bytes(self) -> int:
        return sum(layer.routed_expert_bytes for layer in self.layers)

    @property
    def largest_expert_bytes(self) -> int:
        return max((layer.largest_expert_bytes for layer in self.layers), default=0)


class _CheckpointReader:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.metadata = _scan_safetensors_metadata(root)

    def names_with_prefix(self, prefix: str) -> set[str]:
        return {name for name in self.metadata if name.startswith(prefix)}

    def require(self, name: str, shape: Sequence[int]) -> _TensorMetadata:
        metadata = self.metadata.get(name)
        if metadata is None:
            raise UnsupportedLocalMoeStageError(
                f"required checkpoint tensor {name!r} is missing"
            )
        expected = tuple(int(value) for value in shape)
        if metadata.shape != expected:
            raise UnsupportedLocalMoeStageError(
                f"checkpoint tensor {name!r} has shape {metadata.shape}, expected {expected}"
            )
        if metadata.dtype not in _SAFE_FLOAT_DTYPES:
            raise UnsupportedLocalMoeStageError(
                f"checkpoint tensor {name!r} uses unsupported dtype {metadata.dtype!r}"
            )
        return metadata

    def load(self, name: str) -> torch.Tensor:
        metadata = self.metadata.get(name)
        if metadata is None:
            raise UnsupportedLocalMoeStageError(
                f"required checkpoint tensor {name!r} is missing"
            )
        with safe_open(
            self.root / metadata.file_name,
            framework="pt",
            device="cpu",
        ) as tensors:
            value = tensors.get_tensor(name)
        if (
            value.device.type != "cpu"
            or value.is_meta
            or value.layout != torch.strided
            or not value.is_floating_point()
            or not value.is_contiguous()
            or value.requires_grad
            or value.grad_fn is not None
        ):
            raise UnsupportedLocalMoeStageError(
                f"checkpoint tensor {name!r} is not a detached contiguous floating CPU tensor"
            )
        if tuple(value.shape) != metadata.shape:
            raise UnsupportedLocalMoeStageError(
                f"checkpoint tensor {name!r} changed shape while loading"
            )
        return value


def inspect_local_safetensors_moe_metadata(
    snapshot: str | Path,
    *,
    expected_adapter_id: str | None = None,
) -> LocalSafetensorsMoeMetadata:
    """Validate and describe a certified MoE artifact from local metadata only.

    This is the profiling counterpart of :func:`load_local_safetensors_moe_stage`.
    It reads ``config.json``, safetensors headers and an optional shard index;
    it never calls ``get_tensor`` and never resolves or downloads a Hub model.
    Unknown architectures, incomplete keys, mixed layouts, unsupported or mixed
    routed dtypes, and shape drift all fail closed.
    """

    root = Path(snapshot).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"local model snapshot does not exist: {root}")
    root = root.resolve()
    config_path = root / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"local model snapshot has no config.json: {root}")
    try:
        config_document = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise UnsupportedLocalMoeStageError(f"invalid local config.json: {error}") from error
    if not isinstance(config_document, dict):
        raise UnsupportedLocalMoeStageError("config.json must contain one JSON object")
    raw_model_type = config_document.get("model_type")
    raw_architectures = config_document.get("architectures")
    if raw_model_type not in {"qwen3_moe", "glm4_moe"} or raw_architectures not in (
        ["Qwen3MoeForCausalLM"],
        ["Glm4MoeForCausalLM"],
    ):
        raise UnsupportedLocalMoeStageError(
            "config.json does not declare one certified MoE architecture"
        )
    try:
        config = AutoConfig.from_pretrained(
            str(root),
            local_files_only=True,
            trust_remote_code=False,
        )
    except Exception as error:
        raise UnsupportedLocalMoeStageError(
            f"cannot construct the built-in local HF configuration: {error}"
        ) from error

    total_layers = _required_positive_int(config, "num_hidden_layers")
    try:
        adapter = resolve_selective_stage_adapter(config)
        adapter.validate_source_config(config, total_layers)
    except (UnsupportedSelectiveStageArchitectureError, TypeError, ValueError) as error:
        raise UnsupportedLocalMoeStageError(
            f"uncertified local Hugging Face MoE configuration: {error}"
        ) from error
    spec = _ARCHITECTURES.get(adapter.adapter_id)
    if spec is None:
        raise UnsupportedLocalMoeStageError(
            f"adapter {adapter.adapter_id!r} has no direct MoE stage loader"
        )
    if expected_adapter_id is not None and adapter.adapter_id != expected_adapter_id:
        raise UnsupportedLocalMoeStageError(
            f"adapter identity mismatch: expected {expected_adapter_id!r}, "
            f"got {adapter.adapter_id!r}"
        )

    expert_count = _required_positive_int(config, spec.expert_count_field)
    local_experts = getattr(config, "num_local_experts", expert_count)
    if local_experts != expert_count:
        raise UnsupportedLocalMoeStageError(
            "rank-local or partial expert shards are not accepted"
        )
    experts_per_token = _required_positive_int(config, "num_experts_per_tok")
    if experts_per_token > expert_count:
        raise UnsupportedLocalMoeStageError(
            "num_experts_per_tok exceeds the routed expert count"
        )
    hidden_size = _required_positive_int(config, "hidden_size")
    moe_intermediate = _required_positive_int(config, "moe_intermediate_size")
    dense_prefix = (
        0
        if spec.dense_prefix_field is None
        else _required_nonnegative_int(config, spec.dense_prefix_field)
    )
    shared_count = (
        0
        if spec.shared_count_field is None
        else _required_nonnegative_int(config, spec.shared_count_field)
    )
    dense_intermediate = (
        _required_positive_int(config, "intermediate_size") if dense_prefix else 0
    )

    reader = _CheckpointReader(root)
    layers: list[LocalSafetensorsMoeLayerMetadata] = []
    observed_layout: str | None = None
    observed_dtype: str | None = None
    for layer in range(total_layers):
        if layer < dense_prefix:
            _validate_dense_layer(
                reader,
                layer=layer,
                hidden_size=hidden_size,
                intermediate_size=dense_intermediate,
            )
            layers.append(
                LocalSafetensorsMoeLayerMetadata(
                    layer=layer,
                    storage_layout=None,
                    expert_sizes_bytes=(),
                )
            )
            continue

        layout = _validate_sparse_layer_names_and_shapes(
            reader,
            spec=spec,
            layer=layer,
            expert_count=expert_count,
            hidden_size=hidden_size,
            intermediate_size=moe_intermediate,
            shared_count=shared_count,
        )
        if observed_layout is None:
            observed_layout = layout
        elif layout != observed_layout:
            raise UnsupportedLocalMoeStageError(
                "mixing packed and unpacked expert layouts inside one artifact is unsupported"
            )
        routed_names = _routed_expert_tensor_names(
            layer=layer,
            layout=layout,
            expert_count=expert_count,
        )
        layer_dtypes = {reader.metadata[name].dtype for name in routed_names}
        if len(layer_dtypes) != 1:
            raise UnsupportedLocalMoeStageError(
                f"layer {layer} routed expert tensors use mixed dtypes {sorted(layer_dtypes)}"
            )
        layer_dtype = next(iter(layer_dtypes))
        if observed_dtype is None:
            observed_dtype = layer_dtype
        elif layer_dtype != observed_dtype:
            raise UnsupportedLocalMoeStageError(
                "routed expert dtype changes between sparse layers"
            )
        layers.append(
            LocalSafetensorsMoeLayerMetadata(
                layer=layer,
                storage_layout=layout,
                expert_sizes_bytes=_routed_expert_sizes_from_metadata(
                    reader,
                    layer=layer,
                    layout=layout,
                    expert_count=expert_count,
                ),
            )
        )

    if observed_layout is None or observed_dtype is None:
        raise UnsupportedLocalMoeStageError(
            "certified MoE artifact contains no sparse routed-expert layer"
        )
    return LocalSafetensorsMoeMetadata(
        adapter_id=adapter.adapter_id,
        model_type=spec.model_type,
        architecture=spec.architecture,
        total_layers=total_layers,
        expert_count=expert_count,
        experts_per_token=experts_per_token,
        source_dtype=observed_dtype,
        storage_layout=observed_layout,
        layers=tuple(layers),
    )


def load_local_safetensors_moe_stage(
    snapshot: str | Path,
    *,
    layer_start: int,
    layer_end: int,
    expected_artifact_identity: str | None = None,
    expected_adapter_id: str | None = None,
    expected_total_routed_expert_bytes: int | None = None,
    expected_largest_expert_bytes: int | None = None,
    preload_resident_artifacts: bool = True,
    reserve_bounded_pinned_staging: bool = False,
    expected_resident_streaming_transient_bytes: int | None = None,
    expected_bounded_pinned_staging_reserve_bytes: int | None = None,
    expected_host_ram_peak_bytes: int | None = None,
) -> LocalSafetensorsMoeStage:
    """Load only a certified sparse MoE range from a local checkpoint.

    ``snapshot`` must already exist locally.  The function never accepts a Hub
    repository identifier, calls ``snapshot_download``, or instantiates an HF
    model.  The returned buffers are suitable for the explicit no-clone store
    adoption path.

    The compatibility default preloads router/shared-expert artifacts.  The
    physical RAM-backed runner sets ``preload_resident_artifacts=False``: those
    tensors are still certified from safetensors headers here, but the regular
    selective loader later streams them directly into their device-resident
    destinations one tensor at a time.  This leaves only routed experts as the
    authoritative pageable-RAM working set.
    """

    root = Path(snapshot).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"local model snapshot does not exist: {root}")
    root = root.resolve()
    if not _is_nonnegative_int(layer_start):
        raise ValueError("layer_start must be a non-negative integer")
    if not _is_nonnegative_int(layer_end) or layer_end <= layer_start:
        raise ValueError("layer_end must be greater than layer_start")
    if not isinstance(preload_resident_artifacts, bool):
        raise TypeError("preload_resident_artifacts must be boolean")
    if not isinstance(reserve_bounded_pinned_staging, bool):
        raise TypeError("reserve_bounded_pinned_staging must be boolean")

    config_path = root / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"local model snapshot has no config.json: {root}")
    try:
        config_document = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise UnsupportedLocalMoeStageError(f"invalid local config.json: {error}") from error
    if not isinstance(config_document, dict):
        raise UnsupportedLocalMoeStageError("config.json must contain one JSON object")
    raw_model_type = config_document.get("model_type")
    raw_architectures = config_document.get("architectures")
    if raw_model_type not in {"qwen3_moe", "glm4_moe"} or raw_architectures not in (
        ["Qwen3MoeForCausalLM"],
        ["Glm4MoeForCausalLM"],
    ):
        raise UnsupportedLocalMoeStageError(
            "config.json does not declare one certified MoE architecture"
        )
    try:
        # Reconstruct built-in computed aliases/defaults (for example Qwen3's
        # ``num_experts`` property) without importing or executing remote code.
        config = AutoConfig.from_pretrained(
            str(root),
            local_files_only=True,
            trust_remote_code=False,
        )
    except Exception as error:
        raise UnsupportedLocalMoeStageError(
            f"cannot construct the built-in local HF configuration: {error}"
        ) from error
    total_layers = _required_positive_int(config, "num_hidden_layers")
    if layer_end > total_layers:
        raise ValueError(
            f"layer range [{layer_start}, {layer_end}) exceeds {total_layers} model layers"
        )

    try:
        adapter = resolve_selective_stage_adapter(config)
        adapter.validate_source_config(config, total_layers)
    except (UnsupportedSelectiveStageArchitectureError, TypeError, ValueError) as error:
        raise UnsupportedLocalMoeStageError(
            f"uncertified local Hugging Face MoE configuration: {error}"
        ) from error
    spec = _ARCHITECTURES.get(adapter.adapter_id)
    if spec is None:
        raise UnsupportedLocalMoeStageError(
            f"adapter {adapter.adapter_id!r} has no direct MoE stage loader"
        )
    if expected_adapter_id is not None and adapter.adapter_id != expected_adapter_id:
        raise UnsupportedLocalMoeStageError(
            f"adapter identity mismatch: expected {expected_adapter_id!r}, "
            f"got {adapter.adapter_id!r}"
        )
    for name, value in (
        ("expected_total_routed_expert_bytes", expected_total_routed_expert_bytes),
        ("expected_largest_expert_bytes", expected_largest_expert_bytes),
    ):
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value < 1
        ):
            raise ValueError(f"{name} must be a positive integer")
    for name, value in (
        (
            "expected_resident_streaming_transient_bytes",
            expected_resident_streaming_transient_bytes,
        ),
        (
            "expected_bounded_pinned_staging_reserve_bytes",
            expected_bounded_pinned_staging_reserve_bytes,
        ),
    ):
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value < 0
        ):
            raise ValueError(f"{name} must be a non-negative integer")
    if expected_host_ram_peak_bytes is not None and (
        not isinstance(expected_host_ram_peak_bytes, int)
        or isinstance(expected_host_ram_peak_bytes, bool)
        or expected_host_ram_peak_bytes < 1
    ):
        raise ValueError("expected_host_ram_peak_bytes must be a positive integer")

    expert_count = _required_positive_int(config, spec.expert_count_field)
    local_experts = getattr(config, "num_local_experts", expert_count)
    if local_experts != expert_count:
        raise UnsupportedLocalMoeStageError(
            "rank-local or partial expert shards are not accepted"
        )
    hidden_size = _required_positive_int(config, "hidden_size")
    moe_intermediate = _required_positive_int(config, "moe_intermediate_size")
    top_k = _required_positive_int(config, "num_experts_per_tok")
    if top_k > expert_count:
        raise UnsupportedLocalMoeStageError(
            "num_experts_per_tok exceeds the routed expert count"
        )
    dense_prefix = (
        0
        if spec.dense_prefix_field is None
        else _required_nonnegative_int(config, spec.dense_prefix_field)
    )
    shared_count = (
        0
        if spec.shared_count_field is None
        else _required_nonnegative_int(config, spec.shared_count_field)
    )

    sparse_layers = tuple(
        layer for layer in range(layer_start, layer_end) if layer >= dense_prefix
    )
    dense_layers = tuple(
        layer for layer in range(layer_start, layer_end) if layer < dense_prefix
    )
    if not sparse_layers:
        raise UnsupportedLocalMoeStageError(
            "selected stage contains no sparse MoE layer"
        )

    artifact = model_artifact_reference(str(root))
    if expected_artifact_identity is not None:
        expected = expected_artifact_identity.strip()
        if not expected:
            raise ValueError("expected_artifact_identity cannot be blank")
        if artifact.identity != expected:
            raise UnsupportedLocalMoeStageError(
                f"artifact identity mismatch: expected {expected!r}, got {artifact.identity!r}"
            )

    reader = _CheckpointReader(root)
    for layer in dense_layers:
        _validate_dense_layer(
            reader,
            layer=layer,
            hidden_size=hidden_size,
            intermediate_size=_required_positive_int(config, "intermediate_size"),
        )

    # Validate every routed tensor and its byte budget from safetensors headers
    # before loading a single expert into host RAM. This makes a stale planner
    # profile fail before the potentially multi-GiB allocation peak.
    layer_layouts: dict[int, str] = {}
    preflight_expert_sizes: list[int] = []
    routed_checkpoint_names: set[str] = set()
    observed_layout: str | None = None
    for layer in sparse_layers:
        layout = _validate_sparse_layer_names_and_shapes(
            reader,
            spec=spec,
            layer=layer,
            expert_count=expert_count,
            hidden_size=hidden_size,
            intermediate_size=moe_intermediate,
            shared_count=shared_count,
        )
        if observed_layout is None:
            observed_layout = layout
        elif layout != observed_layout:
            raise UnsupportedLocalMoeStageError(
                "mixing packed and unpacked expert layouts inside one stage is unsupported"
            )
        layer_layouts[layer] = layout
        routed_checkpoint_names.update(
            _routed_expert_tensor_names(
                layer=layer,
                layout=layout,
                expert_count=expert_count,
            )
        )
        preflight_expert_sizes.extend(
            _routed_expert_sizes_from_metadata(
                reader,
                layer=layer,
                layout=layout,
                expert_count=expert_count,
            )
        )
    preflight_total = sum(preflight_expert_sizes)
    preflight_largest = max(preflight_expert_sizes)
    if (
        expected_total_routed_expert_bytes is not None
        and preflight_total != expected_total_routed_expert_bytes
    ):
        raise UnsupportedLocalMoeStageError(
            "routed expert byte budget mismatch: "
            f"expected {expected_total_routed_expert_bytes}, got {preflight_total}"
        )
    if (
        expected_largest_expert_bytes is not None
        and preflight_largest != expected_largest_expert_bytes
    ):
        raise UnsupportedLocalMoeStageError(
            "largest expert byte budget mismatch: "
            f"expected {expected_largest_expert_bytes}, got {preflight_largest}"
        )

    preloaded_resident_names = (
        _preloaded_resident_checkpoint_names(
            sparse_layers=sparse_layers,
            model_type=spec.model_type,
        )
        if preload_resident_artifacts
        else set()
    )
    relevant_checkpoint_names = {
        name
        for name in reader.metadata
        if _checkpoint_tensor_belongs_to_stage(
            name,
            layer_start=layer_start,
            layer_end=layer_end,
            total_layers=total_layers,
        )
    }
    streamed_resident_names = (
        relevant_checkpoint_names
        - routed_checkpoint_names
        - preloaded_resident_names
    )
    resident_streaming_transient_bytes = max(
        (
            _any_tensor_metadata_bytes(reader.metadata[name])
            for name in streamed_resident_names
        ),
        default=0,
    )
    preloaded_resident_bytes = sum(
        _any_tensor_metadata_bytes(reader.metadata[name])
        for name in preloaded_resident_names
    )
    bounded_pinned_staging_reserve_bytes = (
        preflight_largest * 2 if reserve_bounded_pinned_staging else 0
    )
    host_ram_steady_state_bytes = (
        preflight_total
        + preloaded_resident_bytes
        + bounded_pinned_staging_reserve_bytes
    )
    host_ram_peak_upper_bound_bytes = (
        host_ram_steady_state_bytes + resident_streaming_transient_bytes
    )
    if (
        expected_resident_streaming_transient_bytes is not None
        and resident_streaming_transient_bytes
        > expected_resident_streaming_transient_bytes
    ):
        raise MemoryError(
            "resident streaming transient budget would be exceeded before tensor load: "
            f"required {resident_streaming_transient_bytes}, budget "
            f"{expected_resident_streaming_transient_bytes}"
        )
    if (
        expected_bounded_pinned_staging_reserve_bytes is not None
        and bounded_pinned_staging_reserve_bytes
        != expected_bounded_pinned_staging_reserve_bytes
    ):
        raise MemoryError(
            "bounded pinned staging reserve contract mismatch before tensor load: "
            f"required {bounded_pinned_staging_reserve_bytes}, expected "
            f"{expected_bounded_pinned_staging_reserve_bytes}"
        )
    if (
        expected_host_ram_peak_bytes is not None
        and host_ram_peak_upper_bound_bytes > expected_host_ram_peak_bytes
    ):
        raise MemoryError(
            "RAM-backed MoE host RAM peak budget would be exceeded before tensor load: "
            f"required {host_ram_peak_upper_bound_bytes}, budget "
            f"{expected_host_ram_peak_bytes} "
            f"(routed={preflight_total}, "
            f"preloaded_resident={preloaded_resident_bytes}, "
            f"pinned_staging={bounded_pinned_staging_reserve_bytes}, "
            f"streaming_transient={resident_streaming_transient_bytes})"
        )

    records: list[ExpertRecord] = []
    expert_bundles: dict[ExpertKey, TorchExpertBundle] = {}
    routers: dict[int, CpuTensorArtifact] = {}
    shared_experts: dict[int, CpuTensorArtifact] = {}
    observed_dtype: torch.dtype | None = None

    for layer in sparse_layers:
        layout = layer_layouts[layer]

        prefix = f"model.layers.{layer}.mlp"
        if preload_resident_artifacts:
            router_items = [("weight", reader.load(f"{prefix}.gate.weight"))]
            if spec.model_type == "glm4_moe":
                router_items.append(
                    (
                        "e_score_correction_bias",
                        reader.load(f"{prefix}.gate.e_score_correction_bias"),
                    )
                )
            routers[layer] = CpuTensorArtifact(
                content_id=(
                    f"{artifact.identity}|{adapter.adapter_id}|layer={layer}|router-v1"
                ),
                layer=layer,
                kind="router",
                tensors=tuple(router_items),
            )

        layer_bundles = (
            _load_packed_experts(
                reader,
                prefix=prefix,
                layer=layer,
                expert_count=expert_count,
                intermediate_size=moe_intermediate,
                artifact_identity=artifact.identity,
                adapter_id=adapter.adapter_id,
            )
            if layout == "grouped-3d-swiglu"
            else _load_unpacked_experts(
                reader,
                prefix=prefix,
                layer=layer,
                expert_count=expert_count,
                artifact_identity=artifact.identity,
                adapter_id=adapter.adapter_id,
            )
        )
        for bundle in layer_bundles:
            dtype = _validate_expert_bundle(bundle)
            if observed_dtype is None:
                observed_dtype = dtype
            elif dtype != observed_dtype:
                raise UnsupportedLocalMoeStageError(
                    "all routed expert weights in one stage must use one dtype"
                )
            records.append(
                ExpertRecord(
                    key=bundle.key,
                    byte_size=bundle.byte_size,
                    content_id=bundle.content_id,
                )
            )
            expert_bundles[bundle.key] = bundle

        if spec.model_type == "glm4_moe" and preload_resident_artifacts:
            shared_items = tuple(
                (
                    name,
                    reader.load(f"{prefix}.shared_experts.{name}"),
                )
                for name in _CANONICAL_EXPERT_NAMES
            )
            shared = CpuTensorArtifact(
                content_id=(
                    f"{artifact.identity}|{adapter.adapter_id}|layer={layer}|shared-swiglu-v1"
                ),
                layer=layer,
                kind="shared-expert",
                tensors=shared_items,
            )
            for _, tensor in shared.tensors:
                if observed_dtype is not None and tensor.dtype != observed_dtype:
                    raise UnsupportedLocalMoeStageError(
                        "shared and routed expert weights must use one dtype"
                    )
            shared_experts[layer] = shared

    assert observed_layout is not None and observed_dtype is not None
    inventory = MacroStageExpertInventory(tuple(records))
    all_tensors = [
        tensor
        for bundle in expert_bundles.values()
        for _, tensor in bundle.tensors
    ]
    all_tensors.extend(
        tensor for artifact_item in routers.values() for _, tensor in artifact_item.tensors
    )
    all_tensors.extend(
        tensor
        for artifact_item in shared_experts.values()
        for _, tensor in artifact_item.tensors
    )
    return LocalSafetensorsMoeStage(
        schema=LOCAL_MOE_STAGE_SCHEMA,
        adapter_id=adapter.adapter_id,
        model_type=spec.model_type,
        architecture=spec.architecture,
        artifact=artifact,
        layer_start=layer_start,
        layer_end=layer_end,
        sparse_layers=sparse_layers,
        dense_layers=dense_layers,
        storage_layout=observed_layout,
        inventory=inventory,
        expert_bundles=MappingProxyType(dict(expert_bundles)),
        routers=MappingProxyType(dict(routers)),
        shared_experts=MappingProxyType(dict(shared_experts)),
        source_dtype=observed_dtype,
        loaded_backing_bytes=_unique_storage_bytes(all_tensors),
        preloaded_resident_artifacts=preload_resident_artifacts,
        resident_streaming_transient_bytes=resident_streaming_transient_bytes,
        bounded_pinned_staging_reserve_bytes=(
            bounded_pinned_staging_reserve_bytes
        ),
        host_ram_steady_state_bytes=host_ram_steady_state_bytes,
        host_ram_peak_upper_bound_bytes=host_ram_peak_upper_bound_bytes,
    )


def execute_swiglu_expert_cpu(
    bundle: TorchExpertBundle,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    """Execute one canonical extracted expert on CPU as an exact reference.

    This deliberately favors a small auditable definition over a fused kernel.
    Production implementations may fuse the three GEMMs and activation, but
    must preserve this operation and tensor ordering.
    """

    if not isinstance(bundle, TorchExpertBundle):
        raise TypeError("bundle must be TorchExpertBundle")
    if tuple(name for name, _ in bundle.tensors) != _CANONICAL_EXPERT_NAMES:
        raise UnsupportedLocalMoeStageError(
            "bundle must use canonical gate/up/down SwiGLU tensor order"
        )
    if not isinstance(hidden_states, torch.Tensor):
        raise TypeError("hidden_states must be torch.Tensor")
    if (
        hidden_states.device.type != "cpu"
        or hidden_states.is_meta
        or hidden_states.layout != torch.strided
        or not hidden_states.is_floating_point()
        or hidden_states.ndim < 1
    ):
        raise ValueError("hidden_states must be a resident strided floating CPU tensor")
    gate = bundle.tensor("gate_proj.weight")
    up = bundle.tensor("up_proj.weight")
    down = bundle.tensor("down_proj.weight")
    dtype = _validate_expert_bundle(bundle)
    if hidden_states.dtype != dtype:
        raise ValueError("hidden_states and expert weights must use the same dtype")
    if hidden_states.shape[-1] != gate.shape[1]:
        raise ValueError(
            f"hidden width {hidden_states.shape[-1]} does not match expert width {gate.shape[1]}"
        )
    return F.linear(F.silu(F.linear(hidden_states, gate)) * F.linear(hidden_states, up), down)


def _validate_dense_layer(
    reader: _CheckpointReader,
    *,
    layer: int,
    hidden_size: int,
    intermediate_size: int,
) -> None:
    prefix = f"model.layers.{layer}.mlp"
    expected = {
        f"{prefix}.gate_proj.weight": (intermediate_size, hidden_size),
        f"{prefix}.up_proj.weight": (intermediate_size, hidden_size),
        f"{prefix}.down_proj.weight": (hidden_size, intermediate_size),
    }
    actual = reader.names_with_prefix(prefix + ".")
    if actual != set(expected):
        _raise_key_mismatch(layer, "dense MLP", actual, set(expected))
    for name, shape in expected.items():
        reader.require(name, shape)


def _validate_sparse_layer_names_and_shapes(
    reader: _CheckpointReader,
    *,
    spec: _ArchitectureSpec,
    layer: int,
    expert_count: int,
    hidden_size: int,
    intermediate_size: int,
    shared_count: int,
) -> str:
    prefix = f"model.layers.{layer}.mlp"
    router = {f"{prefix}.gate.weight": (expert_count, hidden_size)}
    if spec.model_type == "glm4_moe":
        router[f"{prefix}.gate.e_score_correction_bias"] = (expert_count,)
    shared: dict[str, tuple[int, ...]] = {}
    if spec.model_type == "glm4_moe":
        shared_width = intermediate_size * shared_count
        shared = {
            f"{prefix}.shared_experts.gate_proj.weight": (shared_width, hidden_size),
            f"{prefix}.shared_experts.up_proj.weight": (shared_width, hidden_size),
            f"{prefix}.shared_experts.down_proj.weight": (hidden_size, shared_width),
        }

    packed = {
        f"{prefix}.experts.gate_up_proj": (
            expert_count,
            intermediate_size * 2,
            hidden_size,
        ),
        f"{prefix}.experts.down_proj": (
            expert_count,
            hidden_size,
            intermediate_size,
        ),
    }
    unpacked: dict[str, tuple[int, ...]] = {}
    for expert in range(expert_count):
        expert_prefix = f"{prefix}.experts.{expert}"
        unpacked.update(
            {
                f"{expert_prefix}.gate_proj.weight": (intermediate_size, hidden_size),
                f"{expert_prefix}.up_proj.weight": (intermediate_size, hidden_size),
                f"{expert_prefix}.down_proj.weight": (hidden_size, intermediate_size),
            }
        )

    actual = reader.names_with_prefix(prefix + ".")
    packed_expected = set(router) | set(shared) | set(packed)
    unpacked_expected = set(router) | set(shared) | set(unpacked)
    if actual == packed_expected:
        layout = "grouped-3d-swiglu"
        shapes = {**router, **shared, **packed}
    elif actual == unpacked_expected:
        layout = "unpacked-per-expert-swiglu"
        shapes = {**router, **shared, **unpacked}
    else:
        expected = packed_expected if actual.intersection(packed) else unpacked_expected
        _raise_key_mismatch(layer, "sparse MLP", actual, expected)
        raise AssertionError("unreachable")
    for name, shape in shapes.items():
        reader.require(name, shape)
    return layout


def _routed_expert_sizes_from_metadata(
    reader: _CheckpointReader,
    *,
    layer: int,
    layout: str,
    expert_count: int,
) -> tuple[int, ...]:
    """Return exact per-expert bytes using headers only (no tensor load)."""

    prefix = f"model.layers.{layer}.mlp.experts"
    if layout == "grouped-3d-swiglu":
        names = (
            f"{prefix}.gate_up_proj",
            f"{prefix}.down_proj",
        )
        total = sum(_tensor_metadata_bytes(reader.metadata[name]) for name in names)
        if total % expert_count:
            raise UnsupportedLocalMoeStageError(
                f"packed routed expert bytes do not divide across {expert_count} experts"
            )
        return (total // expert_count,) * expert_count
    if layout != "unpacked-per-expert-swiglu":
        raise UnsupportedLocalMoeStageError(
            f"unsupported routed expert storage layout {layout!r}"
        )
    return tuple(
        sum(
            _tensor_metadata_bytes(
                reader.metadata[f"{prefix}.{expert}.{name}"]
            )
            for name in _CANONICAL_EXPERT_NAMES
        )
        for expert in range(expert_count)
    )


def _routed_expert_tensor_names(
    *,
    layer: int,
    layout: str,
    expert_count: int,
) -> tuple[str, ...]:
    prefix = f"model.layers.{layer}.mlp.experts"
    if layout == "grouped-3d-swiglu":
        return (
            f"{prefix}.gate_up_proj",
            f"{prefix}.down_proj",
        )
    if layout == "unpacked-per-expert-swiglu":
        return tuple(
            f"{prefix}.{expert}.{name}"
            for expert in range(expert_count)
            for name in _CANONICAL_EXPERT_NAMES
        )
    raise UnsupportedLocalMoeStageError(
        f"unsupported routed expert storage layout {layout!r}"
    )


def _preloaded_resident_checkpoint_names(
    *,
    sparse_layers: Sequence[int],
    model_type: str,
) -> set[str]:
    """Return the exact compatibility-only router/shared preload set."""

    names: set[str] = set()
    for layer in sparse_layers:
        prefix = f"model.layers.{layer}.mlp"
        names.add(f"{prefix}.gate.weight")
        if model_type == "glm4_moe":
            names.add(f"{prefix}.gate.e_score_correction_bias")
            names.update(
                f"{prefix}.shared_experts.{name}"
                for name in _CANONICAL_EXPERT_NAMES
            )
    return names


def _checkpoint_tensor_belongs_to_stage(
    name: str,
    *,
    layer_start: int,
    layer_end: int,
    total_layers: int,
) -> bool:
    """Mirror the selective loader's stage ownership using header names only."""

    layer_prefix = "model.layers."
    if name.startswith(layer_prefix):
        remainder = name[len(layer_prefix) :]
        index_text, separator, _ = remainder.partition(".")
        if not separator:
            raise UnsupportedLocalMoeStageError(
                f"invalid checkpoint layer tensor name {name!r}"
            )
        try:
            layer = int(index_text)
        except ValueError as error:
            raise UnsupportedLocalMoeStageError(
                f"invalid checkpoint layer tensor name {name!r}"
            ) from error
        return layer_start <= layer < layer_end
    if name.startswith("model.embed_tokens."):
        return layer_start == 0
    if name.startswith("model.norm."):
        return layer_end == total_layers
    if name.startswith("model."):
        return True
    if name.startswith("lm_head."):
        return layer_end == total_layers
    return False


def _tensor_metadata_bytes(metadata: _TensorMetadata) -> int:
    element_bytes = {"F16": 2, "BF16": 2, "F32": 4}.get(metadata.dtype)
    if element_bytes is None:
        raise UnsupportedLocalMoeStageError(
            f"unsupported safetensors dtype {metadata.dtype!r} in byte preflight"
        )
    elements = 1
    for dimension in metadata.shape:
        elements *= dimension
    return elements * element_bytes


def _any_tensor_metadata_bytes(metadata: _TensorMetadata) -> int:
    """Return exact safetensors storage bytes for a resident tensor header."""

    element_bytes = {
        "BOOL": 1,
        "I8": 1,
        "U8": 1,
        "I16": 2,
        "U16": 2,
        "F16": 2,
        "BF16": 2,
        "I32": 4,
        "U32": 4,
        "F32": 4,
        "I64": 8,
        "U64": 8,
        "F64": 8,
        "F8_E4M3": 1,
        "F8_E5M2": 1,
    }.get(metadata.dtype)
    if element_bytes is None:
        raise UnsupportedLocalMoeStageError(
            f"unsupported safetensors dtype {metadata.dtype!r} in host RAM preflight"
        )
    elements = 1
    for dimension in metadata.shape:
        elements *= dimension
    return elements * element_bytes


def _load_packed_experts(
    reader: _CheckpointReader,
    *,
    prefix: str,
    layer: int,
    expert_count: int,
    intermediate_size: int,
    artifact_identity: str,
    adapter_id: str,
) -> tuple[TorchExpertBundle, ...]:
    gate_up = reader.load(f"{prefix}.experts.gate_up_proj")
    down = reader.load(f"{prefix}.experts.down_proj")
    bundles = []
    for expert in range(expert_count):
        key = ExpertKey(layer, expert)
        bundles.append(
            TorchExpertBundle(
                key=key,
                content_id=(
                    f"{artifact_identity}|{adapter_id}|layer={layer}|expert={expert}|swiglu-v1"
                ),
                tensors=(
                    ("gate_proj.weight", gate_up[expert, :intermediate_size, :]),
                    ("up_proj.weight", gate_up[expert, intermediate_size:, :]),
                    ("down_proj.weight", down[expert]),
                ),
            )
        )
    return tuple(bundles)


def _load_unpacked_experts(
    reader: _CheckpointReader,
    *,
    prefix: str,
    layer: int,
    expert_count: int,
    artifact_identity: str,
    adapter_id: str,
) -> tuple[TorchExpertBundle, ...]:
    bundles = []
    for expert in range(expert_count):
        key = ExpertKey(layer, expert)
        expert_prefix = f"{prefix}.experts.{expert}"
        bundles.append(
            TorchExpertBundle(
                key=key,
                content_id=(
                    f"{artifact_identity}|{adapter_id}|layer={layer}|expert={expert}|swiglu-v1"
                ),
                tensors=tuple(
                    (name, reader.load(f"{expert_prefix}.{name}"))
                    for name in _CANONICAL_EXPERT_NAMES
                ),
            )
        )
    return tuple(bundles)


def _validate_expert_bundle(bundle: TorchExpertBundle) -> torch.dtype:
    if tuple(name for name, _ in bundle.tensors) != _CANONICAL_EXPERT_NAMES:
        raise UnsupportedLocalMoeStageError(
            f"expert {bundle.key} does not use canonical SwiGLU tensor order"
        )
    gate, up, down = (tensor for _, tensor in bundle.tensors)
    if gate.ndim != 2 or up.shape != gate.shape or down.shape != (
        gate.shape[1],
        gate.shape[0],
    ):
        raise UnsupportedLocalMoeStageError(
            f"expert {bundle.key} has inconsistent SwiGLU geometry"
        )
    for tensor in (gate, up, down):
        if (
            tensor.device.type != "cpu"
            or tensor.is_meta
            or tensor.layout != torch.strided
            or not tensor.is_floating_point()
            or not tensor.is_contiguous()
            or tensor.requires_grad
            or tensor.grad_fn is not None
            or tensor.dtype != gate.dtype
        ):
            raise UnsupportedLocalMoeStageError(
                f"expert {bundle.key} must contain detached contiguous CPU weights of one dtype"
            )
    return gate.dtype


def _scan_safetensors_metadata(root: Path) -> dict[str, _TensorMetadata]:
    files = sorted(root.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors checkpoint found under {root}")
    metadata: dict[str, _TensorMetadata] = {}
    actual_weight_map: dict[str, str] = {}
    for file_path in files:
        if not file_path.is_file():
            raise FileNotFoundError(f"checkpoint shard is not a file: {file_path}")
        try:
            with safe_open(file_path, framework="pt", device="cpu") as tensors:
                for name in tensors.keys():
                    if not isinstance(name, str) or not name:
                        raise UnsupportedLocalMoeStageError(
                            f"checkpoint shard {file_path.name} contains an invalid tensor name"
                        )
                    if name in metadata:
                        raise UnsupportedLocalMoeStageError(
                            f"checkpoint tensor {name!r} appears in multiple shards"
                        )
                    tensor_slice = tensors.get_slice(name)
                    metadata[name] = _TensorMetadata(
                        file_name=file_path.name,
                        shape=tuple(int(value) for value in tensor_slice.get_shape()),
                        dtype=str(tensor_slice.get_dtype()),
                    )
                    actual_weight_map[name] = file_path.name
        except UnsupportedLocalMoeStageError:
            raise
        except Exception as error:
            raise UnsupportedLocalMoeStageError(
                f"cannot inspect safetensors shard {file_path.name}: {error}"
            ) from error

    indexes = sorted(root.glob("*.safetensors.index.json"))
    if len(indexes) > 1:
        raise UnsupportedLocalMoeStageError(
            f"multiple safetensors indexes found under {root}"
        )
    if indexes:
        try:
            document = json.loads(indexes[0].read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise UnsupportedLocalMoeStageError(
                f"invalid safetensors index: {error}"
            ) from error
        weight_map = document.get("weight_map") if isinstance(document, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise UnsupportedLocalMoeStageError("safetensors index has no weight_map")
        normalized: dict[str, str] = {}
        for name, file_name in weight_map.items():
            if (
                not isinstance(name, str)
                or not name
                or not isinstance(file_name, str)
                or not file_name
            ):
                raise UnsupportedLocalMoeStageError(
                    "safetensors index weight_map must contain string keys and files"
                )
            relative = Path(file_name)
            if (
                relative.is_absolute()
                or len(relative.parts) != 1
                or relative.suffix != ".safetensors"
            ):
                raise UnsupportedLocalMoeStageError(
                    f"invalid checkpoint shard path in index: {file_name!r}"
                )
            normalized[name] = file_name
        if normalized != actual_weight_map:
            missing = sorted(set(actual_weight_map) - set(normalized))
            extra = sorted(set(normalized) - set(actual_weight_map))
            wrong = sorted(
                name
                for name in set(normalized).intersection(actual_weight_map)
                if normalized[name] != actual_weight_map[name]
            )
            raise UnsupportedLocalMoeStageError(
                "safetensors index does not exactly match shard headers "
                f"(missing={missing[:4]}, extra={extra[:4]}, wrongShard={wrong[:4]})"
            )
    return metadata


def _raise_key_mismatch(
    layer: int,
    kind: str,
    actual: set[str],
    expected: set[str],
) -> None:
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    raise UnsupportedLocalMoeStageError(
        f"layer {layer} {kind} checkpoint keys are not certified "
        f"(missing={missing[:6]}, extra={extra[:6]})"
    )


def _unique_storage_bytes(tensors: Sequence[torch.Tensor]) -> int:
    storages: dict[tuple[int, int], int] = {}
    for tensor in tensors:
        storage = tensor.untyped_storage()
        key = (storage.data_ptr(), storage.nbytes())
        storages[key] = storage.nbytes()
    return sum(storages.values())


def _is_nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _required_positive_int(config: Any, name: str) -> int:
    value = getattr(config, name, None)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise UnsupportedLocalMoeStageError(
            f"config.{name} must be a positive integer"
        )
    return value


def _required_nonnegative_int(config: Any, name: str) -> int:
    value = getattr(config, name, None)
    if not _is_nonnegative_int(value):
        raise UnsupportedLocalMoeStageError(
            f"config.{name} must be a non-negative integer"
        )
    return int(value)


__all__ = [
    "LOCAL_MOE_STAGE_SCHEMA",
    "CpuTensorArtifact",
    "LocalSafetensorsMoeLayerMetadata",
    "LocalSafetensorsMoeMetadata",
    "LocalSafetensorsMoeStage",
    "UnsupportedLocalMoeStageError",
    "execute_swiglu_expert_cpu",
    "inspect_local_safetensors_moe_metadata",
    "load_local_safetensors_moe_stage",
]
