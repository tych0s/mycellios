"""Fail-closed extraction of routed MoE weights from Hugging Face modules.

The extractor bridges certified Transformers model layouts and the RAM-backed
expert cache.  It deliberately recognizes only the Qwen3-MoE and GLM4-MoE
configuration identities already certified by :mod:`model_adapters`.  Routed
experts become canonical SwiGLU tensor bundles while routers and GLM shared
experts remain separate, resident objects.

No model is downloaded and no attribute-name heuristic is used as a fallback.
An unknown architecture, a partial expert shard, an unexpected parameter, or
an ambiguous decoder root aborts extraction before any cache is constructed.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from .model_adapters import (
    UnsupportedSelectiveStageArchitectureError,
    resolve_selective_stage_adapter,
)
from .ram_expert_cache import ExpertKey, ExpertRecord, MacroStageExpertInventory


HF_MOE_EXTRACTOR_SCHEMA = "gdlp-hf-moe-extractor/1"
_CANONICAL_TENSOR_NAMES = (
    "gate_proj.weight",
    "up_proj.weight",
    "down_proj.weight",
)


class UnsupportedHfMoeExpertLayoutError(ValueError):
    """The model does not exactly match a certified MoE tensor layout."""


@dataclass(frozen=True)
class _ArchitectureSpec:
    adapter_id: str
    model_type: str
    architecture: str
    expert_count_field: str
    dense_prefix_field: str | None
    shared_count_field: str | None


_ARCHITECTURES = (
    _ArchitectureSpec(
        adapter_id="transformers-qwen3-moe-v1",
        model_type="qwen3_moe",
        architecture="Qwen3MoeForCausalLM",
        expert_count_field="num_experts",
        dense_prefix_field=None,
        shared_count_field=None,
    ),
    _ArchitectureSpec(
        adapter_id="transformers-glm4-moe-v1",
        model_type="glm4_moe",
        architecture="Glm4MoeForCausalLM",
        expert_count_field="n_routed_experts",
        dense_prefix_field="first_k_dense_replace",
        shared_count_field="n_shared_experts",
    ),
)
_ARCHITECTURE_BY_ADAPTER = {item.adapter_id: item for item in _ARCHITECTURES}


TensorBundle = Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class SharedExpertTensorBundle:
    """One resident shared-expert artifact, never copied into routed bundles.

    ``layers`` may contain more than one layer only when the source model ties
    the exact same tensor storage between those layers.  Such aliases are
    represented once rather than charged or copied twice.
    """

    content_id: str
    layers: tuple[int, ...]
    module_paths: tuple[str, ...]
    tensors: TensorBundle

    @property
    def byte_size(self) -> int:
        return _tensor_bundle_bytes(self.tensors)


@dataclass(frozen=True)
class HfMoeExpertExtraction:
    """Canonical routed weights plus resident objects for one local stage."""

    adapter_id: str
    model_type: str
    architecture: str
    model_identity: str
    storage_layout: str
    global_layer_start: int
    global_layer_end: int
    sparse_layers: tuple[int, ...]
    dense_layers: tuple[int, ...]
    inventory: MacroStageExpertInventory
    expert_tensors: Mapping[ExpertKey, TensorBundle]
    routers: Mapping[int, nn.Module]
    shared_experts: tuple[SharedExpertTensorBundle, ...]
    shared_expert_content_by_layer: Mapping[int, str]
    source_device: torch.device
    source_dtype: torch.dtype

    @property
    def routed_expert_bytes(self) -> int:
        return self.inventory.total_bytes

    @property
    def shared_expert_bytes(self) -> int:
        return sum(bundle.byte_size for bundle in self.shared_experts)


@dataclass
class _SharedBuilder:
    content_id: str
    layers: list[int]
    module_paths: list[str]
    tensors: TensorBundle


def extract_hf_moe_experts(
    model: nn.Module,
    *,
    model_identity: str,
    global_layer_start: int = 0,
) -> HfMoeExpertExtraction:
    """Extract exact routed expert tensors from a resident HF decoder stage.

    ``model`` may be a ``*ForCausalLM`` wrapper exposing ``model.layers`` or
    its decoder object exposing ``layers`` directly.  ``global_layer_start``
    maps a selectively loaded local stage back to global layer identifiers.
    The explicit content-addressed ``model_identity`` prevents cache identities
    from being inferred from a mutable local path.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("model must be torch.nn.Module")
    identity = _nonempty_string(model_identity, "model_identity")
    if (
        not isinstance(global_layer_start, int)
        or isinstance(global_layer_start, bool)
        or global_layer_start < 0
    ):
        raise ValueError("global_layer_start must be a non-negative integer")

    config = getattr(model, "config", None)
    if config is None:
        raise UnsupportedHfMoeExpertLayoutError("model.config is required")
    layers, layer_path = _resolve_decoder_layers(model)
    try:
        adapter = resolve_selective_stage_adapter(config)
        adapter.validate_source_config(config, len(layers))
    except (UnsupportedSelectiveStageArchitectureError, TypeError, ValueError) as error:
        raise UnsupportedHfMoeExpertLayoutError(
            f"uncertified Hugging Face MoE configuration: {error}"
        ) from error

    spec = _ARCHITECTURE_BY_ADAPTER.get(adapter.adapter_id)
    if spec is None:
        raise UnsupportedHfMoeExpertLayoutError(
            f"adapter {adapter.adapter_id!r} has no RAM expert extractor"
        )
    expert_count = _required_positive_int(config, spec.expert_count_field)
    local_expert_count = getattr(config, "num_local_experts", expert_count)
    if local_expert_count != expert_count:
        raise UnsupportedHfMoeExpertLayoutError(
            "rank-local expert shards are not accepted by the RAM cache extractor"
        )
    hidden_size = _required_positive_int(config, "hidden_size")
    intermediate_size = _required_positive_int(config, "moe_intermediate_size")
    top_k = _required_positive_int(config, "num_experts_per_tok")
    if top_k > expert_count:
        raise UnsupportedHfMoeExpertLayoutError(
            "num_experts_per_tok exceeds the routed expert count"
        )

    dense_prefix = (
        0
        if spec.dense_prefix_field is None
        else _required_nonnegative_int(config, spec.dense_prefix_field)
    )
    if dense_prefix > len(layers):
        raise UnsupportedHfMoeExpertLayoutError(
            "dense prefix exceeds the resident decoder layer count"
        )
    shared_count = (
        0
        if spec.shared_count_field is None
        else _required_nonnegative_int(config, spec.shared_count_field)
    )

    records: list[ExpertRecord] = []
    tensor_bundles: dict[ExpertKey, TensorBundle] = {}
    routers: dict[int, nn.Module] = {}
    sparse_layers: list[int] = []
    dense_layers: list[int] = []
    observed_layout: str | None = None
    observed_dtype: torch.dtype | None = None
    observed_device: torch.device | None = None
    shared_builders: dict[tuple[tuple[object, ...], ...], _SharedBuilder] = {}
    shared_by_layer: dict[int, str] = {}

    for local_index, layer in enumerate(layers):
        global_index = global_layer_start + local_index
        if not isinstance(layer, nn.Module):
            raise UnsupportedHfMoeExpertLayoutError(
                f"{layer_path}.{local_index} is not torch.nn.Module"
            )
        mlp = getattr(layer, "mlp", None)
        if not isinstance(mlp, nn.Module):
            raise UnsupportedHfMoeExpertLayoutError(
                f"{layer_path}.{local_index}.mlp is missing"
            )

        if local_index < dense_prefix:
            _validate_dense_mlp(
                mlp,
                hidden_size=hidden_size,
                intermediate_size=_required_positive_int(config, "intermediate_size"),
                path=f"{layer_path}.{local_index}.mlp",
            )
            dense_layers.append(global_index)
            continue

        sparse_layers.append(global_index)
        router = _validate_router(
            mlp,
            expert_count=expert_count,
            top_k=top_k,
            hidden_size=hidden_size,
            require_glm_correction=spec.model_type == "glm4_moe",
            path=f"{layer_path}.{local_index}.mlp.gate",
        )
        routers[global_index] = router
        layer_bundles, layout = _extract_routed_layer(
            mlp,
            expert_count=expert_count,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            path=f"{layer_path}.{local_index}.mlp.experts",
        )
        if observed_layout is None:
            observed_layout = layout
        elif layout != observed_layout:
            raise UnsupportedHfMoeExpertLayoutError(
                "mixing grouped and ModuleList expert layouts inside one stage is unsupported"
            )

        for expert_index, tensors in enumerate(layer_bundles):
            dtype, device = _validate_tensor_group(
                tensors,
                path=f"{layer_path}.{local_index}.mlp.experts[{expert_index}]",
            )
            if observed_dtype is None:
                observed_dtype, observed_device = dtype, device
            elif dtype != observed_dtype or device != observed_device:
                raise UnsupportedHfMoeExpertLayoutError(
                    "all routed and shared expert tensors must use one dtype and device"
                )
            key = ExpertKey(global_index, expert_index)
            byte_size = _tensor_bundle_bytes(tensors)
            records.append(
                ExpertRecord(
                    key=key,
                    byte_size=byte_size,
                    content_id=(
                        f"{identity}|{adapter.adapter_id}|layer={global_index}"
                        f"|expert={expert_index}|swiglu-v1"
                    ),
                )
            )
            tensor_bundles[key] = tensors

        shared_tensors = _extract_shared_expert(
            mlp,
            shared_count=shared_count,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            architecture=spec.architecture,
            path=f"{layer_path}.{local_index}.mlp.shared_experts",
        )
        if shared_tensors is not None:
            dtype, device = _validate_tensor_group(
                shared_tensors,
                path=f"{layer_path}.{local_index}.mlp.shared_experts",
            )
            if dtype != observed_dtype or device != observed_device:
                raise UnsupportedHfMoeExpertLayoutError(
                    "all routed and shared expert tensors must use one dtype and device"
                )
            alias = _tensor_alias_signature(shared_tensors)
            builder = shared_builders.get(alias)
            module_path = f"{layer_path}.{local_index}.mlp.shared_experts"
            if builder is None:
                content_id = (
                    f"{identity}|{adapter.adapter_id}|shared-layer={global_index}|swiglu-v1"
                )
                builder = _SharedBuilder(
                    content_id=content_id,
                    layers=[global_index],
                    module_paths=[module_path],
                    tensors=shared_tensors,
                )
                shared_builders[alias] = builder
            else:
                builder.layers.append(global_index)
                builder.module_paths.append(module_path)
            shared_by_layer[global_index] = builder.content_id

    if not records or observed_layout is None or observed_dtype is None or observed_device is None:
        raise UnsupportedHfMoeExpertLayoutError(
            "the resident stage contains no extractable sparse MoE layer"
        )
    expected_sparse = tuple(range(sparse_layers[0], sparse_layers[-1] + 1))
    if tuple(sparse_layers) != expected_sparse:
        raise UnsupportedHfMoeExpertLayoutError(
            "sparse MoE layers must form one contiguous suffix"
        )

    shared = tuple(
        SharedExpertTensorBundle(
            content_id=builder.content_id,
            layers=tuple(builder.layers),
            module_paths=tuple(builder.module_paths),
            tensors=builder.tensors,
        )
        for builder in shared_builders.values()
    )
    return HfMoeExpertExtraction(
        adapter_id=adapter.adapter_id,
        model_type=spec.model_type,
        architecture=spec.architecture,
        model_identity=identity,
        storage_layout=observed_layout,
        global_layer_start=global_layer_start,
        global_layer_end=global_layer_start + len(layers),
        sparse_layers=tuple(sparse_layers),
        dense_layers=tuple(dense_layers),
        inventory=MacroStageExpertInventory(tuple(records)),
        expert_tensors=MappingProxyType(dict(tensor_bundles)),
        routers=MappingProxyType(dict(routers)),
        shared_experts=shared,
        shared_expert_content_by_layer=MappingProxyType(dict(shared_by_layer)),
        source_device=observed_device,
        source_dtype=observed_dtype,
    )


def hf_moe_extractor_registry_document() -> dict[str, Any]:
    """Return the closed architecture/layout surface for inspection tools."""

    return {
        "schema": HF_MOE_EXTRACTOR_SCHEMA,
        "architectures": [
            {
                "adapter": spec.adapter_id,
                "modelType": spec.model_type,
                "architecture": spec.architecture,
                "decoderRoots": ["model.layers", "layers"],
                "routedExpertLayouts": [
                    "grouped-3d-swiglu",
                    "unpacked-module-list-swiglu",
                ],
                "canonicalTensorNames": list(_CANONICAL_TENSOR_NAMES),
                "densePrefixField": spec.dense_prefix_field,
                "sharedExpertField": spec.shared_count_field,
                "sharedExpertPolicy": "resident-once-not-per-routed-expert",
                "partialExpertShards": False,
                "quantizedOrWrappedExperts": False,
            }
            for spec in _ARCHITECTURES
        ],
    }


def _resolve_decoder_layers(model: nn.Module) -> tuple[nn.ModuleList, str]:
    candidates: list[tuple[nn.ModuleList, str]] = []
    direct = getattr(model, "layers", None)
    if isinstance(direct, nn.ModuleList):
        candidates.append((direct, "layers"))
    decoder = getattr(model, "model", None)
    nested = getattr(decoder, "layers", None)
    if isinstance(nested, nn.ModuleList):
        candidates.append((nested, "model.layers"))
    unique = {id(layers): (layers, path) for layers, path in candidates}
    if len(unique) != 1:
        raise UnsupportedHfMoeExpertLayoutError(
            "model must expose exactly one ModuleList at layers or model.layers"
        )
    layers, path = next(iter(unique.values()))
    if not layers:
        raise UnsupportedHfMoeExpertLayoutError("decoder layer list cannot be empty")
    return layers, path


def _validate_dense_mlp(
    mlp: nn.Module,
    *,
    hidden_size: int,
    intermediate_size: int,
    path: str,
) -> None:
    if getattr(mlp, "experts", None) is not None:
        raise UnsupportedHfMoeExpertLayoutError(
            f"{path} is declared dense but exposes routed experts"
        )
    _canonical_linear_tensors(
        mlp,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        path=path,
    )


def _validate_router(
    mlp: nn.Module,
    *,
    expert_count: int,
    top_k: int,
    hidden_size: int,
    require_glm_correction: bool,
    path: str,
) -> nn.Module:
    router = getattr(mlp, "gate", None)
    if not isinstance(router, nn.Module):
        raise UnsupportedHfMoeExpertLayoutError(f"{path} is missing")
    weight = getattr(router, "weight", None)
    if not isinstance(weight, torch.Tensor) or tuple(weight.shape) != (
        expert_count,
        hidden_size,
    ):
        raise UnsupportedHfMoeExpertLayoutError(f"{path}.weight geometry is invalid")
    if getattr(router, "num_experts", None) != expert_count:
        raise UnsupportedHfMoeExpertLayoutError(f"{path}.num_experts is invalid")
    if getattr(router, "top_k", None) != top_k:
        raise UnsupportedHfMoeExpertLayoutError(f"{path}.top_k is invalid")
    if require_glm_correction:
        correction = getattr(router, "e_score_correction_bias", None)
        if not isinstance(correction, torch.Tensor) or tuple(correction.shape) != (
            expert_count,
        ):
            raise UnsupportedHfMoeExpertLayoutError(
                f"{path}.e_score_correction_bias is required for GLM4-MoE"
            )
    expected_parameters = {"weight"}
    expected_persistent_state = set(expected_parameters)
    if require_glm_correction:
        expected_persistent_state.add("e_score_correction_bias")
    parameters = set(dict(router.named_parameters(recurse=True)))
    persistent_state = set(router.state_dict())
    if (
        parameters != expected_parameters
        or persistent_state != expected_persistent_state
    ):
        raise UnsupportedHfMoeExpertLayoutError(
            f"{path} contains unexpected router parameters or persistent buffers"
        )
    return router


def _extract_routed_layer(
    mlp: nn.Module,
    *,
    expert_count: int,
    hidden_size: int,
    intermediate_size: int,
    path: str,
) -> tuple[tuple[TensorBundle, ...], str]:
    experts = getattr(mlp, "experts", None)
    if isinstance(experts, nn.ModuleList):
        if len(experts) != expert_count:
            raise UnsupportedHfMoeExpertLayoutError(
                f"{path} contains {len(experts)} experts, expected {expert_count}"
            )
        return (
            tuple(
                _canonical_linear_tensors(
                    expert,
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    path=f"{path}.{index}",
                )
                for index, expert in enumerate(experts)
            ),
            "unpacked-module-list-swiglu",
        )
    if not isinstance(experts, nn.Module):
        raise UnsupportedHfMoeExpertLayoutError(
            f"{path} must be a grouped module or torch.nn.ModuleList"
        )
    return (
        _grouped_expert_tensors(
            experts,
            expert_count=expert_count,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            path=path,
        ),
        "grouped-3d-swiglu",
    )


def _grouped_expert_tensors(
    experts: nn.Module,
    *,
    expert_count: int,
    hidden_size: int,
    intermediate_size: int,
    path: str,
) -> tuple[TensorBundle, ...]:
    if (
        getattr(experts, "num_experts", None) != expert_count
        or getattr(experts, "hidden_dim", None) != hidden_size
        or getattr(experts, "intermediate_dim", None) != intermediate_size
    ):
        raise UnsupportedHfMoeExpertLayoutError(
            f"{path} grouped expert metadata is inconsistent"
        )
    gate_up = getattr(experts, "gate_up_proj", None)
    down = getattr(experts, "down_proj", None)
    if not isinstance(gate_up, torch.Tensor) or tuple(gate_up.shape) != (
        expert_count,
        intermediate_size * 2,
        hidden_size,
    ):
        raise UnsupportedHfMoeExpertLayoutError(
            f"{path}.gate_up_proj geometry is invalid"
        )
    if not isinstance(down, torch.Tensor) or tuple(down.shape) != (
        expert_count,
        hidden_size,
        intermediate_size,
    ):
        raise UnsupportedHfMoeExpertLayoutError(f"{path}.down_proj geometry is invalid")
    parameters = set(dict(experts.named_parameters(recurse=True)))
    buffers = set(dict(experts.named_buffers(recurse=True)))
    if parameters != {"gate_up_proj", "down_proj"} or buffers:
        raise UnsupportedHfMoeExpertLayoutError(
            f"{path} contains unexpected grouped parameters or buffers"
        )
    return tuple(
        MappingProxyType(
            {
                "gate_proj.weight": gate_up[index, :intermediate_size, :],
                "up_proj.weight": gate_up[index, intermediate_size:, :],
                "down_proj.weight": down[index],
            }
        )
        for index in range(expert_count)
    )


def _extract_shared_expert(
    mlp: nn.Module,
    *,
    shared_count: int,
    hidden_size: int,
    intermediate_size: int,
    architecture: str,
    path: str,
) -> TensorBundle | None:
    shared = getattr(mlp, "shared_experts", None)
    if architecture == "Qwen3MoeForCausalLM":
        if shared is not None:
            raise UnsupportedHfMoeExpertLayoutError(
                f"{path} is not part of the certified Qwen3-MoE layout"
            )
        return None
    if not isinstance(shared, nn.Module):
        if shared_count == 0:
            return None
        raise UnsupportedHfMoeExpertLayoutError(f"{path} is required for GLM4-MoE")
    tensors = _canonical_linear_tensors(
        shared,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size * shared_count,
        path=path,
        allow_zero_intermediate=shared_count == 0,
    )
    if shared_count == 0:
        if any(tensor.numel() for tensor in tensors.values()):
            raise UnsupportedHfMoeExpertLayoutError(
                f"{path} must be empty when n_shared_experts is zero"
            )
        return None
    return tensors


def _canonical_linear_tensors(
    module: nn.Module,
    *,
    hidden_size: int,
    intermediate_size: int,
    path: str,
    allow_zero_intermediate: bool = False,
) -> TensorBundle:
    if intermediate_size < 0 or (intermediate_size == 0 and not allow_zero_intermediate):
        raise UnsupportedHfMoeExpertLayoutError(
            f"{path} intermediate size must be positive"
        )
    projections = {
        "gate_proj": (intermediate_size, hidden_size),
        "up_proj": (intermediate_size, hidden_size),
        "down_proj": (hidden_size, intermediate_size),
    }
    tensors: dict[str, torch.Tensor] = {}
    for name, shape in projections.items():
        projection = getattr(module, name, None)
        if not isinstance(projection, nn.Linear):
            raise UnsupportedHfMoeExpertLayoutError(f"{path}.{name} must be nn.Linear")
        if projection.bias is not None or tuple(projection.weight.shape) != shape:
            raise UnsupportedHfMoeExpertLayoutError(
                f"{path}.{name} must be bias-free with shape {shape}"
            )
        tensors[f"{name}.weight"] = projection.weight
    parameters = set(dict(module.named_parameters(recurse=True)))
    buffers = set(dict(module.named_buffers(recurse=True)))
    if parameters != set(_CANONICAL_TENSOR_NAMES) or buffers:
        raise UnsupportedHfMoeExpertLayoutError(
            f"{path} contains unexpected parameters or persistent buffers"
        )
    return MappingProxyType(tensors)


def _validate_tensor_group(
    tensors: TensorBundle,
    *,
    path: str,
) -> tuple[torch.dtype, torch.device]:
    if tuple(tensors) != _CANONICAL_TENSOR_NAMES:
        raise UnsupportedHfMoeExpertLayoutError(
            f"{path} does not expose the canonical SwiGLU tensor order"
        )
    first = next(iter(tensors.values()))
    dtype, device = first.dtype, first.device
    for name, tensor in tensors.items():
        if (
            tensor.is_meta
            or tensor.layout != torch.strided
            or not tensor.is_floating_point()
            or tensor.dtype != dtype
            or tensor.device != device
        ):
            raise UnsupportedHfMoeExpertLayoutError(
                f"{path}.{name} must be a resident strided floating tensor on one device"
            )
    return dtype, device


def _tensor_bundle_bytes(tensors: TensorBundle) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())


def _tensor_alias_signature(tensors: TensorBundle) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            name,
            str(tensor.device),
            str(tensor.dtype),
            tensor.untyped_storage().data_ptr(),
            tensor.storage_offset(),
            tuple(tensor.shape),
            tuple(tensor.stride()),
        )
        for name, tensor in tensors.items()
    )


def _required_positive_int(config: Any, name: str) -> int:
    value = getattr(config, name, None)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise UnsupportedHfMoeExpertLayoutError(
            f"config.{name} must be a positive integer"
        )
    return value


def _required_nonnegative_int(config: Any, name: str) -> int:
    value = getattr(config, name, None)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise UnsupportedHfMoeExpertLayoutError(
            f"config.{name} must be a non-negative integer"
        )
    return value


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


__all__ = [
    "HF_MOE_EXTRACTOR_SCHEMA",
    "HfMoeExpertExtraction",
    "SharedExpertTensorBundle",
    "UnsupportedHfMoeExpertLayoutError",
    "extract_hf_moe_experts",
    "hf_moe_extractor_registry_document",
]
