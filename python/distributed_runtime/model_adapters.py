from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from torch import nn
import torch


class UnsupportedSelectiveStageArchitectureError(ValueError):
    """The checkpoint has no explicitly certified selective-stage adapter."""


@dataclass(frozen=True)
class SelectiveStageModelParts:
    decoder: nn.Module
    layers: nn.ModuleList
    embedding: nn.Module
    final_norm: nn.Module
    head: nn.Module


@dataclass(frozen=True)
class CheckpointTensorAssignment:
    checkpoint_name: str
    destination: torch.Tensor


@dataclass(frozen=True)
class SelectiveStageAdapter:
    """Exact semantic contract for one Transformers decoder family.

    Sharing tensor names is not enough to make two architectures equivalent.
    The adapter therefore seals both the public configuration identity and the
    family-specific modules whose execution must survive range slicing.
    Unknown architectures fail closed instead of falling through a generic
    ``model.layers`` heuristic.
    """

    adapter_id: str
    model_type: str
    architectures: tuple[str, ...]
    required_layer_modules: tuple[str, ...]
    semantic_features: tuple[str, ...]
    full_attention_only: bool = True

    def validate_source_config(self, config: Any, total_layers: int) -> None:
        if getattr(config, "model_type", None) != self.model_type:
            raise UnsupportedSelectiveStageArchitectureError(
                f"adapter {self.adapter_id} does not support model_type "
                f"{getattr(config, 'model_type', None)!r}"
            )
        declared = getattr(config, "architectures", None)
        if (
            not isinstance(declared, Sequence)
            or isinstance(declared, (str, bytes, bytearray))
            or len(declared) != 1
            or declared[0] not in self.architectures
        ):
            raise UnsupportedSelectiveStageArchitectureError(
                f"adapter {self.adapter_id} requires one of architectures "
                f"{self.architectures!r}, got {declared!r}"
            )
        actual_layers = _positive_integer(
            getattr(config, "num_hidden_layers", None), "num_hidden_layers"
        )
        if actual_layers != total_layers:
            raise ValueError(
                f"model has {actual_layers} layers, but the stage plan declares "
                f"{total_layers}"
            )
        _positive_integer(getattr(config, "hidden_size", None), "hidden_size")
        attention_heads = _positive_integer(
            getattr(config, "num_attention_heads", None), "num_attention_heads"
        )
        kv_heads = _positive_integer(
            getattr(config, "num_key_value_heads", attention_heads),
            "num_key_value_heads",
        )
        if kv_heads > attention_heads or attention_heads % kv_heads != 0:
            raise ValueError("num_key_value_heads must divide num_attention_heads")
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            hidden_size = int(config.hidden_size)
            if hidden_size % attention_heads != 0:
                raise ValueError("hidden_size must divide evenly across attention heads")
            head_dim = hidden_size // attention_heads
        _positive_integer(head_dim, "head_dim")

        layer_types = getattr(config, "layer_types", None)
        if layer_types is not None:
            if (
                not isinstance(layer_types, Sequence)
                or isinstance(layer_types, (str, bytes, bytearray))
                or len(layer_types) != total_layers
            ):
                raise ValueError("layer_types must contain one entry per global layer")
            if self.full_attention_only and set(layer_types) != {"full_attention"}:
                raise UnsupportedSelectiveStageArchitectureError(
                    f"adapter {self.adapter_id} is certified only for full_attention layers"
                )
        if self.full_attention_only and getattr(config, "use_sliding_window", False):
            raise UnsupportedSelectiveStageArchitectureError(
                f"adapter {self.adapter_id} is not certified for sliding-window attention"
            )
        if self.model_type == "qwen3_moe":
            experts = _positive_integer(getattr(config, "num_experts", None), "num_experts")
            selected = _positive_integer(
                getattr(config, "num_experts_per_tok", None), "num_experts_per_tok"
            )
            if selected > experts:
                raise ValueError("num_experts_per_tok cannot exceed num_experts")
            _positive_integer(
                getattr(config, "moe_intermediate_size", None),
                "moe_intermediate_size",
            )
            if getattr(config, "decoder_sparse_step", None) != 1 or getattr(
                config, "mlp_only_layers", None
            ) not in (None, []):
                raise UnsupportedSelectiveStageArchitectureError(
                    "transformers-qwen3-moe-v1 currently requires every layer to use MoE"
                )
            ep_plan = getattr(config, "base_model_ep_plan", None)
            required_ep_plan = {
                "layers.*.mlp.gate": "ep_router",
                "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
                "layers.*.mlp.experts.down_proj": "grouped_gemm",
                "layers.*.mlp.experts": "moe_tp_experts",
            }
            if not isinstance(ep_plan, dict) or any(
                ep_plan.get(name) != strategy
                for name, strategy in required_ep_plan.items()
            ):
                raise UnsupportedSelectiveStageArchitectureError(
                    "Qwen3Moe base_model_ep_plan does not match the certified layout"
                )
        if self.model_type == "glm4_moe":
            experts = _positive_integer(
                getattr(config, "n_routed_experts", None), "n_routed_experts"
            )
            selected = _positive_integer(
                getattr(config, "num_experts_per_tok", None), "num_experts_per_tok"
            )
            if selected > experts:
                raise ValueError("num_experts_per_tok cannot exceed n_routed_experts")
            _positive_integer(
                getattr(config, "moe_intermediate_size", None),
                "moe_intermediate_size",
            )
            shared = getattr(config, "n_shared_experts", None)
            if not isinstance(shared, int) or isinstance(shared, bool) or shared < 0:
                raise ValueError("n_shared_experts must be a non-negative integer")
            dense = getattr(config, "first_k_dense_replace", None)
            if (
                not isinstance(dense, int)
                or isinstance(dense, bool)
                or not 0 <= dense <= total_layers
            ):
                raise ValueError("first_k_dense_replace is outside the model layer range")
            ep_plan = getattr(config, "base_model_ep_plan", None)
            required_ep_plan = {
                "layers.*.mlp.gate": "ep_router",
                "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
                "layers.*.mlp.experts.down_proj": "grouped_gemm",
                "layers.*.mlp.experts": "moe_tp_experts",
            }
            if not isinstance(ep_plan, dict) or any(
                ep_plan.get(name) != strategy
                for name, strategy in required_ep_plan.items()
            ):
                raise UnsupportedSelectiveStageArchitectureError(
                    "Glm4Moe base_model_ep_plan does not match the certified layout"
                )

    def slice_config(
        self,
        config: Any,
        *,
        layer_start: int,
        layer_end: int,
        total_layers: int,
    ) -> None:
        """Retain the selected global semantics after layers are renumbered."""

        self.validate_source_config(config, total_layers)
        for name, value in vars(config).items():
            if isinstance(value, (list, tuple)) and len(value) == total_layers:
                setattr(config, name, value[layer_start:layer_end])
        if self.model_type == "glm4_moe":
            # GLM chooses dense versus sparse construction from the local layer
            # index. Shift its global dense prefix so a nonzero range builds the
            # same layer classes as the monolithic checkpoint.
            config.first_k_dense_replace = max(
                0,
                int(config.first_k_dense_replace) - layer_start,
            )
        config.num_hidden_layers = layer_end - layer_start

    def inspect_constructed_model(
        self,
        model: Any,
        *,
        local_layers: int,
    ) -> SelectiveStageModelParts:
        decoder = getattr(model, "model", None)
        layers = getattr(decoder, "layers", None)
        embedding = getattr(decoder, "embed_tokens", None)
        final_norm = getattr(decoder, "norm", None)
        head = getattr(model, "lm_head", None)
        if (
            not isinstance(decoder, nn.Module)
            or not isinstance(layers, nn.ModuleList)
            or not isinstance(embedding, nn.Module)
            or not isinstance(final_norm, nn.Module)
            or not isinstance(head, nn.Module)
        ):
            raise TypeError(
                f"adapter {self.adapter_id} requires model.layers, model.norm, "
                "model.embed_tokens and lm_head modules"
            )
        if len(layers) != local_layers:
            raise ValueError("local model constructor did not honor num_hidden_layers")
        for index, layer in enumerate(layers):
            for path in self.required_layer_modules:
                if not isinstance(_module_at_path(layer, path), nn.Module):
                    raise TypeError(
                        f"adapter {self.adapter_id} layer {index} is missing module {path}"
                    )
            self._validate_family_layer(layer, index, model.config)
        return SelectiveStageModelParts(
            decoder=decoder,
            layers=layers,
            embedding=embedding,
            final_norm=final_norm,
            head=head,
        )

    def _validate_family_layer(self, layer: nn.Module, index: int, config: Any) -> None:
        if self.model_type not in ("qwen3", "qwen3_moe", "glm4_moe"):
            return
        attention = _module_at_path(layer, "self_attn")
        head_dim = int(config.head_dim)
        attention_heads = int(config.num_attention_heads)
        kv_heads = int(config.num_key_value_heads)
        expected = {
            "q_proj": attention_heads * head_dim,
            "k_proj": kv_heads * head_dim,
            "v_proj": kv_heads * head_dim,
        }
        for name, out_features in expected.items():
            projection = getattr(attention, name, None)
            if not isinstance(projection, nn.Linear) or projection.out_features != out_features:
                raise TypeError(
                    f"Qwen3 layer {index} {name} does not match its head geometry"
                )
        if self.model_type != "glm4_moe" or getattr(config, "use_qk_norm", False):
            for name in ("q_norm", "k_norm"):
                norm = getattr(attention, name, None)
                weight = getattr(norm, "weight", None)
                if (
                    not isinstance(norm, nn.Module)
                    or weight is None
                    or weight.numel() != head_dim
                ):
                    raise TypeError(
                        f"{self.model_type} layer {index} {name} does not preserve "
                        "per-head normalization"
                    )
        if self.model_type == "qwen3_moe":
            mlp = _module_at_path(layer, "mlp")
            gate = getattr(mlp, "gate", None)
            experts = getattr(mlp, "experts", None)
            gate_weight = getattr(gate, "weight", None)
            gate_up = getattr(experts, "gate_up_proj", None)
            down = getattr(experts, "down_proj", None)
            num_experts = int(config.num_experts)
            hidden_size = int(config.hidden_size)
            intermediate = int(config.moe_intermediate_size)
            if gate_weight is None or tuple(gate_weight.shape) != (
                num_experts,
                hidden_size,
            ):
                raise TypeError(f"Qwen3Moe layer {index} router geometry is invalid")
            if gate_up is None or tuple(gate_up.shape) != (
                num_experts,
                intermediate * 2,
                hidden_size,
            ):
                raise TypeError(
                    f"Qwen3Moe layer {index} stacked gate/up experts are invalid"
                )
            if down is None or tuple(down.shape) != (
                num_experts,
                hidden_size,
                intermediate,
            ):
                raise TypeError(
                    f"Qwen3Moe layer {index} stacked down experts are invalid"
                )
        if self.model_type == "glm4_moe":
            mlp = _module_at_path(layer, "mlp")
            dense = index < int(config.first_k_dense_replace)
            if dense:
                for name in ("gate_proj", "up_proj", "down_proj"):
                    if not isinstance(getattr(mlp, name, None), nn.Linear):
                        raise TypeError(
                            f"Glm4Moe dense layer {index} is missing {name}"
                        )
                return
            gate = getattr(mlp, "gate", None)
            experts = getattr(mlp, "experts", None)
            shared = getattr(mlp, "shared_experts", None)
            num_experts = int(config.n_routed_experts)
            hidden_size = int(config.hidden_size)
            intermediate = int(config.moe_intermediate_size)
            if tuple(getattr(getattr(gate, "weight", None), "shape", ())) != (
                num_experts,
                hidden_size,
            ):
                raise TypeError(f"Glm4Moe layer {index} router geometry is invalid")
            if tuple(getattr(getattr(experts, "gate_up_proj", None), "shape", ())) != (
                num_experts,
                intermediate * 2,
                hidden_size,
            ) or tuple(getattr(getattr(experts, "down_proj", None), "shape", ())) != (
                num_experts,
                hidden_size,
                intermediate,
            ):
                raise TypeError(f"Glm4Moe layer {index} grouped experts are invalid")
            if int(config.n_shared_experts) > 0 and not all(
                isinstance(getattr(shared, name, None), nn.Linear)
                for name in ("gate_proj", "up_proj", "down_proj")
            ):
                raise TypeError(f"Glm4Moe layer {index} shared experts are invalid")

    def checkpoint_assignments(
        self,
        local_name: str,
        target: torch.Tensor,
        *,
        layer_start: int,
        checkpoint_names: set[str],
    ) -> tuple[CheckpointTensorAssignment, ...] | None:
        """Map unpacked Hub expert tensors into the resident grouped layout."""

        if self.model_type not in ("qwen3_moe", "glm4_moe"):
            return None
        prefix = "model.layers."
        if not local_name.startswith(prefix):
            return None
        remainder = local_name[len(prefix) :]
        local_index_text, separator, suffix = remainder.partition(".")
        if not separator or suffix not in (
            "mlp.experts.gate_up_proj",
            "mlp.experts.down_proj",
        ):
            return None
        original_index = layer_start + int(local_index_text)
        packed_name = f"{prefix}{original_index}.{suffix}"
        if packed_name in checkpoint_names:
            return None
        if target.ndim != 3:
            raise TypeError(f"Qwen3Moe grouped target {local_name} must be rank three")
        num_experts = int(target.shape[0])
        assignments: list[CheckpointTensorAssignment] = []
        expert_prefix = f"{prefix}{original_index}.mlp.experts"
        if suffix.endswith("gate_up_proj"):
            if int(target.shape[1]) % 2 != 0:
                raise TypeError("Qwen3Moe grouped gate/up width must be even")
            intermediate = int(target.shape[1]) // 2
            for expert in range(num_experts):
                assignments.extend(
                    (
                        CheckpointTensorAssignment(
                            f"{expert_prefix}.{expert}.gate_proj.weight",
                            target[expert, :intermediate, :],
                        ),
                        CheckpointTensorAssignment(
                            f"{expert_prefix}.{expert}.up_proj.weight",
                            target[expert, intermediate:, :],
                        ),
                    )
                )
        else:
            for expert in range(num_experts):
                assignments.append(
                    CheckpointTensorAssignment(
                        f"{expert_prefix}.{expert}.down_proj.weight",
                        target[expert],
                    )
                )
        return tuple(assignments)


_COMMON_DECODER_MODULES = (
    "self_attn",
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
    "input_layernorm",
    "post_attention_layernorm",
)


_ADAPTERS = (
    SelectiveStageAdapter(
        adapter_id="transformers-llama-v1",
        model_type="llama",
        architectures=("LlamaForCausalLM",),
        required_layer_modules=_COMMON_DECODER_MODULES,
        semantic_features=(
            "causal-decoder",
            "grouped-query-attention",
            "rotary-position",
            "rms-norm",
        ),
    ),
    SelectiveStageAdapter(
        adapter_id="transformers-qwen3-v1",
        model_type="qwen3",
        architectures=("Qwen3ForCausalLM",),
        required_layer_modules=(
            *_COMMON_DECODER_MODULES,
            "self_attn.q_norm",
            "self_attn.k_norm",
        ),
        semantic_features=(
            "causal-decoder",
            "grouped-query-attention",
            "rotary-position",
            "rms-norm",
            "qk-rms-norm",
        ),
    ),
    SelectiveStageAdapter(
        adapter_id="transformers-qwen3-moe-v1",
        model_type="qwen3_moe",
        architectures=("Qwen3MoeForCausalLM",),
        required_layer_modules=(
            "self_attn",
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "self_attn.q_norm",
            "self_attn.k_norm",
            "mlp",
            "mlp.gate",
            "mlp.experts",
            "input_layernorm",
            "post_attention_layernorm",
        ),
        semantic_features=(
            "causal-decoder",
            "grouped-query-attention",
            "rotary-position",
            "rms-norm",
            "qk-rms-norm",
            "sparse-moe",
            "layer-local-router",
            "resident-expert-set",
            "ep-plan-metadata-only",
        ),
    ),
    SelectiveStageAdapter(
        adapter_id="transformers-glm4-moe-v1",
        model_type="glm4_moe",
        architectures=("Glm4MoeForCausalLM",),
        required_layer_modules=(
            "self_attn",
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp",
            "input_layernorm",
            "post_attention_layernorm",
        ),
        semantic_features=(
            "causal-decoder",
            "grouped-query-attention",
            "partial-rotary-position",
            "rms-norm",
            "hybrid-dense-moe",
            "layer-local-router",
            "resident-expert-set",
            "shared-expert",
            "ep-plan-metadata-only",
        ),
    ),
)


def selective_stage_adapters() -> tuple[SelectiveStageAdapter, ...]:
    return _ADAPTERS


def resolve_selective_stage_adapter(config: Any) -> SelectiveStageAdapter:
    model_type = getattr(config, "model_type", None)
    declared = getattr(config, "architectures", None)
    matches = tuple(adapter for adapter in _ADAPTERS if adapter.model_type == model_type)
    if len(matches) != 1:
        raise UnsupportedSelectiveStageArchitectureError(
            "no certified selective-stage adapter for "
            f"model_type={model_type!r}, architectures={declared!r}"
        )
    adapter = matches[0]
    # Layer count is validated by the loader against the stage plan. Resolve
    # still checks the architecture discriminator so an alias cannot inherit a
    # family adapter accidentally.
    if (
        not isinstance(declared, Sequence)
        or isinstance(declared, (str, bytes, bytearray))
        or len(declared) != 1
        or declared[0] not in adapter.architectures
    ):
        raise UnsupportedSelectiveStageArchitectureError(
            f"adapter {adapter.adapter_id} does not certify architectures={declared!r}"
        )
    return adapter


def adapter_registry_document() -> dict[str, Any]:
    """Deterministic inspection surface used by certification tooling."""

    return {
        "schema": "gdlp-transformers-stage-adapters/1",
        "adapters": [
            {
                "id": adapter.adapter_id,
                "modelType": adapter.model_type,
                "architectures": list(adapter.architectures),
                "requiredLayerModules": list(adapter.required_layer_modules),
                "semanticFeatures": list(adapter.semantic_features),
                "attentionScope": (
                    "full-only" if adapter.full_attention_only else "mixed"
                ),
                "parallelismScope": (
                    "pipeline-only"
                    if adapter.model_type in ("qwen3_moe", "glm4_moe")
                    else "pipeline-stage"
                ),
            }
            for adapter in _ADAPTERS
        ],
    }


def _module_at_path(module: nn.Module, path: str) -> Any:
    current: Any = module
    for component in path.split("."):
        current = getattr(current, component, None)
        if current is None:
            return None
    return current


def _positive_integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


__all__ = [
    "CheckpointTensorAssignment",
    "SelectiveStageAdapter",
    "SelectiveStageModelParts",
    "UnsupportedSelectiveStageArchitectureError",
    "adapter_registry_document",
    "resolve_selective_stage_adapter",
    "selective_stage_adapters",
]
