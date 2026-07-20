from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass, replace
import gc
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Protocol

from huggingface_hub import snapshot_download
from safetensors import safe_open
import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.cache_utils import DynamicLayer

from .model_adapters import (
    SelectiveStageAdapter,
    resolve_selective_stage_adapter,
)

MAX_PHYSICAL_STAGE_BATCH_SIZE = 8


@dataclass(frozen=True)
class ModelArtifactReference:
    """Path-independent coordinates for one immutable checkpoint snapshot."""

    identity: str
    canonical_source: str
    canonical_revision: str | None
    snapshot_identity: int


STAGE_QUANTIZE_MODES = (None, "dynamic-int8")
STAGE_COMPILE_MODES = (None, "default", "reduce-overhead")


@dataclass(frozen=True)
class StageModelSpec:
    model_name: str
    layer_start: int
    layer_end: int
    total_layers: int
    threads: int
    revision: str | None = None
    artifact_identity: str | None = None
    canonical_model_source: str | None = None
    canonical_model_revision: str | None = None
    # Opt-in experimental execution variants. ``None`` keeps the reference
    # FP32 eager path byte-for-byte identical. ``quantize="dynamic-int8"`` is an
    # APPROXIMATE mode: it may change greedy tokens versus the FP32 reference.
    quantize: str | None = None
    compile_mode: str | None = None

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError("model_name cannot be empty")
        if self.total_layers < 1:
            raise ValueError("total_layers must be positive")
        if not 0 <= self.layer_start < self.layer_end <= self.total_layers:
            raise ValueError(
                "layer range must satisfy 0 <= layer_start < layer_end <= total_layers"
            )
        if self.threads < 1:
            raise ValueError("threads must be positive")
        if self.revision is not None and not self.revision.strip():
            raise ValueError("revision cannot be blank")
        if self.quantize not in STAGE_QUANTIZE_MODES:
            raise ValueError(f"quantize must be one of {STAGE_QUANTIZE_MODES}")
        if self.compile_mode not in STAGE_COMPILE_MODES:
            raise ValueError(f"compile_mode must be one of {STAGE_COMPILE_MODES}")
        for name, value in (
            ("artifact_identity", self.artifact_identity),
            ("canonical_model_source", self.canonical_model_source),
            ("canonical_model_revision", self.canonical_model_revision),
        ):
            if value is not None and not value.strip():
                raise ValueError(f"{name} cannot be blank")
        if self.artifact_identity is None and (
            self.canonical_model_source is not None
            or self.canonical_model_revision is not None
        ):
            raise ValueError(
                "canonical model coordinates require an explicit artifact identity"
            )

    @property
    def first(self) -> bool:
        return self.layer_start == 0

    @property
    def last(self) -> bool:
        return self.layer_end == self.total_layers


class StageRunnerContract(Protocol):
    """Execution contract consumed by the physical pipeline stage.

    A runner may own one process (``StageRunner``) or coordinate a local
    tensor-parallel cell.  Keeping the wire stage coupled to this small
    contract lets both implementations share the exact BEGIN/forward/
    TRUNCATE/END lifecycle.
    """

    spec: StageModelSpec
    hidden_size: int
    parameter_bytes: int
    loader: str
    executor_manifest: Any

    def begin(self, request_id: int) -> None: ...

    def end(self, request_id: int) -> None: ...

    def truncate(self, request_id: int, token_count: int) -> None: ...

    def sequence_length(self, request_id: int) -> int: ...

    def forward_hidden(
        self,
        request_id: int,
        hidden: torch.Tensor,
        *,
        token_mode: str = "last",
    ) -> tuple[torch.Tensor, int | tuple[int, ...] | None]: ...

    def close(self) -> None: ...


class StageRunner:
    MAX_PHYSICAL_BATCH_SIZE = MAX_PHYSICAL_STAGE_BATCH_SIZE

    def __init__(self, spec: StageModelSpec) -> None:
        torch.set_num_threads(spec.threads)
        model = _load_selective_stage_model(spec)
        self._initialize_from_loaded_model(
            spec,
            model,
            loader="selective-safetensors",
            device_kinds=("cpu",),
        )

    def _initialize_from_loaded_model(
        self,
        spec: StageModelSpec,
        model: nn.Module,
        *,
        loader: str,
        device_kinds: tuple[str, ...],
        semantic_features: tuple[str, ...] | None = None,
    ) -> None:
        """Adopt one already-loaded, adapter-certified local model.

        The RAM-backed MoE loader constructs its local Hugging Face model on
        the meta device, removes routed expert parameters, and only then
        materializes resident tensors.  Keeping the request/KV lifecycle in
        this initializer lets that runner reuse the exact same StageRunner
        implementation without first creating a forbidden full expert copy.
        """

        if not isinstance(loader, str) or not loader.strip():
            raise ValueError("loader cannot be empty")
        if not device_kinds or any(
            not isinstance(kind, str) or not kind.strip() for kind in device_kinds
        ):
            raise ValueError("device_kinds must contain non-empty strings")
        adapter = getattr(model, "_gdlp_selective_stage_adapter", None)
        if not isinstance(adapter, SelectiveStageAdapter):
            raise TypeError("selective model loader did not return a certified family adapter")
        self.model_adapter = adapter
        selected = list(model.model.layers)
        # Every stage owns an independent DynamicCache. Local layer indexes keep
        # cache lookup and sequence-length accounting dense and O(number of local layers).
        for local_index, layer in enumerate(selected):
            layer.self_attn.layer_idx = local_index
        self.base = model.model
        self.head = model.lm_head if spec.last else None
        self.hidden_size = int(model.config.hidden_size)
        self.parameter_bytes = _unique_parameter_bytes(self.base, self.head)
        self.loader = loader
        self.spec = spec
        self._physical_batch_cache_supported = _supports_dynamic_tensor_batching(
            self.base.config
        )
        self.model_forward_calls = 0
        self.physical_batch_calls = 0
        self.physical_batch_items = 0
        self.max_observed_physical_batch_size = 1
        from .executor_abi import build_stage_executor_manifest

        artifact = model_artifact_reference(
            getattr(model, "_gdlp_resolved_snapshot", spec.model_name),
            None if hasattr(model, "_gdlp_resolved_snapshot") else spec.revision,
            artifact_identity=spec.artifact_identity,
            canonical_source=spec.canonical_model_source,
            canonical_revision=spec.canonical_model_revision,
        )

        weight_dtypes = tuple(
            sorted(
                {
                    str(parameter.dtype).removeprefix("torch.")
                    for module in (self.base, self.head)
                    if module is not None
                    for parameter in module.parameters()
                }
            )
        )
        self.executor_manifest = build_stage_executor_manifest(
            engine="python-torch",
            engine_version=torch.__version__,
            adapter=adapter.adapter_id,
            model_identity=artifact.identity,
            model_source=artifact.canonical_source,
            model_revision=artifact.canonical_revision,
            artifact_format="safetensors",
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
            max_batch_size=(
                self.MAX_PHYSICAL_BATCH_SIZE
                if self._physical_batch_cache_supported
                else 1
            ),
            device_kinds=device_kinds,
            compute_apis=("torch",),
            weight_dtypes=weight_dtypes,
            features=(
                "layer-range",
                "rank-local-kv",
                "rollback",
                "selective-load",
                *(adapter.semantic_features if semantic_features is None else semantic_features),
                *(
                    ("physical-tensor-batching",)
                    if self._physical_batch_cache_supported
                    else ()
                ),
            ),
        )
        # Opt-in spike variants. Applied after the manifest so the default
        # metadata path stays identical when both flags are absent.
        if spec.quantize == "dynamic-int8":
            self.base, self.head = _apply_dynamic_int8(self.base, self.head)
            self.loader += "+dynamic-int8"
        if spec.compile_mode is not None:
            inductor_mode = None if spec.compile_mode == "default" else spec.compile_mode
            self.base = torch.compile(self.base, backend="inductor", mode=inductor_mode)
            if self.head is not None:
                self.head = torch.compile(
                    self.head, backend="inductor", mode=inductor_mode
                )
            self.loader += f"+compile-{spec.compile_mode}"
        self.caches: dict[int, Any] = {}
        self.tokens_seen: dict[int, int] = {}
        self.active_requests: set[int] = set()
        del model
        gc.collect()

    def begin(self, request_id: int) -> None:
        if request_id in self.active_requests:
            raise ValueError(f"request {request_id} is already active")
        self.caches.pop(request_id, None)
        self.tokens_seen[request_id] = 0
        self.active_requests.add(request_id)

    def end(self, request_id: int) -> None:
        self.caches.pop(request_id, None)
        self.tokens_seen.pop(request_id, None)
        self.active_requests.discard(request_id)

    def close(self) -> None:
        """Release request state; resident model tensors follow process lifetime."""

        self.caches.clear()
        self.tokens_seen.clear()
        self.active_requests.clear()

    def truncate(self, request_id: int, token_count: int) -> None:
        """Crop this stage's local KV cache to an accepted speculative prefix."""

        self._require_active(request_id)
        current = self.tokens_seen[request_id]
        if not isinstance(token_count, int) or isinstance(token_count, bool):
            raise TypeError("token_count must be an integer")
        if not 0 <= token_count <= current:
            raise ValueError(
                f"cannot truncate request {request_id} from {current} to {token_count} tokens"
            )
        cache = self.caches.get(request_id)
        if cache is not None and token_count < current:
            crop = getattr(cache, "crop", None)
            if not callable(crop):
                raise TypeError("model cache does not support speculative rollback")
            crop(token_count)
        self.tokens_seen[request_id] = token_count

    def sequence_length(self, request_id: int) -> int:
        self._require_active(request_id)
        return self.tokens_seen[request_id]

    def _require_active(self, request_id: int) -> None:
        if request_id not in self.active_requests:
            raise ValueError(f"request {request_id} has not received BEGIN")

    @torch.inference_mode()
    def forward_ids(self, request_id: int, input_ids: torch.Tensor) -> torch.Tensor:
        if not self.spec.first:
            raise RuntimeError("only the first stage accepts token IDs")
        self._require_active(request_id)
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
            raise ValueError("input_ids must have shape [1, tokens] with at least one token")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("input_ids must contain integer token IDs")
        output = self.base(
            input_ids=input_ids,
            past_key_values=self.caches.get(request_id),
            use_cache=True,
        )
        self.model_forward_calls += 1
        self.caches[request_id] = output.past_key_values
        self.tokens_seen[request_id] += int(input_ids.shape[1])
        return output.last_hidden_state

    @torch.inference_mode()
    def forward_ids_batch(
        self,
        request_ids: Sequence[int],
        input_ids: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, ...]:
        """Run compatible token inputs in one physical model forward.

        Requests retain independent KV ownership.  Their caches are merged only
        for the duration of the forward and are split into fresh rank-one cache
        tensors before this method returns.
        """

        if not self.spec.first:
            raise RuntimeError("only the first stage accepts token IDs")
        ids, tensors, token_count = self._validate_physical_batch_inputs(
            request_ids,
            input_ids,
            expected_hidden_size=None,
            integer=True,
        )
        cache = self._merge_dynamic_caches(ids)
        output = self.base(
            input_ids=torch.cat(tensors, dim=0),
            past_key_values=cache,
            use_cache=True,
        )
        self._commit_physical_batch(ids, output.past_key_values, token_count)
        return tuple(
            output.last_hidden_state[index : index + 1].contiguous()
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
        if (
            hidden.ndim != 3
            or hidden.shape[0] != 1
            or hidden.shape[1] < 1
            or hidden.shape[2] != self.hidden_size
        ):
            raise ValueError(
                f"hidden state must have shape [1, tokens, {self.hidden_size}]"
            )
        if not hidden.is_floating_point():
            raise TypeError("hidden state must be floating point")
        if token_mode not in ("none", "last", "all"):
            raise ValueError("token_mode must be none, last or all")
        output = self.base(
            inputs_embeds=hidden,
            past_key_values=self.caches.get(request_id),
            use_cache=True,
        )
        self.model_forward_calls += 1
        self.caches[request_id] = output.past_key_values
        self.tokens_seen[request_id] += int(hidden.shape[1])
        if self.head is None or token_mode == "none":
            return output.last_hidden_state, None
        selected = (
            output.last_hidden_state
            if token_mode == "all"
            else output.last_hidden_state[:, -1:, :]
        )
        logits = self.head(selected)
        tokens = torch.argmax(logits, dim=-1).reshape(-1).tolist()
        if token_mode == "all":
            return output.last_hidden_state, tuple(int(token) for token in tokens)
        return output.last_hidden_state, int(tokens[-1])

    def physical_batch_key(
        self,
        request_id: int,
        *,
        token_count: int,
        token_mode: str,
    ) -> tuple[int, int, str] | None:
        """Return the exact cache/shape key accepted by ``forward_hidden_batch``.

        ``None`` deliberately means sequential fallback.  In particular, the
        pinned Transformers cache API cannot reconstruct hybrid/sliding cache
        metadata from independent requests without touching backend internals,
        so those layouts are never presented as physically batchable here.
        """

        self._require_active(request_id)
        if not self._physical_batch_cache_supported:
            return None
        if (
            not isinstance(token_count, int)
            or isinstance(token_count, bool)
            or token_count < 1
        ):
            raise ValueError("token_count must be a positive integer")
        if token_mode not in ("none", "last", "all"):
            raise ValueError("token_mode must be none, last or all")
        current = self.tokens_seen[request_id]
        cache = self.caches.get(request_id)
        if cache is not None:
            if not isinstance(cache, DynamicCache):
                return None
            if cache.get_seq_length() != current:
                return None
        elif current != 0:
            return None
        return current, token_count, token_mode

    @torch.inference_mode()
    def forward_hidden_batch(
        self,
        request_ids: Sequence[int],
        hidden_states: Sequence[torch.Tensor],
        *,
        token_mode: str = "last",
    ) -> tuple[
        tuple[torch.Tensor, int | tuple[int, ...] | None],
        ...,
    ]:
        """Execute one real tensor batch and split outputs/KV by request."""

        if token_mode not in ("none", "last", "all"):
            raise ValueError("token_mode must be none, last or all")
        ids, tensors, token_count = self._validate_physical_batch_inputs(
            request_ids,
            hidden_states,
            expected_hidden_size=self.hidden_size,
            integer=False,
        )
        keys = {
            self.physical_batch_key(
                request_id,
                token_count=token_count,
                token_mode=token_mode,
            )
            for request_id in ids
        }
        if None in keys or len(keys) != 1:
            raise ValueError(
                "physical batching requires equal cache length, token count and cache layout"
            )
        cache = self._merge_dynamic_caches(ids)
        output = self.base(
            inputs_embeds=torch.cat(tensors, dim=0),
            past_key_values=cache,
            use_cache=True,
        )
        self._commit_physical_batch(ids, output.past_key_values, token_count)

        token_rows: list[int | tuple[int, ...] | None]
        if self.head is None or token_mode == "none":
            token_rows = [None] * len(ids)
        else:
            selected = (
                output.last_hidden_state
                if token_mode == "all"
                else output.last_hidden_state[:, -1:, :]
            )
            predicted = torch.argmax(self.head(selected), dim=-1)
            if token_mode == "all":
                token_rows = [
                    tuple(int(token) for token in predicted[index].reshape(-1).tolist())
                    for index in range(len(ids))
                ]
            else:
                token_rows = [
                    int(predicted[index].reshape(-1)[-1].item())
                    for index in range(len(ids))
                ]
        return tuple(
            (
                output.last_hidden_state[index : index + 1].contiguous(),
                token_rows[index],
            )
            for index in range(len(ids))
        )

    def _validate_physical_batch_inputs(
        self,
        request_ids: Sequence[int],
        tensors: Sequence[torch.Tensor],
        *,
        expected_hidden_size: int | None,
        integer: bool,
    ) -> tuple[tuple[int, ...], tuple[torch.Tensor, ...], int]:
        ids = tuple(request_ids)
        values = tuple(tensors)
        if not 2 <= len(ids) <= self.MAX_PHYSICAL_BATCH_SIZE:
            raise ValueError(
                "physical batch size must be between 2 and "
                f"{self.MAX_PHYSICAL_BATCH_SIZE}"
            )
        if len(values) != len(ids):
            raise ValueError("request_ids and tensors must have equal length")
        if len(set(ids)) != len(ids):
            raise ValueError("physical batch request IDs must be unique")
        for request_id in ids:
            self._require_active(request_id)

        first = values[0]
        if first.ndim != 2 + (expected_hidden_size is not None) or first.shape[0] != 1:
            kind = "hidden state" if expected_hidden_size is not None else "input_ids"
            raise ValueError(f"each {kind} must have batch size one")
        token_count = int(first.shape[1])
        if token_count < 1:
            raise ValueError("physical batch inputs need at least one token")
        expected_shape = tuple(first.shape)
        for value in values:
            if tuple(value.shape) != expected_shape:
                raise ValueError("physical batch tensors must have identical shapes")
            if value.dtype != first.dtype or value.device != first.device:
                raise ValueError("physical batch tensors must share dtype and device")
            if integer:
                if value.dtype not in (torch.int32, torch.int64):
                    raise TypeError("input_ids must contain integer token IDs")
            elif not value.is_floating_point():
                raise TypeError("hidden states must be floating point")
        if expected_hidden_size is not None and int(first.shape[2]) != expected_hidden_size:
            raise ValueError(
                f"hidden state must have shape [1, tokens, {expected_hidden_size}]"
            )
        return ids, values, token_count

    def _merge_dynamic_caches(self, request_ids: tuple[int, ...]) -> DynamicCache:
        if not self._physical_batch_cache_supported:
            raise ValueError("this model cache layout cannot be physically batched")
        caches = tuple(self.caches.get(request_id) for request_id in request_ids)
        current = self.tokens_seen[request_ids[0]]
        if any(self.tokens_seen[request_id] != current for request_id in request_ids):
            raise ValueError("physical batching requires equal cache lengths")

        merged = DynamicCache(config=self.base.config)
        if current == 0:
            if any(
                cache is not None
                and (
                    not isinstance(cache, DynamicCache)
                    or cache.get_seq_length() != 0
                )
                for cache in caches
            ):
                raise ValueError("empty physical batch has inconsistent caches")
            return merged
        if any(not isinstance(cache, DynamicCache) for cache in caches):
            raise ValueError("physical batching requires DynamicCache request state")
        dynamic_caches = tuple(cache for cache in caches if isinstance(cache, DynamicCache))
        if any(len(cache.layers) != len(merged.layers) for cache in dynamic_caches):
            raise ValueError("physical batch caches have different layer counts")

        for layer_index, target in enumerate(merged.layers):
            sources = tuple(cache.layers[layer_index] for cache in dynamic_caches)
            if type(target) is not DynamicLayer or any(
                type(source) is not DynamicLayer for source in sources
            ):
                raise ValueError("physical batching only supports plain DynamicCache layers")
            if any(
                not source.is_initialized
                or source.keys is None
                or source.values is None
                or int(source.keys.shape[0]) != 1
                or int(source.values.shape[0]) != 1
                or source.get_seq_length() != current
                for source in sources
            ):
                raise ValueError("physical batch cache tensors are inconsistent")
            key_shapes = {tuple(source.keys.shape[1:]) for source in sources}
            value_shapes = {tuple(source.values.shape[1:]) for source in sources}
            if len(key_shapes) != 1 or len(value_shapes) != 1:
                raise ValueError("physical batch cache tensor shapes differ")
            target.update(
                torch.cat([source.keys for source in sources], dim=0),
                torch.cat([source.values for source in sources], dim=0),
            )
        return merged

    def _commit_physical_batch(
        self,
        request_ids: tuple[int, ...],
        cache: Any,
        token_count: int,
    ) -> None:
        if not isinstance(cache, DynamicCache):
            raise TypeError("physical model forward did not return DynamicCache")
        if len(cache.layers) == 0 or any(
            type(layer) is not DynamicLayer
            or not layer.is_initialized
            or layer.keys is None
            or layer.values is None
            or int(layer.keys.shape[0]) != len(request_ids)
            or int(layer.values.shape[0]) != len(request_ids)
            for layer in cache.layers
        ):
            raise TypeError("physical model forward returned an unsupported cache layout")

        split = [DynamicCache(config=self.base.config) for _ in request_ids]
        for layer_index, source in enumerate(cache.layers):
            for batch_index, target_cache in enumerate(split):
                target = target_cache.layers[layer_index]
                if type(target) is not DynamicLayer:
                    raise TypeError("physical cache split changed the cache layout")
                target.update(
                    source.keys[batch_index : batch_index + 1].clone(),
                    source.values[batch_index : batch_index + 1].clone(),
                )
        for request_id, request_cache in zip(request_ids, split, strict=True):
            self.caches[request_id] = request_cache
            self.tokens_seen[request_id] += token_count
        self.model_forward_calls += 1
        self.physical_batch_calls += 1
        self.physical_batch_items += len(request_ids)
        self.max_observed_physical_batch_size = max(
            self.max_observed_physical_batch_size,
            len(request_ids),
        )


def _supports_dynamic_tensor_batching(config: Any) -> bool:
    """Whether the pinned Transformers cache can be losslessly split/merged.

    Plain ``DynamicLayer`` stores all sequence positions and therefore has a
    complete public tensor representation. Sliding, hybrid, static and offload
    layers carry additional cursor/window state and intentionally remain on the
    sequential path until their cache API exposes a lossless batch split.
    """

    try:
        cache = DynamicCache(config=config)
    except (AttributeError, TypeError, ValueError):
        return False
    return bool(cache.layers) and all(type(layer) is DynamicLayer for layer in cache.layers)


def _apply_dynamic_int8(
    base: nn.Module, head: nn.Module | None
) -> tuple[nn.Module, nn.Module | None]:
    """Replace every ``nn.Linear`` with a dynamically quantized INT8 kernel.

    APPROXIMATE mode: activations stay FP32 on the wire and in the KV cache,
    but matmul weights are stored INT8 and requantized per batch, so greedy
    tokens may drift from the FP32 reference. ``inplace=True`` avoids a full
    deepcopy of the resident stage. A bare ``nn.Linear`` head is wrapped in a
    pass-through container because ``convert`` only swaps child modules.
    """

    from torch.ao.quantization import quantize_dynamic

    base = quantize_dynamic(base, {nn.Linear}, dtype=torch.qint8, inplace=True)
    if head is not None:
        head = quantize_dynamic(
            nn.Sequential(head), {nn.Linear}, dtype=torch.qint8, inplace=True
        )
    return base, head


def load_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)


def resolve_model_snapshot(model_name: str, revision: str | None = None) -> str:
    """Resolve one immutable local snapshot before child processes are spawned."""
    local = Path(model_name)
    if local.is_dir():
        return str(local.resolve())
    if local.is_absolute():
        raise FileNotFoundError(f"local model directory does not exist: {local}")
    return str(
        snapshot_download(
            repo_id=model_name,
            revision=revision,
            allow_patterns=[
                "*.safetensors",
                "*.safetensors.index.json",
                "config.json",
                "generation_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "added_tokens.json",
                "*.model",
                "*.tiktoken",
                "chat_template*",
                "merges.txt",
                "vocab.json",
                "vocab.txt",
            ],
        )
    )


def model_snapshot_identity(model_name: str, revision: str | None = None) -> int:
    """Return a deterministic uint64 identity for an immutable model snapshot.

    Hub snapshots are addressed by their content-derived commit identifier, so
    hashing that identifier avoids rereading multi-gigabyte weights on every
    machine. Arbitrary local checkpoints do not have that guarantee; for those
    we hash the exact config and safetensors bytes. Absolute paths are never
    part of the identity, which makes independently cached copies agree.
    """

    snapshot = Path(resolve_model_snapshot(model_name, revision))
    return int.from_bytes(_model_snapshot_digest(snapshot)[:8], "big", signed=False)


def model_artifact_reference(
    model_name: str,
    revision: str | None = None,
    *,
    artifact_identity: str | None = None,
    canonical_source: str | None = None,
    canonical_revision: str | None = None,
) -> ModelArtifactReference:
    """Resolve stable executor coordinates without embedding a host cache path.

    An orchestrator can calculate the identity once and pass the three explicit
    fields to every process. Direct callers remain safe: Hub snapshots use the
    commit in their cache path (constant time), while arbitrary local folders
    are content-hashed because a path alone is not an immutable identity.
    """

    snapshot = Path(resolve_model_snapshot(model_name, revision))
    commit = _hub_snapshot_commit(snapshot)
    if artifact_identity is None:
        digest = _model_snapshot_digest(snapshot)
        identity = "sha256:" + digest.hex()
    else:
        identity = artifact_identity.strip()
        if not identity:
            raise ValueError("artifact_identity cannot be blank")
        digest = _identity_digest(identity)

    if canonical_source is None:
        repository = _hub_snapshot_repository(snapshot)
        if repository is not None:
            source = f"hf://{repository}"
        elif commit is not None:
            source = f"hf-snapshot://{commit}"
        else:
            source = f"content-addressed://{identity}"
    else:
        source = canonical_source.strip()
        if not source:
            raise ValueError("canonical_source cannot be blank")

    if canonical_revision is not None:
        stable_revision = canonical_revision.strip()
        if not stable_revision:
            raise ValueError("canonical_revision cannot be blank")
    elif commit is not None:
        stable_revision = commit
    elif artifact_identity is None:
        stable_revision = identity
    else:
        stable_revision = None
    return ModelArtifactReference(
        identity=identity,
        canonical_source=source,
        canonical_revision=stable_revision,
        snapshot_identity=int.from_bytes(digest[:8], "big", signed=False),
    )


def _model_snapshot_digest(snapshot: Path) -> bytes:
    commit = _hub_snapshot_commit(snapshot)
    digest = hashlib.sha256()
    if commit is not None:
        digest.update(b"gdlp-hub-snapshot-v1\0")
        digest.update(commit.encode("ascii"))
        return digest.digest()

    digest.update(b"gdlp-local-model-v1\0")
    files = sorted(
        (
            path
            for path in snapshot.rglob("*")
            if path.is_file()
            and (path.name == "config.json" or path.name.endswith(".safetensors"))
        ),
        key=lambda path: path.relative_to(snapshot).as_posix(),
    )
    if not files:
        raise FileNotFoundError(
            f"model snapshot {snapshot} contains no config.json or safetensors files"
        )
    for path in files:
        relative = path.relative_to(snapshot).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
    return digest.digest()


def _identity_digest(identity: str) -> bytes:
    if identity.startswith("sha256:"):
        candidate = identity.removeprefix("sha256:")
        if len(candidate) == 64 and all(
            character in "0123456789abcdef" for character in candidate
        ):
            return bytes.fromhex(candidate)
    return hashlib.sha256(b"gdlp-explicit-model-identity-v1\0" + identity.encode()).digest()


def _hub_snapshot_commit(snapshot: Path) -> str | None:
    """Extract a Hub commit only from the canonical ``snapshots/<sha>`` layout."""

    candidate = snapshot.name.lower()
    if snapshot.parent.name != "snapshots":
        return None
    if len(candidate) < 32 or any(
        character not in "0123456789abcdef" for character in candidate
    ):
        return None
    return candidate


def _hub_snapshot_repository(snapshot: Path) -> str | None:
    if _hub_snapshot_commit(snapshot) is None:
        return None
    model_directory = snapshot.parent.parent.name
    if not model_directory.startswith("models--"):
        return None
    coordinates = model_directory.removeprefix("models--").split("--")
    if len(coordinates) != 2 or not all(coordinates):
        return None
    return "/".join(coordinates)


def _load_selective_stage_model(spec: StageModelSpec):
    """Instantiate only this stage and stream only its tensors from safetensors.

    The previous prototype loaded the entire checkpoint in every process and
    discarded unused layers afterwards. That makes a large model impossible on
    a low-memory contributor even if its assigned shard is small. Here the
    architecture is first reduced to the local layer count; checkpoint tensors
    are then copied one at a time from memory-mapped safetensors files.
    """

    snapshot_name = resolve_model_snapshot(spec.model_name, spec.revision)
    resolved_spec = replace(spec, model_name=snapshot_name, revision=None)
    config = AutoConfig.from_pretrained(snapshot_name)
    adapter = resolve_selective_stage_adapter(config)
    adapter.validate_source_config(config, spec.total_layers)
    local_config = copy.deepcopy(config)
    local_layers = spec.layer_end - spec.layer_start
    adapter.slice_config(
        local_config,
        layer_start=spec.layer_start,
        layer_end=spec.layer_end,
        total_layers=spec.total_layers,
    )
    original_vocab_size = int(local_config.vocab_size)
    original_pad_token_id = getattr(local_config, "pad_token_id", None)
    # Intermediate stages never look up token IDs or project logits. Avoid even
    # temporarily allocating a full vocabulary matrix on those contributors.
    if not spec.first and not spec.last:
        local_config.vocab_size = 1
        local_config.pad_token_id = 0
    model = AutoModelForCausalLM.from_config(local_config, dtype=torch.float32)
    adapter.inspect_constructed_model(model, local_layers=local_layers)
    if not spec.last:
        model.model.norm = nn.Identity()
        # Drop an untied vocabulary projection before streaming checkpoint tensors;
        # it is never retained or executed on a non-final contributor.
        model.lm_head = nn.Identity()
    if not spec.first:
        # On a last stage lm_head keeps the original tied vocabulary parameter;
        # replacing the unused lookup table releases the duplicate module path.
        model.model.embed_tokens = nn.Embedding(1, model.config.hidden_size)
    if not spec.first and not spec.last:
        # The reduced vocabulary is only a construction-time allocation trick.
        # Preserve the original semantic config for any forward code that reads it.
        model.config.vocab_size = original_vocab_size
        model.config.pad_token_id = original_pad_token_id
    _load_stage_parameters_from_safetensors(model, resolved_spec, adapter=adapter)
    model.model.config.num_hidden_layers = local_layers
    model.config.num_hidden_layers = local_layers
    model._gdlp_selective_stage_adapter = adapter
    model._gdlp_resolved_snapshot = snapshot_name
    return model.eval()


def _slice_layer_specific_config(config: Any, spec: StageModelSpec) -> None:
    # Architectures such as Gemma/Qwen hybrids may keep one entry per layer for
    # attention type. Preserve the selected original pattern after renumbering.
    for name, value in vars(config).items():
        if isinstance(value, (list, tuple)) and len(value) == spec.total_layers:
            setattr(config, name, value[spec.layer_start : spec.layer_end])


def _load_stage_parameters_from_safetensors(
    model: Any,
    spec: StageModelSpec,
    *,
    adapter: SelectiveStageAdapter | None = None,
    preloaded_checkpoint_tensors: Mapping[str, torch.Tensor] | None = None,
    ignored_checkpoint_names: set[str] | frozenset[str] = frozenset(),
) -> None:
    """Load every required local tensor, with explicit RAM-MoE exceptions.

    ``preloaded_checkpoint_tensors`` is used by the direct MoE loader for the
    small router/shared artifacts it has already authenticated and read.
    ``ignored_checkpoint_names`` is deliberately name-exact: it may account
    only for routed expert tensors that another certified owner retains.  This
    prevents a broad prefix filter from silently omitting attention or router
    state.
    """

    root = _checkpoint_root(spec)
    checkpoint_files = _checkpoint_key_map(root)
    checkpoint_names = set(checkpoint_files)
    preloaded = dict(preloaded_checkpoint_tensors or {})
    ignored = set(ignored_checkpoint_names)
    unknown_preloaded = set(preloaded) - checkpoint_names
    unknown_ignored = ignored - checkpoint_names
    if unknown_preloaded or unknown_ignored:
        raise KeyError(
            "external checkpoint accounting names are absent from the artifact: "
            f"preloaded={sorted(unknown_preloaded)}, ignored={sorted(unknown_ignored)}"
        )
    if set(preloaded).intersection(ignored):
        raise ValueError("a checkpoint tensor cannot be both preloaded and ignored")
    tied_embeddings = bool(getattr(model.config, "tie_word_embeddings", False))
    targets: list[tuple[str, torch.Tensor, str]] = []
    seen_tensors: set[int] = set()
    # state_dict(keep_vars=True) includes parameters and persistent buffers while
    # preserving aliases. Loading only named_parameters silently misses checkpointed
    # buffers on architectures that keep rotary or scaling state persistently.
    for local_name, target in model.state_dict(keep_vars=True).items():
        identity = id(target)
        if identity in seen_tensors:
            continue
        seen_tensors.add(identity)
        assignments = (
            adapter.checkpoint_assignments(
                local_name,
                target,
                layer_start=spec.layer_start,
                checkpoint_names=checkpoint_names,
            )
            if adapter is not None
            else None
        )
        if assignments is not None:
            targets.extend(
                (local_name, assignment.destination, assignment.checkpoint_name)
                for assignment in assignments
            )
            continue
        checkpoint_name = _checkpoint_name(
            local_name,
            spec,
            checkpoint_files,
            tied_embeddings=tied_embeddings,
        )
        if checkpoint_name is None:
            continue
        targets.append((local_name, target, checkpoint_name))

    loaded_checkpoint_names = {checkpoint_name for _, _, checkpoint_name in targets}
    _validate_checkpoint_coverage(
        checkpoint_files,
        loaded_checkpoint_names,
        spec,
        tied_embeddings=tied_embeddings,
        externally_accounted_checkpoint_names=ignored,
    )

    target_checkpoint_names = {
        checkpoint_name for _, _, checkpoint_name in targets
    }
    unused_preloaded = set(preloaded) - target_checkpoint_names
    if unused_preloaded:
        raise KeyError(
            "preloaded checkpoint tensors have no resident destination: "
            f"{sorted(unused_preloaded)}"
        )

    by_file: dict[Path, list[tuple[str, torch.Tensor, str]]] = {}
    for local_name, target, checkpoint_name in targets:
        preloaded_value = preloaded.get(checkpoint_name)
        if preloaded_value is not None:
            if not isinstance(preloaded_value, torch.Tensor):
                raise TypeError(
                    f"preloaded checkpoint tensor {checkpoint_name!r} must be torch.Tensor"
                )
            if tuple(preloaded_value.shape) != tuple(target.shape):
                raise ValueError(
                    f"shape mismatch for {local_name}: preloaded "
                    f"{tuple(preloaded_value.shape)}, stage {tuple(target.shape)}"
                )
            with torch.no_grad():
                target.copy_(preloaded_value)
            continue
        relative_file = checkpoint_files.get(checkpoint_name)
        if relative_file is None:
            raise KeyError(
                f"checkpoint tensor {checkpoint_name!r} required by {local_name!r} is missing"
            )
        by_file.setdefault(root / relative_file, []).append(
            (local_name, target, checkpoint_name)
        )

    with torch.no_grad():
        for file_path, file_targets in by_file.items():
            with safe_open(file_path, framework="pt", device="cpu") as tensors:
                for local_name, target, checkpoint_name in file_targets:
                    value = tensors.get_tensor(checkpoint_name)
                    if tuple(value.shape) != tuple(target.shape):
                        raise ValueError(
                            f"shape mismatch for {local_name}: checkpoint {tuple(value.shape)}, "
                            f"stage {tuple(target.shape)}"
                        )
                    # copy_ performs dtype conversion directly into the resident
                    # destination. An explicit value.to(...) would allocate another
                    # full-size tensor, which is especially damaging for vocab matrices.
                    target.copy_(value)
                    del value


def _checkpoint_root(spec: StageModelSpec) -> Path:
    return Path(resolve_model_snapshot(spec.model_name, spec.revision))


def _checkpoint_key_map(root: Path) -> dict[str, str]:
    indexes = sorted(root.glob("*.safetensors.index.json"))
    if len(indexes) > 1:
        raise ValueError(f"multiple safetensors indexes found under {root}")
    if indexes:
        document = json.loads(indexes[0].read_text(encoding="utf-8"))
        weight_map = document.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("safetensors index has no weight_map")
        mapping: dict[str, str] = {}
        for key, value in weight_map.items():
            if not isinstance(key, str) or not key or not isinstance(value, str) or not value:
                raise ValueError("safetensors index weight_map must contain string keys and files")
            mapping[key] = _validate_checkpoint_file(root, value)
        return mapping

    files = sorted(root.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(
            f"no safetensors checkpoint found under {root}; selective loading "
            "does not fall back to a full PyTorch checkpoint"
        )
    mapping: dict[str, str] = {}
    for file_path in files:
        with safe_open(file_path, framework="pt", device="cpu") as tensors:
            for key in tensors.keys():
                if key in mapping:
                    raise ValueError(f"checkpoint tensor {key!r} appears in multiple files")
                mapping[key] = file_path.name
    return mapping


def _checkpoint_name(
    local_name: str,
    spec: StageModelSpec,
    checkpoint_files: dict[str, str],
    *,
    tied_embeddings: bool = False,
) -> str | None:
    layer_prefix = "model.layers."
    if local_name.startswith(layer_prefix):
        remainder = local_name[len(layer_prefix) :]
        local_index_text, separator, suffix = remainder.partition(".")
        if not separator:
            raise ValueError(f"invalid local layer parameter name {local_name!r}")
        original_index = spec.layer_start + int(local_index_text)
        return f"{layer_prefix}{original_index}.{suffix}"
    if local_name.startswith("model.embed_tokens."):
        if not spec.first:
            return None
        if local_name == "model.embed_tokens.weight" and tied_embeddings:
            return _tied_embedding_checkpoint_name(checkpoint_files)
        return local_name
    if local_name.startswith("model.norm."):
        return local_name if spec.last else None
    if local_name.startswith("model."):
        # Model-level persistent state outside layers (for example a saved rotary
        # buffer) is shared configuration state and is required by every stage.
        return local_name
    if local_name.startswith("lm_head."):
        if not spec.last:
            return None
        if local_name == "lm_head.weight" and tied_embeddings:
            return _tied_embedding_checkpoint_name(checkpoint_files)
        return local_name
    raise KeyError(f"selective loader does not recognize parameter {local_name!r}")


def _tied_embedding_checkpoint_name(checkpoint_files: dict[str, str]) -> str:
    # Use one canonical source on every stage so first-stage lookup and last-stage
    # projection cannot diverge when an unusual checkpoint stores both aliases.
    if "model.embed_tokens.weight" in checkpoint_files:
        return "model.embed_tokens.weight"
    if "lm_head.weight" in checkpoint_files:
        return "lm_head.weight"
    return "model.embed_tokens.weight"


def _validate_checkpoint_coverage(
    checkpoint_files: dict[str, str],
    loaded_checkpoint_names: set[str],
    spec: StageModelSpec,
    *,
    tied_embeddings: bool,
    externally_accounted_checkpoint_names: set[str] | frozenset[str] = frozenset(),
) -> None:
    relevant = {
        name
        for name in checkpoint_files
        if _checkpoint_tensor_belongs_to_stage(name, spec)
    }
    externally_accounted = set(externally_accounted_checkpoint_names)
    outside_stage = externally_accounted - relevant
    if outside_stage:
        raise KeyError(
            "externally accounted checkpoint tensors are outside this stage: "
            f"{sorted(outside_stage)}"
        )
    accounted = set(loaded_checkpoint_names) | externally_accounted
    if tied_embeddings and accounted.intersection(
        {"model.embed_tokens.weight", "lm_head.weight"}
    ):
        accounted.update({"model.embed_tokens.weight", "lm_head.weight"})
    omitted = sorted(relevant - accounted)
    if omitted:
        examples = ", ".join(repr(name) for name in omitted[:8])
        suffix = "" if len(omitted) <= 8 else f" (and {len(omitted) - 8} more)"
        raise KeyError(
            f"stage checkpoint contains {len(omitted)} unconsumed required tensors: "
            f"{examples}{suffix}"
        )


def _checkpoint_tensor_belongs_to_stage(name: str, spec: StageModelSpec) -> bool:
    layer_prefix = "model.layers."
    if name.startswith(layer_prefix):
        remainder = name[len(layer_prefix) :]
        index_text, separator, _ = remainder.partition(".")
        if not separator:
            raise ValueError(f"invalid checkpoint layer tensor name {name!r}")
        try:
            layer_index = int(index_text)
        except ValueError as error:
            raise ValueError(f"invalid checkpoint layer tensor name {name!r}") from error
        return spec.layer_start <= layer_index < spec.layer_end
    if name.startswith("model.embed_tokens."):
        return spec.first
    if name.startswith("model.norm."):
        return spec.last
    if name.startswith("model."):
        return True
    if name.startswith("lm_head."):
        return spec.last
    return False


def _validate_checkpoint_file(root: Path, value: str) -> str:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"checkpoint shard must be a relative path under {root}: {value!r}")
    if relative.suffix != ".safetensors":
        raise ValueError(f"checkpoint shard is not a safetensors file: {value!r}")
    if not (root / relative).is_file():
        raise FileNotFoundError(f"checkpoint shard listed by index is missing: {root / relative}")
    return str(relative)


def _unique_parameter_bytes(*modules: nn.Module | None) -> int:
    seen: set[int] = set()
    total = 0
    for module in modules:
        if module is None:
            continue
        for parameter in module.parameters():
            identity = id(parameter)
            if identity in seen:
                continue
            seen.add(identity)
            total += parameter.numel() * parameter.element_size()
    return total


@torch.inference_mode()
def reference_generate(
    model_name: str,
    input_ids: torch.Tensor,
    output_tokens: int,
    threads: int,
) -> tuple[list[int], dict[str, float]]:
    import time

    if output_tokens < 0:
        raise ValueError("output_tokens cannot be negative")
    if threads < 1:
        raise ValueError("threads must be positive")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
        raise ValueError("input_ids must have shape [1, tokens] with at least one token")
    if output_tokens == 0:
        return [], {"ttft_ms": 0.0, "total_ms": 0.0, "tpot_ms": 0.0}

    torch.set_num_threads(threads)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).eval()
    started = time.perf_counter()
    output = model(input_ids=input_ids, use_cache=True)
    cache = output.past_key_values
    tokens: list[int] = []
    token_times: list[float] = []
    for index in range(output_tokens):
        token = torch.argmax(output.logits[:, -1, :], dim=-1)
        tokens.append(int(token.item()))
        token_times.append(time.perf_counter())
        if index + 1 < output_tokens:
            output = model(input_ids=token[:, None], past_key_values=cache, use_cache=True)
            cache = output.past_key_values
    finished = token_times[-1]
    metrics = {
        "ttft_ms": (token_times[0] - started) * 1_000,
        "total_ms": (finished - started) * 1_000,
        "tpot_ms": mean_intervals_ms(token_times),
    }
    del model
    gc.collect()
    return tokens, metrics


def mean_intervals_ms(times: list[float]) -> float:
    if len(times) < 2:
        return 0.0
    return sum((right - left) * 1_000 for left, right in zip(times, times[1:])) / (
        len(times) - 1
    )
