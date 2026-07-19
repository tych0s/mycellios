from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import gc
import hashlib
import json
from pathlib import Path
from typing import Any, Protocol

from huggingface_hub import snapshot_download
from safetensors import safe_open
import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class StageModelSpec:
    model_name: str
    layer_start: int
    layer_end: int
    total_layers: int
    threads: int
    revision: str | None = None

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
    def __init__(self, spec: StageModelSpec) -> None:
        torch.set_num_threads(spec.threads)
        model = _load_selective_stage_model(spec)
        if not hasattr(model, "model") or not hasattr(model.model, "layers"):
            raise TypeError("the distributed prototype requires a decoder model with model.layers")
        selected = list(model.model.layers)
        # Every stage owns an independent DynamicCache. Local layer indexes keep
        # cache lookup and sequence-length accounting dense and O(number of local layers).
        for local_index, layer in enumerate(selected):
            layer.self_attn.layer_idx = local_index
        self.base = model.model
        self.head = model.lm_head if spec.last else None
        self.hidden_size = int(model.config.hidden_size)
        self.parameter_bytes = _unique_parameter_bytes(self.base, self.head)
        self.loader = "selective-safetensors"
        self.spec = spec
        from .executor_abi import (
            build_stage_executor_manifest,
            model_identity_for_source,
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
            adapter="transformers-selective-safetensors",
            model_identity=model_identity_for_source(spec.model_name, spec.revision),
            model_source=spec.model_name,
            model_revision=spec.revision,
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
            device_kinds=("cpu",),
            compute_apis=("torch",),
            weight_dtypes=weight_dtypes,
            features=(
                "layer-range",
                "rank-local-kv",
                "rollback",
                "selective-load",
            ),
        )
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
        self.caches[request_id] = output.past_key_values
        self.tokens_seen[request_id] += int(input_ids.shape[1])
        return output.last_hidden_state

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
    commit = _hub_snapshot_commit(snapshot)
    digest = hashlib.sha256()
    if commit is not None:
        digest.update(b"gdlp-hub-snapshot-v1\0")
        digest.update(commit.encode("ascii"))
    else:
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
    return int.from_bytes(digest.digest()[:8], "big", signed=False)


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
    actual_layers = int(getattr(config, "num_hidden_layers", 0) or 0)
    if actual_layers != spec.total_layers:
        raise ValueError(
            f"model has {actual_layers} layers, but the stage plan declares "
            f"{spec.total_layers}"
        )
    local_config = copy.deepcopy(config)
    local_layers = spec.layer_end - spec.layer_start
    _slice_layer_specific_config(local_config, spec)
    local_config.num_hidden_layers = local_layers
    original_vocab_size = int(local_config.vocab_size)
    original_pad_token_id = getattr(local_config, "pad_token_id", None)
    # Intermediate stages never look up token IDs or project logits. Avoid even
    # temporarily allocating a full vocabulary matrix on those contributors.
    if not spec.first and not spec.last:
        local_config.vocab_size = 1
        local_config.pad_token_id = 0
    model = AutoModelForCausalLM.from_config(local_config, dtype=torch.float32)
    if (
        not hasattr(model, "model")
        or not hasattr(model.model, "layers")
        or not hasattr(model.model, "norm")
        or not hasattr(model.model, "embed_tokens")
        or not hasattr(model, "lm_head")
    ):
        raise TypeError(
            "selective loading currently requires a decoder with model.layers, "
            "model.norm, model.embed_tokens and lm_head"
        )
    if len(model.model.layers) != local_layers:
        raise ValueError("local model constructor did not honor num_hidden_layers")
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
    _load_stage_parameters_from_safetensors(model, resolved_spec)
    model.model.config.num_hidden_layers = local_layers
    model.config.num_hidden_layers = local_layers
    return model.eval()


def _slice_layer_specific_config(config: Any, spec: StageModelSpec) -> None:
    # Architectures such as Gemma/Qwen hybrids may keep one entry per layer for
    # attention type. Preserve the selected original pattern after renumbering.
    for name, value in vars(config).items():
        if isinstance(value, (list, tuple)) and len(value) == spec.total_layers:
            setattr(config, name, value[spec.layer_start : spec.layer_end])


def _load_stage_parameters_from_safetensors(model: Any, spec: StageModelSpec) -> None:
    root = _checkpoint_root(spec)
    checkpoint_files = _checkpoint_key_map(root)
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
    )

    by_file: dict[Path, list[tuple[str, torch.Tensor, str]]] = {}
    for local_name, target, checkpoint_name in targets:
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
) -> None:
    relevant = {
        name
        for name in checkpoint_files
        if _checkpoint_tensor_belongs_to_stage(name, spec)
    }
    accounted = set(loaded_checkpoint_names)
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
