from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
import struct
from typing import Any, Iterable

from transformers import AutoConfig

from .model import (
    _checkpoint_key_map,
    _hub_snapshot_commit,
    model_snapshot_identity,
    resolve_model_snapshot,
)


PROFILE_SCHEMA = "gdlp-model-profile/1"
_MAX_SAFETENSORS_HEADER_BYTES = 64 * 1024 * 1024
_LAYER_MARKERS = frozenset({"layer", "layers", "h", "block", "blocks"})


@dataclass(frozen=True)
class TensorMetadata:
    name: str
    file: str
    dtype: str
    shape: tuple[int, ...]
    bytes: int


@dataclass(frozen=True)
class ModelProfileOptions:
    layer_prefix: str | None = None
    runtime_overhead_bytes_per_stage: int = 256 * 1024 * 1024
    decode_ms_per_layer_at_unit: float = 1.0
    prefill_ms_per_token_at_unit: float = 0.05
    kv_element_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.layer_prefix is not None and not self.layer_prefix.strip():
            raise ValueError("layer_prefix cannot be blank")
        if self.runtime_overhead_bytes_per_stage < 0:
            raise ValueError("runtime_overhead_bytes_per_stage cannot be negative")
        if self.decode_ms_per_layer_at_unit <= 0:
            raise ValueError("decode_ms_per_layer_at_unit must be positive")
        if self.prefill_ms_per_token_at_unit <= 0:
            raise ValueError("prefill_ms_per_token_at_unit must be positive")
        if self.kv_element_bytes is not None and self.kv_element_bytes not in (1, 2, 4, 8):
            raise ValueError("kv_element_bytes must be one of 1, 2, 4 or 8")


def compile_model_profile(
    model_name: str,
    *,
    revision: str | None = None,
    options: ModelProfileOptions | None = None,
) -> dict[str, Any]:
    """Compile an exact weight layout plus a conservative planner profile.

    Safetensors headers are read directly, so compiling a very large checkpoint
    does not materialize its tensors.  Compute timings remain explicit seed
    estimates until a calibration run replaces them.
    """

    options = options or ModelProfileOptions()
    snapshot = Path(resolve_model_snapshot(model_name, revision))
    config = AutoConfig.from_pretrained(snapshot)
    total_layers = _positive_int_config(
        config,
        "num_hidden_layers",
        "n_layer",
        "num_layers",
    )
    hidden_size = _positive_int_config(config, "hidden_size", "n_embd", "d_model")
    attention_heads = _positive_int_config(config, "num_attention_heads", "n_head")
    kv_heads = _optional_positive_int_config(config, "num_key_value_heads") or attention_heads
    head_dim = _optional_positive_int_config(config, "head_dim") or hidden_size // attention_heads
    if hidden_size % attention_heads != 0 and not _optional_positive_int_config(config, "head_dim"):
        raise ValueError("hidden size is not divisible by attention heads and head_dim is absent")

    tensors = read_checkpoint_metadata(snapshot)
    prefix = _select_layer_prefix(tensors, total_layers, options.layer_prefix)
    layer_tensors = _partition_layer_tensors(tensors, prefix, total_layers)
    missing = [index for index, values in enumerate(layer_tensors) if not values]
    if missing:
        raise ValueError(f"layer prefix {prefix!r} has no tensors for layers {missing}")

    assigned = {tensor.name for values in layer_tensors for tensor in values}
    endpoints = [tensor for tensor in tensors if tensor.name not in assigned]
    embedding_tensors = [tensor for tensor in endpoints if _is_embedding_tensor(tensor.name)]
    head_tensors = [tensor for tensor in endpoints if _is_lm_head_tensor(tensor.name)]
    endpoint_assigned = {tensor.name for tensor in (*embedding_tensors, *head_tensors)}
    residual_tensors = [tensor for tensor in endpoints if tensor.name not in endpoint_assigned]

    embedding_bytes = sum(tensor.bytes for tensor in embedding_tensors)
    lm_head_bytes = sum(tensor.bytes for tensor in head_tensors) + sum(
        tensor.bytes for tensor in residual_tensors
    )
    tied = bool(getattr(config, "tie_word_embeddings", False))
    duplicated_tied_head_bytes = 0
    if tied and embedding_bytes > 0 and not head_tensors:
        # Separate first/last stages each need the shared matrix resident.  The
        # planner deliberately budgets that duplication even if storage has one tensor.
        duplicated_tied_head_bytes = embedding_bytes
        lm_head_bytes += duplicated_tied_head_bytes

    kv_element_bytes = options.kv_element_bytes or _dtype_bytes(config)
    kv_bytes_per_token = 2 * kv_heads * head_dim * kv_element_bytes
    profile_layers = [
        {
            "index": index,
            "weightBytes": sum(tensor.bytes for tensor in values),
            "activationElements": hidden_size,
            "kvBytesPerToken": kv_bytes_per_token,
            "decodeMsAtUnit": options.decode_ms_per_layer_at_unit,
            "prefillMsPerTokenAtUnit": options.prefill_ms_per_token_at_unit,
        }
        for index, values in enumerate(layer_tensors)
    ]

    architecture = _first_string(getattr(config, "architectures", None))
    selective_compatible = prefix.endswith("model.layers") or prefix == "model.layers"
    compatibility_reasons: list[str] = []
    if not selective_compatible:
        compatibility_reasons.append(
            f"current selective backend expects model.layers, checkpoint uses {prefix}"
        )
    if not embedding_tensors:
        compatibility_reasons.append("embedding tensor was not recognized")
    if not head_tensors and not tied:
        compatibility_reasons.append("untied language-model head was not recognized")

    storage_bytes = sum(tensor.bytes for tensor in tensors)
    planned_weight_bytes = (
        sum(layer["weightBytes"] for layer in profile_layers)
        + embedding_bytes
        + lm_head_bytes
    )
    return {
        "schema": PROFILE_SCHEMA,
        "source": {
            "model": model_name,
            "revision": revision,
            "snapshotCommit": _hub_snapshot_commit(snapshot),
            "snapshotIdentityUint64Hex": f"{model_snapshot_identity(str(snapshot)):016x}",
            "format": "safetensors",
            "storageBytes": storage_bytes,
            "tensorCount": len(tensors),
        },
        "inspection": {
            "architecture": architecture,
            "layerPrefix": prefix,
            "hiddenSize": hidden_size,
            "attentionHeads": attention_heads,
            "keyValueHeads": kv_heads,
            "headDim": head_dim,
            "kvElementBytes": kv_element_bytes,
            "calibrationRequired": True,
            "duplicatedTiedHeadBytes": duplicated_tied_head_bytes,
        },
        "compatibility": {
            "selectiveSafetensors": selective_compatible and not compatibility_reasons,
            "requiresAdapter": not (selective_compatible and not compatibility_reasons),
            "reasons": compatibility_reasons,
        },
        "model": {
            "id": model_name,
            "layers": profile_layers,
            "embeddingBytes": embedding_bytes,
            "lmHeadBytes": lm_head_bytes,
            "tiedEmbeddingAndHead": tied,
            "runtimeOverheadBytesPerStage": options.runtime_overhead_bytes_per_stage,
            "embeddingDecodeMsAtUnit": 0.0,
            "lmHeadDecodeMsAtUnit": 0.0,
            "embeddingPrefillMsPerTokenAtUnit": 0.0,
            "lmHeadPrefillMsPerTokenAtUnit": 0.0,
        },
        "accounting": {
            "checkpointStorageBytes": storage_bytes,
            "plannedResidentWeightBytesAcrossSplitEndpoints": planned_weight_bytes,
        },
    }


def read_checkpoint_metadata(snapshot: Path) -> list[TensorMetadata]:
    mapping = _checkpoint_key_map(snapshot)
    headers: dict[str, dict[str, Any]] = {}
    result: list[TensorMetadata] = []
    for name, relative_file in sorted(mapping.items()):
        header = headers.get(relative_file)
        if header is None:
            header = _read_safetensors_header(snapshot / relative_file)
            headers[relative_file] = header
        entry = header.get(name)
        if not isinstance(entry, dict):
            raise ValueError(f"tensor {name!r} is missing from {relative_file}")
        dtype = entry.get("dtype")
        shape = entry.get("shape")
        offsets = entry.get("data_offsets")
        if (
            not isinstance(dtype, str)
            or not isinstance(shape, list)
            or not all(isinstance(value, int) and value >= 0 for value in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(value, int) and value >= 0 for value in offsets)
            or offsets[1] < offsets[0]
        ):
            raise ValueError(f"invalid safetensors metadata for {name!r}")
        result.append(
            TensorMetadata(
                name=name,
                file=relative_file,
                dtype=dtype,
                shape=tuple(shape),
                bytes=offsets[1] - offsets[0],
            )
        )
    if not result:
        raise ValueError("checkpoint contains no tensors")
    return result


def _read_safetensors_header(path: Path) -> dict[str, Any]:
    size = path.stat().st_size
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"truncated safetensors header in {path}")
        (header_bytes,) = struct.unpack("<Q", raw_length)
        if header_bytes < 2 or header_bytes > _MAX_SAFETENSORS_HEADER_BYTES:
            raise ValueError(f"invalid safetensors header length in {path}")
        if 8 + header_bytes > size:
            raise ValueError(f"safetensors header exceeds file size in {path}")
        raw_header = handle.read(header_bytes)
    try:
        document = json.loads(raw_header)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid safetensors JSON header in {path}") from error
    if not isinstance(document, dict):
        raise ValueError(f"safetensors header is not an object in {path}")
    return document


def _select_layer_prefix(
    tensors: Iterable[TensorMetadata],
    total_layers: int,
    override: str | None,
) -> str:
    if override is not None:
        return override.strip().rstrip(".")
    candidates: dict[str, tuple[set[int], int]] = {}
    for tensor in tensors:
        parts = tensor.name.split(".")
        for position in range(1, len(parts)):
            if parts[position - 1].lower() not in _LAYER_MARKERS:
                continue
            if not re.fullmatch(r"\d+", parts[position]):
                continue
            index = int(parts[position])
            if index >= total_layers:
                continue
            prefix = ".".join(parts[:position])
            indexes, total_bytes = candidates.get(prefix, (set(), 0))
            candidates[prefix] = (indexes | {index}, total_bytes + tensor.bytes)
    complete = [
        (prefix, total_bytes)
        for prefix, (indexes, total_bytes) in candidates.items()
        if indexes == set(range(total_layers))
    ]
    if not complete:
        summary = {prefix: len(indexes) for prefix, (indexes, _) in candidates.items()}
        raise ValueError(
            f"could not infer a complete {total_layers}-layer tensor prefix; candidates={summary}"
        )
    return max(complete, key=lambda item: item[1])[0]


def _partition_layer_tensors(
    tensors: Iterable[TensorMetadata],
    prefix: str,
    total_layers: int,
) -> list[list[TensorMetadata]]:
    result: list[list[TensorMetadata]] = [[] for _ in range(total_layers)]
    pattern = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.")
    for tensor in tensors:
        match = pattern.match(tensor.name)
        if not match:
            continue
        index = int(match.group(1))
        if 0 <= index < total_layers:
            result[index].append(tensor)
    return result


def _is_embedding_tensor(name: str) -> bool:
    lowered = name.lower()
    return any(
        marker in lowered
        for marker in (
            "embed_tokens.",
            "word_embeddings.",
            "wte.",
            "embed_in.",
            "tok_embeddings.",
        )
    )


def _is_lm_head_tensor(name: str) -> bool:
    lowered = name.lower()
    return (
        lowered.startswith("lm_head.")
        or ".lm_head." in lowered
        or lowered.startswith("output.")
        or lowered.endswith(".output.weight")
        or lowered.startswith("embed_out.")
    )


def _positive_int_config(config: Any, *names: str) -> int:
    value = _optional_positive_int_config(config, *names)
    if value is None:
        raise ValueError(f"model config has no positive integer among {names}")
    return value


def _optional_positive_int_config(config: Any, *names: str) -> int | None:
    for name in names:
        value = getattr(config, name, None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def _dtype_bytes(config: Any) -> int:
    value = getattr(config, "dtype", None) or getattr(config, "torch_dtype", None)
    normalized = str(value).lower()
    if any(marker in normalized for marker in ("float64", "double", "int64")):
        return 8
    if any(marker in normalized for marker in ("float32", "int32")):
        return 4
    if any(marker in normalized for marker in ("float16", "bfloat16", "half", "int16")):
        return 2
    if any(marker in normalized for marker in ("float8", "int8", "uint8")):
        return 1
    # Runtime KV is conservatively assumed FP16 when a checkpoint omits dtype.
    return 2


def _first_string(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return next((item for item in value if isinstance(item, str)), None)
    return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile a GDLP model profile without loading checkpoint tensors"
    )
    parser.add_argument("model", help="Hugging Face model id or local snapshot")
    parser.add_argument("--revision")
    parser.add_argument("--layer-prefix")
    parser.add_argument("--runtime-overhead-mib", type=float, default=256.0)
    parser.add_argument("--decode-ms-per-layer", type=float, default=1.0)
    parser.add_argument("--prefill-ms-per-token-per-layer", type=float, default=0.05)
    parser.add_argument("--kv-element-bytes", type=int, choices=(1, 2, 4, 8))
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    runtime_bytes = args.runtime_overhead_mib * 1024 * 1024
    if not math.isfinite(runtime_bytes) or runtime_bytes < 0:
        raise ValueError("runtime overhead must be a finite non-negative value")
    result = compile_model_profile(
        args.model,
        revision=args.revision,
        options=ModelProfileOptions(
            layer_prefix=args.layer_prefix,
            runtime_overhead_bytes_per_stage=round(runtime_bytes),
            decode_ms_per_layer_at_unit=args.decode_ms_per_layer,
            prefill_ms_per_token_at_unit=args.prefill_ms_per_token_per_layer,
            kv_element_bytes=args.kv_element_bytes,
        ),
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
