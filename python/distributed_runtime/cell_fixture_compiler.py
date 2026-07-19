from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile
from typing import Any, BinaryIO, Mapping, Sequence

from safetensors import safe_open
import torch
from transformers import AutoConfig

from .cell_parallel import (
    balanced_shard,
    llama_attention_shard_plan,
    weighted_shard,
)
from .cell_stage import _read_manifest
from .model import _checkpoint_key_map, _hub_snapshot_commit, resolve_model_snapshot


CELL_FIXTURE_SCHEMA = "gdlp-llama-cell-stage/2"
CELL_FIXTURE_SOURCE_SCHEMA = "gdlp-hf-cell-fixture-source/1"

_OUTPUT_DTYPES: dict[str, tuple[torch.dtype, str, int]] = {
    "float32": (torch.float32, "F32", 4),
    "float16": (torch.float16, "F16", 2),
    "bfloat16": (torch.bfloat16, "BF16", 2),
}

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

_HF_SUFFIXES = {
    "input_norm": "input_layernorm.weight",
    "post_attention_norm": "post_attention_layernorm.weight",
    "query": "self_attn.q_proj.weight",
    "key": "self_attn.k_proj.weight",
    "value": "self_attn.v_proj.weight",
    "output": "self_attn.o_proj.weight",
    "gate": "mlp.gate_proj.weight",
    "up": "mlp.up_proj.weight",
    "down": "mlp.down_proj.weight",
}

_UNSUPPORTED_SUFFIXES = (
    "self_attn.q_proj.bias",
    "self_attn.k_proj.bias",
    "self_attn.v_proj.bias",
    "self_attn.o_proj.bias",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "mlp.gate_proj.bias",
    "mlp.up_proj.bias",
    "mlp.down_proj.bias",
)


@dataclass(frozen=True)
class CellFixtureCompilation:
    destination: str
    schema: str
    source_model: str
    source_revision: str | None
    snapshot_commit: str | None
    layer_start: int
    layer_end: int
    world_size: int
    rank_weights: tuple[float, ...]
    output_dtype: str
    source_tensor_count: int
    source_tensor_bytes: int
    output_tensor_count: int
    output_tensor_bytes: int
    manifest_sha256: str
    shard_sha256: tuple[str, ...]
    rank_fixed_bytes: tuple[int, ...]
    rank_kv_bytes_per_token: tuple[int, ...]


@dataclass(frozen=True)
class _LlamaShape:
    total_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_epsilon: float
    rope_theta: float
    model_type: str | None
    architecture: str | None


class _CheckpointTensorReader:
    """Lazily mmap checkpoint shards and materialize one requested tensor."""

    def __init__(self, root: Path, key_map: Mapping[str, str]) -> None:
        self._root = root
        self._key_map = key_map
        self._stack = ExitStack()
        self._files: dict[str, Any] = {}

    def get_tensor(self, name: str) -> torch.Tensor:
        relative = self._key_map.get(name)
        if relative is None:
            raise KeyError(f"required checkpoint tensor is missing: {name}")
        tensors = self._files.get(relative)
        if tensors is None:
            tensors = self._stack.enter_context(
                safe_open(self._root / relative, framework="pt", device="cpu")
            )
            self._files[relative] = tensors
        try:
            return tensors.get_tensor(name)
        except KeyError as error:
            raise KeyError(
                f"checkpoint index maps {name!r} to {relative!r}, but the tensor is absent"
            ) from error

    def close(self) -> None:
        self._stack.close()

    def __enter__(self) -> _CheckpointTensorReader:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class _StreamingSafeTensorWriter:
    """Write one typed SafeTensors shard in a bounded-memory order."""

    def __init__(
        self,
        path: Path,
        tensors: Sequence[tuple[str, tuple[int, ...]]],
        output_dtype: str,
    ) -> None:
        if sys.byteorder != "little":
            raise RuntimeError("streaming SafeTensors output requires a little-endian host")
        self.path = path
        try:
            self.torch_dtype, safe_dtype, element_bytes = _OUTPUT_DTYPES[output_dtype]
        except KeyError as error:
            raise ValueError("output dtype must be float32, float16 or bfloat16") from error
        self._tensors = tuple(tensors)
        self._next = 0
        offset = 0
        header: dict[str, dict[str, Any]] = {}
        for name, shape in self._tensors:
            size = math.prod(shape) * element_bytes
            header[name] = {
                "dtype": safe_dtype,
                "shape": list(shape),
                "data_offsets": [offset, offset + size],
            }
            offset += size
        self.data_bytes = offset
        self.sha256_hex: str | None = None
        self._digest = hashlib.sha256()
        raw_header = json.dumps(
            header,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        raw_header += b" " * (-len(raw_header) % 8)
        self._data_start = 8 + len(raw_header)
        self._handle: BinaryIO = path.open("xb")
        self._write_raw(struct.pack("<Q", len(raw_header)))
        self._write_raw(raw_header)

    def write(self, name: str, value: torch.Tensor) -> None:
        if self._next >= len(self._tensors):
            raise RuntimeError(f"unexpected tensor after end of SafeTensors stream: {name}")
        expected_name, expected_shape = self._tensors[self._next]
        if name != expected_name:
            raise RuntimeError(
                f"out-of-order SafeTensors write: got {name!r}, expected {expected_name!r}"
            )
        if value.dtype != self.torch_dtype or value.device.type != "cpu":
            raise TypeError(
                f"streaming SafeTensors values must be CPU {self.torch_dtype} tensors"
            )
        if tuple(value.shape) != expected_shape:
            raise ValueError(
                f"output tensor {name!r} has shape {tuple(value.shape)}, "
                f"expected {expected_shape}"
            )
        contiguous = value.detach().contiguous()
        # NumPy has no native bfloat16 dtype. Viewing every supported floating
        # format as uint8 preserves its exact little-endian bit pattern.
        self._write_raw(memoryview(contiguous.view(torch.uint8).numpy()).cast("B"))
        self._next += 1

    def finish(self) -> None:
        if self._next != len(self._tensors):
            missing = [name for name, _ in self._tensors[self._next :]]
            raise RuntimeError(f"SafeTensors stream ended before tensors {missing}")
        expected_size = self._data_start + self.data_bytes
        if self._handle.tell() != expected_size:
            raise RuntimeError(
                f"SafeTensors stream size {self._handle.tell()} != expected {expected_size}"
            )
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()
        self.sha256_hex = self._digest.hexdigest()

    def _write_raw(self, value: bytes | memoryview) -> None:
        self._digest.update(value)
        self._handle.write(value)

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()


def compile_hf_llama_cell_fixture(
    model_name: str,
    destination: str | Path,
    *,
    layer_start: int,
    layer_end: int,
    world_size: int,
    revision: str | None = None,
    output_dtype: str = "float32",
    rank_weights: Sequence[float] | None = None,
) -> CellFixtureCompilation:
    """Compile a contiguous HF Llama layer range into a TP cell fixture.

    The checkpoint is never instantiated as a Transformers model. Its index or
    SafeTensors headers are inspected once, and then exactly one selected source
    tensor is materialized at a time. Slicing happens before FP32 conversion so
    peak compilation memory is bounded by one source tensor plus one rank-local
    output slice, rather than the full model or full selected stage. The output
    dtype is explicit so GPU ranks do not inflate FP16/BF16 checkpoints to
    FP32 merely to satisfy the transport prototype.
    """

    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name cannot be blank")
    for name, value in (("layer_start", layer_start), ("layer_end", layer_end)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer")
    if not isinstance(world_size, int) or isinstance(world_size, bool):
        raise TypeError("world_size must be an integer")
    if world_size < 2:
        raise ValueError("a tensor-parallel cell requires at least two members")
    normalized_rank_weights = _normalize_rank_weights(world_size, rank_weights)
    if output_dtype not in _OUTPUT_DTYPES:
        raise ValueError("output_dtype must be float32, float16 or bfloat16")
    output_torch_dtype, _, output_element_bytes = _OUTPUT_DTYPES[output_dtype]

    snapshot = Path(resolve_model_snapshot(model_name, revision)).resolve()
    config = AutoConfig.from_pretrained(str(snapshot), local_files_only=True)
    shape = _llama_shape(config)
    if not 0 <= layer_start < layer_end <= shape.total_layers:
        raise ValueError(
            f"layer range [{layer_start}, {layer_end}) is outside "
            f"[0, {shape.total_layers})"
        )
    if layer_end - layer_start < 2:
        raise ValueError("a version-2 cell fixture requires at least two layers")
    _validate_world_size(shape, world_size, normalized_rank_weights)

    key_map = _checkpoint_key_map(snapshot)
    _validate_checkpoint_layout(key_map, layer_start, layer_end)
    destination_path = Path(destination).resolve()
    if destination_path.exists():
        raise FileExistsError(f"cell fixture destination already exists: {destination_path}")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination_path.name}.compile-",
            dir=destination_path.parent,
        )
    )

    writers: list[_StreamingSafeTensorWriter] = []
    source_tensor_bytes = 0
    source_dtypes: set[str] = set()
    source_tensor_count = 0
    try:
        for rank in range(world_size):
            writers.append(
                _StreamingSafeTensorWriter(
                    temporary / f"rank-{rank:03d}.safetensors",
                    _rank_tensor_layout(
                        shape,
                        layer_end - layer_start,
                        rank,
                        world_size,
                        normalized_rank_weights,
                    ),
                    output_dtype,
                )
            )

        with _CheckpointTensorReader(snapshot, key_map) as reader:
            for local_index, source_index in enumerate(range(layer_start, layer_end)):
                for logical_name in _TENSOR_NAMES:
                    checkpoint_name = _checkpoint_tensor_name(source_index, logical_name)
                    source = reader.get_tensor(checkpoint_name)
                    _validate_source_tensor(source, checkpoint_name, logical_name, shape)
                    source_tensor_count += 1
                    source_tensor_bytes += source.numel() * source.element_size()
                    source_dtypes.add(str(source.dtype).removeprefix("torch."))
                    for rank, writer in enumerate(writers):
                        local = _local_tensor(
                            source,
                            logical_name,
                            shape,
                            rank,
                            world_size,
                            output_torch_dtype,
                            normalized_rank_weights,
                        )
                        writer.write(f"layers.{local_index}.{logical_name}", local)
                        del local
                    del source

        for writer in writers:
            writer.finish()

        shard_sha256 = []
        for writer in writers:
            if writer.sha256_hex is None:
                raise RuntimeError("cell shard writer has no completed SHA-256")
            shard_sha256.append(writer.sha256_hex)
        rank_fixed_bytes = [writer.data_bytes for writer in writers]
        rank_kv_bytes_per_token = [
            _rank_kv_bytes_per_token(
                shape,
                layer_end - layer_start,
                rank,
                world_size,
                output_element_bytes,
                normalized_rank_weights,
            )
            for rank in range(world_size)
        ]
        manifest = _manifest(
            shape,
            layer_end - layer_start,
            world_size,
            shard_sha256,
            rank_fixed_bytes,
            rank_kv_bytes_per_token,
            output_dtype,
            normalized_rank_weights,
        )
        (temporary / "cell.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest_sha256 = _sha256_file(temporary / "cell.json")
        output_tensor_count = (layer_end - layer_start) * len(_TENSOR_NAMES) * world_size
        output_tensor_bytes = sum(writer.data_bytes for writer in writers)
        provenance = {
            "schema": CELL_FIXTURE_SOURCE_SCHEMA,
            "targetSchema": CELL_FIXTURE_SCHEMA,
            "source": {
                "model": model_name,
                "revision": revision,
                "snapshotCommit": _hub_snapshot_commit(snapshot),
                "modelType": shape.model_type,
                "architecture": shape.architecture,
                "format": "safetensors",
                "dtypes": sorted(source_dtypes),
            },
            "layerRange": {
                "start": layer_start,
                "end": layer_end,
                "count": layer_end - layer_start,
            },
            "worldSize": world_size,
            "rankWeights": list(
                normalized_rank_weights
                or tuple(1.0 for _ in range(world_size))
            ),
            "sourceTensorCount": source_tensor_count,
            "sourceTensorBytes": source_tensor_bytes,
            "outputTensorCount": output_tensor_count,
            "outputTensorBytes": output_tensor_bytes,
            "outputDtype": output_dtype,
            "shardSha256": shard_sha256,
            "cellManifestSha256": manifest_sha256,
            "rankFixedBytes": rank_fixed_bytes,
            "rankKvBytesPerToken": rank_kv_bytes_per_token,
            "memoryPolicy": (
                "one-source-tensor-at-a-time; slice-before-output-dtype-conversion"
            ),
        }
        (temporary / "source.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        # Use the runtime's own reader as the final compatibility gate before
        # publishing the atomically completed fixture directory.
        _read_manifest(temporary)
        for rank in range(world_size):
            with safe_open(
                temporary / f"rank-{rank:03d}.safetensors",
                framework="pt",
                device="cpu",
            ) as shard:
                expected = {
                    f"layers.{layer}.{name}"
                    for layer in range(layer_end - layer_start)
                    for name in _TENSOR_NAMES
                }
                if set(shard.keys()) != expected:
                    raise RuntimeError(f"rank {rank} fixture tensor set is incomplete")

        # ``rename`` deliberately fails if another actor creates the target
        # while compilation is running; never replace a user's directory.
        temporary.rename(destination_path)
        return CellFixtureCompilation(
            destination=str(destination_path),
            schema=CELL_FIXTURE_SCHEMA,
            source_model=model_name,
            source_revision=revision,
            snapshot_commit=_hub_snapshot_commit(snapshot),
            layer_start=layer_start,
            layer_end=layer_end,
            world_size=world_size,
            rank_weights=tuple(
                normalized_rank_weights
                or tuple(1.0 for _ in range(world_size))
            ),
            output_dtype=output_dtype,
            source_tensor_count=source_tensor_count,
            source_tensor_bytes=source_tensor_bytes,
            output_tensor_count=output_tensor_count,
            output_tensor_bytes=output_tensor_bytes,
            manifest_sha256=manifest_sha256,
            shard_sha256=tuple(shard_sha256),
            rank_fixed_bytes=tuple(rank_fixed_bytes),
            rank_kv_bytes_per_token=tuple(rank_kv_bytes_per_token),
        )
    finally:
        for writer in writers:
            writer.close()
        if temporary.exists():
            shutil.rmtree(temporary)


def _llama_shape(config: Any) -> _LlamaShape:
    total_layers = _positive_integer(config, "num_hidden_layers")
    hidden_size = _positive_integer(config, "hidden_size")
    intermediate_size = _positive_integer(config, "intermediate_size")
    num_attention_heads = _positive_integer(config, "num_attention_heads")
    num_key_value_heads = _optional_positive_integer(config, "num_key_value_heads")
    if num_key_value_heads is None:
        num_key_value_heads = num_attention_heads
    head_dim = _optional_positive_integer(config, "head_dim")
    if head_dim is None:
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "hidden_size is not divisible by num_attention_heads and head_dim is absent"
            )
        head_dim = hidden_size // num_attention_heads
    rms_norm_epsilon = _positive_number(config, "rms_norm_eps")
    rope_scaling = getattr(config, "rope_scaling", None)
    configured_theta = _optional_positive_number(config, "rope_theta")
    if configured_theta is None and isinstance(rope_scaling, dict):
        raw_theta = rope_scaling.get("rope_theta")
        if (
            isinstance(raw_theta, (int, float))
            and not isinstance(raw_theta, bool)
            and math.isfinite(float(raw_theta))
            and raw_theta > 0
        ):
            configured_theta = float(raw_theta)
    rope_theta = configured_theta or 10_000.0

    activation = getattr(config, "hidden_act", "silu")
    if activation not in ("silu", None):
        raise ValueError(f"unsupported Llama MLP activation for cell runtime: {activation!r}")
    if bool(getattr(config, "attention_bias", False)):
        raise ValueError("attention projection bias is unsupported by the cell runtime")
    if bool(getattr(config, "mlp_bias", False)):
        raise ValueError("MLP projection bias is unsupported by the cell runtime")
    # Transformers 5 normalizes ordinary RoPE into a dictionary named
    # ``rope_scaling``/``rope_parameters`` even when no scaling is active.
    # Only that default representation is equivalent to this runtime's theta.
    if isinstance(rope_scaling, dict):
        rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", "default"))
        allowed_keys = {"rope_type", "type", "rope_theta"}
        scaling_is_default = (
            rope_type == "default"
            and set(rope_scaling).issubset(allowed_keys)
            and math.isclose(
                float(rope_scaling.get("rope_theta", rope_theta)),
                rope_theta,
            )
        )
    else:
        scaling_is_default = rope_scaling is None
    if not scaling_is_default:
        raise ValueError("RoPE scaling is unsupported by the current cell runtime")
    if bool(getattr(config, "use_sliding_window", False)):
        raise ValueError("sliding-window attention is unsupported by the cell runtime")
    layer_types = getattr(config, "layer_types", None)
    if layer_types is not None and any(value != "full_attention" for value in layer_types):
        raise ValueError("hybrid attention layer types are unsupported by the cell runtime")
    partial_rotary = getattr(config, "partial_rotary_factor", 1.0)
    if not math.isclose(float(partial_rotary), 1.0):
        raise ValueError("partial rotary embeddings are unsupported by the cell runtime")
    if head_dim % 2:
        raise ValueError("the current Llama RoPE implementation requires an even head_dim")

    architecture = None
    architectures = getattr(config, "architectures", None)
    if isinstance(architectures, (list, tuple)) and architectures:
        architecture = str(architectures[0])
    model_type = getattr(config, "model_type", None)
    return _LlamaShape(
        total_layers=total_layers,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        rms_norm_epsilon=rms_norm_epsilon,
        rope_theta=rope_theta,
        model_type=None if model_type is None else str(model_type),
        architecture=architecture,
    )


def _normalize_rank_weights(
    world_size: int,
    rank_weights: Sequence[float] | None,
) -> tuple[float, ...] | None:
    if rank_weights is None:
        return None
    if isinstance(rank_weights, (str, bytes, bytearray)):
        raise TypeError("rank_weights must be an ordered numeric sequence")
    values = tuple(rank_weights)
    if len(values) != world_size:
        raise ValueError("rank_weights length must equal world_size")
    normalized: list[float] = []
    for value in values:
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or value <= 0
        ):
            raise ValueError("rank_weights must contain finite positive numbers")
        normalized.append(float(value))
    return tuple(normalized)


def _validate_world_size(
    shape: _LlamaShape,
    world_size: int,
    rank_weights: Sequence[float] | None,
) -> None:
    for rank in range(world_size):
        llama_attention_shard_plan(
            shape.hidden_size,
            shape.num_attention_heads,
            shape.num_key_value_heads,
            shape.head_dim,
            rank,
            world_size,
            rank_weights,
        )
        if rank_weights is None:
            balanced_shard(shape.intermediate_size, rank, world_size)
        else:
            weighted_shard(shape.intermediate_size, rank, rank_weights)


def _validate_checkpoint_layout(
    key_map: Mapping[str, str],
    layer_start: int,
    layer_end: int,
) -> None:
    missing: list[str] = []
    unsupported: list[str] = []
    for layer in range(layer_start, layer_end):
        prefix = f"model.layers.{layer}."
        for logical_name in _TENSOR_NAMES:
            name = _checkpoint_tensor_name(layer, logical_name)
            if name not in key_map:
                missing.append(name)
        for suffix in _UNSUPPORTED_SUFFIXES:
            name = prefix + suffix
            if name in key_map:
                unsupported.append(name)
    if missing:
        raise ValueError(f"checkpoint is missing required Llama tensors: {missing}")
    if unsupported:
        raise ValueError(f"checkpoint uses unsupported Llama tensors: {unsupported}")


def _checkpoint_tensor_name(layer: int, logical_name: str) -> str:
    return f"model.layers.{layer}.{_HF_SUFFIXES[logical_name]}"


def _source_shape(logical_name: str, shape: _LlamaShape) -> tuple[int, ...]:
    hidden = shape.hidden_size
    query = shape.num_attention_heads * shape.head_dim
    key_value = shape.num_key_value_heads * shape.head_dim
    return {
        "input_norm": (hidden,),
        "post_attention_norm": (hidden,),
        "query": (query, hidden),
        "key": (key_value, hidden),
        "value": (key_value, hidden),
        "output": (hidden, query),
        "gate": (shape.intermediate_size, hidden),
        "up": (shape.intermediate_size, hidden),
        "down": (hidden, shape.intermediate_size),
    }[logical_name]


def _validate_source_tensor(
    value: torch.Tensor,
    checkpoint_name: str,
    logical_name: str,
    shape: _LlamaShape,
) -> None:
    expected = _source_shape(logical_name, shape)
    if tuple(value.shape) != expected:
        raise ValueError(
            f"checkpoint tensor {checkpoint_name!r} has shape {tuple(value.shape)}, "
            f"expected {expected}"
        )
    if not value.is_floating_point():
        raise TypeError(f"checkpoint tensor {checkpoint_name!r} is not floating point")


def _local_shape(
    logical_name: str,
    shape: _LlamaShape,
    rank: int,
    world_size: int,
    rank_weights: Sequence[float] | None,
) -> tuple[int, ...]:
    plan = llama_attention_shard_plan(
        shape.hidden_size,
        shape.num_attention_heads,
        shape.num_key_value_heads,
        shape.head_dim,
        rank,
        world_size,
        rank_weights,
    )
    query = plan.local_query_heads * shape.head_dim
    key_value = plan.local_key_value_heads * shape.head_dim
    intermediate = (
        balanced_shard(shape.intermediate_size, rank, world_size)
        if rank_weights is None
        else weighted_shard(shape.intermediate_size, rank, rank_weights)
    ).size
    return {
        "input_norm": (shape.hidden_size,),
        "post_attention_norm": (shape.hidden_size,),
        "query": (query, shape.hidden_size),
        "key": (key_value, shape.hidden_size),
        "value": (key_value, shape.hidden_size),
        "output": (shape.hidden_size, query),
        "gate": (intermediate, shape.hidden_size),
        "up": (intermediate, shape.hidden_size),
        "down": (shape.hidden_size, intermediate),
    }[logical_name]


def _rank_tensor_layout(
    shape: _LlamaShape,
    layer_count: int,
    rank: int,
    world_size: int,
    rank_weights: Sequence[float] | None,
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    return tuple(
        (
            f"layers.{layer}.{logical_name}",
            _local_shape(
                logical_name, shape, rank, world_size, rank_weights
            ),
        )
        for layer in range(layer_count)
        for logical_name in _TENSOR_NAMES
    )


def _local_tensor(
    source: torch.Tensor,
    logical_name: str,
    shape: _LlamaShape,
    rank: int,
    world_size: int,
    output_dtype: torch.dtype,
    rank_weights: Sequence[float] | None,
) -> torch.Tensor:
    attention = llama_attention_shard_plan(
        shape.hidden_size,
        shape.num_attention_heads,
        shape.num_key_value_heads,
        shape.head_dim,
        rank,
        world_size,
        rank_weights,
    )
    intermediate = (
        balanced_shard(shape.intermediate_size, rank, world_size)
        if rank_weights is None
        else weighted_shard(shape.intermediate_size, rank, rank_weights)
    )
    if logical_name in ("input_norm", "post_attention_norm"):
        local = source
    elif logical_name == "query":
        start = attention.query_heads.start * shape.head_dim
        end = attention.query_heads.end * shape.head_dim
        local = source[start:end, :]
    elif logical_name in ("key", "value"):
        # These ranges are disjoint on the ordinary GQA path. For MQA or GQA
        # with more ranks than KV heads they may overlap deliberately: each
        # rank file receives only the KV heads referenced by its local Q slice.
        start = attention.key_value_heads.start * shape.head_dim
        end = attention.key_value_heads.end * shape.head_dim
        local = source[start:end, :]
    elif logical_name == "output":
        start = attention.query_heads.start * shape.head_dim
        end = attention.query_heads.end * shape.head_dim
        local = source[:, start:end]
    elif logical_name in ("gate", "up"):
        local = source[intermediate.start : intermediate.end, :]
    elif logical_name == "down":
        local = source[:, intermediate.start : intermediate.end]
    else:
        raise KeyError(f"unknown cell tensor {logical_name!r}")
    return local.detach().to(device="cpu", dtype=output_dtype).contiguous()


def _manifest(
    shape: _LlamaShape,
    layer_count: int,
    world_size: int,
    shard_sha256: Sequence[str],
    rank_fixed_bytes: Sequence[int],
    rank_kv_bytes_per_token: Sequence[int],
    output_dtype: str,
    rank_weights: Sequence[float] | None,
) -> dict[str, Any]:
    if any(
        len(values) != world_size
        for values in (shard_sha256, rank_fixed_bytes, rank_kv_bytes_per_token)
    ):
        raise ValueError("one completed shard profile is required per cell member")
    value = {
        "schema": CELL_FIXTURE_SCHEMA,
        "worldSize": world_size,
        "layerCount": layer_count,
        "dtype": output_dtype,
        "layers": [
            {
                "index": index,
                "hiddenSize": shape.hidden_size,
                "intermediateSize": shape.intermediate_size,
                "numAttentionHeads": shape.num_attention_heads,
                "numKeyValueHeads": shape.num_key_value_heads,
                "headDim": shape.head_dim,
                "rmsNormEpsilon": shape.rms_norm_epsilon,
                "ropeTheta": shape.rope_theta,
            }
            for index in range(layer_count)
        ],
        "shards": [f"rank-{rank:03d}.safetensors" for rank in range(world_size)],
        "shardSha256": list(shard_sha256),
        "rankFixedBytes": list(rank_fixed_bytes),
        "rankKvBytesPerToken": list(rank_kv_bytes_per_token),
    }
    if rank_weights is not None:
        value["rankWeights"] = list(rank_weights)
    return value


def _rank_kv_bytes_per_token(
    shape: _LlamaShape,
    layer_count: int,
    rank: int,
    world_size: int,
    element_bytes: int,
    rank_weights: Sequence[float] | None,
) -> int:
    plan = llama_attention_shard_plan(
        shape.hidden_size,
        shape.num_attention_heads,
        shape.num_key_value_heads,
        shape.head_dim,
        rank,
        world_size,
        rank_weights,
    )
    return 2 * plan.key_value_heads.size * shape.head_dim * element_bytes * layer_count


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_integer(config: Any, name: str) -> int:
    value = _optional_positive_integer(config, name)
    if value is None:
        raise ValueError(f"model config has no positive integer {name!r}")
    return value


def _optional_positive_integer(config: Any, name: str) -> int | None:
    value = getattr(config, name, None)
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def _positive_number(config: Any, name: str) -> float:
    value = _optional_positive_number(config, name)
    if value is None:
        raise ValueError(f"model config has no finite positive number {name!r}")
    return value


def _optional_positive_number(config: Any, name: str) -> float | None:
    value = getattr(config, name, None)
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and value > 0
    ):
        return float(value)
    return None


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile HF Llama SafeTensors into a GDLP /2 TP cell fixture."
    )
    parser.add_argument("model_name")
    parser.add_argument("destination")
    parser.add_argument("--revision")
    parser.add_argument("--layer-start", type=int, required=True)
    parser.add_argument("--layer-end", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument(
        "--rank-weight",
        action="append",
        type=float,
        default=None,
        help=(
            "Relative capacity in rank order; repeat exactly world-size times. "
            "Omit for the backward-compatible equal partition."
        ),
    )
    parser.add_argument(
        "--output-dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = compile_hf_llama_cell_fixture(
        args.model_name,
        args.destination,
        revision=args.revision,
        layer_start=args.layer_start,
        layer_end=args.layer_end,
        world_size=args.world_size,
        output_dtype=args.output_dtype,
        rank_weights=args.rank_weight,
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CELL_FIXTURE_SCHEMA",
    "CELL_FIXTURE_SOURCE_SCHEMA",
    "CellFixtureCompilation",
    "compile_hf_llama_cell_fixture",
    "main",
]
