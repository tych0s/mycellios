from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


STAGE_EXECUTOR_SCHEMA = "gdlp-stage-executor/1"
STAGE_KV_FORK_REPORT_SCHEMA = "gdlp-stage-kv-fork-report/1"
STAGE_TENSOR_LAYOUT = "batch-token-hidden-row-major"
STAGE_ENDIANNESS = "little"
_PHASES = ("prefill", "decode", "verify")
_OPERATIONS = (
    "load",
    "begin",
    "prefill",
    "decode",
    "verify",
    "truncate",
    "end",
    "health",
    "unload",
)
_DTYPES = ("float32", "float16", "bfloat16")
_CODECS = (
    "fp32",
    "fp16",
    "int8",
    "int8-grouped",
    "int8-hadamard",
    "int8-grouped-deflate",
    "int8-hadamard-deflate",
)


@dataclass(frozen=True, slots=True)
class StageKVForkReport:
    """Sealed physical accounting captured immediately after one KV fork.

    ``logical_bytes`` is the sum of logical KV represented by every active
    request after the fork. ``unique_physical_bytes`` deduplicates shared
    storage across those requests. The remaining fields are operation deltas;
    workspace is optional because several backends cannot observe allocator
    peaks without perturbing global device statistics.
    """

    logical_bytes: int
    unique_physical_bytes: int
    copied_bytes: int
    newly_reserved_bytes: int
    peak_workspace_bytes: int | None

    def __post_init__(self) -> None:
        for name, value in (
            ("logical_bytes", self.logical_bytes),
            ("unique_physical_bytes", self.unique_physical_bytes),
            ("copied_bytes", self.copied_bytes),
            ("newly_reserved_bytes", self.newly_reserved_bytes),
        ):
            _nonnegative_integer(value, name)
        if self.peak_workspace_bytes is not None:
            _nonnegative_integer(
                self.peak_workspace_bytes,
                "peak_workspace_bytes",
            )

    def to_document(self) -> dict[str, int | str | None]:
        return {
            "schema": STAGE_KV_FORK_REPORT_SCHEMA,
            "logicalBytes": self.logical_bytes,
            "uniquePhysicalBytes": self.unique_physical_bytes,
            "copiedBytes": self.copied_bytes,
            "newlyReservedBytes": self.newly_reserved_bytes,
            "peakWorkspaceBytes": self.peak_workspace_bytes,
        }


@runtime_checkable
class StageKVPhysicalAccounting(Protocol):
    """Optional COW-aware extension to the legacy stage runner contract."""

    def last_fork_report(self) -> StageKVForkReport | None: ...

    def unique_physical_cache_bytes(self, request_ids: Sequence[int]) -> int: ...

    def project_incremental_physical_cache_bytes(
        self,
        parent_request_id: int,
        *,
        new_leaf_count: int,
        delta_tokens: int,
    ) -> int: ...

    def project_tree_incremental_physical_cache_bytes(
        self,
        parent_request_id: int,
        *,
        delta_tokens_by_leaf: Sequence[int],
    ) -> int: ...

    def project_request_incremental_physical_cache_bytes(
        self,
        request_id: int,
        additional_tokens: int,
    ) -> int: ...


@dataclass(frozen=True)
class StageExecutorManifest:
    """Language-neutral contract for one executable contiguous model stage.

    This document deliberately describes semantics, not Python objects or GPU
    pointers. Future Mycellios executors can implement the same operations and
    tensor layout without importing this module. The executor id seals every
    compatibility-relevant field.
    """

    executor_id: str
    engine: str
    engine_version: str | None
    adapter: str
    model_identity: str
    model_source: str
    model_revision: str | None
    artifact_format: str
    layer_start: int
    layer_end: int
    total_layers: int
    hidden_size: int
    activation_dtype: str
    activation_codecs: tuple[str, ...]
    kv_format: str
    supports_truncate: bool
    phases: tuple[str, ...]
    operations: tuple[str, ...]
    max_batch_size: int
    max_context_tokens: int | None
    device_kinds: tuple[str, ...]
    compute_apis: tuple[str, ...]
    weight_dtypes: tuple[str, ...]
    features: tuple[str, ...]

    @property
    def first(self) -> bool:
        return self.layer_start == 0

    @property
    def last(self) -> bool:
        return self.layer_end == self.total_layers

    def to_document(self) -> dict[str, Any]:
        return {
            "schema": STAGE_EXECUTOR_SCHEMA,
            "executorId": self.executor_id,
            "engine": {
                "id": self.engine,
                "version": self.engine_version,
                "adapter": self.adapter,
            },
            "model": {
                "identity": self.model_identity,
                "source": self.model_source,
                "revision": self.model_revision,
                "artifactFormat": self.artifact_format,
            },
            "stage": {
                "layerStart": self.layer_start,
                "layerEnd": self.layer_end,
                "totalLayers": self.total_layers,
                "first": self.first,
                "last": self.last,
            },
            "tensor": {
                "layout": STAGE_TENSOR_LAYOUT,
                "dtype": self.activation_dtype,
                "hiddenSize": self.hidden_size,
                "endianness": STAGE_ENDIANNESS,
                "activationCodecs": list(self.activation_codecs),
            },
            "kv": {
                "ownership": "stage-local",
                "format": self.kv_format,
                "supportsTruncate": self.supports_truncate,
            },
            "execution": {
                "phases": list(self.phases),
                "operations": list(self.operations),
                "maxBatchSize": self.max_batch_size,
                "maxContextTokens": self.max_context_tokens,
                "deviceKinds": list(self.device_kinds),
                "computeApis": list(self.compute_apis),
                "weightDtypes": list(self.weight_dtypes),
                "features": list(self.features),
            },
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_document())).hexdigest()


def build_stage_executor_manifest(
    *,
    engine: str,
    engine_version: str | None,
    adapter: str,
    model_identity: str,
    model_source: str,
    model_revision: str | None,
    artifact_format: str,
    layer_start: int,
    layer_end: int,
    total_layers: int,
    hidden_size: int,
    activation_dtype: str,
    activation_codecs: Sequence[str],
    kv_format: str = "backend-native",
    supports_truncate: bool = True,
    phases: Sequence[str] = _PHASES,
    operations: Sequence[str] = _OPERATIONS,
    max_batch_size: int = 1,
    max_context_tokens: int | None = None,
    device_kinds: Sequence[str] = ("cpu",),
    compute_apis: Sequence[str] = ("torch",),
    weight_dtypes: Sequence[str] = ("float32",),
    features: Sequence[str] = ("layer-range", "rank-local-kv", "rollback"),
) -> StageExecutorManifest:
    without_identity = {
        "schema": STAGE_EXECUTOR_SCHEMA,
        "engine": {
            "id": engine,
            "version": engine_version,
            "adapter": adapter,
        },
        "model": {
            "identity": model_identity,
            "source": model_source,
            "revision": model_revision,
            "artifactFormat": artifact_format,
        },
        "stage": {
            "layerStart": layer_start,
            "layerEnd": layer_end,
            "totalLayers": total_layers,
            "first": layer_start == 0,
            "last": layer_end == total_layers,
        },
        "tensor": {
            "layout": STAGE_TENSOR_LAYOUT,
            "dtype": activation_dtype,
            "hiddenSize": hidden_size,
            "endianness": STAGE_ENDIANNESS,
            "activationCodecs": list(activation_codecs),
        },
        "kv": {
            "ownership": "stage-local",
            "format": kv_format,
            "supportsTruncate": supports_truncate,
        },
        "execution": {
            "phases": list(phases),
            "operations": list(operations),
            "maxBatchSize": max_batch_size,
            "maxContextTokens": max_context_tokens,
            "deviceKinds": list(device_kinds),
            "computeApis": list(compute_apis),
            "weightDtypes": list(weight_dtypes),
            "features": list(features),
        },
    }
    executor_id = hashlib.sha256(_canonical_json(without_identity)).hexdigest()[:32]
    return parse_stage_executor_manifest(
        {**without_identity, "executorId": executor_id}
    )


def model_identity_for_source(source: str, revision: str | None) -> str:
    """Stable cross-backend identity when a stronger artifact digest is absent."""

    _string(source, "model source")
    _nullable_string(revision, "model revision")
    commit = _hub_commit_from_coordinates(source, revision)
    if commit is not None:
        return "sha256:" + hashlib.sha256(
            b"gdlp-hub-snapshot-v1\0" + commit.encode("ascii")
        ).hexdigest()
    return "sha256:" + hashlib.sha256(
        _canonical_json({"source": source, "revision": revision})
    ).hexdigest()


def _hub_commit_from_coordinates(source: str, revision: str | None) -> str | None:
    candidates = [revision.strip().lower()] if revision is not None else []
    parts = [part for part in source.replace("\\", "/").split("/") if part]
    for index, part in enumerate(parts[:-1]):
        if part.lower() == "snapshots":
            candidates.append(parts[index + 1].lower())
    for candidate in candidates:
        if len(candidate) >= 32 and all(
            character in "0123456789abcdef" for character in candidate
        ):
            return candidate
    return None


def parse_stage_kv_fork_report(value: object) -> StageKVForkReport:
    """Parse the exact language-neutral COW accounting document."""

    document = _exact_mapping(
        value,
        (
            "schema",
            "logicalBytes",
            "uniquePhysicalBytes",
            "copiedBytes",
            "newlyReservedBytes",
            "peakWorkspaceBytes",
        ),
        "stage KV fork report",
    )
    if document.get("schema") != STAGE_KV_FORK_REPORT_SCHEMA:
        raise ValueError("unsupported stage KV fork report schema")
    workspace = document.get("peakWorkspaceBytes")
    if workspace is not None:
        workspace = _nonnegative_integer(workspace, "peakWorkspaceBytes")
    return StageKVForkReport(
        logical_bytes=_nonnegative_integer(
            document.get("logicalBytes"), "logicalBytes"
        ),
        unique_physical_bytes=_nonnegative_integer(
            document.get("uniquePhysicalBytes"), "uniquePhysicalBytes"
        ),
        copied_bytes=_nonnegative_integer(
            document.get("copiedBytes"), "copiedBytes"
        ),
        newly_reserved_bytes=_nonnegative_integer(
            document.get("newlyReservedBytes"), "newlyReservedBytes"
        ),
        peak_workspace_bytes=workspace,
    )


def parse_stage_executor_manifest(value: object) -> StageExecutorManifest:
    document = _mapping(value, "stage executor manifest")
    if set(document) != {
        "schema",
        "executorId",
        "engine",
        "model",
        "stage",
        "tensor",
        "kv",
        "execution",
    }:
        raise ValueError("stage executor manifest has unknown or missing fields")
    if document.get("schema") != STAGE_EXECUTOR_SCHEMA:
        raise ValueError("unsupported stage executor schema")
    engine = _exact_mapping(
        document.get("engine"), ("id", "version", "adapter"), "engine"
    )
    model = _exact_mapping(
        document.get("model"),
        ("identity", "source", "revision", "artifactFormat"),
        "model",
    )
    stage = _exact_mapping(
        document.get("stage"),
        ("layerStart", "layerEnd", "totalLayers", "first", "last"),
        "stage",
    )
    tensor = _exact_mapping(
        document.get("tensor"),
        ("layout", "dtype", "hiddenSize", "endianness", "activationCodecs"),
        "tensor",
    )
    kv = _exact_mapping(
        document.get("kv"),
        ("ownership", "format", "supportsTruncate"),
        "kv",
    )
    execution = _exact_mapping(
        document.get("execution"),
        (
            "phases",
            "operations",
            "maxBatchSize",
            "maxContextTokens",
            "deviceKinds",
            "computeApis",
            "weightDtypes",
            "features",
        ),
        "execution",
    )
    layer_start = _nonnegative_integer(stage.get("layerStart"), "layerStart")
    layer_end = _positive_integer(stage.get("layerEnd"), "layerEnd")
    total_layers = _positive_integer(stage.get("totalLayers"), "totalLayers")
    if not layer_start < layer_end <= total_layers:
        raise ValueError("stage executor layer range is invalid")
    if stage.get("first") is not (layer_start == 0):
        raise ValueError("stage executor first flag does not match its range")
    if stage.get("last") is not (layer_end == total_layers):
        raise ValueError("stage executor last flag does not match its range")
    activation_dtype = _string(tensor.get("dtype"), "tensor dtype")
    if activation_dtype not in _DTYPES:
        raise ValueError("stage executor tensor dtype is unsupported")
    if tensor.get("layout") != STAGE_TENSOR_LAYOUT:
        raise ValueError("stage executor tensor layout is unsupported")
    if tensor.get("endianness") != STAGE_ENDIANNESS:
        raise ValueError("stage executor tensor endianness is unsupported")
    activation_codecs = _string_tuple(
        tensor.get("activationCodecs"), "activation codecs"
    )
    if any(codec not in _CODECS for codec in activation_codecs):
        raise ValueError("stage executor activation codec is unsupported")
    phases = _string_tuple(execution.get("phases"), "execution phases")
    if any(phase not in _PHASES for phase in phases) or not {
        "prefill",
        "decode",
    }.issubset(phases):
        raise ValueError("stage executor phases are incompatible with GDLP")
    operations = _string_tuple(execution.get("operations"), "executor operations")
    if any(operation not in _OPERATIONS for operation in operations):
        raise ValueError("stage executor operation is unsupported")
    required_operations = {
        "load",
        "begin",
        "prefill",
        "decode",
        "truncate",
        "end",
        "health",
        "unload",
    }
    if not required_operations.issubset(operations):
        raise ValueError("stage executor omits required operations")
    supports_truncate = kv.get("supportsTruncate")
    if supports_truncate is not True or kv.get("ownership") != "stage-local":
        raise ValueError("stage executor must own truncatable local KV")
    max_context = execution.get("maxContextTokens")
    if max_context is not None:
        max_context = _positive_integer(max_context, "maxContextTokens")
    manifest = StageExecutorManifest(
        executor_id=_hex_string(document.get("executorId"), 32, "executorId"),
        engine=_string(engine.get("id"), "engine id"),
        engine_version=_nullable_string(engine.get("version"), "engine version"),
        adapter=_string(engine.get("adapter"), "engine adapter"),
        model_identity=_string(model.get("identity"), "model identity"),
        model_source=_string(model.get("source"), "model source"),
        model_revision=_nullable_string(model.get("revision"), "model revision"),
        artifact_format=_string(model.get("artifactFormat"), "artifact format"),
        layer_start=layer_start,
        layer_end=layer_end,
        total_layers=total_layers,
        hidden_size=_positive_integer(tensor.get("hiddenSize"), "hiddenSize"),
        activation_dtype=activation_dtype,
        activation_codecs=activation_codecs,
        kv_format=_string(kv.get("format"), "KV format"),
        supports_truncate=supports_truncate,
        phases=phases,
        operations=operations,
        max_batch_size=_positive_integer(execution.get("maxBatchSize"), "maxBatchSize"),
        max_context_tokens=max_context,
        device_kinds=_string_tuple(execution.get("deviceKinds"), "device kinds"),
        compute_apis=_string_tuple(execution.get("computeApis"), "compute APIs"),
        weight_dtypes=_string_tuple(execution.get("weightDtypes"), "weight dtypes"),
        features=_string_tuple(execution.get("features"), "features"),
    )
    expected_id = _manifest_identity(manifest.to_document())
    if manifest.executor_id != expected_id:
        raise ValueError("stage executor identity does not match its contract")
    return manifest


def validate_executor_chain(
    manifests: Sequence[StageExecutorManifest | Mapping[str, Any]],
) -> tuple[StageExecutorManifest, ...]:
    """Validate model, tensor and layer continuity for a complete route."""

    if isinstance(manifests, (str, bytes, bytearray)) or len(manifests) < 1:
        raise ValueError("executor chain cannot be empty")
    parsed = tuple(
        value
        if isinstance(value, StageExecutorManifest)
        else parse_stage_executor_manifest(value)
        for value in manifests
    )
    first = parsed[0]
    if not first.first or not parsed[-1].last:
        raise ValueError("executor chain must cover model endpoints")
    for left, right in zip(parsed, parsed[1:]):
        if left.layer_end != right.layer_start:
            raise ValueError("executor chain layer ranges are not contiguous")
        if (
            left.model_identity != right.model_identity
            or left.total_layers != right.total_layers
        ):
            raise ValueError("executor chain model identity does not match")
        if (
            left.hidden_size != right.hidden_size
            or left.activation_dtype != right.activation_dtype
        ):
            raise ValueError("executor chain tensor contract does not match")
        if not set(left.activation_codecs).intersection(right.activation_codecs):
            raise ValueError("executor chain has no common activation codec")
    if any(value.layer_end != value.total_layers for value in parsed[-1:]):
        raise ValueError("executor chain does not cover every model layer")
    return parsed


def _manifest_identity(document: Mapping[str, Any]) -> str:
    body = dict(document)
    body.pop("executorId", None)
    return hashlib.sha256(_canonical_json(body)).hexdigest()[:32]


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _exact_mapping(
    value: object,
    fields: Sequence[str],
    name: str,
) -> Mapping[str, Any]:
    result = _mapping(value, name)
    if set(result) != set(fields):
        raise ValueError(f"stage executor {name} has unknown or missing fields")
    return result


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _nullable_string(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _string(value, name)


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or len(value) < 1
    ):
        raise ValueError(f"{name} must be a non-empty string list")
    result = tuple(_string(item, name) for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} cannot contain duplicates")
    return result


def _positive_integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _hex_string(value: object, size: int, name: str) -> str:
    text = _string(value, name)
    if len(text) != size or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be lowercase hexadecimal")
    return text


__all__ = [
    "STAGE_EXECUTOR_SCHEMA",
    "STAGE_ENDIANNESS",
    "STAGE_KV_FORK_REPORT_SCHEMA",
    "STAGE_TENSOR_LAYOUT",
    "StageExecutorManifest",
    "StageKVForkReport",
    "StageKVPhysicalAccounting",
    "build_stage_executor_manifest",
    "model_identity_for_source",
    "parse_stage_executor_manifest",
    "parse_stage_kv_fork_report",
    "validate_executor_chain",
]
