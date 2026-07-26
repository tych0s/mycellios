from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import struct
import sys
import tempfile
from typing import Any, BinaryIO, Mapping, Sequence

from safetensors import safe_open
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from .executor_abi import STAGE_ENDIANNESS, STAGE_TENSOR_LAYOUT
from .model import (
    StageModelSpec,
    _checkpoint_key_map,
    _checkpoint_name,
    _hub_snapshot_commit,
    _validate_checkpoint_coverage,
    model_artifact_reference,
    resolve_stage_model_snapshot,
)
from .model_adapters import (
    ADAPTER_REGISTRY_ID,
    SelectiveStageAdapter,
    canonical_adapter_registry_bytes,
    resolve_selective_stage_adapter,
    validate_adapter_registry_document,
)


STAGE_ARTIFACT_SCHEMA = "mycellios-safetensors-stage-package/2"
STAGE_ARTIFACT_FORMAT = "mycellios-stage-safetensors"
STAGE_ARTIFACT_MANIFEST = "mycellios-stage.json"
STAGE_ARTIFACT_CONFIG = "config.json"
STAGE_ARTIFACT_WEIGHTS = "stage.safetensors"
STAGE_ARTIFACT_ADAPTER_REGISTRY = "model-adapter-registry.json"
STAGE_TENSOR_ABI = "mycellios-transformers-global-stage-tensors/1"

_COPY_BUFFER_BYTES = 8 * 1024 * 1024
_MAX_SAFETENSORS_HEADER_BYTES = 256 * 1024 * 1024
_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
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
    "C64": 8,
    "C128": 16,
}
_TENSOR_ROLES = {
    "input-embedding",
    "layer-weights",
    "final-normalization",
    "output-head",
    "shared-model-state",
}


@dataclass(frozen=True, slots=True)
class StageArtifactCompilation:
    destination: str
    schema: str
    package_id: str
    artifact_identity: str
    manifest_sha256: str
    model_identity: str
    family: str
    adapter: str
    layer_start: int
    layer_end: int
    total_layers: int
    tensor_count: int
    tensor_bytes: int
    weights_sha256: str
    weights_size_bytes: int


@dataclass(frozen=True, slots=True)
class SafeTensorsStageArtifact:
    """Strict, canonical view of a sealed Mycellios stage manifest."""

    document_json: str

    @property
    def package_id(self) -> str:
        return str(self.to_document()["packageId"])

    @property
    def model_identity(self) -> str:
        return str(self.to_document()["model"]["identity"])

    @property
    def family(self) -> str:
        return str(self.to_document()["model"]["family"])

    @property
    def adapter(self) -> str:
        return str(self.to_document()["model"]["adapter"])

    @property
    def layer_start(self) -> int:
        return int(self.to_document()["stage"]["layerStart"])

    @property
    def layer_end(self) -> int:
        return int(self.to_document()["stage"]["layerEnd"])

    @property
    def total_layers(self) -> int:
        return int(self.to_document()["stage"]["totalLayers"])

    def to_document(self) -> dict[str, Any]:
        value = json.loads(self.document_json)
        if not isinstance(value, dict):
            raise TypeError("stage artifact document is not an object")
        return value


@dataclass(frozen=True, slots=True)
class VerifiedStageArtifact:
    root: Path
    manifest: SafeTensorsStageArtifact
    config_path: Path
    weights_path: Path
    adapter_registry_path: Path


@dataclass(frozen=True, slots=True)
class _ExpectedTensor:
    name: str
    shape: tuple[int, ...]
    roles: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _SafeTensorLayout:
    name: str
    dtype: str
    shape: tuple[int, ...]
    data_start: int
    data_end: int

    @property
    def size_bytes(self) -> int:
        return self.data_end - self.data_start


@dataclass(frozen=True, slots=True)
class _SelectedSourceTensor:
    expected: _ExpectedTensor
    source_path: Path
    source_layout: _SafeTensorLayout


def compile_safetensors_stage_artifact(
    model_name: str,
    destination: str | os.PathLike[str],
    *,
    layer_start: int,
    layer_end: int,
    revision: str | None = None,
) -> StageArtifactCompilation:
    """Compile one exact contiguous model stage without materializing its weights.

    A small architecture skeleton is built on the PyTorch ``meta`` device to
    derive the same family-specific tensor contract used by the runtime. Source
    SafeTensors headers are then inspected and the selected byte ranges are
    copied directly into one new SafeTensors file. No source tensor, unrelated
    layer, or complete model is ever materialized by this compiler.
    """

    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name cannot be blank")
    for name, value in (("layer_start", layer_start), ("layer_end", layer_end)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer")
    if revision is not None and (not isinstance(revision, str) or not revision.strip()):
        raise ValueError("revision cannot be blank")
    if sys.byteorder != STAGE_ENDIANNESS:
        raise RuntimeError(
            f"stage artifact compiler requires a {STAGE_ENDIANNESS}-endian host"
        )

    source_config = AutoConfig.from_pretrained(model_name, revision=revision)
    total_layers = _positive_integer(
        getattr(source_config, "num_hidden_layers", None), "num_hidden_layers"
    )
    if not 0 <= layer_start < layer_end <= total_layers:
        raise ValueError(
            f"layer range [{layer_start}, {layer_end}) is outside [0, {total_layers})"
        )
    snapshot = Path(
        resolve_stage_model_snapshot(
            model_name,
            revision=revision,
            layer_start=layer_start,
            layer_end=layer_end,
            total_layers=total_layers,
        )
    ).resolve()
    config_path = snapshot / STAGE_ARTIFACT_CONFIG
    if not config_path.is_file():
        raise FileNotFoundError(f"model checkpoint has no config.json: {snapshot}")
    config_bytes = config_path.read_bytes()
    config_document = _json_object(config_bytes, "model config")
    config = AutoConfig.from_pretrained(str(snapshot), local_files_only=True)
    resolved_total_layers = _positive_integer(
        getattr(config, "num_hidden_layers", None), "num_hidden_layers"
    )
    if resolved_total_layers != total_layers:
        raise ValueError("resolved stage snapshot changed its total layer count")

    key_map = _checkpoint_key_map(snapshot)
    _validate_checkpoint_namespace(key_map, total_layers)
    adapter, expected = _expected_stage_tensors(
        config,
        key_map,
        layer_start=layer_start,
        layer_end=layer_end,
        total_layers=total_layers,
    )
    selected = _select_source_tensors(snapshot, key_map, expected)
    source_dtypes = tuple(sorted({value.source_layout.dtype for value in selected}))
    canonical_config = _canonical_json(config_document) + b"\n"
    config_sha256 = hashlib.sha256(canonical_config).hexdigest()
    source_contract_sha256 = hashlib.sha256(
        _canonical_json(
            {
                "configSha256": config_sha256,
                "weightMap": {name: key_map[name] for name in sorted(key_map)},
            }
        )
    ).hexdigest()
    stable_revision = _hub_snapshot_commit(snapshot) or revision
    # This identity names the complete immutable source model and therefore
    # must be identical to the profile compiled by the coordinator. The stage
    # package receives its own independent package_id below.
    model_identity = model_artifact_reference(str(snapshot)).identity
    quantization = _quantization_document(config_document, source_dtypes)

    destination_path = Path(destination).resolve()
    if destination_path.exists():
        raise FileExistsError(
            f"stage artifact destination already exists: {destination_path}"
        )
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination_path.name}.compile-",
            dir=destination_path.parent,
        )
    )
    try:
        _write_bytes_atomic_payload(temporary / STAGE_ARTIFACT_CONFIG, canonical_config)
        _write_bytes_atomic_payload(
            temporary / STAGE_ARTIFACT_ADAPTER_REGISTRY,
            canonical_adapter_registry_bytes(),
        )
        weights_path = temporary / STAGE_ARTIFACT_WEIGHTS
        _write_stage_safetensors(weights_path, selected)
        weights_size = weights_path.stat().st_size
        weights_sha256 = _sha256_file(weights_path)
        config_size = (temporary / STAGE_ARTIFACT_CONFIG).stat().st_size
        files = [
            {
                "path": STAGE_ARTIFACT_CONFIG,
                "role": "model-config",
                "sizeBytes": config_size,
                "sha256": _sha256_file(temporary / STAGE_ARTIFACT_CONFIG),
            },
            {
                "path": STAGE_ARTIFACT_ADAPTER_REGISTRY,
                "role": "model-adapter-registry",
                "sizeBytes": (
                    temporary / STAGE_ARTIFACT_ADAPTER_REGISTRY
                ).stat().st_size,
                "sha256": _sha256_file(
                    temporary / STAGE_ARTIFACT_ADAPTER_REGISTRY
                ),
            },
            {
                "path": STAGE_ARTIFACT_WEIGHTS,
                "role": "stage-weights",
                "sizeBytes": weights_size,
                "sha256": weights_sha256,
            },
        ]
        tensor_documents = [
            {
                "name": item.expected.name,
                "dtype": item.source_layout.dtype,
                "shape": list(item.source_layout.shape),
                "sizeBytes": item.source_layout.size_bytes,
                "roles": list(item.expected.roles),
            }
            for item in selected
        ]
        tensor_bytes = sum(item["sizeBytes"] for item in tensor_documents)
        architecture = _single_architecture(config)
        hidden_size = _positive_integer(
            getattr(config, "hidden_size", None), "hidden_size"
        )
        body = {
            "schema": STAGE_ARTIFACT_SCHEMA,
            "packageKind": "executable-contiguous-layer-range",
            "artifactFormat": STAGE_ARTIFACT_FORMAT,
            "model": {
                "identity": model_identity,
                "source": model_name,
                "revision": stable_revision,
                "snapshotCommit": _hub_snapshot_commit(snapshot),
                "sourceContractSha256": source_contract_sha256,
                "configSha256": config_sha256,
                "family": adapter.model_type,
                "architecture": architecture,
                "adapter": adapter.adapter_id,
                "adapterContractId": adapter.adapter_contract_id,
                "adapterRegistryId": ADAPTER_REGISTRY_ID,
                "quantization": quantization,
            },
            "stage": {
                "layerStart": layer_start,
                "layerEnd": layer_end,
                "totalLayers": total_layers,
                "first": layer_start == 0,
                "last": layer_end == total_layers,
            },
            "tensorAbi": {
                "id": STAGE_TENSOR_ABI,
                "format": "safetensors",
                "naming": "huggingface-global-layer-names",
                "endianness": STAGE_ENDIANNESS,
                "activationLayout": STAGE_TENSOR_LAYOUT,
                "hiddenSize": hidden_size,
                "weightDtypes": list(source_dtypes),
            },
            "artifact": {
                "fileCount": len(files),
                "payloadSizeBytes": sum(item["sizeBytes"] for item in files),
                "tensorCount": len(tensor_documents),
                "tensorBytes": tensor_bytes,
            },
            "files": files,
            "tensors": tensor_documents,
        }
        package_id = hashlib.sha256(_canonical_json(body)).hexdigest()
        manifest_document = {**body, "packageId": package_id}
        manifest_path = temporary / STAGE_ARTIFACT_MANIFEST
        _write_bytes_atomic_payload(
            manifest_path,
            json.dumps(
                manifest_document,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ).encode("utf-8")
            + b"\n",
        )

        verified = verify_stage_artifact(
            temporary,
            expected_layer_start=layer_start,
            expected_layer_end=layer_end,
            expected_total_layers=total_layers,
        )
        if verified.manifest.package_id != package_id:
            raise RuntimeError("published package identity changed during verification")
        manifest_sha256 = _sha256_file(manifest_path)
        temporary.replace(destination_path)
        return StageArtifactCompilation(
            destination=str(destination_path),
            schema=STAGE_ARTIFACT_SCHEMA,
            package_id=package_id,
            artifact_identity=f"sha256:{package_id}",
            manifest_sha256=manifest_sha256,
            model_identity=model_identity,
            family=adapter.model_type,
            adapter=adapter.adapter_id,
            layer_start=layer_start,
            layer_end=layer_end,
            total_layers=total_layers,
            tensor_count=len(tensor_documents),
            tensor_bytes=tensor_bytes,
            weights_sha256=weights_sha256,
            weights_size_bytes=weights_size,
        )
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def prepare_safetensors_stage_artifact(
    model_name: str,
    destination: str | os.PathLike[str],
    *,
    layer_start: int,
    layer_end: int,
    revision: str | None = None,
) -> StageArtifactCompilation:
    """Reuse one authenticated stage package or compile it atomically.

    Desktop contributors call this entrypoint during ``runtime.prepare``.  A
    cache hit is never trusted from its directory name: the complete manifest,
    file digests, tensor contract and exact layer range are revalidated before
    the package is exposed to a model process.
    """

    destination_path = Path(destination).resolve()
    if not destination_path.exists():
        return compile_safetensors_stage_artifact(
            model_name,
            destination_path,
            layer_start=layer_start,
            layer_end=layer_end,
            revision=revision,
        )
    verified = verify_stage_artifact(
        destination_path,
        expected_layer_start=layer_start,
        expected_layer_end=layer_end,
    )
    document = verified.manifest.to_document()
    model = _exact_mapping(
        document.get("model"),
        {
            "identity",
            "source",
            "revision",
            "snapshotCommit",
            "sourceContractSha256",
            "configSha256",
            "family",
            "architecture",
            "adapter",
            "adapterContractId",
            "adapterRegistryId",
            "quantization",
        },
        "stage artifact model",
    )
    if model.get("source") != model_name:
        raise ValueError("cached stage artifact belongs to a different model source")
    artifact = _exact_mapping(
        document.get("artifact"),
        {"fileCount", "payloadSizeBytes", "tensorCount", "tensorBytes"},
        "stage artifact accounting",
    )
    files = {
        str(item["role"]): item
        for item in _sequence(document.get("files"), "stage artifact files")
        if isinstance(item, dict)
    }
    weights = files.get("stage-weights")
    if weights is None:
        raise ValueError("cached stage artifact has no stage weights")
    return StageArtifactCompilation(
        destination=str(destination_path),
        schema=STAGE_ARTIFACT_SCHEMA,
        package_id=verified.manifest.package_id,
        artifact_identity=f"sha256:{verified.manifest.package_id}",
        manifest_sha256=_sha256_file(destination_path / STAGE_ARTIFACT_MANIFEST),
        model_identity=verified.manifest.model_identity,
        family=verified.manifest.family,
        adapter=verified.manifest.adapter,
        layer_start=verified.manifest.layer_start,
        layer_end=verified.manifest.layer_end,
        total_layers=verified.manifest.total_layers,
        tensor_count=_nonnegative_integer(
            artifact.get("tensorCount"), "artifact tensor count"
        ),
        tensor_bytes=_nonnegative_integer(
            artifact.get("tensorBytes"), "artifact tensor bytes"
        ),
        weights_sha256=_sha256(weights.get("sha256"), "stage weights digest"),
        weights_size_bytes=_nonnegative_integer(
            weights.get("sizeBytes"), "stage weights size"
        ),
    )


def parse_stage_artifact_manifest(value: object) -> SafeTensorsStageArtifact:
    document = _exact_mapping(
        value,
        (
            "schema",
            "packageId",
            "packageKind",
            "artifactFormat",
            "model",
            "stage",
            "tensorAbi",
            "artifact",
            "files",
            "tensors",
        ),
        "stage artifact manifest",
    )
    if document.get("schema") != STAGE_ARTIFACT_SCHEMA:
        raise ValueError("unsupported stage artifact schema")
    if document.get("packageKind") != "executable-contiguous-layer-range":
        raise ValueError("stage artifact is not an executable contiguous range")
    if document.get("artifactFormat") != STAGE_ARTIFACT_FORMAT:
        raise ValueError("unsupported stage artifact format")
    package_id = _sha256(document.get("packageId"), "packageId")

    model = _exact_mapping(
        document.get("model"),
        (
            "identity",
            "source",
            "revision",
            "snapshotCommit",
            "sourceContractSha256",
            "configSha256",
            "family",
            "architecture",
            "adapter",
            "adapterContractId",
            "adapterRegistryId",
            "quantization",
        ),
        "stage artifact model",
    )
    identity = _string(model.get("identity"), "model identity")
    if not identity.startswith("sha256:") or len(identity) != 71:
        raise ValueError("model identity must be a sha256: artifact identity")
    _sha256(identity.removeprefix("sha256:"), "model identity")
    _string(model.get("source"), "model source")
    _nullable_string(model.get("revision"), "model revision")
    snapshot_commit = _nullable_string(model.get("snapshotCommit"), "snapshot commit")
    if snapshot_commit is not None:
        _hex(snapshot_commit, "snapshot commit", minimum_length=32)
    _sha256(model.get("sourceContractSha256"), "source contract SHA-256")
    _sha256(model.get("configSha256"), "config SHA-256")
    family = _string(model.get("family"), "model family")
    architecture = _string(model.get("architecture"), "model architecture")
    adapter_id = _string(model.get("adapter"), "model adapter")
    adapter_contract_id = _string(
        model.get("adapterContractId"), "model adapter contract identity"
    )
    if (
        not adapter_contract_id.startswith("sha256:")
        or len(adapter_contract_id) != 71
    ):
        raise ValueError("model adapter contract identity must be a sha256 identity")
    _sha256(
        adapter_contract_id.removeprefix("sha256:"),
        "model adapter contract identity",
    )
    adapter_registry_id = _string(
        model.get("adapterRegistryId"), "model adapter registry identity"
    )
    if (
        not adapter_registry_id.startswith("sha256:")
        or len(adapter_registry_id) != 71
    ):
        raise ValueError("model adapter registry identity must be a sha256 identity")
    _sha256(
        adapter_registry_id.removeprefix("sha256:"),
        "model adapter registry identity",
    )
    quantization = _parse_quantization(model.get("quantization"))

    stage = _exact_mapping(
        document.get("stage"),
        ("layerStart", "layerEnd", "totalLayers", "first", "last"),
        "stage artifact range",
    )
    layer_start = _nonnegative_integer(stage.get("layerStart"), "layerStart")
    layer_end = _positive_integer(stage.get("layerEnd"), "layerEnd")
    total_layers = _positive_integer(stage.get("totalLayers"), "totalLayers")
    if not layer_start < layer_end <= total_layers:
        raise ValueError("stage artifact layer range is invalid")
    if stage.get("first") is not (layer_start == 0):
        raise ValueError("stage artifact first flag does not match its range")
    if stage.get("last") is not (layer_end == total_layers):
        raise ValueError("stage artifact last flag does not match its range")

    tensor_abi = _exact_mapping(
        document.get("tensorAbi"),
        (
            "id",
            "format",
            "naming",
            "endianness",
            "activationLayout",
            "hiddenSize",
            "weightDtypes",
        ),
        "stage tensor ABI",
    )
    if tensor_abi.get("id") != STAGE_TENSOR_ABI:
        raise ValueError("unsupported stage tensor ABI")
    if tensor_abi.get("format") != "safetensors":
        raise ValueError("stage tensor ABI is not SafeTensors")
    if tensor_abi.get("naming") != "huggingface-global-layer-names":
        raise ValueError("stage tensor naming is unsupported")
    if tensor_abi.get("endianness") != STAGE_ENDIANNESS:
        raise ValueError("stage tensor endianness is unsupported")
    if tensor_abi.get("activationLayout") != STAGE_TENSOR_LAYOUT:
        raise ValueError("stage activation layout is unsupported")
    _positive_integer(tensor_abi.get("hiddenSize"), "hiddenSize")
    declared_dtypes = _string_sequence(tensor_abi.get("weightDtypes"), "weightDtypes")
    if tuple(sorted(set(declared_dtypes))) != declared_dtypes:
        raise ValueError("weightDtypes must be unique and sorted")

    files_value = _sequence(document.get("files"), "stage artifact files")
    files: list[dict[str, Any]] = []
    for item in files_value:
        parsed = _exact_mapping(
            item, ("path", "role", "sizeBytes", "sha256"), "artifact file"
        )
        path = _safe_relative_path(parsed.get("path"), "artifact file path")
        role = _string(parsed.get("role"), "artifact file role")
        size = _positive_integer(parsed.get("sizeBytes"), "artifact file size")
        digest = _sha256(parsed.get("sha256"), "artifact file SHA-256")
        files.append({"path": path, "role": role, "sizeBytes": size, "sha256": digest})
    expected_files = {
        STAGE_ARTIFACT_CONFIG: "model-config",
        STAGE_ARTIFACT_ADAPTER_REGISTRY: "model-adapter-registry",
        STAGE_ARTIFACT_WEIGHTS: "stage-weights",
    }
    if {item["path"]: item["role"] for item in files} != expected_files:
        raise ValueError(
            "stage artifact must contain exactly config, adapter registry and stage weights"
        )

    tensors_value = _sequence(document.get("tensors"), "stage tensors")
    tensors: list[dict[str, Any]] = []
    for item in tensors_value:
        parsed = _exact_mapping(
            item,
            ("name", "dtype", "shape", "sizeBytes", "roles"),
            "stage tensor",
        )
        name = _string(parsed.get("name"), "tensor name")
        dtype = _string(parsed.get("dtype"), "tensor dtype")
        element_bytes = _DTYPE_BYTES.get(dtype)
        if element_bytes is None:
            raise ValueError(f"unsupported SafeTensors dtype: {dtype}")
        shape = _shape(parsed.get("shape"), f"tensor {name} shape")
        size = _positive_integer(parsed.get("sizeBytes"), f"tensor {name} size")
        if math.prod(shape) * element_bytes != size:
            raise ValueError(f"tensor {name!r} size does not match dtype and shape")
        roles = _string_sequence(parsed.get("roles"), f"tensor {name} roles")
        if not roles or tuple(sorted(set(roles))) != roles:
            raise ValueError(f"tensor {name!r} roles must be unique and sorted")
        if any(role not in _TENSOR_ROLES for role in roles):
            raise ValueError(f"tensor {name!r} has an unsupported role")
        tensors.append(
            {
                "name": name,
                "dtype": dtype,
                "shape": list(shape),
                "sizeBytes": size,
                "roles": list(roles),
            }
        )
    names = [item["name"] for item in tensors]
    if names != sorted(set(names)):
        raise ValueError("stage tensor names must be unique and sorted")
    actual_dtypes = tuple(sorted({item["dtype"] for item in tensors}))
    if actual_dtypes != declared_dtypes:
        raise ValueError("stage tensor dtypes do not match the tensor ABI")
    if tuple(quantization["weightDtypes"]) != declared_dtypes:
        raise ValueError("quantization dtypes do not match the tensor ABI")

    artifact = _exact_mapping(
        document.get("artifact"),
        ("fileCount", "payloadSizeBytes", "tensorCount", "tensorBytes"),
        "stage artifact accounting",
    )
    if _positive_integer(artifact.get("fileCount"), "fileCount") != len(files):
        raise ValueError("stage artifact file count is incorrect")
    if _positive_integer(artifact.get("payloadSizeBytes"), "payloadSizeBytes") != sum(
        item["sizeBytes"] for item in files
    ):
        raise ValueError("stage artifact payload size is incorrect")
    if _positive_integer(artifact.get("tensorCount"), "tensorCount") != len(tensors):
        raise ValueError("stage artifact tensor count is incorrect")
    if _positive_integer(artifact.get("tensorBytes"), "tensorBytes") != sum(
        item["sizeBytes"] for item in tensors
    ):
        raise ValueError("stage artifact tensor bytes are incorrect")

    # These discriminators are checked here as well as during file verification
    # so a manifest for an unknown family cannot be accepted as an opaque blob.
    config_probe = type(
        "_ManifestFamilyProbe",
        (),
        {"model_type": family, "architectures": [architecture]},
    )()
    adapter = resolve_selective_stage_adapter(config_probe)
    if adapter.adapter_id != adapter_id:
        raise ValueError("stage adapter does not match its certified family")
    if (
        adapter.adapter_contract_id != adapter_contract_id
        or adapter_registry_id != ADAPTER_REGISTRY_ID
    ):
        raise ValueError("stage adapter registry contract is not implemented")

    body = {key: value for key, value in document.items() if key != "packageId"}
    expected_id = hashlib.sha256(_canonical_json(body)).hexdigest()
    if package_id != expected_id:
        raise ValueError("stage artifact package identity does not match its contract")
    return SafeTensorsStageArtifact(
        json.dumps(document, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    )


def verify_stage_artifact(
    directory: str | os.PathLike[str],
    *,
    expected_layer_start: int | None = None,
    expected_layer_end: int | None = None,
    expected_total_layers: int | None = None,
    expected_package_id: str | None = None,
) -> VerifiedStageArtifact:
    """Authenticate a stage package and re-derive its family tensor contract."""

    root = Path(directory).resolve()
    if not root.is_dir() or root.is_symlink():
        raise FileNotFoundError(f"stage artifact directory does not exist: {root}")
    manifest_path = root / STAGE_ARTIFACT_MANIFEST
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise FileNotFoundError(f"stage artifact manifest is missing: {manifest_path}")
    manifest_bytes = manifest_path.read_bytes()
    manifest_document = _json_object(manifest_bytes, "stage artifact manifest")
    manifest = parse_stage_artifact_manifest(manifest_document)
    canonical_manifest = (
        json.dumps(
            manifest.to_document(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    if manifest_bytes != canonical_manifest:
        raise ValueError("stage artifact manifest encoding is not canonical")
    if expected_package_id is not None and manifest.package_id != _sha256(
        expected_package_id, "expected package ID"
    ):
        raise ValueError("stage artifact package identity does not match the request")
    for expected, actual, name in (
        (expected_layer_start, manifest.layer_start, "layer start"),
        (expected_layer_end, manifest.layer_end, "layer end"),
        (expected_total_layers, manifest.total_layers, "total layers"),
    ):
        if expected is not None and expected != actual:
            raise ValueError(
                f"stage artifact {name} does not match the runtime request"
            )

    document = manifest.to_document()
    expected_entries = {
        STAGE_ARTIFACT_MANIFEST,
        *(item["path"] for item in document["files"]),
    }
    actual_entries = {path.name for path in root.iterdir()}
    if actual_entries != expected_entries:
        raise ValueError(
            "stage artifact directory has missing or unsealed files: "
            f"expected={sorted(expected_entries)}, actual={sorted(actual_entries)}"
        )
    by_role: dict[str, Path] = {}
    for item in document["files"]:
        path = root / item["path"]
        if not path.is_file() or path.is_symlink():
            raise ValueError(
                f"stage artifact file is not a regular file: {item['path']}"
            )
        if path.stat().st_size != item["sizeBytes"]:
            raise ValueError(f"stage artifact size mismatch: {item['path']}")
        if _sha256_file(path) != item["sha256"]:
            raise ValueError(f"stage artifact digest mismatch: {item['path']}")
        by_role[item["role"]] = path

    config_path = by_role["model-config"]
    weights_path = by_role["stage-weights"]
    adapter_registry_path = by_role["model-adapter-registry"]
    packaged_registry = _json_object(
        adapter_registry_path.read_bytes(), "packaged model adapter registry"
    )
    validate_adapter_registry_document(
        packaged_registry,
        expected_registry_id=document["model"]["adapterRegistryId"],
    )
    config_document = _json_object(config_path.read_bytes(), "stage model config")
    config_sha256 = hashlib.sha256(_canonical_json(config_document) + b"\n").hexdigest()
    model_document = document["model"]
    if config_sha256 != model_document["configSha256"]:
        raise ValueError("stage config digest does not match the model contract")
    config = AutoConfig.from_pretrained(str(root), local_files_only=True)
    adapter = resolve_selective_stage_adapter(config)
    adapter.validate_source_config(config, manifest.total_layers)
    if adapter.model_type != manifest.family or adapter.adapter_id != manifest.adapter:
        raise ValueError("stage config does not match the sealed family adapter")
    if (
        adapter.adapter_contract_id != model_document["adapterContractId"]
        or model_document["adapterRegistryId"] != ADAPTER_REGISTRY_ID
    ):
        raise ValueError("stage config adapter contract is not installed")
    if _single_architecture(config) != model_document["architecture"]:
        raise ValueError("stage config architecture does not match the manifest")
    if _positive_integer(getattr(config, "hidden_size", None), "hidden_size") != int(
        document["tensorAbi"]["hiddenSize"]
    ):
        raise ValueError("stage config hidden size does not match the tensor ABI")

    actual_layout = _read_safetensors_layout(weights_path)
    descriptors = {item["name"]: item for item in document["tensors"]}
    if set(actual_layout) != set(descriptors):
        raise ValueError("stage weights tensor names do not match the manifest")
    for name, expected in descriptors.items():
        actual = actual_layout[name]
        if (
            actual.dtype != expected["dtype"]
            or list(actual.shape) != expected["shape"]
            or actual.size_bytes != expected["sizeBytes"]
        ):
            raise ValueError(f"stage tensor layout mismatch: {name}")

    key_map = _checkpoint_key_map(root)
    _validate_checkpoint_namespace(key_map, manifest.total_layers)
    verified_adapter, expected_tensors = _expected_stage_tensors(
        config,
        key_map,
        layer_start=manifest.layer_start,
        layer_end=manifest.layer_end,
        total_layers=manifest.total_layers,
    )
    if verified_adapter.adapter_id != manifest.adapter:
        raise ValueError("stage tensor contract resolved a different adapter")
    expected_by_name = {item.name: item for item in expected_tensors}
    if set(expected_by_name) != set(descriptors):
        raise ValueError("stage package contains tensors outside its certified range")
    for name, expected in expected_by_name.items():
        descriptor = descriptors[name]
        if list(expected.shape) != descriptor["shape"]:
            raise ValueError(
                f"stage tensor shape is incompatible with the runtime: {name}"
            )
        if list(expected.roles) != descriptor["roles"]:
            raise ValueError(
                f"stage tensor role is incompatible with the runtime: {name}"
            )

    source_dtypes = tuple(sorted({value.dtype for value in actual_layout.values()}))
    if list(source_dtypes) != document["tensorAbi"]["weightDtypes"]:
        raise ValueError("stage weights dtypes do not match the tensor ABI")
    if (
        _quantization_document(config_document, source_dtypes)
        != model_document["quantization"]
    ):
        raise ValueError(
            "stage quantization contract does not match its config and weights"
        )
    return VerifiedStageArtifact(
        root=root,
        manifest=manifest,
        config_path=config_path,
        weights_path=weights_path,
        adapter_registry_path=adapter_registry_path,
    )


def _expected_stage_tensors(
    config: Any,
    key_map: Mapping[str, str],
    *,
    layer_start: int,
    layer_end: int,
    total_layers: int,
) -> tuple[SelectiveStageAdapter, tuple[_ExpectedTensor, ...]]:
    adapter = resolve_selective_stage_adapter(config)
    adapter.validate_source_config(config, total_layers)
    local_config = copy.deepcopy(config)
    adapter.slice_config(
        local_config,
        layer_start=layer_start,
        layer_end=layer_end,
        total_layers=total_layers,
    )
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(local_config, dtype=torch.float32)
    adapter.inspect_constructed_model(model, local_layers=layer_end - layer_start)
    spec = StageModelSpec(
        model_name="sealed-stage-artifact",
        layer_start=layer_start,
        layer_end=layer_end,
        total_layers=total_layers,
        threads=1,
    )
    tied_embeddings = bool(getattr(model.config, "tie_word_embeddings", False))
    expected: dict[str, tuple[tuple[int, ...], set[str]]] = {}
    seen_tensors: set[int] = set()
    checkpoint_names = set(key_map)
    for local_name, target in model.state_dict(keep_vars=True).items():
        identity = id(target)
        if identity in seen_tensors:
            continue
        seen_tensors.add(identity)
        assignments = adapter.checkpoint_assignments(
            local_name,
            target,
            layer_start=layer_start,
            checkpoint_names=checkpoint_names,
        )
        if assignments is None:
            checkpoint_name = _checkpoint_name(
                local_name,
                spec,
                dict(key_map),
                tied_embeddings=tied_embeddings,
            )
            if checkpoint_name is None:
                continue
            values = ((checkpoint_name, tuple(target.shape)),)
        else:
            values = tuple(
                (assignment.checkpoint_name, tuple(assignment.destination.shape))
                for assignment in assignments
            )
        role = _tensor_role(local_name)
        for checkpoint_name, shape in values:
            prior = expected.get(checkpoint_name)
            if prior is None:
                expected[checkpoint_name] = (shape, {role})
            else:
                if prior[0] != shape:
                    raise ValueError(
                        f"checkpoint tensor {checkpoint_name!r} has conflicting runtime shapes"
                    )
                prior[1].add(role)
    _validate_checkpoint_coverage(
        dict(key_map),
        set(expected),
        spec,
        tied_embeddings=tied_embeddings,
    )
    if not expected:
        raise ValueError("stage range resolved no executable checkpoint tensors")
    return adapter, tuple(
        _ExpectedTensor(name, shape, tuple(sorted(roles)))
        for name, (shape, roles) in sorted(expected.items())
    )


def _select_source_tensors(
    snapshot: Path,
    key_map: Mapping[str, str],
    expected: Sequence[_ExpectedTensor],
) -> tuple[_SelectedSourceTensor, ...]:
    layouts: dict[str, dict[str, _SafeTensorLayout]] = {}
    selected: list[_SelectedSourceTensor] = []
    for item in expected:
        relative = key_map.get(item.name)
        if relative is None:
            raise KeyError(f"required stage tensor is absent: {item.name}")
        source_path = (snapshot / relative).resolve()
        try:
            source_path.relative_to(snapshot)
        except ValueError as error:
            raise ValueError("checkpoint tensor path escapes the snapshot") from error
        by_name = layouts.get(relative)
        if by_name is None:
            by_name = _read_safetensors_layout(source_path)
            layouts[relative] = by_name
        source = by_name.get(item.name)
        if source is None:
            raise KeyError(
                f"checkpoint index maps {item.name!r} to {relative!r}, "
                "but the tensor is absent"
            )
        if source.shape != item.shape:
            raise ValueError(
                f"checkpoint tensor {item.name!r} has shape {source.shape}, "
                f"expected {item.shape}"
            )
        selected.append(
            _SelectedSourceTensor(
                expected=item,
                source_path=source_path,
                source_layout=source,
            )
        )
    return tuple(selected)


def _read_safetensors_layout(path: Path) -> dict[str, _SafeTensorLayout]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"SafeTensors file is missing: {path}")
    # The upstream parser performs its own structural validation. Only keys are
    # requested; get_tensor is deliberately never called.
    with safe_open(path, framework="pt", device="cpu") as reader:
        parser_keys = set(reader.keys())
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError(f"SafeTensors file has no complete header length: {path}")
        header_size = struct.unpack("<Q", prefix)[0]
        if (
            header_size < 2
            or header_size > _MAX_SAFETENSORS_HEADER_BYTES
            or 8 + header_size > file_size
        ):
            raise ValueError(f"SafeTensors header length is invalid: {path}")
        raw_header = handle.read(header_size)
        if len(raw_header) != header_size:
            raise ValueError(f"SafeTensors header is truncated: {path}")
    header = _json_object(raw_header, f"SafeTensors header {path.name}")
    metadata = header.pop("__metadata__", None)
    if metadata is not None and (
        not isinstance(metadata, dict)
        or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in metadata.items()
        )
    ):
        raise ValueError("SafeTensors metadata must contain string keys and values")
    if set(header) != parser_keys:
        raise ValueError("SafeTensors parser and raw header disagree on tensor names")
    data_base = 8 + header_size
    data_size = file_size - data_base
    layouts: dict[str, _SafeTensorLayout] = {}
    ranges: list[tuple[int, int, str]] = []
    for name, value in header.items():
        if not isinstance(name, str) or not name:
            raise ValueError("SafeTensors tensor name cannot be blank")
        entry = _exact_mapping(
            value, ("dtype", "shape", "data_offsets"), f"SafeTensors tensor {name}"
        )
        dtype = _string(entry.get("dtype"), f"SafeTensors tensor {name} dtype")
        element_bytes = _DTYPE_BYTES.get(dtype)
        if element_bytes is None:
            raise ValueError(f"unsupported SafeTensors dtype: {dtype}")
        shape = _shape(entry.get("shape"), f"SafeTensors tensor {name} shape")
        offsets = _sequence(
            entry.get("data_offsets"), f"SafeTensors tensor {name} offsets"
        )
        if len(offsets) != 2:
            raise ValueError(f"SafeTensors tensor {name!r} needs two offsets")
        start = _nonnegative_integer(offsets[0], f"{name} data start")
        end = _positive_integer(offsets[1], f"{name} data end")
        if not start < end <= data_size:
            raise ValueError(f"SafeTensors tensor {name!r} offsets are invalid")
        if end - start != math.prod(shape) * element_bytes:
            raise ValueError(
                f"SafeTensors tensor {name!r} byte range does not match its shape"
            )
        ranges.append((start, end, name))
        layouts[name] = _SafeTensorLayout(
            name=name,
            dtype=dtype,
            shape=shape,
            data_start=data_base + start,
            data_end=data_base + end,
        )
    cursor = 0
    for start, end, name in sorted(ranges):
        if start != cursor:
            raise ValueError(
                f"SafeTensors tensor {name!r} leaves a gap or overlaps another tensor"
            )
        cursor = end
    if cursor != data_size:
        raise ValueError("SafeTensors file contains unaccounted data bytes")
    return layouts


def _write_stage_safetensors(
    path: Path, selected: Sequence[_SelectedSourceTensor]
) -> None:
    if not selected:
        raise ValueError("cannot write an empty stage SafeTensors artifact")
    header: dict[str, dict[str, Any]] = {}
    offset = 0
    for item in selected:
        size = item.source_layout.size_bytes
        header[item.expected.name] = {
            "dtype": item.source_layout.dtype,
            "shape": list(item.source_layout.shape),
            "data_offsets": [offset, offset + size],
        }
        offset += size
    raw_header = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    raw_header += b" " * (-len(raw_header) % 8)
    with path.open("xb") as target:
        target.write(struct.pack("<Q", len(raw_header)))
        target.write(raw_header)
        for item in selected:
            _copy_file_range(
                item.source_path,
                target,
                item.source_layout.data_start,
                item.source_layout.size_bytes,
            )
        target.flush()
        os.fsync(target.fileno())


def _copy_file_range(
    source_path: Path,
    target: BinaryIO,
    start: int,
    length: int,
) -> None:
    remaining = length
    with source_path.open("rb") as source:
        source.seek(start)
        while remaining:
            chunk = source.read(min(remaining, _COPY_BUFFER_BYTES))
            if not chunk:
                raise EOFError(f"checkpoint tensor data is truncated in {source_path}")
            target.write(chunk)
            remaining -= len(chunk)


def _validate_checkpoint_namespace(
    key_map: Mapping[str, str], total_layers: int
) -> None:
    for name in key_map:
        if name.startswith("model.layers."):
            remainder = name.removeprefix("model.layers.")
            index_text, separator, suffix = remainder.partition(".")
            if not separator or not suffix:
                raise ValueError(f"invalid checkpoint layer tensor name {name!r}")
            try:
                index = int(index_text)
            except ValueError as error:
                raise ValueError(
                    f"invalid checkpoint layer tensor name {name!r}"
                ) from error
            if not 0 <= index < total_layers:
                raise ValueError(
                    f"checkpoint tensor {name!r} is outside the configured layer range"
                )
            continue
        if (
            name.startswith("model.embed_tokens.")
            or name.startswith("model.norm.")
            or name.startswith("model.")
            or name.startswith("lm_head.")
        ):
            continue
        raise ValueError(
            f"checkpoint tensor {name!r} is outside the certified model namespace"
        )


def _tensor_role(local_name: str) -> str:
    if local_name.startswith("model.layers."):
        return "layer-weights"
    if local_name.startswith("model.embed_tokens."):
        return "input-embedding"
    if local_name.startswith("model.norm."):
        return "final-normalization"
    if local_name.startswith("lm_head."):
        return "output-head"
    if local_name.startswith("model."):
        return "shared-model-state"
    raise KeyError(f"runtime tensor has no stage artifact role: {local_name!r}")


def _quantization_document(
    config_document: Mapping[str, Any], source_dtypes: Sequence[str]
) -> dict[str, Any]:
    dtypes = tuple(sorted(set(source_dtypes)))
    if not dtypes:
        raise ValueError("stage artifact has no weight dtypes")
    raw = config_document.get("quantization_config")
    if raw is None:
        scheme = "native-" + "-".join(value.lower() for value in dtypes)
        config_sha256 = None
    else:
        if not isinstance(raw, dict) or not raw:
            raise ValueError("quantization_config must be a non-empty object")
        method = raw.get("quant_method")
        scheme = (
            _string(method, "quantization method")
            if method is not None
            else "configured"
        )
        config_sha256 = hashlib.sha256(_canonical_json(raw)).hexdigest()
    return {
        "scheme": scheme,
        "configSha256": config_sha256,
        "weightDtypes": list(dtypes),
    }


def _parse_quantization(value: object) -> dict[str, Any]:
    document = _exact_mapping(
        value,
        ("scheme", "configSha256", "weightDtypes"),
        "stage quantization",
    )
    scheme = _string(document.get("scheme"), "quantization scheme")
    digest = document.get("configSha256")
    if digest is not None:
        digest = _sha256(digest, "quantization config SHA-256")
    dtypes = _string_sequence(
        document.get("weightDtypes"), "quantization weight dtypes"
    )
    if not dtypes or tuple(sorted(set(dtypes))) != dtypes:
        raise ValueError("quantization weight dtypes must be unique and sorted")
    return {
        "scheme": scheme,
        "configSha256": digest,
        "weightDtypes": list(dtypes),
    }


def _single_architecture(config: Any) -> str:
    architectures = getattr(config, "architectures", None)
    if (
        not isinstance(architectures, Sequence)
        or isinstance(architectures, (str, bytes, bytearray))
        or len(architectures) != 1
        or not isinstance(architectures[0], str)
        or not architectures[0]
    ):
        raise ValueError("model config must declare exactly one architecture")
    return architectures[0]


def _write_bytes_atomic_payload(path: Path, value: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_COPY_BUFFER_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _json_object(value: bytes, name: str) -> dict[str, Any]:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"{name} has duplicate field {key!r}")
            result[key] = item
        return result

    try:
        document = json.loads(value, object_pairs_hook=no_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not valid UTF-8 JSON") from error
    if not isinstance(document, dict):
        raise ValueError(f"{name} must be a JSON object")
    return document


def _exact_mapping(value: object, fields: Sequence[str], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    document = dict(value)
    if set(document) != set(fields):
        raise ValueError(f"{name} has unknown or missing fields")
    return document


def _sequence(value: object, name: str) -> tuple[Any, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be an array")
    return tuple(value)


def _string_sequence(value: object, name: str) -> tuple[str, ...]:
    values = _sequence(value, name)
    if any(not isinstance(item, str) or not item for item in values):
        raise ValueError(f"{name} must contain non-empty strings")
    return tuple(values)


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _nullable_string(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _string(value, name)


def _nonnegative_integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _shape(value: object, name: str) -> tuple[int, ...]:
    values = _sequence(value, name)
    if not values:
        raise ValueError(f"{name} cannot be empty")
    return tuple(_positive_integer(item, name) for item in values)


def _hex(value: object, name: str, *, minimum_length: int) -> str:
    text = _string(value, name)
    if len(text) < minimum_length or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError(f"{name} must be lowercase hexadecimal")
    return text


def _sha256(value: object, name: str) -> str:
    text = _hex(value, name, minimum_length=64)
    if len(text) != 64:
        raise ValueError(f"{name} must be a SHA-256 digest")
    return text


def _safe_relative_path(value: object, name: str) -> str:
    text = _string(value, name)
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or len(path.parts) != 1:
        raise ValueError(f"{name} must be a direct relative file path")
    return text


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compile one certified checkpoint layer range into a sealed native "
            "Mycellios SafeTensors stage package."
        )
    )
    parser.add_argument("model_name")
    parser.add_argument("destination")
    parser.add_argument("--revision")
    parser.add_argument("--layer-start", type=int, required=True)
    parser.add_argument("--layer-end", type=int, required=True)
    parser.add_argument(
        "--reuse-verified",
        action="store_true",
        help="reuse an existing package only after full fail-closed verification",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    compiler = (
        prepare_safetensors_stage_artifact
        if args.reuse_verified
        else compile_safetensors_stage_artifact
    )
    result = compiler(
        args.model_name,
        args.destination,
        revision=args.revision,
        layer_start=args.layer_start,
        layer_end=args.layer_end,
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "STAGE_ARTIFACT_ADAPTER_REGISTRY",
    "STAGE_ARTIFACT_CONFIG",
    "STAGE_ARTIFACT_FORMAT",
    "STAGE_ARTIFACT_MANIFEST",
    "STAGE_ARTIFACT_SCHEMA",
    "STAGE_ARTIFACT_WEIGHTS",
    "STAGE_TENSOR_ABI",
    "SafeTensorsStageArtifact",
    "StageArtifactCompilation",
    "VerifiedStageArtifact",
    "compile_safetensors_stage_artifact",
    "main",
    "parse_stage_artifact_manifest",
    "prepare_safetensors_stage_artifact",
    "verify_stage_artifact",
]
