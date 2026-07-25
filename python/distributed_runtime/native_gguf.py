"""Native Mycellios GGUF layer packages.

This module owns the subset of GGUF needed to turn a complete model artifact
into an authenticated, contiguous GDLP stage.  It deliberately does not invoke
an external converter, splitter or model-serving daemon:

* the parser validates GGUF v2/v3 metadata and tensor descriptors itself;
* the writer copies only the selected raw tensor payloads into a new GGUF;
* the package identity seals source, layer range, config and every tensor;
* the materializer dequantizes only the selected stage tensors and emits a
  short-lived SafeTensors snapshot consumed by the native Python stage runner.

The first executable revision certifies dense Llama and Qwen3 tensor naming.
Raw stage extraction preserves common K-quant payloads, but execution is
fail-closed unless the quantization has a native decoder below.  That makes an
unsupported quantization an explicit capability gap rather than silently
falling back to a whole-model external runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
from typing import Any, BinaryIO, Mapping, Sequence
import uuid

import numpy as np
import torch
from safetensors.torch import save_file


NATIVE_GGUF_STAGE_SCHEMA = "gdlp-native-gguf-stage/1"
NATIVE_GGUF_STAGE_MANIFEST = "native-stage.json"
NATIVE_GGUF_STAGE_WEIGHTS = "native-stage.gguf"
NATIVE_GGUF_STAGE_CONFIG = "config.json"
NATIVE_GGUF_FLEET_SCHEMA = "gdlp-native-gguf-fleet/1"
NATIVE_GGUF_FLEET_MANIFEST = "native-fleet.json"
SUPPORTED_ARCHITECTURES = frozenset(("llama", "qwen3"))
DEFAULT_ALIGNMENT = 32
MAX_METADATA_ITEMS = 1_000_000
MAX_TENSORS = 2_000_000
MAX_STRING_BYTES = 64 * 1024 * 1024
MAX_ARRAY_ITEMS = 50_000_000


class NativeGgufError(ValueError):
    """The artifact cannot be accepted by the native GGUF contract."""


@dataclass(frozen=True, slots=True)
class GgufArray:
    element_type: int
    values: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class GgufMetadata:
    key: str
    value_type: int
    value: Any


@dataclass(frozen=True, slots=True)
class GgufTensor:
    name: str
    dimensions: tuple[int, ...]
    ggml_type: int
    relative_offset: int
    size_bytes: int
    data_offset: int

    @property
    def element_count(self) -> int:
        return math.prod(self.dimensions)

    @property
    def torch_shape(self) -> tuple[int, ...]:
        return tuple(reversed(self.dimensions))


@dataclass(frozen=True, slots=True)
class GgufDocument:
    path: Path
    version: int
    metadata: tuple[GgufMetadata, ...]
    tensors: tuple[GgufTensor, ...]
    alignment: int
    data_offset: int
    file_sha256: str

    def metadata_map(self) -> dict[str, Any]:
        return {entry.key: entry.value for entry in self.metadata}

    @property
    def architecture(self) -> str:
        value = self.metadata_map().get("general.architecture")
        if not isinstance(value, str) or not value:
            raise NativeGgufError("GGUF general.architecture is missing")
        return value


@dataclass(frozen=True, slots=True)
class NativeGgufStagePackage:
    root: Path
    package_id: str
    model_source: str
    model_revision: str | None
    source_gguf_sha256: str
    stage_gguf_sha256: str
    config_sha256: str
    architecture: str
    layer_start: int
    layer_end: int
    total_layers: int
    tensor_names: tuple[str, ...]

    @property
    def artifact_identity(self) -> str:
        """Canonical full-model identity shared by every stage range."""

        return _native_model_identity(
            self.source_gguf_sha256,
            self.config_sha256,
        )

    @property
    def package_identity(self) -> str:
        """Identity of this exact range package, distinct across stages."""

        return "sha256:" + self.package_id


@dataclass(frozen=True, slots=True)
class NativeGgufFleet:
    root: Path
    fleet_id: str
    model_source: str
    model_revision: str | None
    source_gguf_sha256: str
    config_sha256: str
    total_layers: int
    stages: tuple[NativeGgufStagePackage, ...]

    @property
    def artifact_identity(self) -> str:
        return _native_model_identity(
            self.source_gguf_sha256,
            self.config_sha256,
        )


# ggml_type -> (elements per block, encoded bytes per block).  Keeping the
# complete common table lets the writer preserve quantized tensors without
# interpreting them.  Execution support is intentionally narrower below.
_GGML_LAYOUTS: dict[int, tuple[int, int]] = {
    0: (1, 4),       # F32
    1: (1, 2),       # F16
    2: (32, 18),     # Q4_0
    3: (32, 20),     # Q4_1
    6: (32, 22),     # Q5_0
    7: (32, 24),     # Q5_1
    8: (32, 34),     # Q8_0
    9: (32, 40),     # Q8_1
    10: (256, 84),   # Q2_K
    11: (256, 110),  # Q3_K
    12: (256, 144),  # Q4_K
    13: (256, 176),  # Q5_K
    14: (256, 210),  # Q6_K
    15: (256, 292),  # Q8_K
    24: (1, 1),      # I8
    25: (1, 2),      # I16
    26: (1, 4),      # I32
    27: (1, 8),      # I64
    28: (1, 8),      # F64
    30: (1, 2),      # BF16
}
_EXECUTABLE_GGML_TYPES = frozenset(
    (0, 1, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 30)
)
_SCALAR_FORMATS: dict[int, str] = {
    0: "<B",
    1: "<b",
    2: "<H",
    3: "<h",
    4: "<I",
    5: "<i",
    6: "<f",
    7: "<B",
    10: "<Q",
    11: "<q",
    12: "<d",
}
_LAYER_TENSOR = re.compile(r"^blk\.(?P<layer>[0-9]+)\.")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def parse_gguf(path_value: str | os.PathLike[str]) -> GgufDocument:
    """Parse and bounds-check one GGUF v2/v3 file without external libraries."""

    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    file_size = path.stat().st_size
    with path.open("rb") as stream:
        reader = _Reader(stream, file_size)
        if reader.read(4) != b"GGUF":
            raise NativeGgufError("artifact is not GGUF")
        version = reader.u32()
        if version not in (2, 3):
            raise NativeGgufError(f"unsupported GGUF version {version}")
        tensor_count = reader.u64()
        metadata_count = reader.u64()
        if tensor_count > MAX_TENSORS:
            raise NativeGgufError("GGUF tensor count is unreasonable")
        if metadata_count > MAX_METADATA_ITEMS:
            raise NativeGgufError("GGUF metadata count is unreasonable")

        metadata: list[GgufMetadata] = []
        seen_metadata: set[str] = set()
        for _ in range(metadata_count):
            key = reader.string()
            if not key or key in seen_metadata:
                raise NativeGgufError("GGUF metadata keys must be non-empty and unique")
            seen_metadata.add(key)
            value_type = reader.u32()
            metadata.append(
                GgufMetadata(key, value_type, reader.metadata_value(value_type))
            )

        descriptors: list[tuple[str, tuple[int, ...], int, int]] = []
        seen_tensors: set[str] = set()
        for _ in range(tensor_count):
            name = reader.string()
            if not name or name in seen_tensors:
                raise NativeGgufError("GGUF tensor names must be non-empty and unique")
            seen_tensors.add(name)
            dimension_count = reader.u32()
            if not 1 <= dimension_count <= 4:
                raise NativeGgufError(f"GGUF tensor {name!r} has invalid rank")
            dimensions = tuple(reader.u64() for _ in range(dimension_count))
            if any(value < 1 for value in dimensions):
                raise NativeGgufError(f"GGUF tensor {name!r} has an empty dimension")
            ggml_type = reader.u32()
            relative_offset = reader.u64()
            descriptors.append((name, dimensions, ggml_type, relative_offset))

        metadata_values = {entry.key: entry.value for entry in metadata}
        alignment_value = metadata_values.get("general.alignment", DEFAULT_ALIGNMENT)
        if (
            not isinstance(alignment_value, int)
            or isinstance(alignment_value, bool)
            or not _power_of_two(alignment_value)
            or not 1 <= alignment_value <= 4096
        ):
            raise NativeGgufError("GGUF general.alignment is invalid")
        data_offset = _align(reader.position, alignment_value)
        if data_offset > file_size:
            raise NativeGgufError("GGUF tensor data section is truncated")

    tensors: list[GgufTensor] = []
    occupied: list[tuple[int, int, str]] = []
    for name, dimensions, ggml_type, relative_offset in descriptors:
        if relative_offset % alignment_value:
            raise NativeGgufError(f"GGUF tensor {name!r} is not aligned")
        size_bytes = _tensor_size_bytes(dimensions, ggml_type, name)
        absolute = data_offset + relative_offset
        end = absolute + size_bytes
        if absolute < data_offset or end > file_size:
            raise NativeGgufError(f"GGUF tensor {name!r} exceeds the artifact")
        occupied.append((absolute, end, name))
        tensors.append(
            GgufTensor(
                name=name,
                dimensions=dimensions,
                ggml_type=ggml_type,
                relative_offset=relative_offset,
                size_bytes=size_bytes,
                data_offset=absolute,
            )
        )
    occupied.sort()
    for left, right in zip(occupied, occupied[1:]):
        if left[1] > right[0]:
            raise NativeGgufError(
                f"GGUF tensors {left[2]!r} and {right[2]!r} overlap"
            )

    return GgufDocument(
        path=path,
        version=version,
        metadata=tuple(metadata),
        tensors=tuple(tensors),
        alignment=alignment_value,
        data_offset=data_offset,
        file_sha256=_sha256_file(path),
    )


def build_native_gguf_stage(
    source_gguf: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    *,
    config_source: str | os.PathLike[str],
    layer_start: int,
    layer_end: int,
    total_layers: int | None = None,
    model_source: str,
    model_revision: str | None,
    _source_document: GgufDocument | None = None,
) -> NativeGgufStagePackage:
    """Atomically build one authenticated contiguous native GGUF stage."""

    requested_source = Path(source_gguf).expanduser().resolve()
    source = _source_document or parse_gguf(requested_source)
    if source.path != requested_source:
        raise NativeGgufError("preparsed GGUF source differs from requested source")
    architecture = source.architecture
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise NativeGgufError(
            f"native GGUF stages do not certify architecture {architecture!r}"
        )
    metadata_map = source.metadata_map()
    declared_layers = metadata_map.get(f"{architecture}.block_count")
    if not isinstance(declared_layers, int) or isinstance(declared_layers, bool):
        raise NativeGgufError("GGUF block_count is missing")
    if total_layers is None:
        total_layers = declared_layers
    if total_layers != declared_layers:
        raise NativeGgufError("requested total_layers differs from GGUF block_count")
    _validate_range(layer_start, layer_end, total_layers)
    if not isinstance(model_source, str) or not model_source.strip():
        raise NativeGgufError("model_source must be non-empty")
    if model_revision is not None and (
        not isinstance(model_revision, str) or not model_revision.strip()
    ):
        raise NativeGgufError("model_revision must be null or non-empty")

    config_path = Path(config_source).expanduser().resolve()
    if config_path.is_dir():
        config_path = config_path / NATIVE_GGUF_STAGE_CONFIG
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config_document = _load_config(config_path)
    _validate_config_against_gguf(
        config_document,
        architecture=architecture,
        total_layers=total_layers,
        metadata=metadata_map,
    )

    selected = _select_stage_tensors(
        source.tensors,
        layer_start=layer_start,
        layer_end=layer_end,
        total_layers=total_layers,
    )
    target = Path(destination).expanduser().resolve()
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.part-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        stage_path = temporary / NATIVE_GGUF_STAGE_WEIGHTS
        _write_stage_gguf(
            source,
            stage_path,
            selected,
            layer_start=layer_start,
            layer_end=layer_end,
            total_layers=total_layers,
        )
        copied_config = temporary / NATIVE_GGUF_STAGE_CONFIG
        shutil.copyfile(config_path, copied_config)
        tensor_records = [
            {
                "name": tensor.name,
                "dimensions": list(tensor.dimensions),
                "ggmlType": tensor.ggml_type,
                "sizeBytes": tensor.size_bytes,
            }
            for tensor in selected
        ]
        without_id = {
            "schema": NATIVE_GGUF_STAGE_SCHEMA,
            "model": {
                "source": model_source,
                "revision": model_revision,
                "architecture": architecture,
                "sourceGgufSha256": source.file_sha256,
            },
            "stage": {
                "layerStart": layer_start,
                "layerEnd": layer_end,
                "totalLayers": total_layers,
                "first": layer_start == 0,
                "last": layer_end == total_layers,
            },
            "files": {
                NATIVE_GGUF_STAGE_WEIGHTS: _file_record(stage_path),
                NATIVE_GGUF_STAGE_CONFIG: _file_record(copied_config),
            },
            "tensors": tensor_records,
            "execution": {
                "engine": "mycellios-native-gguf",
                "externalRuntimeRequired": False,
                "materialization": "stage-only-dequantize-to-torch",
                "tensorLayout": (
                    "llama-rope-qk-permuted"
                    if architecture == "llama"
                    else "huggingface-row-major"
                ),
                "executableGgmlTypes": sorted(_EXECUTABLE_GGML_TYPES),
            },
        }
        package_id = hashlib.sha256(_canonical_json(without_id)).hexdigest()
        manifest = {**without_id, "packageId": package_id}
        _write_json_atomic(temporary / NATIVE_GGUF_STAGE_MANIFEST, manifest)
        verified = verify_native_gguf_stage(temporary)
        os.replace(temporary, target)
        return NativeGgufStagePackage(
            root=target,
            package_id=verified.package_id,
            model_source=verified.model_source,
            model_revision=verified.model_revision,
            source_gguf_sha256=verified.source_gguf_sha256,
            stage_gguf_sha256=verified.stage_gguf_sha256,
            config_sha256=verified.config_sha256,
            architecture=verified.architecture,
            layer_start=verified.layer_start,
            layer_end=verified.layer_end,
            total_layers=verified.total_layers,
            tensor_names=verified.tensor_names,
        )
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def build_native_gguf_fleet(
    source_gguf: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    *,
    config_source: str | os.PathLike[str],
    model_source: str,
    model_revision: str | None,
    ranges: Sequence[tuple[int, int]] | None = None,
    layers_per_stage: int | None = None,
) -> NativeGgufFleet:
    """Compile a complete contiguous stage fleet with one source scan.

    Either explicit ``ranges`` or ``layers_per_stage`` must be supplied.  The
    complete fleet is published atomically only after every child package and
    the fleet-level identity verify.
    """

    source = parse_gguf(source_gguf)
    metadata = source.metadata_map()
    declared_layers = metadata.get(f"{source.architecture}.block_count")
    if not isinstance(declared_layers, int) or isinstance(declared_layers, bool):
        raise NativeGgufError("GGUF block_count is missing")
    if (ranges is None) == (layers_per_stage is None):
        raise NativeGgufError(
            "exactly one of ranges or layers_per_stage must be supplied"
        )
    if layers_per_stage is not None:
        if (
            not isinstance(layers_per_stage, int)
            or isinstance(layers_per_stage, bool)
            or layers_per_stage < 1
        ):
            raise NativeGgufError("layers_per_stage must be positive")
        selected_ranges = tuple(
            (start, min(declared_layers, start + layers_per_stage))
            for start in range(0, declared_layers, layers_per_stage)
        )
    else:
        assert ranges is not None
        selected_ranges = tuple(ranges)
    _validate_complete_ranges(selected_ranges, declared_layers)

    target = Path(destination).expanduser().resolve()
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.part-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        stage_records: list[dict[str, Any]] = []
        fleet_config_sha256: str | None = None
        for index, (layer_start, layer_end) in enumerate(selected_ranges):
            relative = f"stage-{index:04d}-{layer_start:05d}-{layer_end:05d}"
            package = build_native_gguf_stage(
                source.path,
                temporary / relative,
                config_source=config_source,
                layer_start=layer_start,
                layer_end=layer_end,
                total_layers=declared_layers,
                model_source=model_source,
                model_revision=model_revision,
                _source_document=source,
            )
            if fleet_config_sha256 is None:
                fleet_config_sha256 = package.config_sha256
            elif package.config_sha256 != fleet_config_sha256:
                raise NativeGgufError("native GGUF fleet configs are inconsistent")
            stage_records.append(
                {
                    "path": relative,
                    "packageId": package.package_id,
                    "layerStart": layer_start,
                    "layerEnd": layer_end,
                }
            )
        assert fleet_config_sha256 is not None
        without_id = {
            "schema": NATIVE_GGUF_FLEET_SCHEMA,
            "model": {
                "source": model_source,
                "revision": model_revision,
                "sourceGgufSha256": source.file_sha256,
                "configSha256": fleet_config_sha256,
                "architecture": source.architecture,
            },
            "totalLayers": declared_layers,
            "stages": stage_records,
        }
        fleet_id = hashlib.sha256(_canonical_json(without_id)).hexdigest()
        _write_json_atomic(
            temporary / NATIVE_GGUF_FLEET_MANIFEST,
            {**without_id, "fleetId": fleet_id},
        )
        verified = verify_native_gguf_fleet(temporary)
        os.replace(temporary, target)
        return NativeGgufFleet(
            root=target,
            fleet_id=verified.fleet_id,
            model_source=verified.model_source,
            model_revision=verified.model_revision,
            source_gguf_sha256=verified.source_gguf_sha256,
            config_sha256=verified.config_sha256,
            total_layers=verified.total_layers,
            stages=tuple(
                replace(stage, root=target / stage.root.name)
                for stage in verified.stages
            ),
        )
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def verify_native_gguf_fleet(
    fleet_root: str | os.PathLike[str],
    *,
    expected_fleet_id: str | None = None,
) -> NativeGgufFleet:
    """Verify fleet identity, exact child set and complete contiguous coverage."""

    root = Path(fleet_root).expanduser().resolve()
    manifest_path = root / NATIVE_GGUF_FLEET_MANIFEST
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if any(entry.is_symlink() for entry in root.iterdir()):
        raise NativeGgufError("native GGUF fleet cannot contain symbolic links")
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema") != NATIVE_GGUF_FLEET_SCHEMA:
        raise NativeGgufError("native GGUF fleet manifest schema is invalid")
    fleet_id = _digest(document.get("fleetId"), "fleet ID")
    without_id = dict(document)
    without_id.pop("fleetId", None)
    if hashlib.sha256(_canonical_json(without_id)).hexdigest() != fleet_id:
        raise NativeGgufError("native GGUF fleet identity does not match manifest")
    if expected_fleet_id is not None and expected_fleet_id != fleet_id:
        raise NativeGgufError("native GGUF fleet differs from launch contract")
    if set(document) != {"schema", "fleetId", "model", "totalLayers", "stages"}:
        raise NativeGgufError("native GGUF fleet has unknown or missing fields")
    model = _exact_mapping(
        document.get("model"),
        (
            "source",
            "revision",
            "sourceGgufSha256",
            "configSha256",
            "architecture",
        ),
        "fleet model",
    )
    model_source = _nonempty(model.get("source"), "fleet model source")
    revision_value = model.get("revision")
    if revision_value is not None and (
        not isinstance(revision_value, str) or not revision_value.strip()
    ):
        raise NativeGgufError("fleet model revision must be null or non-empty")
    source_digest = _digest(
        model.get("sourceGgufSha256"),
        "fleet source GGUF SHA-256",
    )
    config_digest = _digest(
        model.get("configSha256"),
        "fleet config SHA-256",
    )
    architecture = _nonempty(model.get("architecture"), "fleet architecture")
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise NativeGgufError("native GGUF fleet architecture is unsupported")
    total_layers = _positive_integer(document.get("totalLayers"), "totalLayers")
    stage_values = document.get("stages")
    if not isinstance(stage_values, list) or not stage_values:
        raise NativeGgufError("native GGUF fleet stages are empty")
    stages: list[NativeGgufStagePackage] = []
    declared_paths: list[str] = []
    declared_ranges: list[tuple[int, int]] = []
    for value in stage_values:
        record = _exact_mapping(
            value,
            ("path", "packageId", "layerStart", "layerEnd"),
            "fleet stage",
        )
        relative_text = _nonempty(record.get("path"), "fleet stage path")
        relative = Path(relative_text)
        if (
            relative.is_absolute()
            or len(relative.parts) != 1
            or relative.name != relative_text
            or relative_text == NATIVE_GGUF_FLEET_MANIFEST
        ):
            raise NativeGgufError("native GGUF fleet stage path is invalid")
        package_id = _digest(record.get("packageId"), "fleet stage package ID")
        layer_start = _integer(record.get("layerStart"), "layerStart")
        layer_end = _integer(record.get("layerEnd"), "layerEnd")
        package = verify_native_gguf_stage(
            root / relative,
            expected_package_id=package_id,
            expected_layer_start=layer_start,
            expected_layer_end=layer_end,
            expected_total_layers=total_layers,
        )
        if package.source_gguf_sha256 != source_digest:
            raise NativeGgufError("fleet stages do not share one source GGUF")
        if package.config_sha256 != config_digest:
            raise NativeGgufError("fleet stages do not share one model config")
        if package.model_source != model_source or package.model_revision != revision_value:
            raise NativeGgufError("fleet stage model coordinates are inconsistent")
        if package.architecture != architecture:
            raise NativeGgufError("fleet stage architecture is inconsistent")
        declared_paths.append(relative_text)
        declared_ranges.append((layer_start, layer_end))
        stages.append(package)
    if len(set(declared_paths)) != len(declared_paths):
        raise NativeGgufError("native GGUF fleet repeats a stage path")
    _validate_complete_ranges(tuple(declared_ranges), total_layers)
    actual_entries = {entry.name for entry in root.iterdir()}
    expected_entries = {NATIVE_GGUF_FLEET_MANIFEST, *declared_paths}
    if actual_entries != expected_entries:
        raise NativeGgufError("native GGUF fleet contains unsealed entries")
    if any((root / name).is_symlink() for name in expected_entries):
        raise NativeGgufError("native GGUF fleet cannot contain symbolic links")
    return NativeGgufFleet(
        root=root,
        fleet_id=fleet_id,
        model_source=model_source,
        model_revision=revision_value,
        source_gguf_sha256=source_digest,
        config_sha256=config_digest,
        total_layers=total_layers,
        stages=tuple(stages),
    )


def verify_native_gguf_stage(
    package_root: str | os.PathLike[str],
    *,
    expected_package_id: str | None = None,
    expected_layer_start: int | None = None,
    expected_layer_end: int | None = None,
    expected_total_layers: int | None = None,
) -> NativeGgufStagePackage:
    """Verify identity, exact file set, range and tensor coverage."""

    root = Path(package_root).expanduser().resolve()
    manifest_path = root / NATIVE_GGUF_STAGE_MANIFEST
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    expected_entries = {
        NATIVE_GGUF_STAGE_MANIFEST,
        NATIVE_GGUF_STAGE_WEIGHTS,
        NATIVE_GGUF_STAGE_CONFIG,
    }
    if {entry.name for entry in root.iterdir()} != expected_entries:
        raise NativeGgufError("native GGUF stage contains unsealed entries")
    if any((root / name).is_symlink() for name in expected_entries):
        raise NativeGgufError("native GGUF stage cannot contain symbolic links")
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema") != NATIVE_GGUF_STAGE_SCHEMA:
        raise NativeGgufError("native GGUF stage manifest schema is invalid")
    package_id = document.get("packageId")
    if not isinstance(package_id, str) or not _SHA256.fullmatch(package_id):
        raise NativeGgufError("native GGUF packageId is invalid")
    without_id = dict(document)
    without_id.pop("packageId", None)
    actual_id = hashlib.sha256(_canonical_json(without_id)).hexdigest()
    if actual_id != package_id:
        raise NativeGgufError("native GGUF package identity does not match manifest")
    if expected_package_id is not None and expected_package_id != package_id:
        raise NativeGgufError("native GGUF package identity differs from launch contract")

    model = _exact_mapping(
        document.get("model"),
        ("source", "revision", "architecture", "sourceGgufSha256"),
        "model",
    )
    architecture = _nonempty(model.get("architecture"), "architecture")
    model_source = _nonempty(model.get("source"), "model source")
    model_revision_value = model.get("revision")
    if model_revision_value is not None and (
        not isinstance(model_revision_value, str) or not model_revision_value.strip()
    ):
        raise NativeGgufError("model revision must be null or non-empty")
    model_revision = model_revision_value
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise NativeGgufError("native GGUF architecture is unsupported")
    source_digest = _digest(model.get("sourceGgufSha256"), "source GGUF SHA-256")
    execution = _exact_mapping(
        document.get("execution"),
        (
            "engine",
            "externalRuntimeRequired",
            "materialization",
            "tensorLayout",
            "executableGgmlTypes",
        ),
        "execution",
    )
    if execution.get("engine") != "mycellios-native-gguf":
        raise NativeGgufError("native GGUF execution engine is invalid")
    if execution.get("externalRuntimeRequired") is not False:
        raise NativeGgufError("native GGUF package cannot require an external runtime")
    if execution.get("materialization") != "stage-only-dequantize-to-torch":
        raise NativeGgufError("native GGUF materialization contract is invalid")
    expected_layout = (
        "llama-rope-qk-permuted"
        if architecture == "llama"
        else "huggingface-row-major"
    )
    if execution.get("tensorLayout") != expected_layout:
        raise NativeGgufError("native GGUF tensor layout is invalid")
    executable_types = execution.get("executableGgmlTypes")
    if executable_types != sorted(_EXECUTABLE_GGML_TYPES):
        raise NativeGgufError("native GGUF executable type contract is invalid")
    stage = _exact_mapping(
        document.get("stage"),
        ("layerStart", "layerEnd", "totalLayers", "first", "last"),
        "stage",
    )
    layer_start = _integer(stage.get("layerStart"), "layerStart")
    layer_end = _integer(stage.get("layerEnd"), "layerEnd")
    total_layers = _integer(stage.get("totalLayers"), "totalLayers")
    _validate_range(layer_start, layer_end, total_layers)
    if stage.get("first") is not (layer_start == 0):
        raise NativeGgufError("native GGUF first-stage flag is inconsistent")
    if stage.get("last") is not (layer_end == total_layers):
        raise NativeGgufError("native GGUF last-stage flag is inconsistent")
    for expected, actual, name in (
        (expected_layer_start, layer_start, "layer_start"),
        (expected_layer_end, layer_end, "layer_end"),
        (expected_total_layers, total_layers, "total_layers"),
    ):
        if expected is not None and expected != actual:
            raise NativeGgufError(f"native GGUF {name} differs from launch contract")

    files = _exact_mapping(
        document.get("files"),
        (NATIVE_GGUF_STAGE_WEIGHTS, NATIVE_GGUF_STAGE_CONFIG),
        "files",
    )
    stage_record = _verify_file_record(root, NATIVE_GGUF_STAGE_WEIGHTS, files)
    config_record = _verify_file_record(root, NATIVE_GGUF_STAGE_CONFIG, files)
    parsed = parse_gguf(root / NATIVE_GGUF_STAGE_WEIGHTS)
    if parsed.file_sha256 != stage_record["sha256"]:
        raise NativeGgufError("native GGUF parser digest differs from manifest")
    if parsed.architecture != architecture:
        raise NativeGgufError("native GGUF architecture differs from manifest")
    metadata = parsed.metadata_map()
    expected_metadata = {
        "mycellios.stage.layer_start": layer_start,
        "mycellios.stage.layer_end": layer_end,
        "mycellios.stage.total_layers": total_layers,
        "mycellios.stage.source_sha256": source_digest,
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise NativeGgufError(f"native GGUF metadata {key!r} is inconsistent")

    tensors_value = document.get("tensors")
    if not isinstance(tensors_value, list) or not tensors_value:
        raise NativeGgufError("native GGUF manifest tensor list is empty")
    manifest_tensors: list[dict[str, Any]] = []
    for value in tensors_value:
        record = _exact_mapping(
            value, ("name", "dimensions", "ggmlType", "sizeBytes"), "tensor"
        )
        dimensions = record.get("dimensions")
        if not isinstance(dimensions, list) or not dimensions:
            raise NativeGgufError("native GGUF tensor dimensions are invalid")
        manifest_tensors.append(
            {
                "name": _nonempty(record.get("name"), "tensor name"),
                "dimensions": [_positive_integer(item, "tensor dimension") for item in dimensions],
                "ggmlType": _integer(record.get("ggmlType"), "tensor type"),
                "sizeBytes": _positive_integer(record.get("sizeBytes"), "tensor size"),
            }
        )
    parsed_records = [
        {
            "name": tensor.name,
            "dimensions": list(tensor.dimensions),
            "ggmlType": tensor.ggml_type,
            "sizeBytes": tensor.size_bytes,
        }
        for tensor in parsed.tensors
    ]
    if parsed_records != manifest_tensors:
        raise NativeGgufError("native GGUF tensor directory differs from manifest")
    _validate_selected_tensor_coverage(
        parsed.tensors,
        layer_start=layer_start,
        layer_end=layer_end,
        total_layers=total_layers,
    )
    _validate_config_against_gguf(
        _load_config(root / NATIVE_GGUF_STAGE_CONFIG),
        architecture=architecture,
        total_layers=total_layers,
        metadata=metadata,
    )
    return NativeGgufStagePackage(
        root=root,
        package_id=package_id,
        model_source=model_source,
        model_revision=model_revision,
        source_gguf_sha256=source_digest,
        stage_gguf_sha256=stage_record["sha256"],
        config_sha256=config_record["sha256"],
        architecture=architecture,
        layer_start=layer_start,
        layer_end=layer_end,
        total_layers=total_layers,
        tensor_names=tuple(tensor.name for tensor in parsed.tensors),
    )


def materialize_native_gguf_stage(
    package_root: str | os.PathLike[str],
    destination: str | os.PathLike[str],
) -> Path:
    """Materialize only one verified stage as a temporary HF/SafeTensors snapshot."""

    package = verify_native_gguf_stage(package_root)
    target = Path(destination).expanduser().resolve()
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.part-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        parsed = parse_gguf(package.root / NATIVE_GGUF_STAGE_WEIGHTS)
        config = _load_config(package.root / NATIVE_GGUF_STAGE_CONFIG)
        state: dict[str, torch.Tensor] = {}
        with parsed.path.open("rb") as stream:
            for tensor in parsed.tensors:
                stream.seek(tensor.data_offset)
                raw = stream.read(tensor.size_bytes)
                if len(raw) != tensor.size_bytes:
                    raise NativeGgufError(f"GGUF tensor {tensor.name!r} is truncated")
                decoded = dequantize_gguf_tensor(tensor, raw)
                if tensor.name == "rope_freqs.weight":
                    _validate_derived_rope_factors(
                        decoded,
                        architecture=package.architecture,
                        config=config,
                    )
                    continue
                checkpoint_name = _hf_checkpoint_name(tensor.name)
                if checkpoint_name in state:
                    raise NativeGgufError(
                        f"multiple GGUF tensors map to {checkpoint_name!r}"
                    )
                state[checkpoint_name] = _restore_huggingface_tensor_layout(
                    tensor.name,
                    decoded,
                    architecture=package.architecture,
                    config=config,
                )
        save_file(state, temporary / "model.safetensors")
        shutil.copyfile(
            package.root / NATIVE_GGUF_STAGE_CONFIG,
            temporary / NATIVE_GGUF_STAGE_CONFIG,
        )
        materialization = {
            "schema": "gdlp-native-gguf-materialization/1",
            "packageId": package.package_id,
            "stageGgufSha256": package.stage_gguf_sha256,
            "weightsSha256": _sha256_file(temporary / "model.safetensors"),
            "tensorCount": len(state),
        }
        _write_json_atomic(temporary / "native-materialization.json", materialization)
        os.replace(temporary, target)
        return target
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def dequantize_gguf_tensor(tensor: GgufTensor, raw: bytes) -> torch.Tensor:
    """Decode one tensor with Mycellios-owned, deterministic CPU kernels."""

    if len(raw) != tensor.size_bytes:
        raise NativeGgufError(f"GGUF tensor {tensor.name!r} byte size is invalid")
    if tensor.ggml_type not in _EXECUTABLE_GGML_TYPES:
        raise NativeGgufError(
            f"GGUF tensor {tensor.name!r} uses unsupported executable ggml type "
            f"{tensor.ggml_type}"
        )
    count = tensor.element_count
    ggml_type = tensor.ggml_type
    if ggml_type == 0:
        values = np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=True)
        result = torch.from_numpy(values)
    elif ggml_type == 1:
        values = np.frombuffer(raw, dtype="<f2").astype(np.float32)
        result = torch.from_numpy(values)
    elif ggml_type == 30:
        words = np.frombuffer(raw, dtype="<u2").copy()
        result = torch.from_numpy(words).view(torch.bfloat16).to(torch.float32)
    elif ggml_type in (24, 25, 26, 27):
        dtype = {24: "<i1", 25: "<i2", 26: "<i4", 27: "<i8"}[ggml_type]
        values = np.frombuffer(raw, dtype=dtype).astype(np.float32)
        result = torch.from_numpy(values)
    elif ggml_type == 28:
        values = np.frombuffer(raw, dtype="<f8").astype(np.float32)
        result = torch.from_numpy(values)
    elif ggml_type in (10, 11, 12, 13, 14, 15):
        result = torch.from_numpy(_dequantize_k_blocks(raw, ggml_type))
    else:
        result = torch.from_numpy(_dequantize_blocks(raw, ggml_type))
    if result.numel() != count:
        raise NativeGgufError(
            f"GGUF tensor {tensor.name!r} decoded {result.numel()} values, expected {count}"
        )
    return result.reshape(tensor.torch_shape).contiguous()


def _restore_huggingface_tensor_layout(
    gguf_name: str,
    value: torch.Tensor,
    *,
    architecture: str,
    config: Mapping[str, Any],
) -> torch.Tensor:
    """Undo architecture-specific conversion transforms after dequantization.

    The canonical Llama GGUF layout interleaves the two rotary components of Q
    and K. Transformers expects the original Hugging Face row
    order, so a direct name mapping is not sufficient even though shapes match.
    Qwen3's canonical converter does not apply this transform.
    """

    if architecture != "llama":
        return value
    if not (
        gguf_name.endswith(".attn_q.weight")
        or gguf_name.endswith(".attn_q.bias")
        or gguf_name.endswith(".attn_k.weight")
        or gguf_name.endswith(".attn_k.bias")
    ):
        return value
    attention_heads = config.get("num_attention_heads")
    kv_heads = config.get("num_key_value_heads", attention_heads)
    if not isinstance(attention_heads, int) or isinstance(attention_heads, bool):
        raise NativeGgufError("Llama config num_attention_heads is invalid")
    if not isinstance(kv_heads, int) or isinstance(kv_heads, bool):
        raise NativeGgufError("Llama config num_key_value_heads is invalid")
    heads = kv_heads if ".attn_k." in gguf_name else attention_heads
    if heads < 1 or value.shape[0] % (heads * 2):
        raise NativeGgufError(
            f"GGUF tensor {gguf_name!r} cannot restore Llama rotary layout"
        )
    trailing = tuple(value.shape[1:])
    restored = (
        value.reshape(heads, value.shape[0] // heads // 2, 2, *trailing)
        .swapaxes(1, 2)
        .reshape(value.shape)
    )
    return restored.contiguous()


def _validate_derived_rope_factors(
    value: torch.Tensor,
    *,
    architecture: str,
    config: Mapping[str, Any],
) -> None:
    """Prove that a GGUF-only RoPE tensor is recreated by the sealed config."""

    scaling = config.get("rope_scaling")
    if architecture != "llama" or not isinstance(scaling, Mapping):
        raise NativeGgufError(
            "rope_freqs.weight is unsupported without a sealed Llama3 RoPE config"
        )
    rope_type = scaling.get("rope_type", scaling.get("type"))
    if rope_type != "llama3":
        raise NativeGgufError(
            "rope_freqs.weight is supported only for certified Llama3 RoPE"
        )

    def number(mapping: Mapping[str, Any], key: str) -> float:
        candidate = mapping.get(key)
        if (
            not isinstance(candidate, (int, float))
            or isinstance(candidate, bool)
            or not math.isfinite(float(candidate))
            or float(candidate) <= 0
        ):
            raise NativeGgufError(f"Llama3 RoPE config {key!r} is invalid")
        return float(candidate)

    hidden_size = config.get("hidden_size")
    attention_heads = config.get("num_attention_heads")
    head_dim_value = config.get("head_dim")
    if head_dim_value is None:
        if (
            not isinstance(hidden_size, int)
            or isinstance(hidden_size, bool)
            or not isinstance(attention_heads, int)
            or isinstance(attention_heads, bool)
            or attention_heads < 1
            or hidden_size % attention_heads
        ):
            raise NativeGgufError("Llama3 RoPE head dimension is invalid")
        head_dim = hidden_size // attention_heads
    elif (
        not isinstance(head_dim_value, int)
        or isinstance(head_dim_value, bool)
        or head_dim_value < 2
    ):
        raise NativeGgufError("Llama3 RoPE head dimension is invalid")
    else:
        head_dim = head_dim_value
    if head_dim % 2:
        raise NativeGgufError("Llama3 RoPE head dimension must be even")
    base_value = config.get("rope_theta", 10_000.0)
    if (
        not isinstance(base_value, (int, float))
        or isinstance(base_value, bool)
        or not math.isfinite(float(base_value))
        or float(base_value) <= 0
    ):
        raise NativeGgufError("Llama3 rope_theta is invalid")
    factor = number(scaling, "factor")
    low_factor = number(scaling, "low_freq_factor")
    high_factor = number(scaling, "high_freq_factor")
    original_context = number(scaling, "original_max_position_embeddings")
    if high_factor <= low_factor:
        raise NativeGgufError(
            "Llama3 high_freq_factor must exceed low_freq_factor"
        )

    positions = np.arange(0, head_dim, 2, dtype=np.float64)
    frequencies = 1.0 / (
        float(base_value) ** (positions / float(head_dim))
    )
    low_wavelength = original_context / low_factor
    high_wavelength = original_context / high_factor
    expected: list[float] = []
    for frequency in frequencies:
        wavelength = 2.0 * math.pi / float(frequency)
        if wavelength < high_wavelength:
            expected.append(1.0)
        elif wavelength > low_wavelength:
            expected.append(factor)
        else:
            smooth = (
                original_context / wavelength - low_factor
            ) / (high_factor - low_factor)
            expected.append(1.0 / ((1.0 - smooth) / factor + smooth))
    actual = value.detach().cpu().to(torch.float32).reshape(-1)
    expected_tensor = torch.tensor(expected, dtype=torch.float32)
    if actual.shape != expected_tensor.shape or not torch.allclose(
        actual,
        expected_tensor,
        rtol=1e-5,
        atol=1e-6,
    ):
        raise NativeGgufError(
            "rope_freqs.weight differs from the sealed Llama3 RoPE config"
        )


def _dequantize_blocks(raw: bytes, ggml_type: int) -> np.ndarray:
    block_elements, block_bytes = _GGML_LAYOUTS[ggml_type]
    matrix = np.frombuffer(raw, dtype=np.uint8).reshape(-1, block_bytes)
    scales = matrix[:, :2].copy().view("<f2").reshape(-1).astype(np.float32)
    if not np.isfinite(scales).all():
        raise NativeGgufError("quantized GGUF tensor contains non-finite scales")
    if ggml_type == 8:  # Q8_0
        quants = matrix[:, 2:34].view(np.int8).astype(np.float32)
        return (quants * scales[:, None]).reshape(-1)

    if ggml_type in (2, 3):
        quant_offset = 2 if ggml_type == 2 else 4
        packed = matrix[:, quant_offset : quant_offset + 16]
        low = (packed & 0x0F).astype(np.float32)
        high = (packed >> 4).astype(np.float32)
        quants = np.concatenate((low, high), axis=1)
        if ggml_type == 2:
            quants -= 8.0
            return (quants * scales[:, None]).reshape(-1)
        minimums = matrix[:, 2:4].copy().view("<f2").reshape(-1).astype(np.float32)
        return (quants * scales[:, None] + minimums[:, None]).reshape(-1)

    if ggml_type in (6, 7):
        minimum_offset = 2 if ggml_type == 6 else 4
        high_offset = minimum_offset
        packed_offset = high_offset + 4
        high_bits = matrix[:, high_offset : high_offset + 4]
        packed = matrix[:, packed_offset : packed_offset + 16]
        low = packed & 0x0F
        high_nibbles = packed >> 4
        quants = np.empty((matrix.shape[0], block_elements), dtype=np.float32)
        # GGML stores high-bit i in qh bit i, while the low nibbles store
        # positions [0..15] and [16..31] in the low/high half of each byte.
        qh = high_bits[:, 0].astype(np.uint32)
        qh |= high_bits[:, 1].astype(np.uint32) << 8
        qh |= high_bits[:, 2].astype(np.uint32) << 16
        qh |= high_bits[:, 3].astype(np.uint32) << 24
        for index in range(16):
            quants[:, index] = low[:, index] | (((qh >> index) & 1) << 4)
            quants[:, index + 16] = high_nibbles[:, index] | (
                ((qh >> (index + 16)) & 1) << 4
            )
        if ggml_type == 6:
            quants -= 16.0
            return (quants * scales[:, None]).reshape(-1)
        minimums = matrix[:, 2:4].copy().view("<f2").reshape(-1).astype(np.float32)
        return (quants * scales[:, None] + minimums[:, None]).reshape(-1)
    raise NativeGgufError(f"native decoder is missing for ggml type {ggml_type}")


def _dequantize_k_blocks(raw: bytes, ggml_type: int) -> np.ndarray:
    """Decode GGML's 256-element K-quant super-blocks."""

    block_elements, block_bytes = _GGML_LAYOUTS[ggml_type]
    if block_elements != 256 or len(raw) % block_bytes:
        raise NativeGgufError("K-quantized GGUF tensor has an invalid block size")
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, block_bytes)
    decoders = {
        10: _decode_q2_k_blocks,
        11: _decode_q3_k_blocks,
        12: _decode_q4_k_blocks,
        13: _decode_q5_k_blocks,
        14: _decode_q6_k_blocks,
        15: _decode_q8_k_blocks,
    }
    values = decoders[ggml_type](blocks)
    if values.shape != (blocks.shape[0], 256) or not np.isfinite(values).all():
        raise NativeGgufError("K-quantized GGUF tensor decoded invalid values")
    return values.astype(np.float32, copy=False).reshape(-1)


def _half(raw: np.ndarray) -> float:
    return float(raw.copy().view("<f2")[0])


def _halves(raw: np.ndarray) -> np.ndarray:
    return raw.copy().view("<f2").reshape(raw.shape[0]).astype(np.float32)


def _decode_q2_k_blocks(blocks: np.ndarray) -> np.ndarray:
    scales = blocks[:, :16]
    quants = blocks[:, 16:80]
    d = _halves(blocks[:, 80:82])
    minimum = _halves(blocks[:, 82:84])
    output = np.empty((blocks.shape[0], 256), dtype=np.float32)
    scale_index = 0
    cursor = 0
    for base in (0, 32):
        q = quants[:, base : base + 32]
        for shift in (0, 2, 4, 6):
            for offset in (0, 16):
                encoded = scales[:, scale_index]
                scale_index += 1
                values = ((q[:, offset : offset + 16] >> shift) & 3).astype(
                    np.float32
                )
                output[:, cursor : cursor + 16] = (
                    d[:, None] * (encoded & 15)[:, None] * values
                    - minimum[:, None] * (encoded >> 4)[:, None]
                )
                cursor += 16
    return output


def _decode_q3_k_blocks(blocks: np.ndarray) -> np.ndarray:
    high_mask = blocks[:, :32]
    quants = blocks[:, 32:96]
    words = blocks[:, 96:108].copy().view("<u4").reshape(-1, 3)
    d = _halves(blocks[:, 108:110])
    mask_low = np.uint32(0x03030303)
    mask_nibble = np.uint32(0x0F0F0F0F)
    temporary = words[:, 2]
    expanded = np.empty((blocks.shape[0], 4), dtype="<u4")
    expanded[:, 0] = (words[:, 0] & mask_nibble) | (
        ((temporary >> 0) & mask_low) << 4
    )
    expanded[:, 1] = (words[:, 1] & mask_nibble) | (
        ((temporary >> 2) & mask_low) << 4
    )
    expanded[:, 2] = ((words[:, 0] >> 4) & mask_nibble) | (
        ((temporary >> 4) & mask_low) << 4
    )
    expanded[:, 3] = ((words[:, 1] >> 4) & mask_nibble) | (
        ((temporary >> 6) & mask_low) << 4
    )
    scales = expanded.view(np.uint8).reshape(-1, 16).astype(np.int16) - 32
    output = np.empty((blocks.shape[0], 256), dtype=np.float32)
    scale_index = 0
    mask_bit = 1
    cursor = 0
    for base in (0, 32):
        q = quants[:, base : base + 32]
        for shift in (0, 2, 4, 6):
            for offset in (0, 16):
                low = ((q[:, offset : offset + 16] >> shift) & 3).astype(
                    np.int16
                )
                signed = low - np.where(
                    high_mask[:, offset : offset + 16] & mask_bit,
                    0,
                    4,
                )
                output[:, cursor : cursor + 16] = (
                    d[:, None] * scales[:, scale_index, None] * signed
                )
                scale_index += 1
                cursor += 16
            mask_bit <<= 1
    return output


def _k_scale_min_blocks(
    index: int,
    packed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if index < 4:
        return packed[:, index] & 63, packed[:, index + 4] & 63
    scale = (packed[:, index + 4] & 15) | (
        (packed[:, index - 4] >> 6) << 4
    )
    minimum = (packed[:, index + 4] >> 4) | (
        (packed[:, index] >> 6) << 4
    )
    return scale, minimum


def _decode_q4_k_blocks(blocks: np.ndarray) -> np.ndarray:
    d = _halves(blocks[:, :2])
    minimum = _halves(blocks[:, 2:4])
    scales = blocks[:, 4:16]
    quants = blocks[:, 16:144]
    output = np.empty((blocks.shape[0], 256), dtype=np.float32)
    cursor = 0
    for pair in range(4):
        q = quants[:, pair * 32 : (pair + 1) * 32]
        for local, values in enumerate((q & 15, q >> 4)):
            scale, encoded_min = _k_scale_min_blocks(pair * 2 + local, scales)
            output[:, cursor : cursor + 32] = (
                d[:, None] * scale[:, None] * values
                - minimum[:, None] * encoded_min[:, None]
            )
            cursor += 32
    return output


def _decode_q5_k_blocks(blocks: np.ndarray) -> np.ndarray:
    d = _halves(blocks[:, :2])
    minimum = _halves(blocks[:, 2:4])
    scales = blocks[:, 4:16]
    high = blocks[:, 16:48]
    low = blocks[:, 48:176]
    output = np.empty((blocks.shape[0], 256), dtype=np.float32)
    cursor = 0
    for pair, bits in enumerate(((1, 2), (4, 8), (16, 32), (64, 128))):
        q = low[:, pair * 32 : (pair + 1) * 32]
        for local, (values, bit) in enumerate(
            ((q & 15, bits[0]), (q >> 4, bits[1]))
        ):
            scale, encoded_min = _k_scale_min_blocks(pair * 2 + local, scales)
            quant = values.astype(np.int16) + np.where(high & bit, 16, 0)
            output[:, cursor : cursor + 32] = (
                d[:, None] * scale[:, None] * quant
                - minimum[:, None] * encoded_min[:, None]
            )
            cursor += 32
    return output


def _decode_q6_k_blocks(blocks: np.ndarray) -> np.ndarray:
    low = blocks[:, :128]
    high = blocks[:, 128:192]
    scales = blocks[:, 192:208].view(np.int8).astype(np.int16)
    d = _halves(blocks[:, 208:210])
    output = np.empty((blocks.shape[0], 256), dtype=np.float32)
    for segment in range(2):
        ql = low[:, segment * 64 : (segment + 1) * 64]
        qh = high[:, segment * 32 : (segment + 1) * 32]
        local_scales = scales[:, segment * 8 : (segment + 1) * 8]
        scale_pairs = tuple(
            np.repeat(local_scales[:, group * 2 : group * 2 + 2], 16, axis=1)
            for group in range(4)
        )
        quants = (
            (ql[:, :32] & 15) | (((qh >> 0) & 3) << 4),
            (ql[:, 32:64] & 15) | (((qh >> 2) & 3) << 4),
            (ql[:, :32] >> 4) | (((qh >> 4) & 3) << 4),
            (ql[:, 32:64] >> 4) | (((qh >> 6) & 3) << 4),
        )
        for group, (quant, local_scale) in enumerate(
            zip(quants, scale_pairs, strict=True)
        ):
            start = segment * 128 + group * 32
            output[:, start : start + 32] = (
                d[:, None]
                * local_scale
                * (quant.astype(np.int16) - 32)
            )
    return output


def _decode_q8_k_blocks(blocks: np.ndarray) -> np.ndarray:
    d = blocks[:, :4].copy().view("<f4").reshape(-1).astype(np.float32)
    return d[:, None] * blocks[:, 4:260].view(np.int8).astype(np.float32)


def _decode_q2_k(block: np.ndarray) -> np.ndarray:
    scales = block[:16]
    quants = block[16:80]
    d = _half(block[80:82])
    minimum = _half(block[82:84])
    output: list[np.ndarray] = []
    scale_index = 0
    for base in (0, 32):
        q = quants[base : base + 32]
        for shift in (0, 2, 4, 6):
            for half in (q[:16], q[16:]):
                encoded_scale = int(scales[scale_index])
                scale_index += 1
                local_d = d * (encoded_scale & 0x0F)
                local_min = minimum * (encoded_scale >> 4)
                values = ((half >> shift) & 0x03).astype(np.float32)
                output.append(local_d * values - local_min)
    return np.concatenate(output)


def _decode_q3_k(block: np.ndarray) -> np.ndarray:
    high_mask = block[:32]
    quants = block[32:96]
    packed_scales = block[96:108]
    d = _half(block[108:110])
    scale_words = [
        int.from_bytes(packed_scales[offset : offset + 4].tobytes(), "little")
        for offset in (0, 4, 8)
    ]
    mask_low = 0x03030303
    mask_nibble = 0x0F0F0F0F
    temporary = scale_words[2]
    expanded_words = (
        (scale_words[0] & mask_nibble)
        | (((temporary >> 0) & mask_low) << 4),
        (scale_words[1] & mask_nibble)
        | (((temporary >> 2) & mask_low) << 4),
        ((scale_words[0] >> 4) & mask_nibble)
        | (((temporary >> 4) & mask_low) << 4),
        ((scale_words[1] >> 4) & mask_nibble)
        | (((temporary >> 6) & mask_low) << 4),
    )
    scales = np.frombuffer(
        b"".join(word.to_bytes(4, "little") for word in expanded_words),
        dtype=np.uint8,
    ).astype(np.int16) - 32
    output: list[np.ndarray] = []
    scale_index = 0
    mask_bit = 1
    for base in (0, 32):
        q = quants[base : base + 32]
        for shift in (0, 2, 4, 6):
            for offset in (0, 16):
                low = ((q[offset : offset + 16] >> shift) & 0x03).astype(
                    np.int16
                )
                high = high_mask[offset : offset + 16] & mask_bit
                signed = low - np.where(high != 0, 0, 4)
                output.append(
                    (d * int(scales[scale_index]) * signed).astype(np.float32)
                )
                scale_index += 1
            mask_bit <<= 1
    return np.concatenate(output)


def _k_scale_min(index: int, packed: np.ndarray) -> tuple[int, int]:
    if index < 4:
        return int(packed[index] & 63), int(packed[index + 4] & 63)
    scale = int(packed[index + 4] & 0x0F) | (
        int(packed[index - 4] >> 6) << 4
    )
    minimum = int(packed[index + 4] >> 4) | (
        int(packed[index] >> 6) << 4
    )
    return scale, minimum


def _decode_q4_k(block: np.ndarray) -> np.ndarray:
    d = _half(block[:2])
    minimum = _half(block[2:4])
    scales = block[4:16]
    quants = block[16:144]
    output: list[np.ndarray] = []
    for pair in range(4):
        q = quants[pair * 32 : (pair + 1) * 32]
        for local, values in enumerate((q & 0x0F, q >> 4)):
            scale, encoded_min = _k_scale_min(pair * 2 + local, scales)
            output.append(
                (
                    d * scale * values.astype(np.float32)
                    - minimum * encoded_min
                ).astype(np.float32)
            )
    return np.concatenate(output)


def _decode_q5_k(block: np.ndarray) -> np.ndarray:
    d = _half(block[:2])
    minimum = _half(block[2:4])
    scales = block[4:16]
    high = block[16:48]
    low = block[48:176]
    output: list[np.ndarray] = []
    high_bits = ((1, 2), (4, 8), (16, 32), (64, 128))
    for pair, bits in enumerate(high_bits):
        q = low[pair * 32 : (pair + 1) * 32]
        pairs = ((q & 0x0F, bits[0]), (q >> 4, bits[1]))
        for local, (values, bit) in enumerate(pairs):
            scale, encoded_min = _k_scale_min(pair * 2 + local, scales)
            quant = values.astype(np.int16) + np.where(high & bit, 16, 0)
            output.append(
                (d * scale * quant - minimum * encoded_min).astype(np.float32)
            )
    return np.concatenate(output)


def _decode_q6_k(block: np.ndarray) -> np.ndarray:
    low = block[:128]
    high = block[128:192]
    scales = block[192:208].view(np.int8).astype(np.int16)
    d = _half(block[208:210])
    output = np.empty(256, dtype=np.float32)
    for segment in range(2):
        ql = low[segment * 64 : (segment + 1) * 64]
        qh = high[segment * 32 : (segment + 1) * 32]
        local_scales = scales[segment * 8 : (segment + 1) * 8]
        for index in range(32):
            scale_pair = index // 16
            quantized = (
                int(ql[index] & 0x0F) | (int((qh[index] >> 0) & 0x03) << 4),
                int(ql[index + 32] & 0x0F)
                | (int((qh[index] >> 2) & 0x03) << 4),
                int(ql[index] >> 4) | (int((qh[index] >> 4) & 0x03) << 4),
                int(ql[index + 32] >> 4)
                | (int((qh[index] >> 6) & 0x03) << 4),
            )
            positions = (index, index + 32, index + 64, index + 96)
            for group, (position, quant) in enumerate(
                zip(positions, quantized, strict=True)
            ):
                output[segment * 128 + position] = (
                    d * int(local_scales[scale_pair + group * 2]) * (quant - 32)
                )
    return output


def _decode_q8_k(block: np.ndarray) -> np.ndarray:
    d = float(block[:4].copy().view("<f4")[0])
    return d * block[4:260].view(np.int8).astype(np.float32)


def _write_stage_gguf(
    source: GgufDocument,
    destination: Path,
    tensors: Sequence[GgufTensor],
    *,
    layer_start: int,
    layer_end: int,
    total_layers: int,
) -> None:
    architecture_prefix = source.architecture + "."
    retained_general = frozenset(
        (
            "general.architecture",
            "general.alignment",
            "general.file_type",
            "general.quantization_version",
        )
    )
    metadata = [
        entry
        for entry in source.metadata
        if entry.key in retained_general or entry.key.startswith(architecture_prefix)
    ]
    metadata.extend(
        (
            GgufMetadata("general.alignment", 4, source.alignment),
            GgufMetadata("mycellios.stage.layer_start", 4, layer_start),
            GgufMetadata("mycellios.stage.layer_end", 4, layer_end),
            GgufMetadata("mycellios.stage.total_layers", 4, total_layers),
            GgufMetadata("mycellios.stage.source_sha256", 8, source.file_sha256),
        )
    )
    deduplicated: dict[str, GgufMetadata] = {}
    for entry in metadata:
        deduplicated[entry.key] = entry
    metadata = list(deduplicated.values())

    offsets: list[int] = []
    cursor = 0
    for tensor in tensors:
        cursor = _align(cursor, source.alignment)
        offsets.append(cursor)
        cursor += tensor.size_bytes

    with destination.open("wb") as output, source.path.open("rb") as input_stream:
        output.write(b"GGUF")
        output.write(struct.pack("<IQQ", source.version, len(tensors), len(metadata)))
        for entry in metadata:
            _write_string(output, entry.key)
            output.write(struct.pack("<I", entry.value_type))
            _write_metadata_value(output, entry.value_type, entry.value)
        for tensor, offset in zip(tensors, offsets, strict=True):
            _write_string(output, tensor.name)
            output.write(struct.pack("<I", len(tensor.dimensions)))
            for dimension in tensor.dimensions:
                output.write(struct.pack("<Q", dimension))
            output.write(struct.pack("<IQ", tensor.ggml_type, offset))
        _pad_to_alignment(output, source.alignment)
        data_start = output.tell()
        for tensor, offset in zip(tensors, offsets, strict=True):
            desired = data_start + offset
            if output.tell() > desired:
                raise NativeGgufError("native GGUF writer produced overlapping tensors")
            output.write(b"\0" * (desired - output.tell()))
            input_stream.seek(tensor.data_offset)
            _copy_exact(input_stream, output, tensor.size_bytes)
        output.flush()
        os.fsync(output.fileno())


def _select_stage_tensors(
    tensors: Sequence[GgufTensor],
    *,
    layer_start: int,
    layer_end: int,
    total_layers: int,
) -> tuple[GgufTensor, ...]:
    by_name = {tensor.name: tensor for tensor in tensors}
    selected: list[GgufTensor] = []
    output_present = any(name.startswith("output.") for name in by_name)
    for tensor in tensors:
        match = _LAYER_TENSOR.match(tensor.name)
        if match:
            layer = int(match.group("layer"))
            if layer_start <= layer < layer_end:
                selected.append(tensor)
            continue
        first = tensor.name.startswith(
            ("token_embd.", "token_types.", "position_embd.")
        )
        last = tensor.name.startswith(("output.", "output_norm."))
        shared = tensor.name.startswith(("rope_freqs.",))
        tied_last = (
            layer_end == total_layers
            and not output_present
            and tensor.name == "token_embd.weight"
        )
        if (
            (layer_start == 0 and first)
            or (layer_end == total_layers and last)
            or tied_last
            or shared
        ):
            selected.append(tensor)
    _validate_selected_tensor_coverage(
        selected,
        layer_start=layer_start,
        layer_end=layer_end,
        total_layers=total_layers,
    )
    return tuple(selected)


def _validate_selected_tensor_coverage(
    tensors: Sequence[GgufTensor],
    *,
    layer_start: int,
    layer_end: int,
    total_layers: int,
) -> None:
    layers: dict[int, int] = {}
    names: set[str] = set()
    for tensor in tensors:
        if tensor.name in names:
            raise NativeGgufError("native GGUF stage repeats a tensor")
        names.add(tensor.name)
        match = _LAYER_TENSOR.match(tensor.name)
        if match:
            layer = int(match.group("layer"))
            if not layer_start <= layer < layer_end:
                raise NativeGgufError("native GGUF stage contains an out-of-range layer")
            layers[layer] = layers.get(layer, 0) + 1
    missing = [layer for layer in range(layer_start, layer_end) if layer not in layers]
    if missing:
        raise NativeGgufError(f"native GGUF stage omits layers {missing}")
    if layer_start == 0 and "token_embd.weight" not in names:
        raise NativeGgufError("first native GGUF stage omits token_embd.weight")
    if layer_end == total_layers:
        if "output_norm.weight" not in names:
            raise NativeGgufError("last native GGUF stage omits output_norm.weight")
        if "output.weight" not in names and "token_embd.weight" not in names:
            raise NativeGgufError("last native GGUF stage omits output projection")


def _hf_checkpoint_name(name: str) -> str:
    direct = {
        "token_embd.weight": "model.embed_tokens.weight",
        "output_norm.weight": "model.norm.weight",
        "output_norm.bias": "model.norm.bias",
        "output.weight": "lm_head.weight",
        "output.bias": "lm_head.bias",
    }
    if name in direct:
        return direct[name]
    match = _LAYER_TENSOR.match(name)
    if not match:
        if name.startswith("rope_freqs."):
            return "model." + name
        raise NativeGgufError(f"native GGUF tensor mapping is unsupported for {name!r}")
    layer = int(match.group("layer"))
    suffix = name[match.end() :]
    mappings = {
        "attn_norm.weight": "input_layernorm.weight",
        "attn_norm.bias": "input_layernorm.bias",
        "attn_q.weight": "self_attn.q_proj.weight",
        "attn_q.bias": "self_attn.q_proj.bias",
        "attn_k.weight": "self_attn.k_proj.weight",
        "attn_k.bias": "self_attn.k_proj.bias",
        "attn_v.weight": "self_attn.v_proj.weight",
        "attn_v.bias": "self_attn.v_proj.bias",
        "attn_output.weight": "self_attn.o_proj.weight",
        "attn_output.bias": "self_attn.o_proj.bias",
        "attn_q_norm.weight": "self_attn.q_norm.weight",
        "attn_k_norm.weight": "self_attn.k_norm.weight",
        "ffn_norm.weight": "post_attention_layernorm.weight",
        "ffn_norm.bias": "post_attention_layernorm.bias",
        "ffn_gate.weight": "mlp.gate_proj.weight",
        "ffn_gate.bias": "mlp.gate_proj.bias",
        "ffn_up.weight": "mlp.up_proj.weight",
        "ffn_up.bias": "mlp.up_proj.bias",
        "ffn_down.weight": "mlp.down_proj.weight",
        "ffn_down.bias": "mlp.down_proj.bias",
    }
    mapped = mappings.get(suffix)
    if mapped is None:
        raise NativeGgufError(
            f"native GGUF layer tensor mapping is unsupported for {name!r}"
        )
    return f"model.layers.{layer}.{mapped}"


def _validate_config_against_gguf(
    config: Mapping[str, Any],
    *,
    architecture: str,
    total_layers: int,
    metadata: Mapping[str, Any],
) -> None:
    configured_layers = config.get("num_hidden_layers")
    hidden_size = config.get("hidden_size")
    gguf_hidden = metadata.get(f"{architecture}.embedding_length")
    if configured_layers != total_layers:
        raise NativeGgufError("config num_hidden_layers differs from GGUF")
    if not isinstance(hidden_size, int) or hidden_size < 1 or hidden_size != gguf_hidden:
        raise NativeGgufError("config hidden_size differs from GGUF")
    model_type = config.get("model_type")
    accepted_model_types = {
        "llama": frozenset(("llama",)),
        "qwen3": frozenset(("qwen3",)),
    }[architecture]
    if model_type not in accepted_model_types:
        raise NativeGgufError("config model_type differs from GGUF architecture")


def _load_config(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise NativeGgufError("config.json must contain an object")
    return value


def _tensor_size_bytes(
    dimensions: Sequence[int], ggml_type: int, name: str
) -> int:
    layout = _GGML_LAYOUTS.get(ggml_type)
    if layout is None:
        raise NativeGgufError(
            f"GGUF tensor {name!r} uses unsupported storage ggml type {ggml_type}"
        )
    elements = math.prod(dimensions)
    block_elements, block_bytes = layout
    if dimensions[0] % block_elements:
        raise NativeGgufError(
            f"GGUF tensor {name!r} row width is not divisible by its block size"
        )
    return elements // block_elements * block_bytes


class _Reader:
    def __init__(self, stream: BinaryIO, size: int) -> None:
        self.stream = stream
        self.size = size

    @property
    def position(self) -> int:
        return self.stream.tell()

    def read(self, size: int) -> bytes:
        if size < 0 or self.position + size > self.size:
            raise NativeGgufError("GGUF value exceeds the artifact")
        value = self.stream.read(size)
        if len(value) != size:
            raise NativeGgufError("GGUF file is truncated")
        return value

    def u32(self) -> int:
        return struct.unpack("<I", self.read(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.read(8))[0]

    def string(self) -> str:
        size = self.u64()
        if size > MAX_STRING_BYTES:
            raise NativeGgufError("GGUF string is too large")
        try:
            return self.read(size).decode("utf-8")
        except UnicodeDecodeError as error:
            raise NativeGgufError("GGUF string is not UTF-8") from error

    def metadata_value(self, value_type: int) -> Any:
        if value_type == 8:
            return self.string()
        if value_type == 9:
            element_type = self.u32()
            if element_type == 9:
                raise NativeGgufError("nested GGUF arrays are unsupported")
            count = self.u64()
            if count > MAX_ARRAY_ITEMS:
                raise NativeGgufError("GGUF array is too large")
            return GgufArray(
                element_type,
                tuple(self.metadata_value(element_type) for _ in range(count)),
            )
        format_value = _SCALAR_FORMATS.get(value_type)
        if format_value is None:
            raise NativeGgufError(
                f"unsupported GGUF metadata type {value_type}"
            )
        value = struct.unpack(format_value, self.read(struct.calcsize(format_value)))[0]
        if value_type == 7:
            if value not in (0, 1):
                raise NativeGgufError("GGUF boolean is invalid")
            return bool(value)
        return value


def _write_metadata_value(output: BinaryIO, value_type: int, value: Any) -> None:
    if value_type == 8:
        if not isinstance(value, str):
            raise NativeGgufError("GGUF string is invalid")
        _write_string(output, value)
        return
    if value_type == 9:
        if not isinstance(value, GgufArray):
            raise NativeGgufError("GGUF array value is invalid")
        output.write(struct.pack("<IQ", value.element_type, len(value.values)))
        for item in value.values:
            _write_metadata_value(output, value.element_type, item)
        return
    format_value = _SCALAR_FORMATS.get(value_type)
    if format_value is None:
        raise NativeGgufError(f"unsupported GGUF metadata type {value_type}")
    if value_type == 7:
        value = int(bool(value))
    output.write(struct.pack(format_value, value))


def _write_string(output: BinaryIO, value: str) -> None:
    encoded = value.encode("utf-8")
    output.write(struct.pack("<Q", len(encoded)))
    output.write(encoded)


def _copy_exact(source: BinaryIO, destination: BinaryIO, count: int) -> None:
    remaining = count
    while remaining:
        block = source.read(min(8 * 1024 * 1024, remaining))
        if not block:
            raise NativeGgufError("GGUF tensor payload is truncated")
        destination.write(block)
        remaining -= len(block)


def _pad_to_alignment(output: BinaryIO, alignment: int) -> None:
    padding = _align(output.tell(), alignment) - output.tell()
    if padding:
        output.write(b"\0" * padding)


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _validate_range(layer_start: int, layer_end: int, total_layers: int) -> None:
    for value, name in (
        (layer_start, "layer_start"),
        (layer_end, "layer_end"),
        (total_layers, "total_layers"),
    ):
        _integer(value, name)
    if not 0 <= layer_start < layer_end <= total_layers:
        raise NativeGgufError("native GGUF layer range is invalid")


def _validate_complete_ranges(
    ranges: Sequence[tuple[int, int]],
    total_layers: int,
) -> None:
    if not ranges:
        raise NativeGgufError("native GGUF fleet ranges are empty")
    cursor = 0
    for value in ranges:
        if (
            not isinstance(value, tuple)
            or len(value) != 2
            or any(
                not isinstance(item, int) or isinstance(item, bool)
                for item in value
            )
        ):
            raise NativeGgufError("native GGUF fleet range is invalid")
        layer_start, layer_end = value
        _validate_range(layer_start, layer_end, total_layers)
        if layer_start != cursor:
            raise NativeGgufError(
                "native GGUF fleet ranges must be ordered, contiguous and complete"
            )
        cursor = layer_end
    if cursor != total_layers:
        raise NativeGgufError(
            "native GGUF fleet ranges must be ordered, contiguous and complete"
        )


def _file_record(path: Path) -> dict[str, Any]:
    return {"sizeBytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _verify_file_record(
    root: Path, name: str, files: Mapping[str, Any]
) -> dict[str, Any]:
    record = _exact_mapping(files.get(name), ("sizeBytes", "sha256"), name)
    expected_size = _positive_integer(record.get("sizeBytes"), f"{name} size")
    expected_digest = _digest(record.get("sha256"), f"{name} SHA-256")
    path = root / name
    if not path.is_file() or path.stat().st_size != expected_size:
        raise NativeGgufError(f"native GGUF package file {name!r} size differs")
    if _sha256_file(path) != expected_digest:
        raise NativeGgufError(f"native GGUF package file {name!r} digest differs")
    return {"sizeBytes": expected_size, "sha256": expected_digest}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.part-{uuid.uuid4().hex}")
    payload = _canonical_json(value) + b"\n"
    try:
        with temporary.open("xb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _native_model_identity(source_digest: str, config_digest: str) -> str:
    return "sha256:" + hashlib.sha256(
        _canonical_json(
            {
                "schema": "gdlp-native-gguf-model-identity/1",
                "sourceGgufSha256": source_digest,
                "configSha256": config_digest,
            }
        )
    ).hexdigest()


def _exact_mapping(
    value: object, keys: Sequence[str], name: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(keys):
        raise NativeGgufError(f"native GGUF {name} has unknown or missing fields")
    return value


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise NativeGgufError(f"{name} must be non-empty")
    return value


def _integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise NativeGgufError(f"{name} must be an integer")
    return value


def _positive_integer(value: object, name: str) -> int:
    result = _integer(value, name)
    if result < 1:
        raise NativeGgufError(f"{name} must be positive")
    return result


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise NativeGgufError(f"{name} must be a lowercase SHA-256 digest")
    return value


__all__ = [
    "GgufArray",
    "GgufDocument",
    "GgufMetadata",
    "GgufTensor",
    "NATIVE_GGUF_STAGE_CONFIG",
    "NATIVE_GGUF_FLEET_MANIFEST",
    "NATIVE_GGUF_FLEET_SCHEMA",
    "NATIVE_GGUF_STAGE_MANIFEST",
    "NATIVE_GGUF_STAGE_SCHEMA",
    "NATIVE_GGUF_STAGE_WEIGHTS",
    "NativeGgufError",
    "NativeGgufFleet",
    "NativeGgufStagePackage",
    "build_native_gguf_fleet",
    "build_native_gguf_stage",
    "dequantize_gguf_tensor",
    "materialize_native_gguf_stage",
    "parse_gguf",
    "verify_native_gguf_fleet",
    "verify_native_gguf_stage",
]
