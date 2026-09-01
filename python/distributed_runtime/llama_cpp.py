from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import struct
import subprocess
from typing import Any, Callable, Mapping, Sequence

from .compatibility import ExecutorCompatibilityKey, compatibility_key_for_manifest
from .executor_abi import StageExecutorManifest, parse_stage_executor_manifest


LLAMA_CPP_DEPLOYMENT_SCHEMA = "gdlp-llama-cpp-deployment/1"
LLAMA_LAYER_PACKAGE_SCHEMA = "gdlp-llama-layer-package/1"
LLAMA_STAGE_ADAPTER = "gdlp-llama-layer-stage/1"
LLAMA_LAYER_ARTIFACT_FORMAT = "gdlp-layer-gguf"
GGUF_SPLIT_ROLE = "complete-model-storage-only"

_VERSION_RE = re.compile(
    r"(?m)^version:\s*(?P<build>[0-9]+)\s*\((?P<commit>[0-9a-fA-F]+)\)\s*$"
)
_COMPILER_RE = re.compile(r"(?m)^built with\s+(?P<compiler>.+?)\s*$")
_DEVICE_RE = re.compile(
    r"(?m)^\s*(?P<id>[A-Za-z][A-Za-z0-9_-]*[0-9]+):\s*"
    r"(?P<name>.+?)\s*\((?P<total>[0-9]+) MiB,\s*"
    r"(?P<free>[0-9]+) MiB free\)\s*$"
)
_BACKEND_DEVICE_RE = re.compile(
    r"(?m)^ggml_(?P<backend>[a-zA-Z0-9_]+):\s*"
    r"(?P<index>[0-9]+)\s*=\s*(?P<name>.+?)(?:\s*\|\s*(?P<properties>.+))?$"
)
_SPLIT_FILE_RE = re.compile(
    r"^(?P<prefix>.+)-(?P<part>[0-9]{5})-of-(?P<count>[0-9]{5})\.gguf$",
    re.IGNORECASE,
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_GGUF_VALUE_SIZES = {
    0: 1,   # uint8
    1: 1,   # int8
    2: 2,   # uint16
    3: 2,   # int16
    4: 4,   # uint32
    5: 4,   # int32
    6: 4,   # float32
    7: 1,   # bool
    10: 8,  # uint64
    11: 8,  # int64
    12: 8,  # float64
}
_GGUF_SCALAR_FORMATS = {
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
_FILE_TYPE_NAMES = {
    0: "f32",
    1: "f16",
    2: "q4_0",
    3: "q4_1",
    7: "q8_0",
    8: "q5_0",
    9: "q5_1",
    10: "q2_k",
    11: "q3_k_s",
    12: "q3_k_m",
    13: "q3_k_l",
    14: "q4_k_s",
    15: "q4_k_m",
    16: "q5_k_s",
    17: "q5_k_m",
    18: "q6_k",
}


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


CommandRunner = Callable[[Sequence[str], Path, float], CommandResult]


@dataclass(frozen=True)
class LlamaCppDevice:
    identifier: str
    backend: str
    name: str
    total_memory_mib: int
    free_memory_mib: int
    properties: tuple[tuple[str, bool | int | str], ...] = ()

    def to_document(self) -> dict[str, Any]:
        return {
            "id": self.identifier,
            "backend": self.backend,
            "name": self.name,
            "totalMemoryMiB": self.total_memory_mib,
            "freeMemoryMiB": self.free_memory_mib,
            "properties": {key: value for key, value in self.properties},
        }


@dataclass(frozen=True)
class LlamaCppRuntimeProbe:
    runtime_id: str
    build_number: int
    build_commit: str
    compiler: str
    binary_set_sha256: str
    devices: tuple[LlamaCppDevice, ...]
    rpc_backend_present: bool
    native_partial_stage_abi: bool = False

    @property
    def engine_version(self) -> str:
        return f"b{self.build_number}-{self.build_commit}"

    def to_document(self) -> dict[str, Any]:
        return {
            "runtimeId": self.runtime_id,
            "id": "llama.cpp",
            "version": self.engine_version,
            "buildNumber": self.build_number,
            "buildCommit": self.build_commit,
            "compiler": self.compiler,
            "binarySetSha256": self.binary_set_sha256,
            "capabilities": {
                "wholeModel": True,
                "rpcTensorOffload": self.rpc_backend_present,
                "nativePartialStageAbi": self.native_partial_stage_abi,
                "ordinaryGgufSplitRole": GGUF_SPLIT_ROLE,
            },
        }


@dataclass(frozen=True)
class GGUFArtifact:
    file_name: str
    identity: str
    file_sha256: str
    metadata_sha256: str
    size_bytes: int
    gguf_version: int
    tensor_count: int
    metadata_count: int
    architecture: str
    name: str | None
    file_type: int | None
    quantization: str
    block_count: int
    embedding_length: int
    context_length: int | None
    storage_layout: str
    split_part: int | None = None
    split_count: int | None = None

    @property
    def execution_scope(self) -> str:
        if self.storage_layout == "single-file":
            return "complete-model"
        return "complete-model-storage-shard"

    def to_document(self) -> dict[str, Any]:
        return {
            "kind": "gguf",
            "identity": self.identity,
            "fileName": self.file_name,
            "fileSha256": self.file_sha256,
            "metadataSha256": self.metadata_sha256,
            "sizeBytes": self.size_bytes,
            "gguf": {
                "version": self.gguf_version,
                "tensorCount": self.tensor_count,
                "metadataCount": self.metadata_count,
                "architecture": self.architecture,
                "name": self.name,
                "fileType": self.file_type,
                "quantization": self.quantization,
                "blockCount": self.block_count,
                "embeddingLength": self.embedding_length,
                "contextLength": self.context_length,
            },
            "storage": {
                "layout": self.storage_layout,
                "splitPart": self.split_part,
                "splitCount": self.split_count,
                "executionScope": self.execution_scope,
                "ordinaryGgufSplitIsLayerStage": False,
            },
        }


@dataclass(frozen=True)
class LlamaBenchmarkEvidence:
    benchmark_id: str
    runtime_id: str
    artifact_identity: str
    parameters_json: str
    records_json: tuple[str, ...]

    @property
    def parameters(self) -> dict[str, Any]:
        return json.loads(self.parameters_json)

    @property
    def records(self) -> tuple[dict[str, Any], ...]:
        return tuple(json.loads(value) for value in self.records_json)

    def to_document(self) -> dict[str, Any]:
        return {
            "benchmarkId": self.benchmark_id,
            "runtimeId": self.runtime_id,
            "artifactIdentity": self.artifact_identity,
            "parameters": self.parameters,
            "records": list(self.records),
        }


@dataclass(frozen=True)
class LayerPackageFile:
    path: str
    role: str
    size_bytes: int
    sha256: str

    def to_document(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "role": self.role,
            "sizeBytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class LlamaLayerPackage:
    package_id: str
    model_identity: str
    model_source: str
    model_revision: str | None
    architecture: str
    quantization: str
    layer_start: int
    layer_end: int
    total_layers: int
    hidden_size: int
    weight_dtype: str
    activation_dtype: str
    activation_codecs: tuple[str, ...]
    kv_format: str
    files: tuple[LayerPackageFile, ...]

    def to_document(self) -> dict[str, Any]:
        return {
            "schema": LLAMA_LAYER_PACKAGE_SCHEMA,
            "packageId": self.package_id,
            "packageKind": "executable-contiguous-layer-range",
            "artifactFormat": LLAMA_LAYER_ARTIFACT_FORMAT,
            "model": {
                "identity": self.model_identity,
                "source": self.model_source,
                "revision": self.model_revision,
                "architecture": self.architecture,
                "quantization": self.quantization,
            },
            "stage": {
                "layerStart": self.layer_start,
                "layerEnd": self.layer_end,
                "totalLayers": self.total_layers,
            },
            "tensor": {
                "hiddenSize": self.hidden_size,
                "weightDtype": self.weight_dtype,
                "activationDtype": self.activation_dtype,
                "activationCodecs": list(self.activation_codecs),
                "kvFormat": self.kv_format,
            },
            "files": [value.to_document() for value in self.files],
            "ordinaryGgufSplitAcceptedAsStage": False,
        }


@dataclass(frozen=True)
class LlamaCppDeploymentManifest:
    document_json: str

    @property
    def deployment_id(self) -> str:
        return self.to_document()["deploymentId"]

    @property
    def mode(self) -> str:
        return self.to_document()["execution"]["mode"]

    @property
    def stage_executor(self) -> StageExecutorManifest | None:
        value = self.to_document()["execution"]["stageExecutor"]
        if value is None:
            return None
        return parse_stage_executor_manifest(value)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.document_json.encode("utf-8")).hexdigest()

    def to_document(self) -> dict[str, Any]:
        return json.loads(self.document_json)


def probe_llama_cpp(
    runtime_directory: str | os.PathLike[str],
    *,
    runner: CommandRunner | None = None,
    timeout_seconds: float = 20.0,
) -> LlamaCppRuntimeProbe:
    """Probe a stock llama.cpp build without invoking a command shell.

    A stock build is deliberately advertised as a whole-model engine.  RPC
    devices can offload tensors, but neither RPC nor llama-gguf-split implies
    the GDLP partial Stage Executor ABI.
    """

    root = Path(runtime_directory).resolve()
    cli = _required_executable(root, "llama-cli")
    bench = _required_executable(root, "llama-bench")
    command_runner = runner or _default_command_runner
    version_result = _run_checked(
        command_runner, (str(cli), "--version"), root, timeout_seconds
    )
    version_text = version_result.stdout + "\n" + version_result.stderr
    version_match = _VERSION_RE.search(version_text)
    compiler_match = _COMPILER_RE.search(version_text)
    if version_match is None or compiler_match is None:
        raise ValueError("llama.cpp version output is not recognized")

    device_result = _run_checked(
        command_runner, (str(bench), "--list-devices"), root, timeout_seconds
    )
    devices = _parse_devices(device_result.stdout, device_result.stderr)
    build_number = int(version_match.group("build"))
    if build_number < 1:
        raise ValueError("llama.cpp build number must be positive")
    build_commit = version_match.group("commit").lower()
    compiler = compiler_match.group("compiler").strip()
    binary_set_sha256 = _binary_set_digest(root, (cli, bench))
    rpc_backend_present = (
        _optional_executable(root, "ggml-rpc-server") is not None
        or "loaded RPC backend" in device_result.stderr
    )
    runtime_body = {
        "buildNumber": build_number,
        "buildCommit": build_commit,
        "compiler": compiler,
        "binarySetSha256": binary_set_sha256,
        "nativePartialStageAbi": False,
    }
    runtime_id = hashlib.sha256(_canonical_json(runtime_body)).hexdigest()[:32]
    return LlamaCppRuntimeProbe(
        runtime_id=runtime_id,
        build_number=build_number,
        build_commit=build_commit,
        compiler=compiler,
        binary_set_sha256=binary_set_sha256,
        devices=devices,
        rpc_backend_present=rpc_backend_present,
        native_partial_stage_abi=False,
    )


def inspect_gguf(path_value: str | os.PathLike[str]) -> GGUFArtifact:
    path = Path(path_value).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    size_bytes = path.stat().st_size
    with path.open("rb") as stream:
        reader = _GGUFReader(stream, size_bytes)
        if reader.read_exact(4) != b"GGUF":
            raise ValueError("artifact is not a GGUF file")
        version = reader.read_u32()
        if version not in (2, 3):
            raise ValueError(f"unsupported GGUF version: {version}")
        tensor_count = reader.read_u64()
        metadata_count = reader.read_u64()
        if metadata_count > 1_000_000:
            raise ValueError("GGUF metadata count is unreasonable")
        selected: dict[str, Any] = {}
        for _ in range(metadata_count):
            key = reader.read_string()
            value_type = reader.read_u32()
            keep = _retain_gguf_key(key)
            value = reader.read_value(value_type, keep=keep)
            if keep:
                selected[key] = value
        metadata_sha256 = reader.consumed_sha256

    architecture = _nonempty_string(
        selected.get("general.architecture"), "GGUF general.architecture"
    )
    block_count = _positive_int(
        selected.get(f"{architecture}.block_count"), "GGUF block count"
    )
    embedding_length = _positive_int(
        selected.get(f"{architecture}.embedding_length"),
        "GGUF embedding length",
    )
    context_value = selected.get(f"{architecture}.context_length")
    context_length = (
        None if context_value is None else _positive_int(context_value, "GGUF context length")
    )
    name_value = selected.get("general.name")
    name = None if name_value is None else _nonempty_string(name_value, "GGUF name")
    file_type_value = selected.get("general.file_type")
    file_type = (
        None
        if file_type_value is None
        else _nonnegative_int(file_type_value, "GGUF file type")
    )
    quantization = (
        "unspecified"
        if file_type is None
        else _FILE_TYPE_NAMES.get(file_type, f"gguf-file-type-{file_type}")
    )
    file_sha256 = _sha256_file(path)
    split_match = _SPLIT_FILE_RE.fullmatch(path.name)
    storage_layout = "single-file"
    split_part = None
    split_count = None
    if split_match is not None:
        storage_layout = "gguf-split-shard"
        split_part = int(split_match.group("part"))
        split_count = int(split_match.group("count"))
        if not 1 <= split_part <= split_count:
            raise ValueError("GGUF split file numbering is invalid")
    return GGUFArtifact(
        file_name=path.name,
        identity="sha256:" + file_sha256,
        file_sha256=file_sha256,
        metadata_sha256=metadata_sha256,
        size_bytes=size_bytes,
        gguf_version=version,
        tensor_count=tensor_count,
        metadata_count=metadata_count,
        architecture=architecture,
        name=name,
        file_type=file_type,
        quantization=quantization,
        block_count=block_count,
        embedding_length=embedding_length,
        context_length=context_length,
        storage_layout=storage_layout,
        split_part=split_part,
        split_count=split_count,
    )


def benchmark_llama_cpp(
    runtime_directory: str | os.PathLike[str],
    runtime: LlamaCppRuntimeProbe,
    artifact: GGUFArtifact,
    artifact_path: str | os.PathLike[str],
    *,
    prompt_tokens: int = 128,
    generation_tokens: int = 32,
    repetitions: int = 1,
    gpu_layers: int = -1,
    device: str | None = None,
    no_warmup: bool = False,
    runner: CommandRunner | None = None,
    timeout_seconds: float = 300.0,
) -> LlamaBenchmarkEvidence:
    root = Path(runtime_directory).resolve()
    bench = _required_executable(root, "llama-bench")
    model_path = Path(artifact_path).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if _sha256_file(model_path) != artifact.file_sha256:
        raise ValueError("benchmark GGUF does not match the inspected artifact identity")
    prompt_tokens = _positive_int(prompt_tokens, "prompt tokens")
    generation_tokens = _positive_int(generation_tokens, "generation tokens")
    repetitions = _positive_int(repetitions, "repetitions")
    if not isinstance(gpu_layers, int) or isinstance(gpu_layers, bool) or gpu_layers < -1:
        raise ValueError("gpu layers must be -1 or a non-negative integer")
    if device is not None:
        device = _nonempty_string(device, "device")
        if device not in {value.identifier for value in runtime.devices}:
            raise ValueError("benchmark device was not reported by the runtime probe")
    arguments = [
        str(bench),
        "-m",
        str(model_path),
        "-p",
        str(prompt_tokens),
        "-n",
        str(generation_tokens),
        "-r",
        str(repetitions),
        "-ngl",
        str(gpu_layers),
        "-o",
        "json",
    ]
    if no_warmup:
        arguments.append("--no-warmup")
    if device is not None:
        arguments.extend(("--device", device))
    result = _run_checked(
        runner or _default_command_runner,
        tuple(arguments),
        root,
        timeout_seconds,
    )
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ValueError("llama-bench stdout is not valid JSON") from error
    if not isinstance(value, list) or len(value) < 1:
        raise ValueError("llama-bench JSON must contain at least one record")
    normalized_records = tuple(
        _normalize_benchmark_record(record, runtime, artifact) for record in value
    )
    if not any(
        record.get("n_prompt") == prompt_tokens and record.get("n_gen") == 0
        for record in normalized_records
    ):
        raise ValueError("llama-bench JSON omits the requested prompt benchmark")
    if not any(
        record.get("n_gen") == generation_tokens and record.get("n_prompt") == 0
        for record in normalized_records
    ):
        raise ValueError("llama-bench JSON omits the requested generation benchmark")
    parameters = {
        "promptTokens": prompt_tokens,
        "generationTokens": generation_tokens,
        "repetitions": repetitions,
        "gpuLayers": gpu_layers,
        "device": device,
        "warmup": not no_warmup,
        "output": "json",
    }
    body = {
        "runtimeId": runtime.runtime_id,
        "artifactIdentity": artifact.identity,
        "parameters": parameters,
        "records": normalized_records,
    }
    benchmark_id = hashlib.sha256(_canonical_json(body)).hexdigest()[:32]
    return LlamaBenchmarkEvidence(
        benchmark_id=benchmark_id,
        runtime_id=runtime.runtime_id,
        artifact_identity=artifact.identity,
        parameters_json=_canonical_json(parameters).decode("utf-8"),
        records_json=tuple(
            _canonical_json(record).decode("utf-8") for record in normalized_records
        ),
    )


def build_layer_package(
    *,
    model_identity: str,
    model_source: str,
    model_revision: str | None,
    architecture: str,
    quantization: str,
    layer_start: int,
    layer_end: int,
    total_layers: int,
    hidden_size: int,
    weight_dtype: str,
    activation_dtype: str,
    activation_codecs: Sequence[str],
    kv_format: str,
    files: Sequence[LayerPackageFile | Mapping[str, Any]],
) -> LlamaLayerPackage:
    body = _layer_package_body(
        model_identity=model_identity,
        model_source=model_source,
        model_revision=model_revision,
        architecture=architecture,
        quantization=quantization,
        layer_start=layer_start,
        layer_end=layer_end,
        total_layers=total_layers,
        hidden_size=hidden_size,
        weight_dtype=weight_dtype,
        activation_dtype=activation_dtype,
        activation_codecs=activation_codecs,
        kv_format=kv_format,
        files=files,
    )
    package_id = hashlib.sha256(_canonical_json(body)).hexdigest()[:32]
    return parse_layer_package_manifest({**body, "packageId": package_id})


def parse_layer_package_manifest(value: object) -> LlamaLayerPackage:
    document = _exact_mapping(
        value,
        (
            "schema",
            "packageId",
            "packageKind",
            "artifactFormat",
            "model",
            "stage",
            "tensor",
            "files",
            "ordinaryGgufSplitAcceptedAsStage",
        ),
        "llama layer package",
    )
    if document.get("schema") != LLAMA_LAYER_PACKAGE_SCHEMA:
        raise ValueError("unsupported llama layer package schema")
    if document.get("packageKind") != "executable-contiguous-layer-range":
        raise ValueError("llama layer package is not an executable layer range")
    if document.get("artifactFormat") != LLAMA_LAYER_ARTIFACT_FORMAT:
        raise ValueError("llama layer package artifact format is unsupported")
    if document.get("ordinaryGgufSplitAcceptedAsStage") is not False:
        raise ValueError("ordinary gguf-split shards are not executable layer stages")
    model = _exact_mapping(
        document.get("model"),
        ("identity", "source", "revision", "architecture", "quantization"),
        "llama layer package model",
    )
    stage = _exact_mapping(
        document.get("stage"),
        ("layerStart", "layerEnd", "totalLayers"),
        "llama layer package stage",
    )
    tensor = _exact_mapping(
        document.get("tensor"),
        ("hiddenSize", "weightDtype", "activationDtype", "activationCodecs", "kvFormat"),
        "llama layer package tensor",
    )
    layer_start = _nonnegative_int(stage.get("layerStart"), "layer start")
    layer_end = _positive_int(stage.get("layerEnd"), "layer end")
    total_layers = _positive_int(stage.get("totalLayers"), "total layers")
    if not layer_start < layer_end <= total_layers:
        raise ValueError("llama layer package range is invalid")
    files_value = document.get("files")
    if not isinstance(files_value, Sequence) or isinstance(
        files_value, (str, bytes, bytearray)
    ) or len(files_value) < 1:
        raise ValueError("llama layer package must contain files")
    files = tuple(_parse_layer_package_file(item) for item in files_value)
    if len({item.path for item in files}) != len(files):
        raise ValueError("llama layer package file paths must be unique")
    if not any(item.role == "layer-weights" for item in files):
        raise ValueError("llama layer package must contain layer weights")
    if layer_start > 0 and any(item.role == "input-embedding" for item in files):
        raise ValueError("non-first layer package cannot contain input embeddings")
    if layer_end < total_layers and any(item.role == "output-head" for item in files):
        raise ValueError("non-last layer package cannot contain an output head")
    activation_codecs = _string_tuple(
        tensor.get("activationCodecs"), "activation codecs"
    )
    package = LlamaLayerPackage(
        package_id=_hex_digest(document.get("packageId"), 32, "package id"),
        model_identity=_model_identity(model.get("identity")),
        model_source=_nonempty_string(model.get("source"), "model source"),
        model_revision=_nullable_string(model.get("revision"), "model revision"),
        architecture=_nonempty_string(model.get("architecture"), "model architecture"),
        quantization=_nonempty_string(model.get("quantization"), "quantization"),
        layer_start=layer_start,
        layer_end=layer_end,
        total_layers=total_layers,
        hidden_size=_positive_int(tensor.get("hiddenSize"), "hidden size"),
        weight_dtype=_nonempty_string(tensor.get("weightDtype"), "weight dtype"),
        activation_dtype=_nonempty_string(
            tensor.get("activationDtype"), "activation dtype"
        ),
        activation_codecs=activation_codecs,
        kv_format=_nonempty_string(tensor.get("kvFormat"), "KV format"),
        files=files,
    )
    expected_id = hashlib.sha256(
        _canonical_json(_document_without(document, "packageId"))
    ).hexdigest()[:32]
    if package.package_id != expected_id:
        raise ValueError("llama layer package identity does not match its contract")
    return package


def verify_layer_package_files(
    package_value: LlamaLayerPackage | Mapping[str, Any],
    base_directory: str | os.PathLike[str],
) -> LlamaLayerPackage:
    package = (
        package_value
        if isinstance(package_value, LlamaLayerPackage)
        else parse_layer_package_manifest(package_value)
    )
    root = Path(base_directory).resolve()
    for expected in package.files:
        actual = root.joinpath(*PurePosixPath(expected.path).parts).resolve()
        if not actual.is_file():
            raise FileNotFoundError(actual)
        if actual.stat().st_size != expected.size_bytes:
            raise ValueError(f"layer package size mismatch: {expected.path}")
        if _sha256_file(actual) != expected.sha256:
            raise ValueError(f"layer package digest mismatch: {expected.path}")
    return package


def build_whole_model_deployment(
    runtime: LlamaCppRuntimeProbe,
    artifact: GGUFArtifact,
    *,
    benchmark: LlamaBenchmarkEvidence | None = None,
) -> LlamaCppDeploymentManifest:
    if artifact.storage_layout != "single-file":
        raise ValueError(
            "an individual gguf-split file is a whole-model storage shard, not "
            "a complete deployment or an executable layer stage"
        )
    if benchmark is not None:
        _validate_benchmark_binding(benchmark, runtime, artifact.identity)
    execution = {
        "mode": "whole-model",
        "interface": "llama.cpp-whole-model",
        "requiresExternalStageAdapter": False,
        "stageExecutor": None,
        "ordinaryGgufSplitRole": GGUF_SPLIT_ROLE,
    }
    return _build_deployment_document(
        runtime,
        artifact.to_document(),
        execution,
        benchmark,
    )


def build_partial_stage_deployment(
    runtime: LlamaCppRuntimeProbe,
    package_value: LlamaLayerPackage | Mapping[str, Any],
    stage_executor_value: StageExecutorManifest | Mapping[str, Any],
) -> LlamaCppDeploymentManifest:
    """Bind an explicit layer package to an actual external Stage ABI adapter.

    This function never derives a stage from a normal GGUF or gguf-split file.
    The stock llama.cpp binaries probed above do not implement the ABI; an
    executor manifest from a real adapter is mandatory.
    """

    package = (
        package_value
        if isinstance(package_value, LlamaLayerPackage)
        else parse_layer_package_manifest(package_value)
    )
    executor = (
        stage_executor_value
        if isinstance(stage_executor_value, StageExecutorManifest)
        else parse_stage_executor_manifest(stage_executor_value)
    )
    _validate_partial_binding(runtime, package, executor)
    execution = {
        "mode": "partial-stage",
        "interface": "gdlp-stage-executor/1",
        "requiresExternalStageAdapter": True,
        "stageExecutor": executor.to_document(),
        "ordinaryGgufSplitRole": GGUF_SPLIT_ROLE,
    }
    return _build_deployment_document(
        runtime,
        {"kind": "layer-package", "manifest": package.to_document()},
        execution,
        None,
    )


def parse_llama_cpp_deployment(value: object) -> LlamaCppDeploymentManifest:
    document = _exact_mapping(
        value,
        (
            "schema",
            "deploymentId",
            "engine",
            "artifact",
            "execution",
            "devices",
            "benchmark",
        ),
        "llama.cpp deployment",
    )
    if document.get("schema") != LLAMA_CPP_DEPLOYMENT_SCHEMA:
        raise ValueError("unsupported llama.cpp deployment schema")
    deployment_id = _hex_digest(document.get("deploymentId"), 32, "deployment id")
    expected_id = hashlib.sha256(
        _canonical_json(_document_without(document, "deploymentId"))
    ).hexdigest()[:32]
    if deployment_id != expected_id:
        raise ValueError("llama.cpp deployment identity does not match its contract")
    engine = _parse_runtime_document(document.get("engine"))
    devices = _parse_device_documents(document.get("devices"))
    runtime = LlamaCppRuntimeProbe(
        runtime_id=engine["runtimeId"],
        build_number=engine["buildNumber"],
        build_commit=engine["buildCommit"],
        compiler=engine["compiler"],
        binary_set_sha256=engine["binarySetSha256"],
        devices=devices,
        rpc_backend_present=engine["rpcBackendPresent"],
        native_partial_stage_abi=engine["nativePartialStageAbi"],
    )
    execution = _exact_mapping(
        document.get("execution"),
        (
            "mode",
            "interface",
            "requiresExternalStageAdapter",
            "stageExecutor",
            "ordinaryGgufSplitRole",
        ),
        "llama.cpp execution",
    )
    if execution.get("ordinaryGgufSplitRole") != GGUF_SPLIT_ROLE:
        raise ValueError("ordinary gguf-split role is invalid")
    artifact = _mapping(document.get("artifact"), "llama.cpp artifact")
    mode = execution.get("mode")
    if mode == "whole-model":
        if execution.get("interface") != "llama.cpp-whole-model":
            raise ValueError("whole-model interface is invalid")
        if execution.get("requiresExternalStageAdapter") is not False:
            raise ValueError("whole-model deployment cannot require a Stage adapter")
        if execution.get("stageExecutor") is not None:
            raise ValueError("whole-model deployment cannot advertise a Stage ABI")
        parsed_artifact = _parse_gguf_artifact_document(artifact)
        benchmark_value = document.get("benchmark")
        if benchmark_value is not None:
            _parse_benchmark_document(
                benchmark_value,
                runtime,
                str(parsed_artifact["identity"]),
            )
    elif mode == "partial-stage":
        if execution.get("interface") != "gdlp-stage-executor/1":
            raise ValueError("partial-stage interface is invalid")
        if execution.get("requiresExternalStageAdapter") is not True:
            raise ValueError("partial-stage deployment requires an external adapter")
        if set(artifact) != {"kind", "manifest"} or artifact.get("kind") != "layer-package":
            raise ValueError("partial-stage deployment requires a layer package")
        package = parse_layer_package_manifest(artifact.get("manifest"))
        executor = parse_stage_executor_manifest(execution.get("stageExecutor"))
        _validate_partial_binding(runtime, package, executor)
        if document.get("benchmark") is not None:
            raise ValueError("whole-model benchmark cannot certify a partial stage")
    else:
        raise ValueError("llama.cpp deployment mode is unsupported")
    return LlamaCppDeploymentManifest(
        _canonical_json(document).decode("utf-8")
    )


def compatibility_key_for_llama_partial(
    deployment_value: LlamaCppDeploymentManifest | Mapping[str, Any],
    *,
    device_kind: str,
    compute_api: str,
    weight_dtype: str,
    activation_codec: str,
    parallelism_mode: str = "pipeline-stage",
    world_size: int = 1,
    partitioning: str = "contiguous-layer-range/1",
) -> ExecutorCompatibilityKey:
    deployment = (
        deployment_value
        if isinstance(deployment_value, LlamaCppDeploymentManifest)
        else parse_llama_cpp_deployment(deployment_value)
    )
    executor = deployment.stage_executor
    if deployment.mode != "partial-stage" or executor is None:
        raise ValueError(
            "whole-model llama.cpp deployments do not fit the Stage Executor registry"
        )
    package = parse_layer_package_manifest(
        deployment.to_document()["artifact"]["manifest"]
    )
    return compatibility_key_for_manifest(
        executor,
        model_architecture=package.architecture,
        quantization=package.quantization,
        device_kind=device_kind,
        compute_api=compute_api,
        weight_dtype=weight_dtype,
        activation_codec=activation_codec,
        parallelism_mode=parallelism_mode,
        world_size=world_size,
        partitioning=partitioning,
    )


def _default_command_runner(
    arguments: Sequence[str], cwd: Path, timeout_seconds: float
) -> CommandResult:
    completed = subprocess.run(
        [str(value) for value in arguments],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_seconds,
        check=False,
        shell=False,
    )
    return CommandResult(completed.returncode, completed.stdout, completed.stderr)


def _run_checked(
    runner: CommandRunner,
    arguments: Sequence[str],
    cwd: Path,
    timeout_seconds: float,
) -> CommandResult:
    if not isinstance(timeout_seconds, (int, float)) or isinstance(timeout_seconds, bool):
        raise ValueError("command timeout must be numeric")
    if timeout_seconds <= 0:
        raise ValueError("command timeout must be positive")
    result = runner(tuple(arguments), cwd, float(timeout_seconds))
    if not isinstance(result, CommandResult):
        raise TypeError("command runner must return CommandResult")
    if result.returncode != 0:
        output = (result.stderr or result.stdout).strip()
        if len(output) > 1000:
            output = output[-1000:]
        raise RuntimeError(
            f"llama.cpp command failed with exit code {result.returncode}: {output}"
        )
    return result


def _parse_devices(stdout: str, stderr: str) -> tuple[LlamaCppDevice, ...]:
    properties_by_key: dict[tuple[str, int], tuple[tuple[str, Any], ...]] = {}
    names_by_key: dict[tuple[str, int], str] = {}
    for match in _BACKEND_DEVICE_RE.finditer(stderr):
        backend = match.group("backend").lower()
        index = int(match.group("index"))
        names_by_key[(backend, index)] = match.group("name").strip()
        pairs: list[tuple[str, Any]] = []
        for raw in (match.group("properties") or "").split("|"):
            if not raw.strip() or ":" not in raw:
                continue
            key, raw_value = raw.split(":", 1)
            pairs.append((key.strip(), _coerce_property(raw_value.strip())))
        properties_by_key[(backend, index)] = tuple(sorted(pairs))
    devices: list[LlamaCppDevice] = []
    for match in _DEVICE_RE.finditer(stdout):
        identifier = match.group("id")
        prefix_match = re.match(r"([A-Za-z_]+)([0-9]+)$", identifier)
        if prefix_match is None:
            raise ValueError("llama.cpp device identifier is not recognized")
        backend = prefix_match.group(1).lower()
        index = int(prefix_match.group(2))
        total = int(match.group("total"))
        free = int(match.group("free"))
        if total < 1 or free < 0 or free > total:
            raise ValueError("llama.cpp device memory report is invalid")
        public_name = match.group("name").strip()
        diagnostic_name = names_by_key.get((backend, index))
        if diagnostic_name is not None and not (
            diagnostic_name.startswith(public_name)
            or public_name.startswith(diagnostic_name)
        ):
            raise ValueError("llama.cpp device listings disagree")
        devices.append(
            LlamaCppDevice(
                identifier=identifier,
                backend=backend,
                name=public_name,
                total_memory_mib=total,
                free_memory_mib=free,
                properties=properties_by_key.get((backend, index), ()),
            )
        )
    if len({item.identifier for item in devices}) != len(devices):
        raise ValueError("llama.cpp reported duplicate devices")
    return tuple(devices)


def _normalize_benchmark_record(
    value: object,
    runtime: LlamaCppRuntimeProbe,
    artifact: GGUFArtifact,
) -> dict[str, Any]:
    record = _mapping(value, "llama-bench record")
    build_number = _positive_int(record.get("build_number"), "benchmark build number")
    build_commit = _nonempty_string(
        record.get("build_commit"), "benchmark build commit"
    ).lower()
    if build_number != runtime.build_number or build_commit != runtime.build_commit:
        raise ValueError("llama-bench build does not match the runtime probe")
    model_size = _positive_int(record.get("model_size"), "benchmark model size")
    if model_size > artifact.size_bytes:
        raise ValueError("llama-bench model size exceeds the GGUF artifact")
    _positive_int(record.get("model_n_params"), "benchmark model parameters")
    _nonnegative_int(record.get("n_prompt"), "benchmark prompt tokens")
    _nonnegative_int(record.get("n_gen"), "benchmark generated tokens")
    if record.get("n_prompt") == 0 and record.get("n_gen") == 0:
        raise ValueError("benchmark record must measure prompt or generation tokens")
    avg_ns = record.get("avg_ns")
    avg_ts = record.get("avg_ts")
    if not isinstance(avg_ns, (int, float)) or isinstance(avg_ns, bool) or avg_ns <= 0:
        raise ValueError("benchmark average nanoseconds must be positive")
    if not isinstance(avg_ts, (int, float)) or isinstance(avg_ts, bool) or avg_ts <= 0:
        raise ValueError("benchmark token rate must be positive")
    samples_ns = record.get("samples_ns")
    samples_ts = record.get("samples_ts")
    if not isinstance(samples_ns, list) or not samples_ns:
        raise ValueError("benchmark nanosecond samples are missing")
    if not isinstance(samples_ts, list) or len(samples_ts) != len(samples_ns):
        raise ValueError("benchmark token-rate samples do not match")
    if any(
        not isinstance(item, (int, float)) or isinstance(item, bool) or item <= 0
        for item in (*samples_ns, *samples_ts)
    ):
        raise ValueError("benchmark samples must be positive numbers")
    normalized = json.loads(_canonical_json(record))
    normalized.pop("model_filename", None)
    normalized["artifact_identity"] = artifact.identity
    normalized["artifact_file_name"] = artifact.file_name
    return normalized


def _build_deployment_document(
    runtime: LlamaCppRuntimeProbe,
    artifact: Mapping[str, Any],
    execution: Mapping[str, Any],
    benchmark: LlamaBenchmarkEvidence | None,
) -> LlamaCppDeploymentManifest:
    without_id = {
        "schema": LLAMA_CPP_DEPLOYMENT_SCHEMA,
        "engine": runtime.to_document(),
        "artifact": artifact,
        "execution": execution,
        "devices": [device.to_document() for device in runtime.devices],
        "benchmark": None if benchmark is None else benchmark.to_document(),
    }
    deployment_id = hashlib.sha256(_canonical_json(without_id)).hexdigest()[:32]
    return parse_llama_cpp_deployment(
        {**without_id, "deploymentId": deployment_id}
    )


def _parse_runtime_document(value: object) -> dict[str, Any]:
    engine = _exact_mapping(
        value,
        (
            "runtimeId",
            "id",
            "version",
            "buildNumber",
            "buildCommit",
            "compiler",
            "binarySetSha256",
            "capabilities",
        ),
        "llama.cpp engine",
    )
    if engine.get("id") != "llama.cpp":
        raise ValueError("deployment engine is not llama.cpp")
    build_number = _positive_int(engine.get("buildNumber"), "engine build number")
    build_commit = _hex_text(engine.get("buildCommit"), "engine build commit")
    version = _nonempty_string(engine.get("version"), "engine version")
    if version != f"b{build_number}-{build_commit}":
        raise ValueError("llama.cpp engine version does not match its build")
    capabilities = _exact_mapping(
        engine.get("capabilities"),
        (
            "wholeModel",
            "rpcTensorOffload",
            "nativePartialStageAbi",
            "ordinaryGgufSplitRole",
        ),
        "llama.cpp capabilities",
    )
    if capabilities.get("wholeModel") is not True:
        raise ValueError("llama.cpp whole-model capability is required")
    if capabilities.get("ordinaryGgufSplitRole") != GGUF_SPLIT_ROLE:
        raise ValueError("llama.cpp gguf-split capability is invalid")
    if capabilities.get("nativePartialStageAbi") is not False:
        raise ValueError("stock llama.cpp must not advertise a native partial Stage ABI")
    if not isinstance(capabilities.get("rpcTensorOffload"), bool):
        raise ValueError("llama.cpp RPC capability must be boolean")
    runtime_id = _hex_digest(engine.get("runtimeId"), 32, "runtime id")
    binary_digest = _hex_digest(
        engine.get("binarySetSha256"), 64, "binary-set SHA-256"
    )
    compiler = _nonempty_string(engine.get("compiler"), "engine compiler")
    expected_runtime_id = hashlib.sha256(
        _canonical_json(
            {
                "buildNumber": build_number,
                "buildCommit": build_commit,
                "compiler": compiler,
                "binarySetSha256": binary_digest,
                "nativePartialStageAbi": False,
            }
        )
    ).hexdigest()[:32]
    if runtime_id != expected_runtime_id:
        raise ValueError("llama.cpp runtime identity does not match its build")
    return {
        "runtimeId": runtime_id,
        "buildNumber": build_number,
        "buildCommit": build_commit,
        "compiler": compiler,
        "binarySetSha256": binary_digest,
        "rpcBackendPresent": capabilities.get("rpcTensorOffload"),
        "nativePartialStageAbi": False,
    }


def _parse_device_documents(value: object) -> tuple[LlamaCppDevice, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("llama.cpp devices must be a list")
    devices: list[LlamaCppDevice] = []
    for item in value:
        document = _exact_mapping(
            item,
            ("id", "backend", "name", "totalMemoryMiB", "freeMemoryMiB", "properties"),
            "llama.cpp device",
        )
        properties = _mapping(document.get("properties"), "llama.cpp device properties")
        parsed_properties: list[tuple[str, bool | int | str]] = []
        for key, property_value in properties.items():
            property_key = _nonempty_string(key, "device property key")
            if not isinstance(property_value, (bool, int, str)):
                raise ValueError("llama.cpp device property has unsupported type")
            parsed_properties.append((property_key, property_value))
        total = _positive_int(document.get("totalMemoryMiB"), "device total memory")
        free = _nonnegative_int(document.get("freeMemoryMiB"), "device free memory")
        if free > total:
            raise ValueError("device free memory exceeds total memory")
        devices.append(
            LlamaCppDevice(
                identifier=_nonempty_string(document.get("id"), "device id"),
                backend=_nonempty_string(document.get("backend"), "device backend"),
                name=_nonempty_string(document.get("name"), "device name"),
                total_memory_mib=total,
                free_memory_mib=free,
                properties=tuple(sorted(parsed_properties)),
            )
        )
    return tuple(devices)


def _parse_gguf_artifact_document(value: object) -> Mapping[str, Any]:
    document = _exact_mapping(
        value,
        (
            "kind",
            "identity",
            "fileName",
            "fileSha256",
            "metadataSha256",
            "sizeBytes",
            "gguf",
            "storage",
        ),
        "GGUF artifact",
    )
    if document.get("kind") != "gguf":
        raise ValueError("whole-model artifact must be GGUF")
    digest = _hex_digest(document.get("fileSha256"), 64, "GGUF SHA-256")
    if document.get("identity") != "sha256:" + digest:
        raise ValueError("GGUF identity does not match its digest")
    _hex_digest(document.get("metadataSha256"), 64, "GGUF metadata SHA-256")
    _positive_int(document.get("sizeBytes"), "GGUF size")
    gguf = _exact_mapping(
        document.get("gguf"),
        (
            "version",
            "tensorCount",
            "metadataCount",
            "architecture",
            "name",
            "fileType",
            "quantization",
            "blockCount",
            "embeddingLength",
            "contextLength",
        ),
        "GGUF metadata",
    )
    if gguf.get("version") not in (2, 3):
        raise ValueError("GGUF version is unsupported")
    _nonnegative_int(gguf.get("tensorCount"), "GGUF tensor count")
    _nonnegative_int(gguf.get("metadataCount"), "GGUF metadata count")
    _nonempty_string(gguf.get("architecture"), "GGUF architecture")
    _nullable_string(gguf.get("name"), "GGUF name")
    if gguf.get("fileType") is not None:
        _nonnegative_int(gguf.get("fileType"), "GGUF file type")
    _nonempty_string(gguf.get("quantization"), "GGUF quantization")
    _positive_int(gguf.get("blockCount"), "GGUF block count")
    _positive_int(gguf.get("embeddingLength"), "GGUF embedding length")
    if gguf.get("contextLength") is not None:
        _positive_int(gguf.get("contextLength"), "GGUF context length")
    storage = _exact_mapping(
        document.get("storage"),
        (
            "layout",
            "splitPart",
            "splitCount",
            "executionScope",
            "ordinaryGgufSplitIsLayerStage",
        ),
        "GGUF storage",
    )
    if storage.get("layout") != "single-file":
        raise ValueError("whole-model manifest requires one complete GGUF file")
    if storage.get("splitPart") is not None or storage.get("splitCount") is not None:
        raise ValueError("single-file GGUF cannot have split numbering")
    if storage.get("executionScope") != "complete-model":
        raise ValueError("GGUF execution scope is invalid")
    if storage.get("ordinaryGgufSplitIsLayerStage") is not False:
        raise ValueError("ordinary gguf-split cannot be an executable layer stage")
    return document


def _parse_benchmark_document(
    value: object,
    runtime: LlamaCppRuntimeProbe,
    artifact_identity: str,
) -> LlamaBenchmarkEvidence:
    document = _exact_mapping(
        value,
        ("benchmarkId", "runtimeId", "artifactIdentity", "parameters", "records"),
        "llama.cpp benchmark evidence",
    )
    benchmark_id = _hex_digest(document.get("benchmarkId"), 32, "benchmark id")
    runtime_id = _hex_digest(document.get("runtimeId"), 32, "benchmark runtime id")
    benchmark_artifact = _model_identity(document.get("artifactIdentity"))
    parameters = _exact_mapping(
        document.get("parameters"),
        (
            "promptTokens",
            "generationTokens",
            "repetitions",
            "gpuLayers",
            "device",
            "warmup",
            "output",
        ),
        "benchmark parameters",
    )
    prompt_tokens = _positive_int(parameters.get("promptTokens"), "prompt tokens")
    generation_tokens = _positive_int(
        parameters.get("generationTokens"), "generation tokens"
    )
    _positive_int(parameters.get("repetitions"), "benchmark repetitions")
    gpu_layers = parameters.get("gpuLayers")
    if not isinstance(gpu_layers, int) or isinstance(gpu_layers, bool) or gpu_layers < -1:
        raise ValueError("benchmark gpuLayers must be -1 or non-negative")
    _nullable_string(parameters.get("device"), "benchmark device")
    if not isinstance(parameters.get("warmup"), bool):
        raise ValueError("benchmark warmup flag must be boolean")
    if parameters.get("output") != "json":
        raise ValueError("benchmark evidence must use JSON output")
    records_value = document.get("records")
    if not isinstance(records_value, Sequence) or isinstance(
        records_value, (str, bytes, bytearray)
    ) or len(records_value) < 1:
        raise ValueError("benchmark evidence must contain records")
    records: list[Mapping[str, Any]] = []
    for item in records_value:
        record = _mapping(item, "benchmark record")
        if record.get("artifact_identity") != artifact_identity:
            raise ValueError("benchmark record belongs to another artifact")
        if record.get("build_number") != runtime.build_number:
            raise ValueError("benchmark record belongs to another build")
        if str(record.get("build_commit", "")).lower() != runtime.build_commit:
            raise ValueError("benchmark record belongs to another build commit")
        _positive_int(record.get("model_size"), "benchmark model size")
        _positive_int(record.get("model_n_params"), "benchmark model parameters")
        _nonempty_string(record.get("artifact_file_name"), "benchmark artifact file name")
        n_prompt = _nonnegative_int(record.get("n_prompt"), "benchmark prompt tokens")
        n_gen = _nonnegative_int(record.get("n_gen"), "benchmark generated tokens")
        if n_prompt == 0 and n_gen == 0:
            raise ValueError("benchmark record must measure prompt or generation tokens")
        for field, name in (("avg_ns", "average nanoseconds"), ("avg_ts", "token rate")):
            measured = record.get(field)
            if (
                not isinstance(measured, (int, float))
                or isinstance(measured, bool)
                or measured <= 0
            ):
                raise ValueError(f"benchmark {name} must be positive")
        samples_ns = record.get("samples_ns")
        samples_ts = record.get("samples_ts")
        if not isinstance(samples_ns, list) or not samples_ns:
            raise ValueError("benchmark nanosecond samples are missing")
        if not isinstance(samples_ts, list) or len(samples_ts) != len(samples_ns):
            raise ValueError("benchmark token-rate samples do not match")
        records.append(record)
    if not any(
        record.get("n_prompt") == prompt_tokens and record.get("n_gen") == 0
        for record in records
    ):
        raise ValueError("benchmark evidence omits its requested prompt measurement")
    if not any(
        record.get("n_gen") == generation_tokens and record.get("n_prompt") == 0
        for record in records
    ):
        raise ValueError("benchmark evidence omits its requested generation measurement")
    evidence = LlamaBenchmarkEvidence(
        benchmark_id=benchmark_id,
        runtime_id=runtime_id,
        artifact_identity=benchmark_artifact,
        parameters_json=_canonical_json(parameters).decode("utf-8"),
        records_json=tuple(
            _canonical_json(record).decode("utf-8") for record in records
        ),
    )
    _validate_benchmark_binding(evidence, runtime, artifact_identity)
    return evidence


def _validate_partial_binding(
    runtime: LlamaCppRuntimeProbe,
    package: LlamaLayerPackage,
    executor: StageExecutorManifest,
) -> None:
    if executor.engine != "llama.cpp":
        raise ValueError("partial layer package executor must use llama.cpp")
    if executor.engine_version != runtime.engine_version:
        raise ValueError("partial executor build does not match the probed llama.cpp runtime")
    if executor.adapter != LLAMA_STAGE_ADAPTER:
        raise ValueError("partial executor is not the GDLP llama.cpp Stage adapter")
    if executor.artifact_format != LLAMA_LAYER_ARTIFACT_FORMAT:
        raise ValueError("partial executor artifact format does not match the layer package")
    if (
        executor.model_identity != package.model_identity
        or executor.model_source != package.model_source
        or executor.model_revision != package.model_revision
    ):
        raise ValueError("partial executor model does not match the layer package")
    if (
        executor.layer_start != package.layer_start
        or executor.layer_end != package.layer_end
        or executor.total_layers != package.total_layers
    ):
        raise ValueError("partial executor range does not match the layer package")
    if (
        executor.hidden_size != package.hidden_size
        or executor.activation_dtype != package.activation_dtype
        or executor.kv_format != package.kv_format
    ):
        raise ValueError("partial executor tensor contract does not match the layer package")
    if not set(executor.activation_codecs).intersection(package.activation_codecs):
        raise ValueError("partial executor and layer package have no common activation codec")
    if package.weight_dtype not in executor.weight_dtypes:
        raise ValueError("partial executor does not advertise the package weight dtype")
    if "explicit-layer-package" not in executor.features:
        raise ValueError("partial executor lacks explicit layer-package capability")
    available_compute_apis = {"cpu", *(device.backend for device in runtime.devices)}
    if not available_compute_apis.intersection(executor.compute_apis):
        raise ValueError("partial executor has no compute API available on the probed host")


def _validate_benchmark_binding(
    benchmark: LlamaBenchmarkEvidence,
    runtime: LlamaCppRuntimeProbe,
    artifact_identity: str,
) -> None:
    if benchmark.runtime_id != runtime.runtime_id:
        raise ValueError("benchmark belongs to another llama.cpp runtime")
    if benchmark.artifact_identity != artifact_identity:
        raise ValueError("benchmark belongs to another model artifact")
    body = {
        "runtimeId": benchmark.runtime_id,
        "artifactIdentity": benchmark.artifact_identity,
        "parameters": benchmark.parameters,
        "records": list(benchmark.records),
    }
    expected = hashlib.sha256(_canonical_json(body)).hexdigest()[:32]
    if benchmark.benchmark_id != expected:
        raise ValueError("benchmark identity does not match its evidence")


def _layer_package_body(**values: Any) -> dict[str, Any]:
    raw_files = values["files"]
    parsed_files = tuple(
        item if isinstance(item, LayerPackageFile) else _parse_layer_package_file(item)
        for item in raw_files
    )
    return {
        "schema": LLAMA_LAYER_PACKAGE_SCHEMA,
        "packageKind": "executable-contiguous-layer-range",
        "artifactFormat": LLAMA_LAYER_ARTIFACT_FORMAT,
        "model": {
            "identity": values["model_identity"],
            "source": values["model_source"],
            "revision": values["model_revision"],
            "architecture": values["architecture"],
            "quantization": values["quantization"],
        },
        "stage": {
            "layerStart": values["layer_start"],
            "layerEnd": values["layer_end"],
            "totalLayers": values["total_layers"],
        },
        "tensor": {
            "hiddenSize": values["hidden_size"],
            "weightDtype": values["weight_dtype"],
            "activationDtype": values["activation_dtype"],
            "activationCodecs": list(values["activation_codecs"]),
            "kvFormat": values["kv_format"],
        },
        "files": [item.to_document() for item in parsed_files],
        "ordinaryGgufSplitAcceptedAsStage": False,
    }


def _parse_layer_package_file(value: object) -> LayerPackageFile:
    document = _exact_mapping(
        value,
        ("path", "role", "sizeBytes", "sha256"),
        "llama layer package file",
    )
    path = _portable_relative_path(document.get("path"))
    if _SPLIT_FILE_RE.fullmatch(PurePosixPath(path).name):
        raise ValueError(
            "ordinary gguf-split files are complete-model storage shards, not layer weights"
        )
    role = _nonempty_string(document.get("role"), "layer package file role")
    if role not in {"layer-weights", "input-embedding", "output-head", "auxiliary"}:
        raise ValueError("layer package file role is unsupported")
    return LayerPackageFile(
        path=path,
        role=role,
        size_bytes=_positive_int(document.get("sizeBytes"), "layer package file size"),
        sha256=_hex_digest(document.get("sha256"), 64, "layer package file SHA-256"),
    )


class _GGUFReader:
    def __init__(self, stream: Any, file_size: int) -> None:
        self._stream = stream
        self._file_size = file_size
        self._digest = hashlib.sha256()

    @property
    def consumed_sha256(self) -> str:
        return self._digest.hexdigest()

    def read_exact(self, size: int) -> bytes:
        if size < 0 or size > self._file_size:
            raise ValueError("GGUF value size is invalid")
        value = self._stream.read(size)
        if len(value) != size:
            raise ValueError("GGUF file is truncated")
        self._digest.update(value)
        return value

    def skip_exact(self, size: int) -> None:
        if size < 0 or size > self._file_size:
            raise ValueError("GGUF value size is invalid")
        remaining = size
        while remaining:
            block = self.read_exact(min(remaining, 1024 * 1024))
            remaining -= len(block)

    def read_u32(self) -> int:
        return struct.unpack("<I", self.read_exact(4))[0]

    def read_u64(self) -> int:
        return struct.unpack("<Q", self.read_exact(8))[0]

    def read_string(self) -> str:
        size = self.read_u64()
        if size > self._file_size:
            raise ValueError("GGUF string size is invalid")
        try:
            return self.read_exact(size).decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("GGUF string is not UTF-8") from error

    def read_value(self, value_type: int, *, keep: bool) -> Any:
        if value_type == 8:
            value = self.read_string()
            return value if keep else None
        if value_type == 9:
            element_type = self.read_u32()
            count = self.read_u64()
            if count > 20_000_000:
                raise ValueError("GGUF array length is unreasonable")
            if element_type == 9:
                raise ValueError("GGUF nested arrays are unsupported")
            if not keep and element_type in _GGUF_VALUE_SIZES:
                byte_count = count * _GGUF_VALUE_SIZES[element_type]
                self.skip_exact(byte_count)
                return None
            values = [self.read_value(element_type, keep=keep) for _ in range(count)]
            return values if keep else None
        format_value = _GGUF_SCALAR_FORMATS.get(value_type)
        if format_value is None:
            raise ValueError(f"unsupported GGUF metadata value type: {value_type}")
        result = struct.unpack(format_value, self.read_exact(struct.calcsize(format_value)))[0]
        if value_type == 7:
            if result not in (0, 1):
                raise ValueError("GGUF boolean value is invalid")
            result = bool(result)
        return result if keep else None


def _retain_gguf_key(key: str) -> bool:
    return key in {
        "general.architecture",
        "general.name",
        "general.file_type",
        "general.quantization_version",
    } or key.endswith((".block_count", ".embedding_length", ".context_length"))


def _required_executable(root: Path, stem: str) -> Path:
    value = _optional_executable(root, stem)
    if value is None:
        raise FileNotFoundError(f"missing llama.cpp executable: {stem}")
    return value


def _optional_executable(root: Path, stem: str) -> Path | None:
    candidates = (root / f"{stem}.exe", root / stem)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _binary_set_digest(root: Path, required: Sequence[Path]) -> str:
    candidates = set(path.resolve() for path in required)
    for pattern in (
        "llama*.exe",
        "ggml*.exe",
        "llama*.dll",
        "ggml*.dll",
        "libomp*.dll",
    ):
        candidates.update(path.resolve() for path in root.glob(pattern) if path.is_file())
    digest = hashlib.sha256()
    for path in sorted(candidates, key=lambda value: value.name.lower()):
        file_digest = _sha256_file(path)
        digest.update(path.name.lower().encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_digest.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


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
    value: object, fields: Sequence[str], name: str
) -> Mapping[str, Any]:
    result = _mapping(value, name)
    if set(result) != set(fields):
        raise ValueError(f"{name} has unknown or missing fields")
    return result


def _document_without(value: Mapping[str, Any], key: str) -> dict[str, Any]:
    result = dict(value)
    result.pop(key, None)
    return result


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _nullable_string(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _nonempty_string(value, name)


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{name} must be a string list")
    result = tuple(_nonempty_string(item, name) for item in value)
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{name} must be a non-empty unique string list")
    return result


def _positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _hex_digest(value: object, length: int, name: str) -> str:
    text = _nonempty_string(value, name)
    if len(text) != length or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be {length} lowercase hexadecimal characters")
    return text


def _hex_text(value: object, name: str) -> str:
    text = _nonempty_string(value, name).lower()
    if any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be hexadecimal")
    return text


def _model_identity(value: object) -> str:
    text = _nonempty_string(value, "model identity")
    if not text.startswith("sha256:") or not _SHA256_RE.fullmatch(text[7:]):
        raise ValueError("model identity must be a SHA-256 artifact identity")
    return text


def _portable_relative_path(value: object) -> str:
    text = _nonempty_string(value, "layer package file path").replace("\\", "/")
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError("layer package file path must be portable and relative")
    return str(path)


def _coerce_property(value: str) -> bool | int | str:
    if value == "1":
        return True
    if value == "0":
        return False
    if re.fullmatch(r"-?[0-9]+", value):
        return int(value)
    return value


__all__ = [
    "GGUF_SPLIT_ROLE",
    "LLAMA_CPP_DEPLOYMENT_SCHEMA",
    "LLAMA_LAYER_ARTIFACT_FORMAT",
    "LLAMA_LAYER_PACKAGE_SCHEMA",
    "LLAMA_STAGE_ADAPTER",
    "CommandResult",
    "GGUFArtifact",
    "LayerPackageFile",
    "LlamaBenchmarkEvidence",
    "LlamaCppDeploymentManifest",
    "LlamaCppDevice",
    "LlamaCppRuntimeProbe",
    "LlamaLayerPackage",
    "benchmark_llama_cpp",
    "build_layer_package",
    "build_partial_stage_deployment",
    "build_whole_model_deployment",
    "compatibility_key_for_llama_partial",
    "inspect_gguf",
    "parse_layer_package_manifest",
    "parse_llama_cpp_deployment",
    "probe_llama_cpp",
    "verify_layer_package_files",
]
