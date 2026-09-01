from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

from .executor_abi import model_identity_for_source


NAKSHATRA_REPOSITORY = "https://github.com/fthrvi/nakshatra.git"
NAKSHATRA_COMMIT = "0c16119713396ec6052400f3eb049c5e7a66cd94"
NAKSHATRA_LLAMA_CPP_COMMIT = "c46583b86bed573c4ff30685dae59874f124e664"
NAKSHATRA_STDIO_PROTOCOL = "nakshatra-stdio-le/1"
NAKSHATRA_PACKAGE_SCHEMA = "gdlp-nakshatra-stage-package/1"
NAKSHATRA_PACKAGE_MANIFEST = "nakshatra-stage.json"
NAKSHATRA_PACKAGE_ARTIFACT = "stage.gguf"


@dataclass(frozen=True)
class NakshatraStagePackage:
    """A verified, content-sealed layer-range artifact for Nakshatra.

    Nakshatra's pinned daemon does not report the real partial-model range in
    ``INFO``.  Consequently this manifest, including the exact sub-GGUF hash,
    is the authority for the model and stage contract.  The daemon probe is
    still useful for checking the live hidden/vocabulary dimensions.
    """

    root: Path
    package_id: str
    manifest_sha256: str
    model_identity: str
    pipeline_id: int
    model_source: str
    model_revision: str | None
    model_content_sha256: str
    layer_start: int
    layer_end: int
    total_layers: int
    mode: str
    hidden_size: int
    vocab_size: int
    has_token_embeddings: bool
    has_lm_head: bool
    artifact_sha256: str
    artifact_bytes: int
    weight_type: str
    max_context_tokens: int

    @property
    def artifact_path(self) -> Path:
        return self.root / NAKSHATRA_PACKAGE_ARTIFACT


def seal_nakshatra_stage_package(
    source_sub_gguf: str | Path,
    destination: str | Path,
    *,
    model_source: str,
    model_revision: str | None,
    model_content_sha256: str,
    pipeline_id: int,
    layer_start: int,
    layer_end: int,
    total_layers: int,
    hidden_size: int,
    vocab_size: int,
    weight_type: str,
    max_context_tokens: int,
    keep_token_embeddings: bool = False,
) -> NakshatraStagePackage:
    """Copy one upstream-generated sub-GGUF into an atomic sealed package.

    Slicing remains an external Nakshatra operation at the pinned revision;
    this project neither vendors nor silently rewrites its patched GGUF logic.
    The package records both the full-model provenance digest supplied by the
    caller and the independently measured digest of the actual stage artifact.
    """

    source = Path(source_sub_gguf).resolve()
    target = Path(destination).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"sub-GGUF does not exist: {source}")
    _validate_gguf_magic(source)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"destination already exists: {target}")

    values = _validated_contract_values(
        model_source=model_source,
        model_revision=model_revision,
        model_content_sha256=model_content_sha256,
        pipeline_id=pipeline_id,
        layer_start=layer_start,
        layer_end=layer_end,
        total_layers=total_layers,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        weight_type=weight_type,
        max_context_tokens=max_context_tokens,
        keep_token_embeddings=keep_token_embeddings,
    )

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.", dir=str(target.parent))
    )
    try:
        artifact = temporary / NAKSHATRA_PACKAGE_ARTIFACT
        artifact_sha256, artifact_bytes = _copy_and_hash(source, artifact)
        body = _package_body(
            **values,
            artifact_sha256=artifact_sha256,
            artifact_bytes=artifact_bytes,
        )
        document = {
            **body,
            "packageId": hashlib.sha256(_canonical_json(body)).hexdigest(),
        }
        manifest = temporary / NAKSHATRA_PACKAGE_MANIFEST
        raw_manifest = _canonical_json(document) + b"\n"
        with manifest.open("xb") as handle:
            handle.write(raw_manifest)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return load_nakshatra_stage_package(target)


def load_nakshatra_stage_package(
    package: str | Path,
    *,
    expected_package_id: str | None = None,
    expected_manifest_sha256: str | None = None,
) -> NakshatraStagePackage:
    """Parse and fully verify a sealed package before launching its daemon."""

    root = Path(package).resolve()
    manifest = root / NAKSHATRA_PACKAGE_MANIFEST
    if not root.is_dir() or not manifest.is_file():
        raise FileNotFoundError(
            f"Nakshatra package must contain {NAKSHATRA_PACKAGE_MANIFEST}: {root}"
        )
    raw_manifest = manifest.read_bytes()
    manifest_sha256 = hashlib.sha256(raw_manifest).hexdigest()
    if expected_manifest_sha256 is not None:
        _sha256(expected_manifest_sha256, "expected manifest SHA-256")
        if manifest_sha256 != expected_manifest_sha256:
            raise ValueError("Nakshatra package manifest SHA-256 does not match")
    try:
        document = json.loads(raw_manifest)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Nakshatra package manifest is not valid UTF-8 JSON") from error
    if not isinstance(document, Mapping):
        raise ValueError("Nakshatra package manifest must be an object")
    if set(document) != {
        "schema",
        "packageId",
        "upstream",
        "model",
        "stage",
        "artifact",
        "compatibility",
    }:
        raise ValueError("Nakshatra package has unknown or missing fields")
    if document.get("schema") != NAKSHATRA_PACKAGE_SCHEMA:
        raise ValueError("unsupported Nakshatra package schema")

    package_id = _sha256(document.get("packageId"), "packageId")
    body = dict(document)
    body.pop("packageId")
    if hashlib.sha256(_canonical_json(body)).hexdigest() != package_id:
        raise ValueError("Nakshatra package identity does not match its manifest")
    if expected_package_id is not None:
        _sha256(expected_package_id, "expected package id")
        if package_id != expected_package_id:
            raise ValueError("Nakshatra package id does not match the launch contract")

    upstream = _exact_mapping(
        document.get("upstream"),
        ("repository", "commit", "llamaCppBaseline", "protocol"),
        "upstream",
    )
    if upstream != {
        "repository": NAKSHATRA_REPOSITORY,
        "commit": NAKSHATRA_COMMIT,
        "llamaCppBaseline": NAKSHATRA_LLAMA_CPP_COMMIT,
        "protocol": NAKSHATRA_STDIO_PROTOCOL,
    }:
        raise ValueError("Nakshatra package does not use the pinned upstream contract")

    model = _exact_mapping(
        document.get("model"),
        ("identity", "pipelineId", "source", "revision", "contentSha256", "architecture"),
        "model",
    )
    stage = _exact_mapping(
        document.get("stage"),
        (
            "layerStart",
            "layerEnd",
            "totalLayers",
            "mode",
            "hiddenSize",
            "vocabSize",
            "hasTokenEmbeddings",
            "hasLmHead",
        ),
        "stage",
    )
    artifact = _exact_mapping(
        document.get("artifact"),
        ("file", "sha256", "bytes", "format", "weightType"),
        "artifact",
    )
    compatibility = _exact_mapping(
        document.get("compatibility"),
        (
            "activationDtype",
            "endianness",
            "maxContextTokens",
            "maxBatchTokens",
            "maxBatchSequences",
            "maxConcurrentRequests",
        ),
        "compatibility",
    )

    model_source = _string(model.get("source"), "model source")
    model_revision = _nullable_string(model.get("revision"), "model revision")
    model_identity = _string(model.get("identity"), "model identity")
    expected_identity = model_identity_for_source(model_source, model_revision)
    if model_identity != expected_identity:
        raise ValueError("Nakshatra model identity does not match source and revision")
    if model.get("architecture") != "llama":
        raise ValueError("the pinned Nakshatra adapter supports only Llama architecture")
    model_content_sha256 = _sha256(
        model.get("contentSha256"), "model content SHA-256"
    )
    pipeline_id = _uint64(model.get("pipelineId"), "pipelineId")

    layer_start = _nonnegative_integer(stage.get("layerStart"), "layerStart")
    layer_end = _positive_integer(stage.get("layerEnd"), "layerEnd")
    total_layers = _positive_integer(stage.get("totalLayers"), "totalLayers")
    if not layer_start < layer_end <= total_layers:
        raise ValueError("Nakshatra stage layer range is invalid")
    mode = _string(stage.get("mode"), "stage mode")
    expected_mode = _stage_mode(layer_start, layer_end, total_layers)
    if mode != expected_mode:
        raise ValueError("Nakshatra stage mode does not match its layer range")
    hidden_size = _positive_integer(stage.get("hiddenSize"), "hiddenSize")
    vocab_size = _positive_integer(stage.get("vocabSize"), "vocabSize")
    has_token_embeddings = _boolean(
        stage.get("hasTokenEmbeddings"), "hasTokenEmbeddings"
    )
    has_lm_head = _boolean(stage.get("hasLmHead"), "hasLmHead")
    if layer_start == 0 and not has_token_embeddings:
        raise ValueError("first Nakshatra stage must contain token embeddings")
    if has_lm_head is not (layer_end == total_layers):
        raise ValueError("Nakshatra lm-head flag does not match the final stage")

    if artifact.get("file") != NAKSHATRA_PACKAGE_ARTIFACT:
        raise ValueError("Nakshatra package artifact filename is unsupported")
    if artifact.get("format") != "sub-gguf-nakshatra":
        raise ValueError("Nakshatra package artifact format is unsupported")
    artifact_sha256 = _sha256(artifact.get("sha256"), "artifact SHA-256")
    artifact_bytes = _positive_integer(artifact.get("bytes"), "artifact bytes")
    weight_type = _string(artifact.get("weightType"), "weight type")

    if compatibility.get("activationDtype") != "float32":
        raise ValueError("the pinned Nakshatra protocol requires float32 activations")
    if compatibility.get("endianness") != "little":
        raise ValueError("the pinned Nakshatra protocol requires little-endian values")
    max_context_tokens = _positive_integer(
        compatibility.get("maxContextTokens"), "maxContextTokens"
    )
    if compatibility.get("maxBatchTokens") != max_context_tokens:
        raise ValueError("Nakshatra batch-token limit must equal its context limit")
    if compatibility.get("maxBatchSequences") != 1:
        raise ValueError("the pinned Nakshatra daemon supports one batch sequence")
    if compatibility.get("maxConcurrentRequests") != 1:
        raise ValueError("the pinned Nakshatra daemon supports one concurrent request")

    artifact_path = root / NAKSHATRA_PACKAGE_ARTIFACT
    if not artifact_path.is_file():
        raise FileNotFoundError(f"sealed sub-GGUF is missing: {artifact_path}")
    _validate_gguf_magic(artifact_path)
    measured_sha256, measured_bytes = _hash_file(artifact_path)
    if measured_bytes != artifact_bytes:
        raise ValueError("sealed sub-GGUF byte length does not match its manifest")
    if measured_sha256 != artifact_sha256:
        raise ValueError("sealed sub-GGUF SHA-256 does not match its manifest")

    return NakshatraStagePackage(
        root=root,
        package_id=package_id,
        manifest_sha256=manifest_sha256,
        model_identity=model_identity,
        pipeline_id=pipeline_id,
        model_source=model_source,
        model_revision=model_revision,
        model_content_sha256=model_content_sha256,
        layer_start=layer_start,
        layer_end=layer_end,
        total_layers=total_layers,
        mode=mode,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        has_token_embeddings=has_token_embeddings,
        has_lm_head=has_lm_head,
        artifact_sha256=artifact_sha256,
        artifact_bytes=artifact_bytes,
        weight_type=weight_type,
        max_context_tokens=max_context_tokens,
    )


def _validated_contract_values(**values: Any) -> dict[str, Any]:
    model_source = _string(values["model_source"], "model source")
    model_revision = _nullable_string(values["model_revision"], "model revision")
    model_content_sha256 = _sha256(
        values["model_content_sha256"], "model content SHA-256"
    )
    pipeline_id = _uint64(values["pipeline_id"], "pipelineId")
    layer_start = _nonnegative_integer(values["layer_start"], "layerStart")
    layer_end = _positive_integer(values["layer_end"], "layerEnd")
    total_layers = _positive_integer(values["total_layers"], "totalLayers")
    if not layer_start < layer_end <= total_layers:
        raise ValueError("Nakshatra stage layer range is invalid")
    hidden_size = _positive_integer(values["hidden_size"], "hiddenSize")
    vocab_size = _positive_integer(values["vocab_size"], "vocabSize")
    weight_type = _string(values["weight_type"], "weight type")
    max_context_tokens = _positive_integer(
        values["max_context_tokens"], "maxContextTokens"
    )
    keep_token_embeddings = _boolean(
        values["keep_token_embeddings"], "keepTokenEmbeddings"
    )
    return {
        "model_source": model_source,
        "model_revision": model_revision,
        "model_content_sha256": model_content_sha256,
        "model_identity": model_identity_for_source(model_source, model_revision),
        "pipeline_id": pipeline_id,
        "layer_start": layer_start,
        "layer_end": layer_end,
        "total_layers": total_layers,
        "mode": _stage_mode(layer_start, layer_end, total_layers),
        "hidden_size": hidden_size,
        "vocab_size": vocab_size,
        "has_token_embeddings": layer_start == 0 or keep_token_embeddings,
        "has_lm_head": layer_end == total_layers,
        "weight_type": weight_type,
        "max_context_tokens": max_context_tokens,
    }


def _package_body(
    *,
    model_source: str,
    model_revision: str | None,
    model_content_sha256: str,
    model_identity: str,
    pipeline_id: int,
    layer_start: int,
    layer_end: int,
    total_layers: int,
    mode: str,
    hidden_size: int,
    vocab_size: int,
    has_token_embeddings: bool,
    has_lm_head: bool,
    artifact_sha256: str,
    artifact_bytes: int,
    weight_type: str,
    max_context_tokens: int,
) -> dict[str, Any]:
    return {
        "schema": NAKSHATRA_PACKAGE_SCHEMA,
        "upstream": {
            "repository": NAKSHATRA_REPOSITORY,
            "commit": NAKSHATRA_COMMIT,
            "llamaCppBaseline": NAKSHATRA_LLAMA_CPP_COMMIT,
            "protocol": NAKSHATRA_STDIO_PROTOCOL,
        },
        "model": {
            "identity": model_identity,
            "pipelineId": pipeline_id,
            "source": model_source,
            "revision": model_revision,
            "contentSha256": model_content_sha256,
            "architecture": "llama",
        },
        "stage": {
            "layerStart": layer_start,
            "layerEnd": layer_end,
            "totalLayers": total_layers,
            "mode": mode,
            "hiddenSize": hidden_size,
            "vocabSize": vocab_size,
            "hasTokenEmbeddings": has_token_embeddings,
            "hasLmHead": has_lm_head,
        },
        "artifact": {
            "file": NAKSHATRA_PACKAGE_ARTIFACT,
            "sha256": artifact_sha256,
            "bytes": artifact_bytes,
            "format": "sub-gguf-nakshatra",
            "weightType": weight_type,
        },
        "compatibility": {
            "activationDtype": "float32",
            "endianness": "little",
            "maxContextTokens": max_context_tokens,
            "maxBatchTokens": max_context_tokens,
            "maxBatchSequences": 1,
            "maxConcurrentRequests": 1,
        },
    }


def _stage_mode(layer_start: int, layer_end: int, total_layers: int) -> str:
    if layer_end == total_layers:
        # The StageRunner adapter always submits embeddings (EMBD_DECODE).
        # A one-piece [0,total) model therefore needs Nakshatra's `last`
        # response mode so it emits tokens rather than hidden states.
        return "last"
    if layer_start == 0:
        return "first"
    return "middle"


def _copy_and_hash(source: Path, destination: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    with source.open("rb") as reader, destination.open("xb") as writer:
        while chunk := reader.read(8 * 1024 * 1024):
            digest.update(chunk)
            writer.write(chunk)
            total += len(chunk)
        writer.flush()
        os.fsync(writer.fileno())
    return digest.hexdigest(), total


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
            total += len(chunk)
    return digest.hexdigest(), total


def _validate_gguf_magic(path: Path) -> None:
    with path.open("rb") as handle:
        if handle.read(4) != b"GGUF":
            raise ValueError(f"artifact is not a GGUF file: {path}")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _exact_mapping(
    value: object, fields: tuple[str, ...], name: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        raise ValueError(f"Nakshatra {name} has unknown or missing fields")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _nullable_string(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _string(value, name)


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _positive_integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _uint64(value: object, name: str) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not 0 <= value <= (1 << 64) - 1
    ):
        raise ValueError(f"{name} must be an unsigned 64-bit integer")
    return value


def _sha256(value: object, name: str) -> str:
    text = _string(value, name)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return text


__all__ = [
    "NAKSHATRA_COMMIT",
    "NAKSHATRA_LLAMA_CPP_COMMIT",
    "NAKSHATRA_PACKAGE_ARTIFACT",
    "NAKSHATRA_PACKAGE_MANIFEST",
    "NAKSHATRA_PACKAGE_SCHEMA",
    "NAKSHATRA_REPOSITORY",
    "NAKSHATRA_STDIO_PROTOCOL",
    "NakshatraStagePackage",
    "load_nakshatra_stage_package",
    "seal_nakshatra_stage_package",
]
