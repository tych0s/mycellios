from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence

from .executor_abi import (
    StageExecutorManifest,
    parse_stage_executor_manifest,
    validate_executor_chain,
)


EXECUTOR_CERTIFICATION_SCHEMA = "gdlp-executor-certification/1"
_EVIDENCE_LEVELS = (
    "contract",
    "unit",
    "loopback-physical",
    "hardware-physical",
)
_PARITY_LEVELS = ("not-measured", "tolerance", "exact-greedy")
_PARALLELISM_MODES = (
    "pipeline-stage",
    "tensor-parallel-cell",
    "expert-parallel-cell",
    "context-parallel-cell",
)


@dataclass(frozen=True)
class ExecutorCompatibilityKey:
    """One exact executable combination, never a capability wildcard."""

    executor_id: str
    model_identity: str
    model_architecture: str
    artifact_format: str
    quantization: str
    engine: str
    engine_version: str | None
    adapter: str
    layer_start: int
    layer_end: int
    total_layers: int
    device_kind: str
    compute_api: str
    weight_dtype: str
    activation_dtype: str
    activation_codec: str
    parallelism_mode: str
    world_size: int
    partitioning: str
    operations: tuple[str, ...]
    features: tuple[str, ...]

    def to_document(self) -> dict[str, Any]:
        return {
            "executorId": self.executor_id,
            "model": {
                "identity": self.model_identity,
                "architecture": self.model_architecture,
                "artifactFormat": self.artifact_format,
                "quantization": self.quantization,
            },
            "engine": {
                "id": self.engine,
                "version": self.engine_version,
                "adapter": self.adapter,
            },
            "stage": {
                "layerStart": self.layer_start,
                "layerEnd": self.layer_end,
                "totalLayers": self.total_layers,
            },
            "runtime": {
                "deviceKind": self.device_kind,
                "computeApi": self.compute_api,
                "weightDtype": self.weight_dtype,
                "activationDtype": self.activation_dtype,
                "activationCodec": self.activation_codec,
                "parallelismMode": self.parallelism_mode,
                "worldSize": self.world_size,
                "partitioning": self.partitioning,
                "operations": list(self.operations),
                "features": list(self.features),
            },
        }


@dataclass(frozen=True)
class CompatibilityEvidence:
    level: str
    parity: str
    test_id: str
    artifact_sha256: str | None
    device_fingerprint: str | None

    def to_document(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "parity": self.parity,
            "testId": self.test_id,
            "artifactSha256": self.artifact_sha256,
            "deviceFingerprint": self.device_fingerprint,
        }


@dataclass(frozen=True)
class ExecutorCertification:
    certification_id: str
    key: ExecutorCompatibilityKey
    evidence: CompatibilityEvidence

    def to_document(self) -> dict[str, Any]:
        return {
            "schema": EXECUTOR_CERTIFICATION_SCHEMA,
            "certificationId": self.certification_id,
            "key": self.key.to_document(),
            "evidence": self.evidence.to_document(),
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_document())).hexdigest()


class CompatibilityNotCertifiedError(ValueError):
    def __init__(
        self,
        key: ExecutorCompatibilityKey,
        minimum_evidence: str,
    ) -> None:
        super().__init__(
            "executor combination is not certified: "
            f"{key.engine}/{key.adapter} {key.model_architecture} "
            f"{key.device_kind}/{key.compute_api} {key.parallelism_mode}x{key.world_size} "
            f"requires {minimum_evidence}"
        )
        self.key = key
        self.minimum_evidence = minimum_evidence


class ExecutorCompatibilityRegistry:
    """Fail-closed registry keyed by the complete physical execution tuple.

    Adapter-registry membership is only a software contract. Evidence levels
    are accepted exactly as recorded here: contract, unit and loopback results
    can never satisfy a ``hardware-physical`` requirement.
    """

    def __init__(self, certifications: Iterable[ExecutorCertification] = ()) -> None:
        self._by_key: dict[ExecutorCompatibilityKey, list[ExecutorCertification]] = {}
        self._by_id: dict[str, ExecutorCertification] = {}
        for certification in certifications:
            self.add(certification)

    def add(self, certification: ExecutorCertification | Mapping[str, Any]) -> None:
        parsed = parse_executor_certification(
            certification.to_document()
            if isinstance(certification, ExecutorCertification)
            else certification
        )
        previous = self._by_id.get(parsed.certification_id)
        if previous is not None:
            if previous != parsed:
                raise ValueError("certification id collision")
            return
        self._by_id[parsed.certification_id] = parsed
        self._by_key.setdefault(parsed.key, []).append(parsed)
        self._by_key[parsed.key].sort(
            key=lambda item: (
                _EVIDENCE_LEVELS.index(item.evidence.level),
                _PARITY_LEVELS.index(item.evidence.parity),
                item.certification_id,
            ),
            reverse=True,
        )

    def resolve(
        self,
        key: ExecutorCompatibilityKey,
        *,
        minimum_evidence: str = "unit",
        require_exact_greedy: bool = False,
    ) -> ExecutorCertification | None:
        _evidence_level(minimum_evidence)
        minimum_index = _EVIDENCE_LEVELS.index(minimum_evidence)
        for certification in self._by_key.get(key, ()):
            if _EVIDENCE_LEVELS.index(certification.evidence.level) < minimum_index:
                continue
            if require_exact_greedy and certification.evidence.parity != "exact-greedy":
                continue
            return certification
        return None

    def require(
        self,
        key: ExecutorCompatibilityKey,
        *,
        minimum_evidence: str = "unit",
        require_exact_greedy: bool = False,
    ) -> ExecutorCertification:
        certification = self.resolve(
            key,
            minimum_evidence=minimum_evidence,
            require_exact_greedy=require_exact_greedy,
        )
        if certification is None:
            raise CompatibilityNotCertifiedError(key, minimum_evidence)
        return certification

    def documents(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            self._by_id[identifier].to_document()
            for identifier in sorted(self._by_id)
        )


def compatibility_key_for_manifest(
    manifest_value: StageExecutorManifest | Mapping[str, Any],
    *,
    model_architecture: str,
    quantization: str,
    device_kind: str,
    compute_api: str,
    weight_dtype: str,
    activation_codec: str,
    parallelism_mode: str,
    world_size: int,
    partitioning: str,
) -> ExecutorCompatibilityKey:
    """Bind an ABI manifest to one selected physical backend configuration."""

    manifest = (
        manifest_value
        if isinstance(manifest_value, StageExecutorManifest)
        else parse_stage_executor_manifest(manifest_value)
    )
    selected = {
        "device kind": (device_kind, manifest.device_kinds),
        "compute API": (compute_api, manifest.compute_apis),
        "weight dtype": (weight_dtype, manifest.weight_dtypes),
        "activation codec": (activation_codec, manifest.activation_codecs),
    }
    for name, (value, advertised) in selected.items():
        _string(value, name)
        if value not in advertised:
            raise ValueError(f"selected {name} is not advertised by executor")
    _string(model_architecture, "model architecture")
    _string(quantization, "quantization")
    _string(partitioning, "partitioning")
    if parallelism_mode not in _PARALLELISM_MODES:
        raise ValueError("parallelism mode is unsupported")
    _positive_integer(world_size, "world size")
    if parallelism_mode == "pipeline-stage" and world_size != 1:
        raise ValueError("pipeline-stage compatibility requires world size one")
    if parallelism_mode != "pipeline-stage" and world_size < 2:
        raise ValueError("cell compatibility requires multiple members")
    return ExecutorCompatibilityKey(
        executor_id=manifest.executor_id,
        model_identity=manifest.model_identity,
        model_architecture=model_architecture,
        artifact_format=manifest.artifact_format,
        quantization=quantization,
        engine=manifest.engine,
        engine_version=manifest.engine_version,
        adapter=manifest.adapter,
        layer_start=manifest.layer_start,
        layer_end=manifest.layer_end,
        total_layers=manifest.total_layers,
        device_kind=device_kind,
        compute_api=compute_api,
        weight_dtype=weight_dtype,
        activation_dtype=manifest.activation_dtype,
        activation_codec=activation_codec,
        parallelism_mode=parallelism_mode,
        world_size=world_size,
        partitioning=partitioning,
        operations=tuple(sorted(manifest.operations)),
        features=tuple(sorted(manifest.features)),
    )


def build_executor_certification(
    key: ExecutorCompatibilityKey,
    *,
    evidence_level: str,
    parity: str,
    test_id: str,
    artifact_sha256: str | None = None,
    device_fingerprint: str | None = None,
) -> ExecutorCertification:
    _validate_key(key)
    evidence = CompatibilityEvidence(
        level=_evidence_level(evidence_level),
        parity=_parity(parity),
        test_id=_string(test_id, "test id"),
        artifact_sha256=_nullable_sha256(artifact_sha256, "artifact SHA-256"),
        device_fingerprint=_nullable_string(device_fingerprint, "device fingerprint"),
    )
    body = {
        "schema": EXECUTOR_CERTIFICATION_SCHEMA,
        "key": key.to_document(),
        "evidence": evidence.to_document(),
    }
    certification_id = hashlib.sha256(_canonical_json(body)).hexdigest()[:32]
    return ExecutorCertification(
        certification_id=certification_id,
        key=key,
        evidence=evidence,
    )


def parse_executor_certification(value: object) -> ExecutorCertification:
    document = _exact_mapping(
        value,
        ("schema", "certificationId", "key", "evidence"),
        "executor certification",
    )
    if document.get("schema") != EXECUTOR_CERTIFICATION_SCHEMA:
        raise ValueError("unsupported executor certification schema")
    key = _parse_key(document.get("key"))
    evidence_document = _exact_mapping(
        document.get("evidence"),
        ("level", "parity", "testId", "artifactSha256", "deviceFingerprint"),
        "certification evidence",
    )
    evidence = CompatibilityEvidence(
        level=_evidence_level(evidence_document.get("level")),
        parity=_parity(evidence_document.get("parity")),
        test_id=_string(evidence_document.get("testId"), "test id"),
        artifact_sha256=_nullable_sha256(
            evidence_document.get("artifactSha256"), "artifact SHA-256"
        ),
        device_fingerprint=_nullable_string(
            evidence_document.get("deviceFingerprint"), "device fingerprint"
        ),
    )
    parsed = build_executor_certification(
        key,
        evidence_level=evidence.level,
        parity=evidence.parity,
        test_id=evidence.test_id,
        artifact_sha256=evidence.artifact_sha256,
        device_fingerprint=evidence.device_fingerprint,
    )
    certification_id = _hex_string(
        document.get("certificationId"), 32, "certification id"
    )
    if parsed.certification_id != certification_id:
        raise ValueError("executor certification identity does not match its contract")
    return parsed


def validate_certified_executor_chain(
    manifests: Sequence[StageExecutorManifest | Mapping[str, Any]],
    keys: Sequence[ExecutorCompatibilityKey],
    registry: ExecutorCompatibilityRegistry,
    *,
    minimum_evidence: str = "unit",
    require_exact_greedy: bool = False,
) -> tuple[ExecutorCertification, ...]:
    parsed = validate_executor_chain(manifests)
    if len(parsed) != len(keys):
        raise ValueError("executor chain and compatibility key counts differ")
    certifications: list[ExecutorCertification] = []
    for manifest, key in zip(parsed, keys):
        if manifest.executor_id != key.executor_id:
            raise ValueError("compatibility key belongs to another executor")
        certifications.append(
            registry.require(
                key,
                minimum_evidence=minimum_evidence,
                require_exact_greedy=require_exact_greedy,
            )
        )
    return tuple(certifications)


def _parse_key(value: object) -> ExecutorCompatibilityKey:
    document = _exact_mapping(
        value,
        ("executorId", "model", "engine", "stage", "runtime"),
        "compatibility key",
    )
    model = _exact_mapping(
        document.get("model"),
        ("identity", "architecture", "artifactFormat", "quantization"),
        "compatibility model",
    )
    engine = _exact_mapping(
        document.get("engine"),
        ("id", "version", "adapter"),
        "compatibility engine",
    )
    stage = _exact_mapping(
        document.get("stage"),
        ("layerStart", "layerEnd", "totalLayers"),
        "compatibility stage",
    )
    runtime = _exact_mapping(
        document.get("runtime"),
        (
            "deviceKind",
            "computeApi",
            "weightDtype",
            "activationDtype",
            "activationCodec",
            "parallelismMode",
            "worldSize",
            "partitioning",
            "operations",
            "features",
        ),
        "compatibility runtime",
    )
    key = ExecutorCompatibilityKey(
        executor_id=_hex_string(document.get("executorId"), 32, "executor id"),
        model_identity=_string(model.get("identity"), "model identity"),
        model_architecture=_string(model.get("architecture"), "model architecture"),
        artifact_format=_string(model.get("artifactFormat"), "artifact format"),
        quantization=_string(model.get("quantization"), "quantization"),
        engine=_string(engine.get("id"), "engine id"),
        engine_version=_nullable_string(engine.get("version"), "engine version"),
        adapter=_string(engine.get("adapter"), "engine adapter"),
        layer_start=_nonnegative_integer(stage.get("layerStart"), "layer start"),
        layer_end=_positive_integer(stage.get("layerEnd"), "layer end"),
        total_layers=_positive_integer(stage.get("totalLayers"), "total layers"),
        device_kind=_string(runtime.get("deviceKind"), "device kind"),
        compute_api=_string(runtime.get("computeApi"), "compute API"),
        weight_dtype=_string(runtime.get("weightDtype"), "weight dtype"),
        activation_dtype=_string(runtime.get("activationDtype"), "activation dtype"),
        activation_codec=_string(runtime.get("activationCodec"), "activation codec"),
        parallelism_mode=_string(runtime.get("parallelismMode"), "parallelism mode"),
        world_size=_positive_integer(runtime.get("worldSize"), "world size"),
        partitioning=_string(runtime.get("partitioning"), "partitioning"),
        operations=_sorted_string_tuple(runtime.get("operations"), "operations"),
        features=_sorted_string_tuple(runtime.get("features"), "features"),
    )
    _validate_key(key)
    return key


def _validate_key(key: ExecutorCompatibilityKey) -> None:
    if not isinstance(key, ExecutorCompatibilityKey):
        raise ValueError("compatibility key is invalid")
    _hex_string(key.executor_id, 32, "executor id")
    for name, value in (
        ("model identity", key.model_identity),
        ("model architecture", key.model_architecture),
        ("artifact format", key.artifact_format),
        ("quantization", key.quantization),
        ("engine", key.engine),
        ("adapter", key.adapter),
        ("device kind", key.device_kind),
        ("compute API", key.compute_api),
        ("weight dtype", key.weight_dtype),
        ("activation dtype", key.activation_dtype),
        ("activation codec", key.activation_codec),
        ("partitioning", key.partitioning),
    ):
        _string(value, name)
    _nullable_string(key.engine_version, "engine version")
    _nonnegative_integer(key.layer_start, "layer start")
    _positive_integer(key.layer_end, "layer end")
    _positive_integer(key.total_layers, "total layers")
    if not key.layer_start < key.layer_end <= key.total_layers:
        raise ValueError("compatibility layer range is invalid")
    _positive_integer(key.world_size, "world size")
    if key.parallelism_mode not in _PARALLELISM_MODES:
        raise ValueError("parallelism mode is unsupported")
    if key.parallelism_mode == "pipeline-stage" and key.world_size != 1:
        raise ValueError("pipeline-stage compatibility requires world size one")
    if key.parallelism_mode != "pipeline-stage" and key.world_size < 2:
        raise ValueError("cell compatibility requires multiple members")
    if key.operations != tuple(sorted(set(key.operations))):
        raise ValueError("operations must be a unique sorted tuple")
    if key.features != tuple(sorted(set(key.features))):
        raise ValueError("features must be a unique sorted tuple")
    for value in (*key.operations, *key.features):
        _string(value, "compatibility operation or feature")


def _exact_mapping(
    value: object,
    fields: Sequence[str],
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        raise ValueError(f"{name} has unknown or missing fields")
    return value


def _sorted_string_tuple(value: object, name: str) -> tuple[str, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or len(value) < 1
    ):
        raise ValueError(f"{name} must be a non-empty list")
    result = tuple(_string(item, name) for item in value)
    if result != tuple(sorted(set(result))):
        raise ValueError(f"{name} must be unique and sorted")
    return result


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _nullable_string(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _string(value, name)


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


def _nullable_sha256(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _hex_string(value, 64, name)


def _evidence_level(value: object) -> str:
    if value not in _EVIDENCE_LEVELS:
        raise ValueError("evidence level is unsupported")
    return str(value)


def _parity(value: object) -> str:
    if value not in _PARITY_LEVELS:
        raise ValueError("parity level is unsupported")
    return str(value)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


__all__ = [
    "EXECUTOR_CERTIFICATION_SCHEMA",
    "CompatibilityEvidence",
    "CompatibilityNotCertifiedError",
    "ExecutorCertification",
    "ExecutorCompatibilityKey",
    "ExecutorCompatibilityRegistry",
    "build_executor_certification",
    "compatibility_key_for_manifest",
    "parse_executor_certification",
    "validate_certified_executor_chain",
]
