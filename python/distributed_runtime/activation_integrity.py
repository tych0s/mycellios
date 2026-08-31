"""Commit-first activation sketches for heterogeneous stage verification.

Adapted and substantially modified from ``shard/challenge.py`` at revision
fcf728096948c7686bcf0897e9acb75d1abda1d5 (Apache-2.0). Mycellios uses a
domain-separated hash stream for projection indexes, strict bounded wire
validation and a Torch-free verifier. Tensor production imports Torch lazily.

This module supplies a verification primitive. It does not by itself prove a
physical multi-host run or authorize settlement.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import secrets
import sys
from collections.abc import Mapping, Sequence
from typing import Any

SCHEMA = "mycellios-activation-sketch/1"
DEFAULT_SAMPLE_COUNT = 256
MAX_SAMPLE_COUNT = 1_024
MAX_ELEMENT_COUNT = 1 << 40
DEFAULT_COSINE_THRESHOLD = 0.99
DEFAULT_RELATIVE_NORM_THRESHOLD = 0.05
_SEED_BYTES = 16
_INDEX_DOMAIN = b"mycellios-activation-index/1\x00"
_COMMITMENT_DOMAIN = b"mycellios-activation-seed/1\x00"


class ActivationSketchError(ValueError):
    """The challenge or sketch is malformed and must fail closed."""


def generate_seed() -> str:
    """Return a fresh 128-bit verifier seed encoded as lowercase hex."""
    return secrets.token_hex(_SEED_BYTES)


def seed_commitment(seed: str) -> str:
    """Bind a seed before revealing which activation coordinates are sampled."""
    return "sha256:" + hashlib.sha256(_COMMITMENT_DOMAIN + _parse_seed(seed)).hexdigest()


def verify_seed_commitment(seed: str, commitment: str) -> bool:
    """Compare a revealed seed with its earlier commitment in constant time."""
    try:
        expected = seed_commitment(seed)
    except ActivationSketchError:
        return False
    return isinstance(commitment, str) and hmac.compare_digest(expected, commitment)


def projection_indices(seed: str, element_count: int,
                       sample_count: int = DEFAULT_SAMPLE_COUNT) -> tuple[int, ...]:
    """Derive device-independent, unique indexes from a bounded hash stream."""
    seed_bytes = _parse_seed(seed)
    if (not isinstance(element_count, int) or isinstance(element_count, bool)
            or element_count < 1 or element_count > MAX_ELEMENT_COUNT):
        raise ActivationSketchError("activation_element_count_is_invalid")
    if (not isinstance(sample_count, int) or isinstance(sample_count, bool)
            or sample_count < 1 or sample_count > min(MAX_SAMPLE_COUNT, element_count)):
        raise ActivationSketchError("activation_sample_count_is_invalid")
    selected: list[int] = []
    seen: set[int] = set()
    counter = 0
    while len(selected) < sample_count:
        digest = hashlib.sha256(
            _INDEX_DOMAIN + seed_bytes + counter.to_bytes(8, "big")
        ).digest()
        counter += 1
        index = int.from_bytes(digest[:8], "big") % element_count
        if index not in seen:
            seen.add(index)
            selected.append(index)
    return tuple(selected)


def make_activation_sketch(activation: Any, seed: str,
                           sample_count: int = DEFAULT_SAMPLE_COUNT) -> dict[str, Any]:
    """Create a bounded sketch from a Torch tensor without moving it wholesale."""
    try:
        import torch
    except ImportError as error:  # pragma: no cover - environment capability
        raise ActivationSketchError("activation_sketch_torch_is_unavailable") from error
    if not isinstance(activation, torch.Tensor):
        raise ActivationSketchError("activation_tensor_is_invalid")
    flattened = activation.detach().to(dtype=torch.float32).flatten()
    element_count = int(flattened.numel())
    indices = projection_indices(seed, element_count, sample_count)
    index_tensor = torch.tensor(indices, dtype=torch.long, device=flattened.device)
    projection = flattened.index_select(0, index_tensor).cpu().tolist()
    norm = float(torch.linalg.vector_norm(flattened).item())
    if not math.isfinite(norm) or any(not math.isfinite(float(value)) for value in projection):
        raise ActivationSketchError("activation_tensor_is_not_finite")
    return {
        "schema": SCHEMA,
        "seed": seed,
        "seedCommitment": seed_commitment(seed),
        "elementCount": element_count,
        "sampleCount": sample_count,
        "norm": norm,
        "projection": projection,
    }


def compare_activation_sketches(
    suspect: Mapping[str, Any], trusted: Mapping[str, Any], *,
    cosine_threshold: float = DEFAULT_COSINE_THRESHOLD,
    relative_norm_threshold: float = DEFAULT_RELATIVE_NORM_THRESHOLD,
) -> dict[str, Any]:
    """Compare two wire sketches without Torch; malformed input fails closed."""
    try:
        left = _parse_sketch(suspect)
        right = _parse_sketch(trusted)
        _validate_threshold(cosine_threshold, "activation_cosine_threshold_is_invalid")
        _validate_threshold(relative_norm_threshold, "activation_norm_threshold_is_invalid")
        if left["seed"] != right["seed"]:
            raise ActivationSketchError("activation_sketch_seed_mismatch")
        if left["seedCommitment"] != right["seedCommitment"]:
            raise ActivationSketchError("activation_sketch_commitment_mismatch")
        if (left["elementCount"] != right["elementCount"]
                or left["sampleCount"] != right["sampleCount"]):
            raise ActivationSketchError("activation_sketch_shape_mismatch")
        a, b = left["projection"], right["projection"]
        dot = math.fsum(x * y for x, y in zip(a, b, strict=True))
        magnitude_a = math.sqrt(math.fsum(x * x for x in a))
        magnitude_b = math.sqrt(math.fsum(y * y for y in b))
        cosine = (1.0 if magnitude_a == magnitude_b else 0.0) if (
            magnitude_a == 0 or magnitude_b == 0
        ) else max(-1.0, min(1.0, dot / (magnitude_a * magnitude_b)))
        denominator = max(left["norm"], right["norm"], sys.float_info.min)
        relative_norm = abs(left["norm"] - right["norm"]) / denominator
        passed = cosine >= cosine_threshold and relative_norm < relative_norm_threshold
        return {"passed": passed, "cosine": cosine, "relativeNorm": relative_norm,
                **({} if passed else {"error": "activation_sketch_diverged"})}
    except (ActivationSketchError, KeyError, TypeError, ValueError, OverflowError) as error:
        return {"passed": False, "cosine": 0.0, "relativeNorm": 1.0,
                "error": str(error) or "activation_sketch_is_malformed"}


def _parse_seed(seed: str) -> bytes:
    if not isinstance(seed, str) or len(seed) != _SEED_BYTES * 2:
        raise ActivationSketchError("activation_seed_is_invalid")
    try:
        decoded = bytes.fromhex(seed)
    except ValueError as error:
        raise ActivationSketchError("activation_seed_is_invalid") from error
    if len(decoded) != _SEED_BYTES or seed != seed.lower():
        raise ActivationSketchError("activation_seed_is_invalid")
    return decoded


def _parse_sketch(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or value.get("schema") != SCHEMA:
        raise ActivationSketchError("activation_sketch_schema_is_invalid")
    seed, commitment = value.get("seed"), value.get("seedCommitment")
    if not verify_seed_commitment(seed, commitment):
        raise ActivationSketchError("activation_sketch_commitment_is_invalid")
    element_count, sample_count = value.get("elementCount"), value.get("sampleCount")
    projection, norm = value.get("projection"), value.get("norm")
    projection_indices(seed, element_count, sample_count)
    if (not isinstance(projection, Sequence)
            or isinstance(projection, (str, bytes, bytearray))
            or len(projection) != sample_count):
        raise ActivationSketchError("activation_sketch_projection_is_invalid")
    numbers = [float(item) for item in projection]
    norm_number = float(norm)
    if norm_number < 0 or not math.isfinite(norm_number):
        raise ActivationSketchError("activation_sketch_norm_is_invalid")
    if any(not math.isfinite(item) for item in numbers):
        raise ActivationSketchError("activation_sketch_projection_is_invalid")
    return {"seed": seed, "seedCommitment": commitment, "elementCount": element_count,
            "sampleCount": sample_count, "norm": norm_number, "projection": numbers}


def _validate_threshold(value: float, error: str) -> None:
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(float(value)) or not 0 < float(value) <= 1):
        raise ActivationSketchError(error)


def main() -> int:
    """Read ``{suspect, trusted}`` JSON from stdin and emit a verdict."""
    try:
        request = json.load(sys.stdin)
        verdict = compare_activation_sketches(
            request["suspect"], request["trusted"],
            cosine_threshold=float(request.get("cosineThreshold", DEFAULT_COSINE_THRESHOLD)),
            relative_norm_threshold=float(request.get(
                "relativeNormThreshold", DEFAULT_RELATIVE_NORM_THRESHOLD)),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        json.dump({"passed": False,
                   "error": f"activation_challenge_request_invalid:{error}"}, sys.stdout)
        return 2
    json.dump(verdict, sys.stdout, separators=(",", ":"), sort_keys=True)
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
