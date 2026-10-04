"""Publish a model-independent ONNX stage after checking its declared tensor ABI.

The operator supplies a manifest with schema mycellios-browser-layer/2, exact
modelDigest and graphSha256, plus a native-output canary. A native adapter must
be registered separately for a model family to join full-model generation.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys
from urllib.parse import urlparse

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from distributed_runtime.browser_expert_owner import BrowserExpertOwner
from distributed_runtime.model import model_artifact_reference


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _shape(template: list, tokens: int, past: int) -> tuple[int, ...]:
    dimensions = tuple(tokens if value == "tokens" else
                       past if value == "past" else
                       tokens + past if value == "total" else value
                       for value in template)
    if not dimensions or any(not isinstance(value, int) or value < 0
                             for value in dimensions):
        raise ValueError("invalid declared tensor shape")
    return dimensions


def _tensor(encoded: str, shape: tuple[int, ...]) -> np.ndarray:
    raw = base64.b64decode(encoded, validate=True)
    if len(raw) != 4 * int(np.prod(shape)):
        raise ValueError("canary tensor length does not match declared shape")
    values = np.frombuffer(raw, dtype="<f4").reshape(shape)
    if not np.isfinite(values).all():
        raise ValueError("canary tensor contains non-finite values")
    return values


def _verify_graph(graph: Path, manifest: dict) -> None:
    session = ort.InferenceSession(str(graph), providers=["CPUExecutionProvider"])
    canary = manifest["canary"]
    tokens = canary["tokens"]
    hidden = manifest["hidden"]
    auxiliary = manifest["auxiliary"]
    states = manifest["state"]
    feeds = {hidden["inputName"]: _tensor(canary["hiddenBase64"],
                                          (1, tokens, hidden["width"]))}
    if set(canary["auxiliaryBase64"]) != {item["name"] for item in auxiliary}:
        raise ValueError("canary auxiliary names do not match the manifest")
    for item in auxiliary:
        feeds[item["name"]] = _tensor(canary["auxiliaryBase64"][item["name"]],
                                      _shape(item["shape"], tokens, 0))
    for item in states:
        if item["shape"][item["tokenAxis"]] != "past":
            raise ValueError("state token axis must be past")
        feeds[item["inputName"]] = np.zeros(_shape(item["shape"], tokens, 0),
                                             dtype=np.float32)
    expected_outputs = [hidden["outputName"], *[item["outputName"] for item in states],
                        *[item["deltaOutputName"] for item in states]]
    if set(feeds) != set(session.get_inputs()[i].name for i in range(len(session.get_inputs()))) \
            or not set(expected_outputs).issubset(
                {output.name for output in session.get_outputs()}):
        raise ValueError("ONNX graph IO does not match the manifest")
    values = dict(zip(expected_outputs, session.run(expected_outputs, feeds)))
    actual = values[hidden["outputName"]]
    expected = _tensor(canary["expectedBase64"], (1, tokens, hidden["width"]))
    if actual.shape != expected.shape or not np.isfinite(actual).all() \
            or not np.allclose(actual, expected, atol=2e-4, rtol=0):
        raise ValueError("ONNX canary differs from the native reference")
    for item in states:
        full = values[item["outputName"]]
        delta = values[item["deltaOutputName"]]
        expected_shape = list(_shape(item["shape"], tokens, 0))
        expected_shape[item["tokenAxis"]] = tokens
        if full.shape != tuple(expected_shape) or delta.shape != tuple(expected_shape) \
                or not np.isfinite(full).all() or not np.isfinite(delta).all():
            raise ValueError("ONNX state output differs from declared shape")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--coordinator-url", required=True)
    parser.add_argument("--config-output", type=Path)
    args = parser.parse_args()
    model = args.model.resolve()
    graph = args.graph.resolve()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema") != "mycellios-browser-layer/2" \
            or "artifactId" in manifest:
        raise ValueError("unsupported browser layer manifest")
    identity = model_artifact_reference(str(model)).identity
    if manifest.get("modelDigest") != identity or manifest.get("graphSha256") != _sha256(graph):
        raise ValueError("manifest does not match the exact model and graph")
    parsed = urlparse(args.coordinator_url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc \
            or parsed.path not in ("", "/") or parsed.query or parsed.fragment \
            or parsed.username or parsed.password \
            or (parsed.scheme == "http" and parsed.hostname not in
                ("localhost", "127.0.0.1", "::1")):
        raise ValueError("coordinator URL must be an HTTPS or loopback origin")
    _verify_graph(graph, manifest)
    owner = BrowserExpertOwner("browser-layer-publisher", args.coordinator_url)
    owner._request("PUT", f"/internal/v1/mobile/layers/graphs/{manifest['graphSha256']}",
                   graph.read_bytes(), content_type="application/octet-stream")
    registration = owner._request_json("POST", "/internal/v1/mobile/layers/register", manifest)
    artifact_id = registration["artifactId"]
    if args.config_output is not None:
        config = {"schema": "mycellios-browser-layer-bridge/1",
                  "artifactIdentity": identity,
                  "coordinatorUrl": args.coordinator_url.rstrip("/"),
                  "layer": manifest["layer"], "artifactId": artifact_id,
                  "maxContextTokens": manifest["maxContextTokens"]}
        args.config_output.parent.mkdir(parents=True, exist_ok=True)
        args.config_output.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"artifactId": artifact_id,
                      "config": str(args.config_output.resolve())
                      if args.config_output is not None else None}))


if __name__ == "__main__":
    main()
