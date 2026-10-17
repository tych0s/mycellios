"""Publish a checked decoder ONNX layer for browser assignment by Mycellios.

Run the layer exporter first. This command binds its graph to the exact native
model snapshot, registers the numerical canary, and writes an opt-in stage
configuration. The browser receives only the assigned layer after joining.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from distributed_runtime.browser_expert_owner import BrowserExpertOwner
from distributed_runtime.model import model_artifact_reference


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor(values: list[float]) -> str:
    import numpy as np
    return base64.b64encode(np.asarray(values, dtype="<f4").tobytes()).decode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--coordinator-url", required=True)
    parser.add_argument("--config-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path)
    parser.add_argument("--max-context-tokens", type=int, default=4096)
    args = parser.parse_args()
    model_path = args.model.resolve()
    graph_path = args.graph.resolve()
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    if fixture.get("schema") != "mycellios-local-browser-layer-fixture/2" \
            or fixture.get("architecture") not in ("qwen3", "llama"):
        raise ValueError("unsupported browser layer fixture")
    layer = fixture.get("layer")
    if not isinstance(layer, int) or isinstance(layer, bool) or layer < 0:
        raise ValueError("fixture has no valid layer index")
    checkpoint = model_path / "model.safetensors"
    if fixture.get("modelSha256") != _sha256(checkpoint):
        raise ValueError("fixture was not exported from this exact checkpoint")
    graph_sha256 = _sha256(graph_path)
    if fixture.get("onnxSha256") != graph_sha256:
        raise ValueError("ONNX graph differs from the verified fixture")
    if not 1 <= args.max_context_tokens <= 131_072:
        raise ValueError("unsupported context token limit")
    parsed = urlparse(args.coordinator_url)
    if (parsed.scheme not in ("http", "https") or not parsed.netloc
            or parsed.path not in ("", "/") or parsed.query or parsed.fragment
            or parsed.username or parsed.password
            or (parsed.scheme == "http" and parsed.hostname not in
                ("localhost", "127.0.0.1", "::1"))):
        raise ValueError("coordinator URL must be an HTTPS or loopback origin")
    steps = fixture.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("fixture has no numerical canary")
    first = steps[0]
    tokens = first["tokens"]
    if not isinstance(tokens, int) or tokens < 1 or tokens > 64:
        raise ValueError("invalid canary token count")
    owner = BrowserExpertOwner("browser-layer-publisher", args.coordinator_url)
    owner._request("PUT", f"/internal/v1/mobile/layers/graphs/{graph_sha256}",
                   graph_path.read_bytes(), content_type="application/octet-stream")
    manifest = {
        "schema": "mycellios-browser-layer/2",
        "graphSha256": graph_sha256,
        "modelDigest": model_artifact_reference(str(model_path)).identity,
        "layer": layer,
        "hidden": {"inputName": "hidden", "outputName": "output",
                   "width": fixture["hiddenSize"]},
        "auxiliary": [
            {"name": "cos", "shape": [1, "tokens", fixture["headDim"]]},
            {"name": "sin", "shape": [1, "tokens", fixture["headDim"]]},
            {"name": "attention_mask", "shape": [1, 1, "tokens", "total"]},
        ],
        "state": [
            {"name": "key", "inputName": "past_key", "outputName": "key",
             "deltaOutputName": "new_key", "shape": [1, fixture["keyValueHeads"],
              "past", fixture["headDim"]], "tokenAxis": 2},
            {"name": "value", "inputName": "past_value", "outputName": "value",
             "deltaOutputName": "new_value", "shape": [1, fixture["keyValueHeads"],
              "past", fixture["headDim"]], "tokenAxis": 2},
        ],
        "maxContextTokens": args.max_context_tokens,
        "canary": {
            "tokens": tokens,
            "hiddenBase64": _tensor(first["hidden"]),
            "auxiliaryBase64": {
                "cos": _tensor(first["cos"]),
                "sin": _tensor(first["sin"]),
                "attention_mask": _tensor(first["mask"]),
            },
            "expectedBase64": _tensor(first["expected"]),
        },
    }
    if args.manifest_output is not None:
        args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    registration = owner._request_json("POST", "/internal/v1/mobile/layers/register", manifest)
    artifact_id = registration.get("artifactId")
    if not isinstance(artifact_id, str) or len(artifact_id) != 64:
        raise ValueError("coordinator returned no layer artifact ID")
    config = {
        "schema": "mycellios-browser-layer-bridge/1",
        "artifactIdentity": model_artifact_reference(str(model_path)).identity,
        "coordinatorUrl": args.coordinator_url.rstrip("/"),
        "layer": layer, "artifactId": artifact_id,
        "maxContextTokens": args.max_context_tokens,
    }
    args.config_output.parent.mkdir(parents=True, exist_ok=True)
    args.config_output.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"artifactId": artifact_id,
                      "config": str(args.config_output.resolve())}))


if __name__ == "__main__":
    main()
