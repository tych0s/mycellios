"""Local synthetic browser ABI check with arbitrary names and no state."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

import numpy as np
import onnx
from onnx import TensorProto, helper

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from distributed_runtime.browser_expert_owner import BrowserExpertOwner
from distributed_runtime.model import model_artifact_reference


def _tensor(values: list[float]) -> str:
    return base64.b64encode(np.asarray(values, dtype="<f4").tobytes()).decode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--coordinator-url", required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("runtime/browser-generic-abi-receipt.json"))
    args = parser.parse_args()
    artifact_dir = Path("runtime/browser-generic-abi")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    graph_path = artifact_dir / "stage.onnx"
    inputs = [helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, "tokens", 2])
              for name in ("activation", "bias")]
    graph = helper.make_graph(
        [helper.make_node("Add", ["activation", "bias"], ["result"])],
        "synthetic-stateless-stage", inputs,
        [helper.make_tensor_value_info("result", TensorProto.FLOAT, [1, "tokens", 2])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    onnx.save(model, graph_path)
    manifest_path = artifact_dir / "manifest.json"
    manifest_path.write_text(json.dumps({
        "schema": "mycellios-browser-layer/2",
        "graphSha256": hashlib.sha256(graph_path.read_bytes()).hexdigest(),
        "modelDigest": model_artifact_reference(str(args.model.resolve())).identity,
        "layer": 0,
        "hidden": {"inputName": "activation", "outputName": "result", "width": 2},
        "auxiliary": [{"name": "bias", "shape": [1, "tokens", 2]}],
        "state": [], "maxContextTokens": 8,
        "canary": {"tokens": 1, "hiddenBase64": _tensor([1, 2]),
                   "auxiliaryBase64": {"bias": _tensor([3, 4])},
                   "expectedBase64": _tensor([4, 6])},
    }), encoding="utf-8")
    publisher = Path(__file__).resolve().parents[1] / "scripts" / "publish-browser-layer.py"
    result = subprocess.run([sys.executable, str(publisher),
                             "--model", str(args.model), "--graph", str(graph_path),
                             "--manifest", str(manifest_path),
                             "--coordinator-url", args.coordinator_url],
                            check=True, capture_output=True, text=True)
    artifact_id = json.loads(result.stdout)["artifactId"]
    owner = BrowserExpertOwner("browser-generic-abi-test", args.coordinator_url)
    ready = owner._request_json("POST", "/internal/v1/mobile/layers/prepare",
                                {"artifactId": artifact_id})
    request_id = str(uuid4())
    try:
        checks = []
        for position, hidden in enumerate(([5, 6], [7, 8])):
            response = owner._request_json("POST", "/internal/v1/mobile/layers/execute", {
                "artifactId": artifact_id, "requestId": request_id,
                "position": position, "tokens": 1,
                "hiddenBase64": _tensor(hidden),
                "auxiliaryBase64": {"bias": _tensor([3, 4])},
            })
            output = np.frombuffer(base64.b64decode(response["outputBase64"]),
                                   dtype="<f4")
            expected = np.asarray(hidden, dtype=np.float32) + [3, 4]
            if response["stateBase64"] != {} or not np.array_equal(output, expected):
                raise AssertionError("generic browser stage changed its output or state")
            checks.append(output.tolist())
    finally:
        owner._request_json("POST", "/internal/v1/mobile/layers/reset",
                            {"artifactId": artifact_id, "requestId": request_id})
    receipt = {"schema": "mycellios-browser-generic-abi-receipt/1",
               "synthetic": True, "artifactId": artifact_id,
               "backend": ready["backend"], "outputs": checks}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
