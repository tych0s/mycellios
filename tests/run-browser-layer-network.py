"""Check browser-returned Qwen3 layer output and KV over the real coordinator."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import sys
from uuid import uuid4

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from distributed_runtime.browser_expert_owner import BrowserExpertOwner


def encoded(values) -> str:
    return base64.b64encode(np.asarray(values, dtype="<f4").tobytes()).decode("ascii")


def decoded(value: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(value, validate=True), dtype="<f4")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, default=Path("runtime/browser-layer-network-receipt.json"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    owner = BrowserExpertOwner("browser-layer-network-test", config["coordinatorUrl"])
    artifact_id = config["artifactId"]
    ready = owner._request_json("POST", "/internal/v1/mobile/layers/prepare",
                                {"artifactId": artifact_id})
    if ready.get("resident") is not True:
        raise AssertionError("browser layer was not admitted")
    request_id = str(uuid4())
    checks = []
    position = 0
    try:
        for step in fixture["steps"]:
            response = owner._request_json("POST", "/internal/v1/mobile/layers/execute", {
                "artifactId": artifact_id, "requestId": request_id,
                "position": position, "tokens": step["tokens"],
                "hiddenBase64": encoded(step["hidden"]),
                "auxiliaryBase64": {
                    "cos": encoded(step["cos"]),
                    "sin": encoded(step["sin"]),
                    "attention_mask": encoded(step["mask"]),
                },
            })
            errors = {
                "output": float(np.max(np.abs(decoded(response["outputBase64"])
                                               - np.asarray(step["expected"], dtype=np.float32)))),
                "key": float(np.max(np.abs(decoded(response["stateBase64"]["key"])
                                            - np.asarray(step["expectedNewKey"], dtype=np.float32)))),
                "value": float(np.max(np.abs(decoded(response["stateBase64"]["value"])
                                              - np.asarray(step["expectedNewValue"], dtype=np.float32)))),
            }
            if errors["output"] > 2e-4 or errors["key"] > 5e-4 or errors["value"] > 2e-4:
                raise AssertionError(f"browser/native Qwen3 layer differs: {errors}")
            position += step["tokens"]
            checks.append({"position": position, "errors": errors,
                           "durationMs": response["durationMs"]})
    finally:
        owner._request_json("POST", "/internal/v1/mobile/layers/reset",
                            {"artifactId": artifact_id, "requestId": request_id})
    receipt = {"schema": "mycellios-browser-layer-network-receipt/1",
               "artifactId": artifact_id, "workerId": ready["workerId"],
               "backend": ready["backend"], "checks": checks}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"backend": ready["backend"], "steps": len(checks),
                      "maxOutputError": max(step["errors"]["output"] for step in checks),
                      "maxKeyError": max(step["errors"]["key"] for step in checks)}))


if __name__ == "__main__":
    main()
