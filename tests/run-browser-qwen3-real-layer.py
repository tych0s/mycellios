"""Compare the cached real Qwen3 0.6B layer-0 output through browser and Torch."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time
from urllib.request import urlopen

import torch

from distributed_runtime.model import StageModelSpec, StageRunner, model_artifact_reference


def workers(url: str) -> list[dict]:
    with urlopen(url + "/mobile/v1/workers", timeout=10) as response:
        return json.load(response)["data"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--coordinator", default="http://127.0.0.1:8770")
    parser.add_argument("--output", default="runtime/browser-qwen3-real-layer-receipt.json")
    args = parser.parse_args()
    if not os.environ.get("MYCELLIOS_INTERNAL_TOKEN"):
        parser.error("set MYCELLIOS_INTERNAL_TOKEN for the local coordinator")
    before = workers(args.coordinator)
    ready = [worker for worker in before if worker["status"] == "online"
             and worker["visible"] and worker["verifiedTasks"] > 0]
    if len({worker["clientId"] for worker in ready}) < 2:
        parser.error("two distinct visible validated browser workers are required")

    snapshot = Path(args.snapshot).resolve()
    artifact = model_artifact_reference(str(snapshot))
    model_config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    if model_config.get("model_type") != "qwen3" or model_config.get("hidden_size") != 1024:
        parser.error("expected a local Qwen3 0.6B safetensors snapshot")
    torch.set_num_threads(4)
    prompt = torch.tensor([[1, 5, 9, 3]], dtype=torch.long)
    spec = StageModelSpec(
        str(snapshot), 0, 1, model_config["num_hidden_layers"], 4,
        artifact_identity=artifact.identity,
    )
    with tempfile.TemporaryDirectory() as directory:
        bridge_path = Path(directory, "bridge.json")
        bridge_path.write_text(json.dumps({
            "schema": "mycellios-browser-dense-bridge/1",
            "artifactIdentity": artifact.identity,
            "coordinatorUrl": args.coordinator,
            "layer": 0,
        }), encoding="utf-8")
        os.environ["MYCELLIOS_BROWSER_DENSE_BRIDGE_FILE"] = str(bridge_path)
        browser_stage = None
        native_stage = None
        try:
            browser_stage = StageRunner(spec, device="cpu")
            os.environ.pop("MYCELLIOS_BROWSER_DENSE_BRIDGE_FILE", None)
            native_stage = StageRunner(spec, device="cpu")
            browser_samples_ms: list[float] = []
            native_samples_ms: list[float] = []
            max_error = 0.0
            for repeat in range(3):
                request_id = 901 + repeat
                browser_stage.begin(request_id)
                native_stage.begin(request_id)
                try:
                    started = time.perf_counter()
                    actual = browser_stage.forward_ids(request_id, prompt)
                    browser_samples_ms.append((time.perf_counter() - started) * 1000)
                    started = time.perf_counter()
                    expected = native_stage.forward_ids(request_id, prompt)
                    native_samples_ms.append((time.perf_counter() - started) * 1000)
                finally:
                    browser_stage.end(request_id)
                    native_stage.end(request_id)
                max_error = max(max_error, float((actual - expected).abs().max()))
                torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
            wrapper = browser_stage.base.layers[0].mlp
            if wrapper.browser_forwards != 3 or wrapper.native_forwards != 0:
                raise AssertionError("the actual Qwen3 MLP did not run in browsers")
        finally:
            os.environ.pop("MYCELLIOS_BROWSER_DENSE_BRIDGE_FILE", None)
            if browser_stage is not None:
                browser_stage.close()
            if native_stage is not None:
                native_stage.close()

    after = workers(args.coordinator)
    verified_delta = {
        worker["clientId"]: worker["verifiedTasks"]
        - next((entry["verifiedTasks"] for entry in before
                if entry["clientId"] == worker["clientId"]), 0)
        for worker in after
    }
    if sum(delta > 0 for delta in verified_delta.values()) < 2:
        raise AssertionError("both browsers must verify the real Qwen3 layer")
    receipt = {
        "schema": "mycellios-local-browser-qwen3-real-layer/1",
        "modelArtifactIdentity": artifact.identity,
        "model": "Qwen/Qwen3-0.6B local safetensors snapshot",
        "layer": 0,
        "promptIds": prompt.tolist()[0],
        "activationShape": list(actual.shape),
        "maxAbsError": max_error,
        "browserForwards": wrapper.browser_forwards,
        "nativeFallbackForwards": wrapper.native_forwards,
        "browserVerifiedTaskDeltas": verified_delta,
        "browserBackends": sorted(worker["backend"] for worker in after
                                   if verified_delta.get(worker["clientId"], 0) > 0),
        "browserStageSamplesMs": [round(value, 3) for value in browser_samples_ms],
        "nativeStageSamplesMs": [round(value, 3) for value in native_samples_ms],
        "samePhysicalHost": True,
        "completeModelGeneration": False,
        "physicalTwoHostProof": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
