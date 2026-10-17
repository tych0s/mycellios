"""Same-host end-to-end Qwen3 dense generation through two browser workers."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import tempfile
import time
from urllib.request import urlopen

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from distributed_runtime.model import StageModelSpec, StageRunner, model_artifact_reference


def workers(url: str) -> list[dict]:
    with urlopen(url + "/mobile/v1/workers", timeout=10) as response:
        return json.load(response)["data"]


def reference_tokens(model, prompt: torch.Tensor, count: int) -> list[int]:
    cache = None
    current = prompt
    result: list[int] = []
    with torch.inference_mode():
        for _ in range(count):
            output = model(input_ids=current, past_key_values=cache, use_cache=True)
            cache = output.past_key_values
            token = int(torch.argmax(output.logits[:, -1, :], dim=-1).item())
            result.append(token)
            current = torch.tensor([[token]], dtype=torch.long)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coordinator", default="http://127.0.0.1:8770")
    parser.add_argument("--output", default="runtime/browser-dense-generation-receipt.json")
    parser.add_argument("--expect-fallback", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("MYCELLIOS_INTERNAL_TOKEN"):
        parser.error("set MYCELLIOS_INTERNAL_TOKEN for the local coordinator")
    before = workers(args.coordinator)
    ready = [worker for worker in before if worker["status"] == "online"
             and worker["visible"] and worker["verifiedTasks"] > 0]
    if not args.expect_fallback and len({worker["clientId"] for worker in ready}) < 2:
        parser.error("two distinct visible validated browser workers are required")

    torch.manual_seed(877)
    torch.set_num_threads(4)
    config = Qwen3Config(
        vocab_size=32, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=64,
        architectures=["Qwen3ForCausalLM"],
    )
    model = Qwen3ForCausalLM(config).eval()
    prompt = torch.tensor([[1, 5, 9, 3]], dtype=torch.long)
    expected = reference_tokens(model, prompt, 3)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        model.save_pretrained(root, safe_serialization=True)
        del model
        gc.collect()
        artifact = model_artifact_reference(str(root))
        bridge = {
            "schema": "mycellios-browser-dense-bridge/1",
            "artifactIdentity": artifact.identity,
            "coordinatorUrl": args.coordinator,
            "layer": 0,
        }
        bridge_path = root / "dense-bridge.json"
        bridge_path.write_text(json.dumps(bridge), encoding="utf-8")
        os.environ["MYCELLIOS_BROWSER_DENSE_BRIDGE_FILE"] = str(bridge_path)
        first = None
        last = None
        try:
            first = StageRunner(StageModelSpec(
                str(root), 0, 1, 2, 1, artifact_identity=artifact.identity,
            ), device="cpu")
            last = StageRunner(StageModelSpec(
                str(root), 1, 2, 2, 1, artifact_identity=artifact.identity,
            ), device="cpu")
            wrapper = first.base.layers[0].mlp
            first.begin(877)
            last.begin(877)
            actual: list[int] = []
            started = time.perf_counter()
            try:
                current = prompt
                for _ in range(3):
                    hidden = first.forward_ids(877, current)
                    _, token = last.forward_hidden(877, hidden)
                    assert token is not None
                    actual.append(token)
                    current = torch.tensor([[token]], dtype=torch.long)
            finally:
                first.end(877)
                last.end(877)
            elapsed_ms = (time.perf_counter() - started) * 1000
            browser_forwards = wrapper.browser_forwards
            native_forwards = wrapper.native_forwards
        finally:
            os.environ.pop("MYCELLIOS_BROWSER_DENSE_BRIDGE_FILE", None)
            if first is not None:
                first.close()
            if last is not None:
                last.close()

    after = workers(args.coordinator)
    verified_delta = {
        worker["clientId"]: worker["verifiedTasks"]
        - next((entry["verifiedTasks"] for entry in before
                if entry["clientId"] == worker["clientId"]), 0)
        for worker in after
    }
    expected_browser = 0 if args.expect_fallback else 3
    expected_native = 3 if args.expect_fallback else 0
    if (actual != expected or browser_forwards != expected_browser
            or native_forwards != expected_native):
        raise AssertionError(f"dense browser generation failed: {actual}, {expected}, "
                             f"browser={browser_forwards}, native={native_forwards}")
    if not args.expect_fallback and sum(delta > 0 for delta in verified_delta.values()) < 2:
        raise AssertionError("both browser replicas must verify real model work")
    receipt = {
        "schema": "mycellios-local-browser-dense-generation/1",
        "model": "generated Qwen3ForCausalLM tiny fixture",
        "modelArtifactIdentity": artifact.identity,
        "promptIds": prompt.tolist()[0],
        "referenceTokenIds": expected,
        "browserRouteTokenIds": actual,
        "exactTokenIdsMatch": True,
        "browserForwards": browser_forwards,
        "nativeFallbackForwards": native_forwards,
        "browserVerifiedTaskDeltas": verified_delta,
        "browserBackends": sorted(worker["backend"] for worker in after
                                   if verified_delta.get(worker["clientId"], 0) > 0),
        "generationElapsedMs": round(elapsed_ms, 3),
        "samePhysicalHost": True,
        "completeModelGeneration": True,
        "physicalTwoHostProof": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
