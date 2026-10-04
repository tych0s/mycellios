"""Local two-browser, complete tiny-MoE generation certification.

Start ``tests/serve-browser-split-probe.ts`` and two independent visible
``/browser/`` sessions first.  This tests the real coordinator/browser route;
it is deliberately not a physical two-computer performance claim.
"""

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
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from distributed_runtime.browser_moe_bridge import attach_browser_moe_bridge
from distributed_runtime.model import StageModelSpec, model_artifact_reference
from distributed_runtime.ram_backed_moe_stage import RamBackedMoeStageRunner
from distributed_runtime.ram_expert_cache import PredictiveCacheConfig


def workers(url: str) -> list[dict]:
    with urlopen(url + "/mobile/v1/workers", timeout=10) as response:
        return json.load(response)["data"]


def run_tokens(first, last, prompt: torch.Tensor, count: int) -> tuple[list[int], list[str]]:
    first.begin(871)
    last.begin(871)
    tokens: list[int] = []
    paths: list[str] = []
    try:
        current = prompt
        for _ in range(count):
            hidden = first.forward_ids(871, current)
            plan = first._expert_modules[0].last_resident_mesh_plan
            paths.extend(dispatch.path for dispatch in plan.dispatches)
            _, token = last.forward_hidden(871, hidden)
            assert token is not None
            tokens.append(token)
            current = torch.tensor([[token]], dtype=torch.long)
    finally:
        first.end(871)
        last.end(871)
    return tokens, paths


def reference_tokens(model, prompt: torch.Tensor, count: int) -> list[int]:
    output: list[int] = []
    cache = None
    current = prompt
    with torch.inference_mode():
        for _ in range(count):
            result = model(input_ids=current, past_key_values=cache, use_cache=True)
            cache = result.past_key_values
            token = int(torch.argmax(result.logits[:, -1, :], dim=-1).item())
            output.append(token)
            current = torch.tensor([[token]], dtype=torch.long)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coordinator", default="http://127.0.0.1:8770")
    parser.add_argument("--output", default="runtime/browser-moe-generation-receipt.json")
    args = parser.parse_args()
    if not os.environ.get("MYCELLIOS_INTERNAL_TOKEN"):
        parser.error("set MYCELLIOS_INTERNAL_TOKEN for the local coordinator")
    before = workers(args.coordinator)
    ready = [worker for worker in before if worker["status"] == "online"
             and worker["visible"] and worker["verifiedTasks"] > 0]
    if len({worker["clientId"] for worker in ready}) < 2:
        parser.error("two distinct visible validated browser workers are required")

    torch.manual_seed(871)
    torch.set_num_threads(4)
    config = Qwen3MoeConfig(
        vocab_size=32, hidden_size=32, intermediate_size=64,
        moe_intermediate_size=16, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        num_experts=1, num_experts_per_tok=1,
        max_position_embeddings=64,
        architectures=["Qwen3MoeForCausalLM"],
    )
    model = Qwen3MoeForCausalLM(config).eval()
    prompt = torch.tensor([[1, 5, 9, 3]], dtype=torch.long)
    expected = reference_tokens(model, prompt, 3)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        model.save_pretrained(root, safe_serialization=True)
        del model
        gc.collect()
        artifact = model_artifact_reference(str(root))
        bridge = {
            "schema": "mycellios-browser-moe-bridge/1",
            "artifactIdentity": artifact.identity,
            "coordinatorUrl": args.coordinator,
            "layer": 0, "expert": 0,
            # These values only make a tiny fixture choose the remote path;
            # they are not a measured performance profile.
            "localComputeMsPerToken": 100,
            "browserComputeMsPerToken": 0.01,
            "ramToDeviceGbytesPerSecond": 0.001,
            "roundTripMs": 0.01,
            "bandwidthMbps": 10_000,
            "localVramBudgetBytes": 100_000,
            "browserVramBudgetBytes": 100_000,
        }
        bridge_path = root / "bridge.json"
        bridge_path.write_text(json.dumps(bridge), encoding="utf-8")
        cache = PredictiveCacheConfig(
            capacity_bytes=16 * 32 * 3 * 4 * 2,
            prefetch_reserve_bytes=16 * 32 * 3 * 4,
            pcie_bandwidth_gbytes_per_second=8,
        )
        first = RamBackedMoeStageRunner(
            StageModelSpec(str(root), 0, 1, 2, 1, artifact_identity=artifact.identity),
            cache, device="cpu",
        )
        last = RamBackedMoeStageRunner(
            StageModelSpec(str(root), 1, 2, 2, 1, artifact_identity=artifact.identity),
            cache, device="cpu",
        )
        try:
            os.environ["MYCELLIOS_BROWSER_MOE_BRIDGE_FILE"] = str(bridge_path)
            attach_browser_moe_bridge(first)
            started = time.perf_counter()
            actual, paths = run_tokens(first, last, prompt, 3)
            elapsed_ms = (time.perf_counter() - started) * 1000
        finally:
            os.environ.pop("MYCELLIOS_BROWSER_MOE_BRIDGE_FILE", None)
            first.close()
            last.close()

    after = workers(args.coordinator)
    verified_delta = {
        worker["clientId"]: worker["verifiedTasks"]
        - next((entry["verifiedTasks"] for entry in before
                if entry["clientId"] == worker["clientId"]), 0)
        for worker in after
    }
    if actual != expected or "remote-resident" not in paths:
        raise AssertionError(f"browser generation mismatch: {actual}, {expected}, {paths}")
    if sum(delta > 0 for delta in verified_delta.values()) < 2:
        raise AssertionError("two browser replicas did not verify model work")
    receipt = {
        "schema": "mycellios-local-browser-moe-generation/1",
        "model": "generated Qwen3MoeForCausalLM tiny fixture",
        "modelArtifactIdentity": artifact.identity,
        "promptIds": prompt.tolist()[0],
        "referenceTokenIds": expected,
        "browserRouteTokenIds": actual,
        "exactTokenIdsMatch": True,
        "routePaths": paths,
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
