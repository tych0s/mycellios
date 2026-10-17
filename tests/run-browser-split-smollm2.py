"""Local browser-stage probe using the real block-0 SmolLM2 GGUF MLP weights.

The native mesh owns routing and reduction. Two visible browser workers execute
the assigned SwiGLU projection through the existing coordinator contract.
This does not claim a complete transformer-layer or multi-host inference run.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
from statistics import median
import time
from urllib.request import Request, urlopen

import torch
import torch.nn.functional as F

from distributed_runtime.browser_expert_owner import BrowserExpertOwner
from distributed_runtime.native_gguf import dequantize_gguf_tensor, parse_gguf
from distributed_runtime.ram_expert_cache import ExpertKey, ExpertRecord
from distributed_runtime.resident_expert_mesh import (
    AuthoritativeRouting,
    MeshLinkProfile,
    MeshNodeProfile,
    ResidentExpertMesh,
    ResidentExpertReplica,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coordinator", default="http://127.0.0.1:8770")
    parser.add_argument("--gguf", default="runtime/models/SmolLM2-135M-Instruct-F16.gguf")
    parser.add_argument("--output", default="runtime/browser-split-probe-result.json")
    parser.add_argument("--require-webgpu", action="store_true")
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--diagnostic-executions", type=int, default=0)
    args = parser.parse_args()
    if args.rows < 1:
        parser.error("--rows must be positive")
    if args.repeats < 1 or args.diagnostic_executions < 0:
        parser.error("--repeats must be positive and --diagnostic-executions nonnegative")
    torch.set_num_threads(4)

    document = parse_gguf(args.gguf)
    names = {
        "gate": "blk.0.ffn_gate.weight",
        "up": "blk.0.ffn_up.weight",
        "down": "blk.0.ffn_down.weight",
    }
    tensors = {tensor.name: tensor for tensor in document.tensors}
    weights: dict[str, torch.Tensor] = {}
    with open(args.gguf, "rb") as source:
        for role, name in names.items():
            tensor = tensors[name]
            source.seek(tensor.data_offset)
            weights[role] = dequantize_gguf_tensor(tensor, source.read(tensor.size_bytes))

    gate, up, down = weights["gate"], weights["up"], weights["down"]
    intermediate, hidden_size = gate.shape
    assert up.shape == (intermediate, hidden_size)
    assert down.shape == (hidden_size, intermediate)
    weight_bytes = 3 * hidden_size * intermediate * 4
    packed = torch.cat((gate.reshape(-1), up.reshape(-1), down.reshape(-1)))
    content_id = "sha256:" + hashlib.sha256(packed.numpy().tobytes()).hexdigest()

    key = ExpertKey(0, 0)
    owner = BrowserExpertOwner(
        "browser-stage",
        args.coordinator,
        internal_token="local-browser-split-probe",
        timeout_seconds=90,
    )
    published_at = time.perf_counter()
    artifact_id = owner.publish_swiglu_expert(
        key=key,
        content_id=content_id,
        model_id="SmolLM2-135M-Instruct-F16",
        model_digest="sha256:" + document.file_sha256,
        gate_projection=gate,
        up_projection=up,
        down_projection=down,
    )
    publish_ms = (time.perf_counter() - published_at) * 1000

    mesh = ResidentExpertMesh(
        coordinator_id="native-root",
        experts=(ExpertRecord(key, weight_bytes, content_id),),
        nodes=(
            MeshNodeProfile("native-root", 1_000_000_000, 0, 100.0,
                            expert_workspace_bytes_per_token=4 * intermediate),
            MeshNodeProfile("browser-stage", 1_000_000_000, 0, 1.0,
                            expert_workspace_bytes_per_token=4 * intermediate),
        ),
        links=(MeshLinkProfile("native-root", "browser-stage", 0.1, 1_000),),
        local_ram_keys=(),
        local_gpu_keys=(),
        replicas=(ResidentExpertReplica(key, "browser-stage", content_id),),
        local_weight_buffer_bytes=0,
        activation_bytes_per_token=hidden_size * 4,
        require_local_ram_fallback=False,
    )
    # A reproducible representative activation at the actual model width.
    # It is not claimed to be an activation captured from a full prompt run.
    positions = torch.arange(args.rows * hidden_size, dtype=torch.float32)
    activation = torch.sin(positions.reshape(args.rows, hidden_size) * 0.013)
    routing = AuthoritativeRouting(
        torch.zeros((args.rows, 1), dtype=torch.long),
        torch.ones((args.rows, 1), dtype=torch.float32),
    )
    expected = F.linear(F.silu(F.linear(activation, gate)) * F.linear(activation, up), down)
    route_samples_ms: list[float] = []
    max_abs_error = 0.0
    try:
        for _ in range(args.repeats):
            started = time.perf_counter()
            actual, plan = mesh.execute_layer(activation, 0, routing, {"browser-stage": owner})
            route_samples_ms.append((time.perf_counter() - started) * 1000)
            paths = sorted({dispatch.path for dispatch in plan.dispatches})
            if paths != ["remote-resident"]:
                raise AssertionError(f"expected browser-owned remote stage, got {paths}")
            torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
            max_abs_error = max(max_abs_error, float((actual - expected).abs().max()))
    finally:
        mesh.close()

    direct_http_ms: list[float] = []
    browser_reported_ms: list[float] = []
    for _ in range(args.diagnostic_executions):
        payload = json.dumps({
            "artifactId": artifact_id,
            "rows": args.rows,
            "hiddenSize": hidden_size,
            "activationsBase64": base64.b64encode(activation.numpy().tobytes()).decode("ascii"),
        }).encode("utf-8")
        request = Request(
            args.coordinator + "/internal/v1/mobile/experts/execute",
            data=payload,
            method="POST",
            headers={
                "authorization": "Bearer local-browser-split-probe",
                "content-type": "application/json",
            },
        )
        started = time.perf_counter()
        with urlopen(request, timeout=90) as response:
            diagnostic = json.load(response)
        direct_http_ms.append((time.perf_counter() - started) * 1000)
        if args.require_webgpu and diagnostic["backend"] != "webgpu":
            raise AssertionError("diagnostic execution fell back from WebGPU")
        browser_reported_ms.append(float(diagnostic["durationMs"]))
        diagnostic_output = torch.frombuffer(bytearray(base64.b64decode(
            diagnostic["outputBase64"])), dtype=torch.float32).reshape(args.rows, hidden_size)
        torch.testing.assert_close(diagnostic_output, expected, rtol=2e-4, atol=2e-4)
        max_abs_error = max(max_abs_error, float((diagnostic_output - expected).abs().max()))

    with urlopen(args.coordinator + "/mobile/v1/workers", timeout=10) as response:
        workers = json.load(response)["data"]
    residents = [
        worker for worker in workers
        if any(item["artifactId"] == artifact_id for item in worker["residentExperts"])
    ]
    if len(residents) != 2:
        raise AssertionError(f"expected two browser replicas, got {len(residents)}")
    if len({worker["clientId"] for worker in residents}) != 2:
        raise AssertionError("browser replicas must have distinct client IDs")
    if any(worker["status"] != "online" or not worker["visible"]
           or worker["verifiedTasks"] < 1 + args.repeats + args.diagnostic_executions
           for worker in residents):
        raise AssertionError("browser replicas did not verify every execution")
    backends = sorted(worker["backend"] for worker in residents)
    if args.require_webgpu and backends != ["webgpu", "webgpu"]:
        raise AssertionError(f"browser workers used {backends}, expected WebGPU")

    result = {
        "schema": "mycellios-local-browser-split-probe/1",
        "modelSha256": document.file_sha256,
        "model": "SmolLM2-135M-Instruct-F16",
        "assignedPart": "block-0 SwiGLU MLP",
        "layer": 0,
        "rows": args.rows,
        "hiddenSize": hidden_size,
        "intermediateSize": intermediate,
        "weightBytes": weight_bytes,
        "artifactId": artifact_id,
        "routePaths": paths,
        "browserReplicaCount": len(residents),
        "browserBackends": backends,
        "browserGpuDescriptions": sorted(worker["capabilities"]["gpuDescription"] for worker in residents),
        "browserVerifiedTasks": sorted(worker["verifiedTasks"] for worker in residents),
        "activationBytesEachWay": activation.numel() * 4,
        "publishMs": round(publish_ms, 3),
        "routeSamplesMs": [round(value, 3) for value in route_samples_ms],
        "routeMedianMs": round(median(route_samples_ms), 3),
        "routeWarmMedianMs": round(median(route_samples_ms[1:]), 3) if len(route_samples_ms) > 1 else None,
        "directHttpSamplesMs": [round(value, 3) for value in direct_http_ms],
        "browserReportedExecutionSamplesMs": [round(value, 3) for value in browser_reported_ms],
        "maxAbsError": max_abs_error,
        "numericalParity": True,
        "samePhysicalHost": True,
        "completeModelGeneration": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
