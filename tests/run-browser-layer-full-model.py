"""Generate with a model while one decoder layer runs in a browser."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import torch
from transformers import AutoConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from distributed_runtime.browser_expert_owner import BrowserExpertOwnerError
from distributed_runtime.browser_layer_bridge import attach_browser_layer_bridge
from distributed_runtime.model import StageModelSpec, StageRunner


def generate(runner: StageRunner, request_id: int, prompt: list[int],
             *, inject_failure: bool = False) -> tuple[list[int], list[torch.Tensor]]:
    runner.begin(request_id)
    outputs: list[torch.Tensor] = []
    tokens: list[int] = []
    bridge = getattr(runner, "_browser_layer_bridge", None)
    original = bridge.owner._request_json if bridge is not None else None
    try:
        for index in range(4):
            if inject_failure and index == 2 and bridge is not None:
                def fail_execute(method, path, payload):
                    if path.endswith("/execute"):
                        raise BrowserExpertOwnerError("injected browser disconnect")
                    return original(method, path, payload)
                bridge.owner._request_json = fail_execute
            ids = prompt if index == 0 else [tokens[-1]]
            hidden, predicted = runner.forward_ids_with_tokens(
                request_id, torch.tensor([ids], dtype=torch.long), token_mode="last",
            )
            outputs.append(hidden.detach().cpu().clone())
            tokens.append(int(predicted))
    finally:
        if bridge is not None:
            bridge.owner._request_json = original
        runner.end(request_id)
    return tokens, outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--bridge-config", type=Path, required=True)
    parser.add_argument("--auto-discovery", action="store_true")
    parser.add_argument("--receipt", type=Path,
                        default=Path("runtime/browser-layer-full-model-receipt.json"))
    args = parser.parse_args()
    total_layers = int(AutoConfig.from_pretrained(args.model).num_hidden_layers)
    os.environ.pop("MYCELLIOS_BROWSER_LAYER_BRIDGE_FILE", None)
    os.environ.pop("MYCELLIOS_BROWSER_LAYER_COORDINATOR_URL", None)
    runner = StageRunner(StageModelSpec(
        model_name=str(args.model.resolve()), layer_start=0, layer_end=total_layers,
        total_layers=total_layers, threads=4, kv_cache="dynamic", decode_attention="stock",
    ), device="cpu")
    try:
        reference_tokens, reference_hidden = generate(runner, 700, [1, 2, 3])
        if args.auto_discovery:
            config = json.loads(args.bridge_config.read_text(encoding="utf-8"))
            os.environ["MYCELLIOS_BROWSER_LAYER_COORDINATOR_URL"] = config["coordinatorUrl"]
        else:
            os.environ["MYCELLIOS_BROWSER_LAYER_BRIDGE_FILE"] = str(args.bridge_config.resolve())
        if not attach_browser_layer_bridge(runner):
            raise AssertionError("browser layer bridge was not attached")
        bridge = runner._browser_layer_bridge
        regular_tokens, regular_hidden = generate(runner, 701, [1, 2, 3])
        regular_forwards = bridge.browser_forwards
        recovered_tokens, recovered_hidden = generate(
            runner, 702, [1, 2, 3], inject_failure=True)
        if regular_forwards != 4 or bridge.browser_forwards != 6:
            raise AssertionError("browser did not execute the expected model steps")
        if bridge.native_forwards != 2:
            raise AssertionError("injected browser failure did not use native fallback")
        for name, tokens, outputs in (
            ("browser", regular_tokens, regular_hidden),
            ("recovered", recovered_tokens, recovered_hidden),
        ):
            if tokens != reference_tokens:
                raise AssertionError(f"{name} changed generated token IDs")
            errors = [float(torch.max(torch.abs(a - b)).item())
                      for a, b in zip(outputs, reference_hidden)]
            if max(errors) > 2e-3:
                raise AssertionError(f"{name} hidden states diverged: {errors}")
        receipt = {
            "schema": "mycellios-browser-layer-full-model-receipt/1",
            "model": str(args.model.resolve()),
            "tokens": reference_tokens,
            "browserForwards": bridge.browser_forwards,
            "nativeFallbackForwards": bridge.native_forwards,
            "browserMaxHiddenError": max(float(torch.max(torch.abs(a - b)).item())
                                         for a, b in zip(regular_hidden, reference_hidden)),
            "recoveredMaxHiddenError": max(float(torch.max(torch.abs(a - b)).item())
                                           for a, b in zip(recovered_hidden, reference_hidden)),
        }
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(receipt))
    finally:
        runner.close()


if __name__ == "__main__":
    main()
