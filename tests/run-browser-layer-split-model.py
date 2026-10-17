"""Verify one browser-owned layer inside a two-stage native Qwen3 pipeline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from distributed_runtime.model import StageModelSpec, StageRunner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--bridge-config", type=Path, required=True)
    parser.add_argument("--receipt", type=Path,
                        default=Path("runtime/browser-layer-split-model-receipt.json"))
    args = parser.parse_args()
    os.environ["MYCELLIOS_BROWSER_LAYER_BRIDGE_FILE"] = str(args.bridge_config.resolve())
    model = str(args.model.resolve())
    first = StageRunner(StageModelSpec(
        model_name=model, layer_start=0, layer_end=1,
        total_layers=28, threads=4, kv_cache="dynamic", decode_attention="stock",
    ), device="cpu")
    second = StageRunner(StageModelSpec(
        model_name=model, layer_start=1, layer_end=28,
        total_layers=28, threads=4, kv_cache="dynamic", decode_attention="stock",
    ), device="cpu")
    try:
        if getattr(first, "_browser_layer_bridge", None) is not None:
            raise AssertionError("browser layer attached outside its assigned stage")
        bridge = getattr(second, "_browser_layer_bridge", None)
        if bridge is None:
            raise AssertionError("browser layer did not attach to its stage")
        first.begin(1000)
        second.begin(1000)
        tokens: list[int] = []
        for index in range(4):
            ids = [1, 2, 3] if index == 0 else [tokens[-1]]
            hidden = first.forward_ids(1000, torch.tensor([ids], dtype=torch.long))
            _, predicted = second.forward_hidden(1000, hidden, token_mode="last")
            tokens.append(int(predicted))
        if tokens != [16, 17, 18, 19] or bridge.browser_forwards != 4:
            raise AssertionError(f"split model did not use browser stage: {tokens}")
        if bridge.native_forwards != 0:
            raise AssertionError("split model unexpectedly used native fallback")
        receipt = {"schema": "mycellios-browser-layer-split-model-receipt/1",
                   "nativeRanges": [[0, 1], [1, 28]],
                   "browserLayer": 1,
                   "tokens": tokens, "browserForwards": bridge.browser_forwards}
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(receipt))
    finally:
        if 1000 in first.active_requests:
            first.end(1000)
        if 1000 in second.active_requests:
            second.end(1000)
        first.close()
        second.close()


if __name__ == "__main__":
    main()
