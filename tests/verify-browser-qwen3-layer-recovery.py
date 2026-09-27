"""Verify native continuation from the actual browser-returned Qwen3 KV deltas."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers import Qwen3Config
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--fixture", default="runtime/browser-qwen3-real-layer-fixture.json")
    parser.add_argument("--browser-receipt", required=True)
    args = parser.parse_args()
    model_path = Path(args.model)
    fixture = json.loads(Path(args.fixture).read_text(encoding="utf-8"))
    browser = json.loads(Path(args.browser_receipt).read_text(encoding="utf-8"))
    if browser.get("schema") != "mycellios-local-browser-layer-receipt/1" \
            or not browser.get("realCheckpoint") \
            or browser.get("backend") not in ("wasm", "webgpu"):
        raise ValueError("browser receipt is not a real Qwen3 layer result")
    if len(browser["steps"]) != len(fixture["steps"]):
        raise ValueError("browser receipt has an unexpected step count")
    config = Qwen3Config.from_pretrained(model_path)
    config._attn_implementation = "eager"
    layer = Qwen3DecoderLayer(config, layer_idx=0).eval()
    with safe_open(model_path / "model.safetensors", framework="pt", device="cpu") as weights:
        prefix = "model.layers.0."
        state = {name[len(prefix):]: weights.get_tensor(name).float()
                 for name in weights.keys() if name.startswith(prefix)}
    layer.load_state_dict(state, strict=True)
    torch.set_num_threads(4)
    width = fixture["hiddenSize"]
    head_dim = fixture["headDim"]
    heads = fixture["keyValueHeads"]
    recovered = []
    for cut in range(1, len(fixture["steps"])):
        cache = DynamicCache()
        total = 0
        for index in range(cut):
            step = fixture["steps"][index]
            result = browser["steps"][index]
            tokens = step["tokens"]
            shape = (1, heads, tokens, head_dim)
            new_key = np.asarray(result["newKey"], dtype=np.float32).reshape(shape)
            new_value = np.asarray(result["newValue"], dtype=np.float32).reshape(shape)
            cache.update(torch.from_numpy(new_key), torch.from_numpy(new_value), 0)
            total += tokens
        step = fixture["steps"][cut]
        tokens = step["tokens"]
        next_total = total + tokens
        hidden = torch.tensor(step["hidden"], dtype=torch.float32).reshape(1, tokens, width)
        cos = torch.tensor(step["cos"], dtype=torch.float32).reshape(1, tokens, head_dim)
        sin = torch.tensor(step["sin"], dtype=torch.float32).reshape(1, tokens, head_dim)
        mask = torch.tensor(step["mask"], dtype=torch.float32).reshape(1, 1, tokens, next_total)
        expected = np.asarray(step["expected"], dtype=np.float32)
        with torch.inference_mode():
            actual = layer(hidden, attention_mask=mask,
                           position_embeddings=(cos, sin), past_key_values=cache)
        error = float(np.max(np.abs(actual.numpy().reshape(-1) - expected)))
        recovered.append({"afterBrowserSteps": cut, "maxAbsError": error})
        if error > 2e-4:
            raise AssertionError(f"native continuation differs after {cut} browser steps: {error}")
    with (model_path / "model.safetensors").open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    summary = {
        "schema": "mycellios-local-browser-layer-recovery/1",
        "backend": browser["backend"],
        "modelSha256": digest,
        "checks": recovered,
        "maxAbsError": max(item["maxAbsError"] for item in recovered),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
