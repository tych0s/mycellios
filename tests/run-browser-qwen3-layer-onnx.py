"""Export and check one complete Qwen3 transformer layer with explicit KV state.

This is a local stage-format experiment, not a deployed browser worker. The
resulting ONNX graph is intended for a subsequent WebGPU browser execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch import nn
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer, Qwen3RotaryEmbedding, apply_rotary_pos_emb, repeat_kv,
)
from safetensors import safe_open
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer, LlamaRotaryEmbedding,
    apply_rotary_pos_emb as llama_rotary, repeat_kv as llama_repeat_kv,
)


class Qwen3LayerWithKv(nn.Module):
    def __init__(self, layer: nn.Module) -> None:
        super().__init__()
        self.layer = layer

    def forward(self, hidden, cos, sin, past_key, past_value, attention_mask):
        attention = self.layer.self_attn
        batch, tokens, width = hidden.shape
        normalized = self.layer.input_layernorm(hidden)
        shape = (batch, tokens, -1, attention.head_dim)
        query = attention.q_norm(attention.q_proj(normalized).reshape(shape)).transpose(1, 2)
        key = attention.k_norm(attention.k_proj(normalized).reshape(shape)).transpose(1, 2)
        value = attention.v_proj(normalized).reshape(shape).transpose(1, 2)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        new_key, new_value = key, value
        key = torch.cat((past_key, key), dim=2)
        value = torch.cat((past_value, value), dim=2)
        repeated_key = repeat_kv(key, attention.num_key_value_groups)
        repeated_value = repeat_kv(value, attention.num_key_value_groups)
        scores = torch.matmul(query, repeated_key.transpose(2, 3)) * attention.scaling
        scores = scores + attention_mask
        weights = torch.softmax(scores.float(), dim=-1).to(query.dtype)
        attended = torch.matmul(weights, repeated_value)
        attended = attended.transpose(1, 2).reshape(batch, tokens, -1)
        hidden = hidden + attention.o_proj(attended)
        hidden = hidden + self.layer.mlp(self.layer.post_attention_layernorm(hidden))
        return hidden, key, value, new_key, new_value


class LlamaLayerWithKv(nn.Module):
    def __init__(self, layer: nn.Module) -> None:
        super().__init__()
        self.layer = layer

    def forward(self, hidden, cos, sin, past_key, past_value, attention_mask):
        attention = self.layer.self_attn
        batch, tokens, _ = hidden.shape
        normalized = self.layer.input_layernorm(hidden)
        shape = (batch, tokens, -1, attention.head_dim)
        query = attention.q_proj(normalized).reshape(shape).transpose(1, 2)
        key = attention.k_proj(normalized).reshape(shape).transpose(1, 2)
        value = attention.v_proj(normalized).reshape(shape).transpose(1, 2)
        query, key = llama_rotary(query, key, cos, sin)
        new_key, new_value = key, value
        key = torch.cat((past_key, key), dim=2)
        value = torch.cat((past_value, value), dim=2)
        scores = torch.matmul(query, llama_repeat_kv(key, attention.num_key_value_groups)
                              .transpose(2, 3)) * attention.scaling
        weights = torch.softmax(scores + attention_mask, dim=-1).to(query.dtype)
        attended = torch.matmul(weights, llama_repeat_kv(value, attention.num_key_value_groups))
        attended = attended.transpose(1, 2).reshape(batch, tokens, -1)
        hidden = hidden + attention.o_proj(attended)
        hidden = hidden + self.layer.mlp(self.layer.post_attention_layernorm(hidden))
        return hidden, key, value, new_key, new_value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runtime/browser-qwen3-layer.onnx")
    parser.add_argument("--receipt", default="runtime/browser-qwen3-layer-onnx-receipt.json")
    parser.add_argument("--fixture", default="runtime/browser-qwen3-layer-fixture.json")
    parser.add_argument("--model", help="Local Qwen3 checkpoint directory for a real-weight layer")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--architecture", choices=("qwen3", "llama"), default="qwen3")
    args = parser.parse_args()
    torch.manual_seed(877)
    torch.set_num_threads(4)
    if args.model:
        model_path = Path(args.model)
        config_class = Qwen3Config if args.architecture == "qwen3" else LlamaConfig
        layer_class = Qwen3DecoderLayer if args.architecture == "qwen3" else LlamaDecoderLayer
        rotary_class = Qwen3RotaryEmbedding if args.architecture == "qwen3" else LlamaRotaryEmbedding
        config = config_class.from_pretrained(model_path)
        config._attn_implementation = "eager"
        if not 0 <= args.layer < config.num_hidden_layers:
            raise ValueError("selected layer is outside the checkpoint")
        layer = layer_class(config, layer_idx=args.layer).eval()
        checkpoint = model_path / "model.safetensors"
        with safe_open(checkpoint, framework="pt", device="cpu") as weights:
            prefix = f"model.layers.{args.layer}."
            state = {name[len(prefix):]: weights.get_tensor(name).float()
                     for name in weights.keys() if name.startswith(prefix)}
        layer.load_state_dict(state, strict=True)
        rotary = rotary_class(config)
        model_name = str(model_path)
        digest = hashlib.sha256()
        with checkpoint.open("rb") as source:
            for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        model_digest = digest.hexdigest()
    else:
        config_class = Qwen3Config if args.architecture == "qwen3" else LlamaConfig
        model_class = Qwen3ForCausalLM if args.architecture == "qwen3" else LlamaForCausalLM
        config = config_class(
            vocab_size=32, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=8, max_position_embeddings=64,
            _attn_implementation="eager",
        )
        model = model_class(config).eval()
        if not 0 <= args.layer < config.num_hidden_layers:
            raise ValueError("selected layer is outside the generated model")
        layer = model.model.layers[args.layer]
        rotary = model.model.rotary_emb
        model_name = f"generated tiny {args.architecture}"
        model_digest = None
    wrapped = (Qwen3LayerWithKv(layer) if args.architecture == "qwen3"
               else LlamaLayerWithKv(layer)).eval()
    heads = config.num_key_value_heads
    dim = layer.self_attn.head_dim
    dummy = torch.zeros(1, 1, config.hidden_size)
    dummy_cos = torch.ones(1, 1, dim)
    dummy_sin = torch.zeros(1, 1, dim)
    dummy_kv = torch.zeros(1, heads, 2, dim)
    dummy_mask = torch.zeros(1, 1, 1, 3)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapped, (dummy, dummy_cos, dummy_sin, dummy_kv, dummy_kv, dummy_mask), output,
        input_names=["hidden", "cos", "sin", "past_key", "past_value", "attention_mask"],
        output_names=["output", "key", "value", "new_key", "new_value"], opset_version=18,
        dynamic_axes={
            "hidden": {1: "tokens"}, "cos": {1: "tokens"}, "sin": {1: "tokens"},
            "past_key": {2: "past_tokens"}, "past_value": {2: "past_tokens"},
            "attention_mask": {2: "tokens", 3: "total_tokens"},
            "output": {1: "tokens"}, "key": {2: "total_tokens"},
            "value": {2: "total_tokens"},
            "new_key": {2: "tokens"}, "new_value": {2: "tokens"},
        },
        dynamo=False,
    )
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    cache = DynamicCache()
    browser_key = np.zeros((1, heads, 0, dim), dtype=np.float32)
    browser_value = np.zeros_like(browser_key)
    returned_key = browser_key.copy()
    returned_value = browser_value.copy()
    checks = []
    fixtures = []
    for tokens in (3, *([1] * 10)):
        hidden = torch.randn(1, tokens, config.hidden_size)
        start = browser_key.shape[2]
        positions = torch.arange(start, start + tokens).unsqueeze(0)
        cos, sin = rotary(hidden, positions)
        total = start + tokens
        causal = torch.arange(total)[None, :] <= torch.arange(start, total)[:, None]
        mask = torch.where(causal, 0.0, -1e9)[None, None, :, :]
        with torch.inference_mode():
            expected = layer(hidden, attention_mask=mask, position_embeddings=(cos, sin),
                             past_key_values=cache)
            fallback_cache = DynamicCache()
            if start:
                fallback_cache.update(torch.from_numpy(returned_key),
                                      torch.from_numpy(returned_value), args.layer)
            recovered = layer(hidden, attention_mask=mask,
                              position_embeddings=(cos, sin),
                              past_key_values=fallback_cache)
        recovery_error = float(torch.max(torch.abs(recovered - expected)).item())
        native_new_key = cache.layers[args.layer].keys[:, :, -tokens:, :].numpy()
        native_new_value = cache.layers[args.layer].values[:, :, -tokens:, :].numpy()
        fixtures.append({
            "tokens": tokens,
            "hidden": hidden.reshape(-1).tolist(),
            "cos": cos.reshape(-1).tolist(),
            "sin": sin.reshape(-1).tolist(),
            "mask": mask.reshape(-1).tolist(),
            "expected": expected.reshape(-1).tolist(),
            "expectedNewKey": native_new_key.reshape(-1).tolist(),
            "expectedNewValue": native_new_value.reshape(-1).tolist(),
        })
        actual, browser_key, browser_value, new_key, new_value = session.run(None, {
            "hidden": hidden.numpy(), "cos": cos.numpy(), "sin": sin.numpy(),
            "past_key": browser_key, "past_value": browser_value,
            "attention_mask": mask.numpy(),
        })
        error = float(np.max(np.abs(actual - expected.numpy())))
        key_error = float(np.max(np.abs(new_key - native_new_key)))
        value_error = float(np.max(np.abs(new_value - native_new_value)))
        returned_key = np.concatenate((returned_key, new_key), axis=2)
        returned_value = np.concatenate((returned_value, new_value), axis=2)
        assembled_key_error = float(np.max(np.abs(returned_key - browser_key)))
        assembled_value_error = float(np.max(np.abs(returned_value - browser_value)))
        checks.append({"tokens": tokens, "totalTokens": total,
                       "maxAbsError": error, "newKeyError": key_error,
                       "newValueError": value_error,
                       "recoveryError": recovery_error})
        if error > 2e-5:
            raise AssertionError(f"{args.architecture} ONNX layer differs from native: {error}")
        if key_error > 5e-4 or value_error > 2e-5:
            raise AssertionError(f"{args.architecture} ONNX incremental cache differs from native: "
                                 f"key={key_error}, value={value_error}, tokens={tokens}")
        if assembled_key_error > 1e-6 or assembled_value_error > 1e-6:
            raise AssertionError("returned incremental cache does not reconstruct browser cache")
        if recovery_error > 2e-4:
            raise AssertionError(f"native fallback differs after browser cache: {recovery_error}")
    onnx_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    receipt = {"schema": "mycellios-local-browser-layer-onnx/2",
               "architecture": args.architecture,
               "model": model_name, "modelSha256": model_digest, "checks": checks,
               "layer": args.layer, "onnxSha256": onnx_sha256,
               "onnxBytes": output.stat().st_size, "browserExecuted": False}
    receipt_path = Path(args.receipt)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    fixture_path = Path(args.fixture)
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    fixture_path.write_text(json.dumps({
        "schema": "mycellios-local-browser-layer-fixture/2",
        "architecture": args.architecture,
        "layer": args.layer, "modelSha256": model_digest,
        "onnxSha256": onnx_sha256,
        "hiddenSize": config.hidden_size, "headDim": dim, "keyValueHeads": heads,
        "steps": fixtures,
    }) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
