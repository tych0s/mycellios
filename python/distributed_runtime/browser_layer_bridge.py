"""Opt-in decoder layer execution on one browser node in a native stage.

The stage keeps its native weights and authoritative KV. Browser-returned KV is
committed only after the coordinator validates the complete response. A failed
browser request continues through the native layer for that generation.
"""

from __future__ import annotations

import base64
from contextlib import contextmanager
from contextvars import ContextVar
import json
import os
from pathlib import Path
import re
from typing import TYPE_CHECKING, Protocol
from urllib.parse import urlparse
from uuid import uuid4, uuid5

import torch
from torch import nn

from .browser_expert_owner import BrowserExpertOwner, BrowserExpertOwnerError

if TYPE_CHECKING:
    from .model import StageRunner


_request_id: ContextVar[int | None] = ContextVar("browser_layer_request", default=None)


@contextmanager
def browser_layer_request(request_id: int):
    token = _request_id.set(request_id)
    try:
        yield
    finally:
        _request_id.reset(token)


def run_browser_stage(runner: "StageRunner", request_id: int, **kwargs):
    with browser_layer_request(request_id):
        return runner.base(**kwargs)


def reset_browser_stage(runner: "StageRunner", request_id: int,
                        *, disable: bool = False, end: bool = False) -> None:
    bridge = getattr(runner, "_browser_layer_bridge", None)
    if bridge is not None:
        if end:
            bridge.end(request_id)
        else:
            bridge.reset(request_id, disable=disable)


def close_browser_stage(runner: "StageRunner") -> None:
    bridge = getattr(runner, "_browser_layer_bridge", None)
    if bridge is not None:
        for request_id in tuple(runner.active_requests):
            bridge.end(request_id)


def initialize_browser_layer_stage(runner: "StageRunner") -> None:
    try:
        attach_browser_layer_bridge(runner)
    except BaseException:
        runner.close()
        raise


class BrowserLayerAdapter(Protocol):
    """Native boundary for a model family using the browser tensor ABI."""

    def supports(self, layer: nn.Module) -> bool: ...

    def position(self, cache: object, local_index: int) -> int: ...

    def auxiliary(self, hidden: torch.Tensor, attention_mask: object,
                  position_embeddings: object, position: int,
                  tokens: int) -> dict[str, str]: ...

    def commit(self, layer: nn.Module, response: dict, cache: object,
               local_index: int, hidden: torch.Tensor, tokens: int) -> None: ...


_adapters: dict[str, BrowserLayerAdapter] = {}


def register_browser_layer_adapter(name: str, adapter: BrowserLayerAdapter) -> None:
    """Register a trusted native adapter before constructing a stage runner."""
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", name) or name in _adapters:
        raise ValueError("invalid or duplicate browser layer adapter name")
    _adapters[name] = adapter


def _adapter_for(layer: nn.Module) -> BrowserLayerAdapter | None:
    return next((adapter for adapter in _adapters.values() if adapter.supports(layer)), None)


class RopeKvAdapter:
    """Qwen3 and Llama share a hidden, rotary-input, incremental-KV boundary."""

    def supports(self, layer: nn.Module) -> bool:
        return type(layer).__name__ in ("Qwen3DecoderLayer", "LlamaDecoderLayer") \
            and type(getattr(layer, "mlp", None)).__name__ in ("Qwen3MLP", "LlamaMLP")

    def position(self, cache: object, local_index: int) -> int:
        return int(cache.layers[local_index].get_seq_length())

    def auxiliary(self, hidden: torch.Tensor, attention_mask: object,
                  position_embeddings: object, position: int,
                  tokens: int) -> dict[str, str]:
        if not isinstance(position_embeddings, tuple) or len(position_embeddings) != 2:
            raise ValueError("rotary position embeddings are required")
        cos, sin = position_embeddings
        if attention_mask is None:
            rows = torch.arange(position, position + tokens, device=hidden.device)
            columns = torch.arange(position + tokens, device=hidden.device)
            mask = torch.where(columns[None, :] <= rows[:, None], 0.0, -1e9)
            mask = mask.reshape(1, 1, tokens, position + tokens)
        elif isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 4:
            mask = attention_mask[:, :, :, :position + tokens]
            if tuple(mask.shape) != (1, 1, tokens, position + tokens):
                raise ValueError("unsupported attention mask shape")
        else:
            raise ValueError("unsupported attention mask")
        return {"cos": _encode(cos), "sin": _encode(sin),
                "attention_mask": _encode(mask)}

    def commit(self, layer: nn.Module, response: dict, cache: object,
               local_index: int, hidden: torch.Tensor, tokens: int) -> None:
        state = response.get("stateBase64")
        if not isinstance(state, dict) or set(state) != {"key", "value"}:
            raise BrowserExpertOwnerError("browser layer returned the wrong state")
        heads = int(layer.self_attn.config.num_key_value_heads)
        head_dim = int(layer.self_attn.head_dim)
        key = _decode(state["key"], (1, heads, tokens, head_dim))
        value = _decode(state["value"], (1, heads, tokens, head_dim))
        cache.update(key.to(device=hidden.device, dtype=hidden.dtype),
                     value.to(device=hidden.device, dtype=hidden.dtype), local_index)


register_browser_layer_adapter("rope_kv", RopeKvAdapter())


class BrowserDecoderLayer(nn.Module):
    """One complete decoder layer, backed by a browser when it is resident."""

    def __init__(self, native: nn.Module, owner: BrowserExpertOwner,
                 artifact_id: str, local_index: int, max_context_tokens: int,
                 adapter: BrowserLayerAdapter | None = None) -> None:
        super().__init__()
        self.native = native
        self.owner = owner
        self.artifact_id = artifact_id
        self.local_index = local_index
        self.max_context_tokens = max_context_tokens
        self.adapter = adapter or _adapter_for(native)
        if self.adapter is None:
            raise ValueError("no browser layer adapter supports this native layer")
        self.namespace = uuid4()
        self.disabled_requests: set[int] = set()
        self.started_requests: set[int] = set()
        self.browser_forwards = 0
        self.native_forwards = 0

    def request_uuid(self, request_id: int) -> str:
        return str(uuid5(self.namespace, str(request_id)))

    def reset(self, request_id: int, *, disable: bool = False) -> None:
        if disable:
            self.disabled_requests.add(request_id)
        if request_id not in self.started_requests:
            return
        self.started_requests.discard(request_id)
        try:
            self.owner._request_json("POST", "/internal/v1/mobile/layers/reset", {
                "artifactId": self.artifact_id, "requestId": self.request_uuid(request_id),
            })
        except BrowserExpertOwnerError:
            pass

    def end(self, request_id: int) -> None:
        self.reset(request_id)
        self.disabled_requests.discard(request_id)

    def forward(self, hidden_states: torch.Tensor, attention_mask=None,
                position_ids=None, past_key_values=None, use_cache=False,
                position_embeddings=None, **kwargs) -> torch.Tensor:
        request_id = _request_id.get()
        native_args = (hidden_states,)
        native_kwargs = dict(attention_mask=attention_mask, position_ids=position_ids,
                             past_key_values=past_key_values, use_cache=use_cache,
                             position_embeddings=position_embeddings, **kwargs)
        if request_id is None or request_id in self.disabled_requests:
            self.native_forwards += 1
            return self.native(*native_args, **native_kwargs)
        if (hidden_states.ndim != 3 or hidden_states.shape[0] != 1
                or past_key_values is None or not use_cache):
            self.disabled_requests.add(request_id)
            self.native_forwards += 1
            return self.native(*native_args, **native_kwargs)
        try:
            position = self.adapter.position(past_key_values, self.local_index)
        except (AttributeError, IndexError, TypeError, ValueError):
            self.disabled_requests.add(request_id)
            self.native_forwards += 1
            return self.native(*native_args, **native_kwargs)
        tokens = int(hidden_states.shape[1])
        if position + tokens > self.max_context_tokens or tokens < 1:
            self.disabled_requests.add(request_id)
            self.native_forwards += 1
            return self.native(*native_args, **native_kwargs)
        # A browser request starts at zero. A restored/forked native cache has
        # no corresponding browser KV and must stay on the native path.
        if position > 0 and request_id not in self.started_requests:
            self.disabled_requests.add(request_id)
            self.native_forwards += 1
            return self.native(*native_args, **native_kwargs)
        try:
            auxiliary = self.adapter.auxiliary(hidden_states, attention_mask,
                                               position_embeddings, position, tokens)
            if position == 0:
                ready = self.owner._request_json("POST", "/internal/v1/mobile/layers/prepare",
                                                 {"artifactId": self.artifact_id})
                if ready.get("resident") is not True:
                    raise BrowserExpertOwnerError("browser layer is not resident")
            self.started_requests.add(request_id)
            response = self.owner._request_json("POST", "/internal/v1/mobile/layers/execute", {
                "artifactId": self.artifact_id,
                "requestId": self.request_uuid(request_id),
                "position": position, "tokens": tokens,
                "hiddenBase64": _encode(hidden_states),
                "auxiliaryBase64": auxiliary,
            })
            width = int(hidden_states.shape[-1])
            output = _decode(response.get("outputBase64"), (1, tokens, width))
            if (response.get("artifactId") != self.artifact_id
                    or response.get("requestId") != self.request_uuid(request_id)
                    or response.get("position") != position
                    or response.get("tokens") != tokens):
                raise BrowserExpertOwnerError("browser layer lease identity mismatch")
            self.adapter.commit(self.native, response, past_key_values,
                                self.local_index, hidden_states, tokens)
            self.browser_forwards += 1
            return output.to(device=hidden_states.device, dtype=hidden_states.dtype)
        except (BrowserExpertOwnerError, ValueError, KeyError, TypeError):
            self.reset(request_id, disable=True)
            self.native_forwards += 1
            return self.native(*native_args, **native_kwargs)


def attach_browser_layer_bridge(runner: "StageRunner") -> bool:
    path = os.environ.get("MYCELLIOS_BROWSER_LAYER_BRIDGE_FILE", "").strip()
    if path:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    else:
        coordinator_url = os.environ.get(
            "MYCELLIOS_BROWSER_LAYER_COORDINATOR_URL", "").strip()
        if not coordinator_url:
            return False
        if runner.compute_dtype != torch.float32 or runner.spec.quantize \
                or runner.spec.compile_mode:
            return False
        identity = runner.executor_manifest.model_identity
        _require_coordinator_origin(coordinator_url)
        owner = BrowserExpertOwner("browser-layer-discovery", coordinator_url,
                                   timeout_seconds=10)
        result = owner._request_json(
            "GET", "/internal/v1/mobile/layers/lookup"
            f"?modelDigest={identity}&layerStart={runner.spec.layer_start}"
            f"&layerEnd={runner.spec.layer_end}")
        candidates = result.get("data")
        if not isinstance(candidates, list):
            raise ValueError("browser layer registry returned invalid data")
        matched: list[dict] = []
        seen_layers: set[int] = set()
        for item in candidates:
            if not isinstance(item, dict) or set(item) != {
                    "artifactId", "modelDigest", "layer", "maxContextTokens"}:
                raise ValueError("browser layer registry returned invalid identity")
            layer = item["layer"]
            if (not isinstance(layer, int) or isinstance(layer, bool)
                    or not runner.spec.layer_start <= layer < runner.spec.layer_end
                    or item["modelDigest"] != identity):
                raise ValueError("browser layer registry returned a wrong model range")
            if layer in seen_layers:
                raise ValueError("multiple browser artifacts registered for one model layer")
            seen_layers.add(layer)
            if _adapter_for(runner.base.layers[layer - runner.spec.layer_start]):
                matched.append(item)
        if not matched:
            return False
        selected = min(matched, key=lambda item: item["layer"])
        document = {
            "schema": "mycellios-browser-layer-bridge/1",
            "artifactIdentity": identity, "coordinatorUrl": coordinator_url,
            "layer": selected["layer"], "artifactId": selected["artifactId"],
            "maxContextTokens": selected["maxContextTokens"],
        }
    if not isinstance(document, dict) or set(document) != {
        "schema", "artifactIdentity", "coordinatorUrl", "layer", "artifactId",
        "maxContextTokens",
    } or document["schema"] != "mycellios-browser-layer-bridge/1":
        raise ValueError("unsupported browser layer bridge configuration")
    identity = document["artifactIdentity"]
    artifact_id = document["artifactId"]
    if (not isinstance(identity, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", identity)
            or identity != runner.executor_manifest.model_identity
            or not isinstance(artifact_id, str)
            or not re.fullmatch(r"[0-9a-f]{64}", artifact_id)):
        raise ValueError("browser layer requires the exact model and artifact identity")
    url = document["coordinatorUrl"]
    if not isinstance(url, str):
        raise ValueError("browser layer coordinator URL must be an origin")
    _require_coordinator_origin(url)
    layer_index = document["layer"]
    context = document["maxContextTokens"]
    if (not isinstance(layer_index, int) or isinstance(layer_index, bool)
            or not isinstance(context, int) or isinstance(context, bool)
            or context < 1):
        raise ValueError("browser layer index and context are invalid")
    if not runner.spec.layer_start <= layer_index < runner.spec.layer_end:
        return False
    if runner.compute_dtype != torch.float32 or runner.spec.quantize or runner.spec.compile_mode:
        raise ValueError("browser layer currently requires eager float32 native execution")
    local_index = layer_index - runner.spec.layer_start
    native = runner.base.layers[local_index]
    adapter = _adapter_for(native)
    if adapter is None:
        raise ValueError("browser layer requires a registered native decoder adapter")
    owner = BrowserExpertOwner("browser-layer", url)
    bridge = BrowserDecoderLayer(native, owner, artifact_id, local_index, context, adapter)
    runner.base.layers[local_index] = bridge
    runner._browser_layer_bridge = bridge
    return True


def _require_coordinator_origin(url: str) -> None:
    parsed = urlparse(url)
    if (parsed.scheme not in ("https", "http") or not parsed.netloc
            or parsed.path not in ("", "/") or parsed.username or parsed.password
            or parsed.query or parsed.fragment
            or (parsed.scheme == "http" and parsed.hostname not in
                ("localhost", "127.0.0.1", "::1"))):
        raise ValueError("browser layer coordinator URL must be an origin")


# Kept for older local test harnesses and bridge fixtures.
BrowserQwen3Layer = BrowserDecoderLayer


def _encode(tensor: torch.Tensor) -> str:
    data = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    return base64.b64encode(data.tobytes(order="C")).decode("ascii")


def _decode(value: object, shape: tuple[int, ...]) -> torch.Tensor:
    if not isinstance(value, str):
        raise BrowserExpertOwnerError("browser layer returned no tensor")
    try:
        raw = base64.b64decode(value, validate=True)
    except (ValueError, base64.binascii.Error) as error:
        raise BrowserExpertOwnerError("browser layer returned invalid base64") from error
    expected = 4
    for dimension in shape:
        expected *= dimension
    if len(raw) != expected:
        raise BrowserExpertOwnerError("browser layer returned a wrong tensor shape")
    result = torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(shape)
    if not bool(torch.isfinite(result).all()):
        raise BrowserExpertOwnerError("browser layer returned non-finite values")
    return result
