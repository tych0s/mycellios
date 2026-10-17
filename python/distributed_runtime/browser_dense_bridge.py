"""Opt-in browser execution for one Qwen3 dense SwiGLU model layer.

The native stage retains attention, KV state and a complete local MLP.  A
validated pair of foreground browsers may execute the stateless MLP projection
through the existing replicated expert contract.  The local MLP is used when
browser residency or execution disappears.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import torch
from torch import nn

from .browser_expert_owner import BrowserExpertOwner, BrowserExpertOwnerError
from .ram_expert_cache import ExpertKey
from .resident_expert_mesh import (
    ExpertResidentSlotUnavailableError,
    OwnerExpertBatchItem,
)

if TYPE_CHECKING:
    from .model import StageRunner


def initialize_dense_stage_with_browser_bridge(runner: "StageRunner", spec, model, **kwargs) -> None:
    """Finish normal stage initialization, then attach the optional bridge."""

    runner._initialize_from_loaded_model(spec, model, **kwargs)
    try:
        attach_browser_dense_bridge(runner)
    except BaseException:
        runner.close()
        raise


class BrowserDenseMlp(nn.Module):
    """Qwen3 MLP wrapper retaining the exact native module as fallback."""

    def __init__(self, native: nn.Module, owner: BrowserExpertOwner, key: ExpertKey,
                 content_id: str) -> None:
        super().__init__()
        self.native = native
        self.owner = owner
        self.key = key
        self.content_id = content_id
        self.browser_forwards = 0
        self.native_forwards = 0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        if len(shape) < 2:
            raise ValueError("dense browser activation must include hidden dimension")
        if self.owner.is_expert_resident(self.key, self.content_id):
            try:
                flat = hidden_states.reshape(-1, shape[-1])
                result = self.owner.execute_batch((
                    OwnerExpertBatchItem(self.key, self.content_id, flat),
                ))[0].output
                self.browser_forwards += 1
                return result.to(device=hidden_states.device, dtype=hidden_states.dtype).reshape(shape)
            except (ExpertResidentSlotUnavailableError, BrowserExpertOwnerError):
                pass
        self.native_forwards += 1
        return self.native(hidden_states)


def attach_browser_dense_bridge(runner: "StageRunner") -> bool:
    """Attach a browser MLP only to the selected exact Qwen3 stage."""

    path = os.environ.get("MYCELLIOS_BROWSER_DENSE_BRIDGE_FILE", "").strip()
    if not path:
        return False
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict) or set(document) != {
        "schema", "artifactIdentity", "coordinatorUrl", "layer",
    } or document["schema"] != "mycellios-browser-dense-bridge/1":
        raise ValueError("unsupported browser dense bridge configuration")
    identity = document["artifactIdentity"]
    if not isinstance(identity, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", identity):
        raise ValueError("browser dense bridge requires an exact SHA-256 model identity")
    if identity != runner.executor_manifest.model_identity:
        raise ValueError("browser dense bridge artifact identity differs from the active model")
    url = document["coordinatorUrl"]
    if not isinstance(url, str):
        raise ValueError("browser dense bridge coordinator URL must be an origin")
    parsed = urlparse(url)
    if (parsed.scheme not in ("https", "http") or not parsed.netloc
            or parsed.path not in ("", "/") or parsed.username or parsed.password
            or parsed.query or parsed.fragment):
        raise ValueError("browser dense bridge coordinator URL must be an origin")
    if parsed.scheme == "http" and parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
        raise ValueError("remote browser dense coordinator requires HTTPS")
    layer_index = document["layer"]
    if not isinstance(layer_index, int) or isinstance(layer_index, bool) or layer_index < 0:
        raise ValueError("browser dense bridge layer must be a non-negative integer")
    if not runner.spec.layer_start <= layer_index < runner.spec.layer_end:
        return False
    local_index = layer_index - runner.spec.layer_start
    layer = runner.base.layers[local_index]
    native = layer.mlp
    if type(native).__name__ != "Qwen3MLP":
        raise ValueError("browser dense bridge currently supports Qwen3MLP only")
    activation_name = getattr(native.act_fn, "__name__", type(native.act_fn).__name__).lower()
    if activation_name not in ("silu", "siluactivation"):
        raise ValueError("browser dense bridge requires SiLU activation")
    for projection in (native.gate_proj, native.up_proj, native.down_proj):
        if projection.bias is not None:
            raise ValueError("browser dense bridge does not support projection biases")
    gate = native.gate_proj.weight.detach()
    up = native.up_proj.weight.detach()
    down = native.down_proj.weight.detach()
    hidden = int(gate.shape[1])
    intermediate = int(gate.shape[0])
    if tuple(up.shape) != (intermediate, hidden) or tuple(down.shape) != (hidden, intermediate):
        raise ValueError("browser dense projection shapes do not match SwiGLU")
    if 3 * hidden * intermediate * 4 > 512 * 1024 * 1024:
        raise ValueError("browser dense expert exceeds the coordinator weight limit")
    key = ExpertKey(layer_index, 0)
    content_id = identity + f":dense-mlp:{layer_index}"
    owner = BrowserExpertOwner("browser-dense", url)
    owner.publish_swiglu_expert(
        key=key,
        content_id=content_id,
        model_id=Path(runner.spec.model_name).name[:200],
        model_digest=identity,
        gate_projection=gate,
        up_projection=up,
        down_projection=down,
    )
    layer.mlp = BrowserDenseMlp(native, owner, key, content_id)
    return True
