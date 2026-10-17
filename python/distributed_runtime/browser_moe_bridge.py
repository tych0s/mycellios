"""Opt-in placement of one real routed MoE expert on browser workers.

The model and its router stay in the native stage.  A browser is considered
only when two validated foreground replicas hold the exact published expert;
the local RAM copy remains the authoritative fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from .browser_expert_owner import BrowserExpertOwner
from .ram_expert_cache import ExpertKey
from .resident_expert_mesh import (
    MeshLinkProfile,
    MeshNodeProfile,
    ResidentExpertMesh,
    ResidentExpertReplica,
)

if TYPE_CHECKING:
    from .ram_backed_moe_stage import RamBackedMoeStageRunner


_SCHEMA = "mycellios-browser-moe-bridge/1"
_MAX_BROWSER_WEIGHT_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True)
class BrowserMoeBridgeConfig:
    artifact_identity: str
    coordinator_url: str
    layer: int
    expert: int
    local_compute_ms_per_token: float
    browser_compute_ms_per_token: float
    ram_to_device_gbytes_per_second: float
    round_trip_ms: float
    bandwidth_mbps: float
    local_vram_budget_bytes: int
    browser_vram_budget_bytes: int

    @classmethod
    def from_file(cls, path: str | Path) -> "BrowserMoeBridgeConfig":
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(document, dict) or document.get("schema") != _SCHEMA:
            raise ValueError("unsupported browser MoE bridge configuration")
        fields = {
            "artifact_identity": "artifactIdentity",
            "coordinator_url": "coordinatorUrl",
            "layer": "layer",
            "expert": "expert",
            "local_compute_ms_per_token": "localComputeMsPerToken",
            "browser_compute_ms_per_token": "browserComputeMsPerToken",
            "ram_to_device_gbytes_per_second": "ramToDeviceGbytesPerSecond",
            "round_trip_ms": "roundTripMs",
            "bandwidth_mbps": "bandwidthMbps",
            "local_vram_budget_bytes": "localVramBudgetBytes",
            "browser_vram_budget_bytes": "browserVramBudgetBytes",
        }
        if set(document) != {"schema", *fields.values()}:
            raise ValueError("browser MoE bridge fields do not match the schema")
        config = cls(**{name: document[key] for name, key in fields.items()})
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", config.artifact_identity):
            raise ValueError("browser MoE bridge requires an exact artifact identity")
        parsed = urlparse(config.coordinator_url)
        if parsed.scheme not in ("https", "http") or not parsed.netloc or parsed.path not in ("", "/"):
            raise ValueError("browser MoE coordinator URL must be an origin")
        if parsed.scheme == "http" and parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
            raise ValueError("remote browser MoE coordinator requires HTTPS")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("browser MoE coordinator URL cannot contain credentials or parameters")
        for name in ("layer", "expert"):
            value = getattr(config, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"browser MoE {name} must be a non-negative integer")
        for name in (
            "local_compute_ms_per_token", "browser_compute_ms_per_token",
            "ram_to_device_gbytes_per_second", "round_trip_ms", "bandwidth_mbps",
        ):
            value = getattr(config, name)
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f"browser MoE {name} must be finite and positive")
        for name in ("local_vram_budget_bytes", "browser_vram_budget_bytes"):
            value = getattr(config, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be positive")
        return config


def attach_browser_moe_bridge(runner: "RamBackedMoeStageRunner") -> bool:
    """Attach the selected expert if this process owns its sparse layer.

    No environment variable means the existing native route is unchanged.
    Publishing uses the internal coordinator token; browsers never receive it.
    """

    config_path = os.environ.get("MYCELLIOS_BROWSER_MOE_BRIDGE_FILE", "").strip()
    if not config_path:
        return False
    config = BrowserMoeBridgeConfig.from_file(config_path)
    if runner.spec.artifact_identity != config.artifact_identity:
        raise ValueError("browser MoE bridge artifact identity differs from the active model")
    if config.layer not in runner.sparse_layers:
        if runner.spec.layer_start <= config.layer < runner.spec.layer_end:
            raise ValueError("browser MoE bridge selected a dense layer")
        return False

    module = runner._expert_module_for_layer(config.layer)
    key = ExpertKey(config.layer, config.expert)
    records = tuple(
        runner.expert_scheduler.inventory.record(ExpertKey(config.layer, index))
        for index in range(module.num_experts)
    )
    if key not in {record.key for record in records}:
        raise ValueError("browser MoE expert is absent from the model")
    bundle = runner.expert_store.ram_bundle(key)
    gate = bundle.tensor("gate_proj.weight")
    up = bundle.tensor("up_proj.weight")
    down = bundle.tensor("down_proj.weight")
    activation_name = getattr(module.act_fn, "__name__", type(module.act_fn).__name__).lower()
    if activation_name not in ("silu", "siluactivation"):
        raise ValueError("browser MoE bridge supports only SiLU experts")
    hidden_size = int(gate.shape[1])
    intermediate_size = int(gate.shape[0])
    browser_bytes = 3 * hidden_size * intermediate_size * 4
    if browser_bytes > _MAX_BROWSER_WEIGHT_BYTES:
        raise ValueError("browser expert exceeds the coordinator weight limit")
    local_workspace = 4 * intermediate_size * gate.element_size()
    browser_workspace = 4 * intermediate_size * 4
    if config.browser_vram_budget_bytes < browser_bytes + browser_workspace:
        raise ValueError("browser VRAM budget cannot hold the selected expert")
    record = runner.expert_scheduler.inventory.record(key)
    if config.local_vram_budget_bytes < max(record.byte_size for record in records) + local_workspace:
        raise ValueError("local VRAM budget cannot hold the fallback expert")
    mesh = ResidentExpertMesh(
        coordinator_id="native",
        experts=records,
        nodes=(
            MeshNodeProfile(
                "native", config.local_vram_budget_bytes, 0, config.local_compute_ms_per_token,
                ram_to_device_gbytes_per_second=config.ram_to_device_gbytes_per_second,
                expert_workspace_bytes_per_token=local_workspace,
            ),
            MeshNodeProfile(
                "browser", config.browser_vram_budget_bytes, 0,
                config.browser_compute_ms_per_token,
                expert_workspace_bytes_per_token=browser_workspace,
            ),
        ),
        links=(MeshLinkProfile(
            "native", "browser", config.round_trip_ms, config.bandwidth_mbps,
        ),),
        local_ram_keys=tuple(record.key for record in records),
        local_gpu_keys=(),
        replicas=(ResidentExpertReplica(key, "browser", record.content_id),),
        local_weight_buffer_bytes=max(record.byte_size for record in records),
        activation_bytes_per_token=hidden_size * gate.element_size(),
    )
    try:
        owner = BrowserExpertOwner("browser", config.coordinator_url)
        owner.publish_swiglu_expert(
            key=key,
            content_id=record.content_id,
            model_id=Path(runner.spec.model_name).name[:200],
            model_digest=config.artifact_identity,
            gate_projection=gate,
            up_projection=up,
            down_projection=down,
        )
        runner.attach_resident_expert_mesh(config.layer, mesh, {"browser": owner})
        runner.adopt_browser_expert_mesh(mesh)
    except BaseException:
        mesh.close()
        raise
    return True
