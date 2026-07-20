from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import metadata as importlib_metadata
import json
from pathlib import Path
import struct
import time
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as distributed


EXPERT_PARALLEL_CELL_SCHEMA = "gdlp-expert-parallel-cell/1"
EXPERT_PARALLEL_INSPECTION_SCHEMA = "gdlp-expert-parallel-inspection/1"
EXPERT_PARALLEL_GATE_SCHEMA = "gdlp-expert-parallel-physical-gate/1"

CERTIFIED_TRANSFORMERS_VERSION = "5.14.1"
CERTIFIED_ACCELERATE_VERSION = "1.14.0"
CERTIFIED_TORCH_VERSION = "2.13.0"

QWEN3_MOE_EP_PLAN = {
    "layers.*.mlp.gate": "ep_router",
    "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
    "layers.*.mlp.experts.down_proj": "grouped_gemm",
    "layers.*.mlp.experts": "moe_tp_experts",
}
QWEN3_MOE_EXPANDED_EP_PLAN = {
    f"model.{name}": strategy for name, strategy in QWEN3_MOE_EP_PLAN.items()
}

_MODEL_ARCHITECTURE = "Qwen3MoeForCausalLM"
_ENGINE_ID = "transformers-native-expert-parallel"
_ADAPTER_ID = "hf-native-qwen3-moe-ep-v1"
_HASH_CHUNK_BYTES = 8 * 1024 * 1024


class UnsupportedExpertParallelCellError(RuntimeError):
    """Raised before launch when a model or runtime is not certified for EP."""


@dataclass(frozen=True)
class ExpertParallelCheckpointInspection:
    checkpoint: Path
    model_identity: str
    model_source: str
    model_type: str
    architectures: tuple[str, ...]
    total_layers: int
    hidden_size: int
    num_experts: int
    top_k: int
    ep_plan: tuple[tuple[str, str], ...]
    world_size: int
    backend: str
    device_kind: str
    checkpoint_parameter_bytes: int
    checkpoint_expert_bytes: int
    projected_rank_local_parameter_bytes: int

    @property
    def local_experts(self) -> int:
        return self.num_experts // self.world_size

    def to_document(self) -> dict[str, Any]:
        return {
            "schema": EXPERT_PARALLEL_INSPECTION_SCHEMA,
            "model": {
                "identity": self.model_identity,
                "source": self.model_source,
                "artifactFormat": "safetensors",
                "modelType": self.model_type,
                "architectures": list(self.architectures),
                "totalLayers": self.total_layers,
                "hiddenSize": self.hidden_size,
            },
            "experts": {
                "globalCount": self.num_experts,
                "localCount": self.local_experts,
                "topK": self.top_k,
                "baseModelEpPlan": dict(self.ep_plan),
            },
            "distributed": {
                "scope": "low-latency-cell-only",
                "wanAllowed": False,
                "backend": self.backend,
                "deviceKind": self.device_kind,
                "worldSize": self.world_size,
            },
            "memory": {
                "checkpointParameterBytes": self.checkpoint_parameter_bytes,
                "checkpointExpertBytes": self.checkpoint_expert_bytes,
                "projectedRankLocalParameterBytes": self.projected_rank_local_parameter_bytes,
                "rankLocalParameterBytes": None,
                "rankLocalMeasurement": "requires-physical-rank-load",
            },
            "capabilities": {
                "fullModelCell": True,
                "partialStage": False,
                "stageRunner": False,
                "expertParallelAcrossWan": False,
            },
        }


@dataclass(frozen=True)
class ExpertParallelCellManifest:
    document: Mapping[str, Any]

    @property
    def executor_id(self) -> str:
        return str(self.document["executorId"])

    @property
    def cell_contract_id(self) -> str:
        return str(self.document["cellContractId"])

    def to_document(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.document))


class HfNativeExpertParallelCellExecutor:
    """One complete HF MoE model executed by a tightly coupled EP cell.

    Every rank runs the complete transformer graph. Only the expert tensors are
    rank-local; embeddings, attention, routers, norms and LM head remain
    replicated. The expert outputs are summed by the native HF all-reduce hook.
    This class is intentionally not a GDLP StageRunner and must never be routed
    over the residential WAN.
    """

    def __init__(
        self,
        *,
        model: Any,
        inspection: ExpertParallelCheckpointInspection,
        manifest: ExpertParallelCellManifest,
        load_ms: float,
    ) -> None:
        self.model = model
        self.inspection = inspection
        self.manifest = manifest
        self.load_ms = load_ms

    @classmethod
    def load(
        cls,
        checkpoint: str | Path,
        *,
        model_source: str | None = None,
        backend: str | None = None,
        device_kind: str | None = None,
        dtype: torch.dtype = torch.float32,
        local_files_only: bool = True,
        require_certified_versions: bool = True,
    ) -> "HfNativeExpertParallelCellExecutor":
        if not distributed.is_available() or not distributed.is_initialized():
            raise UnsupportedExpertParallelCellError(
                "HF native EP requires an initialized intra-cell process group"
            )
        rank = distributed.get_rank()
        world_size = distributed.get_world_size()
        actual_backend = _backend_name(distributed.get_backend())
        requested_backend = _backend_name(backend or actual_backend)
        if requested_backend != actual_backend:
            raise UnsupportedExpertParallelCellError(
                f"initialized backend {actual_backend!r} does not match {requested_backend!r}"
            )
        requested_device = device_kind or ("cuda" if torch.cuda.is_available() else "cpu")
        _validate_transport(requested_backend, requested_device, world_size)
        versions = runtime_versions()
        if require_certified_versions:
            _require_certified_runtime(versions)

        inspection = inspect_expert_parallel_checkpoint(
            checkpoint,
            world_size=world_size,
            backend=requested_backend,
            device_kind=requested_device,
            model_source=model_source,
        )

        from transformers import AutoModelForCausalLM
        from transformers.distributed import DistributedConfig

        distributed_config = DistributedConfig(
            tp_size=world_size,
            enable_expert_parallel=True,
        )
        started = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            inspection.checkpoint,
            distributed_config=distributed_config,
            tp_plan="auto",
            local_files_only=local_files_only,
            trust_remote_code=False,
            dtype=dtype,
        ).eval()
        load_ms = (time.perf_counter() - started) * 1_000

        _validate_loaded_model(model, inspection, rank)
        parameter_bytes = _module_parameter_bytes(model)
        expert_parameter_bytes = _module_parameter_bytes(
            model,
            name_contains=".mlp.experts.",
        )
        model_device = _model_device(model)
        if model_device.type != requested_device:
            raise UnsupportedExpertParallelCellError(
                f"loaded model device {model_device.type!r} does not match {requested_device!r}"
            )
        _validate_collective_layout(
            inspection=inspection,
            rank=rank,
            parameter_bytes=parameter_bytes,
            expert_parameter_bytes=expert_parameter_bytes,
        )
        manifest = build_expert_parallel_cell_manifest(
            inspection,
            rank=rank,
            device=str(model_device),
            rank_local_parameter_bytes=parameter_bytes,
            rank_local_expert_parameter_bytes=expert_parameter_bytes,
            versions=versions,
        )
        return cls(
            model=model,
            inspection=inspection,
            manifest=manifest,
            load_ms=load_ms,
        )

    @property
    def device(self) -> torch.device:
        return _model_device(self.model)

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> Any:
        if not distributed.is_initialized():
            raise RuntimeError("the EP process group was destroyed before forward")
        if input_ids.ndim != 2 or input_ids.numel() < 1:
            raise ValueError("input_ids must have shape [batch, tokens]")
        return self.model(input_ids=input_ids.to(self.device), **kwargs)

    @torch.no_grad()
    def greedy_next_token(self, input_ids: torch.Tensor) -> torch.Tensor:
        output = self.forward(input_ids, use_cache=False, logits_to_keep=1)
        return output.logits[:, -1, :].argmax(dim=-1)


def runtime_versions() -> dict[str, str]:
    return {
        "torch": torch.__version__,
        "transformers": _distribution_version("transformers"),
        "accelerate": _distribution_version("accelerate"),
    }


def inspect_expert_parallel_checkpoint(
    checkpoint: str | Path,
    *,
    world_size: int,
    backend: str,
    device_kind: str,
    model_source: str | None = None,
) -> ExpertParallelCheckpointInspection:
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.is_dir():
        raise UnsupportedExpertParallelCellError(
            "expert-parallel checkpoint must be a resolved local directory"
        )
    _validate_transport(backend, device_kind, world_size)
    model_identity, files = _checkpoint_identity(checkpoint_path)

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        checkpoint_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    if getattr(config, "model_type", None) != "qwen3_moe":
        raise UnsupportedExpertParallelCellError(
            "HF native EP adapter currently certifies only model_type='qwen3_moe'"
        )
    architectures = _string_tuple(getattr(config, "architectures", None), "architectures")
    if architectures != (_MODEL_ARCHITECTURE,):
        raise UnsupportedExpertParallelCellError(
            f"qwen3_moe architectures must be exactly [{_MODEL_ARCHITECTURE!r}]"
        )
    ep_plan = getattr(config, "base_model_ep_plan", None)
    if not isinstance(ep_plan, Mapping) or dict(ep_plan) != QWEN3_MOE_EP_PLAN:
        raise UnsupportedExpertParallelCellError(
            "Qwen3Moe base_model_ep_plan does not match the certified HF EP layout"
        )
    num_experts = _positive_integer(getattr(config, "num_experts", None), "num_experts")
    top_k = _positive_integer(
        getattr(config, "num_experts_per_tok", None),
        "num_experts_per_tok",
    )
    if top_k > num_experts:
        raise UnsupportedExpertParallelCellError(
            "num_experts_per_tok cannot exceed num_experts"
        )
    if num_experts % world_size != 0:
        raise UnsupportedExpertParallelCellError(
            f"num_experts must be divisible by world_size: {num_experts} % {world_size} != 0"
        )
    if getattr(config, "decoder_sparse_step", None) != 1 or getattr(
        config, "mlp_only_layers", None
    ) not in (None, []):
        raise UnsupportedExpertParallelCellError(
            "the v1 Qwen3Moe EP adapter requires every decoder layer to be sparse MoE"
        )

    parameter_bytes, expert_bytes = _safetensors_parameter_bytes(files)
    if expert_bytes % world_size != 0:
        raise UnsupportedExpertParallelCellError(
            "checkpoint expert bytes cannot be divided evenly across EP ranks"
        )
    projected_local = parameter_bytes - expert_bytes + expert_bytes // world_size
    stable_source = model_source or f"local-qwen3-moe@{model_identity[7:23]}"
    return ExpertParallelCheckpointInspection(
        checkpoint=checkpoint_path,
        model_identity=model_identity,
        model_source=_nonempty_string(stable_source, "model_source"),
        model_type="qwen3_moe",
        architectures=architectures,
        total_layers=_positive_integer(
            getattr(config, "num_hidden_layers", None), "num_hidden_layers"
        ),
        hidden_size=_positive_integer(getattr(config, "hidden_size", None), "hidden_size"),
        num_experts=num_experts,
        top_k=top_k,
        ep_plan=tuple(sorted((str(name), str(strategy)) for name, strategy in ep_plan.items())),
        world_size=world_size,
        backend=_backend_name(backend),
        device_kind=device_kind,
        checkpoint_parameter_bytes=parameter_bytes,
        checkpoint_expert_bytes=expert_bytes,
        projected_rank_local_parameter_bytes=projected_local,
    )


def build_expert_parallel_cell_manifest(
    inspection: ExpertParallelCheckpointInspection,
    *,
    rank: int,
    device: str,
    rank_local_parameter_bytes: int,
    rank_local_expert_parameter_bytes: int,
    versions: Mapping[str, str] | None = None,
) -> ExpertParallelCellManifest:
    if not isinstance(rank, int) or isinstance(rank, bool) or not 0 <= rank < inspection.world_size:
        raise ValueError("rank is outside the EP cell")
    parameter_bytes = _positive_integer(
        rank_local_parameter_bytes, "rank_local_parameter_bytes"
    )
    expert_bytes = _positive_integer(
        rank_local_expert_parameter_bytes, "rank_local_expert_parameter_bytes"
    )
    if expert_bytes >= parameter_bytes:
        raise ValueError("rank-local experts cannot consume every model parameter byte")
    local_count = inspection.local_experts
    local_start = rank * local_count
    runtime = dict(versions or runtime_versions())
    if set(runtime) != {"torch", "transformers", "accelerate"}:
        raise ValueError("runtime versions must name torch, transformers and accelerate")

    contract = {
        "engine": {
            "id": _ENGINE_ID,
            "adapter": _ADAPTER_ID,
            "torchVersion": _nonempty_string(runtime["torch"], "torch version"),
            "transformersVersion": _nonempty_string(
                runtime["transformers"], "transformers version"
            ),
            "accelerateVersion": _nonempty_string(
                runtime["accelerate"], "accelerate version"
            ),
        },
        "model": {
            "identity": inspection.model_identity,
            "source": inspection.model_source,
            "artifactFormat": "safetensors",
            "modelType": inspection.model_type,
            "architectures": list(inspection.architectures),
            "totalLayers": inspection.total_layers,
            "hiddenSize": inspection.hidden_size,
        },
        "experts": {
            "globalCount": inspection.num_experts,
            "localCount": local_count,
            "topK": inspection.top_k,
            "plan": dict(inspection.ep_plan),
            "router": "global-replicated",
            "weights": "contiguous-rank-local",
            "combine": "all-reduce-sum",
        },
        "distributed": {
            "scope": "low-latency-cell-only",
            "wanAllowed": False,
            "backend": inspection.backend,
            "deviceKind": inspection.device_kind,
            "worldSize": inspection.world_size,
            "tpSize": inspection.world_size,
            "fsdpSize": 1,
        },
        "capabilities": {
            "fullModelCell": True,
            "partialStage": False,
            "stageRunner": False,
            "expertParallelAcrossWan": False,
            "rankLocalKv": True,
            "replicatedNonExpertParameters": True,
        },
        "limitations": [
            "collective forward requires every rank for every token",
            "attention embeddings norms routers and lm-head remain replicated",
            "native HF loader does not expose a certified partial-layer stage here",
            "expert parallel collectives are confined to one low-latency cell",
        ],
    }
    cell_contract_id = hashlib.sha256(_canonical_json(contract)).hexdigest()[:32]
    body = {
        "schema": EXPERT_PARALLEL_CELL_SCHEMA,
        "cellContractId": cell_contract_id,
        **contract,
        "rank": {
            "index": rank,
            "device": _nonempty_string(device, "device"),
            "localExpertRange": {"start": local_start, "end": local_start + local_count},
        },
        "memory": {
            "rankLocalParameterBytes": parameter_bytes,
            "rankLocalExpertParameterBytes": expert_bytes,
            "measurement": "physical-loaded-parameters",
        },
    }
    executor_id = hashlib.sha256(_canonical_json(body)).hexdigest()[:32]
    return validate_expert_parallel_cell_manifest({**body, "executorId": executor_id})


def validate_expert_parallel_cell_manifest(
    value: Mapping[str, Any],
) -> ExpertParallelCellManifest:
    document = _mapping(value, "EP cell manifest")
    if set(document) != {
        "schema",
        "executorId",
        "cellContractId",
        "engine",
        "model",
        "experts",
        "distributed",
        "capabilities",
        "limitations",
        "rank",
        "memory",
    }:
        raise ValueError("EP cell manifest has unknown or missing fields")
    if document.get("schema") != EXPERT_PARALLEL_CELL_SCHEMA:
        raise ValueError("unsupported EP cell manifest schema")
    engine = _exact_mapping(
        document.get("engine"),
        ("id", "adapter", "torchVersion", "transformersVersion", "accelerateVersion"),
        "engine",
    )
    if engine.get("id") != _ENGINE_ID or engine.get("adapter") != _ADAPTER_ID:
        raise ValueError("EP cell engine is unsupported")
    for field in ("torchVersion", "transformersVersion", "accelerateVersion"):
        _nonempty_string(engine.get(field), field)
    model = _exact_mapping(
        document.get("model"),
        (
            "identity",
            "source",
            "artifactFormat",
            "modelType",
            "architectures",
            "totalLayers",
            "hiddenSize",
        ),
        "model",
    )
    if model.get("artifactFormat") != "safetensors" or model.get("modelType") != "qwen3_moe":
        raise ValueError("EP cell model artifact is unsupported")
    identity = _nonempty_string(model.get("identity"), "model identity")
    if not identity.startswith("sha256:") or len(identity) != 71:
        raise ValueError("EP cell model identity must be sha256")
    _nonempty_string(model.get("source"), "model source")
    if _string_tuple(model.get("architectures"), "architectures") != (_MODEL_ARCHITECTURE,):
        raise ValueError("EP cell architecture is unsupported")
    _positive_integer(model.get("totalLayers"), "totalLayers")
    _positive_integer(model.get("hiddenSize"), "hiddenSize")

    experts = _exact_mapping(
        document.get("experts"),
        ("globalCount", "localCount", "topK", "plan", "router", "weights", "combine"),
        "experts",
    )
    global_count = _positive_integer(experts.get("globalCount"), "globalCount")
    local_count = _positive_integer(experts.get("localCount"), "localCount")
    top_k = _positive_integer(experts.get("topK"), "topK")
    if top_k > global_count or experts.get("plan") != QWEN3_MOE_EP_PLAN:
        raise ValueError("EP cell expert routing plan is invalid")
    if (
        experts.get("router") != "global-replicated"
        or experts.get("weights") != "contiguous-rank-local"
        or experts.get("combine") != "all-reduce-sum"
    ):
        raise ValueError("EP cell expert execution semantics are invalid")

    dist = _exact_mapping(
        document.get("distributed"),
        ("scope", "wanAllowed", "backend", "deviceKind", "worldSize", "tpSize", "fsdpSize"),
        "distributed",
    )
    world_size = _positive_integer(dist.get("worldSize"), "worldSize")
    if dist.get("tpSize") != world_size or dist.get("fsdpSize") != 1:
        raise ValueError("EP cell parallel dimensions are invalid")
    if global_count != local_count * world_size:
        raise ValueError("EP cell local experts do not cover the global set")
    if dist.get("scope") != "low-latency-cell-only" or dist.get("wanAllowed") is not False:
        raise ValueError("expert parallelism cannot cross the WAN")
    _validate_transport(
        _nonempty_string(dist.get("backend"), "backend"),
        _nonempty_string(dist.get("deviceKind"), "deviceKind"),
        world_size,
    )

    capabilities = _exact_mapping(
        document.get("capabilities"),
        (
            "fullModelCell",
            "partialStage",
            "stageRunner",
            "expertParallelAcrossWan",
            "rankLocalKv",
            "replicatedNonExpertParameters",
        ),
        "capabilities",
    )
    expected_capabilities = {
        "fullModelCell": True,
        "partialStage": False,
        "stageRunner": False,
        "expertParallelAcrossWan": False,
        "rankLocalKv": True,
        "replicatedNonExpertParameters": True,
    }
    if dict(capabilities) != expected_capabilities:
        raise ValueError("EP cell capabilities overclaim the executor")
    limitations = document.get("limitations")
    if not isinstance(limitations, list) or len(limitations) < 4 or any(
        not isinstance(item, str) or not item for item in limitations
    ):
        raise ValueError("EP cell limitations must be explicit")

    rank_document = _exact_mapping(
        document.get("rank"), ("index", "device", "localExpertRange"), "rank"
    )
    rank = rank_document.get("index")
    if not isinstance(rank, int) or isinstance(rank, bool) or not 0 <= rank < world_size:
        raise ValueError("EP cell rank is invalid")
    _nonempty_string(rank_document.get("device"), "rank device")
    local_range = _exact_mapping(
        rank_document.get("localExpertRange"), ("start", "end"), "localExpertRange"
    )
    if local_range.get("start") != rank * local_count or local_range.get("end") != (rank + 1) * local_count:
        raise ValueError("EP cell local expert range is invalid")
    memory = _exact_mapping(
        document.get("memory"),
        ("rankLocalParameterBytes", "rankLocalExpertParameterBytes", "measurement"),
        "memory",
    )
    parameter_bytes = _positive_integer(
        memory.get("rankLocalParameterBytes"), "rankLocalParameterBytes"
    )
    expert_bytes = _positive_integer(
        memory.get("rankLocalExpertParameterBytes"), "rankLocalExpertParameterBytes"
    )
    if expert_bytes >= parameter_bytes or memory.get("measurement") != "physical-loaded-parameters":
        raise ValueError("EP cell memory evidence is invalid")

    contract = {
        name: document[name]
        for name in (
            "engine",
            "model",
            "experts",
            "distributed",
            "capabilities",
            "limitations",
        )
    }
    expected_contract_id = hashlib.sha256(_canonical_json(contract)).hexdigest()[:32]
    if document.get("cellContractId") != expected_contract_id:
        raise ValueError("EP cell contract identity does not match")
    body = dict(document)
    body.pop("executorId", None)
    expected_executor_id = hashlib.sha256(_canonical_json(body)).hexdigest()[:32]
    if document.get("executorId") != expected_executor_id:
        raise ValueError("EP cell executor identity does not match")
    return ExpertParallelCellManifest(document=json.loads(json.dumps(document)))


def _validate_loaded_model(
    model: Any,
    inspection: ExpertParallelCheckpointInspection,
    rank: int,
) -> None:
    config = getattr(model, "config", None)
    distributed_config = getattr(config, "distributed_config", None)
    if (
        distributed_config is None
        or distributed_config.enable_expert_parallel is not True
        or distributed_config.tp_size != inspection.world_size
        or distributed_config.fsdp_size != 1
    ):
        raise UnsupportedExpertParallelCellError(
            "loaded model did not retain the requested expert-parallel dimensions"
        )
    if dict(getattr(model, "tp_plan", {})) != QWEN3_MOE_EXPANDED_EP_PLAN:
        raise UnsupportedExpertParallelCellError(
            "loaded model did not apply the certified expanded HF EP plan"
        )
    if getattr(model, "_tp_size", None) != inspection.world_size:
        raise UnsupportedExpertParallelCellError("loaded model TP size is inconsistent")
    layers = getattr(getattr(model, "model", None), "layers", None)
    if not isinstance(layers, torch.nn.ModuleList) or len(layers) != inspection.total_layers:
        raise UnsupportedExpertParallelCellError("loaded model layer set is incomplete")
    for layer_index, layer in enumerate(layers):
        experts = getattr(getattr(layer, "mlp", None), "experts", None)
        gate_up = getattr(experts, "gate_up_proj", None)
        down = getattr(experts, "down_proj", None)
        if (
            not isinstance(gate_up, torch.nn.Parameter)
            or not isinstance(down, torch.nn.Parameter)
            or gate_up.ndim != 3
            or down.ndim != 3
            or gate_up.shape[0] != inspection.local_experts
            or down.shape[0] != inspection.local_experts
            or getattr(experts, "num_experts", None) != inspection.local_experts
        ):
            raise UnsupportedExpertParallelCellError(
                f"layer {layer_index} does not hold the expected rank-local expert shard"
            )
        router = getattr(getattr(layer, "mlp", None), "gate", None)
        if getattr(router, "num_experts", None) != inspection.num_experts:
            raise UnsupportedExpertParallelCellError(
                f"layer {layer_index} router is not globally replicated"
            )


def _validate_collective_layout(
    *,
    inspection: ExpertParallelCheckpointInspection,
    rank: int,
    parameter_bytes: int,
    expert_parameter_bytes: int,
) -> None:
    local = {
        "rank": rank,
        "identity": inspection.model_identity,
        "localExperts": inspection.local_experts,
        "start": rank * inspection.local_experts,
        "end": (rank + 1) * inspection.local_experts,
        "parameterBytes": parameter_bytes,
        "expertBytes": expert_parameter_bytes,
    }
    gathered: list[dict[str, Any] | None] = [None] * inspection.world_size
    distributed.all_gather_object(gathered, local)
    if any(item is None for item in gathered):
        raise UnsupportedExpertParallelCellError("EP layout certification missed a rank")
    ordered = sorted((dict(item) for item in gathered if item is not None), key=lambda item: item["rank"])
    if [item["rank"] for item in ordered] != list(range(inspection.world_size)):
        raise UnsupportedExpertParallelCellError("EP layout ranks are duplicated or incomplete")
    if any(item["identity"] != inspection.model_identity for item in ordered):
        raise UnsupportedExpertParallelCellError("EP ranks loaded different model identities")
    if [(item["start"], item["end"]) for item in ordered] != [
        (index * inspection.local_experts, (index + 1) * inspection.local_experts)
        for index in range(inspection.world_size)
    ]:
        raise UnsupportedExpertParallelCellError("EP expert shards overlap or leave a gap")
    if len({item["expertBytes"] for item in ordered}) != 1:
        raise UnsupportedExpertParallelCellError("EP ranks hold unequal expert byte counts")


def _require_certified_runtime(versions: Mapping[str, str]) -> None:
    expected = {
        "transformers": CERTIFIED_TRANSFORMERS_VERSION,
        "accelerate": CERTIFIED_ACCELERATE_VERSION,
    }
    for package, version in expected.items():
        if versions.get(package) != version:
            raise UnsupportedExpertParallelCellError(
                f"{package} {versions.get(package)!r} is not the certified EP version {version!r}"
            )
    torch_public = str(versions.get("torch", "")).split("+", 1)[0]
    if torch_public != CERTIFIED_TORCH_VERSION:
        raise UnsupportedExpertParallelCellError(
            f"torch {versions.get('torch')!r} is not the certified EP version {CERTIFIED_TORCH_VERSION!r}"
        )


def _validate_transport(backend: str, device_kind: str, world_size: int) -> None:
    size = _positive_integer(world_size, "world_size")
    if size < 2:
        raise UnsupportedExpertParallelCellError(
            "expert parallelism requires at least two intra-cell ranks"
        )
    normalized_backend = _backend_name(backend)
    if device_kind not in ("cpu", "cuda"):
        raise UnsupportedExpertParallelCellError(
            "HF native EP currently certifies only cpu or cuda device kinds"
        )
    expected = "gloo" if device_kind == "cpu" else "nccl"
    if normalized_backend != expected:
        raise UnsupportedExpertParallelCellError(
            f"device kind {device_kind!r} requires backend {expected!r}, got {normalized_backend!r}"
        )


def _checkpoint_identity(checkpoint: Path) -> tuple[str, tuple[Path, ...]]:
    config = checkpoint / "config.json"
    if not config.is_file():
        raise UnsupportedExpertParallelCellError("local EP checkpoint is missing config.json")
    tensor_files = tuple(sorted(checkpoint.glob("*.safetensors"), key=lambda path: path.name))
    if not tensor_files:
        raise UnsupportedExpertParallelCellError(
            "HF native EP certification requires a local safetensors checkpoint"
        )
    files = (config, *tensor_files)
    index = checkpoint / "model.safetensors.index.json"
    if index.is_file():
        files = (config, index, *tensor_files)
    digest = hashlib.sha256(b"gdlp-hf-native-ep-checkpoint-v1\0")
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            while chunk := handle.read(_HASH_CHUNK_BYTES):
                digest.update(chunk)
    return f"sha256:{digest.hexdigest()}", tuple(tensor_files)


def _safetensors_parameter_bytes(files: Sequence[Path]) -> tuple[int, int]:
    total = 0
    experts = 0
    names: set[str] = set()
    for path in files:
        with path.open("rb") as handle:
            length_bytes = handle.read(8)
            if len(length_bytes) != 8:
                raise UnsupportedExpertParallelCellError(
                    f"safetensors header is truncated: {path.name}"
                )
            header_length = struct.unpack("<Q", length_bytes)[0]
            if header_length < 2 or header_length > path.stat().st_size - 8:
                raise UnsupportedExpertParallelCellError(
                    f"safetensors header length is invalid: {path.name}"
                )
            try:
                header = json.loads(handle.read(header_length))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise UnsupportedExpertParallelCellError(
                    f"safetensors header is invalid JSON: {path.name}"
                ) from error
        if not isinstance(header, Mapping):
            raise UnsupportedExpertParallelCellError("safetensors header must be an object")
        for name, metadata in header.items():
            if name == "__metadata__":
                continue
            if name in names:
                raise UnsupportedExpertParallelCellError(
                    f"duplicate safetensors parameter {name!r}"
                )
            names.add(name)
            if not isinstance(metadata, Mapping):
                raise UnsupportedExpertParallelCellError(
                    f"safetensors metadata for {name!r} is invalid"
                )
            offsets = metadata.get("data_offsets")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or not all(isinstance(value, int) and value >= 0 for value in offsets)
                or offsets[1] < offsets[0]
            ):
                raise UnsupportedExpertParallelCellError(
                    f"safetensors offsets for {name!r} are invalid"
                )
            size = offsets[1] - offsets[0]
            total += size
            if ".mlp.experts." in name:
                experts += size
    if total < 1 or experts < 1 or experts >= total:
        raise UnsupportedExpertParallelCellError(
            "checkpoint parameter bytes do not describe a sparse MoE model"
        )
    return total, experts


def _module_parameter_bytes(module: Any, name_contains: str | None = None) -> int:
    total = 0
    for name, parameter in module.named_parameters():
        if name_contains is None or name_contains in name:
            total += parameter.numel() * parameter.element_size()
    return total


def _model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration as error:
        raise UnsupportedExpertParallelCellError("loaded EP model has no parameters") from error


def _distribution_version(name: str) -> str:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError as error:
        raise UnsupportedExpertParallelCellError(
            f"HF native EP requires package {name!r}"
        ) from error


def _backend_name(value: Any) -> str:
    text = str(value).lower()
    if text.startswith("backend."):
        text = text.split(".", 1)[1]
    return text


def _positive_integer(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence) or not value:
        raise UnsupportedExpertParallelCellError(f"{name} must be a non-empty string list")
    result = tuple(_nonempty_string(item, name) for item in value)
    if len(set(result)) != len(result):
        raise UnsupportedExpertParallelCellError(f"{name} cannot contain duplicates")
    return result


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _exact_mapping(value: Any, fields: Sequence[str], name: str) -> Mapping[str, Any]:
    mapping = _mapping(value, name)
    if set(mapping) != set(fields):
        raise ValueError(f"EP cell {name} has unknown or missing fields")
    return mapping


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


__all__ = [
    "CERTIFIED_ACCELERATE_VERSION",
    "CERTIFIED_TORCH_VERSION",
    "CERTIFIED_TRANSFORMERS_VERSION",
    "EXPERT_PARALLEL_CELL_SCHEMA",
    "EXPERT_PARALLEL_GATE_SCHEMA",
    "EXPERT_PARALLEL_INSPECTION_SCHEMA",
    "ExpertParallelCellManifest",
    "ExpertParallelCheckpointInspection",
    "HfNativeExpertParallelCellExecutor",
    "QWEN3_MOE_EP_PLAN",
    "QWEN3_MOE_EXPANDED_EP_PLAN",
    "UnsupportedExpertParallelCellError",
    "build_expert_parallel_cell_manifest",
    "inspect_expert_parallel_checkpoint",
    "runtime_versions",
    "validate_expert_parallel_cell_manifest",
]
