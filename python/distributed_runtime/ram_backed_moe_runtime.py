"""Sealed process-local configuration for the RAM-backed MoE runner.

The portable manifest never contains a host path.  The launcher binds its
stage contract to the existing ``--model`` path and these flags.  This module
is intentionally light: parsing and validation do not import Torch,
Transformers or the physical runner, so malformed contracts fail before model
resolution or device allocation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any, Literal

from .ram_expert_cache import PredictiveCacheConfig

if TYPE_CHECKING:
    from .model import StageModelSpec, StageRunnerContract


ARTIFACT_SCHEMA = "gdlp-local-safetensors-moe-stage/1"
CACHE_SCHEMA = "gdlp-predictive-expert-cache/1"
_ADAPTER_CONFIG_IDENTITIES = {
    "transformers-qwen3-moe-v1": ("qwen3_moe", ["Qwen3MoeForCausalLM"]),
    "transformers-glm4-moe-v1": ("glm4_moe", ["Glm4MoeForCausalLM"]),
}


@dataclass(frozen=True, kw_only=True)
class RamBackedMoeRuntimeConfig:
    artifact_identity: str
    adapter_id: str
    device: str
    cache_config: PredictiveCacheConfig
    expected_resident_parameter_bytes: int
    expected_total_routed_expert_bytes: int
    expected_largest_expert_bytes: int
    expected_resident_streaming_transient_bytes: int
    expected_bounded_pinned_staging_reserve_bytes: int
    expected_host_ram_peak_upper_bound_bytes: int
    pin_memory: bool = False
    allow_cpu_fallback: bool = False
    enable_online_prefetch: bool = True
    prefetch_deadline_ms: float = 1_000.0
    execution_mode: Literal["production-cuda", "certification-cpu"] = (
        "production-cuda"
    )

    def __post_init__(self) -> None:
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", self.artifact_identity):
            raise ValueError("RAM-backed MoE artifact identity must be lowercase SHA-256")
        if self.adapter_id not in _ADAPTER_CONFIG_IDENTITIES:
            raise ValueError("RAM-backed MoE adapter is unsupported")
        if self.execution_mode == "production-cuda":
            if not re.fullmatch(r"cuda(?::(?:0|[1-9][0-9]*))?", self.device):
                raise ValueError("production RAM-backed MoE device must be cuda[:index]")
        elif self.execution_mode == "certification-cpu":
            if self.device != "cpu":
                raise ValueError("CPU certification mode requires device=cpu")
        else:
            raise ValueError("RAM-backed MoE execution mode is unsupported")
        if self.pin_memory:
            raise ValueError("RAM-backed MoE pin-memory must remain false")
        if self.allow_cpu_fallback:
            raise ValueError("RAM-backed MoE CPU fallback is forbidden")
        if not isinstance(self.enable_online_prefetch, bool):
            raise TypeError("enable_online_prefetch must be boolean")
        if not math.isfinite(self.prefetch_deadline_ms) or self.prefetch_deadline_ms <= 0:
            raise ValueError("prefetch_deadline_ms must be finite and positive")
        for name, value in (
            (
                "expected_resident_parameter_bytes",
                self.expected_resident_parameter_bytes,
            ),
            (
                "expected_total_routed_expert_bytes",
                self.expected_total_routed_expert_bytes,
            ),
            ("expected_largest_expert_bytes", self.expected_largest_expert_bytes),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in (
            (
                "expected_resident_streaming_transient_bytes",
                self.expected_resident_streaming_transient_bytes,
            ),
            (
                "expected_bounded_pinned_staging_reserve_bytes",
                self.expected_bounded_pinned_staging_reserve_bytes,
            ),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if (
            not isinstance(self.expected_host_ram_peak_upper_bound_bytes, int)
            or isinstance(self.expected_host_ram_peak_upper_bound_bytes, bool)
            or self.expected_host_ram_peak_upper_bound_bytes < 1
        ):
            raise ValueError(
                "expected_host_ram_peak_upper_bound_bytes must be a positive integer"
            )
        if (
            self.expected_largest_expert_bytes
            > self.expected_total_routed_expert_bytes
        ):
            raise ValueError("largest expert cannot exceed the routed expert set")
        if (
            self.cache_config.prefetch_reserve_bytes
            != self.expected_largest_expert_bytes
            or self.cache_config.capacity_bytes
            < self.expected_largest_expert_bytes * 2
        ):
            raise ValueError(
                "physical cache must contain one exact and one prefetch expert buffer"
            )
        expected_host_peak = (
            self.expected_total_routed_expert_bytes
            + self.expected_bounded_pinned_staging_reserve_bytes
            + self.expected_resident_streaming_transient_bytes
        )
        if self.expected_host_ram_peak_upper_bound_bytes != expected_host_peak:
            raise ValueError(
                "host RAM peak must equal routed experts plus pinned staging reserve "
                "plus one resident streaming tensor"
            )
        required_staging_reserve = (
            self.expected_largest_expert_bytes * 2
            if self.execution_mode == "production-cuda"
            else 0
        )
        if (
            self.expected_bounded_pinned_staging_reserve_bytes
            != required_staging_reserve
        ):
            raise ValueError(
                "bounded pinned staging reserve must be exactly two largest experts "
                "for production CUDA and zero for CPU certification"
            )

    @classmethod
    def for_cpu_certification(
        cls,
        *,
        artifact_identity: str,
        adapter_id: str,
        cache_config: PredictiveCacheConfig,
        expected_resident_parameter_bytes: int,
        expected_total_routed_expert_bytes: int,
        expected_largest_expert_bytes: int,
        expected_resident_streaming_transient_bytes: int,
        expected_bounded_pinned_staging_reserve_bytes: int = 0,
        expected_host_ram_peak_upper_bound_bytes: int | None = None,
        enable_online_prefetch: bool = True,
    ) -> "RamBackedMoeRuntimeConfig":
        """Explicit non-production mode used only by physical CPU parity gates."""

        return cls(
            artifact_identity=artifact_identity,
            adapter_id=adapter_id,
            device="cpu",
            cache_config=cache_config,
            expected_resident_parameter_bytes=expected_resident_parameter_bytes,
            expected_total_routed_expert_bytes=expected_total_routed_expert_bytes,
            expected_largest_expert_bytes=expected_largest_expert_bytes,
            expected_resident_streaming_transient_bytes=(
                expected_resident_streaming_transient_bytes
            ),
            expected_bounded_pinned_staging_reserve_bytes=(
                expected_bounded_pinned_staging_reserve_bytes
            ),
            expected_host_ram_peak_upper_bound_bytes=(
                expected_host_ram_peak_upper_bound_bytes
                if expected_host_ram_peak_upper_bound_bytes is not None
                else expected_total_routed_expert_bytes
                + expected_bounded_pinned_staging_reserve_bytes
                + expected_resident_streaming_transient_bytes
            ),
            enable_online_prefetch=enable_online_prefetch,
            execution_mode="certification-cpu",
        )


def add_ram_backed_moe_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ram-moe-artifact-schema")
    parser.add_argument("--ram-moe-artifact-identity")
    parser.add_argument(
        "--ram-moe-adapter-id",
        choices=tuple(_ADAPTER_CONFIG_IDENTITIES),
    )
    parser.add_argument("--ram-moe-device")
    parser.add_argument("--ram-moe-pin-memory", choices=("true", "false"))
    parser.add_argument("--ram-moe-allow-cpu-fallback", choices=("true", "false"))
    parser.add_argument("--ram-moe-cache-schema")
    parser.add_argument("--ram-moe-cache-capacity-bytes", type=int)
    parser.add_argument("--ram-moe-prefetch-reserve-bytes", type=int)
    parser.add_argument("--ram-moe-pcie-bandwidth-gbytes-per-second", type=float)
    parser.add_argument("--ram-moe-hotness-decay", type=float)
    parser.add_argument("--ram-moe-min-prefetch-confidence", type=float)
    parser.add_argument("--ram-moe-resident-parameter-budget-bytes", type=int)
    parser.add_argument("--ram-moe-total-routed-expert-bytes", type=int)
    parser.add_argument("--ram-moe-largest-expert-bytes", type=int)
    parser.add_argument("--ram-moe-resident-streaming-transient-bytes", type=int)
    parser.add_argument("--ram-moe-bounded-pinned-staging-reserve-bytes", type=int)
    parser.add_argument("--ram-moe-host-ram-peak-upper-bound-bytes", type=int)


def ram_backed_moe_config_from_args(
    args: argparse.Namespace,
) -> RamBackedMoeRuntimeConfig | None:
    names = (
        "ram_moe_artifact_schema",
        "ram_moe_artifact_identity",
        "ram_moe_adapter_id",
        "ram_moe_device",
        "ram_moe_pin_memory",
        "ram_moe_allow_cpu_fallback",
        "ram_moe_cache_schema",
        "ram_moe_cache_capacity_bytes",
        "ram_moe_prefetch_reserve_bytes",
        "ram_moe_pcie_bandwidth_gbytes_per_second",
        "ram_moe_hotness_decay",
        "ram_moe_min_prefetch_confidence",
        "ram_moe_resident_parameter_budget_bytes",
        "ram_moe_total_routed_expert_bytes",
        "ram_moe_largest_expert_bytes",
        "ram_moe_resident_streaming_transient_bytes",
        "ram_moe_bounded_pinned_staging_reserve_bytes",
        "ram_moe_host_ram_peak_upper_bound_bytes",
    )
    values = tuple(getattr(args, name, None) for name in names)
    if not any(value is not None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError("RAM-backed MoE flags must be supplied together")
    if args.ram_moe_artifact_schema != ARTIFACT_SCHEMA:
        raise ValueError("unsupported RAM-backed MoE artifact schema")
    if args.ram_moe_cache_schema != CACHE_SCHEMA:
        raise ValueError("unsupported RAM-backed MoE cache schema")
    if args.ram_moe_pin_memory != "false":
        raise ValueError("RAM-backed MoE pin-memory must remain false")
    if args.ram_moe_allow_cpu_fallback != "false":
        raise ValueError("RAM-backed MoE CPU fallback is forbidden")
    config = RamBackedMoeRuntimeConfig(
        artifact_identity=args.ram_moe_artifact_identity,
        adapter_id=args.ram_moe_adapter_id,
        device=args.ram_moe_device,
        pin_memory=False,
        allow_cpu_fallback=False,
        cache_config=PredictiveCacheConfig(
            capacity_bytes=args.ram_moe_cache_capacity_bytes,
            prefetch_reserve_bytes=args.ram_moe_prefetch_reserve_bytes,
            pcie_bandwidth_gbytes_per_second=(
                args.ram_moe_pcie_bandwidth_gbytes_per_second
            ),
            hotness_decay=args.ram_moe_hotness_decay,
            min_prefetch_confidence=args.ram_moe_min_prefetch_confidence,
        ),
        expected_resident_parameter_bytes=(
            args.ram_moe_resident_parameter_budget_bytes
        ),
        expected_total_routed_expert_bytes=(
            args.ram_moe_total_routed_expert_bytes
        ),
        expected_largest_expert_bytes=args.ram_moe_largest_expert_bytes,
        expected_resident_streaming_transient_bytes=(
            args.ram_moe_resident_streaming_transient_bytes
        ),
        expected_bounded_pinned_staging_reserve_bytes=(
            args.ram_moe_bounded_pinned_staging_reserve_bytes
        ),
        expected_host_ram_peak_upper_bound_bytes=(
            args.ram_moe_host_ram_peak_upper_bound_bytes
        ),
    )
    validate_ram_backed_moe_binding(
        config,
        model_name=args.model,
        revision=getattr(args, "revision", None),
        pipeline_snapshot_identity=getattr(args, "pipeline_snapshot_identity", None),
        standard_artifact_identity=getattr(args, "model_artifact_identity", None),
        canonical_model_source=getattr(args, "model_canonical_source", None),
        canonical_model_revision=getattr(args, "model_canonical_revision", None),
    )
    return config


def validate_ram_backed_moe_binding(
    config: RamBackedMoeRuntimeConfig,
    *,
    model_name: str,
    revision: str | None,
    pipeline_snapshot_identity: int | None,
    standard_artifact_identity: str | None = None,
    canonical_model_source: str | None = None,
    canonical_model_revision: str | None = None,
) -> Path:
    snapshot = Path(model_name).expanduser()
    if not snapshot.is_absolute():
        raise ValueError("RAM-backed MoE --model must be an absolute host-local path")
    if revision is not None:
        raise ValueError("RAM-backed MoE local snapshots cannot use --revision")
    if any(
        value is not None
        for value in (
            standard_artifact_identity,
            canonical_model_source,
            canonical_model_revision,
        )
    ):
        raise ValueError(
            "RAM-backed MoE identity cannot be combined with standard model identity flags"
        )
    if (
        pipeline_snapshot_identity is None
        or not isinstance(pipeline_snapshot_identity, int)
        or isinstance(pipeline_snapshot_identity, bool)
        or not 0 <= pipeline_snapshot_identity <= (1 << 64) - 1
    ):
        raise ValueError(
            "RAM-backed MoE requires --pipeline-snapshot-identity as uint64"
        )
    expected_pipeline = int(config.artifact_identity.removeprefix("sha256:")[:16], 16)
    if pipeline_snapshot_identity != expected_pipeline:
        raise ValueError("RAM-backed MoE artifact and pipeline identities do not match")
    if not snapshot.is_dir():
        raise FileNotFoundError(f"RAM-backed MoE snapshot does not exist: {snapshot}")
    config_path = snapshot / "config.json"
    try:
        document: Any = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read RAM-backed MoE config.json: {error}") from error
    model_type, architectures = _ADAPTER_CONFIG_IDENTITIES[config.adapter_id]
    if (
        not isinstance(document, dict)
        or document.get("model_type") != model_type
        or document.get("architectures") != architectures
    ):
        raise ValueError("RAM-backed MoE adapter does not match local config.json")
    return snapshot.resolve()


def build_ram_backed_moe_stage_runner(
    spec: StageModelSpec,
    config: RamBackedMoeRuntimeConfig,
    *,
    pipeline_snapshot_identity: int,
) -> StageRunnerContract:
    validate_ram_backed_moe_binding(
        config,
        model_name=spec.model_name,
        revision=spec.revision,
        pipeline_snapshot_identity=pipeline_snapshot_identity,
    )
    if spec.artifact_identity != config.artifact_identity:
        raise ValueError("stage spec and RAM-backed artifact identities do not match")
    from .ram_backed_moe_stage import RamBackedMoeStageRunner

    runner = RamBackedMoeStageRunner(
        spec,
        config.cache_config,
        device=config.device,
        pin_memory=config.pin_memory,
        allow_cpu_fallback=config.allow_cpu_fallback,
        enable_online_prefetch=config.enable_online_prefetch,
        prefetch_deadline_ms=config.prefetch_deadline_ms,
        expected_adapter_id=config.adapter_id,
        expected_resident_parameter_bytes=(
            config.expected_resident_parameter_bytes
        ),
        expected_total_routed_expert_bytes=(
            config.expected_total_routed_expert_bytes
        ),
        expected_largest_expert_bytes=config.expected_largest_expert_bytes,
        expected_resident_streaming_transient_bytes=(
            config.expected_resident_streaming_transient_bytes
        ),
        expected_bounded_pinned_staging_reserve_bytes=(
            config.expected_bounded_pinned_staging_reserve_bytes
        ),
        expected_host_ram_peak_upper_bound_bytes=(
            config.expected_host_ram_peak_upper_bound_bytes
        ),
        bounded_pinned_staging=(
            config.expected_bounded_pinned_staging_reserve_bytes > 0
        ),
    )
    if runner.resident_parameter_bytes > config.expected_resident_parameter_bytes:
        runner.close()
        raise MemoryError("RAM-backed resident parameter budget was exceeded")
    if runner.ram_parameter_bytes != config.expected_total_routed_expert_bytes:
        runner.close()
        raise MemoryError("RAM-backed routed expert byte contract changed after load")
    if runner.largest_expert_bytes != config.expected_largest_expert_bytes:
        runner.close()
        raise MemoryError("RAM-backed largest expert byte contract changed after load")
    if (
        runner.resident_streaming_transient_bytes
        > config.expected_resident_streaming_transient_bytes
    ):
        runner.close()
        raise MemoryError("RAM-backed resident streaming transient budget was exceeded")
    if (
        runner.bounded_pinned_staging_reserve_bytes
        != config.expected_bounded_pinned_staging_reserve_bytes
    ):
        runner.close()
        raise MemoryError("RAM-backed bounded pinned staging reserve changed after load")
    if (
        runner.host_ram_peak_upper_bound_bytes
        > config.expected_host_ram_peak_upper_bound_bytes
    ):
        runner.close()
        raise MemoryError("RAM-backed host RAM peak budget was exceeded")
    if config.execution_mode == "production-cuda" and (
        runner.compute_device.type != "cuda" or runner.expert_store.cpu_fallback
    ):
        runner.close()
        raise RuntimeError("production RAM-backed MoE runner did not remain on CUDA")
    return runner


__all__ = [
    "ARTIFACT_SCHEMA",
    "CACHE_SCHEMA",
    "RamBackedMoeRuntimeConfig",
    "add_ram_backed_moe_arguments",
    "build_ram_backed_moe_stage_runner",
    "ram_backed_moe_config_from_args",
    "validate_ram_backed_moe_binding",
]
