"""Canonical Mycellios stage runner for authenticated native GGUF packages.

The persistent artifact stays in Mycellios' own GGUF stage format. At process
startup this module verifies the sealed package once, constructs only the local
Transformers range and streams one GGUF tensor at a time directly into its
resident destination. No external inference daemon, all-tensor FP32 dictionary,
temporary SafeTensors checkpoint or whole-model checkpoint is involved.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import re
import time
from typing import Any

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM

from .dense_tiering import DenseTieringConfig
from .device import resolve_torch_execution_device
from .executor_abi import build_stage_executor_manifest
from .model import (
    StageModelSpec,
    StageRunner,
    _checkpoint_name,
    _validate_checkpoint_coverage,
)
from .model_adapters import (
    SelectiveStageAdapter,
    resolve_selective_stage_adapter,
)
from .native_gguf import (
    NATIVE_GGUF_STAGE_CONFIG,
    NATIVE_GGUF_STAGE_WEIGHTS,
    NativeGgufError,
    NativeGgufStagePackage,
    _hf_checkpoint_name,
    _load_config,
    _restore_huggingface_tensor_layout,
    _validate_derived_rope_factors,
    dequantize_gguf_tensor,
    parse_gguf,
    verify_native_gguf_stage,
)
from .native_gguf_disk_tiering import (
    NativeGgufDiskTierRuntime,
    decide_native_gguf_disk_tiering,
)


_PACKAGE_ID = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class NativeGgufRuntimeConfig:
    """Sealed launch binding for one native GGUF stage package."""

    package: str
    package_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.package, str) or not self.package.strip():
            raise ValueError("native GGUF package path cannot be empty")
        if not isinstance(self.package_id, str) or not _PACKAGE_ID.fullmatch(
            self.package_id
        ):
            raise ValueError("native GGUF package id must be a lowercase SHA-256")


def add_native_gguf_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the mutually-required native GGUF stage launch arguments."""

    parser.add_argument(
        "--native-gguf-package",
        help="authenticated Mycellios native GGUF stage package",
    )
    parser.add_argument(
        "--native-gguf-package-id",
        help="sealed SHA-256 package identity expected by the coordinator",
    )


def native_gguf_runtime_from_args(
    args: argparse.Namespace,
) -> NativeGgufRuntimeConfig | None:
    package = getattr(args, "native_gguf_package", None)
    package_id = getattr(args, "native_gguf_package_id", None)
    if package is None and package_id is None:
        return None
    if package is None or package_id is None:
        raise ValueError(
            "native GGUF package path and package id must be supplied together"
        )
    return NativeGgufRuntimeConfig(package=package, package_id=package_id)


class NativeGgufStageRunner(StageRunner):
    """Execute one exact native GGUF range through the standard stage ABI."""

    def __init__(
        self,
        spec: StageModelSpec,
        runtime: NativeGgufRuntimeConfig,
        *,
        device: str = "auto",
        dense_tiering: DenseTieringConfig | None = None,
    ) -> None:
        package = verify_native_gguf_stage(
            runtime.package,
            expected_package_id=runtime.package_id,
            expected_layer_start=spec.layer_start,
            expected_layer_end=spec.layer_end,
            expected_total_layers=spec.total_layers,
        )
        if spec.artifact_identity != package.artifact_identity:
            raise NativeGgufError(
                "stage artifact identity differs from the native GGUF package"
            )
        if spec.stage_package_identity != package.package_identity:
            raise NativeGgufError(
                "stage package identity differs from the native GGUF package"
            )
        if spec.canonical_model_source != package.model_source:
            raise NativeGgufError(
                "canonical model source differs from the native GGUF package"
            )
        if spec.canonical_model_revision != package.model_revision:
            raise NativeGgufError(
                "canonical model revision differs from the native GGUF package"
            )

        torch.set_num_threads(spec.threads)
        execution_device = resolve_torch_execution_device(device)
        compute_dtype = (
            torch.float16 if execution_device.accelerated else torch.float32
        )
        if spec.quantize == "dynamic-int8" and execution_device.accelerated:
            raise ValueError("dynamic-int8 stage quantization is CPU-only")
        tiering_config = dense_tiering or DenseTieringConfig()
        model, startup = _load_native_gguf_stage_model(
            spec,
            package,
            resident_dtype=compute_dtype,
            host_ram_budget_bytes=tiering_config.host_ram_budget_bytes,
            dense_tiering=tiering_config,
            execution_device=execution_device,
        )
        storage_tier = getattr(
            model,
            "_mycellios_native_gguf_disk_tiering",
            None,
        )
        if storage_tier is not None and not isinstance(
            storage_tier,
            NativeGgufDiskTierRuntime,
        ):
            raise TypeError("native GGUF disk-tier runtime is invalid")
        try:
            self._initialize_from_loaded_model(
                spec,
                model,
                loader="native-gdlp-gguf-stage",
                device_kinds=(execution_device.device.type,),
                execution_device=execution_device,
                compute_dtype=compute_dtype,
                move_model=storage_tier is None,
                dense_tiering=tiering_config,
            )
        except BaseException:
            if storage_tier is not None:
                storage_tier.close()
            raise
        if storage_tier is not None:
            self.dense_tiering = storage_tier
            self.parameter_bytes = int(
                storage_tier.snapshot()["totalWeightBytes"]
            )
            self.loader += "+authenticated-storage-tier"
        self.native_gguf_startup = startup

        inherited = self.executor_manifest
        features = tuple(
            dict.fromkeys(
                (
                    *inherited.features,
                    "native-gguf-parser",
                    "native-gguf-stage-package",
                    "stage-only-dequantization",
                    "streaming-tensor-load",
                    "no-safetensors-rematerialization",
                    "external-runtime-free",
                    *(
                        (
                            "authenticated-storage-offsets",
                            "bounded-storage-to-ram-layer-residency",
                            "synchronous-storage-materialization",
                            "no-storage-prefetch",
                        )
                        if storage_tier is not None
                        else ()
                    ),
                )
            )
        )
        self.executor_manifest = build_stage_executor_manifest(
            engine="mycellios-native-gguf-torch",
            engine_version=f"torch-{torch.__version__}/gdlp-native-gguf-1",
            adapter=inherited.adapter,
            model_identity=package.artifact_identity,
            model_source=package.model_source,
            model_revision=package.model_revision,
            artifact_format="gdlp-layer-gguf",
            layer_start=spec.layer_start,
            layer_end=spec.layer_end,
            total_layers=spec.total_layers,
            hidden_size=inherited.hidden_size,
            activation_dtype=inherited.activation_dtype,
            activation_codecs=inherited.activation_codecs,
            kv_format=inherited.kv_format,
            supports_truncate=inherited.supports_truncate,
            phases=inherited.phases,
            operations=inherited.operations,
            max_batch_size=inherited.max_batch_size,
            max_context_tokens=inherited.max_context_tokens,
            device_kinds=inherited.device_kinds,
            compute_apis=inherited.compute_apis,
            weight_dtypes=inherited.weight_dtypes,
            features=features,
        )

    def execution_snapshot(self) -> dict[str, Any]:
        snapshot = super().execution_snapshot()
        snapshot["nativeGgufStartup"] = dict(self.native_gguf_startup)
        return snapshot


def _load_native_gguf_stage_model(
    spec: StageModelSpec,
    package: NativeGgufStagePackage,
    *,
    resident_dtype: torch.dtype = torch.float32,
    host_ram_budget_bytes: int = 0,
    dense_tiering: DenseTieringConfig | None = None,
    execution_device: Any | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Build one local family adapter and stream its authenticated GGUF tensors."""

    parsed = parse_gguf(package.root / NATIVE_GGUF_STAGE_WEIGHTS)
    if parsed.file_sha256 != package.stage_gguf_sha256:
        raise NativeGgufError("native GGUF changed after package authentication")
    tiering_config = dense_tiering or DenseTieringConfig(
        host_ram_budget_bytes=host_ram_budget_bytes,
    )
    gguf_config = _load_config(package.root / NATIVE_GGUF_STAGE_CONFIG)
    decision = decide_native_gguf_disk_tiering(
        parsed,
        architecture=package.architecture,
        resident_dtype=resident_dtype,
        config=tiering_config,
        derived_static_bytes=_derived_rotary_buffer_bytes(gguf_config),
    )
    config = AutoConfig.from_pretrained(
        package.root,
        local_files_only=True,
        trust_remote_code=False,
    )
    adapter = resolve_selective_stage_adapter(config)
    adapter.validate_source_config(config, spec.total_layers)
    local_config = copy.deepcopy(config)
    local_layers = spec.layer_end - spec.layer_start
    adapter.slice_config(
        local_config,
        layer_start=spec.layer_start,
        layer_end=spec.layer_end,
        total_layers=spec.total_layers,
    )
    original_vocab_size = int(local_config.vocab_size)
    original_pad_token_id = getattr(local_config, "pad_token_id", None)
    if not spec.first and not spec.last:
        local_config.vocab_size = 1
        local_config.pad_token_id = 0
    if decision.enabled:
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(
                local_config,
                dtype=resident_dtype,
            )
    else:
        model = AutoModelForCausalLM.from_config(
            local_config,
            dtype=resident_dtype,
        )
    adapter.inspect_constructed_model(model, local_layers=local_layers)
    if not spec.last:
        model.model.norm = nn.Identity()
        model.lm_head = nn.Identity()
    if not spec.first:
        model.model.embed_tokens = (
            nn.Identity()
            if decision.enabled
            else nn.Embedding(1, model.config.hidden_size)
        )
    if not spec.first and not spec.last:
        model.config.vocab_size = original_vocab_size
        model.config.pad_token_id = original_pad_token_id

    if decision.enabled:
        if execution_device is None:
            raise TypeError(
                "native GGUF disk tier requires the resolved execution device"
            )
        storage_tier = NativeGgufDiskTierRuntime(
            model,
            spec,
            package,
            parsed,
            adapter,
            execution_device=execution_device,
            compute_dtype=resident_dtype,
            config=tiering_config,
            decision=decision,
            gguf_config=gguf_config,
        )
        model._mycellios_native_gguf_disk_tiering = storage_tier
        storage = storage_tier.snapshot()["storageToRam"]
        startup = {
            "schema": "mycellios-native-gguf-startup/1",
            "loadMode": "authenticated-offset-layer-cache",
            "tensorCount": len(
                tuple(
                    tensor
                    for tensor in parsed.tensors
                    if not tensor.name.startswith("rope_freqs.")
                )
            ),
            "encodedBytesRead": storage["artifactBytesMaterialized"],
            "authenticatedFileBytes": storage["fileBytesAuthenticated"],
            "residentSourceBytes": decision.full_resident_weight_bytes,
            "largestEncodedTensorBytes": max(
                tensor.size_bytes for tensor in parsed.tensors
            ),
            "largestDecodedTensorBytes": max(
                tensor.element_count * 4 for tensor in parsed.tensors
            ),
            "maxExplicitLiveTensorBytes":
                decision.explicit_host_working_set_upper_bound_bytes,
            "avoidedFullStageTensorMapBytes":
                decision.full_resident_weight_bytes,
            "temporarySafetensorsBytesWritten": 0,
            "tensorReadOperations": storage["tensorReadOperations"],
            "materializeAndCopyNanoseconds":
                storage["materializeAndCopyNanoseconds"],
            "prefetchEnabled": False,
        }
    else:
        startup = _stream_native_gguf_parameters(
            model,
            spec,
            package,
            adapter=adapter,
            parsed=parsed,
        )
    model.model.config.num_hidden_layers = local_layers
    model.config.num_hidden_layers = local_layers
    model._gdlp_selective_stage_adapter = adapter
    # _initialize_from_loaded_model uses this authenticated directory only to
    # resolve the already-supplied global identity. It never opens weights.
    model._gdlp_resolved_snapshot = str(package.root)
    if not decision.enabled:
        model._mycellios_dense_storage_evidence = {
            "schema": "mycellios-storage-to-ram/1",
            "format": "gguf",
            "artifactBytesMaterialized": startup["encodedBytesRead"],
            "tensorReadOperations": startup["tensorReadOperations"],
            "materializeAndCopyNanoseconds": startup[
                "materializeAndCopyNanoseconds"
            ],
            "physicalDiskBytes": None,
            "osPageCacheHits": None,
        }
    return model.eval(), startup


def _derived_rotary_buffer_bytes(config: dict[str, Any]) -> int:
    head_dim = config.get("head_dim")
    if head_dim is None:
        hidden = config.get("hidden_size")
        heads = config.get("num_attention_heads")
        if (
            not isinstance(hidden, int)
            or isinstance(hidden, bool)
            or not isinstance(heads, int)
            or isinstance(heads, bool)
            or heads < 1
            or hidden % heads
        ):
            raise NativeGgufError(
                "native GGUF config cannot derive rotary buffer size"
            )
        head_dim = hidden // heads
    if (
        not isinstance(head_dim, int)
        or isinstance(head_dim, bool)
        or head_dim < 2
        or head_dim % 2
    ):
        raise NativeGgufError("native GGUF config head_dim is invalid")
    # inv_freq and original_inv_freq are derived FP32 buffers, each with
    # head_dim / 2 elements in the certified Llama and Qwen3 implementations.
    return head_dim * torch.empty((), dtype=torch.float32).element_size()


def _stream_native_gguf_parameters(
    model: nn.Module,
    spec: StageModelSpec,
    package: NativeGgufStagePackage,
    *,
    adapter: SelectiveStageAdapter,
    parsed: Any | None = None,
) -> dict[str, Any]:
    """Copy exactly one decoded tensor at a time into its resident destination."""

    parsed = parsed or parse_gguf(package.root / NATIVE_GGUF_STAGE_WEIGHTS)
    if parsed.file_sha256 != package.stage_gguf_sha256:
        raise NativeGgufError(
            "native GGUF changed after package authentication"
        )
    config = _load_config(package.root / NATIVE_GGUF_STAGE_CONFIG)
    gguf_by_checkpoint: dict[str, Any] = {}
    rope_tensors: dict[str, Any] = {}
    for tensor in parsed.tensors:
        if tensor.name.startswith("rope_freqs."):
            rope_tensors[tensor.name] = tensor
            continue
        checkpoint_name = _hf_checkpoint_name(tensor.name)
        if checkpoint_name in gguf_by_checkpoint:
            raise NativeGgufError(
                f"multiple GGUF tensors map to {checkpoint_name!r}"
            )
        gguf_by_checkpoint[checkpoint_name] = tensor

    checkpoint_files = {
        checkpoint_name: NATIVE_GGUF_STAGE_WEIGHTS
        for checkpoint_name in gguf_by_checkpoint
    }
    tied_embeddings = bool(getattr(model.config, "tie_word_embeddings", False))
    targets: list[tuple[str, torch.Tensor, str]] = []
    seen_tensors: set[int] = set()
    for local_name, target in model.state_dict(keep_vars=True).items():
        identity = id(target)
        if identity in seen_tensors:
            continue
        assignments = adapter.checkpoint_assignments(
            local_name,
            target,
            layer_start=spec.layer_start,
            checkpoint_names=set(checkpoint_files),
        )
        if assignments is not None:
            seen_tensors.add(identity)
            targets.extend(
                (local_name, assignment.destination, assignment.checkpoint_name)
                for assignment in assignments
            )
            continue
        checkpoint_name = _checkpoint_name(
            local_name,
            spec,
            checkpoint_files,
            tied_embeddings=tied_embeddings,
        )
        if checkpoint_name is not None:
            seen_tensors.add(identity)
            targets.append((local_name, target, checkpoint_name))

    loaded_checkpoint_names = {
        checkpoint_name for _, _, checkpoint_name in targets
    }
    _validate_checkpoint_coverage(
        checkpoint_files,
        loaded_checkpoint_names,
        spec,
        tied_embeddings=tied_embeddings,
    )
    target_by_checkpoint: dict[str, list[tuple[str, torch.Tensor]]] = {}
    for local_name, target, checkpoint_name in targets:
        if checkpoint_name not in checkpoint_files:
            raise KeyError(
                f"GGUF tensor {checkpoint_name!r} required by {local_name!r} is missing"
            )
        target_by_checkpoint.setdefault(checkpoint_name, []).append(
            (local_name, target)
        )
    unused = set(checkpoint_files) - set(target_by_checkpoint)
    if unused:
        raise KeyError(
            "native GGUF contains required stage tensors with no resident "
            f"destination: {sorted(unused)}"
        )

    tensor_count = 0
    total_encoded_bytes = 0
    total_resident_source_bytes = 0
    largest_encoded_bytes = 0
    largest_decoded_bytes = 0
    max_explicit_live_bytes = 0
    seen_checkpoint_names: set[str] = set()
    materialize_and_copy_nanoseconds = 0
    tensor_read_operations = 0
    with parsed.path.open("rb") as stream, torch.no_grad():
        for tensor in parsed.tensors:
            started = time.perf_counter_ns()
            stream.seek(tensor.data_offset)
            raw = stream.read(tensor.size_bytes)
            if len(raw) != tensor.size_bytes:
                raise NativeGgufError(
                    f"GGUF tensor {tensor.name!r} is truncated"
                )
            decoded = dequantize_gguf_tensor(tensor, raw)
            decoded_bytes = decoded.numel() * decoded.element_size()
            explicit_live_bytes = len(raw) + decoded_bytes
            largest_encoded_bytes = max(largest_encoded_bytes, len(raw))
            largest_decoded_bytes = max(largest_decoded_bytes, decoded_bytes)
            total_encoded_bytes += len(raw)
            if tensor.name in rope_tensors:
                _validate_derived_rope_factors(
                    decoded,
                    architecture=package.architecture,
                    config=config,
                )
                max_explicit_live_bytes = max(
                    max_explicit_live_bytes,
                    explicit_live_bytes,
                )
                del decoded, raw
                materialize_and_copy_nanoseconds += max(
                    0, time.perf_counter_ns() - started
                )
                tensor_read_operations += 1
                continue
            checkpoint_name = _hf_checkpoint_name(tensor.name)
            restored = _restore_huggingface_tensor_layout(
                tensor.name,
                decoded,
                architecture=package.architecture,
                config=config,
            )
            restored_bytes = restored.numel() * restored.element_size()
            if restored.untyped_storage().data_ptr() != decoded.untyped_storage().data_ptr():
                explicit_live_bytes += restored_bytes
            max_explicit_live_bytes = max(
                max_explicit_live_bytes,
                explicit_live_bytes,
            )
            for local_name, target in target_by_checkpoint[checkpoint_name]:
                if tuple(restored.shape) != tuple(target.shape):
                    raise ValueError(
                        f"shape mismatch for {local_name}: GGUF "
                        f"{tuple(restored.shape)}, stage {tuple(target.shape)}"
                    )
                target.copy_(restored)
            seen_checkpoint_names.add(checkpoint_name)
            tensor_count += 1
            tensor_read_operations += 1
            total_resident_source_bytes += restored_bytes
            del restored, decoded, raw
            materialize_and_copy_nanoseconds += max(
                0, time.perf_counter_ns() - started
            )
    if seen_checkpoint_names != set(target_by_checkpoint):
        missing = sorted(set(target_by_checkpoint) - seen_checkpoint_names)
        raise KeyError(f"native GGUF did not stream required tensors: {missing}")
    return {
        "schema": "mycellios-native-gguf-startup/1",
        "loadMode": "one-tensor-at-a-time",
        "tensorCount": tensor_count,
        "encodedBytesRead": total_encoded_bytes,
        "residentSourceBytes": total_resident_source_bytes,
        "largestEncodedTensorBytes": largest_encoded_bytes,
        "largestDecodedTensorBytes": largest_decoded_bytes,
        # This is exact for the explicit raw/decoded/layout tensors owned by
        # this loop. NumPy/Torch kernel-internal workspace is not observable,
        # so it is deliberately not mislabeled as process peak RSS.
        "maxExplicitLiveTensorBytes": max_explicit_live_bytes,
        "avoidedFullStageTensorMapBytes": total_resident_source_bytes,
        "temporarySafetensorsBytesWritten": 0,
        "tensorReadOperations": tensor_read_operations,
        "materializeAndCopyNanoseconds": materialize_and_copy_nanoseconds,
    }


def build_native_gguf_stage_runner(
    spec: StageModelSpec,
    runtime: NativeGgufRuntimeConfig,
    *,
    device: str = "auto",
    dense_tiering: DenseTieringConfig | None = None,
) -> NativeGgufStageRunner:
    """Small construction seam shared by CLI, server and local engine paths."""

    return NativeGgufStageRunner(
        spec,
        runtime,
        device=device,
        dense_tiering=dense_tiering,
    )


def native_gguf_launch_document(
    runtime: NativeGgufRuntimeConfig,
) -> dict[str, Any]:
    """Return a path-free launch identity suitable for logs and telemetry."""

    package = verify_native_gguf_stage(
        runtime.package,
        expected_package_id=runtime.package_id,
    )
    return {
        "runtime": "mycellios-native-gguf",
        "packageId": package.package_id,
        "packageIdentity": package.package_identity,
        "modelIdentity": package.artifact_identity,
        "modelSource": package.model_source,
        "modelRevision": package.model_revision,
        "layerStart": package.layer_start,
        "layerEnd": package.layer_end,
        "totalLayers": package.total_layers,
        "architecture": package.architecture,
        "externalRuntimeRequired": False,
    }
