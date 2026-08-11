"""Opt-in paged KV/COW prototype for one Hugging Face stage.

This module deliberately does *not* replace :class:`StageRunner`.  It is an
isolated adapter around the ``PagedAttentionCache`` shipped by the exact
Transformers version pinned by this repository.  The adapter proves the part
that ``DynamicCache`` cannot provide safely: complete KV blocks are shared by
reference across forks and only an incomplete tail block is copied.

The integration surface is intentionally narrow and fail-closed:

* one process and one device (CPU or CUDA), without tensor parallelism;
* full-attention, stage-local configurations only;
* ``paged|eager`` or ``paged|sdpa`` model attention;
* one model forward at a time; packed tree verification is not implemented;
* opaque, adapter-unique cache keys disable unrelated-request prefix reuse;
* the physical cache pool is preallocated by Transformers;
* Transformers does not expose per-operation workspace peaks, so
  ``peak_workspace`` is reported as ``None`` instead of being estimated.

``forward_model`` accepts either token IDs (first stage) or ``inputs_embeds``
(an intermediate selective stage).  It owns allocation and commit.  If the
model raises after a cache write may have started, that request is discarded;
publishing a possibly half-written KV state would be unsafe.

Some observability and rollback operations necessarily inspect the
``PagedAttentionCache`` block manager, whose public API currently has no
block-ID/refcount snapshot, request rename, or crop operation.  Construction
therefore seals the implementation to Transformers 5.14.1 and validates the
expected manager shape before accepting work.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
import copy
from dataclasses import dataclass
import gc
import hashlib
import json
from math import ceil
from time import perf_counter_ns
from typing import Any

import torch
import transformers
from torch import nn
from transformers.configuration_utils import PreTrainedConfig

from .executor_abi import StageKVForkReport, build_stage_executor_manifest
from .model import (
    StageModelSpec,
    _load_selective_stage_model,
    _unique_parameter_bytes,
    model_artifact_reference,
)
from .model_adapters import SelectiveStageAdapter


SUPPORTED_TRANSFORMERS_VERSION = "5.14.1"
SUPPORTED_ATTENTION_BACKENDS = frozenset(("eager", "sdpa"))
MAX_REQUEST_ID = 2**63 - 1
MAX_OPAQUE_CACHE_KEY = 2**31 - 1
PAGED_STAGE_RUNTIME_SCHEMA = "mycellios-hf-paged-stage/2"
DEFAULT_PAGED_BLOCK_SIZE = 16
DEFAULT_PAGED_NUM_BLOCKS = 256
DEFAULT_PAGED_MAX_BATCH_TOKENS = 256
DEFAULT_PAGED_MAX_ACTIVE_REQUESTS = 8
DEFAULT_PAGED_MAX_SEQUENCE_TOKENS = 2048
MAX_PAGED_BLOCK_SIZE = 256
MAX_PAGED_NUM_BLOCKS = 1_048_576
MAX_PAGED_BATCH_TOKENS = 65_536
MAX_PAGED_ACTIVE_REQUESTS = 4_096
MAX_PAGED_SEQUENCE_TOKENS = 1_048_576


@dataclass(frozen=True)
class _ContinuousBatchingTypes:
    config: type[Any]
    cache: type[Any]
    distributed_helper: type[Any]
    request_state: type[Any]


def _require_supported_transformers_version() -> None:
    installed_version = transformers.__version__
    if installed_version != SUPPORTED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            "HF paged stage adapter is sealed to transformers "
            f"{SUPPORTED_TRANSFORMERS_VERSION}, found {installed_version}; "
            "continuous-batching symbols were not imported"
        )


def _load_continuous_batching_types() -> _ContinuousBatchingTypes:
    """Load the sealed optional HF API only after its exact version is known."""

    _require_supported_transformers_version()
    try:
        from transformers.generation.configuration_utils import (
            ContinuousBatchingConfig,
        )
        from transformers.generation.continuous_batching.cache import (
            PagedAttentionCache,
        )
        from transformers.generation.continuous_batching.distributed import (
            DistributedHelper,
        )
        from transformers.generation.continuous_batching.requests import RequestState
    except (AttributeError, ImportError) as error:
        raise RuntimeError(
            "transformers 5.14.1 is missing the sealed continuous-batching API "
            "required by the HF paged stage adapter"
        ) from error
    return _ContinuousBatchingTypes(
        config=ContinuousBatchingConfig,
        cache=PagedAttentionCache,
        distributed_helper=DistributedHelper,
        request_state=RequestState,
    )


@dataclass(frozen=True)
class HFPagedStageRuntimeConfig:
    """Closed, content-sealed settings for the native paged KV backend.

    ``cpu_spill_bytes`` bounds only the tensor payload retained by the
    adapter-owned CPU spill tier. It is not an RSS, allocator-overhead or pinned
    allocator-cache limit. A request is copied completely and integrity-checked
    before its device block references are released. Restore verifies the CPU
    source, allocates and fills a private block table, and only then publishes
    the request. A single target payload can be held as an explicitly measured
    restore workspace while its persistent spill slot is reused.
    """

    schema: str = PAGED_STAGE_RUNTIME_SCHEMA
    device: str = "cpu"
    attention_backend: str = "eager"
    block_size: int = DEFAULT_PAGED_BLOCK_SIZE
    num_blocks: int = DEFAULT_PAGED_NUM_BLOCKS
    max_batch_tokens: int = DEFAULT_PAGED_MAX_BATCH_TOKENS
    max_active_requests: int = DEFAULT_PAGED_MAX_ACTIVE_REQUESTS
    max_sequence_tokens: int = DEFAULT_PAGED_MAX_SEQUENCE_TOKENS
    cpu_spill_bytes: int = 0

    def __post_init__(self) -> None:
        if self.schema != PAGED_STAGE_RUNTIME_SCHEMA:
            raise ValueError(
                f"paged runtime schema must be {PAGED_STAGE_RUNTIME_SCHEMA}"
            )
        try:
            device = torch.device(self.device)
        except (TypeError, RuntimeError) as error:
            raise ValueError("paged device must be cpu or cuda[:index]") from error
        if device.type not in ("cpu", "cuda") or (
            device.type == "cpu" and device.index is not None
        ):
            raise ValueError("paged device must be cpu or cuda[:index]")
        if self.device != str(device):
            raise ValueError("paged device must use its canonical torch spelling")
        if self.attention_backend not in SUPPORTED_ATTENTION_BACKENDS:
            raise ValueError(
                "paged attention backend must be one of "
                f"{sorted(SUPPORTED_ATTENTION_BACKENDS)}"
            )
        for name, value, minimum, maximum in (
            ("block_size", self.block_size, 4, MAX_PAGED_BLOCK_SIZE),
            ("num_blocks", self.num_blocks, 1, MAX_PAGED_NUM_BLOCKS),
            (
                "max_batch_tokens",
                self.max_batch_tokens,
                1,
                MAX_PAGED_BATCH_TOKENS,
            ),
            (
                "max_active_requests",
                self.max_active_requests,
                1,
                MAX_PAGED_ACTIVE_REQUESTS,
            ),
            (
                "max_sequence_tokens",
                self.max_sequence_tokens,
                1,
                MAX_PAGED_SEQUENCE_TOKENS,
            ),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or not minimum <= value <= maximum
            ):
                raise ValueError(
                    f"{name} must be an integer between {minimum} and {maximum}"
                )
        if ceil(self.max_sequence_tokens / self.block_size) > self.num_blocks:
            raise ValueError(
                "paged num_blocks cannot hold one maximum-length request"
            )
        if (
            not isinstance(self.cpu_spill_bytes, int)
            or isinstance(self.cpu_spill_bytes, bool)
            or self.cpu_spill_bytes < 0
        ):
            raise ValueError("paged cpu_spill_bytes must be a non-negative integer")
    def to_document(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "device": self.device,
            "attentionBackend": self.attention_backend,
            "blockSize": self.block_size,
            "numBlocks": self.num_blocks,
            "maxBatchTokens": self.max_batch_tokens,
            "maxActiveRequests": self.max_active_requests,
            "maxSequenceTokens": self.max_sequence_tokens,
            "cpuSpillBytes": self.cpu_spill_bytes,
            "cpuSpill": {
                "supported": True,
                "enabled": self.cpu_spill_bytes > 0,
                "maxBytes": self.cpu_spill_bytes,
                "storedPayloadLimitBytes": self.cpu_spill_bytes,
                "restoreWorkspaceUpperBoundBytes": self.cpu_spill_bytes,
                "maxCpuPayloadUpperBoundBytes": 2 * self.cpu_spill_bytes,
                "budgetScope": "stored-tensor-payload",
                "rssBounded": False,
                "allocatorOverheadIncluded": False,
                "restoreWorkspaceIncluded": False,
                "transferByteCounters": "successful-full-payloads-only",
                "policy": "integrity-checked-request-lru",
                "restore": "verify-source-allocate-copy-publish",
            },
        }

    @property
    def configuration_id(self) -> str:
        encoded = json.dumps(
            self.to_document(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def add_paged_kv_arguments(parser: argparse.ArgumentParser) -> None:
    """Add an explicit all-or-nothing paged backend contract to a CLI."""

    parser.add_argument(
        "--paged-kv",
        action="store_true",
        help="Use Mycellios' native Transformers paged KV/COW stage backend.",
    )
    parser.add_argument("--paged-device")
    parser.add_argument(
        "--paged-attention-backend",
        choices=tuple(sorted(SUPPORTED_ATTENTION_BACKENDS)),
    )
    parser.add_argument("--paged-block-size", type=int)
    parser.add_argument("--paged-num-blocks", type=int)
    parser.add_argument("--paged-max-batch-tokens", type=int)
    parser.add_argument("--paged-max-active-requests", type=int)
    parser.add_argument("--paged-max-sequence-tokens", type=int)
    parser.add_argument(
        "--paged-cpu-spill-bytes",
        type=int,
        help=(
            "Maximum stored tensor-payload bytes in the integrity-checked CPU "
            "spill tier; this is not an RSS or allocator-overhead limit."
        ),
    )


def paged_kv_config_from_args(
    args: argparse.Namespace,
) -> HFPagedStageRuntimeConfig | None:
    """Parse a closed paged contract without silently enabling partial flags."""

    values = {
        "device": getattr(args, "paged_device", None),
        "attention_backend": getattr(args, "paged_attention_backend", None),
        "block_size": getattr(args, "paged_block_size", None),
        "num_blocks": getattr(args, "paged_num_blocks", None),
        "max_batch_tokens": getattr(args, "paged_max_batch_tokens", None),
        "max_active_requests": getattr(args, "paged_max_active_requests", None),
        "max_sequence_tokens": getattr(args, "paged_max_sequence_tokens", None),
        "cpu_spill_bytes": getattr(args, "paged_cpu_spill_bytes", None),
    }
    enabled = bool(getattr(args, "paged_kv", False))
    if not enabled:
        if any(value is not None for value in values.values()):
            raise ValueError("paged KV settings require --paged-kv")
        return None
    return HFPagedStageRuntimeConfig(
        device=values["device"] or "cpu",
        attention_backend=values["attention_backend"] or "eager",
        block_size=(
            DEFAULT_PAGED_BLOCK_SIZE
            if values["block_size"] is None
            else values["block_size"]
        ),
        num_blocks=(
            DEFAULT_PAGED_NUM_BLOCKS
            if values["num_blocks"] is None
            else values["num_blocks"]
        ),
        max_batch_tokens=(
            DEFAULT_PAGED_MAX_BATCH_TOKENS
            if values["max_batch_tokens"] is None
            else values["max_batch_tokens"]
        ),
        max_active_requests=(
            DEFAULT_PAGED_MAX_ACTIVE_REQUESTS
            if values["max_active_requests"] is None
            else values["max_active_requests"]
        ),
        max_sequence_tokens=(
            DEFAULT_PAGED_MAX_SEQUENCE_TOKENS
            if values["max_sequence_tokens"] is None
            else values["max_sequence_tokens"]
        ),
        cpu_spill_bytes=(
            0 if values["cpu_spill_bytes"] is None else values["cpu_spill_bytes"]
        ),
    )


class PagedStageCorruptionError(RuntimeError):
    """Raised when block tables/refcounts no longer match adapter state."""


@dataclass(frozen=True)
class PagedCacheMetrics:
    """Physical/logical accounting after one completed operation.

    ``unique_physical_bytes`` is active block occupancy, not process VRAM.  The
    complete KV pool (including Hugging Face's two trash blocks) is allocated
    at construction and is exposed separately as ``pool_reserved_bytes``.

    ``copied_bytes`` and ``newly_reserved_bytes`` are deltas for the operation
    that returned this record.  ``peak_workspace`` is optional because the HF
    cache manager does not expose it for eager/SDPA or ``copy_cache``.

    Spill byte counters include only complete successful payload copies.
    Timings use ``perf_counter_ns`` around each attempted D2H spill or device
    materialisation. CPU payload fields count tensor storage owned by this
    adapter, never Python/PyTorch allocator overhead or process RSS.
    """

    logical_bytes: int
    unique_physical_bytes: int
    copied_bytes: int
    newly_reserved_bytes: int
    peak_workspace: int | None
    pool_reserved_bytes: int
    active_requests: int
    free_blocks: int
    spilled_bytes: int = 0
    spilled_requests: int = 0
    spill_count: int = 0
    restore_count: int = 0
    spill_failures: int = 0
    restore_failures: int = 0
    spill_bytes_transferred: int = 0
    restore_bytes_transferred: int = 0
    spill_time_ns: int = 0
    restore_time_ns: int = 0
    restore_workspace_bytes: int = 0
    peak_restore_workspace_bytes: int = 0
    current_cpu_payload_bytes: int = 0
    peak_cpu_payload_bytes: int = 0


@dataclass(frozen=True)
class PagedRequestSnapshot:
    request_id: int
    sequence_length: int
    block_ids: tuple[int, ...]
    ref_counts: tuple[int, ...]
    residency: str = "device"
    spill_bytes: int = 0


@dataclass(frozen=True)
class PagedCacheSnapshot:
    requests: tuple[PagedRequestSnapshot, ...]
    free_blocks: int
    total_blocks: int
    block_size: int
    spilled_bytes: int = 0
    spill_limit_bytes: int = 0


@dataclass(frozen=True)
class PagedForwardResult:
    output: Any
    metrics: PagedCacheMetrics


@dataclass
class _RequestRecord:
    request_id: int
    backend_id: str
    cache_keys: list[int]


@dataclass(frozen=True)
class _SpilledRequest:
    request_id: int
    backend_id: str
    cache_keys: tuple[int, ...]
    key_blocks: tuple[torch.Tensor, ...]
    value_blocks: tuple[torch.Tensor, ...]
    byte_count: int
    digest: str


@dataclass(frozen=True)
class _PendingForward:
    request_id: int
    old_length: int
    query_length: int
    cache_keys: tuple[int, ...]
    newly_allocated_blocks: int
    read_index: tuple[torch.Tensor, ...]
    write_index: tuple[torch.Tensor, ...]
    position_ids: torch.Tensor
    attention_mask: torch.Tensor


class HFPagedStageCache:
    """Own a bounded, stage-local Hugging Face paged KV cache.

    Use :meth:`for_model` to switch an inference model to ``paged|eager`` (the
    first Windows/CUDA backend to connect) or ``paged|sdpa`` and construct the
    cache with matching device/dtype.  ``begin``, ``fork``, ``promote``,
    ``truncate`` and ``end`` implement the request lifecycle; actual cache
    writes must go through :meth:`forward_model`.
    """

    def __init__(
        self,
        config: PreTrainedConfig,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
        block_size: int = 16,
        num_blocks: int = 256,
        max_batch_tokens: int = 256,
        max_active_requests: int = 8,
        max_sequence_tokens: int = 2048,
        attention_backend: str = "eager",
        cpu_spill_bytes: int = 0,
    ) -> None:
        continuous_batching = _load_continuous_batching_types()
        if attention_backend not in SUPPORTED_ATTENTION_BACKENDS:
            raise ValueError(
                "attention_backend must be one of "
                f"{sorted(SUPPORTED_ATTENTION_BACKENDS)}, got {attention_backend!r}"
            )
        for name, value, minimum in (
            ("block_size", block_size, 4),
            ("num_blocks", num_blocks, 1),
            ("max_batch_tokens", max_batch_tokens, 1),
            ("max_active_requests", max_active_requests, 1),
            ("max_sequence_tokens", max_sequence_tokens, 1),
            ("cpu_spill_bytes", cpu_spill_bytes, 0),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
            if value < minimum:
                raise ValueError(f"{name} must be at least {minimum}")
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise TypeError("dtype must be a floating-point torch dtype")

        max_blocks_per_request = ceil(max_sequence_tokens / block_size)
        if max_blocks_per_request > num_blocks:
            raise ValueError(
                "num_blocks cannot hold one maximum-length request: "
                f"need {max_blocks_per_request}, got {num_blocks}"
            )

        resolved_device = torch.device(device)
        if resolved_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA cache requested but torch.cuda.is_available() is false")
        if resolved_device.type not in ("cpu", "cuda"):
            raise ValueError("prototype supports only CPU or CUDA devices")

        cb_config = continuous_batching.config(
            block_size=block_size,
            num_blocks=num_blocks,
            max_batch_tokens=max_batch_tokens,
            max_memory_percent=0.95,
            max_requests_per_batch=max_active_requests,
            max_blocks_per_request=max_blocks_per_request,
            allow_block_sharing=True,
            use_async_batching=False,
            use_cuda_graph=False,
        )
        distributed_helper = continuous_batching.distributed_helper(
            device_mesh=None,
            cpu_group_timeout=30.0,
        )
        cache = continuous_batching.cache(
            config=config,
            continuous_batching_config=cb_config,
            device=resolved_device,
            distributed_helper=distributed_helper,
            tp_plan={},
            dtype=dtype,
        )
        if not cache.use_prefix_sharing:
            raise ValueError(
                "paged COW requires a stage-local config containing only full-attention layers"
            )
        if (
            cache.num_groups != 1
            or len(cache.group_cache_managers) != 1
            or not cache.group_cache_managers[0].uses_block_sharing
        ):
            raise RuntimeError("unsupported Transformers paged-cache manager layout")
        block_manager = getattr(cache, "_block_manager", None)
        required_manager_fields = (
            "_id_to_block",
            "_uninit_block_ids",
            "_init_block_ids",
            "get_free_blocks",
            "free_blocks",
        )
        if block_manager is None or any(
            not hasattr(block_manager, field) for field in required_manager_fields
        ):
            raise RuntimeError("Transformers BlockManager API does not match the sealed adapter")

        self.config = config
        self.device = resolved_device
        self.dtype = dtype
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.max_batch_tokens = max_batch_tokens
        self.max_active_requests = max_active_requests
        self.max_sequence_tokens = max_sequence_tokens
        self.cpu_spill_limit_bytes = cpu_spill_bytes
        self.attention_backend = attention_backend
        self.attention_implementation = f"paged|{attention_backend}"
        self.cache = cache
        self._request_state_type = continuous_batching.request_state
        self._requests: dict[int, _RequestRecord] = {}
        self._spilled_requests: dict[int, _SpilledRequest] = {}
        self._restore_workspaces: dict[int, _SpilledRequest] = {}
        self._spilled_bytes = 0
        self._spill_count = 0
        self._restore_count = 0
        self._spill_failures = 0
        self._restore_failures = 0
        self._spill_bytes_transferred = 0
        self._restore_bytes_transferred = 0
        self._spill_time_ns = 0
        self._restore_time_ns = 0
        self._restore_workspace_bytes = 0
        self._peak_restore_workspace_bytes = 0
        self._peak_cpu_payload_bytes = 0
        self._access_clock = 0
        self._last_access: dict[int, int] = {}
        self._pending: _PendingForward | None = None
        self._next_cache_key = 1
        self._closed = False
        self._poisoned = False

        per_page_elements = cache.num_key_value_heads * cache.head_dim
        self.bytes_per_token = (
            2 * len(cache.key_cache) * per_page_elements * dtype.itemsize
        )
        self.bytes_per_block = self.bytes_per_token * block_size
        self.pool_reserved_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in (*cache.key_cache, *cache.value_cache)
        )
        self._assert_integrity()

    @classmethod
    def for_model(
        cls,
        model: nn.Module,
        *,
        attention_backend: str = "eager",
        cache_vocab_size: int | None = None,
        block_size: int = 16,
        num_blocks: int = 256,
        max_batch_tokens: int = 256,
        max_active_requests: int = 8,
        max_sequence_tokens: int = 2048,
        cpu_spill_bytes: int = 0,
    ) -> "HFPagedStageCache":
        """Configure one inference model and build a matching cache adapter."""

        _require_supported_transformers_version()
        if attention_backend not in SUPPORTED_ATTENTION_BACKENDS:
            raise ValueError(
                "attention_backend must be one of "
                f"{sorted(SUPPORTED_ATTENTION_BACKENDS)}, got {attention_backend!r}"
            )
        if cache_vocab_size is not None and (
            not isinstance(cache_vocab_size, int)
            or isinstance(cache_vocab_size, bool)
            or cache_vocab_size < 1
        ):
            raise ValueError("cache_vocab_size must be a positive integer or None")
        set_attention = getattr(model, "set_attn_implementation", None)
        if not callable(set_attention):
            raise TypeError("model does not support set_attn_implementation")
        parameters = list(model.parameters())
        if not parameters:
            raise ValueError("model has no parameters")
        model_devices = {parameter.device for parameter in parameters}
        model_dtypes = {
            parameter.dtype for parameter in parameters if parameter.dtype.is_floating_point
        }
        if len(model_devices) != 1 or len(model_dtypes) != 1:
            raise ValueError("model parameters must use one device and one floating dtype")
        set_attention(f"paged|{attention_backend}")
        model.eval()
        cache_config = model.config
        if cache_vocab_size is not None:
            # The HF memory handler includes a vocabulary projection peak.
            # Non-final selective stages have no lm_head, so retaining the
            # global vocabulary here would reject valid cache pools on 4 GB
            # devices even though that tensor is never computed locally.
            cache_config = copy.deepcopy(model.config)
            cache_config.vocab_size = cache_vocab_size
        return cls(
            cache_config,
            device=next(iter(model_devices)),
            dtype=next(iter(model_dtypes)),
            block_size=block_size,
            num_blocks=num_blocks,
            max_batch_tokens=max_batch_tokens,
            max_active_requests=max_active_requests,
            max_sequence_tokens=max_sequence_tokens,
            attention_backend=attention_backend,
            cpu_spill_bytes=cpu_spill_bytes,
        )

    def begin(self, request_id: int) -> PagedCacheMetrics:
        self._guard()
        self._require_idle()
        request_id = self._validate_request_id(request_id)
        if (
            request_id in self._requests
            or request_id in self._spilled_requests
            or request_id in self._restore_workspaces
        ):
            raise ValueError(f"request {request_id} is already active")
        if self._active_request_count() >= self.max_active_requests:
            raise ValueError("maximum active request count reached")
        backend_id = self._backend_id(request_id)
        allocated = self.cache.allocate_blocks(0, backend_id, allocated_blocks=0)
        if allocated != 0:
            self._poisoned = True
            raise PagedStageCorruptionError("zero-block BEGIN unexpectedly allocated cache")
        self._requests[request_id] = _RequestRecord(request_id, backend_id, [])
        self._touch(request_id)
        self._after_mutation()
        return self._metrics()

    def end(self, request_id: int) -> PagedCacheMetrics:
        self._guard()
        self._require_idle()
        request_id = self._validate_request_id(request_id)
        record = self._requests.get(request_id)
        if record is not None:
            try:
                self.cache.free_blocks(record.backend_id)
            except BaseException as free_error:
                manager = self.cache.group_cache_managers[0]
                if record.backend_id not in manager.block_table:
                    self._poisoned = True
                    raise PagedStageCorruptionError(
                        "request end failed after changing allocator state"
                    ) from free_error
                raise
            del self._requests[record.request_id]
        else:
            spilled = self._spilled_requests.pop(request_id, None)
            if spilled is None:
                raise ValueError(f"unknown paged request {request_id}")
            self._spilled_bytes -= spilled.byte_count
        self._last_access.pop(request_id, None)
        self._after_mutation()
        return self._metrics()

    def close(self) -> None:
        """Release request references. The preallocated tensors follow object lifetime."""

        if self._closed:
            return
        try:
            self.cache.free_all_requests()
        finally:
            self._requests.clear()
            self._spilled_requests.clear()
            self._restore_workspaces.clear()
            self._spilled_bytes = 0
            self._restore_workspace_bytes = 0
            self._last_access.clear()
            self._pending = None
            self._closed = True

    def sequence_length(self, request_id: int) -> int:
        self._guard()
        record = self._require_logical_request(request_id)
        return len(record.cache_keys)

    def metrics(self) -> PagedCacheMetrics:
        self._guard()
        return self._metrics()

    def logical_cache_bytes(self, request_id: int) -> int:
        """Return logical (unrounded, potentially shared) KV for one request."""

        self._guard()
        record = self._require_logical_request(request_id)
        return len(record.cache_keys) * self.bytes_per_token

    def spill(self, request_id: int) -> PagedCacheMetrics:
        """Move one idle resident request into the integrity-checked CPU tier."""

        self._guard()
        self._require_idle()
        request_id = self._validate_request_id(request_id)
        if request_id in self._spilled_requests:
            self._touch(request_id)
            return self._metrics()
        record = self._require_request(request_id)
        manager = self.cache.group_cache_managers[0]
        if not manager.block_table[record.backend_id]:
            raise ValueError("cannot spill a request without resident KV blocks")
        self._spill_resident_request(record)
        return self._metrics()

    def restore(self, request_id: int) -> PagedCacheMetrics:
        """Restore one spilled request before it resumes model execution."""

        self._guard()
        self._require_idle()
        request_id = self._validate_request_id(request_id)
        if request_id in self._requests:
            self._touch(request_id)
            return self._metrics()
        self._restore_spilled_request(request_id)
        return self._metrics()

    def unique_physical_bytes(self, request_ids: Sequence[int]) -> int:
        """Return cross-tier physical occupancy, deduplicating resident prefixes."""

        self._guard()
        ids = tuple(request_ids)
        if len(set(ids)) != len(ids):
            raise ValueError("request_ids must be unique")
        manager = self.cache.group_cache_managers[0]
        blocks: set[int] = set()
        spilled_bytes = 0
        for request_id in ids:
            record = self._require_logical_request(request_id)
            if isinstance(record, _SpilledRequest):
                spilled_bytes += record.byte_count
            else:
                blocks.update(manager.block_table[record.backend_id])
        return len(blocks) * self.bytes_per_block + spilled_bytes

    def project_incremental_physical_bytes(
        self,
        parent_request_id: int,
        *,
        new_leaf_count: int,
        delta_tokens: int,
    ) -> int:
        """Project extra unique blocks for new COW leaves and their deltas."""

        self._guard()
        self._require_idle()
        parent = self._require_logical_request(parent_request_id)
        for name, value in (
            ("new_leaf_count", new_leaf_count),
            ("delta_tokens", delta_tokens),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} cannot be negative")
        if new_leaf_count == 0:
            return 0
        if self._active_request_count() + new_leaf_count > self.max_active_requests:
            raise ValueError("projection exceeds maximum active request count")
        parent_length = len(parent.cache_keys)
        projected_length = parent_length + delta_tokens
        if projected_length > self.max_sequence_tokens:
            raise ValueError(
                f"projected leaf would reach {projected_length} tokens, "
                f"limit is {self.max_sequence_tokens}"
            )

        # Complete parent blocks remain shared. Each leaf needs storage from
        # the parent's first incomplete block (if any) through its own delta.
        blocks_per_leaf = (
            ceil(projected_length / self.block_size)
            - parent_length // self.block_size
        )
        required_blocks = new_leaf_count * blocks_per_leaf
        return required_blocks * self.bytes_per_block

    def project_tree_incremental_physical_bytes(
        self,
        parent_request_id: int,
        *,
        delta_tokens_by_leaf: Sequence[int],
    ) -> int:
        """Project all heterogeneous leaves in one atomic free-block check."""

        self._guard()
        self._require_idle()
        parent = self._require_logical_request(parent_request_id)
        deltas = tuple(delta_tokens_by_leaf)
        if self._active_request_count() + len(deltas) > self.max_active_requests:
            raise ValueError("projection exceeds maximum active request count")
        parent_length = len(parent.cache_keys)
        shared_complete_blocks = parent_length // self.block_size
        required_blocks = 0
        for delta_tokens in deltas:
            if not isinstance(delta_tokens, int) or isinstance(delta_tokens, bool):
                raise TypeError("delta_tokens_by_leaf must contain integers")
            if delta_tokens < 0:
                raise ValueError("delta_tokens_by_leaf cannot contain negatives")
            projected_length = parent_length + delta_tokens
            if projected_length > self.max_sequence_tokens:
                raise ValueError(
                    f"projected leaf would reach {projected_length} tokens, "
                    f"limit is {self.max_sequence_tokens}"
                )
            required_blocks += (
                ceil(projected_length / self.block_size) - shared_complete_blocks
            )
        return required_blocks * self.bytes_per_block

    def project_request_incremental_physical_bytes(
        self,
        request_id: int,
        additional_tokens: int,
    ) -> int:
        """Project new pool blocks for an already materialised request."""

        self._guard()
        self._require_idle()
        record = self._require_logical_request(request_id)
        if not isinstance(additional_tokens, int) or isinstance(additional_tokens, bool):
            raise TypeError("additional_tokens must be an integer")
        if additional_tokens < 0:
            raise ValueError("additional_tokens cannot be negative")
        current_length = len(record.cache_keys)
        projected_length = current_length + additional_tokens
        if projected_length > self.max_sequence_tokens:
            raise ValueError(
                f"projected request would reach {projected_length} tokens, "
                f"limit is {self.max_sequence_tokens}"
            )
        required_blocks = (
            ceil(projected_length / self.block_size)
            - ceil(current_length / self.block_size)
        )
        return required_blocks * self.bytes_per_block

    def available_physical_bytes(self) -> int:
        """Return a conservative one-request allocation ceiling.

        The stage ABI asks for capacity without identifying the request that
        will grow. Therefore this reports free blocks plus only the minimum
        spill-reclaimable blocks across every possible resident request being
        protected. It can underestimate a particular request, but it cannot
        count that same request as an eviction candidate.
        """

        self._guard()
        self._require_idle()
        free_blocks = self.cache.get_num_free_blocks()
        if not self._requests or self.cpu_spill_limit_bytes <= 0:
            return free_blocks * self.bytes_per_block
        guaranteed_reclaimable = min(
            self._reclaimable_block_count(protected_request_ids={request_id})
            for request_id in self._requests
        )
        return (free_blocks + guaranteed_reclaimable) * self.bytes_per_block

    def snapshot(self) -> PagedCacheSnapshot:
        """Return immutable block/refcount observability for tests and benchmarks."""

        self._guard()
        manager = self.cache.group_cache_managers[0]
        block_manager = self.cache._block_manager
        requests = []
        for request_id in sorted(self._requests):
            record = self._requests[request_id]
            block_ids = tuple(manager.block_table[record.backend_id])
            ref_counts = tuple(
                block_manager._id_to_block[block_id].ref_count for block_id in block_ids
            )
            requests.append(
                PagedRequestSnapshot(
                    request_id=request_id,
                    sequence_length=len(record.cache_keys),
                    block_ids=block_ids,
                    ref_counts=ref_counts,
                )
            )
        for request_id in sorted(self._spilled_requests):
            record = self._spilled_requests[request_id]
            requests.append(
                PagedRequestSnapshot(
                    request_id=request_id,
                    sequence_length=len(record.cache_keys),
                    block_ids=(),
                    ref_counts=(),
                    residency="cpu-spill",
                    spill_bytes=record.byte_count,
                )
            )
        requests.sort(key=lambda request: request.request_id)
        return PagedCacheSnapshot(
            requests=tuple(requests),
            free_blocks=self.cache.get_num_free_blocks(),
            total_blocks=self.num_blocks,
            block_size=self.block_size,
            spilled_bytes=self._spilled_bytes,
            spill_limit_bytes=self.cpu_spill_limit_bytes,
        )

    def fork(self, child_request_id: int, parent_request_id: int) -> PagedCacheMetrics:
        """Fork one request using shared complete blocks and a private tail."""

        self._guard()
        self._require_idle()
        child_request_id = self._validate_request_id(child_request_id)
        parent = self._ensure_resident_request(parent_request_id)
        if child_request_id == parent.request_id:
            raise ValueError("fork child and parent request IDs must differ")
        if (
            child_request_id in self._requests
            or child_request_id in self._spilled_requests
            or child_request_id in self._restore_workspaces
        ):
            raise ValueError(f"fork child request {child_request_id} is already active")
        if self._active_request_count() >= self.max_active_requests:
            raise ValueError("maximum active request count reached")
        private_tail_blocks = 1 if len(parent.cache_keys) % self.block_size else 0
        if private_tail_blocks and self.cpu_spill_limit_bytes > 0:
            self._ensure_free_blocks(
                private_tail_blocks,
                protected_request_ids={parent.request_id},
            )
        if self.cache.compute_max_num_forks(parent.backend_id) < 1:
            raise MemoryError("paged cache has no capacity for a private fork tail")

        child_backend_id = self._backend_id(child_request_id)
        try:
            source_blocks, destination_blocks = self.cache.fork_request(
                parent.backend_id, [child_backend_id]
            )
            if len(source_blocks) != len(destination_blocks):
                raise PagedStageCorruptionError("fork copy source/destination mismatch")
            if source_blocks:
                self.cache.copy_cache(source_blocks, destination_blocks)
        except BaseException:
            manager = self.cache.group_cache_managers[0]
            if child_backend_id in manager.block_table:
                self.cache.free_blocks(child_backend_id)
            raise

        self._requests[child_request_id] = _RequestRecord(
            child_request_id,
            child_backend_id,
            parent.cache_keys.copy(),
        )
        self._touch(parent.request_id)
        self._touch(child_request_id)
        self._after_mutation()
        copied_bytes = len(source_blocks) * self.bytes_per_block
        newly_reserved = len(destination_blocks) * self.bytes_per_block
        return self._metrics(
            copied_bytes=copied_bytes,
            newly_reserved_bytes=newly_reserved,
        )

    def promote(self, parent_request_id: int, child_request_id: int) -> PagedCacheMetrics:
        """Replace the parent with a selected child's KV state without copying.

        Promotion is tier-aware: a spilled child remains spilled and a resident
        child keeps its block table. It never restores parent and child
        sequentially, which could otherwise evict them back and forth when the
        device pool is full.
        """

        self._guard()
        self._require_idle()
        parent_request_id = self._validate_request_id(parent_request_id)
        child_request_id = self._validate_request_id(child_request_id)
        parent = self._require_logical_request(parent_request_id)
        child = self._require_logical_request(child_request_id)
        if parent.request_id == child.request_id:
            raise ValueError("promote parent and child request IDs must differ")

        manager = self.cache.group_cache_managers[0]
        if isinstance(child, _SpilledRequest):
            self._validate_spilled_request(child)
            if isinstance(parent, _RequestRecord):
                if manager.block_table.get(parent.backend_id) is None:
                    self._poisoned = True
                    raise PagedStageCorruptionError(
                        "resident promote parent has no block table"
                    )
                try:
                    self.cache.free_blocks(parent.backend_id)
                except BaseException as free_error:
                    if parent.backend_id not in manager.block_table:
                        self._poisoned = True
                        raise PagedStageCorruptionError(
                            "promote parent release changed allocator state "
                            "before failing"
                        ) from free_error
                    raise
                del self._requests[parent.request_id]
            else:
                del self._spilled_requests[parent.request_id]
                self._spilled_bytes -= parent.byte_count

            del self._spilled_requests[child.request_id]
            self._spilled_requests[parent.request_id] = _SpilledRequest(
                request_id=parent.request_id,
                backend_id=parent.backend_id,
                cache_keys=child.cache_keys,
                key_blocks=child.key_blocks,
                value_blocks=child.value_blocks,
                byte_count=child.byte_count,
                digest=child.digest,
            )
        else:
            child_blocks = manager.block_table.get(child.backend_id)
            if child_blocks is None:
                self._poisoned = True
                raise PagedStageCorruptionError(
                    "resident promote child has no block table"
                )
            moved_blocks = child_blocks
            if isinstance(parent, _RequestRecord):
                if manager.block_table.get(parent.backend_id) is None:
                    self._poisoned = True
                    raise PagedStageCorruptionError(
                        "resident promote parent has no block table"
                    )
                try:
                    self.cache.free_blocks(parent.backend_id)
                except BaseException as free_error:
                    if parent.backend_id not in manager.block_table:
                        self._poisoned = True
                        raise PagedStageCorruptionError(
                            "promote parent release changed allocator state "
                            "before failing"
                        ) from free_error
                    raise
                parent.cache_keys = child.cache_keys
            else:
                del self._spilled_requests[parent.request_id]
                self._spilled_bytes -= parent.byte_count
                parent = _RequestRecord(
                    request_id=parent.request_id,
                    backend_id=parent.backend_id,
                    cache_keys=list(child.cache_keys),
                )
                self._requests[parent.request_id] = parent
            manager.block_table.pop(child.backend_id)
            manager.block_table[parent.backend_id] = moved_blocks
            del self._requests[child.request_id]

        self._last_access.pop(child.request_id, None)
        self._touch(parent.request_id)
        self._after_mutation()
        return self._metrics()

    def truncate(self, request_id: int, token_count: int) -> PagedCacheMetrics:
        """Crop to an accepted prefix, privatizing a retained complete tail."""

        self._guard()
        self._require_idle()
        record = self._ensure_resident_request(request_id)
        if not isinstance(token_count, int) or isinstance(token_count, bool):
            raise TypeError("token_count must be an integer")
        current_length = len(record.cache_keys)
        if not 0 <= token_count <= current_length:
            raise ValueError(
                f"cannot truncate request {request_id} from {current_length} to {token_count}"
            )
        if token_count == current_length:
            return self._metrics()

        manager = self.cache.group_cache_managers[0]
        block_manager = self.cache._block_manager
        old_blocks = manager.block_table[record.backend_id]
        retained_count = ceil(token_count / self.block_size) if token_count else 0
        retained = old_blocks[:retained_count]
        removed = old_blocks[retained_count:]
        copied_bytes = 0
        newly_reserved_bytes = 0

        # A complete block is immutable and may be shared. If rollback lands
        # inside it, create a private incomplete copy before future appends.
        if token_count % self.block_size and retained:
            retained_tail = retained[-1]
            retained_tail_meta = block_manager._id_to_block[retained_tail]
            if retained_tail_meta.is_complete:
                parent_block = retained[-2] if len(retained) > 1 else None
                self._ensure_free_blocks(
                    1,
                    protected_request_ids={record.request_id},
                )
                replacement = block_manager.get_free_blocks(
                    1,
                    last_block_id=parent_block,
                    shareable=True,
                    group_id=0,
                )
                if replacement is None:
                    raise MemoryError("paged cache cannot privatize the truncated tail")
                replacement_id = replacement[0]
                try:
                    self.cache.copy_cache([retained_tail], [replacement_id])
                except BaseException:
                    block_manager.free_blocks([replacement_id], shareable=True)
                    raise
                retained[-1] = replacement_id
                copied_bytes = self.bytes_per_block
                newly_reserved_bytes = self.bytes_per_block
                removed = [retained_tail, *removed]

        manager.block_table[record.backend_id] = retained
        if removed:
            block_manager.free_blocks(removed, shareable=True)
        del record.cache_keys[token_count:]
        self._touch(record.request_id)
        self._after_mutation()
        return self._metrics(
            copied_bytes=copied_bytes,
            newly_reserved_bytes=newly_reserved_bytes,
        )

    def forward_model(
        self,
        model: nn.Module,
        request_id: int,
        *,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **model_kwargs: Any,
    ) -> PagedForwardResult:
        """Run and atomically publish one cache-writing model forward.

        ``inputs_embeds`` is the intended hook for a selectively loaded middle
        stage.  The current pipeline still needs an integration change before
        it can route its hidden activations through this adapter.
        """

        self._guard()
        self._require_idle()
        record = self._ensure_resident_request(request_id)
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("provide exactly one of input_ids or inputs_embeds")
        input_tensor = input_ids if input_ids is not None else inputs_embeds
        assert input_tensor is not None
        expected_rank = 2 if input_ids is not None else 3
        if input_tensor.ndim != expected_rank or input_tensor.shape[0] != 1:
            expected = "[1, tokens]" if input_ids is not None else "[1, tokens, hidden]"
            raise ValueError(f"model input must have shape {expected}")
        query_length = int(input_tensor.shape[1])
        if query_length < 1:
            raise ValueError("model input must contain at least one token")
        if input_tensor.device != self.device:
            raise ValueError(
                f"model input is on {input_tensor.device}, cache is on {self.device}"
            )
        if input_ids is not None and input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("input_ids must contain integer token IDs")
        if inputs_embeds is not None and not inputs_embeds.dtype.is_floating_point:
            raise TypeError("inputs_embeds must be floating point")
        if model.training:
            raise RuntimeError("paged stage prototype only supports eval-mode inference")
        implementation = getattr(model.config, "_attn_implementation", None)
        if implementation != self.attention_implementation:
            raise RuntimeError(
                "model attention implementation does not match cache: "
                f"expected {self.attention_implementation!r}, got {implementation!r}"
            )
        reserved_kwargs = {
            "attention_mask",
            "position_ids",
            "past_key_values",
            "use_cache",
            "cache",
            "read_index",
            "write_index",
        }
        conflicts = reserved_kwargs.intersection(model_kwargs)
        if conflicts:
            raise ValueError(f"reserved paged-forward kwargs supplied: {sorted(conflicts)}")

        pending = self._prepare_forward(record, query_length)
        try:
            with torch.inference_mode():
                output = model(
                    input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                    attention_mask=pending.attention_mask,
                    position_ids=pending.position_ids,
                    use_cache=False,
                    cache=self.cache,
                    read_index=list(pending.read_index),
                    write_index=list(pending.write_index),
                    **model_kwargs,
                )
            metrics = self._commit_forward(pending)
        except BaseException:
            self._discard_pending_request()
            raise
        return PagedForwardResult(output=output, metrics=metrics)

    def _prepare_forward(
        self, record: _RequestRecord, query_length: int
    ) -> _PendingForward:
        if query_length > self.max_batch_tokens:
            raise ValueError(
                f"query has {query_length} tokens, limit is {self.max_batch_tokens}"
            )
        old_length = len(record.cache_keys)
        new_length = old_length + query_length
        if new_length > self.max_sequence_tokens:
            raise ValueError(
                f"request would reach {new_length} tokens, limit is {self.max_sequence_tokens}"
            )
        old_blocks = ceil(old_length / self.block_size) if old_length else 0
        new_blocks = ceil(new_length / self.block_size)
        blocks_to_allocate = new_blocks - old_blocks
        self._ensure_free_blocks(
            blocks_to_allocate,
            protected_request_ids={record.request_id},
        )
        allocated = self.cache.allocate_blocks(
            blocks_to_allocate,
            record.backend_id,
            allocated_blocks=old_blocks,
        )
        if allocated is None:
            raise MemoryError("paged cache has insufficient free blocks")
        if allocated != blocks_to_allocate:
            self._poisoned = True
            raise PagedStageCorruptionError(
                "full-attention allocator returned an unexpected block count"
            )

        write_lists: list[list[int]] = [[] for _ in range(self.cache.num_groups)]
        read_lists: list[list[int]] | None
        read_lists = None if old_length == 0 else [[] for _ in range(self.cache.num_groups)]
        try:
            self.cache.extend_read_and_write_indices(
                record.backend_id,
                old_length,
                query_length,
                read_lists,
                write_lists,
            )
            read_index = tuple(
                torch.empty(0, dtype=torch.int64, device=self.device)
                for _ in range(self.cache.num_groups)
            )
            if read_lists is not None:
                read_index = tuple(
                    torch.tensor(values, dtype=torch.int64, device=self.device)
                    for values in read_lists
                )
            write_index = tuple(
                torch.tensor(values, dtype=torch.int64, device=self.device)
                for values in write_lists
            )
            position_ids = torch.arange(
                old_length,
                new_length,
                dtype=torch.int64,
                device=self.device,
            ).unsqueeze(0)
            attention_mask = self._causal_mask(old_length, query_length)
            cache_keys = tuple(self._take_cache_keys(query_length))
        except BaseException:
            if blocks_to_allocate:
                manager = self.cache.group_cache_managers[0]
                just_allocated = manager.block_table[record.backend_id][-blocks_to_allocate:]
                del manager.block_table[record.backend_id][-blocks_to_allocate:]
                self.cache._block_manager.free_blocks(just_allocated, shareable=True)
            raise

        pending = _PendingForward(
            request_id=record.request_id,
            old_length=old_length,
            query_length=query_length,
            cache_keys=cache_keys,
            newly_allocated_blocks=blocks_to_allocate,
            read_index=read_index,
            write_index=write_index,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
        self._pending = pending
        return pending

    def _commit_forward(self, pending: _PendingForward) -> PagedCacheMetrics:
        if self._pending is not pending:
            self._poisoned = True
            raise PagedStageCorruptionError("paged forward commit token does not match")
        record = self._require_request(pending.request_id)
        if len(record.cache_keys) != pending.old_length:
            self._poisoned = True
            raise PagedStageCorruptionError("request length changed during paged forward")
        record.cache_keys.extend(pending.cache_keys)
        new_length = len(record.cache_keys)
        newly_complete = (
            new_length // self.block_size - pending.old_length // self.block_size
        )
        if newly_complete:
            state = self._request_state_type(
                request_id=record.backend_id,
                initial_tokens=record.cache_keys.copy(),
            )
            self.cache.mark_shareable_blocks_as_complete(state, newly_complete)
        self._pending = None
        self._touch(record.request_id)
        self._after_mutation()
        return self._metrics(
            newly_reserved_bytes=(
                pending.newly_allocated_blocks * self.bytes_per_block
            )
        )

    def _discard_pending_request(self) -> None:
        pending = self._pending
        self._pending = None
        if pending is None:
            return
        record = self._requests.pop(pending.request_id, None)
        if record is None:
            self._poisoned = True
            return
        self._last_access.pop(pending.request_id, None)
        try:
            self.cache.free_blocks(record.backend_id)
            self._assert_integrity()
        except BaseException:
            self._poisoned = True

    def _causal_mask(self, past_length: int, query_length: int) -> torch.Tensor:
        key_length = past_length + query_length
        query_positions = torch.arange(
            past_length,
            key_length,
            device=self.device,
        ).unsqueeze(1)
        key_positions = torch.arange(key_length, device=self.device).unsqueeze(0)
        future = key_positions > query_positions
        mask = torch.zeros(
            (1, 1, query_length, key_length),
            dtype=self.dtype,
            device=self.device,
        )
        return mask.masked_fill(future.unsqueeze(0).unsqueeze(0), torch.finfo(self.dtype).min)

    def _take_cache_keys(self, count: int) -> list[int]:
        end = self._next_cache_key + count
        if end - 1 > MAX_OPAQUE_CACHE_KEY:
            raise RuntimeError("opaque paged-cache key space exhausted")
        keys = list(range(self._next_cache_key, end))
        self._next_cache_key = end
        return keys

    def _metrics(
        self,
        *,
        copied_bytes: int = 0,
        newly_reserved_bytes: int = 0,
    ) -> PagedCacheMetrics:
        logical_tokens = sum(len(record.cache_keys) for record in self._requests.values())
        logical_tokens += sum(
            len(record.cache_keys) for record in self._spilled_requests.values()
        )
        unique_blocks = len(self._unique_block_ids())
        return PagedCacheMetrics(
            logical_bytes=logical_tokens * self.bytes_per_token,
            unique_physical_bytes=unique_blocks * self.bytes_per_block,
            copied_bytes=copied_bytes,
            newly_reserved_bytes=newly_reserved_bytes,
            peak_workspace=None,
            pool_reserved_bytes=self.pool_reserved_bytes,
            active_requests=self._active_request_count(),
            free_blocks=self.cache.get_num_free_blocks(),
            spilled_bytes=self._spilled_bytes,
            spilled_requests=len(self._spilled_requests),
            spill_count=self._spill_count,
            restore_count=self._restore_count,
            spill_failures=self._spill_failures,
            restore_failures=self._restore_failures,
            spill_bytes_transferred=self._spill_bytes_transferred,
            restore_bytes_transferred=self._restore_bytes_transferred,
            spill_time_ns=self._spill_time_ns,
            restore_time_ns=self._restore_time_ns,
            restore_workspace_bytes=self._restore_workspace_bytes,
            peak_restore_workspace_bytes=self._peak_restore_workspace_bytes,
            current_cpu_payload_bytes=(
                self._spilled_bytes + self._restore_workspace_bytes
            ),
            peak_cpu_payload_bytes=self._peak_cpu_payload_bytes,
        )

    def _unique_block_ids(self) -> set[int]:
        unique: set[int] = set()
        for manager in self.cache.group_cache_managers:
            for blocks in manager.block_table.values():
                unique.update(blocks)
        return unique

    def _guard(self) -> None:
        if self._closed:
            raise RuntimeError("paged stage cache is closed")
        if self._poisoned:
            raise PagedStageCorruptionError("paged stage cache is poisoned")
        try:
            self._assert_integrity()
        except BaseException as error:
            self._poisoned = True
            if isinstance(error, PagedStageCorruptionError):
                raise
            raise PagedStageCorruptionError("paged stage integrity check failed") from error

    def _after_mutation(self) -> None:
        try:
            self._assert_integrity()
        except BaseException as error:
            self._poisoned = True
            if isinstance(error, PagedStageCorruptionError):
                raise
            raise PagedStageCorruptionError("paged stage mutation corrupted state") from error

    def _assert_integrity(self) -> None:
        if self._closed:
            return
        if self._active_request_count() > self.max_active_requests:
            raise PagedStageCorruptionError("active request limit exceeded")
        request_sets = (
            set(self._requests),
            set(self._spilled_requests),
            set(self._restore_workspaces),
        )
        if any(
            left.intersection(right)
            for index, left in enumerate(request_sets)
            for right in request_sets[index + 1 :]
        ):
            raise PagedStageCorruptionError(
                "request appears in more than one KV residency tier"
            )
        expected_spilled_bytes = sum(
            record.byte_count for record in self._spilled_requests.values()
        )
        if expected_spilled_bytes != self._spilled_bytes:
            raise PagedStageCorruptionError(
                "CPU spill accounting does not match stored requests"
            )
        if not 0 <= self._spilled_bytes <= self.cpu_spill_limit_bytes:
            raise PagedStageCorruptionError("CPU spill tier exceeded its byte limit")
        expected_workspace_bytes = sum(
            record.byte_count for record in self._restore_workspaces.values()
        )
        if expected_workspace_bytes != self._restore_workspace_bytes:
            raise PagedStageCorruptionError(
                "restore workspace accounting does not match transitioning requests"
            )
        if self._restore_workspace_bytes < 0:
            raise PagedStageCorruptionError("restore workspace bytes cannot be negative")
        if (
            len(self._restore_workspaces) > 1
            or self._restore_workspace_bytes > self.cpu_spill_limit_bytes
        ):
            raise PagedStageCorruptionError(
                "restore workspace exceeded its one-target payload bound"
            )
        current_cpu_payload = self._spilled_bytes + self._restore_workspace_bytes
        if current_cpu_payload > 2 * self.cpu_spill_limit_bytes:
            raise PagedStageCorruptionError(
                "combined stored and restore-workspace payload exceeded its bound"
            )
        if self._peak_cpu_payload_bytes < current_cpu_payload:
            raise PagedStageCorruptionError(
                "CPU payload peak accounting moved below current payload"
            )
        active_ids = (
            set(self._requests)
            .union(self._spilled_requests)
            .union(self._restore_workspaces)
        )
        if set(self._last_access) != active_ids:
            raise PagedStageCorruptionError("request access ledger differs from active requests")
        expected_backend_ids = {record.backend_id for record in self._requests.values()}
        if len(expected_backend_ids) != len(self._requests):
            raise PagedStageCorruptionError("duplicate backend request IDs")

        observed_refs: Counter[int] = Counter()
        for manager in self.cache.group_cache_managers:
            if set(manager.block_table) != expected_backend_ids:
                raise PagedStageCorruptionError("adapter requests and block tables differ")
            for request_id, record in self._requests.items():
                blocks = manager.block_table[record.backend_id]
                required = (
                    ceil(len(record.cache_keys) / self.block_size)
                    if record.cache_keys
                    else 0
                )
                if len(blocks) != required:
                    raise PagedStageCorruptionError(
                        f"request {request_id} has {len(blocks)} blocks, expected {required}"
                    )
                if len(set(blocks)) != len(blocks):
                    raise PagedStageCorruptionError(
                        f"request {request_id} repeats a physical block"
                    )
                observed_refs.update(blocks)

        block_manager = self.cache._block_manager
        for block_id, count in observed_refs.items():
            block = block_manager._id_to_block.get(block_id)
            if block is None:
                raise PagedStageCorruptionError(f"active block {block_id} has no metadata")
            if block.ref_count != count:
                raise PagedStageCorruptionError(
                    f"block {block_id} refcount is {block.ref_count}, expected {count}"
                )
        for block_id, block in block_manager._id_to_block.items():
            expected = observed_refs.get(block_id, 0)
            if block.ref_count != expected:
                raise PagedStageCorruptionError(
                    f"block {block_id} refcount is {block.ref_count}, expected {expected}"
                )
        if self.cache.get_num_free_blocks() + len(observed_refs) != self.num_blocks:
            raise PagedStageCorruptionError("free and active block counts do not cover the pool")

    def _active_request_count(self) -> int:
        return (
            len(self._requests)
            + len(self._spilled_requests)
            + len(self._restore_workspaces)
        )

    def _touch(self, request_id: int) -> None:
        self._access_clock += 1
        self._last_access[request_id] = self._access_clock

    def _record_cpu_payload_peak(self) -> None:
        self._peak_cpu_payload_bytes = max(
            self._peak_cpu_payload_bytes,
            self._spilled_bytes + self._restore_workspace_bytes,
        )

    def _ensure_free_blocks(
        self,
        required_blocks: int,
        *,
        protected_request_ids: set[int],
    ) -> None:
        planned = self._plan_spills(
            required_blocks,
            protected_request_ids=protected_request_ids,
        )
        self._spill_candidates_atomically(planned)

    def _plan_spills(
        self,
        required_blocks: int,
        *,
        protected_request_ids: set[int],
        spill_credit_bytes: int = 0,
    ) -> tuple[_RequestRecord, ...]:
        """Return an LRU spill plan without changing residency or refcounts."""

        free_blocks = self.cache.get_num_free_blocks()
        if required_blocks <= free_blocks:
            return ()
        if self.cpu_spill_limit_bytes <= 0:
            raise MemoryError("paged cache has insufficient free blocks")
        missing = required_blocks - free_blocks
        planned, released_blocks = self._select_spill_candidates(
            protected_request_ids=protected_request_ids,
            available_spill_bytes=(
                self.cpu_spill_limit_bytes
                - self._spilled_bytes
                + spill_credit_bytes
            ),
            stop_after_blocks=missing,
        )
        if released_blocks < missing:
            raise MemoryError(
                "paged cache cannot free enough blocks within the bounded CPU spill tier"
            )
        return planned

    def _reclaimable_block_count(
        self,
        *,
        protected_request_ids: set[int],
    ) -> int:
        if self.cpu_spill_limit_bytes <= 0:
            return 0
        _, released_blocks = self._select_spill_candidates(
            protected_request_ids=protected_request_ids,
            available_spill_bytes=(
                self.cpu_spill_limit_bytes - self._spilled_bytes
            ),
            stop_after_blocks=None,
        )
        return released_blocks

    def _select_spill_candidates(
        self,
        *,
        protected_request_ids: set[int],
        available_spill_bytes: int,
        stop_after_blocks: int | None,
    ) -> tuple[tuple[_RequestRecord, ...], int]:
        """Select candidates and count only blocks whose last ref is released."""

        if available_spill_bytes < 0:
            raise PagedStageCorruptionError("CPU spill accounting exceeded its limit")
        manager = self.cache.group_cache_managers[0]
        block_manager = self.cache._block_manager
        candidates = sorted(
            (
                record
                for request_id, record in self._requests.items()
                if request_id not in protected_request_ids
                and manager.block_table[record.backend_id]
            ),
            key=lambda record: (self._last_access[record.request_id], record.request_id),
        )
        planned: list[_RequestRecord] = []
        planned_refs: Counter[int] = Counter()
        planned_bytes = 0
        released_blocks = 0
        for candidate in candidates:
            block_ids = manager.block_table[candidate.backend_id]
            candidate_bytes = len(block_ids) * self.bytes_per_block
            if planned_bytes + candidate_bytes > available_spill_bytes:
                continue
            planned.append(candidate)
            planned_bytes += candidate_bytes
            planned_refs.update(block_ids)
            released_blocks = sum(
                1
                for block_id, selected_refs in planned_refs.items()
                if selected_refs == block_manager._id_to_block[block_id].ref_count
            )
            if (
                stop_after_blocks is not None
                and released_blocks >= stop_after_blocks
            ):
                break
        return tuple(planned), released_blocks

    def _spill_candidates_atomically(
        self,
        planned: Sequence[_RequestRecord],
    ) -> tuple[int, ...]:
        """Apply a preflighted plan or restore every completed eviction."""

        if not planned:
            return ()
        access_before = dict(self._last_access)
        clock_before = self._access_clock
        spilled_ids: list[int] = []
        try:
            for candidate in planned:
                self._spill_resident_request(candidate)
                spilled_ids.append(candidate.request_id)
        except BaseException as error:
            try:
                self._rollback_spilled_candidates(tuple(spilled_ids))
                self._last_access = access_before
                self._access_clock = clock_before
                self._assert_integrity()
            except BaseException as rollback_error:
                self._poisoned = True
                raise PagedStageCorruptionError(
                    "CPU spill plan failed and residency rollback was incomplete"
                ) from rollback_error
            raise error
        return tuple(spilled_ids)

    def _rollback_spilled_candidates(self, request_ids: Sequence[int]) -> None:
        for request_id in reversed(tuple(request_ids)):
            spilled = self._spilled_requests.get(request_id)
            if spilled is None:
                raise PagedStageCorruptionError(
                    f"spill rollback lost request {request_id}"
                )
            block_count = self._validate_spilled_request(spilled)
            if block_count > self.cache.get_num_free_blocks():
                raise PagedStageCorruptionError(
                    "spill rollback cannot recover the original device residency"
                )
            try:
                record = self._copy_spilled_to_device(spilled, block_count)
                self._requests[request_id] = record
                del self._spilled_requests[request_id]
                self._spilled_bytes -= spilled.byte_count
                self._restore_count += 1
                self._restore_bytes_transferred += spilled.byte_count
                self._touch(request_id)
                self._after_mutation()
            except BaseException:
                self._restore_failures += 1
                raise

    def _spill_resident_request(self, record: _RequestRecord) -> None:
        started_ns = perf_counter_ns()
        try:
            manager = self.cache.group_cache_managers[0]
            block_ids = tuple(manager.block_table[record.backend_id])
            byte_count = len(block_ids) * self.bytes_per_block
            if self._spilled_bytes + byte_count > self.cpu_spill_limit_bytes:
                raise MemoryError(
                    "request KV exceeds the remaining bounded CPU spill capacity"
                )
            key_blocks = tuple(
                self._copy_blocks_to_cpu(tensor, block_ids)
                for tensor in self.cache.key_cache
            )
            value_blocks = tuple(
                self._copy_blocks_to_cpu(tensor, block_ids)
                for tensor in self.cache.value_cache
            )
            actual_bytes = sum(
                tensor.numel() * tensor.element_size()
                for tensor in (*key_blocks, *value_blocks)
            )
            if actual_bytes != byte_count:
                raise PagedStageCorruptionError(
                    f"CPU spill copied {actual_bytes} bytes, expected {byte_count}"
                )
            cache_keys = tuple(record.cache_keys)
            digest = self._spill_digest(cache_keys, key_blocks, value_blocks)
            spilled = _SpilledRequest(
                request_id=record.request_id,
                backend_id=record.backend_id,
                cache_keys=cache_keys,
                key_blocks=key_blocks,
                value_blocks=value_blocks,
                byte_count=byte_count,
                digest=digest,
            )

            try:
                self.cache.free_blocks(record.backend_id)
            except BaseException as free_error:
                # HF removes the block table before decrementing its refs. If
                # an unexpected failure happens after that point, retain the
                # completed CPU copy as the logical request and poison the
                # adapter instead of silently publishing a missing resident.
                if record.backend_id not in manager.block_table:
                    del self._requests[record.request_id]
                    self._spilled_requests[record.request_id] = spilled
                    self._spilled_bytes += byte_count
                    self._record_cpu_payload_peak()
                    self._touch(record.request_id)
                    self._poisoned = True
                    raise PagedStageCorruptionError(
                        "device block release failed after changing allocator state"
                    ) from free_error
                raise
            del self._requests[record.request_id]
            self._spilled_requests[record.request_id] = spilled
            self._spilled_bytes += byte_count
            self._record_cpu_payload_peak()
            self._spill_count += 1
            self._spill_bytes_transferred += byte_count
            self._touch(record.request_id)
            self._after_mutation()
        except BaseException:
            self._spill_failures += 1
            raise
        finally:
            self._spill_time_ns += perf_counter_ns() - started_ns

    def _restore_spilled_request(
        self,
        request_id: int,
        *,
        protected_request_ids: set[int] | None = None,
    ) -> _RequestRecord:
        """Restore one request while transactionally reusing its spill slot."""

        request_id = self._validate_request_id(request_id)
        spilled = self._spilled_requests.get(request_id)
        if spilled is None:
            raise ValueError(f"unknown paged request {request_id}")
        detached = False
        candidate_ids: tuple[int, ...] = ()
        access_before = dict(self._last_access)
        clock_before = self._access_clock
        try:
            block_count = self._validate_spilled_request(spilled)
            protected = set(protected_request_ids or ())
            protected.add(request_id)
            planned = self._plan_spills(
                block_count,
                protected_request_ids=protected,
                spill_credit_bytes=spilled.byte_count,
            )

            del self._spilled_requests[request_id]
            self._spilled_bytes -= spilled.byte_count
            self._restore_workspaces[request_id] = spilled
            self._restore_workspace_bytes += spilled.byte_count
            self._record_cpu_payload_peak()
            self._peak_restore_workspace_bytes = max(
                self._peak_restore_workspace_bytes,
                self._restore_workspace_bytes,
            )
            detached = True
            self._after_mutation()

            candidate_ids = self._spill_candidates_atomically(planned)
            record = self._copy_spilled_to_device(spilled, block_count)
            self._requests[request_id] = record
            del self._restore_workspaces[request_id]
            self._restore_workspace_bytes -= spilled.byte_count
            detached = False
            self._restore_count += 1
            self._restore_bytes_transferred += spilled.byte_count
            self._touch(request_id)
            self._after_mutation()
            return record
        except BaseException as error:
            self._restore_failures += 1
            if detached:
                try:
                    manager = self.cache.group_cache_managers[0]
                    self._requests.pop(request_id, None)
                    if spilled.backend_id in manager.block_table:
                        self.cache.free_blocks(spilled.backend_id)
                    self._rollback_spilled_candidates(candidate_ids)
                    del self._restore_workspaces[request_id]
                    self._restore_workspace_bytes -= spilled.byte_count
                    self._spilled_requests[request_id] = spilled
                    self._spilled_bytes += spilled.byte_count
                    self._last_access = access_before
                    self._access_clock = clock_before
                    self._assert_integrity()
                except BaseException as rollback_error:
                    self._poisoned = True
                    raise PagedStageCorruptionError(
                        "CPU restore failed and residency rollback was incomplete"
                    ) from rollback_error
            raise error

    def _validate_spilled_request(self, spilled: _SpilledRequest) -> int:
        try:
            observed_digest = self._spill_digest(
                spilled.cache_keys,
                spilled.key_blocks,
                spilled.value_blocks,
            )
        except BaseException:
            self._poisoned = True
            raise
        if observed_digest != spilled.digest:
            self._poisoned = True
            raise PagedStageCorruptionError(
                f"CPU spill integrity check failed for request {spilled.request_id}"
            )
        if (
            len(spilled.key_blocks) != len(self.cache.key_cache)
            or len(spilled.value_blocks) != len(self.cache.value_cache)
            or not spilled.key_blocks
        ):
            self._poisoned = True
            raise PagedStageCorruptionError(
                "CPU spill layer count does not match the device cache"
            )
        block_count = len(spilled.key_blocks[0])
        expected_shape = (
            block_count,
            self.block_size,
            self.cache.num_key_value_heads,
            self.cache.head_dim,
        )
        for source, destination in zip(
            (*spilled.key_blocks, *spilled.value_blocks),
            (*self.cache.key_cache, *self.cache.value_cache),
            strict=True,
        ):
            if source.shape != expected_shape or source.dtype != destination.dtype:
                self._poisoned = True
                raise PagedStageCorruptionError(
                    "CPU spill tensor geometry or dtype does not match the device cache"
                )
        actual_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in (*spilled.key_blocks, *spilled.value_blocks)
        )
        if actual_bytes != spilled.byte_count:
            self._poisoned = True
            raise PagedStageCorruptionError(
                f"CPU spill byte count is {actual_bytes}, expected {spilled.byte_count}"
            )
        if (
            block_count * self.bytes_per_block != spilled.byte_count
            or ceil(len(spilled.cache_keys) / self.block_size) != block_count
        ):
            self._poisoned = True
            raise PagedStageCorruptionError(
                "CPU spill block geometry does not match its integrity-checked payload"
            )
        return block_count

    def _copy_spilled_to_device(
        self,
        spilled: _SpilledRequest,
        block_count: int,
    ) -> _RequestRecord:
        started_ns = perf_counter_ns()
        try:
            allocated = self.cache.allocate_blocks(
                block_count,
                spilled.backend_id,
                allocated_blocks=0,
            )
            if allocated != block_count:
                if allocated is not None:
                    self.cache.free_blocks(spilled.backend_id)
                raise MemoryError(
                    "paged cache could not allocate all restored KV blocks"
                )

            manager = self.cache.group_cache_managers[0]
            try:
                destination_blocks = tuple(manager.block_table[spilled.backend_id])
                if len(destination_blocks) != block_count:
                    raise PagedStageCorruptionError(
                        "restored request received an unexpected block table"
                    )
                self._copy_cpu_blocks_to_device(
                    spilled.key_blocks,
                    self.cache.key_cache,
                    destination_blocks,
                )
                self._copy_cpu_blocks_to_device(
                    spilled.value_blocks,
                    self.cache.value_cache,
                    destination_blocks,
                )
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                complete_blocks = len(spilled.cache_keys) // self.block_size
                if complete_blocks:
                    state = self._request_state_type(
                        request_id=spilled.backend_id,
                        initial_tokens=list(spilled.cache_keys),
                    )
                    self.cache.mark_shareable_blocks_as_complete(
                        state,
                        complete_blocks,
                    )
            except BaseException:
                self.cache.free_blocks(spilled.backend_id)
                raise
            return _RequestRecord(
                request_id=spilled.request_id,
                backend_id=spilled.backend_id,
                cache_keys=list(spilled.cache_keys),
            )
        finally:
            self._restore_time_ns += perf_counter_ns() - started_ns

    def _copy_blocks_to_cpu(
        self,
        tensor: torch.Tensor,
        block_ids: tuple[int, ...],
    ) -> torch.Tensor:
        shape = (
            len(block_ids),
            self.block_size,
            self.cache.num_key_value_heads,
            self.cache.head_dim,
        )
        output = torch.empty(
            shape,
            dtype=tensor.dtype,
            device="cpu",
            pin_memory=self.device.type == "cuda",
        )
        source = tensor.view(
            -1,
            self.block_size,
            self.cache.num_key_value_heads,
            self.cache.head_dim,
        )
        for index, block_id in enumerate(block_ids):
            output[index].copy_(source[block_id], non_blocking=False)
        return output.contiguous()

    def _copy_cpu_blocks_to_device(
        self,
        source_tensors: tuple[torch.Tensor, ...],
        destination_tensors: list[torch.Tensor],
        destination_blocks: tuple[int, ...],
    ) -> None:
        if len(source_tensors) != len(destination_tensors):
            raise PagedStageCorruptionError("CPU spill layer count does not match cache")
        for source, destination in zip(source_tensors, destination_tensors, strict=True):
            expected_shape = (
                len(destination_blocks),
                self.block_size,
                self.cache.num_key_value_heads,
                self.cache.head_dim,
            )
            if source.shape != expected_shape or source.dtype != destination.dtype:
                raise PagedStageCorruptionError(
                    "CPU spill tensor geometry or dtype does not match the device cache"
                )
            destination_view = destination.view(
                -1,
                self.block_size,
                self.cache.num_key_value_heads,
                self.cache.head_dim,
            )
            if len(source) != len(destination_blocks):
                raise PagedStageCorruptionError(
                    "CPU spill block count does not match restored block table"
                )
            for index, block_id in enumerate(destination_blocks):
                destination_view[block_id].copy_(source[index], non_blocking=False)

    @staticmethod
    def _spill_digest(
        cache_keys: tuple[int, ...],
        key_blocks: tuple[torch.Tensor, ...],
        value_blocks: tuple[torch.Tensor, ...],
    ) -> str:
        digest = hashlib.sha256()
        digest.update(
            json.dumps(
                {
                    "cacheKeys": cache_keys,
                    "keyShapes": [tuple(tensor.shape) for tensor in key_blocks],
                    "valueShapes": [tuple(tensor.shape) for tensor in value_blocks],
                    "keyDtypes": [str(tensor.dtype) for tensor in key_blocks],
                    "valueDtypes": [str(tensor.dtype) for tensor in value_blocks],
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        for tensor in (*key_blocks, *value_blocks):
            if tensor.device.type != "cpu" or not tensor.is_contiguous():
                raise PagedStageCorruptionError(
                    "CPU spill tensors must remain contiguous CPU storage"
                )
            digest.update(memoryview(tensor.view(torch.uint8).numpy()))
        return digest.hexdigest()

    def _require_idle(self) -> None:
        if self._pending is not None:
            raise RuntimeError("another paged forward is still pending")

    def _require_request(self, request_id: int) -> _RequestRecord:
        request_id = self._validate_request_id(request_id)
        record = self._requests.get(request_id)
        if record is None:
            raise ValueError(f"unknown paged request {request_id}")
        return record

    def _require_logical_request(
        self,
        request_id: int,
    ) -> _RequestRecord | _SpilledRequest:
        request_id = self._validate_request_id(request_id)
        record = self._requests.get(request_id)
        if record is not None:
            return record
        spilled = self._spilled_requests.get(request_id)
        if spilled is not None:
            return spilled
        raise ValueError(f"unknown paged request {request_id}")

    def _ensure_resident_request(self, request_id: int) -> _RequestRecord:
        request_id = self._validate_request_id(request_id)
        record = self._requests.get(request_id)
        if record is None:
            record = self._restore_spilled_request(request_id)
        self._touch(request_id)
        return record

    @staticmethod
    def _validate_request_id(request_id: int) -> int:
        if not isinstance(request_id, int) or isinstance(request_id, bool):
            raise TypeError("request_id must be an integer")
        if not 0 <= request_id <= MAX_REQUEST_ID:
            raise ValueError(f"request_id must be between 0 and {MAX_REQUEST_ID}")
        return request_id

    @staticmethod
    def _backend_id(request_id: int) -> str:
        return f"gdlp-paged:{request_id}"


class HFPagedStageRunner:
    """Opt-in selective stage runner backed by real paged COW KV.

    It fulfills the sequential ``StageRunnerContract`` surface and the optional
    ``StageKVPhysicalAccounting`` ABI. Packed tree verification remains a
    separate certification task and is not advertised in the manifest.
    """

    MAX_PHYSICAL_BATCH_SIZE = 1

    def __init__(
        self,
        spec: StageModelSpec,
        *,
        device: torch.device | str = "cpu",
        attention_backend: str = "eager",
        block_size: int = 16,
        num_blocks: int = 256,
        max_batch_tokens: int = 256,
        max_active_requests: int = 8,
        max_sequence_tokens: int = 2048,
        cpu_spill_bytes: int = 0,
    ) -> None:
        _require_supported_transformers_version()
        runtime_config = HFPagedStageRuntimeConfig(
            device=str(torch.device(device)),
            attention_backend=attention_backend,
            block_size=block_size,
            num_blocks=num_blocks,
            max_batch_tokens=max_batch_tokens,
            max_active_requests=max_active_requests,
            max_sequence_tokens=max_sequence_tokens,
            cpu_spill_bytes=cpu_spill_bytes,
        )
        resolved_device = torch.device(runtime_config.device)
        if resolved_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA paged runner requested but CUDA is unavailable")
        if resolved_device.type not in ("cpu", "cuda"):
            raise ValueError("paged stage runner supports only CPU or CUDA")

        torch.set_num_threads(spec.threads)
        model = _load_selective_stage_model(spec)
        model.to(resolved_device)
        adapter = getattr(model, "_gdlp_selective_stage_adapter", None)
        if not isinstance(adapter, SelectiveStageAdapter):
            raise TypeError("selective model loader did not return a certified adapter")
        for local_index, layer in enumerate(model.model.layers):
            layer.self_attn.layer_idx = local_index

        self.spec = spec
        self.model_adapter = adapter
        self.base = model.model
        self.head = model.lm_head if spec.last else None
        self.hidden_size = int(model.config.hidden_size)
        self.parameter_bytes = _unique_parameter_bytes(self.base, self.head)
        self.loader = "selective-safetensors-hf-paged-cow"
        self.device = resolved_device
        self.model_dtype = next(
            parameter.dtype
            for parameter in self.base.parameters()
            if parameter.dtype.is_floating_point
        )
        self.max_active_requests = max_active_requests
        self.maximum_context = max_sequence_tokens
        self.model_forward_calls = 0
        self.physical_batch_calls = 0
        self.physical_batch_items = 0
        self.max_observed_physical_batch_size = 1
        self._last_fork_report: StageKVForkReport | None = None
        self.runtime_config = runtime_config

        self.paged_cache = HFPagedStageCache.for_model(
            self.base,
            attention_backend=runtime_config.attention_backend,
            cache_vocab_size=None if self.head is not None else 1,
            block_size=runtime_config.block_size,
            num_blocks=runtime_config.num_blocks,
            max_batch_tokens=runtime_config.max_batch_tokens,
            max_active_requests=runtime_config.max_active_requests,
            max_sequence_tokens=runtime_config.max_sequence_tokens,
            cpu_spill_bytes=runtime_config.cpu_spill_bytes,
        )

        artifact = model_artifact_reference(
            getattr(model, "_gdlp_resolved_snapshot", spec.model_name),
            None if hasattr(model, "_gdlp_resolved_snapshot") else spec.revision,
            artifact_identity=spec.artifact_identity,
            canonical_source=spec.canonical_model_source,
            canonical_revision=spec.canonical_model_revision,
        )
        weight_dtypes = tuple(
            sorted(
                {
                    str(parameter.dtype).removeprefix("torch.")
                    for module in (self.base, self.head)
                    if module is not None
                    for parameter in module.parameters()
                }
            )
        )
        features = tuple(
            dict.fromkeys(
                (
                    "layer-range",
                    "rank-local-kv",
                    "rollback",
                    "selective-load",
                    "physical-kv-accounting",
                    "paged-kv-shared-prefix",
                    "fork-complete-block-zero-copy",
                    "fork-tail-copy",
                    "bounded-cpu-kv-spill",
                    "integrity-checked-atomic-kv-restore",
                    *adapter.semantic_features,
                )
            )
        )
        compute_apis = (
            ("torch", "cuda") if resolved_device.type == "cuda" else ("torch",)
        )
        self.executor_manifest = build_stage_executor_manifest(
            engine="python-transformers-paged",
            engine_version=(
                f"torch-{torch.__version__}/transformers-{transformers.__version__}"
            ),
            adapter=adapter.adapter_id,
            model_identity=artifact.identity,
            model_source=artifact.canonical_source,
            model_revision=artifact.canonical_revision,
            artifact_format="safetensors",
            layer_start=spec.layer_start,
            layer_end=spec.layer_end,
            total_layers=spec.total_layers,
            hidden_size=self.hidden_size,
            activation_dtype=str(self.model_dtype).removeprefix("torch."),
            activation_codecs=(
                "fp32",
                "fp16",
                "int8",
                "int8-grouped",
                "int8-hadamard",
            ),
            kv_format="transformers-paged-cow-v1",
            max_batch_size=1,
            max_context_tokens=max_sequence_tokens,
            device_kinds=(resolved_device.type,),
            compute_apis=compute_apis,
            weight_dtypes=weight_dtypes,
            features=features,
        )
        del model
        gc.collect()

    @classmethod
    def from_runtime_config(
        cls,
        spec: StageModelSpec,
        runtime: HFPagedStageRuntimeConfig,
    ) -> HFPagedStageRunner:
        if not isinstance(runtime, HFPagedStageRuntimeConfig):
            raise TypeError("paged runtime must be HFPagedStageRuntimeConfig")
        return cls(
            spec,
            device=runtime.device,
            attention_backend=runtime.attention_backend,
            block_size=runtime.block_size,
            num_blocks=runtime.num_blocks,
            max_batch_tokens=runtime.max_batch_tokens,
            max_active_requests=runtime.max_active_requests,
            max_sequence_tokens=runtime.max_sequence_tokens,
            cpu_spill_bytes=runtime.cpu_spill_bytes,
        )

    def execution_snapshot(self) -> dict[str, Any]:
        """Publish observed bounded pool state and the honest spill contract."""

        runtime = self.runtime_config.to_document()
        metrics = self.paged_cache.metrics()
        return {
            "backend": "hf-paged-cow",
            "configurationId": self.runtime_config.configuration_id,
            "device": str(self.device),
            "attentionBackend": self.runtime_config.attention_backend,
            "kvPool": {
                "blockSize": self.paged_cache.block_size,
                "numBlocks": self.paged_cache.num_blocks,
                "reservedBytes": self.paged_cache.pool_reserved_bytes,
                "freeBlocks": self.paged_cache.cache.get_num_free_blocks(),
                "residentBytes": metrics.unique_physical_bytes,
            },
            "limits": {
                "maxBatchTokens": self.runtime_config.max_batch_tokens,
                "maxActiveRequests": self.runtime_config.max_active_requests,
                "maxSequenceTokens": self.runtime_config.max_sequence_tokens,
            },
            "cpuSpill": runtime["cpuSpill"],
            "cpuSpillState": {
                "usedBytes": metrics.spilled_bytes,
                "storedPayloadBytes": metrics.spilled_bytes,
                "requests": metrics.spilled_requests,
                "spillCount": metrics.spill_count,
                "restoreCount": metrics.restore_count,
                "spillFailures": metrics.spill_failures,
                "restoreFailures": metrics.restore_failures,
                "spillBytesTransferred": metrics.spill_bytes_transferred,
                "restoreBytesTransferred": metrics.restore_bytes_transferred,
                "spillTimeNs": metrics.spill_time_ns,
                "restoreTimeNs": metrics.restore_time_ns,
                "restoreWorkspaceBytes": metrics.restore_workspace_bytes,
                "peakRestoreWorkspaceBytes": metrics.peak_restore_workspace_bytes,
                "currentCpuPayloadBytes": metrics.current_cpu_payload_bytes,
                "peakCpuPayloadBytes": metrics.peak_cpu_payload_bytes,
                "restoreWorkspaceUpperBoundBytes": (
                    self.runtime_config.cpu_spill_bytes
                ),
                "maxCpuPayloadUpperBoundBytes": (
                    2 * self.runtime_config.cpu_spill_bytes
                ),
                "storage": (
                    "pinned-cpu" if self.device.type == "cuda" else "cpu"
                ),
            },
        }

    def begin(self, request_id: int) -> None:
        self.paged_cache.begin(request_id)

    def end(self, request_id: int) -> None:
        self.paged_cache.end(request_id)

    def spill(self, request_id: int) -> None:
        self.paged_cache.spill(request_id)

    def restore(self, request_id: int) -> None:
        self.paged_cache.restore(request_id)

    def close(self) -> None:
        self.paged_cache.close()

    def truncate(self, request_id: int, token_count: int) -> None:
        self.paged_cache.truncate(request_id, token_count)

    def sequence_length(self, request_id: int) -> int:
        return self.paged_cache.sequence_length(request_id)

    def request_cache_bytes(self, request_id: int) -> int:
        """Return logical KV bytes for compatibility with the legacy ABI."""

        return self.paged_cache.logical_cache_bytes(request_id)

    def project_request_cache_bytes(
        self,
        request_id: int,
        additional_tokens: int,
    ) -> int:
        if not isinstance(additional_tokens, int) or isinstance(
            additional_tokens, bool
        ):
            raise TypeError("additional_tokens must be an integer")
        if additional_tokens < 0:
            raise ValueError("additional_tokens cannot be negative")
        current = self.sequence_length(request_id)
        projected = current + additional_tokens
        if projected > self.maximum_context:
            raise ValueError(
                f"projected request would reach {projected} tokens, "
                f"limit is {self.maximum_context}"
            )
        return projected * self.paged_cache.bytes_per_token

    def unique_physical_cache_bytes(self, request_ids: Sequence[int]) -> int:
        return self.paged_cache.unique_physical_bytes(request_ids)

    def project_incremental_physical_cache_bytes(
        self,
        parent_request_id: int,
        *,
        new_leaf_count: int,
        delta_tokens: int,
    ) -> int:
        return self.paged_cache.project_incremental_physical_bytes(
            parent_request_id,
            new_leaf_count=new_leaf_count,
            delta_tokens=delta_tokens,
        )

    def project_tree_incremental_physical_cache_bytes(
        self,
        parent_request_id: int,
        *,
        delta_tokens_by_leaf: Sequence[int],
    ) -> int:
        return self.paged_cache.project_tree_incremental_physical_bytes(
            parent_request_id,
            delta_tokens_by_leaf=delta_tokens_by_leaf,
        )

    def project_request_incremental_physical_cache_bytes(
        self,
        request_id: int,
        additional_tokens: int,
    ) -> int:
        return self.paged_cache.project_request_incremental_physical_bytes(
            request_id,
            additional_tokens,
        )

    def available_physical_cache_bytes(self) -> int:
        """Return free plus conservatively spill-reclaimable KV payload bytes."""

        return self.paged_cache.available_physical_bytes()

    def last_fork_report(self) -> StageKVForkReport | None:
        return self._last_fork_report

    def fork_request(
        self,
        child_request_id: int,
        parent_request_id: int,
        *,
        max_cache_bytes: int,
    ) -> int:
        if not isinstance(max_cache_bytes, int) or isinstance(max_cache_bytes, bool):
            raise TypeError("max_cache_bytes must be an integer")
        if max_cache_bytes < 0:
            raise ValueError("max_cache_bytes cannot be negative")
        projected = self.project_incremental_physical_cache_bytes(
            parent_request_id,
            new_leaf_count=1,
            delta_tokens=0,
        )
        if projected > max_cache_bytes:
            raise ValueError(
                "fork cache exceeds the physical preflight byte budget: "
                f"{projected} > {max_cache_bytes}"
            )
        metrics = self.paged_cache.fork(child_request_id, parent_request_id)
        self._last_fork_report = StageKVForkReport(
            logical_bytes=metrics.logical_bytes,
            unique_physical_bytes=metrics.unique_physical_bytes,
            copied_bytes=metrics.copied_bytes,
            newly_reserved_bytes=metrics.newly_reserved_bytes,
            peak_workspace_bytes=metrics.peak_workspace,
        )
        return metrics.copied_bytes

    def promote_request(self, parent_request_id: int, child_request_id: int) -> None:
        self.paged_cache.promote(parent_request_id, child_request_id)

    @torch.inference_mode()
    def forward_ids(self, request_id: int, input_ids: torch.Tensor) -> torch.Tensor:
        if not self.spec.first:
            raise RuntimeError("only the first stage accepts token IDs")
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
            raise ValueError("input_ids must have shape [1, tokens] with at least one token")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("input_ids must contain integer token IDs")
        local_ids = input_ids.to(device=self.device)
        result = self.paged_cache.forward_model(
            self.base,
            request_id,
            input_ids=local_ids,
        )
        self.model_forward_calls += 1
        return result.output.last_hidden_state

    @torch.inference_mode()
    def forward_ids_with_tokens(
        self,
        request_id: int,
        input_ids: torch.Tensor,
        *,
        token_mode: str = "last",
    ) -> tuple[torch.Tensor, int | tuple[int, ...]]:
        if not self.spec.first or not self.spec.last or self.head is None:
            raise RuntimeError("token projection requires one complete local stage")
        if token_mode not in ("last", "all"):
            raise ValueError("token_mode must be last or all")
        hidden = self.forward_ids(request_id, input_ids)
        selected = hidden if token_mode == "all" else hidden[:, -1:, :]
        predicted = torch.argmax(self.head(selected), dim=-1).reshape(-1)
        tokens = predicted.detach().cpu().tolist()
        if token_mode == "all":
            return hidden, tuple(int(token) for token in tokens)
        return hidden, int(tokens[-1])

    @torch.inference_mode()
    def forward_hidden(
        self,
        request_id: int,
        hidden: torch.Tensor,
        *,
        token_mode: str = "last",
    ) -> tuple[torch.Tensor, int | tuple[int, ...] | None]:
        if (
            hidden.ndim != 3
            or hidden.shape[0] != 1
            or hidden.shape[1] < 1
            or hidden.shape[2] != self.hidden_size
        ):
            raise ValueError(
                f"hidden state must have shape [1, tokens, {self.hidden_size}]"
            )
        if not hidden.is_floating_point():
            raise TypeError("hidden state must be floating point")
        if token_mode not in ("none", "last", "all"):
            raise ValueError("token_mode must be none, last or all")
        local_hidden = hidden.to(device=self.device, dtype=self.model_dtype)
        result = self.paged_cache.forward_model(
            self.base,
            request_id,
            inputs_embeds=local_hidden,
        )
        self.model_forward_calls += 1
        output_hidden = result.output.last_hidden_state
        if self.head is None or token_mode == "none":
            return output_hidden, None
        selected = (
            output_hidden if token_mode == "all" else output_hidden[:, -1:, :]
        )
        predicted = torch.argmax(self.head(selected), dim=-1).reshape(-1)
        tokens = predicted.detach().cpu().tolist()
        if token_mode == "all":
            return output_hidden, tuple(int(token) for token in tokens)
        return output_hidden, int(tokens[-1])


__all__ = [
    "HFPagedStageRuntimeConfig",
    "HFPagedStageCache",
    "HFPagedStageRunner",
    "PAGED_STAGE_RUNTIME_SCHEMA",
    "PagedCacheMetrics",
    "PagedCacheSnapshot",
    "PagedForwardResult",
    "PagedRequestSnapshot",
    "PagedStageCorruptionError",
    "add_paged_kv_arguments",
    "paged_kv_config_from_args",
]
