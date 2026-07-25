"""Native local sibling-model drafter for exact speculative decoding.

The drafter never serves tokens directly.  It proposes deterministic greedy
token ids and the distributed target model remains the sole authority that
verifies and emits them.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import ctypes
import hashlib
import json
from numbers import Integral
import os
import re
import threading
from time import perf_counter_ns
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .model import model_artifact_reference
from .speculation import MAX_DRAFT_TOKENS


LOCAL_DRAFT_MODEL_SCHEMA = "mycellios-local-draft-model/1"
SUPPORTED_DRAFT_DTYPES = frozenset(("auto", "float32", "float16", "bfloat16"))
_SHA256_IDENTITY = re.compile(r"sha256:[0-9a-f]{64}")


@dataclass(frozen=True)
class LocalDraftModelRuntimeConfig:
    """Sealed coordinates and placement for one local sibling drafter."""

    source: str
    artifact_identity: str
    max_draft_tokens: int
    revision: str | None = None
    canonical_source: str | None = None
    canonical_revision: str | None = None
    device: str = "auto"
    dtype: str = "auto"
    parameter_bytes: int = 0
    memory_reservation_bytes: int = 0
    schema: str = LOCAL_DRAFT_MODEL_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != LOCAL_DRAFT_MODEL_SCHEMA:
            raise ValueError(
                f"local draft model schema must be {LOCAL_DRAFT_MODEL_SCHEMA}"
            )
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("local draft model source cannot be blank")
        if self.source != self.source.strip():
            raise ValueError("local draft model source must be normalized")
        if self.revision is not None and (
            not isinstance(self.revision, str)
            or not self.revision.strip()
            or self.revision != self.revision.strip()
        ):
            raise ValueError("local draft model revision must be normalized or None")
        if not isinstance(self.artifact_identity, str) or not _SHA256_IDENTITY.fullmatch(
            self.artifact_identity
        ):
            raise ValueError("local draft model artifact identity must be sha256")
        for name, value in (
            ("canonical_source", self.canonical_source),
            ("canonical_revision", self.canonical_revision),
        ):
            if value is not None and (
                not isinstance(value, str)
                or not value.strip()
                or value != value.strip()
            ):
                raise ValueError(f"local draft model {name} must be normalized or None")
        if (
            not isinstance(self.max_draft_tokens, Integral)
            or isinstance(self.max_draft_tokens, bool)
            or not 1 <= int(self.max_draft_tokens) <= MAX_DRAFT_TOKENS
        ):
            raise ValueError(
                f"local draft max tokens must be between 1 and {MAX_DRAFT_TOKENS}"
            )
        _canonical_device(self.device, allow_auto=True)
        if self.dtype not in SUPPORTED_DRAFT_DTYPES:
            raise ValueError(
                f"local draft dtype must be one of {sorted(SUPPORTED_DRAFT_DTYPES)}"
            )
        if self.device == "cpu" and self.dtype == "float16":
            raise ValueError("local draft float16 is not supported on CPU")
        for name, value in (
            ("parameter_bytes", self.parameter_bytes),
            ("memory_reservation_bytes", self.memory_reservation_bytes),
        ):
            if (
                not isinstance(value, Integral)
                or isinstance(value, bool)
                or int(value) <= 0
            ):
                raise ValueError(f"local draft {name} must be a positive integer")
        if self.memory_reservation_bytes < self.parameter_bytes:
            raise ValueError(
                "local draft memory reservation cannot be smaller than parameter bytes"
            )

    def to_document(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "source": self.source,
            "revision": self.revision,
            "artifactIdentity": self.artifact_identity,
            "canonicalSource": self.canonical_source,
            "canonicalRevision": self.canonical_revision,
            "device": self.device,
            "dtype": self.dtype,
            "parameterBytes": int(self.parameter_bytes),
            "memoryReservationBytes": int(self.memory_reservation_bytes),
            "maxDraftTokens": int(self.max_draft_tokens),
            "verificationAuthority": "distributed-target-model",
            "tokenizerCompatibility": "exact-id-to-token-vocabulary",
        }

    @property
    def configuration_id(self) -> str:
        encoded = json.dumps(
            self.to_document(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def add_local_draft_model_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the all-or-nothing sibling-drafter contract to the root CLI."""

    parser.add_argument("--draft-model-source")
    parser.add_argument("--draft-model-revision")
    parser.add_argument("--draft-model-artifact-identity")
    parser.add_argument("--draft-model-canonical-source")
    parser.add_argument("--draft-model-canonical-revision")
    parser.add_argument("--draft-model-parameter-bytes", type=int)
    parser.add_argument("--draft-model-memory-reservation-bytes", type=int)
    parser.add_argument("--draft-model-device")
    parser.add_argument(
        "--draft-model-dtype",
        choices=tuple(sorted(SUPPORTED_DRAFT_DTYPES)),
    )


def local_draft_model_config_from_args(
    args: argparse.Namespace,
) -> LocalDraftModelRuntimeConfig | None:
    """Return a sealed config only when ``draft-model`` is selected."""

    selected = getattr(args, "speculation", None) == "draft-model"
    fields = {
        "source": getattr(args, "draft_model_source", None),
        "revision": getattr(args, "draft_model_revision", None),
        "artifact_identity": getattr(args, "draft_model_artifact_identity", None),
        "canonical_source": getattr(args, "draft_model_canonical_source", None),
        "canonical_revision": getattr(args, "draft_model_canonical_revision", None),
        "parameter_bytes": getattr(args, "draft_model_parameter_bytes", None),
        "memory_reservation_bytes": getattr(
            args,
            "draft_model_memory_reservation_bytes",
            None,
        ),
        "device": getattr(args, "draft_model_device", None),
        "dtype": getattr(args, "draft_model_dtype", None),
    }
    supplied = any(value is not None for value in fields.values())
    if not selected:
        if supplied:
            raise ValueError(
                "local draft model coordinates require --speculation draft-model"
            )
        return None
    if (
        fields["source"] is None
        or fields["artifact_identity"] is None
        or fields["parameter_bytes"] is None
        or fields["memory_reservation_bytes"] is None
    ):
        raise ValueError(
            "draft-model requires source, sha256 artifact identity, parameter bytes "
            "and a memory reservation"
        )
    return LocalDraftModelRuntimeConfig(
        source=fields["source"],
        revision=fields["revision"],
        artifact_identity=fields["artifact_identity"],
        canonical_source=fields["canonical_source"],
        canonical_revision=fields["canonical_revision"],
        device=fields["device"] or "auto",
        dtype=fields["dtype"] or "auto",
        parameter_bytes=fields["parameter_bytes"],
        memory_reservation_bytes=fields["memory_reservation_bytes"],
        max_draft_tokens=getattr(args, "speculative_max_draft_tokens", 0),
    )


@dataclass(frozen=True)
class LocalDraftModelStats:
    draft_calls: int
    draft_failures: int
    drafted_tokens: int
    draft_time_ns: int
    truncated_context_calls: int
    bypassed_calls: int
    circuit_open: bool
    circuit_reason: str | None
    cache_hits: int
    cache_misses: int
    cache_rejections: int
    cached_kv_bytes: int


@dataclass(frozen=True)
class _DraftPrefixCache:
    tokens: tuple[int, ...]
    past_key_values: Any
    next_logits: torch.Tensor
    tensor_bytes: int


class LocalDraftModelProvider:
    """Greedy local drafter whose every token is verified by the target model."""

    strategy = "draft-model"
    # The adaptive adapter may decide/probe before paying for model execution.
    defer_until_selected = True

    def __init__(
        self,
        config: LocalDraftModelRuntimeConfig,
        *,
        target_tokenizer: Any,
        draft_tokenizer: Any,
        model: Any,
        actual_artifact_identity: str,
        resolved_device: torch.device,
        max_cached_requests: int = 1,
    ) -> None:
        if actual_artifact_identity != config.artifact_identity:
            raise ValueError(
                "local draft model artifact identity does not match the sealed contract"
            )
        target_vocab = _tokenizer_vocabulary(target_tokenizer)
        draft_vocab = _tokenizer_vocabulary(draft_tokenizer)
        if target_vocab != draft_vocab:
            raise ValueError(
                "local draft model tokenizer vocabulary is not identical to the target"
            )
        if _special_token_contract(target_tokenizer) != _special_token_contract(
            draft_tokenizer
        ):
            raise ValueError(
                "local draft model special-token ids are not identical to the target"
            )
        vocabulary_span = max(target_vocab.values(), default=-1) + 1
        valid_token_ids = frozenset(target_vocab.values())
        model_vocab = _model_vocabulary_size(model)
        if model_vocab < vocabulary_span:
            raise ValueError(
                "local draft model logits cannot represent the target vocabulary"
            )

        self.config = config
        self.model = model.eval()
        self.device = resolved_device
        self.vocabulary_fingerprint = _vocabulary_fingerprint(target_vocab)
        self.vocabulary_span = vocabulary_span
        self.valid_token_ids = valid_token_ids
        self._dense_vocabulary = valid_token_ids == frozenset(range(vocabulary_span))
        self._valid_token_mask: torch.Tensor | None = None
        self.model_vocab_size = model_vocab
        self.eos_token_ids = frozenset(_token_id_tuple(draft_tokenizer.eos_token_id))
        self.max_context_tokens = _model_context_limit(model)
        if self.max_context_tokens <= 1:
            raise ValueError(
                "local draft model context must leave room for at least one draft token"
            )
        self.max_draft_tokens = min(
            int(config.max_draft_tokens),
            self.max_context_tokens - 1,
        )
        self.parameter_bytes = _model_parameter_bytes(model)
        if self.parameter_bytes != int(config.parameter_bytes):
            raise ValueError(
                "local draft model parameter bytes do not match the sealed contract"
            )
        if self.parameter_bytes > int(config.memory_reservation_bytes):
            raise ValueError(
                "local draft model exceeds its sealed memory reservation"
            )
        self.resolved_dtype = _model_parameter_dtype(model)
        if (
            not isinstance(max_cached_requests, Integral)
            or isinstance(max_cached_requests, bool)
            or not 1 <= int(max_cached_requests) <= 1024
        ):
            raise ValueError("local draft cached request limit must be between 1 and 1024")
        self.max_cached_requests = int(max_cached_requests)
        self._lock = threading.Lock()
        self._draft_calls = 0
        self._draft_failures = 0
        self._drafted_tokens = 0
        self._draft_time_ns = 0
        self._truncated_context_calls = 0
        self._bypassed_calls = 0
        self._circuit_open = False
        self._circuit_reason: str | None = None
        self._draft_cache: OrderedDict[str | int, _DraftPrefixCache] = OrderedDict()
        self._draft_cache_bytes = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_rejections = 0

    def draft(
        self,
        token_history: Sequence[int],
        max_tokens: int | None = None,
    ) -> tuple[int, ...]:
        return self._draft_for_key(None, token_history, max_tokens)

    def draft_for_request(
        self,
        request_id: str | int,
        token_history: Sequence[int],
        max_tokens: int | None = None,
    ) -> tuple[int, ...]:
        if (
            not isinstance(request_id, (str, Integral))
            or isinstance(request_id, bool)
            or (isinstance(request_id, str) and not request_id)
            or (isinstance(request_id, Integral) and int(request_id) < 0)
        ):
            raise ValueError("local draft request id must be non-empty or non-negative")
        key: str | int = (
            int(request_id) if isinstance(request_id, Integral) else request_id
        )
        return self._draft_for_key(key, token_history, max_tokens)

    def _draft_for_key(
        self,
        request_id: str | int | None,
        token_history: Sequence[int],
        max_tokens: int | None,
    ) -> tuple[int, ...]:
        history = _token_history(token_history)
        if not history:
            raise ValueError("local draft model requires non-empty token history")
        if any(token not in self.valid_token_ids for token in history):
            raise ValueError("token history contains an id outside the shared vocabulary")
        limit = self.max_draft_tokens if max_tokens is None else _draft_limit(max_tokens)
        limit = min(limit, self.max_draft_tokens)
        if limit == 0:
            return ()

        started_ns = perf_counter_ns()
        with self._lock:
            if self._circuit_open:
                self._bypassed_calls += 1
                return ()
            self._draft_calls += 1
            try:
                available_context = self.max_context_tokens - limit
                context = history[-available_context:]
                if len(context) != len(history):
                    self._truncated_context_calls += 1
                result = self._greedy_draft(context, limit, request_id)
                self._drafted_tokens += len(result)
                return result
            except Exception as error:
                self._draft_failures += 1
                self._circuit_open = True
                self._circuit_reason = type(error).__name__
                self._draft_cache.clear()
                self._draft_cache_bytes = 0
                _release_device_cache(self.device)
                return ()
            finally:
                self._draft_time_ns += perf_counter_ns() - started_ns

    @torch.inference_mode()
    def _greedy_draft(
        self,
        context: tuple[int, ...],
        limit: int,
        request_id: str | int | None,
    ) -> tuple[int, ...]:
        cached = self._matching_cache(request_id, context)
        if cached is None:
            full_ids = torch.tensor([context], dtype=torch.long, device=self.device)
            output = self.model(
                input_ids=full_ids,
                attention_mask=torch.ones_like(full_ids),
                use_cache=True,
                return_dict=True,
            )
            state_tokens = context
            past_key_values = _cacheable_past_key_values(
                getattr(output, "past_key_values", None)
            )
            next_logits = _last_model_logits(output)
        elif len(cached.tokens) == len(context):
            state_tokens = cached.tokens
            past_key_values = cached.past_key_values
            next_logits = cached.next_logits
        else:
            suffix = context[len(cached.tokens) :]
            suffix_ids = torch.tensor([suffix], dtype=torch.long, device=self.device)
            output = self.model(
                input_ids=suffix_ids,
                attention_mask=torch.ones(
                    (1, len(context)),
                    dtype=torch.long,
                    device=self.device,
                ),
                past_key_values=cached.past_key_values,
                use_cache=True,
                return_dict=True,
            )
            state_tokens = context
            past_key_values = _cacheable_past_key_values(
                getattr(output, "past_key_values", None)
            )
            next_logits = _last_model_logits(output)

        drafted: list[int] = []
        for index in range(limit):
            if int(next_logits.shape[-1]) < self.vocabulary_span:
                raise RuntimeError("local draft model logits vocabulary changed at runtime")
            candidate_logits = next_logits[: self.vocabulary_span]
            if not self._dense_vocabulary:
                if (
                    self._valid_token_mask is None
                    or self._valid_token_mask.device != candidate_logits.device
                ):
                    mask = torch.zeros(
                        self.vocabulary_span,
                        dtype=torch.bool,
                        device=candidate_logits.device,
                    )
                    mask[list(self.valid_token_ids)] = True
                    self._valid_token_mask = mask
                candidate_logits = candidate_logits.masked_fill(
                    ~self._valid_token_mask,
                    float("-inf"),
                )
            token = int(torch.argmax(candidate_logits).item())
            drafted.append(token)
            if token in self.eos_token_ids or index + 1 == limit:
                break

            next_id = torch.tensor([[token]], dtype=torch.long, device=self.device)
            if past_key_values is None:
                full_ids = torch.tensor(
                    [(*context, *drafted)],
                    dtype=torch.long,
                    device=self.device,
                )
                attention_mask = torch.ones_like(full_ids)
                output = self.model(
                    input_ids=full_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    return_dict=True,
                )
            else:
                attention_mask = torch.ones(
                    (1, len(context) + len(drafted)),
                    dtype=torch.long,
                    device=self.device,
                )
                output = self.model(
                    input_ids=next_id,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
            state_tokens = (*context, *drafted)
            past_key_values = _cacheable_past_key_values(
                getattr(output, "past_key_values", None)
            )
            next_logits = _last_model_logits(output)
        self._remember_cache(
            request_id,
            state_tokens,
            past_key_values,
            next_logits,
        )
        return tuple(drafted)

    def _matching_cache(
        self,
        request_id: str | int | None,
        context: tuple[int, ...],
    ) -> _DraftPrefixCache | None:
        if request_id is None:
            return None
        cached = self._draft_cache.get(request_id)
        if (
            cached is None
            or len(cached.tokens) > len(context)
            or context[: len(cached.tokens)] != cached.tokens
        ):
            self._cache_misses += 1
            return None
        self._draft_cache.move_to_end(request_id)
        self._cache_hits += 1
        return cached

    def _remember_cache(
        self,
        request_id: str | int | None,
        tokens: tuple[int, ...],
        past_key_values: Any,
        next_logits: torch.Tensor,
    ) -> None:
        if request_id is None or past_key_values is None:
            return
        detached_logits = next_logits.detach()
        tensor_bytes = _tensor_tree_bytes(past_key_values) + _tensor_tree_bytes(
            detached_logits
        )
        existing = self._draft_cache.pop(request_id, None)
        if existing is not None:
            self._draft_cache_bytes -= existing.tensor_bytes
        if self.parameter_bytes + tensor_bytes > self.config.memory_reservation_bytes:
            self._cache_rejections += 1
            return
        while (
            self._draft_cache
            and (
                len(self._draft_cache) >= self.max_cached_requests
                or self.parameter_bytes
                + self._draft_cache_bytes
                + tensor_bytes
                > self.config.memory_reservation_bytes
            )
        ):
            _key, evicted = self._draft_cache.popitem(last=False)
            self._draft_cache_bytes -= evicted.tensor_bytes
        if (
            self.parameter_bytes
            + self._draft_cache_bytes
            + tensor_bytes
            > self.config.memory_reservation_bytes
        ):
            self._cache_rejections += 1
            return
        self._draft_cache[request_id] = _DraftPrefixCache(
            tokens=tokens,
            past_key_values=past_key_values,
            next_logits=detached_logits,
            tensor_bytes=tensor_bytes,
        )
        self._draft_cache_bytes += tensor_bytes

    def release_request(self, request_id: str | int) -> None:
        with self._lock:
            cached = self._draft_cache.pop(request_id, None)
            if cached is not None:
                self._draft_cache_bytes -= cached.tensor_bytes

    def stats(self) -> LocalDraftModelStats:
        with self._lock:
            return LocalDraftModelStats(
                draft_calls=self._draft_calls,
                draft_failures=self._draft_failures,
                drafted_tokens=self._drafted_tokens,
                draft_time_ns=self._draft_time_ns,
                truncated_context_calls=self._truncated_context_calls,
                bypassed_calls=self._bypassed_calls,
                circuit_open=self._circuit_open,
                circuit_reason=self._circuit_reason,
                cache_hits=self._cache_hits,
                cache_misses=self._cache_misses,
                cache_rejections=self._cache_rejections,
                cached_kv_bytes=self._draft_cache_bytes,
            )

    def execution_snapshot(self) -> dict[str, Any]:
        stats = self.stats()
        return {
            "schema": self.config.schema,
            "strategy": self.strategy,
            "configurationId": self.config.configuration_id,
            "artifactIdentity": self.config.artifact_identity,
            "canonicalSource": self.config.canonical_source,
            "canonicalRevision": self.config.canonical_revision,
            "device": str(self.device),
            "configuredDtype": self.config.dtype,
            "resolvedDtype": self.resolved_dtype,
            "configuredMaxDraftTokens": int(self.config.max_draft_tokens),
            "maxDraftTokens": self.max_draft_tokens,
            "maxContextTokens": self.max_context_tokens,
            "modelVocabularySize": self.model_vocab_size,
            "targetVocabularySpan": self.vocabulary_span,
            "vocabularyFingerprint": self.vocabulary_fingerprint,
            "parameterBytes": self.parameter_bytes,
            "memoryReservationBytes": int(self.config.memory_reservation_bytes),
            "verificationAuthority": "distributed-target-model",
            "draftCalls": stats.draft_calls,
            "draftFailures": stats.draft_failures,
            "draftedTokens": stats.drafted_tokens,
            "draftTimeNs": stats.draft_time_ns,
            "truncatedContextCalls": stats.truncated_context_calls,
            "bypassedCalls": stats.bypassed_calls,
            "circuitOpen": stats.circuit_open,
            "circuitReason": stats.circuit_reason,
            "cacheHits": stats.cache_hits,
            "cacheMisses": stats.cache_misses,
            "cacheRejections": stats.cache_rejections,
            "cachedKvBytes": stats.cached_kv_bytes,
            "maxCachedRequests": self.max_cached_requests,
        }


def load_local_draft_model_provider(
    config: LocalDraftModelRuntimeConfig,
    *,
    target_tokenizer: Any,
    max_cached_requests: int = 1,
) -> LocalDraftModelProvider:
    """Resolve, authenticate and load one local sibling model."""

    reference = model_artifact_reference(config.source, config.revision)
    if reference.identity != config.artifact_identity:
        raise ValueError(
            "resolved local draft model does not match its sha256 artifact identity"
        )
    if (
        config.canonical_source is not None
        and reference.canonical_source != config.canonical_source
    ):
        raise ValueError("resolved local draft model canonical source mismatch")
    if (
        config.canonical_revision is not None
        and reference.canonical_revision != config.canonical_revision
    ):
        raise ValueError("resolved local draft model canonical revision mismatch")

    device = _resolve_device(config.device)
    dtype = _resolve_dtype(config.dtype, device)
    _ensure_memory_reservation(config.memory_reservation_bytes, device)
    snapshot = config.source
    if config.revision is not None:
        # model_artifact_reference has already resolved and verified the exact
        # immutable snapshot. Resolve once more through the same cache only to
        # obtain its local path for offline loading.
        from .model import resolve_model_snapshot

        snapshot = resolve_model_snapshot(config.source, config.revision)
    else:
        from .model import resolve_model_snapshot

        snapshot = resolve_model_snapshot(config.source)
    draft_tokenizer = AutoTokenizer.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
        dtype=dtype,
    ).to(device)
    return LocalDraftModelProvider(
        config,
        target_tokenizer=target_tokenizer,
        draft_tokenizer=draft_tokenizer,
        model=model,
        actual_artifact_identity=reference.identity,
        resolved_device=device,
        max_cached_requests=max_cached_requests,
    )


def _canonical_device(value: object, *, allow_auto: bool) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError("local draft device must be normalized")
    if allow_auto and value == "auto":
        return value
    try:
        device = torch.device(value)
    except (TypeError, RuntimeError) as error:
        raise ValueError("local draft device is invalid") from error
    if device.type not in ("cpu", "cuda", "mps", "xpu"):
        raise ValueError("local draft device must be auto, cpu, cuda, mps or xpu")
    if value != str(device):
        raise ValueError("local draft device must use canonical torch spelling")
    return value


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available():
            return torch.device("mps")
        xpu = getattr(torch, "xpu", None)
        if xpu is not None and xpu.is_available():
            return torch.device("xpu")
        return torch.device("cpu")
    device = torch.device(_canonical_device(value, allow_auto=False))
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("local draft CUDA device is unavailable")
        index = torch.cuda.current_device() if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise RuntimeError("local draft CUDA device index is unavailable")
    elif device.type == "mps" and (
        not bool(getattr(torch.backends, "mps", None))
        or not torch.backends.mps.is_available()
    ):
        raise RuntimeError("local draft MPS device is unavailable")
    elif device.type == "xpu":
        xpu = getattr(torch, "xpu", None)
        if xpu is None or not xpu.is_available():
            raise RuntimeError("local draft XPU device is unavailable")
        count = getattr(xpu, "device_count", lambda: 0)()
        index = getattr(xpu, "current_device", lambda: 0)() if device.index is None else device.index
        if index >= count:
            raise RuntimeError("local draft XPU device index is unavailable")
    return device


def _resolve_dtype(value: str, device: torch.device) -> torch.dtype | str:
    if value == "auto":
        return "auto"
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[value]
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("local draft float16 is not supported on CPU")
    return dtype


def _tokenizer_vocabulary(tokenizer: Any) -> dict[str, int]:
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if not callable(get_vocab):
        raise ValueError("tokenizer does not expose get_vocab")
    raw = get_vocab()
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("tokenizer vocabulary must be a non-empty mapping")
    result: dict[str, int] = {}
    seen_ids: set[int] = set()
    for token, token_id in raw.items():
        if (
            not isinstance(token, str)
            or not isinstance(token_id, Integral)
            or isinstance(token_id, bool)
            or int(token_id) < 0
        ):
            raise ValueError("tokenizer vocabulary contains an invalid entry")
        normalized = int(token_id)
        if normalized in seen_ids:
            raise ValueError("tokenizer vocabulary reuses a token id")
        seen_ids.add(normalized)
        result[token] = normalized
    return result


def _vocabulary_fingerprint(vocabulary: Mapping[str, int]) -> str:
    digest = hashlib.sha256(b"mycellios-token-vocabulary-v1\0")
    for token, token_id in sorted(vocabulary.items(), key=lambda item: (item[1], item[0])):
        encoded = token.encode("utf-8")
        digest.update(int(token_id).to_bytes(8, "big"))
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return "sha256:" + digest.hexdigest()


def _special_token_contract(tokenizer: Any) -> tuple[tuple[str, tuple[int, ...]], ...]:
    return tuple(
        (
            name,
            _token_id_tuple(getattr(tokenizer, name)),
        )
        for name in (
            "bos_token_id",
            "eos_token_id",
            "pad_token_id",
            "unk_token_id",
            "all_special_ids",
        )
    )


def _token_id_tuple(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    values = value if isinstance(value, (list, tuple, set, frozenset)) else (value,)
    result = []
    for token_id in values:
        if (
            not isinstance(token_id, Integral)
            or isinstance(token_id, bool)
            or int(token_id) < 0
        ):
            raise ValueError("special token id must be a non-negative integer")
        result.append(int(token_id))
    return tuple(sorted(set(result)))


def _model_vocabulary_size(model: Any) -> int:
    output = getattr(model, "get_output_embeddings", lambda: None)()
    for value in (
        getattr(output, "out_features", None),
        getattr(output, "num_embeddings", None),
        getattr(getattr(model, "config", None), "vocab_size", None),
    ):
        if isinstance(value, Integral) and not isinstance(value, bool) and int(value) > 0:
            return int(value)
    raise ValueError("local draft model does not expose a positive vocabulary size")


def _model_context_limit(model: Any) -> int:
    config = getattr(model, "config", None)
    for name in (
        "max_position_embeddings",
        "max_sequence_length",
        "n_positions",
        "seq_length",
    ):
        value = getattr(config, name, None)
        if isinstance(value, Integral) and not isinstance(value, bool) and int(value) > 1:
            return int(value)
    return 1_048_576


def _model_parameter_bytes(model: Any) -> int:
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return 0
    unique: set[int] = set()
    total = 0
    for parameter in parameters():
        identity = id(parameter)
        if identity in unique:
            continue
        unique.add(identity)
        total += int(parameter.numel()) * int(parameter.element_size())
    return total


def _model_parameter_dtype(model: Any) -> str:
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return "unknown"
    dtypes = {
        str(parameter.dtype).removeprefix("torch.")
        for parameter in parameters()
    }
    if not dtypes:
        return "none"
    if len(dtypes) == 1:
        return next(iter(dtypes))
    return "mixed:" + ",".join(sorted(dtypes))


def _last_model_logits(output: Any) -> torch.Tensor:
    logits = getattr(output, "logits", None)
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise RuntimeError("local draft model did not return rank-3 logits")
    return logits[0, -1]


def _cacheable_past_key_values(value: Any) -> Any | None:
    if value is None:
        return None
    if not isinstance(value, (tuple, list)):
        to_legacy = getattr(value, "to_legacy_cache", None)
        if not callable(to_legacy):
            return None
        value = to_legacy()
    normalized = _tensor_tree_tuple(value)
    return normalized if normalized else None


def _tensor_tree_tuple(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        return tuple(_tensor_tree_tuple(item) for item in value)
    if value is None:
        return None
    raise TypeError("local draft model returned an unsupported KV cache value")


def _tensor_tree_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.numel()) * int(value.element_size())
    if isinstance(value, (tuple, list)):
        return sum(_tensor_tree_bytes(item) for item in value)
    if value is None:
        return 0
    raise TypeError("local draft cache contains an unsupported value")


def _ensure_memory_reservation(
    reservation_bytes: int,
    device: torch.device,
) -> None:
    available = _available_memory_bytes(device)
    if available is None:
        raise RuntimeError(
            f"local draft memory availability is not measurable for {device.type}"
        )
    if int(reservation_bytes) > available:
        raise RuntimeError(
            "local draft memory reservation exceeds currently available memory"
        )


def _available_memory_bytes(device: torch.device) -> int | None:
    if device.type == "cuda":
        free, _total = torch.cuda.mem_get_info(device)
        return int(free)
    if device.type == "xpu":
        xpu = getattr(torch, "xpu", None)
        mem_get_info = getattr(xpu, "mem_get_info", None)
        if callable(mem_get_info):
            free, _total = mem_get_info(device)
            return int(free)
        return None
    if device.type in ("cpu", "mps"):
        if os.name == "nt":
            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(_MemoryStatusEx)
            kernel32 = ctypes.windll.kernel32
            if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return None
            return int(status.ullAvailPhys)
        try:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        except (AttributeError, OSError, TypeError, ValueError):
            return None
        return page_size * available_pages
    return None


def _release_device_cache(device: torch.device) -> None:
    try:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "xpu":
            empty_cache = getattr(getattr(torch, "xpu", None), "empty_cache", None)
            if callable(empty_cache):
                empty_cache()
    except Exception:
        # The optional acceleration path is already disabled. Cache cleanup
        # must never replace the exact autoregressive fallback with a failure.
        pass


def _token_history(value: Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise ValueError("token history must be a sequence of token ids")
    result = []
    for token in value:
        if (
            not isinstance(token, Integral)
            or isinstance(token, bool)
            or int(token) < 0
        ):
            raise ValueError("token history must contain non-negative integer ids")
        result.append(int(token))
    return tuple(result)


def _draft_limit(value: object) -> int:
    if (
        not isinstance(value, Integral)
        or isinstance(value, bool)
        or not 0 <= int(value) <= MAX_DRAFT_TOKENS
    ):
        raise ValueError(
            f"max_tokens must be an integer between 0 and {MAX_DRAFT_TOKENS}"
        )
    return int(value)


__all__ = [
    "LOCAL_DRAFT_MODEL_SCHEMA",
    "LocalDraftModelProvider",
    "LocalDraftModelRuntimeConfig",
    "LocalDraftModelStats",
    "add_local_draft_model_arguments",
    "load_local_draft_model_provider",
    "local_draft_model_config_from_args",
]
