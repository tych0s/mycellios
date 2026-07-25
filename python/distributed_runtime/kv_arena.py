"""Append-in-place KV cache: the decode step stops copying the whole history.

Why this exists
---------------
``transformers``' default ``DynamicLayer.update`` runs

    self.keys = torch.cat([self.keys, key_states], dim=-2)

on **every decode step, for every layer**.  One token therefore reallocates and
rewrites the entire key/value history: O(n) bytes per step, O(n^2) per sequence,
on top of the weight read that the step actually needs.  Profiling one CPU decode
step of the project testbed (SmolLM2-135M, 1 thread, 30 layers) attributes only
47.8% of the wall clock to ``aten::mm`` -- the real GEMV work.  The rest is layout
churn, and this concat is a first-class part of it.

What this module changes
------------------------
``ArenaLayer`` keeps a preallocated arena per layer and *writes new positions into
it*, growing by amortised doubling.  ``keys``/``values`` stay live views over the
written prefix, so every consumer -- attention, the physical batching merge/split,
storage accounting, speculative rollback -- reads exactly the tensor a
concatenating layer would have produced.

Exactness
---------
No arithmetic changes.  The same values are stored in the same order and attention
receives the same numbers, so greedy tokens are bit-identical to the concatenating
path.  ``tests/test_kv_arena.py`` asserts value identity against ``DynamicCache``
directly, and the pipeline benchmark keeps asserting token equality against the
monolithic FP32 reference.

Rollback
--------
``crop`` is the speculative-rollback primitive and is the one place where the
arena is strictly better than a slice: rejecting a draft only moves the write
cursor back, so accepted history is never copied and the freed capacity is reused
by the next wave.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers.cache_utils import (
    DYNAMIC_LAYER_TYPE_MAPPING,
    Cache,
    DynamicCache,
    DynamicLayer,
    get_layer_types_and_kwargs,
)

MINIMUM_ARENA_TOKENS = 64


class ArenaLayer(DynamicLayer):
    """``DynamicLayer`` that appends into a preallocated arena instead of concatenating.

    The public contract is unchanged: ``keys`` and ``values`` are ``[batch, heads,
    tokens, head_dim]`` tensors holding every cached position.  They are views over
    the arena rather than freshly concatenated tensors.
    """

    # Class-level defaults so an untouched layer (never updated, but deep-copied
    # or inspected by the KV accounting walk) behaves like an empty DynamicLayer.
    _key_arena: torch.Tensor | None = None
    _value_arena: torch.Tensor | None = None
    _length: int = 0

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        super().lazy_initialization(key_states, value_states)
        self._key_arena = None
        self._value_arena = None
        self._length = 0

    @property
    def arena_capacity_tokens(self) -> int:
        """Positions currently reserved, whether written or not."""

        return 0 if self._key_arena is None else int(self._key_arena.shape[-2])

    def _reserve(self, template: torch.Tensor, needed: int, *, exact: bool = False) -> None:
        capacity = self.arena_capacity_tokens
        batch_matches = (
            self._key_arena is not None
            and self._key_arena.shape[0] == template.shape[0]
            and self._key_arena.shape[1] == template.shape[1]
            and self._key_arena.shape[3] == template.shape[3]
            and self._key_arena.dtype == template.dtype
            and self._key_arena.device == template.device
        )
        if batch_matches and capacity >= needed:
            return
        if exact:
            # A batch arena lives for exactly one forward, so power-of-two doubling
            # would allocate (and fault in) up to twice the batch for nothing.
            target = needed
        else:
            target = max(MINIMUM_ARENA_TOKENS, capacity if batch_matches else 0)
            while target < needed:
                target *= 2
        shape = (template.shape[0], template.shape[1], target, template.shape[3])
        keys = torch.empty(shape, dtype=template.dtype, device=template.device)
        values = torch.empty(shape, dtype=template.dtype, device=template.device)
        if self._length and batch_matches:
            keys[..., : self._length, :] = self._key_arena[..., : self._length, :]
            values[..., : self._length, :] = self._value_arena[..., : self._length, :]
        elif self._length:
            # Geometry changed under us (batch/dtype/device); the previous prefix
            # cannot be reused, so the caller is starting a new logical cache.
            self._length = 0
        self._key_arena = keys
        self._value_arena = values

    def _publish(self) -> None:
        self.keys = self._key_arena[..., : self._length, :]
        self.values = self._value_arena[..., : self._length, :]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        added = int(key_states.shape[-2])
        self._reserve(key_states, self._length + added)
        end = self._length + added
        self._key_arena[..., self._length : end, :] = key_states
        self._value_arena[..., self._length : end, :] = value_states
        self._length = end
        self._publish()
        return self.keys, self.values

    def crop(self, max_length: int) -> None:
        """Roll back to an accepted prefix by moving the cursor, not by copying."""

        if self._key_arena is None:
            super().crop(max_length)
            return
        if max_length <= 0:
            max_length = self._length - abs(max_length)
        if self._length <= max_length:
            return
        self._length = max(0, int(max_length))
        self._publish()

    def reset(self) -> None:
        if self._key_arena is None:
            super().reset()
            return
        self._length = 0
        self._publish()

    def _adopt(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        """Re-seat the arena around tensors produced outside the append path."""

        self._key_arena = None
        self._value_arena = None
        self._length = 0
        if keys.shape[-2]:
            self._reserve(keys, int(keys.shape[-2]))
            self._key_arena[..., : keys.shape[-2], :] = keys
            self._value_arena[..., : values.shape[-2], :] = values
            self._length = int(keys.shape[-2])
            self._publish()
        else:
            self.keys = keys
            self.values = values

    def batch_repeat_interleave(self, repeats: int) -> None:
        if self.get_seq_length() > 0:
            self._adopt(
                self.keys.repeat_interleave(repeats, dim=0),
                self.values.repeat_interleave(repeats, dim=0),
            )

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        if self.get_seq_length() > 0:
            self._adopt(self.keys[indices, ...], self.values[indices, ...])

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        if self.get_seq_length() > 0:
            self._adopt(
                self.keys.index_select(0, beam_idx.to(self.keys.device)),
                self.values.index_select(0, beam_idx.to(self.values.device)),
            )

    def __deepcopy__(self, memo: dict[int, Any]) -> "ArenaLayer":
        """Fork a request without copying reserved-but-unwritten capacity.

        Exact speculative forks deep-copy the parent KV so branches never alias.
        Copying the whole arena would charge the fork for capacity the branch has
        not used, so only the live prefix is materialised.
        """

        clone = self.__class__.__new__(self.__class__)
        memo[id(self)] = clone
        for name, value in self.__dict__.items():
            if name in ("keys", "values", "_key_arena", "_value_arena"):
                continue
            setattr(clone, name, value)
        clone._key_arena = None
        clone._value_arena = None
        clone._length = 0
        if self._key_arena is not None and self._length:
            clone._reserve(self._key_arena, self._length)
            clone._key_arena[..., : self._length, :] = self._key_arena[..., : self._length, :]
            clone._value_arena[..., : self._length, :] = self._value_arena[..., : self._length, :]
            clone._length = self._length
            clone._publish()
        elif self._key_arena is not None:
            clone.keys = self.keys.clone()
            clone.values = self.values.clone()
        else:
            clone.keys = self.keys.clone() if self.keys is not None else None
            clone.values = self.values.clone() if self.values is not None else None
        return clone


class ArenaCache(DynamicCache):
    """``DynamicCache`` whose plain attention layers append in place.

    Constructed with the same ``config=`` keyword as ``DynamicCache`` so it is a
    drop-in at every call site.  Layer *types* are resolved exactly as the parent
    does; only layers that would have been a plain ``DynamicLayer`` are swapped
    for an ``ArenaLayer``.  Sliding and hybrid layers keep their own class,
    because their window bookkeeping is not an append-only cursor.
    """

    def __init__(self, config: Any = None, **kwargs: Any) -> None:
        layers: list[Any] = []
        if config is not None:
            decoder_config = config.get_text_config(decoder=True)
            layer_types, layer_kwargs = get_layer_types_and_kwargs(decoder_config)
            for layer_type in layer_types:
                mapped = DYNAMIC_LAYER_TYPE_MAPPING[layer_type]
                layers.append(
                    ArenaLayer(**layer_kwargs) if mapped is DynamicLayer else mapped(**layer_kwargs)
                )
        if layers:
            Cache.__init__(self, layers=layers)
        else:
            Cache.__init__(self, layer_class_to_replicate=ArenaLayer)


def stack_into_arena(
    layer: ArenaLayer,
    key_slices: list[torch.Tensor],
    value_slices: list[torch.Tensor],
    *,
    headroom: int = 0,
) -> None:
    """Gather per-request KV rows into one batched arena with a single copy each.

    The generic path costs two full copies of the batch: ``torch.cat`` materialises
    it, and the receiving layer then concatenates that result onto its own empty
    tensor.  Writing each request straight into a reserved arena row costs one.

    ``headroom`` reserves room for the positions the forward is about to append.
    Getting this wrong is expensive rather than merely suboptimal: an arena sized
    exactly to the history has to grow on the model's very first append, and that
    growth copies the whole batch -- turning the optimisation into a regression
    that gets worse as the conversation gets longer.
    """

    if not key_slices:
        raise ValueError("cannot stack an empty physical batch")
    if headroom < 0:
        raise ValueError("headroom cannot be negative")
    first = key_slices[0]
    batch = len(key_slices)
    tokens = int(first.shape[-2])
    if not layer.is_initialized:
        layer.lazy_initialization(first, value_slices[0])
    template = first.new_empty((batch, int(first.shape[1]), 0, int(first.shape[3])))
    layer._length = 0
    layer._reserve(template, tokens + headroom, exact=True)
    for index, (keys, values) in enumerate(zip(key_slices, value_slices, strict=True)):
        layer._key_arena[index : index + 1, :, :tokens, :] = keys
        layer._value_arena[index : index + 1, :, :tokens, :] = values
    layer._length = tokens
    layer._publish()


def is_arena_layer(layer: Any) -> bool:
    return type(layer) is ArenaLayer


def arena_reserved_bytes(cache: Any) -> int:
    """Bytes reserved by arenas in ``cache`` (written prefix plus spare capacity)."""

    layers = getattr(cache, "layers", ())
    total = 0
    for layer in layers:
        arena = getattr(layer, "_key_arena", None)
        if arena is None:
            continue
        total += arena.numel() * arena.element_size()
        values = getattr(layer, "_value_arena", None)
        if values is not None:
            total += values.numel() * values.element_size()
    return total


__all__ = [
    "ArenaCache",
    "ArenaLayer",
    "MINIMUM_ARENA_TOKENS",
    "arena_reserved_bytes",
    "is_arena_layer",
    "stack_into_arena",
]
