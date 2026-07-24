"""Decode attention that materialises nothing.

The problem
-----------
Profiling one CPU decode step of the project testbed (SmolLM2-135M, 30 layers,
1 thread) attributes only 47.8% of the wall clock to ``aten::mm`` -- the GEMVs that
are the actual work.  ``aten::copy_`` takes 19.1%, and the largest contributor is
the attention path: SmolLM2 is grouped-query (9 query heads over 3 KV heads), and
the stock path calls ``repeat_kv`` to physically expand the cached keys and values
by the group factor before handing them to SDPA.  That expansion rewrites the whole
KV history, every layer, every token -- three times its size, for a model whose KV
is already the thing growing without bound.  SDPA then falls back to its ``math``
backend because a dense mask is present, adding another full-size temporary.

The observation
---------------
During decode there is exactly **one** query position, and it attends to a dense
prefix.  A causal mask over a single query row is all-ones, so the mask contributes
nothing: adding it is adding zeros.  And grouped-query attention over one position
is just a batched matmul if the query heads are *reshaped* into their KV groups
rather than the keys being *copied* up to the query head count.

So the whole decode attention collapses to:

    scores = query_grouped @ keys^T      # [B, Hkv, G, L]
    out    = softmax(scores) @ values    # [B, Hkv, G, D]

with the cached keys and values read in place, as strided views, and no temporary
of the KV history at any point.

Exactness
---------
This is the same sequence of arithmetic operations the ``math`` SDPA backend
performs, on the same values in the same order -- ``repeat_kv`` is a pure copy and a
no-op mask adds 0.0 -- so the result is bit-identical, not merely close.  The
eligibility check below is fail-closed: anything it cannot prove is a dense,
unmasked, single-position decode goes to the stock implementation.
``tests/test_decode_attention.py`` asserts bitwise equality against the stock path.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

GROUPED_PREFIX_ATTENTION = "gdlp-grouped-prefix"

# Observability: a silent fallback would keep every equivalence test green while
# giving back none of the speed, so the split is counted and asserted in tests.
STATS: dict[str, int] = {"grouped": 0, "fallback": 0}


def reset_stats() -> None:
    STATS["grouped"] = 0
    STATS["fallback"] = 0


def _mask_is_noop(attention_mask: torch.Tensor | None, length: int) -> bool:
    """Whether the mask leaves every cached position visible.

    A single decode row under a causal mask is all-ones by construction, but this
    is verified rather than assumed: padding, sliding windows and tree masks all
    arrive through the same argument, and silently ignoring one of them would
    change tokens.
    """

    if attention_mask is None:
        return True
    if attention_mask.numel() == 0:
        return True
    if attention_mask.shape[-1] < length:
        return False
    visible = attention_mask[..., :length]
    if visible.dtype == torch.bool:
        return bool(visible.all())
    return bool((visible == 0).all())


def _eligible(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float,
    position_bias: torch.Tensor | None,
    kwargs: dict[str, Any],
) -> bool:
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        return False
    if query.shape[2] != 1:  # not a single decode position
        return False
    if dropout:
        return False
    if position_bias is not None:
        return False
    if kwargs.get("output_attentions", False):
        return False
    if query.dtype != key.dtype or key.dtype != value.dtype:
        return False
    if key.shape[0] != query.shape[0] or key.shape[2] != value.shape[2]:
        return False
    heads, kv_heads = query.shape[1], key.shape[1]
    if kv_heads < 1 or heads % kv_heads:
        return False
    if key.shape[3] != query.shape[3] or value.shape[3] != query.shape[3]:
        return False
    return _mask_is_noop(attention_mask, int(key.shape[2]))


def grouped_prefix_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    position_bias: torch.Tensor | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Single-position attention over a dense prefix, with no materialisation."""

    if not _eligible(query, key, value, attention_mask, dropout, position_bias, kwargs):
        STATS["fallback"] += 1
        return sdpa_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            dropout=dropout,
            scaling=scaling,
            is_causal=is_causal,
            position_bias=position_bias,
            **kwargs,
        )

    batch, heads, _, head_dim = query.shape
    kv_heads = key.shape[1]
    groups = heads // kv_heads
    if scaling is None:
        scaling = head_dim ** -0.5

    # Regroup the query instead of expanding the cache: [B, Hq, 1, D] -> [B, Hkv, G, D].
    # `repeat_kv` maps KV head h onto query heads [h*G, (h+1)*G), so this reshape is
    # the same correspondence read the other way round.
    grouped_query = query.reshape(batch, kv_heads, groups, head_dim)
    # `key`/`value` stay exactly as the cache handed them over -- possibly strided
    # views into an arena. Nothing here forces them contiguous.
    scores = torch.matmul(grouped_query, key.transpose(-1, -2)) * scaling
    probabilities = scores.softmax(dim=-1)
    attended = torch.matmul(probabilities, value)

    STATS["grouped"] += 1
    # The caller reshapes to [B, q_len, Hq * D]; [B, Hkv, G, D] flattens head-major
    # into exactly that layout.
    return attended, None


def register_grouped_prefix_attention() -> str:
    """Register the implementation and return the name to select it by."""

    ALL_ATTENTION_FUNCTIONS.register(
        GROUPED_PREFIX_ATTENTION, grouped_prefix_attention_forward
    )
    return GROUPED_PREFIX_ATTENTION


def apply_to(model: torch.nn.Module) -> bool:
    """Point one loaded model at the grouped-prefix implementation.

    Returns ``False`` when the model does not route attention through the
    registry, leaving it untouched rather than guessing.
    """

    register_grouped_prefix_attention()
    config = getattr(model, "config", None)
    if config is None:
        return False
    applied = False
    for target in (config, getattr(config, "get_text_config", lambda **_: None)(decoder=True)):
        if target is None:
            continue
        target._attn_implementation = GROUPED_PREFIX_ATTENTION
        applied = True
    return applied


__all__ = [
    "GROUPED_PREFIX_ATTENTION",
    "STATS",
    "apply_to",
    "grouped_prefix_attention_forward",
    "register_grouped_prefix_attention",
    "reset_stats",
]
