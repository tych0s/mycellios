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

Prefill
-------
The same file also fixes the other end of the step. Prefill arrives with a *materialised*
causal mask, which forces SDPA onto its math backend; profiling shows that backend then
spends about a quarter of the whole prefill inside ``isneginf`` and ``where``, doing
bookkeeping over an O(n^2) mask tensor. Since the mask is only ever "attend to positions
at or before mine", handing SDPA ``is_causal=True`` and no mask at all lets it dispatch to
the fused kernel. Measured bit-identical (max deviation exactly 0.0) and 2.04x at 4096
tokens. The mask is *verified* to be plain causal, never assumed, and the verdict is
memoised on the mask object so the check is paid once per forward rather than per layer.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, sdpa_mask
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

GROUPED_PREFIX_ATTENTION = "gdlp-grouped-prefix"

# Observability: a silent fallback would keep every equivalence test green while
# giving back none of the speed, so the split is counted and asserted in tests.
STATS: dict[str, int] = {"grouped": 0, "causal_prefill": 0, "fallback": 0}


def reset_stats() -> None:
    for key in STATS:
        STATS[key] = 0


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


class _CausalMaskMemo:
    """Remember the verdict for the mask object a forward pass is reusing.

    Proving a mask is the plain causal mask costs one O(n^2) comparison. Paying
    that per layer would cost more than it saves, but a forward pass builds the
    mask once and hands the *same tensor* to every layer, so one verification
    covers all of them. Identity plus the tensor's version counter is what makes
    reuse sound: a different object, or an in-place edit to this one, misses.
    """

    def __init__(self) -> None:
        self._tensor: torch.Tensor | None = None
        self._version: int | None = None
        self._verdict = False

    def lookup(self, mask: torch.Tensor) -> bool | None:
        if self._tensor is mask and self._version == mask._version:
            return self._verdict
        return None

    def remember(self, mask: torch.Tensor, verdict: bool) -> None:
        self._tensor = mask
        self._version = mask._version
        self._verdict = verdict


_CAUSAL_MEMO = _CausalMaskMemo()


def _is_plain_causal_mask(mask: torch.Tensor, query_length: int, key_length: int) -> bool:
    """Whether ``mask`` is exactly 'attend to every position at or before mine'.

    Verified, never assumed: padding, sliding windows, prefix-LM and tree masks all
    arrive through this argument, and treating one of them as causal would silently
    change tokens.
    """

    if query_length != key_length or mask.shape[-1] != key_length or mask.shape[-2] != query_length:
        return False
    if mask.ndim != 4 or mask.shape[0] != 1:
        # Per-sequence masks in a padded batch are exactly the case that is not causal.
        return False
    cached = _CAUSAL_MEMO.lookup(mask)
    if cached is not None:
        return cached
    upper = torch.ones(query_length, key_length, dtype=torch.bool, device=mask.device).triu(1)
    if mask.dtype == torch.bool:
        verdict = bool(torch.equal(mask[0, 0], ~upper))
    else:
        blocked = torch.isneginf(mask[0, 0])
        verdict = bool(torch.equal(blocked, upper)) and bool((mask[0, 0][~upper] == 0).all())
    _CAUSAL_MEMO.remember(mask, verdict)
    return verdict


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
    if heads == kv_heads:
        # Nothing to regroup. With no expansion to avoid, torch's fused kernel beats
        # two explicit matmuls plus a Python-level softmax -- measured a regression of
        # up to 1.7x per layer. The gate has to be closed against losing, not only
        # against being wrong.
        return False
    if key.shape[3] != query.shape[3] or value.shape[3] != query.shape[3]:
        return False
    return _mask_is_noop(attention_mask, int(key.shape[2]))


def _prefill_eligible(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float,
    position_bias: torch.Tensor | None,
    kwargs: dict[str, Any],
) -> bool:
    """Whether this is a whole-prompt causal prefill that the fused kernel can take.

    ``is_causal=True`` aligns the mask to the upper left, so it is only the right
    answer when the query covers the entire cache. A *chunked* prefill, where earlier
    chunks already sit in the cache, has more keys than queries and must not take this
    path -- it would silently attend to the wrong window.
    """

    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        return False
    if dropout or position_bias is not None or kwargs.get("output_attentions", False):
        return False
    if query.device.type != "cpu":
        # The flag being worked around is only slow in torch's CPU implementation.
        return False
    query_length, key_length = int(query.shape[2]), int(key.shape[2])
    if query_length < 2 or key_length != query_length or int(value.shape[2]) != key_length:
        return False
    heads, kv_heads = int(query.shape[1]), int(key.shape[1])
    if kv_heads < 1 or heads % kv_heads or heads == kv_heads:
        return False
    if query.dtype != key.dtype or key.dtype != value.dtype:
        return False
    if key.shape[0] != query.shape[0] or key.shape[3] != query.shape[3]:
        return False
    if attention_mask is None:
        return True
    return _is_plain_causal_mask(attention_mask, query_length, key_length)


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
        if _prefill_eligible(query, key, value, attention_mask, dropout, position_bias, kwargs):
            # Whole-prompt prefill on CPU. Transformers hands SDPA `enable_gqa=True`
            # here (it does that whenever the mask is None), and torch's CPU
            # implementation of that flag is pathologically slow: measured 2002 ms per
            # layer at 4096 tokens against 257 ms for the same attention with the keys
            # and values expanded explicitly. Paying one expansion to reach the fused
            # causal kernel is ~7.8x cheaper, and prefill *is* the time-to-first-token.
            groups = query.shape[1] // key.shape[1]
            expanded_key = torch.repeat_interleave(key, groups, dim=1)
            expanded_value = torch.repeat_interleave(value, groups, dim=1)
            attended = torch.nn.functional.scaled_dot_product_attention(
                query,
                expanded_key,
                expanded_value,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True,
                scale=scaling,
            )
            STATS["causal_prefill"] += 1
            return attended.transpose(1, 2).contiguous(), None
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
    """Register the implementation and return the name to select it by.

    Registering the *mask builder* under the same name is not optional. Transformers
    treats an attention implementation it does not recognise as a custom backend that
    builds its own masks, and skips mask construction entirely
    (``masking_utils.create_causal_mask`` returns ``None``). Every layer would then be
    handed ``attention_mask=None``, and the stock fallback reacts to that by inferring
    ``is_causal=True`` and *slicing the cache down to the query length*
    (``sdpa_attention.py``: ``key = key[:, :, :q_length, :]``) -- silently dropping the
    KV history. Whole-prompt prefill and single-token decode are unaffected, which is
    what makes it dangerous: it only corrupts ``1 < q_len < kv_len``, which is exactly
    chunked prefill and the speculative VERIFY wave.
    """

    ALL_ATTENTION_FUNCTIONS.register(
        GROUPED_PREFIX_ATTENTION, grouped_prefix_attention_forward
    )
    ALL_MASK_ATTENTION_FUNCTIONS.register(GROUPED_PREFIX_ATTENTION, sdpa_mask)
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
