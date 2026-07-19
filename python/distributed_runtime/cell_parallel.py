from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
import torch.distributed as distributed
from torch.nn import functional as functional


@dataclass(frozen=True)
class ShardRange:
    rank: int
    world_size: int
    start: int
    end: int
    total: int

    @property
    def size(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class LlamaAttentionShardPlan:
    rank: int
    world_size: int
    hidden_size: int
    head_dim: int
    query_heads: ShardRange
    key_value_heads: ShardRange
    query_output_sizes: tuple[int, ...]
    query_to_local_key_value: tuple[int, ...]

    @property
    def local_query_heads(self) -> int:
        return self.query_heads.size

    @property
    def local_key_value_heads(self) -> int:
        return self.key_value_heads.size

    @property
    def query_heads_per_key_value_head(self) -> int:
        return self.query_heads.total // self.key_value_heads.total


def balanced_shard(total: int, rank: int, world_size: int) -> ShardRange:
    if not isinstance(total, int) or isinstance(total, bool) or total < 1:
        raise ValueError("total must be a positive integer")
    if not isinstance(world_size, int) or isinstance(world_size, bool) or world_size < 1:
        raise ValueError("world_size must be a positive integer")
    if world_size > total:
        raise ValueError("world_size cannot exceed the sharded dimension")
    if not isinstance(rank, int) or isinstance(rank, bool) or not 0 <= rank < world_size:
        raise ValueError("rank must be between zero and world_size - 1")
    base, remainder = divmod(total, world_size)
    size = base + (1 if rank < remainder else 0)
    start = rank * base + min(rank, remainder)
    return ShardRange(rank, world_size, start, start + size, total)


def weighted_shard(
    total: int,
    rank: int,
    rank_weights: Sequence[float],
) -> ShardRange:
    """Partition an integer dimension proportionally without empty ranks.

    The result is deterministic Hamilton apportionment over the units left
    after reserving one unit for every rank.  Ties are resolved by lower rank,
    so the same sealed weights produce identical slices on every host.  Equal
    weights are intentionally identical to :func:`balanced_shard`.
    """

    if isinstance(rank_weights, (str, bytes, bytearray)):
        raise TypeError("rank_weights must be an ordered numeric sequence")
    weights = tuple(rank_weights)
    world_size = len(weights)
    if world_size < 1:
        raise ValueError("rank_weights cannot be empty")
    if not isinstance(total, int) or isinstance(total, bool) or total < 1:
        raise ValueError("total must be a positive integer")
    if world_size > total:
        raise ValueError("world_size cannot exceed the sharded dimension")
    if not isinstance(rank, int) or isinstance(rank, bool) or not 0 <= rank < world_size:
        raise ValueError("rank must be between zero and world_size - 1")
    normalized: list[float] = []
    for value in weights:
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or value <= 0
        ):
            raise ValueError("rank_weights must contain finite positive numbers")
        normalized.append(float(value))
    remaining = total - world_size
    weight_sum = math.fsum(normalized)
    quotas = [remaining * value / weight_sum for value in normalized]
    extras = [math.floor(value) for value in quotas]
    unassigned = remaining - sum(extras)
    order = sorted(
        range(world_size),
        key=lambda member: (-(quotas[member] - extras[member]), member),
    )
    for member in order[:unassigned]:
        extras[member] += 1
    sizes = tuple(1 + value for value in extras)
    start = sum(sizes[:rank])
    return ShardRange(rank, world_size, start, start + sizes[rank], total)


def shard_sizes(
    total: int,
    world_size: int,
    rank_weights: Sequence[float] | None = None,
) -> tuple[int, ...]:
    if rank_weights is None:
        return tuple(
            balanced_shard(total, rank, world_size).size for rank in range(world_size)
        )
    if len(rank_weights) != world_size:
        raise ValueError("rank_weights length must equal world_size")
    return tuple(weighted_shard(total, rank, rank_weights).size for rank in range(world_size))


def _planned_shard(
    total: int,
    rank: int,
    world_size: int,
    rank_weights: Sequence[float] | None,
) -> ShardRange:
    if rank_weights is None:
        return balanced_shard(total, rank, world_size)
    if len(rank_weights) != world_size:
        raise ValueError("rank_weights length must equal world_size")
    return weighted_shard(total, rank, rank_weights)


def llama_attention_shard_plan(
    hidden_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    rank: int,
    world_size: int,
    rank_weights: Sequence[float] | None = None,
) -> LlamaAttentionShardPlan:
    """Plan exact Llama GQA/MQA shards, including Q-only over-decomposition.

    With no more ranks than KV heads, complete query groups follow a unique KV
    owner and K/V weights plus cache remain unreplicated. With additional ranks
    (up to the query-head count), Q heads are balanced and each rank receives
    only the contiguous KV heads its Q slice references. That can replicate a
    boundary KV head, which is required for exact MQA/GQA without communicating
    K/V activations on every token.
    """
    for name, value in (
        ("hidden_size", hidden_size),
        ("num_attention_heads", num_attention_heads),
        ("num_key_value_heads", num_key_value_heads),
        ("head_dim", head_dim),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if num_attention_heads % num_key_value_heads != 0:
        raise ValueError("attention heads must be divisible by key/value heads")
    if num_attention_heads * head_dim != hidden_size:
        raise ValueError("attention heads times head_dim must equal hidden_size")
    query_per_kv = num_attention_heads // num_key_value_heads

    if world_size <= num_key_value_heads:
        # Keep whole GQA groups together whenever every rank can own at least
        # one KV head. This is the zero-replication path: each checkpoint KV
        # row and each cache entry exists on exactly one member.
        kv = _planned_shard(
            num_key_value_heads, rank, world_size, rank_weights
        )
        query = ShardRange(
            rank=rank,
            world_size=world_size,
            start=kv.start * query_per_kv,
            end=kv.end * query_per_kv,
            total=num_attention_heads,
        )
        query_sizes = tuple(
            _planned_shard(
                num_key_value_heads, member, world_size, rank_weights
            ).size
            * query_per_kv
            * head_dim
            for member in range(world_size)
        )
    else:
        # MQA and narrow-GQA can still use more ranks than KV heads. Split Q
        # heads exactly once, then give each rank the smallest contiguous KV
        # range referenced by its Q slice. Adjacent ranks may intentionally
        # replicate a boundary KV head, but never unrelated KV weights/cache.
        query = _planned_shard(
            num_attention_heads, rank, world_size, rank_weights
        )
        first_kv = query.start // query_per_kv
        last_kv_exclusive = (query.end - 1) // query_per_kv + 1
        kv = ShardRange(
            rank=rank,
            world_size=world_size,
            start=first_kv,
            end=last_kv_exclusive,
            total=num_key_value_heads,
        )
        query_sizes = tuple(
            _planned_shard(
                num_attention_heads, member, world_size, rank_weights
            ).size * head_dim
            for member in range(world_size)
        )

    query_to_local_key_value = tuple(
        (global_query // query_per_kv) - kv.start
        for global_query in range(query.start, query.end)
    )
    return LlamaAttentionShardPlan(
        rank=rank,
        world_size=world_size,
        hidden_size=hidden_size,
        head_dim=head_dim,
        query_heads=query,
        key_value_heads=kv,
        query_output_sizes=query_sizes,
        query_to_local_key_value=query_to_local_key_value,
    )


def shard_llama_attention(
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    value_weight: torch.Tensor,
    output_weight: torch.Tensor,
    *,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    rank: int,
    world_size: int,
    rank_weights: Sequence[float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, LlamaAttentionShardPlan]:
    hidden_size = int(query_weight.shape[1])
    plan = llama_attention_shard_plan(
        hidden_size,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        rank,
        world_size,
        rank_weights,
    )
    expected = (
        (num_attention_heads * head_dim, hidden_size),
        (num_key_value_heads * head_dim, hidden_size),
        (num_key_value_heads * head_dim, hidden_size),
        (hidden_size, num_attention_heads * head_dim),
    )
    actual = tuple(tuple(value.shape) for value in (
        query_weight,
        key_weight,
        value_weight,
        output_weight,
    ))
    if actual != expected:
        raise ValueError(f"attention weight shapes {actual} do not match {expected}")
    q_start = plan.query_heads.start * head_dim
    q_end = plan.query_heads.end * head_dim
    kv_start = plan.key_value_heads.start * head_dim
    kv_end = plan.key_value_heads.end * head_dim
    return (
        query_weight[q_start:q_end, :].contiguous(),
        key_weight[kv_start:kv_end, :].contiguous(),
        value_weight[kv_start:kv_end, :].contiguous(),
        output_weight[:, q_start:q_end].contiguous(),
        plan,
    )


def shard_column_linear(
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    rank: int,
    world_size: int,
    rank_weights: Sequence[float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, ShardRange]:
    """Offline split of output features for Q/K/V, gate or up projections."""

    _validate_linear_weight(weight)
    if bias is not None and (bias.ndim != 1 or bias.shape[0] != weight.shape[0]):
        raise ValueError("bias must match the output dimension")
    shard = _planned_shard(
        int(weight.shape[0]), rank, world_size, rank_weights
    )
    local_weight = weight[shard.start : shard.end, :].contiguous()
    local_bias = (
        None if bias is None else bias[shard.start : shard.end].contiguous()
    )
    return local_weight, local_bias, shard


def shard_row_linear(
    weight: torch.Tensor,
    rank: int,
    world_size: int,
    rank_weights: Sequence[float] | None = None,
) -> tuple[torch.Tensor, ShardRange]:
    """Offline split of input features for attention output or down projections."""

    _validate_linear_weight(weight)
    shard = _planned_shard(
        int(weight.shape[1]), rank, world_size, rank_weights
    )
    return weight[:, shard.start : shard.end].contiguous(), shard


@torch.inference_mode()
def column_parallel_linear(
    inputs: torch.Tensor,
    local_weight: torch.Tensor,
    local_bias: torch.Tensor | None,
    output_sizes: Sequence[int],
    *,
    group: object | None = None,
    gather_output: bool = True,
) -> torch.Tensor:
    """Execute an output-feature shard and optionally gather the full result.

    Inputs are replicated, but only one output slice of the weight and bias is
    resident on each member.  Uneven dimensions are padded only for the
    collective and trimmed immediately afterwards.
    """

    _validate_inputs(inputs, local_weight)
    sizes = _validate_collective_sizes(output_sizes, int(local_weight.shape[0]), group)
    if local_bias is not None and (
        local_bias.ndim != 1 or local_bias.shape[0] != local_weight.shape[0]
    ):
        raise ValueError("local_bias must match local output features")
    local_output = functional.linear(inputs, local_weight, local_bias)
    if not gather_output or len(sizes) == 1:
        return local_output
    maximum = max(sizes)
    if local_output.shape[-1] < maximum:
        local_output = functional.pad(
            local_output,
            (0, maximum - int(local_output.shape[-1])),
        )
    gathered = [torch.empty_like(local_output) for _ in sizes]
    distributed.all_gather(gathered, local_output, group=group)
    return torch.cat(
        [value[..., :size] for value, size in zip(gathered, sizes)],
        dim=-1,
    )


@torch.inference_mode()
def row_parallel_linear(
    local_inputs: torch.Tensor,
    local_weight: torch.Tensor,
    bias: torch.Tensor | None,
    input_sizes: Sequence[int],
    *,
    group: object | None = None,
) -> torch.Tensor:
    """Execute an input-feature shard and sum partial outputs on every member."""

    _validate_inputs(local_inputs, local_weight)
    _validate_collective_sizes(input_sizes, int(local_weight.shape[1]), group)
    if bias is not None and (
        bias.ndim != 1 or bias.shape[0] != local_weight.shape[0]
    ):
        raise ValueError("bias must match the replicated output dimension")
    output = functional.linear(local_inputs, local_weight, None)
    if len(input_sizes) > 1:
        distributed.all_reduce(output, op=distributed.ReduceOp.SUM, group=group)
    if bias is not None:
        output.add_(bias)
    return output


@torch.inference_mode()
def llama_gated_mlp_tensor_parallel(
    inputs: torch.Tensor,
    local_gate_weight: torch.Tensor,
    local_up_weight: torch.Tensor,
    local_down_weight: torch.Tensor,
    intermediate_sizes: Sequence[int],
    *,
    local_gate_bias: torch.Tensor | None = None,
    local_up_bias: torch.Tensor | None = None,
    down_bias: torch.Tensor | None = None,
    group: object | None = None,
) -> torch.Tensor:
    """Exact tensor-parallel Llama-style SiLU-gated MLP subgraph.

    Gate/up projections are column-sharded and remain local through the
    elementwise product.  The down projection is row-sharded and performs one
    all-reduce, so no member needs a full intermediate matrix or activation.
    """

    _validate_inputs(inputs, local_gate_weight)
    _validate_inputs(inputs, local_up_weight)
    if local_gate_weight.shape != local_up_weight.shape:
        raise ValueError("local gate and up weights must have equal shapes")
    if (
        local_down_weight.ndim != 2
        or local_down_weight.shape[1] != local_gate_weight.shape[0]
    ):
        raise ValueError("local down weight must consume the local intermediate shard")
    for name, bias in (
        ("local_gate_bias", local_gate_bias),
        ("local_up_bias", local_up_bias),
    ):
        if bias is not None and (
            bias.ndim != 1 or bias.shape[0] != local_gate_weight.shape[0]
        ):
            raise ValueError(f"{name} must match the local intermediate shard")
    gate = functional.linear(inputs, local_gate_weight, local_gate_bias)
    up = functional.linear(inputs, local_up_weight, local_up_bias)
    local_hidden = functional.silu(gate) * up
    return row_parallel_linear(
        local_hidden,
        local_down_weight,
        down_bias,
        intermediate_sizes,
        group=group,
    )


@torch.inference_mode()
def llama_attention_tensor_parallel(
    inputs: torch.Tensor,
    local_query_weight: torch.Tensor,
    local_key_weight: torch.Tensor,
    local_value_weight: torch.Tensor,
    local_output_weight: torch.Tensor,
    plan: LlamaAttentionShardPlan,
    *,
    past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
    rope_theta: float = 10_000.0,
    output_bias: torch.Tensor | None = None,
    group: object | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Tensor-parallel GQA/MQA with rank-local, minimally replicated KV cache."""

    if inputs.ndim != 3:
        raise ValueError("attention inputs must have shape [batch, tokens, hidden]")
    if inputs.shape[-1] != plan.hidden_size:
        raise ValueError("attention input hidden size does not match the shard plan")
    if not math.isfinite(rope_theta) or rope_theta <= 0:
        raise ValueError("rope_theta must be finite and positive")
    local_query_features = plan.local_query_heads * plan.head_dim
    local_kv_features = plan.local_key_value_heads * plan.head_dim
    expected_shapes = (
        (local_query_features, plan.hidden_size),
        (local_kv_features, plan.hidden_size),
        (local_kv_features, plan.hidden_size),
        (plan.hidden_size, local_query_features),
    )
    actual_shapes = tuple(
        tuple(value.shape)
        for value in (
            local_query_weight,
            local_key_weight,
            local_value_weight,
            local_output_weight,
        )
    )
    if actual_shapes != expected_shapes:
        raise ValueError(
            f"local attention weights {actual_shapes} do not match {expected_shapes}"
        )
    for value in (
        local_query_weight,
        local_key_weight,
        local_value_weight,
        local_output_weight,
    ):
        if value.dtype != inputs.dtype or value.device != inputs.device:
            raise ValueError("attention inputs and weights must share dtype and device")

    batch, tokens, _ = inputs.shape
    query = functional.linear(inputs, local_query_weight)
    key = functional.linear(inputs, local_key_weight)
    value = functional.linear(inputs, local_value_weight)
    query = query.view(batch, tokens, plan.local_query_heads, plan.head_dim).transpose(1, 2)
    key = key.view(batch, tokens, plan.local_key_value_heads, plan.head_dim).transpose(1, 2)
    value = value.view(batch, tokens, plan.local_key_value_heads, plan.head_dim).transpose(1, 2)

    past_tokens = 0
    if past_key_value is not None:
        past_key, past_value = past_key_value
        expected_prefix = (batch, plan.local_key_value_heads)
        if (
            past_key.ndim != 4
            or past_value.shape != past_key.shape
            or tuple(past_key.shape[:2]) != expected_prefix
            or past_key.shape[-1] != plan.head_dim
            or past_key.dtype != inputs.dtype
            or past_key.device != inputs.device
        ):
            raise ValueError("past key/value cache does not match this attention shard")
        past_tokens = int(past_key.shape[2])
    cosine, sine = _llama_rope(
        past_tokens,
        tokens,
        plan.head_dim,
        rope_theta,
        dtype=inputs.dtype,
        device=inputs.device,
    )
    query = query * cosine + _rotate_half(query) * sine
    key = key * cosine + _rotate_half(key) * sine
    if past_key_value is not None:
        key = torch.cat((past_key_value[0], key), dim=2)
        value = torch.cat((past_key_value[1], value), dim=2)
    present = (key, value)

    repeats = plan.query_heads_per_key_value_head
    aligned_mapping = tuple(
        local_key_value
        for local_key_value in range(plan.local_key_value_heads)
        for _ in range(repeats)
    )
    if plan.query_to_local_key_value == aligned_mapping:
        # Fast zero-replication path for ordinary GQA shards.
        expanded_key = key.repeat_interleave(repeats, dim=1)
        expanded_value = value.repeat_interleave(repeats, dim=1)
    else:
        # A rank can own only part of a GQA group when world_size exceeds the
        # KV-head count. Select the referenced local KV head for every local Q
        # head; the compiler has already materialized the minimal KV range.
        query_to_key_value = torch.tensor(
            plan.query_to_local_key_value,
            dtype=torch.long,
            device=inputs.device,
        )
        expanded_key = key.index_select(1, query_to_key_value)
        expanded_value = value.index_select(1, query_to_key_value)
    scores = torch.matmul(query, expanded_key.transpose(-1, -2)) / math.sqrt(plan.head_dim)
    query_positions = torch.arange(
        past_tokens,
        past_tokens + tokens,
        device=inputs.device,
    )
    key_positions = torch.arange(key.shape[2], device=inputs.device)
    causal = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
    scores = scores.masked_fill(~causal[None, None, :, :], torch.finfo(scores.dtype).min)
    probabilities = torch.softmax(scores.float(), dim=-1).to(dtype=inputs.dtype)
    context = torch.matmul(probabilities, expanded_value)
    context = context.transpose(1, 2).contiguous().view(batch, tokens, local_query_features)
    output = row_parallel_linear(
        context,
        local_output_weight,
        output_bias,
        plan.query_output_sizes,
        group=group,
    )
    return output, present


@torch.inference_mode()
def llama_decoder_layer_tensor_parallel(
    inputs: torch.Tensor,
    input_norm_weight: torch.Tensor,
    post_attention_norm_weight: torch.Tensor,
    local_query_weight: torch.Tensor,
    local_key_weight: torch.Tensor,
    local_value_weight: torch.Tensor,
    local_output_weight: torch.Tensor,
    attention_plan: LlamaAttentionShardPlan,
    local_gate_weight: torch.Tensor,
    local_up_weight: torch.Tensor,
    local_down_weight: torch.Tensor,
    intermediate_sizes: Sequence[int],
    *,
    past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
    rms_norm_epsilon: float = 1e-6,
    rope_theta: float = 10_000.0,
    group: object | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """One exact dense Llama-style decoder layer split across cell members."""

    if (
        input_norm_weight.shape != (attention_plan.hidden_size,)
        or post_attention_norm_weight.shape != (attention_plan.hidden_size,)
    ):
        raise ValueError("RMSNorm weights must match hidden_size")
    if not math.isfinite(rms_norm_epsilon) or rms_norm_epsilon <= 0:
        raise ValueError("rms_norm_epsilon must be finite and positive")
    normalized = _rms_norm(inputs, input_norm_weight, rms_norm_epsilon)
    attention, present = llama_attention_tensor_parallel(
        normalized,
        local_query_weight,
        local_key_weight,
        local_value_weight,
        local_output_weight,
        attention_plan,
        past_key_value=past_key_value,
        rope_theta=rope_theta,
        group=group,
    )
    hidden = inputs + attention
    mlp_input = _rms_norm(hidden, post_attention_norm_weight, rms_norm_epsilon)
    mlp = llama_gated_mlp_tensor_parallel(
        mlp_input,
        local_gate_weight,
        local_up_weight,
        local_down_weight,
        intermediate_sizes,
        group=group,
    )
    return hidden + mlp, present


def _rms_norm(inputs: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    variance = inputs.float().pow(2).mean(dim=-1, keepdim=True)
    normalized = inputs * torch.rsqrt(variance + epsilon).to(dtype=inputs.dtype)
    return normalized * weight


def _llama_rope(
    position_start: int,
    tokens: int,
    head_dim: int,
    theta: float,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if head_dim % 2 != 0:
        raise ValueError("Llama rotary head_dim must be even")
    frequencies = 1.0 / (
        theta
        ** (
            torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
            / head_dim
        )
    )
    positions = torch.arange(
        position_start,
        position_start + tokens,
        dtype=torch.float32,
        device=device,
    )
    angles = torch.outer(positions, frequencies)
    embedding = torch.cat((angles, angles), dim=-1)
    shape = (1, 1, tokens, head_dim)
    return (
        embedding.cos().to(dtype=dtype).view(shape),
        embedding.sin().to(dtype=dtype).view(shape),
    )


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _validate_linear_weight(weight: torch.Tensor) -> None:
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise ValueError("weight must be a two-dimensional tensor")
    if not weight.is_floating_point():
        raise TypeError("weight must be floating point")


def _validate_inputs(inputs: torch.Tensor, weight: torch.Tensor) -> None:
    _validate_linear_weight(weight)
    if not isinstance(inputs, torch.Tensor) or inputs.ndim < 1:
        raise ValueError("inputs must have at least one dimension")
    if not inputs.is_floating_point():
        raise TypeError("inputs must be floating point")
    if inputs.shape[-1] != weight.shape[1]:
        raise ValueError("input features do not match the local weight")
    if inputs.dtype != weight.dtype or inputs.device != weight.device:
        raise ValueError("inputs and local weight must share dtype and device")


def _validate_collective_sizes(
    values: Sequence[int],
    local_size: int,
    group: object | None,
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError("collective sizes must be a sequence of positive integers")
    sizes = tuple(values)
    if not sizes or any(
        not isinstance(value, int) or isinstance(value, bool) or value < 1
        for value in sizes
    ):
        raise ValueError("collective sizes must be a sequence of positive integers")
    if len(sizes) == 1:
        if sizes[0] != local_size:
            raise ValueError("local shard size does not match collective sizes")
        return sizes
    if not distributed.is_available() or not distributed.is_initialized():
        raise RuntimeError("torch.distributed must be initialized for a multi-member cell")
    world_size = distributed.get_world_size(group)
    rank = distributed.get_rank(group)
    if world_size != len(sizes):
        raise ValueError("collective world size does not match the shard plan")
    if sizes[rank] != local_size:
        raise ValueError("local shard size does not match this rank")
    return sizes


__all__ = [
    "ShardRange",
    "LlamaAttentionShardPlan",
    "balanced_shard",
    "column_parallel_linear",
    "llama_gated_mlp_tensor_parallel",
    "llama_attention_shard_plan",
    "llama_attention_tensor_parallel",
    "llama_decoder_layer_tensor_parallel",
    "row_parallel_linear",
    "shard_column_linear",
    "shard_llama_attention",
    "shard_row_linear",
    "shard_sizes",
    "weighted_shard",
]
