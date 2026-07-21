"""Exact physical KV projection for a prefix-closed COW tree wave.

The projector mirrors the block semantics sealed by
``HFPagedStageCache.fork`` without importing Torch or Transformers:

* the real parent remains live and is never mutated;
* one carrier is forked from that intact parent;
* complete blocks are shared by reference;
* an incomplete tail is copied into one private block at every fork;
* the pending target token is appended once to the carrier;
* candidate prefixes are appended once according to
  :mod:`distributed_runtime.prefix_closed_schedule`;
* nested forks happen only at trie divergences.

All live tree lanes are retained because exact verification still needs every
leaf result.  Consequently occupancy is monotonic during this projection and
``peak_live_blocks`` is the final unique physical occupancy, including the
intact parent.  This module owns no tensors, cache blocks or request IDs and is
not an execution engine.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral

from .prefix_closed_schedule import (
    DEFAULT_MAX_DEPTH,
    DEFAULT_MAX_NODES,
    DEFAULT_MAX_PATHS,
    PrefixComputeStep,
    PrefixDivergenceStep,
    PrefixForkStep,
    plan_prefix_closed_schedule,
)


UINT32_MAX = (1 << 32) - 1
UINT64_MAX = (1 << 64) - 1


@dataclass(frozen=True, slots=True)
class CowKVPhysicalCost:
    """Physical occupancy for one execution strategy.

    ``incremental_*`` excludes the already-live parent blocks.
    ``peak_live_*`` includes them.  Fork copies and append allocations are
    disjoint and sum exactly to ``incremental_blocks``.
    """

    incremental_blocks: int
    incremental_bytes: int
    peak_live_blocks: int
    peak_live_bytes: int
    copied_tail_blocks: int
    append_allocated_blocks: int
    fork_count: int
    leaf_lane_count: int

    def __post_init__(self) -> None:
        if self.incremental_blocks != (
            self.copied_tail_blocks + self.append_allocated_blocks
        ):
            raise ValueError(
                "incremental_blocks must equal copied tails plus append allocations"
            )


@dataclass(frozen=True, slots=True)
class PrefixClosedCowProjection:
    """Exact prefix-closed projection and its flat-leaf baseline."""

    parent_tokens: int
    block_tokens: int
    bytes_per_block: int
    pending_token: int
    candidate_paths: tuple[tuple[int, ...], ...]
    parent_blocks: int
    parent_bytes: int
    prefix_closed: CowKVPhysicalCost
    naive_flat: CowKVPhysicalCost
    saved_incremental_blocks: int
    saved_incremental_bytes: int

    @property
    def physical_block_reduction_ratio(self) -> float:
        """Fraction of the flat wave's incremental blocks that are avoided."""

        if self.naive_flat.incremental_blocks == 0:
            return 0.0
        return self.saved_incremental_blocks / self.naive_flat.incremental_blocks


def project_prefix_closed_cow_kv(
    *,
    parent_tokens: int,
    block_tokens: int,
    bytes_per_block: int,
    pending_token: int,
    candidate_paths: Sequence[Sequence[int]],
    max_paths: int = DEFAULT_MAX_PATHS,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_nodes: int = DEFAULT_MAX_NODES,
) -> PrefixClosedCowProjection:
    """Project exact unique physical KV blocks for one tree wave.

    ``parent_tokens`` is the KV length before the visible pending target token,
    matching ``PhysicalTreeWave.base_kv_tokens``.  ``pending_token`` is a token
    ID and contributes exactly one append before the candidate paths.  Its
    value does not affect occupancy, but validating and retaining it prevents a
    caller from accidentally projecting a wave with different root semantics.

    The flat baseline forks every leaf directly from the intact parent and
    appends ``(pending_token, *path)`` independently.  Both strategies retain
    the real parent and every leaf until resolution, so their peaks are
    directly comparable.
    """

    parent_length = _bounded_integer(
        "parent_tokens", parent_tokens, minimum=0, maximum=UINT32_MAX
    )
    block_size = _bounded_integer(
        "block_tokens", block_tokens, minimum=1, maximum=UINT32_MAX
    )
    block_bytes = _bounded_integer(
        "bytes_per_block", bytes_per_block, minimum=1, maximum=UINT64_MAX
    )
    pending = _bounded_integer(
        "pending_token", pending_token, minimum=0, maximum=UINT32_MAX
    )

    schedule = plan_prefix_closed_schedule(
        candidate_paths,
        shared_root_tokens=1,
        max_paths=max_paths,
        max_depth=max_depth,
        max_nodes=max_nodes,
    )
    longest_leaf = max(len(path) for path in schedule.candidate_paths)
    if parent_length + 1 + longest_leaf > UINT32_MAX:
        raise ValueError(
            "parent, pending token and longest candidate path exceed the uint32 "
            "sequence-length limit"
        )

    parent_blocks = _ceil_blocks(parent_length, block_size)

    # Lane 0 becomes the carrier.  Forking copies only an incomplete parent
    # tail; appending the pending target token may then fill it or allocate the
    # first block after a complete boundary.
    lane_lengths: dict[int, int] = {0: parent_length}
    copied_tail_blocks = int(parent_length % block_size != 0)
    append_allocated_blocks = _append_block_delta(
        parent_length, 1, block_size
    )
    lane_lengths[0] = parent_length + 1

    for operation in schedule.operations:
        if isinstance(operation, PrefixDivergenceStep):
            _require_lane_prefix_length(
                lane_lengths,
                operation.lane_id,
                parent_length,
                operation.prefix,
            )
            continue

        if isinstance(operation, PrefixForkStep):
            source_length = _require_lane_prefix_length(
                lane_lengths,
                operation.source_lane_id,
                parent_length,
                operation.prefix,
            )
            if operation.target_lane_id in lane_lengths:
                raise RuntimeError(
                    f"prefix schedule reuses lane {operation.target_lane_id}"
                )
            copied_tail_blocks += int(source_length % block_size != 0)
            lane_lengths[operation.target_lane_id] = source_length
            continue

        if isinstance(operation, PrefixComputeStep):
            expected_before = parent_length + len(operation.prefix)
            try:
                old_length = lane_lengths[operation.lane_id]
            except KeyError as exc:
                raise RuntimeError(
                    f"prefix schedule computes on unknown lane {operation.lane_id}"
                ) from exc
            if old_length != expected_before:
                raise RuntimeError(
                    "prefix schedule lane length does not match compute prefix: "
                    f"lane {operation.lane_id} has {old_length}, expected "
                    f"{expected_before} before {operation.prefix!r}"
                )
            append_allocated_blocks += _append_block_delta(
                old_length, 1, block_size
            )
            lane_lengths[operation.lane_id] = old_length + 1
            continue

        raise TypeError(f"unknown prefix schedule operation {type(operation)!r}")

    if len(lane_lengths) != schedule.cost.prefix_closed_lane_count:
        raise RuntimeError("prefix schedule lane count changed during COW projection")
    for leaf in schedule.leaves:
        expected_length = parent_length + 1 + len(leaf.path)
        if lane_lengths.get(leaf.lane_id) != expected_length:
            raise RuntimeError(
                f"leaf {leaf.path!r} did not finish at its exact projected length"
            )

    prefix_incremental = copied_tail_blocks + append_allocated_blocks
    prefix_cost = _cost(
        parent_blocks=parent_blocks,
        incremental_blocks=prefix_incremental,
        bytes_per_block=block_bytes,
        copied_tail_blocks=copied_tail_blocks,
        append_allocated_blocks=append_allocated_blocks,
        fork_count=1 + schedule.cost.prefix_closed_fork_steps,
        leaf_lane_count=schedule.cost.prefix_closed_lane_count,
    )

    # Every flat leaf starts from the same immutable parent.  Its fork copies
    # the parent's partial tail (if present), then its full verification input
    # grows independently.
    naive_copied_tail_blocks = (
        len(schedule.candidate_paths) if parent_length % block_size else 0
    )
    naive_append_allocated_blocks = sum(
        _append_block_delta(parent_length, 1 + len(path), block_size)
        for path in schedule.candidate_paths
    )
    naive_incremental = (
        naive_copied_tail_blocks + naive_append_allocated_blocks
    )
    naive_cost = _cost(
        parent_blocks=parent_blocks,
        incremental_blocks=naive_incremental,
        bytes_per_block=block_bytes,
        copied_tail_blocks=naive_copied_tail_blocks,
        append_allocated_blocks=naive_append_allocated_blocks,
        fork_count=len(schedule.candidate_paths),
        leaf_lane_count=len(schedule.candidate_paths),
    )

    saved_blocks = naive_incremental - prefix_incremental
    if saved_blocks < 0:
        raise RuntimeError(
            "prefix-closed COW projection exceeded the exact flat baseline"
        )
    saved_bytes = _checked_bytes(saved_blocks, block_bytes, "saved bytes")
    return PrefixClosedCowProjection(
        parent_tokens=parent_length,
        block_tokens=block_size,
        bytes_per_block=block_bytes,
        pending_token=pending,
        candidate_paths=schedule.candidate_paths,
        parent_blocks=parent_blocks,
        parent_bytes=_checked_bytes(parent_blocks, block_bytes, "parent bytes"),
        prefix_closed=prefix_cost,
        naive_flat=naive_cost,
        saved_incremental_blocks=saved_blocks,
        saved_incremental_bytes=saved_bytes,
    )


def _require_lane_prefix_length(
    lane_lengths: dict[int, int],
    lane_id: int,
    parent_tokens: int,
    prefix: tuple[int, ...],
) -> int:
    try:
        actual = lane_lengths[lane_id]
    except KeyError as exc:
        raise RuntimeError(f"prefix schedule forks unknown lane {lane_id}") from exc
    expected = parent_tokens + 1 + len(prefix)
    if actual != expected:
        raise RuntimeError(
            "prefix schedule lane length does not match divergence prefix: "
            f"lane {lane_id} has {actual}, expected {expected} for {prefix!r}"
        )
    return actual


def _cost(
    *,
    parent_blocks: int,
    incremental_blocks: int,
    bytes_per_block: int,
    copied_tail_blocks: int,
    append_allocated_blocks: int,
    fork_count: int,
    leaf_lane_count: int,
) -> CowKVPhysicalCost:
    peak_blocks = parent_blocks + incremental_blocks
    return CowKVPhysicalCost(
        incremental_blocks=incremental_blocks,
        incremental_bytes=_checked_bytes(
            incremental_blocks, bytes_per_block, "incremental bytes"
        ),
        peak_live_blocks=peak_blocks,
        peak_live_bytes=_checked_bytes(
            peak_blocks, bytes_per_block, "peak live bytes"
        ),
        copied_tail_blocks=copied_tail_blocks,
        append_allocated_blocks=append_allocated_blocks,
        fork_count=fork_count,
        leaf_lane_count=leaf_lane_count,
    )


def _append_block_delta(length: int, token_count: int, block_tokens: int) -> int:
    return _ceil_blocks(length + token_count, block_tokens) - _ceil_blocks(
        length, block_tokens
    )


def _ceil_blocks(token_count: int, block_tokens: int) -> int:
    if token_count == 0:
        return 0
    return (token_count + block_tokens - 1) // block_tokens


def _checked_bytes(blocks: int, bytes_per_block: int, name: str) -> int:
    value = blocks * bytes_per_block
    if value > UINT64_MAX:
        raise ValueError(f"{name} exceeds the uint64 byte-accounting limit")
    return value


def _bounded_integer(
    name: str,
    value: object,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if (
        not isinstance(value, Integral)
        or isinstance(value, bool)
        or int(value) < minimum
        or int(value) > maximum
    ):
        raise ValueError(
            f"{name} must be an integer in [{minimum}, {maximum}]"
        )
    return int(value)


__all__ = [
    "CowKVPhysicalCost",
    "PrefixClosedCowProjection",
    "project_prefix_closed_cow_kv",
]
