"""Pure prefix-closed planner for exact speculative candidate paths.

The current physical tree executes each leaf from the common parent, repeating
all shared prefixes.  This module turns leaf-only ``candidate_paths`` into a
deterministic trie schedule where:

* every non-empty prefix has one node and one compute step;
* unary chains stay on the same logical lane;
* a fork is emitted only at a node with two or more children;
* all secondary lanes are forked before the primary lane mutates;
* node, leaf, lane and operation IDs are independent of input ordering.

It owns no tensors, sockets or request IDs.  Lane IDs are relative planner IDs
and must be mapped to physical request IDs by a future integration.  The cost
report counts token-step work, not wall time, GPU kernels or WAN latency.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral


UINT32_MAX = (1 << 32) - 1
DEFAULT_MAX_PATHS = 64
DEFAULT_MAX_DEPTH = 64
DEFAULT_MAX_NODES = 4_096
HARD_MAX_PATHS = 4_096
HARD_MAX_DEPTH = 256
HARD_MAX_NODES = 65_536


@dataclass(frozen=True, slots=True)
class PrefixComputeStep:
    sequence_index: int
    node_id: int
    parent_node_id: int
    lane_id: int
    token: int
    prefix: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PrefixForkStep:
    sequence_index: int
    at_node_id: int
    source_lane_id: int
    target_lane_id: int
    target_child_node_id: int
    prefix: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PrefixDivergenceStep:
    sequence_index: int
    node_id: int
    lane_id: int
    prefix: tuple[int, ...]
    child_node_ids: tuple[int, ...]
    primary_child_node_id: int
    fork_target_lane_ids: tuple[int, ...]


PrefixScheduleStep = PrefixComputeStep | PrefixForkStep | PrefixDivergenceStep


@dataclass(frozen=True, slots=True)
class PrefixNode:
    node_id: int
    parent_node_id: int | None
    lane_id: int
    token: int | None
    depth: int
    prefix: tuple[int, ...]
    child_node_ids: tuple[int, ...]
    descendant_leaf_ids: tuple[int, ...]
    terminal: bool


@dataclass(frozen=True, slots=True)
class PrefixLeaf:
    leaf_id: int
    node_id: int
    lane_id: int
    path: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PrefixCostReport:
    """Flat-leaf versus prefix-closed theoretical token-step cost."""

    leaf_count: int
    unique_prefix_node_count: int
    divergence_count: int
    maximum_depth: int
    shared_root_tokens: int
    flat_leaf_token_steps: int
    unique_prefix_token_steps: int
    saved_token_steps: int
    flat_leaf_fork_steps: int
    prefix_closed_fork_steps: int
    prefix_closed_lane_count: int

    @property
    def compute_reduction_ratio(self) -> float:
        if self.flat_leaf_token_steps == 0:
            return 0.0
        return self.saved_token_steps / self.flat_leaf_token_steps


@dataclass(frozen=True, slots=True)
class PrefixClosedSchedule:
    candidate_paths: tuple[tuple[int, ...], ...]
    nodes: tuple[PrefixNode, ...]
    leaves: tuple[PrefixLeaf, ...]
    operations: tuple[PrefixScheduleStep, ...]
    cost: PrefixCostReport

    @property
    def root(self) -> PrefixNode:
        return self.nodes[0]

    def node(self, node_id: int) -> PrefixNode:
        if not isinstance(node_id, int) or isinstance(node_id, bool):
            raise TypeError("node_id must be an integer")
        if not 0 <= node_id < len(self.nodes):
            raise ValueError(f"unknown prefix node {node_id}")
        return self.nodes[node_id]

    def leaf(self, path: Sequence[int]) -> PrefixLeaf:
        normalized = _path(path, maximum_depth=HARD_MAX_DEPTH)
        for leaf in self.leaves:
            if leaf.path == normalized:
                return leaf
        raise ValueError(f"unknown candidate path {normalized}")


def plan_prefix_closed_schedule(
    candidate_paths: Sequence[Sequence[int]],
    *,
    shared_root_tokens: int = 0,
    max_paths: int = DEFAULT_MAX_PATHS,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_nodes: int = DEFAULT_MAX_NODES,
) -> PrefixClosedSchedule:
    """Validate leaf paths and return their deterministic prefix schedule.

    ``shared_root_tokens`` accounts for already-known work prepended to every
    flat leaf, such as MacroWave's pending token.  It affects only the cost
    report; those tokens are outside ``candidate_paths`` and therefore do not
    create schedule nodes.
    """

    path_limit = _bounded_limit(
        "max_paths",
        max_paths,
        hard_maximum=HARD_MAX_PATHS,
    )
    depth_limit = _bounded_limit(
        "max_depth",
        max_depth,
        hard_maximum=HARD_MAX_DEPTH,
    )
    node_limit = _bounded_limit(
        "max_nodes",
        max_nodes,
        hard_maximum=HARD_MAX_NODES,
    )
    root_tokens = _nonnegative_integer(
        "shared_root_tokens",
        shared_root_tokens,
        maximum=UINT32_MAX,
    )
    if isinstance(candidate_paths, (str, bytes, bytearray)) or not isinstance(
        candidate_paths, Sequence
    ):
        raise ValueError("candidate_paths must be a sequence of token paths")
    if not candidate_paths:
        raise ValueError("candidate_paths must not be empty")
    if len(candidate_paths) > path_limit:
        raise ValueError(
            f"candidate_paths contains {len(candidate_paths)} paths, limit is {path_limit}"
        )

    paths = tuple(sorted(_path(path, maximum_depth=depth_limit) for path in candidate_paths))
    if len(set(paths)) != len(paths):
        raise ValueError("candidate_paths must not contain duplicates")
    for index, path in enumerate(paths[:-1]):
        successor = paths[index + 1]
        if len(path) < len(successor) and successor[: len(path)] == path:
            raise ValueError("a candidate path cannot be a prefix of another path")

    prefix_set: set[tuple[int, ...]] = {()}
    for path in paths:
        prefix_set.update(path[:depth] for depth in range(1, len(path) + 1))
    if len(prefix_set) > node_limit:
        raise ValueError(
            f"candidate trie contains {len(prefix_set)} nodes, limit is {node_limit}"
        )

    ordered_prefixes = tuple(sorted(prefix_set))
    node_id_by_prefix = {
        prefix: node_id for node_id, prefix in enumerate(ordered_prefixes)
    }
    children_by_prefix: dict[tuple[int, ...], list[tuple[int, ...]]] = {
        prefix: [] for prefix in ordered_prefixes
    }
    for prefix in ordered_prefixes[1:]:
        children_by_prefix[prefix[:-1]].append(prefix)
    for children in children_by_prefix.values():
        children.sort(key=lambda prefix: prefix[-1])

    leaf_id_by_path = {path: leaf_id for leaf_id, path in enumerate(paths)}
    node_lane: dict[tuple[int, ...], int] = {(): 0}
    operations: list[PrefixScheduleStep] = []
    next_lane_id = 1

    def append_operation(operation_type, **values) -> None:
        operations.append(operation_type(sequence_index=len(operations), **values))

    def visit(prefix: tuple[int, ...]) -> None:
        nonlocal next_lane_id
        children = children_by_prefix[prefix]
        if not children:
            return
        parent_node_id = node_id_by_prefix[prefix]
        parent_lane = node_lane[prefix]
        primary = children[0]
        node_lane[primary] = parent_lane

        fork_lanes: list[int] = []
        if len(children) > 1:
            secondary_lanes: dict[tuple[int, ...], int] = {}
            for child in children[1:]:
                lane_id = next_lane_id
                next_lane_id += 1
                secondary_lanes[child] = lane_id
                node_lane[child] = lane_id
                fork_lanes.append(lane_id)
            append_operation(
                PrefixDivergenceStep,
                node_id=parent_node_id,
                lane_id=parent_lane,
                prefix=prefix,
                child_node_ids=tuple(node_id_by_prefix[child] for child in children),
                primary_child_node_id=node_id_by_prefix[primary],
                fork_target_lane_ids=tuple(fork_lanes),
            )
            # Clone every secondary while the source lane still represents the
            # divergence prefix. The primary compute is emitted afterwards.
            for child in children[1:]:
                append_operation(
                    PrefixForkStep,
                    at_node_id=parent_node_id,
                    source_lane_id=parent_lane,
                    target_lane_id=secondary_lanes[child],
                    target_child_node_id=node_id_by_prefix[child],
                    prefix=prefix,
                )

        for child in children:
            append_operation(
                PrefixComputeStep,
                node_id=node_id_by_prefix[child],
                parent_node_id=parent_node_id,
                lane_id=node_lane[child],
                token=child[-1],
                prefix=child,
            )
            visit(child)

    visit(())

    terminal_paths = set(paths)
    nodes = tuple(
        PrefixNode(
            node_id=node_id_by_prefix[prefix],
            parent_node_id=(
                None if not prefix else node_id_by_prefix[prefix[:-1]]
            ),
            lane_id=node_lane[prefix],
            token=None if not prefix else prefix[-1],
            depth=len(prefix),
            prefix=prefix,
            child_node_ids=tuple(
                node_id_by_prefix[child] for child in children_by_prefix[prefix]
            ),
            descendant_leaf_ids=tuple(
                leaf_id
                for path, leaf_id in leaf_id_by_path.items()
                if len(prefix) <= len(path) and path[: len(prefix)] == prefix
            ),
            terminal=prefix in terminal_paths,
        )
        for prefix in ordered_prefixes
    )
    leaves = tuple(
        PrefixLeaf(
            leaf_id=leaf_id,
            node_id=node_id_by_prefix[path],
            lane_id=node_lane[path],
            path=path,
        )
        for path, leaf_id in leaf_id_by_path.items()
    )
    compute_steps = sum(
        isinstance(operation, PrefixComputeStep) for operation in operations
    )
    fork_steps = sum(
        isinstance(operation, PrefixForkStep) for operation in operations
    )
    divergence_steps = sum(
        isinstance(operation, PrefixDivergenceStep) for operation in operations
    )
    if compute_steps != len(prefix_set) - 1:
        raise RuntimeError("prefix planner did not compute every unique prefix exactly once")

    flat_token_steps = sum(len(path) + root_tokens for path in paths)
    prefix_token_steps = compute_steps + root_tokens
    saved_token_steps = flat_token_steps - prefix_token_steps
    if saved_token_steps < 0:
        raise RuntimeError("prefix planner produced a negative theoretical saving")
    cost = PrefixCostReport(
        leaf_count=len(paths),
        unique_prefix_node_count=compute_steps,
        divergence_count=divergence_steps,
        maximum_depth=max(len(path) for path in paths),
        shared_root_tokens=root_tokens,
        flat_leaf_token_steps=flat_token_steps,
        unique_prefix_token_steps=prefix_token_steps,
        saved_token_steps=saved_token_steps,
        flat_leaf_fork_steps=len(paths),
        prefix_closed_fork_steps=fork_steps,
        prefix_closed_lane_count=next_lane_id,
    )
    return PrefixClosedSchedule(
        candidate_paths=paths,
        nodes=nodes,
        leaves=leaves,
        operations=tuple(operations),
        cost=cost,
    )


def _path(values: Sequence[int], *, maximum_depth: int) -> tuple[int, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise ValueError("each candidate path must be a sequence of token ids")
    if not values:
        raise ValueError("candidate paths must not be empty")
    if len(values) > maximum_depth:
        raise ValueError(
            f"candidate path depth {len(values)} exceeds limit {maximum_depth}"
        )
    return tuple(
        _nonnegative_integer("candidate token", token, maximum=UINT32_MAX)
        for token in values
    )


def _bounded_limit(name: str, value: object, *, hard_maximum: int) -> int:
    normalized = _nonnegative_integer(name, value, maximum=hard_maximum)
    if normalized < 1:
        raise ValueError(f"{name} must be positive")
    return normalized


def _nonnegative_integer(name: str, value: object, *, maximum: int) -> int:
    if (
        not isinstance(value, Integral)
        or isinstance(value, bool)
        or int(value) < 0
        or int(value) > maximum
    ):
        raise ValueError(f"{name} must be an integer in [0, {maximum}]")
    return int(value)


__all__ = [
    "DEFAULT_MAX_DEPTH",
    "DEFAULT_MAX_NODES",
    "DEFAULT_MAX_PATHS",
    "PrefixClosedSchedule",
    "PrefixComputeStep",
    "PrefixCostReport",
    "PrefixDivergenceStep",
    "PrefixForkStep",
    "PrefixLeaf",
    "PrefixNode",
    "PrefixScheduleStep",
    "plan_prefix_closed_schedule",
]
