"""Deterministic compressed-trie schedule for exact speculative paths.

This module is deliberately pure: it owns no tensors, request IDs, sockets or
threads.  It turns candidate token paths into a schedule that a future runtime
can map to physical KV-cache lanes.

Compared with a node-at-a-time prefix schedule, it adds two useful invariants:

* every maximal unary chain is one :class:`PrefixComputeSegment`;
* for a breadth-first frontier, every required fork is emitted before any
  compute segment mutates a lane in that frontier.

Segments in the same dependency frontier are kept contiguous when they share
``(start_depth, token_count)``.  Such a group is only a structural batching
opportunity; it is not a claim that a model backend can already execute it in
one kernel or WAN message.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import groupby
from numbers import Integral


UINT32_MAX = (1 << 32) - 1
DEFAULT_MAX_PATHS = 64
DEFAULT_MAX_DEPTH = 64
DEFAULT_MAX_NODES = 4_096
HARD_MAX_PATHS = 4_096
HARD_MAX_DEPTH = 256
HARD_MAX_NODES = 65_536
_MISSING = object()


@dataclass(frozen=True, slots=True)
class PrefixSegmentNode:
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
class PrefixSegmentLeaf:
    leaf_id: int
    node_id: int
    lane_id: int
    candidate_path: tuple[int, ...]
    full_path: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PrefixSegmentFork:
    sequence_index: int
    fork_id: int
    frontier_index: int
    divergence_node_id: int
    source_lane_id: int
    target_lane_id: int
    target_child_node_id: int
    prefix: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PrefixComputeSegment:
    sequence_index: int
    segment_id: int
    frontier_index: int
    compatible_group_id: int
    lane_id: int
    parent_node_id: int
    node_ids: tuple[int, ...]
    tokens: tuple[int, ...]
    start_depth: int
    end_depth: int

    @property
    def token_count(self) -> int:
        return len(self.tokens)

    @property
    def end_node_id(self) -> int:
        return self.node_ids[-1]


PrefixSegmentOperation = PrefixSegmentFork | PrefixComputeSegment


@dataclass(frozen=True, slots=True)
class PrefixCompatibleFrontierGroup:
    """Contiguous segments with equal input depth and segment length."""

    group_id: int
    frontier_index: int
    start_depth: int
    token_count: int
    segment_ids: tuple[int, ...]
    first_operation_index: int
    last_operation_index: int

    @property
    def segment_count(self) -> int:
        return len(self.segment_ids)


@dataclass(frozen=True, slots=True)
class PrefixSegmentCostReport:
    leaf_count: int
    maximum_depth: int
    shared_prefix_token_count: int
    flat_token_steps: int
    unique_token_steps: int
    saved_token_steps: int
    node_calls: int
    segment_calls: int
    # Divergence forks inside the trie.  A physical runtime additionally forks
    # one carrier from the immutable real parent before executing this plan.
    fork_count: int
    lane_count: int
    frontier_count: int
    compatible_frontier_groups: tuple[PrefixCompatibleFrontierGroup, ...]

    @property
    def compute_reduction_ratio(self) -> float:
        if self.flat_token_steps == 0:
            return 0.0
        return self.saved_token_steps / self.flat_token_steps

    @property
    def call_reduction_ratio(self) -> float:
        if self.node_calls == 0:
            return 0.0
        return (self.node_calls - self.segment_calls) / self.node_calls

    @property
    def compatible_frontier_group_count(self) -> int:
        return len(self.compatible_frontier_groups)

    @property
    def carrier_fork_count(self) -> int:
        return 1

    @property
    def total_physical_fork_count(self) -> int:
        return self.carrier_fork_count + self.fork_count


@dataclass(frozen=True, slots=True)
class PrefixSegmentSchedule:
    shared_prefix_tokens: tuple[int, ...]
    candidate_paths: tuple[tuple[int, ...], ...]
    nodes: tuple[PrefixSegmentNode, ...]
    leaves: tuple[PrefixSegmentLeaf, ...]
    forks: tuple[PrefixSegmentFork, ...]
    segments: tuple[PrefixComputeSegment, ...]
    operations: tuple[PrefixSegmentOperation, ...]
    compatible_frontier_groups: tuple[PrefixCompatibleFrontierGroup, ...]
    cost: PrefixSegmentCostReport

    @property
    def root(self) -> PrefixSegmentNode:
        return self.nodes[0]

    def node(self, node_id: int) -> PrefixSegmentNode:
        if not isinstance(node_id, int) or isinstance(node_id, bool):
            raise TypeError("node_id must be an integer")
        if not 0 <= node_id < len(self.nodes):
            raise ValueError(f"unknown prefix-segment node {node_id}")
        return self.nodes[node_id]

    def leaf(self, candidate_path: Sequence[int]) -> PrefixSegmentLeaf:
        normalized = _token_path(
            "candidate path",
            candidate_path,
            maximum_depth=HARD_MAX_DEPTH,
            allow_empty=False,
        )
        for leaf in self.leaves:
            if leaf.candidate_path == normalized:
                return leaf
        raise ValueError(f"unknown candidate path {normalized}")

    def reconstruct_full_leaf_paths(self) -> tuple[tuple[int, ...], ...]:
        """Rebuild every full leaf by following node parent links."""

        reconstructed: list[tuple[int, ...]] = []
        for leaf in self.leaves:
            tokens: list[int] = []
            node = self.node(leaf.node_id)
            while node.parent_node_id is not None:
                assert node.token is not None
                tokens.append(node.token)
                node = self.node(node.parent_node_id)
            reconstructed.append(tuple(reversed(tokens)))
        return tuple(reconstructed)

    def reconstruct_candidate_paths(self) -> tuple[tuple[int, ...], ...]:
        shared_count = len(self.shared_prefix_tokens)
        full_paths = self.reconstruct_full_leaf_paths()
        for full_path in full_paths:
            if full_path[:shared_count] != self.shared_prefix_tokens:
                raise RuntimeError("leaf reconstruction lost the shared prefix")
        return tuple(full_path[shared_count:] for full_path in full_paths)


@dataclass(frozen=True, slots=True)
class _SegmentSpec:
    frontier_index: int
    lane_id: int
    parent_prefix: tuple[int, ...]
    prefixes: tuple[tuple[int, ...], ...]

    @property
    def start_depth(self) -> int:
        return len(self.parent_prefix)

    @property
    def token_count(self) -> int:
        return len(self.prefixes)


def plan_prefix_segment_schedule(
    candidate_paths: Sequence[Sequence[int]],
    *,
    shared_prefix_tokens: Sequence[int] = (),
    max_paths: int = DEFAULT_MAX_PATHS,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_nodes: int = DEFAULT_MAX_NODES,
) -> PrefixSegmentSchedule:
    """Return a bounded, canonical compressed-trie execution schedule.

    ``candidate_paths`` are suffixes after ``shared_prefix_tokens``.  The
    shared prefix is included in both the node graph and compute segments, so a
    pending token such as ``(pending_token,)`` is computed once before lanes
    diverge.  ``max_depth`` bounds the complete shared-plus-candidate path.
    """

    path_limit = _bounded_limit("max_paths", max_paths, HARD_MAX_PATHS)
    depth_limit = _bounded_limit("max_depth", max_depth, HARD_MAX_DEPTH)
    node_limit = _bounded_limit("max_nodes", max_nodes, HARD_MAX_NODES)
    shared = _token_path(
        "shared_prefix_tokens",
        shared_prefix_tokens,
        maximum_depth=depth_limit,
        allow_empty=True,
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

    remaining_depth = depth_limit - len(shared)
    paths = tuple(
        sorted(
            _token_path(
                "candidate path",
                path,
                maximum_depth=remaining_depth,
                allow_empty=False,
            )
            for path in candidate_paths
        )
    )
    if len(set(paths)) != len(paths):
        raise ValueError("candidate_paths must not contain duplicates")
    for index, path in enumerate(paths[:-1]):
        successor = paths[index + 1]
        if len(path) < len(successor) and successor[: len(path)] == path:
            raise ValueError("a candidate path cannot be a prefix of another path")

    full_paths = tuple(shared + path for path in paths)
    prefix_set: set[tuple[int, ...]] = {()}
    for full_path in full_paths:
        prefix_set.update(
            full_path[:depth] for depth in range(1, len(full_path) + 1)
        )
    if len(prefix_set) > node_limit:
        raise ValueError(
            f"candidate trie contains {len(prefix_set)} nodes, limit is {node_limit}"
        )

    # Depth-first IDs would also be canonical.  Breadth-first IDs make parent
    # ordering and frontier diagnostics more obvious while staying independent
    # of input order.
    ordered_prefixes = ((),) + tuple(
        sorted((prefix for prefix in prefix_set if prefix), key=lambda item: (len(item), item))
    )
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

    leaf_id_by_full_path = {
        full_path: leaf_id for leaf_id, full_path in enumerate(full_paths)
    }
    lane_by_prefix: dict[tuple[int, ...], int] = {(): 0}
    operations: list[PrefixSegmentOperation] = []
    forks: list[PrefixSegmentFork] = []
    segments: list[PrefixComputeSegment] = []
    groups: list[PrefixCompatibleFrontierGroup] = []
    next_lane_id = 1
    frontier_index = 0
    frontier: tuple[tuple[int, ...], ...] = ((),)

    while frontier:
        segment_specs: list[_SegmentSpec] = []

        # Phase 1: assign outgoing lanes and emit every clone for the complete
        # dependency frontier.  No compute operation is allowed in this phase.
        for parent_prefix in sorted(frontier):
            children = children_by_prefix[parent_prefix]
            if not children:
                raise RuntimeError("leaf node unexpectedly entered a compute frontier")
            source_lane = lane_by_prefix[parent_prefix]
            primary = children[0]
            lane_by_prefix[primary] = source_lane
            for child in children[1:]:
                target_lane = next_lane_id
                next_lane_id += 1
                lane_by_prefix[child] = target_lane
                fork = PrefixSegmentFork(
                    sequence_index=len(operations),
                    fork_id=len(forks),
                    frontier_index=frontier_index,
                    divergence_node_id=node_id_by_prefix[parent_prefix],
                    source_lane_id=source_lane,
                    target_lane_id=target_lane,
                    target_child_node_id=node_id_by_prefix[child],
                    prefix=parent_prefix,
                )
                forks.append(fork)
                operations.append(fork)

        # Phase 2: discover maximal unary chains.  Lane assignment here is a
        # pure planner update; actual compute operations are emitted only after
        # every frontier fork above.
        for parent_prefix in sorted(frontier):
            for first_child in children_by_prefix[parent_prefix]:
                lane_id = lane_by_prefix[first_child]
                chain = [first_child]
                cursor = first_child
                while len(children_by_prefix[cursor]) == 1:
                    cursor = children_by_prefix[cursor][0]
                    lane_by_prefix[cursor] = lane_id
                    chain.append(cursor)
                segment_specs.append(
                    _SegmentSpec(
                        frontier_index=frontier_index,
                        lane_id=lane_id,
                        parent_prefix=parent_prefix,
                        prefixes=tuple(chain),
                    )
                )

        # Phase 3: compatible segments stay adjacent.  Dependency frontiers
        # are never mixed even when their shapes happen to match.
        segment_specs.sort(
            key=lambda spec: (
                spec.start_depth,
                spec.token_count,
                spec.prefixes[0],
            )
        )
        next_frontier: list[tuple[int, ...]] = []
        for compatibility, compatible_specs_iterator in groupby(
            segment_specs,
            key=lambda spec: (spec.start_depth, spec.token_count),
        ):
            compatible_specs = tuple(compatible_specs_iterator)
            group_id = len(groups)
            first_operation_index = len(operations)
            group_segment_ids: list[int] = []
            for spec in compatible_specs:
                segment = PrefixComputeSegment(
                    sequence_index=len(operations),
                    segment_id=len(segments),
                    frontier_index=frontier_index,
                    compatible_group_id=group_id,
                    lane_id=spec.lane_id,
                    parent_node_id=node_id_by_prefix[spec.parent_prefix],
                    node_ids=tuple(
                        node_id_by_prefix[prefix] for prefix in spec.prefixes
                    ),
                    tokens=tuple(prefix[-1] for prefix in spec.prefixes),
                    start_depth=spec.start_depth,
                    end_depth=spec.start_depth + spec.token_count,
                )
                segments.append(segment)
                operations.append(segment)
                group_segment_ids.append(segment.segment_id)
                endpoint = spec.prefixes[-1]
                if len(children_by_prefix[endpoint]) > 1:
                    next_frontier.append(endpoint)
            groups.append(
                PrefixCompatibleFrontierGroup(
                    group_id=group_id,
                    frontier_index=frontier_index,
                    start_depth=compatibility[0],
                    token_count=compatibility[1],
                    segment_ids=tuple(group_segment_ids),
                    first_operation_index=first_operation_index,
                    last_operation_index=len(operations) - 1,
                )
            )

        frontier = tuple(sorted(next_frontier))
        frontier_index += 1

    if len(lane_by_prefix) != len(prefix_set):
        raise RuntimeError("prefix-segment planner did not assign every trie node")
    computed_node_ids = tuple(
        node_id for segment in segments for node_id in segment.node_ids
    )
    if len(computed_node_ids) != len(prefix_set) - 1 or len(set(computed_node_ids)) != len(
        computed_node_ids
    ):
        raise RuntimeError("prefix-segment planner did not compute each node once")

    descendants_by_prefix: dict[tuple[int, ...], tuple[int, ...]] = {}
    for prefix in reversed(ordered_prefixes):
        descendants: list[int] = []
        leaf_id = leaf_id_by_full_path.get(prefix)
        if leaf_id is not None:
            descendants.append(leaf_id)
        for child in children_by_prefix[prefix]:
            descendants.extend(descendants_by_prefix[child])
        descendants_by_prefix[prefix] = tuple(sorted(descendants))

    terminal_paths = set(full_paths)
    nodes = tuple(
        PrefixSegmentNode(
            node_id=node_id_by_prefix[prefix],
            parent_node_id=(
                None if not prefix else node_id_by_prefix[prefix[:-1]]
            ),
            lane_id=lane_by_prefix[prefix],
            token=None if not prefix else prefix[-1],
            depth=len(prefix),
            prefix=prefix,
            child_node_ids=tuple(
                node_id_by_prefix[child] for child in children_by_prefix[prefix]
            ),
            descendant_leaf_ids=descendants_by_prefix[prefix],
            terminal=prefix in terminal_paths,
        )
        for prefix in ordered_prefixes
    )
    leaves = tuple(
        PrefixSegmentLeaf(
            leaf_id=leaf_id,
            node_id=node_id_by_prefix[full_path],
            lane_id=lane_by_prefix[full_path],
            candidate_path=paths[leaf_id],
            full_path=full_path,
        )
        for leaf_id, full_path in enumerate(full_paths)
    )

    flat_token_steps = sum(len(full_path) for full_path in full_paths)
    unique_token_steps = len(prefix_set) - 1
    saved_token_steps = flat_token_steps - unique_token_steps
    if saved_token_steps < 0:
        raise RuntimeError("prefix-segment planner produced negative compute saving")
    cost = PrefixSegmentCostReport(
        leaf_count=len(paths),
        maximum_depth=max(len(full_path) for full_path in full_paths),
        shared_prefix_token_count=len(shared),
        flat_token_steps=flat_token_steps,
        unique_token_steps=unique_token_steps,
        saved_token_steps=saved_token_steps,
        node_calls=unique_token_steps,
        segment_calls=len(segments),
        fork_count=len(forks),
        lane_count=next_lane_id,
        frontier_count=frontier_index,
        compatible_frontier_groups=tuple(groups),
    )
    return PrefixSegmentSchedule(
        shared_prefix_tokens=shared,
        candidate_paths=paths,
        nodes=nodes,
        leaves=leaves,
        forks=tuple(forks),
        segments=tuple(segments),
        operations=tuple(operations),
        compatible_frontier_groups=tuple(groups),
        cost=cost,
    )


def target_argmax_by_prefix_from_node_outputs(
    schedule: PrefixSegmentSchedule,
    target_argmax_by_node: Mapping[int, int],
    *,
    root_target_argmax: int | object = _MISSING,
) -> dict[tuple[int, ...], int]:
    """Map one exact target output per computed node to relative prefixes.

    Transformer output after the final shared-prefix token predicts relative
    prefix ``()``.  Output after every candidate-trie node predicts the
    candidate prefix ending at that node.  Outputs before the final shared
    token are validated as part of their segment but intentionally do not enter
    the returned MacroWave map.

    With an empty shared prefix there is no compute node that can predict
    ``()``; callers must then supply ``root_target_argmax`` from the already
    materialised parent state.
    """

    _require_schedule(schedule)
    if not isinstance(target_argmax_by_node, Mapping):
        raise ValueError("target_argmax_by_node must be a mapping")
    normalized: dict[int, int] = {}
    for raw_node_id, raw_target in target_argmax_by_node.items():
        node_id = _bounded_identifier("target node id", raw_node_id, len(schedule.nodes))
        if node_id == 0:
            raise ValueError("the virtual root has no compute output")
        target = _token(raw_target)
        existing = normalized.get(node_id)
        if existing is not None and existing != target:
            raise ValueError(
                f"contradictory target argmax for node {node_id}: {existing} != {target}"
            )
        normalized[node_id] = target

    expected_node_ids = set(range(1, len(schedule.nodes)))
    actual_node_ids = set(normalized)
    if actual_node_ids != expected_node_ids:
        missing = sorted(expected_node_ids - actual_node_ids)
        extra = sorted(actual_node_ids - expected_node_ids)
        raise ValueError(
            "node targets do not cover the schedule; "
            f"missing={missing!r}, extra={extra!r}"
        )

    shared_count = len(schedule.shared_prefix_tokens)
    relative_targets: dict[tuple[int, ...], int] = {}
    if shared_count == 0:
        if root_target_argmax is _MISSING:
            raise ValueError(
                "root_target_argmax is required when shared_prefix_tokens is empty"
            )
        relative_targets[()] = _token(root_target_argmax)
    elif root_target_argmax is not _MISSING:
        raise ValueError(
            "root_target_argmax must be omitted when a shared prefix predicts ()"
        )

    for node in schedule.nodes[1:]:
        if node.depth < shared_count:
            continue
        prefix = node.prefix[shared_count:]
        target = normalized[node.node_id]
        existing = relative_targets.get(prefix)
        if existing is not None and existing != target:
            raise ValueError(
                "contradictory target argmax for relative prefix "
                f"{prefix!r}: {existing} != {target}"
            )
        relative_targets[prefix] = target

    expected_prefixes = _expected_relative_target_prefixes(schedule)
    if set(relative_targets) != expected_prefixes:
        missing = sorted(
            expected_prefixes - set(relative_targets), key=lambda item: (len(item), item)
        )
        extra = sorted(
            set(relative_targets) - expected_prefixes, key=lambda item: (len(item), item)
        )
        raise ValueError(
            "node targets do not cover candidate prefixes; "
            f"missing={missing!r}, extra={extra!r}"
        )
    return {
        prefix: relative_targets[prefix]
        for prefix in sorted(relative_targets, key=lambda item: (len(item), item))
    }


def target_argmax_by_prefix_from_segment_outputs(
    schedule: PrefixSegmentSchedule,
    target_argmax_by_segment: Mapping[int, Sequence[int]],
    *,
    root_target_argmax: int | object = _MISSING,
) -> dict[tuple[int, ...], int]:
    """Expand exact per-segment output vectors and build the relative map."""

    _require_schedule(schedule)
    if not isinstance(target_argmax_by_segment, Mapping):
        raise ValueError("target_argmax_by_segment must be a mapping")
    normalized: dict[int, tuple[int, ...]] = {}
    for raw_segment_id, raw_targets in target_argmax_by_segment.items():
        segment_id = _bounded_identifier(
            "target segment id", raw_segment_id, len(schedule.segments)
        )
        targets = _token_path(
            "segment target outputs",
            raw_targets,
            maximum_depth=HARD_MAX_DEPTH,
            allow_empty=False,
        )
        segment = schedule.segments[segment_id]
        if len(targets) != segment.token_count:
            raise ValueError(
                f"segment {segment_id} returned {len(targets)} targets, "
                f"expected {segment.token_count}"
            )
        existing = normalized.get(segment_id)
        if existing is not None and existing != targets:
            raise ValueError(
                f"contradictory target outputs for segment {segment_id}"
            )
        normalized[segment_id] = targets

    expected_segment_ids = set(range(len(schedule.segments)))
    if set(normalized) != expected_segment_ids:
        missing = sorted(expected_segment_ids - set(normalized))
        extra = sorted(set(normalized) - expected_segment_ids)
        raise ValueError(
            "segment targets do not cover the schedule; "
            f"missing={missing!r}, extra={extra!r}"
        )

    node_targets: dict[int, int] = {}
    for segment in schedule.segments:
        for node_id, target in zip(segment.node_ids, normalized[segment.segment_id]):
            existing = node_targets.get(node_id)
            if existing is not None and existing != target:
                raise ValueError(
                    f"contradictory target argmax for node {node_id}: "
                    f"{existing} != {target}"
                )
            node_targets[node_id] = target
    return target_argmax_by_prefix_from_node_outputs(
        schedule,
        node_targets,
        root_target_argmax=root_target_argmax,
    )


def target_argmax_by_prefix_from_flat_leaf_results(
    schedule: PrefixSegmentSchedule,
    target_argmax_by_leaf: Mapping[tuple[int, ...], Sequence[int]],
) -> dict[tuple[int, ...], int]:
    """Merge legacy flat-leaf vectors for exact parity/cross-check tests.

    Each leaf vector has ``len(candidate_path) + 1`` entries: the first predicts
    relative prefix ``()`` and the last predicts the full candidate path.
    Shared prefixes across leaves must report identical targets.
    """

    _require_schedule(schedule)
    if not isinstance(target_argmax_by_leaf, Mapping):
        raise ValueError("target_argmax_by_leaf must be a mapping")
    normalized: dict[tuple[int, ...], tuple[int, ...]] = {}
    for raw_path, raw_targets in target_argmax_by_leaf.items():
        path = _token_path(
            "flat leaf path",
            raw_path,
            maximum_depth=HARD_MAX_DEPTH,
            allow_empty=False,
        )
        targets = _token_path(
            "flat leaf target outputs",
            raw_targets,
            maximum_depth=HARD_MAX_DEPTH + 1,
            allow_empty=False,
        )
        if len(targets) != len(path) + 1:
            raise ValueError(
                f"flat leaf {path!r} returned {len(targets)} targets, "
                f"expected {len(path) + 1}"
            )
        existing = normalized.get(path)
        if existing is not None and existing != targets:
            raise ValueError(f"contradictory target outputs for flat leaf {path!r}")
        normalized[path] = targets

    expected_paths = set(schedule.candidate_paths)
    if set(normalized) != expected_paths:
        missing = sorted(expected_paths - set(normalized))
        extra = sorted(set(normalized) - expected_paths)
        raise ValueError(
            "flat targets do not cover the candidate leaves; "
            f"missing={missing!r}, extra={extra!r}"
        )

    merged: dict[tuple[int, ...], int] = {}
    for path in schedule.candidate_paths:
        for index, target in enumerate(normalized[path]):
            prefix = path[:index]
            existing = merged.get(prefix)
            if existing is not None and existing != target:
                raise ValueError(
                    "contradictory target argmax for shared relative prefix "
                    f"{prefix!r}: {existing} != {target}"
                )
            merged[prefix] = target
    expected_prefixes = _expected_relative_target_prefixes(schedule)
    if set(merged) != expected_prefixes:
        missing = sorted(expected_prefixes - set(merged), key=lambda item: (len(item), item))
        extra = sorted(set(merged) - expected_prefixes, key=lambda item: (len(item), item))
        raise ValueError(
            "flat targets do not cover candidate prefixes; "
            f"missing={missing!r}, extra={extra!r}"
        )
    return {
        prefix: merged[prefix]
        for prefix in sorted(merged, key=lambda item: (len(item), item))
    }


def _expected_relative_target_prefixes(
    schedule: PrefixSegmentSchedule,
) -> set[tuple[int, ...]]:
    return {()} | {
        path[:depth]
        for path in schedule.candidate_paths
        for depth in range(1, len(path) + 1)
    }


def _require_schedule(schedule: object) -> PrefixSegmentSchedule:
    if not isinstance(schedule, PrefixSegmentSchedule):
        raise ValueError("schedule must be a PrefixSegmentSchedule")
    return schedule


def _bounded_identifier(name: str, value: object, count: int) -> int:
    if (
        not isinstance(value, Integral)
        or isinstance(value, bool)
        or int(value) < 0
        or int(value) >= count
    ):
        raise ValueError(f"{name} must be an integer in [0, {count - 1}]")
    return int(value)


def _token_path(
    name: str,
    values: Sequence[int],
    *,
    maximum_depth: int,
    allow_empty: bool,
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a sequence of token ids")
    if not values and not allow_empty:
        raise ValueError(f"{name} must not be empty")
    if len(values) > maximum_depth:
        raise ValueError(f"{name} depth {len(values)} exceeds limit {maximum_depth}")
    return tuple(_token(token) for token in values)


def _token(value: object) -> int:
    if (
        not isinstance(value, Integral)
        or isinstance(value, bool)
        or int(value) < 0
        or int(value) > UINT32_MAX
    ):
        raise ValueError(f"token id must be an integer in [0, {UINT32_MAX}]")
    return int(value)


def _bounded_limit(name: str, value: object, hard_maximum: int) -> int:
    if (
        not isinstance(value, Integral)
        or isinstance(value, bool)
        or int(value) < 1
        or int(value) > hard_maximum
    ):
        raise ValueError(f"{name} must be an integer in [1, {hard_maximum}]")
    return int(value)


__all__ = [
    "DEFAULT_MAX_DEPTH",
    "DEFAULT_MAX_NODES",
    "DEFAULT_MAX_PATHS",
    "PrefixCompatibleFrontierGroup",
    "PrefixComputeSegment",
    "PrefixSegmentCostReport",
    "PrefixSegmentFork",
    "PrefixSegmentLeaf",
    "PrefixSegmentNode",
    "PrefixSegmentOperation",
    "PrefixSegmentSchedule",
    "plan_prefix_segment_schedule",
    "target_argmax_by_prefix_from_flat_leaf_results",
    "target_argmax_by_prefix_from_node_outputs",
    "target_argmax_by_prefix_from_segment_outputs",
]
