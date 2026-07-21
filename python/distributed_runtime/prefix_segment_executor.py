"""Pure physical executor prototype for :mod:`prefix_segment_schedule`.

The planner deliberately knows only logical lane numbers.  This module adds
the smallest backend-facing layer needed to prove that the plan can be mapped
to real KV-cache request IDs without ever computing on the real parent:

* lane ``0`` is first materialised by a *carrier* FORK from the intact parent;
* later FORKs clone the physical request currently bound to their source lane;
* every segment computes on a physical lane request, never on the parent;
* the stable logical owner is metadata and is never confused with a physical
  FORK source;
* a leaf can later be promoted into the intact parent while all other lanes are
  ended.

There are no sockets, tensors or runtime imports here.  Callbacks are the
mutation boundary.  Any validation error before the first callback is safe for
same-route fallback.  Once a mutation callback is attempted, an exception is
conservatively classified as ``route_fatal`` because a remote/backend mutation
may already have happened even when the callback raised.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Protocol

from .prefix_segment_schedule import (
    HARD_MAX_DEPTH,
    HARD_MAX_NODES,
    HARD_MAX_PATHS,
    PrefixComputeSegment,
    PrefixSegmentFork,
    PrefixSegmentSchedule,
    plan_prefix_segment_schedule,
    target_argmax_by_prefix_from_node_outputs,
)


UINT32_MAX = (1 << 32) - 1
UINT64_MAX = (1 << 64) - 1
_MISSING = object()


class PrefixSegmentExecutionError(RuntimeError):
    """Fail-closed execution failure with an explicit fallback boundary."""

    def __init__(
        self,
        message: str,
        *,
        route_fatal: bool,
        phase: str,
        completed_operation_count: int,
        attempted_operation: object | None = None,
    ) -> None:
        super().__init__(message)
        self.route_fatal = route_fatal
        self.phase = phase
        self.completed_operation_count = completed_operation_count
        self.attempted_operation = attempted_operation

    @property
    def fallback_allowed(self) -> bool:
        return not self.route_fatal


@dataclass(frozen=True, slots=True)
class PrefixPhysicalLaneBinding:
    lane_id: int
    physical_request_id: int
    logical_owner_request_id: int


@dataclass(frozen=True, slots=True)
class PrefixPhysicalFork:
    """One backend clone operation.

    ``fork_source_physical_request_id`` is the request whose KV is cloned.
    ``logical_owner_request_id`` is only the stable owner of the speculative
    route.  The two IDs are intentionally separate and may be different.
    """

    execution_index: int
    schedule_sequence_index: int | None
    fork_id: int | None
    frontier_index: int | None
    logical_owner_request_id: int
    source_lane_id: int | None
    target_lane_id: int
    fork_source_physical_request_id: int
    target_physical_request_id: int
    divergence_node_id: int
    prefix: tuple[int, ...]
    carrier: bool


@dataclass(frozen=True, slots=True)
class PrefixPhysicalCompute:
    """One maximal unary-chain compute on a physical lane request."""

    execution_index: int
    schedule_sequence_index: int
    segment_id: int
    frontier_index: int
    compatible_group_id: int
    logical_owner_request_id: int
    lane_id: int
    physical_request_id: int
    parent_node_id: int
    node_ids: tuple[int, ...]
    tokens: tuple[int, ...]
    start_depth: int
    end_depth: int

    @property
    def token_count(self) -> int:
        return len(self.tokens)


PrefixPhysicalOperation = PrefixPhysicalFork | PrefixPhysicalCompute


@dataclass(frozen=True, slots=True)
class PrefixPhysicalComputeBatch:
    """One exact backend call for a compatible planner frontier group.

    Every contained operation is still a complete
    :class:`PrefixPhysicalCompute` and remains individually visible in the
    successful execution trace.  The wrapper changes callback granularity,
    not execution semantics or output ordering.
    """

    execution_index: int
    frontier_index: int
    compatible_group_id: int
    start_depth: int
    token_count: int
    operations: tuple[PrefixPhysicalCompute, ...]

    @property
    def segment_ids(self) -> tuple[int, ...]:
        return tuple(operation.segment_id for operation in self.operations)


class PrefixSegmentExecutorBackend(Protocol):
    """Typed mutation callbacks required by the pure executor."""

    def fork(self, operation: PrefixPhysicalFork) -> None:
        """Clone source KV into the target request while preserving source."""

    def compute(self, operation: PrefixPhysicalCompute) -> Sequence[int]:
        """Append ``tokens`` and return one exact target argmax per node."""


class PrefixSegmentBatchExecutorBackend(PrefixSegmentExecutorBackend, Protocol):
    """Optional exact compatible-group batch extension."""

    def compute_batch(
        self,
        batch: PrefixPhysicalComputeBatch,
    ) -> Sequence[Sequence[int]]:
        """Return one exact target vector per operation, in input order."""


@dataclass(frozen=True, slots=True)
class PrefixExecutedSegmentOutput:
    segment_id: int
    lane_id: int
    physical_request_id: int
    node_ids: tuple[int, ...]
    target_argmax: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PrefixExecutedNodeOutput:
    node_id: int
    segment_id: int
    lane_id: int
    physical_request_id: int
    target_argmax: int


@dataclass(frozen=True, slots=True)
class PrefixPhysicalLeaf:
    leaf_id: int
    candidate_path: tuple[int, ...]
    full_path: tuple[int, ...]
    lane_id: int
    physical_request_id: int
    logical_owner_request_id: int


@dataclass(frozen=True, slots=True)
class EndPrefixLeaf:
    logical_owner_request_id: int
    lane_id: int
    physical_request_id: int
    candidate_path: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PromotePrefixLeaf:
    logical_owner_request_id: int
    intact_parent_physical_request_id: int
    winner_lane_id: int
    winner_physical_request_id: int
    winner_candidate_path: tuple[int, ...]


PrefixLeafCommitOperation = EndPrefixLeaf | PromotePrefixLeaf


@dataclass(frozen=True, slots=True)
class PrefixLeafCommitPlan:
    logical_owner_request_id: int
    intact_parent_physical_request_id: int
    winner: PrefixPhysicalLeaf
    loser_ends: tuple[EndPrefixLeaf, ...]
    promotion: PromotePrefixLeaf
    operations: tuple[PrefixLeafCommitOperation, ...]


@dataclass(frozen=True, slots=True)
class PrefixSegmentExecutionResult:
    schedule: PrefixSegmentSchedule
    logical_owner_request_id: int
    intact_parent_physical_request_id: int
    lane_bindings: tuple[PrefixPhysicalLaneBinding, ...]
    operations: tuple[PrefixPhysicalOperation, ...]
    segment_outputs: tuple[PrefixExecutedSegmentOutput, ...]
    node_outputs: tuple[PrefixExecutedNodeOutput, ...]
    leaves: tuple[PrefixPhysicalLeaf, ...]

    @property
    def parent_was_compute_target(self) -> bool:
        return any(
            isinstance(operation, PrefixPhysicalCompute)
            and operation.physical_request_id
            == self.intact_parent_physical_request_id
            for operation in self.operations
        )

    @property
    def target_argmax_by_node(self) -> dict[int, int]:
        return {output.node_id: output.target_argmax for output in self.node_outputs}

    def target_argmax_by_prefix(
        self,
        *,
        root_target_argmax: int | object = _MISSING,
    ) -> dict[tuple[int, ...], int]:
        kwargs: dict[str, int] = {}
        if root_target_argmax is not _MISSING:
            kwargs["root_target_argmax"] = _bounded_int(
                "root_target_argmax", root_target_argmax, maximum=UINT32_MAX
            )
        return target_argmax_by_prefix_from_node_outputs(
            self.schedule,
            self.target_argmax_by_node,
            **kwargs,
        )

    def plan_leaf_commit(
        self,
        winning_candidate_path: Sequence[int],
    ) -> PrefixLeafCommitPlan:
        winner_path = _tokens(
            "winning_candidate_path",
            winning_candidate_path,
            maximum=UINT32_MAX,
            allow_empty=False,
        )
        matching = tuple(
            leaf for leaf in self.leaves if leaf.candidate_path == winner_path
        )
        if len(matching) != 1:
            raise ValueError(f"unknown winning candidate path {winner_path!r}")
        winner = matching[0]
        loser_ends = tuple(
            EndPrefixLeaf(
                logical_owner_request_id=self.logical_owner_request_id,
                lane_id=leaf.lane_id,
                physical_request_id=leaf.physical_request_id,
                candidate_path=leaf.candidate_path,
            )
            for leaf in self.leaves
            if leaf is not winner
        )
        promotion = PromotePrefixLeaf(
            logical_owner_request_id=self.logical_owner_request_id,
            intact_parent_physical_request_id=self.intact_parent_physical_request_id,
            winner_lane_id=winner.lane_id,
            winner_physical_request_id=winner.physical_request_id,
            winner_candidate_path=winner.candidate_path,
        )
        return PrefixLeafCommitPlan(
            logical_owner_request_id=self.logical_owner_request_id,
            intact_parent_physical_request_id=self.intact_parent_physical_request_id,
            winner=winner,
            loser_ends=loser_ends,
            promotion=promotion,
            operations=(*loser_ends, promotion),
        )


@dataclass(frozen=True, slots=True)
class _PreparedExecution:
    lane_bindings: tuple[PrefixPhysicalLaneBinding, ...]
    fork_callback: object
    compute_callback: object
    compute_batch_callback: object | None


def execute_prefix_segment_schedule(
    schedule: PrefixSegmentSchedule,
    *,
    logical_owner_request_id: int,
    intact_parent_physical_request_id: int,
    lane_physical_request_ids: Sequence[int],
    backend: PrefixSegmentExecutorBackend,
) -> PrefixSegmentExecutionResult:
    """Execute a logical segment schedule against typed physical callbacks.

    All inputs and the complete schedule are preflighted before ``backend`` is
    called.  That makes a preflight exception fallback-safe.  Mutation calls
    are then issued in one canonical order: carrier FORK first, followed by the
    planner operations verbatim (whose frontier invariant places nested FORKs
    before compute).
    """

    try:
        owner_id = _bounded_int(
            "logical_owner_request_id",
            logical_owner_request_id,
            maximum=UINT64_MAX,
        )
        parent_id = _bounded_int(
            "intact_parent_physical_request_id",
            intact_parent_physical_request_id,
            maximum=UINT64_MAX,
        )
        prepared = _prepare_execution(
            schedule,
            owner_id=owner_id,
            parent_id=parent_id,
            lane_physical_request_ids=lane_physical_request_ids,
            backend=backend,
        )
    except PrefixSegmentExecutionError:
        raise
    except Exception as exc:
        raise PrefixSegmentExecutionError(
            f"prefix-segment preflight failed: {exc}",
            route_fatal=False,
            phase="preflight",
            completed_operation_count=0,
        ) from exc

    physical_by_lane = {
        binding.lane_id: binding.physical_request_id
        for binding in prepared.lane_bindings
    }
    completed_operations: list[PrefixPhysicalOperation] = []
    segment_outputs: list[PrefixExecutedSegmentOutput] = []
    node_outputs: list[PrefixExecutedNodeOutput] = []

    carrier = PrefixPhysicalFork(
        execution_index=0,
        schedule_sequence_index=None,
        fork_id=None,
        frontier_index=None,
        logical_owner_request_id=owner_id,
        source_lane_id=None,
        target_lane_id=0,
        fork_source_physical_request_id=parent_id,
        target_physical_request_id=physical_by_lane[0],
        divergence_node_id=0,
        prefix=(),
        carrier=True,
    )
    _apply_fork(
        prepared.fork_callback,
        carrier,
        completed_operations=completed_operations,
        phase="carrier_fork",
    )

    schedule_operation_index = 0
    while schedule_operation_index < len(schedule.operations):
        operation = schedule.operations[schedule_operation_index]
        if isinstance(operation, PrefixSegmentFork):
            physical_fork = PrefixPhysicalFork(
                execution_index=len(completed_operations),
                schedule_sequence_index=operation.sequence_index,
                fork_id=operation.fork_id,
                frontier_index=operation.frontier_index,
                logical_owner_request_id=owner_id,
                source_lane_id=operation.source_lane_id,
                target_lane_id=operation.target_lane_id,
                fork_source_physical_request_id=physical_by_lane[
                    operation.source_lane_id
                ],
                target_physical_request_id=physical_by_lane[
                    operation.target_lane_id
                ],
                divergence_node_id=operation.divergence_node_id,
                prefix=operation.prefix,
                carrier=False,
            )
            _apply_fork(
                prepared.fork_callback,
                physical_fork,
                completed_operations=completed_operations,
                phase="nested_fork",
            )
            schedule_operation_index += 1
            continue

        if not isinstance(operation, PrefixComputeSegment):
            # The complete operation vector was preflighted, so reaching this
            # branch after a carrier mutation means memory changed underneath
            # us or an invariant was violated.  Quarantine the route.
            raise PrefixSegmentExecutionError(
                f"unknown prefix-segment operation {operation!r}",
                route_fatal=True,
                phase="dispatch",
                completed_operation_count=len(completed_operations),
                attempted_operation=operation,
            )
        compatible_group_id = operation.compatible_group_id
        grouped_schedule_operations: list[PrefixComputeSegment] = []
        while schedule_operation_index < len(schedule.operations):
            candidate = schedule.operations[schedule_operation_index]
            if (
                not isinstance(candidate, PrefixComputeSegment)
                or candidate.compatible_group_id != compatible_group_id
            ):
                break
            grouped_schedule_operations.append(candidate)
            schedule_operation_index += 1

        physical_computes = tuple(
            PrefixPhysicalCompute(
                execution_index=len(completed_operations) + offset,
                schedule_sequence_index=segment.sequence_index,
                segment_id=segment.segment_id,
                frontier_index=segment.frontier_index,
                compatible_group_id=segment.compatible_group_id,
                logical_owner_request_id=owner_id,
                lane_id=segment.lane_id,
                physical_request_id=physical_by_lane[segment.lane_id],
                parent_node_id=segment.parent_node_id,
                node_ids=segment.node_ids,
                tokens=segment.tokens,
                start_depth=segment.start_depth,
                end_depth=segment.end_depth,
            )
            for offset, segment in enumerate(grouped_schedule_operations)
        )
        if prepared.compute_batch_callback is None:
            target_vectors = tuple(
                _apply_compute(
                    prepared.compute_callback,
                    physical_compute,
                    completed_operations=completed_operations,
                )
                for physical_compute in physical_computes
            )
        else:
            compute_batch = PrefixPhysicalComputeBatch(
                execution_index=physical_computes[0].execution_index,
                frontier_index=physical_computes[0].frontier_index,
                compatible_group_id=compatible_group_id,
                start_depth=physical_computes[0].start_depth,
                token_count=physical_computes[0].token_count,
                operations=physical_computes,
            )
            target_vectors = _apply_compute_batch(
                prepared.compute_batch_callback,
                compute_batch,
                completed_operations=completed_operations,
            )
        for physical_compute, targets in zip(physical_computes, target_vectors):
            segment_outputs.append(
                PrefixExecutedSegmentOutput(
                    segment_id=physical_compute.segment_id,
                    lane_id=physical_compute.lane_id,
                    physical_request_id=physical_compute.physical_request_id,
                    node_ids=physical_compute.node_ids,
                    target_argmax=targets,
                )
            )
            node_outputs.extend(
                PrefixExecutedNodeOutput(
                    node_id=node_id,
                    segment_id=physical_compute.segment_id,
                    lane_id=physical_compute.lane_id,
                    physical_request_id=physical_compute.physical_request_id,
                    target_argmax=target,
                )
                for node_id, target in zip(physical_compute.node_ids, targets)
            )

    try:
        if len(segment_outputs) != len(schedule.segments):
            raise RuntimeError("execution did not return every segment output")
        if len(node_outputs) != len(schedule.nodes) - 1:
            raise RuntimeError("execution did not return every node output")
        leaves = tuple(
            PrefixPhysicalLeaf(
                leaf_id=leaf.leaf_id,
                candidate_path=leaf.candidate_path,
                full_path=leaf.full_path,
                lane_id=leaf.lane_id,
                physical_request_id=physical_by_lane[leaf.lane_id],
                logical_owner_request_id=owner_id,
            )
            for leaf in schedule.leaves
        )
        result = PrefixSegmentExecutionResult(
            schedule=schedule,
            logical_owner_request_id=owner_id,
            intact_parent_physical_request_id=parent_id,
            lane_bindings=prepared.lane_bindings,
            operations=tuple(completed_operations),
            segment_outputs=tuple(segment_outputs),
            node_outputs=tuple(node_outputs),
            leaves=leaves,
        )
        if result.parent_was_compute_target:
            raise RuntimeError("the intact parent became a compute target")
        return result
    except PrefixSegmentExecutionError:
        raise
    except Exception as exc:
        raise PrefixSegmentExecutionError(
            f"prefix-segment result finalization failed: {exc}",
            route_fatal=True,
            phase="finalize",
            completed_operation_count=len(completed_operations),
        ) from exc


def _apply_fork(
    callback: object,
    operation: PrefixPhysicalFork,
    *,
    completed_operations: list[PrefixPhysicalOperation],
    phase: str,
) -> None:
    try:
        callback(operation)  # type: ignore[operator]
    except BaseException as exc:
        raise PrefixSegmentExecutionError(
            f"prefix-segment {phase} failed: {exc}",
            route_fatal=True,
            phase=phase,
            completed_operation_count=len(completed_operations),
            attempted_operation=operation,
        ) from exc
    completed_operations.append(operation)


def _apply_compute(
    callback: object,
    operation: PrefixPhysicalCompute,
    *,
    completed_operations: list[PrefixPhysicalOperation],
) -> tuple[int, ...]:
    try:
        raw_targets = callback(operation)  # type: ignore[operator]
        targets = _tokens(
            f"segment {operation.segment_id} target outputs",
            raw_targets,
            maximum=UINT32_MAX,
            allow_empty=False,
        )
        if len(targets) != operation.token_count:
            raise ValueError(
                f"segment {operation.segment_id} returned {len(targets)} targets, "
                f"expected {operation.token_count}"
            )
    except BaseException as exc:
        raise PrefixSegmentExecutionError(
            f"prefix-segment compute failed: {exc}",
            route_fatal=True,
            phase="compute",
            completed_operation_count=len(completed_operations),
            attempted_operation=operation,
        ) from exc
    completed_operations.append(operation)
    return targets


def _apply_compute_batch(
    callback: object,
    batch: PrefixPhysicalComputeBatch,
    *,
    completed_operations: list[PrefixPhysicalOperation],
) -> tuple[tuple[int, ...], ...]:
    try:
        raw_vectors = callback(batch)  # type: ignore[operator]
        if isinstance(raw_vectors, (str, bytes, bytearray)) or not isinstance(
            raw_vectors, Sequence
        ):
            raise ValueError("batch target outputs must be a sequence of vectors")
        if len(raw_vectors) != len(batch.operations):
            raise ValueError(
                f"compatible group {batch.compatible_group_id} returned "
                f"{len(raw_vectors)} segment vectors, expected "
                f"{len(batch.operations)}"
            )
        target_vectors: list[tuple[int, ...]] = []
        for operation, raw_targets in zip(batch.operations, raw_vectors):
            targets = _tokens(
                f"segment {operation.segment_id} batched target outputs",
                raw_targets,
                maximum=UINT32_MAX,
                allow_empty=False,
            )
            if len(targets) != operation.token_count:
                raise ValueError(
                    f"segment {operation.segment_id} returned {len(targets)} "
                    f"batched targets, expected {operation.token_count}"
                )
            target_vectors.append(targets)
    except BaseException as exc:
        raise PrefixSegmentExecutionError(
            f"prefix-segment batch compute failed: {exc}",
            route_fatal=True,
            phase="compute_batch",
            completed_operation_count=len(completed_operations),
            attempted_operation=batch,
        ) from exc
    completed_operations.extend(batch.operations)
    return tuple(target_vectors)


def _prepare_execution(
    schedule: PrefixSegmentSchedule,
    *,
    owner_id: int,
    parent_id: int,
    lane_physical_request_ids: Sequence[int],
    backend: PrefixSegmentExecutorBackend,
) -> _PreparedExecution:
    if not isinstance(schedule, PrefixSegmentSchedule):
        raise TypeError("schedule must be a PrefixSegmentSchedule")
    if isinstance(lane_physical_request_ids, (str, bytes, bytearray)) or not isinstance(
        lane_physical_request_ids, Sequence
    ):
        raise TypeError("lane_physical_request_ids must be a sequence")
    physical_ids = tuple(
        _bounded_int("physical lane request id", value, maximum=UINT64_MAX)
        for value in lane_physical_request_ids
    )
    if len(physical_ids) != schedule.cost.lane_count:
        raise ValueError(
            f"received {len(physical_ids)} physical lane IDs, "
            f"expected {schedule.cost.lane_count}"
        )
    if len(set(physical_ids)) != len(physical_ids):
        raise ValueError("physical lane request IDs must be unique")
    if parent_id in physical_ids:
        raise ValueError("a physical lane request ID aliases the intact parent")

    fork_callback = getattr(backend, "fork", None)
    compute_callback = getattr(backend, "compute", None)
    if not callable(fork_callback):
        raise TypeError("backend.fork must be callable")
    if not callable(compute_callback):
        raise TypeError("backend.compute must be callable")
    compute_batch_callback = getattr(backend, "compute_batch", None)
    if compute_batch_callback is not None and not callable(compute_batch_callback):
        raise TypeError("backend.compute_batch must be callable when provided")

    _validate_schedule_for_execution(schedule)
    lane_bindings = tuple(
        PrefixPhysicalLaneBinding(
            lane_id=lane_id,
            physical_request_id=physical_id,
            logical_owner_request_id=owner_id,
        )
        for lane_id, physical_id in enumerate(physical_ids)
    )
    return _PreparedExecution(
        lane_bindings=lane_bindings,
        fork_callback=fork_callback,
        compute_callback=compute_callback,
        compute_batch_callback=compute_batch_callback,
    )


def _validate_schedule_for_execution(schedule: PrefixSegmentSchedule) -> None:
    # Internal consistency is not enough: leaf metadata and target mappings are
    # consumed after mutation, so a forged-but-self-consistent dataclass must
    # not reach the carrier FORK.  Regenerate the public canonical planner
    # output under the hard bounds and require full structural equality.
    try:
        canonical = plan_prefix_segment_schedule(
            schedule.candidate_paths,
            shared_prefix_tokens=schedule.shared_prefix_tokens,
            max_paths=HARD_MAX_PATHS,
            max_depth=HARD_MAX_DEPTH,
            max_nodes=HARD_MAX_NODES,
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"schedule cannot be regenerated canonically: {exc}") from exc
    if schedule != canonical:
        raise ValueError("schedule is not the complete canonical planner output")

    lane_count = schedule.cost.lane_count
    if lane_count < 1:
        raise ValueError("schedule must contain at least one lane")
    if schedule.root.node_id != 0 or schedule.root.prefix != ():
        raise ValueError("schedule virtual root is malformed")
    if schedule.root.lane_id != 0:
        raise ValueError("schedule virtual root must use lane 0")
    if schedule.cost.fork_count != len(schedule.forks):
        raise ValueError("schedule fork count is inconsistent")
    if schedule.cost.segment_calls != len(schedule.segments):
        raise ValueError("schedule segment count is inconsistent")
    if len(schedule.operations) != len(schedule.forks) + len(schedule.segments):
        raise ValueError("schedule operation count is inconsistent")
    grouped_segment_ids: list[int] = []
    for expected_group_id, group in enumerate(
        schedule.compatible_frontier_groups
    ):
        if group.group_id != expected_group_id:
            raise ValueError("compatible group IDs are not contiguous")
        if not (
            0
            <= group.first_operation_index
            <= group.last_operation_index
            < len(schedule.operations)
        ):
            raise ValueError("compatible group operation range is invalid")
        group_operations = schedule.operations[
            group.first_operation_index : group.last_operation_index + 1
        ]
        if len(group_operations) != len(group.segment_ids):
            raise ValueError("compatible group operation range has the wrong size")
        for expected_segment_id, operation in zip(
            group.segment_ids, group_operations
        ):
            if not isinstance(operation, PrefixComputeSegment):
                raise ValueError("compatible group contains a non-compute operation")
            if operation.segment_id != expected_segment_id:
                raise ValueError("compatible group segment order is inconsistent")
            if operation.compatible_group_id != group.group_id:
                raise ValueError("segment points at a different compatible group")
            if operation.frontier_index != group.frontier_index:
                raise ValueError("compatible group mixes dependency frontiers")
            if operation.start_depth != group.start_depth:
                raise ValueError("compatible group mixes start depths")
            if operation.token_count != group.token_count:
                raise ValueError("compatible group mixes token counts")
            grouped_segment_ids.append(operation.segment_id)
    if grouped_segment_ids != [
        operation.segment_id
        for operation in schedule.operations
        if isinstance(operation, PrefixComputeSegment)
    ]:
        raise ValueError(
            "compatible groups do not cover compute operations in exact order"
        )

    lane_prefix: dict[int, tuple[int, ...]] = {0: ()}
    compute_seen_by_frontier: set[int] = set()
    seen_fork_ids: set[int] = set()
    seen_segment_ids: set[int] = set()
    computed_node_ids: list[int] = []

    for sequence_index, operation in enumerate(schedule.operations):
        if operation.sequence_index != sequence_index:
            raise ValueError("schedule operation indices are not contiguous")
        if isinstance(operation, PrefixSegmentFork):
            if operation.frontier_index in compute_seen_by_frontier:
                raise ValueError("a nested fork follows compute in its frontier")
            if operation.fork_id in seen_fork_ids:
                raise ValueError("schedule reuses a fork id")
            seen_fork_ids.add(operation.fork_id)
            if not 0 <= operation.source_lane_id < lane_count:
                raise ValueError("fork source lane is outside the schedule")
            if not 0 <= operation.target_lane_id < lane_count:
                raise ValueError("fork target lane is outside the schedule")
            if operation.source_lane_id not in lane_prefix:
                raise ValueError("fork source lane is not materialised")
            if operation.target_lane_id in lane_prefix:
                raise ValueError("fork target lane is already materialised")
            divergence = schedule.node(operation.divergence_node_id)
            if divergence.prefix != operation.prefix:
                raise ValueError("fork divergence node and prefix disagree")
            if lane_prefix[operation.source_lane_id] != operation.prefix:
                raise ValueError("fork source lane is not at the divergence prefix")
            lane_prefix[operation.target_lane_id] = operation.prefix
            continue

        if not isinstance(operation, PrefixComputeSegment):
            raise ValueError(f"unknown schedule operation {operation!r}")
        compute_seen_by_frontier.add(operation.frontier_index)
        if operation.segment_id in seen_segment_ids:
            raise ValueError("schedule reuses a segment id")
        seen_segment_ids.add(operation.segment_id)
        if not 0 <= operation.lane_id < lane_count:
            raise ValueError("compute lane is outside the schedule")
        if operation.lane_id not in lane_prefix:
            raise ValueError("compute lane is not materialised")
        if not operation.tokens or len(operation.node_ids) != len(operation.tokens):
            raise ValueError("segment node/token vectors are malformed")
        parent = schedule.node(operation.parent_node_id)
        if lane_prefix[operation.lane_id] != parent.prefix:
            raise ValueError("compute lane does not hold its parent prefix")
        if operation.start_depth != parent.depth:
            raise ValueError("segment start depth disagrees with its parent")
        cursor = parent.prefix
        for node_id, token in zip(operation.node_ids, operation.tokens):
            node = schedule.node(node_id)
            cursor = (*cursor, token)
            if node.prefix != cursor or node.token != token:
                raise ValueError("segment does not follow contiguous trie nodes")
            if node.lane_id != operation.lane_id:
                raise ValueError("segment node is assigned to a different lane")
            computed_node_ids.append(node_id)
        if operation.end_depth != len(cursor):
            raise ValueError("segment end depth is inconsistent")
        lane_prefix[operation.lane_id] = cursor

    if seen_fork_ids != set(range(len(schedule.forks))):
        raise ValueError("schedule fork IDs do not cover the fork vector")
    if seen_segment_ids != set(range(len(schedule.segments))):
        raise ValueError("schedule segment IDs do not cover the segment vector")
    if set(lane_prefix) != set(range(lane_count)):
        raise ValueError("schedule did not materialise every lane")
    if len(computed_node_ids) != len(set(computed_node_ids)):
        raise ValueError("schedule computes a node more than once")
    if set(computed_node_ids) != set(range(1, len(schedule.nodes))):
        raise ValueError("schedule does not compute every non-root node")
    if len(schedule.leaves) != lane_count:
        raise ValueError("prefix-free leaves must have one distinct final lane")
    leaf_lanes: set[int] = set()
    for leaf in schedule.leaves:
        if leaf.lane_id in leaf_lanes:
            raise ValueError("two leaves alias the same final lane")
        leaf_lanes.add(leaf.lane_id)
        if lane_prefix.get(leaf.lane_id) != leaf.full_path:
            raise ValueError("leaf lane does not hold its full path")


def _bounded_int(name: str, value: object, *, maximum: int) -> int:
    if (
        not isinstance(value, Integral)
        or isinstance(value, bool)
        or int(value) < 0
        or int(value) > maximum
    ):
        raise ValueError(f"{name} must be an integer in [0, {maximum}]")
    return int(value)


def _tokens(
    name: str,
    values: Sequence[int],
    *,
    maximum: int,
    allow_empty: bool,
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a sequence of integers")
    if not values and not allow_empty:
        raise ValueError(f"{name} must not be empty")
    return tuple(_bounded_int(name, value, maximum=maximum) for value in values)


__all__ = [
    "EndPrefixLeaf",
    "PrefixExecutedNodeOutput",
    "PrefixExecutedSegmentOutput",
    "PrefixLeafCommitOperation",
    "PrefixLeafCommitPlan",
    "PrefixPhysicalCompute",
    "PrefixPhysicalComputeBatch",
    "PrefixPhysicalFork",
    "PrefixPhysicalLaneBinding",
    "PrefixPhysicalLeaf",
    "PrefixPhysicalOperation",
    "PrefixSegmentExecutionError",
    "PrefixSegmentExecutionResult",
    "PrefixSegmentBatchExecutorBackend",
    "PrefixSegmentExecutorBackend",
    "PromotePrefixLeaf",
    "execute_prefix_segment_schedule",
]
