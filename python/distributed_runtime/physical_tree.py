"""Pure coordinator for exact, flat physical MacroWave trees.

The coordinator in this module deliberately owns no sockets and no tensor
backend.  It turns one already validated :class:`MacroWaveProposal` into typed
commands that a pipeline engine can apply locally and enqueue on its ordered
wire stream.  One physical request is allocated per candidate *leaf*; every
leaf is a direct child of the real request.

This is the flat-leaf MVP described in ``docs/PHYSICAL_SPARSE_TREE_EXECUTION``.
Its lifecycle is backend-agnostic: runners may use safe copies or the optional
paged/COW ABI.  It still does not implement prefix-closed execution or tree
attention.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
import math
from numbers import Integral
import time

from .macro_wave import MacroWaveState
from .macro_wave_adapter import (
    MacroWaveProposal,
    MacroWaveResolution,
    resolve_macro_wave,
)


UINT32_MAX = (1 << 32) - 1
UINT64_MAX = (1 << 64) - 1


class PhysicalTreeError(RuntimeError):
    """Base class for a fail-closed physical-tree coordination error."""


class PhysicalTreeProtocolError(PhysicalTreeError):
    """A virtual return violated the sealed wave contract."""


class UnknownVirtualRequestError(PhysicalTreeProtocolError):
    """A return used an ID which is neither live nor a retained tombstone."""


class TreePhase(str, Enum):
    PREPARED = "prepared"
    FORKING = "forking"
    VERIFYING = "verifying"
    RESOLVING = "resolving"
    CLEANING = "cleaning"
    PROMOTING = "promoting"
    TRUNCATING = "truncating"
    COMMITTED = "committed"
    ABORTING = "aborting"
    DRAINING = "draining"
    ABORTED = "aborted"


@dataclass(frozen=True)
class ForkCommand:
    child_request_id: int
    parent_request_id: int

    @property
    def request_id(self) -> int:
        return self.child_request_id


@dataclass(frozen=True)
class VerifyCommand:
    request_id: int
    step: int
    input_tokens: tuple[int, ...]

    @property
    def token_count(self) -> int:
        return len(self.input_tokens)


@dataclass(frozen=True)
class EndCommand:
    request_id: int


@dataclass(frozen=True)
class PromoteCommand:
    parent_request_id: int
    child_request_id: int

    @property
    def request_id(self) -> int:
        return self.parent_request_id


@dataclass(frozen=True)
class TruncateCommand:
    request_id: int
    keep_tokens: int

    @property
    def token_count(self) -> int:
        return self.keep_tokens


@dataclass(frozen=True)
class CancelCommand:
    request_id: int


PhysicalTreeCommand = (
    ForkCommand
    | VerifyCommand
    | EndCommand
    | PromoteCommand
    | TruncateCommand
    | CancelCommand
)
CleanupCommand = EndCommand | PromoteCommand | TruncateCommand


@dataclass
class PhysicalLeaf:
    path: tuple[int, ...]
    virtual_request_id: int
    inherited_step: int
    expected_targets: int
    ordinal: int
    fork_sent: bool = False
    verify_sent: bool = False
    verify_sent_at: float | None = None
    result: tuple[int, ...] | None = None
    return_tombstone: bool = False
    return_drained: bool = False
    physically_closed: bool = False


@dataclass
class PhysicalTreeWave:
    parent_request_id: int
    proposal: MacroWaveProposal
    base_kv_tokens: int
    pending_token: int
    inherited_step: int
    leaves: dict[tuple[int, ...], PhysicalLeaf]
    virtual_routes: dict[int, PhysicalLeaf]
    target_argmax_by_prefix: dict[tuple[int, ...], int]
    deadline_at: float
    phase: TreePhase = TreePhase.PREPARED
    carrier_request_id: int | None = None
    resolution: MacroWaveResolution | None = None
    cleanup_commands: tuple[CleanupCommand, ...] = ()
    cleanup_index: int = 0
    abort_reason: str | None = None
    route_fatal: bool = False
    cancel_requested: bool = False
    parent_cancel_sent: bool = False

    @property
    def ordered_leaves(self) -> tuple[PhysicalLeaf, ...]:
        return tuple(sorted(self.leaves.values(), key=lambda leaf: leaf.ordinal))

    @property
    def cleanup_complete(self) -> bool:
        return self.cleanup_index == len(self.cleanup_commands)


@dataclass(frozen=True)
class TreeCommitPlan:
    parent_request_id: int
    resolution: MacroWaveResolution
    carrier_request_id: int
    target_argmax_by_prefix: tuple[tuple[tuple[int, ...], int], ...]
    commands: tuple[CleanupCommand, ...]
    next_step: int


@dataclass(frozen=True)
class TreeAbortPlan:
    parent_request_id: int
    reason: str
    commands: tuple[CleanupCommand, ...]
    route_fatal: bool
    oldest_virtual_request_id: int | None = None

    @property
    def fallback_allowed(self) -> bool:
        return not self.route_fatal


@dataclass(frozen=True)
class CancellationPlan:
    parent_request_id: int
    commands: tuple[CancelCommand, ...]
    deferred_until_commit: bool
    draining_virtual_request_ids: tuple[int, ...]


@dataclass(frozen=True)
class TombstoneDrain:
    parent_request_id: int
    virtual_request_id: int
    wave_drained: bool


PhysicalTreeResult = TreeCommitPlan | TreeAbortPlan | TombstoneDrain | None


def _bounded_int(name: str, value: object, *, maximum: int) -> int:
    if (
        not isinstance(value, Integral)
        or isinstance(value, bool)
        or int(value) < 0
        or int(value) > maximum
    ):
        raise ValueError(f"{name} must be an integer in [0, {maximum}]")
    return int(value)


def _tokens(values: Sequence[int], name: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError(f"{name} must be a sequence of token ids")
    return tuple(_bounded_int(name, value, maximum=UINT32_MAX) for value in values)


def _finite_time(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite non-negative number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return normalized


class PhysicalTreeCoordinator:
    """State machine for exact physical leaves and their virtual routes.

    Virtual IDs are supplied by the caller and must be strictly increasing
    across the lifetime of this coordinator.  Real and virtual IDs must come
    from the caller's same never-reused allocator; this coordinator additionally
    rejects every collision that is live in its indexes.  Issuing a command
    consumes that operation exactly once; FORK/PROMOTE are never offered for
    retry.
    """

    def __init__(self) -> None:
        self.tree_by_parent: dict[int, PhysicalTreeWave] = {}
        self.leaf_route: dict[int, tuple[int, PhysicalLeaf]] = {}
        self._virtual_id_high_water = -1

    def prepare_wave(
        self,
        *,
        parent_request_id: int,
        proposal: MacroWaveProposal,
        base_kv_tokens: int,
        pending_token: int,
        inherited_step: int,
        virtual_request_ids: Sequence[int],
        deadline_at: float,
    ) -> PhysicalTreeWave:
        """Validate and register one flat leaf wave without issuing commands."""

        parent = _bounded_int(
            "parent_request_id", parent_request_id, maximum=UINT64_MAX
        )
        base = _bounded_int("base_kv_tokens", base_kv_tokens, maximum=UINT32_MAX)
        pending = _bounded_int("pending_token", pending_token, maximum=UINT32_MAX)
        step = _bounded_int(
            "inherited_step", inherited_step, maximum=UINT32_MAX - 1
        )
        deadline = _finite_time("deadline_at", deadline_at)
        if not isinstance(proposal, MacroWaveProposal):
            raise ValueError("proposal must be a MacroWaveProposal")
        if proposal.tree.state is not MacroWaveState.OPEN:
            raise ValueError("proposal must still be open")
        if parent in self.tree_by_parent:
            raise ValueError(f"parent request {parent} already has a physical tree wave")
        if parent in self.leaf_route:
            raise ValueError("a real parent ID cannot collide with a live virtual ID")

        paths = tuple(
            sorted(
                (_tokens(path, "candidate path") for path in proposal.candidate_paths),
                key=lambda path: (len(path), path),
            )
        )
        self._validate_leaf_paths(paths)
        covered_prefixes = {
            path[:depth]
            for path in paths
            for depth in range(len(path) + 1)
        }
        if covered_prefixes != set(proposal.prefixes()):
            raise ValueError("candidate leaves do not exactly cover the proposal tree")
        raw_ids = tuple(virtual_request_ids)
        if len(raw_ids) != len(paths):
            raise ValueError("one virtual request ID is required for every candidate leaf")
        virtual_ids = tuple(
            _bounded_int("virtual_request_id", value, maximum=UINT64_MAX)
            for value in raw_ids
        )
        if any(left >= right for left, right in zip(virtual_ids, virtual_ids[1:])):
            raise ValueError("virtual request IDs must be strictly increasing")
        if virtual_ids and virtual_ids[0] <= self._virtual_id_high_water:
            raise ValueError("virtual request IDs must be globally monotonic and never reused")
        if parent in virtual_ids:
            raise ValueError("a virtual request ID cannot equal its real parent ID")
        if any(value in self.tree_by_parent for value in virtual_ids):
            raise ValueError("a virtual request ID cannot equal a registered parent ID")

        visible_prefix = proposal.tree.base_prefix_tokens
        if not visible_prefix:
            raise ValueError("proposal base prefix must contain the pending token")
        if visible_prefix[-1] != pending:
            raise ValueError("pending_token must be the last visible proposal token")
        if base != len(visible_prefix) - 1:
            raise ValueError(
                "base_kv_tokens must equal visible proposal length minus pending token"
            )

        leaves: dict[tuple[int, ...], PhysicalLeaf] = {}
        routes: dict[int, PhysicalLeaf] = {}
        for ordinal, (path, virtual_id) in enumerate(zip(paths, virtual_ids)):
            leaf = PhysicalLeaf(
                path=path,
                virtual_request_id=virtual_id,
                inherited_step=step,
                expected_targets=1 + len(path),
                ordinal=ordinal,
            )
            leaves[path] = leaf
            routes[virtual_id] = leaf

        wave = PhysicalTreeWave(
            parent_request_id=parent,
            proposal=proposal,
            base_kv_tokens=base,
            pending_token=pending,
            inherited_step=step,
            leaves=leaves,
            virtual_routes=routes,
            target_argmax_by_prefix={},
            deadline_at=deadline,
        )
        self.tree_by_parent[parent] = wave
        for virtual_id, leaf in routes.items():
            self.leaf_route[virtual_id] = (parent, leaf)
        if virtual_ids:
            self._virtual_id_high_water = virtual_ids[-1]
        return wave

    def wave(self, parent_request_id: int) -> PhysicalTreeWave:
        parent = _bounded_int(
            "parent_request_id", parent_request_id, maximum=UINT64_MAX
        )
        try:
            return self.tree_by_parent[parent]
        except KeyError as exc:
            raise KeyError(f"parent request {parent} has no physical tree wave") from exc

    def next_fork_command(self, parent_request_id: int) -> ForkCommand | None:
        wave = self.wave(parent_request_id)
        if wave.phase is TreePhase.PREPARED:
            wave.phase = TreePhase.FORKING
        if wave.phase is not TreePhase.FORKING:
            raise PhysicalTreeError(f"cannot issue FORK while wave is {wave.phase.value}")
        for leaf in wave.ordered_leaves:
            if not leaf.fork_sent:
                leaf.fork_sent = True
                return ForkCommand(leaf.virtual_request_id, wave.parent_request_id)
        return None

    def next_verify_command(
        self,
        parent_request_id: int,
        *,
        now: float | None = None,
    ) -> VerifyCommand | None:
        wave = self.wave(parent_request_id)
        if wave.phase is TreePhase.FORKING:
            if any(not leaf.fork_sent for leaf in wave.ordered_leaves):
                raise PhysicalTreeError("all FORK commands must be issued before VERIFY")
            wave.phase = TreePhase.VERIFYING
        if wave.phase is not TreePhase.VERIFYING:
            raise PhysicalTreeError(f"cannot issue VERIFY while wave is {wave.phase.value}")
        issued_at = time.perf_counter() if now is None else _finite_time("now", now)
        for leaf in wave.ordered_leaves:
            if not leaf.verify_sent:
                leaf.verify_sent = True
                leaf.verify_sent_at = issued_at
                return VerifyCommand(
                    request_id=leaf.virtual_request_id,
                    step=wave.inherited_step,
                    input_tokens=(wave.pending_token, *leaf.path),
                )
        return None

    def accept_verify_result(
        self,
        virtual_request_id: int,
        *,
        step: int,
        target_tokens: Sequence[int],
        now: float | None = None,
    ) -> PhysicalTreeResult:
        """Record one virtual return and resolve only after every leaf returns."""

        virtual_id = _bounded_int(
            "virtual_request_id", virtual_request_id, maximum=UINT64_MAX
        )
        try:
            parent, leaf = self.leaf_route[virtual_id]
        except KeyError as exc:
            raise UnknownVirtualRequestError(
                f"unknown virtual request ID {virtual_id}"
            ) from exc
        wave = self.tree_by_parent[parent]
        observed_at = time.perf_counter() if now is None else _finite_time("now", now)
        if (
            wave.phase in (TreePhase.VERIFYING, TreePhase.DRAINING)
            and observed_at >= wave.deadline_at
            and any(
                current.verify_sent
                and current.result is None
                and not current.return_drained
                for current in wave.ordered_leaves
            )
        ):
            return self._timeout_plan(wave)

        if leaf.return_tombstone:
            if leaf.return_drained:
                raise PhysicalTreeProtocolError(
                    f"duplicate return for tombstoned virtual request {virtual_id}"
                )
            leaf.return_drained = True
            self.leaf_route.pop(virtual_id, None)
            drained = not self._outstanding_tombstones(wave)
            if drained and wave.cleanup_complete:
                wave.phase = TreePhase.ABORTED
            elif drained and wave.phase is TreePhase.DRAINING:
                wave.phase = TreePhase.ABORTED
            return TombstoneDrain(parent, virtual_id, drained)

        if wave.phase is not TreePhase.VERIFYING or not leaf.verify_sent:
            return self._abort_fatal(
                wave,
                f"VERIFY_RESULT for {virtual_id} arrived outside its verifying state",
            )
        if leaf.result is not None:
            return self._abort_fatal(
                wave,
                f"duplicate VERIFY_RESULT for virtual request {virtual_id}",
            )

        try:
            observed_step = _bounded_int("step", step, maximum=UINT32_MAX)
        except ValueError as exc:
            return self._abort_fatal(wave, f"invalid VERIFY_RESULT step: {exc}")
        if observed_step != leaf.inherited_step:
            return self._abort_fatal(
                wave,
                f"VERIFY_RESULT step {observed_step} != inherited step {leaf.inherited_step}",
            )
        try:
            targets = _tokens(target_tokens, "target_tokens")
        except (TypeError, ValueError) as exc:
            return self._abort_fatal(wave, f"invalid VERIFY_RESULT targets: {exc}")
        if len(targets) != leaf.expected_targets:
            return self._abort_fatal(
                wave,
                "VERIFY_RESULT target count "
                f"{len(targets)} != expected {leaf.expected_targets}",
            )
        leaf.result = targets
        if any(current.result is None for current in wave.ordered_leaves):
            return None
        return self._resolve(wave)

    def confirm_cleanup_command(
        self,
        parent_request_id: int,
        command: CleanupCommand,
    ) -> CancelCommand | None:
        """Confirm one locally-applied and wire-enqueued cleanup operation.

        Commands must be confirmed in the exact order in the returned plan.
        The optional return is a deferred parent cancellation which may only be
        issued after an in-progress promotion/truncation has completed.
        """

        wave = self.wave(parent_request_id)
        if wave.cleanup_complete:
            raise PhysicalTreeError("the wave has no cleanup command left to confirm")
        expected = wave.cleanup_commands[wave.cleanup_index]
        if command != expected:
            raise PhysicalTreeError(
                f"cleanup order violation: expected {expected!r}, got {command!r}"
            )

        if isinstance(command, EndCommand):
            leaf = wave.virtual_routes[command.request_id]
            leaf.physically_closed = True
            if not leaf.return_tombstone or leaf.return_drained:
                self.leaf_route.pop(command.request_id, None)
        elif isinstance(command, PromoteCommand):
            leaf = wave.virtual_routes[command.child_request_id]
            leaf.physically_closed = True
            self.leaf_route.pop(command.child_request_id, None)

        wave.cleanup_index += 1
        if not wave.cleanup_complete:
            following = wave.cleanup_commands[wave.cleanup_index]
            if isinstance(following, PromoteCommand):
                wave.phase = TreePhase.PROMOTING
            elif isinstance(following, TruncateCommand):
                wave.phase = TreePhase.TRUNCATING
            elif wave.abort_reason is not None:
                wave.phase = TreePhase.ABORTING
            else:
                wave.phase = TreePhase.CLEANING
            return None

        if wave.abort_reason is not None:
            wave.phase = (
                TreePhase.DRAINING
                if self._outstanding_tombstones(wave)
                else TreePhase.ABORTED
            )
            if wave.cancel_requested and not wave.parent_cancel_sent:
                wave.parent_cancel_sent = True
                return CancelCommand(wave.parent_request_id)
            return None
        wave.phase = TreePhase.COMMITTED
        if wave.cancel_requested:
            wave.phase = TreePhase.ABORTED
            wave.parent_cancel_sent = True
            return CancelCommand(wave.parent_request_id)
        return None

    def request_cancel(self, parent_request_id: int) -> CancellationPlan:
        """Plan child-before-parent cancellation and retain in-flight tombstones."""

        wave = self.wave(parent_request_id)
        if wave.phase in (
            TreePhase.CLEANING,
            TreePhase.PROMOTING,
            TreePhase.TRUNCATING,
            TreePhase.ABORTING,
        ):
            wave.cancel_requested = True
            return CancellationPlan(wave.parent_request_id, (), True, ())
        if wave.phase is TreePhase.DRAINING:
            if wave.parent_cancel_sent or wave.abort_reason == "cancelled":
                return CancellationPlan(
                    wave.parent_request_id,
                    (),
                    False,
                    self._outstanding_tombstones(wave),
                )
            wave.cancel_requested = True
            wave.parent_cancel_sent = True
            return CancellationPlan(
                wave.parent_request_id,
                (CancelCommand(wave.parent_request_id),),
                False,
                self._outstanding_tombstones(wave),
            )
        if wave.phase in (TreePhase.COMMITTED,):
            wave.phase = TreePhase.ABORTED
            wave.cancel_requested = True
            wave.parent_cancel_sent = True
            return CancellationPlan(
                wave.parent_request_id,
                (CancelCommand(wave.parent_request_id),),
                False,
                (),
            )
        if wave.phase is TreePhase.ABORTED:
            return CancellationPlan(wave.parent_request_id, (), False, ())

        created = tuple(
            sorted(
                (leaf for leaf in wave.ordered_leaves if leaf.fork_sent),
                key=lambda leaf: leaf.virtual_request_id,
            )
        )
        commands = tuple(
            [*(CancelCommand(leaf.virtual_request_id) for leaf in created),
             CancelCommand(wave.parent_request_id)]
        )
        draining: list[int] = []
        for leaf in wave.ordered_leaves:
            if leaf.verify_sent and leaf.result is None:
                leaf.return_tombstone = True
                draining.append(leaf.virtual_request_id)
            else:
                self.leaf_route.pop(leaf.virtual_request_id, None)
        self._rollback_if_open(wave)
        wave.abort_reason = "cancelled"
        wave.cancel_requested = True
        wave.parent_cancel_sent = True
        wave.phase = TreePhase.DRAINING if draining else TreePhase.ABORTED
        return CancellationPlan(
            wave.parent_request_id,
            commands,
            False,
            tuple(sorted(draining)),
        )

    def abort_before_wire(
        self,
        parent_request_id: int,
        reason: str,
    ) -> TreeAbortPlan:
        """Abort a locally rejected wave while fallback on this route is safe.

        This is intentionally the only ``fallback_allowed`` abort path.  It is
        legal solely before a FORK has been issued, so every physical stage and
        the real parent's KV are demonstrably untouched.
        """

        wave = self.wave(parent_request_id)
        if wave.phase is not TreePhase.PREPARED or any(
            leaf.fork_sent for leaf in wave.ordered_leaves
        ):
            raise PhysicalTreeError("same-route fallback is only safe before first FORK")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must be a non-empty string")
        self._rollback_if_open(wave)
        wave.abort_reason = reason.strip()
        wave.phase = TreePhase.ABORTED
        for leaf in wave.ordered_leaves:
            self.leaf_route.pop(leaf.virtual_request_id, None)
        return TreeAbortPlan(
            parent_request_id=wave.parent_request_id,
            reason=wave.abort_reason,
            commands=(),
            route_fatal=False,
        )

    def check_timeouts(self, now: float) -> tuple[TreeAbortPlan, ...]:
        """Fail the entire route when the oldest virtual return misses deadline."""

        observed_at = _finite_time("now", now)
        expired: list[TreeAbortPlan] = []
        for wave in tuple(self.tree_by_parent.values()):
            if wave.phase not in (TreePhase.VERIFYING, TreePhase.DRAINING):
                continue
            outstanding = [
                leaf
                for leaf in wave.ordered_leaves
                if leaf.verify_sent and leaf.result is None and not leaf.return_drained
            ]
            if not outstanding or observed_at < wave.deadline_at:
                continue
            oldest = min(
                outstanding,
                key=lambda leaf: (
                    float("inf") if leaf.verify_sent_at is None else leaf.verify_sent_at,
                    leaf.ordinal,
                ),
            )
            expired.append(self._timeout_plan(wave, oldest=oldest))
        return tuple(expired)

    def retire_wave(self, parent_request_id: int) -> PhysicalTreeWave:
        """Remove a terminal wave while retaining the global virtual-ID ledger."""

        wave = self.wave(parent_request_id)
        if wave.phase not in (TreePhase.COMMITTED, TreePhase.ABORTED):
            raise PhysicalTreeError(f"cannot retire a wave while it is {wave.phase.value}")
        if any(virtual_id in self.leaf_route for virtual_id in wave.virtual_routes):
            raise PhysicalTreeError("cannot retire a wave with live virtual routes")
        return self.tree_by_parent.pop(wave.parent_request_id)

    def _resolve(self, wave: PhysicalTreeWave) -> TreeCommitPlan | TreeAbortPlan:
        wave.phase = TreePhase.RESOLVING
        target: dict[tuple[int, ...], int] = {}
        for leaf in wave.ordered_leaves:
            if leaf.result is None:
                return self._abort_fatal(wave, "a physical leaf result is missing")
            for index, token in enumerate(leaf.result):
                prefix = leaf.path[:index]
                existing = target.get(prefix)
                if existing is not None and existing != token:
                    return self._abort_fatal(
                        wave,
                        "contradictory target argmax for shared prefix "
                        f"{prefix!r}: {existing} != {token}",
                    )
                target[prefix] = token

        expected_prefixes = set(wave.proposal.prefixes())
        if set(target) != expected_prefixes:
            missing = sorted(expected_prefixes - set(target), key=lambda item: (len(item), item))
            extra = sorted(set(target) - expected_prefixes, key=lambda item: (len(item), item))
            return self._abort_fatal(
                wave,
                f"physical targets do not cover the proposal; missing={missing!r}, extra={extra!r}",
            )
        wave.target_argmax_by_prefix = target
        try:
            resolution = resolve_macro_wave(wave.proposal, target)
        except (KeyError, RuntimeError, ValueError) as exc:
            return self._abort_fatal(wave, f"exact MacroWave resolution failed: {exc}")

        accepted = resolution.commit_tokens
        carriers = [
            leaf
            for leaf in wave.ordered_leaves
            if leaf.path[: len(accepted)] == accepted
        ]
        if not carriers:
            raise PhysicalTreeError("exact resolution has no physical carrier leaf")
        carrier = min(carriers, key=lambda leaf: leaf.virtual_request_id)
        losers = sorted(
            (leaf for leaf in wave.ordered_leaves if leaf is not carrier),
            key=lambda leaf: leaf.virtual_request_id,
        )
        commands: list[CleanupCommand] = [
            EndCommand(leaf.virtual_request_id) for leaf in losers
        ]
        commands.append(
            PromoteCommand(wave.parent_request_id, carrier.virtual_request_id)
        )
        if resolution.truncate_required:
            commands.append(
                TruncateCommand(
                    wave.parent_request_id,
                    wave.base_kv_tokens + 1 + resolution.accepted_draft_tokens,
                )
            )

        wave.resolution = resolution
        wave.carrier_request_id = carrier.virtual_request_id
        wave.cleanup_commands = tuple(commands)
        wave.cleanup_index = 0
        wave.phase = (
            TreePhase.CLEANING
            if isinstance(wave.cleanup_commands[0], EndCommand)
            else TreePhase.PROMOTING
        )
        ordered_targets = tuple(
            sorted(target.items(), key=lambda item: (len(item[0]), item[0]))
        )
        return TreeCommitPlan(
            parent_request_id=wave.parent_request_id,
            resolution=resolution,
            carrier_request_id=carrier.virtual_request_id,
            target_argmax_by_prefix=ordered_targets,
            commands=wave.cleanup_commands,
            next_step=wave.inherited_step + 1,
        )

    def _abort_fatal(self, wave: PhysicalTreeWave, reason: str) -> TreeAbortPlan:
        """Quarantine a route whose physical return contract is inconsistent."""

        self._rollback_if_open(wave)
        wave.abort_reason = reason
        wave.route_fatal = True
        wave.cleanup_commands = ()
        wave.cleanup_index = 0
        wave.phase = TreePhase.ABORTED
        for leaf in wave.ordered_leaves:
            self.leaf_route.pop(leaf.virtual_request_id, None)
        return TreeAbortPlan(
            parent_request_id=wave.parent_request_id,
            reason=reason,
            commands=(),
            route_fatal=True,
        )

    def _timeout_plan(
        self,
        wave: PhysicalTreeWave,
        *,
        oldest: PhysicalLeaf | None = None,
    ) -> TreeAbortPlan:
        outstanding = [
            leaf
            for leaf in wave.ordered_leaves
            if leaf.verify_sent and leaf.result is None and not leaf.return_drained
        ]
        if not outstanding:
            raise PhysicalTreeError("cannot time out a wave with no outstanding return")
        if oldest is None:
            oldest = min(
                outstanding,
                key=lambda leaf: (
                    float("inf") if leaf.verify_sent_at is None else leaf.verify_sent_at,
                    leaf.ordinal,
                ),
            )
        reason = (
            "physical tree VERIFY_RESULT timeout for oldest virtual request "
            f"{oldest.virtual_request_id}"
        )
        plan = self._abort_fatal(wave, reason)
        return TreeAbortPlan(
            parent_request_id=plan.parent_request_id,
            reason=plan.reason,
            commands=plan.commands,
            route_fatal=True,
            oldest_virtual_request_id=oldest.virtual_request_id,
        )

    @staticmethod
    def _validate_leaf_paths(paths: tuple[tuple[int, ...], ...]) -> None:
        if not paths or any(not path for path in paths):
            raise ValueError("candidate paths must contain non-empty leaves")
        if len(set(paths)) != len(paths):
            raise ValueError("candidate leaf paths must not contain duplicates")
        for index, path in enumerate(paths):
            for other_index, other in enumerate(paths):
                if (
                    index != other_index
                    and len(path) < len(other)
                    and other[: len(path)] == path
                ):
                    raise ValueError("one candidate leaf path cannot prefix another")

    @staticmethod
    def _rollback_if_open(wave: PhysicalTreeWave) -> None:
        if wave.proposal.tree.state is MacroWaveState.OPEN:
            wave.proposal.tree.rollback()

    @staticmethod
    def _outstanding_tombstones(wave: PhysicalTreeWave) -> tuple[int, ...]:
        return tuple(
            leaf.virtual_request_id
            for leaf in wave.ordered_leaves
            if leaf.return_tombstone and not leaf.return_drained
        )


__all__ = [
    "CancelCommand",
    "CancellationPlan",
    "CleanupCommand",
    "EndCommand",
    "ForkCommand",
    "PhysicalLeaf",
    "PhysicalTreeCommand",
    "PhysicalTreeCoordinator",
    "PhysicalTreeError",
    "PhysicalTreeProtocolError",
    "PhysicalTreeResult",
    "PhysicalTreeWave",
    "PromoteCommand",
    "TombstoneDrain",
    "TreeAbortPlan",
    "TreeCommitPlan",
    "TreePhase",
    "TruncateCommand",
    "UnknownVirtualRequestError",
    "VerifyCommand",
]
