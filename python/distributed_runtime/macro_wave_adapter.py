"""Compatibility bridge from linear speculation to exact MacroWave trees.

The existing speculative decoder exposes a linear ``DraftProvider`` and an
``AdaptiveSpeculationController``.  MacroWave represents candidates as a
prefix tree.  This module keeps those layers independent while giving callers
one explicit translation contract:

* a legacy draft becomes a tree with exactly one candidate path (width one);
* optional alternative paths share their common prefixes in the same tree;
* exact verification returns separately named commit, truncate, correction,
  and bonus values; and
* the existing linear controller can still select and record a wave unchanged.

No tensor or wire behaviour lives here. ``DistributedPipelineEngine`` now uses
this adapter for physical linear waves and drives its KV/TRUNCATE operations
from the returned commit. The general branching tree stays logical until a
paged KV backend can materialize multiple branches in every stage.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from numbers import Integral

from .macro_wave import (
    BranchIdentity,
    GreedyAcceptance,
    KVVersion,
    MacroWaveCommit,
    MacroWaveTree,
    WaveIdentity,
    verify_greedy_exact,
)
from .speculation import (
    AdaptiveSpeculationController,
    DraftProvider,
    SpeculationDecision,
    TreeDraftProvider,
)


def _tokens(values: Sequence[int], name: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError(f"{name} must be a sequence of token ids")
    normalized: list[int] = []
    for value in values:
        if not isinstance(value, Integral) or isinstance(value, bool) or int(value) < 0:
            raise ValueError(f"{name} must contain non-negative integer token ids")
        normalized.append(int(value))
    return tuple(normalized)


def _count(name: str, value: object, *, minimum: int = 0) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool) or int(value) < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _strategy(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("strategy must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class MacroWaveProposal:
    """A candidate tree plus a stable token-prefix to branch index."""

    tree: MacroWaveTree
    candidate_paths: tuple[tuple[int, ...], ...]
    strategy: str
    _prefix_branches: tuple[tuple[tuple[int, ...], BranchIdentity], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.tree, MacroWaveTree):
            raise ValueError("tree must be a MacroWaveTree")
        if not self.candidate_paths:
            raise ValueError("candidate_paths must not be empty")
        object.__setattr__(self, "strategy", _strategy(self.strategy))

    @property
    def wave_identity(self) -> WaveIdentity:
        return self.tree.identity

    @property
    def width(self) -> int:
        """Maximum sibling count at any prefix in the candidate tree."""

        return max(
            len(self.tree.children(branch.identity))
            for branch in self.tree.branches()
        )

    @property
    def max_depth(self) -> int:
        return max(len(path) for path in self.candidate_paths)

    @property
    def is_linear(self) -> bool:
        return len(self.candidate_paths) == 1 and self.width == 1

    @property
    def linear_tokens(self) -> tuple[int, ...]:
        if not self.is_linear:
            raise ValueError("proposal is branched, not linear")
        return self.candidate_paths[0]

    def branch_for_prefix(self, prefix_tokens: Sequence[int]) -> BranchIdentity:
        prefix = _tokens(prefix_tokens, "prefix_tokens")
        for observed_prefix, branch in self._prefix_branches:
            if observed_prefix == prefix:
                return branch
        raise KeyError(f"candidate prefix is not present in this wave: {prefix!r}")

    def prefixes(self) -> tuple[tuple[int, ...], ...]:
        return tuple(prefix for prefix, _branch in self._prefix_branches)


class ContinuationKind(str, Enum):
    """Why exact verification produced the one non-draft output token."""

    CORRECTION = "correction"
    BONUS = "bonus"


@dataclass(frozen=True)
class MacroWaveResolution:
    """Unambiguous result of verifying and committing one MacroWave.

    ``commit_tokens`` are draft tokens whose KV may be kept.  Every draft token
    after ``truncate_draft_to`` must be discarded.  Exactly one of
    ``correction_token`` and ``bonus_token`` is populated.  That token is
    visible output but remains the pending token for the next target step.
    """

    acceptance: GreedyAcceptance
    commit: MacroWaveCommit
    commit_tokens: tuple[int, ...]
    truncate_draft_to: int
    truncate_required: bool
    continuation_kind: ContinuationKind
    correction_token: int | None
    bonus_token: int | None
    emitted_tokens: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.commit_tokens != self.acceptance.accepted_tokens:
            raise ValueError("commit_tokens must equal the exact accepted draft prefix")
        if self.truncate_draft_to != len(self.commit_tokens):
            raise ValueError("truncate_draft_to must equal the committed draft count")
        if self.emitted_tokens != self.acceptance.emitted_tokens:
            raise ValueError("emitted_tokens must equal the exact greedy output")
        populated = int(self.correction_token is not None) + int(
            self.bonus_token is not None
        )
        if populated != 1:
            raise ValueError("exactly one correction_token or bonus_token is required")
        if self.continuation_kind is ContinuationKind.BONUS:
            if self.bonus_token != self.acceptance.continuation_token:
                raise ValueError("bonus_token must be the accepted leaf continuation")
            if self.correction_token is not None or self.truncate_required:
                raise ValueError("a bonus cannot also request correction or truncation")
        else:
            if self.correction_token != self.acceptance.continuation_token:
                raise ValueError("correction_token must be the mismatch continuation")
            if self.bonus_token is not None or not self.truncate_required:
                raise ValueError("a correction must request truncation")

    @property
    def pending_token(self) -> int:
        return self.acceptance.continuation_token

    @property
    def accepted_draft_tokens(self) -> int:
        return len(self.commit_tokens)


@dataclass(frozen=True)
class MacroWavePreparation:
    """Controller decision and the optional width-one wave it selected."""

    decision: SpeculationDecision
    available_draft_tokens: tuple[int, ...]
    selected_draft_tokens: tuple[int, ...]
    proposal: MacroWaveProposal | None
    is_probe: bool = False

    def __post_init__(self) -> None:
        if (self.proposal is None) != (not self.selected_draft_tokens):
            raise ValueError("proposal presence must match selected_draft_tokens")
        if self.proposal is not None:
            if not self.proposal.is_linear:
                raise ValueError("the legacy controller may only prepare a linear wave")
            if self.proposal.linear_tokens != self.selected_draft_tokens:
                raise ValueError("proposal tokens must match the selected draft")
        if self.is_probe and self.decision.enabled:
            raise ValueError("an enabled controller decision is not a probe")

    @property
    def enabled(self) -> bool:
        return self.proposal is not None


def branched_candidates_to_macro_wave(
    candidate_paths: Sequence[Sequence[int]],
    *,
    request_id: str | int,
    ordinal: int,
    base_prefix_tokens: Sequence[int],
    parent_kv_version: KVVersion = KVVersion(0),
    strategy: str = "external",
) -> MacroWaveProposal:
    """Build one exact prefix tree from one or more candidate token paths."""

    if isinstance(candidate_paths, (str, bytes, bytearray)):
        raise ValueError("candidate_paths must be a sequence of token paths")
    paths = tuple(_tokens(path, "candidate path") for path in candidate_paths)
    if not paths or any(not path for path in paths):
        raise ValueError("candidate_paths must contain non-empty token paths")
    if len(set(paths)) != len(paths):
        raise ValueError("candidate_paths must not contain duplicates")
    for index, path in enumerate(paths):
        for other_index, other in enumerate(paths):
            if index != other_index and len(path) < len(other) and other[: len(path)] == path:
                raise ValueError(
                    "candidate paths must be leaves; one path cannot prefix another"
                )

    prefix = _tokens(base_prefix_tokens, "base_prefix_tokens")
    if not isinstance(parent_kv_version, KVVersion):
        raise ValueError("parent_kv_version must be a KVVersion")
    tree = MacroWaveTree.create(
        request_id,
        _count("ordinal", ordinal),
        prefix,
        parent_kv_version,
    )
    for path in paths:
        tree.ensure_path(path)

    prefix_branches = tuple(
        sorted(
            (
                (tree.candidate_tokens(branch.identity), branch.identity)
                for branch in tree.branches()
            ),
            key=lambda item: (len(item[0]), item[0]),
        )
    )
    return MacroWaveProposal(
        tree=tree,
        candidate_paths=paths,
        strategy=_strategy(strategy),
        _prefix_branches=prefix_branches,
    )


def linear_draft_to_macro_wave(
    draft_tokens: Sequence[int],
    *,
    request_id: str | int,
    ordinal: int,
    base_prefix_tokens: Sequence[int],
    parent_kv_version: KVVersion = KVVersion(0),
    strategy: str = "legacy-linear",
) -> MacroWaveProposal:
    """Convert a current linear draft into an exact width-one MacroWave."""

    draft = _tokens(draft_tokens, "draft_tokens")
    if not draft:
        raise ValueError("draft_tokens must not be empty")
    proposal = branched_candidates_to_macro_wave(
        (draft,),
        request_id=request_id,
        ordinal=ordinal,
        base_prefix_tokens=base_prefix_tokens,
        parent_kv_version=parent_kv_version,
        strategy=strategy,
    )
    if proposal.width != 1:
        raise RuntimeError("a linear draft must produce a width-one MacroWave")
    return proposal


def prepare_tree_macro_wave(
    provider: TreeDraftProvider,
    token_history: Sequence[int],
    *,
    request_id: str | int,
    ordinal: int,
    parent_kv_version: KVVersion = KVVersion(0),
    max_tokens: int | None = None,
    max_branches: int | None = None,
) -> MacroWaveProposal | None:
    """Ask a tree drafter for bounded leaves and build an exact proposal.

    This function is intentionally policy-free: a runtime controller may
    lower ``max_tokens`` or ``max_branches`` from latency, load and KV budget
    measurements before calling it.  Every provider limit is checked again so
    an untrusted or buggy drafter cannot silently exceed the sealed wave.
    """

    if not isinstance(provider, TreeDraftProvider):
        raise ValueError("provider must implement TreeDraftProvider")
    history = _tokens(token_history, "token_history")
    token_limit = (
        int(provider.max_draft_tokens)
        if max_tokens is None
        else min(_count("max_tokens", max_tokens), int(provider.max_draft_tokens))
    )
    branch_limit = (
        int(provider.max_branches)
        if max_branches is None
        else min(_count("max_branches", max_branches), int(provider.max_branches))
    )
    if token_limit == 0 or branch_limit == 0:
        return None

    raw_paths = provider.draft_paths(
        history,
        max_tokens=token_limit,
        max_branches=branch_limit,
    )
    if isinstance(raw_paths, (str, bytes, bytearray)):
        raise ValueError("provider paths must be a sequence of token paths")
    paths = tuple(_tokens(path, "provider candidate path") for path in raw_paths)
    if not paths:
        return None
    if len(paths) > branch_limit:
        raise ValueError("provider returned more paths than max_branches")
    if any(len(path) > token_limit for path in paths):
        raise ValueError("provider returned a path longer than max_tokens")

    return branched_candidates_to_macro_wave(
        paths,
        request_id=request_id,
        ordinal=ordinal,
        base_prefix_tokens=history,
        parent_kv_version=parent_kv_version,
        strategy=provider.strategy,
    )


def prepare_linear_macro_wave(
    provider: DraftProvider,
    controller: AdaptiveSpeculationController,
    token_history: Sequence[int],
    *,
    request_id: str | int,
    ordinal: int,
    parent_kv_version: KVVersion = KVVersion(0),
    max_tokens: int | None = None,
    allow_probe: bool = False,
) -> MacroWavePreparation:
    """Run the current provider/controller gate and adapt its selected draft.

    ``allow_probe`` mirrors the engine's explicit warm-up probe path.  It never
    changes the controller's decision; ``is_probe`` identifies that override.
    """

    if not isinstance(provider, DraftProvider):
        raise ValueError("provider must implement DraftProvider")
    if not isinstance(controller, AdaptiveSpeculationController):
        raise ValueError("controller must be an AdaptiveSpeculationController")
    history = _tokens(token_history, "token_history")
    if max_tokens is not None:
        limit = _count("max_tokens", max_tokens)
    else:
        limit = None

    available = _tokens(
        provider.draft(history, max_tokens=limit),
        "provider draft",
    )
    if limit is not None and len(available) > limit:
        raise ValueError("provider returned more tokens than max_tokens")
    if len(available) > int(provider.max_draft_tokens):
        raise ValueError("provider returned more tokens than max_draft_tokens")

    decision = controller.decide(
        history_tokens=len(history),
        available_draft_tokens=len(available),
    )
    selected_size = decision.candidate_size if decision.enabled else 0
    is_probe = False
    if selected_size == 0 and allow_probe:
        selected_size = controller.next_probe_size(
            history_tokens=len(history),
            available_draft_tokens=len(available),
        ) or 0
        is_probe = selected_size > 0
    if selected_size > len(available):
        raise ValueError("controller selected more draft tokens than are available")

    selected = available[:selected_size]
    proposal = None
    if selected:
        proposal = linear_draft_to_macro_wave(
            selected,
            request_id=request_id,
            ordinal=ordinal,
            base_prefix_tokens=history,
            parent_kv_version=parent_kv_version,
            strategy=provider.strategy,
        )
    return MacroWavePreparation(
        decision=decision,
        available_draft_tokens=available,
        selected_draft_tokens=selected,
        proposal=proposal,
        is_probe=is_probe,
    )


def resolve_macro_wave(
    proposal: MacroWaveProposal,
    target_argmax_by_prefix: Mapping[tuple[int, ...], int],
) -> MacroWaveResolution:
    """Verify prefix-keyed target argmax values and commit the exact path."""

    if not isinstance(proposal, MacroWaveProposal):
        raise ValueError("proposal must be a MacroWaveProposal")
    if not isinstance(target_argmax_by_prefix, Mapping):
        raise ValueError("target_argmax_by_prefix must be a mapping")

    known_prefixes = set(proposal.prefixes())
    by_branch: dict[BranchIdentity, int] = {}
    for raw_prefix, target_token in target_argmax_by_prefix.items():
        prefix = _tokens(raw_prefix, "target prefix")
        if prefix not in known_prefixes:
            raise ValueError(f"target argmax contains an unknown candidate prefix: {prefix!r}")
        branch = proposal.branch_for_prefix(prefix)
        if branch in by_branch:
            raise ValueError(f"duplicate target argmax for candidate prefix: {prefix!r}")
        by_branch[branch] = target_token

    acceptance = verify_greedy_exact(proposal.tree, by_branch)
    commit = proposal.tree.commit_greedy(acceptance)
    is_bonus = acceptance.stop_reason == "candidate_exhausted"
    kind = ContinuationKind.BONUS if is_bonus else ContinuationKind.CORRECTION
    return MacroWaveResolution(
        acceptance=acceptance,
        commit=commit,
        commit_tokens=acceptance.accepted_tokens,
        truncate_draft_to=len(acceptance.accepted_tokens),
        truncate_required=not is_bonus,
        continuation_kind=kind,
        correction_token=None if is_bonus else acceptance.continuation_token,
        bonus_token=acceptance.continuation_token if is_bonus else None,
        emitted_tokens=acceptance.emitted_tokens,
    )


def resolve_linear_macro_wave(
    proposal: MacroWaveProposal,
    target_tokens: Sequence[int],
) -> MacroWaveResolution:
    """Resolve the current ``draft length + 1`` verification vector exactly."""

    if not isinstance(proposal, MacroWaveProposal) or not proposal.is_linear:
        raise ValueError("proposal must be a linear MacroWaveProposal")
    draft = proposal.linear_tokens
    targets = _tokens(target_tokens, "target_tokens")
    if len(targets) != len(draft) + 1:
        raise ValueError(
            "verification target vector must contain one prediction per draft plus bonus"
        )
    target_by_prefix = {
        draft[:index]: target
        for index, target in enumerate(targets)
    }
    return resolve_macro_wave(proposal, target_by_prefix)


def record_linear_resolution(
    controller: AdaptiveSpeculationController,
    proposal: MacroWaveProposal,
    resolution: MacroWaveResolution,
    *,
    latency_seconds: float,
    transferred_bytes: int,
) -> None:
    """Feed a width-one MacroWave result into the unchanged legacy controller."""

    if not isinstance(controller, AdaptiveSpeculationController):
        raise ValueError("controller must be an AdaptiveSpeculationController")
    if not isinstance(proposal, MacroWaveProposal) or not proposal.is_linear:
        raise ValueError("the legacy controller can only record a linear proposal")
    if not isinstance(resolution, MacroWaveResolution):
        raise ValueError("resolution must be a MacroWaveResolution")
    if resolution.acceptance.wave_identity != proposal.wave_identity:
        raise ValueError("resolution belongs to another proposal")
    controller.record_verification(
        proposed_tokens=len(proposal.linear_tokens),
        accepted_tokens=resolution.accepted_draft_tokens,
        latency_seconds=latency_seconds,
        transferred_bytes=transferred_bytes,
    )


__all__ = [
    "ContinuationKind",
    "MacroWavePreparation",
    "MacroWaveProposal",
    "MacroWaveResolution",
    "branched_candidates_to_macro_wave",
    "linear_draft_to_macro_wave",
    "prepare_tree_macro_wave",
    "prepare_linear_macro_wave",
    "record_linear_resolution",
    "resolve_linear_macro_wave",
    "resolve_macro_wave",
]
