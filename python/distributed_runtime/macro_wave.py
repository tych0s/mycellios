"""Logical foundation for exact speculative MacroWave execution.

This module deliberately has no dependency on the current pipeline engine or
wire protocol.  It models the pieces those integrations will eventually need:

* stable wave and branch identities;
* a prefix tree whose branches fork logical KV versions;
* copy-on-write commit/rollback without copying tensor payloads;
* token-exact greedy verification of a candidate tree; and
* a cost controller that chooses wave depth and width from live measurements.

The KV ledger stores parent links and token deltas only.  A later tensor-backed
implementation can attach paged KV handles to the same versions without
changing the tree or verification contracts defined here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import hashlib
import math
from numbers import Integral


def _integer(name: str, value: object, *, minimum: int = 0) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    normalized = int(value)
    if normalized < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return normalized


def _finite(
    name: str,
    value: object,
    *,
    minimum: float = 0.0,
    strictly_positive: bool = False,
) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(normalized) or normalized < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    if strictly_positive and normalized == 0.0:
        raise ValueError(f"{name} must be positive")
    return normalized


def _token_id(value: object, name: str = "token_id") -> int:
    return _integer(name, value, minimum=0)


def _token_tuple(values: Sequence[int], name: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError(f"{name} must be a sequence of token ids")
    return tuple(_token_id(value, f"{name} item") for value in values)


@dataclass(frozen=True, order=True)
class KVVersion:
    """Stable identifier for one logical KV snapshot."""

    value: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _integer("KV version", self.value))

    def __str__(self) -> str:
        return f"kv:{self.value}"


@dataclass(frozen=True, order=True)
class WaveIdentity:
    """Deterministic identity for one request's speculative wave."""

    request_id: str
    ordinal: int
    prefix_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ValueError("request_id must be a non-empty string")
        object.__setattr__(self, "request_id", self.request_id.strip())
        object.__setattr__(self, "ordinal", _integer("ordinal", self.ordinal))
        digest = str(self.prefix_digest).lower()
        if len(digest) != 32 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("prefix_digest must be 32 lowercase hexadecimal characters")
        object.__setattr__(self, "prefix_digest", digest)

    @classmethod
    def for_prefix(
        cls,
        request_id: str | int,
        ordinal: int,
        prefix_tokens: Sequence[int],
        parent_kv_version: KVVersion,
    ) -> WaveIdentity:
        if isinstance(request_id, bool) or not isinstance(request_id, (str, int)):
            raise ValueError("request_id must be a string or integer")
        normalized_request = str(request_id).strip()
        if not normalized_request:
            raise ValueError("request_id must not be empty")
        normalized_ordinal = _integer("ordinal", ordinal)
        normalized_prefix = _token_tuple(prefix_tokens, "prefix_tokens")
        if not isinstance(parent_kv_version, KVVersion):
            raise ValueError("parent_kv_version must be a KVVersion")

        digest = hashlib.blake2b(digest_size=16)
        digest.update(b"gdlp-macro-wave-prefix-v1\0")
        digest.update(parent_kv_version.value.to_bytes(8, "big", signed=False))
        digest.update(len(normalized_prefix).to_bytes(8, "big", signed=False))
        for token in normalized_prefix:
            digest.update(token.to_bytes(8, "big", signed=False))
        return cls(normalized_request, normalized_ordinal, digest.hexdigest())

    @property
    def key(self) -> str:
        return f"{self.request_id}:{self.ordinal}:{self.prefix_digest}"


@dataclass(frozen=True, order=True)
class BranchIdentity:
    """Identity of a node in a wave tree.

    ``path`` contains child slots, not token ids.  Two siblings may therefore
    never collide even if a future tree-building policy changes token ordering.
    """

    wave: WaveIdentity
    path: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.wave, WaveIdentity):
            raise ValueError("wave must be a WaveIdentity")
        normalized = tuple(
            _integer("branch path slot", slot) for slot in self.path
        )
        object.__setattr__(self, "path", normalized)

    @property
    def depth(self) -> int:
        return len(self.path)

    @property
    def parent(self) -> BranchIdentity | None:
        if not self.path:
            return None
        return BranchIdentity(self.wave, self.path[:-1])

    def child(self, slot: int) -> BranchIdentity:
        return BranchIdentity(self.wave, (*self.path, _integer("child slot", slot)))

    @property
    def key(self) -> str:
        suffix = "root" if not self.path else ".".join(str(slot) for slot in self.path)
        return f"{self.wave.key}/{suffix}"


class KVSnapshotState(str, Enum):
    COMMITTED = "committed"
    SPECULATIVE = "speculative"
    ROLLED_BACK = "rolled-back"


class MacroBranchState(str, Enum):
    COMMITTED = "committed"
    SPECULATIVE = "speculative"
    ROLLED_BACK = "rolled-back"


class MacroWaveState(str, Enum):
    OPEN = "open"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled-back"


@dataclass(frozen=True)
class LogicalKVSnapshot:
    version: KVVersion
    parent_version: KVVersion | None
    delta_tokens: tuple[int, ...]
    owner_branch: BranchIdentity | None
    state: KVSnapshotState


@dataclass
class _KVRecord:
    version: KVVersion
    parent_version: KVVersion | None
    delta_tokens: tuple[int, ...]
    owner_branch: BranchIdentity | None
    state: KVSnapshotState

    def public(self) -> LogicalKVSnapshot:
        return LogicalKVSnapshot(
            version=self.version,
            parent_version=self.parent_version,
            delta_tokens=self.delta_tokens,
            owner_branch=self.owner_branch,
            state=self.state,
        )


@dataclass(frozen=True)
class KVFinalizeResult:
    committed_head: KVVersion
    newly_committed: tuple[KVVersion, ...]
    rolled_back: tuple[KVVersion, ...]


class LogicalKVLedger:
    """Copy-on-write ledger containing references and token deltas only."""

    def __init__(
        self,
        base_prefix_tokens: Sequence[int],
        base_version: KVVersion = KVVersion(0),
    ) -> None:
        if not isinstance(base_version, KVVersion):
            raise ValueError("base_version must be a KVVersion")
        prefix = _token_tuple(base_prefix_tokens, "base_prefix_tokens")
        self._base_version = base_version
        self._head_version = base_version
        self._next_version = base_version.value + 1
        self._records: dict[KVVersion, _KVRecord] = {
            base_version: _KVRecord(
                version=base_version,
                parent_version=None,
                delta_tokens=prefix,
                owner_branch=None,
                state=KVSnapshotState.COMMITTED,
            )
        }
        self._finalized = False

    @property
    def base_version(self) -> KVVersion:
        return self._base_version

    @property
    def head_version(self) -> KVVersion:
        return self._head_version

    @property
    def physical_copy_count(self) -> int:
        """Tensor copies performed by this logical implementation (always zero)."""

        return 0

    @property
    def finalized(self) -> bool:
        return self._finalized

    def snapshots(self) -> tuple[LogicalKVSnapshot, ...]:
        return tuple(
            self._records[version].public()
            for version in sorted(self._records, key=lambda item: item.value)
        )

    def snapshot(self, version: KVVersion) -> LogicalKVSnapshot:
        return self._record(version).public()

    def fork(
        self,
        parent_version: KVVersion,
        *,
        owner_branch: BranchIdentity,
        token_id: int,
    ) -> KVVersion:
        if self._finalized:
            raise RuntimeError("KV ledger is finalized")
        if not isinstance(owner_branch, BranchIdentity):
            raise ValueError("owner_branch must be a BranchIdentity")
        parent = self._record(parent_version)
        if parent.state is KVSnapshotState.ROLLED_BACK:
            raise RuntimeError("cannot fork a rolled-back KV version")
        token = _token_id(token_id)
        version = KVVersion(self._next_version)
        self._next_version += 1
        self._records[version] = _KVRecord(
            version=version,
            parent_version=parent.version,
            delta_tokens=(token,),
            owner_branch=owner_branch,
            state=KVSnapshotState.SPECULATIVE,
        )
        return version

    def resolve_tokens(self, version: KVVersion) -> tuple[int, ...]:
        record = self._record(version)
        chunks: list[tuple[int, ...]] = []
        visited: set[KVVersion] = set()
        while True:
            if record.version in visited:
                raise RuntimeError("logical KV parent cycle detected")
            visited.add(record.version)
            chunks.append(record.delta_tokens)
            if record.parent_version is None:
                break
            record = self._record(record.parent_version)
        return tuple(token for chunk in reversed(chunks) for token in chunk)

    def finalize(self, accepted_head: KVVersion) -> KVFinalizeResult:
        """Commit one ancestry chain and roll back every other speculative fork."""

        if self._finalized:
            raise RuntimeError("KV ledger is already finalized")
        accepted = self._ancestry(accepted_head)
        if self._base_version not in accepted:
            raise RuntimeError("accepted KV version does not descend from the base")

        newly_committed: list[KVVersion] = []
        rolled_back: list[KVVersion] = []
        for version in sorted(self._records, key=lambda item: item.value):
            record = self._records[version]
            if record.state is not KVSnapshotState.SPECULATIVE:
                continue
            if version in accepted:
                record.state = KVSnapshotState.COMMITTED
                newly_committed.append(version)
            else:
                record.state = KVSnapshotState.ROLLED_BACK
                rolled_back.append(version)

        self._head_version = accepted_head
        self._finalized = True
        return KVFinalizeResult(
            committed_head=accepted_head,
            newly_committed=tuple(newly_committed),
            rolled_back=tuple(rolled_back),
        )

    def rollback(self) -> KVFinalizeResult:
        return self.finalize(self._base_version)

    def _record(self, version: KVVersion) -> _KVRecord:
        if not isinstance(version, KVVersion):
            raise ValueError("version must be a KVVersion")
        try:
            return self._records[version]
        except KeyError as exc:
            raise KeyError(f"unknown logical KV version {version}") from exc

    def _ancestry(self, version: KVVersion) -> set[KVVersion]:
        record = self._record(version)
        ancestry: set[KVVersion] = set()
        while True:
            if record.version in ancestry:
                raise RuntimeError("logical KV parent cycle detected")
            ancestry.add(record.version)
            if record.parent_version is None:
                return ancestry
            record = self._record(record.parent_version)


@dataclass(frozen=True)
class MacroBranch:
    identity: BranchIdentity
    parent_identity: BranchIdentity | None
    token_id: int | None
    parent_kv_version: KVVersion
    kv_version: KVVersion
    state: MacroBranchState

    @property
    def depth(self) -> int:
        return self.identity.depth


@dataclass
class _BranchRecord:
    identity: BranchIdentity
    parent_identity: BranchIdentity | None
    token_id: int | None
    parent_kv_version: KVVersion
    kv_version: KVVersion
    state: MacroBranchState

    def public(self) -> MacroBranch:
        return MacroBranch(
            identity=self.identity,
            parent_identity=self.parent_identity,
            token_id=self.token_id,
            parent_kv_version=self.parent_kv_version,
            kv_version=self.kv_version,
            state=self.state,
        )


@dataclass(frozen=True)
class GreedyAcceptance:
    wave_identity: WaveIdentity
    accepted_branches: tuple[BranchIdentity, ...]
    accepted_tokens: tuple[int, ...]
    continuation_token: int
    emitted_tokens: tuple[int, ...]
    terminal_prefix_branch: BranchIdentity
    stop_reason: str


@dataclass(frozen=True)
class MacroWaveCommit:
    wave_identity: WaveIdentity
    committed_kv_version: KVVersion
    committed_branch_ids: tuple[BranchIdentity, ...]
    rolled_back_branch_ids: tuple[BranchIdentity, ...]
    rolled_back_kv_versions: tuple[KVVersion, ...]
    kv_prefix_tokens: tuple[int, ...]
    visible_prefix_tokens: tuple[int, ...]
    pending_token: int


@dataclass(frozen=True)
class MacroWaveRollback:
    wave_identity: WaveIdentity
    restored_kv_version: KVVersion
    restored_prefix_tokens: tuple[int, ...]
    rolled_back_branch_ids: tuple[BranchIdentity, ...]
    rolled_back_kv_versions: tuple[KVVersion, ...]


class MacroWaveTree:
    """Prefix tree for one speculative wave and its logical KV forks."""

    def __init__(
        self,
        identity: WaveIdentity,
        base_prefix_tokens: Sequence[int],
        parent_kv_version: KVVersion,
    ) -> None:
        if not isinstance(identity, WaveIdentity):
            raise ValueError("identity must be a WaveIdentity")
        if not isinstance(parent_kv_version, KVVersion):
            raise ValueError("parent_kv_version must be a KVVersion")
        prefix = _token_tuple(base_prefix_tokens, "base_prefix_tokens")
        expected = WaveIdentity.for_prefix(
            identity.request_id,
            identity.ordinal,
            prefix,
            parent_kv_version,
        )
        if expected.prefix_digest != identity.prefix_digest:
            raise ValueError("wave identity does not match its prefix and parent KV version")

        self.identity = identity
        self.base_prefix_tokens = prefix
        self.ledger = LogicalKVLedger(prefix, parent_kv_version)
        self.state = MacroWaveState.OPEN
        self.root_identity = BranchIdentity(identity)
        self._records: dict[BranchIdentity, _BranchRecord] = {
            self.root_identity: _BranchRecord(
                identity=self.root_identity,
                parent_identity=None,
                token_id=None,
                parent_kv_version=parent_kv_version,
                kv_version=parent_kv_version,
                state=MacroBranchState.COMMITTED,
            )
        }
        self._children: dict[BranchIdentity, list[BranchIdentity]] = {
            self.root_identity: []
        }

    @classmethod
    def create(
        cls,
        request_id: str | int,
        ordinal: int,
        base_prefix_tokens: Sequence[int],
        parent_kv_version: KVVersion = KVVersion(0),
    ) -> MacroWaveTree:
        identity = WaveIdentity.for_prefix(
            request_id,
            ordinal,
            base_prefix_tokens,
            parent_kv_version,
        )
        return cls(identity, base_prefix_tokens, parent_kv_version)

    def branch(self, identity: BranchIdentity) -> MacroBranch:
        return self._record(identity).public()

    def branches(self) -> tuple[MacroBranch, ...]:
        return tuple(
            self._records[identity].public()
            for identity in sorted(self._records, key=lambda item: item.path)
        )

    def children(self, parent: BranchIdentity) -> tuple[MacroBranch, ...]:
        self._record(parent)
        return tuple(self._records[identity].public() for identity in self._children[parent])

    def add_branch(self, parent: BranchIdentity, token_id: int) -> BranchIdentity:
        self._require_open()
        parent_record = self._record(parent)
        if parent_record.state is MacroBranchState.ROLLED_BACK:
            raise RuntimeError("cannot extend a rolled-back branch")
        token = _token_id(token_id)
        for child_identity in self._children[parent]:
            if self._records[child_identity].token_id == token:
                raise ValueError("a parent cannot contain duplicate candidate tokens")

        child_identity = parent.child(len(self._children[parent]))
        kv_version = self.ledger.fork(
            parent_record.kv_version,
            owner_branch=child_identity,
            token_id=token,
        )
        self._records[child_identity] = _BranchRecord(
            identity=child_identity,
            parent_identity=parent,
            token_id=token,
            parent_kv_version=parent_record.kv_version,
            kv_version=kv_version,
            state=MacroBranchState.SPECULATIVE,
        )
        self._children[parent].append(child_identity)
        self._children[child_identity] = []
        return child_identity

    def ensure_path(
        self,
        candidate_tokens: Sequence[int],
        *,
        parent: BranchIdentity | None = None,
    ) -> BranchIdentity:
        self._require_open()
        current = self.root_identity if parent is None else parent
        self._record(current)
        for token in _token_tuple(candidate_tokens, "candidate_tokens"):
            match = next(
                (
                    child
                    for child in self._children[current]
                    if self._records[child].token_id == token
                ),
                None,
            )
            current = match if match is not None else self.add_branch(current, token)
        return current

    def candidate_tokens(self, branch: BranchIdentity) -> tuple[int, ...]:
        record = self._record(branch)
        tokens: list[int] = []
        while record.parent_identity is not None:
            if record.token_id is None:
                raise RuntimeError("non-root branch has no token")
            tokens.append(record.token_id)
            record = self._record(record.parent_identity)
        return tuple(reversed(tokens))

    def full_prefix(self, branch: BranchIdentity) -> tuple[int, ...]:
        return (*self.base_prefix_tokens, *self.candidate_tokens(branch))

    def commit_greedy(self, acceptance: GreedyAcceptance) -> MacroWaveCommit:
        self._require_open()
        if not isinstance(acceptance, GreedyAcceptance):
            raise ValueError("acceptance must be a GreedyAcceptance")
        if acceptance.wave_identity != self.identity:
            raise ValueError("acceptance belongs to another wave")
        if len(acceptance.accepted_branches) != len(acceptance.accepted_tokens):
            raise ValueError("accepted branch and token counts must match")
        if acceptance.emitted_tokens != (
            *acceptance.accepted_tokens,
            acceptance.continuation_token,
        ):
            raise ValueError("greedy acceptance emitted token contract is invalid")

        parent = self.root_identity
        accepted_versions: list[KVVersion] = []
        for index, branch_identity in enumerate(acceptance.accepted_branches):
            record = self._record(branch_identity)
            if record.parent_identity != parent:
                raise ValueError("accepted branches must form one contiguous prefix")
            if record.token_id != acceptance.accepted_tokens[index]:
                raise ValueError("accepted branch token does not match acceptance")
            accepted_versions.append(record.kv_version)
            parent = branch_identity
        if parent != acceptance.terminal_prefix_branch:
            raise ValueError("terminal prefix branch does not match accepted path")

        committed_head = accepted_versions[-1] if accepted_versions else self.ledger.base_version
        finalized = self.ledger.finalize(committed_head)
        accepted_set = set(acceptance.accepted_branches)
        rolled_back_branches: list[BranchIdentity] = []
        for identity, record in self._records.items():
            if identity == self.root_identity:
                continue
            if identity in accepted_set:
                record.state = MacroBranchState.COMMITTED
            else:
                record.state = MacroBranchState.ROLLED_BACK
                rolled_back_branches.append(identity)
        self.state = MacroWaveState.COMMITTED

        kv_prefix = self.ledger.resolve_tokens(committed_head)
        visible_prefix = (*kv_prefix, acceptance.continuation_token)
        return MacroWaveCommit(
            wave_identity=self.identity,
            committed_kv_version=committed_head,
            committed_branch_ids=acceptance.accepted_branches,
            rolled_back_branch_ids=tuple(
                sorted(rolled_back_branches, key=lambda item: item.path)
            ),
            rolled_back_kv_versions=finalized.rolled_back,
            kv_prefix_tokens=kv_prefix,
            visible_prefix_tokens=visible_prefix,
            pending_token=acceptance.continuation_token,
        )

    def rollback(self) -> MacroWaveRollback:
        self._require_open()
        finalized = self.ledger.rollback()
        rolled_back_branches: list[BranchIdentity] = []
        for identity, record in self._records.items():
            if identity == self.root_identity:
                continue
            record.state = MacroBranchState.ROLLED_BACK
            rolled_back_branches.append(identity)
        self.state = MacroWaveState.ROLLED_BACK
        return MacroWaveRollback(
            wave_identity=self.identity,
            restored_kv_version=self.ledger.base_version,
            restored_prefix_tokens=self.base_prefix_tokens,
            rolled_back_branch_ids=tuple(
                sorted(rolled_back_branches, key=lambda item: item.path)
            ),
            rolled_back_kv_versions=finalized.rolled_back,
        )

    def _record(self, identity: BranchIdentity) -> _BranchRecord:
        if not isinstance(identity, BranchIdentity):
            raise ValueError("branch identity must be a BranchIdentity")
        if identity.wave != self.identity:
            raise KeyError("branch belongs to another wave")
        try:
            return self._records[identity]
        except KeyError as exc:
            raise KeyError(f"unknown branch {identity.key}") from exc

    def _require_open(self) -> None:
        if self.state is not MacroWaveState.OPEN:
            raise RuntimeError(f"macro wave is already {self.state.value}")


def verify_greedy_exact(
    tree: MacroWaveTree,
    target_argmax_by_prefix: Mapping[BranchIdentity, int],
) -> GreedyAcceptance:
    """Follow the target model's exact greedy path through a candidate tree.

    The mapping contains the target argmax for the prefix represented by each
    branch.  The verifier accepts matching candidate edges until the first
    mismatch.  At a leaf, the mapped argmax is the standard speculative bonus
    token.  A missing argmax on the selected path fails closed because emitting
    anything else would weaken exactness.
    """

    if not isinstance(tree, MacroWaveTree):
        raise ValueError("tree must be a MacroWaveTree")
    tree._require_open()
    if not isinstance(target_argmax_by_prefix, Mapping):
        raise ValueError("target_argmax_by_prefix must be a mapping")

    current = tree.root_identity
    accepted_branches: list[BranchIdentity] = []
    accepted_tokens: list[int] = []
    while True:
        if current not in target_argmax_by_prefix:
            raise ValueError(
                f"missing target argmax for selected prefix {current.key}"
            )
        target_token = _token_id(
            target_argmax_by_prefix[current],
            "target argmax token",
        )
        children = tree.children(current)
        match = next((child for child in children if child.token_id == target_token), None)
        if match is None:
            reason = "candidate_exhausted" if not children else "candidate_mismatch"
            emitted = (*accepted_tokens, target_token)
            return GreedyAcceptance(
                wave_identity=tree.identity,
                accepted_branches=tuple(accepted_branches),
                accepted_tokens=tuple(accepted_tokens),
                continuation_token=target_token,
                emitted_tokens=emitted,
                terminal_prefix_branch=current,
                stop_reason=reason,
            )
        accepted_branches.append(match.identity)
        if match.token_id is None:
            raise RuntimeError("candidate branch has no token")
        accepted_tokens.append(match.token_id)
        current = match.identity


def _candidate_nodes(width: int, depth: int, maximum: int) -> int | None:
    total = 0
    level = 1
    for _ in range(depth):
        level *= width
        total += level
        if total > maximum:
            return None
    return total


@dataclass(frozen=True)
class MacroWaveCostConfig:
    candidate_depths: tuple[int, ...] = (1, 2, 4, 8)
    candidate_widths: tuple[int, ...] = (1, 2, 4)
    minimum_speedup: float = 1.02
    max_candidate_nodes: int = 4096

    def __post_init__(self) -> None:
        depths = tuple(
            sorted(
                {
                    _integer("candidate depth", value, minimum=1)
                    for value in self.candidate_depths
                }
            )
        )
        widths = tuple(
            sorted(
                {
                    _integer("candidate width", value, minimum=1)
                    for value in self.candidate_widths
                }
            )
        )
        if not depths or not widths:
            raise ValueError("candidate depths and widths must not be empty")
        threshold = _finite(
            "minimum_speedup",
            self.minimum_speedup,
            strictly_positive=True,
        )
        if threshold < 1.0:
            raise ValueError("minimum_speedup must be >= 1")
        maximum = _integer("max_candidate_nodes", self.max_candidate_nodes, minimum=1)
        object.__setattr__(self, "candidate_depths", depths)
        object.__setattr__(self, "candidate_widths", widths)
        object.__setattr__(self, "minimum_speedup", threshold)
        object.__setattr__(self, "max_candidate_nodes", maximum)


@dataclass(frozen=True)
class MacroWaveCostObservation:
    """Measurements consumed by :class:`MacroWaveCostController`.

    ``route_rtt_ms`` is the fixed propagation/synchronization cost paid once
    per verification wave.  Per-node byte values should already include all
    WAN boundaries traversed by one candidate.  ``acceptance_by_width`` is
    measured probability that a tree of the given width contains the exact
    target token at one depth level.
    """

    route_rtt_ms: float
    bandwidth_bytes_per_second: float
    acceptance_by_width: tuple[tuple[int, float], ...]
    activation_bytes_per_node: int
    metadata_bytes_per_node: int
    kv_bytes_per_node: int
    workspace_bytes_per_node: int
    rollback_ms_per_node: float
    target_base_ms: float
    target_ms_per_node: float
    vram_budget_bytes: int
    vram_reserved_bytes: int = 0
    fixed_wire_bytes: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "route_rtt_ms",
            _finite("route_rtt_ms", self.route_rtt_ms),
        )
        object.__setattr__(
            self,
            "bandwidth_bytes_per_second",
            _finite(
                "bandwidth_bytes_per_second",
                self.bandwidth_bytes_per_second,
                strictly_positive=True,
            ),
        )
        normalized_acceptance: list[tuple[int, float]] = []
        observed_widths: set[int] = set()
        for raw_width, raw_probability in self.acceptance_by_width:
            width = _integer("acceptance width", raw_width, minimum=1)
            if width in observed_widths:
                raise ValueError("acceptance_by_width contains a duplicate width")
            probability = _finite("acceptance probability", raw_probability)
            if probability > 1.0:
                raise ValueError("acceptance probability must be <= 1")
            observed_widths.add(width)
            normalized_acceptance.append((width, probability))
        if not normalized_acceptance:
            raise ValueError("acceptance_by_width must not be empty")
        normalized_acceptance.sort()
        object.__setattr__(self, "acceptance_by_width", tuple(normalized_acceptance))

        for name in (
            "activation_bytes_per_node",
            "metadata_bytes_per_node",
            "kv_bytes_per_node",
            "workspace_bytes_per_node",
            "vram_budget_bytes",
            "vram_reserved_bytes",
            "fixed_wire_bytes",
        ):
            object.__setattr__(self, name, _integer(name, getattr(self, name)))
        for name in ("rollback_ms_per_node", "target_base_ms", "target_ms_per_node"):
            object.__setattr__(self, name, _finite(name, getattr(self, name)))
        if self.vram_reserved_bytes > self.vram_budget_bytes:
            raise ValueError("vram_reserved_bytes cannot exceed vram_budget_bytes")

    def acceptance_for_width(self, width: int) -> float | None:
        normalized = _integer("width", width, minimum=1)
        return dict(self.acceptance_by_width).get(normalized)


@dataclass(frozen=True)
class MacroWaveCostEstimate:
    depth: int
    width: int
    candidate_nodes: int
    level_acceptance: float
    expected_accepted_tokens: float
    expected_emitted_tokens: float
    expected_rollback_nodes: float
    wire_bytes: int
    vram_required_bytes: int
    fits_vram: bool
    predicted_latency_ms: float
    predicted_ms_per_token: float
    predicted_speedup: float


@dataclass(frozen=True)
class MacroWavePlan:
    enabled: bool
    reason: str
    estimate: MacroWaveCostEstimate | None
    baseline_latency_ms: float

    @property
    def depth(self) -> int:
        return self.estimate.depth if self.estimate is not None else 0

    @property
    def width(self) -> int:
        return self.estimate.width if self.estimate is not None else 0

    @property
    def predicted_speedup(self) -> float | None:
        return self.estimate.predicted_speedup if self.estimate is not None else None


class MacroWaveCostController:
    """Select the lowest predicted latency per exact emitted token."""

    def __init__(self, config: MacroWaveCostConfig | None = None) -> None:
        self.config = config or MacroWaveCostConfig()

    def baseline_latency_ms(self, observation: MacroWaveCostObservation) -> float:
        self._validate_observation(observation)
        wire_bytes = (
            observation.fixed_wire_bytes
            + observation.activation_bytes_per_node
            + observation.metadata_bytes_per_node
        )
        return (
            observation.route_rtt_ms
            + observation.target_base_ms
            + observation.target_ms_per_node
            + 1000.0 * wire_bytes / observation.bandwidth_bytes_per_second
        )

    def estimates(
        self,
        observation: MacroWaveCostObservation,
    ) -> tuple[MacroWaveCostEstimate, ...]:
        self._validate_observation(observation)
        baseline = self.baseline_latency_ms(observation)
        estimates: list[MacroWaveCostEstimate] = []
        for width in self.config.candidate_widths:
            acceptance = observation.acceptance_for_width(width)
            if acceptance is None:
                continue
            for depth in self.config.candidate_depths:
                nodes = _candidate_nodes(
                    width,
                    depth,
                    self.config.max_candidate_nodes,
                )
                if nodes is None:
                    continue
                expected_accepted = sum(
                    acceptance**level for level in range(1, depth + 1)
                )
                expected_emitted = 1.0 + expected_accepted
                expected_rollback = max(0.0, nodes - expected_accepted)
                wire_bytes = (
                    observation.fixed_wire_bytes
                    + nodes
                    * (
                        observation.activation_bytes_per_node
                        + observation.metadata_bytes_per_node
                    )
                )
                vram_required = (
                    observation.vram_reserved_bytes
                    + nodes
                    * (
                        observation.kv_bytes_per_node
                        + observation.workspace_bytes_per_node
                    )
                )
                latency = (
                    observation.route_rtt_ms
                    + observation.target_base_ms
                    + nodes * observation.target_ms_per_node
                    + 1000.0
                    * wire_bytes
                    / observation.bandwidth_bytes_per_second
                    + expected_rollback * observation.rollback_ms_per_node
                )
                per_token = latency / expected_emitted
                estimates.append(
                    MacroWaveCostEstimate(
                        depth=depth,
                        width=width,
                        candidate_nodes=nodes,
                        level_acceptance=acceptance,
                        expected_accepted_tokens=expected_accepted,
                        expected_emitted_tokens=expected_emitted,
                        expected_rollback_nodes=expected_rollback,
                        wire_bytes=wire_bytes,
                        vram_required_bytes=vram_required,
                        fits_vram=vram_required <= observation.vram_budget_bytes,
                        predicted_latency_ms=latency,
                        predicted_ms_per_token=per_token,
                        predicted_speedup=baseline / per_token,
                    )
                )
        return tuple(estimates)

    def choose(self, observation: MacroWaveCostObservation) -> MacroWavePlan:
        baseline = self.baseline_latency_ms(observation)
        all_estimates = self.estimates(observation)
        feasible = [estimate for estimate in all_estimates if estimate.fits_vram]
        if not feasible:
            return MacroWavePlan(
                enabled=False,
                reason="insufficient_vram",
                estimate=None,
                baseline_latency_ms=baseline,
            )
        best = min(
            feasible,
            key=lambda estimate: (
                estimate.predicted_ms_per_token,
                estimate.vram_required_bytes,
                estimate.candidate_nodes,
                estimate.depth,
                estimate.width,
            ),
        )
        if best.predicted_speedup < self.config.minimum_speedup:
            return MacroWavePlan(
                enabled=False,
                reason="not_beneficial",
                estimate=best,
                baseline_latency_ms=baseline,
            )
        return MacroWavePlan(
            enabled=True,
            reason="beneficial",
            estimate=best,
            baseline_latency_ms=baseline,
        )

    @staticmethod
    def _validate_observation(observation: MacroWaveCostObservation) -> None:
        if not isinstance(observation, MacroWaveCostObservation):
            raise ValueError("observation must be a MacroWaveCostObservation")
