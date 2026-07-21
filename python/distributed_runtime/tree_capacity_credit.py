"""Pure advance-credit ledger for a physically protected PrefixWave KV pool.

The current stage accounting ABI exposes total/free blocks (or bytes).  That is
enough for an immediate transactional quote, but not for a long-lived credit:
an ordinary admission can consume the quoted blocks between the quote and the
first FORK.  This module specifies the extra invariant needed to remove
PREPARE/COMMIT from the hot path:

* ``headroom_blocks`` is a dedicated KV subpool;
* ordinary admission is limited to the non-headroom partition;
* the backend reports how many physical blocks are currently owned by credit
  mutations, so partition drift can be detected rather than inferred;
* credits reserve bounded slices of headroom ahead of time and are exactly
  bound to one model/route/stage and one PrefixWave identity.

The implementation is deliberately independent of sockets and tensor backends.
Every public transition is lock-serialized and consumes an explicit monotonic
time plus an immutable physical observation, making schedules reproducible in
tests.  A real adapter must update the physical allocator and this ledger under
the same admission barrier (or expose genuinely separate allocator arenas).

``topology_digest`` is only a content/identity binding.  It is not a MAC and a
``lease_id`` is not an authentication credential.  Peer authentication belongs
to the transport; this module intentionally does not simulate cryptographic
security.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import heapq
import math
from numbers import Integral
from threading import RLock


UINT32_MAX = (1 << 32) - 1
UINT64_MAX = (1 << 64) - 1
TOPOLOGY_DIGEST_BYTES = 32
MAX_ID_UTF8_BYTES = 256


class TreeCapacityCreditError(RuntimeError):
    """Base class for an advance-credit failure."""


class TreeCapacityCreditValidationError(TreeCapacityCreditError, ValueError):
    """An input is not a canonical bounded credit value."""


class TreeCapacityCreditUnavailable(TreeCapacityCreditError):
    """Protected headroom cannot fit another credit."""


class TreeCapacityCreditReplay(TreeCapacityCreditError):
    """A pre-mutation replay or a reference to a closed/unknown credit."""


class TreeCapacityCreditBindingError(TreeCapacityCreditError):
    """A wave does not exactly match the credit to which it refers."""


class TreeCapacityCreditFatal(TreeCapacityCreditError):
    """The route must be quarantined because post-mutation safety is unknown."""


def _bounded_integer(name: str, value: object, minimum: int, maximum: int) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TreeCapacityCreditValidationError(f"{name} must be an integer")
    normalized = int(value)
    if normalized < minimum or normalized > maximum:
        raise TreeCapacityCreditValidationError(
            f"{name} must be in [{minimum}, {maximum}]"
        )
    return normalized


def _bounded_text(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise TreeCapacityCreditValidationError(f"{name} must be a string")
    if not value or value != value.strip() or "\x00" in value:
        raise TreeCapacityCreditValidationError(
            f"{name} must be non-empty, trimmed and contain no NUL"
        )
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise TreeCapacityCreditValidationError(
            f"{name} must be valid UTF-8"
        ) from exc
    if len(encoded) > MAX_ID_UTF8_BYTES:
        raise TreeCapacityCreditValidationError(
            f"{name} exceeds {MAX_ID_UTF8_BYTES} UTF-8 bytes"
        )
    return value


def _time_value(name: str, value: object) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TreeCapacityCreditValidationError(f"{name} must be a number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0:
        raise TreeCapacityCreditValidationError(
            f"{name} must be finite and non-negative"
        )
    return normalized


@dataclass(frozen=True, slots=True)
class TreeCapacityCreditBinding:
    """Exact one-wave scope of an advance KV credit."""

    model_id: str
    route_id: str
    stage_id: str
    parent_request_id: int
    step: int
    topology_digest: bytes
    block_count: int
    lane_count: int

    def __post_init__(self) -> None:
        for name in ("model_id", "route_id", "stage_id"):
            object.__setattr__(self, name, _bounded_text(name, getattr(self, name)))
        object.__setattr__(
            self,
            "parent_request_id",
            _bounded_integer(
                "parent_request_id", self.parent_request_id, 1, UINT64_MAX
            ),
        )
        object.__setattr__(
            self, "step", _bounded_integer("step", self.step, 0, UINT32_MAX)
        )
        if not isinstance(self.topology_digest, bytes):
            raise TreeCapacityCreditValidationError(
                "topology_digest must be immutable bytes"
            )
        if len(self.topology_digest) != TOPOLOGY_DIGEST_BYTES:
            raise TreeCapacityCreditValidationError(
                f"topology_digest must contain {TOPOLOGY_DIGEST_BYTES} bytes"
            )
        object.__setattr__(
            self,
            "block_count",
            _bounded_integer("block_count", self.block_count, 1, UINT32_MAX),
        )
        object.__setattr__(
            self,
            "lane_count",
            _bounded_integer("lane_count", self.lane_count, 1, UINT32_MAX),
        )


@dataclass(frozen=True, slots=True, order=True)
class TreeCapacityCreditKey:
    """Replay identity; nonces are globally one-shot within one epoch."""

    epoch: int
    lease_id: str
    nonce: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "epoch", _bounded_integer("epoch", self.epoch, 1, UINT64_MAX)
        )
        object.__setattr__(self, "lease_id", _bounded_text("lease_id", self.lease_id))
        object.__setattr__(
            self, "nonce", _bounded_integer("nonce", self.nonce, 1, UINT64_MAX)
        )


@dataclass(frozen=True, slots=True, order=True)
class PhysicalCreditKVUsage:
    """Physical headroom blocks currently attributable to one credit."""

    key: TreeCapacityCreditKey
    block_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.key, TreeCapacityCreditKey):
            raise TreeCapacityCreditValidationError(
                "physical credit usage key must be a TreeCapacityCreditKey"
            )
        object.__setattr__(
            self,
            "block_count",
            _bounded_integer(
                "credit usage block_count", self.block_count, 1, UINT32_MAX
            ),
        )


@dataclass(frozen=True, slots=True)
class PhysicalKVObservation:
    """Atomic counters supplied by the physical KV allocator.

    ``credit_usages`` attributes protected blocks to exact MUTATING credit
    identities.  Merely deriving an aggregate from total free blocks would not
    prove physical separation or enforce each credit's individual ceiling.
    ``revision`` is the allocator's monotonic observation/version counter.
    """

    epoch: int
    revision: int
    total_blocks: int
    free_blocks: int
    credit_usages: tuple[PhysicalCreditKVUsage, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "epoch", _bounded_integer("epoch", self.epoch, 1, UINT64_MAX)
        )
        object.__setattr__(
            self,
            "revision",
            _bounded_integer("revision", self.revision, 0, UINT64_MAX),
        )
        total = _bounded_integer("total_blocks", self.total_blocks, 1, UINT32_MAX)
        free = _bounded_integer("free_blocks", self.free_blocks, 0, total)
        if not isinstance(self.credit_usages, tuple):
            raise TreeCapacityCreditValidationError(
                "credit_usages must be an immutable tuple"
            )
        previous_key: TreeCapacityCreditKey | None = None
        used = 0
        for usage in self.credit_usages:
            if not isinstance(usage, PhysicalCreditKVUsage):
                raise TreeCapacityCreditValidationError(
                    "credit_usages must contain PhysicalCreditKVUsage values"
                )
            if usage.key.epoch != self.epoch:
                raise TreeCapacityCreditValidationError(
                    "physical credit usage belongs to another epoch"
                )
            if previous_key is not None and usage.key <= previous_key:
                raise TreeCapacityCreditValidationError(
                    "credit_usages must be unique and canonically key-sorted"
                )
            previous_key = usage.key
            used += usage.block_count
            if used > total:
                raise TreeCapacityCreditValidationError(
                    "aggregate physical credit usage exceeds total blocks"
                )
        object.__setattr__(self, "total_blocks", total)
        object.__setattr__(self, "free_blocks", free)

    @property
    def credit_blocks_in_use(self) -> int:
        return sum(usage.block_count for usage in self.credit_usages)


class TreeCapacityCreditState(str, Enum):
    ISSUED = "issued"
    CONSUMED = "consumed"
    MUTATING = "mutating"
    RELEASED = "released"
    EXPIRED = "expired"
    COMPLETED = "completed"


@dataclass(frozen=True, slots=True)
class TreeCapacityCredit:
    key: TreeCapacityCreditKey
    binding: TreeCapacityCreditBinding
    issued_at: float
    expires_at: float


@dataclass(slots=True)
class _CreditRecord:
    credit: TreeCapacityCredit
    state: TreeCapacityCreditState = TreeCapacityCreditState.ISSUED
    consumed_at: float | None = None
    mutation_started_at: float | None = None
    closed_at: float | None = None


@dataclass(frozen=True, slots=True)
class TreeCapacityCreditRecordSnapshot:
    key: TreeCapacityCreditKey
    binding: TreeCapacityCreditBinding
    state: TreeCapacityCreditState
    issued_at: float
    expires_at: float
    consumed_at: float | None
    mutation_started_at: float | None
    closed_at: float | None


@dataclass(frozen=True, slots=True)
class TreeCapacityCreditSnapshot:
    """Deterministically ordered, immutable observability for tests/runtime."""

    epoch: int
    total_blocks: int
    headroom_blocks: int
    ordinary_capacity_blocks: int
    reserved_blocks: int
    mutating_limit_blocks: int
    protected_free_floor_blocks: int
    ordinary_free_blocks: int | None
    fatal: bool
    fatal_reason: str | None
    last_observation: PhysicalKVObservation | None
    records: tuple[TreeCapacityCreditRecordSnapshot, ...]


class TreeCapacityCreditLedger:
    """Thread-safe one-epoch ledger for advance PrefixWave KV credits."""

    def __init__(
        self,
        *,
        epoch: int,
        total_blocks: int,
        headroom_blocks: int,
        max_lanes_per_credit: int,
        max_ttl_seconds: float = 300.0,
        max_credits_per_epoch: int = 65_536,
    ) -> None:
        self.epoch = _bounded_integer("epoch", epoch, 1, UINT64_MAX)
        self.total_blocks = _bounded_integer(
            "total_blocks", total_blocks, 1, UINT32_MAX
        )
        self.headroom_blocks = _bounded_integer(
            "headroom_blocks", headroom_blocks, 1, self.total_blocks
        )
        self.max_lanes_per_credit = _bounded_integer(
            "max_lanes_per_credit", max_lanes_per_credit, 1, UINT32_MAX
        )
        self.max_ttl_seconds = _time_value("max_ttl_seconds", max_ttl_seconds)
        if self.max_ttl_seconds <= 0:
            raise TreeCapacityCreditValidationError(
                "max_ttl_seconds must be positive"
            )
        self.max_credits_per_epoch = _bounded_integer(
            "max_credits_per_epoch", max_credits_per_epoch, 1, UINT32_MAX
        )
        self._records: dict[TreeCapacityCreditKey, _CreditRecord] = {}
        self._nonce_keys: dict[int, TreeCapacityCreditKey] = {}
        self._expiry_heap: list[tuple[float, TreeCapacityCreditKey]] = []
        self._reserved_block_count = 0
        self._mutating_block_limit = 0
        self._last_observation: PhysicalKVObservation | None = None
        self._fatal_reason: str | None = None
        self._lock = RLock()

    @property
    def ordinary_capacity_blocks(self) -> int:
        return self.total_blocks - self.headroom_blocks

    @property
    def fatal(self) -> bool:
        with self._lock:
            return self._fatal_reason is not None

    def issue(
        self,
        binding: TreeCapacityCreditBinding,
        *,
        lease_id: str,
        nonce: int,
        ttl_seconds: float,
        now: float,
        observation: PhysicalKVObservation,
    ) -> TreeCapacityCredit:
        """Reserve a bounded slice of protected headroom without allocating KV."""

        if not isinstance(binding, TreeCapacityCreditBinding):
            raise TreeCapacityCreditValidationError(
                "binding must be a TreeCapacityCreditBinding"
            )
        lease_id = _bounded_text("lease_id", lease_id)
        nonce = _bounded_integer("nonce", nonce, 1, UINT64_MAX)
        ttl = _time_value("ttl_seconds", ttl_seconds)
        if ttl <= 0 or ttl > self.max_ttl_seconds:
            raise TreeCapacityCreditValidationError(
                f"ttl_seconds must be in (0, {self.max_ttl_seconds}]"
            )
        now = _time_value("now", now)
        key = TreeCapacityCreditKey(self.epoch, lease_id, nonce)

        with self._lock:
            self._advance(now, observation)
            if nonce in self._nonce_keys:
                previous = self._records[self._nonce_keys[nonce]]
                if previous.state in {
                    TreeCapacityCreditState.MUTATING,
                    TreeCapacityCreditState.COMPLETED,
                }:
                    self._fatal("credit nonce replay after mutation began")
                raise TreeCapacityCreditReplay(
                    "credit nonce is already spent in this epoch"
                )
            if binding.block_count > self.headroom_blocks:
                raise TreeCapacityCreditUnavailable(
                    "credit exceeds protected KV headroom"
                )
            if binding.lane_count > self.max_lanes_per_credit:
                raise TreeCapacityCreditUnavailable(
                    "credit exceeds the sealed lane limit"
                )
            if len(self._records) >= self.max_credits_per_epoch:
                raise TreeCapacityCreditUnavailable(
                    "credit epoch record limit reached; rotate the epoch"
                )
            reserved = self._reserved_blocks()
            if reserved + binding.block_count > self.headroom_blocks:
                raise TreeCapacityCreditUnavailable(
                    "aggregate credits exceed protected KV headroom"
                )
            expires_at = now + ttl
            if not math.isfinite(expires_at):
                raise TreeCapacityCreditValidationError("credit expiry overflow")
            credit = TreeCapacityCredit(
                key=key,
                binding=binding,
                issued_at=now,
                expires_at=expires_at,
            )
            self._records[key] = _CreditRecord(credit=credit)
            self._nonce_keys[nonce] = key
            self._reserved_block_count += binding.block_count
            heapq.heappush(self._expiry_heap, (expires_at, key))
            return credit

    def consume(
        self,
        key: TreeCapacityCreditKey,
        binding: TreeCapacityCreditBinding,
        *,
        now: float,
        observation: PhysicalKVObservation,
    ) -> TreeCapacityCredit:
        """Claim a credit; an exact retry is idempotent until mutation starts."""

        now = _time_value("now", now)
        with self._lock:
            self._advance(now, observation)
            record = self._record(key)
            self._require_binding(record, binding, operation="consume")
            if record.state is TreeCapacityCreditState.ISSUED:
                record.state = TreeCapacityCreditState.CONSUMED
                record.consumed_at = now
                return record.credit
            if record.state is TreeCapacityCreditState.CONSUMED:
                return record.credit
            if record.state in {
                TreeCapacityCreditState.MUTATING,
                TreeCapacityCreditState.COMPLETED,
            }:
                self._fatal("credit consume replay after mutation began")
            raise TreeCapacityCreditReplay(
                f"cannot consume a {record.state.value} credit"
            )

    def release(
        self,
        key: TreeCapacityCreditKey,
        binding: TreeCapacityCreditBinding,
        *,
        now: float,
        observation: PhysicalKVObservation,
    ) -> bool:
        """Release before mutation; exact retries are idempotent.

        Returns ``True`` only for the transition that actually releases a live
        reservation.  Releasing an already RELEASED/EXPIRED pre-mutation
        tombstone returns ``False``.
        """

        now = _time_value("now", now)
        with self._lock:
            self._advance(now, observation)
            record = self._record(key)
            self._require_binding(record, binding, operation="release")
            if record.state in {
                TreeCapacityCreditState.ISSUED,
                TreeCapacityCreditState.CONSUMED,
            }:
                record.state = TreeCapacityCreditState.RELEASED
                record.closed_at = now
                self._reserved_block_count -= record.credit.binding.block_count
                return True
            if record.state in {
                TreeCapacityCreditState.RELEASED,
                TreeCapacityCreditState.EXPIRED,
            }:
                return False
            self._fatal("credit cancellation/release after mutation began")

    def begin_mutation(
        self,
        key: TreeCapacityCreditKey,
        binding: TreeCapacityCreditBinding,
        *,
        now: float,
        observation: PhysicalKVObservation,
    ) -> TreeCapacityCredit:
        """Cross the irreversible boundary for one consumed credit.

        The caller must serialize this transition with its physical headroom
        allocator.  From this point, expiry, cancellation and every replay are
        fatal because remote/local KV may already have changed.
        """

        now = _time_value("now", now)
        with self._lock:
            self._advance(now, observation)
            record = self._record(key)
            self._require_binding(record, binding, operation="begin mutation")
            if record.state is TreeCapacityCreditState.CONSUMED:
                record.state = TreeCapacityCreditState.MUTATING
                record.mutation_started_at = now
                self._mutating_block_limit += record.credit.binding.block_count
                return record.credit
            if record.state in {
                TreeCapacityCreditState.MUTATING,
                TreeCapacityCreditState.COMPLETED,
            }:
                self._fatal("credit mutation replay after mutation began")
            raise TreeCapacityCreditReplay(
                f"cannot mutate with a {record.state.value} credit"
            )

    def complete_mutation(
        self,
        key: TreeCapacityCreditKey,
        binding: TreeCapacityCreditBinding,
        *,
        now: float,
        observation: PhysicalKVObservation,
    ) -> None:
        """Retire a mutation only after its protected blocks were freed/migrated.

        The supplied observation must fit the remaining MUTATING credits.  This
        prevents the ledger from making the headroom reusable while blocks from
        the completed wave are still charged to it.
        """

        now = _time_value("now", now)
        with self._lock:
            self._ensure_healthy()
            self._expire(now)
            record = self._record(key)
            self._require_binding(record, binding, operation="complete mutation")
            if record.state is not TreeCapacityCreditState.MUTATING:
                if record.state is TreeCapacityCreditState.COMPLETED:
                    self._fatal("credit completion replay after mutation began")
                raise TreeCapacityCreditReplay(
                    f"cannot complete a {record.state.value} credit"
                )
            remaining_mutating = self._mutating_limit_blocks() - record.credit.binding.block_count
            self._observe(
                observation,
                mutating_limit_override=remaining_mutating,
                excluded_mutating_key=key,
            )
            record.state = TreeCapacityCreditState.COMPLETED
            record.closed_at = now
            self._reserved_block_count -= record.credit.binding.block_count
            self._mutating_block_limit -= record.credit.binding.block_count

    def advance(
        self,
        *,
        now: float,
        observation: PhysicalKVObservation,
    ) -> None:
        """Process TTLs and validate the current physical partition counters."""

        now = _time_value("now", now)
        with self._lock:
            self._advance(now, observation)

    def ordinary_admissible_blocks(
        self,
        *,
        now: float,
        observation: PhysicalKVObservation,
    ) -> int:
        """Return only free blocks in the ordinary, non-headroom partition."""

        now = _time_value("now", now)
        with self._lock:
            self._advance(now, observation)
            return self._ordinary_free_blocks(observation)

    def snapshot(self) -> TreeCapacityCreditSnapshot:
        """Return immutable state; this method never advances time implicitly."""

        with self._lock:
            records = tuple(
                TreeCapacityCreditRecordSnapshot(
                    key=record.credit.key,
                    binding=record.credit.binding,
                    state=record.state,
                    issued_at=record.credit.issued_at,
                    expires_at=record.credit.expires_at,
                    consumed_at=record.consumed_at,
                    mutation_started_at=record.mutation_started_at,
                    closed_at=record.closed_at,
                )
                for _, record in sorted(self._records.items())
            )
            observation = self._last_observation
            credit_in_use = (
                0 if observation is None else observation.credit_blocks_in_use
            )
            return TreeCapacityCreditSnapshot(
                epoch=self.epoch,
                total_blocks=self.total_blocks,
                headroom_blocks=self.headroom_blocks,
                ordinary_capacity_blocks=self.ordinary_capacity_blocks,
                reserved_blocks=self._reserved_blocks(),
                mutating_limit_blocks=self._mutating_limit_blocks(),
                protected_free_floor_blocks=self.headroom_blocks - credit_in_use,
                ordinary_free_blocks=(
                    None
                    if observation is None
                    else self._ordinary_free_blocks(observation)
                ),
                fatal=self._fatal_reason is not None,
                fatal_reason=self._fatal_reason,
                last_observation=observation,
                records=records,
            )

    def _advance(
        self, now: float, observation: PhysicalKVObservation
    ) -> None:
        self._ensure_healthy()
        self._expire(now)
        self._observe(observation)

    def _ensure_healthy(self) -> None:
        if self._fatal_reason is not None:
            raise TreeCapacityCreditFatal(self._fatal_reason)

    def _fatal(self, reason: str) -> None:
        if self._fatal_reason is None:
            self._fatal_reason = reason
        raise TreeCapacityCreditFatal(self._fatal_reason)

    def _expire(self, now: float) -> None:
        # Heap tuple ordering makes the fatal choice deterministic if several
        # leases age out in the same tick, without scanning all epoch tombstones
        # on every hot-path operation.
        while self._expiry_heap and self._expiry_heap[0][0] <= now:
            _, key = heapq.heappop(self._expiry_heap)
            record = self._records[key]
            if record.state in {
                TreeCapacityCreditState.ISSUED,
                TreeCapacityCreditState.CONSUMED,
            }:
                record.state = TreeCapacityCreditState.EXPIRED
                record.closed_at = record.credit.expires_at
                self._reserved_block_count -= record.credit.binding.block_count
            elif record.state is TreeCapacityCreditState.MUTATING:
                self._fatal("credit TTL expired after mutation began")

    def _observe(
        self,
        observation: PhysicalKVObservation,
        *,
        mutating_limit_override: int | None = None,
        excluded_mutating_key: TreeCapacityCreditKey | None = None,
    ) -> None:
        if not isinstance(observation, PhysicalKVObservation):
            raise TreeCapacityCreditValidationError(
                "observation must be a PhysicalKVObservation"
            )
        if observation.epoch != self.epoch:
            self._fatal("physical KV epoch drift")
        if observation.total_blocks != self.total_blocks:
            self._fatal("physical KV total-block drift")
        previous = self._last_observation
        if previous is not None:
            if observation.revision < previous.revision:
                self._fatal("physical KV observation revision moved backwards")
            if observation.revision == previous.revision and observation != previous:
                self._fatal("physical KV counters changed without a new revision")

        mutating_limit = (
            self._mutating_limit_blocks()
            if mutating_limit_override is None
            else mutating_limit_override
        )
        for usage in observation.credit_usages:
            record = self._records.get(usage.key)
            if (
                record is None
                or usage.key == excluded_mutating_key
                or record.state is not TreeCapacityCreditState.MUTATING
            ):
                self._fatal(
                    "physical KV usage is not owned by a live mutating credit"
                )
            if usage.block_count > record.credit.binding.block_count:
                self._fatal("physical KV usage exceeds its credit block limit")
        if observation.credit_blocks_in_use > mutating_limit:
            self._fatal("credit KV usage exceeds all mutating credit limits")
        if observation.credit_blocks_in_use > self.headroom_blocks:
            self._fatal("credit KV usage exceeds protected headroom")

        physically_used = observation.total_blocks - observation.free_blocks
        if observation.credit_blocks_in_use > physically_used:
            self._fatal("credit KV usage exceeds total physical occupancy")
        ordinary_used = physically_used - observation.credit_blocks_in_use
        if ordinary_used > self.ordinary_capacity_blocks:
            self._fatal("ordinary KV occupancy crossed the protected headroom floor")

        self._last_observation = observation

    def _ordinary_free_blocks(self, observation: PhysicalKVObservation) -> int:
        physically_used = observation.total_blocks - observation.free_blocks
        ordinary_used = physically_used - observation.credit_blocks_in_use
        return self.ordinary_capacity_blocks - ordinary_used

    def _record(self, key: TreeCapacityCreditKey) -> _CreditRecord:
        if not isinstance(key, TreeCapacityCreditKey):
            raise TreeCapacityCreditValidationError(
                "key must be a TreeCapacityCreditKey"
            )
        if key.epoch != self.epoch:
            raise TreeCapacityCreditReplay("credit belongs to another epoch")
        record = self._records.get(key)
        if record is None:
            previous_key = self._nonce_keys.get(key.nonce)
            if previous_key is not None:
                previous = self._records[previous_key]
                if previous.state in {
                    TreeCapacityCreditState.MUTATING,
                    TreeCapacityCreditState.COMPLETED,
                }:
                    self._fatal("credit identity replay after mutation began")
            raise TreeCapacityCreditReplay("unknown or spent credit identity")
        return record

    def _require_binding(
        self,
        record: _CreditRecord,
        binding: TreeCapacityCreditBinding,
        *,
        operation: str,
    ) -> None:
        if not isinstance(binding, TreeCapacityCreditBinding):
            raise TreeCapacityCreditValidationError(
                "binding must be a TreeCapacityCreditBinding"
            )
        if binding == record.credit.binding:
            return
        if record.state in {
            TreeCapacityCreditState.MUTATING,
            TreeCapacityCreditState.COMPLETED,
        }:
            self._fatal(f"credit {operation} binding mismatch after mutation began")
        raise TreeCapacityCreditBindingError(
            f"credit {operation} does not match its sealed binding"
        )

    def _reserved_blocks(self) -> int:
        return self._reserved_block_count

    def _mutating_limit_blocks(self) -> int:
        return self._mutating_block_limit


__all__ = [
    "PhysicalCreditKVUsage",
    "PhysicalKVObservation",
    "TOPOLOGY_DIGEST_BYTES",
    "TreeCapacityCredit",
    "TreeCapacityCreditBinding",
    "TreeCapacityCreditBindingError",
    "TreeCapacityCreditError",
    "TreeCapacityCreditFatal",
    "TreeCapacityCreditKey",
    "TreeCapacityCreditLedger",
    "TreeCapacityCreditRecordSnapshot",
    "TreeCapacityCreditReplay",
    "TreeCapacityCreditSnapshot",
    "TreeCapacityCreditState",
    "TreeCapacityCreditUnavailable",
    "TreeCapacityCreditValidationError",
]
