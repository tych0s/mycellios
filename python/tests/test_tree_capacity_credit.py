from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import random
import unittest

from distributed_runtime.tree_capacity_credit import (
    PhysicalCreditKVUsage,
    PhysicalKVObservation,
    TreeCapacityCreditBinding,
    TreeCapacityCreditBindingError,
    TreeCapacityCreditFatal,
    TreeCapacityCreditLedger,
    TreeCapacityCreditReplay,
    TreeCapacityCreditState,
    TreeCapacityCreditUnavailable,
    TreeCapacityCreditValidationError,
)


def _binding(
    *,
    blocks: int = 8,
    lanes: int = 4,
    parent: int = 101,
    step: int = 7,
    digest_byte: int = 0x5A,
    model: str = "glm-test",
    route: str = "route-a",
    stage: str = "stage-1",
) -> TreeCapacityCreditBinding:
    return TreeCapacityCreditBinding(
        model_id=model,
        route_id=route,
        stage_id=stage,
        parent_request_id=parent,
        step=step,
        topology_digest=bytes([digest_byte]) * 32,
        block_count=blocks,
        lane_count=lanes,
    )


def _observation(
    revision: int = 0,
    *,
    total: int = 100,
    free: int = 100,
    credit: int = 0,
    credit_key=None,
    credit_usages=(),
    epoch: int = 3,
) -> PhysicalKVObservation:
    if credit and credit_key is None:
        raise ValueError("credit_key is required for non-zero credit usage")
    usages = list(credit_usages)
    if credit:
        usages.append((credit_key, credit))
    canonical_usages = tuple(
        PhysicalCreditKVUsage(key=key, block_count=blocks)
        for key, blocks in sorted(usages)
    )
    return PhysicalKVObservation(
        epoch=epoch,
        revision=revision,
        total_blocks=total,
        free_blocks=free,
        credit_usages=canonical_usages,
    )


def _ledger(
    *,
    total: int = 100,
    headroom: int = 20,
    lanes: int = 16,
    ttl: float = 100.0,
) -> TreeCapacityCreditLedger:
    return TreeCapacityCreditLedger(
        epoch=3,
        total_blocks=total,
        headroom_blocks=headroom,
        max_lanes_per_credit=lanes,
        max_ttl_seconds=ttl,
    )


class TreeCapacityCreditLedgerTests(unittest.TestCase):
    def test_binding_is_exactly_scoped_and_bounded(self) -> None:
        binding = _binding()
        self.assertEqual(binding.block_count, 8)
        self.assertEqual(binding.lane_count, 4)
        for change in (
            {"model_id": "glm-other"},
            {"route_id": "route-b"},
            {"stage_id": "stage-2"},
            {"parent_request_id": 102},
            {"step": 8},
            {"topology_digest": b"x" * 32},
            {"block_count": 9},
            {"lane_count": 5},
        ):
            self.assertNotEqual(binding, replace(binding, **change))

        with self.assertRaises(TreeCapacityCreditValidationError):
            replace(binding, topology_digest=bytearray(32))
        with self.assertRaises(TreeCapacityCreditValidationError):
            replace(binding, topology_digest=b"short")
        with self.assertRaises(TreeCapacityCreditValidationError):
            replace(binding, route_id=" route-a")
        with self.assertRaises(TreeCapacityCreditValidationError):
            replace(binding, block_count=0)

    def test_multiple_credits_share_only_bounded_headroom(self) -> None:
        ledger = _ledger(headroom=20)
        first = ledger.issue(
            _binding(blocks=7),
            lease_id="lease-a",
            nonce=1,
            ttl_seconds=10,
            now=0,
            observation=_observation(),
        )
        second = ledger.issue(
            _binding(blocks=13, parent=102),
            lease_id="lease-b",
            nonce=2,
            ttl_seconds=10,
            now=0,
            observation=_observation(),
        )
        self.assertEqual(ledger.snapshot().reserved_blocks, 20)
        with self.assertRaises(TreeCapacityCreditUnavailable):
            ledger.issue(
                _binding(blocks=1, parent=103),
                lease_id="lease-c",
                nonce=3,
                ttl_seconds=10,
                now=0,
                observation=_observation(),
            )
        self.assertTrue(
            ledger.release(
                first.key,
                first.binding,
                now=1,
                observation=_observation(1),
            )
        )
        third = ledger.issue(
            _binding(blocks=7, parent=103),
            lease_id="lease-c",
            nonce=3,
            ttl_seconds=10,
            now=1,
            observation=_observation(1),
        )
        self.assertEqual(third.binding.block_count, 7)
        self.assertEqual(ledger.snapshot().reserved_blocks, 20)
        self.assertEqual(second.key.nonce, 2)

    def test_ordinary_admission_never_includes_protected_headroom(self) -> None:
        ledger = _ledger(total=100, headroom=20)
        credit = ledger.issue(
            _binding(blocks=8),
            lease_id="lease",
            nonce=1,
            ttl_seconds=20,
            now=0,
            observation=_observation(),
        )
        # All blocks physically free, but ordinary work can see only its arena.
        self.assertEqual(
            ledger.ordinary_admissible_blocks(
                now=0,
                observation=_observation(),
            ),
            80,
        )
        # Thirty ordinary blocks have been allocated.
        self.assertEqual(
            ledger.ordinary_admissible_blocks(
                now=1,
                observation=_observation(1, free=70),
            ),
            50,
        )
        ledger.consume(
            credit.key,
            credit.binding,
            now=1,
            observation=_observation(1, free=70),
        )
        ledger.begin_mutation(
            credit.key,
            credit.binding,
            now=1,
            observation=_observation(1, free=70),
        )
        # Five more blocks came from the credit arena. Ordinary free capacity is
        # still fifty, not total free (sixty-five).
        self.assertEqual(
            ledger.ordinary_admissible_blocks(
                now=2,
                observation=_observation(
                    2, free=65, credit=5, credit_key=credit.key
                ),
            ),
            50,
        )
        snapshot = ledger.snapshot()
        self.assertEqual(snapshot.protected_free_floor_blocks, 15)
        self.assertEqual(snapshot.ordinary_free_blocks, 50)

    def test_consume_and_release_are_idempotent_before_mutation(self) -> None:
        ledger = _ledger()
        credit = ledger.issue(
            _binding(),
            lease_id="lease",
            nonce=1,
            ttl_seconds=10,
            now=0,
            observation=_observation(),
        )
        first = ledger.consume(
            credit.key,
            credit.binding,
            now=1,
            observation=_observation(1),
        )
        second = ledger.consume(
            credit.key,
            credit.binding,
            now=1,
            observation=_observation(1),
        )
        self.assertIs(first, second)
        self.assertTrue(
            ledger.release(
                credit.key,
                credit.binding,
                now=2,
                observation=_observation(2),
            )
        )
        self.assertFalse(
            ledger.release(
                credit.key,
                credit.binding,
                now=2,
                observation=_observation(2),
            )
        )
        self.assertEqual(ledger.snapshot().reserved_blocks, 0)
        self.assertFalse(ledger.fatal)

    def test_nonce_is_one_shot_for_whole_epoch_even_after_release(self) -> None:
        ledger = _ledger()
        credit = ledger.issue(
            _binding(),
            lease_id="lease-a",
            nonce=44,
            ttl_seconds=10,
            now=0,
            observation=_observation(),
        )
        ledger.release(
            credit.key,
            credit.binding,
            now=1,
            observation=_observation(1),
        )
        with self.assertRaises(TreeCapacityCreditReplay):
            ledger.issue(
                _binding(parent=102),
                lease_id="lease-b",
                nonce=44,
                ttl_seconds=10,
                now=1,
                observation=_observation(1),
            )
        self.assertFalse(ledger.fatal)

    def test_binding_mismatch_before_mutation_rejects_without_poisoning(self) -> None:
        ledger = _ledger()
        credit = ledger.issue(
            _binding(),
            lease_id="lease",
            nonce=1,
            ttl_seconds=10,
            now=0,
            observation=_observation(),
        )
        with self.assertRaises(TreeCapacityCreditBindingError):
            ledger.consume(
                credit.key,
                replace(credit.binding, step=credit.binding.step + 1),
                now=1,
                observation=_observation(1),
            )
        self.assertFalse(ledger.fatal)
        ledger.consume(
            credit.key,
            credit.binding,
            now=1,
            observation=_observation(1),
        )

    def test_every_replay_or_cancel_after_mutation_is_fatal(self) -> None:
        operations = (
            "consume",
            "release",
            "begin",
            "wrong-binding",
            "changed-lease",
            "reissue-nonce",
        )
        for operation in operations:
            with self.subTest(operation=operation):
                ledger = _ledger()
                credit = ledger.issue(
                    _binding(),
                    lease_id="lease",
                    nonce=1,
                    ttl_seconds=20,
                    now=0,
                    observation=_observation(),
                )
                ledger.consume(
                    credit.key,
                    credit.binding,
                    now=1,
                    observation=_observation(1),
                )
                ledger.begin_mutation(
                    credit.key,
                    credit.binding,
                    now=2,
                    observation=_observation(2),
                )
                with self.assertRaises(TreeCapacityCreditFatal):
                    if operation == "consume":
                        ledger.consume(
                            credit.key,
                            credit.binding,
                            now=2,
                            observation=_observation(2),
                        )
                    elif operation == "release":
                        ledger.release(
                            credit.key,
                            credit.binding,
                            now=2,
                            observation=_observation(2),
                        )
                    elif operation == "begin":
                        ledger.begin_mutation(
                            credit.key,
                            credit.binding,
                            now=2,
                            observation=_observation(2),
                        )
                    elif operation == "wrong-binding":
                        ledger.consume(
                            credit.key,
                            replace(credit.binding, stage_id="stage-other"),
                            now=2,
                            observation=_observation(2),
                        )
                    elif operation == "changed-lease":
                        ledger.consume(
                            replace(credit.key, lease_id="changed-lease"),
                            credit.binding,
                            now=2,
                            observation=_observation(2),
                        )
                    else:
                        ledger.issue(
                            replace(credit.binding, parent_request_id=102),
                            lease_id="changed-lease",
                            nonce=credit.key.nonce,
                            ttl_seconds=10,
                            now=2,
                            observation=_observation(2),
                        )
                self.assertTrue(ledger.fatal)

    def test_expiry_releases_before_mutation_but_is_fatal_after(self) -> None:
        ledger = _ledger()
        credit = ledger.issue(
            _binding(),
            lease_id="lease",
            nonce=1,
            ttl_seconds=5,
            now=10,
            observation=_observation(),
        )
        ledger.advance(now=15, observation=_observation(1))
        snapshot = ledger.snapshot()
        self.assertEqual(snapshot.records[0].state, TreeCapacityCreditState.EXPIRED)
        self.assertEqual(snapshot.reserved_blocks, 0)
        self.assertFalse(
            ledger.release(
                credit.key,
                credit.binding,
                now=16,
                observation=_observation(2),
            )
        )

        mutated = _ledger()
        credit = mutated.issue(
            _binding(),
            lease_id="lease",
            nonce=1,
            ttl_seconds=5,
            now=10,
            observation=_observation(),
        )
        mutated.consume(
            credit.key,
            credit.binding,
            now=11,
            observation=_observation(1),
        )
        mutated.begin_mutation(
            credit.key,
            credit.binding,
            now=12,
            observation=_observation(2),
        )
        with self.assertRaises(TreeCapacityCreditFatal):
            mutated.advance(now=15, observation=_observation(3))
        self.assertIn("TTL", mutated.snapshot().fatal_reason or "")

    def test_floor_epoch_total_and_revision_drift_fail_closed(self) -> None:
        bad_observations = (
            _observation(1, free=19),  # 81 ordinary blocks cross the 80 floor.
            _observation(1, epoch=4),
            _observation(1, total=101, free=101),
        )
        for observation in bad_observations:
            with self.subTest(observation=observation):
                ledger = _ledger()
                with self.assertRaises(TreeCapacityCreditFatal):
                    ledger.advance(now=1, observation=observation)
                self.assertTrue(ledger.snapshot().fatal)

        ledger = _ledger()
        ledger.advance(now=1, observation=_observation(5, free=90))
        with self.assertRaises(TreeCapacityCreditFatal):
            ledger.advance(now=2, observation=_observation(4, free=90))

        ledger = _ledger()
        ledger.advance(now=1, observation=_observation(5, free=90))
        with self.assertRaises(TreeCapacityCreditFatal):
            ledger.advance(now=2, observation=_observation(5, free=91))

    def test_credit_counter_cannot_drift_past_mutating_limits(self) -> None:
        ledger = _ledger()
        credit = ledger.issue(
            _binding(blocks=8),
            lease_id="lease",
            nonce=1,
            ttl_seconds=20,
            now=0,
            observation=_observation(),
        )
        with self.assertRaises(TreeCapacityCreditFatal):
            ledger.advance(
                now=1,
                observation=_observation(
                    1, free=99, credit=1, credit_key=credit.key
                ),
            )

        ledger = _ledger()
        credit = ledger.issue(
            _binding(blocks=8),
            lease_id="lease",
            nonce=1,
            ttl_seconds=20,
            now=0,
            observation=_observation(),
        )
        ledger.consume(
            credit.key,
            credit.binding,
            now=1,
            observation=_observation(1),
        )
        ledger.begin_mutation(
            credit.key,
            credit.binding,
            now=1,
            observation=_observation(1),
        )
        with self.assertRaises(TreeCapacityCreditFatal):
            ledger.advance(
                now=2,
                observation=_observation(
                    2, free=91, credit=9, credit_key=credit.key
                ),
            )

        # Aggregate accounting alone would miss this: the two limits sum to
        # twenty, but the first eight-block credit must not borrow from the
        # second twelve-block credit.
        ledger = _ledger()
        first = ledger.issue(
            _binding(blocks=8),
            lease_id="first",
            nonce=1,
            ttl_seconds=20,
            now=0,
            observation=_observation(),
        )
        second = ledger.issue(
            _binding(blocks=12, parent=102),
            lease_id="second",
            nonce=2,
            ttl_seconds=20,
            now=0,
            observation=_observation(),
        )
        for item in (first, second):
            ledger.consume(
                item.key,
                item.binding,
                now=1,
                observation=_observation(1),
            )
            ledger.begin_mutation(
                item.key,
                item.binding,
                now=1,
                observation=_observation(1),
            )
        with self.assertRaises(TreeCapacityCreditFatal):
            ledger.advance(
                now=2,
                observation=_observation(
                    2, free=91, credit=9, credit_key=first.key
                ),
            )

    def test_completion_requires_physical_credit_usage_to_be_replenished(self) -> None:
        ledger = _ledger()
        first = ledger.issue(
            _binding(blocks=8),
            lease_id="first",
            nonce=1,
            ttl_seconds=20,
            now=0,
            observation=_observation(),
        )
        second = ledger.issue(
            _binding(blocks=6, parent=102),
            lease_id="second",
            nonce=2,
            ttl_seconds=20,
            now=0,
            observation=_observation(),
        )
        for credit in (first, second):
            ledger.consume(
                credit.key,
                credit.binding,
                now=1,
                observation=_observation(1),
            )
            ledger.begin_mutation(
                credit.key,
                credit.binding,
                now=1,
                observation=_observation(1),
            )
        ledger.advance(
            now=2,
            observation=_observation(
                2,
                free=90,
                credit_usages=((first.key, 8), (second.key, 2)),
            ),
        )
        # Completing the first is safe only after aggregate protected use fits
        # wholly inside the remaining second credit's six-block ceiling.
        ledger.complete_mutation(
            first.key,
            first.binding,
            now=3,
            observation=_observation(
                3, free=94, credit=6, credit_key=second.key
            ),
        )
        snapshot = ledger.snapshot()
        self.assertEqual(snapshot.reserved_blocks, 6)
        self.assertEqual(snapshot.mutating_limit_blocks, 6)
        ledger.complete_mutation(
            second.key,
            second.binding,
            now=4,
            observation=_observation(4, free=100, credit=0),
        )
        self.assertEqual(ledger.snapshot().reserved_blocks, 0)

        failed = _ledger()
        credit = failed.issue(
            _binding(blocks=8),
            lease_id="lease",
            nonce=1,
            ttl_seconds=20,
            now=0,
            observation=_observation(),
        )
        failed.consume(
            credit.key,
            credit.binding,
            now=1,
            observation=_observation(1),
        )
        failed.begin_mutation(
            credit.key,
            credit.binding,
            now=1,
            observation=_observation(1),
        )
        with self.assertRaises(TreeCapacityCreditFatal):
            failed.complete_mutation(
                credit.key,
                credit.binding,
                now=2,
                observation=_observation(
                    2, free=95, credit=5, credit_key=credit.key
                ),
            )

    def test_release_begin_interleavings_have_an_explicit_safety_boundary(self) -> None:
        release_first = _ledger()
        credit = release_first.issue(
            _binding(),
            lease_id="lease",
            nonce=1,
            ttl_seconds=20,
            now=0,
            observation=_observation(),
        )
        release_first.consume(
            credit.key,
            credit.binding,
            now=1,
            observation=_observation(1),
        )
        release_first.release(
            credit.key,
            credit.binding,
            now=2,
            observation=_observation(2),
        )
        with self.assertRaises(TreeCapacityCreditReplay):
            release_first.begin_mutation(
                credit.key,
                credit.binding,
                now=2,
                observation=_observation(2),
            )
        self.assertFalse(release_first.fatal)

        begin_first = _ledger()
        credit = begin_first.issue(
            _binding(),
            lease_id="lease",
            nonce=1,
            ttl_seconds=20,
            now=0,
            observation=_observation(),
        )
        begin_first.consume(
            credit.key,
            credit.binding,
            now=1,
            observation=_observation(1),
        )
        begin_first.begin_mutation(
            credit.key,
            credit.binding,
            now=2,
            observation=_observation(2),
        )
        with self.assertRaises(TreeCapacityCreditFatal):
            begin_first.release(
                credit.key,
                credit.binding,
                now=2,
                observation=_observation(2),
            )

    def test_concurrent_issuance_cannot_overreserve(self) -> None:
        ledger = _ledger(headroom=20)
        observation = _observation()

        def issue(nonce: int) -> bool:
            try:
                ledger.issue(
                    _binding(blocks=4, parent=1000 + nonce),
                    lease_id=f"lease-{nonce:02d}",
                    nonce=nonce,
                    ttl_seconds=50,
                    now=0,
                    observation=observation,
                )
                return True
            except TreeCapacityCreditUnavailable:
                return False

        with ThreadPoolExecutor(max_workers=16) as pool:
            results = tuple(pool.map(issue, range(1, 65)))
        self.assertEqual(sum(results), 5)
        self.assertEqual(ledger.snapshot().reserved_blocks, 20)
        self.assertFalse(ledger.fatal)

    def test_random_property_never_overreserves_headroom(self) -> None:
        rng = random.Random(0xC0FFEE)
        ledger = _ledger(headroom=31, ttl=10_000)
        live = []
        nonce = 0
        revision = 0
        now = 0.0
        for _ in range(1_000):
            now += 0.01
            action = rng.randrange(5)
            if action < 2:
                nonce += 1
                blocks = rng.randint(1, 12)
                try:
                    credit = ledger.issue(
                        _binding(
                            blocks=blocks,
                            lanes=rng.randint(1, 8),
                            parent=10_000 + nonce,
                            digest_byte=nonce % 256,
                        ),
                        lease_id=f"lease-{nonce}",
                        nonce=nonce,
                        ttl_seconds=100,
                        now=now,
                        observation=_observation(revision),
                    )
                    live.append(credit)
                except TreeCapacityCreditUnavailable:
                    pass
            elif live:
                index = rng.randrange(len(live))
                credit = live[index]
                revision += 1
                states = {
                    record.key: record.state for record in ledger.snapshot().records
                }
                state = states[credit.key]
                if state is TreeCapacityCreditState.ISSUED and rng.randrange(2):
                    ledger.consume(
                        credit.key,
                        credit.binding,
                        now=now,
                        observation=_observation(revision),
                    )
                elif state is TreeCapacityCreditState.CONSUMED and rng.randrange(2):
                    ledger.begin_mutation(
                        credit.key,
                        credit.binding,
                        now=now,
                        observation=_observation(revision),
                    )
                elif state is TreeCapacityCreditState.MUTATING:
                    ledger.complete_mutation(
                        credit.key,
                        credit.binding,
                        now=now,
                        observation=_observation(revision),
                    )
                    live.pop(index)
                else:
                    ledger.release(
                        credit.key,
                        credit.binding,
                        now=now,
                        observation=_observation(revision),
                    )
                    live.pop(index)
            else:
                revision += 1
                ledger.advance(now=now, observation=_observation(revision))

            snapshot = ledger.snapshot()
            self.assertGreaterEqual(snapshot.reserved_blocks, 0)
            self.assertLessEqual(snapshot.reserved_blocks, snapshot.headroom_blocks)
            self.assertLessEqual(
                snapshot.mutating_limit_blocks, snapshot.reserved_blocks
            )
            recomputed_reserved = sum(
                record.binding.block_count
                for record in snapshot.records
                if record.state
                in {
                    TreeCapacityCreditState.ISSUED,
                    TreeCapacityCreditState.CONSUMED,
                    TreeCapacityCreditState.MUTATING,
                }
            )
            recomputed_mutating = sum(
                record.binding.block_count
                for record in snapshot.records
                if record.state is TreeCapacityCreditState.MUTATING
            )
            self.assertEqual(snapshot.reserved_blocks, recomputed_reserved)
            self.assertEqual(snapshot.mutating_limit_blocks, recomputed_mutating)
            self.assertEqual(snapshot.ordinary_free_blocks, 69)

    def test_snapshot_is_stable_and_key_sorted(self) -> None:
        ledger = _ledger()
        for lease_id, nonce, parent in (
            ("z", 9, 109),
            ("a", 7, 107),
            ("m", 8, 108),
        ):
            ledger.issue(
                _binding(blocks=2, parent=parent),
                lease_id=lease_id,
                nonce=nonce,
                ttl_seconds=20,
                now=0,
                observation=_observation(),
            )
        first = ledger.snapshot()
        second = ledger.snapshot()
        self.assertEqual(first, second)
        self.assertEqual(
            [record.key.lease_id for record in first.records],
            ["a", "m", "z"],
        )
        self.assertIsInstance(first.records, tuple)

    def test_credit_limits_ttl_and_record_count_are_fail_closed(self) -> None:
        ledger = TreeCapacityCreditLedger(
            epoch=3,
            total_blocks=100,
            headroom_blocks=20,
            max_lanes_per_credit=4,
            max_ttl_seconds=5,
            max_credits_per_epoch=1,
        )
        with self.assertRaises(TreeCapacityCreditUnavailable):
            ledger.issue(
                _binding(lanes=5),
                lease_id="lease",
                nonce=1,
                ttl_seconds=5,
                now=0,
                observation=_observation(),
            )
        with self.assertRaises(TreeCapacityCreditValidationError):
            ledger.issue(
                _binding(lanes=4),
                lease_id="lease",
                nonce=1,
                ttl_seconds=6,
                now=0,
                observation=_observation(),
            )
        credit = ledger.issue(
            _binding(lanes=4),
            lease_id="lease",
            nonce=1,
            ttl_seconds=5,
            now=0,
            observation=_observation(),
        )
        ledger.release(
            credit.key,
            credit.binding,
            now=1,
            observation=_observation(1),
        )
        with self.assertRaises(TreeCapacityCreditUnavailable):
            ledger.issue(
                _binding(parent=102),
                lease_id="lease-2",
                nonce=2,
                ttl_seconds=5,
                now=1,
                observation=_observation(1),
            )


if __name__ == "__main__":
    unittest.main()
