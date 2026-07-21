"""Session KV retention: exact multi-turn reuse of the live pipeline cache.

Unit tests cover the longest-common-prefix bound and the retained-session
state machine (checkout, retention, TTL, budget, cancel release) against a
fake runner and a real socket pair, so the frames the root would send
(TRUNCATE / END) are asserted byte-for-byte through the actual protocol.

The integration tests (RUN_DISTRIBUTED_MODEL_TESTS=1) prove token-exact
parity: every turn generated with KV reuse must equal the same turn generated
by a full re-prefill of the complete history, including edited-history
divergence fallbacks and cancellation.
"""

from __future__ import annotations

import itertools
import os
import random
import socket
import threading
import unittest
from concurrent.futures import CancelledError as FutureCancelledError

import torch

from distributed_runtime.engine import (
    DistributedPipelineEngine,
    GenerationInput,
    PipelineEngineConfig,
    _GenerationJob,
    _longest_common_prefix,
)
from distributed_runtime.protocol import FrameType, TensorCodec, recv_frame


class LongestCommonPrefixTests(unittest.TestCase):
    def test_prefix_lengths(self) -> None:
        self.assertEqual(_longest_common_prefix((), ()), 0)
        self.assertEqual(_longest_common_prefix((1, 2, 3), (1, 2, 3)), 3)
        self.assertEqual(_longest_common_prefix((1, 2, 3), (1, 2, 3, 4, 5)), 3)
        self.assertEqual(_longest_common_prefix((1, 2, 3, 9), (1, 2, 3, 4, 5)), 3)
        self.assertEqual(_longest_common_prefix((9, 2), (1, 2)), 0)


class SessionConfigurationTests(unittest.TestCase):
    def test_retention_configuration_is_bounded(self) -> None:
        common = dict(model_name="fake", boundaries=(0, 2, 4))
        configured = PipelineEngineConfig(
            **common,
            max_retained_sessions=4,
            max_retained_session_tokens=10_000,
            retained_session_ttl_seconds=30.0,
        )
        self.assertEqual(configured.max_retained_sessions, 4)
        for invalid in (
            {"max_retained_sessions": -1},
            {"max_retained_sessions": True},
            {"max_retained_session_tokens": -5},
            {"retained_session_ttl_seconds": 0.0},
            {"retained_session_ttl_seconds": float("nan")},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(
                (TypeError, ValueError)
            ):
                PipelineEngineConfig(**common, **invalid)

    def test_generation_input_session_key_is_validated(self) -> None:
        ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        accepted = GenerationInput(1, ids, 4, session_key="chat-a")
        self.assertEqual(accepted.session_key, "chat-a")
        self.assertIsNone(GenerationInput(1, ids, 4).session_key)
        for invalid in ("", "   ", 123, "x" * 257):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                GenerationInput(1, ids, 4, session_key=invalid)


class _SessionRunner:
    """Root-runner double: tracks per-wire KV length and END calls only."""

    def __init__(self) -> None:
        self.tokens: dict[int, int] = {}
        self.ended: list[int] = []

    def begin(self, request_id: int) -> None:
        self.tokens[request_id] = 0

    def end(self, request_id: int) -> None:
        self.ended.append(request_id)
        self.tokens.pop(request_id, None)

    def truncate(self, request_id: int, token_count: int) -> None:
        if token_count > self.tokens[request_id]:
            raise ValueError("cannot truncate beyond the current KV")
        self.tokens[request_id] = token_count

    def sequence_length(self, request_id: int) -> int:
        return self.tokens[request_id]


def _bare_engine(
    *,
    max_retained_sessions: int = 4,
    max_retained_session_tokens: int = 0,
    retained_session_ttl_seconds: float = 60.0,
) -> DistributedPipelineEngine:
    engine = DistributedPipelineEngine.__new__(DistributedPipelineEngine)
    engine.config = PipelineEngineConfig(
        model_name="fake",
        boundaries=(0, 2, 4),
        max_retained_sessions=max_retained_sessions,
        max_retained_session_tokens=max_retained_session_tokens,
        retained_session_ttl_seconds=retained_session_ttl_seconds,
    )
    engine._retained_sessions = {}
    engine._session_retained_tokens = 0
    engine._session_reuse_hits = 0
    engine._session_reuse_misses = 0
    engine._session_busy_fallbacks = 0
    engine._session_divergence_fallbacks = 0
    engine._session_ttl_evictions = 0
    engine._session_budget_evictions = 0
    engine._session_cancel_releases = 0
    engine._session_reused_token_total = 0
    return engine


def _finished_job(
    ids: list[int],
    key: str,
    *,
    wire_id: int,
    step: int,
    token_ids: tuple[int, ...],
    kv_valid: int,
) -> _GenerationJob:
    job = _GenerationJob(
        request=GenerationInput(
            client_id=1,
            input_ids=torch.tensor([ids], dtype=torch.long),
            max_new_tokens=8,
            session_key=key,
        ),
        callback=None,
    )
    job.wire_id = wire_id
    job.step = step
    job.token_ids = list(token_ids)
    job.kv_valid = kv_valid
    return job


class SessionStateMachineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = _bare_engine()
        self.runner = _SessionRunner()
        self.downstream, self.peer = socket.socketpair()
        self.peer.settimeout(1.0)
        self.addCleanup(self.downstream.close)
        self.addCleanup(self.peer.close)

    def _peer_has_no_frame(self) -> bool:
        self.peer.settimeout(0.05)
        try:
            return self.peer.recv(1) == b""
        except (TimeoutError, socket.timeout):
            return True
        finally:
            self.peer.settimeout(1.0)

    def _retain_first_turn(self, key: str = "chat-a", wire_id: int = 7) -> None:
        # Turn 1 served prompt [1,2,3,4] and emitted [50,51]; the last emitted
        # token was never forwarded, so the physical/valid KV holds 5 tokens.
        self.runner.begin(wire_id)
        self.runner.tokens[wire_id] = 5
        job = _finished_job(
            [1, 2, 3, 4],
            key,
            wire_id=wire_id,
            step=3,
            token_ids=(50, 51),
            kv_valid=5,
        )
        self.assertTrue(
            self.engine._try_retain_session(job, self.runner, self.downstream)
        )

    def test_pure_extension_reuses_without_truncate(self) -> None:
        self._retain_first_turn()
        session = self.engine._retained_sessions["chat-a"]
        self.assertEqual(session.served_ids, (1, 2, 3, 4, 50, 51))
        self.assertEqual(session.kv_tokens, 5)
        self.assertEqual(session.kv_valid_tokens, 5)
        self.assertEqual(session.step, 4)
        self.assertEqual(self.engine._session_retained_tokens, 5)

        turn2 = _finished_job(
            [1, 2, 3, 4, 50, 51, 60, 61],
            "chat-a",
            wire_id=0,
            step=0,
            token_ids=(),
            kv_valid=0,
        )
        checked_out = self.engine._checkout_session(
            turn2, self.runner, self.downstream
        )
        self.assertIs(checked_out, session)
        self.assertTrue(session.busy)
        # LCP is 6 but only 5 KV tokens are valid: prefill resumes at 5.
        self.assertEqual(turn2.prefill_offset, 5)
        self.assertEqual(turn2.reused_tokens, 5)
        self.assertEqual(turn2.kv_valid, 5)
        self.assertEqual(self.runner.tokens[7], 5)
        self.assertTrue(self._peer_has_no_frame(), "no TRUNCATE frame expected")
        self.assertEqual(self.engine._session_reuse_hits, 1)

        # A concurrent turn of the same chat must not touch the busy wire.
        rival = _finished_job([1, 2], "chat-a", wire_id=0, step=0, token_ids=(), kv_valid=0)
        self.assertIsNone(
            self.engine._checkout_session(rival, self.runner, self.downstream)
        )
        self.assertEqual(self.engine._session_busy_fallbacks, 1)
        self.assertFalse(
            self.engine._try_retain_session(rival, self.runner, self.downstream),
            "a rival turn that did not check the session out retires normally",
        )

    def test_midway_divergence_truncates_to_common_prefix(self) -> None:
        self._retain_first_turn()
        edited = _finished_job(
            [1, 2, 999, 4, 50, 51],
            "chat-a",
            wire_id=0,
            step=0,
            token_ids=(),
            kv_valid=0,
        )
        session = self.engine._checkout_session(edited, self.runner, self.downstream)
        self.assertIsNotNone(session)
        self.assertEqual(edited.prefill_offset, 2)
        frame = recv_frame(self.peer)
        self.assertEqual(frame.frame_type, FrameType.TRUNCATE)
        self.assertEqual(frame.request_id, 7)
        self.assertEqual(frame.token_count, 2)
        self.assertEqual(self.runner.tokens[7], 2)
        self.assertEqual(self.engine._session_retained_tokens, 2)

    def test_first_token_divergence_releases_and_full_prefills(self) -> None:
        self._retain_first_turn()
        divergent = _finished_job(
            [999, 2, 3], "chat-a", wire_id=0, step=0, token_ids=(), kv_valid=0
        )
        self.assertIsNone(
            self.engine._checkout_session(divergent, self.runner, self.downstream)
        )
        frame = recv_frame(self.peer)
        self.assertEqual(frame.frame_type, FrameType.END)
        self.assertEqual(frame.request_id, 7)
        self.assertEqual(self.runner.ended, [7])
        self.assertEqual(self.engine._retained_sessions, {})
        self.assertEqual(self.engine._session_retained_tokens, 0)
        self.assertEqual(self.engine._session_divergence_fallbacks, 1)

    def test_ttl_expiry_sends_end(self) -> None:
        self._retain_first_turn()
        self.engine._retained_sessions["chat-a"].last_used -= 3_600
        self.engine._expire_retained_sessions(self.runner, self.downstream)
        frame = recv_frame(self.peer)
        self.assertEqual(frame.frame_type, FrameType.END)
        self.assertEqual(frame.request_id, 7)
        self.assertEqual(self.engine._retained_sessions, {})
        self.assertEqual(self.engine._session_ttl_evictions, 1)

    def test_session_budget_evicts_least_recently_used(self) -> None:
        self.engine = _bare_engine(max_retained_sessions=1)
        self._retain_first_turn(key="chat-a", wire_id=7)
        self._retain_first_turn(key="chat-b", wire_id=8)
        frame = recv_frame(self.peer)
        self.assertEqual(frame.frame_type, FrameType.END)
        self.assertEqual(frame.request_id, 7, "LRU chat-a must be evicted first")
        self.assertEqual(list(self.engine._retained_sessions), ["chat-b"])
        self.assertEqual(self.engine._session_budget_evictions, 1)
        self.assertEqual(self.engine._session_retained_tokens, 5)

    def test_token_budget_evicts_sessions(self) -> None:
        self.engine = _bare_engine(max_retained_sessions=8, max_retained_session_tokens=6)
        self._retain_first_turn(key="chat-a", wire_id=7)
        self._retain_first_turn(key="chat-b", wire_id=8)
        frame = recv_frame(self.peer)
        self.assertEqual(frame.frame_type, FrameType.END)
        self.assertEqual(frame.request_id, 7)
        self.assertEqual(self.engine._session_retained_tokens, 5)

    def test_cancelled_turn_drops_the_retained_session(self) -> None:
        self._retain_first_turn()
        turn2 = _finished_job(
            [1, 2, 3, 4, 50, 51, 60],
            "chat-a",
            wire_id=0,
            step=0,
            token_ids=(),
            kv_valid=0,
        )
        self.assertIsNotNone(
            self.engine._checkout_session(turn2, self.runner, self.downstream)
        )
        turn2.cancel_requested.set()
        self.engine._drop_job_session(turn2)
        self.assertEqual(self.engine._retained_sessions, {})
        self.assertEqual(self.engine._session_retained_tokens, 0)
        self.assertEqual(self.engine._session_cancel_releases, 1)
        self.assertFalse(
            self.engine._try_retain_session(turn2, self.runner, self.downstream),
            "a cancelled turn must never retain",
        )


@unittest.skipUnless(
    os.environ.get("RUN_DISTRIBUTED_MODEL_TESTS") == "1",
    "set RUN_DISTRIBUTED_MODEL_TESTS=1 to run the physical session-reuse parity",
)
class SessionReuseIntegrationTests(unittest.TestCase):
    """Exact parity of KV reuse against full re-prefill on the real pipeline."""

    MODEL_NAME = os.environ.get(
        "DISTRIBUTED_TEST_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct"
    )
    engine: DistributedPipelineEngine

    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = DistributedPipelineEngine(
            PipelineEngineConfig(
                model_name=cls.MODEL_NAME,
                boundaries=(0, 15, 30),
                codec=TensorCodec.FP32,
                threads_per_stage=2,
                max_active_sequences=2,
                max_pending_requests=16,
                prefill_chunk_tokens=3,
                speculative_max_draft_tokens=2,
                max_retained_sessions=4,
                retained_session_ttl_seconds=300.0,
            )
        )
        cls.client_ids = itertools.count(1)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.engine.close()

    def _generate(self, ids: list[int], key: str | None = None, max_new: int = 5):
        request = GenerationInput(
            client_id=next(self.client_ids),
            input_ids=torch.tensor([ids], dtype=torch.long),
            max_new_tokens=max_new,
            session_key=key,
        )
        return self.engine.generate([request])[0]

    def test_a_ten_multiturn_conversations_are_token_exact(self) -> None:
        rng = random.Random(7)
        hits_before = self.engine.session_stats["reuse_hits"]
        for conversation in range(10):
            key = f"parity-chat-{conversation}"
            history = [rng.randrange(200, 2000) for _ in range(rng.randrange(4, 9))]
            for turn in range(3):
                if turn > 0:
                    history = history + [
                        rng.randrange(200, 2000)
                        for _ in range(rng.randrange(2, 5))
                    ]
                expected = self._generate(list(history))
                actual = self._generate(list(history), key=key)
                self.assertEqual(
                    actual.token_ids,
                    expected.token_ids,
                    f"conversation {conversation} turn {turn} diverged from "
                    "the full re-prefill reference",
                )
                if turn > 0:
                    self.assertGreater(
                        actual.reused_kv_tokens,
                        0,
                        f"conversation {conversation} turn {turn} did not reuse KV",
                    )
                history = history + list(actual.token_ids)
        stats = self.engine.session_stats
        self.assertGreaterEqual(stats["reuse_hits"] - hits_before, 20)
        self.assertTrue(self.engine.healthy)

    def test_b_edited_history_truncates_to_common_prefix(self) -> None:
        key = "edited-chat"
        history = [300 + index for index in range(8)]
        first = self._generate(list(history), key=key)
        follow_up = history + list(first.token_ids) + [500, 501]
        edited = list(follow_up)
        edited[3] = 999  # user edited a message in the middle of the history
        expected = self._generate(list(edited))
        actual = self._generate(list(edited), key=key)
        self.assertEqual(actual.token_ids, expected.token_ids)
        self.assertEqual(
            actual.reused_kv_tokens,
            3,
            "reuse must stop exactly at the longest common prefix",
        )

    def test_c_fully_divergent_history_falls_back_to_full_prefill(self) -> None:
        key = "restart-chat"
        history = [400 + index for index in range(6)]
        self._generate(list(history), key=key)
        fallbacks_before = self.engine.session_stats["divergence_fallbacks"]
        divergent = [999] + [700 + index for index in range(5)]
        expected = self._generate(list(divergent))
        actual = self._generate(list(divergent), key=key)
        self.assertEqual(actual.token_ids, expected.token_ids)
        self.assertEqual(actual.reused_kv_tokens, 0)
        self.assertEqual(
            self.engine.session_stats["divergence_fallbacks"],
            fallbacks_before + 1,
        )

    def test_d_cancel_frees_the_session_and_next_turn_stays_exact(self) -> None:
        key = "cancelled-chat"
        history = [600 + index for index in range(6)]
        first = self._generate(list(history), key=key)
        follow_up = history + list(first.token_ids) + [610, 611]

        cancelled_id = next(self.client_ids)

        def cancel_on_first_token(
            client_id: int, _token: int, _index: int, _arrived: float
        ) -> None:
            self.engine.cancel(client_id)

        future = self.engine.submit(
            [
                GenerationInput(
                    client_id=cancelled_id,
                    input_ids=torch.tensor([follow_up], dtype=torch.long),
                    max_new_tokens=6,
                    session_key=key,
                )
            ],
            cancel_on_first_token,
        )[0]
        with self.assertRaises(FutureCancelledError):
            future.result(timeout=60)
        self.assertGreaterEqual(self.engine.session_stats["cancel_releases"], 1)

        # The chat's KV was freed everywhere: the next turn must re-prefill in
        # full and still match the reference exactly.
        expected = self._generate(list(follow_up))
        actual = self._generate(list(follow_up), key=key)
        self.assertEqual(actual.token_ids, expected.token_ids)
        self.assertEqual(actual.reused_kv_tokens, 0)
        self.assertTrue(self.engine.healthy)


if __name__ == "__main__":
    unittest.main()
