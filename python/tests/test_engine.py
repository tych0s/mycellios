from __future__ import annotations

from collections import deque
import os
import queue
import socket
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from distributed_runtime.engine import (
    DistributedPipelineEngine,
    GenerationInput,
    PipelineEngineConfig,
    _GenerationJob,
    _PreparedRootWave,
    _resolve_verified_tokens,
    _speculation_load_profile,
)
from distributed_runtime.macro_wave import KVVersion, MacroWaveState
from distributed_runtime.macro_wave_adapter import linear_draft_to_macro_wave
from distributed_runtime.model import load_tokenizer, reference_generate
from distributed_runtime.protocol import (
    Frame,
    FrameType,
    LinkEmulator,
    TensorCodec,
    decode_tensor,
    recv_frame,
    token_payload,
    verify_result_payload,
)
from distributed_runtime.speculation import (
    AdaptiveSpeculationConfig,
    AdaptiveSpeculationController,
)


class EngineUnitTests(unittest.TestCase):
    def test_exact_speculative_resolution_accepts_prefix_and_returns_correction(self) -> None:
        self.assertEqual(
            _resolve_verified_tokens((10, 20), (10, 20, 30)),
            (2, (10, 20, 30)),
        )
        self.assertEqual(
            _resolve_verified_tokens((10, 20), (99, 20, 30)),
            (0, (99,)),
        )
        self.assertEqual(
            _resolve_verified_tokens((10, 20), (10, 99, 30)),
            (1, (10, 99)),
        )
        with self.assertRaisesRegex(ValueError, "plus bonus"):
            _resolve_verified_tokens((10, 20), (10, 20))

    def test_prefill_and_speculation_configuration_is_bounded(self) -> None:
        common = dict(model_name="fake", boundaries=(0, 2, 4))
        configured = PipelineEngineConfig(
            **common,
            prefill_chunk_tokens=32,
            sealed_wave_tokens=9,
            max_prefill_chunk_tokens=32,
            speculative_max_draft_tokens=8,
            speculation_minimum_speedup=1.1,
        )
        self.assertEqual(configured.prefill_chunk_tokens, 32)
        self.assertEqual(configured.speculative_max_draft_tokens, 8)
        self.assertEqual(configured.sealed_wave_token_limit, 9)
        self.assertEqual(configured.prefill_token_limit, 32)
        self.assertEqual(configured.root_batch_window_ms, 0.5)
        self.assertEqual(configured.route_probe_interval_seconds, 5.0)
        self.assertEqual(configured.route_probe_timeout_seconds, 10.0)
        for invalid in (-1, 17, True):
            with self.subTest(max_draft_tokens=invalid), self.assertRaises(
                (TypeError, ValueError)
            ):
                PipelineEngineConfig(
                    **common,
                    speculative_max_draft_tokens=invalid,
                )
        with self.assertRaisesRegex(ValueError, "must be supplied together"):
            PipelineEngineConfig(**common, sealed_wave_tokens=1)
        with self.assertRaisesRegex(ValueError, "smaller than the VERIFY"):
            PipelineEngineConfig(
                **common,
                speculative_max_draft_tokens=4,
                sealed_wave_tokens=4,
                max_prefill_chunk_tokens=32,
            )
        with self.assertRaisesRegex(ValueError, "require a draft provider"):
            PipelineEngineConfig(
                **common,
                sealed_wave_tokens=2,
                max_prefill_chunk_tokens=32,
            )
        for invalid in (-1, 101, float("inf")):
            with self.subTest(root_batch_window_ms=invalid), self.assertRaises(
                ValueError
            ):
                PipelineEngineConfig(**common, root_batch_window_ms=invalid)
        for invalid in (-1, float("inf"), float("nan")):
            with self.subTest(route_probe_interval_seconds=invalid), self.assertRaises(
                ValueError
            ):
                PipelineEngineConfig(
                    **common,
                    route_probe_interval_seconds=invalid,
                )
        for invalid in (0, -1, float("inf"), float("nan")):
            with self.subTest(route_probe_timeout_seconds=invalid), self.assertRaises(
                ValueError
            ):
                PipelineEngineConfig(
                    **common,
                    route_probe_timeout_seconds=invalid,
                )

    def test_speculation_measurements_are_isolated_by_active_load(self) -> None:
        self.assertEqual(_speculation_load_profile(1), "load-1")
        self.assertEqual(_speculation_load_profile(2), "load-2")
        self.assertEqual(_speculation_load_profile(3), "load-3-4")
        self.assertEqual(_speculation_load_profile(4), "load-3-4")
        self.assertEqual(_speculation_load_profile(8), "load-5-plus")
        with self.assertRaises(ValueError):
            _speculation_load_profile(0)

        engine = DistributedPipelineEngine.__new__(DistributedPipelineEngine)
        engine.speculation_controller = AdaptiveSpeculationController(
            AdaptiveSpeculationConfig(max_draft_tokens=2)
        )
        engine.speculation_controller.record_rtt(17.0)
        engine._speculation_controllers = {}
        load_one = engine._speculation_controller_for_profile_locked("load-1")
        load_four = engine._speculation_controller_for_profile_locked("load-3-4")
        self.assertIs(load_one, engine.speculation_controller)
        self.assertIsNot(load_one, load_four)
        self.assertEqual(load_four.rtt_ewma_ms, 17.0)
        self.assertIs(
            engine._speculation_controller_for_profile_locked("load-3-4"),
            load_four,
        )

    def test_decode_preparation_uses_width_one_macro_wave_and_preserves_probe_metrics(self) -> None:
        engine = _root_batch_test_engine(prefill_chunk_tokens=0)
        engine.config.speculative_max_draft_tokens = 2
        engine.config.speculation_probe = True
        engine.draft_provider = _FixedDraftProvider((20, 21))
        controller = AdaptiveSpeculationController(
            AdaptiveSpeculationConfig(
                max_draft_tokens=2,
                candidate_sizes=(2,),
                min_token_history=0,
                min_classic_observations=1,
                min_verify_observations=1,
            )
        )
        controller.record_classic(latency_seconds=0.1, transferred_bytes=100)
        engine.speculation_controller = controller
        engine._speculation_lock = threading.Lock()
        engine._speculation_controllers = {}
        engine._speculation_enabled_decisions = 0
        engine._speculation_disabled_decisions = 0
        engine._speculation_probe_waves = 0
        engine._speculation_decision_reasons = {}
        engine._speculation_selected_sizes = {}
        runner = _FakeRootBatchRunner()
        runner.active = {7: 3}
        job = _GenerationJob(
            GenerationInput(601, torch.tensor([[1, 2, 3]]), 5),
            None,
            wire_id=7,
            step=2,
            token_ids=[10],
        )

        wave = engine._prepare_decode_wave(
            job,
            runner,
            active_sequences=1,
        )

        self.assertEqual(wave.frame_type, FrameType.VERIFY)
        self.assertEqual(wave.input_ids.tolist(), [[10, 20, 21]])
        proposal = job.verify_proposal
        self.assertIsNotNone(proposal)
        assert proposal is not None
        self.assertTrue(proposal.is_linear)
        self.assertEqual(proposal.width, 1)
        self.assertEqual(proposal.linear_tokens, (20, 21))
        self.assertEqual(proposal.wave_identity.request_id, "7")
        self.assertEqual(proposal.wave_identity.ordinal, 2)
        self.assertEqual(proposal.tree.ledger.base_version, KVVersion(3))
        self.assertEqual(job.verify_base_tokens, 3)
        self.assertEqual(engine._speculation_enabled_decisions, 0)
        self.assertEqual(engine._speculation_disabled_decisions, 1)
        self.assertEqual(engine._speculation_probe_waves, 1)
        self.assertEqual(
            engine._speculation_decision_reasons,
            {"verification_warmup": 1},
        )
        self.assertEqual(engine._speculation_selected_sizes, {2: 1})

    def test_route_probe_measures_the_full_control_path_and_feeds_speculation(self) -> None:
        engine = DistributedPipelineEngine.__new__(DistributedPipelineEngine)
        engine.config = SimpleNamespace(
            route_probe_interval_seconds=1.0,
            route_probe_timeout_seconds=2.0,
        )
        engine.pipeline_id = 55
        engine._route_probe_lock = threading.Lock()
        engine._route_probe_sent_at = None
        engine._route_probe_deadline_at = None
        engine._route_probe_pending_step = None
        engine._route_probe_sequence = 0
        engine._route_probe_next_at = 0.0
        engine._route_probe_returns = queue.Queue()
        engine._route_rtt_ms = None
        engine._route_probe_count = 0
        engine._speculation_lock = threading.Lock()
        engine.speculation_controller = AdaptiveSpeculationController(
            AdaptiveSpeculationConfig(max_draft_tokens=2)
        )
        engine._speculation_controllers = {}
        root, child = socket.socketpair()
        try:
            engine._send_route_probe_if_due(root, LinkEmulator())
            ping = recv_frame(child)
            self.assertEqual(ping.frame_type, FrameType.PING)
            self.assertEqual(ping.request_id, 55)
            self.assertEqual(ping.step, 1)
            engine._consume_route_pong(
                Frame(FrameType.PONG, 0, 55, ping.step, 0, 0, b""),
                time.perf_counter(),
            )
            route_rtt_ms, count = engine._route_probe_snapshot()
            self.assertEqual(count, 1)
            self.assertIsNotNone(route_rtt_ms)
            self.assertGreaterEqual(float(route_rtt_ms), 0.0)
            self.assertEqual(
                engine.speculation_controller.rtt_ewma_ms,
                route_rtt_ms,
            )
            with self.assertRaisesRegex(RuntimeError, "unsolicited"):
                engine._consume_route_pong(
                    Frame(FrameType.PONG, 0, 55, ping.step, 0, 0, b""),
                    time.perf_counter(),
                )

            with engine._route_probe_lock:
                engine._route_probe_next_at = 0.0
            engine._send_route_probe_if_due(root, LinkEmulator())
            second_ping = recv_frame(child)
            self.assertEqual(second_ping.step, 2)
            with engine._route_probe_lock:
                engine._route_probe_deadline_at = time.perf_counter() - 1.0
            with self.assertRaisesRegex(TimeoutError, "probe 2 timed out"):
                engine._check_route_probe_timeout()
        finally:
            root.close()
            child.close()

    def test_timely_queued_pong_wins_over_a_late_scheduler_timeout_check(self) -> None:
        engine = DistributedPipelineEngine.__new__(DistributedPipelineEngine)
        engine.config = SimpleNamespace(
            route_probe_interval_seconds=1.0,
            route_probe_timeout_seconds=2.0,
        )
        engine.pipeline_id = 77
        engine._route_probe_lock = threading.Lock()
        engine._route_probe_sent_at = 10.0
        engine._route_probe_deadline_at = 20.0
        engine._route_probe_pending_step = 7
        engine._route_probe_next_at = 0.0
        engine._route_probe_returns = queue.Queue()
        engine._route_probe_returns.put(
            (Frame(FrameType.PONG, 0, 77, 7, 0, 0, b""), 19.5)
        )
        engine._route_rtt_ms = None
        engine._route_probe_count = 0
        engine._speculation_lock = threading.Lock()
        engine.speculation_controller = None
        engine._speculation_controllers = {}

        with mock.patch(
            "distributed_runtime.engine.time.perf_counter",
            return_value=21.0,
        ):
            engine._check_route_probe_timeout()

        self.assertEqual(engine._route_probe_count, 1)
        self.assertEqual(engine._route_rtt_ms, 9_500.0)
        self.assertIsNone(engine._route_probe_sent_at)
        self.assertIsNone(engine._route_probe_pending_step)

    def test_route_probe_is_not_sent_after_scheduler_stop(self) -> None:
        engine = DistributedPipelineEngine.__new__(DistributedPipelineEngine)
        engine.config = SimpleNamespace(
            route_probe_interval_seconds=1.0,
            route_probe_timeout_seconds=2.0,
        )
        engine.pipeline_id = 88
        engine._closed = False
        engine._scheduler_stop = threading.Event()
        engine._scheduler_stop.set()
        engine._route_probe_lock = threading.Lock()
        engine._route_probe_sent_at = None
        engine._route_probe_deadline_at = None
        engine._route_probe_pending_step = None
        engine._route_probe_sequence = 0
        engine._route_probe_next_at = 0.0
        root, child = socket.socketpair()
        child.settimeout(0.05)
        try:
            engine._send_route_probe_if_due(root, LinkEmulator())
            with self.assertRaises(socket.timeout):
                child.recv(1)
            self.assertEqual(engine._route_probe_sequence, 0)
            self.assertIsNone(engine._route_probe_sent_at)
        finally:
            root.close()
            child.close()

    def test_idle_scheduler_sentinel_sends_shutdown_without_a_final_ping(self) -> None:
        engine = DistributedPipelineEngine.__new__(DistributedPipelineEngine)
        engine.config = SimpleNamespace(
            route_probe_interval_seconds=1.0,
            route_probe_timeout_seconds=2.0,
            one_way_delay_ms=0.0,
            bandwidth_mbps=0.0,
            max_active_sequences=1,
        )
        engine.pipeline_id = 99
        engine._runner = _FakeRootBatchRunner()
        root, child = socket.socketpair()
        child.settimeout(0.05)
        engine._downstream = root
        engine._received_frames = queue.Queue()
        engine._route_probe_returns = queue.Queue()
        engine._submission_queue = queue.Queue()
        engine._submission_queue.put(None)
        engine._deferred_batches = deque()
        engine._scheduler_stop = threading.Event()
        engine._route_probe_lock = threading.Lock()
        engine._route_probe_sent_at = None
        engine._route_probe_deadline_at = None
        engine._route_probe_pending_step = None
        engine._route_probe_sequence = 0
        engine._route_probe_next_at = 0.0
        engine._route_rtt_ms = None
        engine._route_probe_count = 0
        engine._closed = False
        engine._shutdown_sent = False
        engine._state_lock = threading.Lock()
        engine._fatal_error = None
        engine._jobs_by_client = {}
        try:
            engine._scheduler_loop()
            shutdown = recv_frame(child)
            self.assertEqual(shutdown.frame_type, FrameType.SHUTDOWN)
            with self.assertRaises(socket.timeout):
                child.recv(1)
            self.assertEqual(engine._route_probe_sequence, 0)
            self.assertIsNone(engine._fatal_error)
        finally:
            root.close()
            child.close()

    def test_root_admission_and_decode_use_one_physical_forward_per_wave(self) -> None:
        engine = _root_batch_test_engine(prefill_chunk_tokens=0)
        runner = _FakeRootBatchRunner()
        root, child = socket.socketpair()
        active: dict[int, _GenerationJob] = {}
        jobs = [
            _GenerationJob(
                GenerationInput(client_id, torch.tensor([[1, 2, 3]]), 3),
                None,
            )
            for client_id in (101, 202)
        ]
        try:
            engine._admit_batch(
                jobs,
                active,
                runner,
                root,
                LinkEmulator(),
            )
            initial = tuple(recv_frame(child) for _ in range(4))
            self.assertEqual(
                [frame.frame_type for frame in initial],
                [
                    FrameType.BEGIN,
                    FrameType.BEGIN,
                    FrameType.ACTIVATION,
                    FrameType.ACTIVATION,
                ],
            )
            self.assertEqual(runner.batch_calls, [((1, 2), (3, 3))])
            self.assertEqual(runner.sequential_calls, [])
            self.assertEqual([job.prefill_offset for job in jobs], [3, 3])
            self.assertTrue(
                all(decode_tensor(frame).shape == (1, 3, 4) for frame in initial[2:])
            )

            prepared = []
            arrived = time.perf_counter()
            for request_id, token in ((1, 10), (2, 11)):
                wave = engine._handle_return_value(
                    (
                        Frame(
                            FrameType.TOKEN,
                            0,
                            request_id,
                            0,
                            0,
                            0,
                            token_payload(token),
                        ),
                        arrived,
                    ),
                    active,
                    runner,
                    root,
                )
                self.assertIsNotNone(wave)
                prepared.append(wave)
            engine._dispatch_root_waves(
                prepared,
                runner,
                root,
                LinkEmulator(),
            )
            decode_frames = (recv_frame(child), recv_frame(child))
            self.assertEqual(
                [frame.frame_type for frame in decode_frames],
                [FrameType.ACTIVATION, FrameType.ACTIVATION],
            )
            self.assertEqual([frame.step for frame in decode_frames], [1, 1])
            self.assertEqual(runner.batch_calls[-1], ((1, 2), (1, 1)))
            self.assertEqual(engine.root_batch_stats["physical_batch_calls"], 2)
            self.assertEqual(engine.root_batch_stats["physical_batch_items"], 4)
            self.assertEqual(engine.root_batch_stats["model_forward_calls"], 2)
            self.assertEqual(engine.root_batch_stats["sequential_items"], 0)
        finally:
            root.close()
            child.close()

    def test_root_chunked_prefill_batches_acks_and_falls_back_by_exact_key(self) -> None:
        engine = _root_batch_test_engine(prefill_chunk_tokens=2)
        runner = _FakeRootBatchRunner()
        root, child = socket.socketpair()
        jobs = [
            _GenerationJob(
                GenerationInput(client_id, torch.tensor([[1, 2, 3, 4]]), 2),
                None,
                wire_id=request_id,
                prefill_offset=2,
            )
            for client_id, request_id in ((101, 1), (202, 2))
        ]
        runner.active = {1: 2, 2: 2}
        active = {1: jobs[0], 2: jobs[1]}
        try:
            waves = []
            for request_id in (1, 2):
                wave = engine._handle_return_value(
                    (
                        Frame(FrameType.PREFILL_ACK, 0, request_id, 0, 0, 0, b""),
                        time.perf_counter(),
                    ),
                    active,
                    runner,
                    root,
                )
                self.assertIsNotNone(wave)
                waves.append(wave)
            engine._dispatch_root_waves(waves, runner, root, LinkEmulator())
            frames = (recv_frame(child), recv_frame(child))
            self.assertEqual(
                [frame.frame_type for frame in frames],
                [FrameType.ACTIVATION, FrameType.ACTIVATION],
            )
            self.assertEqual(runner.batch_calls, [((1, 2), (2, 2))])
            self.assertEqual([job.prefill_offset for job in jobs], [4, 4])

            unsupported = _FakeRootBatchRunner(batchable=False)
            unsupported.active = {11: 0, 22: 0}
            fallback_engine = _root_batch_test_engine(prefill_chunk_tokens=0)
            fallback_jobs = [
                _PreparedRootWave(
                    _GenerationJob(
                        GenerationInput(client_id, torch.tensor([[1, 2]]), 2),
                        None,
                        wire_id=request_id,
                    ),
                    torch.tensor([[1, 2]]),
                    FrameType.ACTIVATION,
                    prefill_end=2,
                )
                for client_id, request_id in ((301, 11), (302, 22))
            ]
            fallback_engine._dispatch_root_waves(
                fallback_jobs,
                unsupported,
                root,
                LinkEmulator(),
            )
            fallback_frames = (recv_frame(child), recv_frame(child))
            self.assertEqual(
                [frame.request_id for frame in fallback_frames], [11, 22]
            )
            self.assertEqual(unsupported.batch_calls, [])
            self.assertEqual(unsupported.sequential_calls, [(11, 2), (22, 2)])
            self.assertEqual(fallback_engine.root_batch_stats["sequential_items"], 2)
            self.assertEqual(fallback_engine.root_batch_stats["physical_batch_calls"], 0)
        finally:
            root.close()
            child.close()

    def test_speculative_rejection_truncates_before_batched_correction_wave(self) -> None:
        engine = _root_batch_test_engine(prefill_chunk_tokens=0)
        controller = _RecordingSpeculationController()
        engine.speculation_controller = controller
        engine._speculation_lock = threading.Lock()
        engine._speculation_controllers = {"load-1": controller}
        runner = _FakeRootBatchRunner()
        runner.active = {1: 5, 2: 5}
        proposals = [
            linear_draft_to_macro_wave(
                (20,),
                request_id=request_id,
                ordinal=1,
                base_prefix_tokens=(1, 2, 3, 10),
                parent_kv_version=KVVersion(3),
            )
            for request_id in (1, 2)
        ]
        jobs = [
            _GenerationJob(
                GenerationInput(client_id, torch.tensor([[1, 2, 3]]), 4),
                None,
                wire_id=request_id,
                step=1,
                started_at=time.perf_counter() - 1,
                token_ids=[10],
                arrivals=[time.perf_counter() - 0.5],
                wave_started_at=time.perf_counter() - 0.1,
                verify_proposal=proposal,
                verify_base_tokens=3,
            )
            for (client_id, request_id), proposal in zip(
                ((401, 1), (402, 2)),
                proposals,
                strict=True,
            )
        ]
        active = {1: jobs[0], 2: jobs[1]}
        root, child = socket.socketpair()
        try:
            waves = []
            for request_id in (1, 2):
                wave = engine._handle_return_value(
                    (
                        Frame(
                            FrameType.VERIFY_RESULT,
                            0,
                            request_id,
                            1,
                            2,
                            0,
                            verify_result_payload((99, 77)),
                        ),
                        time.perf_counter(),
                    ),
                    active,
                    runner,
                    root,
                )
                self.assertIsNotNone(wave)
                waves.append(wave)
            truncates = (recv_frame(child), recv_frame(child))
            self.assertEqual(
                [frame.frame_type for frame in truncates],
                [FrameType.TRUNCATE, FrameType.TRUNCATE],
            )
            self.assertEqual([frame.token_count for frame in truncates], [4, 4])
            self.assertEqual(runner.truncations, [(1, 4), (2, 4)])

            engine._dispatch_root_waves(waves, runner, root, LinkEmulator())
            corrections = (recv_frame(child), recv_frame(child))
            self.assertEqual(
                [frame.frame_type for frame in corrections],
                [FrameType.ACTIVATION, FrameType.ACTIVATION],
            )
            self.assertEqual(runner.batch_calls, [((1, 2), (1, 1))])
            self.assertEqual([job.token_ids for job in jobs], [[10, 99], [10, 99]])
            self.assertTrue(all(job.verify_proposal is None for job in jobs))
            self.assertTrue(
                all(
                    proposal.tree.state is MacroWaveState.COMMITTED
                    for proposal in proposals
                )
            )
            self.assertEqual(controller.verifications, [(1, 0), (1, 0)])
        finally:
            root.close()
            child.close()

    def test_speculative_bonus_commits_without_truncate_and_remains_pending(self) -> None:
        engine = _root_batch_test_engine(prefill_chunk_tokens=0)
        controller = _RecordingSpeculationController()
        engine.speculation_controller = controller
        engine._speculation_lock = threading.Lock()
        engine._speculation_controllers = {"load-1": controller}
        runner = _FakeRootBatchRunner()
        runner.active = {1: 6}
        proposal = linear_draft_to_macro_wave(
            (20, 21),
            request_id=1,
            ordinal=1,
            base_prefix_tokens=(1, 2, 3, 10),
            parent_kv_version=KVVersion(3),
        )
        job = _GenerationJob(
            GenerationInput(501, torch.tensor([[1, 2, 3]]), 8),
            None,
            wire_id=1,
            step=1,
            started_at=time.perf_counter() - 1,
            token_ids=[10],
            arrivals=[time.perf_counter() - 0.5],
            wave_started_at=time.perf_counter() - 0.1,
            verify_proposal=proposal,
            verify_base_tokens=3,
        )
        active = {1: job}
        root, child = socket.socketpair()
        try:
            wave = engine._handle_return_value(
                (
                    Frame(
                        FrameType.VERIFY_RESULT,
                        0,
                        1,
                        1,
                        3,
                        0,
                        verify_result_payload((20, 21, 99)),
                    ),
                    time.perf_counter(),
                ),
                active,
                runner,
                root,
            )
            self.assertIsNotNone(wave)
            assert wave is not None
            self.assertEqual(wave.frame_type, FrameType.ACTIVATION)
            self.assertEqual(wave.input_ids.tolist(), [[99]])
            self.assertEqual(job.token_ids, [10, 20, 21, 99])
            self.assertIsNone(job.verify_proposal)
            self.assertEqual(job.verify_base_tokens, 0)
            self.assertEqual(runner.truncations, [])
            self.assertEqual(proposal.tree.state, MacroWaveState.COMMITTED)
            self.assertEqual(controller.verifications, [(2, 2)])

            engine._dispatch_root_waves(
                [wave],
                runner,
                root,
                LinkEmulator(),
            )
            continuation = recv_frame(child)
            self.assertEqual(continuation.frame_type, FrameType.ACTIVATION)
            self.assertEqual(continuation.step, 2)
            self.assertEqual(runner.active[1], 7)
        finally:
            root.close()
            child.close()

    def test_speculative_eos_stops_without_dispatching_correction_or_truncate(self) -> None:
        engine = _root_batch_test_engine(prefill_chunk_tokens=0)
        controller = _RecordingSpeculationController()
        engine.speculation_controller = controller
        engine._speculation_lock = threading.Lock()
        engine._speculation_controllers = {"load-1": controller}
        runner = _FakeRootBatchRunner()
        runner.active = {1: 5}
        proposal = linear_draft_to_macro_wave(
            (20,),
            request_id=1,
            ordinal=1,
            base_prefix_tokens=(1, 2, 3, 10),
            parent_kv_version=KVVersion(3),
        )
        job = _GenerationJob(
            GenerationInput(
                701,
                torch.tensor([[1, 2, 3]]),
                8,
                frozenset({99}),
            ),
            None,
            wire_id=1,
            step=1,
            started_at=time.perf_counter() - 1,
            token_ids=[10],
            arrivals=[time.perf_counter() - 0.5],
            wave_started_at=time.perf_counter() - 0.1,
            verify_proposal=proposal,
            verify_base_tokens=3,
        )
        engine._jobs_by_client = {job.request.client_id: job}
        active = {1: job}
        root, child = socket.socketpair()
        try:
            wave = engine._handle_return_value(
                (
                    Frame(
                        FrameType.VERIFY_RESULT,
                        0,
                        1,
                        1,
                        2,
                        0,
                        verify_result_payload((99, 77)),
                    ),
                    time.perf_counter(),
                ),
                active,
                runner,
                root,
            )

            self.assertIsNone(wave)
            self.assertEqual(active, {})
            self.assertEqual(job.token_ids, [10, 99])
            self.assertIsNone(job.verify_proposal)
            self.assertEqual(job.verify_base_tokens, 0)
            self.assertEqual(runner.truncations, [])
            self.assertNotIn(1, runner.active)
            self.assertEqual(proposal.tree.state, MacroWaveState.COMMITTED)
            self.assertEqual(controller.verifications, [(1, 0)])
            terminal = recv_frame(child)
            self.assertEqual(terminal.frame_type, FrameType.END)
            output = job.future.result(timeout=0)
            self.assertEqual(output.finish_reason, "stop")
            self.assertEqual(output.token_ids, (10, 99))
        finally:
            root.close()
            child.close()

    def test_cancelling_inflight_macro_wave_rolls_back_logical_branches(self) -> None:
        engine = _root_batch_test_engine(prefill_chunk_tokens=0)
        runner = _FakeRootBatchRunner()
        runner.active = {1: 5}
        proposal = linear_draft_to_macro_wave(
            (20,),
            request_id=1,
            ordinal=1,
            base_prefix_tokens=(1, 2, 3, 10),
            parent_kv_version=KVVersion(3),
        )
        job = _GenerationJob(
            GenerationInput(801, torch.tensor([[1, 2, 3]]), 8),
            None,
            wire_id=1,
            step=1,
            verify_proposal=proposal,
            verify_base_tokens=3,
        )
        job.cancel_requested.set()
        engine._jobs_by_client = {job.request.client_id: job}
        active = {1: job}
        root, child = socket.socketpair()
        try:
            wave = engine._handle_return_value(
                (
                    Frame(
                        FrameType.VERIFY_RESULT,
                        0,
                        1,
                        1,
                        2,
                        0,
                        verify_result_payload((20, 99)),
                    ),
                    time.perf_counter(),
                ),
                active,
                runner,
                root,
            )

            self.assertIsNone(wave)
            self.assertEqual(active, {})
            self.assertTrue(job.future.cancelled())
            self.assertIsNone(job.verify_proposal)
            self.assertEqual(job.verify_base_tokens, 0)
            self.assertEqual(proposal.tree.state, MacroWaveState.ROLLED_BACK)
            cancel = recv_frame(child)
            self.assertEqual(cancel.frame_type, FrameType.CANCEL)
        finally:
            root.close()
            child.close()


@unittest.skipUnless(
    os.environ.get("RUN_DISTRIBUTED_MODEL_TESTS") == "1",
    "set RUN_DISTRIBUTED_MODEL_TESTS=1 to run the continuous scheduler integration",
)
class ContinuousSchedulerIntegrationTests(unittest.TestCase):
    MODEL_NAME = os.environ.get(
        "DISTRIBUTED_TEST_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct"
    )

    def test_root_and_child_reconstruct_physical_batches_for_equal_chats(self) -> None:
        tokenizer = load_tokenizer(self.MODEL_NAME)
        input_ids = tokenizer(
            "The capital of France is", return_tensors="pt"
        ).input_ids
        expected = reference_generate(self.MODEL_NAME, input_ids, 4, 2)[0]
        engine = DistributedPipelineEngine(
            PipelineEngineConfig(
                model_name=self.MODEL_NAME,
                boundaries=(0, 15, 30),
                codec=TensorCodec.FP32,
                threads_per_stage=2,
                max_active_sequences=2,
                max_pending_requests=8,
                prefill_chunk_tokens=2,
                speculative_max_draft_tokens=0,
                root_batch_window_ms=10,
            )
        )
        try:
            outputs = engine.generate(
                [
                    GenerationInput(701, input_ids.clone(), 4),
                    GenerationInput(702, input_ids.clone(), 4),
                ]
            )
            self.assertEqual([list(output.token_ids) for output in outputs], [expected, expected])
            stats = engine.root_batch_stats
            self.assertGreaterEqual(int(stats["physical_batch_calls"]), 2)
            self.assertGreaterEqual(int(stats["physical_batch_items"]), 4)
            self.assertEqual(stats["max_physical_batch_size"], 2)
            self.assertEqual(stats["sequential_items"], 0)
            route_stats = engine.speculation_stats
            self.assertGreaterEqual(int(route_stats["route_probe_count"]), 1)
            self.assertIsNotNone(route_stats["route_rtt_ms"])
            self.assertGreaterEqual(float(route_stats["route_rtt_ms"]), 0.0)
            self.assertTrue(engine.healthy)
        finally:
            engine.close()
        child_metrics = [
            metric
            for metric in engine.stage_metrics
            if "request_id" in metric and "physical_batch_calls" in metric
        ]
        self.assertEqual(len(child_metrics), 2)
        self.assertTrue(
            all(int(metric["physical_batch_calls"]) > 0 for metric in child_metrics)
        )

    def test_new_chat_enters_after_first_token_without_waiting_for_long_chat(self) -> None:
        tokenizer = load_tokenizer(self.MODEL_NAME)
        long_ids = tokenizer(
            "Write a numbered list of computer facts:", return_tensors="pt"
        ).input_ids
        short_ids = tokenizer("The capital of France is", return_tensors="pt").input_ids
        expected_long = reference_generate(self.MODEL_NAME, long_ids, 12, 2)[0]
        expected_short = reference_generate(self.MODEL_NAME, short_ids, 4, 2)[0]

        engine_config = PipelineEngineConfig(
                model_name=self.MODEL_NAME,
                boundaries=(0, 15, 30),
                codec=TensorCodec.FP32,
                threads_per_stage=2,
                max_active_sequences=2,
                max_pending_requests=8,
                prefill_chunk_tokens=2,
                speculative_max_draft_tokens=3,
            )
        engine = DistributedPipelineEngine(
            engine_config,
            draft_provider=_ReferenceDraftProvider(
                (
                    (tuple(int(token) for token in long_ids.reshape(-1)), tuple(expected_long)),
                    (tuple(int(token) for token in short_ids.reshape(-1)), tuple(expected_short)),
                )
            ),
            speculation_controller=AdaptiveSpeculationController(
                AdaptiveSpeculationConfig(
                    max_draft_tokens=3,
                    candidate_sizes=(3,),
                    min_token_history=0,
                    min_classic_observations=1,
                    min_verify_observations=2,
                )
            ),
        )
        short_future = []
        submitted = threading.Event()
        completion_times: dict[str, float] = {}

        def on_long(_client_id: int, _token_id: int, step: int, _arrived: float) -> None:
            if step == 0 and not submitted.is_set():
                short_future.extend(
                    engine.submit(
                        [GenerationInput(202, short_ids, 4)],
                    )
                )
                submitted.set()

        try:
            long_future = engine.submit(
                [GenerationInput(101, long_ids, 12)], on_long
            )[0]
            long_future.add_done_callback(
                lambda _: completion_times.__setitem__("long", time.perf_counter())
            )
            self.assertTrue(submitted.wait(20), "second chat was not admitted from token callback")
            short_future[0].add_done_callback(
                lambda _: completion_times.__setitem__("short", time.perf_counter())
            )
            short_output = short_future[0].result(timeout=30)
            long_output = long_future.result(timeout=30)
            self.assertEqual(list(short_output.token_ids), expected_short)
            self.assertEqual(list(long_output.token_ids), expected_long)
            self.assertLess(
                completion_times["short"],
                completion_times["long"],
                "short chat should finish while the older long chat is still active",
            )
            self.assertTrue(engine.healthy)
            self.assertGreater(
                int(engine.speculation_stats["verification_observations"]),
                0,
            )
            speculation = engine.speculation_stats
            self.assertGreater(
                int(speculation["selected_candidate_sizes"].get("3", 0)),
                0,
            )
            self.assertGreater(
                int(speculation["proposed_tokens"]),
                int(speculation["accepted_tokens"]),
                "the physical route must exercise a mid-wave mismatch and TRUNCATE",
            )
        finally:
            engine.close()


def _root_batch_test_engine(*, prefill_chunk_tokens: int) -> DistributedPipelineEngine:
    engine = DistributedPipelineEngine.__new__(DistributedPipelineEngine)
    engine.config = SimpleNamespace(
        max_active_sequences=8,
        prefill_chunk_tokens=prefill_chunk_tokens,
        speculative_max_draft_tokens=0,
        speculation_probe=False,
        root_batch_window_ms=0.5,
        codec=TensorCodec.FP32,
    )
    engine._deferred_batches = deque()
    engine._request_counter = 0
    engine._callback_lock = threading.Lock()
    engine._callback_routes = {}
    engine._state_lock = threading.Lock()
    engine._jobs_by_client = {}
    engine.draft_provider = None
    engine.speculation_controller = None
    engine._root_ready_items = 0
    engine._root_model_forward_calls = 0
    engine._root_physical_batch_calls = 0
    engine._root_physical_batch_items = 0
    engine._root_sequential_items = 0
    engine._root_max_physical_batch_size = 1
    return engine


class _FakeRootBatchRunner:
    MAX_PHYSICAL_BATCH_SIZE = 8

    def __init__(self, *, batchable: bool = True) -> None:
        self.batchable = batchable
        self.active: dict[int, int] = {}
        self.batch_calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        self.sequential_calls: list[tuple[int, int]] = []
        self.truncations: list[tuple[int, int]] = []

    def begin(self, request_id: int) -> None:
        if request_id in self.active:
            raise ValueError("duplicate BEGIN")
        self.active[request_id] = 0

    def end(self, request_id: int) -> None:
        self.active.pop(request_id, None)

    def truncate(self, request_id: int, token_count: int) -> None:
        if not 0 <= token_count <= self.active[request_id]:
            raise ValueError("invalid truncate")
        self.active[request_id] = token_count
        self.truncations.append((request_id, token_count))

    def sequence_length(self, request_id: int) -> int:
        return self.active[request_id]

    def physical_batch_key(
        self,
        request_id: int,
        *,
        token_count: int,
        token_mode: str,
    ) -> tuple[int, int, str] | None:
        if not self.batchable:
            return None
        return self.active[request_id], token_count, token_mode

    def forward_ids(self, request_id: int, input_ids: torch.Tensor) -> torch.Tensor:
        token_count = int(input_ids.shape[1])
        self.sequential_calls.append((request_id, token_count))
        self.active[request_id] += token_count
        return input_ids.to(torch.float32).unsqueeze(-1).repeat(1, 1, 4)

    def forward_ids_batch(
        self,
        request_ids: tuple[int, ...],
        input_ids: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        token_counts = tuple(int(value.shape[1]) for value in input_ids)
        self.batch_calls.append((request_ids, token_counts))
        outputs = []
        for request_id, value, token_count in zip(
            request_ids, input_ids, token_counts, strict=True
        ):
            self.active[request_id] += token_count
            outputs.append(value.to(torch.float32).unsqueeze(-1).repeat(1, 1, 4))
        return tuple(outputs)


class _RecordingSpeculationController(AdaptiveSpeculationController):
    def __init__(self) -> None:
        super().__init__(AdaptiveSpeculationConfig(max_draft_tokens=8))
        self.verifications: list[tuple[int, int]] = []

    def record_verification(
        self,
        *,
        proposed_tokens: int,
        accepted_tokens: int,
        latency_seconds: float,
        transferred_bytes: int,
    ) -> None:
        if latency_seconds <= 0 or transferred_bytes <= 0:
            raise ValueError("invalid verification observation")
        self.verifications.append((proposed_tokens, accepted_tokens))


class _FixedDraftProvider:
    strategy = "fixed-engine-test"

    def __init__(self, tokens: tuple[int, ...]) -> None:
        self.tokens = tokens
        self.max_draft_tokens = len(tokens)

    def draft(
        self,
        token_history: tuple[int, ...],
        max_tokens: int | None = None,
    ) -> tuple[int, ...]:
        del token_history
        limit = self.max_draft_tokens if max_tokens is None else max_tokens
        return self.tokens[:limit]


class _ReferenceDraftProvider:
    strategy = "test-reference"
    max_draft_tokens = 3

    def __init__(
        self,
        cases: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...],
    ) -> None:
        self.cases = cases

    def draft(
        self,
        token_history: tuple[int, ...],
        max_tokens: int | None = None,
    ) -> tuple[int, ...]:
        if max_tokens == 0:
            return ()
        history = tuple(int(token) for token in token_history)
        for prompt, reference in self.cases:
            if history[: len(prompt)] != prompt:
                continue
            generated = len(history) - len(prompt)
            if not 0 <= generated < len(reference):
                return ()
            limit = self.max_draft_tokens if max_tokens is None else max_tokens
            candidates = list(reference[generated : generated + limit])
            # Later three-position probes intentionally miss at their second
            # draft. The physical integration therefore covers a committed
            # prefix, target correction and TRUNCATE on root and child before
            # continuing to the exact greedy sequence.
            if generated >= 4 and len(candidates) >= 2:
                candidates[1] = (candidates[1] + 1) % 49_152
            return tuple(candidates)
        return ()


if __name__ == "__main__":
    unittest.main()
