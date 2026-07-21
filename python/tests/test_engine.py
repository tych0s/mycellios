from __future__ import annotations

from collections import deque
from dataclasses import replace
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
    PipelineShutdownError,
    _GenerationJob,
    _CommittedPhysicalTreeWave,
    _InflightWave,
    _PreparedPhysicalTreeWave,
    _PreparedRootWave,
    _prefill_frame_byte_reservation,
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
    TreePrepareRejection,
    TreePrepareStatus,
    decode_branch_request_id,
    decode_tree_prepare,
    decode_tensor,
    recv_frame,
    encode_tree_prepare_quote,
    token_payload,
    verify_result_payload,
)
from distributed_runtime.physical_tree import PhysicalTreeCoordinator
from distributed_runtime.speculation import (
    AdaptiveSpeculationConfig,
    AdaptiveSpeculationController,
    NgramTreeDraftProvider,
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
        self.assertEqual(configured.prefill_inflight_chunks, 1)
        self.assertEqual(configured.prefill_inflight_bytes, 0)
        self.assertEqual(configured.speculative_max_draft_tokens, 8)
        self.assertEqual(configured.sealed_wave_token_limit, 9)
        self.assertEqual(configured.prefill_token_limit, 32)
        self.assertEqual(configured.root_batch_window_ms, 0.5)
        self.assertEqual(configured.route_probe_interval_seconds, 5.0)
        self.assertEqual(configured.route_probe_timeout_seconds, 10.0)
        for invalid in (0, 65, -1, True):
            with self.subTest(prefill_inflight_chunks=invalid), self.assertRaises(
                ValueError
            ):
                PipelineEngineConfig(
                    **common,
                    prefill_inflight_chunks=invalid,
                )
        for invalid in (-1, (1 << 30) + 1, True):
            with self.subTest(prefill_inflight_bytes=invalid), self.assertRaises(
                ValueError
            ):
                PipelineEngineConfig(
                    **common,
                    prefill_inflight_bytes=invalid,
                )
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
        with self.assertRaisesRegex(ValueError, "three sealed speculative limits"):
            DistributedPipelineEngine(
                PipelineEngineConfig(
                    **common,
                    speculative_max_draft_tokens=2,
                ),
                tree_draft_provider=NgramTreeDraftProvider(
                    max_draft_tokens=2,
                    max_branches=2,
                ),
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

    def test_close_orders_protocol_shutdown_before_child_wait_and_hard_socket_close(self) -> None:
        events: list[str] = []
        process = _LifecycleProcess(events, exit_code=0)
        engine = _close_lifecycle_test_engine(events, (process,))

        engine.close()

        self.assertLess(events.index("shutdown-sent"), events.index("half-close-write"))
        self.assertLess(events.index("half-close-write"), events.index("child-join"))
        self.assertLess(events.index("child-join"), events.index("socket-close"))
        self.assertNotIn("hard-close", events)
        self.assertEqual(process.exitcode, 0)
        self.assertTrue(engine.shutdown_status["clean"])
        self.assertFalse(engine.shutdown_status["hard_fallback_used"])

    def test_close_fails_and_exposes_nonzero_child_exit_code(self) -> None:
        events: list[str] = []
        process = _LifecycleProcess(events, exit_code=7)
        engine = _close_lifecycle_test_engine(events, (process,))

        with self.assertRaises(PipelineShutdownError) as raised:
            engine.close()

        report = raised.exception.report
        self.assertFalse(report["clean"])
        self.assertFalse(report["hard_fallback_used"])
        self.assertEqual(report["child_processes"][0]["exit_code"], 7)
        self.assertFalse(report["child_processes"][0]["alive"])
        with self.assertRaises(PipelineShutdownError) as repeated:
            engine.close()
        self.assertIs(repeated.exception, engine._close_error)

    def test_close_uses_bounded_hard_fallback_for_stubborn_child(self) -> None:
        events: list[str] = []
        process = _LifecycleProcess(events, exit_code=0, stubborn=True)
        engine = _close_lifecycle_test_engine(events, (process,))

        with self.assertRaises(PipelineShutdownError) as raised:
            engine.close()

        self.assertIn("hard-close", events)
        self.assertIn("child-terminate", events)
        self.assertIn("socket-close", events)
        self.assertTrue(raised.exception.report["hard_fallback_used"])
        self.assertEqual(raised.exception.report["child_processes"][0]["exit_code"], -15)
        self.assertFalse(raised.exception.report["child_processes"][0]["alive"])

    def test_context_manager_preserves_original_exception_over_cleanup_failure(self) -> None:
        engine = DistributedPipelineEngine.__new__(DistributedPipelineEngine)
        engine.close = mock.Mock(side_effect=RuntimeError("secondary cleanup"))
        original = ValueError("original body failure")

        with self.assertRaises(ValueError) as raised:
            with engine:
                raise original

        self.assertIs(raised.exception, original)
        self.assertTrue(
            any("secondary cleanup" in note for note in original.__notes__)
        )

    def test_constructor_preserves_start_error_over_cleanup_failure(self) -> None:
        config = PipelineEngineConfig(model_name="fake", boundaries=(0, 2, 4))
        artifact = SimpleNamespace(
            identity="sha256:" + "1" * 64,
            canonical_source="content-addressed://fake",
            canonical_revision=None,
            snapshot_identity=1,
        )
        model_config = SimpleNamespace(
            num_hidden_layers=4,
            hidden_size=8,
            max_position_embeddings=128,
        )
        with (
            mock.patch(
                "distributed_runtime.engine.AutoConfig.from_pretrained",
                return_value=model_config,
            ),
            mock.patch(
                "distributed_runtime.engine.resolve_model_snapshot",
                return_value="fake-snapshot",
            ),
            mock.patch(
                "distributed_runtime.engine.model_artifact_reference",
                return_value=artifact,
            ),
            mock.patch.object(
                DistributedPipelineEngine,
                "_start",
                side_effect=ValueError("original start failure"),
            ),
            mock.patch.object(
                DistributedPipelineEngine,
                "close",
                side_effect=RuntimeError("secondary cleanup"),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "original start failure") as raised:
                DistributedPipelineEngine(config)

        self.assertTrue(
            any("secondary cleanup" in note for note in raised.exception.__notes__)
        )

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
        for job in jobs:
            _seed_inflight_wave(
                engine,
                job,
                FrameType.PREFILL,
                prefill_end=2,
            )
        try:
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
                self.assertIsNone(wave)
            self.assertEqual(
                engine._dispatch_prefill_credit_round(
                    active, runner, root, LinkEmulator()
                ),
                2,
            )
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
                    step=0,
                    prefill_end=2,
                    reserved_bytes=_prefill_frame_byte_reservation(
                        TensorCodec.FP32, 2, 4
                    ),
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

    def test_prefill_credit_window_fills_one_chunk_per_request_and_round(self) -> None:
        engine = _root_batch_test_engine(
            prefill_chunk_tokens=2,
            prefill_inflight_chunks=3,
        )
        runner = _FakeRootBatchRunner()
        root, child = socket.socketpair()
        active: dict[int, _GenerationJob] = {}
        jobs = [
            _GenerationJob(
                GenerationInput(client_id, torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]), 1),
                None,
            )
            for client_id in (101, 202)
        ]
        try:
            engine._admit_batch(jobs, active, runner, root, LinkEmulator())
            begins = (recv_frame(child), recv_frame(child))
            first_round = (recv_frame(child), recv_frame(child))
            self.assertEqual(
                [frame.frame_type for frame in begins],
                [FrameType.BEGIN, FrameType.BEGIN],
            )
            self.assertEqual(
                engine._dispatch_prefill_credit_round(
                    active, runner, root, LinkEmulator()
                ),
                2,
            )
            second_round = (recv_frame(child), recv_frame(child))
            self.assertEqual(
                engine._dispatch_prefill_credit_round(
                    active, runner, root, LinkEmulator()
                ),
                2,
            )
            third_round = (recv_frame(child), recv_frame(child))
            self.assertEqual(
                engine._dispatch_prefill_credit_round(
                    active, runner, root, LinkEmulator()
                ),
                0,
            )
            frames = (*first_round, *second_round, *third_round)
            by_request = {
                request_id: [frame for frame in frames if frame.request_id == request_id]
                for request_id in (1, 2)
            }
            self.assertEqual(
                [[frame.step for frame in by_request[request_id]] for request_id in (1, 2)],
                [[0, 1, 2], [0, 1, 2]],
            )
            self.assertTrue(
                all(
                    frame.frame_type == FrameType.PREFILL
                    for frame in frames
                )
            )
            self.assertEqual([job.prefill_offset for job in jobs], [6, 6])
            self.assertEqual([len(job.inflight_waves) for job in jobs], [3, 3])
            stats = engine.prefill_window_stats
            self.assertEqual(stats["configured_chunks_per_request"], 3)
            self.assertEqual(stats["global_chunk_ceiling"], 24)
            self.assertEqual(stats["high_water_chunks"], 6)
            self.assertEqual(stats["max_request_chunks"], 3)
            self.assertEqual(stats["dispatched_chunks"], 6)
        finally:
            root.close()
            child.close()

    def test_prefill_credit_window_rejects_out_of_order_ack_without_consuming_it(self) -> None:
        engine = _root_batch_test_engine(
            prefill_chunk_tokens=2,
            prefill_inflight_chunks=3,
        )
        runner = _FakeRootBatchRunner()
        root, child = socket.socketpair()
        active: dict[int, _GenerationJob] = {}
        job = _GenerationJob(
            GenerationInput(101, torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]), 1),
            None,
        )
        try:
            engine._admit_batch([job], active, runner, root, LinkEmulator())
            recv_frame(child)
            recv_frame(child)
            for _ in range(2):
                self.assertEqual(
                    engine._dispatch_prefill_credit_round(
                        active, runner, root, LinkEmulator()
                    ),
                    1,
                )
                recv_frame(child)
            with self.assertRaisesRegex(RuntimeError, "expected FIFO step 0"):
                engine._handle_return_value(
                    (
                        Frame(FrameType.PREFILL_ACK, 0, 1, 1, 0, 0, b""),
                        time.perf_counter(),
                    ),
                    active,
                    runner,
                    root,
                )
            self.assertEqual(
                [flight.step for flight in job.inflight_waves],
                [0, 1, 2],
            )
            self.assertEqual(job.step, 0)
        finally:
            root.close()
            child.close()

    def test_prefill_byte_credit_is_per_request_and_strict(self) -> None:
        reservation = _prefill_frame_byte_reservation(TensorCodec.FP32, 2, 4)
        engine = _root_batch_test_engine(
            prefill_chunk_tokens=2,
            prefill_inflight_chunks=4,
            prefill_inflight_bytes=reservation,
        )
        runner = _FakeRootBatchRunner()
        root, child = socket.socketpair()
        active: dict[int, _GenerationJob] = {}
        job = _GenerationJob(
            GenerationInput(101, torch.tensor([[1, 2, 3, 4, 5, 6]]), 1),
            None,
        )
        try:
            engine._admit_batch([job], active, runner, root, LinkEmulator())
            recv_frame(child)
            first = recv_frame(child)
            self.assertEqual(first.step, 0)
            self.assertEqual(
                engine._dispatch_prefill_credit_round(
                    active, runner, root, LinkEmulator()
                ),
                0,
            )
            self.assertIsNone(
                engine._handle_return_value(
                    (
                        Frame(FrameType.PREFILL_ACK, 0, 1, 0, 0, 0, b""),
                        time.perf_counter(),
                    ),
                    active,
                    runner,
                    root,
                )
            )
            self.assertEqual(
                engine._dispatch_prefill_credit_round(
                    active, runner, root, LinkEmulator()
                ),
                1,
            )
            self.assertEqual(recv_frame(child).step, 1)
            stats = engine.prefill_window_stats
            self.assertEqual(stats["configured_bytes_per_request"], reservation)
            self.assertEqual(stats["max_request_reserved_bytes"], reservation)
            self.assertLessEqual(stats["max_request_reserved_bytes"], reservation)
        finally:
            root.close()
            child.close()

    def test_prefill_timeout_uses_oldest_outstanding_chunk(self) -> None:
        engine = _root_batch_test_engine(
            prefill_chunk_tokens=2,
            prefill_inflight_chunks=2,
        )
        engine.config.socket_timeout_seconds = 1.0
        job = _GenerationJob(
            GenerationInput(101, torch.tensor([[1, 2, 3, 4, 5, 6]]), 1),
            None,
            wire_id=1,
        )
        old = time.monotonic() - 2.0
        job.inflight_waves.extend(
            (
                _InflightWave(0, FrameType.PREFILL, 2, time.perf_counter(), old, 64, 64),
                _InflightWave(
                    1,
                    FrameType.PREFILL,
                    4,
                    time.perf_counter(),
                    time.monotonic(),
                    64,
                    64,
                ),
            )
        )
        with self.assertRaisesRegex(TimeoutError, "request 101"):
            engine._check_pipeline_timeouts({1: job})

    def test_cancelled_window_drains_all_fifo_tombstones_before_retirement(self) -> None:
        engine = _root_batch_test_engine(
            prefill_chunk_tokens=2,
            prefill_inflight_chunks=3,
        )
        runner = _FakeRootBatchRunner()
        root, child = socket.socketpair()
        active: dict[int, _GenerationJob] = {}
        callbacks: list[int] = []
        job = _GenerationJob(
            GenerationInput(101, torch.tensor([[1, 2, 3, 4, 5, 6]]), 1),
            lambda _client, token, _index, _arrived: callbacks.append(token),
        )
        try:
            engine._admit_batch([job], active, runner, root, LinkEmulator())
            recv_frame(child)
            data_frames = [recv_frame(child)]
            for _ in range(2):
                self.assertEqual(
                    engine._dispatch_prefill_credit_round(
                        active, runner, root, LinkEmulator()
                    ),
                    1,
                )
                data_frames.append(recv_frame(child))
            self.assertEqual(
                [frame.frame_type for frame in data_frames],
                [FrameType.PREFILL, FrameType.PREFILL, FrameType.ACTIVATION],
            )
            job.cancel_requested.set()
            engine._send_requested_cancellations(active, runner, root)
            self.assertEqual(recv_frame(child).frame_type, FrameType.CANCEL)
            for step in (0, 1):
                self.assertIsNone(
                    engine._handle_return_value(
                        (
                            Frame(FrameType.PREFILL_ACK, 0, 1, step, 0, 0, b""),
                            time.perf_counter(),
                        ),
                        active,
                        runner,
                        root,
                    )
                )
                self.assertIn(1, active)
            self.assertIsNone(
                engine._handle_return_value(
                    (
                        Frame(
                            FrameType.TOKEN,
                            0,
                            1,
                            2,
                            0,
                            0,
                            token_payload(99),
                        ),
                        time.perf_counter(),
                    ),
                    active,
                    runner,
                    root,
                )
            )
            self.assertEqual(active, {})
            self.assertTrue(job.future.cancelled())
            self.assertEqual(callbacks, [])
            self.assertNotIn(1, runner.active)
            self.assertEqual(engine.prefill_window_stats["current_chunks"], 0)
            self.assertEqual(engine.prefill_window_stats["current_bytes"], 0)
        finally:
            root.close()
            child.close()

    def test_prefill_window_changes_timing_not_exact_wire_sequence_or_output(self) -> None:
        def execute(window: int) -> tuple[list[tuple[int, FrameType, list]], tuple[int, ...]]:
            engine = _root_batch_test_engine(
                prefill_chunk_tokens=2,
                prefill_inflight_chunks=window,
            )
            runner = _FakeRootBatchRunner()
            root, child = socket.socketpair()
            active: dict[int, _GenerationJob] = {}
            job = _GenerationJob(
                GenerationInput(101, torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]), 1),
                None,
            )
            trace: list[tuple[int, FrameType, list]] = []
            try:
                engine._admit_batch([job], active, runner, root, LinkEmulator())
                self.assertEqual(recv_frame(child).frame_type, FrameType.BEGIN)
                first = recv_frame(child)
                trace.append((first.step, first.frame_type, decode_tensor(first).tolist()))
                while active:
                    while engine._dispatch_prefill_credit_round(
                        active, runner, root, LinkEmulator()
                    ):
                        sent = recv_frame(child)
                        trace.append((sent.step, sent.frame_type, decode_tensor(sent).tolist()))
                    flight = job.inflight_waves[0]
                    if flight.frame_type == FrameType.PREFILL:
                        returned = Frame(
                            FrameType.PREFILL_ACK,
                            0,
                            1,
                            flight.step,
                            0,
                            0,
                            b"",
                        )
                    else:
                        returned = Frame(
                            FrameType.TOKEN,
                            0,
                            1,
                            flight.step,
                            0,
                            0,
                            token_payload(99),
                        )
                    engine._handle_return_value(
                        (returned, time.perf_counter()),
                        active,
                        runner,
                        root,
                    )
                self.assertEqual(recv_frame(child).frame_type, FrameType.END)
                return trace, job.future.result(timeout=0).token_ids
            finally:
                root.close()
                child.close()

        stop_and_wait = execute(1)
        pipelined = execute(3)
        self.assertEqual(pipelined, stop_and_wait)

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
        for job in jobs:
            _seed_inflight_wave(engine, job, FrameType.VERIFY)
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
        _seed_inflight_wave(engine, job, FrameType.VERIFY)
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
        _seed_inflight_wave(engine, job, FrameType.VERIFY)
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
        _seed_inflight_wave(engine, job, FrameType.VERIFY)
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

    def test_physical_tree_default_is_sequential_and_commits_before_publish(self) -> None:
        engine = _root_batch_test_engine(
            prefill_chunk_tokens=0,
            speculative_max_draft_tokens=2,
            max_speculative_branches=8,
            max_speculative_branch_tokens=128,
            max_speculative_kv_bytes=4096,
        )
        engine.tree_draft_provider = _FixedTreeDraftProvider(
            ((20, 21), (20, 22))
        )
        engine._request_counter = 7
        runner = _FakeRootBatchRunner(batchable=True)
        runner.active = {7: 3}
        streamed: list[tuple[int, int]] = []
        job = _GenerationJob(
            GenerationInput(
                901,
                torch.tensor([[1, 2, 3]]),
                4,
                frozenset({99}),
            ),
            lambda client, token, _index, _arrived: streamed.append((client, token)),
            wire_id=7,
            step=1,
            started_at=time.perf_counter() - 1,
            token_ids=[10],
            arrivals=[time.perf_counter() - 0.5],
            prefill_offset=3,
            prefill_acked_offset=3,
        )
        engine._jobs_by_client = {901: job}
        engine._callback_routes = {7: job}
        active = {7: job}
        root, child = socket.socketpair()
        try:
            prepared = engine._prepare_decode_wave(job, runner, active_sequences=1)
            self.assertIsInstance(prepared, _PreparedPhysicalTreeWave)
            _dispatch_and_ready_tree_quote(
                engine, prepared, active, runner, root, child
            )
            outbound = tuple(recv_frame(child) for _ in range(4))
            self.assertEqual(
                [frame.frame_type for frame in outbound],
                [FrameType.FORK, FrameType.FORK, FrameType.VERIFY, FrameType.VERIFY],
            )
            self.assertEqual([frame.request_id for frame in outbound], [8, 9, 8, 9])
            self.assertEqual(
                [decode_branch_request_id(frame) for frame in outbound[:2]],
                [7, 7],
            )
            # Shape compatibility alone is not a batch-exact certificate.
            self.assertEqual(runner.batch_calls, [])
            self.assertEqual(runner.sequential_calls, [(8, 3), (9, 3)])
            self.assertEqual(set(engine._callback_routes), {7})

            self.assertIsNone(
                engine._handle_return_value(
                    (
                        Frame(
                            FrameType.VERIFY_RESULT,
                            0,
                            9,
                            1,
                            3,
                            0,
                            verify_result_payload((20, 99, 55)),
                        ),
                        time.perf_counter(),
                    ),
                    active,
                    runner,
                    root,
                )
            )
            self.assertEqual(streamed, [])
            self.assertIsNone(
                engine._handle_return_value(
                    (
                        Frame(
                            FrameType.VERIFY_RESULT,
                            0,
                            8,
                            1,
                            3,
                            0,
                            verify_result_payload((20, 99, 77)),
                        ),
                        time.perf_counter(),
                    ),
                    active,
                    runner,
                    root,
                )
            )
            cleanup = tuple(recv_frame(child) for _ in range(4))
            self.assertEqual(
                [frame.frame_type for frame in cleanup],
                [
                    FrameType.END,
                    FrameType.PROMOTE,
                    FrameType.TRUNCATE,
                    FrameType.END,
                ],
            )
            self.assertEqual(cleanup[0].request_id, 9)
            self.assertEqual(cleanup[1].request_id, 7)
            self.assertEqual(decode_branch_request_id(cleanup[1]), 8)
            self.assertEqual(cleanup[2].token_count, 5)
            self.assertEqual(streamed, [(901, 20), (901, 99)])
            self.assertEqual(job.step, 2)
            self.assertEqual(job.next_step, 2)
            self.assertEqual(job.future.result(timeout=0).token_ids, (10, 20, 99))
            self.assertEqual(engine._leaf_routes, {})
            self.assertEqual(engine._physical_tree_live_children, set())
            self.assertEqual(engine.speculation_stats["physical_tree"]["committed_waves"], 1)
        finally:
            root.close()
            child.close()

    def test_physical_tree_width_one_matches_linear_mismatch_and_truncates(self) -> None:
        engine = _root_batch_test_engine(
            prefill_chunk_tokens=0,
            speculative_max_draft_tokens=2,
            max_speculative_branches=4,
            max_speculative_branch_tokens=128,
            max_speculative_kv_bytes=2048,
        )
        engine.tree_draft_provider = _FixedTreeDraftProvider(((20,),))
        engine._request_counter = 7
        runner = _FakeRootBatchRunner()
        runner.active = {7: 3}
        job = _GenerationJob(
            GenerationInput(
                902,
                torch.tensor([[1, 2, 3]]),
                3,
                frozenset({99}),
            ),
            None,
            wire_id=7,
            step=4,
            started_at=time.perf_counter() - 1,
            token_ids=[10],
            arrivals=[time.perf_counter() - 0.5],
            prefill_offset=3,
            prefill_acked_offset=3,
        )
        engine._jobs_by_client = {902: job}
        engine._callback_routes = {7: job}
        active = {7: job}
        root, child = socket.socketpair()
        try:
            prepared = engine._prepare_decode_wave(job, runner, active_sequences=1)
            self.assertIsInstance(prepared, _PreparedPhysicalTreeWave)
            _dispatch_and_ready_tree_quote(
                engine, prepared, active, runner, root, child
            )
            fork, verify = recv_frame(child), recv_frame(child)
            self.assertEqual((fork.frame_type, verify.frame_type), (FrameType.FORK, FrameType.VERIFY))
            self.assertIsNone(
                engine._handle_return_value(
                    (
                        Frame(
                            FrameType.VERIFY_RESULT,
                            0,
                            8,
                            4,
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
            )
            controls = tuple(recv_frame(child) for _ in range(3))
            self.assertEqual(
                [frame.frame_type for frame in controls],
                [FrameType.PROMOTE, FrameType.TRUNCATE, FrameType.END],
            )
            self.assertEqual(controls[1].token_count, 4)
            self.assertEqual(runner.truncations, [(7, 4)])
            self.assertEqual(job.future.result(timeout=0).token_ids, (10, 99))
            self.assertEqual(job.future.result(timeout=0).finish_reason, "stop")
            self.assertEqual(job.step, 5)
            self.assertEqual(job.next_step, 5)
        finally:
            root.close()
            child.close()

    def test_physical_tree_batches_only_with_both_sealed_certificates(self) -> None:
        engine, runner, job, active = _prepared_tree_test_case(
            ((20,), (21,)), max_new_tokens=4
        )
        runner.executor_manifest.features = (
            "exact-tree-verify-batching",
            "bounded-tree-verify-workspace",
        )
        root, child = socket.socketpair()
        try:
            prepared = engine._prepare_decode_wave(job, runner, active_sequences=1)
            assert isinstance(prepared, _PreparedPhysicalTreeWave)
            _dispatch_and_ready_tree_quote(
                engine, prepared, active, runner, root, child
            )
            for _ in range(4):
                recv_frame(child)
            self.assertEqual(runner.batch_calls, [((8, 9), (2, 2))])
            self.assertEqual(runner.sequential_calls, [])
            self.assertEqual(engine._physical_tree_batch_calls, 1)
            self.assertEqual(engine._physical_tree_batch_items, 2)
        finally:
            engine._retire_job(job, runner, exception=RuntimeError("test cleanup"))
            root.close()
            child.close()

    def test_physical_tree_cancel_drains_virtual_tombstones_before_future(self) -> None:
        engine, runner, job, active = _prepared_tree_test_case(
            ((20,), (21,)), max_new_tokens=4
        )
        root, child = socket.socketpair()
        try:
            prepared = engine._prepare_decode_wave(job, runner, active_sequences=1)
            assert isinstance(prepared, _PreparedPhysicalTreeWave)
            _dispatch_and_ready_tree_quote(
                engine, prepared, active, runner, root, child
            )
            for _ in range(4):
                recv_frame(child)
            job.cancel_requested.set()
            engine._send_requested_cancellations(active, runner, root)
            cancels = tuple(recv_frame(child) for _ in range(3))
            self.assertEqual([frame.frame_type for frame in cancels], [FrameType.CANCEL] * 3)
            self.assertEqual([frame.request_id for frame in cancels], [8, 9, 7])
            self.assertIn(7, active)
            self.assertFalse(job.future.done())

            for index, request_id in enumerate((8, 9)):
                self.assertIsNone(
                    engine._handle_return_value(
                        (
                            Frame(
                                FrameType.VERIFY_RESULT,
                                0,
                                request_id,
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
                )
                if index == 0:
                    self.assertFalse(job.future.done())
            self.assertEqual(active, {})
            self.assertTrue(job.future.cancelled())
            self.assertEqual(engine._leaf_routes, {})
            self.assertEqual(engine._physical_tree_live_children, set())
        finally:
            root.close()
            child.close()

    def test_physical_tree_cancel_before_first_fork_only_cancels_parent(self) -> None:
        engine, runner, job, active = _prepared_tree_test_case(
            ((20,), (21,)), max_new_tokens=4
        )
        root, child = socket.socketpair()
        try:
            prepared = engine._prepare_decode_wave(job, runner, active_sequences=1)
            assert isinstance(prepared, _PreparedPhysicalTreeWave)
            job.cancel_requested.set()
            engine._dispatch_root_waves([prepared], runner, root, LinkEmulator())
            self.assertEqual(runner.forks, [])
            self.assertEqual(engine._physical_tree.tree_by_parent, {})
            engine._send_requested_cancellations(active, runner, root)
            cancel = recv_frame(child)
            self.assertEqual(cancel.frame_type, FrameType.CANCEL)
            self.assertEqual(cancel.request_id, 7)
            self.assertEqual(active, {})
            self.assertTrue(job.future.cancelled())
        finally:
            root.close()
            child.close()

    def test_tree_quote_orders_prepare_commit_before_any_kv_mutation(self) -> None:
        engine, runner, job, active = _prepared_tree_test_case(
            ((30, 31), (20,)), max_new_tokens=5
        )
        root, child = socket.socketpair()
        try:
            prepared = engine._prepare_decode_wave(job, runner, active_sequences=1)
            self.assertIsInstance(prepared, _PreparedPhysicalTreeWave)
            assert isinstance(prepared, _PreparedPhysicalTreeWave)
            root_state = dict(runner.active)

            engine._dispatch_root_waves([prepared], runner, root, LinkEmulator())
            prepare = recv_frame(child)
            self.assertEqual(prepare.frame_type, FrameType.TREE_PREPARE)
            quote = decode_tree_prepare(prepare)
            self.assertEqual(quote.path_lengths, (1, 2))
            self.assertEqual(runner.active, root_state)
            self.assertEqual(runner.forks, [])
            self.assertEqual(engine._physical_tree.tree_by_parent, {})
            self.assertEqual(prepared.proposal.tree.state, MacroWaveState.OPEN)
            child.settimeout(0.02)
            with self.assertRaises(socket.timeout):
                recv_frame(child)
            child.settimeout(None)

            ready = replace(quote, stage_count=engine.stages - 1)
            awaiting_commit = engine._handle_return_value(
                (
                    Frame(
                        FrameType.TREE_PREPARE_RESULT,
                        0,
                        prepare.request_id,
                        prepare.step,
                        prepare.token_count,
                        0,
                        encode_tree_prepare_quote(ready),
                    ),
                    time.perf_counter(),
                ),
                active,
                runner,
                root,
            )
            self.assertIsNone(awaiting_commit)
            commit = recv_frame(child)
            self.assertEqual(commit.frame_type, FrameType.TREE_RESERVATION_COMMIT)
            self.assertEqual(runner.active, root_state)
            self.assertEqual(runner.forks, [])
            self.assertEqual(engine._physical_tree.tree_by_parent, {})
            child.settimeout(0.02)
            with self.assertRaises(socket.timeout):
                recv_frame(child)
            child.settimeout(None)

            committed = engine._handle_return_value(
                (
                    Frame(
                        FrameType.TREE_RESERVATION_COMMIT_RESULT,
                        0,
                        commit.request_id,
                        commit.step,
                        engine.stages - 1,
                        0,
                        commit.payload,
                    ),
                    time.perf_counter(),
                ),
                active,
                runner,
                root,
            )
            self.assertIsInstance(committed, _CommittedPhysicalTreeWave)
            assert isinstance(committed, _CommittedPhysicalTreeWave)
            engine._dispatch_root_waves([committed], runner, root, LinkEmulator())
            outbound = tuple(recv_frame(child) for _ in range(4))
            self.assertEqual(
                [frame.frame_type for frame in outbound],
                [FrameType.FORK, FrameType.FORK, FrameType.VERIFY, FrameType.VERIFY],
            )
            self.assertEqual(len(runner.forks), 2)
            self.assertIsNone(engine._pending_tree_reservation)
        finally:
            engine._retire_job(job, runner, exception=RuntimeError("test cleanup"))
            root.close()
            child.close()

    def test_tree_quote_remote_reject_cancels_and_falls_back_same_step(self) -> None:
        engine, runner, job, active = _prepared_tree_test_case(
            ((20,), (21,)), max_new_tokens=4
        )
        root, child = socket.socketpair()
        try:
            prepared = engine._prepare_decode_wave(job, runner, active_sequences=1)
            self.assertIsInstance(prepared, _PreparedPhysicalTreeWave)
            assert isinstance(prepared, _PreparedPhysicalTreeWave)
            root_state = dict(runner.active)
            engine._dispatch_root_waves([prepared], runner, root, LinkEmulator())
            prepare = recv_frame(child)
            quote = decode_tree_prepare(prepare)
            rejected = replace(
                quote,
                status=TreePrepareStatus.REJECT,
                rejection=TreePrepareRejection.KV_BYTES,
                stage_count=engine.stages - 1,
                rejecting_layer_start=1,
                required=5_000,
                limit=4_096,
                total_projected_bytes=5_000,
            )
            fallback = engine._handle_return_value(
                (
                    Frame(
                        FrameType.TREE_PREPARE_RESULT,
                        0,
                        prepare.request_id,
                        prepare.step,
                        prepare.token_count,
                        0,
                        encode_tree_prepare_quote(rejected),
                    ),
                    time.perf_counter(),
                ),
                active,
                runner,
                root,
            )

            self.assertIsInstance(fallback, _PreparedRootWave)
            assert isinstance(fallback, _PreparedRootWave)
            cancel = recv_frame(child)
            self.assertEqual(cancel.frame_type, FrameType.TREE_RESERVATION_CANCEL)
            self.assertEqual(cancel.request_id, job.wire_id)
            self.assertEqual(fallback.step, prepared.step)
            self.assertEqual(fallback.frame_type, FrameType.ACTIVATION)
            self.assertEqual(job.step, prepared.step)
            self.assertEqual(job.next_step, prepared.step)
            self.assertEqual(runner.active, root_state)
            self.assertEqual(runner.forks, [])
            self.assertEqual(engine._physical_tree.tree_by_parent, {})
            self.assertIsNone(engine._pending_tree_reservation)
            self.assertEqual(
                prepared.proposal.tree.state, MacroWaveState.ROLLED_BACK
            )
            quote_stats = engine.speculation_stats["physical_tree"]["capacity_quote"]
            self.assertEqual(quote_stats["rejected"], 1)
            self.assertEqual(quote_stats["rejections"], {"kv_bytes": 1})
        finally:
            engine._retire_job(job, runner, exception=RuntimeError("test cleanup"))
            root.close()
            child.close()

    def test_tree_quote_nonce_step_and_stage_identity_fail_closed(self) -> None:
        cases = ("nonce", "step", "stage count")
        for mismatch in cases:
            with self.subTest(mismatch=mismatch):
                engine, runner, job, active = _prepared_tree_test_case(
                    ((20,),), max_new_tokens=4
                )
                root, child = socket.socketpair()
                try:
                    prepared = engine._prepare_decode_wave(
                        job, runner, active_sequences=1
                    )
                    assert isinstance(prepared, _PreparedPhysicalTreeWave)
                    engine._dispatch_root_waves(
                        [prepared], runner, root, LinkEmulator()
                    )
                    prepare = recv_frame(child)
                    quote = decode_tree_prepare(prepare)
                    result_step = prepare.step
                    result_quote = replace(quote, stage_count=engine.stages - 1)
                    if mismatch == "nonce":
                        result_quote = replace(result_quote, nonce=quote.nonce + 1)
                    elif mismatch == "step":
                        result_step += 1
                    else:
                        result_quote = replace(
                            result_quote, stage_count=engine.stages
                        )
                    with self.assertRaisesRegex(RuntimeError, mismatch):
                        engine._handle_return_value(
                            (
                                Frame(
                                    FrameType.TREE_PREPARE_RESULT,
                                    0,
                                    prepare.request_id,
                                    result_step,
                                    prepare.token_count,
                                    0,
                                    encode_tree_prepare_quote(result_quote),
                                ),
                                time.perf_counter(),
                            ),
                            active,
                            runner,
                            root,
                        )
                    child.settimeout(0.02)
                    with self.assertRaises(socket.timeout):
                        recv_frame(child)
                    self.assertEqual(runner.forks, [])
                    self.assertEqual(engine._physical_tree.tree_by_parent, {})
                    self.assertEqual(
                        engine.speculation_stats["physical_tree"]["capacity_quote"]
                        ["protocol_failures"],
                        1,
                    )
                finally:
                    engine._retire_job(
                        job, runner, exception=RuntimeError("test cleanup")
                    )
                    root.close()
                    child.close()

    def test_tree_quote_pending_cancel_releases_before_parent_cancel(self) -> None:
        engine, runner, job, active = _prepared_tree_test_case(
            ((20,),), max_new_tokens=4
        )
        root, child = socket.socketpair()
        try:
            prepared = engine._prepare_decode_wave(job, runner, active_sequences=1)
            assert isinstance(prepared, _PreparedPhysicalTreeWave)
            engine._dispatch_root_waves([prepared], runner, root, LinkEmulator())
            self.assertEqual(recv_frame(child).frame_type, FrameType.TREE_PREPARE)

            job.cancel_requested.set()
            engine._send_requested_cancellations(active, runner, root)
            reservation_cancel, parent_cancel = recv_frame(child), recv_frame(child)
            self.assertEqual(
                [reservation_cancel.frame_type, parent_cancel.frame_type],
                [FrameType.TREE_RESERVATION_CANCEL, FrameType.CANCEL],
            )
            self.assertEqual(
                [reservation_cancel.request_id, parent_cancel.request_id],
                [job.wire_id, job.wire_id],
            )
            self.assertIsNone(engine._pending_tree_reservation)
            self.assertEqual(
                prepared.proposal.tree.state, MacroWaveState.ROLLED_BACK
            )
            self.assertEqual(runner.forks, [])
            self.assertEqual(active, {})
            self.assertTrue(job.future.cancelled())
            self.assertEqual(
                engine.speculation_stats["physical_tree"]["capacity_quote"]
                ["cancelled"],
                1,
            )
        finally:
            root.close()
            child.close()

    def test_tree_quote_timeout_cancels_and_late_result_fails_closed(self) -> None:
        engine, runner, job, active = _prepared_tree_test_case(
            ((20,),), max_new_tokens=4
        )
        root, child = socket.socketpair()
        try:
            prepared = engine._prepare_decode_wave(job, runner, active_sequences=1)
            assert isinstance(prepared, _PreparedPhysicalTreeWave)
            engine._dispatch_root_waves([prepared], runner, root, LinkEmulator())
            prepare = recv_frame(child)
            quote = decode_tree_prepare(prepare)
            assert engine._pending_tree_reservation is not None
            engine._pending_tree_reservation.deadline_at = time.perf_counter() - 1

            with self.assertRaisesRegex(TimeoutError, "TREE_PREPARE_RESULT timeout"):
                engine._check_pipeline_timeouts(active)
            cancel = recv_frame(child)
            self.assertEqual(cancel.frame_type, FrameType.TREE_RESERVATION_CANCEL)
            self.assertIsNone(engine._pending_tree_reservation)
            self.assertEqual(
                prepared.proposal.tree.state, MacroWaveState.ROLLED_BACK
            )
            self.assertEqual(runner.forks, [])
            self.assertEqual(
                engine.speculation_stats["physical_tree"]["capacity_quote"]
                ["timeouts"],
                1,
            )

            late = replace(quote, stage_count=engine.stages - 1)
            with self.assertRaisesRegex(RuntimeError, "late or unsolicited"):
                engine._handle_return_value(
                    (
                        Frame(
                            FrameType.TREE_PREPARE_RESULT,
                            0,
                            prepare.request_id,
                            prepare.step,
                            prepare.token_count,
                            0,
                            encode_tree_prepare_quote(late),
                        ),
                        time.perf_counter(),
                    ),
                    active,
                    runner,
                    root,
                )
            self.assertEqual(runner.forks, [])
        finally:
            engine._retire_job(job, runner, exception=RuntimeError("test cleanup"))
            root.close()
            child.close()

    def test_physical_tree_timeout_and_queued_arrival_race_fail_closed(self) -> None:
        # A return timestamped before its deadline is accepted even if the
        # scheduler does not process it until after wall-clock expiry.
        engine, runner, job, active = _prepared_tree_test_case(
            ((20,),), max_new_tokens=4
        )
        root, child = socket.socketpair()
        try:
            prepared = engine._prepare_decode_wave(job, runner, active_sequences=1)
            assert isinstance(prepared, _PreparedPhysicalTreeWave)
            _dispatch_and_ready_tree_quote(
                engine, prepared, active, runner, root, child
            )
            recv_frame(child)
            recv_frame(child)
            wave = engine._physical_tree.wave(7)
            arrived = time.perf_counter()
            wave.deadline_at = arrived + 0.001
            engine._record_queued_tree_return(8, arrived)
            time.sleep(0.003)
            engine._check_pipeline_timeouts(active)
            continuation = engine._handle_return_value(
                (
                    Frame(
                        FrameType.VERIFY_RESULT,
                        0,
                        8,
                        1,
                        2,
                        0,
                        verify_result_payload((20, 99)),
                    ),
                    arrived,
                ),
                active,
                runner,
                root,
            )
            self.assertIsNotNone(continuation)
            for _ in range(1):
                recv_frame(child)  # PROMOTE; accepted leaf needs no TRUNCATE.
        finally:
            engine._retire_job(job, runner, exception=RuntimeError("test cleanup"))
            root.close()
            child.close()

        # With no queued return, the oldest missing virtual ID expires.
        engine, runner, job, active = _prepared_tree_test_case(
            ((20,),), max_new_tokens=3
        )
        root, child = socket.socketpair()
        try:
            prepared = engine._prepare_decode_wave(job, runner, active_sequences=1)
            assert isinstance(prepared, _PreparedPhysicalTreeWave)
            _dispatch_and_ready_tree_quote(
                engine, prepared, active, runner, root, child
            )
            recv_frame(child)
            recv_frame(child)
            engine._physical_tree.wave(7).deadline_at = time.perf_counter() - 1
            with self.assertRaisesRegex(TimeoutError, "virtual request 8"):
                engine._check_pipeline_timeouts(active)
        finally:
            engine._retire_job(job, runner, exception=RuntimeError("test cleanup"))
            root.close()
            child.close()

    def test_physical_tree_return_deadline_starts_after_last_verify_send(self) -> None:
        engine, runner, job, active = _prepared_tree_test_case(
            ((20,), (21,)), max_new_tokens=4
        )
        engine.config.socket_timeout_seconds = 0.02
        original_forward = runner.forward_ids

        def slow_forward(request_id: int, input_ids: torch.Tensor) -> torch.Tensor:
            time.sleep(0.015)
            return original_forward(request_id, input_ids)

        runner.forward_ids = slow_forward  # type: ignore[method-assign]
        root, child = socket.socketpair()
        try:
            prepared = engine._prepare_decode_wave(job, runner, active_sequences=1)
            assert isinstance(prepared, _PreparedPhysicalTreeWave)
            started = time.perf_counter()
            _dispatch_and_ready_tree_quote(
                engine, prepared, active, runner, root, child
            )
            for _ in range(4):
                recv_frame(child)
            wave = engine._physical_tree.wave(7)
            sent_times = tuple(
                leaf.verify_sent_at for leaf in wave.ordered_leaves
            )
            self.assertTrue(all(value is not None for value in sent_times))
            observed = tuple(float(value) for value in sent_times if value is not None)
            self.assertGreater(observed[-1] - started, engine.config.socket_timeout_seconds)
            self.assertGreaterEqual(
                wave.deadline_at - observed[-1],
                engine.config.socket_timeout_seconds - 0.001,
            )
            engine._check_pipeline_timeouts(active)
        finally:
            engine._retire_job(job, runner, exception=RuntimeError("test cleanup"))
            root.close()
            child.close()

    def test_physical_tree_late_arrival_and_recovery_are_fail_closed(self) -> None:
        engine, runner, job, active = _prepared_tree_test_case(
            ((20,),), max_new_tokens=3
        )
        root, child = socket.socketpair()
        try:
            prepared = engine._prepare_decode_wave(job, runner, active_sequences=1)
            assert isinstance(prepared, _PreparedPhysicalTreeWave)
            _dispatch_and_ready_tree_quote(
                engine, prepared, active, runner, root, child
            )
            recv_frame(child)
            recv_frame(child)
            deadline = time.perf_counter() - 0.01
            engine._physical_tree.wave(7).deadline_at = deadline
            arrived = time.perf_counter()
            engine._record_queued_tree_return(8, arrived)
            engine._check_pipeline_timeouts(active)
            with self.assertRaisesRegex(RuntimeError, "must be replayed"):
                engine._handle_return_value(
                    (
                        Frame(
                            FrameType.VERIFY_RESULT,
                            0,
                            8,
                            1,
                            2,
                            0,
                            verify_result_payload((20, 99)),
                        ),
                        arrived,
                    ),
                    active,
                    runner,
                    root,
                )
            with self.assertRaisesRegex(RuntimeError, "not replay-recovery eligible"):
                _ = engine.recovery_identity
        finally:
            engine._retire_job(job, runner, exception=RuntimeError("test cleanup"))
            root.close()
            child.close()

    def test_second_tree_during_quote_falls_back_without_another_prepare(self) -> None:
        engine = _root_batch_test_engine(
            prefill_chunk_tokens=0,
            speculative_max_draft_tokens=1,
            max_speculative_branches=8,
            max_speculative_branch_tokens=128,
            max_speculative_kv_bytes=4096,
        )
        engine.tree_draft_provider = _FixedTreeDraftProvider(((20,),))
        engine._request_counter = 2
        runner = _FakeRootBatchRunner()
        runner.active = {1: 3, 2: 3}
        jobs = [
            _GenerationJob(
                GenerationInput(910 + request_id, torch.tensor([[1, 2, 3]]), 4),
                None,
                wire_id=request_id,
                step=1,
                token_ids=[10],
                prefill_offset=3,
                prefill_acked_offset=3,
            )
            for request_id in (1, 2)
        ]
        active = {1: jobs[0], 2: jobs[1]}
        root, child = socket.socketpair()
        try:
            prepared = [
                engine._prepare_decode_wave(job, runner, active_sequences=2)
                for job in jobs
            ]
            self.assertTrue(
                all(isinstance(wave, _PreparedPhysicalTreeWave) for wave in prepared)
            )
            engine._dispatch_root_waves(prepared, runner, root, LinkEmulator())
            first = recv_frame(child)
            self.assertEqual(first.frame_type, FrameType.TREE_PREPARE)
            self.assertEqual(first.request_id, 1)
            child.settimeout(0.02)
            with self.assertRaises(socket.timeout):
                recv_frame(child)
            child.settimeout(None)

            # The second request has already fallen back logically, but its
            # root/remote KV mutation stays behind the global capacity barrier.
            engine._cancel_pending_tree_reservation(
                jobs[0], root, reason="unit test releases first quote"
            )
            engine._dispatch_root_waves([], runner, root, LinkEmulator())
            cancel, second = recv_frame(child), recv_frame(child)
            self.assertEqual(cancel.frame_type, FrameType.TREE_RESERVATION_CANCEL)
            self.assertEqual(second.frame_type, FrameType.ACTIVATION)
            self.assertEqual(second.request_id, 2)
            child.settimeout(0.02)
            with self.assertRaises(socket.timeout):
                recv_frame(child)
            self.assertEqual(runner.forks, [])
            self.assertEqual(engine._physical_tree.tree_by_parent, {})
            self.assertEqual(
                engine.speculation_stats["physical_tree"]["capacity_quote"]
                ["singleflight_fallbacks"],
                1,
            )
        finally:
            for job in jobs:
                engine._retire_job(job, runner, exception=RuntimeError("test cleanup"))
            root.close()
            child.close()

    def test_second_tree_while_first_verify_is_pending_uses_classic_path(self) -> None:
        engine, runner, first_job, active = _prepared_tree_test_case(
            ((20,),), max_new_tokens=4
        )
        second_job = _GenerationJob(
            GenerationInput(905, torch.tensor([[1, 2, 3]]), 4),
            None,
            wire_id=6,
            step=1,
            token_ids=[10],
            prefill_offset=3,
            prefill_acked_offset=3,
        )
        runner.active[6] = 3
        active[6] = second_job
        engine._jobs_by_client[905] = second_job
        engine._callback_routes[6] = second_job
        root, child = socket.socketpair()
        try:
            first = engine._prepare_decode_wave(
                first_job, runner, active_sequences=2
            )
            assert isinstance(first, _PreparedPhysicalTreeWave)
            _dispatch_and_ready_tree_quote(
                engine, first, active, runner, root, child
            )
            first_fork, first_verify = recv_frame(child), recv_frame(child)
            self.assertEqual(
                [first_fork.frame_type, first_verify.frame_type],
                [FrameType.FORK, FrameType.VERIFY],
            )
            self.assertIsNone(engine._pending_tree_reservation)
            self.assertIn(first_job.wire_id, engine._physical_tree.tree_by_parent)
            self.assertTrue(engine._physical_tree_live_children)

            second = engine._prepare_decode_wave(
                second_job, runner, active_sequences=2
            )
            self.assertIsInstance(second, _PreparedRootWave)
            assert isinstance(second, _PreparedRootWave)
            engine._dispatch_root_waves([second], runner, root, LinkEmulator())
            classic = recv_frame(child)
            self.assertEqual(classic.frame_type, FrameType.ACTIVATION)
            self.assertEqual(classic.request_id, second_job.wire_id)
            child.settimeout(0.02)
            with self.assertRaises(socket.timeout):
                recv_frame(child)
            self.assertEqual(
                engine.speculation_stats["physical_tree"]["capacity_quote"]
                ["singleflight_fallbacks"],
                1,
            )
        finally:
            engine._retire_job(
                first_job, runner, exception=RuntimeError("test cleanup")
            )
            engine._retire_job(
                second_job, runner, exception=RuntimeError("test cleanup")
            )
            root.close()
            child.close()

    def test_physical_tree_capacity_miss_falls_back_before_any_fork(self) -> None:
        engine = _root_batch_test_engine(
            prefill_chunk_tokens=0,
            speculative_max_draft_tokens=2,
            max_speculative_branches=8,
            max_speculative_branch_tokens=128,
            max_speculative_kv_bytes=100,
        )
        engine.tree_draft_provider = _FixedTreeDraftProvider(
            ((20, 21), (20, 22))
        )
        runner = _FakeRootBatchRunner()
        runner.active = {7: 3}
        job = _GenerationJob(
            GenerationInput(903, torch.tensor([[1, 2, 3]]), 5),
            None,
            wire_id=7,
            step=1,
            token_ids=[10],
            prefill_offset=3,
            prefill_acked_offset=3,
        )
        prepared = engine._prepare_decode_wave(job, runner, active_sequences=1)
        self.assertIsInstance(prepared, _PreparedRootWave)
        self.assertEqual(prepared.frame_type, FrameType.ACTIVATION)
        self.assertEqual(runner.forks, [])
        self.assertEqual(engine._physical_tree.tree_by_parent, {})


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


def _root_batch_test_engine(
    *,
    prefill_chunk_tokens: int,
    prefill_inflight_chunks: int = 1,
    prefill_inflight_bytes: int = 0,
    speculative_max_draft_tokens: int = 0,
    max_speculative_branches: int = 0,
    max_speculative_branch_tokens: int = 0,
    max_speculative_kv_bytes: int = 0,
) -> DistributedPipelineEngine:
    engine = DistributedPipelineEngine.__new__(DistributedPipelineEngine)
    engine.config = SimpleNamespace(
        boundaries=(0, 1, 2),
        max_active_sequences=8,
        prefill_chunk_tokens=prefill_chunk_tokens,
        prefill_inflight_chunks=prefill_inflight_chunks,
        prefill_inflight_bytes=prefill_inflight_bytes,
        speculative_max_draft_tokens=speculative_max_draft_tokens,
        max_speculative_branches=max_speculative_branches,
        max_speculative_branch_tokens=max_speculative_branch_tokens,
        max_speculative_kv_bytes=max_speculative_kv_bytes,
        sealed_wave_tokens=None,
        socket_timeout_seconds=2.0,
        speculation_probe=False,
        root_batch_window_ms=0.5,
        codec=TensorCodec.FP32,
    )
    engine.hidden_size = 4
    engine.maximum_context = 128
    engine._deferred_batches = deque()
    engine._request_counter = 0
    engine._callback_lock = threading.Lock()
    engine._callback_routes = {}
    engine._leaf_routes = {}
    engine._tree_return_lock = threading.Lock()
    engine._queued_tree_returns = {}
    engine._physical_tree_live_children = set()
    engine._physical_tree = PhysicalTreeCoordinator()
    engine._state_lock = threading.Lock()
    engine._jobs_by_client = {}
    engine.draft_provider = None
    engine.tree_draft_provider = None
    engine.speculation_controller = None
    engine._root_ready_items = 0
    engine._root_model_forward_calls = 0
    engine._root_physical_batch_calls = 0
    engine._root_physical_batch_items = 0
    engine._root_sequential_items = 0
    engine._root_max_physical_batch_size = 1
    engine._physical_tree_prepared_waves = 0
    engine._physical_tree_prepared_leaves = 0
    engine._physical_tree_committed_waves = 0
    engine._physical_tree_aborted_waves = 0
    engine._physical_tree_batch_calls = 0
    engine._physical_tree_batch_items = 0
    engine._pending_tree_reservation = None
    engine._tree_quote_nonce_counter = 0
    engine._queued_tree_prepare_results = deque()
    engine._queued_tree_commit_results = deque()
    engine._tree_barrier_deferred_waves = deque()
    engine._physical_tree_quote_requests = 0
    engine._physical_tree_quote_ready = 0
    engine._physical_tree_quote_rejected = 0
    engine._physical_tree_quote_cancelled = 0
    engine._physical_tree_quote_timeouts = 0
    engine._physical_tree_quote_singleflight_fallbacks = 0
    engine._physical_tree_quote_protocol_failures = 0
    engine._physical_tree_quote_rtt_seconds = 0.0
    engine._physical_tree_quote_rejections = {}
    engine._prefill_current_chunks = 0
    engine._prefill_current_bytes = 0
    engine._prefill_current_reserved_bytes = 0
    engine._prefill_high_water_chunks = 0
    engine._prefill_high_water_bytes = 0
    engine._prefill_high_water_reserved_bytes = 0
    engine._prefill_max_request_chunks = 0
    engine._prefill_max_request_bytes = 0
    engine._prefill_max_request_reserved_bytes = 0
    engine._prefill_dispatched_chunks = 0
    engine._prefill_completed_chunks = 0
    engine._prefill_acknowledged_chunks = 0
    return engine


def _prepared_tree_test_case(
    paths: tuple[tuple[int, ...], ...],
    *,
    max_new_tokens: int,
) -> tuple[
    DistributedPipelineEngine,
    "_FakeRootBatchRunner",
    _GenerationJob,
    dict[int, _GenerationJob],
]:
    maximum_depth = max(len(path) for path in paths)
    engine = _root_batch_test_engine(
        prefill_chunk_tokens=0,
        speculative_max_draft_tokens=maximum_depth,
        max_speculative_branches=8,
        max_speculative_branch_tokens=128,
        max_speculative_kv_bytes=4096,
    )
    engine.tree_draft_provider = _FixedTreeDraftProvider(paths)
    engine._request_counter = 7
    runner = _FakeRootBatchRunner()
    runner.active = {7: 3}
    job = _GenerationJob(
        GenerationInput(904, torch.tensor([[1, 2, 3]]), max_new_tokens),
        None,
        wire_id=7,
        step=1,
        started_at=time.perf_counter() - 1,
        token_ids=[10],
        arrivals=[time.perf_counter() - 0.5],
        prefill_offset=3,
        prefill_acked_offset=3,
    )
    engine._jobs_by_client = {904: job}
    engine._callback_routes = {7: job}
    return engine, runner, job, {7: job}


def _dispatch_and_ready_tree_quote(
    engine: DistributedPipelineEngine,
    prepared: _PreparedPhysicalTreeWave,
    active: dict[int, _GenerationJob],
    runner: "_FakeRootBatchRunner",
    root: socket.socket,
    child: socket.socket,
) -> Frame:
    """Drive PREPARE -> READY -> COMMIT, then materialise the physical leaves."""

    engine._dispatch_root_waves([prepared], runner, root, LinkEmulator())
    prepare = recv_frame(child)
    if prepare.frame_type != FrameType.TREE_PREPARE:
        raise AssertionError(f"expected TREE_PREPARE, got {prepare.frame_type.name}")
    quote = decode_tree_prepare(prepare)
    result_quote = replace(quote, stage_count=engine.stages - 1)
    awaiting_commit = engine._handle_return_value(
        (
            Frame(
                FrameType.TREE_PREPARE_RESULT,
                0,
                prepare.request_id,
                prepare.step,
                prepare.token_count,
                0,
                encode_tree_prepare_quote(result_quote),
            ),
            time.perf_counter(),
        ),
        active,
        runner,
        root,
    )
    if awaiting_commit is not None:
        raise AssertionError("READY quote must wait for the route COMMIT result")
    commit = recv_frame(child)
    if commit.frame_type != FrameType.TREE_RESERVATION_COMMIT:
        raise AssertionError(
            f"expected TREE_RESERVATION_COMMIT, got {commit.frame_type.name}"
        )
    committed = engine._handle_return_value(
        (
            Frame(
                FrameType.TREE_RESERVATION_COMMIT_RESULT,
                0,
                commit.request_id,
                commit.step,
                engine.stages - 1,
                0,
                commit.payload,
            ),
            time.perf_counter(),
        ),
        active,
        runner,
        root,
    )
    if not isinstance(committed, _CommittedPhysicalTreeWave):
        raise AssertionError("COMMIT result did not arm the physical tree")
    engine._dispatch_root_waves([committed], runner, root, LinkEmulator())
    return prepare


def _seed_inflight_wave(
    engine: DistributedPipelineEngine,
    job: _GenerationJob,
    frame_type: FrameType,
    *,
    prefill_end: int | None = None,
    outbound_bytes: int = 64,
) -> None:
    step = job.step
    started = time.perf_counter() - 0.1
    flight = _InflightWave(
        step=step,
        frame_type=frame_type,
        prefill_end=prefill_end,
        started_at=started,
        sent_at=time.monotonic() - 0.1,
        outbound_bytes=outbound_bytes,
        reserved_bytes=outbound_bytes,
    )
    job.inflight_waves.append(flight)
    job.next_step = step + 1
    job.last_sent_at = flight.sent_at
    job.last_outbound_bytes = outbound_bytes
    if prefill_end is not None:
        job.prefill_offset = max(job.prefill_offset, prefill_end)
        job.prefill_inflight_bytes += outbound_bytes
        job.prefill_reserved_bytes += outbound_bytes
        engine._prefill_current_chunks += 1
        engine._prefill_current_bytes += outbound_bytes
        engine._prefill_current_reserved_bytes += outbound_bytes


class _FakeRootBatchRunner:
    MAX_PHYSICAL_BATCH_SIZE = 8

    def __init__(
        self,
        *,
        batchable: bool = True,
        tree_batch_exact: bool = False,
    ) -> None:
        self.batchable = batchable
        self.executor_manifest = SimpleNamespace(
            features=(
                (
                    "exact-tree-verify-batching",
                    "bounded-tree-verify-workspace",
                )
                if tree_batch_exact
                else ()
            )
        )
        self.active: dict[int, int] = {}
        self.batch_calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        self.sequential_calls: list[tuple[int, int]] = []
        self.truncations: list[tuple[int, int]] = []
        self.forks: list[tuple[int, int, int]] = []
        self.promotions: list[tuple[int, int]] = []
        self.ends: list[int] = []

    def begin(self, request_id: int) -> None:
        if request_id in self.active:
            raise ValueError("duplicate BEGIN")
        self.active[request_id] = 0

    def end(self, request_id: int) -> None:
        self.ends.append(request_id)
        self.active.pop(request_id, None)

    def truncate(self, request_id: int, token_count: int) -> None:
        if not 0 <= token_count <= self.active[request_id]:
            raise ValueError("invalid truncate")
        self.active[request_id] = token_count
        self.truncations.append((request_id, token_count))

    def sequence_length(self, request_id: int) -> int:
        return self.active[request_id]

    def request_cache_bytes(self, request_id: int) -> int:
        return self.active[request_id] * 16

    def project_request_cache_bytes(
        self,
        request_id: int,
        additional_tokens: int,
    ) -> int:
        return (self.active[request_id] + additional_tokens) * 16

    def fork_request(
        self,
        child_request_id: int,
        parent_request_id: int,
        *,
        max_cache_bytes: int,
    ) -> int:
        copied = self.request_cache_bytes(parent_request_id)
        if copied > max_cache_bytes:
            raise ValueError("fork exceeds budget")
        if child_request_id in self.active:
            raise ValueError("duplicate fork child")
        self.active[child_request_id] = self.active[parent_request_id]
        self.forks.append((child_request_id, parent_request_id, copied))
        return copied

    def promote_request(
        self,
        parent_request_id: int,
        child_request_id: int,
    ) -> None:
        self.active[parent_request_id] = self.active.pop(child_request_id)
        self.promotions.append((parent_request_id, child_request_id))

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


class _FixedTreeDraftProvider:
    strategy = "fixed-tree-engine-test"

    def __init__(self, paths: tuple[tuple[int, ...], ...]) -> None:
        self.paths = paths
        self.max_draft_tokens = max(len(path) for path in paths)
        self.max_branches = len(paths)

    def draft_paths(
        self,
        token_history: tuple[int, ...],
        max_tokens: int | None = None,
        max_branches: int | None = None,
    ) -> tuple[tuple[int, ...], ...]:
        del token_history
        token_limit = self.max_draft_tokens if max_tokens is None else max_tokens
        branch_limit = self.max_branches if max_branches is None else max_branches
        return tuple(
            path for path in self.paths if len(path) <= token_limit
        )[:branch_limit]


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


class _LifecycleSocket:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def shutdown(self, how: int) -> None:
        self.events.append(
            "half-close-write" if how == socket.SHUT_WR else "hard-close"
        )

    def close(self) -> None:
        self.events.append("socket-close")


class _LifecycleScheduler:
    name = "scheduler"

    def __init__(
        self,
        events: list[str],
        engine: DistributedPipelineEngine,
    ) -> None:
        self.events = events
        self.engine = engine
        self.alive = True

    def is_alive(self) -> bool:
        return self.alive

    def join(self, timeout: float | None = None) -> None:
        del timeout
        self.events.append("scheduler-join")
        self.events.append("shutdown-sent")
        self.engine._shutdown_sent = True
        self.alive = False


class _LifecycleProcess:
    name = "child-stage"
    pid = 1234

    def __init__(
        self,
        events: list[str],
        *,
        exit_code: int,
        stubborn: bool = False,
    ) -> None:
        self.events = events
        self.target_exit_code = exit_code
        self.stubborn = stubborn
        self.alive = True
        self.exitcode: int | None = None

    def is_alive(self) -> bool:
        return self.alive

    def join(self, timeout: float | None = None) -> None:
        del timeout
        self.events.append("child-join")
        if not self.stubborn:
            self.alive = False
            self.exitcode = self.target_exit_code

    def terminate(self) -> None:
        self.events.append("child-terminate")
        self.alive = False
        self.exitcode = -15


class _LifecycleRunner:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def close(self) -> None:
        self.events.append("runner-close")


def _close_lifecycle_test_engine(
    events: list[str],
    processes: tuple[_LifecycleProcess, ...],
) -> DistributedPipelineEngine:
    engine = DistributedPipelineEngine.__new__(DistributedPipelineEngine)
    engine._state_lock = threading.Lock()
    engine._closed = False
    engine._close_error = None
    engine._shutdown_report = None
    engine._fatal_error = None
    engine._scheduler_stop = threading.Event()
    engine._submission_queue = queue.Queue()
    engine._received_frames = queue.Queue()
    engine._shutdown_sent = False
    engine._downstream = _LifecycleSocket(events)
    engine._return_socket = None
    engine._return_listener = None
    engine._receiver_thread = None
    engine._control_thread = None
    engine._processes = list(processes)
    engine._metrics_queue = None
    engine.stage_metrics = []
    engine._runner = _LifecycleRunner(events)
    engine._scheduler_thread = _LifecycleScheduler(events, engine)
    return engine


if __name__ == "__main__":
    unittest.main()
