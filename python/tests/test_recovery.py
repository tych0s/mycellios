from __future__ import annotations

from concurrent.futures import CancelledError, Future
from dataclasses import replace
from types import SimpleNamespace
import threading
import unittest

import torch

from distributed_runtime.engine import (
    GenerationInput,
    GenerationOutput,
    PipelineEngineConfig,
    PipelineRecoveryIdentity,
)
from distributed_runtime.protocol import TensorCodec
from distributed_runtime.recovery import (
    PipelineRecoveryError,
    RecoveringPipelineEngine,
)


class RecoveringPipelineEngineTests(unittest.TestCase):
    def test_physical_tree_limits_are_part_of_recovery_identity_v4(self) -> None:
        disabled = _identity()
        enabled = replace(
            disabled,
            max_speculative_branches=2,
            max_speculative_branch_tokens=128,
            max_speculative_kv_bytes=4096,
        )
        self.assertEqual(disabled.schema_version, 4)
        self.assertNotEqual(disabled, enabled)

    def test_executor_contract_requires_one_sealed_id_per_stage(self) -> None:
        common = {"model_name": "fixture", "boundaries": (0, 2, 4)}
        with self.assertRaisesRegex(ValueError, "one id per stage"):
            PipelineEngineConfig(**common, stage_executor_ids=("1" * 32,))
        with self.assertRaisesRegex(ValueError, "32 lowercase hex"):
            PipelineEngineConfig(
                **common,
                stage_executor_ids=("1" * 32, "NOT-A-SEALED-EXECUTOR-ID"),
            )

    def test_midstream_failure_replays_prefix_without_duplicate_tokens(self) -> None:
        identity = _identity()
        expected = {7: (10, 11, 12, 13)}
        prompt_lengths = {7: 2}
        first = _ScriptedEngine(
            identity,
            expected,
            prompt_lengths,
            emit_before_failure=2,
        )
        replacements: list[_ScriptedEngine] = []

        def factory() -> _ScriptedEngine:
            engine = _ScriptedEngine(identity, expected, prompt_lengths)
            replacements.append(engine)
            return engine

        engine = RecoveringPipelineEngine(
            factory,
            initial_engine=first,
            max_retries=1,
        )
        observed: list[tuple[int, int]] = []
        try:
            output = engine.generate(
                [GenerationInput(7, torch.tensor([[1, 2]]), 4)],
                lambda _client, token, step, _arrived: observed.append((step, token)),
            )[0]
        finally:
            engine.close()

        self.assertEqual(output.token_ids, expected[7])
        self.assertEqual(output.finish_reason, "length")
        self.assertEqual(observed, [(0, 10), (1, 11), (2, 12), (3, 13)])
        self.assertEqual(len(replacements), 1)
        self.assertEqual(replacements[0].observed_inputs[0].tolist(), [[1, 2, 10, 11]])
        self.assertEqual(replacements[0].observed_max_new_tokens, [2])
        stats = engine.recovery_stats
        self.assertEqual(stats["route_recovery_successes"], 1)
        self.assertEqual(stats["recovered_requests"], 1)
        self.assertEqual(stats["replayed_output_tokens"], 2)

    def test_one_failed_batch_recovers_multiple_requests_together(self) -> None:
        identity = _identity()
        expected = {101: (20, 21, 22), 202: (30, 31)}
        prompt_lengths = {101: 1, 202: 2}
        first = _ScriptedEngine(
            identity,
            expected,
            prompt_lengths,
            emit_before_failure=1,
        )
        replacement = _ScriptedEngine(identity, expected, prompt_lengths)
        engine = RecoveringPipelineEngine(
            lambda: replacement,
            initial_engine=first,
            max_retries=1,
        )
        callbacks: dict[int, list[int]] = {101: [], 202: []}
        try:
            outputs = engine.generate(
                [
                    GenerationInput(101, torch.tensor([[5]]), 3),
                    GenerationInput(202, torch.tensor([[6, 7]]), 2),
                ],
                lambda client, token, _step, _arrived: callbacks[client].append(token),
            )
        finally:
            engine.close()

        self.assertEqual([output.token_ids for output in outputs], [expected[101], expected[202]])
        self.assertEqual(callbacks, {101: [20, 21, 22], 202: [30, 31]})
        self.assertEqual(len(replacement.observed_batches), 1)
        self.assertEqual(set(replacement.observed_batches[0]), {101, 202})
        self.assertEqual(engine.recovery_stats["recovered_requests"], 2)

    def test_standby_with_different_child_executor_contract_fails_closed(self) -> None:
        expected = {9: (40, 41)}
        prompt_lengths = {9: 1}
        first = _ScriptedEngine(
            _identity(),
            expected,
            prompt_lengths,
            emit_before_failure=1,
        )
        incompatible = _ScriptedEngine(
            _identity(stage_executor_ids=("1" * 32, "9" * 32)),
            expected,
            prompt_lengths,
        )
        engine = RecoveringPipelineEngine(
            lambda: incompatible,
            initial_engine=first,
            max_retries=1,
        )
        try:
            future = engine.submit(
                [GenerationInput(9, torch.tensor([[8]]), 2)]
            )[0]
            with self.assertRaisesRegex(
                PipelineRecoveryError,
                "immutable-compatible recovery route",
            ):
                future.result(timeout=2)
            self.assertFalse(engine.healthy)
            self.assertIn("recovery identity differs", engine.fatal_error or "")
            self.assertGreaterEqual(engine.recovery_stats["compatibility_rejections"], 1)
        finally:
            engine.close()

    def test_remote_route_without_explicit_executor_contract_is_rejected(self) -> None:
        route = _ScriptedEngine(_identity(), {1: (2,)}, {1: 1})
        route.config.spawn_local_stages = False
        route.config.stage_executor_ids = None
        with self.assertRaisesRegex(
            PipelineRecoveryError,
            "remote recovery requires explicit sealed stage_executor_ids",
        ):
            RecoveringPipelineEngine(lambda: route, initial_engine=route)
        self.assertTrue(route.closed)

    def test_request_fails_after_max_retries_instead_of_looping(self) -> None:
        identity = _identity()
        expected = {11: (50, 51, 52)}
        prompt_lengths = {11: 1}
        first = _ScriptedEngine(
            identity,
            expected,
            prompt_lengths,
            emit_before_failure=1,
        )
        built: list[_ScriptedEngine] = []

        def factory() -> _ScriptedEngine:
            route = _ScriptedEngine(
                identity,
                expected,
                prompt_lengths,
                emit_before_failure=1,
            )
            built.append(route)
            return route

        engine = RecoveringPipelineEngine(
            factory,
            initial_engine=first,
            max_retries=1,
        )
        seen: list[int] = []
        try:
            future = engine.submit(
                [GenerationInput(11, torch.tensor([[3]]), 3)],
                lambda _client, token, _step, _arrived: seen.append(token),
            )[0]
            with self.assertRaisesRegex(PipelineRecoveryError, "exceeded max_retries=1"):
                future.result(timeout=2)
        finally:
            engine.close()

        self.assertEqual(seen, [50, 51])
        self.assertEqual(len(built), 1)
        self.assertEqual(engine.recovery_stats["request_retry_attempts"], 2)

    def test_eos_received_before_transport_failure_completes_without_replay(self) -> None:
        identity = _identity()
        expected = {12: (60, 99, 61)}
        prompt_lengths = {12: 1}
        first = _ScriptedEngine(
            identity,
            expected,
            prompt_lengths,
            emit_before_failure=2,
        )
        factory_calls = 0

        def factory() -> _ScriptedEngine:
            nonlocal factory_calls
            factory_calls += 1
            return _ScriptedEngine(identity, expected, prompt_lengths)

        engine = RecoveringPipelineEngine(
            factory,
            initial_engine=first,
            max_retries=1,
        )
        try:
            output = engine.generate(
                [
                    GenerationInput(
                        12,
                        torch.tensor([[4]]),
                        3,
                        eos_token_ids=frozenset((99,)),
                    )
                ]
            )[0]
        finally:
            engine.close()

        self.assertEqual(output.token_ids, (60, 99))
        self.assertEqual(output.finish_reason, "stop")
        self.assertEqual(factory_calls, 0)

    def test_simultaneous_failed_submissions_build_only_one_replacement_route(self) -> None:
        identity = _identity()
        expected = {71: (10, 11), 72: (20, 21)}
        prompt_lengths = {71: 1, 72: 1}
        first = _ConcurrentFailEngine(identity, expected, prompt_lengths)
        replacement = _ScriptedEngine(identity, expected, prompt_lengths)
        factory_calls = 0

        def factory() -> _ScriptedEngine:
            nonlocal factory_calls
            factory_calls += 1
            return replacement

        engine = RecoveringPipelineEngine(
            factory,
            initial_engine=first,
            max_retries=1,
        )
        try:
            first_future = engine.submit(
                [GenerationInput(71, torch.tensor([[1]]), 2)]
            )[0]
            second_future = engine.submit(
                [GenerationInput(72, torch.tensor([[2]]), 2)]
            )[0]
            self.assertEqual(first_future.result(timeout=2).token_ids, expected[71])
            self.assertEqual(second_future.result(timeout=2).token_ids, expected[72])
        finally:
            engine.close()

        self.assertEqual(factory_calls, 1)
        self.assertEqual(engine.recovery_stats["route_recovery_successes"], 1)

    def test_cancel_during_an_attempt_cancels_the_wrapper_future(self) -> None:
        identity = _identity()
        route = _BlockingEngine(identity, {81: (1,)}, {81: 1})
        engine = RecoveringPipelineEngine(
            lambda: route,
            initial_engine=route,
            max_retries=1,
        )
        try:
            future = engine.submit(
                [GenerationInput(81, torch.tensor([[3]]), 1)]
            )[0]
            self.assertTrue(route.submitted.wait(1))
            self.assertTrue(engine.cancel(81))
            with self.assertRaises(CancelledError):
                future.result(timeout=2)
            self.assertFalse(engine.cancel(81))
        finally:
            engine.close()


class _ScriptedEngine:
    def __init__(
        self,
        identity: PipelineRecoveryIdentity,
        expected: dict[int, tuple[int, ...]],
        prompt_lengths: dict[int, int],
        *,
        emit_before_failure: int | None = None,
    ) -> None:
        self.recovery_identity = identity
        self.expected = expected
        self.prompt_lengths = prompt_lengths
        self.emit_before_failure = emit_before_failure
        self.config = SimpleNamespace(
            boundaries=identity.boundaries,
            codec=TensorCodec(identity.codec),
            max_pending_requests=32,
            spawn_local_stages=True,
            stage_executor_ids=None,
        )
        self.maximum_context = identity.maximum_context
        self.total_layers = identity.total_layers
        self.hidden_size = identity.hidden_size
        self.model_snapshot = "fixture"
        self.model_artifact = SimpleNamespace(identity=identity.artifact_identity)
        self.pipeline_id = identity.pipeline_snapshot_identity
        self.root_parameter_bytes = 1234
        self.speculation_stats = {"configured": False, "enabled": False}
        self.root_batch_stats = {"physical_batch_calls": 0}
        self.stage_metrics: list[dict[str, object]] = []
        self.healthy = True
        self.fatal_error: str | None = None
        self.closed = False
        self.observed_inputs: list[torch.Tensor] = []
        self.observed_max_new_tokens: list[int] = []
        self.observed_batches: list[tuple[int, ...]] = []

    def submit(self, requests, callback):
        self.observed_batches.append(tuple(request.client_id for request in requests))
        futures = [Future() for _ in requests]
        suffixes: list[tuple[int, ...]] = []
        for request in requests:
            self.observed_inputs.append(request.input_ids.clone())
            self.observed_max_new_tokens.append(request.max_new_tokens)
            prompt_length = self.prompt_lengths[request.client_id]
            replayed = tuple(int(value) for value in request.input_ids[0, prompt_length:])
            expected = self.expected[request.client_id]
            if replayed != expected[: len(replayed)]:
                raise AssertionError(f"invalid replayed prefix {replayed!r}")
            available = expected[len(replayed) : len(replayed) + request.max_new_tokens]
            if self.emit_before_failure is not None:
                available = available[: self.emit_before_failure]
            suffixes.append(tuple(available))

        for request, suffix in zip(requests, suffixes):
            for step, token in enumerate(suffix):
                callback(request.client_id, token, step, 0.0)

        if self.emit_before_failure is not None:
            self.healthy = False
            self.fatal_error = "synthetic route failure"
            for future in futures:
                future.set_exception(RuntimeError(self.fatal_error))
        else:
            for request, suffix, future in zip(requests, suffixes, futures):
                reason = (
                    "stop"
                    if suffix and suffix[-1] in request.eos_token_ids
                    else "length"
                )
                future.set_result(
                    GenerationOutput(
                        request.client_id,
                        suffix,
                        reason,
                        1.0,
                        1.0,
                        1.0,
                    )
                )
        return futures

    def cancel(self, _client_id: int) -> bool:
        return False

    def close(self) -> None:
        self.closed = True


class _ConcurrentFailEngine(_ScriptedEngine):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.barrier = threading.Barrier(2)

    def submit(self, requests, callback):
        futures = [Future() for _ in requests]
        self.barrier.wait(timeout=1)
        for request, future in zip(requests, futures):
            prompt_length = self.prompt_lengths[request.client_id]
            replayed = tuple(int(value) for value in request.input_ids[0, prompt_length:])
            token = self.expected[request.client_id][len(replayed)]
            callback(request.client_id, token, 0, 0.0)
            future.set_exception(RuntimeError("simultaneous synthetic failure"))
        self.healthy = False
        self.fatal_error = "simultaneous synthetic failure"
        return futures


class _BlockingEngine(_ScriptedEngine):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.submitted = threading.Event()
        self.pending: dict[int, Future] = {}

    def submit(self, requests, _callback):
        futures = [Future() for _ in requests]
        self.pending.update(
            (request.client_id, future)
            for request, future in zip(requests, futures)
        )
        self.submitted.set()
        return futures

    def cancel(self, client_id: int) -> bool:
        future = self.pending.get(client_id)
        return future.cancel() if future is not None else False

    def close(self) -> None:
        super().close()
        for future in self.pending.values():
            future.cancel()


def _identity(
    *, artifact_identity: str = "sha256:same",
    stage_executor_ids: tuple[str, ...] = ("1" * 32, "2" * 32),
) -> PipelineRecoveryIdentity:
    return PipelineRecoveryIdentity(
        schema_version=4,
        artifact_identity=artifact_identity,
        canonical_model_source="hf://fixture/model",
        canonical_model_revision="a" * 40,
        pipeline_snapshot_identity=123,
        boundaries=(0, 2, 4),
        codec=int(TensorCodec.FP32),
        total_layers=4,
        hidden_size=16,
        maximum_context=128,
        root_stage_loader="fixture",
        stage_executor_ids=stage_executor_ids,
        threads_per_stage=1,
        prefill_chunk_tokens=0,
        prefill_inflight_chunks=1,
        prefill_inflight_bytes=0,
        max_speculative_branches=0,
        max_speculative_branch_tokens=0,
        max_speculative_kv_bytes=0,
        sealed_wave_tokens=1,
        max_prefill_chunk_tokens=0,
        speculative_max_draft_tokens=0,
        speculation_minimum_speedup=1.05,
        speculation_probe=True,
    )


if __name__ == "__main__":
    unittest.main()
