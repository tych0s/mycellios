from __future__ import annotations

import os
import threading
import time
import unittest

from distributed_runtime.engine import (
    DistributedPipelineEngine,
    GenerationInput,
    PipelineEngineConfig,
    _resolve_verified_tokens,
    _speculation_load_profile,
)
from distributed_runtime.model import load_tokenizer, reference_generate
from distributed_runtime.protocol import TensorCodec
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
            speculative_max_draft_tokens=8,
            speculation_minimum_speedup=1.1,
        )
        self.assertEqual(configured.prefill_chunk_tokens, 32)
        self.assertEqual(configured.speculative_max_draft_tokens, 8)
        for invalid in (-1, 17, True):
            with self.subTest(max_draft_tokens=invalid), self.assertRaises(
                (TypeError, ValueError)
            ):
                PipelineEngineConfig(
                    **common,
                    speculative_max_draft_tokens=invalid,
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
        engine._speculation_controllers = {}
        load_one = engine._speculation_controller_for_profile_locked("load-1")
        load_four = engine._speculation_controller_for_profile_locked("load-3-4")
        self.assertIs(load_one, engine.speculation_controller)
        self.assertIsNot(load_one, load_four)
        self.assertIs(
            engine._speculation_controller_for_profile_locked("load-3-4"),
            load_four,
        )


@unittest.skipUnless(
    os.environ.get("RUN_DISTRIBUTED_MODEL_TESTS") == "1",
    "set RUN_DISTRIBUTED_MODEL_TESTS=1 to run the continuous scheduler integration",
)
class ContinuousSchedulerIntegrationTests(unittest.TestCase):
    MODEL_NAME = os.environ.get(
        "DISTRIBUTED_TEST_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct"
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
                speculative_max_draft_tokens=1,
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
                    max_draft_tokens=1,
                    candidate_sizes=(1,),
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
        finally:
            engine.close()


class _ReferenceDraftProvider:
    strategy = "test-reference"
    max_draft_tokens = 1

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
            candidate = reference[generated]
            # Later probes intentionally miss so the physical integration also
            # traverses TRUNCATE and proves the target correction remains exact.
            if generated >= 4:
                candidate = (candidate + 1) % 49_152
            return (candidate,)
        return ()


if __name__ == "__main__":
    unittest.main()
