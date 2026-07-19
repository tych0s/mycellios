from __future__ import annotations

import asyncio
from concurrent.futures import Future
from types import SimpleNamespace
import unittest

import torch

from distributed_runtime.engine import (
    GenerationOutput,
    PipelineEngineConfig,
    balanced_boundaries,
    parse_boundaries,
)
from distributed_runtime.protocol import TensorCodec
from distributed_runtime.server import (
    ContinuousMicroBatcher,
    DistributedOpenAIServer,
    IncrementalTokenDecoder,
    PendingGeneration,
)


class EngineConfigurationTests(unittest.TestCase):
    def test_balanced_and_explicit_boundaries(self) -> None:
        self.assertEqual(balanced_boundaries(30, 4), (0, 8, 15, 22, 30))
        self.assertEqual(parse_boundaries("0,9,30", 30), (0, 9, 30))
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            parse_boundaries("0,10,10,30", 30)
        with self.assertRaisesRegex(ValueError, "between 2"):
            balanced_boundaries(4, 5)

    def test_remote_pipeline_requires_routable_ports(self) -> None:
        common = dict(
            model_name="fake",
            boundaries=(0, 2, 4),
            spawn_local_stages=False,
            return_port=20_002,
        )
        with self.assertRaisesRegex(ValueError, "first_stage_port is required"):
            PipelineEngineConfig(**common)
        with self.assertRaisesRegex(ValueError, "non-zero first_stage_port"):
            PipelineEngineConfig(**common, first_stage_port=0)
        with self.assertRaisesRegex(ValueError, "return_port"):
            PipelineEngineConfig(
                **{**common, "return_port": 0},
                first_stage_port=20_001,
            )


class IncrementalDecoderTests(unittest.TestCase):
    def test_incomplete_utf8_is_held_until_the_next_token(self) -> None:
        decoder = IncrementalTokenDecoder(_FakeTokenizer())
        self.assertEqual(decoder.push(1), "caf")
        self.assertEqual(decoder.push(2), "é")
        self.assertEqual(decoder.finish(), "")


class ContinuousMicroBatcherTests(unittest.IsolatedAsyncioTestCase):
    async def test_two_arrivals_share_one_generation_round(self) -> None:
        engine = _FakeEngine()
        batcher = ContinuousMicroBatcher(
            engine,
            max_batch_size=4,
            batch_window_ms=10,
            eos_token_ids=frozenset((99,)),
        )
        await batcher.start()
        first = PendingGeneration(1, torch.tensor([[1, 2]]), 2, 2)
        second = PendingGeneration(2, torch.tensor([[3]]), 2, 1)
        try:
            await asyncio.gather(batcher.submit(first), batcher.submit(second))
            first_events = await _read_until_done(first)
            second_events = await _read_until_done(second)
        finally:
            await batcher.close()

        self.assertEqual(engine.batch_sizes, [2])
        self.assertEqual([kind for kind, _ in first_events], ["token", "token", "done"])
        self.assertEqual([kind for kind, _ in second_events], ["token", "token", "done"])
        self.assertEqual(batcher.batches, 1)
        self.assertEqual(batcher.requests, 2)
        self.assertTrue(engine.closed)

    async def test_new_batch_is_dispatched_while_previous_generation_is_active(self) -> None:
        engine = _ControllableEngine()
        batcher = ContinuousMicroBatcher(
            engine,
            max_batch_size=1,
            batch_window_ms=0,
            eos_token_ids=frozenset(),
        )
        await batcher.start()
        first = PendingGeneration(11, torch.tensor([[1]]), 1, 1)
        second = PendingGeneration(22, torch.tensor([[2]]), 1, 1)
        try:
            await batcher.submit(first)
            await _eventually(lambda: len(engine.submissions) == 1)
            await batcher.submit(second)
            await _eventually(lambda: len(engine.submissions) == 2)
            self.assertFalse(engine.submissions[0][0].done())
            engine.complete_all()
            await _read_until_done(first)
            await _read_until_done(second)
        finally:
            await batcher.close()


class RequestValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.server = DistributedOpenAIServer(
            _PrepareEngine(),
            _TemplateTokenizer(),
            public_model_name="distributed-small",
            max_batch_size=4,
            batch_window_ms=0,
            max_output_tokens=32,
        )

    def valid(self):
        return {
            "model": "distributed-small",
            "messages": [{"role": "user", "content": "hola"}],
            "temperature": 0,
            "max_tokens": 4,
            "stream": False,
        }

    def test_json_types_are_not_coerced(self) -> None:
        cases = (
            ({**self.valid(), "stream": "false"}, "stream"),
            ({**self.valid(), "n": 1.9}, "n"),
            ({**self.valid(), "max_tokens": 1.9}, "max_tokens"),
        )
        for body, message in cases:
            with self.subTest(body=body), self.assertRaisesRegex(ValueError, message):
                self.server._prepare_request(body)

    def test_model_is_required_and_unsupported_fields_are_rejected(self) -> None:
        missing = self.valid()
        missing.pop("model")
        with self.assertRaisesRegex(ValueError, "model is required"):
            self.server._prepare_request(missing)
        with self.assertRaisesRegex(ValueError, "unsupported request field"):
            self.server._prepare_request({**self.valid(), "tools": []})

    def test_valid_developer_message_is_normalized(self) -> None:
        body = self.valid()
        body["messages"] = [{"role": "developer", "content": "sé breve"}]
        pending, stream = self.server._prepare_request(body)
        self.assertFalse(stream)
        self.assertEqual(pending.max_new_tokens, 4)


async def _read_until_done(pending: PendingGeneration):
    events = []
    while True:
        event = await asyncio.wait_for(pending.events.get(), 2)
        events.append(event)
        if event[0] in ("done", "error"):
            return events


async def _eventually(predicate) -> None:
    for _ in range(100):
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("condition was not reached")


class _FakeTokenizer:
    def decode(self, token_ids, **_: object) -> str:
        return {
            (1,): "caf\ufffd",
            (1, 2): "café",
        }[tuple(token_ids)]


class _FakeEngine:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []
        self.closed = False
        self.config = SimpleNamespace(codec=TensorCodec.FP16)

    def submit(self, requests, callback):
        self.batch_sizes.append(len(requests))
        futures = []
        for request in requests:
            for step, token_id in enumerate((10, 11)):
                callback(request.client_id, token_id, step, 1.0 + step)
            future = Future()
            future.set_result(
                GenerationOutput(
                    request.client_id,
                    (10, 11),
                    "length",
                    1.0,
                    2.0,
                    3.0,
                )
            )
            futures.append(future)
        return futures

    def cancel(self, _client_id: int) -> bool:
        return False

    def close(self) -> None:
        self.closed = True


class _ControllableEngine(_FakeEngine):
    def __init__(self) -> None:
        super().__init__()
        self.submissions: list[list[Future]] = []
        self.requests = []

    def submit(self, requests, callback):
        futures = [Future() for _ in requests]
        self.requests.extend(requests)
        self.submissions.append(futures)
        return futures

    def complete_all(self) -> None:
        for request, future in zip(self.requests, [item for batch in self.submissions for item in batch]):
            if not future.done():
                future.set_result(
                    GenerationOutput(request.client_id, (10,), "length", 1, 0, 1)
                )


class _PrepareEngine(_FakeEngine):
    maximum_context = 128


class _TemplateTokenizer:
    eos_token_id = 99

    def apply_chat_template(self, messages, **_: object):
        self.messages = messages
        return torch.tensor([[1, 2, 3]], dtype=torch.long)


if __name__ == "__main__":
    unittest.main()
