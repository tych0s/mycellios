from __future__ import annotations

import json
import unittest

import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestServer

from distributed_runtime.api_benchmark import (
    parse_final_stream_evidence,
    percentile,
    positive_csv,
    stream_request,
)


class ApiBenchmarkUtilityTests(unittest.TestCase):
    def test_concurrency_list_is_strictly_positive(self) -> None:
        self.assertEqual(positive_csv("1, 2,8"), (1, 2, 8))
        for raw in ("", "0,1", "1,nope"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                positive_csv(raw)

    def test_percentile_interpolates(self) -> None:
        self.assertEqual(percentile([7], 0.95), 7)
        self.assertAlmostEqual(percentile([0, 10], 0.95), 9.5)
        with self.assertRaises(ValueError):
            percentile([], 0.95)

    def test_final_evidence_uses_actual_tokens_and_checks_consistency(self) -> None:
        document = {
            "choices": [{"finish_reason": "stop", "delta": {}}],
            "usage": {
                "prompt_tokens": 7,
                "completion_tokens": 3,
                "total_tokens": 10,
            },
            "distribution_metrics": {
                "ttft_ms": 5.0,
                "tpot_ms": 2.0,
                "pipeline_ms": 9.0,
                "output_token_ids_sha256": "sha256:" + "a" * 64,
                "output_token_ids_hash_scheme": "gdlp-output-token-ids-v1",
            },
        }
        parsed = parse_final_stream_evidence(document)
        self.assertEqual(parsed["completion_tokens"], 3)
        self.assertEqual(parsed["server_token_tpot_ms"], 2.0)

        broken = {**document, "usage": {**document["usage"], "total_tokens": 11}}
        with self.assertRaisesRegex(RuntimeError, "total_tokens"):
            parse_final_stream_evidence(broken)

        broken = {
            **document,
            "distribution_metrics": {
                **document["distribution_metrics"],
                "pipeline_ms": 99.0,
            },
        }
        with self.assertRaisesRegex(RuntimeError, "internally inconsistent"):
            parse_final_stream_evidence(broken)

    def test_final_evidence_rejects_nominal_or_ambiguous_token_data(self) -> None:
        base = {
            "choices": [{"finish_reason": "length", "delta": {}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "distribution_metrics": {
                "ttft_ms": 1.0,
                "tpot_ms": 0.0,
                "pipeline_ms": 1.0,
                "output_token_ids_sha256": "sha256:" + "0" * 64,
                "output_token_ids_hash_scheme": "gdlp-output-token-ids-v1",
            },
        }
        for change in (
            {"usage": {"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1}},
            {"distribution_metrics": {**base["distribution_metrics"], "output_token_ids_hash_scheme": "unknown"}},
            {"distribution_metrics": {**base["distribution_metrics"], "output_token_ids_sha256": "not-a-hash"}},
        ):
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                parse_final_stream_evidence({**base, **change})


class ApiBenchmarkHttpTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_parser_uses_early_eos_actual_tokens_across_fragments(self) -> None:
        async def completion(_: web.Request) -> web.StreamResponse:
            response = web.StreamResponse(headers={"content-type": "text/event-stream"})
            await response.prepare(_)
            content = {
                "choices": [
                    {"delta": {"content": "ok"}, "finish_reason": None, "index": 0}
                ]
            }
            final = {
                "choices": [{"delta": {}, "finish_reason": "stop", "index": 0}],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 2,
                    "total_tokens": 7,
                },
                "distribution_metrics": {
                    "ttft_ms": 4.0,
                    "tpot_ms": 3.0,
                    "pipeline_ms": 7.0,
                    "output_token_ids_sha256": "sha256:" + "1" * 64,
                    "output_token_ids_hash_scheme": "gdlp-output-token-ids-v1",
                },
            }
            wire = (
                "data: "
                + json.dumps(content, separators=(",", ":"))
                + "\n\ndata: "
                + json.dumps(final, separators=(",", ":"))
                + "\n\ndata: [DONE]\n\n"
            ).encode()
            for boundary in (3, 17, 41, len(wire)):
                chunk, wire = wire[:boundary], wire[boundary:]
                if chunk:
                    await response.write(chunk)
            if wire:
                await response.write(wire)
            return response

        app = web.Application()
        app.router.add_post("/v1/chat/completions", completion)
        async with TestServer(app) as server, aiohttp.ClientSession() as session:
            value = await stream_request(
                session,
                str(server.make_url("/v1/chat/completions")),
                "distributed-test",
                "prompt",
                8,
                "early-eos",
            )

        self.assertEqual(value.completion_tokens, 2)
        self.assertEqual(value.finish_reason, "stop")
        self.assertEqual(value.output_token_ids_sha256, "sha256:" + "1" * 64)
        self.assertTrue(value.text_nonempty)


if __name__ == "__main__":
    unittest.main()
