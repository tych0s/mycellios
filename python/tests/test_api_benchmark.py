from __future__ import annotations

import argparse
import json
import unittest

import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestServer

from distributed_runtime.api_benchmark import (
    OpenLoopSample,
    mean_inflight,
    parse_final_stream_evidence,
    percentile,
    positive_csv,
    positive_float_csv,
    stream_request,
    summarize_open_loop,
    summary_stats,
)


def _sample(index: int, started: float, finished: float, *, tokens: int | None = 8,
            error: str | None = None) -> OpenLoopSample:
    return OpenLoopSample(
        index=index,
        scheduled_offset_s=started,
        started_offset_s=started,
        finished_offset_s=finished,
        response_ms=(finished - started) * 1_000,
        server_token_ttft_ms=None if error else 10.0,
        server_token_tpot_ms=None if error else 5.0,
        completion_tokens=None if error else tokens,
        finish_reason=None if error else "length",
        text_nonempty=error is None,
        error=error,
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


class OpenLoopBenchmarkTests(unittest.TestCase):
    def test_rate_list_is_strictly_positive_finite(self) -> None:
        self.assertEqual(positive_float_csv("0.6, 1.0,1.2"), (0.6, 1.0, 1.2))
        for raw in ("", "0", "-1", "1,nope", "inf"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                positive_float_csv(raw)

    def test_summary_stats_reports_none_when_empty(self) -> None:
        self.assertIsNone(summary_stats([]))
        stats = summary_stats([1, 2, 3, 4])
        self.assertEqual(stats["count"], 4)
        self.assertEqual((stats["min"], stats["max"]), (1.0, 4.0))

    def test_mean_inflight_is_overlap_over_window(self) -> None:
        # two requests each occupying 2s of a 3s window -> 4/3 mean in-flight
        samples = [_sample(0, 0.0, 2.0), _sample(1, 1.0, 3.0)]
        self.assertAlmostEqual(mean_inflight(samples, 0.0, 3.0), 4 / 3)
        self.assertEqual(mean_inflight(samples, 2.0, 2.0), 0.0)

    def test_summarize_open_loop_discards_warmup_and_counts_errors(self) -> None:
        args = argparse.Namespace(warmup_seconds=0.5, duration_seconds=3.0, seed=7)
        samples = [
            _sample(0, 0.0, 2.0),  # starts before warmup -> excluded from steady state
            _sample(1, 1.0, 3.0),  # steady
            _sample(2, 2.0, 2.1, error="RuntimeError: boom"),
        ]
        out = summarize_open_loop(samples, 1.0, args)
        self.assertEqual(out["offered_requests"], 3)
        self.assertEqual(out["completed_requests"], 2)
        self.assertEqual(out["errors"], 1)
        self.assertEqual(out["error_examples"], ["RuntimeError: boom"])
        steady = out["steady_state"]
        self.assertEqual(steady["requests"], 1)  # only the request that starts >= warmup
        self.assertEqual(steady["completion_tokens_per_request"]["count"], 1)
        self.assertEqual(
            steady["aggregate_completion_tokens_per_second"], 8 / (3.0 - 0.5)
        )


if __name__ == "__main__":
    unittest.main()
