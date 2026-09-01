from __future__ import annotations

import asyncio
from concurrent.futures import Future
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from aiohttp.test_utils import TestClient, TestServer
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
    DistributedMycelliosServer,
    IncrementalTokenDecoder,
    OUTPUT_TOKEN_HASH_SCHEME,
    PendingGeneration,
    build_server,
    chunk_payload,
    output_token_ids_sha256,
    parse_args as parse_server_args,
    parse_remote_recovery_standby_routes,
    resolve_execution_tokenizer_snapshot,
    tree_draft_provider_from_args,
)


class EngineConfigurationTests(unittest.TestCase):
    def test_root_stage_resolves_tokenizer_from_authenticated_model_coordinates(
        self,
    ) -> None:
        package_identity = f"sha256:{'a' * 64}"
        manifest = SimpleNamespace(
            to_document=lambda: {
                "model": {
                    "source": "HMellor/tiny-random-LlamaForCausalLM",
                    "revision": "9408c553e5c189a7dcdc5a5dbd2feb476b061759",
                }
            }
        )
        verified = SimpleNamespace(manifest=manifest)
        with (
            patch(
                "distributed_runtime.stage_artifact.verify_stage_artifact",
                return_value=verified,
            ) as verify,
            patch(
                "distributed_runtime.server.resolve_model_metadata_snapshot",
                return_value="/cache/authenticated-tokenizer",
            ) as resolve,
        ):
            tokenizer_snapshot = resolve_execution_tokenizer_snapshot(
                "/cache/root-stage-package",
                package_identity,
            )

        self.assertEqual(tokenizer_snapshot, "/cache/authenticated-tokenizer")
        verify.assert_called_once_with(
            "/cache/root-stage-package",
            expected_package_id="a" * 64,
        )
        resolve.assert_called_once_with(
            "HMellor/tiny-random-LlamaForCausalLM",
            "9408c553e5c189a7dcdc5a5dbd2feb476b061759",
        )

    def test_unpacked_model_keeps_its_existing_tokenizer_snapshot(self) -> None:
        with patch(
            "distributed_runtime.server.resolve_model_metadata_snapshot",
        ) as resolve:
            self.assertEqual(
                resolve_execution_tokenizer_snapshot("/cache/full-model", None),
                "/cache/full-model",
            )
        resolve.assert_not_called()

    def test_server_cli_rejects_known_external_backends(self) -> None:
        for argument, backend in (
            ("--nakshatra-package", "nakshatra"),
            ("--llama-cpp-server", "llama.cpp"),
            ("--ollama-url", "ollama"),
            ("--vllm-endpoint", "vllm"),
        ):
            with self.subTest(argument=argument):
                with self.assertRaisesRegex(
                    ValueError,
                    f"mycellios_native_runtime_forbids_external_backend:{backend}",
                ):
                    parse_server_args([argument, "research-only"])

    def test_remote_recovery_cli_accepts_only_complete_unique_standby_contracts(
        self,
    ) -> None:
        executor_ids = ["1" * 32, "2" * 32]
        document = {
            "schema": "gdlp-recovery-standby-route/1",
            "routeId": "standby-one",
            "firstStage": {"host": "10.0.0.8", "port": 20_001},
            "stageExecutorIds": executor_ids,
        }
        raw = json.dumps(document, separators=(",", ":"), sort_keys=True)
        args = parse_server_args(
            [
                "--first-stage-host",
                "10.0.0.7",
                "--first-stage-port",
                "20000",
                "--return-port",
                "20002",
                "--recovery-max-retries",
                "2",
                "--stage-executor-id",
                executor_ids[0],
                "--stage-executor-id",
                executor_ids[1],
                "--recovery-standby-route",
                raw,
            ]
        )
        routes = parse_remote_recovery_standby_routes(
            args.recovery_standby_route
        )
        self.assertEqual(len(routes), 1)
        self.assertEqual(routes[0].route_id, "standby-one")
        self.assertEqual(routes[0].stage_executor_ids, tuple(executor_ids))

        with self.assertRaisesRegex(ValueError, "routeId values must be unique"):
            parse_remote_recovery_standby_routes([raw, raw])

        same_endpoint = {
            **document,
            "routeId": "standby-two",
        }
        with self.assertRaisesRegex(
            ValueError,
            "first-stage endpoints must be unique",
        ):
            parse_remote_recovery_standby_routes(
                [raw, json.dumps(same_endpoint)]
            )

    def test_remote_recovery_cli_rejects_unknown_fields_and_weak_ids(self) -> None:
        base = {
            "schema": "gdlp-recovery-standby-route/1",
            "routeId": "standby-one",
            "firstStage": {"host": "127.0.0.1", "port": 20_001},
            "stageExecutorIds": ["1" * 32, "2" * 32],
        }
        with self.assertRaisesRegex(ValueError, "unknown or missing fields"):
            parse_remote_recovery_standby_routes(
                [json.dumps({**base, "ignored": True})]
            )
        with self.assertRaisesRegex(ValueError, "32 lowercase hex"):
            parse_remote_recovery_standby_routes(
                [json.dumps({**base, "stageExecutorIds": ["weak"]})]
            )

    def test_remote_recovery_fails_before_model_loading_without_independent_contract(
        self,
    ) -> None:
        primary_ids = ["1" * 32, "2" * 32]
        base_arguments = [
            "--first-stage-host",
            "127.0.0.1",
            "--first-stage-port",
            "20001",
            "--return-port",
            "20002",
            "--recovery-max-retries",
            "1",
            "--stage-executor-id",
            primary_ids[0],
            "--stage-executor-id",
            primary_ids[1],
        ]
        mismatched = {
            "schema": "gdlp-recovery-standby-route/1",
            "routeId": "mismatched",
            "firstStage": {"host": "127.0.0.1", "port": 20_003},
            "stageExecutorIds": [primary_ids[0], "9" * 32],
        }
        with self.assertRaisesRegex(
            ValueError,
            "does not match the configured active route",
        ):
            build_server(
                parse_server_args(
                    [
                        *base_arguments,
                        "--recovery-standby-route",
                        json.dumps(mismatched),
                    ]
                )
            )

        reused = {
            **mismatched,
            "routeId": "reused",
            "firstStage": {"host": "127.0.0.1", "port": 20_001},
            "stageExecutorIds": primary_ids,
        }
        with self.assertRaisesRegex(ValueError, "reuses the primary endpoint"):
            build_server(
                parse_server_args(
                    [
                        *base_arguments,
                        "--recovery-standby-route",
                        json.dumps(reused),
                    ]
                )
            )

    def test_server_rejects_partial_wave_limits_before_model_loading(self) -> None:
        args = parse_server_args(["--sealed-wave-tokens", "1"])
        with self.assertRaisesRegex(ValueError, "must be supplied together"):
            build_server(args)

    def test_paged_slots_are_derived_or_rejected_before_model_loading(self) -> None:
        automatic = parse_server_args(["--paged-kv"])
        with patch(
            "distributed_runtime.server.resolve_model_snapshot",
            side_effect=RuntimeError("model resolution reached"),
        ):
            with self.assertRaisesRegex(RuntimeError, "model resolution reached"):
                build_server(automatic)
        self.assertEqual(automatic.paged_max_active_requests, 36)

        explicit_shortfall = parse_server_args(
            [
                "--paged-kv",
                "--paged-max-active-requests",
                "35",
            ]
        )
        with patch("distributed_runtime.server.resolve_model_snapshot") as resolve:
            with self.assertRaisesRegex(
                ValueError,
                "active sequences, sealed speculative branches and retained sessions",
            ):
                build_server(explicit_shortfall)
        resolve.assert_not_called()

    def test_native_draft_tree_cli_preserves_every_sealed_limit(self) -> None:
        args = parse_server_args(
            [
                "--speculation",
                "draft-tree",
                "--speculative-max-draft-tokens",
                "4",
                "--max-speculative-branches",
                "6",
                "--max-speculative-branch-tokens",
                "32768",
                "--max-speculative-kv-bytes",
                str(384 * 1024 * 1024),
                "--sealed-wave-tokens",
                "5",
            ]
        )
        provider = tree_draft_provider_from_args(args)
        self.assertIsNotNone(provider)
        assert provider is not None
        self.assertEqual(provider.strategy, "ngram-tree")
        self.assertEqual(provider.max_draft_tokens, 4)
        self.assertEqual(provider.max_branches, 6)
        self.assertIsNone(
            tree_draft_provider_from_args(parse_server_args(["--speculation", "ngram"]))
        )

    def test_native_draft_tree_cli_fails_closed_before_model_loading(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires sealed positive"):
            build_server(
                parse_server_args(
                    [
                        "--speculation",
                        "draft-tree",
                        "--sealed-wave-tokens",
                        "5",
                        "--max-prefill-chunk-tokens",
                        "32",
                    ]
                )
            )
        complete_limits = [
            "--speculation",
            "draft-tree",
            "--speculative-max-draft-tokens",
            "4",
            "--max-speculative-branches",
            "4",
            "--max-speculative-branch-tokens",
            "8192",
            "--max-speculative-kv-bytes",
            str(64 * 1024 * 1024),
        ]
        with self.assertRaisesRegex(ValueError, "explicit sealed-wave-tokens"):
            build_server(parse_server_args(complete_limits))
        with self.assertRaisesRegex(ValueError, "must equal draft depth plus one"):
            build_server(
                parse_server_args(
                    [
                        *complete_limits,
                        "--sealed-wave-tokens",
                        "6",
                        "--max-prefill-chunk-tokens",
                        "32",
                    ]
                )
            )

    def test_native_draft_model_wave_limit_fails_before_model_loading(self) -> None:
        common = [
            "--speculation",
            "draft-model",
            "--speculative-max-draft-tokens",
            "4",
            "--draft-model-source",
            "local-draft",
            "--draft-model-artifact-identity",
            f"sha256:{'a' * 64}",
            "--draft-model-parameter-bytes",
            "16",
            "--draft-model-memory-reservation-bytes",
            str(64 * 1024 * 1024),
        ]
        with patch("distributed_runtime.server.resolve_model_snapshot") as resolve:
            with self.assertRaisesRegex(ValueError, "explicit sealed-wave-tokens"):
                build_server(parse_server_args(common))
        resolve.assert_not_called()

        with patch("distributed_runtime.server.resolve_model_snapshot") as resolve:
            with self.assertRaisesRegex(
                ValueError,
                "cannot be smaller than the VERIFY input",
            ):
                build_server(
                    parse_server_args(
                        [
                            *common,
                            "--sealed-wave-tokens",
                            "1",
                            "--max-prefill-chunk-tokens",
                            "32",
                        ]
                    )
                )
        resolve.assert_not_called()

        with patch("distributed_runtime.server.resolve_model_snapshot") as resolve:
            with self.assertRaisesRegex(
                ValueError,
                "must equal draft depth plus one",
            ):
                build_server(
                    parse_server_args(
                        [
                            *common,
                            "--sealed-wave-tokens",
                            "6",
                            "--max-prefill-chunk-tokens",
                            "32",
                        ]
                    )
                )
        resolve.assert_not_called()

    def test_server_rejects_invalid_prefill_pipeline_credits_before_model_loading(self) -> None:
        for arguments, message in (
            (["--prefill-inflight-chunks", "0"], "between 1 and 64"),
            (["--prefill-inflight-chunks", "65"], "between 1 and 64"),
            (["--prefill-inflight-bytes", "-1"], "between 0 and 1 GiB"),
            (
                ["--prefill-inflight-bytes", str(1024 * 1024 * 1024 + 1)],
                "between 0 and 1 GiB",
            ),
        ):
            with self.subTest(arguments=arguments):
                args = parse_server_args(arguments)
                with self.assertRaisesRegex(ValueError, message):
                    build_server(args)

    def test_speculative_conveyor_cli_is_bounded_and_opt_in(self) -> None:
        defaults = parse_server_args([])
        self.assertEqual(defaults.speculative_inflight_waves, 1)
        self.assertEqual(defaults.speculative_inflight_bytes, 0)

        invalid_cases = (
            (["--speculative-inflight-waves", "0"], "between 1 and 16"),
            (["--speculative-inflight-waves", "17"], "between 1 and 16"),
            (["--speculative-inflight-bytes", "-1"], "between 0 and 1 GiB"),
            (
                [
                    "--speculative-inflight-bytes",
                    str(1024 * 1024 * 1024 + 1),
                ],
                "between 0 and 1 GiB",
            ),
            (
                ["--speculative-inflight-waves", "2"],
                "more than one in-flight wave and a positive byte ceiling",
            ),
            (
                ["--speculative-inflight-bytes", "4096"],
                "more than one in-flight wave and a positive byte ceiling",
            ),
            (
                [
                    "--speculative-inflight-waves",
                    "2",
                    "--speculative-inflight-bytes",
                    "4096",
                ],
                "requires linear ngram or draft-model speculation",
            ),
            (
                [
                    "--speculation",
                    "ngram",
                    "--speculative-inflight-waves",
                    "2",
                    "--speculative-inflight-bytes",
                    "4096",
                ],
                "requires max-active-sequences=1",
            ),
        )
        for arguments, message in invalid_cases:
            with self.subTest(arguments=arguments):
                with patch("distributed_runtime.server.resolve_model_snapshot") as resolve:
                    with self.assertRaisesRegex(ValueError, message):
                        build_server(parse_server_args(arguments))
                resolve.assert_not_called()

        valid = parse_server_args(
            [
                "--max-batch-size",
                "1",
                "--max-active-sequences",
                "1",
                "--speculation",
                "ngram",
                "--speculative-inflight-waves",
                "3",
                "--speculative-inflight-bytes",
                str(64 * 1024 * 1024),
            ]
        )
        with patch(
            "distributed_runtime.server.resolve_model_snapshot",
            side_effect=RuntimeError("model resolution reached"),
        ):
            with self.assertRaisesRegex(RuntimeError, "model resolution reached"):
                build_server(valid)

    def test_server_rejects_partial_or_unbounded_tree_limits_before_model_loading(self) -> None:
        for arguments, message in (
            (["--max-speculative-branches", "1"], "must all be zero"),
            (
                [
                    "--max-speculative-branches",
                    "65",
                    "--max-speculative-branch-tokens",
                    "1",
                    "--max-speculative-kv-bytes",
                    "1",
                ],
                "max-speculative-branches must be between 0 and 64",
            ),
            (
                [
                    "--max-speculative-branches",
                    "1",
                    "--max-speculative-branch-tokens",
                    "1048577",
                    "--max-speculative-kv-bytes",
                    "1",
                ],
                "max-speculative-branch-tokens must be between 0 and 1048576",
            ),
            (
                [
                    "--max-speculative-branches",
                    "1",
                    "--max-speculative-branch-tokens",
                    "1",
                    "--max-speculative-kv-bytes",
                    str((1 << 40) + 1),
                ],
                f"max-speculative-kv-bytes must be between 0 and {1 << 40}",
            ),
        ):
            with self.subTest(arguments=arguments):
                with self.assertRaisesRegex(ValueError, message):
                    build_server(parse_server_args(arguments))

    def test_balanced_and_explicit_boundaries(self) -> None:
        self.assertEqual(balanced_boundaries(30, 1), (0, 30))
        self.assertEqual(balanced_boundaries(30, 4), (0, 8, 15, 22, 30))
        self.assertEqual(parse_boundaries("0,30", 30), (0, 30))
        self.assertEqual(parse_boundaries("0,9,30", 30), (0, 9, 30))
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            parse_boundaries("0,10,10,30", 30)
        with self.assertRaisesRegex(ValueError, "between 1"):
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


class OutputTokenDigestTests(unittest.TestCase):
    def test_digest_is_domain_separated_counted_and_stable(self) -> None:
        self.assertEqual(
            output_token_ids_sha256([10, 11, 99]),
            "sha256:75f9dc144ab4bc0c4a03e57a78e590d6d970c8624f16b98753e3c745673df4f6",
        )
        self.assertNotEqual(
            output_token_ids_sha256([10, 11]),
            output_token_ids_sha256([10, 11, 0]),
        )

    def test_digest_rejects_values_that_are_not_uint32(self) -> None:
        for token_ids in ([-1], [2**32], [True]):
            with self.subTest(token_ids=token_ids), self.assertRaisesRegex(
                ValueError, "uint32"
            ):
                output_token_ids_sha256(token_ids)

    def test_final_stream_chunk_can_carry_real_usage_and_sealed_metrics(self) -> None:
        usage = {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
        metrics = {
            "ttft_ms": 4.0,
            "tpot_ms": 2.0,
            "pipeline_ms": 6.0,
            "output_token_ids_sha256": output_token_ids_sha256([10, 11]),
            "output_token_ids_hash_scheme": OUTPUT_TOKEN_HASH_SCHEME,
        }
        payload = chunk_payload(
            "chatcmpl-test",
            123,
            "distributed-small",
            "",
            finish_reason="length",
            usage=usage,
            distribution_metrics=metrics,
        )
        self.assertEqual(payload["usage"], usage)
        self.assertEqual(payload["distribution_metrics"], metrics)


class HealthStatusCodeTests(unittest.IsolatedAsyncioTestCase):
    """Un motor muerto tiene que decirlo en el CÓDIGO, no sólo en el cuerpo.

    Antes, `/health` devolvía siempre 200 y metía la palabra `degraded` dentro
    del JSON. Todo supervisor, balanceador y sonda mira el código de estado, así
    que un motor con error fatal —que no va a servir ni una petición más— pasaba
    por sano. Y en el despliegue real (Salad) la recuperación automática está
    prohibida, así que nadie se enteraba nunca: outage silencioso.
    """

    async def _health(self, engine):
        server = DistributedMycelliosServer(
            engine,
            _HttpTokenizer(),
            public_model_name="distributed-test",
            max_batch_size=2,
            batch_window_ms=0,
            max_output_tokens=8,
        )
        async with TestClient(TestServer(server.create_app())) as client:
            response = await client.get("/health")
            return response.status, await response.json()

    async def test_a_healthy_engine_answers_200_ready(self) -> None:
        status, body = await self._health(_HttpEngine())
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "ready")

    async def test_a_dead_engine_answers_503_not_200(self) -> None:
        engine = _HttpEngine()
        engine.closed = True          # `healthy` pasa a False
        status, body = await self._health(engine)
        self.assertEqual(body["status"], "degraded")
        self.assertEqual(
            status, 503,
            "un motor degradado devolvía 200: el outage era invisible para "
            "cualquier supervisor que mire el código de estado",
        )

    async def test_a_recovering_engine_stays_200(self) -> None:
        # Recuperarse NO es estar caído: el motor está trabajando en volver, y
        # sacarlo del balanceador por eso alargaría el corte en vez de acortarlo.
        engine = _HttpEngine()
        engine.closed = True
        engine.recovery_stats = {"configured": True, "state": "recovering"}
        status, body = await self._health(engine)
        self.assertEqual(body["status"], "recovering")
        self.assertEqual(status, 200)


class HttpInferenceEvidenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_http_health_non_stream_and_sse_expose_sealed_evidence(self) -> None:
        engine = _HttpEngine()
        server = DistributedMycelliosServer(
            engine,
            _HttpTokenizer(),
            public_model_name="distributed-test",
            max_batch_size=2,
            batch_window_ms=0,
            max_output_tokens=8,
        )

        async with TestClient(TestServer(server.create_app())) as client:
            health_response = await client.get("/health")
            self.assertEqual(health_response.status, 200)
            health = await health_response.json()
            self.assertEqual(
                {
                    key: health[key]
                    for key in (
                        "artifact_identity",
                        "canonical_model_source",
                        "canonical_model_revision",
                        "pipeline_snapshot_identity",
                    )
                },
                {
                    "artifact_identity": _HttpEngine.ARTIFACT_IDENTITY,
                    "canonical_model_source": "hf://example/distributed-test",
                    "canonical_model_revision": "0123456789abcdef",
                    "pipeline_snapshot_identity": str(_HttpEngine.PIPELINE_ID),
                },
            )
            self.assertEqual(health["status"], "ready")
            self.assertEqual(health["prefill_inflight_chunks"], 3)
            self.assertEqual(health["prefill_inflight_bytes"], 4096)
            self.assertEqual(health["max_speculative_branches"], 0)
            self.assertEqual(health["max_speculative_branch_tokens"], 0)
            self.assertEqual(health["max_speculative_kv_bytes"], 0)
            self.assertEqual(health["speculative_inflight_waves"], 3)
            self.assertEqual(health["speculative_inflight_bytes"], 8192)
            self.assertEqual(
                health["speculative_window"]["high_water_waves"],
                2,
            )
            self.assertEqual(health["prefill_window"]["high_water_chunks"], 2)

            non_stream_response = await client.post(
                "/v1/chat/completions",
                json=_http_request(stream=False),
            )
            self.assertEqual(non_stream_response.status, 200)
            non_stream = await non_stream_response.json()
            self._assert_completion_evidence(non_stream)
            self.assertEqual(non_stream["choices"][0]["message"]["content"], "hello")

            stream_response = await client.post(
                "/v1/chat/completions",
                json=_http_request(stream=True),
            )
            self.assertEqual(stream_response.status, 200)
            events, done = await _read_sse(stream_response)
            self.assertTrue(done)
            final = next(
                event
                for event in events
                if event.get("choices", [{}])[0].get("finish_reason") == "length"
            )
            self._assert_completion_evidence(final)

        self.assertTrue(engine.closed)

    async def test_real_http_mismatch_fails_closed_for_non_stream_and_sse(self) -> None:
        engine = _HttpEngine(emitted_token_ids=(10,), output_token_ids=(10, 11))
        server = DistributedMycelliosServer(
            engine,
            _HttpTokenizer(),
            public_model_name="distributed-test",
            max_batch_size=2,
            batch_window_ms=0,
            max_output_tokens=8,
        )

        async with TestClient(TestServer(server.create_app())) as client:
            non_stream_response = await client.post(
                "/v1/chat/completions",
                json=_http_request(stream=False),
            )
            self.assertEqual(non_stream_response.status, 500)
            non_stream = await non_stream_response.json()
            self.assertEqual(non_stream["error"]["type"], "pipeline_evidence_error")
            self.assertNotIn("usage", non_stream)
            self.assertNotIn("distribution_metrics", non_stream)

            stream_response = await client.post(
                "/v1/chat/completions",
                json=_http_request(stream=True),
            )
            self.assertEqual(stream_response.status, 200)
            events, done = await _read_sse(stream_response)
            self.assertTrue(done)
            evidence_error = next(event["error"] for event in events if "error" in event)
            self.assertEqual(evidence_error["type"], "pipeline_evidence_error")
            self.assertFalse(any("usage" in event for event in events))
            self.assertFalse(any("distribution_metrics" in event for event in events))

    def _assert_completion_evidence(self, document) -> None:
        self.assertEqual(
            document["usage"],
            {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        )
        self.assertEqual(
            document["distribution_metrics"],
            {
                "ttft_ms": 4.0,
                "tpot_ms": 2.0,
                "pipeline_ms": 6.0,
                "reused_kv_tokens": 0,
                "output_token_ids_sha256": output_token_ids_sha256([10, 11]),
                "output_token_ids_hash_scheme": OUTPUT_TOKEN_HASH_SCHEME,
            },
        )


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
        self.server = DistributedMycelliosServer(
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


class _HttpEngine:
    ARTIFACT_IDENTITY = "sha256:" + "ab" * 32
    PIPELINE_ID = 2**63 + 123

    def __init__(
        self,
        *,
        emitted_token_ids: tuple[int, ...] = (10, 11),
        output_token_ids: tuple[int, ...] = (10, 11),
    ) -> None:
        self.emitted_token_ids = emitted_token_ids
        self.output_token_ids = output_token_ids
        self.closed = False
        self.maximum_context = 128
        self.config = SimpleNamespace(
            boundaries=(0, 2, 4),
            codec=TensorCodec.FP16,
            prefill_chunk_tokens=16,
            prefill_inflight_chunks=3,
            prefill_inflight_bytes=4096,
            max_speculative_branches=0,
            max_speculative_branch_tokens=0,
            max_speculative_kv_bytes=0,
            speculative_inflight_waves=3,
            speculative_inflight_bytes=8192,
            sealed_wave_token_limit=4,
            prefill_token_limit=64,
        )
        self.model_artifact = SimpleNamespace(
            identity=self.ARTIFACT_IDENTITY,
            canonical_source="hf://example/distributed-test",
            canonical_revision="0123456789abcdef",
        )
        self.pipeline_id = self.PIPELINE_ID
        self.speculation_stats = {"configured": False}
        self.speculative_window_stats = {
            "configured": True,
            "configured_waves_per_request": 3,
            "configured_bytes_per_request": 8192,
            "high_water_waves": 2,
        }
        self.prefill_window_stats = {
            "configured_chunks": 3,
            "configured_bytes": 4096,
            "current_chunks": 0,
            "current_bytes": 0,
            "current_reserved_bytes": 0,
            "high_water_chunks": 2,
            "high_water_bytes": 2048,
            "high_water_reserved_bytes": 2048,
            "dispatched_chunks": 2,
            "completed_chunks": 2,
            "acknowledged_chunks": 2,
        }
        self.root_batch_stats = {"active": 0}
        self.root_parameter_bytes = 1_024

    @property
    def healthy(self) -> bool:
        return not self.closed

    @property
    def fatal_error(self) -> None:
        return None

    @property
    def stages(self) -> int:
        return len(self.config.boundaries) - 1

    def submit(self, requests, callback):
        futures = []
        for request in requests:
            for step, token_id in enumerate(self.emitted_token_ids):
                callback(request.client_id, token_id, step, 1.0 + step)
            future = Future()
            future.set_result(
                GenerationOutput(
                    request.client_id,
                    self.output_token_ids,
                    "length",
                    4.0,
                    2.0,
                    6.0,
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


class _HttpTokenizer(_TemplateTokenizer):
    all_special_ids: tuple[int, ...] = ()

    def decode(self, token_ids, **_: object) -> str:
        return {
            (10,): "hel",
            (10, 11): "hello",
        }[tuple(token_ids)]


def _http_request(*, stream: bool) -> dict[str, object]:
    return {
        "model": "distributed-test",
        "messages": [{"role": "user", "content": "say hello"}],
        "temperature": 0,
        "max_tokens": 2,
        "stream": stream,
    }


async def _read_sse(response) -> tuple[list[dict[str, object]], bool]:
    events: list[dict[str, object]] = []
    done = False
    for line in (await response.text()).splitlines():
        if not line.startswith("data: "):
            continue
        payload = line.removeprefix("data: ")
        if payload == "[DONE]":
            done = True
        else:
            events.append(json.loads(payload))
    return events, done


if __name__ == "__main__":
    unittest.main()
