from __future__ import annotations

import json
import multiprocessing as mp
import os
import platform
import socket
import sys
import threading
import unittest

import torch
import transformers
from transformers import AutoConfig

from distributed_runtime.engine import (
    DistributedPipelineEngine,
    GenerationInput,
    PipelineEngineConfig,
    balanced_boundaries,
)
from distributed_runtime.model import load_tokenizer, reference_generate, resolve_model_snapshot
from distributed_runtime.protocol import TensorCodec
from distributed_runtime.recovery import (
    RecoveringPipelineEngine,
    RemoteRecoveryStandbyEngineFactory,
    RemoteRecoveryStandbyRoute,
)
from distributed_runtime.stage import (
    StageModelSpec,
    StageProcessConfig,
    run_stage_process,
)


@unittest.skipUnless(
    os.environ.get("RUN_PHYSICAL_RECOVERY_TESTS") == "1",
    "set RUN_PHYSICAL_RECOVERY_TESTS=1 to kill and recover a physical stage",
)
class PhysicalMidstreamRecoveryTests(unittest.TestCase):
    MODEL_NAME = os.environ.get(
        "DISTRIBUTED_TEST_MODEL",
        "HuggingFaceTB/SmolLM2-135M-Instruct",
    )

    def test_killed_child_is_recreated_and_greedy_stream_remains_exact(self) -> None:
        snapshot = resolve_model_snapshot(self.MODEL_NAME)
        model_config = AutoConfig.from_pretrained(snapshot)
        total_layers = int(model_config.num_hidden_layers)
        config = PipelineEngineConfig(
            model_name=snapshot,
            boundaries=balanced_boundaries(total_layers, 2),
            codec=TensorCodec.FP32,
            threads_per_stage=2,
            max_active_sequences=2,
            max_pending_requests=8,
            prefill_chunk_tokens=2,
            root_batch_window_ms=1,
        )
        tokenizer = load_tokenizer(snapshot)
        input_ids = tokenizer(
            "The capital of France is",
            return_tensors="pt",
        ).input_ids
        output_tokens = 6
        expected = reference_generate(snapshot, input_ids, output_tokens, 2)[0]
        routes: list[DistributedPipelineEngine] = []

        def factory() -> DistributedPipelineEngine:
            route = DistributedPipelineEngine(config)
            routes.append(route)
            return route

        initial = factory()
        engine = RecoveringPipelineEngine(
            factory,
            initial_engine=initial,
            max_retries=1,
        )
        recovery_identity = engine.recovery_identity
        killed = threading.Event()
        observed: list[tuple[int, int]] = []

        def on_token(_client: int, token: int, step: int, _arrived: float) -> None:
            observed.append((step, token))
            if step == 1 and not killed.is_set():
                killed.set()
                # This is deliberately a physical process death, not an injected
                # future exception. The socket EOF must drive route recovery.
                routes[0]._processes[0].terminate()

        try:
            output = engine.generate(
                [GenerationInput(707, input_ids, output_tokens)],
                on_token,
            )[0]
            stats = engine.recovery_stats
        finally:
            engine.close()

        self.assertTrue(killed.is_set())
        self.assertGreaterEqual(len(routes), 2)
        self.assertEqual(list(output.token_ids), expected)
        self.assertEqual(observed, list(enumerate(expected)))
        self.assertEqual(stats["route_recovery_successes"], 1)
        self.assertEqual(stats["recovered_requests"], 1)
        print(
            "GDLP_PHYSICAL_RECOVERY_SMOKE "
            + json.dumps(
                {
                    "schema": "gdlp-physical-recovery-smoke/1",
                    "model": self.MODEL_NAME,
                    "modelRevision": recovery_identity.canonical_model_revision,
                    "modelIdentity": recovery_identity.artifact_identity,
                    "boundaries": list(config.boundaries),
                    "stageExecutorIds": list(recovery_identity.stage_executor_ids),
                    "codec": "fp32",
                    "killedAfterVisibleTokens": 2,
                    "completionTokens": output_tokens,
                    "outputTokenIds": list(output.token_ids),
                    "exact": list(output.token_ids) == expected,
                    "callbacksExact": observed == list(enumerate(expected)),
                    "routeRecoverySuccesses": stats["route_recovery_successes"],
                    "replayedOutputTokens": stats["replayed_output_tokens"],
                    "recoveryMs": stats["last_recovery_duration_ms"],
                    "environment": {
                        "platform": platform.platform(),
                        "python": platform.python_version(),
                        "torch": torch.__version__,
                        "transformers": transformers.__version__,
                        "process": sys.executable,
                        "topology": "single-host root plus one spawned CPU child",
                    },
                    "limits": [
                        "greedy temperature=0 only",
                        "single Windows host; not a LAN or WAN claim",
                        "CPU FP32; no GPU or quantized recovery claim",
                        "one killed child and one recovery attempt",
                        "short six-token equality probe; not a quality corpus",
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    def test_remote_loopback_route_promotes_idle_standby_exactly(self) -> None:
        """Kill one external stage and continue through another external stage.

        Both stages are real spawned processes, but the test remains explicitly
        loopback-only. It does not claim LAN/WAN failover latency.
        """

        snapshot = resolve_model_snapshot(self.MODEL_NAME)
        model_config = AutoConfig.from_pretrained(snapshot)
        total_layers = int(model_config.num_hidden_layers)
        boundaries = balanced_boundaries(total_layers, 2)
        bootstrap_config = PipelineEngineConfig(
            model_name=snapshot,
            boundaries=boundaries,
            codec=TensorCodec.FP32,
            threads_per_stage=2,
            device="cpu",
            max_active_sequences=1,
            max_pending_requests=4,
            prefill_chunk_tokens=2,
        )
        bootstrap = DistributedPipelineEngine(bootstrap_config)
        try:
            identity = bootstrap.recovery_identity
            model_artifact = bootstrap.model_artifact
            pipeline_id = bootstrap.pipeline_id
        finally:
            bootstrap.close()

        primary_port, standby_port, return_port = (
            _reserve_loopback_port(),
            _reserve_loopback_port(),
            _reserve_loopback_port(),
        )
        context = mp.get_context("spawn")
        metrics = context.Queue()
        processes: list[mp.Process] = []
        for name, listen_port in (
            ("remote-primary", primary_port),
            ("remote-standby", standby_port),
        ):
            config = StageProcessConfig(
                spec=StageModelSpec(
                    snapshot,
                    boundaries[1],
                    boundaries[2],
                    total_layers,
                    2,
                    artifact_identity=model_artifact.identity,
                    canonical_model_source=model_artifact.canonical_source,
                    canonical_model_revision=model_artifact.canonical_revision,
                ),
                pipeline_id=pipeline_id,
                listen_host="127.0.0.1",
                listen_port=listen_port,
                next_host=None,
                next_port=None,
                next_layer_end=None,
                return_host="127.0.0.1",
                return_port=return_port,
                codec=TensorCodec.FP32,
                one_way_delay_ms=0,
                bandwidth_mbps=0,
                device="cpu",
                connect_timeout_seconds=60,
            )
            ready = context.Event()
            process = context.Process(
                target=run_stage_process,
                args=(config, ready, metrics),
                name=name,
            )
            process.start()
            self.assertTrue(ready.wait(60), f"{name} never began listening")
            self.assertTrue(process.is_alive(), f"{name} exited before activation")
            processes.append(process)

        common = dict(
            model_name=snapshot,
            boundaries=boundaries,
            codec=TensorCodec.FP32,
            threads_per_stage=2,
            device="cpu",
            spawn_local_stages=False,
            return_bind_host="127.0.0.1",
            return_advertise_host="127.0.0.1",
            return_port=return_port,
            artifact_identity=model_artifact.identity,
            canonical_model_source=model_artifact.canonical_source,
            canonical_model_revision=model_artifact.canonical_revision,
            pipeline_snapshot_identity=pipeline_id,
            stage_executor_ids=identity.stage_executor_ids,
            max_active_sequences=1,
            max_pending_requests=4,
            prefill_chunk_tokens=2,
        )
        primary_config = PipelineEngineConfig(
            **common,
            first_stage_host="127.0.0.1",
            first_stage_port=primary_port,
        )
        standby_config = PipelineEngineConfig(
            **common,
            first_stage_host="127.0.0.1",
            first_stage_port=standby_port,
        )
        tokenizer = load_tokenizer(snapshot)
        input_ids = tokenizer(
            "The capital of France is",
            return_tensors="pt",
        ).input_ids
        output_tokens = 6
        expected = reference_generate(snapshot, input_ids, output_tokens, 2)[0]
        initial = DistributedPipelineEngine(primary_config)
        route = RemoteRecoveryStandbyRoute(
            route_id="physical-loopback-standby",
            first_stage_host="127.0.0.1",
            first_stage_port=standby_port,
            stage_executor_ids=identity.stage_executor_ids,
        )
        engine = RecoveringPipelineEngine(
            lambda: DistributedPipelineEngine(primary_config),
            initial_engine=initial,
            max_retries=1,
            standby_factories=(
                RemoteRecoveryStandbyEngineFactory(
                    route,
                    lambda: DistributedPipelineEngine(standby_config),
                ),
            ),
        )
        observed: list[tuple[int, int]] = []
        killed = threading.Event()

        def on_token(_client: int, token: int, step: int, _arrived: float) -> None:
            observed.append((step, token))
            if step == 1 and not killed.is_set():
                killed.set()
                processes[0].terminate()

        try:
            output = engine.generate(
                [GenerationInput(808, input_ids, output_tokens)],
                on_token,
            )[0]
            stats = engine.recovery_stats
        finally:
            engine.close()
            for process in processes:
                process.join(timeout=10)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
            metrics.close()

        self.assertTrue(killed.is_set())
        self.assertEqual(list(output.token_ids), expected)
        self.assertEqual(observed, list(enumerate(expected)))
        self.assertEqual(stats["active_route_id"], route.route_id)
        self.assertEqual(stats["last_promoted_route_id"], route.route_id)
        self.assertEqual(stats["duplicate_tokens_suppressed"], 2)
        self.assertEqual(stats["checkpoint_kind"], "visible-token-prefix-not-kv")
        print(
            "GDLP_PHYSICAL_REMOTE_RECOVERY_SMOKE "
            + json.dumps(
                {
                    "schema": "gdlp-physical-remote-recovery-smoke/1",
                    "topology": "single-host root plus two external loopback stages",
                    "primaryKilledAfterVisibleTokens": 2,
                    "promotedRouteId": stats["last_promoted_route_id"],
                    "recoveryIdentitySha256": stats["recovery_identity_sha256"],
                    "visiblePrefixSha256": stats["last_visible_prefix_sha256"],
                    "outputTokenIds": list(output.token_ids),
                    "exact": list(output.token_ids) == expected,
                    "callbacksExact": observed == list(enumerate(expected)),
                    "limits": [
                        "greedy temperature=0 only",
                        "loopback only; no LAN or WAN claim",
                        "CPU FP32; no GPU recovery claim",
                        "visible-token prefix recomputation; no KV transfer claim",
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )


def _reserve_loopback_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
    finally:
        sock.close()


if __name__ == "__main__":
    unittest.main()
