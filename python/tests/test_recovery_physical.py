from __future__ import annotations

import json
import os
import platform
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
from distributed_runtime.recovery import RecoveringPipelineEngine


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


if __name__ == "__main__":
    unittest.main()
