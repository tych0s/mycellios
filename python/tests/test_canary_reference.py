from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from distributed_runtime.canary_reference import (
    build_reference,
    greedy_reference,
    normalize_messages,
    prompt_token_ids_sha256,
)
from distributed_runtime.model import ModelArtifactReference


class CanaryReferenceTests(unittest.TestCase):
    def test_messages_match_server_developer_normalization(self) -> None:
        self.assertEqual(
            normalize_messages(
                [
                    {"role": "developer", "content": "Sé breve"},
                    {"role": "user", "content": "Hola"},
                ]
            ),
            [
                {"role": "system", "content": "Sé breve"},
                {"role": "user", "content": "Hola"},
            ],
        )
        with self.assertRaises(ValueError):
            normalize_messages([{"role": "user", "content": "x", "extra": True}])

    def test_prompt_digest_is_counted_domain_separated_and_validates_uint32(self) -> None:
        self.assertRegex(prompt_token_ids_sha256([10, 11]), r"^sha256:[0-9a-f]{64}$")
        self.assertNotEqual(prompt_token_ids_sha256([10]), prompt_token_ids_sha256([10, 0]))
        with self.assertRaisesRegex(ValueError, "uint32"):
            prompt_token_ids_sha256([-1])

    def test_greedy_reference_stops_on_eos_and_reports_actual_count(self) -> None:
        model = _FakeAutoregressiveModel([7, 99, 42])
        tokens, reason, metrics = greedy_reference(
            model,
            torch.tensor([[1, 2]], dtype=torch.long),
            max_tokens=8,
            eos_token_ids=frozenset((99,)),
        )
        self.assertEqual(tokens, [7, 99])
        self.assertEqual(reason, "stop")
        self.assertEqual(model.calls, 2)
        self.assertGreaterEqual(metrics["pipelineMs"], metrics["ttftMs"])

    def test_reference_exports_strong_artifact_and_decimal_snapshot_identity(self) -> None:
        artifact = ModelArtifactReference(
            identity=f"sha256:{'a' * 64}",
            canonical_source="hf://example/model",
            canonical_revision="b" * 40,
            snapshot_identity=12_345_678_901_234_567_890,
        )
        model = _FakeAutoregressiveModel([7, 99])
        tokenizer = _FakeTokenizer()
        with tempfile.TemporaryDirectory() as temporary:
            messages = Path(temporary) / "messages.json"
            messages.write_text(
                json.dumps([{"role": "user", "content": "Hola"}]),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                model="example/model",
                revision="b" * 40,
                messages_json=messages,
                max_tokens=2,
                threads=1,
                device="cpu",
                dtype="float32",
                tokenizer_id="example/tokenizer",
            )
            with (
                patch(
                    "distributed_runtime.canary_reference.resolve_model_snapshot",
                    return_value="C:/sealed/snapshot",
                ),
                patch(
                    "distributed_runtime.canary_reference.model_artifact_reference",
                    return_value=artifact,
                ),
                patch(
                    "distributed_runtime.canary_reference.load_tokenizer",
                    return_value=tokenizer,
                ),
                patch(
                    "distributed_runtime.canary_reference.AutoModelForCausalLM.from_pretrained",
                    return_value=model,
                ),
            ):
                reference = build_reference(args)

        self.assertEqual(
            reference["model"],
            {
                "requestedId": "example/model",
                "requestedRevision": "b" * 40,
                "artifactIdentity": f"sha256:{'a' * 64}",
                "canonicalSource": "hf://example/model",
                "canonicalRevision": "b" * 40,
                "snapshotIdentity": "12345678901234567890",
                "tokenizerId": "example/tokenizer",
                "device": "cpu",
                "dtype": "float32",
            },
        )


class _FakeAutoregressiveModel:
    def __init__(self, outputs: list[int]) -> None:
        self.outputs = outputs
        self.calls = 0

    def __call__(self, **_: object):
        token = self.outputs[self.calls]
        self.calls += 1
        logits = torch.full((1, 1, 128), -1_000.0)
        logits[0, 0, token] = 1_000.0
        return SimpleNamespace(logits=logits, past_key_values=(self.calls,))

    def to(self, _: str):
        return self

    def eval(self):
        return self


class _FakeTokenizer:
    eos_token_id = 99

    def apply_chat_template(self, *_: object, **__: object) -> torch.Tensor:
        return torch.tensor([[1, 2]], dtype=torch.long)


if __name__ == "__main__":
    unittest.main()
