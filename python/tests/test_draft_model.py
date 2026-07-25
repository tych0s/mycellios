from __future__ import annotations

import argparse
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from distributed_runtime.draft_model import (
    LOCAL_DRAFT_MODEL_SCHEMA,
    LocalDraftModelProvider,
    LocalDraftModelRuntimeConfig,
    add_local_draft_model_arguments,
    load_local_draft_model_provider,
    local_draft_model_config_from_args,
)
from distributed_runtime.model import ModelArtifactReference


ARTIFACT_IDENTITY = f"sha256:{'a' * 64}"
PARAMETER_BYTES = 16
MEMORY_RESERVATION_BYTES = 64 * 1024 * 1024


class _Tokenizer:
    def __init__(
        self,
        vocabulary: dict[str, int] | None = None,
        *,
        eos_token_id: int = 3,
    ) -> None:
        self._vocabulary = vocabulary or {
            "<bos>": 0,
            "one": 1,
            "two": 2,
            "<eos>": 3,
        }
        self.bos_token_id = 0
        self.eos_token_id = eos_token_id
        self.pad_token_id = 0
        self.unk_token_id = None
        self.all_special_ids = [0, eos_token_id]

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocabulary)


class _GreedyModel:
    def __init__(
        self,
        *,
        vocabulary_size: int = 4,
        context_tokens: int = 4,
        invalid_hole_wins: bool = False,
    ) -> None:
        self.config = SimpleNamespace(
            vocab_size=vocabulary_size,
            max_position_embeddings=context_tokens,
        )
        self.invalid_hole_wins = invalid_hole_wins
        self.to_device: torch.device | None = None
        self.weight = torch.zeros(4, dtype=torch.float32)

    def eval(self) -> "_GreedyModel":
        return self

    def to(self, device: torch.device) -> "_GreedyModel":
        self.to_device = device
        return self

    def parameters(self) -> tuple[torch.Tensor, ...]:
        return (self.weight,)

    def get_output_embeddings(self) -> SimpleNamespace:
        return SimpleNamespace(out_features=self.config.vocab_size)

    def __call__(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
        return_dict: bool,
        past_key_values: object | None = None,
    ) -> SimpleNamespace:
        del attention_mask, use_cache, return_dict, past_key_values
        last = int(input_ids[0, -1].item())
        next_token = (last + 1) % self.config.vocab_size
        logits = torch.full(
            (1, int(input_ids.shape[1]), self.config.vocab_size),
            -10.0,
            dtype=torch.float32,
            device=input_ids.device,
        )
        logits[0, -1, next_token] = 10.0
        if self.invalid_hole_wins:
            logits[0, -1, 1] = 100.0
            logits[0, -1, 2] = 90.0
        return SimpleNamespace(logits=logits, past_key_values=None)


class _FailingModel(_GreedyModel):
    def __call__(self, **_kwargs: object) -> SimpleNamespace:
        raise RuntimeError("synthetic accelerator failure")


class _CachedGreedyModel(_GreedyModel):
    def __init__(self) -> None:
        super().__init__(context_tokens=32)
        self.call_lengths: list[int] = []

    def __call__(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
        return_dict: bool,
        past_key_values: object | None = None,
    ) -> SimpleNamespace:
        del attention_mask, use_cache, return_dict
        self.call_lengths.append(int(input_ids.shape[1]))
        prior = 0
        if isinstance(past_key_values, tuple):
            prior = int(past_key_values[0][0].shape[-2])
        total = prior + int(input_ids.shape[1])
        last = int(input_ids[0, -1].item())
        logits = torch.full(
            (1, int(input_ids.shape[1]), self.config.vocab_size),
            -10.0,
            dtype=torch.float32,
        )
        logits[0, -1, (last + 1) % self.config.vocab_size] = 10.0
        cache = (
            (
                torch.zeros((1, 1, total, 1), dtype=torch.float32),
                torch.zeros((1, 1, total, 1), dtype=torch.float32),
            ),
        )
        return SimpleNamespace(logits=logits, past_key_values=cache)


def _config(**overrides: object) -> LocalDraftModelRuntimeConfig:
    values: dict[str, object] = {
        "source": "local-draft",
        "artifact_identity": ARTIFACT_IDENTITY,
        "canonical_source": f"content-addressed://{ARTIFACT_IDENTITY}",
        "canonical_revision": ARTIFACT_IDENTITY,
        "max_draft_tokens": 3,
        "device": "cpu",
        "dtype": "float32",
        "parameter_bytes": PARAMETER_BYTES,
        "memory_reservation_bytes": MEMORY_RESERVATION_BYTES,
    }
    values.update(overrides)
    return LocalDraftModelRuntimeConfig(**values)  # type: ignore[arg-type]


class LocalDraftModelConfigurationTests(unittest.TestCase):
    def test_cli_is_all_or_nothing_and_binds_depth(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--speculation", default="off")
        parser.add_argument("--speculative-max-draft-tokens", type=int, default=0)
        add_local_draft_model_arguments(parser)

        self.assertIsNone(local_draft_model_config_from_args(parser.parse_args([])))
        with self.assertRaisesRegex(ValueError, "require --speculation"):
            local_draft_model_config_from_args(
                parser.parse_args(["--draft-model-source", "unexpected"])
            )
        with self.assertRaisesRegex(ValueError, "requires source"):
            local_draft_model_config_from_args(
                parser.parse_args(
                    [
                        "--speculation",
                        "draft-model",
                        "--speculative-max-draft-tokens",
                        "3",
                    ]
                )
            )

        config = local_draft_model_config_from_args(
            parser.parse_args(
                [
                    "--speculation",
                    "draft-model",
                    "--speculative-max-draft-tokens",
                    "3",
                    "--draft-model-source",
                    "local-draft",
                    "--draft-model-artifact-identity",
                    ARTIFACT_IDENTITY,
                    "--draft-model-parameter-bytes",
                    str(PARAMETER_BYTES),
                    "--draft-model-memory-reservation-bytes",
                    str(MEMORY_RESERVATION_BYTES),
                    "--draft-model-device",
                    "cpu",
                    "--draft-model-dtype",
                    "float32",
                ]
            )
        )
        self.assertIsNotNone(config)
        assert config is not None
        self.assertEqual(config.max_draft_tokens, 3)
        self.assertEqual(config.schema, LOCAL_DRAFT_MODEL_SCHEMA)
        self.assertEqual(config.parameter_bytes, PARAMETER_BYTES)
        self.assertEqual(
            config.memory_reservation_bytes,
            MEMORY_RESERVATION_BYTES,
        )

    def test_configuration_rejects_weak_identity_and_cpu_float16(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be sha256"):
            _config(artifact_identity="weak")
        with self.assertRaisesRegex(ValueError, "not supported on CPU"):
            _config(dtype="float16")
        with self.assertRaisesRegex(ValueError, "positive integer"):
            _config(parameter_bytes=0)
        with self.assertRaisesRegex(ValueError, "cannot be smaller"):
            _config(
                parameter_bytes=MEMORY_RESERVATION_BYTES + 1,
                memory_reservation_bytes=MEMORY_RESERVATION_BYTES,
            )


class LocalDraftModelProviderTests(unittest.TestCase):
    def test_greedy_draft_is_bounded_exact_vocab_and_reports_stats(self) -> None:
        provider = LocalDraftModelProvider(
            _config(),
            target_tokenizer=_Tokenizer(),
            draft_tokenizer=_Tokenizer(),
            model=_GreedyModel(),
            actual_artifact_identity=ARTIFACT_IDENTITY,
            resolved_device=torch.device("cpu"),
        )

        self.assertEqual(provider.draft((0, 1), max_tokens=3), (2, 3))
        self.assertEqual(provider.draft((0, 1, 2, 0, 1), max_tokens=1), (2,))
        stats = provider.stats()
        self.assertEqual(stats.draft_calls, 2)
        self.assertEqual(stats.draft_failures, 0)
        self.assertEqual(stats.drafted_tokens, 3)
        self.assertEqual(stats.truncated_context_calls, 2)
        snapshot = provider.execution_snapshot()
        self.assertEqual(snapshot["artifactIdentity"], ARTIFACT_IDENTITY)
        self.assertEqual(snapshot["verificationAuthority"], "distributed-target-model")
        self.assertEqual(snapshot["draftCalls"], 2)
        self.assertEqual(snapshot["resolvedDtype"], "float32")
        self.assertFalse(snapshot["circuitOpen"])
        self.assertGreaterEqual(snapshot["draftTimeNs"], 0)

    def test_sparse_vocabulary_masks_invalid_logit_ids(self) -> None:
        vocabulary = {"<bos>": 0, "two": 2, "<eos>": 3}
        provider = LocalDraftModelProvider(
            _config(max_draft_tokens=1),
            target_tokenizer=_Tokenizer(vocabulary),
            draft_tokenizer=_Tokenizer(vocabulary),
            model=_GreedyModel(invalid_hole_wins=True),
            actual_artifact_identity=ARTIFACT_IDENTITY,
            resolved_device=torch.device("cpu"),
        )
        self.assertEqual(provider.draft((0,), max_tokens=1), (2,))
        with self.assertRaisesRegex(ValueError, "outside the shared vocabulary"):
            provider.draft((1,), max_tokens=1)

    def test_context_limit_reduces_effective_depth(self) -> None:
        provider = LocalDraftModelProvider(
            _config(max_draft_tokens=3),
            target_tokenizer=_Tokenizer(),
            draft_tokenizer=_Tokenizer(),
            model=_GreedyModel(context_tokens=2),
            actual_artifact_identity=ARTIFACT_IDENTITY,
            resolved_device=torch.device("cpu"),
        )
        self.assertEqual(provider.max_draft_tokens, 1)
        self.assertEqual(provider.draft((0, 1), max_tokens=3), (2,))
        self.assertEqual(provider.execution_snapshot()["configuredMaxDraftTokens"], 3)

    def test_inference_failure_opens_circuit_and_preserves_exact_fallback(self) -> None:
        provider = LocalDraftModelProvider(
            _config(),
            target_tokenizer=_Tokenizer(),
            draft_tokenizer=_Tokenizer(),
            model=_FailingModel(),
            actual_artifact_identity=ARTIFACT_IDENTITY,
            resolved_device=torch.device("cpu"),
        )

        self.assertEqual(provider.draft((0, 1), max_tokens=2), ())
        self.assertEqual(provider.draft((0, 1), max_tokens=2), ())
        stats = provider.stats()
        self.assertEqual(stats.draft_calls, 1)
        self.assertEqual(stats.draft_failures, 1)
        self.assertEqual(stats.bypassed_calls, 1)
        self.assertTrue(stats.circuit_open)
        self.assertEqual(stats.circuit_reason, "RuntimeError")

    def test_request_cache_reuses_only_an_exact_verified_prefix(self) -> None:
        model = _CachedGreedyModel()
        provider = LocalDraftModelProvider(
            _config(max_draft_tokens=2),
            target_tokenizer=_Tokenizer(),
            draft_tokenizer=_Tokenizer(),
            model=model,
            actual_artifact_identity=ARTIFACT_IDENTITY,
            resolved_device=torch.device("cpu"),
            max_cached_requests=2,
        )

        self.assertEqual(
            provider.draft_for_request(7, (0, 1), max_tokens=2),
            (2, 3),
        )
        self.assertEqual(model.call_lengths, [2, 1])
        self.assertEqual(
            provider.draft_for_request(7, (0, 1, 2, 3, 0), max_tokens=1),
            (1,),
        )
        self.assertEqual(model.call_lengths, [2, 1, 2])
        stats = provider.stats()
        self.assertEqual(stats.cache_misses, 1)
        self.assertEqual(stats.cache_hits, 1)
        self.assertGreater(stats.cached_kv_bytes, 0)

        provider.release_request(7)
        self.assertEqual(provider.stats().cached_kv_bytes, 0)

    def test_incompatible_tokenizer_or_artifact_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "artifact identity"):
            LocalDraftModelProvider(
                _config(),
                target_tokenizer=_Tokenizer(),
                draft_tokenizer=_Tokenizer(),
                model=_GreedyModel(),
                actual_artifact_identity=f"sha256:{'b' * 64}",
                resolved_device=torch.device("cpu"),
            )
        with self.assertRaisesRegex(ValueError, "vocabulary is not identical"):
            LocalDraftModelProvider(
                _config(),
                target_tokenizer=_Tokenizer(),
                draft_tokenizer=_Tokenizer({"different": 0, "one": 1, "two": 2, "<eos>": 3}),
                model=_GreedyModel(),
                actual_artifact_identity=ARTIFACT_IDENTITY,
                resolved_device=torch.device("cpu"),
            )
        with self.assertRaisesRegex(ValueError, "special-token ids"):
            LocalDraftModelProvider(
                _config(),
                target_tokenizer=_Tokenizer(),
                draft_tokenizer=_Tokenizer(eos_token_id=2),
                model=_GreedyModel(),
                actual_artifact_identity=ARTIFACT_IDENTITY,
                resolved_device=torch.device("cpu"),
            )

    def test_loader_authenticates_snapshot_before_exposing_provider(self) -> None:
        reference = ModelArtifactReference(
            identity=ARTIFACT_IDENTITY,
            canonical_source=f"content-addressed://{ARTIFACT_IDENTITY}",
            canonical_revision=ARTIFACT_IDENTITY,
            snapshot_identity=int("a" * 16, 16),
        )
        model = _GreedyModel()
        with (
            patch(
                "distributed_runtime.draft_model.model_artifact_reference",
                return_value=reference,
            ),
            patch(
                "distributed_runtime.model.resolve_model_snapshot",
                return_value="resolved-draft",
            ),
            patch(
                "distributed_runtime.draft_model.AutoTokenizer.from_pretrained",
                return_value=_Tokenizer(),
            ) as tokenizer_loader,
            patch(
                "distributed_runtime.draft_model.AutoModelForCausalLM.from_pretrained",
                return_value=model,
            ) as model_loader,
            patch("distributed_runtime.draft_model._ensure_memory_reservation"),
        ):
            provider = load_local_draft_model_provider(
                _config(),
                target_tokenizer=_Tokenizer(),
            )

        self.assertEqual(provider.config.artifact_identity, ARTIFACT_IDENTITY)
        tokenizer_loader.assert_called_once_with(
            "resolved-draft",
            local_files_only=True,
            trust_remote_code=False,
        )
        model_loader.assert_called_once_with(
            "resolved-draft",
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.float32,
        )
        self.assertEqual(model.to_device, torch.device("cpu"))

    def test_loader_rejects_capacity_before_transformers_load(self) -> None:
        reference = ModelArtifactReference(
            identity=ARTIFACT_IDENTITY,
            canonical_source=f"content-addressed://{ARTIFACT_IDENTITY}",
            canonical_revision=ARTIFACT_IDENTITY,
            snapshot_identity=int("a" * 16, 16),
        )
        with (
            patch(
                "distributed_runtime.draft_model.model_artifact_reference",
                return_value=reference,
            ),
            patch(
                "distributed_runtime.model.resolve_model_snapshot",
                return_value="resolved-draft",
            ),
            patch(
                "distributed_runtime.draft_model._ensure_memory_reservation",
                side_effect=RuntimeError("insufficient memory"),
            ),
            patch(
                "distributed_runtime.draft_model.AutoTokenizer.from_pretrained",
            ) as tokenizer_loader,
            patch(
                "distributed_runtime.draft_model.AutoModelForCausalLM.from_pretrained",
            ) as model_loader,
        ):
            with self.assertRaisesRegex(RuntimeError, "insufficient memory"):
                load_local_draft_model_provider(
                    _config(),
                    target_tokenizer=_Tokenizer(),
                )
        tokenizer_loader.assert_not_called()
        model_loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
