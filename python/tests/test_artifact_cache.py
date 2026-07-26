from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from safetensors.torch import save_file
import torch
from transformers import AutoModelForCausalLM, LlamaConfig

from distributed_runtime.artifact_cache import ContentAddressedStageCache
from distributed_runtime.stage_artifact import (
    STAGE_ARTIFACT_MANIFEST,
    compile_safetensors_stage_artifact,
    verify_stage_artifact,
)


class ArtifactCacheTests(unittest.TestCase):
    def test_blob_transfer_resumes_at_the_confirmed_offset_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = (b"mycellios-resumable-stage-" * 8_192) + b"done"
            digest = hashlib.sha256(value).hexdigest()
            split = len(value) // 3
            cache = ContentAddressedStageCache(root / "cache")
            first = cache.begin_blob(digest, len(value))
            self.assertEqual(
                first.append(
                    0,
                    value[:split],
                    chunk_sha256=hashlib.sha256(value[:split]).hexdigest(),
                ),
                split,
            )

            resumed = ContentAddressedStageCache(root / "cache").begin_blob(
                f"sha256:{digest}",
                len(value),
            )
            self.assertEqual(resumed.confirmed_offset, split)
            resumed.append(
                split,
                value[split:],
                chunk_sha256=hashlib.sha256(value[split:]).hexdigest(),
            )
            final = resumed.finalize()
            self.assertEqual(final.read_bytes(), value)
            self.assertFalse(cache.partial_path(digest).exists())
            self.assertFalse(cache.partial_metadata_path(digest).exists())

    def test_transfer_rejects_corrupt_chunks_offsets_and_final_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = b"sealed-stage-weights"
            digest = hashlib.sha256(value).hexdigest()
            cache = ContentAddressedStageCache(root / "cache")
            transfer = cache.begin_blob(digest, len(value))

            with self.assertRaisesRegex(ValueError, "chunk digest mismatch"):
                transfer.append(0, value[:5], chunk_sha256="0" * 64)
            self.assertEqual(transfer.confirmed_offset, 0)
            with self.assertRaisesRegex(ValueError, "offset mismatch"):
                transfer.append(
                    2,
                    value[:5],
                    chunk_sha256=hashlib.sha256(value[:5]).hexdigest(),
                )
            transfer.append(
                0,
                b"x" * len(value),
                chunk_sha256=hashlib.sha256(b"x" * len(value)).hexdigest(),
            )
            with self.assertRaisesRegex(ValueError, "completed blob digest mismatch"):
                transfer.finalize()
            self.assertFalse(cache.blob_path(digest).exists())

    def test_stage_import_deduplicates_blobs_and_materializes_verified_package(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            _write_checkpoint(checkpoint)
            stage = root / "stage"
            compiled = compile_safetensors_stage_artifact(
                str(checkpoint),
                stage,
                layer_start=1,
                layer_end=3,
            )
            cache = ContentAddressedStageCache(root / "cache")

            first = cache.import_stage(stage)
            second = cache.import_stage(stage)
            self.assertEqual(first, second)
            self.assertEqual(first.package_id, compiled.package_id)
            self.assertEqual(len(cache.cached_packages()), 1)

            materialized = cache.materialize_stage(
                compiled.artifact_identity,
                root / "materialized",
            )
            verified = verify_stage_artifact(
                materialized,
                expected_package_id=compiled.package_id,
            )
            self.assertEqual(verified.manifest.package_id, compiled.package_id)

    def test_manifest_commit_waits_for_every_verified_blob(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            _write_checkpoint(checkpoint)
            stage = root / "stage"
            compiled = compile_safetensors_stage_artifact(
                str(checkpoint),
                stage,
                layer_start=0,
                layer_end=2,
            )
            manifest_bytes = (stage / STAGE_ARTIFACT_MANIFEST).read_bytes()
            manifest = json.loads(manifest_bytes)
            cache = ContentAddressedStageCache(root / "cache")

            with self.assertRaisesRegex(FileNotFoundError, "blob is missing"):
                cache.commit_manifest(manifest_bytes)
            for item in manifest["files"]:
                data = (stage / item["path"]).read_bytes()
                transfer = cache.begin_blob(item["sha256"], item["sizeBytes"])
                transfer.append(
                    0,
                    data,
                    chunk_sha256=hashlib.sha256(data).hexdigest(),
                )
                transfer.finalize()
            cached = cache.commit_manifest(manifest_bytes)
            self.assertEqual(cached.package_id, compiled.package_id)


def _write_checkpoint(root: Path) -> None:
    root.mkdir(parents=True)
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=2,
        tie_word_embeddings=False,
        attention_bias=False,
        mlp_bias=False,
    )
    config.architectures = ["LlamaForCausalLM"]
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    values: dict[str, torch.Tensor] = {}
    for index, (name, parameter) in enumerate(model.state_dict().items()):
        values[name] = torch.full(
            tuple(parameter.shape),
            (index + 1) / 100,
            dtype=torch.float16,
        )
    config.save_pretrained(root)
    save_file(values, str(root / "model.safetensors"))


if __name__ == "__main__":
    unittest.main()
