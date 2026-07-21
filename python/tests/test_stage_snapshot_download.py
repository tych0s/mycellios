from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from distributed_runtime.model import resolve_stage_model_snapshot


class StageSnapshotDownloadTests(unittest.TestCase):
    def test_indexed_checkpoint_downloads_only_files_needed_by_middle_stage(self) -> None:
        with TemporaryDirectory() as temporary:
            snapshot = Path(temporary)
            embedding_weight = "model.embed_" + "tokens.weight"
            (snapshot / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            embedding_weight: "weights-00001.safetensors",
                            "model.layers.0.self_attn.q_proj.weight": "weights-00001.safetensors",
                            "model.layers.1.self_attn.q_proj.weight": "weights-00002.safetensors",
                            "model.layers.2.self_attn.q_proj.weight": "weights-00003.safetensors",
                            "model.layers.3.self_attn.q_proj.weight": "weights-00004.safetensors",
                            "model.norm.weight": "weights-00004.safetensors",
                            "lm_head.weight": "weights-00004.safetensors",
                        }
                    }
                ),
                encoding="utf-8",
            )
            with patch("distributed_runtime.model.snapshot_download", return_value=str(snapshot)) as download:
                resolved = resolve_stage_model_snapshot(
                    "example/model",
                    revision="commit",
                    layer_start=1,
                    layer_end=2,
                    total_layers=4,
                )

            self.assertEqual(resolved, str(snapshot))
            self.assertEqual(download.call_count, 2)
            requested = download.call_args_list[1].kwargs["allow_patterns"]
            self.assertIn("weights-00002.safetensors", requested)
            self.assertNotIn("weights-00001.safetensors", requested)
            self.assertNotIn("weights-00003.safetensors", requested)
            self.assertNotIn("weights-00004.safetensors", requested)

    def test_last_stage_fetches_projection_and_tied_embedding_sources(self) -> None:
        with TemporaryDirectory() as temporary:
            snapshot = Path(temporary)
            (snapshot / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "model.embed_tokens.weight": "first.safetensors",
                            "model.layers.0.mlp.down_proj.weight": "first.safetensors",
                            "model.layers.1.mlp.down_proj.weight": "last.safetensors",
                            "model.norm.weight": "last.safetensors",
                        }
                    }
                ),
                encoding="utf-8",
            )
            with patch("distributed_runtime.model.snapshot_download", return_value=str(snapshot)) as download:
                resolve_stage_model_snapshot(
                    "example/tied-model",
                    revision=None,
                    layer_start=1,
                    layer_end=2,
                    total_layers=2,
                )

            requested = download.call_args_list[1].kwargs["allow_patterns"]
            self.assertIn("first.safetensors", requested)
            self.assertIn("last.safetensors", requested)


if __name__ == "__main__":
    unittest.main()
