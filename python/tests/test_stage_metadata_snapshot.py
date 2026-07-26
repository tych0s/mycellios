"""Un nodo con SÓLO metadatos debe declarar la misma identidad que uno completo.

Esto es lo que desbloquea «un modelo mayor que cualquier host». La descarga
selectiva por capa estaba escrita y probada desde hacía tiempo, pero no se
ejecutaba nunca: `stage_cli` pre-resolvía el checkpoint COMPLETO para calcular
la identidad del artefacto, y al pasarle al `StageModelSpec` una ruta local ya
descargada, `resolve_stage_model_snapshot` hacía cortocircuito (devuelve tal
cual cualquier directorio). Cada nodo se traía el modelo entero.

El arreglo separa las dos cosas: los metadatos (kilobytes) bastan para la
identidad, y los pesos los pide el cargador selectivo capa a capa. Sólo funciona
si se cumple la invariante que fijan estas pruebas: **la identidad de un
snapshot del Hub sale del commit de la ruta de caché, no de los bytes.** Si eso
dejara de ser cierto, dos etapas con shards distintos calcularían identidades
distintas y el pipeline se rechazaría a sí mismo al arrancar.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from distributed_runtime.model import (
    _METADATA_PATTERNS,
    model_artifact_reference,
    resolve_model_metadata_snapshot,
)


def _hub_cache_snapshot(root: Path, commit: str) -> Path:
    """Reproduce el layout de caché del Hub: .../snapshots/<commit>/."""
    snapshot = root / "models--example--model" / "snapshots" / commit
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(
        json.dumps({"model_type": "llama", "num_hidden_layers": 4}), encoding="utf-8"
    )
    return snapshot


class MetadataSnapshotTests(unittest.TestCase):
    def test_requests_metadata_only_and_never_the_weights(self) -> None:
        with TemporaryDirectory() as temporary:
            with patch(
                "distributed_runtime.model.snapshot_download",
                return_value=temporary,
            ) as download:
                resolve_model_metadata_snapshot("example/model", "commit")
            patterns = download.call_args.kwargs["allow_patterns"]
            self.assertNotIn("*.safetensors", patterns)
            self.assertIn("config.json", patterns)
            self.assertIn("*.safetensors.index.json", patterns)
            self.assertEqual(patterns, list(_METADATA_PATTERNS))

    def test_a_local_directory_is_returned_untouched(self) -> None:
        with TemporaryDirectory() as temporary:
            with patch("distributed_runtime.model.snapshot_download") as download:
                resolved = resolve_model_metadata_snapshot(temporary)
            self.assertEqual(Path(resolved), Path(temporary).resolve())
            download.assert_not_called()

    def test_a_missing_absolute_path_fails_loudly(self) -> None:
        with self.assertRaises(FileNotFoundError):
            resolve_model_metadata_snapshot(str(Path("/no/such/checkpoint").resolve()))


class IdentityIsIndependentOfDownloadedWeightsTests(unittest.TestCase):
    """La invariante que sostiene todo el arreglo."""

    def test_metadata_only_node_agrees_with_a_fully_downloaded_node(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            commit = "0123456789abcdef0123456789abcdef01234567"

            # Nodo A: sólo metadatos.
            thin = _hub_cache_snapshot(root / "thin", commit)

            # Nodo B: el mismo commit, pero además con pesos en disco.
            fat = _hub_cache_snapshot(root / "fat", commit)
            (fat / "model-00001-of-00002.safetensors").write_bytes(b"\x01" * 4096)
            (fat / "model-00002-of-00002.safetensors").write_bytes(b"\x02" * 8192)

            # Nodo C: mismo commit, otros shards (una etapa distinta).
            other = _hub_cache_snapshot(root / "other", commit)
            (other / "model-00002-of-00002.safetensors").write_bytes(b"\x02" * 8192)

            references = [model_artifact_reference(str(path)) for path in (thin, fat, other)]
            identities = {reference.identity for reference in references}
            self.assertEqual(
                len(identities), 1,
                "tres nodos del mismo commit con shards distintos deben declarar "
                "la MISMA identidad; si no, el pipeline se rechaza a sí mismo",
            )
            snapshot_ids = {reference.snapshot_identity for reference in references}
            self.assertEqual(len(snapshot_ids), 1)
            self.assertEqual({r.canonical_revision for r in references}, {commit})

    def test_a_different_commit_gives_a_different_identity(self) -> None:
        # El control: la identidad tiene que seguir distinguiendo modelos.
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = _hub_cache_snapshot(root / "a", "1" * 40)
            second = _hub_cache_snapshot(root / "b", "2" * 40)
            self.assertNotEqual(
                model_artifact_reference(str(first)).identity,
                model_artifact_reference(str(second)).identity,
            )


class StageCliUsesTheMetadataResolverTests(unittest.TestCase):
    def test_stage_cli_no_longer_pre_downloads_the_whole_checkpoint(self) -> None:
        import inspect

        from distributed_runtime import stage_cli

        source = inspect.getsource(stage_cli)
        self.assertIn("resolve_model_metadata_snapshot(args.model", source)
        self.assertNotIn(
            "resolve_model_snapshot(args.model",
            source,
            "stage_cli volvería a descargar el checkpoint entero y a "
            "cortocircuitar la carga selectiva",
        )


if __name__ == "__main__":
    unittest.main()
