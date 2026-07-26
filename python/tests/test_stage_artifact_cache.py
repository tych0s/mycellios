from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, redirect_stdout
import io
import json
from pathlib import Path
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from safetensors.torch import save_file
import torch
from transformers import AutoModelForCausalLM, LlamaConfig

from distributed_runtime.model import StageModelSpec, StageRunner
from distributed_runtime.stage_artifact import (
    STAGE_ARTIFACT_CONFIG,
    STAGE_ARTIFACT_MANIFEST,
    STAGE_ARTIFACT_WEIGHTS,
    compile_safetensors_stage_artifact,
    verify_stage_artifact,
)
from distributed_runtime.stage_artifact_cache import (
    StageArtifactCache,
    StageArtifactCacheLimits,
    StageArtifactIntegrityError,
    StageArtifactTransportError,
    _PartialState,
    main,
)


class StageArtifactCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._fixture_directory = tempfile.TemporaryDirectory()
        fixture_root = Path(cls._fixture_directory.name)
        checkpoint = fixture_root / "checkpoint"
        _write_tiny_checkpoint(checkpoint)
        cls.source_package = fixture_root / "source-package"
        cls.compilation = compile_safetensors_stage_artifact(
            str(checkpoint),
            cls.source_package,
            layer_start=0,
            layer_end=2,
        )
        cls.source_files = {
            f"/{path.name}": path.read_bytes() for path in cls.source_package.iterdir()
        }

    @classmethod
    def tearDownClass(cls) -> None:
        cls._fixture_directory.cleanup()

    def test_local_source_prepares_a_stage_runner_and_selectively_cleans_it(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = StageArtifactCache(Path(directory) / "cache", limits=_test_limits())

            result = cache.acquire(
                self.source_package / STAGE_ARTIFACT_MANIFEST,
                expected_package_id=self.compilation.package_id,
            )

            self.assertTrue(result.materialized)
            self.assertGreater(result.downloaded_bytes, 0)
            self.assertEqual(result.resumed_bytes, 0)
            verified = verify_stage_artifact(
                result.package_directory,
                expected_package_id=self.compilation.package_id,
            )
            self.assertEqual(verified.manifest.package_id, self.compilation.package_id)
            runner = StageRunner(
                StageModelSpec(
                    model_name=result.package_directory,
                    layer_start=0,
                    layer_end=2,
                    total_layers=2,
                    threads=1,
                    artifact_identity=self.compilation.model_identity,
                    stage_package_identity=result.artifact_identity,
                ),
                device="cpu",
            )
            runner.close()
            for item in result.objects:
                self.assertTrue(cache.object_path(item.sha256).is_file())
            with self.assertRaisesRegex(
                RuntimeError, "referenced by a materialized package"
            ):
                cache.cleanup(digest=self.compilation.weights_sha256)

            cleaned = cache.cleanup(
                package_id=result.package_id,
                prune_package_objects=True,
            )

            self.assertTrue(cleaned.package_removed)
            self.assertEqual(
                set(cleaned.objects_removed),
                {item.sha256 for item in result.objects},
            )
            self.assertFalse(Path(result.package_directory).exists())

    def test_local_partial_resumes_from_its_confirmed_offset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = StageArtifactCache(Path(directory) / "cache", limits=_test_limits())
            original_commit = cache._commit_partial_chunk
            interrupted = False

            def interrupt_after_two_weight_chunks(
                handle,
                *,
                state_path,
                state,
                chunk,
            ):
                nonlocal interrupted
                original_commit(
                    handle,
                    state_path=state_path,
                    state=state,
                    chunk=chunk,
                )
                if (
                    state.digest == self.compilation.weights_sha256
                    and state.confirmed_offset == 32
                    and not interrupted
                ):
                    interrupted = True
                    raise StageArtifactTransportError(
                        "injected local byte transport cut"
                    )

            with (
                patch.object(
                    cache,
                    "_commit_partial_chunk",
                    side_effect=interrupt_after_two_weight_chunks,
                ),
                self.assertRaisesRegex(StageArtifactTransportError, "injected local"),
            ):
                cache.acquire(self.source_package / STAGE_ARTIFACT_MANIFEST)

            part_path, state_path = cache._partial_paths(
                self.compilation.weights_sha256
            )
            self.assertEqual(part_path.stat().st_size, 32)
            self.assertEqual(
                json.loads(state_path.read_text(encoding="utf-8"))["confirmedOffset"],
                32,
            )

            result = cache.acquire(self.source_package / STAGE_ARTIFACT_MANIFEST)

            self.assertEqual(result.resumed_bytes, 32)
            verify_stage_artifact(result.package_directory)

    def test_declared_payload_size_limit_fails_before_payload_acquisition(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = StageArtifactCache(
                Path(directory) / "cache",
                limits=StageArtifactCacheLimits(
                    max_manifest_bytes=1024 * 1024,
                    max_payload_bytes=8,
                    max_package_bytes=8 * 1024 * 1024,
                ),
            )

            with self.assertRaisesRegex(StageArtifactIntegrityError, "file size limit"):
                cache.acquire(self.source_package / STAGE_ARTIFACT_MANIFEST)

            self.assertEqual(
                [
                    path
                    for path in (cache.root / "objects").rglob("*")
                    if path.is_file()
                ],
                [],
            )

    def test_http_cut_resumes_from_the_exact_durable_byte_offset(self) -> None:
        fixture = _HttpArtifactFixture(
            self.source_files,
            cut_target=f"/{STAGE_ARTIFACT_WEIGHTS}",
            cut_at=37,
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            _serve_fixture(fixture) as manifest_url,
        ):
            cache = StageArtifactCache(Path(directory) / "cache", limits=_test_limits())
            weights_digest = self.compilation.weights_sha256

            with self.assertRaisesRegex(
                StageArtifactTransportError, "stopped|transport"
            ):
                cache.acquire(manifest_url)

            part_path, state_path = cache._partial_paths(weights_digest)
            state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(state["confirmedOffset"], 37)
            self.assertEqual(part_path.stat().st_size, 37)

            result = cache.acquire(manifest_url)

            self.assertEqual(result.resumed_bytes, 37)
            self.assertFalse(part_path.exists())
            self.assertFalse(state_path.exists())
            verify_stage_artifact(result.package_directory)
            weight_requests = fixture.requests_for(STAGE_ARTIFACT_WEIGHTS)
            self.assertEqual(len(weight_requests), 2)
            self.assertIsNone(weight_requests[0]["range"])
            self.assertEqual(weight_requests[1]["range"], "bytes=37-")
            self.assertEqual(weight_requests[1]["if_range"], '"fixture-v1"')

    def test_resume_rejects_a_server_that_ignores_range_without_appending(
        self,
    ) -> None:
        fixture = _HttpArtifactFixture(
            self.source_files,
            cut_target=f"/{STAGE_ARTIFACT_WEIGHTS}",
            cut_at=41,
            ignore_range=True,
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            _serve_fixture(fixture) as manifest_url,
        ):
            cache = StageArtifactCache(Path(directory) / "cache", limits=_test_limits())
            with self.assertRaises(StageArtifactTransportError):
                cache.acquire(manifest_url)
            part_path, state_path = cache._partial_paths(
                self.compilation.weights_sha256
            )

            with self.assertRaisesRegex(StageArtifactIntegrityError, "ignored Range"):
                cache.acquire(manifest_url)

            self.assertEqual(part_path.stat().st_size, 41)
            self.assertEqual(
                json.loads(state_path.read_text(encoding="utf-8"))["confirmedOffset"],
                41,
            )

    def test_resume_rejects_changed_etag_without_appending(self) -> None:
        fixture = _HttpArtifactFixture(
            self.source_files,
            cut_target=f"/{STAGE_ARTIFACT_WEIGHTS}",
            cut_at=43,
            change_etag_on_range=True,
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            _serve_fixture(fixture) as manifest_url,
        ):
            cache = StageArtifactCache(Path(directory) / "cache", limits=_test_limits())
            with self.assertRaises(StageArtifactTransportError):
                cache.acquire(manifest_url)
            part_path, state_path = cache._partial_paths(
                self.compilation.weights_sha256
            )

            with self.assertRaisesRegex(StageArtifactIntegrityError, "ETag changed"):
                cache.acquire(manifest_url)

            self.assertEqual(part_path.stat().st_size, 43)
            self.assertEqual(
                json.loads(state_path.read_text(encoding="utf-8"))["confirmedOffset"],
                43,
            )

    def test_wrong_payload_hash_is_rejected_and_never_published(self) -> None:
        fixture = _HttpArtifactFixture(
            self.source_files,
            corrupt_target=f"/{STAGE_ARTIFACT_WEIGHTS}",
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            _serve_fixture(fixture) as manifest_url,
        ):
            cache = StageArtifactCache(Path(directory) / "cache", limits=_test_limits())

            with self.assertRaisesRegex(
                StageArtifactIntegrityError, "SHA-256 mismatch"
            ):
                cache.acquire(manifest_url)

            self.assertFalse(
                cache.object_path(self.compilation.weights_sha256).exists()
            )
            part_path, state_path = cache._partial_paths(
                self.compilation.weights_sha256
            )
            self.assertFalse(part_path.exists())
            self.assertFalse(state_path.exists())
            self.assertFalse(
                (cache.root / "packages" / self.compilation.package_id).exists()
            )

    def test_concurrent_acquisition_deduplicates_payload_downloads(self) -> None:
        # Este test va sobre DEDUPLICACION, no sobre trocear, asi que usa el
        # tamano de trozo de PRODUCCION (`transfer_chunk_bytes`, 1 MiB por
        # defecto en stage_artifact_cache.py) en vez de los dieciseis bytes que
        # heredaba de `_test_limits()`.
        #
        # Con dieciseis bytes hacia 537 escrituras atomicas de estado —cada una
        # crea, sincroniza, renombra y borra un fichero temporal— para mover
        # 8.489 bytes. Ese coste es FIJO POR TROZO, y es el 71 % del tiempo del
        # test. Como este es el unico test del fichero con fecha limite de reloj
        # (`future.result(timeout=20)`, y ademas `lock_timeout_seconds=10`),
        # bastaba con que la maquina se cargara para pasar de 4 s a 37-49 s y
        # fallar. Reproducido exactamente inyectando 60 ms por escritura
        # atomica: 37,88 s y falla, con las 537 escrituras.
        #
        # NO es aflojar el test: no se toca ningun presupuesto de tiempo. Se le
        # quita un coste que nunca quiso ejercitar. Comprobado por PRUEBA DE
        # MUTACION que conserva su poder de deteccion: sustituyendo
        # `_exclusive_file_lock` por un contextmanager que no excluye nada, el
        # test FALLA igual con 16 B, con 64 KiB y con 1 MiB. Y la carrera se
        # sigue produciendo, porque la crea el `delay_seconds=0.05` del fixture,
        # no el tamano de trozo: con 1 MiB varios candados siguen esperandose.
        #
        # La cobertura de reanudacion y offsets se queda intacta: los cuatro
        # tests que asertan sobre ella siguen con los dieciseis bytes por
        # defecto.
        fixture = _HttpArtifactFixture(
            self.source_files,
            delay_seconds=0.05,
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            _serve_fixture(fixture) as manifest_url,
        ):
            cache = StageArtifactCache(
                Path(directory) / "cache",
                limits=_test_limits(transfer_chunk_bytes=1024 * 1024),
            )
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(cache.acquire, manifest_url) for _ in range(2)
                ]
                results = [future.result(timeout=20) for future in futures]

            self.assertEqual(results[0].package_directory, results[1].package_directory)
            self.assertEqual(
                sorted(result.materialized for result in results),
                [False, True],
            )
            self.assertEqual(len(fixture.requests_for(STAGE_ARTIFACT_CONFIG)), 1)
            self.assertEqual(len(fixture.requests_for(STAGE_ARTIFACT_WEIGHTS)), 1)

    def test_existing_cache_object_and_destination_are_never_overwritten(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = StageArtifactCache(root / "cache", limits=_test_limits())
            object_path = cache.object_path(self.compilation.weights_sha256)
            object_path.parent.mkdir(parents=True, exist_ok=True)
            object_path.write_bytes(b"user-owned-sentinel")

            with self.assertRaisesRegex(StageArtifactIntegrityError, "wrong size"):
                cache.acquire(self.source_package / STAGE_ARTIFACT_MANIFEST)
            self.assertEqual(object_path.read_bytes(), b"user-owned-sentinel")

            cache.cleanup(digest=self.compilation.weights_sha256)
            destination = root / "existing-destination"
            destination.mkdir()
            marker = destination / "keep.txt"
            marker.write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                cache.acquire(
                    self.source_package / STAGE_ARTIFACT_MANIFEST,
                    destination=destination,
                )
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_confirmed_offset_never_runs_ahead_of_the_durable_payload(self) -> None:
        """El invariante del que depende toda la reanudacion, fijado como RELACION.

        Los cuatro tests de reanudacion que ya habia fijan NUMEROS concretos —que
        el offset vale 32, o 37—, y eso no protege la propiedad que de verdad
        importa: **el offset confirmado en disco nunca puede ir por delante de los
        bytes durables del `.part`**.

        La asimetria es total. Un offset ATRASADO lo arregla el lector: trunca lo
        que sobra y reanuda. Uno ADELANTADO es TERMINAL — el lector lanza
        "partial payload is shorter than its confirmed offset" en cada `acquire`
        posterior de ese digest, para siempre.

        Y estaba sin cubrir: se comprobo que una implementacion que publica el
        estado ANTES de que los bytes sean durables pasa la suite entera en verde,
        10 de 10. Este test es lo que hace que deje de pasar.
        """
        with tempfile.TemporaryDirectory() as directory:
            cache = StageArtifactCache(Path(directory) / "cache", limits=_test_limits())
            original_write = cache._write_partial_state
            violations: list[tuple[str, int, int]] = []

            def check_before_publishing(state_path, state):
                # En el instante de publicar, el offset que se va a escribir tiene
                # que caber ya en los bytes que hay en el `.part`.
                part_path = state_path.with_suffix("")
                durable = part_path.stat().st_size if part_path.exists() else 0
                if state.confirmed_offset > durable:
                    violations.append(
                        (state.digest, state.confirmed_offset, durable)
                    )
                original_write(state_path, state)

            with patch.object(
                cache, "_write_partial_state", check_before_publishing
            ):
                cache.acquire(self.source_package / STAGE_ARTIFACT_MANIFEST)

            self.assertEqual(
                violations,
                [],
                "el estado se publico por delante de los bytes durables del .part",
            )

    def test_a_cut_before_the_first_state_write_stays_recoverable(self) -> None:
        """Un corte en la ventana de creacion no puede encallar el digest.

        `_load_or_create_partial` escribe el estado y despues crea el `.part`. Al
        reves, un corte entre las dos lineas dejaba un `.part` huerfano sin estado,
        y eso es TERMINAL: cada `acquire` posterior de ese digest falla con
        "partial payload exists without confirmed offset state" hasta que alguien
        corra un `cleanup --digest` a mano. La ventana se midio en 7,83 ms de
        mediana por fichero de carga, y es un ancho FIJO por fichero.

        Aqui se fabrican las dos caras de esa ventana y se comprueba que la que
        deja el orden actual es recuperable.
        """
        with tempfile.TemporaryDirectory() as directory:
            cache = StageArtifactCache(Path(directory) / "cache", limits=_test_limits())
            digest = self.compilation.weights_sha256
            part_path, state_path = cache._partial_paths(digest)
            argumentos = {
                "digest": digest,
                "expected_size": 64,
                "source": "fuente-de-prueba",
                "part_path": part_path,
                "state_path": state_path,
            }

            # SE CORTA EN LA PRIMERA ESCRITURA DE ESTADO, que es el instante que
            # decide. Con el orden actual el `.part` todavia no existe, asi que no
            # queda nada que estorbe. Con el orden invertido ya se habria creado y
            # quedaria huerfano — y este assert es lo que lo detecta.
            with patch.object(
                cache,
                "_write_partial_state",
                side_effect=OSError("corte inyectado al publicar el estado"),
            ):
                with self.assertRaises(OSError):
                    cache._load_or_create_partial(**argumentos)

            self.assertFalse(
                part_path.exists(),
                "el `.part` se creo antes de publicar el estado: un corte aqui deja"
                " un huerfano que encalla el digest para siempre",
            )

            # Y por tanto el siguiente intento arranca limpio en vez de encallar.
            state = cache._load_or_create_partial(**argumentos)
            self.assertEqual(state.confirmed_offset, 0)
            self.assertTrue(part_path.exists())

            # El estado que dejaba el orden ANTIGUO sigue siendo terminal a
            # proposito: un `.part` sin estado es corrupcion real y no se debe
            # adivinar que hacer con ella. Por eso el arreglo es no producirlo,
            # no tolerarlo.
            state_path.unlink()
            with self.assertRaisesRegex(
                StageArtifactIntegrityError, "without confirmed offset state"
            ):
                cache._load_or_create_partial(**argumentos)

    def test_cli_acquire_returns_a_control_plane_ready_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = main(
                    [
                        "acquire",
                        "--cache-root",
                        str(Path(directory) / "cache"),
                        "--manifest",
                        str(self.source_package / STAGE_ARTIFACT_MANIFEST),
                        "--expected-package-id",
                        self.compilation.package_id,
                    ]
                )

            self.assertEqual(exit_code, 0)
            document = json.loads(output.getvalue())
            self.assertEqual(
                document["artifact_identity"],
                self.compilation.artifact_identity,
            )
            verify_stage_artifact(document["package_directory"])


def _test_limits(
    *, transfer_chunk_bytes: int = 16
) -> StageArtifactCacheLimits:
    """Limites compartidos por los tests de la cache.

    `transfer_chunk_bytes=16` es DELIBERADO y por defecto: trocear el payload en
    pedazos de dieciseis bytes obliga a que haya cientos de confirmaciones
    parciales, que es lo unico que ejercita de verdad la reanudacion por offset,
    los cortes a mitad de transferencia y el estado parcial. Cuatro tests
    asertan sobre eso y lo necesitan.

    Pero es un tamano PATOLOGICO en coste de E/S, y se heredaba sin querer. Cada
    trozo confirmado hace, en `_commit_partial_chunk`, un `fsync` del `.part` mas
    una escritura atomica del estado — que crea un fichero temporal nuevo, lo
    sincroniza, lo renombra y lo borra. Medido en este proyecto: **537
    escrituras atomicas para 8.489 bytes de payload, el 71 % del tiempo del
    test**. Ese coste es FIJO POR TROZO, asi que dieciseis bytes lo multiplican
    por sesenta y cinco mil frente al valor de produccion.

    Por eso el parametro: un test que NO va sobre trocear no debe pagarlo. El
    unico con fecha limite de reloj —el de adquisicion concurrente— pasaba de
    4 segundos a 37-49 y fallaba en cuanto la maquina se cargaba.
    """
    return StageArtifactCacheLimits(
        max_manifest_bytes=1024 * 1024,
        max_payload_bytes=4 * 1024 * 1024,
        max_package_bytes=8 * 1024 * 1024,
        transfer_chunk_bytes=transfer_chunk_bytes,
        http_timeout_seconds=5,
        lock_timeout_seconds=10,
    )


def _write_tiny_checkpoint(root: Path) -> None:
    root.mkdir(parents=True)
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=2,
        tie_word_embeddings=False,
        attention_bias=False,
        mlp_bias=False,
    )
    config.architectures = ["LlamaForCausalLM"]
    config.save_pretrained(root)
    torch.manual_seed(11)
    model = AutoModelForCausalLM.from_config(config, dtype=torch.float16)
    state = {
        name: value.detach().cpu().contiguous().clone()
        for name, value in model.state_dict().items()
    }
    save_file(state, root / "model.safetensors")
    del model


class _HttpArtifactFixture:
    def __init__(
        self,
        files: dict[str, bytes],
        *,
        cut_target: str | None = None,
        cut_at: int = 0,
        ignore_range: bool = False,
        change_etag_on_range: bool = False,
        corrupt_target: str | None = None,
        delay_seconds: float = 0,
    ) -> None:
        self.files = dict(files)
        self.cut_target = cut_target
        self.cut_at = cut_at
        self.ignore_range = ignore_range
        self.change_etag_on_range = change_etag_on_range
        self.corrupt_target = corrupt_target
        self.delay_seconds = delay_seconds
        self.cut_done = False
        self.requests: list[dict[str, str | None]] = []
        self.lock = threading.Lock()

    def record(self, path: str, range_header: str | None, if_range: str | None) -> None:
        with self.lock:
            self.requests.append(
                {
                    "path": path,
                    "range": range_header,
                    "if_range": if_range,
                }
            )

    def requests_for(self, name: str) -> list[dict[str, str | None]]:
        with self.lock:
            return [dict(item) for item in self.requests if item["path"] == f"/{name}"]


@contextmanager
def _serve_fixture(
    fixture: _HttpArtifactFixture,
):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:
            value = fixture.files.get(self.path)
            if value is None:
                self.send_error(404)
                return
            range_header = self.headers.get("Range")
            if_range = self.headers.get("If-Range")
            fixture.record(self.path, range_header, if_range)
            if (
                fixture.corrupt_target == self.path
                and self.path != f"/{STAGE_ARTIFACT_MANIFEST}"
            ):
                value = bytes([value[0] ^ 1]) + value[1:]
            offset = 0
            if range_header is not None:
                offset = int(range_header.removeprefix("bytes=").removesuffix("-"))
            ignore_range = range_header is not None and fixture.ignore_range
            if ignore_range:
                offset = 0
                self.send_response(200)
            elif range_header is not None:
                self.send_response(206)
                self.send_header(
                    "Content-Range",
                    f"bytes {offset}-{len(value) - 1}/{len(value)}",
                )
            else:
                self.send_response(200)
            body = value[offset:]
            if self.path != f"/{STAGE_ARTIFACT_MANIFEST}":
                etag = (
                    '"fixture-v2"'
                    if range_header is not None and fixture.change_etag_on_range
                    else '"fixture-v1"'
                )
                self.send_header("ETag", etag)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if fixture.delay_seconds:
                time.sleep(fixture.delay_seconds)
            should_cut = (
                fixture.cut_target == self.path
                and range_header is None
                and not fixture.cut_done
            )
            if should_cut:
                fixture.cut_done = True
                self.wfile.write(body[: fixture.cut_at])
                self.wfile.flush()
                try:
                    self.connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                self.connection.close()
                return
            self.wfile.write(body)
            self.wfile.flush()

        def log_message(self, _format: str, *_args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield (f"http://127.0.0.1:{server.server_address[1]}/{STAGE_ARTIFACT_MANIFEST}")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
