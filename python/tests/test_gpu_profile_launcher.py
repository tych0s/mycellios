"""Las tres trampas de Salad que hacen que un contenedor no arranque, en silencio.

Ninguna da error al crear: el contenedor se acepta, se queda `stopped` con
`start_time == finish_time`, y parece un fallo de arranque del proceso. Costaron
dos contenedores muertos antes de encontrarlas. Estos tests las fijan para que
no vuelvan por descuido.
"""
from __future__ import annotations

import base64
import importlib.util
import pathlib
import sys
import unittest

_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[2]
    / "scripts"
    / "salad"
    / "run_gpu_profile.py"
)
_spec = importlib.util.spec_from_file_location("run_gpu_profile", _SCRIPT)
assert _spec and _spec.loader
launcher = importlib.util.module_from_spec(_spec)
sys.modules["run_gpu_profile"] = launcher
_spec.loader.exec_module(launcher)

_REPO = pathlib.Path(__file__).resolve().parents[2]


class EnvChunkingTests(unittest.TestCase):
    def test_no_chunk_exceeds_the_env_var_limit(self) -> None:
        env: dict[str, str] = {}
        launcher.chunk_env(
            env, "X", (_REPO / "scripts" / "profile_forward.py").read_bytes()
        )
        self.assertGreater(len(env), 1, "no troceó: el fichero cabe en una variable?")
        for name, value in env.items():
            if name.endswith("_PARTS"):
                continue
            self.assertLessEqual(len(value), launcher.ENV_CHUNK, name)

    def test_chunks_reassemble_byte_identical(self) -> None:
        original = (_REPO / "deploy" / "salad" / "gpu_profile_probe.py").read_bytes()
        env: dict[str, str] = {}
        launcher.chunk_env(env, "X", original)
        joined = "".join(env[f"X_{i}"] for i in range(int(env["X_PARTS"])))
        self.assertEqual(base64.b64decode(joined), original)

    def test_parts_counter_matches_the_chunks(self) -> None:
        env: dict[str, str] = {}
        count = launcher.chunk_env(env, "X", b"a" * 5_000)
        self.assertEqual(int(env["X_PARTS"]), count)
        self.assertIn(f"X_{count - 1}", env)
        self.assertNotIn(f"X_{count}", env)


class BootstrapTests(unittest.TestCase):
    def test_bootstrap_does_not_depend_on_shell_binaries(self) -> None:
        """La imagen de PyTorch garantiza Python; `bash` y `base64` no.

        Depender de ellos añade dos modos de fallo que no aportan nada y que se
        manifiestan igual que cualquier otro: el contenedor no arranca.
        """
        command = launcher.build_command()
        self.assertEqual(command[0], "python")
        joined = " ".join(command)
        self.assertNotIn("bash", joined)
        self.assertNotIn("base64 -d", joined)

    def test_bootstrap_reads_both_scripts_and_runs_as_main(self) -> None:
        command = launcher.build_command()
        body = command[-1]
        self.assertIn("GDLP_FORWARD", body)
        self.assertIn("GDLP_PROBE", body)
        # La sonda tiene su lógica bajo `if __name__ == "__main__"`.
        self.assertIn("'__name__':'__main__'", body.replace(" ", ""))


class BillingSafetyTests(unittest.TestCase):
    def test_the_launcher_declares_autostart_and_never_restart(self) -> None:
        """`autostart_policy` es LA trampa: sin él el contenedor nunca arranca.

        Se comprueba sobre el fuente y no sobre una llamada real para que el
        test no dependa de la red ni gaste dinero.
        """
        source = _SCRIPT.read_text(encoding="utf-8")
        self.assertIn('"autostart_policy": True', source)
        # `never` evita que un proceso que sale reviva y siga facturando.
        self.assertIn('"restart_policy": "never"', source)

    def test_cleanup_runs_even_on_failure(self) -> None:
        source = _SCRIPT.read_text(encoding="utf-8")
        self.assertIn("finally:", source)
        # El borrado tiene que estar DENTRO del finally, no después del try.
        finally_block = source[source.index("finally:") :]
        self.assertIn("delete_container_verified", finally_block)

    def test_network_errors_do_not_escape_and_abort_the_teardown(self) -> None:
        """`HTTPError` sola no basta: un timeout escapaba y dejaba gasto abierto.

        Es el fallo que ya se había corregido en `latency_map.py` y seguía vivo
        en los orquestadores.
        """
        source = _SCRIPT.read_text(encoding="utf-8")
        self.assertIn("urllib.error.URLError", source)
        self.assertIn("OSError", source)

    def test_deletion_is_verified_by_listing_not_by_status_code(self) -> None:
        """Un 2xx no prueba que el recurso muriera; hay que comprobar la lista."""
        source = _SCRIPT.read_text(encoding="utf-8")
        body = source[source.index("def delete_container_verified") :]
        body = body[: body.index("\ndef ")] if "\ndef " in body else body
        self.assertIn('call("GET", base)', body)
        self.assertIn("not in names", body)
        # Y si no se puede confirmar, tiene que decir cómo borrarlo a mano.
        self.assertIn("curl -X DELETE", body)

    def test_the_probe_carries_its_own_absolute_ttl(self) -> None:
        """Tercera capa: el TTL vive DENTRO del contenedor.

        Si el lanzador muere de forma dura —o lo mata un límite de tiempo
        externo antes de que corra su `finally`—, esto es lo único que impide
        que una GPU facture indefinidamente. Pasó en el primer intento.
        """
        probe = (_REPO / "deploy" / "salad" / "gpu_profile_probe.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("GDLP_PROBE_TTL", probe)
        self.assertIn("dejar de facturar", probe)


if __name__ == "__main__":
    unittest.main()
