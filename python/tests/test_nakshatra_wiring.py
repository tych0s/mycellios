"""El brazo D no puede correr PyTorch y reportarse como motor nativo.

Es el fallo de exp13 trasladado al motor: medir un mecanismo que nunca se
activó. Aquí sería aún más difícil de detectar que en el caso original, porque
los tokens saldrían idénticos —el motor nativo es token-exacto por diseño— y
solo cambiaría el tiempo, que es justo lo que se está midiendo.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest

_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[2]
    / "scripts"
    / "salad"
    / "pipeline_orchestrate.py"
)
_spec = importlib.util.spec_from_file_location("pipeline_orchestrate", _SCRIPT)
assert _spec and _spec.loader
orchestrate = importlib.util.module_from_spec(_spec)
sys.modules["pipeline_orchestrate"] = orchestrate
_spec.loader.exec_module(orchestrate)


_COMPLETE = {
    "package": "/opt/pkg/model.nk",
    "daemon_bin": "/opt/bin/nakshatrad",
    "pipeline_id": 42,
    "context_tokens": 4096,
}


class FailClosedTests(unittest.TestCase):
    def test_no_config_means_no_flags(self) -> None:
        self.assertEqual(orchestrate.nakshatra_stage_args(None), [])
        self.assertEqual(orchestrate.nakshatra_stage_args({}), [])

    def test_partial_config_aborts_instead_of_falling_back(self) -> None:
        """La aserción que protege la medida."""
        for missing in _COMPLETE:
            partial = {k: v for k, v in _COMPLETE.items() if k != missing}
            with self.assertRaises(SystemExit) as caught:
                orchestrate.nakshatra_stage_args(partial)
            self.assertIn(missing, str(caught.exception))

    def test_the_abort_explains_why_it_matters(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            orchestrate.nakshatra_stage_args({"package": "x"})
        message = str(caught.exception)
        self.assertIn("PyTorch", message)
        self.assertIn("invalida la medida", message)


class FlagShapeTests(unittest.TestCase):
    def test_complete_config_emits_every_required_flag(self) -> None:
        args = orchestrate.nakshatra_stage_args(_COMPLETE)
        for flag in (
            "--nakshatra-package",
            "--nakshatra-daemon-bin",
            "--nakshatra-pipeline-id",
            "--nakshatra-context-tokens",
            "--nakshatra-gpu-layers",
            "--nakshatra-compute-api",
        ):
            self.assertIn(flag, args)

    def test_defaults_target_the_gpu_not_the_cpu(self) -> None:
        # El defecto de `stage_cli` es `cpu` y 0 capas en GPU, que para el brazo
        # D sería medir el motor nativo con la GPU apagada.
        args = orchestrate.nakshatra_stage_args(_COMPLETE)
        self.assertEqual(args[args.index("--nakshatra-compute-api") + 1], "cuda")
        self.assertEqual(args[args.index("--nakshatra-gpu-layers") + 1], "-1")

    def test_optional_identity_flags_only_appear_when_given(self) -> None:
        self.assertNotIn(
            "--nakshatra-package-id", orchestrate.nakshatra_stage_args(_COMPLETE)
        )
        args = orchestrate.nakshatra_stage_args(
            {**_COMPLETE, "package_id": "pkg-1", "manifest_sha256": "abc"}
        )
        self.assertIn("--nakshatra-package-id", args)
        self.assertIn("--nakshatra-manifest-sha256", args)


class NodePlanTests(unittest.TestCase):
    def test_stages_get_the_engine_and_the_root_does_not(self) -> None:
        """La raíz sigue en PyTorch: tokeniza, planifica y muestrea.

        `server.py` no expone estas banderas, así que pasárselas fallaría al
        arrancar. Aislar el motor en la etapa es además lo que quiere el brazo D.
        """
        plans = orchestrate.node_plan(
            2,
            boundaries=[0, 4, 8],
            total=8,
            model="m",
            codec="fp16",
            ragged=True,
            nakshatra=_COMPLETE,
        )
        root_role, _, root_args, _ = plans[0]
        stage_role, _, stage_args, _ = plans[1]
        self.assertEqual(root_role, "root")
        self.assertEqual(stage_role, "stage")
        self.assertFalse(any(a.startswith("--nakshatra") for a in root_args))
        self.assertIn("--nakshatra-package", stage_args)

    def test_without_nakshatra_the_plan_is_unchanged(self) -> None:
        plans = orchestrate.node_plan(
            2, boundaries=[0, 4, 8], total=8, model="m", codec="fp16", ragged=False
        )
        for _, _, args, _ in plans:
            self.assertFalse(any(a.startswith("--nakshatra") for a in args))


if __name__ == "__main__":
    unittest.main()
