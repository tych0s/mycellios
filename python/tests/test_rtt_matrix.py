"""El ensamblador de la matriz RTT omite lo no medido en vez de inventarlo."""
from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest

_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[2] / "scripts" / "build_rtt_matrix.py"
)
_spec = importlib.util.spec_from_file_location("build_rtt_matrix", _SCRIPT)
assert _spec and _spec.loader
build_rtt_matrix = importlib.util.module_from_spec(_spec)
sys.modules["build_rtt_matrix"] = build_rtt_matrix
_spec.loader.exec_module(build_rtt_matrix)


def _peer(p50: float, *, n: int = 30, p90: float | None = None, complete: bool = True):
    return {"p50_ms": p50, "n": n, "p90_ms": p90, "complete": complete, "url": "u"}


class MeasuredEdgesTests(unittest.TestCase):
    def test_rtt_is_halved_into_one_way(self) -> None:
        """La sonda mide ida y vuelta; el planificador espera one-way.

        Confundirlos duplica el coste de cada salto y sesgaría toda la
        colocación hacia cadenas más cortas de lo necesario.
        """
        stats = {
            "n0": {"peers": {"n1": _peer(80.0)}},
            "n1": {"peers": {"n0": _peer(80.0)}},
        }
        links, _ = build_rtt_matrix.build_links(stats)
        self.assertEqual(len(links), 2)
        self.assertAlmostEqual(links[0]["oneWayLatencyMs"], 40.0)

    def test_jitter_is_derived_from_p90_not_folded_into_latency(self) -> None:
        stats = {"n0": {"peers": {"n1": _peer(40.0, p90=60.0)}}}
        links, _ = build_rtt_matrix.build_links(stats)
        self.assertAlmostEqual(links[0]["oneWayLatencyMs"], 20.0)
        self.assertAlmostEqual(links[0]["jitterP95Ms"], 10.0)


class OmissionTests(unittest.TestCase):
    def test_unmeasured_edge_is_omitted_not_invented(self) -> None:
        """La regla central: omitida cae al centinela y se evita; inventada, se elige."""
        stats = {
            "n0": {"peers": {"n1": {"n": 0, "p50_ms": None, "error": "timeout"}}},
        }
        links, warnings = build_rtt_matrix.build_links(stats)
        self.assertEqual(links, [])
        self.assertTrue(any("Se OMITE" in warning for warning in warnings))

    def test_partial_measurement_is_omitted_by_default(self) -> None:
        stats = {"n0": {"peers": {"n1": _peer(40.0, n=3, complete=False)}}}
        links, warnings = build_rtt_matrix.build_links(stats)
        self.assertEqual(links, [])
        self.assertTrue(any("incompleta" in warning for warning in warnings))

    def test_partial_can_be_opted_into_explicitly(self) -> None:
        stats = {"n0": {"peers": {"n1": _peer(40.0, n=3, complete=False)}}}
        links, _ = build_rtt_matrix.build_links(stats, require_complete=False)
        self.assertEqual(len(links), 1)

    def test_matrix_holes_are_reported(self) -> None:
        # n1 no midió contra n0: el hueco tiene que ser visible, porque decide
        # si el planificador puede usar esa cadena o no.
        stats = {"n0": {"peers": {"n1": _peer(40.0)}}, "n1": {"peers": {}}}
        _, warnings = build_rtt_matrix.build_links(stats)
        self.assertTrue(any("HUECO" in warning for warning in warnings))


class AsymmetryTests(unittest.TestCase):
    def test_asymmetric_routes_are_preserved_and_flagged(self) -> None:
        """Promediar A->B con B->A ocultaría rutas asimétricas, que existen."""
        stats = {
            "n0": {"peers": {"n1": _peer(40.0)}},
            "n1": {"peers": {"n0": _peer(200.0)}},
        }
        links, warnings = build_rtt_matrix.build_links(stats)
        forward = next(l for l in links if l["from"] == "n0")
        backward = next(l for l in links if l["from"] == "n1")
        self.assertAlmostEqual(forward["oneWayLatencyMs"], 20.0)
        self.assertAlmostEqual(backward["oneWayLatencyMs"], 100.0)
        self.assertTrue(any("ASIMÉTRICA" in warning for warning in warnings))

    def test_symmetric_routes_do_not_warn(self) -> None:
        stats = {
            "n0": {"peers": {"n1": _peer(40.0)}},
            "n1": {"peers": {"n0": _peer(44.0)}},
        }
        _, warnings = build_rtt_matrix.build_links(stats)
        self.assertFalse(any("ASIMÉTRICA" in warning for warning in warnings))


class OutputShapeTests(unittest.TestCase):
    def test_links_match_the_planner_contract(self) -> None:
        # Las claves son las de `DirectedLinkProfile` en src/distribution/types.ts.
        stats = {"n0": {"peers": {"n1": _peer(40.0)}}}
        links, _ = build_rtt_matrix.build_links(stats)
        self.assertEqual(
            set(links[0]),
            {
                "from",
                "to",
                "oneWayLatencyMs",
                "jitterP95Ms",
                "bandwidthMbps",
                "lossRate",
                "availability",
            },
        )


if __name__ == "__main__":
    unittest.main()
