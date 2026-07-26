"""La invariante de `kv_valid` se hace cumplir en el punto donde se vuelve durable.

Bloqueante de medición número 2. `kv_valid` se adelanta en el DESPACHO de la
ola, no en el commit. En la ruta secuencial es sólido —la ola de decode reenvía
un token ya emitido—, pero con varias olas en vuelo una ola despachada tras una
divergencia adelanta posiciones que nunca se comprometen.

Lo peligroso no es el valor en sí: es que se persiste en la sesión retenida y
gobierna la reutilización de prefijo del turno SIGUIENTE. Un `kv_valid` inflado
hace reutilizar KV que no codifica la secuencia servida, y la divergencia
aparece en otro turno, sin traza que la conecte con su causa.

Estos tests fijan la invariante declarada por el propio tipo:

    kv_valid <= len(prompt) + len(emitidos)
"""
from __future__ import annotations

import unittest


class KvValidInvariantTests(unittest.TestCase):
    """El invariante como propiedad, independiente del motor.

    Se prueba la aritmética de la comprobación —no se arranca un pipeline
    entero— porque lo que se quiere blindar es la CONDICIÓN, y ésa se puede
    afirmar sin GPU ni sockets.
    """

    @staticmethod
    def _violates(kv_valid: int, prompt_tokens: int, emitted_tokens: int) -> bool:
        served = prompt_tokens + emitted_tokens
        return kv_valid > served

    def test_sequential_decode_stays_within_the_invariant(self) -> None:
        # Prefill de 10 + 5 decodes: cada decode reenvía un token ya emitido.
        self.assertFalse(self._violates(kv_valid=15, prompt_tokens=10, emitted_tokens=5))

    def test_kv_valid_may_lag_the_served_sequence(self) -> None:
        # Quedarse corto es seguro: se reutiliza menos prefijo del posible.
        self.assertFalse(self._violates(kv_valid=8, prompt_tokens=10, emitted_tokens=5))

    def test_pipelined_dispatch_past_a_divergence_violates(self) -> None:
        """El caso que el guardián existe para atrapar.

        Tres olas en vuelo, la primera diverge: las otras dos ya habían
        adelantado `kv_valid` por tokens que nunca se emitieron.
        """
        self.assertTrue(self._violates(kv_valid=18, prompt_tokens=10, emitted_tokens=5))

    def test_exact_equality_is_allowed(self) -> None:
        # El caso normal al final de un turno limpio.
        self.assertFalse(self._violates(kv_valid=15, prompt_tokens=10, emitted_tokens=5))

    def test_empty_generation_cannot_claim_validity_beyond_the_prompt(self) -> None:
        self.assertFalse(self._violates(kv_valid=10, prompt_tokens=10, emitted_tokens=0))
        self.assertTrue(self._violates(kv_valid=11, prompt_tokens=10, emitted_tokens=0))


class GuardIsWiredTests(unittest.TestCase):
    def test_the_engine_raises_rather_than_retaining_a_bad_session(self) -> None:
        """El guardián está en el punto donde el error se vuelve durable.

        Se comprueba sobre el fuente para que nadie lo mueva a un sitio donde
        ya no proteja la reutilización del turno siguiente, ni lo degrade a un
        aviso.
        """
        import pathlib

        source = (
            pathlib.Path(__file__).resolve().parents[1]
            / "distributed_runtime"
            / "engine.py"
        ).read_text(encoding="utf-8")
        self.assertIn("kv_valid invariant violated", source)
        # Tiene que LANZAR: degradarlo a warning devolvería el fallo silencioso.
        marker = source.index("kv_valid invariant violated")
        preceding = source[max(0, marker - 400) : marker]
        self.assertIn("raise RuntimeError", preceding)


if __name__ == "__main__":
    unittest.main()
