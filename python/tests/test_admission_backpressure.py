"""Contrapresión: una cola llena debe decirse distinto de un fallo.

Antes, `submit` levantaba un `RuntimeError` pelado para la cola llena y el
servidor lo traducía —junto con cualquier otro `RuntimeError`— a un 503
`service_unavailable`. Dos consecuencias, las dos malas:

  - Un cliente (o un balanceador delante) no puede distinguir «estoy saturado,
    reintenta» de «estoy roto, sácame del pool». Un 503 dice lo segundo.
  - Un fallo interno de verdad quedaba disfrazado de carga.

Contexto medido: bajo sobrecarga con cola plana, TODA petición admitida expiró y
el goodput puede caer a cero cuando una cola saturada sigue admitiendo trabajo.
Rechazar lo que no se puede servir es lo que mantiene servido lo que sí.
"""
import unittest

from distributed_runtime.engine import QueueFullError
from distributed_runtime.server import (
    OVERLOAD_RETRY_AFTER_SECONDS,
    overloaded_response,
)


class QueueFullErrorTests(unittest.TestCase):
    def test_is_a_runtime_error_so_old_handlers_still_catch_it(self):
        error = QueueFullError("lleno", pending=32, capacity=32)
        self.assertIsInstance(error, RuntimeError)

    def test_carries_the_numbers_needed_to_act_on_it(self):
        error = QueueFullError("lleno", pending=30, capacity=32)
        self.assertEqual(error.pending, 30)
        self.assertEqual(error.capacity, 32)

    def test_keeps_its_message(self):
        self.assertEqual(str(QueueFullError("cola llena", pending=1, capacity=1)), "cola llena")


class OverloadResponseTests(unittest.TestCase):
    def test_answers_429_not_503(self):
        response = overloaded_response(QueueFullError("lleno", pending=32, capacity=32))
        self.assertEqual(response.status, 429)

    def test_tells_the_client_when_to_come_back(self):
        response = overloaded_response(QueueFullError("lleno", pending=32, capacity=32))
        self.assertEqual(response.headers["Retry-After"], str(OVERLOAD_RETRY_AFTER_SECONDS))
        self.assertGreater(OVERLOAD_RETRY_AFTER_SECONDS, 0)

    def test_reports_the_occupancy_so_a_caller_can_back_off_proportionally(self):
        import json

        response = overloaded_response(QueueFullError("lleno", pending=30, capacity=32))
        body = json.loads(response.body)
        self.assertEqual(body["error"]["type"], "overloaded")
        self.assertEqual(body["error"]["pending"], 30)
        self.assertEqual(body["error"]["capacity"], 32)


class EngineRaisesQueueFullTests(unittest.TestCase):
    """La ruta real del motor debe levantar el tipo nuevo, no un RuntimeError."""

    def test_engine_and_recovery_agree_on_the_exception_type(self):
        import inspect

        from distributed_runtime import engine, recovery

        for module in (engine, recovery):
            source = inspect.getsource(module)
            self.assertIn(
                "raise QueueFullError(",
                source,
                f"{module.__name__} debe señalar la cola llena con QueueFullError",
            )
            self.assertNotIn(
                'raise RuntimeError("pipeline request queue is full")',
                source,
                f"{module.__name__} conserva el RuntimeError pelado: "
                "el borde HTTP no puede distinguirlo de un fallo real",
            )


class ErrorPathMapsQueueFullTests(unittest.TestCase):
    """La cola llena NO llega por `submit()`, y ahí estaba el fallo.

    `ContinuousMicroBatcher.submit()` sólo hace `put_nowait` en una
    `asyncio.Queue` **sin límite**, así que nunca puede levantar `QueueFullError`.
    La levanta `engine.submit()` dentro del bucle de despacho, y viaja como
    evento `("error", ...)` del pending. Mapearla sólo en el borde de entrada
    dejaba la contrapresión SIN EFECTO: el cliente recibía un 500 genérico y no
    sabía que debía reintentar.
    """

    def test_submit_cannot_raise_queue_full_because_its_queue_is_unbounded(self):
        import inspect

        from distributed_runtime.server import ContinuousMicroBatcher

        source = inspect.getsource(ContinuousMicroBatcher.submit)
        self.assertIn("put_nowait", source)
        self.assertNotIn(
            "engine.submit", source,
            "si `submit` llamara al motor, el catch del borde bastaría; "
            "como no lo hace, la cola llena tiene que mapearse en la ruta de eventos",
        )

    def test_the_event_path_maps_queue_full_to_the_overload_response(self):
        import inspect

        from distributed_runtime import server

        source = inspect.getsource(server)
        self.assertIn(
            "isinstance(payload, QueueFullError)", source,
            "la ruta de eventos debe distinguir la cola llena del fallo genérico",
        )
        self.assertIn("overloaded_response(payload)", source)


if __name__ == "__main__":
    unittest.main()
