from __future__ import annotations

from argparse import Namespace
import unittest

from distributed_runtime.conversation_benchmark import DEFAULT_TURNS, validate_args


class ConversationBenchmarkTests(unittest.TestCase):
    def test_arguments_define_bounded_growing_chats(self) -> None:
        args = Namespace(
            conversations="1, 4",
            iterations=2,
            turns=len(DEFAULT_TURNS),
            output_tokens=16,
            timeout_seconds=10.0,
        )
        self.assertEqual(validate_args(args), (1, 4))

    def test_rejects_invalid_scenarios(self) -> None:
        valid = {
            "conversations": "1",
            "iterations": 1,
            "turns": 1,
            "output_tokens": 1,
            "timeout_seconds": 10.0,
        }
        for change in (
            {"iterations": 0},
            {"turns": 0},
            {"turns": len(DEFAULT_TURNS) + 1},
            {"output_tokens": 0},
            {"timeout_seconds": float("nan")},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_args(Namespace(**(valid | change)))


if __name__ == "__main__":
    unittest.main()
