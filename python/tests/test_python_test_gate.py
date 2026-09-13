from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


class PythonTestGateTests(unittest.TestCase):
    def run_fixture(self, outcome: str, strict: bool = False):
        gate = Path(__file__).resolve().parents[2] / "scripts" / "run-python-tests.py"
        program = textwrap.dedent(f"""
            import runpy
            import sys
            import unittest
            from unittest.mock import patch
            class InjectedFailure(unittest.TestCase):
                def id(self):
                    return ('test_external_cell.ExternalTensorParallelCellTests.'
                            'test_external_rank_cli_and_anchor_rank_zero_execute_one_logical_stage')
                def runTest(self):
                    {outcome}
            suite = unittest.TestSuite([InjectedFailure()])
            sys.argv = [sys.argv[1], *sys.argv[2:]]
            with patch.object(unittest.defaultTestLoader, 'discover', return_value=suite):
                runpy.run_path(sys.argv[0], run_name='__main__')
        """)
        return subprocess.run(
            [sys.executable, "-c", program, str(gate), *(["--strict"] if strict else [])],
            capture_output=True, text=True, timeout=15,
        )

    def test_previously_exempt_parity_failures_fail_the_command(self):
        for strict in (False, True):
            with self.subTest(strict=strict):
                result = self.run_fixture("self.fail('injected parity regression')", strict)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn("failures=1", result.stdout)

    def test_startup_errors_cannot_turn_into_a_success(self):
        result = self.run_fixture("raise RuntimeError('Connection refused')")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("errors=1", result.stdout)

    def test_unavailable_physical_tests_are_explicitly_counted_as_skipped(self):
        result = self.run_fixture("self.skipTest('physical host unavailable')")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("skipped=1", result.stdout)


if __name__ == "__main__":
    unittest.main()
