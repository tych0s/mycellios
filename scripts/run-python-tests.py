#!/usr/bin/env python
"""Run the native Python suite; every failure or error fails the command.

Physical tests retain their explicit prerequisite skips. Numerical parity and
process-startup failures are never converted into successful test runs.
The --strict flag remains accepted for existing callers; strict is the default.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import unittest

PYTHON_DIR = Path(__file__).resolve().parents[1] / "python"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="Fail on any failure or error (the default).")
    parser.parse_args()
    os.chdir(PYTHON_DIR)
    sys.path.insert(0, str(PYTHON_DIR))
    suite = unittest.defaultTestLoader.discover("tests")
    result = unittest.TextTestRunner(verbosity=1, stream=sys.stdout).run(suite)
    load_errors = [
        test.id() for test, _ in result.errors
        if "_FailedTest" in type(test).__name__
    ]
    if load_errors:
        print("Modules that failed to import (their tests were not counted):")
        for name in load_errors:
            print(f"  - {name}")
    print(f"tests={result.testsRun} failures={len(result.failures)} "
          f"errors={len(result.errors)} skipped={len(result.skipped)}")
    print("strict: " + ("PASS" if result.wasSuccessful() else "FAIL"))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
