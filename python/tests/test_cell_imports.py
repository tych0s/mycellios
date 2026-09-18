from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


class CellMemberImportTests(unittest.TestCase):
    def test_rank_entrypoints_do_not_load_the_dense_model_stack(self) -> None:
        # A spawned rank executes Torch/SafeTensors directly. Importing the
        # dense Hugging Face loader just to resolve an annotation needlessly
        # consumes its startup deadline and duplicates the loader in each rank.
        script = textwrap.dedent("""
            import importlib.abc
            import sys

            class RejectDenseLoader(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == "distributed_runtime.model" or fullname.split(".")[0] == "transformers":
                        raise AssertionError(f"cell entrypoint imported dense loader: {fullname}")
                    return None

            sys.meta_path.insert(0, RejectDenseLoader())
            from distributed_runtime import cell_stage, external_cell, cell_member_cli
            assert callable(cell_stage._cell_member_main)
            assert callable(external_cell.run_external_cell_member)
            assert callable(cell_member_cli.main)
        """)
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
