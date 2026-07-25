from __future__ import annotations

import unittest

from distributed_runtime.installed_stage_canary import (
    SCHEMA,
    run_installed_stage_canary,
)
from distributed_runtime.model_adapters import ADAPTER_REGISTRY_ID


class InstalledStageCanaryTests(unittest.TestCase):
    def test_real_stage_runner_kv_and_batching_are_deterministic(self) -> None:
        evidence = run_installed_stage_canary()

        self.assertEqual(evidence["schema"], SCHEMA)
        self.assertIs(evidence["ok"], True)
        self.assertEqual(evidence["engine"], "python-torch")
        self.assertEqual(evidence["adapter"], "transformers-llama-v1")
        self.assertEqual(evidence["adapterRegistryId"], ADAPTER_REGISTRY_ID)
        self.assertRegex(evidence["adapterContractId"], r"^sha256:[0-9a-f]{64}$")
        self.assertEqual(evidence["loader"], "selective-safetensors")
        self.assertEqual(evidence["batchSize"], 2)
        self.assertGreaterEqual(evidence["physicalBatchCalls"], 2)
        self.assertGreaterEqual(evidence["physicalBatchItems"], 4)
        self.assertEqual(evidence["sequenceTokens"], 3)
        self.assertGreater(evidence["kvBytes"], 0)
        self.assertEqual(evidence["copiedKvBytes"], evidence["kvBytes"])
        self.assertEqual(evidence["batchQueueBatches"], 1)
        self.assertRegex(evidence["outputTokenSha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(
            evidence["parity"],
            {
                "sequentialVsBatch": True,
                "fork": True,
                "rollback": True,
            },
        )


if __name__ == "__main__":
    unittest.main()
