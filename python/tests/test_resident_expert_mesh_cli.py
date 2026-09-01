from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

from distributed_runtime.resident_expert_mesh_cli import (
    REPORT_SCHEMA,
    execute_cli,
    simulate_at_rtt,
    simulate_document,
)


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "config" / "resident-expert-mesh-4gb.example.json"


def example_document() -> dict:
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))


class ResidentExpertMeshCliTests(unittest.TestCase):
    def test_bounded_exact_repair_recovers_a_heterogeneous_greedy_dead_end(
        self,
    ) -> None:
        document = example_document()
        document["model"] = {
            "sparseLayers": 1,
            "expertsPerLayer": 4,
            "expertBytes": 3,
            "activationElements": 1,
            "activationDtypeBytes": 1,
            "topK": 2,
        }
        document["wave"] = {
            "positions": 3,
            "positionsPerSequence": 3,
            "concurrentSequences": 1,
            "committedTokens": 1,
            "routingUnionExpertsPerLayer": 4,
        }
        document["coordinator"].update(
            usableVramBytes=100,
            reservedVramBytes=0,
            weightBufferBytes=3,
            ramToGpuGbytesPerSecond=0.000001,
            expertComputeMsPerToken=1_000,
        )
        document["owners"] = document["owners"][:3]
        for owner, capacity in zip(document["owners"], (5, 11, 23)):
            owner.update(usableVramBytes=capacity, reservedVramBytes=0)

        report = simulate_document(document)

        # Greedy puts the repeated (0, 1) edge on the 23-byte owner first and
        # then cannot place both remaining experts.  The feasible exact repair
        # is 3 experts / 1 expert: 21 and 9 peak bytes respectively.
        self.assertEqual(report["placement"]["capacityLowerBoundOwners"], 1)
        self.assertEqual(report["placement"]["ownersRequired"], 2)
        self.assertEqual(
            report["placement"]["strategy"],
            "trace-coactivation-bounded-exact-repair/1",
        )
        self.assertTrue(report["placement"]["minimumOwnerCountProven"])
        self.assertGreater(report["placement"]["searchStatesEvaluated"], 0)
        self.assertFalse(report["placement"]["searchLimitReached"])
        self.assertEqual(
            report["placement"]["residentBytesByOwner"],
            {"home-owner-02": 3, "home-owner-03": 9},
        )
        self.assertEqual(
            report["primary"]["peakVramBytesByNode"]["home-owner-02"],
            9,
        )
        self.assertEqual(
            report["primary"]["peakVramBytesByNode"]["home-owner-03"],
            21,
        )

    def test_tight_topk_split_reserves_only_each_owners_real_route(self) -> None:
        document = example_document()
        document["model"] = {
            "sparseLayers": 1,
            "expertsPerLayer": 4,
            "expertBytes": 10,
            "activationElements": 1,
            "activationDtypeBytes": 1,
            "topK": 2,
        }
        document["wave"] = {
            "positions": 2,
            "positionsPerSequence": 2,
            "concurrentSequences": 1,
            "committedTokens": 1,
            "routingUnionExpertsPerLayer": 4,
        }
        document["coordinator"].update(
            usableVramBytes=100,
            reservedVramBytes=0,
            weightBufferBytes=10,
            ramToGpuGbytesPerSecond=0.000001,
            expertComputeMsPerToken=1_000,
        )
        document["owners"] = document["owners"][:4]
        for owner in document["owners"]:
            owner.update(usableVramBytes=13, reservedVramBytes=0)

        report = simulate_document(document)

        self.assertEqual(report["placement"]["ownersRequired"], 4)
        for owner_index in range(1, 5):
            self.assertEqual(
                report["primary"]["peakVramBytesByNode"][
                    f"home-owner-{owner_index:02d}"
                ],
                13,
            )

    def test_example_places_every_expert_and_reports_partial_ceiling(self) -> None:
        report = simulate_document(example_document())

        self.assertEqual(report["schema"], REPORT_SCHEMA)
        self.assertEqual(report["model"]["sparseLayers"], 12)
        self.assertEqual(report["model"]["totalExperts"], 144)
        self.assertEqual(report["model"]["activationBytesPerPosition"], 16_384)
        self.assertEqual(report["placement"]["candidateOwnerCount"], 8)
        self.assertEqual(report["placement"]["capacityLowerBoundOwners"], 6)
        self.assertEqual(report["placement"]["ownersRequired"], 7)
        self.assertEqual(report["placement"]["residentReplicaCount"], 144)
        self.assertEqual(
            report["placement"]["strategy"],
            "trace-coactivation-disjoint-greedy/1",
        )
        self.assertTrue(report["placement"]["minimumOwnerCountProven"])
        self.assertEqual(report["placement"]["searchStatesEvaluated"], 0)
        self.assertFalse(report["placement"]["searchLimitReached"])
        self.assertEqual(
            report["placement"]["ownerSubsetStrategy"],
            "exact-enumeration/1",
        )
        self.assertEqual(report["placement"]["ownerSubsetsEvaluated"], 8)
        self.assertEqual(report["placement"]["ownerSubsetTotal"], 8)
        self.assertTrue(
            report["placement"]["ownerSubsetEnumerationComplete"]
        )
        self.assertEqual(
            sum(report["placement"]["residentBytesByOwner"].values()),
            report["model"]["totalResidentExpertBytes"],
        )
        self.assertTrue(report["wave"]["routingCoverageVerified"])
        self.assertEqual(report["wave"]["positions"], 16)
        self.assertEqual(report["wave"]["positionsPerSequence"], 4)
        self.assertEqual(report["wave"]["concurrentSequences"], 4)
        self.assertEqual(report["wave"]["routingUnionExpertsPerLayer"], 12)
        self.assertEqual(
            report["wave"]["throughputNormalization"],
            "per-sequence-committed-tokens",
        )
        self.assertEqual(report["primary"]["roundTripMs"], 15)
        self.assertGreater(report["primary"]["tokensPerSecondPerSequence"], 0)
        self.assertNotIn("aggregateTokensPerSecond", report["primary"])
        self.assertGreater(report["primary"]["activationRoundTripBytesPerWave"], 0)
        self.assertGreater(report["primary"]["routeMetadataBytesPerWave"], 0)
        self.assertTrue(any(report["primary"]["coalescedOwnersByLayer"]))
        self.assertGreater(report["primary"]["hostDeviceStagingBytesPerWave"], 0)
        self.assertTrue(report["primary"]["transportCalibrationRequired"])
        self.assertTrue(report["primary"]["workspaceCalibrationRequired"])
        self.assertEqual(report["primary"]["maxInflightOwnerRpcs"], 32)
        scheduling = report["primary"]["schedulingByLayer"]
        self.assertEqual(len(scheduling), report["model"]["sparseLayers"])
        self.assertEqual(
            [item["layer"] for item in scheduling],
            list(range(report["model"]["sparseLayers"])),
        )
        for item in scheduling:
            self.assertEqual(item["maxInflightOwnerRpcs"], 32)
            self.assertAlmostEqual(
                item["exposedMs"],
                max(
                    item["ownerScheduledMakespanMs"],
                    item["coordinatorNicLowerBoundMs"],
                )
                + item["coordinatorOverheadMs"],
            )
        self.assertAlmostEqual(
            report["primary"]["exposedMsPerWave"],
            sum(item["exposedMs"] for item in scheduling),
        )
        self.assertAlmostEqual(
            report["primary"]["ownerScheduledMakespanMsPerWave"],
            sum(item["ownerScheduledMakespanMs"] for item in scheduling),
        )
        self.assertAlmostEqual(
            report["primary"]["coordinatorNicLowerBoundMsPerWave"],
            sum(item["coordinatorNicLowerBoundMs"] for item in scheduling),
        )
        self.assertAlmostEqual(
            report["primary"]["coordinatorOverheadMsPerWave"],
            sum(item["coordinatorOverheadMs"] for item in scheduling),
        )
        self.assertEqual(
            report["primary"]["coordinationCalibrationRequired"],
            any(item["coordinationCalibrationRequired"] for item in scheduling),
        )
        self.assertIn("mesh-root", report["primary"]["peakVramBytesByNode"])
        self.assertTrue(
            set(report["primary"]["activeRemoteOwnerIds"]).issubset(
                report["primary"]["peakVramBytesByNode"]
            )
        )
        self.assertGreater(report["primary"]["weightAvoidedBytesPerWave"], 0)
        self.assertEqual(
            report["comparison"]["macroWaveColdTokensPerSecondPerSequence"],
            1.22,
        )
        self.assertTrue(report["limitations"]["isPartialUpperBound"])
        self.assertFalse(report["limitations"]["isGlmPrediction"])
        self.assertIn("attention", report["limitations"]["excludes"])
        self.assertIn("stage-to-stage-wan", report["limitations"]["excludes"])

    def test_sensitivity_rows_are_real_simulator_calls(self) -> None:
        document = example_document()
        report = simulate_document(document)
        self.assertEqual(
            [row["roundTripMs"] for row in report["sensitivity"]],
            [1, 5, 15, 30, 60],
        )
        for row in report["sensitivity"]:
            direct = simulate_at_rtt(document, row["roundTripMs"])
            self.assertEqual(row, direct)

        throughput = [
            row["tokensPerSecondPerSequence"] for row in report["sensitivity"]
        ]
        expected = [35.817440, 22.769002, 11.916201, 6.948336, 3.789036]
        for actual, expected_value in zip(throughput, expected):
            self.assertAlmostEqual(actual, expected_value, places=6)
        self.assertEqual(throughput, sorted(throughput, reverse=True))
        self.assertEqual(
            report["sensitivity"][0]["activationRoundTripBytesPerWave"],
            9 * 1024**2,
        )
        self.assertGreater(
            report["sensitivity"][-1]["weightLoadedBytesPerWave"],
            report["sensitivity"][0]["weightLoadedBytesPerWave"],
        )

    def test_input_order_does_not_change_deterministic_placement_or_hash(self) -> None:
        original = example_document()
        reordered = copy.deepcopy(original)
        reordered["owners"].reverse()
        reordered["sensitivityRttMs"].reverse()

        self.assertEqual(simulate_document(original), simulate_document(reordered))

    def test_heterogeneous_owners_prefer_the_faster_capacity(self) -> None:
        document = example_document()
        slow = document["owners"][0]
        slow["roundTripMs"] = 500
        slow["expertComputeMsPerToken"] = 50
        fast = document["owners"][-1]
        fast["roundTripMs"] = 1
        fast["bandwidthMbps"] = 2_000
        fast["egressBandwidthMbps"] = 1_500
        fast["ingressBandwidthMbps"] = 2_500
        fast["expertComputeMsPerToken"] = 0.1
        fast["supportsExactInputCoalescing"] = False
        fast["rowIndexBytesPerAssignment"] = 100

        report = simulate_document(document)

        selected = report["placement"]["ownerIds"]
        self.assertIn("home-owner-08", selected)
        self.assertNotIn("home-owner-01", selected)
        declared = report["primary"]["transportAssumption"]["owners"]
        self.assertFalse(
            declared["home-owner-08"]["exactInputCoalescing"]
        )
        self.assertEqual(
            declared["home-owner-08"]["egressBandwidthMbps"],
            1_500.0,
        )

    def test_smaller_fast_owner_can_replace_larger_slow_prefix_owner(self) -> None:
        document = example_document()
        document["model"] = {
            "sparseLayers": 1,
            "expertsPerLayer": 1,
            "expertBytes": 1,
            "activationElements": 1,
            "activationDtypeBytes": 1,
            "topK": 1,
        }
        document["wave"] = {
            "positions": 1,
            "positionsPerSequence": 1,
            "concurrentSequences": 1,
            "committedTokens": 1,
            "routingUnionExpertsPerLayer": 1,
        }
        document["coordinator"].update(
            usableVramBytes=100,
            reservedVramBytes=0,
            weightBufferBytes=1,
            ramToGpuGbytesPerSecond=0.000001,
            expertComputeMsPerToken=1_000,
        )
        document["owners"] = document["owners"][:2]
        document["owners"][0].update(
            id="slow-large",
            usableVramBytes=5,
            reservedVramBytes=0,
            roundTripMs=500,
            expertComputeMsPerToken=50,
        )
        document["owners"][1].update(
            id="fast-tight",
            usableVramBytes=4,
            reservedVramBytes=0,
            roundTripMs=1,
            expertComputeMsPerToken=0.1,
        )

        report = simulate_document(document)

        self.assertEqual(report["placement"]["capacityLowerBoundOwners"], 1)
        self.assertEqual(report["placement"]["ownersRequired"], 1)
        self.assertEqual(report["placement"]["ownerIds"], ["fast-tight"])
        self.assertTrue(report["placement"]["minimumOwnerCountProven"])
        self.assertEqual(
            report["placement"]["ownerSubsetStrategy"],
            "exact-enumeration/1",
        )
        self.assertEqual(report["placement"]["ownerSubsetsEvaluated"], 2)
        self.assertEqual(report["placement"]["ownerSubsetTotal"], 2)
        self.assertTrue(
            report["placement"]["ownerSubsetEnumerationComplete"]
        )
        self.assertLess(report["primary"]["exposedMsPerWave"], 2)

    def test_subset_score_uses_v1_when_coalesced_metadata_is_larger(self) -> None:
        document = example_document()
        document["model"] = {
            "sparseLayers": 1,
            "expertsPerLayer": 2,
            "expertBytes": 1,
            "activationElements": 1,
            "activationDtypeBytes": 1,
            "topK": 2,
        }
        document["wave"] = {
            "positions": 1,
            "positionsPerSequence": 1,
            "concurrentSequences": 1,
            "committedTokens": 1,
            "routingUnionExpertsPerLayer": 2,
        }
        document["coordinator"].update(
            usableVramBytes=100,
            reservedVramBytes=0,
            weightBufferBytes=1,
            expertComputeMsPerToken=1_000,
        )
        template = document["owners"][0]
        fast = copy.deepcopy(template)
        fast.update(
            id="fast-bad-metadata",
            usableVramBytes=8,
            reservedVramBytes=0,
            roundTripMs=1,
            bandwidthMbps=1_000,
            egressBandwidthMbps=1_000,
            ingressBandwidthMbps=1_000,
            expertComputeMsPerToken=0.1,
            supportsExactInputCoalescing=True,
            rowIndexBytesPerAssignment=1_000_000_000,
        )
        slow = copy.deepcopy(template)
        slow.update(
            id="slow-v1",
            usableVramBytes=8,
            reservedVramBytes=0,
            roundTripMs=10,
            bandwidthMbps=1_000,
            egressBandwidthMbps=1_000,
            ingressBandwidthMbps=1_000,
            expertComputeMsPerToken=10,
            supportsExactInputCoalescing=False,
            rowIndexBytesPerAssignment=4,
        )
        document["owners"] = [fast, slow]

        report = simulate_document(document)

        self.assertEqual(report["placement"]["ownerIds"], ["fast-bad-metadata"])
        self.assertEqual(report["primary"]["coalescedOwnersByLayer"], [[]])
        self.assertEqual(report["primary"]["activationRequestBytesPerWave"], 2)
        self.assertEqual(report["primary"]["activationResponseBytesPerWave"], 2)
        self.assertEqual(report["primary"]["routeMetadataBytesPerWave"], 0)
        self.assertEqual(report["primary"]["transportPayloadBytesPerWave"], 4)
        self.assertEqual(report["primary"]["hostDeviceStagingBytesPerWave"], 8)
        self.assertLess(report["primary"]["exposedMsPerWave"], 2)

    def test_large_owner_subset_space_uses_honest_deterministic_fallback(
        self,
    ) -> None:
        document = example_document()
        document["model"] = {
            "sparseLayers": 1,
            "expertsPerLayer": 4,
            "expertBytes": 10,
            "activationElements": 1,
            "activationDtypeBytes": 1,
            "topK": 1,
        }
        document["wave"] = {
            "positions": 4,
            "positionsPerSequence": 4,
            "concurrentSequences": 1,
            "committedTokens": 1,
            "routingUnionExpertsPerLayer": 4,
        }
        document["coordinator"].update(
            usableVramBytes=100,
            reservedVramBytes=0,
            weightBufferBytes=10,
        )
        template = document["owners"][0]
        document["owners"] = []
        for owner_index in range(12):
            owner = copy.deepcopy(template)
            owner.update(
                id=f"owner-{owner_index:02d}",
                usableVramBytes=13,
                reservedVramBytes=0,
            )
            document["owners"].append(owner)

        report = simulate_document(document)

        self.assertEqual(report["placement"]["ownersRequired"], 4)
        self.assertTrue(report["placement"]["minimumOwnerCountProven"])
        self.assertEqual(
            report["placement"]["ownerSubsetStrategy"],
            "deterministic-capacity-speed-neighborhood/1",
        )
        self.assertEqual(report["placement"]["ownerSubsetTotal"], 495)
        self.assertLess(
            report["placement"]["ownerSubsetsEvaluated"],
            report["placement"]["ownerSubsetTotal"],
        )
        self.assertFalse(
            report["placement"]["ownerSubsetEnumerationComplete"]
        )
        self.assertTrue(report["placement"]["searchLimitReached"])

        reordered = copy.deepcopy(document)
        reordered["owners"].reverse()
        self.assertEqual(report, simulate_document(reordered))

    def test_version_shape_routing_and_capacity_fail_closed(self) -> None:
        unknown = example_document()
        unknown["unexpected"] = True
        with self.assertRaisesRegex(ValueError, "input keys are invalid"):
            simulate_document(unknown)

        schema = example_document()
        schema["schema"] = "gdlp-resident-expert-mesh-simulation/2"
        with self.assertRaisesRegex(ValueError, "schema is not supported"):
            simulate_document(schema)

        uncovered = example_document()
        uncovered["model"]["expertsPerLayer"] = 40
        uncovered["wave"]["routingUnionExpertsPerLayer"] = 33
        with self.assertRaisesRegex(ValueError, "cannot be covered"):
            simulate_document(uncovered)

        insufficient = example_document()
        for owner in insufficient["owners"]:
            owner["usableVramBytes"] = owner["reservedVramBytes"] + 128 * 1024**2
        with self.assertRaisesRegex(ValueError, "cannot hold every"):
            simulate_document(insufficient)

        search_limited = example_document()
        search_limited["model"] = {
            "sparseLayers": 1,
            "expertsPerLayer": 65,
            "expertBytes": 1,
            "activationElements": 1,
            "activationDtypeBytes": 1,
            "topK": 2,
        }
        search_limited["wave"] = {
            "positions": 33,
            "positionsPerSequence": 33,
            "concurrentSequences": 1,
            "committedTokens": 1,
            "routingUnionExpertsPerLayer": 65,
        }
        search_limited["coordinator"].update(
            usableVramBytes=100,
            reservedVramBytes=0,
            weightBufferBytes=1,
        )
        search_limited["owners"] = search_limited["owners"][:1]
        search_limited["owners"][0].update(
            usableVramBytes=70,
            reservedVramBytes=0,
        )
        with self.assertRaisesRegex(ValueError, "repair was not completed"):
            simulate_document(search_limited)

    def test_cli_emits_stable_json_and_honest_markdown(self) -> None:
        first = execute_cli(
            [str(EXAMPLE), "--format", "json"],
            ROOT,
        )
        second = execute_cli(
            [str(EXAMPLE), "--format=json"],
            ROOT,
        )
        self.assertEqual(first, second)
        self.assertEqual(json.loads(first)["schema"], REPORT_SCHEMA)

        markdown = execute_cli(
            [str(EXAMPLE), "--format", "markdown"],
            ROOT,
        )
        self.assertIn("## Actual simulator sensitivity", markdown)
        self.assertIn("Candidate/required owners: **8 / 7**", markdown)
        self.assertIn("**16** = **4** × **4 chats**", markdown)
        self.assertIn("1.22 tok/s", markdown)
        self.assertIn("This is not a GLM prediction", markdown)
        self.assertIn("ceiling that times only sparse experts", markdown)
        self.assertIn("## Scheduling and coordination", markdown)
        self.assertIn("Maximum concurrent owner RPCs: **32**", markdown)
        self.assertIn("Coordination calibration required: **yes**", markdown)
        # Windows PowerShell can expose a CP1252 stdout. Keep the human-readable
        # report encodable there so the real CLI does not fail after simulation.
        markdown.encode("cp1252")


if __name__ == "__main__":
    unittest.main()
