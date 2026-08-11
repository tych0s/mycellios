import { describe, expect, it } from "vitest";

import { runModelCertificationHarness } from "../src/benchlab/model-certification-harness.js";

const sha = (character: string) => `sha256:${character.repeat(64)}`;
const input = () => ({
  distributionManifestId: sha("a"), componentManifestId: sha("b"), adapterContractId: sha("c"),
  sourceRevision: "d".repeat(40), evidenceClass: "automatic", measuredAt: "2026-08-10T12:00:00.000Z",
  hardware: { fingerprint: null, backend: "cpu", deviceFamily: "generic-x64", memoryBytes: 16_000_000_000 },
  codecs: ["fp16/1"], performance: { ttftMs: 20, tpotMs: 8, peakMemoryBytes: 2_000_000_000 },
  cases: [
    { id: "case-b", prompt: "Private second prompt", seed: 2, maximumTokens: 4, referenceTokens: [3, 4], candidateTokens: [3, 4] },
    { id: "case-a", prompt: "Private first prompt", seed: 1, maximumTokens: 4, referenceTokens: [1, 2], candidateTokens: [1, 2] },
  ],
});

describe("model certification harness", () => {
  it("seals deterministic redacted evidence without promoting automatic checks", () => {
    const first = runModelCertificationHarness(input());
    const secondInput = input();
    secondInput.cases.reverse();
    const second = runModelCertificationHarness(secondInput);
    expect(first).toEqual(second);
    expect(first.classification).toBe("software-candidate");
    expect(first.receiptId).toMatch(/^sha256:[0-9a-f]{64}$/);
    expect(JSON.stringify(first)).not.toContain("Private");
  });

  it("records the exact first mismatch", () => {
    const value = input();
    value.cases[0]!.candidateTokens = [3, 9];
    const receipt = runModelCertificationHarness(value);
    expect(receipt.exact).toBe(false);
    expect(receipt.cases.find(({ id }) => id === "case-b")).toMatchObject({ firstMismatchIndex: 1, exact: false });
  });

  it("requires a hardware fingerprint for physical evidence", () => {
    expect(() => runModelCertificationHarness({ ...input(), evidenceClass: "physical" })).toThrow("model_certification_physical_hardware_fingerprint_is_required");
  });
});
