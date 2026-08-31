import { generateKeyPairSync, type KeyObject } from "node:crypto";
import { describe, expect, it } from "vitest";

import {
  EXECUTION_RECEIPT_SCHEMA,
  EXECUTION_RECEIPT_ENVELOPE_SCHEMA,
  STAGE_EXECUTION_OBSERVATION_SCHEMA,
  buildExecutionReceipt,
  encodeExecutionReceiptEnvelope,
  signStageExecutionObservation,
  signExecutionReceiptEnvelope,
  verifyExecutionReceipt,
  verifyExecutionReceiptEnvelope,
  verifyExecutionReceiptEnvelopeBytes,
  type StageExecutionObservation,
} from "../src/contracts/distributed-execution-receipt.js";
import {
  classifyRecoveryFailure,
  type RecoveryEvent,
} from "../src/contracts/recovery-outcome.js";

const digest = (character: string) => `sha256:${character.repeat(64)}` as const;
const executionId = "execution-1";
const requestIdHash = digest("1");
const engineDescriptorDigest = digest("2");
const artifactManifestDigest = digest("3");

function signedStage(
  stage: number,
  inputRoot: `sha256:${string}`,
  outputRoot: `sha256:${string}`,
  keyId: string,
  privateKey: KeyObject,
): StageExecutionObservation {
  return signStageExecutionObservation(
    {
      schema: STAGE_EXECUTION_OBSERVATION_SCHEMA,
      executionId,
      requestIdHash,
      stageId: `stage-${stage}`,
      nodeIdHash: digest(String(stage + 4)),
      topologyGeneration: 7,
      layerStart: stage * 2,
      layerEnd: stage * 2 + 2,
      engineDescriptorDigest,
      artifactManifestDigest,
      inputRoot,
      outputRoot,
      counters: {
        frames: 10,
        inputBytes: 1_000,
        outputBytes: 1_000,
        computeMs: 12.5,
      },
      outcome: "completed",
    },
    { keyId, privateKey },
  );
}

describe("execution receipt chain", () => {
  const firstKeys = generateKeyPairSync("ed25519");
  const secondKeys = generateKeyPairSync("ed25519");
  const envelopeKeys = generateKeyPairSync("ed25519");
  const requestRoot = digest("a");
  const middleRoot = digest("b");
  const resultRoot = digest("c");

  function receipt(stages?: StageExecutionObservation[], recovery: RecoveryEvent[] = []) {
    return buildExecutionReceipt({
      schema: EXECUTION_RECEIPT_SCHEMA,
      sourceSha: "d".repeat(40),
      sourceId: digest("d"),
      createdAt: "2026-08-10T00:00:00.000Z",
      evidenceClass: "wan-direct",
      executionId,
      requestIdHash,
      modelDigest: digest("e"),
      engineDescriptorDigest,
      artifactManifestDigest,
      topologyDigest: digest("f"),
      topologyGeneration: 7,
      transport: "direct",
      totalLayers: 4,
      requestRoot,
      resultRoot,
      outputTokenHash: digest("0"),
      stages: stages ?? [
        signedStage(0, requestRoot, middleRoot, "stage-key-0", firstKeys.privateKey),
        signedStage(1, middleRoot, resultRoot, "stage-key-1", secondKeys.privateKey),
      ],
      recovery,
      metrics: {
        ttftMs: 250,
        tpotMs: 30,
        outputTokens: 32,
        acceptedDraftTokens: 10,
        proposedDraftTokens: 16,
      },
      redactionVersion: 1,
    });
  }

  it("verifies signatures, complete coverage and chained roots", () => {
    const value = receipt();
    const verified = verifyExecutionReceipt(value, {
      pinnedStageKeys: new Map([
        ["stage-key-0", firstKeys.publicKey],
        ["stage-key-1", secondKeys.publicKey],
      ]),
    });
    expect(verified.receiptId).toMatch(/^sha256:[0-9a-f]{64}$/);
    expect(verified.stages).toHaveLength(2);
  });

  it("seals classified recovery lifecycle evidence into the receipt identity", () => {
    const recovery = classifyRecoveryFailure({
      generation: 7,
      activeGeneration: 7,
      role: "middle",
      failureClass: "process-exit",
      checkpointKind: "visible-token-prefix",
      checkpointCompatible: true,
      compatibleStandbyAvailable: true,
      visibleTokens: 9,
      downtimeMs: 330,
      discardedWaves: 2,
      discardedBytes: 8_192,
    });
    const value = receipt(undefined, [recovery]);
    expect(value.recovery).toEqual([expect.objectContaining({
      outcome: "exact-replay",
      replayScope: "visible-token-prefix",
      replayedTokens: 9,
      discardedBytes: 8_192,
    })]);
    expect(receipt().receiptId).not.toBe(value.receiptId);
    expect(verifyExecutionReceipt(value, {
      pinnedStageKeys: new Map([
        ["stage-key-0", firstKeys.publicKey],
        ["stage-key-1", secondKeys.publicKey],
      ]),
    }).receiptId).toBe(value.receiptId);
  });

  it("rejects gaps, overlaps and broken root chains", () => {
    const first = signedStage(
      0,
      requestRoot,
      middleRoot,
      "stage-key-0",
      firstKeys.privateKey,
    );
    const brokenCoverage = signedStage(
      1,
      middleRoot,
      resultRoot,
      "stage-key-1",
      secondKeys.privateKey,
    );
    expect(() => receipt([{ ...brokenCoverage, layerStart: 3 }])).toThrow(
      "execution_receipt_layer_coverage_is_invalid",
    );
    expect(() =>
      receipt([first, { ...brokenCoverage, inputRoot: digest("9") }]),
    ).toThrow("execution_receipt_stage_chain_is_invalid");
  });

  it("rejects failed and zero-work stages before settlement", () => {
    const first = signedStage(
      0,
      requestRoot,
      middleRoot,
      "stage-key-0",
      firstKeys.privateKey,
    );
    const second = signedStage(
      1,
      middleRoot,
      resultRoot,
      "stage-key-1",
      secondKeys.privateKey,
    );
    expect(() => receipt([{ ...first, outcome: "failed" }, second])).toThrow(
      "execution_receipt_contains_failed_stage",
    );
    expect(() => receipt([
      {
        ...first,
        counters: { ...first.counters, frames: 0 },
      },
      second,
    ])).toThrow("execution_receipt_stage_attests_zero_work");
    expect(() => receipt([
      {
        ...first,
        counters: { ...first.counters, inputBytes: 0 },
      },
      second,
    ])).toThrow("execution_receipt_stage_attests_zero_work");
    expect(() => receipt([
      {
        ...first,
        counters: { ...first.counters, outputBytes: 0 },
      },
      second,
    ])).toThrow("execution_receipt_stage_attests_zero_work");
  });

  it("rejects tampering even when the envelope digest is recomputed", () => {
    const value = receipt();
    const tamperedStage = {
      ...value.stages[0]!,
      counters: { ...value.stages[0]!.counters, computeMs: 0 },
    };
    const rebuilt = receipt([tamperedStage, value.stages[1]!]);
    expect(() =>
      verifyExecutionReceipt(rebuilt, {
        pinnedStageKeys: new Map([
          ["stage-key-0", firstKeys.publicKey],
          ["stage-key-1", secondKeys.publicKey],
        ]),
      }),
    ).toThrow("execution_receipt_stage_signature_is_invalid:stage-0");
  });

  it("rejects unknown stage keys and noncanonical fields", () => {
    const value = receipt();
    expect(() =>
      verifyExecutionReceipt(value, {
        pinnedStageKeys: new Map([["stage-key-0", firstKeys.publicKey]]),
      }),
    ).toThrow("execution_receipt_stage_key_is_unknown:stage-key-1");
    expect(() => verifyExecutionReceipt({ ...value, publicIp: "203.0.113.1" }, {
      pinnedStageKeys: new Map(),
    })).toThrow();
  });

  it("signs the complete receipt envelope against its GDLP wire identity", () => {
    const envelope = signExecutionReceiptEnvelope(receipt(), {
      keyId: "coordinator-receipt-key-1",
      privateKey: envelopeKeys.privateKey,
      deploymentGeneration: 7,
      routeDigest: "1".repeat(32),
      waveStrategyDigest: "2".repeat(32),
      waveArtifactDigest: "3".repeat(32),
    });
    expect(envelope.schema).toBe(EXECUTION_RECEIPT_ENVELOPE_SCHEMA);
    expect(
      verifyExecutionReceiptEnvelope(envelope, {
        pinnedEnvelopeKeys: new Map([
          ["coordinator-receipt-key-1", envelopeKeys.publicKey],
        ]),
        pinnedStageKeys: new Map([
          ["stage-key-0", firstKeys.publicKey],
          ["stage-key-1", secondKeys.publicKey],
        ]),
      }),
    ).toEqual(envelope);
    const wire = encodeExecutionReceiptEnvelope(envelope);
    expect(
      verifyExecutionReceiptEnvelopeBytes(wire, {
        pinnedEnvelopeKeys: new Map([
          ["coordinator-receipt-key-1", envelopeKeys.publicKey],
        ]),
        pinnedStageKeys: new Map([
          ["stage-key-0", firstKeys.publicKey],
          ["stage-key-1", secondKeys.publicKey],
        ]),
      }),
    ).toEqual(envelope);
  });

  it("rejects an unknown envelope key and any wire-identity tampering", () => {
    const envelope = signExecutionReceiptEnvelope(receipt(), {
      keyId: "coordinator-receipt-key-1",
      privateKey: envelopeKeys.privateKey,
      deploymentGeneration: 7,
      routeDigest: "1".repeat(32),
      waveStrategyDigest: "2".repeat(32),
      waveArtifactDigest: "3".repeat(32),
    });
    const stageKeys = new Map([
      ["stage-key-0", firstKeys.publicKey],
      ["stage-key-1", secondKeys.publicKey],
    ]);
    expect(() =>
      verifyExecutionReceiptEnvelope(envelope, {
        pinnedEnvelopeKeys: new Map(),
        pinnedStageKeys: stageKeys,
      }),
    ).toThrow(
      "execution_receipt_envelope_key_is_unknown:coordinator-receipt-key-1",
    );
    expect(() =>
      verifyExecutionReceiptEnvelope(
        { ...envelope, waveArtifactDigest: "4".repeat(32) },
        {
          pinnedEnvelopeKeys: new Map([
            ["coordinator-receipt-key-1", envelopeKeys.publicKey],
          ]),
          pinnedStageKeys: stageKeys,
        },
      ),
    ).toThrow("execution_receipt_envelope_signature_is_invalid");
    expect(() =>
      verifyExecutionReceiptEnvelopeBytes(
        Buffer.from(` ${encodeExecutionReceiptEnvelope(envelope).toString()}`),
        {
          pinnedEnvelopeKeys: new Map([
            ["coordinator-receipt-key-1", envelopeKeys.publicKey],
          ]),
          pinnedStageKeys: stageKeys,
        },
      ),
    ).toThrow("execution_receipt_envelope_wire_json_is_not_canonical");
    expect(() =>
      signExecutionReceiptEnvelope(
        { ...receipt(), receiptId: digest("9") },
        {
          keyId: "coordinator-receipt-key-1",
          privateKey: envelopeKeys.privateKey,
          deploymentGeneration: 7,
          routeDigest: "1".repeat(32),
          waveStrategyDigest: "2".repeat(32),
          waveArtifactDigest: "3".repeat(32),
        },
      ),
    ).toThrow("execution_receipt_identity_mismatch");
  });
});
