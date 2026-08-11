import { describe, expect, it } from "vitest";
import {
  classifyRuntimeLinkFailure,
  runtimeLinkRecoveryEventsForReceipt,
  runtimeLinkFailureEvidenceSchema,
} from "../src/contracts/runtime-link-failure.js";

const relayFailure = {
  schema: "mycellios-runtime-link-failure/1",
  streamId: "stream-1",
  sourceNodeId: "node-a",
  destinationNodeId: "node-b",
  generation: 4,
  role: "relay",
  failureClass: "relay-link-lost",
  transportMode: "relay",
  checkpointKind: "stream-offset",
  sourceOffset: 128,
  destinationOffset: 96,
  observedAt: 1_000,
  reason: "authenticated_relay_disconnected",
} as const;

describe("runtime link failure evidence", () => {
  it("classifies offset-sealed relay loss as resume", () => {
    expect(classifyRuntimeLinkFailure(relayFailure, 4, 1_025)).toMatchObject({
      role: "relay",
      failureClass: "relay-link-lost",
      outcome: "resume",
      checkpointKind: "stream-offset",
      replayScope: "stream-offset",
      downtimeMs: 25,
      discardedBytes: 128,
    });
  });

  it("makes direct loss terminal without inventing an offset checkpoint", () => {
    expect(classifyRuntimeLinkFailure({
      ...relayFailure,
      role: "direct",
      failureClass: "direct-link-lost",
      transportMode: "direct",
      checkpointKind: "none",
    }, 4, 1_010)).toMatchObject({
      role: "direct",
      outcome: "terminal",
      replayScope: "none",
    });
  });

  it("rejects cross-role claims and unknown fields", () => {
    expect(() => runtimeLinkFailureEvidenceSchema.parse({
      ...relayFailure,
      role: "direct",
    })).toThrow(/runtime_link_failure_role_binding_is_invalid/);
    expect(() => runtimeLinkFailureEvidenceSchema.parse({
      ...relayFailure,
      recoveryToken: "must-not-leak",
    })).toThrow();
  });

  it("produces deterministic receipt events and rejects lifecycle duplicates", () => {
    const direct = {
      ...relayFailure,
      streamId: "stream-2",
      observedAt: 900,
      role: "direct",
      failureClass: "direct-link-lost",
      transportMode: "direct",
      checkpointKind: "none",
    } as const;
    expect(runtimeLinkRecoveryEventsForReceipt([relayFailure, direct], 4, 1_100)).toEqual([
      expect.objectContaining({ role: "direct", outcome: "terminal", downtimeMs: 200 }),
      expect.objectContaining({ role: "relay", outcome: "resume", downtimeMs: 100 }),
    ]);
    expect(() => runtimeLinkRecoveryEventsForReceipt(
      [relayFailure, structuredClone(relayFailure)],
      4,
      1_100,
    )).toThrow(/runtime_link_failure_receipt_duplicate/);
  });
});
