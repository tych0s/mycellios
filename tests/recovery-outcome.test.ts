import { describe, expect, it } from "vitest";
import parityVectors from "./fixtures/recovery-outcomes.json" with { type: "json" };
import {
  classifyRecoveryFailure,
  RECOVERY_EVENT_SCHEMA,
  recoveryEventSchema,
} from "../src/contracts/recovery-outcome.js";

const base = {
  generation: 7,
  activeGeneration: 7,
  failureClass: "process-exit" as const,
  checkpointKind: "visible-token-prefix" as const,
  checkpointCompatible: true,
  compatibleStandbyAvailable: true,
  visibleTokens: 4,
  downtimeMs: 125,
  discardedWaves: 2,
  discardedBytes: 4_096,
};

describe("recovery outcome classification", () => {
  it.each(parityVectors)("matches shared Python/TypeScript vector $name", ({ observation, event }) => {
    expect(classifyRecoveryFailure(observation)).toEqual(event);
    expect(recoveryEventSchema.parse(event)).toEqual(event);
  });

  it.each(["root", "head", "middle", "tail"] as const)(
    "classifies %s failure as exact visible-prefix replay",
    (role) => {
      expect(classifyRecoveryFailure({ ...base, role })).toMatchObject({
        schema: RECOVERY_EVENT_SCHEMA,
        role,
        outcome: "exact-replay",
        replayScope: "visible-token-prefix",
        replayedTokens: 4,
      });
    },
  );

  it.each(["direct", "relay", "coordinator"] as const)(
    "resumes %s transport from an authenticated offset",
    (role) => {
      expect(classifyRecoveryFailure({
        ...base,
        role,
        failureClass: role === "direct" ? "direct-link-lost"
          : role === "relay" ? "relay-link-lost" : "coordinator-lost",
        checkpointKind: "stream-offset",
      })).toMatchObject({ outcome: "resume", replayScope: "stream-offset" });
    },
  );

  it("safe-retries only an invisible request on a compatible standby", () => {
    expect(classifyRecoveryFailure({
      ...base, role: "root", checkpointKind: "none", visibleTokens: 0,
    })).toMatchObject({ outcome: "safe-retry", replayScope: "full-request" });
  });

  it.each([
    { failureClass: "checkpoint-corrupt", checkpointCompatible: false },
    { failureClass: "identity-mismatch", checkpointCompatible: false },
    { failureClass: "retry-exhausted" },
    { generation: 6 },
  ])("fails closed for corrupt, incompatible, exhausted or superseded state", (change) => {
    expect(classifyRecoveryFailure({ ...base, role: "tail", ...change }))
      .toMatchObject({ outcome: "terminal", replayScope: "none", replayedTokens: 0 });
  });

  it("rejects internally contradictory receipt events", () => {
    expect(() => recoveryEventSchema.parse({
      schema: RECOVERY_EVENT_SCHEMA,
      generation: 1,
      role: "root",
      failureClass: "process-exit",
      outcome: "terminal",
      checkpointKind: "none",
      replayScope: "visible-token-prefix",
      downtimeMs: 1,
      replayedTokens: 1,
      discardedWaves: 0,
      discardedBytes: 0,
    })).toThrow("recovery_terminal_cannot_replay");
  });
});
