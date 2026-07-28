import { describe, expect, it } from "vitest";
import {
  issueWorkerSessionToken,
  verifyWorkerSessionToken,
} from "../src/coordinator/worker-session-token.js";

const claims = {
  workerId: "worker-public-1",
  identityKind: "device" as const,
  identityId: "desktop-public-1",
  credentialFingerprint: `sha256:${"a".repeat(64)}`,
};

describe("worker session tokens", () => {
  it("round-trips scoped claims without exposing the network secret", () => {
    const token = issueWorkerSessionToken("network-secret", claims);

    expect(token).not.toContain("network-secret");
    expect(verifyWorkerSessionToken("network-secret", token)).toEqual({
      schema: "mycellios-worker-session/1",
      ...claims,
    });
  });

  it("rejects tampering and tokens signed by another coordinator", () => {
    const token = issueWorkerSessionToken("network-secret", claims);
    const tampered = `${token.slice(0, -1)}${token.endsWith("a") ? "b" : "a"}`;

    expect(verifyWorkerSessionToken("network-secret", tampered)).toBeNull();
    expect(verifyWorkerSessionToken("another-secret", token)).toBeNull();
  });
});
