import { describe, expect, it } from "vitest";
import { verifyFinalWebNativeTraceability } from "../scripts/verify-final-web-native-traceability.js";

describe("final web/native traceability", () => {
  it("links every FR and AC to real tasks, gates and evidence classes", async () => {
    await expect(verifyFinalWebNativeTraceability()).resolves.toEqual({
      requirements: 16,
      acceptanceCriteria: 20,
      checkLinks: 45,
      consistencyChecks: 2,
    });
  });
});
