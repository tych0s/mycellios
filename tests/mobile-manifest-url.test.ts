import { describe, expect, it } from "vitest";
import { trustedExpertManifestUrl } from "../src/mobile/runtime.js";

describe("mobile expert manifest URL policy", () => {
  const artifactId = "a".repeat(64);
  const origin = "https://node.example.test";

  it("accepts only the exact same-origin manifest route for the offered artifact", () => {
    expect(trustedExpertManifestUrl(
      artifactId,
      `/mobile/v1/experts/${artifactId}/manifest`,
      origin,
    ).href).toBe(`${origin}/mobile/v1/experts/${artifactId}/manifest`);

    for (const candidate of [
      `https://attacker.example/mobile/v1/experts/${artifactId}/manifest`,
      `/mobile/v1/experts/${"b".repeat(64)}/manifest`,
      `/mobile/v1/experts/${artifactId}/manifest?redirect=https://attacker.example`,
      `/mobile/v1/experts/${artifactId}/manifest#fragment`,
    ]) {
      expect(() => trustedExpertManifestUrl(artifactId, candidate, origin)).toThrow(
        "untrusted expert manifest URL",
      );
    }
  });
});
