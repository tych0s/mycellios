import { describe, expect, it } from "vitest";
import { trustedExpertManifestPath } from "../src/mobile/expert-manifest-url.js";

describe("mobile expert manifest URL policy", () => {
  const artifactId = "a".repeat(64);
  it("accepts only the exact same-origin manifest route for the offered artifact", () => {
    expect(trustedExpertManifestPath(
      artifactId,
      `/mobile/v1/experts/${artifactId}/manifest`,
    )).toBe(`/mobile/v1/experts/${artifactId}/manifest`);

    for (const candidate of [
      `https://attacker.example/mobile/v1/experts/${artifactId}/manifest`,
      `/mobile/v1/experts/${"b".repeat(64)}/manifest`,
      `/mobile/v1/experts/${artifactId}/manifest?redirect=https://attacker.example`,
      `/mobile/v1/experts/${artifactId}/manifest#fragment`,
    ]) {
      expect(() => trustedExpertManifestPath(artifactId, candidate)).toThrow(
        "untrusted expert manifest URL",
      );
    }
    expect(() => trustedExpertManifestPath("not-a-digest", "/mobile/v1/experts/not-a-digest/manifest"))
      .toThrow("invalid expert artifact id");
  });
});
