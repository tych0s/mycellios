import { describe, expect, it } from "vitest";
import { resolveNodeSourceRevision } from "../src/node/build-identity.js";

describe("node build identity", () => {
  it("accepts an explicit CI-sealed source revision", async () => {
    await expect(resolveNodeSourceRevision({ MYCELLIOS_SOURCE_REVISION: "a".repeat(40) }, (() => { throw new Error("unused"); }) as never)).resolves.toBe("a".repeat(40));
  });

  it("rejects malformed explicit revisions instead of falling back", async () => {
    await expect(resolveNodeSourceRevision({ MYCELLIOS_SOURCE_REVISION: "not-a-sha" }, (() => "b".repeat(40)) as never)).rejects.toThrow();
  });
});
