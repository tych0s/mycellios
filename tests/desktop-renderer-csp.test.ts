import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

describe("desktop renderer content security policy", () => {
  it("allows public telemetry and bundled data fonts", () => {
    const html = readFileSync(resolve("src/renderer/index.html"), "utf8");

    expect(html).toContain("font-src 'self' data:");
    expect(html).toContain("https://www.mycellios.com");
    expect(html).toContain("wss://www.mycellios.com");
  });
});
