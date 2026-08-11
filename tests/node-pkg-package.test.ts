import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { materializeMacPairingApp } from "../scripts/build-node-pkg.mjs";

const cleanup: string[] = [];
afterEach(async () => Promise.all(cleanup.splice(0).map((path) => rm(path, { recursive: true, force: true }))));

describe("native macOS pairing application", () => {
  it("owns enrollment documents and delegates to the protected native bootstrap", async () => {
    const root = await mkdtemp(join(tmpdir(), "mycellios-pkg-app-")); cleanup.push(root); await materializeMacPairingApp(root);
    const app = join(root, "Applications/Mycellios Pairing.app/Contents");
    const plist = await readFile(join(app, "Info.plist"), "utf8"), launcher = await readFile(join(app, "MacOS/pair-mycellios"), "utf8");
    expect(plist).toContain("application/vnd.mycellios.enrollment+json"); expect(plist).toContain("LSHandlerRank</key><string>Owner");
    expect(launcher).toContain("macos-pairing-main.js"); expect(launcher).not.toContain("enrollmentToken");
  });
});
