import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { buildNodePackageManifest } from "../scripts/node-package-manifest.mjs";

const cleanup: string[] = [];
afterEach(async () => Promise.all(cleanup.splice(0).map((path) => rm(path, { recursive: true, force: true }))));

describe("node package manifest", () => {
  it("seals entrypoint bytes, target and source revision", async () => {
    const directory = await mkdtemp(join(tmpdir(), "mycellios-node-package-")); cleanup.push(directory);
    const entrypointPath = join(directory, "main.js"); await writeFile(entrypointPath, "export {};\n");
    const bootstrapEntrypointPath = join(directory, "install-main.js"); await writeFile(bootstrapEntrypointPath, "export const install = true;\n");
    const helperEntrypointPath = join(directory, "uninstall-main.js"); await writeFile(helperEntrypointPath, "export const helper = true;\n");
    await expect(buildNodePackageManifest({ entrypointPath, entrypoint: "node/main.js", bootstrapEntrypointPath, bootstrapEntrypoint: "node/install-main.js", helperEntrypointPath, helperEntrypoint: "node/uninstall-main.js", target: "linux-x64", sourceRevision: "a".repeat(40) })).resolves.toMatchObject({
      schema: "mycellios-node-package/2", target: "linux-x64", entrypoint: "node/main.js", sourceRevision: "a".repeat(40),
      sha256: "8e609bb71c20b858c77f0e9f90bb1319db8477b13f9f965f1a1e18524bf50881",
      installationBootstrap: { entrypoint: "node/install-main.js", sha256: expect.stringMatching(/^[a-f0-9]{64}$/) },
      uninstallHelper: { entrypoint: "node/uninstall-main.js", sha256: expect.stringMatching(/^[a-f0-9]{64}$/) },
    });
  });

  it("rejects malformed source identity and unsafe entrypoints", async () => {
    const input = { entrypointPath: "package.json", entrypoint: "../main.js", bootstrapEntrypointPath: "package.json", bootstrapEntrypoint: "node/install-main.js", helperEntrypointPath: "package.json", helperEntrypoint: "node/uninstall-main.js", target: "linux-x64", sourceRevision: "bad" };
    await expect(buildNodePackageManifest(input)).rejects.toThrow("node_package_entrypoint_invalid");
    await expect(buildNodePackageManifest({ ...input, entrypoint: "node/main.js" })).rejects.toThrow("node_package_source_revision_invalid");
  });
});
