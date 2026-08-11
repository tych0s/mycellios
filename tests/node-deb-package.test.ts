import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { stageNodeInstaller } from "../scripts/stage-node-installer.mjs";
import { buildNodeDeb } from "../scripts/build-node-deb.mjs";
import { verifyNodeDeb } from "../scripts/verify-node-deb.mjs";
import { buildNativeSourceProvenance } from "../scripts/native-build-provenance.mjs";
import { writeNodeReleaseEvidence } from "../scripts/node-release-evidence.mjs";
import { verifyNodeReleaseEvidence } from "../scripts/verify-node-release-evidence.mjs";

const cleanup: string[] = [];
afterEach(async () => Promise.all(cleanup.splice(0).map((path) => rm(path, { recursive: true, force: true }))));

describe.runIf(process.platform === "linux" && process.arch === "x64")("native Debian package", () => {
  it("binds the exact staged product and opens pairing files without a terminal", async () => {
    const fixture = await stagedFixture(); const deb = join(fixture.root, "mycellios-node.deb");
    await buildNodeDeb({ stagedRoot: fixture.output, output: deb, version: "0.2.77", sourceRevision: "a".repeat(40), sourceDateEpoch: "0" });
    await expect(verifyNodeDeb({ deb, stagedRoot: fixture.output, version: "0.2.77" })).resolves.toEqual({ package: "mycellios-node", version: "0.2.77", architecture: "amd64" });
    const release = await writeNodeReleaseEvidence({ artifact: deb, stagedRoot: fixture.output, outputDirectory: join(fixture.root, "evidence"), sourceDateEpoch: "0",
      builderOs: "linux", builderArch: "x64", workflow: "test", runId: "1", runAttempt: "1" });
    await expect(verifyNodeReleaseEvidence({ artifact: deb, stagedRoot: fixture.output, evidencePath: release.paths.evidencePath,
      sbomPath: release.paths.sbomPath, checksumPath: release.paths.checksumPath })).resolves.toMatchObject({ signature: { state: "pending", requiredForPromotion: true } });
    await expect(verifyNodeReleaseEvidence({ artifact: deb, stagedRoot: fixture.output, evidencePath: release.paths.evidencePath,
      sbomPath: release.paths.sbomPath, checksumPath: release.paths.checksumPath, requireSigned: true })).rejects.toThrow("node_release_artifact_is_not_signed");
    expect((await readFile(deb)).byteLength).toBeGreaterThan(100);
  });
});

async function stagedFixture() {
  const root = await mkdtemp(join(tmpdir(), "mycellios-deb-test-")); cleanup.push(root);
  const dist = join(root, "dist"), runtime = join(root, "runtime"), nodeModules = join(root, "node_modules"), output = join(root, "staged");
  await Promise.all([mkdir(join(dist, "node"), { recursive: true }), mkdir(join(dist, "contracts"), { recursive: true }), mkdir(runtime),
    mkdir(join(nodeModules, "ws"), { recursive: true }), mkdir(join(nodeModules, "zod"), { recursive: true }), mkdir(join(root, "host"))]);
  await Promise.all(["main.js", "install-main.js", "uninstall-main.js"].map((name) => writeFile(join(dist, "node", name), "export {};\n")));
  await writeFile(join(dist, "contracts", "node-control.js"), "export {};\n"); await writeFile(join(runtime, "runtime-manifest.json"), JSON.stringify({ platform: "linux", arch: "x64" }));
  await writeFile(join(runtime, "bin"), "runtime"); await writeFile(join(nodeModules, "ws", "package.json"), "{}"); await writeFile(join(nodeModules, "zod", "package.json"), "{}");
  const nodeExecutable = join(root, "host", "node"); await writeFile(nodeExecutable, "node");
  await stageNodeInstaller({ output, dist, runtime, nodeExecutable, nodeModules, target: "linux-x64", sourceRevision: "a".repeat(40), sourceProvenance: buildNativeSourceProvenance(resolve(".")) });
  return { root, output };
}
