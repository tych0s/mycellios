import { createHash } from "node:crypto";
import { access, mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, join, resolve, sep } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { assembleNodeReleaseAssets } from "../scripts/assemble-node-release-assets.mjs";
import { buildNativeSourceProvenance } from "../scripts/native-build-provenance.mjs";
import { stageNodeInstaller } from "../scripts/stage-node-installer.mjs";
import { writeNodeReleaseEvidence } from "../scripts/node-release-evidence.mjs";
import { preparePublicReleaseTransaction } from "../src/coordinator/public-release-transaction-cli.js";
import { NativeReleaseTransactionStore } from "../src/coordinator/release-upload.js";

const roots: string[] = [];
afterEach(async () => {
  for (const root of roots.splice(0)) {
    if (!resolve(root).startsWith(`${resolve(tmpdir())}${sep}`)
      || !basename(root).startsWith("mycellios-node-assets-")) {
      throw new Error("node_release_test_cleanup_path_unsafe");
    }
    await rm(root, { recursive: true, force: true });
  }
});

describe("native node release asset assembly", () => {
  it("assembles three verified CI jobs into a coherent feed and public manifest", async () => {
    const fixture = await createCiFixture();
    const result = await assembleNodeReleaseAssets(fixture.input);
    expect(result).toMatchObject({ signatureState: "pending", packages: [
      { target: "linux-x64" }, { target: "macos-arm64" }, { target: "windows-x64" },
    ] });
    const feed = JSON.parse(await readFile(join(fixture.outputRoot, "mycellios-node-latest.json"), "utf8"));
    expect(feed).toMatchObject({ schema: "mycellios-node-update-feed/1", version: fixture.version,
      packages: result.packages });
    for (const item of result.packages) {
      const bytes = await readFile(join(fixture.outputRoot, item.name));
      expect(createHash("sha256").update(bytes).digest("hex")).toBe(item.sha256);
    }
    const manifest = await preparePublicReleaseTransaction({
      assetsRoot: fixture.outputRoot,
      outputPath: join(fixture.root, "release-manifest.json"),
      transactionId: "native-release-1234567890",
      sourceId: fixture.input.sourceId as `sha256:${string}`,
      revision: fixture.input.revision,
      version: fixture.version,
    });
    expect(manifest.assets).toHaveLength(4);
    expect(manifest.assets.map(({ fileName }) => fileName)).toEqual([
      `mycellios-node-${fixture.version}-linux-x64.deb`,
      `mycellios-node-${fixture.version}-macos-arm64.pkg`,
      `mycellios-node-${fixture.version}-windows-x64.msi`,
      "mycellios-node-latest.json",
    ]);
    const store = new NativeReleaseTransactionStore({
      storageRoot: join(fixture.root, "transactions"), sourceId: fixture.input.sourceId,
      revision: fixture.input.revision, version: fixture.version,
    });
    await store.initialize();
    for (const asset of manifest.assets) {
      await store.storeChunk({
        identity: manifest,
        channel: asset.channel,
        fileName: asset.fileName,
        metadata: {
          chunkIndex: 0, chunkCount: 1,
          chunkSha256: asset.fileSha256,
          fileSha256: asset.fileSha256,
          fileSize: asset.fileSize,
        },
        body: await readFile(join(fixture.outputRoot, asset.fileName)),
      });
    }
    await store.commit(manifest);
    const published = await store.publicAssetPath("downloads", `mycellios-node-${fixture.version}-windows-x64.msi`);
    expect(published).not.toBeNull();
    expect(await readFile(published!)).toEqual(await readFile(join(fixture.outputRoot,
      `mycellios-node-${fixture.version}-windows-x64.msi`)));
  });

  it("rejects a mixed CI run and leaves no publishable asset directory", async () => {
    const fixture = await createCiFixture();
    const target = "windows-x64", name = `mycellios-node-${fixture.version}-${target}.msi`;
    const path = join(fixture.artifactsRoot, `mycellios-node-${target}`, "evidence", target, `${name}.provenance.json`);
    const evidence = JSON.parse(await readFile(path, "utf8"));
    evidence.builder.runId = "other-run";
    await writeFile(path, `${JSON.stringify(evidence, null, 2)}\n`);
    await expect(assembleNodeReleaseAssets(fixture.input)).rejects.toThrow("node_release_builder_run_mismatch");
    await expect(access(fixture.outputRoot)).rejects.toThrow();
  });

  it("rejects altered package bytes even when the other two jobs are intact", async () => {
    const fixture = await createCiFixture();
    const artifact = join(fixture.artifactsRoot, "mycellios-node-linux-x64",
      `mycellios-node-${fixture.version}-linux-x64.deb`);
    await writeFile(artifact, "changed after evidence was generated");
    await expect(assembleNodeReleaseAssets(fixture.input)).rejects.toThrow("node_release_artifact_evidence_mismatch");
    await expect(access(fixture.outputRoot)).rejects.toThrow();
  });

  it("rejects a CI upload that omitted a hidden runtime file", async () => {
    const fixture = await createCiFixture();
    await rm(join(fixture.artifactsRoot, "mycellios-node-macos-arm64",
      "node-package", "macos-arm64", "runtime", ".mycellios-runtime-marker"));
    await expect(assembleNodeReleaseAssets(fixture.input)).rejects.toThrow("node_installer_layout_digest_mismatch");
    await expect(access(fixture.outputRoot)).rejects.toThrow();
  });
});

async function createCiFixture() {
  const root = await mkdtemp(join(tmpdir(), "mycellios-node-assets-"));
  roots.push(root);
  const artifactsRoot = join(root, "ci"), outputRoot = join(root, "assembled");
  const sourceRoot = resolve(".");
  const sourceProvenance = buildNativeSourceProvenance(sourceRoot);
  const version = sourceProvenance.version;
  const revision = "a".repeat(40);
  for (const [target, extension, platform, arch] of [
    ["linux-x64", "deb", "linux", "x64"],
    ["macos-arm64", "pkg", "darwin", "arm64"],
    ["windows-x64", "msi", "win32", "x64"],
  ] as const) {
    const jobRoot = join(artifactsRoot, `mycellios-node-${target}`);
    const dist = join(jobRoot, "dist"), runtime = join(jobRoot, "runtime");
    const nodeModules = join(jobRoot, "node_modules"), pythonSource = join(jobRoot, "python", "distributed_runtime");
    const host = join(jobRoot, "host"), stagedRoot = join(jobRoot, "node-package", target);
    await Promise.all([
      mkdir(join(dist, "node"), { recursive: true }),
      mkdir(join(dist, "contracts"), { recursive: true }),
      mkdir(runtime, { recursive: true }),
      mkdir(join(nodeModules, "ws"), { recursive: true }),
      mkdir(join(nodeModules, "zod"), { recursive: true }),
      mkdir(pythonSource, { recursive: true }),
      mkdir(host, { recursive: true }),
      mkdir(jobRoot, { recursive: true }),
    ]);
    await Promise.all(["main.js", "install-main.js", "uninstall-main.js", "service-supervisor.js", "restore-main.js"]
      .map((file) => writeFile(join(dist, "node", file), "export {};\n")));
    await writeFile(join(dist, "contracts", "node-control.js"), "export {};\n");
    await writeFile(join(runtime, "runtime-manifest.json"), JSON.stringify({ platform, arch, pythonVersion: "3.12.13" }));
    await writeFile(join(runtime, "python"), "portable runtime");
    await writeFile(join(runtime, ".mycellios-runtime-marker"), "required runtime file");
    await Promise.all(["__init__.py", "physical_probe.py", "model_adapter_registry.json"]
      .map((file) => writeFile(join(pythonSource, file), file.endsWith(".json") ? "{}" : "")));
    await writeFile(join(nodeModules, "ws", "package.json"), '{"version":"1.0.0"}');
    await writeFile(join(nodeModules, "zod", "package.json"), '{"version":"1.0.0"}');
    const nodeExecutable = join(host, "node.exe"), brokerExecutable = join(host, "mycellios-job-broker.exe");
    const serviceWrapperExecutable = join(host, "MycelliosNode.exe");
    await Promise.all([writeFile(nodeExecutable, "node"), writeFile(brokerExecutable, "broker"),
      writeFile(serviceWrapperExecutable, "wrapper")]);
    await stageNodeInstaller({
      output: stagedRoot, dist, runtime, pythonSource, nodeExecutable, nodeModules,
      brokerExecutable, serviceWrapperExecutable, sourceRoot, target,
      sourceRevision: revision, sourceProvenance,
    });
    const name = `mycellios-node-${version}-${target}.${extension}`;
    const artifact = join(jobRoot, name);
    await writeFile(artifact, `fixture package for ${target}`);
    await writeNodeReleaseEvidence({
      artifact, stagedRoot, outputDirectory: join(jobRoot, "evidence", target),
      sourceDateEpoch: "0", builderOs: platform, builderArch: arch,
      workflow: "Native node build", runId: "12345", runAttempt: "1",
    });
  }
  return { root, artifactsRoot, outputRoot, version, input: {
    artifactsRoot, outputRoot, version, revision, sourceId: sourceProvenance.sourceId,
    publishedAt: "2026-09-29T00:00:00.000Z",
  } };
}
