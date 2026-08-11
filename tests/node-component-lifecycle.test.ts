import { generateKeyPairSync } from "node:crypto";
import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { buildComponentUpdateManifest, signComponentUpdateManifest } from "../src/contracts/component-update-manifest.js";
import type { NodeConfiguration } from "../src/contracts/node-configuration.js";
import { buildComponentFilesPackage } from "../src/update/component-files.js";
import { NodeComponentLifecycle, resolveNodePythonProduct } from "../src/node/component-lifecycle.js";

const cleanup: string[] = [];
afterEach(async () => Promise.all(cleanup.splice(0).map((path) => rm(path, { recursive: true, force: true }))));

describe("NodeComponentLifecycle", () => {
  it("pins an exact manifest, canaries staged Python and leaves a boot-health marker", async () => {
    const fixture = await setup();
    const canary = vi.fn(async () => undefined);
    const lifecycle = new NodeComponentLifecycle(fixture.config, "0.2.77", async () => true, async () => async () => undefined, fixture.fetch, canary);

    await expect(lifecycle.update("stable", fixture.manifest.manifestId as `sha256:${string}`)).resolves.toMatchObject({ state: "applied", changedComponents: ["python-product"] });
    expect(canary).toHaveBeenCalledTimes(1);
    expect(await lifecycle.pendingActivationIsCurrent()).toBe(true);
    expect(await resolveNodePythonProduct(fixture.config)).toBeTruthy();
    await lifecycle.markActivationHealthy();
    expect(await lifecycle.pendingActivation()).toBeNull();
  });

  it("rejects a feed manifest different from the command before downloading bytes", async () => {
    const fixture = await setup();
    let artifactRequests = 0;
    const counted = (async (input: string | URL | Request, init?: RequestInit) => {
      if (!String(input).endsWith("/manifest.json")) artifactRequests += 1;
      return fixture.fetch(input, init);
    }) as typeof fetch;
    const lifecycle = new NodeComponentLifecycle(fixture.config, "0.2.77", async () => true, async () => async () => undefined, counted, async () => undefined);
    await expect(lifecycle.update("stable", `sha256:${"f".repeat(64)}`)).rejects.toThrow("component_update_manifest_id_mismatch");
    expect(artifactRequests).toBe(0);
  });

  it("falls back to the bundled runtime when the first activated component fails boot", async () => {
    const fixture = await setup();
    const lifecycle = new NodeComponentLifecycle(fixture.config, "0.2.77", async () => true, async () => async () => undefined, fixture.fetch, async () => undefined);
    await lifecycle.update("stable", fixture.manifest.manifestId as `sha256:${string}`);
    expect(await resolveNodePythonProduct(fixture.config)).toBeTruthy();
    await expect(lifecycle.rollbackFailedActivation()).resolves.toMatchObject({ state: "rolled-back-failed-activation", changedComponents: ["python-product"] });
    expect(await resolveNodePythonProduct(fixture.config)).toBeNull();
    expect(await lifecycle.pendingActivation()).toBeNull();
  });
});

async function setup() {
  const root = await mkdtemp(join(tmpdir(), "mycellios-node-components-"));
  cleanup.push(root);
  const source = join(root, "source");
  await mkdir(join(source, "distributed_runtime"), { recursive: true });
  await writeFile(join(source, "distributed_runtime", "__init__.py"), "REVISION = 1\n");
  const built = buildComponentFilesPackage(source);
  const { privateKey, publicKey } = generateKeyPairSync("ed25519");
  const pinnedKey = { keyId: "stable-test-key", spki: publicKey.export({ format: "der", type: "spki" }).toString("base64url") };
  const manifest = signComponentUpdateManifest(buildComponentUpdateManifest({
    channel: "stable",
    sequence: 1,
    revision: "1".padStart(40, "0"),
    provenance: { baseRevision: "1".padStart(40, "0"), sourceTreeDirty: false, sourceTreeDigest: built.filesManifestSha256 },
    sourceId: `sha256:${"1".padStart(64, "0")}`,
    compatibility: { workerProtocol: { min: 1, max: 1 }, runtimeAbi: "mycellios-distribution-runtime/4", minBootstrapVersion: "0.2.70" },
    components: [{
      id: "python-product", version: "0.2.77", platform: process.platform as "linux" | "darwin" | "win32", arch: process.arch as "x64" | "arm64", restartScope: "runtime",
      artifact: { url: "https://updates.example.test/python-product", sha256: built.artifactSha256, bytes: built.packageBytes.length, format: "json-gzip-v1", filesManifestSha256: built.filesManifestSha256 },
      requirements: { backend: "any", driver: null, runtimeAbi: "mycellios-distribution-runtime/4", workerProtocol: { min: 1, max: 1 }, dependencies: [] },
    }],
  }), { keyId: pinnedKey.keyId, privateKey });
  const config: NodeConfiguration = {
    schema: "mycellios-node-configuration/1",
    revision: 1,
    nodeId: "node-1",
    coordinator: { url: "https://coordinator.example.test", identityPath: join(root, "identity") },
    worker: { configPath: join(root, "worker.json") },
    runtime: { pythonExecutable: "/usr/bin/python3", pythonPath: join(root, "fallback-python"), cachePath: join(root, "cache"), stagePort: 9_850 },
    limits: { maxConcurrency: 1, maxCpuPercent: 90, maxRamMiB: 8_192, maxVramMiB: 0, maxDiskMiB: 32_768, maxTemperatureC: 85 },
    isolation: { mode: "linux-cgroup-v2" },
    updateChannel: "stable",
    componentUpdates: { feedUrl: "https://updates.example.test", pinnedKeys: { dev: [], stable: [pinnedKey] } },
  };
  const fetchFixture = (async (input: string | URL | Request) => String(input).endsWith("/manifest.json")
    ? Response.json(manifest)
    : new Response(new Uint8Array(built.packageBytes), { headers: { "content-length": String(built.packageBytes.length) } })) as typeof fetch;
  return { root, config, manifest, fetch: fetchFixture };
}
