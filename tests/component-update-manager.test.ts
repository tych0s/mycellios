import {
  generateKeyPairSync,
} from "node:crypto";
import {
  access,
  mkdtemp,
  mkdir,
  readFile,
  readdir,
  rm,
  writeFile,
} from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { afterEach, describe, expect, it } from "vitest";

import {
  buildComponentUpdateManifest,
  signComponentUpdateManifest,
  type ComponentUpdateManifest,
} from "../src/contracts/component-update-manifest.js";
import { buildComponentFilesPackage } from "../src/update/component-files.js";
import {
  acquireComponentInUseLease,
  assertManagedComponentPolicy,
  assertResolvedComponentDependencies,
  ComponentUpdateManager,
  readComponentInUseRoots,
  readComponentInstallState,
  resolveActiveComponentRoot,
  rollbackInstalledComponents,
} from "../src/update/component-update-manager.js";

const roots: string[] = [];

afterEach(async () => {
  await Promise.all(
    roots.splice(0).map((root) =>
      rm(root, { recursive: true, force: true }),
    ),
  );
});

async function fixture(
  sequence = 1,
  restartScope: "none" | "runtime" | "agent" | "application" = "runtime",
): Promise<{
  root: string;
  packageBytes: Buffer;
  manifest: ComponentUpdateManifest;
  pinnedKey: { keyId: string; spki: string };
  privateKey: ReturnType<typeof generateKeyPairSync>["privateKey"];
}> {
  const root = await mkdtemp(join(tmpdir(), "mycellios-component-manager-"));
  roots.push(root);
  const source = join(root, "source");
  await mkdir(join(source, "distributed_runtime"), { recursive: true });
  await writeFile(
    join(source, "distributed_runtime", "__init__.py"),
    `REVISION = ${JSON.stringify(sequence)}\n`,
  );
  const built = buildComponentFilesPackage(source);
  const { privateKey, publicKey } = generateKeyPairSync("ed25519");
  const keyId = "mycellios-dev-test-1";
  const manifest = signComponentUpdateManifest(
    buildComponentUpdateManifest({
      channel: "dev",
      sequence,
      revision: sequence.toString(16).padStart(40, "0"),
      provenance: {
        baseRevision: sequence.toString(16).padStart(40, "0"),
        sourceTreeDirty: false,
        sourceTreeDigest: built.filesManifestSha256,
      },
      sourceId: `sha256:${sequence.toString(16).padStart(64, "0")}`,
      compatibility: {
        workerProtocol: { min: 1, max: 1 },
        runtimeAbi: "mycellios-distribution-runtime/4",
        minBootstrapVersion: "0.2.0",
      },
      components: [
        {
          id: "python-product",
          version: `0.2.0-dev.${sequence}`,
          platform: "win32",
          arch: "x64",
          artifact: {
            url: "https://updates.example.test/updates/v1/artifacts/python",
            sha256: built.artifactSha256,
            bytes: built.packageBytes.length,
            format: "json-gzip-v1",
            filesManifestSha256: built.filesManifestSha256,
          },
          requirements: {
            backend: "any",
            driver: null,
            runtimeAbi: "mycellios-distribution-runtime/4",
            workerProtocol: { min: 1, max: 1 },
            dependencies: [],
          },
          restartScope,
        },
      ],
    }),
    { keyId, privateKey },
  );
  return {
    root,
    packageBytes: built.packageBytes,
    manifest,
    pinnedKey: {
      keyId,
      spki: publicKey
        .export({ format: "der", type: "spki" })
        .toString("base64url"),
    },
    privateKey,
  };
}

function fetchFixture(
  manifest: ComponentUpdateManifest,
  packageBytes: Buffer,
): typeof fetch {
  return (async (input: string | URL | Request) => {
    const url = input instanceof Request ? input.url : String(input);
    if (url.endsWith("/manifest.json")) {
      return new Response(JSON.stringify(manifest), {
        status: 200,
        headers: {
          "content-type": "application/json",
        },
      });
    }
    return new Response(new Uint8Array(packageBytes), {
      status: 200,
      headers: {
        "content-type": "application/octet-stream",
        "content-length": String(packageBytes.length),
      },
    });
  }) as typeof fetch;
}

function managerOptions(
  data: Awaited<ReturnType<typeof fixture>>,
  overrides: Partial<ConstructorParameters<typeof ComponentUpdateManager>[0]> =
    {},
): ConstructorParameters<typeof ComponentUpdateManager>[0] {
  return {
    storageRoot: join(data.root, "installed"),
    feedBaseUrl: "https://updates.example.test",
    channel: "dev",
    pinnedKey: data.pinnedKey,
    bootstrapVersion: "0.2.70",
    workerProtocol: { min: 1, max: 1 },
    runtimeAbi: "mycellios-distribution-runtime/4",
    platform: "win32",
    arch: "x64",
    fetch: fetchFixture(data.manifest, data.packageBytes),
    isIdle: () => true,
    onActivate: async () => undefined,
    ...overrides,
  };
}

async function nextRevision(
  data: Awaited<ReturnType<typeof fixture>>,
  sequence: number,
): Promise<{ manifest: ComponentUpdateManifest; packageBytes: Buffer }> {
  const source = join(data.root, `source-v${sequence}`);
  await mkdir(join(source, "distributed_runtime"), { recursive: true });
  await writeFile(join(source, "distributed_runtime", "__init__.py"), `REVISION = ${sequence}\n`);
  const built = buildComponentFilesPackage(source);
  return {
    packageBytes: built.packageBytes,
    manifest: signComponentUpdateManifest(buildComponentUpdateManifest({
      channel: "dev",
      sequence,
      revision: sequence.toString(16).padStart(40, "0"),
      provenance: { baseRevision: sequence.toString(16).padStart(40, "0"), sourceTreeDirty: false, sourceTreeDigest: built.filesManifestSha256 },
      sourceId: `sha256:${sequence.toString(16).padStart(64, "0")}`,
      compatibility: { workerProtocol: { min: 1, max: 1 }, runtimeAbi: "mycellios-distribution-runtime/4", minBootstrapVersion: "0.2.0" },
      components: [{
        id: "python-product", version: `0.2.0-dev.${sequence}`, platform: "win32", arch: "x64", restartScope: "runtime",
        artifact: { url: `https://updates.example.test/updates/v1/artifacts/python-v${sequence}`, sha256: built.artifactSha256, bytes: built.packageBytes.length, format: "json-gzip-v1", filesManifestSha256: built.filesManifestSha256 },
        requirements: { backend: "any", driver: null, runtimeAbi: "mycellios-distribution-runtime/4", workerProtocol: { min: 1, max: 1 }, dependencies: [] },
      }],
    }), { keyId: data.pinnedKey.keyId, privateKey: data.privateKey }),
  };
}

describe("ComponentUpdateManager", () => {
  it("enforces the canonical bootstrap, product and backend component matrix", async () => {
    const data = await fixture();
    const base = data.manifest.components[0]!;
    const dependency = { id: "python-product", minVersion: "0.2.70" };
    const variants = [
      { id: "node-bootstrap", platform: "win32", arch: "x64", backend: "any", driver: null, dependencies: [], restartScope: "agent" },
      { id: "python-product", platform: "linux", arch: "x64", backend: "any", driver: null, dependencies: [], restartScope: "runtime" },
      { id: "runtime-cpu", platform: "linux", arch: "x64", backend: "cpu", driver: null, dependencies: [dependency], restartScope: "runtime" },
      { id: "runtime-cuda", platform: "win32", arch: "x64", backend: "cuda", driver: { api: "nvidia-display", minVersion: "560.0.0" }, dependencies: [dependency], restartScope: "runtime" },
      { id: "runtime-rocm", platform: "win32", arch: "x64", backend: "rocm", driver: { api: "amd-windows", minVersion: "24.20.0" }, dependencies: [dependency], restartScope: "runtime" },
      { id: "runtime-metal", platform: "darwin", arch: "arm64", backend: "metal", driver: null, dependencies: [dependency], restartScope: "runtime" },
    ] as const;
    for (const variant of variants) {
      const component = {
        ...base,
        id: variant.id,
        platform: variant.platform,
        arch: variant.arch,
        restartScope: variant.restartScope,
        requirements: { ...base.requirements, backend: variant.backend, driver: variant.driver, dependencies: [...variant.dependencies] },
      };
      expect(() => assertManagedComponentPolicy(component, variant.platform, variant.arch)).not.toThrow();
    }
    const cuda = {
      ...base, id: "runtime-cuda", platform: "win32" as const, arch: "x64" as const, restartScope: "runtime" as const,
      requirements: { ...base.requirements, backend: "cuda" as const, driver: { api: "nvidia-display", minVersion: "560.0.0" }, dependencies: [dependency] },
    };
    expect(() => assertManagedComponentPolicy({ ...cuda, requirements: { ...cuda.requirements, backend: "cpu" } }, "win32", "x64"))
      .toThrow("component_update_backend_policy_mismatch:runtime-cuda");
    expect(() => assertManagedComponentPolicy({ ...cuda, requirements: { ...cuda.requirements, driver: null } }, "win32", "x64"))
      .toThrow("component_update_driver_policy_mismatch:runtime-cuda");
    expect(() => assertManagedComponentPolicy({ ...cuda, requirements: { ...cuda.requirements, dependencies: [] } }, "win32", "x64"))
      .toThrow("component_update_dependency_policy_mismatch:runtime-cuda");
    expect(() => assertManagedComponentPolicy(cuda, "linux", "x64"))
      .toThrow("component_update_target_not_certified:runtime-cuda:linux/x64");

    const emptyState = { schema: "mycellios-component-install-state/2" as const, channels: { dev: null, stable: null }, active: {}, previous: {}, rejected: { dev: null, stable: null } };
    const runtime = { ...cuda, id: "runtime-cpu", requirements: { ...cuda.requirements, backend: "cpu" as const, driver: null } };
    expect(() => assertResolvedComponentDependencies([runtime], emptyState, new Set()))
      .toThrow("component_update_dependency_missing:runtime-cpu:python-product");
    expect(() => assertResolvedComponentDependencies([
      { ...base, version: "0.2.0" },
      { ...runtime, requirements: { ...runtime.requirements, dependencies: [{ ...dependency, minVersion: "0.3.0" }] } },
    ], emptyState, new Set()))
      .toThrow("component_update_dependency_too_old:runtime-cpu:python-product");
    expect(() => assertResolvedComponentDependencies([
      { ...base, version: "0.2.0" },
      { ...runtime, requirements: { ...runtime.requirements, dependencies: [{ ...dependency, minVersion: "0.1.0", maxVersionExclusive: "0.2.0" }] } },
    ], emptyState, new Set()))
      .toThrow("component_update_dependency_too_new:runtime-cpu:python-product");
  });

  it("pins the exact active root while a runtime process is using it", async () => {
    const data = await fixture();
    const storageRoot = join(data.root, "installed");
    await new ComponentUpdateManager(managerOptions(data)).checkAndApply();
    const active = await resolveActiveComponentRoot(storageRoot, "python-product");
    const lease = await acquireComponentInUseLease(storageRoot, ["python-product"]);
    expect(lease.roots["python-product"]).toBe(active);
    expect([...(await readComponentInUseRoots(storageRoot))]).toEqual([active]);
    expect(await readdir(join(storageRoot, "in-use"))).toHaveLength(1);
    await lease.release();
    await lease.release();
    expect(await readdir(join(storageRoot, "in-use"))).toHaveLength(0);

    const stale = await acquireComponentInUseLease(storageRoot, ["python-product"]);
    expect(await readComponentInUseRoots(storageRoot, () => false)).toEqual(new Set());
    expect(await readdir(join(storageRoot, "in-use"))).toHaveLength(0);
    await stale.release();
  });

  it("keeps an in-use generation beyond active and previous until its lease is released", async () => {
    const data = await fixture();
    const storageRoot = join(data.root, "installed");
    await new ComponentUpdateManager(managerOptions(data)).checkAndApply();
    const firstRoot = await resolveActiveComponentRoot(storageRoot, "python-product");
    const lease = await acquireComponentInUseLease(storageRoot, ["python-product"]);
    const second = await nextRevision(data, 2);
    await new ComponentUpdateManager(managerOptions(data, { fetch: fetchFixture(second.manifest, second.packageBytes) })).checkAndApply();
    const third = await nextRevision(data, 3);
    const thirdManager = new ComponentUpdateManager(managerOptions(data, { fetch: fetchFixture(third.manifest, third.packageBytes) }));
    await thirdManager.checkAndApply();
    await expect(access(firstRoot!)).resolves.toBeUndefined();
    await lease.release();
    await thirdManager.checkAndApply();
    await expect(access(firstRoot!)).rejects.toMatchObject({ code: "ENOENT" });
  });

  it.each(["activation_timeout", "activation_oom", "activation_power_loss"])(
    "keeps the last healthy pointer after simulated %s",
    async (failure) => {
      const data = await fixture();
      const storageRoot = join(data.root, "installed");
      await new ComponentUpdateManager(managerOptions(data)).checkAndApply();
      const healthyRoot = await resolveActiveComponentRoot(storageRoot, "python-product");
      const second = await nextRevision(data, 2);
      const manager = new ComponentUpdateManager(managerOptions(data, {
        fetch: fetchFixture(second.manifest, second.packageBytes),
        onActivate: async () => { throw new Error(failure); },
      }));
      await expect(manager.checkAndApply()).rejects.toThrow(
        "component_update_activation_failed_and_rolled_back",
      );
      expect(await resolveActiveComponentRoot(storageRoot, "python-product")).toBe(healthyRoot);
      expect(await readFile(join(healthyRoot!, "distributed_runtime", "__init__.py"), "utf8"))
        .toContain("REVISION = 1");
    },
  );

  it("installs by digest, switches the pointer, and is idempotent", async () => {
    const data = await fixture();
    let activations = 0;
    const manager = new ComponentUpdateManager(
      managerOptions(data, {
        onActivate: async ({ stagedComponentRoots }) => {
          activations += 1;
          const active = await resolveActiveComponentRoot(
            join(data.root, "installed"),
            "python-product",
          );
          expect(active).toBeNull();
          expect(stagedComponentRoots["python-product"]).toBeTruthy();
        },
      }),
    );

    await expect(manager.checkAndApply()).resolves.toMatchObject({
      state: "applied",
      changedComponents: ["python-product"],
    });
    await expect(manager.checkAndApply()).resolves.toMatchObject({
      state: "up-to-date",
      changedComponents: [],
    });
    expect(activations).toBe(1);

    const active = await resolveActiveComponentRoot(
      join(data.root, "installed"),
      "python-product",
    );
    expect(
      await readFile(
        join(active!, "distributed_runtime", "__init__.py"),
        "utf8",
      ),
    ).toContain("REVISION = 1");
    expect(
      (await readComponentInstallState(join(data.root, "installed"))).channels
        .dev,
    ).toMatchObject({
      sequence: 1,
      manifestId: data.manifest.manifestId,
    });
  });

  it("rolls back only to a complete previous root signed by a still-pinned key", async () => {
    const data = await fixture();
    const storageRoot = join(data.root, "installed");
    await new ComponentUpdateManager(managerOptions(data)).checkAndApply();

    const source = join(data.root, "source-v2");
    await mkdir(join(source, "distributed_runtime"), { recursive: true });
    await writeFile(join(source, "distributed_runtime", "__init__.py"), "REVISION = 2\n");
    const built = buildComponentFilesPackage(source);
    const manifest = signComponentUpdateManifest(buildComponentUpdateManifest({
      channel: "dev",
      sequence: 2,
      revision: "2".padStart(40, "0"),
      provenance: { baseRevision: "2".padStart(40, "0"), sourceTreeDirty: false, sourceTreeDigest: built.filesManifestSha256 },
      sourceId: `sha256:${"2".padStart(64, "0")}`,
      compatibility: { workerProtocol: { min: 1, max: 1 }, runtimeAbi: "mycellios-distribution-runtime/4", minBootstrapVersion: "0.2.0" },
      components: [{
        id: "python-product", version: "0.2.0-dev.2", platform: "win32", arch: "x64", restartScope: "runtime",
        artifact: { url: "https://updates.example.test/updates/v1/artifacts/python-v2", sha256: built.artifactSha256, bytes: built.packageBytes.length, format: "json-gzip-v1", filesManifestSha256: built.filesManifestSha256 },
        requirements: { backend: "any", driver: null, runtimeAbi: "mycellios-distribution-runtime/4", workerProtocol: { min: 1, max: 1 }, dependencies: [] },
      }],
    }), { keyId: data.pinnedKey.keyId, privateKey: data.privateKey });
    await new ComponentUpdateManager(managerOptions(data, { fetch: fetchFixture(manifest, built.packageBytes) })).checkAndApply();
    expect(await readFile(join((await resolveActiveComponentRoot(storageRoot, "python-product"))!, "distributed_runtime", "__init__.py"), "utf8")).toContain("REVISION = 2");

    let preparedRoot = "";
    await expect(rollbackInstalledComponents({
      storageRoot,
      componentIds: ["python-product"],
      expectedChannel: "dev",
      pinnedKeys: [data.pinnedKey],
      onPrepare: async ({ stagedComponentRoots }) => { preparedRoot = stagedComponentRoots["python-product"]!; },
    })).resolves.toMatchObject({ state: "rolled-back", changedComponents: ["python-product"] });
    expect(await readFile(join(preparedRoot, "distributed_runtime", "__init__.py"), "utf8")).toContain("REVISION = 1");
    expect(await readFile(join((await resolveActiveComponentRoot(storageRoot, "python-product"))!, "distributed_runtime", "__init__.py"), "utf8")).toContain("REVISION = 1");

    await expect(rollbackInstalledComponents({
      storageRoot,
      componentIds: ["python-product"],
      expectedChannel: "dev",
      pinnedKeys: [],
      onPrepare: async () => undefined,
    })).rejects.toThrow("component_rollback_key_not_trusted:python-product");
  });

  it("runs an active component only while its signing key remains trusted", async () => {
    const data = await fixture();
    const storageRoot = join(data.root, "installed");
    await new ComponentUpdateManager(managerOptions(data)).checkAndApply();

    await expect(
      resolveActiveComponentRoot(
        storageRoot,
        "python-product",
        "dev",
        [data.pinnedKey],
      ),
    ).resolves.toBeTruthy();

    const { publicKey: replacementPublicKey } =
      generateKeyPairSync("ed25519");
    const replacementKey = {
      keyId: data.pinnedKey.keyId,
      spki: replacementPublicKey
        .export({ format: "der", type: "spki" })
        .toString("base64url"),
    };
    await expect(
      resolveActiveComponentRoot(
        storageRoot,
        "python-product",
        "dev",
        [replacementKey],
      ),
    ).resolves.toBeNull();
    await expect(
      resolveActiveComponentRoot(
        storageRoot,
        "python-product",
        "dev",
        [],
      ),
    ).resolves.toBeNull();
  });

  it("canaries and reactivates the same artifact after a dev key rotation", async () => {
    const data = await fixture();
    const storageRoot = join(data.root, "installed");
    await new ComponentUpdateManager(managerOptions(data)).checkAndApply();

    const { privateKey, publicKey } = generateKeyPairSync("ed25519");
    const rotatedPinnedKey = {
      keyId: data.pinnedKey.keyId,
      spki: publicKey
        .export({ format: "der", type: "spki" })
        .toString("base64url"),
    };
    const rotatedManifest = signComponentUpdateManifest(
      buildComponentUpdateManifest({
        channel: data.manifest.channel,
        sequence: data.manifest.sequence,
        revision: data.manifest.revision,
        provenance: data.manifest.provenance,
        sourceId: data.manifest.sourceId,
        compatibility: data.manifest.compatibility,
        components: data.manifest.components,
      }),
      { keyId: rotatedPinnedKey.keyId, privateKey },
    );
    let activations = 0;
    const manager = new ComponentUpdateManager(
      managerOptions(data, {
        pinnedKey: rotatedPinnedKey,
        fetch: fetchFixture(rotatedManifest, data.packageBytes),
        onPrepare: async () => {
          activations += 1;
        },
      }),
    );

    await expect(manager.checkAndApply()).resolves.toMatchObject({
      state: "applied",
      changedComponents: ["python-product"],
    });
    expect(activations).toBe(1);
    await expect(
      resolveActiveComponentRoot(
        storageRoot,
        "python-product",
        "dev",
        [rotatedPinnedKey],
      ),
    ).resolves.toBeTruthy();
    await expect(
      resolveActiveComponentRoot(
        storageRoot,
        "python-product",
        "dev",
        [data.pinnedKey],
      ),
    ).resolves.toBeNull();
  });

  it("stages but does not change the active pointer while work is active", async () => {
    const data = await fixture();
    const manager = new ComponentUpdateManager(
      managerOptions(data, { isIdle: () => false }),
    );

    await expect(manager.checkAndApply()).resolves.toMatchObject({
      state: "waiting-idle",
      changedComponents: ["python-product"],
    });
    expect(
      await resolveActiveComponentRoot(
        join(data.root, "installed"),
        "python-product",
      ),
    ).toBeNull();
    expect(
      (await readComponentInstallState(join(data.root, "installed"))).channels
        .dev,
    ).toBeNull();
  });

  it("restores the previous pointer when the activation canary fails", async () => {
    const data = await fixture();
    let rollbacks = 0;
    const manager = new ComponentUpdateManager(
      managerOptions(data, {
        onActivate: async () => {
          throw new Error("canary_failed");
        },
        onRollback: async () => {
          rollbacks += 1;
        },
      }),
    );

    await expect(manager.checkAndApply()).rejects.toThrow(
      "component_update_activation_failed_and_rolled_back",
    );
    expect(rollbacks).toBe(1);
    expect(
      await resolveActiveComponentRoot(
        join(data.root, "installed"),
        "python-product",
      ),
    ).toBeNull();
  });

  it("canaries the staged root before exposing it as active", async () => {
    const data = await fixture();
    let prepareCalls = 0;
    let activationCalled = false;
    let rollbackCalled = false;
    const manager = new ComponentUpdateManager(
      managerOptions(data, {
        onPrepare: async ({ stagedComponentRoots }) => {
          prepareCalls += 1;
          expect(
            await resolveActiveComponentRoot(
              join(data.root, "installed"),
              "python-product",
            ),
          ).toBeNull();
          expect(
            await readFile(
              join(
                stagedComponentRoots["python-product"]!,
                "distributed_runtime",
                "__init__.py",
              ),
              "utf8",
            ),
          ).toContain("REVISION = 1");
          throw new Error("pre_activation_canary_failed");
        },
        onActivate: async () => {
          activationCalled = true;
        },
        onRollback: async () => {
          rollbackCalled = true;
        },
      }),
    );

    await expect(manager.checkAndApply()).rejects.toThrow(
      "component_update_pre_activation_canary_failed",
    );
    expect(activationCalled).toBe(false);
    expect(rollbackCalled).toBe(false);
    await expect(manager.checkAndApply()).rejects.toThrow(
      "component_update_manifest_rejected",
    );
    expect(prepareCalls).toBe(1);
    expect(
      (await readComponentInstallState(join(data.root, "installed"))).rejected
        .dev,
    ).toMatchObject({
      manifestId: data.manifest.manifestId,
      attempts: 1,
      retryAfter: null,
    });
    expect(
      await resolveActiveComponentRoot(
        join(data.root, "installed"),
        "python-product",
      ),
    ).toBeNull();
  });

  it("rejects a manifest for a different runtime ABI before download", async () => {
    const data = await fixture();
    let artifactRequests = 0;
    const baseFetch = fetchFixture(data.manifest, data.packageBytes);
    const manager = new ComponentUpdateManager(
      managerOptions(data, {
        runtimeAbi: "mycellios-distribution-runtime/5",
        fetch: (async (input, init) => {
          if (!String(input).endsWith("/manifest.json")) artifactRequests += 1;
          return baseFetch(input, init);
        }) as typeof fetch,
      }),
    );

    await expect(manager.checkAndApply()).rejects.toThrow(
      "component_update_runtime_abi_incompatible",
    );
    expect(artifactRequests).toBe(0);
  });

  it("rejects a managed component with a restart scope the agent cannot honor", async () => {
    const data = await fixture(1, "application");
    const manager = new ComponentUpdateManager(managerOptions(data));

    await expect(manager.checkAndApply()).rejects.toThrow(
      "component_update_restart_scope_invalid:python-product",
    );
  });

  it("repairs a corrupted active component and immutable artifact cache", async () => {
    const data = await fixture();
    const storageRoot = join(data.root, "installed");
    await new ComponentUpdateManager(managerOptions(data)).checkAndApply();
    const active = await resolveActiveComponentRoot(
      storageRoot,
      "python-product",
    );
    await writeFile(
      join(active!, "distributed_runtime", "__init__.py"),
      "CORRUPTED = True\n",
    );
    await writeFile(
      join(
        storageRoot,
        "artifacts",
        data.manifest.components[0]!.artifact.sha256.slice("sha256:".length),
      ),
      "corrupt-cache",
    );
    let artifactRequests = 0;
    const baseFetch = fetchFixture(data.manifest, data.packageBytes);
    const countedFetch = (async (input, init) => {
      if (!String(input).endsWith("/manifest.json")) artifactRequests += 1;
      return baseFetch(input, init);
    }) as typeof fetch;
    const waitingManager = new ComponentUpdateManager(managerOptions(data, {
      isIdle: () => false,
      fetch: countedFetch,
    }));
    await expect(waitingManager.checkAndApply()).resolves.toMatchObject({
      state: "waiting-idle",
      changedComponents: ["python-product"],
    });
    expect(
      await readFile(
        join(active!, "distributed_runtime", "__init__.py"),
        "utf8",
      ),
    ).toContain("CORRUPTED");
    expect(
      await resolveActiveComponentRoot(storageRoot, "python-product"),
    ).toBeNull();

    const manager = new ComponentUpdateManager(managerOptions(data, {
      fetch: countedFetch,
    }));

    await expect(manager.checkAndApply()).resolves.toMatchObject({
      state: "applied",
      changedComponents: ["python-product"],
    });
    expect(artifactRequests).toBe(1);
    const repaired = await resolveActiveComponentRoot(
      storageRoot,
      "python-product",
    );
    expect(
      await readFile(
        join(repaired!, "distributed_runtime", "__init__.py"),
        "utf8",
      ),
    ).toContain("REVISION = 1");
  });

  it("migrates an inferable v1 state and recovers a corrupt primary from last-good", async () => {
    const data = await fixture();
    const storageRoot = join(data.root, "legacy-installed");
    await mkdir(storageRoot, { recursive: true });
    const component = {
      id: "python-product",
      version: "0.2.0-dev.1",
      artifactSha256: data.manifest.components[0]!.artifact.sha256,
      filesManifestSha256:
        data.manifest.components[0]!.artifact.filesManifestSha256,
      manifestId: data.manifest.manifestId,
      activatedAt: new Date().toISOString(),
    };
    await writeFile(
      join(storageRoot, "state.json"),
      `${JSON.stringify({
        schema: "mycellios-component-install-state/1",
        channels: {
          dev: {
            sequence: data.manifest.sequence,
            manifestId: data.manifest.manifestId,
          },
          stable: null,
        },
        active: { "python-product": component },
        previous: {},
      })}\n`,
    );

    const migrated = await readComponentInstallState(storageRoot);
    expect(migrated.schema).toBe("mycellios-component-install-state/2");
    expect(migrated.active["python-product"]?.channel).toBe("dev");

    await writeFile(join(storageRoot, "state.json"), "{not-json");
    const recovered = await readComponentInstallState(storageRoot);
    expect(recovered).toEqual(migrated);
    expect(
      (await readdir(storageRoot)).some((name) =>
        name.startsWith("state.json.invalid-"),
      ),
    ).toBe(true);
  });
});
