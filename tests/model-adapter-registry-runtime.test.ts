import { spawnSync } from "node:child_process";
import { copyFileSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { stripTypeScriptTypes } from "node:module";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { afterEach, describe, expect, it } from "vitest";

const temporaryRoots: string[] = [];
const registryRelativePath = "python/distributed_runtime/model_adapter_registry.json";
const installedRegistry = JSON.parse(readFileSync(resolve(registryRelativePath), "utf8")) as { registryId: string };

afterEach(() => {
  for (const root of temporaryRoots.splice(0)) rmSync(root, { recursive: true, force: true });
});

describe("installed model adapter registry discovery", () => {
  it("imports a relocated compiled module from an unrelated entrypoint and working directory", () => {
    const fixture = createRuntime();
    copyRegistry(fixture.runtime);
    const result = runImport(fixture);
    expect(result.status, result.stderr).toBe(0);
    expect(JSON.parse(result.stdout)).toEqual({
      registryId: installedRegistry.registryId,
      source: join(fixture.runtime, registryRelativePath),
    });
  });

  it("does not let the caller working directory shadow the installed registry", () => {
    const fixture = createRuntime();
    copyRegistry(fixture.runtime);
    const misleadingRegistry = join(fixture.cwd, registryRelativePath);
    mkdirSync(dirname(misleadingRegistry), { recursive: true });
    writeFileSync(misleadingRegistry, "not the installed registry");
    const result = runImport(fixture);
    expect(result.status, result.stderr).toBe(0);
    expect(JSON.parse(result.stdout).source).toBe(join(fixture.runtime, registryRelativePath));
  });

  it("fails closed when the installed registry is missing even if the caller provides another copy", () => {
    const fixture = createRuntime();
    copyRegistry(fixture.cwd);
    const result = runImport(fixture);
    expect(result.status).not.toBe(0);
    expect(result.stderr).toContain("mycellios_model_adapter_registry_is_missing");
  });

  it("retains the explicitly provided packaged resources directory", () => {
    const fixture = createRuntime();
    const resources = join(fixture.root, "resources");
    copyRegistry(resources);
    const result = runImport(fixture, resources);
    expect(result.status, result.stderr).toBe(0);
    expect(JSON.parse(result.stdout).source).toBe(join(resources, registryRelativePath));
  });
});

function createRuntime() {
  const root = mkdtempSync(join(tmpdir(), "mycellios-registry-runtime-"));
  temporaryRoots.push(root);
  const runtime = join(root, "relocated-package");
  const cwd = join(root, "unrelated", "working-directory");
  mkdirSync(cwd, { recursive: true });
  mkdirSync(runtime, { recursive: true });
  writeFileSync(join(runtime, "package.json"), '{"type":"module"}\n');
  for (const module of ["contracts/model-adapter-registry", "core/json"]) {
    const output = join(runtime, "dist", `${module}.js`);
    mkdirSync(dirname(output), { recursive: true });
    writeFileSync(output, stripTypeScriptTypes(readFileSync(resolve("src", `${module}.ts`), "utf8")));
  }
  return { root, runtime, cwd };
}

function copyRegistry(root: string): void {
  const destination = join(root, registryRelativePath);
  mkdirSync(dirname(destination), { recursive: true });
  copyFileSync(resolve(registryRelativePath), destination);
}

function runImport(fixture: ReturnType<typeof createRuntime>, resources?: string) {
  const entrypoint = join(fixture.cwd, "external-importer.mjs");
  const moduleUrl = pathToFileURL(join(fixture.runtime, "dist/contracts/model-adapter-registry.js")).href;
  writeFileSync(entrypoint, [
    ...(resources ? [`process.resourcesPath = ${JSON.stringify(resources)};`] : []),
    `const registry = await import(${JSON.stringify(moduleUrl)});`,
    "process.stdout.write(JSON.stringify({registryId: registry.MODEL_ADAPTER_REGISTRY_ID, source: registry.MODEL_ADAPTER_REGISTRY_SOURCE}));",
  ].join("\n"));
  const environment: NodeJS.ProcessEnv = {};
  if (process.env.SystemRoot) environment.SystemRoot = process.env.SystemRoot;
  return spawnSync(process.execPath, [entrypoint], {
    cwd: fixture.cwd, env: environment, encoding: "utf8", shell: false,
    windowsHide: true, timeout: 10_000, maxBuffer: 1024 * 1024,
  });
}
