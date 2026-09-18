import {
  copyFileSync,
  cpSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { loadCoordinatorConfig } from "../src/core/config.js";
import {
  prepareCoordinatorRelease,
  verifyCoordinatorOutputReceiptDocument,
  verifyCoordinatorReleaseDirectory,
  writeCoordinatorOutputReceipt,
} from "../scripts/coordinator-release-policy.mjs";

const temporaryRoots: string[] = [];

afterEach(() => {
  for (const root of temporaryRoots.splice(0)) {
    rmSync(root, { recursive: true, force: true });
  }
});

describe("native coordinator release staging", () => {
  it("ships only the coordinator JS closure, native Python and production drop-ins", () => {
    const fixture = createFixture();
    const destination = join(fixture, "release");
    mkdirSync(destination);
    const receipt = writeCoordinatorOutputReceipt(fixture);
    const manifest = prepareCoordinatorRelease(fixture, destination, {
      revision: "a".repeat(40),
      populateProductionDependencies: writeProductionDependencies,
    });
    expect(manifest.schema).toBe("mycellios-native-coordinator-release/3");
    expect(manifest.sourceId).toMatch(/^sha256:[0-9a-f]{64}$/);
    expect(manifest.releaseId).toMatch(/^sha256:[0-9a-f]{64}$/);
    expect(receipt.sourceId).toBe(manifest.sourceId);
    expect(receipt.outputs.map((output) => output.path)).toEqual([
      "dist",
      "landing-dist",
      "mobile-dist",
    ]);
    expect(() =>
      verifyCoordinatorOutputReceiptDocument({
        ...receipt,
        unexpected: true,
      }),
    ).toThrow("unexpected or missing fields");
    const tamperedReceipt = structuredClone(receipt);
    tamperedReceipt.outputs[0]!.files[0]!.bytes += 1;
    expect(() =>
      verifyCoordinatorOutputReceiptDocument(tamperedReceipt),
    ).toThrow("outputId does not seal");
    expect(manifest.files.map((entry) => entry.path)).toEqual(
      expect.arrayContaining([
        "mycellios-native-build-provenance.json",
        "mycellios-coordinator-output-receipt.json",
        "dist/coordinator/main.js",
        "dist/core/config.js",
        "dist/contracts/schema.js",
        "node_modules/.package-lock.json",
        "node_modules/fixture-runtime/index.js",
        "python/distributed_runtime/native_gguf_disk_tiering.py",
        "python/mycellios-native-python-manifest.json",
      ]),
    );
    expect(manifest.files.some((entry) =>
      entry.path.includes("simulator")
      || entry.path.includes("gpu_cloud")
      || entry.path.endsWith("benchmark.py"),
    )).toBe(false);
    expect(manifest.files.map((entry) => entry.path).filter((path) => path.startsWith("deploy/")))
      .toEqual([
        "deploy/systemd/mycellios-dynamic-workers.conf",
        "deploy/systemd/mycellios-release-storage.conf",
      ]);
    for (const file of ["mycellios-dynamic-workers.conf", "mycellios-release-storage.conf"]) {
      expect(readFileSync(join(destination, "deploy", "systemd", file), "utf8"))
        .toBe(readFileSync(resolve("deploy", "systemd", file), "utf8"));
    }
    expect(() => verifyCoordinatorReleaseDirectory(destination)).not.toThrow();

    mkdirSync(join(destination, "deploy", "gpu_cloud"));
    writeFileSync(join(destination, "deploy", "gpu_cloud", "entrypoint.py"), "pass\n");
    expect(() => verifyCoordinatorReleaseDirectory(destination)).toThrow(
      "non-production deploy material",
    );
  });

  it("requires the native storage policy instead of silently omitting missing deployment inputs", () => {
    const fixture = createFixture();
    const destination = join(fixture, "release");
    mkdirSync(destination);
    rmSync(join(fixture, "deploy", "systemd", "mycellios-release-storage.conf"));
    writeCoordinatorOutputReceipt(fixture);

    expect(() => prepareCoordinatorRelease(fixture, destination, {
      revision: "f".repeat(40),
      populateProductionDependencies: writeProductionDependencies,
    })).toThrow("Coordinator release input is missing: deploy/systemd/mycellios-release-storage.conf");
  });

  it("loads the shipped storage policy into the native coordinator configuration", () => {
    const source = readFileSync(resolve("deploy/systemd/mycellios-release-storage.conf"), "utf8");
    const environment = Object.fromEntries(source.split(/\r?\n/)
      .filter((line) => line.startsWith("Environment="))
      .map((line) => {
        const assignment = line.slice("Environment=".length);
        const separator = assignment.indexOf("=");
        return [assignment.slice(0, separator), assignment.slice(separator + 1)];
      }));
    const config = loadCoordinatorConfig(environment);
    expect(config.databasePath).toBe(resolve("/var/lib/mycellios/mycellios.db"));
    expect(config.nodeUpdatesPath).toBe(resolve("/var/lib/mycellios/node-updates"));
    expect(config.releaseDownloadsPath).toBe(resolve("/var/lib/mycellios/downloads"));
    expect(environment.MYCELLIOS_BENCHMARK_ROOT).toBe("/var/lib/mycellios/benchmarks");
  });

  it("rejects an output receipt generated before the native source changed", () => {
    const fixture = createFixture();
    const destination = join(fixture, "release");
    mkdirSync(destination);
    writeCoordinatorOutputReceipt(fixture);
    write(
      fixture,
      "src/coordinator/main.ts",
      "export const staleSource = true;\n",
    );

    expect(() =>
      prepareCoordinatorRelease(fixture, destination, {
        revision: "b".repeat(40),
        populateProductionDependencies: writeProductionDependencies,
      }),
    ).toThrow("sourceId is stale");
  });

  it("rejects post-seal output and dependency mutations", () => {
    const fixture = createFixture();
    const firstDestination = join(fixture, "release-output");
    mkdirSync(firstDestination);
    writeCoordinatorOutputReceipt(fixture);
    prepareCoordinatorRelease(fixture, firstDestination, {
      revision: "c".repeat(40),
      populateProductionDependencies: writeProductionDependencies,
    });
    write(
      firstDestination,
      "mobile-dist/index.html",
      "<main>tampered</main>\n",
    );
    expect(() => verifyCoordinatorReleaseDirectory(firstDestination)).toThrow(
      "does not match the exact packaged outputs",
    );

    const secondDestination = join(fixture, "release-dependency");
    mkdirSync(secondDestination);
    prepareCoordinatorRelease(fixture, secondDestination, {
      revision: "d".repeat(40),
      populateProductionDependencies: writeProductionDependencies,
    });
    write(
      secondDestination,
      "node_modules/fixture-runtime/index.js",
      "export const changed = true;\n",
    );
    expect(() => verifyCoordinatorReleaseDirectory(secondDestination)).toThrow(
      "manifest does not match",
    );
  });

  it("rejects source drift that occurs during dependency installation", () => {
    const fixture = createFixture();
    const destination = join(fixture, "release-race");
    mkdirSync(destination);
    writeCoordinatorOutputReceipt(fixture);

    expect(() =>
      prepareCoordinatorRelease(fixture, destination, {
        revision: "e".repeat(40),
        populateProductionDependencies: (releaseRoot) => {
          writeProductionDependencies(releaseRoot);
          write(
            fixture,
            "src/coordinator/main.ts",
            "export const changedDuringPackaging = true;\n",
          );
        },
      }),
    ).toThrow("source changed while the release was being sealed");
  });
});

function createFixture(): string {
  const root = mkdtempSync(join(tmpdir(), "mycellios-coordinator-release-test-"));
  temporaryRoots.push(root);
  for (const portable of [
    ".github",
    "assets",
    "build/icons",
    "config",
    "deploy",
    "landing",
    "scripts",
    "src",
    "python",
  ]) {
    cpSync(resolve(portable), join(root, ...portable.split("/")), {
      recursive: true,
      dereference: true,
    });
  }
  for (const portable of [
    ".gitattributes",
    "package.json",
    "package-lock.json",
    "vite.landing.config.ts",
    "vite.mobile.config.ts",
    "tsconfig.json",
    "tsconfig.build.json",
    "tsconfig.landing.json",
    "tsconfig.mobile.json",
  ]) {
    copyFileSync(resolve(portable), join(root, portable));
  }
  write(root, "dist/coordinator/main.js", 'import "../core/config.js";\n');
  write(root, "dist/core/config.js", 'export { schema } from "../contracts/schema.js";\n');
  write(root, "dist/contracts/schema.js", "export const schema = 1;\n");
  write(root, "dist/simulator/cli.js", "throw new Error('not product');\n");
  write(root, "landing-dist/index.html", "<main>landing</main>\n");
  write(root, "mobile-dist/index.html", "<main>mobile</main>\n");
  write(root, "package.json", '{"name":"fixture","version":"1.0.0"}\n');
  write(root, "package-lock.json", '{"name":"fixture","lockfileVersion":3}\n');
  return root;
}

function write(root: string, portable: string, contents: string): void {
  const path = join(root, ...portable.split("/"));
  mkdirSync(resolve(path, ".."), { recursive: true });
  writeFileSync(path, contents, "utf8");
}

function writeProductionDependencies(destination: string): void {
  write(
    destination,
    "node_modules/.package-lock.json",
    '{"name":"fixture","version":"1.0.0","lockfileVersion":3,"packages":{}}\n',
  );
  write(
    destination,
    "node_modules/fixture-runtime/index.js",
    "export const fixture = true;\n",
  );
}
