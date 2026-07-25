import {
  cpSync,
  mkdirSync,
  mkdtempSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  prepareCoordinatorRelease,
  verifyCoordinatorReleaseDirectory,
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
    const manifest = prepareCoordinatorRelease(fixture, destination, {
      revision: "a".repeat(40),
    });
    expect(manifest.files.map((entry) => entry.path)).toEqual(
      expect.arrayContaining([
        "dist/coordinator/main.js",
        "dist/core/config.js",
        "dist/contracts/schema.js",
        "python/distributed_runtime/native_gguf_disk_tiering.py",
        "python/mycellios-native-python-manifest.json",
      ]),
    );
    expect(manifest.files.some((entry) =>
      entry.path.includes("simulator")
      || entry.path.includes("gpu_cloud")
      || entry.path.endsWith("benchmark.py"),
    )).toBe(false);
    expect(() => verifyCoordinatorReleaseDirectory(destination)).not.toThrow();

    mkdirSync(join(destination, "deploy", "gpu_cloud"));
    writeFileSync(join(destination, "deploy", "gpu_cloud", "entrypoint.py"), "pass\n");
    expect(() => verifyCoordinatorReleaseDirectory(destination)).toThrow(
      "non-production deploy material",
    );
  });
});

function createFixture(): string {
  const root = mkdtempSync(join(tmpdir(), "mycellios-coordinator-release-test-"));
  temporaryRoots.push(root);
  write(root, "dist/coordinator/main.js", 'import "../core/config.js";\n');
  write(root, "dist/core/config.js", 'export { schema } from "../contracts/schema.js";\n');
  write(root, "dist/contracts/schema.js", "export const schema = 1;\n");
  write(root, "dist/simulator/cli.js", "throw new Error('not product');\n");
  write(root, "landing-dist/index.html", "<main>landing</main>\n");
  write(root, "mobile-dist/index.html", "<main>mobile</main>\n");
  write(root, "package.json", '{"name":"fixture","version":"1.0.0"}\n');
  write(root, "package-lock.json", '{"name":"fixture","lockfileVersion":3}\n');
  for (const file of [
    "mycellios-content-hub.conf",
    "mycellios-dynamic-workers.conf",
    "mycellios-release-storage.conf",
    "mycellios-supabase-persistence.conf",
  ]) {
    write(root, `deploy/systemd/${file}`, "[Service]\n");
  }
  cpSync(resolve("python"), join(root, "python"), {
    recursive: true,
    dereference: true,
  });
  return root;
}

function write(root: string, portable: string, contents: string): void {
  const path = join(root, ...portable.split("/"));
  mkdirSync(resolve(path, ".."), { recursive: true });
  writeFileSync(path, contents, "utf8");
}
