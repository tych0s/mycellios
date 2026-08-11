import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

export const NODE_PACKAGE_MANIFEST_SCHEMA = "mycellios-node-package/2";

export async function buildNodePackageManifest({ entrypointPath, entrypoint, bootstrapEntrypointPath, bootstrapEntrypoint, helperEntrypointPath, helperEntrypoint, target, sourceRevision }) {
  if (!/^[a-z0-9]+-[a-z0-9]+$/.test(target)) throw new Error("node_package_target_invalid");
  if (!/^(?!\/)(?!.*(?:^|\/)\.\.(?:\/|$))[A-Za-z0-9._/-]+$/.test(entrypoint)) throw new Error("node_package_entrypoint_invalid");
  if (!/^(?!\/)(?!.*(?:^|\/)\.\.(?:\/|$))[A-Za-z0-9._/-]+$/.test(bootstrapEntrypoint)) throw new Error("node_package_bootstrap_entrypoint_invalid");
  if (!/^(?!\/)(?!.*(?:^|\/)\.\.(?:\/|$))[A-Za-z0-9._/-]+$/.test(helperEntrypoint)) throw new Error("node_package_helper_entrypoint_invalid");
  if (!/^[a-f0-9]{40}$/.test(sourceRevision)) throw new Error("node_package_source_revision_invalid");
  const bytes = await readFile(resolve(entrypointPath));
  const bootstrapBytes = await readFile(resolve(bootstrapEntrypointPath));
  const helperBytes = await readFile(resolve(helperEntrypointPath));
  return {
    schema: NODE_PACKAGE_MANIFEST_SCHEMA,
    target,
    entrypoint,
    sourceRevision,
    sha256: createHash("sha256").update(bytes).digest("hex"),
    installationBootstrap: { entrypoint: bootstrapEntrypoint, sha256: createHash("sha256").update(bootstrapBytes).digest("hex") },
    uninstallHelper: { entrypoint: helperEntrypoint, sha256: createHash("sha256").update(helperBytes).digest("hex") },
  };
}

async function main(argv) {
  const values = Object.fromEntries(argv.map((argument) => {
    const match = /^--([^=]+)=(.+)$/.exec(argument);
    if (!match) throw new Error(`node_package_argument_invalid:${argument}`);
    return [match[1], match[2]];
  }));
  for (const key of ["entrypoint-path", "entrypoint", "bootstrap-entrypoint-path", "bootstrap-entrypoint", "helper-entrypoint-path", "helper-entrypoint", "target", "source-revision", "output"]) {
    if (!values[key]) throw new Error(`node_package_argument_missing:${key}`);
  }
  const manifest = await buildNodePackageManifest({
    entrypointPath: values["entrypoint-path"], entrypoint: values.entrypoint,
    bootstrapEntrypointPath: values["bootstrap-entrypoint-path"], bootstrapEntrypoint: values["bootstrap-entrypoint"],
    helperEntrypointPath: values["helper-entrypoint-path"], helperEntrypoint: values["helper-entrypoint"],
    target: values.target, sourceRevision: values["source-revision"],
  });
  await writeFile(resolve(values.output), `${JSON.stringify(manifest, null, 2)}\n`, { flag: "wx" });
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) {
  await main(process.argv.slice(2));
}
