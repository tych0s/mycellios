import { spawnSync } from "node:child_process";
import { mkdtemp, readFile, readdir, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { assertInstallerPackageTreeMatches } from "./verify-installer-package-tree.mjs";

export async function verifyNodeMsi(input) {
  const root = await mkdtemp(join(tmpdir(), "mycellios-node-msi-verify-")), extracted = join(root, "administrative"), decompiled = join(root, "decompiled.wxs");
  try {
    run("msiexec.exe", ["/a", resolve(input.msi), "/qn", `TARGETDIR=${extracted}`], "node_msi_administrative_extract_failed");
    run("dark.exe", ["-nologo", "-x", join(root, "dark"), "-o", decompiled, resolve(input.msi)], "node_msi_decompile_failed");
    const manifests = await find(extracted, "layout-manifest.json");
    if (manifests.length !== 1) throw new Error("node_msi_product_root_is_ambiguous");
    await assertInstallerPackageTreeMatches(resolve(input.stagedRoot), dirname(manifests[0]));
    const source = await readFile(decompiled, "utf8");
    if (!source.includes(".mycellios-enrollment") || !source.includes("Mycellios.Enrollment") || !source.includes(input.sourceRevision)) {
      throw new Error("node_msi_pairing_or_source_identity_is_missing");
    }
    return { version: input.version, sourceRevision: input.sourceRevision };
  } finally { await rm(root, { recursive: true, force: true }); }
}

async function find(root, name) { const matches = []; async function visit(directory) { for (const entry of await readdir(directory, { withFileTypes: true })) { const path = join(directory, entry.name); if (entry.isDirectory()) await visit(path); else if (entry.isFile() && entry.name === name) matches.push(path); } } await visit(root); return matches; }
function run(executable, arguments_, error) { const result = spawnSync(executable, arguments_, { encoding: "utf8", shell: false }); if (result.error) throw result.error; if (result.status !== 0) throw new Error(`${error}:${result.stderr.trim()}`); }
if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) { const values = Object.fromEntries(process.argv.slice(2).map((argument) => { const match = /^--([^=]+)=(.+)$/.exec(argument); if (!match) throw new Error(`node_msi_verify_argument_is_invalid:${argument}`); return [match[1], match[2]]; }));
  for (const key of ["msi", "staged-root", "version", "source-revision"]) if (!values[key]) throw new Error(`node_msi_verify_argument_is_missing:${key}`);
  await verifyNodeMsi({ msi: values.msi, stagedRoot: values["staged-root"], version: values.version, sourceRevision: values["source-revision"] }); process.stdout.write(`Native node MSI verified: ${values.msi}\n`); }
