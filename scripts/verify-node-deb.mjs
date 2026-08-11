import { spawnSync } from "node:child_process";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { assertDebianMd5Sums, assertInstallerPackageTreeMatches } from "./verify-installer-package-tree.mjs";

export async function verifyNodeDeb(input) {
  const temporary = await mkdtemp(join(tmpdir(), "mycellios-node-deb-verify-"));
  const payload = join(temporary, "payload"), control = join(temporary, "control");
  try {
    for (const arguments_ of [["-x", resolve(input.deb), payload], ["-e", resolve(input.deb), control]]) {
      const result = spawnSync("dpkg-deb", arguments_, { encoding: "utf8", shell: false });
      if (result.error) throw result.error; if (result.status !== 0) throw new Error("node_deb_extract_failed");
    }
    const fields = parseControl(await readFile(join(control, "control"), "utf8"));
    if (fields.get("Package") !== "mycellios-node" || fields.get("Version") !== input.version || fields.get("Architecture") !== "amd64") {
      throw new Error("node_deb_control_identity_mismatch");
    }
    await assertDebianMd5Sums(join(control, "md5sums"), payload);
    await assertInstallerPackageTreeMatches(resolve(input.stagedRoot), join(payload, "opt", "mycellios"));
    const desktop = await readFile(join(payload, "usr/share/applications/mycellios-enrollment.desktop"), "utf8");
    const mime = await readFile(join(payload, "usr/share/mime/packages/mycellios-enrollment.xml"), "utf8");
    if (!desktop.includes("Exec=pkexec /opt/mycellios/install %f") || !desktop.includes("Terminal=false") || !mime.includes("*.mycellios-enrollment")) {
      throw new Error("node_deb_pairing_association_is_invalid");
    }
    return { package: fields.get("Package"), version: fields.get("Version"), architecture: fields.get("Architecture") };
  } finally { await rm(temporary, { recursive: true, force: true }); }
}

function parseControl(text) {
  const result = new Map();
  for (const line of text.split("\n")) { const match = /^([A-Za-z][A-Za-z0-9-]*): (.*)$/.exec(line); if (match) result.set(match[1], match[2]); }
  return result;
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) {
  const values = Object.fromEntries(process.argv.slice(2).map((argument) => { const match = /^--([^=]+)=(.+)$/.exec(argument); if (!match) throw new Error(`node_deb_verify_argument_is_invalid:${argument}`); return [match[1], match[2]]; }));
  for (const key of ["deb", "staged-root", "version"]) if (!values[key]) throw new Error(`node_deb_verify_argument_is_missing:${key}`);
  await verifyNodeDeb({ deb: values.deb, stagedRoot: values["staged-root"], version: values.version });
  process.stdout.write(`Native node DEB verified: ${values.deb}\n`);
}
