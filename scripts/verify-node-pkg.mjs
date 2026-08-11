import { spawnSync } from "node:child_process";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { assertInstallerPackageTreeMatches } from "./verify-installer-package-tree.mjs";

export async function verifyNodePkg(input) {
  const root = await mkdtemp(join(tmpdir(), "mycellios-node-pkg-verify-")), expanded = join(root, "expanded");
  try {
    const result = spawnSync("pkgutil", ["--expand-full", resolve(input.pkg), expanded], { encoding: "utf8", shell: false });
    if (result.error) throw result.error; if (result.status !== 0) throw new Error("node_pkg_expand_failed");
    const payload = join(expanded, "Payload");
    await assertInstallerPackageTreeMatches(resolve(input.stagedRoot), join(payload, "Library/Application Support/Mycellios/Product"));
    const plist = await readFile(join(payload, "Applications/Mycellios Pairing.app/Contents/Info.plist"), "utf8");
    const launcher = await readFile(join(payload, "Applications/Mycellios Pairing.app/Contents/MacOS/pair-mycellios"), "utf8");
    if (!plist.includes("mycellios-enrollment") || !plist.includes("io.mycellios.pairing") || !launcher.includes("macos-pairing-main.js")) {
      throw new Error("node_pkg_pairing_application_is_invalid");
    }
    return { identifier: "io.mycellios.node", version: input.version };
  } finally { await rm(root, { recursive: true, force: true }); }
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) {
  const values = Object.fromEntries(process.argv.slice(2).map((argument) => { const match = /^--([^=]+)=(.+)$/.exec(argument); if (!match) throw new Error(`node_pkg_verify_argument_is_invalid:${argument}`); return [match[1], match[2]]; }));
  for (const key of ["pkg", "staged-root", "version"]) if (!values[key]) throw new Error(`node_pkg_verify_argument_is_missing:${key}`);
  await verifyNodePkg({ pkg: values.pkg, stagedRoot: values["staged-root"], version: values.version }); process.stdout.write(`Native node PKG verified: ${values.pkg}\n`);
}
