import { spawnSync } from "node:child_process";
import { chmod, cp, mkdir, rm, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { verifyNodeInstaller } from "./verify-node-installer.mjs";

export async function buildNodePkg(input) {
  const staged = resolve(input.stagedRoot), output = resolve(input.output);
  if (!/^\d+\.\d+\.\d+(?:-[0-9A-Za-z.+~]+)?$/.test(input.version)) throw new Error("node_pkg_version_is_invalid");
  const layout = await verifyNodeInstaller(staged); if (layout.target !== "macos-arm64") throw new Error("node_pkg_layout_target_mismatch");
  const root = `${output}.root`; await rm(root, { recursive: true, force: true }); await rm(output, { force: true });
  await cp(staged, join(root, "Library/Application Support/Mycellios/Product"), { recursive: true, errorOnExist: true, force: false });
  await materializeMacPairingApp(root);
  const result = spawnSync("pkgbuild", ["--root", root, "--identifier", "io.mycellios.node", "--version", input.version, "--install-location", "/", output], { encoding: "utf8", shell: false,
    env: { ...process.env, SOURCE_DATE_EPOCH: input.sourceDateEpoch ?? "0" } });
  if (result.error) throw result.error; if (result.status !== 0) throw new Error(`node_pkg_build_failed:${result.stderr.trim()}`);
  await rm(root, { recursive: true, force: true }); return { output, layout };
}

export async function materializeMacPairingApp(rootValue) {
  const contents = join(resolve(rootValue), "Applications/Mycellios Pairing.app/Contents"), executable = join(contents, "MacOS/pair-mycellios");
  await mkdir(join(contents, "MacOS"), { recursive: true });
  await writeFile(join(contents, "Info.plist"), [
    '<?xml version="1.0" encoding="UTF-8"?>', '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">',
    '<plist version="1.0"><dict>', '<key>CFBundleIdentifier</key><string>io.mycellios.pairing</string>', '<key>CFBundleName</key><string>Mycellios Pairing</string>',
    '<key>CFBundleExecutable</key><string>pair-mycellios</string>', '<key>CFBundlePackageType</key><string>APPL</string>', '<key>LSBackgroundOnly</key><true/>',
    '<key>CFBundleDocumentTypes</key><array><dict>', '<key>CFBundleTypeName</key><string>Mycellios Enrollment</string>', '<key>CFBundleTypeRole</key><string>Viewer</string>',
    '<key>LSHandlerRank</key><string>Owner</string>', '<key>CFBundleTypeExtensions</key><array><string>mycellios-enrollment</string></array>',
    '<key>CFBundleTypeMIMETypes</key><array><string>application/vnd.mycellios.enrollment+json</string></array>', '</dict></array>', '</dict></plist>', "",
  ].join("\n"), { mode: 0o644 });
  await writeFile(executable, ["#!/bin/sh", "set -eu", 'exec "/Library/Application Support/Mycellios/Product/bin/node" "/Library/Application Support/Mycellios/Product/app/node/macos-pairing-main.js" "$1"', ""].join("\n"), { mode: 0o755 });
  await chmod(executable, 0o755);
}

async function main(argv) { const values = Object.fromEntries(argv.map((argument) => { const match = /^--([^=]+)=(.+)$/.exec(argument); if (!match) throw new Error(`node_pkg_argument_is_invalid:${argument}`); return [match[1], match[2]]; }));
  for (const key of ["staged-root", "output", "version"]) if (!values[key]) throw new Error(`node_pkg_argument_is_missing:${key}`);
  await buildNodePkg({ stagedRoot: values["staged-root"], output: values.output, version: values.version, sourceDateEpoch: values["source-date-epoch"] }); }
if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) await main(process.argv.slice(2));
