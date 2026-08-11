import { createHash } from "node:crypto";
import { spawnSync } from "node:child_process";
import { chmod, cp, mkdir, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { dirname, join, relative, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { verifyNodeInstaller } from "./verify-node-installer.mjs";

export async function buildNodeDeb(input) {
  const stagedRoot = resolve(input.stagedRoot);
  const output = resolve(input.output);
  if (!/^\d+\.\d+\.\d+(?:-[0-9A-Za-z.+~]+)?$/.test(input.version)) throw new Error("node_deb_version_is_invalid");
  if (!/^[a-f0-9]{40}$/.test(input.sourceRevision)) throw new Error("node_deb_source_revision_is_invalid");
  const layout = await verifyNodeInstaller(stagedRoot);
  if (layout.target !== "linux-x64") throw new Error("node_deb_layout_target_mismatch");
  const root = `${output}.root`;
  await rm(root, { recursive: true, force: true }); await rm(output, { force: true });
  const product = join(root, "opt", "mycellios"); const control = join(root, "DEBIAN");
  await mkdir(control, { recursive: true }); await cp(stagedRoot, product, { recursive: true, errorOnExist: true, force: false });
  await writeIntegration(root);
  await writeFile(join(control, "control"), [
    "Package: mycellios-node", `Version: ${input.version}`, "Section: utils", "Priority: optional", "Architecture: amd64",
    "Depends: systemd, policykit-1, shared-mime-info", "Maintainer: Mycellios <support@mycellios.com>",
    "Description: Native Mycellios contribution node", " Headless GPU inference service paired and controlled from the Mycellios web application.", "",
  ].join("\n"), { mode: 0o644 });
  await writeMaintainerScripts(control);
  await writeFile(join(control, "md5sums"), await canonicalMd5Sums(root), { mode: 0o644 });
  const built = spawnSync("dpkg-deb", ["--root-owner-group", "--build", root, output], { encoding: "utf8", shell: false,
    env: { ...process.env, SOURCE_DATE_EPOCH: input.sourceDateEpoch ?? "0" } });
  if (built.error) throw built.error;
  if (built.status !== 0) throw new Error(`node_deb_build_failed:${built.stderr.trim()}`);
  await rm(root, { recursive: true, force: true });
  return { output, layout };
}

async function writeIntegration(root) {
  const applications = join(root, "usr/share/applications"); const mime = join(root, "usr/share/mime/packages");
  await mkdir(applications, { recursive: true }); await mkdir(mime, { recursive: true });
  await writeFile(join(applications, "mycellios-enrollment.desktop"), [
    "[Desktop Entry]", "Type=Application", "Name=Pair this computer with Mycellios", "NoDisplay=true", "Terminal=false",
    "Exec=pkexec /opt/mycellios/install %f", "MimeType=application/vnd.mycellios.enrollment+json;", "Categories=Utility;", "",
  ].join("\n"), { mode: 0o644 });
  await writeFile(join(mime, "mycellios-enrollment.xml"), [
    '<?xml version="1.0" encoding="UTF-8"?>', '<mime-info xmlns="http://www.freedesktop.org/standards/shared-mime-info">',
    '  <mime-type type="application/vnd.mycellios.enrollment+json">', "    <comment>Mycellios one-time computer pairing</comment>",
    '    <glob pattern="*.mycellios-enrollment"/>', "  </mime-type>", "</mime-info>", "",
  ].join("\n"), { mode: 0o644 });
}

async function writeMaintainerScripts(control) {
  const postinst = join(control, "postinst"), prerm = join(control, "prerm");
  await writeFile(postinst, ["#!/bin/sh", "set -e", "command -v update-mime-database >/dev/null && update-mime-database /usr/share/mime || true",
    "command -v update-desktop-database >/dev/null && update-desktop-database /usr/share/applications || true", "exit 0", ""].join("\n"), { mode: 0o755 });
  await writeFile(prerm, ["#!/bin/sh", "set -e", "if [ \"$1\" = remove ] || [ \"$1\" = deconfigure ]; then",
    "  systemctl disable --now mycellios-node.service >/dev/null 2>&1 || true", "  rm -f /etc/systemd/system/mycellios-node.service",
    "  systemctl daemon-reload >/dev/null 2>&1 || true", "fi", "exit 0", ""].join("\n"), { mode: 0o755 });
  await Promise.all([chmod(postinst, 0o755), chmod(prerm, 0o755)]);
}

async function canonicalMd5Sums(root) {
  const paths = [];
  async function visit(directory) {
    const entries = await readdir(directory, { withFileTypes: true }); entries.sort((a, b) => a.name.localeCompare(b.name));
    for (const entry of entries) {
      const absolute = join(directory, entry.name); const path = relative(root, absolute).replaceAll("\\", "/");
      if (path === "DEBIAN" || path.startsWith("DEBIAN/")) continue;
      if (entry.isDirectory()) await visit(absolute);
      else if (entry.isFile()) paths.push(path);
      else throw new Error(`node_deb_special_file_is_forbidden:${path}`);
    }
  }
  await visit(root);
  return `${(await Promise.all(paths.map(async (path) => `${createHash("md5").update(await readFile(join(root, ...path.split("/")))).digest("hex")}  ${path}`))).join("\n")}\n`;
}

async function main(argv) {
  const values = Object.fromEntries(argv.map((argument) => { const match = /^--([^=]+)=(.+)$/.exec(argument); if (!match) throw new Error(`node_deb_argument_is_invalid:${argument}`); return [match[1], match[2]]; }));
  for (const key of ["staged-root", "output", "version", "source-revision"]) if (!values[key]) throw new Error(`node_deb_argument_is_missing:${key}`);
  await buildNodeDeb({ stagedRoot: values["staged-root"], output: values.output, version: values.version, sourceRevision: values["source-revision"], sourceDateEpoch: values["source-date-epoch"] });
}
if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) await main(process.argv.slice(2));
