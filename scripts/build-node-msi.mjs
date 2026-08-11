import { createHash } from "node:crypto";
import { spawnSync } from "node:child_process";
import { mkdir, rm, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { verifyNodeInstaller } from "./verify-node-installer.mjs";

export async function buildNodeMsi(input) {
  const staged = resolve(input.stagedRoot), output = resolve(input.output);
  const layout = await verifyNodeInstaller(staged); if (layout.target !== "windows-x64") throw new Error("node_msi_layout_target_mismatch");
  const source = `${output}.wxs`, object = `${output}.wixobj`; await mkdir(dirname(output), { recursive: true });
  await Promise.all([rm(output, { force: true }), rm(source, { force: true }), rm(object, { force: true })]);
  await writeFile(source, generateWixSource({ stagedRoot: staged, version: input.version, sourceRevision: input.sourceRevision, files: layout.files }), "utf8");
  run("candle.exe", ["-nologo", "-arch", "x64", "-out", object, source], "node_msi_compile_failed");
  run("light.exe", ["-nologo", "-sval", "-out", output, object], "node_msi_link_failed");
  await rm(object, { force: true }); return { output, source, layout };
}

export function generateWixSource(input) {
  if (!/^\d+\.\d+\.\d+$/.test(input.version)) throw new Error("node_msi_version_is_invalid");
  if (!/^[a-f0-9]{40}$/.test(input.sourceRevision)) throw new Error("node_msi_source_revision_is_invalid");
  const root = directoryNode("INSTALLFOLDER", "Mycellios"); const componentIds = [];
  for (const file of input.files) {
    const parts = file.path.split("/"); const name = parts.pop(); if (!name || parts.some((part) => !part || part === "." || part === "..")) throw new Error("node_msi_file_path_is_invalid");
    let directory = root; for (const part of parts) directory = childDirectory(directory, part);
    const id = wixId("cmp", file.path); componentIds.push(id);
    directory.components.push(`<Component Id="${id}" Guid="${guid(`component:${file.path}`)}" Win64="yes"><File Id="${wixId("fil", file.path)}" Source="${xml(`${resolve(input.stagedRoot, ...file.path.split("/"))}`)}" KeyPath="yes" Checksum="yes" /></Component>`);
  }
  const registryId = "cmp_pairing_association"; componentIds.push(registryId);
  root.components.push(`<Component Id="${registryId}" Guid="${guid("component:pairing-association")}" Win64="yes">
<RegistryValue Root="HKLM" Key="Software\\Classes\\.mycellios-enrollment" Value="Mycellios.Enrollment" Type="string" KeyPath="yes" />
<RegistryValue Root="HKLM" Key="Software\\Classes\\Mycellios.Enrollment" Value="Mycellios one-time computer pairing" Type="string" />
<RegistryValue Root="HKLM" Key="Software\\Classes\\Mycellios.Enrollment\\shell\\open\\command" Value="&quot;[System64Folder]WindowsPowerShell\\v1.0\\powershell.exe&quot; -NoProfile -ExecutionPolicy Bypass -File &quot;[INSTALLFOLDER]install.ps1&quot; -Enrollment &quot;%1&quot;" Type="string" />
</Component>`);
  return `<?xml version="1.0" encoding="UTF-8"?>
<Wix xmlns="http://schemas.microsoft.com/wix/2006/wi">
<Product Id="${guid(`product:${input.version}:${input.sourceRevision}`)}" Name="Mycellios Node" Language="1033" Version="${input.version}" Manufacturer="Mycellios" UpgradeCode="${guid("upgrade:io.mycellios.node")}">
<Package Id="${guid(`package:${input.version}:${input.sourceRevision}`)}" InstallerVersion="500" Compressed="yes" InstallScope="perMachine" Platform="x64" Description="Native Mycellios contribution node" />
<MajorUpgrade DowngradeErrorMessage="A newer Mycellios Node is already installed." />
<MediaTemplate EmbedCab="yes" CompressionLevel="high" />
<Property Id="MYCELLIOSSOURCEREVISION" Value="${input.sourceRevision}" />
<Directory Id="TARGETDIR" Name="SourceDir"><Directory Id="ProgramFiles64Folder">${renderDirectory(root)}</Directory></Directory>
<Feature Id="ProductFeature" Title="Mycellios Node" Level="1">${componentIds.map((id) => `<ComponentRef Id="${id}" />`).join("")}</Feature>
</Product></Wix>
`;
}

function directoryNode(id, name) { return { id, name, children: new Map(), components: [] }; }
function childDirectory(parent, name) { let child = parent.children.get(name); if (!child) { child = directoryNode(wixId("dir", `${parent.id}/${name}`), name); parent.children.set(name, child); } return child; }
function renderDirectory(node) { return `<Directory Id="${node.id}" Name="${xml(node.name)}">${node.components.join("")}${[...node.children.values()].map(renderDirectory).join("")}</Directory>`; }
function wixId(prefix, value) { return `${prefix}_${createHash("sha256").update(value).digest("hex").slice(0, 24)}`; }
function guid(value) { const bytes = Buffer.from(createHash("sha256").update(value).digest().subarray(0, 16)); bytes[6] = (bytes[6] & 0x0f) | 0x50; bytes[8] = (bytes[8] & 0x3f) | 0x80; const hex = bytes.toString("hex"); return `${hex.slice(0,8)}-${hex.slice(8,12)}-${hex.slice(12,16)}-${hex.slice(16,20)}-${hex.slice(20)}`; }
function xml(value) { return value.replaceAll("&", "&amp;").replaceAll('"', "&quot;").replaceAll("<", "&lt;").replaceAll(">", "&gt;"); }
function run(executable, arguments_, error) { const result = spawnSync(executable, arguments_, { encoding: "utf8", shell: false }); if (result.error) throw result.error; if (result.status !== 0) throw new Error(`${error}:${result.stderr.trim()}`); }

async function main(argv) { const values = Object.fromEntries(argv.map((argument) => { const match = /^--([^=]+)=(.+)$/.exec(argument); if (!match) throw new Error(`node_msi_argument_is_invalid:${argument}`); return [match[1], match[2]]; }));
  for (const key of ["staged-root", "output", "version", "source-revision"]) if (!values[key]) throw new Error(`node_msi_argument_is_missing:${key}`);
  await buildNodeMsi({ stagedRoot: values["staged-root"], output: values.output, version: values.version, sourceRevision: values["source-revision"] }); }
if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) await main(process.argv.slice(2));
