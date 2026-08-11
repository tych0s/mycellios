import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { NODE_INSTALLER_LAYOUT_SCHEMA, inventoryNodeInstaller } from "./stage-node-installer.mjs";
import { verifyNativeBuildProvenanceDocument } from "./native-build-provenance.mjs";

export async function verifyNodeInstaller(rootValue) {
  const root = resolve(rootValue);
  const manifest = JSON.parse(await readFile(resolve(root, "layout-manifest.json"), "utf8"));
  if (manifest.schema !== NODE_INSTALLER_LAYOUT_SCHEMA || !/^[a-f0-9]{40}$/.test(manifest.sourceRevision)) {
    throw new Error("node_installer_manifest_identity_is_invalid");
  }
  const files = await inventoryNodeInstaller(root, new Set(["layout-manifest.json"]));
  if (JSON.stringify(files) !== JSON.stringify(manifest.files)) throw new Error("node_installer_layout_digest_mismatch");
  for (const entrypoint of Object.values(manifest.entrypoints ?? {})) {
    if (typeof entrypoint !== "string" || !files.some((file) => file.path === entrypoint)) {
      throw new Error("node_installer_entrypoint_is_missing");
    }
  }
  const runtime = await readFile(resolve(root, "runtime", "runtime-manifest.json"));
  if (createHash("sha256").update(runtime).digest("hex") !== manifest.runtimeManifestSha256) {
    throw new Error("node_installer_runtime_manifest_digest_mismatch");
  }
  const provenanceBytes = await readFile(resolve(root, "app", "mycellios-native-build-provenance.json"));
  const provenance = verifyNativeBuildProvenanceDocument(JSON.parse(provenanceBytes.toString("utf8")));
  const identity = JSON.parse(await readFile(resolve(root, "app", "manifest.json"), "utf8"));
  const provenanceSha256 = createHash("sha256").update(provenanceBytes).digest("hex");
  if (manifest.source?.revision !== manifest.sourceRevision || manifest.source?.sourceId !== provenance.sourceId
    || manifest.source?.provenanceSha256 !== provenanceSha256 || manifest.source?.version !== provenance.version || identity.sourceRevision !== manifest.sourceRevision
    || identity.sourceId !== provenance.sourceId || identity.provenanceSha256 !== provenanceSha256) {
    throw new Error("node_installer_source_identity_mismatch");
  }
  return manifest;
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) {
  const root = process.argv.find((argument) => argument.startsWith("--root="))?.slice("--root=".length);
  if (!root) throw new Error("usage: verify-node-installer --root=<path>");
  await verifyNodeInstaller(root);
  process.stdout.write(`Native node installer layout verified: ${root}\n`);
}
