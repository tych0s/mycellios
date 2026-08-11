import { createHash } from "node:crypto";
import { readFile, stat } from "node:fs/promises";
import { basename, join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { buildNodeCycloneDx, NODE_RELEASE_EVIDENCE_SCHEMA } from "./node-release-evidence.mjs";
import { verifyNodeInstaller } from "./verify-node-installer.mjs";

export async function verifyNodeReleaseEvidence(input) {
  const artifact = resolve(input.artifact), staged = resolve(input.stagedRoot);
  const [artifactBytes, artifactStats, evidence, sbom, checksum, layout, layoutBytes, runtime, ws, zod] = await Promise.all([
    readFile(artifact), stat(artifact), readJson(input.evidencePath), readJson(input.sbomPath), readFile(input.checksumPath, "utf8"),
    verifyNodeInstaller(staged), readFile(join(staged, "layout-manifest.json")), readJson(join(staged, "runtime/runtime-manifest.json")),
    readJson(join(staged, "node_modules/ws/package.json")), readJson(join(staged, "node_modules/zod/package.json")),
  ]);
  if (evidence.schema !== NODE_RELEASE_EVIDENCE_SCHEMA) throw new Error("node_release_evidence_schema_is_invalid");
  const artifactSha = sha256(artifactBytes), name = basename(artifact);
  if (evidence.artifact?.name !== name || evidence.artifact?.sha256 !== artifactSha || evidence.artifact?.bytes !== artifactStats.size || evidence.artifact?.target !== layout.target) {
    throw new Error("node_release_artifact_evidence_mismatch");
  }
  if (evidence.source?.revision !== layout.source.revision || evidence.source?.sourceId !== layout.source.sourceId
    || evidence.layout?.sha256 !== sha256(layoutBytes) || evidence.layout?.fileCount !== layout.files.length) {
    throw new Error("node_release_source_or_layout_evidence_mismatch");
  }
  const expectedSbom = buildNodeCycloneDx({ layout, layoutSha256: sha256(layoutBytes), runtime, dependencies: { ws, zod } });
  if (JSON.stringify(sbom) !== JSON.stringify(expectedSbom)) throw new Error("node_release_sbom_mismatch");
  const sbomBytes = await readFile(input.sbomPath);
  if (evidence.sbom?.sha256 !== sha256(sbomBytes) || checksum !== `${artifactSha}  ${name}\n`) throw new Error("node_release_checksum_evidence_mismatch");
  if (evidence.signature?.requiredForPromotion !== true) throw new Error("node_release_signature_policy_is_missing");
  if (input.requireSigned && evidence.signature?.state !== "signed") throw new Error("node_release_artifact_is_not_signed");
  return evidence;
}
async function readJson(path) { return JSON.parse(await readFile(path, "utf8")); }
function sha256(bytes) { return createHash("sha256").update(bytes).digest("hex"); }
if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) { const values = Object.fromEntries(process.argv.slice(2).map((argument) => { const match = /^--([^=]+)=(.+)$/.exec(argument); if (!match) throw new Error(`node_release_verify_argument_is_invalid:${argument}`); return [match[1], match[2]]; }));
  for (const key of ["artifact", "staged-root", "evidence", "sbom", "checksum"]) if (!values[key]) throw new Error(`node_release_verify_argument_is_missing:${key}`);
  await verifyNodeReleaseEvidence({ artifact: values.artifact, stagedRoot: values["staged-root"], evidencePath: values.evidence, sbomPath: values.sbom, checksumPath: values.checksum, requireSigned: values["require-signed"] === "true" }); process.stdout.write(`Native node release evidence verified: ${values.artifact}\n`); }
