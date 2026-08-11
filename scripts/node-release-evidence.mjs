import { createHash } from "node:crypto";
import { mkdir, readFile, stat, writeFile } from "node:fs/promises";
import { basename, extname, join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { verifyNodeInstaller } from "./verify-node-installer.mjs";

export const NODE_RELEASE_EVIDENCE_SCHEMA = "mycellios-node-release-evidence/1";

export async function writeNodeReleaseEvidence(input) {
  const artifact = resolve(input.artifact), stagedRoot = resolve(input.stagedRoot), output = resolve(input.outputDirectory);
  const [artifactBytes, artifactStats, layout, layoutBytes, runtime, ws, zod] = await Promise.all([
    readFile(artifact), stat(artifact), verifyNodeInstaller(stagedRoot), readFile(join(stagedRoot, "layout-manifest.json")),
    readJson(join(stagedRoot, "runtime", "runtime-manifest.json")), readJson(join(stagedRoot, "node_modules", "ws", "package.json")),
    readJson(join(stagedRoot, "node_modules", "zod", "package.json")),
  ]);
  const name = basename(artifact), format = extname(name).slice(1).toLowerCase();
  if (!new Set(["deb", "pkg", "msi"]).has(format)) throw new Error("node_release_artifact_format_is_invalid");
  if (!/^\d+$/.test(input.sourceDateEpoch)) throw new Error("node_release_source_date_epoch_is_invalid");
  const sbom = buildNodeCycloneDx({ layout, layoutSha256: sha256(layoutBytes), runtime, dependencies: { ws, zod } });
  const sbomBytes = Buffer.from(`${JSON.stringify(sbom, null, 2)}\n`);
  const evidence = {
    schema: NODE_RELEASE_EVIDENCE_SCHEMA,
    artifact: { name, format, target: layout.target, bytes: artifactStats.size, sha256: sha256(artifactBytes) },
    source: layout.source,
    layout: { schema: layout.schema, sha256: sha256(layoutBytes), fileCount: layout.files.length },
    sbom: { format: "CycloneDX", specVersion: "1.6", name: `${name}.cdx.json`, sha256: sha256(sbomBytes) },
    builder: { os: input.builderOs, arch: input.builderArch, workflow: input.workflow, runId: input.runId, runAttempt: input.runAttempt },
    builtAt: new Date(Number(input.sourceDateEpoch) * 1_000).toISOString(),
    signature: { state: "pending", requiredForPromotion: true },
  };
  await mkdir(output, { recursive: true });
  const sbomPath = join(output, `${name}.cdx.json`), evidencePath = join(output, `${name}.provenance.json`), checksumPath = join(output, `${name}.sha256`);
  await writeFile(sbomPath, sbomBytes, { flag: "wx" });
  await writeFile(evidencePath, `${JSON.stringify(evidence, null, 2)}\n`, { flag: "wx" });
  await writeFile(checksumPath, `${evidence.artifact.sha256}  ${name}\n`, { flag: "wx" });
  return { evidence, sbom, paths: { sbomPath, evidencePath, checksumPath } };
}

export function buildNodeCycloneDx(input) {
  const version = versionFromProvenance(input.layout);
  const components = [
    component("application", "mycellios-node", version, `pkg:generic/mycellios-node@${version}?target=${encodeURIComponent(input.layout.target)}`),
    component("framework", "node", input.layout.toolchain.nodeVersion.replace(/^v/, ""), `pkg:generic/node@${input.layout.toolchain.nodeVersion.replace(/^v/, "")}`),
    component("framework", "python", input.layout.toolchain.pythonVersion, `pkg:generic/python@${input.layout.toolchain.pythonVersion}`),
    component("library", "ws", String(input.dependencies.ws.version ?? "unknown"), `pkg:npm/ws@${input.dependencies.ws.version ?? "unknown"}`),
    component("library", "zod", String(input.dependencies.zod.version ?? "unknown"), `pkg:npm/zod@${input.dependencies.zod.version ?? "unknown"}`),
    ...pythonComponents(input.runtime),
  ].sort((left, right) => left["bom-ref"].localeCompare(right["bom-ref"]));
  const serial = deterministicUuid(`${input.layout.source.sourceId}:${input.layout.target}:${input.layoutSha256}`);
  return { bomFormat: "CycloneDX", specVersion: "1.6", serialNumber: `urn:uuid:${serial}`, version: 1,
    metadata: { component: components.find((entry) => entry.name === "mycellios-node"), properties: [
      { name: "io.mycellios.source.revision", value: input.layout.source.revision }, { name: "io.mycellios.source.id", value: input.layout.source.sourceId },
      { name: "io.mycellios.layout.sha256", value: input.layoutSha256 }, { name: "io.mycellios.target", value: input.layout.target },
    ] }, components };
}

function pythonComponents(runtime) { const fields = [["torch", runtime.torchVersion], ["transformers", runtime.transformersVersion], ["accelerate", runtime.accelerateVersion],
  ["safetensors", runtime.safetensorsVersion], ["aiohttp", runtime.aiohttpVersion], ["sentencepiece", runtime.sentencepieceVersion], ["numpy", runtime.numpyVersion]];
  return fields.filter(([, version]) => typeof version === "string").map(([name, version]) => component("library", name, version, `pkg:pypi/${name}@${version}`)); }
function component(type, name, version, purl) { return { type, name, version, purl, "bom-ref": purl }; }
function versionFromProvenance(layout) { if (typeof layout.source?.version !== "string" || !layout.source.version) throw new Error("node_release_source_provenance_is_missing"); return layout.source.version; }
function deterministicUuid(value) { const bytes = Buffer.from(createHash("sha256").update(value).digest().subarray(0, 16)); bytes[6] = (bytes[6] & 0x0f) | 0x50; bytes[8] = (bytes[8] & 0x3f) | 0x80; const hex = bytes.toString("hex"); return `${hex.slice(0,8)}-${hex.slice(8,12)}-${hex.slice(12,16)}-${hex.slice(16,20)}-${hex.slice(20)}`; }
async function readJson(path) { return JSON.parse(await readFile(path, "utf8")); }
function sha256(bytes) { return createHash("sha256").update(bytes).digest("hex"); }

async function main(argv) { const values = Object.fromEntries(argv.map((argument) => { const match = /^--([^=]+)=(.+)$/.exec(argument); if (!match) throw new Error(`node_release_evidence_argument_is_invalid:${argument}`); return [match[1], match[2]]; }));
  for (const key of ["artifact", "staged-root", "output-directory", "source-date-epoch", "builder-os", "builder-arch", "workflow", "run-id", "run-attempt"]) if (!values[key]) throw new Error(`node_release_evidence_argument_is_missing:${key}`);
  await writeNodeReleaseEvidence({ artifact: values.artifact, stagedRoot: values["staged-root"], outputDirectory: values["output-directory"], sourceDateEpoch: values["source-date-epoch"], builderOs: values["builder-os"], builderArch: values["builder-arch"], workflow: values.workflow, runId: values["run-id"], runAttempt: values["run-attempt"] }); }
if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) await main(process.argv.slice(2));
