import type { NodeInstallerLayoutManifest } from "./stage-node-installer.mjs";
export const NODE_RELEASE_EVIDENCE_SCHEMA: "mycellios-node-release-evidence/1";
export function buildNodeCycloneDx(input: { layout: NodeInstallerLayoutManifest; layoutSha256: string; runtime: Record<string, unknown>; dependencies: { ws: Record<string, unknown>; zod: Record<string, unknown> } }): Record<string, unknown>;
export function writeNodeReleaseEvidence(input: { artifact: string; stagedRoot: string; outputDirectory: string; sourceDateEpoch: string; builderOs: string; builderArch: string; workflow: string; runId: string; runAttempt: string }): Promise<{ evidence: Record<string, unknown>; sbom: Record<string, unknown>; paths: { sbomPath: string; evidencePath: string; checksumPath: string } }>;
