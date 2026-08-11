import type { NodeInstallerLayoutManifest } from "./stage-node-installer.mjs";
export function buildNodeDeb(input: { stagedRoot: string; output: string; version: string; sourceRevision: string; sourceDateEpoch?: string }): Promise<{ output: string; layout: NodeInstallerLayoutManifest }>;
