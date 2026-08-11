import type { NodeInstallerLayoutManifest } from "./stage-node-installer.mjs";
export function generateWixSource(input: { stagedRoot: string; version: string; sourceRevision: string; files: NodeInstallerLayoutManifest["files"] }): string;
export function buildNodeMsi(input: { stagedRoot: string; output: string; version: string; sourceRevision: string }): Promise<{ output: string; source: string; layout: NodeInstallerLayoutManifest }>;
