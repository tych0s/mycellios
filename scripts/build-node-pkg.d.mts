import type { NodeInstallerLayoutManifest } from "./stage-node-installer.mjs";
export function buildNodePkg(input: { stagedRoot: string; output: string; version: string; sourceDateEpoch?: string }): Promise<{ output: string; layout: NodeInstallerLayoutManifest }>;
export function materializeMacPairingApp(root: string): Promise<void>;
