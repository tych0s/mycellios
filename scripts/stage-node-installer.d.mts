export const NODE_INSTALLER_LAYOUT_SCHEMA: "mycellios-node-installer-layout/2";
import type { NativeBuildProvenance } from "./native-build-provenance.mjs";
export interface StageNodeInstallerInput { output: string; dist: string; runtime: string; nodeExecutable: string; nodeModules: string; target: string; sourceRevision: string; sourceProvenance: NativeBuildProvenance }
export interface NodeInstallerLayoutManifest { schema: typeof NODE_INSTALLER_LAYOUT_SCHEMA; target: string; sourceRevision: string; source: { revision: string; sourceId: `sha256:${string}`; provenanceSha256: string; version: string }; toolchain: { nodeVersion: string; pythonVersion: string; pythonAbi: string; backend: string }; entrypoints: { service: string; install: string; uninstall: string; launcher: string }; runtimeManifestSha256: string; files: Array<{ path: string; bytes: number; sha256: string }> }
export function stageNodeInstaller(input: StageNodeInstallerInput): Promise<NodeInstallerLayoutManifest>;
export function inventoryNodeInstaller(root: string, excluded?: Set<string>): Promise<NodeInstallerLayoutManifest["files"]>;
