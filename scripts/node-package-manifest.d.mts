export const NODE_PACKAGE_MANIFEST_SCHEMA: "mycellios-node-package/2";
export interface NodePackageManifestInput {
  entrypointPath: string;
  entrypoint: string;
  bootstrapEntrypointPath: string;
  bootstrapEntrypoint: string;
  helperEntrypointPath: string;
  helperEntrypoint: string;
  target: string;
  sourceRevision: string;
}
export interface NodePackageManifest {
  schema: typeof NODE_PACKAGE_MANIFEST_SCHEMA;
  target: string;
  entrypoint: string;
  sourceRevision: string;
  sha256: string;
  installationBootstrap: { entrypoint: string; sha256: string };
  uninstallHelper: { entrypoint: string; sha256: string };
}
export function buildNodePackageManifest(input: NodePackageManifestInput): Promise<NodePackageManifest>;
