export interface NativePythonManifestEntry {
  path: string;
  bytes: number;
  sha256: string;
}

export interface NativePythonProductManifest {
  schema: "mycellios-native-python-product/1";
  policyId: string;
  entryModules: string[];
  files: NativePythonManifestEntry[];
}

export const NATIVE_PYTHON_PRODUCT_SCHEMA:
  "mycellios-native-python-product/1";
export const NATIVE_PYTHON_PRODUCT_MANIFEST:
  "mycellios-native-python-manifest.json";
export const NATIVE_PYTHON_ENTRY_MODULES: readonly string[];
export const NATIVE_PYTHON_PRODUCT_FILES: readonly string[];
export const NATIVE_PYTHON_IMPORT_SMOKE_MODULES: readonly string[];

export function prepareNativePythonProductSource(
  sourceRoot: string,
  destinationRoot: string,
): NativePythonProductManifest;
export function assertNativePythonSourceClosure(sourceRoot: string): void;
export function verifyNativePythonProductSource(
  sourceRoot: string,
): NativePythonProductManifest;
export function assertNativePythonProductMatchesSource(
  sourceRoot: string,
  productRoot: string,
): NativePythonProductManifest;
export function buildNativePythonProductManifest(
  sourceRoot: string,
): NativePythonProductManifest;
