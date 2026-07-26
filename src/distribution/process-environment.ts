/**
 * Environment keys that an isolated Mycellios child may inherit from its
 * controller without an explicit per-launch grant.
 *
 * Authentication, cloud, proxy, package-manager, model-hub and MYCELLIOS_*
 * variables are deliberately absent. Paths needed by the packaged runtime are
 * passed explicitly by the trusted launcher.
 */
export const ISOLATED_PROCESS_INHERITED_ENVIRONMENT_KEYS = Object.freeze([
  "PATH",
  "HOME",
  "TMPDIR",
  "TEMP",
  "TMP",
  "LANG",
  "LC_ALL",
  "LC_CTYPE",
  "TZ",
  "SSL_CERT_FILE",
  "SSL_CERT_DIR",
  "REQUESTS_CA_BUNDLE",
  "CURL_CA_BUNDLE",
  "SystemRoot",
  "WINDIR",
  "COMSPEC",
  "PATHEXT",
  "USERPROFILE",
  "LOCALAPPDATA",
  "APPDATA",
  "PROGRAMDATA",
  "PROGRAMFILES",
  "PROGRAMFILES(X86)",
  "PROGRAMW6432",
  "LD_LIBRARY_PATH",
  "DYLD_LIBRARY_PATH",
  "CUDA_PATH",
  "ROCM_PATH",
  "HIP_PATH",
] as const);

export interface IsolatedProcessEnvironmentOptions {
  /** Values selected by the trusted launcher for this exact process. */
  overrides?: NodeJS.ProcessEnv;
  /** Parent environment; injectable so the contract can be tested. */
  source?: NodeJS.ProcessEnv;
  /** Additional inherited names, only for a reviewed integration. */
  additionalInheritedKeys?: readonly string[];
}

/**
 * Build a fresh child environment instead of spreading process.env.
 *
 * Explicit `undefined` removes a normally inherited value. This lets a
 * launcher revoke an optional capability without mutating the parent.
 */
export function buildIsolatedProcessEnvironment(
  options: IsolatedProcessEnvironmentOptions = {},
): NodeJS.ProcessEnv {
  const source = options.source ?? process.env;
  const result: NodeJS.ProcessEnv = {};
  const inherited = new Set<string>([
    ...ISOLATED_PROCESS_INHERITED_ENVIRONMENT_KEYS,
    ...(options.additionalInheritedKeys ?? []),
  ]);

  for (const key of inherited) {
    const value = source[key];
    if (value !== undefined) result[key] = value;
  }
  for (const [key, value] of Object.entries(options.overrides ?? {})) {
    if (value === undefined) delete result[key];
    else result[key] = value;
  }
  return result;
}
