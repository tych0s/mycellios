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

export const EXECUTOR_ISOLATION_SCHEMA = "gdlp-executor-isolation/2" as const;

export interface ExecutorIsolationPolicyOptions {
  maxOutputBytesPerStream?: number;
  stopGraceMs?: number;
}

/**
 * Honest description of the controls enforced by the current native launcher.
 *
 * This contract intentionally records the controls that are not implemented
 * yet. A launch cannot claim an OS sandbox, a process-tree boundary or resource
 * quotas until the corresponding native backend can enforce them.
 */
export interface ExecutorIsolationPolicyV2 {
  schema: typeof EXECUTOR_ISOLATION_SCHEMA;
  environmentPolicy: "inherit-reviewed-system-keys-plus-trusted-overrides";
  inheritedEnvironmentKeys: string[];
  executablePolicy: "exact-prepared-command";
  workspacePolicy: "private-temp-shared-runtime";
  processTreePolicy: "direct-child-only";
  resourceLimitPolicy: "not-enforced";
  maxOutputBytesPerStream: number;
  stopGraceMs: number;
}

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

/**
 * Normalize the policy included in every compiled process and therefore in
 * the launch identity. The compact options form is accepted only at compile
 * time; normalized launch descriptions always carry the complete contract.
 */
export function normalizeExecutorIsolationPolicy(
  value?: ExecutorIsolationPolicyOptions | ExecutorIsolationPolicyV2 | null,
): ExecutorIsolationPolicyV2 {
  if (value !== undefined && value !== null && !isRecord(value)) {
    throw new Error("executor_isolation_policy_must_be_an_object");
  }
  const source = value ?? {};
  const isNormalized = source.schema !== undefined;
  if (isNormalized) {
    assertExactPolicyKeys(source);
    if (source.schema !== EXECUTOR_ISOLATION_SCHEMA) {
      throw new Error("executor_isolation_schema_is_invalid");
    }
    if (
      source.environmentPolicy
      !== "inherit-reviewed-system-keys-plus-trusted-overrides"
    ) {
      throw new Error("executor_isolation_environment_policy_is_unsupported");
    }
    if (
      !Array.isArray(source.inheritedEnvironmentKeys)
      || source.inheritedEnvironmentKeys.some((key) => typeof key !== "string")
      || !sameStrings(
        source.inheritedEnvironmentKeys,
        ISOLATED_PROCESS_INHERITED_ENVIRONMENT_KEYS,
      )
    ) {
      throw new Error("executor_isolation_environment_allowlist_mismatch");
    }
    if (source.executablePolicy !== "exact-prepared-command") {
      throw new Error("executor_isolation_executable_policy_is_unsupported");
    }
    if (source.workspacePolicy !== "private-temp-shared-runtime") {
      throw new Error("executor_isolation_workspace_policy_is_unsupported");
    }
    if (source.processTreePolicy !== "direct-child-only") {
      throw new Error("executor_isolation_process_tree_policy_is_unsupported");
    }
    if (source.resourceLimitPolicy !== "not-enforced") {
      throw new Error("executor_isolation_resource_limit_policy_is_unsupported");
    }
  } else {
    assertExactOptionKeys(source);
  }

  return {
    schema: EXECUTOR_ISOLATION_SCHEMA,
    environmentPolicy: "inherit-reviewed-system-keys-plus-trusted-overrides",
    inheritedEnvironmentKeys: [...ISOLATED_PROCESS_INHERITED_ENVIRONMENT_KEYS],
    executablePolicy: "exact-prepared-command",
    workspacePolicy: "private-temp-shared-runtime",
    processTreePolicy: "direct-child-only",
    resourceLimitPolicy: "not-enforced",
    maxOutputBytesPerStream: boundedInteger(
      source.maxOutputBytesPerStream ?? 64 * 1024,
      1_024,
      16 * 1024 * 1024,
      "executor_isolation_output_limit_is_invalid",
    ),
    stopGraceMs: boundedInteger(
      source.stopGraceMs ?? 5_000,
      1,
      300_000,
      "executor_isolation_stop_grace_is_invalid",
    ),
  };
}

export function validateExecutorIsolationPolicy(
  value: unknown,
): asserts value is ExecutorIsolationPolicyV2 {
  if (!isRecord(value) || value.schema === undefined) {
    throw new Error("executor_isolation_policy_is_not_normalized");
  }
  normalizeExecutorIsolationPolicy(
    value as unknown as ExecutorIsolationPolicyV2,
  );
}

function assertExactPolicyKeys(value: Record<string, unknown>): void {
  const expected = [
    "schema",
    "environmentPolicy",
    "inheritedEnvironmentKeys",
    "executablePolicy",
    "workspacePolicy",
    "processTreePolicy",
    "resourceLimitPolicy",
    "maxOutputBytesPerStream",
    "stopGraceMs",
  ];
  if (!sameStrings(Object.keys(value).sort(), [...expected].sort())) {
    throw new Error("executor_isolation_policy_has_unknown_or_missing_fields");
  }
}

function assertExactOptionKeys(value: Record<string, unknown>): void {
  const allowed = new Set(["maxOutputBytesPerStream", "stopGraceMs"]);
  if (Object.keys(value).some((key) => !allowed.has(key))) {
    throw new Error("executor_isolation_options_have_unknown_fields");
  }
}

function sameStrings(
  left: readonly string[],
  right: readonly string[],
): boolean {
  return left.length === right.length
    && left.every((value, index) => value === right[index]);
}

function boundedInteger(
  value: unknown,
  minimum: number,
  maximum: number,
  message: string,
): number {
  if (
    typeof value !== "number"
    || !Number.isSafeInteger(value)
    || value < minimum
    || value > maximum
  ) {
    throw new Error(message);
  }
  return value;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === "object" && !Array.isArray(value);
}
