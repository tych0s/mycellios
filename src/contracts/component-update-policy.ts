/**
 * One policy shared by the signer, coordinator and native node. A release that can
 * be published must therefore also be installable by every compatible client.
 */
export const MAX_COMPONENT_ARTIFACT_BYTES = 8 * 1024 * 1024;

export const COMPONENT_FILES_POLICY = Object.freeze({
  maxFiles: 4_096,
  maxFileBytes: 16 * 1024 * 1024,
  maxTotalFileBytes: 32 * 1024 * 1024,
  maxCompressedBytes: MAX_COMPONENT_ARTIFACT_BYTES,
  maxDocumentBytes: 48 * 1024 * 1024,
  maxPathBytes: 1_024,
});

export interface ManagedComponentPolicy {
  restartScope: "runtime" | "agent";
  backend: "any" | "cpu" | "cuda" | "rocm" | "metal";
  driverApi: string | null;
  dependencies: readonly string[];
  targets: readonly `${"win32" | "linux" | "darwin"}/${"x64" | "arm64"}`[];
}

export const MANAGED_COMPONENT_POLICIES = Object.freeze({
  "node-bootstrap": Object.freeze({
    restartScope: "agent", backend: "any", driverApi: null, dependencies: [] as const,
    targets: ["win32/x64", "linux/x64", "darwin/arm64"] as const,
  }),
  "python-product": Object.freeze({
    restartScope: "runtime", backend: "any", driverApi: null, dependencies: [] as const,
    targets: ["win32/x64", "linux/x64", "darwin/arm64"] as const,
  }),
  "runtime-cpu": Object.freeze({
    restartScope: "runtime", backend: "cpu", driverApi: null, dependencies: ["python-product"] as const,
    targets: ["win32/x64", "linux/x64", "darwin/arm64"] as const,
  }),
  "runtime-cuda": Object.freeze({
    restartScope: "runtime", backend: "cuda", driverApi: "nvidia-display", dependencies: ["python-product"] as const,
    targets: ["win32/x64"] as const,
  }),
  "runtime-metal": Object.freeze({
    restartScope: "runtime", backend: "metal", driverApi: null, dependencies: ["python-product"] as const,
    targets: ["darwin/arm64"] as const,
  }),
  "runtime-rocm": Object.freeze({
    restartScope: "runtime", backend: "rocm", driverApi: "amd-windows", dependencies: ["python-product"] as const,
    targets: ["win32/x64"] as const,
  }),
} as const satisfies Record<string, ManagedComponentPolicy>);

export type ManagedComponentId = keyof typeof MANAGED_COMPONENT_POLICIES;

export const MANAGED_COMPONENT_RESTART_SCOPES = Object.freeze(Object.fromEntries(
  Object.entries(MANAGED_COMPONENT_POLICIES).map(([id, policy]) => [id, policy.restartScope]),
) as Record<ManagedComponentId, ManagedComponentPolicy["restartScope"]>);
