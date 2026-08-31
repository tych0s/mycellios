export interface DeveloperEnvironmentInput {
  nodeVersion: string;
  npmVersion: string | null;
  pythonVersion: string | null;
  hasLockfile: boolean;
  hasAlternativeLock: boolean;
  gitVersion: string | null;
  powershellVersion: string | null;
  platform: NodeJS.Platform;
}

export interface DeveloperEnvironmentCheck {
  id: string;
  required: boolean;
  ok: boolean;
  detail: string;
  fix: string;
}

export interface DeveloperEnvironmentResult {
  ok: boolean;
  fullRuntimeReady: boolean;
  checks: DeveloperEnvironmentCheck[];
  platform: NodeJS.Platform;
  capabilities: { controlPlane: boolean; pythonRuntime: boolean; portableTwoHostPreflight: boolean; windowsPhysicalScripts: boolean; powershell: boolean };
}

export function evaluateDeveloperEnvironment(input: DeveloperEnvironmentInput): DeveloperEnvironmentResult;
export function inspectDeveloperEnvironment(root?: string): DeveloperEnvironmentResult;
