export interface InstalledStageCanaryEvidence {
  schema: "mycellios-installed-stage-canary/1";
  ok: true;
  pythonVersion: string;
  pythonPrefix: string;
  torchVersion: string;
  transformersVersion: string;
  engine: "python-torch";
  adapter: "transformers-llama-v1";
  adapterContractId: string;
  adapterRegistryId: string;
  loader: "selective-safetensors";
  batchSize: 2;
  physicalBatchCalls: number;
  physicalBatchItems: number;
  sequenceTokens: 3;
  kvBytes: number;
  copiedKvBytes: number;
  batchQueueBatches: 1;
  outputTokenSha256: string;
  parity: {
    sequentialVsBatch: true;
    fork: true;
    rollback: true;
  };
}

export interface InstalledStageCanaryOptions {
  pythonExecutable: string;
  pythonSourceRoot: string;
  expectedPythonPrefix?: string;
  expectedTorchVersion?: string;
  expectedTransformersVersion?: string;
  expectedAdapterContractId?: string;
  expectedAdapterRegistryId?: string;
}

export interface InstalledStageCanaryDependencies {
  existsSync?: (path: string) => boolean;
  readFileSync?: (path: string, encoding: "utf8") => string;
  verifyNativePythonProductSource?: (path: string) => unknown;
  spawnSync?: (
    executable: string,
    args: readonly string[],
    options: object,
  ) => {
    error?: Error;
    status: number | null;
    stdout?: string;
    stderr?: string;
  };
}

export const INSTALLED_STAGE_CANARY_SCHEMA: "mycellios-installed-stage-canary/1";
export const INSTALLED_STAGE_CANARY_MARKER: "MYCELLIOS_INSTALLED_STAGE_CANARY=";

export function runInstalledStageCanary(
  options: InstalledStageCanaryOptions,
  dependencies?: InstalledStageCanaryDependencies,
): InstalledStageCanaryEvidence;

export function validateInstalledStageCanaryEvidence(
  evidence: unknown,
  options?: Partial<InstalledStageCanaryOptions>,
): void;
