import { randomBytes } from "node:crypto";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import type {
  WorkerAcceleratorDiagnostics,
  WorkerPhysicalIdentity,
} from "../contracts/types.js";
import { sha256Text } from "../core/json.js";
import {
  LocalProcessAgent,
  type LaunchAgent,
  type LaunchAgentStartRequest,
  type LaunchProcessHandle,
  type RuntimePreparationProgressEvent,
} from "../distribution/launch-supervisor.js";
import type {
  PythonLaunchProcess,
  PythonPipelineLaunchDescription,
} from "../distribution/python-launcher.js";
import {
  PythonPhysicalProbe,
  type PhysicalProbeCollector,
  type PhysicalProbeV1,
} from "../distribution/physical-probe.js";
import type { WorkerAgentOptions } from "./agent.js";
import type { VerifiedGpuRuntimeEvidence } from "./hardware.js";
import type { NodeConfiguration } from "../contracts/node-configuration.js";
import {
  prepareNodeStageArtifacts,
  type StageArtifactPreparationOptions,
} from "./stage-artifact-preparer.js";

const NODE_ID = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;
const LOOPBACK_HOSTS = new Set(["localhost", "127.0.0.1", "::1", "[::1]"]);

export interface HeadlessWorkerEnvironment {
  configPath: string;
  coordinatorUrl: string;
  networkToken?: string;
  workerCredentialPath?: string;
  nodeId: string;
  provider: "salad" | "generic";
  providerMachineId: string;
  stagePort: number;
  pythonExecutable: string;
  pythonPath: string;
  cachePath: string;
  maxWorkspaceBytes: number;
  windowsJobBrokerExecutable?: string;
  appVersion: string;
}

export interface HeadlessRuntime {
  executor: NonNullable<WorkerAgentOptions["distributedExecutor"]>;
  verifiedGpuRuntime: VerifiedGpuRuntimeEvidence;
  preferredHardwareGpu: NonNullable<WorkerAgentOptions["preferredHardwareGpu"]>;
  acceleration: WorkerAcceleratorDiagnostics;
  physicalProbe: PhysicalProbeV1;
}

export function headlessEnvironmentFromNodeConfiguration(
  config: NodeConfiguration,
  appVersion: string,
): HeadlessWorkerEnvironment {
  return {
    configPath: config.worker.configPath,
    coordinatorUrl: config.coordinator.url.replace(/\/$/, ""),
    nodeId: config.nodeId,
    provider: "generic",
    providerMachineId: config.nodeId,
    stagePort: config.runtime.stagePort,
    pythonExecutable: config.runtime.pythonExecutable,
    pythonPath: config.runtime.pythonPath,
    cachePath: config.runtime.cachePath,
    maxWorkspaceBytes: config.limits.maxDiskMiB * 1024 * 1024,
    ...(config.isolation.mode === "windows-job-object"
      ? { windowsJobBrokerExecutable: config.isolation.brokerExecutable }
      : {}),
    appVersion,
  };
}

type StageArtifactPreparer = (
  description: PythonPipelineLaunchDescription,
  options: StageArtifactPreparationOptions,
) => Promise<PythonLaunchProcess[]>;

/**
 * Product node launch boundary: materialize and verify only this node's sealed
 * stage ranges before allowing any process from the plan to start.
 */
export class HeadlessStageLaunchAgent implements LaunchAgent {
  readonly id: string;

  constructor(
    private readonly processAgent: LaunchAgent,
    private readonly config: Pick<
      HeadlessWorkerEnvironment,
      "nodeId" | "pythonExecutable" | "pythonPath" | "cachePath"
    >,
    private readonly prepareArtifacts: StageArtifactPreparer = prepareNodeStageArtifacts,
  ) {
    this.id = `headless-stage:${config.nodeId}`;
  }

  async prepareRuntime(
    description: PythonPipelineLaunchDescription,
    nodeId: string,
    onProgress?: (event: RuntimePreparationProgressEvent) => void,
  ): Promise<readonly PythonLaunchProcess[]> {
    if (nodeId !== this.config.nodeId) {
      throw new Error("headless_stage_artifact_node_mismatch");
    }
    return await this.prepareArtifacts(description, {
      nodeId,
      pythonExecutable: this.config.pythonExecutable,
      cacheDirectory: this.config.cachePath,
      cwd: process.cwd(),
      environment: {
        PYTHONPATH: this.config.pythonPath,
        HF_HOME: this.config.cachePath,
        TOKENIZERS_PARALLELISM: "false",
      },
      ...(onProgress ? { onProgress } : {}),
    });
  }

  async start(
    request: LaunchAgentStartRequest,
    signal: AbortSignal,
  ): Promise<LaunchProcessHandle> {
    return await this.processAgent.start(request, signal);
  }

  async close(): Promise<void> {
    await this.processAgent.close?.();
  }
}

export function loadHeadlessWorkerEnvironment(
  environment: NodeJS.ProcessEnv = process.env,
  cwd = process.cwd(),
): HeadlessWorkerEnvironment {
  const rawProviderMachineId = environment.SALAD_MACHINE_ID?.trim();
  const explicitNodeId = environment.MYCELLIOS_NODE_ID?.trim();
  if (!explicitNodeId && !rawProviderMachineId) {
    throw new Error("headless_worker_requires_mycellios_node_id_or_salad_machine_id");
  }
  const provider = rawProviderMachineId ? "salad" as const : "generic" as const;
  const providerMachineId = rawProviderMachineId ?? explicitNodeId!;
  const nodeId = explicitNodeId
    ?? `salad-${sha256Text(`gdlp-salad-node-id-v1\0${providerMachineId}`).slice(-32)}`;
  if (!NODE_ID.test(nodeId)) throw new Error("headless_worker_node_id_is_invalid");

  const coordinatorUrl = requiredText(
    environment.GPU_MESH_COORDINATOR,
    "headless_worker_requires_gpu_mesh_coordinator",
  );
  const coordinator = parseCoordinatorUrl(coordinatorUrl);
  const networkToken = loadSecret(
    environment.MYCELLIOS_NETWORK_TOKEN,
    environment.MYCELLIOS_NETWORK_TOKEN_FILE,
    cwd,
  );
  const workerCredentialPath = environment.MYCELLIOS_WORKER_CREDENTIAL_PATH?.trim();
  const loopback = LOOPBACK_HOSTS.has(coordinator.hostname);
  const insecureDevelopment =
    environment.MYCELLIOS_ALLOW_INSECURE_COORDINATOR === "true" && loopback;
  if (coordinator.protocol !== "https:" && !insecureDevelopment) {
    throw new Error("headless_worker_external_coordinator_requires_https");
  }
  if (
    !loopback
    && (!networkToken || networkToken.length < 32)
    && !workerCredentialPath
  ) {
    throw new Error(
      "headless_worker_external_coordinator_requires_scoped_credential_or_32_character_network_token",
    );
  }

  return {
    configPath: resolve(
      cwd,
      environment.GPU_MESH_WORKER_CONFIG?.trim()
        || "./config/worker.salad.example.json",
    ),
    coordinatorUrl: coordinator.href.replace(/\/$/, ""),
    ...(networkToken ? { networkToken } : {}),
    ...(workerCredentialPath
      ? { workerCredentialPath: resolve(cwd, workerCredentialPath) }
      : {}),
    nodeId,
    provider,
    providerMachineId,
    stagePort: positiveInteger(environment.MYCELLIOS_STAGE_PORT, 9_850, 65_535),
    pythonExecutable: environment.MYCELLIOS_PYTHON_EXECUTABLE?.trim() || "python",
    pythonPath: resolve(cwd, environment.MYCELLIOS_PYTHONPATH?.trim() || "python"),
    cachePath: resolve(
      cwd,
      environment.HF_HOME?.trim() || "/var/cache/mycellios/huggingface",
    ),
    maxWorkspaceBytes: 4 * 1024 * 1024 * 1024,
    appVersion: environment.MYCELLIOS_VERSION?.trim() || packageVersion(),
  };
}

export async function createHeadlessRuntime(
  config: HeadlessWorkerEnvironment,
  collector?: PhysicalProbeCollector,
  now: () => Date = () => new Date(),
): Promise<HeadlessRuntime> {
  const runtimeEnvironment = {
    PYTHONPATH: config.pythonPath,
    HF_HOME: config.cachePath,
  };
  const physicalProbe = await (
    collector
      ?? new PythonPhysicalProbe({
        pythonExecutable: config.pythonExecutable,
        cwd: process.cwd(),
        env: runtimeEnvironment,
      })
  ).collect(`headless-${randomBytes(16).toString("hex")}`);
  const device = physicalProbe.devices[0];
  if (!physicalProbe.runtime.cudaApiAvailable || !device) {
    throw new Error("headless_worker_requires_verified_cuda_device");
  }
  const backend = physicalProbe.runtime.rocmVersion ? "rocm" as const : "cuda" as const;
  const verifiedGpuRuntime: VerifiedGpuRuntimeEvidence = {
    status: "gpu-ready",
    backend,
    deviceName: device.name,
  };
  const attestedAt = now().toISOString();
  const physicalIdentity = buildPhysicalIdentity(config, physicalProbe, attestedAt);
  const acceleration: WorkerAcceleratorDiagnostics = {
    schema: "mycellios-accelerator-diagnostics/1",
    appVersion: config.appVersion,
    state: "gpu-ready",
    backend,
    deviceName: device.name,
    gpuVendor: backend === "cuda" ? "nvidia" : "amd",
    gpuModel: device.name,
    phase: "ready",
    progressPct: 100,
    issueCode: null,
    issueSummary: null,
    retryable: false,
    retryAttempt: 0,
    nextRetryAt: null,
    updatedAt: attestedAt,
    recentEvents: [{
      at: attestedAt,
      level: "success",
      message: "Physical GPU probe passed; headless shard executor is ready.",
    }],
  };
  const processAgent = new LocalProcessAgent({
    id: `headless-shard-executor:${config.nodeId}`,
    cwd: process.cwd(),
    env: runtimeEnvironment,
    allowedExecutables: [config.pythonExecutable],
    workspaceRoot: resolve(config.cachePath, "process-workspaces"),
    maxWorkspaceBytes: config.maxWorkspaceBytes,
    ...(config.windowsJobBrokerExecutable
      ? { windowsJobBrokerExecutable: config.windowsJobBrokerExecutable }
      : {}),
  });
  const launchAgent = new HeadlessStageLaunchAgent(processAgent, config);
  return {
    executor: {
      nodeId: config.nodeId,
      stageHost: `${config.nodeId}.relay`,
      stagePort: config.stagePort,
      launchAgent,
      pythonExecutable: config.pythonExecutable,
      computeMode: "gpu-only",
      cpuEligible: false,
      acceleration,
      physicalIdentity,
    },
    verifiedGpuRuntime,
    preferredHardwareGpu: {
      vendor: backend === "cuda" ? "nvidia" : "amd",
      model: device.name,
    },
    acceleration,
    physicalProbe,
  };
}

export function buildPhysicalIdentity(
  config: Pick<HeadlessWorkerEnvironment, "provider" | "providerMachineId">,
  probe: PhysicalProbeV1,
  attestedAt: string,
): WorkerPhysicalIdentity {
  if (probe.devices.length === 0) {
    throw new Error("headless_worker_physical_identity_requires_gpu");
  }
  const providerMachineFingerprintSha256 = sha256Text(
    `gdlp-provider-machine-v1\0${config.provider}\0${config.providerMachineId}`,
  );
  return {
    schema: "gdlp-worker-physical-identity/1",
    provider: config.provider,
    providerMachineFingerprintSha256,
    hostFingerprintSha256: sha256Text(
      [
        "gdlp-provider-bound-host-v1",
        providerMachineFingerprintSha256,
        probe.host.fingerprintSha256,
      ].join("\0"),
    ),
    gpuFingerprintsSha256: probe.devices.map((device) =>
      sha256Text(
        [
          "gdlp-provider-bound-gpu-v1",
          providerMachineFingerprintSha256,
          device.fingerprintSha256,
        ].join("\0"),
      )
    ),
    attestedAt,
  };
}

function parseCoordinatorUrl(value: string): URL {
  let url: URL;
  try {
    url = new URL(value);
  } catch {
    throw new Error("headless_worker_coordinator_url_is_invalid");
  }
  if (
    !["http:", "https:"].includes(url.protocol)
    || url.username
    || url.password
    || url.search
    || url.hash
    || (url.pathname !== "" && url.pathname !== "/")
  ) {
    throw new Error("headless_worker_coordinator_url_is_invalid");
  }
  return url;
}

function loadSecret(
  value: string | undefined,
  file: string | undefined,
  cwd: string,
): string | undefined {
  const inline = value?.trim();
  if (inline) return inline;
  const path = file?.trim();
  if (!path) return undefined;
  const secret = readFileSync(resolve(cwd, path), "utf8").trim();
  if (!secret) throw new Error("headless_worker_network_token_file_is_empty");
  return secret;
}

function requiredText(value: string | undefined, error: string): string {
  const normalized = value?.trim();
  if (!normalized) throw new Error(error);
  return normalized;
}

function positiveInteger(value: string | undefined, fallback: number, maximum: number): number {
  if (value === undefined) return fallback;
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed < 1 || parsed > maximum) {
    throw new Error("headless_worker_positive_integer_is_invalid");
  }
  return parsed;
}

function packageVersion(): string {
  try {
    const packageJson = JSON.parse(
      readFileSync(new URL("../../package.json", import.meta.url), "utf8"),
    ) as { version?: unknown };
    return typeof packageJson.version === "string" && packageJson.version
      ? packageJson.version
      : "headless";
  } catch {
    return "headless";
  }
}
