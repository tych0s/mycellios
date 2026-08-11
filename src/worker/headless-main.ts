import { loadWorkerConfig } from "../core/config.js";
import { WorkerAgent } from "./agent.js";
import {
  loadOrCreateWorkerAdmissionCredential,
  workerAdmissionSigner,
} from "./admission-credential.js";
import {
  createHeadlessRuntime,
  loadHeadlessWorkerEnvironment,
} from "./headless-runtime.js";
import { probeRuntimePerformanceProfile } from "../performance/runtime-profile-probe.js";
import { probeQwen3EngineRuntimeProfile } from "../performance/engine-runtime-profile-probe.js";

const environment = loadHeadlessWorkerEnvironment();
const runtime = await createHeadlessRuntime(environment);
const probedDevice = runtime.physicalProbe.devices[0];
if (!probedDevice) throw new Error("headless_worker_physical_probe_missing_device");
const detectedVramMb = Math.floor(
  Math.min(probedDevice.totalMemoryBytes, probedDevice.runtimeTotalMemoryBytes) / (1024 * 1024),
);
const workerConfig = {
  ...loadWorkerConfig(environment.configPath),
  offeredVramMb: detectedVramMb,
};
const admissionSigner = environment.workerCredentialPath
  ? workerAdmissionSigner(
      loadOrCreateWorkerAdmissionCredential(environment.workerCredentialPath),
    )
  : undefined;
const agent = new WorkerAgent(workerConfig, {
  coordinatorUrl: environment.coordinatorUrl,
  identity: { kind: "device", id: environment.nodeId },
  ...(environment.networkToken ? { networkToken: environment.networkToken } : {}),
  ...(admissionSigner ? { admissionSigner } : {}),
  reconnect: true,
  advertiseDeployment: false,
  agentVersion: environment.appVersion,
  verifiedGpuRuntime: runtime.verifiedGpuRuntime,
  preferredHardwareGpu: runtime.preferredHardwareGpu,
  hardwareCapacityOverride: {
    id: `physical-${probedDevice.index}`,
    vendor: runtime.preferredHardwareGpu.vendor,
    model: probedDevice.name,
    physicalVramMb: detectedVramMb,
  },
  distributedExecutor: runtime.executor,
  runtimePerformanceProfileProbe: () => probeRuntimePerformanceProfile({
    pythonExecutable: environment.pythonExecutable,
    pythonPath: [environment.pythonPath],
    backend: runtime.verifiedGpuRuntime.backend,
    device: "cuda:0",
    precision: "float16",
    expectedDeviceName: runtime.verifiedGpuRuntime.deviceName,
    cwd: process.cwd(),
    env: { HF_HOME: environment.cachePath },
  }),
  engineRuntimeProfileProbe: (challenge) => probeQwen3EngineRuntimeProfile(
    challenge,
    {
      pythonExecutable: environment.pythonExecutable,
      pythonPath: [environment.pythonPath],
      device: "cuda:0",
      precision: "float16",
      cwd: process.cwd(),
      env: { HF_HOME: environment.cachePath },
    },
  ),
});

for (const signal of ["SIGINT", "SIGTERM"] as const) {
  process.once(signal, () => void agent.stop());
}

await agent.start();
