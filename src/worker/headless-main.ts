import { loadWorkerConfig } from "../core/config.js";
import { WorkerAgent } from "./agent.js";
import {
  createHeadlessRuntime,
  loadHeadlessWorkerEnvironment,
} from "./headless-runtime.js";

const environment = loadHeadlessWorkerEnvironment();
const runtime = await createHeadlessRuntime(environment);
const agent = new WorkerAgent(loadWorkerConfig(environment.configPath), {
  coordinatorUrl: environment.coordinatorUrl,
  ...(environment.networkToken ? { networkToken: environment.networkToken } : {}),
  reconnect: true,
  advertiseDeployment: false,
  agentVersion: environment.appVersion,
  verifiedGpuRuntime: runtime.verifiedGpuRuntime,
  preferredHardwareGpu: runtime.preferredHardwareGpu,
  distributedExecutor: runtime.executor,
});

for (const signal of ["SIGINT", "SIGTERM"] as const) {
  process.once(signal, () => void agent.stop());
}

await agent.start();
