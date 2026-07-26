import { resolve } from "node:path";
import { loadWorkerConfig } from "../core/config.js";
import { readNativeRuntimeBuildMetadata } from "../core/native-build-identity.js";
import { WorkerAgent } from "./agent.js";
import {
  loadOrCreateWorkerAdmissionCredential,
  workerAdmissionSigner,
} from "./admission-credential.js";

const configPath = process.env.GPU_MESH_WORKER_CONFIG ?? "./config/worker.example.json";
const coordinatorUrl = process.env.GPU_MESH_COORDINATOR ?? "http://127.0.0.1:8787";
const runtimeMetadata = readNativeRuntimeBuildMetadata(
  resolve(import.meta.dirname, "../.."),
);
const admissionCredentialPath = resolve(
  process.env.MYCELLIOS_WORKER_CREDENTIAL_PATH
    ?? "./runtime/worker-admission-credential.json",
);
const admissionSigner = workerAdmissionSigner(
  loadOrCreateWorkerAdmissionCredential(admissionCredentialPath),
);

const agent = new WorkerAgent(loadWorkerConfig(resolve(configPath)), {
  coordinatorUrl,
  admissionSigner,
  ...(process.env.MYCELLIOS_NETWORK_TOKEN?.trim()
    ? { networkToken: process.env.MYCELLIOS_NETWORK_TOKEN.trim() }
    : {}),
  agentVersion: runtimeMetadata.version,
  ...(runtimeMetadata.buildIdentity
    ? { buildIdentity: runtimeMetadata.buildIdentity }
    : {}),
});

for (const signal of ["SIGINT", "SIGTERM"] as const) {
  process.once(signal, () => void agent.stop());
}

await agent.start();
