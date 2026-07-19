import { resolve } from "node:path";
import { loadWorkerConfig } from "../core/config.js";
import { WorkerAgent } from "./agent.js";

const configPath = process.env.GPU_MESH_WORKER_CONFIG ?? "./config/worker.example.json";
const coordinatorUrl = process.env.GPU_MESH_COORDINATOR ?? "http://127.0.0.1:8787";

const agent = new WorkerAgent(loadWorkerConfig(resolve(configPath)), {
  coordinatorUrl,
});

for (const signal of ["SIGINT", "SIGTERM"] as const) {
  process.once(signal, () => void agent.stop());
}

await agent.start();
