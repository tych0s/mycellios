import { readFileSync } from "node:fs";

const dockerfile = readFileSync(new URL("../Dockerfile.salad", import.meta.url), "utf8");
const packageJson = JSON.parse(
  readFileSync(new URL("../package.json", import.meta.url), "utf8"),
);
const workerConfig = JSON.parse(
  readFileSync(new URL("../config/worker.salad.example.json", import.meta.url), "utf8"),
);
const headlessMain = readFileSync(
  new URL("../src/worker/headless-main.ts", import.meta.url),
  "utf8",
);

const requiredDockerFragments = [
  "pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime",
  "torch.__version__.startswith('2.13.0')",
  "container/salad/package-lock.json",
  "USER 10001:10001",
  'CMD ["node", "dist/worker/headless-main.js"]',
  "GPU_MESH_WORKER_CONFIG=/opt/mycellios/config/worker.salad.example.json",
];
for (const fragment of requiredDockerFragments) {
  if (!dockerfile.includes(fragment)) {
    throw new Error(`salad_container_missing_required_fragment:${fragment}`);
  }
}
for (const forbidden of [
  "MYCELLIOS_NETWORK_TOKEN=",
  "HF_TOKEN=",
  "SALAD_MACHINE_ID=",
]) {
  if (dockerfile.includes(forbidden)) {
    throw new Error(`salad_container_must_not_bake_secret_or_identity:${forbidden}`);
  }
}
if (packageJson.scripts?.["start:worker:headless"] !== "node dist/worker/headless-main.js") {
  throw new Error("salad_container_headless_script_is_missing");
}
if (
  workerConfig.capacityScope !== "host"
  || workerConfig.adapter?.kind !== "mock"
  || workerConfig.offeredVramMb > 512
  || workerConfig.limits?.pauseWhenForeground !== false
) {
  throw new Error("salad_worker_config_is_not_shard_only");
}
for (const fragment of [
  "probedDevice.totalMemoryBytes",
  "probedDevice.runtimeTotalMemoryBytes",
  "offeredVramMb: detectedVramMb",
  "hardwareCapacityOverride",
  'identity: { kind: "device", id: environment.nodeId }',
]) {
  if (!headlessMain.includes(fragment)) {
    throw new Error(`salad_worker_missing_dynamic_physical_capacity:${fragment}`);
  }
}

process.stdout.write("Salad container contract verified.\n");
