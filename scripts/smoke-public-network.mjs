import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { workerConfigSchema } from "../dist/contracts/schemas.js";
import { WorkerAgent } from "../dist/worker/agent.js";

const coordinatorUrl = process.argv[2] ?? "http://127.0.0.1:8787";
const config = workerConfigSchema.parse(
  JSON.parse(await readFile(resolve("config/worker.example.json"), "utf8")),
);
const before = await snapshot();
const agent = new WorkerAgent(config, {
  coordinatorUrl,
  reconnect: false,
  logger: { info() {}, warn() {}, error() {} },
});
const run = agent.start();

try {
  const connected = await waitFor(
    (current) => current.summary.connected > before.summary.connected,
    "the desktop worker to connect",
  );
  await agent.stop();
  await run;
  const disconnected = await waitFor(
    (current) => current.summary.registered === before.summary.registered,
    "the desktop worker to disappear after stopping",
  );
  console.log(JSON.stringify({
    coordinatorUrl,
    before: before.summary,
    connected: connected.summary,
    disconnected: disconnected.summary,
  }, null, 2));
} finally {
  await agent.stop();
  await run;
}

async function snapshot() {
  const response = await fetch(new URL("/public/v1/snapshot", coordinatorUrl));
  if (!response.ok) throw new Error(`Snapshot failed with HTTP ${response.status}`);
  return response.json();
}

async function waitFor(predicate, description, timeoutMs = 10_000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const current = await snapshot();
    if (predicate(current)) return current;
    await new Promise((resolveDelay) => setTimeout(resolveDelay, 150));
  }
  throw new Error(`Timed out waiting for ${description}`);
}
