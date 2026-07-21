import { workerConfigSchema } from "./contracts/schemas.js";
import { createCoordinator } from "./coordinator/server.js";
import { WorkerAgent } from "./worker/agent.js";

const runtime = await createCoordinator(
  {
    host: "127.0.0.1",
    port: 0,
    databasePath: ":memory:",
    requestTimeoutMs: 30_000,
  },
  { logger: false },
);

let agent: WorkerAgent | null = null;
let agentRun: Promise<void> | null = null;

try {
  const address = await runtime.app.listen({ host: "127.0.0.1", port: 0 });
  const config = workerConfigSchema.parse({
    region: "es-mad",
    offeredVramMb: 4_096,
    limits: {
      maxConcurrency: 2,
      maxPowerW: 80,
      maxTemperatureC: 78,
      pauseWhenForeground: true,
    },
    adapter: {
      kind: "mock",
      model: "distributed-small",
      tokensPerSecond: 200,
      ttftMs: 10,
      failureRate: 0,
    },
    deployment: {
      modelDigest: "sha256:distributed-small-demo",
      peakVramMb: 3_000,
      contextLimit: 8_192,
      tokensPerSecond: 200,
      ttftMs: 10,
    },
  });
  agent = new WorkerAgent(config, {
    coordinatorUrl: address,
    heartbeatIntervalMs: 250,
    reconnect: false,
    logger: console,
  });
  agentRun = agent.start();
  await waitUntil(() => runtime.hub.connectedWorkerIds().size === 1, 10_000);

  const response = await fetch(new URL("v1/chat/completions", `${address}/`), {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "idempotency-key": "demo-request-1",
    },
    body: JSON.stringify({
      model: "distributed-small",
      messages: [{ role: "user", content: "Does the distributed network work?" }],
      max_tokens: 80,
      stream: false,
      session_id: "demo-session",
      preferred_region: "es-mad",
    }),
  });
  const body = (await response.json()) as Record<string, unknown>;
  if (!response.ok) throw new Error(`Demo request failed: ${JSON.stringify(body)}`);

  const worker = runtime.store.listWorkers()[0]!;
  console.log(
    JSON.stringify(
      {
        ok: true,
        coordinator: address,
        worker: {
          id: worker.id,
          region: worker.capabilities.region,
          offeredVramMb: worker.capabilities.gpus[0]?.offeredVramMb,
          jobsCompleted: runtime.store.getWorker(worker.id)?.jobsCompleted,
        },
        response: body,
      },
      null,
      2,
    ),
  );
} finally {
  await agent?.stop();
  await agentRun;
  await runtime.close();
}

async function waitUntil(predicate: () => boolean, timeoutMs: number): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (!predicate()) {
    if (Date.now() >= deadline) throw new Error("Timed out waiting for demo worker connection");
    await new Promise((resolve) => setTimeout(resolve, 25));
  }
}
