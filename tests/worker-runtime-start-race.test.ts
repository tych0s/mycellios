import { describe, expect, it } from "vitest";
import { workerConfigSchema } from "../src/contracts/schemas.js";
import type { WorkerEnvelope } from "../src/contracts/types.js";
import { parseWorkerEnvelope } from "../src/contracts/worker-protocol.js";
import type { LaunchAgentStartRequest, LaunchProcessHandle } from "../src/distribution/launch-supervisor.js";
import { normalizeExecutorIsolationPolicy } from "../src/distribution/process-environment.js";
import { WorkerAgent } from "../src/worker/agent.js";

function harness() {
  let release!: (handle: LaunchProcessHandle) => void;
  let launchSignal: AbortSignal | undefined;
  let starts = 0;
  let stops = 0;
  const sent: WorkerEnvelope[] = [];
  const agent = new WorkerAgent(workerConfigSchema.parse({
    region: "test", offeredVramMb: 4_096,
    limits: { maxConcurrency: 1, pauseWhenForeground: false },
    adapter: { kind: "mock", developmentOnly: true, model: "test-model", tokensPerSecond: 1_000, ttftMs: 0, failureRate: 0 },
    deployment: { modelDigest: "sha256:test", contextLimit: 8_192 },
  }), {
    coordinatorUrl: "http://127.0.0.1:9999", reconnect: false, advertiseDeployment: false,
    distributedExecutor: { nodeId: "node", stageHost: "node.relay", stagePort: 9_850, computeMode: "cpu-only", cpuEligible: true,
      launchAgent: { id: "delayed-launch", start: async (_request, signal) => {
        starts += 1;
        launchSignal = signal;
        return new Promise<LaunchProcessHandle>((resolve) => { release = resolve; });
      } },
    },
    logger: { info() {}, warn() {}, error() {} },
  });
  const worker = agent as unknown as {
    registeredWorkerId: string;
    socket: { readyState: number; send(serialized: string): void };
    preparedRuntimeFormation: Pick<LaunchAgentStartRequest, "launchId" | "pipelineId" | "deploymentGeneration">;
    authorizedRuntimeProcesses: Map<string, string>;
    preparedRuntimeProcesses: Map<string, LaunchAgentStartRequest["process"]>;
    runtimeProcesses: Map<string, LaunchProcessHandle>;
    runtimeStartRequests: Map<string, string>;
    startDistributedRuntime(requestId: string, input: unknown): Promise<void>;
    prepareDistributedRuntime(requestId: string, input: unknown, generation: number): Promise<void>;
    sendRuntimeExit(requestId: string, handle: undefined, exit: { code: null; signal: null; error: string }): void;
    stopDistributedRuntime(requestId: string, reason: string): Promise<void>;
    resetDistributedRuntime(reason: string): Promise<void>;
    runtimeTunnel: { close(): Promise<void> };
  };
  const request = {
    launchId: "launch", pipelineId: "pipeline", deploymentGeneration: 1, nodeId: "node",
    process: { kind: "root-engine", processId: "root", anchor: { memberId: "node", endpoint: { host: "node.relay", port: 9_850 } },
      command: { executable: "python", args: ["-m", "runtime"] }, isolation: normalizeExecutorIsolationPolicy() },
  } as LaunchAgentStartRequest;
  worker.registeredWorkerId = "worker";
  worker.socket = { readyState: 1, send: (serialized) => sent.push(JSON.parse(serialized) as WorkerEnvelope) };
  worker.preparedRuntimeFormation = request;
  worker.authorizedRuntimeProcesses.set("root", JSON.stringify(request.process));
  worker.preparedRuntimeProcesses.set("root", request.process);
  let resolveExit!: (exit: { code: number; signal: null }) => void;
  const handle: LaunchProcessHandle = {
    ready: Promise.resolve(),
    exited: new Promise((resolve) => { resolveExit = resolve; }),
    stop: async () => { stops += 1; resolveExit({ code: 0, signal: null }); },
  };
  return { worker, request, sent, release: () => release(handle), starts: () => starts,
    stops: () => stops, signal: () => launchSignal };
}

describe("worker runtime launch races", () => {
  it("keeps long preparation and exit errors within the wire contract without dropping the root exception", async () => {
    const test = harness();
    const detail = `stage_artifact_preparation_failed:Traceback\n${"line of Python traceback\n".repeat(300)}FileNotFoundError: missing shard`;
    const input = new Proxy({}, { get() { throw new Error(detail); } });
    try {
      await test.worker.prepareDistributedRuntime("prepare-error", input, 0);
      test.worker.sendRuntimeExit("start-error", undefined, { code: null, signal: null, error: detail });
      expect(test.sent.map(({ type }) => type)).toEqual(["runtime.prepared", "runtime.exited"]);
      for (const envelope of test.sent) {
        expect(parseWorkerEnvelope(envelope)).not.toBeNull();
        const payload = envelope.payload as { error?: string; exit?: { error?: string } };
        const message = payload.error ?? payload.exit?.error;
        expect(message).toHaveLength(2_048);
        expect(message).toContain("stage_artifact_preparation_failed:Traceback");
        expect(message).toContain("FileNotFoundError: missing shard");
      }
    } finally {
      await test.worker.runtimeTunnel.close();
    }
  });

  it.each(["stop", "reset"] as const)("disposes a late process after %s during asynchronous launch", async (operation) => {
    const test = harness();
    const starting = test.worker.startDistributedRuntime("request", test.request);
    if (operation === "stop") await test.worker.stopDistributedRuntime("request", "cancelled");
    else await test.worker.resetDistributedRuntime("cancelled");
    const aborted = test.signal()?.aborted;
    test.release();
    await starting;
    const processes = test.worker.runtimeProcesses.size;
    const stops = test.stops();
    const ready = test.sent.filter(({ type }) => type === "runtime.ready");
    await test.worker.resetDistributedRuntime("teardown");
    await test.worker.runtimeTunnel.close();
    expect(aborted).toBe(true);
    expect(stops).toBe(1);
    expect(processes).toBe(0);
    expect(ready).toHaveLength(0);
  });

  it("coalesces identical starts while the initial launch is unresolved", async () => {
    const test = harness();
    const starting = test.worker.startDistributedRuntime("request", test.request);
    const duplicate = test.worker.startDistributedRuntime("request", structuredClone(test.request));
    const starts = test.starts();
    test.release();
    // A broken duplicate overwrites the test's pending resolver, so only await the surviving operation.
    if (starts === 1) await starting;
    await duplicate;
    await test.worker.resetDistributedRuntime("teardown");
    await test.worker.runtimeTunnel.close();
    expect(starts).toBe(1);
  });
});
