import { describe, expect, it } from "vitest";
import {
  LaunchCancelledError,
  LaunchProcessExitedError,
  LaunchReadinessTimeoutError,
  LocalProcessAgent,
  PythonLaunchSupervisor,
  type LaunchAgent,
  type LaunchAgentStartRequest,
  type LaunchCapturedOutput,
  type LaunchProcessExit,
  type LaunchProcessHandle,
} from "../src/distribution/launch-supervisor.js";
import {
  compilePythonLaunchDescription,
  type PythonLaunchCompilerOptions,
  type PythonPipelineLaunchDescription,
} from "../src/distribution/python-launcher.js";
import {
  buildRuntimePipelineManifest,
  type RuntimePlanRequest,
} from "../src/distribution/runtime-manifest.js";
import type {
  DistributedModelProfile,
  DistributionPlan,
  DistributionWorkload,
} from "../src/distribution/types.js";

const MIB = 1024 * 1024;

function fixtureDescription(): PythonPipelineLaunchDescription {
  const model: DistributedModelProfile = {
    id: "supervisor-model",
    layers: Array.from({ length: 6 }, (_, index) => ({
      index,
      weightBytes: 20 * MIB,
      activationElements: 256,
      kvBytesPerToken: 64,
      decodeMsAtUnit: 1,
      prefillMsPerTokenAtUnit: 0.1,
    })),
    embeddingBytes: MIB,
    lmHeadBytes: MIB,
    runtimeOverheadBytesPerStage: 2 * MIB,
    embeddingDecodeMsAtUnit: 0.1,
    lmHeadDecodeMsAtUnit: 0.1,
    embeddingPrefillMsPerTokenAtUnit: 0.02,
    lmHeadPrefillMsPerTokenAtUnit: 0.02,
  };
  const workload: DistributionWorkload = {
    promptTokens: 16,
    outputTokens: 8,
    contextTokens: 32,
    concurrentSequences: 2,
    maxStages: 3,
    maxQualityLoss: 0,
    minRouteAvailability: 0.8,
    batchWindowMs: 1,
    p95: false,
  };
  const plan: DistributionPlan = {
    algorithm: "supervisor-fixture",
    codec: "fp16",
    microBatchSize: 2,
    prefillChunkTokens: 8,
    stages: Array.from({ length: 3 }, (_, index) => ({
      nodeId: `node-${index}`,
      layerStart: index * 2,
      layerEnd: (index + 1) * 2,
    })),
  };
  const nodes = Array.from({ length: 3 }, (_, index) => ({
    id: `node-${index}`,
    region: `region-${index}`,
    memoryBytes: 64 * MIB,
    reserveBytes: 4 * MIB,
    decodeScale: 1,
    prefillScale: 1,
    codecScale: 1,
    batchGain: 0.2,
    maxBatchSpeedup: 1.5,
    powerWatts: 60,
    availability: 0.999,
    endpoint: { host: `node-${index}.internal`, port: 23_000 + index },
  }));
  const links = [];
  for (const from of nodes) {
    for (const to of nodes) {
      if (from.id === to.id) continue;
      links.push({
        from: from.id,
        to: to.id,
        oneWayLatencyMs: 2,
        jitterP95Ms: 0,
        bandwidthMbps: 1_000,
        lossRate: 0,
        availability: 0.999,
      });
    }
  }
  const request: RuntimePlanRequest = {
    model,
    modelRevision: "sha256:supervisor-r1",
    topology: { nodes, links },
    workload,
    phasePlans: { prefill: plan, decode: plan },
  };
  const options: PythonLaunchCompilerOptions = {
    apiEndpoint: { host: "0.0.0.0", port: 8_081 },
    returnEndpoint: { host: "root.internal", port: 30_000 },
    returnBindHost: "0.0.0.0",
  };
  return compilePythonLaunchDescription(buildRuntimePipelineManifest(request), options);
}

type FakeBehavior = "ready" | "hang" | "start-fail" | "exit-before-ready";

interface FakeTrace {
  starts: string[];
  stops: string[];
  handles: Map<string, FakeHandle>;
  behaviors: Map<string, FakeBehavior>;
}

class FakeHandle implements LaunchProcessHandle {
  readonly ready: Promise<void>;
  readonly exited: Promise<LaunchProcessExit>;
  private readonly readyDeferred = deferred<void>();
  private readonly exitDeferred = deferred<LaunchProcessExit>();
  private stopped = false;

  constructor(
    readonly processId: string,
    private readonly trace: FakeTrace,
  ) {
    this.ready = this.readyDeferred.promise;
    this.exited = this.exitDeferred.promise;
  }

  markReady(): void {
    this.readyDeferred.resolve();
  }

  exit(exit: LaunchProcessExit): void {
    this.exitDeferred.resolve(exit);
  }

  async stop(_reason: string): Promise<void> {
    if (this.stopped) return;
    this.stopped = true;
    this.trace.stops.push(this.processId);
    this.exit({ code: 0, signal: "SIGTERM" });
  }

  output(): LaunchCapturedOutput {
    return {
      stdout: `stdout:${this.processId}`,
      stderr: `stderr:${this.processId}`,
      stdoutTruncated: false,
      stderrTruncated: false,
    };
  }
}

class FakeAgent implements LaunchAgent {
  constructor(
    readonly id: string,
    private readonly trace: FakeTrace,
  ) {}

  async start(request: LaunchAgentStartRequest, signal: AbortSignal): Promise<LaunchProcessHandle> {
    if (signal.aborted) throw new LaunchCancelledError();
    this.trace.starts.push(request.process.processId);
    const behavior = this.trace.behaviors.get(request.process.processId) ?? "ready";
    if (behavior === "start-fail") throw new Error(`fake_start_failed:${request.process.processId}`);
    const handle = new FakeHandle(request.process.processId, this.trace);
    this.trace.handles.set(request.process.processId, handle);
    if (behavior === "ready") queueMicrotask(() => handle.markReady());
    if (behavior === "exit-before-ready") {
      queueMicrotask(() => handle.exit({ code: 7, signal: null }));
    }
    return handle;
  }
}

function harness(
  description: PythonPipelineLaunchDescription,
  behaviors: Array<[number, FakeBehavior]> = [],
): {
  trace: FakeTrace;
  agents: Map<string, FakeAgent>;
} {
  const trace: FakeTrace = {
    starts: [],
    stops: [],
    handles: new Map(),
    behaviors: new Map(
      behaviors.map(([launchIndex, behavior]) => [
        description.launchOrder[launchIndex]!.processId,
        behavior,
      ]),
    ),
  };
  const agents = new Map(
    description.launchOrder.map((process) => [
      process.anchor.memberId,
      new FakeAgent(`agent:${process.anchor.memberId}`, trace),
    ]),
  );
  return { trace, agents };
}

function supervisor(
  description: PythonPipelineLaunchDescription,
  setup: ReturnType<typeof harness>,
  readinessTimeoutMs = 100,
  maxTelemetryEvents = 1_000,
): PythonLaunchSupervisor {
  return new PythonLaunchSupervisor(description, {
    resolveAgent: (nodeId) => setup.agents.get(nodeId),
    readinessTimeoutMs,
    maxTelemetryEvents,
  });
}

describe("Python launch supervisor", () => {
  it("starts sequentially through injected node agents and stops in reverse order", async () => {
    const description = fixtureDescription();
    const setup = harness(description);
    const runtime = supervisor(description, setup);
    const observed: string[] = [];
    runtime.subscribe((event) => observed.push(event.type));

    const running = await runtime.start();
    expect(running.state).toBe("running");
    expect(setup.trace.starts).toEqual(
      description.launchOrder.map((process) => process.processId),
    );
    expect(setup.trace.starts.at(-1)).toBe(description.route.rootProcessId);
    expect(running.processes.every((process) => process.state === "ready")).toBe(true);
    expect(running.processes.map((process) => process.agentId)).toEqual(
      description.launchOrder.map((process) => `agent:${process.anchor.memberId}`),
    );
    expect(observed.filter((type) => type === "process_ready")).toHaveLength(3);

    const stopped = await runtime.stop();
    expect(stopped.state).toBe("stopped");
    expect(setup.trace.stops).toEqual(
      [...description.launchOrder].reverse().map((process) => process.processId),
    );
    const stoppedAgain = await runtime.stop();
    expect(stoppedAgain.state).toBe("stopped");
    expect(setup.trace.stops).toHaveLength(3);
  });

  it("times out readiness and rolls back the current and prior handles", async () => {
    const description = fixtureDescription();
    const setup = harness(description, [[1, "hang"]]);
    const runtime = supervisor(description, setup, 15);

    await expect(runtime.start()).rejects.toBeInstanceOf(LaunchReadinessTimeoutError);
    const snapshot = runtime.snapshot();
    expect(snapshot.state).toBe("failed");
    expect(snapshot.failure).toContain("launch_readiness_timeout");
    expect(setup.trace.starts).toEqual(
      description.launchOrder.slice(0, 2).map((process) => process.processId),
    );
    expect(setup.trace.stops).toEqual(
      description.launchOrder.slice(0, 2).reverse().map((process) => process.processId),
    );
    expect(snapshot.processes[1]!.state).toBe("failed");
    expect(snapshot.processes[2]!.state).toBe("pending");
  });

  it("rolls back ready processes when a later agent fails to start", async () => {
    const description = fixtureDescription();
    const setup = harness(description, [[2, "start-fail"]]);
    const runtime = supervisor(description, setup);

    await expect(runtime.start()).rejects.toThrow("fake_start_failed");
    expect(runtime.snapshot().state).toBe("failed");
    expect(setup.trace.stops).toEqual(
      description.launchOrder.slice(0, 2).reverse().map((process) => process.processId),
    );
    expect(runtime.snapshot().processes[2]!.state).toBe("failed");
  });

  it("treats exit-before-ready as failure and stops its allocated handle", async () => {
    const description = fixtureDescription();
    const setup = harness(description, [[0, "exit-before-ready"]]);
    const runtime = supervisor(description, setup);

    await expect(runtime.start()).rejects.toBeInstanceOf(LaunchProcessExitedError);
    expect(runtime.snapshot().state).toBe("failed");
    expect(setup.trace.stops).toEqual([description.launchOrder[0]!.processId]);
    expect(runtime.snapshot().processes[0]!.state).toBe("failed");
  });

  it("cancels an in-progress launch and rolls back without labeling it a runtime failure", async () => {
    const description = fixtureDescription();
    const setup = harness(description, [[1, "hang"]]);
    const runtime = supervisor(description, setup, 1_000);
    const controller = new AbortController();
    const start = runtime.start(controller.signal);
    await waitFor(() => setup.trace.starts.length === 2);
    controller.abort();

    await expect(start).rejects.toBeInstanceOf(LaunchCancelledError);
    const snapshot = runtime.snapshot();
    expect(snapshot.state).toBe("stopped");
    expect(snapshot.failure).toBeNull();
    expect(setup.trace.stops).toEqual(
      description.launchOrder.slice(0, 2).reverse().map((process) => process.processId),
    );
    expect(snapshot.telemetry.some((event) => event.type === "supervisor_cancelled")).toBe(true);
  });

  it("detects a process dying after ready and automatically tears down the route", async () => {
    const description = fixtureDescription();
    const setup = harness(description);
    const runtime = supervisor(description, setup);
    await runtime.start();
    const terminal = runtime.waitForTerminal();
    const failedProcess = description.launchOrder[1]!;
    setup.trace.handles.get(failedProcess.processId)!.exit({ code: 42, signal: null });

    const snapshot = await terminal;
    expect(snapshot.state).toBe("failed");
    expect(snapshot.failure).toContain(`launch_process_exited:${failedProcess.processId}`);
    expect(snapshot.processes[1]!.state).toBe("failed");
    expect(setup.trace.stops).toEqual(
      [...description.launchOrder].reverse().map((process) => process.processId),
    );
    expect(snapshot.telemetry.some((event) => event.type === "process_exit")).toBe(true);
    expect(snapshot.telemetry.some((event) => event.type === "supervisor_failed")).toBe(true);
  });

  it("bounds telemetry, exposes captured output and isolates subscriber errors", async () => {
    const description = fixtureDescription();
    const setup = harness(description);
    const runtime = supervisor(description, setup, 100, 5);
    runtime.subscribe(() => {
      throw new Error("observer failure");
    });
    await runtime.start();
    const stopped = await runtime.stop();

    expect(stopped.telemetry).toHaveLength(5);
    expect(stopped.telemetryDropped).toBeGreaterThan(0);
    expect(stopped.processes[0]!.output?.stdout).toContain("stdout:");
    expect(stopped.telemetry.map((event) => event.sequence)).toEqual(
      [...stopped.telemetry.map((event) => event.sequence)].sort((a, b) => a - b),
    );
  });

  it("allows constructing the local opt-in agent without starting a process", () => {
    const local = new LocalProcessAgent({
      id: "explicit-local",
      maxOutputBytesPerStream: 2_048,
      stopGraceMs: 10,
    });
    expect(local.id).toBe("explicit-local");
  });

  it("retains a final readiness marker when bounded process output is truncated", async () => {
    const marker = "FINAL_EXECUTION_MARKER";
    const local = new LocalProcessAgent({
      id: "tail-capture-local",
      maxOutputBytesPerStream: 1_024,
      stopGraceMs: 1_000,
      readyWhen: ({ recentStdout }) => recentStdout.includes(marker),
    });
    const request = {
      launchId: "tail-capture-launch",
      pipelineId: "tail-capture-pipeline",
      nodeId: "local-node",
      process: {
        processId: "tail-capture-process",
        kind: "root-engine",
        command: {
          executable: process.execPath,
          args: [
            "-e",
            `process.stdout.write("x".repeat(4096) + "${marker}\\n"); setInterval(() => {}, 1000);`,
          ],
        },
      },
    } as unknown as LaunchAgentStartRequest;

    const handle = await local.start(request, new AbortController().signal);
    await handle.ready;
    expect(handle.output?.()).toMatchObject({
      stdoutTruncated: true,
    });
    expect(handle.output?.().stdout).toContain(marker);
    await handle.stop("test_complete");
    await handle.exited;
  });

  it("stops from idle idempotently without resolving any agent", async () => {
    const description = fixtureDescription();
    const setup = harness(description);
    const runtime = supervisor(description, setup);
    expect((await runtime.stop()).state).toBe("stopped");
    expect((await runtime.stop()).state).toBe("stopped");
    expect(setup.trace.starts).toEqual([]);
    expect(setup.trace.stops).toEqual([]);
  });
});

function deferred<T>(): {
  promise: Promise<T>;
  resolve(value: T): void;
} {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((res) => {
    resolve = res;
  });
  return { promise, resolve };
}

async function waitFor(predicate: () => boolean, timeoutMs = 1_000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (!predicate()) {
    if (Date.now() >= deadline) throw new Error("test_wait_timeout");
    await new Promise<void>((resolve) => setTimeout(resolve, 1));
  }
}
