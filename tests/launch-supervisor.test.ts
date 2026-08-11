import { describe, expect, it } from "vitest";
import { existsSync } from "node:fs";
import { resolve } from "node:path";
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
import { normalizeExecutorIsolationPolicy } from "../src/distribution/process-environment.js";

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
      allowedExecutables: [process.execPath],
      maxOutputBytesPerStream: 2_048,
      stopGraceMs: 10,
    });
    expect(local.id).toBe("explicit-local");
  });

  it("routes an authorized Windows target through the packaged Job Object broker", async () => {
    const broker = resolve(
      "build",
      "windows-job-broker",
      "mycellios-job-broker.exe",
    );
    if (process.platform !== "win32" || !existsSync(broker)) return;
    const description = fixtureDescription();
    const launchProcess = structuredClone(
      description.launchOrder.find((candidate) => candidate.kind === "remote-stage")!,
    );
    launchProcess.command = {
      executable: process.execPath,
      args: [
        "-e",
        "console.error('stage_ready');setInterval(() => undefined, 1000);",
      ],
    };
    launchProcess.isolation = normalizeExecutorIsolationPolicy({
      stopGraceMs: 1_000,
    });
    const local = new LocalProcessAgent({
      id: "windows-job-local",
      cwd: resolve("."),
      allowedExecutables: [process.execPath],
      windowsJobBrokerExecutable: broker,
      stopGraceMs: 1_000,
    });

    const handle = await local.start(
      {
        launchId: description.launchId,
        pipelineId: description.pipelineId,
        deploymentGeneration: description.deploymentGeneration,
        nodeId: launchProcess.anchor.memberId,
        process: launchProcess,
      },
      new AbortController().signal,
    );
    await handle.ready;
    expect(handle.output?.().stderr).toContain("stage_ready");
    await handle.stop("test_complete");
    await expect(handle.exited).resolves.toMatchObject({
      code: expect.anything(),
    });
  });

  it("rejects a sealed process policy above the local agent ceilings", async () => {
    const description = fixtureDescription();
    const process = structuredClone(description.launchOrder[0]!);
    process.isolation = normalizeExecutorIsolationPolicy({
      maxOutputBytesPerStream: 128 * 1024,
    });
    const local = new LocalProcessAgent({
      id: "bounded-local",
      allowedExecutables: [process.command.executable],
      maxOutputBytesPerStream: 64 * 1024,
    });

    await expect(
      local.start(
        {
          launchId: description.launchId,
          pipelineId: description.pipelineId,
          deploymentGeneration: description.deploymentGeneration,
          nodeId: process.anchor.memberId,
          process,
        },
        new AbortController().signal,
      ),
    ).rejects.toThrow("local_process_output_limit_exceeds_agent_ceiling");
  });

  it("pins the exact prepared interpreter before spawning a local process", async () => {
    const local = new LocalProcessAgent({
      id: "pinned-local",
      allowedExecutables: [process.execPath],
    });
    const request = {
      launchId: "pinned-launch",
      pipelineId: "pinned-pipeline",
      nodeId: "local-node",
      process: {
        processId: "pinned-process",
        kind: "root-engine",
        isolation: normalizeExecutorIsolationPolicy(),
        command: {
          executable: `${process.execPath}.untrusted`,
          args: ["-e", "process.exit(0)"],
        },
      },
    } as unknown as LaunchAgentStartRequest;

    await expect(
      local.start(request, new AbortController().signal),
    ).rejects.toThrow("local_process_executable_is_not_authorized");
  });

  it("captures and restores bounded KV through the private executor workspace socket", async () => {
    const marker = "CHECKPOINT_CONTROL_READY";
    const local = new LocalProcessAgent({
      id: "checkpoint-control-local",
      allowedExecutables: [process.execPath],
      stopGraceMs: 1_000,
      readyWhen: ({ recentStdout }) => recentStdout.includes(marker),
    });
    const childScript = String.raw`
      const net = require("node:net");
      const path = require("node:path").join(process.env.MYCELLIOS_EXECUTOR_WORKSPACE, "activation-checkpoint.sock");
      const server = net.createServer((socket) => {
        let bytes = Buffer.alloc(0);
        socket.on("data", (chunk) => {
          bytes = Buffer.concat([bytes, chunk]);
          const newline = bytes.indexOf(10);
          if (newline < 0) return;
          const header = JSON.parse(bytes.subarray(0, newline).toString("utf8"));
          if (header.operation === "capture") {
            const payload = Buffer.from("live-kv");
            socket.end(JSON.stringify({ok:true,payloadBytes:payload.length,committedPosition:37}) + "\n" + payload);
          } else if (bytes.length >= newline + 1 + header.payloadBytes) {
            const payload = bytes.subarray(newline + 1);
            if (payload.toString() !== "restored-kv" || header.committedPosition !== 37) process.exit(91);
            socket.end(JSON.stringify({ok:true,payloadBytes:0}) + "\n");
          }
        });
      });
      server.listen(path, () => console.log("${marker}"));
      process.on("SIGTERM", () => server.close(() => process.exit(0)));
    `;
    const request = {
      launchId: "checkpoint-control-launch",
      pipelineId: "checkpoint-control-pipeline",
      deploymentGeneration: 1,
      nodeId: "local-node",
      process: {
        processId: "checkpoint-control-process",
        stageId: "stage-a",
        kind: "remote-stage",
        isolation: normalizeExecutorIsolationPolicy({ stopGraceMs: 1_000 }),
        command: { executable: process.execPath, args: ["-e", childScript] },
      },
    } as unknown as LaunchAgentStartRequest;
    const handle = await local.start(request, new AbortController().signal);
    await handle.ready;
    await expect(handle.captureActivationCheckpoint?.(17, 64)).resolves.toEqual({
      payload: Buffer.from("live-kv"),
      committedPosition: 37,
    });
    await expect(handle.restoreActivationCheckpoint?.(
      17, Buffer.from("restored-kv"), 37, 64,
    )).resolves.toBeUndefined();
    await handle.stop("test_complete");
  });

  it("retains a final readiness marker when bounded process output is truncated", async () => {
    const marker = "FINAL_EXECUTION_MARKER";
    const local = new LocalProcessAgent({
      id: "tail-capture-local",
      allowedExecutables: [process.execPath],
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
        isolation: normalizeExecutorIsolationPolicy({
          maxOutputBytesPerStream: 1_024,
          stopGraceMs: 1_000,
        }),
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

  it("terminates the ordinary descendant process tree", async () => {
    const marker = "PROCESS_TREE_READY";
    const local = new LocalProcessAgent({
      id: "process-tree-local",
      allowedExecutables: [process.execPath],
      stopGraceMs: 2_000,
      readyWhen: ({ recentStdout }) => recentStdout.includes(marker),
    });
    const request = {
      launchId: "process-tree-launch",
      pipelineId: "process-tree-pipeline",
      nodeId: "local-node",
      process: {
        processId: "process-tree-root",
        kind: "root-engine",
        isolation: normalizeExecutorIsolationPolicy({
          stopGraceMs: 1_000,
        }),
        command: {
          executable: process.execPath,
          args: [
            "-e",
            `const { spawn } = require("node:child_process"); const descendant = spawn(process.execPath, ["-e", "setInterval(() => {}, 1000)"], { stdio: "ignore" }); process.stdout.write("${marker}:" + JSON.stringify({ root: process.pid, descendant: descendant.pid }) + "\\n"); setInterval(() => {}, 1000);`,
          ],
        },
      },
    } as unknown as LaunchAgentStartRequest;

    const handle = await local.start(request, new AbortController().signal);
    await handle.ready;
    const output = handle.output?.().stdout ?? "";
    const encoded = output.split(`${marker}:`)[1]?.trim();
    const pids = JSON.parse(encoded ?? "{}") as {
      root: number;
      descendant: number;
    };
    expect(isProcessAlive(pids.root)).toBe(true);
    expect(isProcessAlive(pids.descendant)).toBe(true);

    await handle.stop("tree_test_complete");
    await handle.exited;
    expect(await processBecomesDead(pids.root, 3_000)).toBe(true);
    expect(await processBecomesDead(pids.descendant, 3_000)).toBe(true);
  }, 15_000);

  it("terminates a process that exceeds its sealed private workspace quota", async () => {
    const marker = "WORKSPACE_QUOTA_READY";
    const local = new LocalProcessAgent({
      id: "workspace-quota-local",
      allowedExecutables: [process.execPath],
      stopGraceMs: 2_000,
      readyWhen: ({ recentStdout }) => recentStdout.includes(marker),
    });
    const request = {
      launchId: "workspace-quota-launch",
      pipelineId: "workspace-quota-pipeline",
      nodeId: "local-node",
      process: {
        processId: "workspace-quota-process",
        kind: "root-engine",
        isolation: normalizeExecutorIsolationPolicy({
          stopGraceMs: 1_000,
          maxWorkspaceBytes: 64 * 1024,
          maxWorkspaceEntries: 32,
          workspaceCheckIntervalMs: 25,
        }),
        command: {
          executable: process.execPath,
          args: [
            "-e",
            `const { writeFileSync } = require("node:fs"); const { join } = require("node:path"); const workspace = process.env.MYCELLIOS_EXECUTOR_WORKSPACE; process.stdout.write("${marker}:" + workspace + "\\n"); setTimeout(() => writeFileSync(join(workspace, "quota.bin"), Buffer.alloc(128 * 1024)), 10); setInterval(() => {}, 1000);`,
          ],
        },
      },
    } as unknown as LaunchAgentStartRequest;

    const handle = await local.start(request, new AbortController().signal);
    await handle.ready;
    const workspace = (handle.output?.().stdout ?? "")
      .split(`${marker}:`)[1]
      ?.trim();
    expect(workspace).toBeTruthy();
    const exit = await handle.exited;
    expect(exit.error).toContain("executor_workspace_byte_limit_exceeded");
    expect(existsSync(workspace!)).toBe(false);
  }, 15_000);

  it("does not leak an ungranted parent secret into the executor", async () => {
    const inheritedSecretName = "MYCELLIOS_TEST_PARENT_SECRET_DO_NOT_INHERIT";
    const previous = process.env[inheritedSecretName];
    process.env[inheritedSecretName] = "parent-secret";
    try {
      const marker = "ISOLATED_ENVIRONMENT_RESULT";
      const local = new LocalProcessAgent({
        id: "isolated-environment-local",
        allowedExecutables: [process.execPath],
        env: {
          MYCELLIOS_TEST_EXPLICIT_VALUE: "explicit-value",
        },
        readyWhen: ({ recentStdout }) => recentStdout.includes(marker),
      });
      const request = {
        launchId: "isolated-environment-launch",
        pipelineId: "isolated-environment-pipeline",
        nodeId: "local-node",
        process: {
          processId: "isolated-environment-process",
          kind: "root-engine",
          isolation: normalizeExecutorIsolationPolicy(),
          command: {
            executable: process.execPath,
            args: [
            "-e",
              `process.stdout.write("${marker}:" + JSON.stringify({ inherited: process.env.${inheritedSecretName} ?? null, explicit: process.env.MYCELLIOS_TEST_EXPLICIT_VALUE ?? null, workspace: process.env.MYCELLIOS_EXECUTOR_WORKSPACE ?? null, tempMatches: process.env.TMP === process.env.MYCELLIOS_EXECUTOR_WORKSPACE && process.env.TEMP === process.env.MYCELLIOS_EXECUTOR_WORKSPACE && process.env.TMPDIR === process.env.MYCELLIOS_EXECUTOR_WORKSPACE }) + "\\n");`,
            ],
          },
        },
      } as unknown as LaunchAgentStartRequest;

      const handle = await local.start(request, new AbortController().signal);
      await handle.ready;
      await handle.exited;
      const output = handle.output?.().stdout ?? "";
      const encoded = output.split(`${marker}:`)[1]?.trim();
      const observed = JSON.parse(encoded ?? "{}") as {
        inherited: string | null;
        explicit: string | null;
        workspace: string | null;
        tempMatches: boolean;
      };
      expect(observed).toMatchObject({
        inherited: null,
        explicit: "explicit-value",
        tempMatches: true,
      });
      expect(observed.workspace).toBeTruthy();
      expect(existsSync(observed.workspace!)).toBe(false);
    } finally {
      if (previous === undefined) delete process.env[inheritedSecretName];
      else process.env[inheritedSecretName] = previous;
    }
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

function isProcessAlive(pid: number): boolean {
  if (!Number.isSafeInteger(pid) || pid < 1) return false;
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

async function processBecomesDead(
  pid: number,
  timeoutMs: number,
): Promise<boolean> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (!isProcessAlive(pid)) return true;
    await new Promise<void>((resolve) => setTimeout(resolve, 25));
  }
  return !isProcessAlive(pid);
}

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
