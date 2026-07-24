import { EventEmitter } from "node:events";
import { describe, expect, it } from "vitest";
import type { WorkerEnvelope } from "../src/contracts/types.js";
import type { WorkerHub } from "../src/coordinator/worker-hub.js";
import type {
  LaunchAgentStartRequest,
  LaunchCapturedOutput,
} from "../src/distribution/launch-supervisor.js";
import type { PythonPipelineLaunchDescription } from "../src/distribution/python-launcher.js";
import { WorkerTunnelLaunchAgent } from "../src/distribution/worker-tunnel-launch-agent.js";

class FakeWorkerHub extends EventEmitter {
  readonly sent: Array<{ workerId: string; type: string; payload: Record<string, unknown> }> = [];
  readonly readyOutput: LaunchCapturedOutput = {
    stdout: "root online\n",
    stderr: '{"execution":{"backend":"cuda","accelerated":true}}\n',
    stdoutTruncated: false,
    stderrTruncated: false,
  };

  send(workerId: string, type: string, value: unknown): boolean {
    const payload = value as Record<string, unknown>;
    this.sent.push({ workerId, type, payload });
    queueMicrotask(() => {
      if (type === "runtime.prepare") {
        this.emitEnvelope(workerId, "runtime.prepared", {
          requestId: payload.requestId,
          ok: true,
        });
      } else if (type === "runtime.start") {
        this.emitEnvelope(workerId, "runtime.ready", {
          requestId: payload.requestId,
          output: this.readyOutput,
        });
      }
    });
    return true;
  }

  private emitEnvelope(workerId: string, type: string, payload: Record<string, unknown>): void {
    this.emit("envelope", {
      v: 1,
      workerId,
      type,
      payload,
    } as WorkerEnvelope);
  }
}

describe("worker tunnel execution telemetry", () => {
  it("copies the runtime.ready output before resolving the launch handle", async () => {
    const hub = new FakeWorkerHub();
    const description = {
      launchId: "launch-test",
      pipelineId: "pipeline-test",
      launchOrder: [],
    } as unknown as PythonPipelineLaunchDescription;
    const agent = new WorkerTunnelLaunchAgent(
      hub as unknown as WorkerHub,
      "worker-test",
      "node-test",
      description,
      1_000,
    );
    const request = {
      launchId: "launch-test",
      pipelineId: "pipeline-test",
      nodeId: "node-test",
      process: { processId: "stage-1" },
    } as unknown as LaunchAgentStartRequest;

    const handle = await agent.start(request, new AbortController().signal);
    await handle.ready;

    expect(handle.output?.()).toEqual(hub.readyOutput);
    expect(hub.sent.map(({ type }) => type)).toEqual(["runtime.prepare", "runtime.start"]);
    await agent.close();
  });

  it("rejects a disconnected launch through the supervised promises without an orphan rejection", async () => {
    const hub = new FakeWorkerHub();
    const description = {
      launchId: "launch-disconnect",
      pipelineId: "pipeline-disconnect",
      launchOrder: [],
    } as unknown as PythonPipelineLaunchDescription;
    const agent = new WorkerTunnelLaunchAgent(
      hub as unknown as WorkerHub,
      "worker-disconnect",
      "node-disconnect",
      description,
      1_000,
    );
    const request = {
      launchId: "launch-disconnect",
      pipelineId: "pipeline-disconnect",
      nodeId: "node-disconnect",
      process: { processId: "stage-disconnect" },
    } as unknown as LaunchAgentStartRequest;

    const handle = await agent.start(request, new AbortController().signal);
    await handle.ready;
    const exited = handle.exited.catch((error: unknown) => error);
    hub.emit("disconnect", "worker-disconnect");

    await expect(exited).resolves.toMatchObject({
      message: "distributed_worker_disconnected:worker-disconnect",
    });
    await new Promise<void>((resolve) => setImmediate(resolve));
    await agent.close();
  });

  it("does not block route cleanup when a connected worker never acknowledges stop", async () => {
    const hub = new FakeWorkerHub();
    const description = {
      launchId: "launch-stop-timeout",
      pipelineId: "pipeline-stop-timeout",
      launchOrder: [],
    } as unknown as PythonPipelineLaunchDescription;
    const agent = new WorkerTunnelLaunchAgent(
      hub as unknown as WorkerHub,
      "worker-stop-timeout",
      "node-stop-timeout",
      description,
      20,
    );
    const request = {
      launchId: "launch-stop-timeout",
      pipelineId: "pipeline-stop-timeout",
      nodeId: "node-stop-timeout",
      process: { processId: "stage-stop-timeout" },
    } as unknown as LaunchAgentStartRequest;

    const handle = await agent.start(request, new AbortController().signal);
    await handle.ready;
    const exited = handle.exited.catch((error: unknown) => error);

    await expect(handle.stop("route_repair")).resolves.toBeUndefined();
    await expect(exited).resolves.toMatchObject({
      message: "worker_tunnel_stop_timeout:worker-stop-timeout",
    });
    expect(hub.sent.map(({ type }) => type)).toContain("runtime.stop");
    await agent.close();
  });
});
