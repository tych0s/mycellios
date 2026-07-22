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
});
