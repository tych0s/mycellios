import { afterEach, describe, expect, it } from "vitest";
import { createHmac, randomBytes } from "node:crypto";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { createServer, type Socket } from "node:net";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { once } from "node:events";
import {
  CHECKPOINT_CONTROL_SCHEMA,
  CHECKPOINT_CONTROL_ENDPOINT_NAME,
} from "../src/contracts/activation-checkpoint-control.js";
import { activationCheckpointControlRequest } from "../src/distribution/activation-checkpoint-control.js";
import { LocalProcessAgent, type LaunchAgentStartRequest } from "../src/distribution/launch-supervisor.js";
import { normalizeExecutorIsolationPolicy } from "../src/distribution/process-environment.js";

const cleanup: Array<() => Promise<void>> = [];
afterEach(async () => {
  for (const close of cleanup.splice(0).reverse()) await close();
});

async function fixture(onRequest: (socket: Socket, header: Record<string, unknown>) => void) {
  const workspacePath = await mkdtemp(join(tmpdir(), "checkpoint-client-"));
  cleanup.push(() => rm(workspacePath, { recursive: true, force: true }));
  const token = randomBytes(32).toString("hex");
  let connections = 0;
  const sockets = new Set<Socket>();
  const server = createServer((socket) => {
    connections += 1;
    sockets.add(socket);
    socket.once("close", () => sockets.delete(socket));
    let bytes = Buffer.alloc(0);
    socket.on("data", (chunk) => {
      bytes = Buffer.concat([bytes, chunk]);
      const newline = bytes.indexOf(10);
      if (newline < 0) return;
      const header = JSON.parse(bytes.subarray(0, newline).toString("utf8")) as Record<string, unknown>;
      expect(header.token).toBe(token);
      onRequest(socket, header);
    });
  });
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  cleanup.push(async () => {
    for (const socket of sockets) socket.destroy();
    await new Promise<void>((resolveClose) => server.close(() => resolveClose()));
  });
  const port = (server.address() as { port: number }).port;
  const descriptor = {
    schema: CHECKPOINT_CONTROL_SCHEMA, host: "127.0.0.1", port,
    signature: createHmac("sha256", Buffer.from(token, "hex"))
      .update(`${CHECKPOINT_CONTROL_SCHEMA}\n127.0.0.1\n${port}`).digest("hex"),
  };
  const descriptorPath = join(workspacePath, CHECKPOINT_CONTROL_ENDPOINT_NAME);
  await writeFile(descriptorPath, JSON.stringify(descriptor));
  const options = { workspacePath, token, platform: "win32" as const };
  return { options, descriptor, descriptorPath, connectionCount: () => connections };
}

const capture = { operation: "capture", requestId: 17, maxBytes: 64 };

describe("authenticated workspace checkpoint control", () => {
  it("reassembles fragmented headers and binary checkpoint payloads", async () => {
    const value = Buffer.from([0, 255, 17, 10, 128]);
    const f = await fixture((socket) => {
      socket.write('{"ok":tr');
      setImmediate(() => {
        socket.write('ue,"payloadBytes":5,"committedPosition":37}\n');
        socket.write(value.subarray(0, 2));
        setImmediate(() => socket.end(value.subarray(2)));
      });
    });
    await expect(activationCheckpointControlRequest(f.options, capture, undefined, 64))
      .resolves.toEqual({ payload: value, committedPosition: 37 });
  });

  it.each([
    { host: "0.0.0.0" }, { host: "localhost" }, { host: "192.168.1.1" },
    { port: 0 }, { port: 65536 }, { port: 1.5 }, { token: "unexpected-secret" },
    { schema: "unknown" },
  ])("rejects unsafe endpoint fields before opening a connection: %j", async (change) => {
    const f = await fixture(() => { throw new Error("unexpected connection"); });
    await writeFile(f.descriptorPath, JSON.stringify({ ...f.descriptor, ...change }));
    await expect(activationCheckpointControlRequest(f.options, capture, undefined, 64))
      .rejects.toThrow("activation_checkpoint_control_endpoint_is_invalid");
    expect(f.connectionCount()).toBe(0);
  });

  it("rejects altered ports and stale launch tokens without disclosing the current secret", async () => {
    const f = await fixture(() => { throw new Error("unexpected connection"); });
    await writeFile(f.descriptorPath, JSON.stringify({ ...f.descriptor, port: f.descriptor.port === 65535 ? 65534 : f.descriptor.port + 1 }));
    await expect(activationCheckpointControlRequest(f.options, capture, undefined, 64))
      .rejects.toThrow("activation_checkpoint_control_endpoint_authentication_failed");
    await writeFile(f.descriptorPath, JSON.stringify(f.descriptor));
    await expect(activationCheckpointControlRequest({ ...f.options, token: randomBytes(32).toString("hex") }, capture, undefined, 64))
      .rejects.toThrow("activation_checkpoint_control_endpoint_authentication_failed");
    expect(f.connectionCount()).toBe(0);
  });

  it.each(["{" , "null", " ".repeat(4097)])("bounds and validates endpoint JSON", async (descriptor) => {
    const f = await fixture(() => { throw new Error("unexpected connection"); });
    await writeFile(f.descriptorPath, descriptor);
    await expect(activationCheckpointControlRequest(f.options, capture, undefined, 64))
      .rejects.toThrow("activation_checkpoint_control_endpoint_is_invalid");
    expect(f.connectionCount()).toBe(0);
  });

  it.each([
    ["x".repeat(4097), "header_is_too_large"],
    ['{"ok":true,"payloadBytes":65}\n', "payload_size_is_invalid"],
    ['{"ok":true,"payloadBytes":1}\nab', "payload_has_trailing_bytes"],
    ['{"ok":true,"payloadBytes":2}\na', "ended_early"],
  ])("rejects unbounded, trailing and truncated responses", async (response, error) => {
    const f = await fixture((socket) => socket.end(response));
    await expect(activationCheckpointControlRequest(f.options, capture, undefined, 64))
      .rejects.toThrow(`activation_checkpoint_control_${error}`);
  });

  it("redacts an echoed launch token from remote error messages", async () => {
    const f = await fixture((socket, header) => socket.end(JSON.stringify({ ok: false, error: `bad:${header.token}` }) + "\n"));
    await expect(activationCheckpointControlRequest(f.options, capture, undefined, 64))
      .rejects.toThrow("activation_checkpoint_control_failed:bad:[redacted]");
  });

  it("cancels pending requests and closes their socket promptly", async () => {
    let observedClose!: Promise<unknown>;
    let received!: () => void;
    const requestReceived = new Promise<void>((resolveReceived) => { received = resolveReceived; });
    const f = await fixture((socket) => { observedClose = once(socket, "close"); received(); });
    const controller = new AbortController();
    const result = activationCheckpointControlRequest({ ...f.options, signal: controller.signal }, capture, undefined, 64);
    const rejected = expect(result).rejects.toThrow("activation_checkpoint_control_aborted");
    await requestReceived;
    controller.abort();
    await rejected;
    await observedClose;
  });

  it("checks outbound restore and response limits before a connection", async () => {
    const f = await fixture(() => { throw new Error("unexpected connection"); });
    await expect(activationCheckpointControlRequest(f.options, capture, undefined, 513 * 1024 * 1024))
      .rejects.toThrow("activation_checkpoint_response_limit_is_invalid");
    await expect(activationCheckpointControlRequest(f.options, {
      operation: "restore", requestId: 17, maxBytes: 2, payloadBytes: 3, committedPosition: 37,
    }, Buffer.from("abc"), 0)).rejects.toThrow("activation_checkpoint_control_request_is_invalid");
    expect(f.connectionCount()).toBe(0);
  });
});

describe("Python checkpoint control interoperability", () => {
  it.skipIf(!process.env.MYCELLIOS_PYTHON)("captures and restores through a real isolated Python child", async () => {
    const python = process.env.MYCELLIOS_PYTHON!;
    const marker = "CHECKPOINT_PYTHON_READY";
    const script = `
import os, time
from distributed_runtime.checkpoint_control import StageCheckpointControl
class Runner:
    payload = b"python-kv"
    def activation_checkpoint_payload(self, request_id, *, max_bytes):
        assert request_id == 17 and len(self.payload) <= max_bytes
        return self.payload
    def sequence_length(self, request_id):
        return 37
    def restore_activation_checkpoint_payload(self, request_id, payload, committed_position, *, max_bytes):
        assert request_id == 17 and committed_position == 37 and len(payload) <= max_bytes
        self.payload = payload
control = StageCheckpointControl(os.environ["MYCELLIOS_EXECUTOR_WORKSPACE"], Runner())
print("${marker}", flush=True)
try:
    while True:
        control.service_pending()
        time.sleep(0.005)
finally:
    control.close()
`;
    const local = new LocalProcessAgent({
      allowedExecutables: [python], stopGraceMs: 1_000,
      env: { PYTHONPATH: resolve("python") },
      readyWhen: ({ recentStdout }) => recentStdout.includes(marker),
    });
    const handle = await local.start({
      launchId: "python-checkpoint-interop", pipelineId: "python-checkpoint-interop", deploymentGeneration: 1, nodeId: "local",
      process: { processId: "python-checkpoint", kind: "remote-stage", stageId: "stage-a",
        isolation: normalizeExecutorIsolationPolicy({ stopGraceMs: 1_000 }),
        command: { executable: python, args: ["-u", "-c", script] },
      },
    } as unknown as LaunchAgentStartRequest, new AbortController().signal);
    try {
      await handle.ready;
      await expect(handle.captureActivationCheckpoint!(17, 64))
        .resolves.toEqual({ payload: Buffer.from("python-kv"), committedPosition: 37 });
      await handle.restoreActivationCheckpoint!(17, Buffer.from("restored-python-kv"), 37, 64);
      await expect(handle.captureActivationCheckpoint!(17, 64))
        .resolves.toEqual({ payload: Buffer.from("restored-python-kv"), committedPosition: 37 });
    } finally {
      await handle.stop("test_complete");
    }
  }, 20_000);
});
