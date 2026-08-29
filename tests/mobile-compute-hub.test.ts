import type { AddressInfo } from "node:net";
import { createHash } from "node:crypto";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import WebSocket from "ws";
import { afterEach, describe, expect, it } from "vitest";
import type { CoordinatorRuntime } from "../src/coordinator/server.js";
import { createCoordinator } from "../src/coordinator/server.js";
import {
  generateWorkerAdmissionCredential,
  workerAdmissionSigner,
} from "../src/worker/admission-credential.js";

const runtimes: CoordinatorRuntime[] = [];
const artifactDirectories: string[] = [];

afterEach(async () => {
  await Promise.all(runtimes.splice(0).map((runtime) => runtime.close()));
  for (const directory of artifactDirectories.splice(0)) {
    rmSync(directory, { recursive: true, force: true });
  }
});

describe("mobile compute hub", () => {
  it("uses a mobile worker as a hash-verified real SwiGLU expert", async () => {
    const runtime = await createCoordinator(
      {
        host: "127.0.0.1", port: 0, databasePath: ":memory:", requestTimeoutMs: 10_000,
        internalToken: "expert-admin",
        mobileExpertArtifactsPath: newArtifactDirectory(),
      },
      { logger: false },
    );
    runtimes.push(runtime);
    await runtime.app.listen({ host: "127.0.0.1", port: 0 });
    const address = runtime.app.server.address() as AddressInfo;
    const registration = await runtime.app.inject({
      method: "POST", url: "/mobile/v1/register",
      payload: registrationPayload(),
    });
    const credentials = registration.json<{ workerId: string; token: string }>();
    const socket = new WebSocket(
      `ws://127.0.0.1:${address.port}/mobile/v1/connect?workerId=${credentials.workerId}&token=${credentials.token}`,
    );
    expect((await nextMessage(socket)).type).toBe("server.ready");
    const replicaRegistration = await runtime.app.inject({
      method: "POST", url: "/mobile/v1/register",
      payload: { ...registrationPayload(), clientId: "mobile-client-replica", name: "Replica phone" },
    });
    const replicaCredentials = replicaRegistration.json<{ workerId: string; token: string }>();
    const replicaSocket = new WebSocket(
      `ws://127.0.0.1:${address.port}/mobile/v1/connect?workerId=${replicaCredentials.workerId}&token=${replicaCredentials.token}`,
    );
    expect((await nextMessage(replicaSocket)).type).toBe("server.ready");

    const hiddenSize = 3;
    const intermediateSize = 4;
    const gate = new Float32Array([0.2, -0.1, 0.3, -0.4, 0.5, 0.1, 0.6, -0.2, 0.4, 0.1, 0.3, -0.5]);
    const up = new Float32Array([-0.3, 0.2, 0.7, 0.1, -0.6, 0.2, 0.5, 0.4, -0.2, -0.1, 0.8, 0.3]);
    const down = new Float32Array([0.4, -0.2, 0.1, 0.3, -0.5, 0.6, 0.2, -0.1, 0.3, 0.1, -0.4, 0.7]);
    const packed = Buffer.concat([floatBuffer(gate), floatBuffer(up), floatBuffer(down)]);
    const weightsHash = createHash("sha256").update(packed).digest("hex");
    const uploaded = await runtime.app.inject({
      method: "PUT",
      url: `/internal/v1/mobile/experts/weights/${weightsHash}`,
      headers: {
        authorization: "Bearer expert-admin",
        "content-type": "application/octet-stream",
      },
      payload: packed,
    });
    expect(uploaded.statusCode).toBe(201);
    const canary = new Float32Array([-0.75, 0, 0.75]);
    const canaryOutput = swiGlu(canary, 1, hiddenSize, intermediateSize, gate, up, down);
    const artifact = await runtime.app.inject({
      method: "POST",
      url: "/internal/v1/mobile/experts/register",
      headers: { authorization: "Bearer expert-admin" },
      payload: {
        modelId: "tiny-moe", modelDigest: "sha256:model", layer: 2, expert: 7,
        contentId: "sha256:expert-content", weightsHash, hiddenSize, intermediateSize,
        dtype: "float32", activation: "silu",
        canaryInputBase64: floatBuffer(canary).toString("base64"),
        canaryOutputBase64: floatBuffer(canaryOutput).toString("base64"),
      },
    });
    expect(artifact.statusCode).toBe(201);
    const { artifactId } = artifact.json<{ artifactId: string }>();

    const blockedBeforeValidation = await runtime.app.inject({
      method: "POST", url: "/internal/v1/mobile/experts/prepare",
      headers: { authorization: "Bearer expert-admin" }, payload: { artifactId },
    });
    expect(blockedBeforeValidation.statusCode).toBe(503);
    expect(blockedBeforeValidation.json()).toMatchObject({
      error: { message: expect.stringContaining("validated visible mobile workers") },
    });
    await validateMobileWorker(socket);
    await validateMobileWorker(replicaSocket);

    for (const workerSocket of [socket, replicaSocket]) {
      workerSocket.on("message", (raw) => {
        const message = JSON.parse(raw.toString()) as { type: string; payload: Record<string, unknown> };
        if (message.type === "expert.load") {
          workerSocket.send(JSON.stringify({ v: 1, type: "expert.ready", payload: {
            taskId: message.payload.taskId, leaseId: message.payload.leaseId, artifactId,
            canaryOutputBase64: floatBuffer(canaryOutput).toString("base64"),
            backend: "webgpu", durationMs: 2,
          } }));
        }
        if (message.type === "expert.execute") {
          const input = floatArray(String(message.payload.activationsBase64));
          const rows = Number(message.payload.rows);
          const output = swiGlu(input, rows, hiddenSize, intermediateSize, gate, up, down);
          workerSocket.send(JSON.stringify({ v: 1, type: "expert.result", payload: {
            taskId: message.payload.taskId, leaseId: message.payload.leaseId, artifactId,
            rows, hiddenSize, outputBase64: floatBuffer(output).toString("base64"),
            backend: "webgpu", durationMs: 3,
          } }));
        }
      });
    }

    const prepared = await runtime.app.inject({
      method: "POST", url: "/internal/v1/mobile/experts/prepare",
      headers: { authorization: "Bearer expert-admin" }, payload: { artifactId },
    });
    expect(prepared.statusCode).toBe(200);
    expect(prepared.json()).toMatchObject({ resident: true, workerId: credentials.workerId });
    const activations = new Float32Array([0.25, -0.5, 1, -0.1, 0.3, 0.8]);
    const executed = await runtime.app.inject({
      method: "POST", url: "/internal/v1/mobile/experts/execute",
      headers: { authorization: "Bearer expert-admin" }, payload: {
        artifactId, rows: 2, hiddenSize,
        activationsBase64: floatBuffer(activations).toString("base64"),
      },
    });
    expect(executed.statusCode).toBe(200);
    const execution = executed.json<{ outputBase64: string; replicaWorkerIds: string[] }>();
    expect(new Set(execution.replicaWorkerIds)).toEqual(new Set([
      credentials.workerId,
      replicaCredentials.workerId,
    ]));
    const actual = floatArray(execution.outputBase64);
    const expected = swiGlu(activations, 2, hiddenSize, intermediateSize, gate, up, down);
    expect(actual).toHaveLength(expected.length);
    for (let index = 0; index < actual.length; index += 1) {
      expect(actual[index]).toBeCloseTo(expected[index] ?? Number.NaN, 5);
    }
    const dashboard = await runtime.app.inject({ method: "GET", url: "/internal/v1/workers" });
    const mobile = dashboard.json<{ data: Array<{
      id: string; mobile?: { residentExperts: Array<{ artifactId: string; modelId: string }> };
    }> }>().data.find((worker) => worker.id === credentials.workerId);
    expect(mobile?.mobile?.residentExperts).toContainEqual(expect.objectContaining({
      artifactId, modelId: "tiny-moe",
    }));
    socket.close(1000, "user stopped contribution");
    replicaSocket.close(1000, "user stopped contribution");
    await waitUntil(() => runtime.mobileHub.listWorkers().length === 0);
  });

  it("fails closed when only one browser expert replica is available", async () => {
    const rig = await createExpertRig(["honest"]);
    const executed = await rig.runtime.app.inject({
      method: "POST",
      url: "/internal/v1/mobile/experts/execute",
      headers: { authorization: "Bearer expert-admin" },
      payload: rig.executionPayload,
    });
    expect(executed.statusCode).toBe(503);
    expect(executed.json()).toMatchObject({
      error: { message: expect.stringContaining("two distinct validated visible mobile workers") },
    });
    expect(rig.runtime.mobileHub.listWorkers()[0]?.verifiedTasks).toBe(1);
  });

  it("discards mismatched replicated expert tensors", async () => {
    const rig = await createExpertRig(["honest", "mismatch"]);
    const executed = await rig.runtime.app.inject({
      method: "POST",
      url: "/internal/v1/mobile/experts/execute",
      headers: { authorization: "Bearer expert-admin" },
      payload: rig.executionPayload,
    });
    expect(executed.statusCode).toBe(503);
    expect(executed.json()).toMatchObject({
      error: { message: "mobile expert replicas returned different tensors" },
    });
    for (const worker of rig.runtime.mobileHub.listWorkers()) {
      expect(worker.verifiedTasks).toBe(1);
      expect(worker.residentExperts).toEqual([]);
    }
  });

  it("dispatches real matrix work and independently verifies the result", async () => {
    const runtime = await createCoordinator(
      {
        host: "127.0.0.1",
        port: 0,
        databasePath: ":memory:",
        requestTimeoutMs: 10_000,
        mobileExpertArtifactsPath: newArtifactDirectory(),
      },
      { logger: false },
    );
    runtimes.push(runtime);
    await runtime.app.listen({ host: "127.0.0.1", port: 0 });
    const address = runtime.app.server.address() as AddressInfo;

    const registration = await runtime.app.inject({
      method: "POST",
      url: "/mobile/v1/register",
      payload: registrationPayload(),
    });
    expect(registration.statusCode).toBe(201);
    const credentials = registration.json<{ workerId: string; token: string }>();
    const socket = new WebSocket(
      `ws://127.0.0.1:${address.port}/mobile/v1/connect?workerId=${credentials.workerId}&token=${credentials.token}`,
    );

    const ready = await nextMessage(socket);
    expect(ready.type).toBe("server.ready");
    socket.send(JSON.stringify({ v: 1, type: "work.request", payload: {} }));
    const offered = await nextMessage(socket);
    expect(offered.type).toBe("compute.offer");
    const task = offered.payload as {
      taskId: string;
      leaseId: string;
      size: number;
      seed: number;
    };
    expect(task.size).toBe(64);
    socket.send(
      JSON.stringify({
        v: 1,
        type: "compute.accept",
        payload: { taskId: task.taskId, leaseId: task.leaseId },
      }),
    );
    socket.send(
      JSON.stringify({
        v: 1,
        type: "compute.result",
        payload: {
          taskId: task.taskId,
          leaseId: task.leaseId,
          backend: "cpu",
          durationMs: 12,
          estimatedGflops: 0.5,
          samples: independentSamples(task.size, task.seed),
        },
      }),
    );
    const verified = await nextMessage(socket);
    expect(verified.type).toBe("compute.verified");
    expect(verified.payload).toMatchObject({ inferenceReady: true, verifiedTasks: 1 });

    socket.send(JSON.stringify({ v: 1, type: "work.request", payload: {} }));
    await expectNoMessage(socket);

    const workers = await runtime.app.inject({ method: "GET", url: "/mobile/v1/workers" });
    expect(workers.json<{ data: Array<{ verifiedTasks: number; status: string }> }>().data[0]).toMatchObject({
      verifiedTasks: 1,
      status: "online",
    });
    const dashboard = await runtime.app.inject({ method: "GET", url: "/internal/v1/workers" });
    expect(dashboard.json<{ data: Array<{ id: string; deployments: unknown[] }> }>().data).toContainEqual(
      expect.objectContaining({ id: credentials.workerId, deployments: [] }),
    );
    socket.close();
  });

  it("shows only active browser leases and expires disconnected sessions", async () => {
    const runtime = await createCoordinator(
      {
        host: "127.0.0.1",
        port: 0,
        databasePath: ":memory:",
        requestTimeoutMs: 10_000,
        mobileExpertArtifactsPath: newArtifactDirectory(),
      },
      { logger: false, mobileDisconnectedRetentionMs: 60_000 },
    );
    runtimes.push(runtime);
    await runtime.app.listen({ host: "127.0.0.1", port: 0 });
    const address = runtime.app.server.address() as AddressInfo;

    const firstRegistration = await runtime.app.inject({
      method: "POST",
      url: "/mobile/v1/register",
      payload: { ...registrationPayload(), clientId: "browser-installation-old" },
    });
    const first = firstRegistration.json<{ workerId: string; token: string }>();
    const firstSocket = new WebSocket(
      `ws://127.0.0.1:${address.port}/mobile/v1/connect?workerId=${first.workerId}&token=${first.token}`,
    );
    await nextMessage(firstSocket);
    firstSocket.terminate();
    await waitUntil(() => runtime.mobileHub.connectedCount() === 0);

    const secondRegistration = await runtime.app.inject({
      method: "POST",
      url: "/mobile/v1/register",
      payload: { ...registrationPayload(), clientId: "browser-installation-new" },
    });
    const second = secondRegistration.json<{ workerId: string; token: string }>();
    const secondSocket = new WebSocket(
      `ws://127.0.0.1:${address.port}/mobile/v1/connect?workerId=${second.workerId}&token=${second.token}`,
    );
    await nextMessage(secondSocket);

    expect(runtime.mobileHub.listWorkers().map((worker) => worker.id)).toEqual([second.workerId]);
    expect(runtime.mobileHub.expireDisconnectedWorkers(Date.now() + 60_001)).toBe(1);
    expect(runtime.mobileHub.listWorkers().map((worker) => worker.id)).toEqual([second.workerId]);
    secondSocket.close(1000, "user stopped contribution");
  });

  it("requires the optional invitation token and serves the built PWA", async () => {
    const runtime = await createCoordinator(
      {
        host: "127.0.0.1",
        port: 0,
        databasePath: ":memory:",
        requestTimeoutMs: 10_000,
        mobileJoinToken: "invite-test",
        mobileAssetsPath: resolve("tests/fixtures/mobile-assets"),
        mobileExpertArtifactsPath: newArtifactDirectory(),
      },
      { logger: false },
    );
    runtimes.push(runtime);
    const signer = workerAdmissionSigner(generateWorkerAdmissionCredential());
    const challengePayload = {
      identity: { kind: "browser", id: "mobile-client-test" },
      publicKey: signer.publicKey,
      protocol: { min: 1, max: 1 },
      registrationDigest: `sha256:${"a".repeat(64)}`,
    };
    const rejectedChallenge = await runtime.app.inject({
      method: "POST",
      url: "/mobile/v1/admission-challenge",
      payload: challengePayload,
    });
    expect(rejectedChallenge.statusCode).toBe(401);
    const acceptedChallenge = await runtime.app.inject({
      method: "POST",
      url: "/mobile/v1/admission-challenge",
      payload: { ...challengePayload, joinToken: "invite-test" },
    });
    expect(acceptedChallenge.statusCode).toBe(200);

    const rejected = await runtime.app.inject({
      method: "POST",
      url: "/mobile/v1/register",
      payload: registrationPayload(),
    });
    expect(rejected.statusCode).toBe(401);
    const accepted = await runtime.app.inject({
      method: "POST",
      url: "/mobile/v1/register",
      payload: { ...registrationPayload(), joinToken: "invite-test" },
    });
    expect(accepted.statusCode).toBe(201);

    const page = await runtime.app.inject({ method: "GET", url: "/mobile/" });
    expect(page.statusCode).toBe(200);
    expect(page.body).toContain("Contribute power to the network");
    const browserPage = await runtime.app.inject({ method: "GET", url: "/browser/" });
    expect(browserPage.statusCode).toBe(200);
    const manifest = await runtime.app.inject({ method: "GET", url: "/mobile/manifest.webmanifest" });
    expect(manifest.statusCode).toBe(200);
    expect(manifest.json<{ short_name: string }>().short_name).toBe("mycellios");
    const browserManifest = await runtime.app.inject({ method: "GET", url: "/browser/manifest.webmanifest" });
    expect(browserManifest.statusCode).toBe(200);
  });
});

function registrationPayload() {
  return {
    clientId: "mobile-client-test",
    name: "Test phone",
    region: "es-mad",
    platform: "vitest",
    backend: "cpu",
    performanceLevel: "balanced",
    capabilities: {
      webgpu: false,
      wasm: true,
      hardwareConcurrency: 8,
      deviceMemoryGb: 8,
    },
    benchmark: { durationMs: 10, estimatedGflops: 0.4, matrixSize: 48 },
  };
}

async function createExpertRig(modes: Array<"honest" | "mismatch">): Promise<{
  runtime: CoordinatorRuntime;
  executionPayload: { artifactId: string; rows: number; hiddenSize: number; activationsBase64: string };
}> {
  const runtime = await createCoordinator({
    host: "127.0.0.1",
    port: 0,
    databasePath: ":memory:",
    requestTimeoutMs: 10_000,
    internalToken: "expert-admin",
    mobileJoinToken: "trusted-expert-invite",
    mobileExpertArtifactsPath: newArtifactDirectory(),
  }, { logger: false });
  runtimes.push(runtime);
  await runtime.app.listen({ host: "127.0.0.1", port: 0 });
  const address = runtime.app.server.address() as AddressInfo;
  const gate = new Float32Array([0.75]);
  const up = new Float32Array([0.5]);
  const down = new Float32Array([1.25]);
  const packed = Buffer.concat([floatBuffer(gate), floatBuffer(up), floatBuffer(down)]);
  const weightsHash = createHash("sha256").update(packed).digest("hex");
  const upload = await runtime.app.inject({
    method: "PUT",
    url: `/internal/v1/mobile/experts/weights/${weightsHash}`,
    headers: { authorization: "Bearer expert-admin", "content-type": "application/octet-stream" },
    payload: packed,
  });
  expect(upload.statusCode).toBe(201);
  const canary = new Float32Array([0.5]);
  const canaryOutput = swiGlu(canary, 1, 1, 1, gate, up, down);
  const artifact = await runtime.app.inject({
    method: "POST",
    url: "/internal/v1/mobile/experts/register",
    headers: { authorization: "Bearer expert-admin" },
    payload: {
      modelId: "replicated-tiny-moe",
      modelDigest: "sha256:replicated-model",
      layer: 0,
      expert: 0,
      contentId: "sha256:replicated-expert",
      weightsHash,
      hiddenSize: 1,
      intermediateSize: 1,
      dtype: "float32",
      activation: "silu",
      canaryInputBase64: floatBuffer(canary).toString("base64"),
      canaryOutputBase64: floatBuffer(canaryOutput).toString("base64"),
    },
  });
  expect(artifact.statusCode).toBe(201);
  const { artifactId } = artifact.json<{ artifactId: string }>();

  for (const [index, mode] of modes.entries()) {
    const registration = await runtime.app.inject({
      method: "POST",
      url: "/mobile/v1/register",
      payload: {
        ...registrationPayload(),
        clientId: `replica-client-${index}`,
        name: `Replica ${index}`,
        joinToken: "trusted-expert-invite",
      },
    });
    const credentials = registration.json<{ workerId: string; token: string }>();
    const socket = new WebSocket(
      `ws://127.0.0.1:${address.port}/mobile/v1/connect?workerId=${credentials.workerId}&token=${credentials.token}`,
    );
    expect((await nextMessage(socket)).type).toBe("server.ready");
    await validateMobileWorker(socket);
    socket.on("message", (raw) => {
      const message = JSON.parse(raw.toString()) as { type: string; payload: Record<string, unknown> };
      if (message.type === "expert.load") {
        socket.send(JSON.stringify({ v: 1, type: "expert.ready", payload: {
          taskId: message.payload.taskId,
          leaseId: message.payload.leaseId,
          artifactId,
          canaryOutputBase64: floatBuffer(canaryOutput).toString("base64"),
          backend: "webgpu",
          durationMs: 1,
        } }));
      }
      if (message.type === "expert.execute") {
        const input = floatArray(String(message.payload.activationsBase64));
        const output = swiGlu(input, Number(message.payload.rows), 1, 1, gate, up, down);
        if (mode === "mismatch") output[0] = (output[0] ?? 0) + 1;
        socket.send(JSON.stringify({ v: 1, type: "expert.result", payload: {
          taskId: message.payload.taskId,
          leaseId: message.payload.leaseId,
          artifactId,
          rows: message.payload.rows,
          hiddenSize: 1,
          outputBase64: floatBuffer(output).toString("base64"),
          backend: "webgpu",
          durationMs: 1,
        } }));
      }
    });
  }

  const activations = new Float32Array([0.25]);
  return {
    runtime,
    executionPayload: {
      artifactId,
      rows: 1,
      hiddenSize: 1,
      activationsBase64: floatBuffer(activations).toString("base64"),
    },
  };
}

function newArtifactDirectory(): string {
  const directory = mkdtempSync(join(tmpdir(), "mycellios-mobile-experts-"));
  artifactDirectories.push(directory);
  return directory;
}

function nextMessage(socket: WebSocket): Promise<{ type: string; payload: unknown }> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error("Timed out waiting for WebSocket message")), 5_000);
    socket.once("message", (raw) => {
      clearTimeout(timer);
      resolve(JSON.parse(raw.toString()) as { type: string; payload: unknown });
    });
    socket.once("error", (error) => {
      clearTimeout(timer);
      reject(error);
    });
  });
}

async function validateMobileWorker(socket: WebSocket): Promise<void> {
  socket.send(JSON.stringify({ v: 1, type: "work.request", payload: {} }));
  const offered = await nextMessage(socket);
  expect(offered.type).toBe("compute.offer");
  const task = offered.payload as { taskId: string; leaseId: string; size: number; seed: number };
  socket.send(JSON.stringify({
    v: 1,
    type: "compute.accept",
    payload: { taskId: task.taskId, leaseId: task.leaseId },
  }));
  socket.send(JSON.stringify({
    v: 1,
    type: "compute.result",
    payload: {
      taskId: task.taskId,
      leaseId: task.leaseId,
      backend: "cpu",
      durationMs: 12,
      estimatedGflops: 0.5,
      samples: independentSamples(task.size, task.seed),
    },
  }));
  const verified = await nextMessage(socket);
  expect(verified.type).toBe("compute.verified");
  expect(verified.payload).toMatchObject({ inferenceReady: true });
}

function expectNoMessage(socket: WebSocket, durationMs = 120): Promise<void> {
  return new Promise((resolve, reject) => {
    const onMessage = (raw: WebSocket.RawData) => {
      clearTimeout(timer);
      reject(new Error(`Unexpected WebSocket message: ${raw.toString()}`));
    };
    const timer = setTimeout(() => {
      socket.off("message", onMessage);
      resolve();
    }, durationMs);
    socket.once("message", onMessage);
  });
}

async function waitUntil(predicate: () => boolean, timeoutMs = 2_000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (!predicate()) {
    if (Date.now() >= deadline) throw new Error("Timed out waiting for mobile worker state");
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
}

function independentSamples(size: number, seed: number): number[] {
  const positions: Array<readonly [number, number]> = [
    [0, 0],
    [Math.floor(size / 2), Math.floor(size / 3)],
    [size - 1, size - 1],
    [Math.floor(size / 3), size - 2],
  ];
  return positions.map(([row, column]) => {
    let result = 0;
    for (let k = 0; k < size; k += 1) {
      const left = ((row * 3 + k * 5 + seed) % 31 - 15) / 16;
      const right = ((k * 7 + column * 11 + seed * 3) % 29 - 14) / 16;
      result += left * right;
    }
    return result;
  });
}

function floatBuffer(values: Float32Array): Buffer {
  return Buffer.from(values.buffer, values.byteOffset, values.byteLength);
}

function floatArray(value: string): Float32Array {
  const buffer = Buffer.from(value, "base64");
  return new Float32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 4).slice();
}

function swiGlu(
  input: Float32Array,
  rows: number,
  hidden: number,
  intermediate: number,
  gate: Float32Array,
  up: Float32Array,
  down: Float32Array,
): Float32Array {
  const activated = new Float32Array(rows * intermediate);
  for (let row = 0; row < rows; row += 1) {
    for (let neuron = 0; neuron < intermediate; neuron += 1) {
      let gateValue = 0;
      let upValue = 0;
      for (let column = 0; column < hidden; column += 1) {
        gateValue += (input[row * hidden + column] ?? 0) * (gate[neuron * hidden + column] ?? 0);
        upValue += (input[row * hidden + column] ?? 0) * (up[neuron * hidden + column] ?? 0);
      }
      activated[row * intermediate + neuron] = gateValue / (1 + Math.exp(-gateValue)) * upValue;
    }
  }
  const output = new Float32Array(rows * hidden);
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < hidden; column += 1) {
      for (let neuron = 0; neuron < intermediate; neuron += 1) {
        const outputIndex = row * hidden + column;
        output[outputIndex] = (output[outputIndex] ?? 0)
          + (activated[row * intermediate + neuron] ?? 0)
            * (down[column * intermediate + neuron] ?? 0);
      }
    }
  }
  return output;
}
