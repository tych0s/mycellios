import { createHash, randomUUID } from "node:crypto";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import Fastify from "fastify";
import { afterEach, expect, it } from "vitest";
import { BrowserLayerHub, type BrowserLayerPeer } from "../src/coordinator/browser-layer-hub.js";

const directories: string[] = [];
afterEach(() => {
  for (const directory of directories.splice(0)) rmSync(directory, { recursive: true, force: true });
});

function encoded(values: number[]): string {
  return Buffer.from(new Float32Array(values).buffer).toString("base64");
}

it("assigns a content-verified layer to one CPU browser and enforces its KV sequence", async () => {
  const directory = mkdtempSync(join(tmpdir(), "mycellios-layer-"));
  directories.push(directory);
  const app = Fastify();
  app.addContentTypeParser("application/octet-stream", { parseAs: "buffer" },
    (_request, body, done) => done(null, body));
  const peer: BrowserLayerPeer = {
    id: randomUUID(), token: "browser-worker-token", backend: "cpu",
    connected: true, visible: true, validated: true, busy: false,
  };
  const graph = Buffer.from("onnx-fixture-graph");
  const graphSha256 = createHash("sha256").update(graph).digest("hex");
  const canaryOutput = encoded([0.25, -0.5]);
  let artifactId = "";
  let resetCount = 0;
  let verifiedCount = 0;
  const hub = new BrowserLayerHub({
    artifactDirectory: directory,
    peers: () => [peer],
    onExecutionVerified: () => { verifiedCount += 1; },
    send: (_id, type, payload) => {
      const task = payload as Record<string, unknown>;
      if (type === "layer.reset") { resetCount += 1; return; }
      queueMicrotask(() => {
        if (type === "layer.load") {
          hub.handleMessage(peer.id, "layer.ready", {
            taskId: task.taskId, leaseId: task.leaseId, artifactId,
            outputBase64: canaryOutput, backend: "cpu", durationMs: 1,
          });
        } else if (type === "layer.execute") {
          const position = Number(task.position);
          const tokens = Number(task.tokens);
          hub.handleMessage(peer.id, "layer.result", {
            taskId: task.taskId, leaseId: task.leaseId, artifactId,
            requestId: task.requestId, position, tokens,
            outputBase64: encoded([1, 2]),
            stateBase64: { key: encoded([3]), value: encoded([4]) },
            backend: "cpu", durationMs: 2,
          });
        }
      });
    },
  });
  hub.attach(app);
  const graphUpload = await app.inject({ method: "PUT",
    url: `/internal/v1/mobile/layers/graphs/${graphSha256}`,
    headers: { "content-type": "application/octet-stream" }, payload: graph });
  expect(graphUpload.statusCode).toBe(201);
  const manifestInput = {
      schema: "mycellios-browser-layer/2", graphSha256,
      modelDigest: `sha256:${"a".repeat(64)}`, layer: 0,
      hidden: { inputName: "hidden", outputName: "output", width: 2 },
      auxiliary: [{ name: "cos", shape: [1, "tokens", 1] },
        { name: "sin", shape: [1, "tokens", 1] },
        { name: "attention_mask", shape: [1, 1, "tokens", "total"] }],
      state: [{ name: "key", inputName: "past_key", outputName: "key",
        deltaOutputName: "new_key", shape: [1, 1, "past", 1], tokenAxis: 2 },
      { name: "value", inputName: "past_value", outputName: "value",
        deltaOutputName: "new_value", shape: [1, 1, "past", 1], tokenAxis: 2 }],
      maxContextTokens: 8,
      canary: {
        tokens: 1, hiddenBase64: encoded([1, 2]),
        auxiliaryBase64: { cos: encoded([1]), sin: encoded([0]),
          attention_mask: encoded([0]) },
        expectedBase64: canaryOutput,
      },
    };
  const registered = await app.inject({ method: "POST", url: "/internal/v1/mobile/layers/register",
    payload: manifestInput });
  expect(registered.statusCode).toBe(201);
  artifactId = registered.json<{ artifactId: string }>().artifactId;
  const unauthorized = await app.inject({ method: "GET",
    url: `/mobile/v1/layers/${artifactId}/manifest` });
  expect(unauthorized.statusCode).toBe(401);
  const prepared = await app.inject({ method: "POST", url: "/internal/v1/mobile/layers/prepare",
    payload: { artifactId } });
  expect(prepared.statusCode).toBe(200);
  expect(prepared.json()).toMatchObject({ workerId: peer.id, backend: "cpu", resident: true });
  const manifest = await app.inject({ method: "GET",
    url: `/mobile/v1/layers/${artifactId}/manifest`,
    headers: { authorization: `Bearer ${peer.token}` } });
  expect(manifest.statusCode).toBe(200);
  const graphRead = await app.inject({ method: "GET",
    url: `/mobile/v1/layers/graphs/${graphSha256}`,
    headers: { authorization: `Bearer ${peer.token}` } });
  expect(graphRead.rawPayload).toEqual(graph);
  const requestId = randomUUID();
  const first = await app.inject({ method: "POST", url: "/internal/v1/mobile/layers/execute",
    payload: { artifactId, requestId, position: 0, tokens: 1,
      hiddenBase64: encoded([5, 6]), auxiliaryBase64: { cos: encoded([1]),
        sin: encoded([0]), attention_mask: encoded([0]) } } });
  expect(first.statusCode).toBe(200);
  expect(first.json()).toMatchObject({ workerId: peer.id, requestId, position: 0 });
  expect(verifiedCount).toBe(1);
  const duplicate = await app.inject({ method: "POST", url: "/internal/v1/mobile/layers/execute",
    payload: { artifactId, requestId, position: 0, tokens: 1,
      hiddenBase64: encoded([5, 6]), auxiliaryBase64: { cos: encoded([1]),
        sin: encoded([0]), attention_mask: encoded([0]) } } });
  expect(duplicate.statusCode).toBe(503);
  expect(verifiedCount).toBe(1);
  const next = await app.inject({ method: "POST", url: "/internal/v1/mobile/layers/register",
    payload: { ...manifestInput, layer: 1 } });
  expect(next.statusCode).toBe(201);
  const nextArtifactId = next.json<{ artifactId: string }>().artifactId;
  const busy = await app.inject({ method: "POST", url: "/internal/v1/mobile/layers/prepare",
    payload: { artifactId: nextArtifactId } });
  expect(busy.statusCode).toBe(503);
  await app.inject({ method: "POST", url: "/internal/v1/mobile/layers/reset",
    payload: { artifactId, requestId } });
  expect(resetCount).toBe(1);
  artifactId = nextArtifactId;
  const switched = await app.inject({ method: "POST", url: "/internal/v1/mobile/layers/prepare",
    payload: { artifactId } });
  expect(switched.statusCode).toBe(200);
  peer.connected = false;
  hub.disconnect(peer.id);
  const unavailable = await app.inject({ method: "POST",
    url: "/internal/v1/mobile/layers/prepare", payload: { artifactId } });
  expect(unavailable.statusCode).toBe(503);
  const revoked = await app.inject({ method: "GET",
    url: `/mobile/v1/layers/${artifactId}/manifest`,
    headers: { authorization: `Bearer ${peer.token}` } });
  expect(revoked.statusCode).toBe(401);
  hub.close();
  await app.close();
});

it("tries another admitted browser when the preferred GPU fails the model canary", async () => {
  const directory = mkdtempSync(join(tmpdir(), "mycellios-layer-"));
  directories.push(directory);
  const app = Fastify();
  app.addContentTypeParser("application/octet-stream", { parseAs: "buffer" },
    (_request, body, done) => done(null, body));
  const gpu: BrowserLayerPeer = { id: randomUUID(), token: "gpu-token", backend: "webgpu",
    connected: true, visible: true, validated: true, busy: false };
  const cpu: BrowserLayerPeer = { id: randomUUID(), token: "cpu-token", backend: "cpu",
    connected: true, visible: true, validated: true, busy: false };
  const graph = Buffer.from("portable-onnx-layer");
  const graphSha256 = createHash("sha256").update(graph).digest("hex");
  const attempted: string[] = [];
  let artifactId = "";
  const hub = new BrowserLayerHub({
    artifactDirectory: directory,
    peers: () => [cpu, gpu],
    send: (peerId, type, payload) => {
      const task = payload as Record<string, unknown>;
      if (type === "layer.load") attempted.push(peerId);
      queueMicrotask(() => {
        if (type === "layer.load") hub.handleMessage(peerId, "layer.ready", {
          taskId: task.taskId, leaseId: task.leaseId, artifactId,
          outputBase64: encoded(peerId === gpu.id ? [99, 99] : [1, 2]),
          backend: peerId === gpu.id ? "webgpu" : "cpu", durationMs: 1,
        });
        if (type === "layer.execute") hub.handleMessage(peerId, "layer.result", {
          taskId: task.taskId, leaseId: task.leaseId, artifactId,
          requestId: task.requestId, position: task.position, tokens: task.tokens,
          outputBase64: encoded([3, 4]), stateBase64: {}, backend: "cpu", durationMs: 2,
        });
      });
    },
  });
  hub.attach(app);
  const uploaded = await app.inject({ method: "PUT",
    url: `/internal/v1/mobile/layers/graphs/${graphSha256}`,
    headers: { "content-type": "application/octet-stream" }, payload: graph });
  expect(uploaded.statusCode).toBe(201);
  const manifestInput = { schema: "mycellios-browser-layer/2", graphSha256,
      modelDigest: `sha256:${"b".repeat(64)}`, layer: 3,
      hidden: { inputName: "activation", outputName: "result", width: 2 },
      auxiliary: [], state: [], maxContextTokens: 8,
      canary: { tokens: 1, hiddenBase64: encoded([1, 2]),
        auxiliaryBase64: {}, expectedBase64: encoded([1, 2]) } };
  const registered = await app.inject({ method: "POST", url: "/internal/v1/mobile/layers/register",
    payload: manifestInput });
  expect(registered.statusCode).toBe(201);
  artifactId = registered.json<{ artifactId: string }>().artifactId;
  const prepared = await app.inject({ method: "POST", url: "/internal/v1/mobile/layers/prepare",
    payload: { artifactId } });
  expect(prepared.statusCode).toBe(200);
  expect(prepared.json()).toMatchObject({ workerId: cpu.id, backend: "cpu" });
  expect(attempted).toEqual([gpu.id, cpu.id]);
  const executed = await app.inject({ method: "POST", url: "/internal/v1/mobile/layers/execute",
    payload: { artifactId, requestId: randomUUID(), position: 0, tokens: 1,
      hiddenBase64: encoded([1, 2]), auxiliaryBase64: {} } });
  expect(executed.statusCode).toBe(200);
  expect(executed.json()).toMatchObject({ workerId: cpu.id, outputBase64: encoded([3, 4]) });
  const republished = await app.inject({ method: "POST", url: "/internal/v1/mobile/layers/register",
    payload: { ...manifestInput, maxContextTokens: 9 } });
  expect(republished.statusCode).toBe(201);
  const latestId = republished.json<{ artifactId: string }>().artifactId;
  expect(latestId).not.toBe(artifactId);
  hub.close();
  await app.close();
  const restarted = Fastify();
  new BrowserLayerHub({ artifactDirectory: directory, peers: () => [], send: () => undefined })
    .attach(restarted);
  const lookup = await restarted.inject({ method: "GET",
    url: `/internal/v1/mobile/layers/lookup?modelDigest=${manifestInput.modelDigest}&layerStart=3&layerEnd=4` });
  expect(lookup.statusCode).toBe(200);
  expect(lookup.json()).toEqual({ data: [{ artifactId: latestId,
    modelDigest: manifestInput.modelDigest, layer: 3, maxContextTokens: 9 }] });
  const wrongModel = await restarted.inject({ method: "GET",
    url: `/internal/v1/mobile/layers/lookup?modelDigest=sha256:${"c".repeat(64)}&layerStart=3&layerEnd=4` });
  expect(wrongModel.json()).toEqual({ data: [] });
  await restarted.close();
});
