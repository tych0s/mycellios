import { createServer, type RequestListener, type Server } from "node:http";
import { afterEach, describe, expect, it } from "vitest";
import {
  MycelliosPipelineAdapter,
  normalizeMycelliosPipelineBaseUrl,
} from "../src/adapters/mycellios-pipeline.js";

const MODEL_DIGEST = `sha256:${"a".repeat(64)}`;
const ACTIVATION_ID = "pipeline-activation-42";

describe("native Mycellios pipeline adapter", () => {
  const servers: Server[] = [];

  afterEach(async () => {
    await Promise.all(
      servers.splice(0).map(
        (server) => new Promise<void>((resolve) => server.close(() => resolve())),
      ),
    );
  });

  it("validates native pipeline identity and normalizes its SSE stream", async () => {
    let receivedSessionId: string | undefined;
    const baseUrl = await listen((request, response) => {
      if (request.url === "/health") {
        response.setHeader("content-type", "application/json");
        response.end(JSON.stringify({
          status: "ready",
          model: "test-model",
          artifact_identity: MODEL_DIGEST,
          pipeline_snapshot_identity: ACTIVATION_ID,
          stages: 2,
          boundaries: [0, 14, 28],
        }));
        return;
      }
      if (request.url === "/v1/chat/completions") {
        receivedSessionId = request.headers["x-session-id"] as string | undefined;
        response.setHeader("content-type", "text/event-stream");
        response.write('data: {"choices":[{"delta":{"content":"hola "}}]}\n\n');
        response.write('data: {"choices":[{"delta":{"content":"mundo"}}]}\n\n');
        response.write('data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":12,"completion_tokens":2},"distribution_metrics":{"ttft_ms":18.4,"pipeline_ms":41.2,"reused_kv_tokens":9}}\n\n');
        response.end("data: [DONE]\n\n");
      }
    });
    const adapter = new MycelliosPipelineAdapter({
      baseUrl,
      model: "test-model",
      modelDigest: MODEL_DIGEST,
      activationId: ACTIVATION_ID,
    });

    expect(await adapter.probe()).toMatchObject({
      kind: "mycellios-pipeline",
      models: ["test-model"],
    });
    const chunks: Array<{
      text: string;
      metrics?: { reusedKvTokens?: number | undefined };
    }> = [];
    for await (const chunk of adapter.generate(
      {
        jobId: "job",
        request: {
          model: "test-model",
          messages: [{ role: "user", content: "hola" }],
          session_id: "chat-stable",
        },
      },
      new AbortController().signal,
    )) {
      chunks.push(chunk);
    }
    expect(chunks.map((chunk) => chunk.text).join("")).toBe("hola mundo");
    expect(receivedSessionId).toBe("chat-stable");
    expect(chunks.at(-1)).toMatchObject({
      text: "",
      metrics: { reusedKvTokens: 9 },
    });
  });

  it("rejects every external, credentialed or path-prefixed endpoint", () => {
    for (const endpoint of [
      "https://inference.example",
      "http://192.168.1.10:8080",
      "http://user:secret@127.0.0.1:8080",
      "http://127.0.0.1:8080/provider",
      "http://127.0.0.1:8080/?target=external",
    ]) {
      expect(
        () => normalizeMycelliosPipelineBaseUrl(endpoint),
        endpoint,
      ).toThrow(/loopback origins/);
    }
    expect(normalizeMycelliosPipelineBaseUrl("http://127.0.0.1:8080").origin)
      .toBe("http://127.0.0.1:8080");
  });

  it("fails closed when loopback health is not the sealed native artifact", async () => {
    const baseUrl = await listen((_request, response) => {
      response.setHeader("content-type", "application/json");
      response.end(JSON.stringify({
        status: "ready",
        model: "test-model",
        artifact_identity: "sha256:other",
        pipeline_snapshot_identity: ACTIVATION_ID,
        stages: 1,
        boundaries: [0, 28],
      }));
    });
    const adapter = new MycelliosPipelineAdapter({
      baseUrl,
      model: "test-model",
      modelDigest: MODEL_DIGEST,
      activationId: ACTIVATION_ID,
    });
    await expect(adapter.probe()).rejects.toMatchObject({
      code: "mycellios_pipeline_identity_mismatch",
      retryable: false,
    });
  });

  it("fails closed when health reports another pipeline activation", async () => {
    const baseUrl = await listen((_request, response) => {
      response.setHeader("content-type", "application/json");
      response.end(JSON.stringify({
        status: "ready",
        model: "test-model",
        artifact_identity: MODEL_DIGEST,
        pipeline_snapshot_identity: "another-activation",
        stages: 1,
        boundaries: [0, 28],
      }));
    });
    const adapter = new MycelliosPipelineAdapter({
      baseUrl,
      model: "test-model",
      modelDigest: MODEL_DIGEST,
      activationId: ACTIVATION_ID,
    });
    await expect(adapter.probe()).rejects.toMatchObject({
      code: "mycellios_pipeline_identity_mismatch",
      retryable: false,
    });
  });

  async function listen(handler: RequestListener): Promise<string> {
    const server = createServer(handler);
    servers.push(server);
    await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
    const address = server.address();
    if (!address || typeof address === "string") {
      throw new Error("Expected a TCP test address");
    }
    return `http://127.0.0.1:${address.port}`;
  }
});
