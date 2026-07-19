import { createServer, type RequestListener, type Server } from "node:http";
import { afterEach, describe, expect, it } from "vitest";
import { local model runtimeAdapter } from "../src/adapters/local-model-runtime.js";
import {
  normalizeAndValidateBaseUrl,
  OpenAICompatibleAdapter,
} from "../src/adapters/openai-compatible.js";

describe("inference adapters", () => {
  const servers: Server[] = [];

  afterEach(async () => {
    await Promise.all(servers.splice(0).map((server) => new Promise<void>((resolve) => server.close(() => resolve()))));
  });

  it("normalizes an OpenAI-compatible SSE stream", async () => {
    const baseUrl = await listen((request, response) => {
      if (request.url === "/v1/models") {
        response.setHeader("content-type", "application/json");
        response.end(JSON.stringify({ data: [{ id: "test-model" }] }));
        return;
      }
      if (request.url === "/v1/chat/completions") {
        response.setHeader("content-type", "text/event-stream");
        response.write('data: {"choices":[{"delta":{"content":"hola "}}]}\n\n');
        response.write('data: {"choices":[{"delta":{"content":"mundo"}}]}\n\n');
        response.end("data: [DONE]\n\n");
      }
    });
    const adapter = new OpenAICompatibleAdapter({
      baseUrl,
      model: "test-model",
      kind: "externalggufruntime",
    });
    expect((await adapter.probe()).models).toContain("test-model");
    const chunks = [];
    for await (const chunk of adapter.generate(
      {
        jobId: "job",
        request: { model: "test-model", messages: [{ role: "user", content: "hola" }] },
      },
      new AbortController().signal,
    )) {
      chunks.push(chunk.text);
    }
    expect(chunks.join("")).toBe("hola mundo");
  });

  it("normalizes local model runtime NDJSON", async () => {
    const baseUrl = await listen((request, response) => {
      if (request.url === "/api/tags") {
        response.setHeader("content-type", "application/json");
        response.end(JSON.stringify({ models: [{ name: "qwen:test" }] }));
        return;
      }
      if (request.url === "/api/chat") {
        response.setHeader("content-type", "application/x-ndjson");
        response.write('{"message":{"content":"uno "}}\n');
        response.end('{"message":{"content":"dos"},"done":true}\n');
      }
    });
    const adapter = new local model runtimeAdapter({ baseUrl, model: "qwen:test" });
    expect((await adapter.probe()).models).toEqual(["qwen:test"]);
    const chunks = [];
    for await (const chunk of adapter.generate(
      {
        jobId: "job",
        request: { model: "qwen:test", messages: [{ role: "user", content: "hola" }] },
      },
      new AbortController().signal,
    )) {
      chunks.push(chunk.text);
    }
    expect(chunks.join("")).toBe("uno dos");
  });

  it("blocks non-loopback backends unless the host is explicitly allowed", () => {
    expect(() => normalizeAndValidateBaseUrl("http://169.254.169.254", [])).toThrow(
      /not in the explicit allowlist/,
    );
    expect(normalizeAndValidateBaseUrl("https://inference.example", ["inference.example"]).hostname).toBe(
      "inference.example",
    );
  });

  it("requires HTTPS even for explicitly allowed remote inference targets", () => {
    expect(() =>
      normalizeAndValidateBaseUrl("http://inference.example", ["inference.example"]),
    ).toThrow(/must use HTTPS/);
    expect(
      normalizeAndValidateBaseUrl("https://inference.example", ["inference.example"]).protocol,
    ).toBe("https:");
  });

  it("supports provider base URLs that already include their API version", async () => {
    const adapter = new OpenAICompatibleAdapter({
      baseUrl: "https://api.z.ai/api/paas/v4/",
      apiPathPrefix: "",
      model: "glm-4.5-air",
      allowedHosts: ["api.z.ai"],
    });
    expect(adapter.kind).toBe("openai-compatible");
  });

  async function listen(handler: RequestListener): Promise<string> {
    const server = createServer(handler);
    servers.push(server);
    await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
    const address = server.address();
    if (!address || typeof address === "string") throw new Error("Expected a TCP test address");
    return `http://127.0.0.1:${address.port}`;
  }
});
