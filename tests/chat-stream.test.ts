import { describe, expect, it } from "vitest";
import { consumeChatCompletionStream } from "../src/desktop/chat-stream.js";
import type { ChatStreamUpdate } from "../src/desktop/contracts.js";

describe("chat completion stream", () => {
  it("publishes every token before returning the final metrics", async () => {
    const source = [
      'data: {"id":"job_live","model":"qwen","choices":[{"delta":{"role":"assistant"}}],"x_network":{"route_class":"replica","affinity_hit":true}}\n\n',
      'data: {"id":"job_live","model":"qwen","choices":[{"delta":{"content":"hola "}}],"x_network":{"token_index":0}}\n\n',
      'data: {"id":"job_live","model":"qwen","choices":[{"delta":{"content":"mundo"}}],"x_network":{"token_index":1}}\n\n',
      'data: {"id":"job_live","model":"qwen","choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":2,"total_tokens":6},"x_network":{"ttft_ms":120,"active_ms":500}}\n\n',
      "data: [DONE]\n\n",
    ].join("");
    const response = streamingResponse([source.slice(0, 37), source.slice(37, 181), source.slice(181)]);
    const updates: ChatStreamUpdate[] = [];

    const result = await consumeChatCompletionStream(response, "fallback", (update) => updates.push(update));

    expect(updates.filter((update) => update.delta).map((update) => update.text)).toEqual(["hola ", "hola mundo"]);
    expect(updates[0]).toMatchObject({ requestId: "job_live", routeClass: "replica", affinityHit: true });
    expect(updates.at(-1)).toMatchObject({ text: "hola mundo", outputTokens: 2, ttftMs: 120 });
    expect(result).toEqual({
      requestId: "job_live",
      model: "qwen",
      text: "hola mundo",
      promptTokens: 4,
      outputTokens: 2,
      totalTokens: 6,
      routeClass: "replica",
      affinityHit: true,
      ttftMs: 120,
      activeMs: 500,
    });
  });

  it("preserves an error emitted after the stream opens", async () => {
    const response = streamingResponse([
      'data: {"error":{"code":"worker_failed","message":"El worker perdió el modelo"}}\n\n',
      "data: [DONE]\n\n",
    ]);
    await expect(consumeChatCompletionStream(response, "qwen", () => undefined)).rejects.toThrow("El worker perdió el modelo");
  });
});

function streamingResponse(chunks: string[]): Response {
  const encoder = new TextEncoder();
  return new Response(new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
      controller.close();
    },
  }), { headers: { "content-type": "text/event-stream; charset=utf-8" } });
}
