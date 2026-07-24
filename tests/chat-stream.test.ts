import { describe, expect, it } from "vitest";
import {
  consumeChatCompletionStream,
  consumeChatCompletionStreamWithRecovery,
} from "../src/desktop/chat-stream.js";
import type { ChatStreamUpdate } from "../src/desktop/contracts.js";

describe("chat completion stream", () => {
  it("publishes every token before returning the final metrics", async () => {
    const source = [
      'data: {"id":"job_live","model":"qwen","choices":[{"delta":{"role":"assistant"}}],"x_network":{"session_id":"chat-stable","route_class":"replica","affinity_hit":true}}\n\n',
      'data: {"id":"job_live","model":"qwen","choices":[{"delta":{"content":"hola "}}],"x_network":{"token_index":0}}\n\n',
      'data: {"id":"job_live","model":"qwen","choices":[{"delta":{"content":"mundo"}}],"x_network":{"token_index":1}}\n\n',
      'data: {"id":"job_live","model":"qwen","choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":2,"total_tokens":6},"x_network":{"ttft_ms":120,"active_ms":500,"reused_kv_tokens":3}}\n\n',
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
      sessionId: "chat-stable",
      reusedKvTokens: 3,
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

  it("retries once before token zero when a distributed stage disconnects", async () => {
    let attempts = 0;
    const updates: ChatStreamUpdate[] = [];
    const result = await consumeChatCompletionStreamWithRecovery(
      async () => {
        attempts += 1;
        if (attempts === 1) {
          return streamingResponse([
            'data: {"id":"job-1","choices":[{"delta":{"role":"assistant"}}],"x_network":{"session_id":"session","route_class":"replica","affinity_hit":false}}\n\n',
            'data: {"error":{"code":"pipeline_stage_disconnected","message":"AMD stage disconnected"}}\n\n',
          ]);
        }
        return streamingResponse([
          ": mycellios-heartbeat 1\n\n",
          'data: {"id":"job-2","model":"qwen","choices":[{"delta":{"role":"assistant"}}],"x_network":{"session_id":"session","route_class":"replica","affinity_hit":false}}\n\n',
          'data: {"id":"job-2","model":"qwen","choices":[{"delta":{"content":"recuperado"}}],"x_network":{"token_index":0}}\n\n',
          'data: {"id":"job-2","model":"qwen","choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":1,"total_tokens":5},"x_network":{"ttft_ms":80,"active_ms":120}}\n\n',
          "data: [DONE]\n\n",
        ]);
      },
      "qwen",
      (update) => updates.push(update),
      { sessionId: "session", retryDelayMs: 0 },
    );

    expect(attempts).toBe(2);
    expect(updates).toContainEqual(expect.objectContaining({
      phase: "recovering",
      attempt: 2,
      outputTokens: 0,
    }));
    expect(result.text).toBe("recuperado");
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
