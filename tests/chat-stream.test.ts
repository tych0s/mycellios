import { describe, expect, it } from "vitest";
import {
  consumeChatCompletionStream,
  consumeChatCompletionStreamWithRecovery,
} from "../src/core/chat-stream.js";
import type { ChatStreamUpdate } from "../src/contracts/control-api.js";

describe("chat completion stream", () => {
  it("publishes every token before returning the final metrics", async () => {
    const source = [
      'data: {"id":"job_live","model":"qwen","choices":[{"delta":{"role":"assistant"}}],"x_network":{"session_id":"chat-stable","route_class":"replica","affinity_hit":true}}\n\n',
      'data: {"id":"job_live","model":"qwen","choices":[{"delta":{"content":"hola "}}],"x_network":{"token_index":0}}\n\n',
      'data: {"id":"job_live","model":"qwen","choices":[{"delta":{"content":"mundo"}}],"x_network":{"token_index":1}}\n\n',
      `data: {"id":"job_live","model":"qwen","choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":2,"total_tokens":6},"x_network":{"ttft_ms":120,"active_ms":500,"reused_kv_tokens":3,"execution_receipt_id":"sha256:${"a".repeat(64)}","trust_policy":"trusted-only","boundary_policy":"pinned-edges","pinned_identity_count":2}}\n\n`,
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
      executionReceiptId: `sha256:${"a".repeat(64)}`,
      trustPolicy: "trusted-only",
      boundaryPolicy: "pinned-edges",
      pinnedIdentityCount: 2,
    });
    expect(updates.at(-1)).toMatchObject({
      trustPolicy: "trusted-only",
      boundaryPolicy: "pinned-edges",
      pinnedIdentityCount: 2,
    });
  });

  it("preserves an error emitted after the stream opens", async () => {
    const response = streamingResponse([
      'data: {"error":{"code":"worker_failed","message":"El worker perdió el modelo"}}\n\n',
      "data: [DONE]\n\n",
    ]);
    await expect(consumeChatCompletionStream(response, "qwen", () => undefined)).rejects.toThrow("El worker perdió el modelo");
  });

  it("preserves the declared recovery mode through progress and completion", async () => {
    const response = streamingResponse([
      'data: {"id":"job-recovery","model":"qwen","choices":[{"delta":{}}],"x_network":{"phase":"recovering","status_message":"Replaying prefix","attempt":2,"recovery_mode":"deterministic-prefix-replay"}}\n\n',
      'data: {"id":"job-recovery","model":"qwen","choices":[{"delta":{"content":"ok"}}],"x_network":{"token_index":0}}\n\n',
      'data: {"id":"job-recovery","model":"qwen","choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4},"x_network":{"ttft_ms":30,"active_ms":60,"recovery_mode":"deterministic-prefix-replay","recovery_attempts":2,"replayed_token_events":4}}\n\n',
      "data: [DONE]\n\n",
    ]);
    const updates: ChatStreamUpdate[] = [];
    const result = await consumeChatCompletionStream(response, "fallback", (update) => updates.push(update));
    expect(updates).toContainEqual(expect.objectContaining({
      phase: "recovering",
      recoveryMode: "deterministic-prefix-replay",
    }));
    expect(result).toMatchObject({
      recoveryMode: "deterministic-prefix-replay",
      recoveryAttempts: 2,
      replayedTokenEvents: 4,
    });
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

  it("abandons a half-open connection and resumes after a network change", async () => {
    let attempts = 0;
    const updates: ChatStreamUpdate[] = [];
    const result = await consumeChatCompletionStreamWithRecovery(
      async (_attempt, signal) => {
        attempts += 1;
        if (attempts === 1) {
          return new Promise<Response>((_resolve, reject) => {
            signal.addEventListener("abort", () => {
              reject(new DOMException("The connection was aborted", "AbortError"));
            }, { once: true });
          });
        }
        return streamingResponse([
          'data: {"id":"job-vpn","model":"qwen","choices":[{"delta":{"role":"assistant"}}],"x_network":{"session_id":"vpn-session","route_class":"replica","affinity_hit":false}}\n\n',
          'data: {"id":"job-vpn","model":"qwen","choices":[{"delta":{"content":"reconectado"}}],"x_network":{"token_index":0}}\n\n',
          'data: {"id":"job-vpn","model":"qwen","choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":1,"total_tokens":5},"x_network":{"ttft_ms":20,"active_ms":40}}\n\n',
          "data: [DONE]\n\n",
        ]);
      },
      "qwen",
      (update) => updates.push(update),
      {
        sessionId: "vpn-session",
        connectionTimeoutMs: 5,
        retryDelayMs: 0,
      },
    );

    expect(attempts).toBe(2);
    expect(updates).toContainEqual(expect.objectContaining({
      phase: "recovering",
      attempt: 2,
      maximumAttempts: 8,
    }));
    expect(result.text).toBe("reconectado");
  });

  it("keeps reconnecting through more than one transient network failure", async () => {
    let attempts = 0;
    const result = await consumeChatCompletionStreamWithRecovery(
      async () => {
        attempts += 1;
        if (attempts < 4) throw new TypeError("fetch failed");
        return streamingResponse([
          'data: {"id":"job-back","model":"qwen","choices":[{"delta":{"content":"online"}}],"x_network":{"session_id":"session","route_class":"replica","token_index":0}}\n\n',
          'data: {"id":"job-back","model":"qwen","choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2},"x_network":{"ttft_ms":10,"active_ms":20}}\n\n',
          "data: [DONE]\n\n",
        ]);
      },
      "qwen",
      () => undefined,
      { sessionId: "session", maximumAttempts: 4, retryDelayMs: 0 },
    );

    expect(attempts).toBe(4);
    expect(result.text).toBe("online");
  });

  it("keeps received text visible while replaying after a mid-response disconnect", async () => {
    let attempts = 0;
    const updates: ChatStreamUpdate[] = [];
    const result = await consumeChatCompletionStreamWithRecovery(
      async () => {
        attempts += 1;
        if (attempts === 1) {
          return streamingResponse([
            'data: {"id":"job-first","model":"qwen","choices":[{"delta":{"content":"hola "}}],"x_network":{"session_id":"session","route_class":"replica","token_index":0}}\n\n',
            'data: {"error":{"code":"pipeline_stage_disconnected","message":"route lost"}}\n\n',
          ]);
        }
        return streamingResponse([
          'data: {"id":"job-replayed","model":"qwen","choices":[{"delta":{"content":"hola "}}],"x_network":{"session_id":"session","route_class":"replica","token_index":0}}\n\n',
          'data: {"id":"job-replayed","model":"qwen","choices":[{"delta":{"content":"mundo"}}],"x_network":{"session_id":"session","route_class":"replica","token_index":1}}\n\n',
          'data: {"id":"job-replayed","model":"qwen","choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3},"x_network":{"ttft_ms":10,"active_ms":30}}\n\n',
          "data: [DONE]\n\n",
        ]);
      },
      "qwen",
      (update) => updates.push(update),
      { sessionId: "session", retryDelayMs: 0 },
    );

    const recoveryUpdates = updates.filter((update) => update.phase === "recovering");
    expect(recoveryUpdates.some((update) => update.text === "hola ")).toBe(true);
    expect(result.text).toBe("hola mundo");
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
