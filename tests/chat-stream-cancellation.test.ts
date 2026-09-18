import { describe, expect, it, vi } from "vitest";
import { consumeChatCompletionStream, consumeChatCompletionStreamWithRecovery } from "../src/core/chat-stream.js";

const encoder = new TextEncoder();
const headers = { "content-type": "text/event-stream" };

describe("chat cancellation and transport cleanup", () => {
  it("parses CRLF event boundaries and UTF-8 characters split across network chunks", async () => {
    const bytes = encoder.encode('data: {"choices":[{"delta":{"content":"¡Hola!"}}]}\r\n\r\ndata: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"completion_tokens":1}}\r\n\r\ndata: [DONE]\r\n\r\n');
    const body = new ReadableStream<Uint8Array>({ start(stream) { for (const byte of bytes) stream.enqueue(new Uint8Array([byte])); stream.close(); } });
    await expect(consumeChatCompletionStream(new Response(body, { headers }), "model", () => undefined)).resolves.toMatchObject({ text: "¡Hola!", outputTokens: 1 });
  });

  it("does not open a connection when already cancelled", async () => {
    const controller = new AbortController();
    controller.abort();
    const open = vi.fn();
    await expect(consumeChatCompletionStreamWithRecovery(open, "model", () => undefined, { sessionId: "s", signal: controller.signal })).rejects.toMatchObject({ name: "AbortError" });
    expect(open).not.toHaveBeenCalled();
  });

  it("aborts an outstanding connection without treating Stop as a network retry", async () => {
    const controller = new AbortController();
    const open = vi.fn(async (_attempt: number, signal: AbortSignal) => new Promise<Response>((_resolve, reject) => {
      signal.addEventListener("abort", () => reject(signal.reason), { once: true });
      controller.abort();
    }));
    await expect(consumeChatCompletionStreamWithRecovery(open, "model", () => undefined, { sessionId: "s", signal: controller.signal })).rejects.toMatchObject({ name: "AbortError" });
    expect(open).toHaveBeenCalledTimes(1);
  });

  it("cancels a stalled reader immediately and releases its lock", async () => {
    const controller = new AbortController();
    const cancel = vi.fn();
    let opened!: () => void;
    const ready = new Promise<void>((resolve) => { opened = resolve; });
    const body = new ReadableStream<Uint8Array>({ pull() { opened(); }, cancel });
    const result = consumeChatCompletionStreamWithRecovery(async () => new Response(body, { headers }), "model", () => undefined, { sessionId: "s", signal: controller.signal });
    const assertion = expect(result).rejects.toMatchObject({ name: "AbortError" });
    await ready;
    controller.abort();
    await assertion;
    expect(cancel).toHaveBeenCalledOnce();
    expect(body.locked).toBe(false);
  });

  it("does not emit buffered tokens after the user cancels", async () => {
    const controller = new AbortController();
    const updates: string[] = [];
    const cancel = vi.fn();
    const body = new ReadableStream<Uint8Array>({
      start(stream) {
        stream.enqueue(encoder.encode('data: {"choices":[{"delta":{"content":"first"}}]}\n\ndata: {"choices":[{"delta":{"content":"second"}}]}\n\n'));
      }, cancel,
    });
    await expect(consumeChatCompletionStream(new Response(body, { headers }), "model", (update) => { updates.push(update.text); controller.abort(); }, Date.now(), 25_000, undefined, controller.signal)).rejects.toMatchObject({ name: "AbortError" });
    expect(updates).toEqual(["first"]);
    expect(cancel).toHaveBeenCalledOnce();
    expect(body.locked).toBe(false);
  });

  it("interrupts retry backoff without opening a new request", async () => {
    const controller = new AbortController();
    const open = vi.fn(async () => { throw new TypeError("Network lost"); });
    let recovering!: () => void;
    const ready = new Promise<void>((resolve) => { recovering = resolve; });
    const result = consumeChatCompletionStreamWithRecovery(open, "model", (update) => { if (update.phase === "recovering") recovering(); }, { sessionId: "s", signal: controller.signal, retryDelayMs: 8_000 });
    const assertion = expect(result).rejects.toMatchObject({ name: "AbortError" });
    await ready;
    controller.abort();
    await assertion;
    expect(open).toHaveBeenCalledOnce();
  });

  it("closes malformed streams instead of leaving a worker generating", async () => {
    const cancel = vi.fn();
    const body = new ReadableStream<Uint8Array>({ start(stream) { stream.enqueue(encoder.encode("data: {broken\n\n")); }, cancel });
    await expect(consumeChatCompletionStream(new Response(body, { headers }), "model", () => undefined)).rejects.toThrow("fragmento");
    expect(cancel).toHaveBeenCalledOnce();
    expect(body.locked).toBe(false);
  });
});
