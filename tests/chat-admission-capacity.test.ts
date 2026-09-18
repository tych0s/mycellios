import { IncomingMessage, ServerResponse } from "node:http";
import { Socket } from "node:net";
import { afterEach, describe, expect, it, vi } from "vitest";
import { waitForChatCapacity } from "../src/coordinator/chat-admission.js";
import type { MeshService } from "../src/coordinator/mesh-service.js";

afterEach(() => vi.useRealTimers());
const request = { model: "model", messages: [{ role: "user" as const, content: "hello" }] };

describe("chat admission capacity waiting", () => {
  it("releases its deadline immediately when the client disconnects", async () => {
    vi.useFakeTimers();
    const response = new ServerResponse(new IncomingMessage(new Socket()));
    const service = { hasCapacity: vi.fn(() => false) } as unknown as MeshService;
    const waiting = waitForChatCapacity(service, request, "session", 45_000, response);
    const rejected = expect(waiting).rejects.toMatchObject({ code: "client_disconnected" });
    response.emit("close");
    await rejected;
    expect(vi.getTimerCount()).toBe(0);
    expect(response.listenerCount("close")).toBe(0);
    expect(service.hasCapacity).toHaveBeenCalledTimes(1);
  });

  it("does not probe capacity when the response is already destroyed", async () => {
    const response = new ServerResponse(new IncomingMessage(new Socket()));
    response.destroy();
    const service = { hasCapacity: vi.fn(() => true) } as unknown as MeshService;
    await expect(waitForChatCapacity(service, request, "session", 45_000, response))
      .rejects.toMatchObject({ code: "client_disconnected" });
    expect(service.hasCapacity).not.toHaveBeenCalled();
    expect(response.listenerCount("close")).toBe(0);
  });

  it("removes its close listener when capacity returns", async () => {
    vi.useFakeTimers();
    const response = new ServerResponse(new IncomingMessage(new Socket()));
    const service = { hasCapacity: vi.fn().mockReturnValueOnce(false).mockReturnValue(true) } as unknown as MeshService;
    const waiting = waitForChatCapacity(service, request, "session", 45_000, response);
    await vi.advanceTimersByTimeAsync(250);
    await waiting;
    expect(response.listenerCount("close")).toBe(0);
    expect(vi.getTimerCount()).toBe(0);
  });
});
