import { describe, expect, it } from "vitest";
import {
  parseWorkerEnvelope,
  workerEnvelopeSchema,
  workerEnvelopeValidationIssues,
} from "../src/contracts/worker-protocol.js";

describe("worker protocol deterministic abuse fuzzing", () => {
  it("fails closed without throwing for seeded arbitrary JSON values", () => {
    const random = seededRandom(0x6d796365);
    for (let index = 0; index < 5_000; index += 1) {
      const candidate = arbitraryJson(random, 0);
      let parsed: ReturnType<typeof parseWorkerEnvelope>;
      expect(() => {
        parsed = parseWorkerEnvelope(candidate);
      }).not.toThrow();
      if (parsed! !== null) {
        expect(workerEnvelopeSchema.safeParse(candidate).success).toBe(true);
      }
      expect(() => workerEnvelopeValidationIssues(candidate)).not.toThrow();
    }
  });

  it.each([
    null,
    [],
    {},
    { v: 1, workerId: "worker", type: "worker.hello", payload: {}, extra: true },
    { v: 2, workerId: "worker", type: "worker.hello", payload: {} },
    { v: 1, workerId: "", type: "worker.hello", payload: {} },
    { v: 1, workerId: "x".repeat(257), type: "worker.hello", payload: {} },
    { v: 1, workerId: "worker", type: "worker.hello", payload: { extra: true } },
    { v: 1, workerId: "worker", type: "__proto__", payload: {} },
    {
      v: 1,
      workerId: "worker",
      type: "runtime.stream.data",
      payload: {
        streamId: "stream",
        generation: 0,
        recoveryToken: "a".repeat(16),
        offset: 0,
        data: "A".repeat(70_000),
      },
    },
  ])("rejects directed malformed or oversized envelope %#", (candidate) => {
    expect(parseWorkerEnvelope(candidate)).toBeNull();
  });

  it("normalizes prototype-shaped JSON without polluting shared objects", () => {
    const candidate = JSON.parse(
      '{"v":1,"workerId":"worker","type":"worker.hello","payload":{},"__proto__":{"polluted":true}}',
    ) as unknown;
    expect(parseWorkerEnvelope(candidate)).toEqual({
      v: 1,
      workerId: "worker",
      type: "worker.hello",
      payload: {},
    });
    expect(Object.prototype).not.toHaveProperty("polluted");
  });
});

function seededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return (state >>> 0) / 0x1_0000_0000;
  };
}

function arbitraryJson(random: () => number, depth: number): unknown {
  const scalar = (): unknown => {
    switch (Math.floor(random() * 6)) {
      case 0: return null;
      case 1: return random() < 0.5;
      case 2: return Math.floor((random() - 0.5) * 2_000_000);
      case 3: return Number.NaN;
      case 4: return randomText(random, Math.floor(random() * 320));
      default: return "";
    }
  };
  if (depth >= 4 || random() < 0.48) return scalar();
  if (random() < 0.5) {
    return Array.from(
      { length: Math.floor(random() * 6) },
      () => arbitraryJson(random, depth + 1),
    );
  }
  const value: Record<string, unknown> = {};
  const keys = Math.floor(random() * 6);
  for (let index = 0; index < keys; index += 1) {
    value[randomText(random, 1 + Math.floor(random() * 18))] = arbitraryJson(random, depth + 1);
  }
  return value;
}

function randomText(random: () => number, length: number): string {
  const alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-€{}[]";
  let result = "";
  for (let index = 0; index < length; index += 1) {
    result += alphabet[Math.floor(random() * alphabet.length)];
  }
  return result;
}
