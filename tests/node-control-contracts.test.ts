import { randomUUID } from "node:crypto";
import { describe, expect, it } from "vitest";
import {
  nodeCommandSchema,
  nodeControlOpenApiSchemas,
  nodeEnrollmentRedeemSchema,
  parseAuthorizedNodeCommand,
} from "../src/contracts/node-control.js";

const now = new Date("2026-08-09T20:00:00.000Z");

function command(overrides: Record<string, unknown> = {}) {
  return {
    schema: "mycellios-node-command/1",
    id: randomUUID(),
    nodeId: "node-1",
    actor: { kind: "account", id: "account-1", scopes: ["node:control"] },
    generation: 4,
    issuedAt: "2026-08-09T19:59:30.000Z",
    expiresAt: "2026-08-09T20:01:00.000Z",
    nonce: "abcdefghijklmnopqrstuv",
    type: "pause",
    payload: { version: 1 },
    ...overrides,
  };
}

const context = {
  now,
  minimumGeneration: 4,
  consumedNonces: new Set<string>(),
  expectedNodeId: "node-1",
};

describe("node control contracts", () => {
  it("exports strict OpenAPI-compatible JSON schemas", () => {
    expect(Object.keys(nodeControlOpenApiSchemas)).toHaveLength(8);
    expect(JSON.stringify(nodeControlOpenApiSchemas.NodeCommand)).toContain("additionalProperties");
    expect(nodeCommandSchema.safeParse({ ...command(), unexpected: true }).success).toBe(false);
  });

  it("accepts an authorized current command", () => {
    expect(parseAuthorizedNodeCommand(command(), context).type).toBe("pause");
  });

  it("requires node:limits for strict schedule and model policy changes", () => {
    const input = command({ type: "set-policy", actor: { kind: "account", id: "account-1", scopes: ["node:limits"] },
      payload: { version: 1, policy: { schedule: [{ days: [1, 2, 3, 4, 5], startMinuteUtc: 480, endMinuteUtc: 1080 }], modelAllowlist: ["org/model"] } } });
    expect(parseAuthorizedNodeCommand(input, context).type).toBe("set-policy");
    expect(() => parseAuthorizedNodeCommand({ ...input, actor: { kind: "account", id: "account-1", scopes: ["node:control"] }, nonce: "different_nonce_123456789" }, context)).toThrow("node_command_scope_denied");
  });

  it.each([
    ["replay", command(), { ...context, consumedNonces: new Set(["abcdefghijklmnopqrstuv"]) }, "node_command_replay"],
    ["expired TTL", command({ expiresAt: "2026-08-09T19:59:59.000Z" }), context, "node_command_expired"],
    ["wrong node", command({ nodeId: "node-2" }), context, "node_command_wrong_node"],
    ["generation downgrade", command({ generation: 3 }), context, "node_command_generation_downgrade"],
    ["wrong scope", command({ actor: { kind: "account", id: "account-1", scopes: ["node:read"] } }), context, "node_command_scope_denied"],
    ["wrong actor", command({ actor: { kind: "node", id: "node-1", scopes: ["node:control"] } }), context, "node_command_actor_denied"],
  ])("rejects %s", (_name, input, validation, error) => {
    expect(() => parseAuthorizedNodeCommand(input, validation)).toThrow(error);
  });

  it("rejects unknown enrollment fields and malformed proof material", () => {
    const result = nodeEnrollmentRedeemSchema.safeParse({
      schema: "mycellios-node-enrollment-redeem/1",
      enrollmentToken: "a".repeat(32),
      identity: { kind: "device", id: "node-1" },
      publicKey: { algorithm: "ed25519", spki: "x".repeat(40) },
      protocol: { min: 1, max: 1 },
      registrationDigest: `sha256:${"a".repeat(64)}`,
      nonce: "abcdefghijklmnopqrstuv",
      recoverableToken: "must-not-exist",
    });
    expect(result.success).toBe(false);
  });
});
