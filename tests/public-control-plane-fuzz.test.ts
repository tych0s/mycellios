import { randomUUID } from "node:crypto";
import { describe, expect, it } from "vitest";
import {
  nodeCommandResultSchema,
  nodeCommandSchema,
  nodeEnrollmentBundleSchema,
  nodeEnrollmentRedeemSchema,
  nodeEventSchema,
  nodeSnapshotSchema,
} from "../src/contracts/node-control.js";
import { componentUpdateManifestSchema, parseComponentUpdateManifest } from "../src/contracts/component-update-manifest.js";
import { modelDistributionManifestSchema, parseModelDistributionManifest } from "../src/contracts/model-distribution-manifest.js";
import { NodeCommandStore } from "../src/coordinator/node-command-store.js";
import { NodeEnrollmentStore } from "../src/coordinator/node-enrollment-store.js";
import { verifyArtifactSwarmManifest } from "../src/model-fabric/artifact-swarm.js";
import { MeshDatabase } from "../src/storage/database.js";
import { computeFilesManifestSha256, verifyComponentFilesPackage } from "../src/update/component-files.js";

const actor = { kind: "account" as const, id: "account-fuzz", scopes: ["node:identity" as const] };
const digest = (char: string) => `sha256:${char.repeat(64)}` as const;

describe("public control-plane deterministic stateful fuzzing", () => {
  it("keeps public JSON parsers total for seeded arbitrary values", () => {
    const random = seededRandom(0x503544);
    const schemas = [
      nodeEnrollmentBundleSchema,
      nodeEnrollmentRedeemSchema,
      nodeCommandSchema,
      nodeCommandResultSchema,
      nodeSnapshotSchema,
      nodeEventSchema,
      componentUpdateManifestSchema,
      modelDistributionManifestSchema,
    ];
    for (let index = 0; index < 3_000; index += 1) {
      const candidate = arbitraryJson(random, 0);
      for (const schema of schemas) expect(() => schema.safeParse(candidate)).not.toThrow();
      expect(() => attempt(() => parseComponentUpdateManifest(candidate))).not.toThrow();
      expect(() => attempt(() => parseModelDistributionManifest(candidate))).not.toThrow();
      expect(() => attempt(() => verifyArtifactSwarmManifest(candidate as never))).not.toThrow();
    }
    expect(Object.prototype).not.toHaveProperty("polluted");
  });

  it("bounds malformed archive parsing and rejects traversal-shaped paths", () => {
    const random = seededRandom(0x61726368);
    for (let index = 0; index < 1_000; index += 1) {
      const bytes = Uint8Array.from({ length: Math.floor(random() * 512) }, () => Math.floor(random() * 256));
      expect(() => attempt(() => verifyComponentFilesPackage(bytes))).not.toThrow();
    }
    for (const path of ["../escape", "a/../../escape", "/absolute", "C:/windows/system32", "a\\..\\escape", "nul", "a\0b"]) {
      expect(() => computeFilesManifestSha256([{ path, mode: "0644", bytes: 1, sha256: digest("a") }])).toThrow();
    }
  });

  it("never accepts enrollment or command replay across a randomized state sequence", () => {
    const database = new MeshDatabase(":memory:");
    const now = Date.parse("2026-08-10T12:00:01.000Z");
    const enrollments = new NodeEnrollmentStore(database, () => now);
    const commands = new NodeCommandStore(database, () => now);
    const random = seededRandom(0x73746174);

    for (let index = 0; index < 64; index += 1) {
      const issued = enrollments.issue({ accountId: actor.id, actor, expiresInSeconds: 120 });
      if (random() < 0.8) enrollments.confirm({ enrollmentId: issued.enrollmentId, accountId: actor.id, actorId: actor.id });
      const input = {
        enrollmentToken: issued.enrollmentToken,
        nonce: issued.nonce,
        identityKind: "device" as const,
        identityId: `node-${index}`,
        publicKeyFingerprint: digest(index % 2 === 0 ? "a" : "b"),
      };
      const first = enrollments.consume(input);
      const replay = enrollments.consume(input);
      if (first.state === "consumed") expect(replay.state).toBe("already-consumed");
      else expect(["unconfirmed", "expired", "nonce-mismatch", "credential-conflict"]).toContain(first.state);

      const command = commandFixture(index);
      expect(commands.enqueue(command, 3).state).toBe("queued");
      expect(commands.enqueue(command, 3).state).toBe("queued");
      expect(() => commands.enqueue({ ...commandFixture(index + 10_000), nonce: command.nonce }, 3))
        .toThrow("node_command_replay");
    }

    const events = commands.eventsAfter("node-fuzz", null);
    for (const event of events.slice(0, 32)) {
      const forged = `${event.cursor.slice(0, -1)}${event.cursor.endsWith("0") ? "1" : "0"}`;
      expect(() => commands.eventsAfter("node-fuzz", forged)).toThrow(/node_event_cursor/);
    }
    database.close();
  });

  it("treats payload-shaped code as data and rejects unknown command fields", () => {
    const marker = { executed: false };
    const candidate = {
      ...commandFixture(999),
      payload: { version: 1, command: "touch /tmp/mycellios-fuzz-pwned", run: () => { marker.executed = true; } },
    };
    expect(nodeCommandSchema.safeParse(candidate).success).toBe(false);
    expect(marker.executed).toBe(false);
  });
});

function commandFixture(index: number) {
  return {
    schema: "mycellios-node-command/1" as const,
    id: randomUUID(),
    nodeId: "node-fuzz",
    actor: { kind: "account" as const, id: "account-fuzz", scopes: ["node:control" as const] },
    generation: 3,
    issuedAt: "2026-08-10T12:00:00.000Z",
    expiresAt: "2026-08-10T12:05:00.000Z",
    nonce: `nonce_${index.toString(16).padStart(24, "0")}`,
    type: "pause" as const,
    payload: { version: 1 as const },
  };
}

function attempt(operation: () => unknown): void {
  try { operation(); } catch { /* rejection is the expected fail-closed outcome */ }
}

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
  if (depth >= 4 || random() < 0.5) {
    const scalars: unknown[] = [null, true, false, "", randomText(random, Math.floor(random() * 384)), Math.floor((random() - 0.5) * 2e9)];
    return scalars[Math.floor(random() * scalars.length)];
  }
  if (random() < 0.5) return Array.from({ length: Math.floor(random() * 7) }, () => arbitraryJson(random, depth + 1));
  const value: Record<string, unknown> = {};
  for (let index = 0; index < Math.floor(random() * 7); index += 1) {
    value[randomText(random, 1 + Math.floor(random() * 24))] = arbitraryJson(random, depth + 1);
  }
  return value;
}

function randomText(random: () => number, length: number): string {
  const alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-/\\:{}[]\0";
  let result = "";
  for (let index = 0; index < length; index += 1) result += alphabet[Math.floor(random() * alphabet.length)];
  return result;
}
