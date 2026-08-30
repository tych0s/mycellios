import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { createHash } from "node:crypto";
import type { ChatCompletionRequest } from "../src/contracts/types.js";
import {
  API_KEY_PREFIX,
  ApiAccessError,
  ApiAccessManager,
} from "../src/coordinator/api-access.js";
import { MeshDatabase } from "../src/storage/database.js";

const request: ChatCompletionRequest = {
  model: "real-model",
  messages: [{ role: "user", content: "hola" }],
  max_tokens: 10,
  stream: false,
};

describe("API access accounting", () => {
  let database: MeshDatabase;
  let access: ApiAccessManager;

  beforeEach(() => {
    database = new MeshDatabase(":memory:");
    access = new ApiAccessManager(database, {
      starterTokens: 100,
      requestsPerMinute: 2,
      maxConcurrent: 1,
      maxActiveKeys: 2,
    });
  });

  afterEach(() => database.close());

  it("creates hashed, revocable keys and only returns the full secret once", () => {
    const created = access.createKey("user-1", "Production");
    expect(created.secret).toMatch(/^myc_live_[A-Za-z0-9_-]{10}_[A-Za-z0-9_-]{43}$/);
    expect(created.prefix).toHaveLength(10);
    expect(access.listKeys("user-1")).toEqual([
      expect.objectContaining({ id: created.id, name: "Production" }),
    ]);
    expect(JSON.stringify(access.listKeys("user-1"))).not.toContain(created.secret);
    const stored = database.raw.prepare("SELECT secret_hash FROM api_keys WHERE id = ?")
      .get(created.id) as { secret_hash: string };
    expect(stored.secret_hash).toMatch(/^scrypt\$[a-f0-9]{32}\$[a-f0-9]{64}$/);
    expect(access.authenticateKey(created.secret)).toEqual({
      kind: "api_key",
      userId: "user-1",
      apiKeyId: created.id,
    });
    expect(access.revokeKey("user-1", created.id)).toBe(true);
    expect(access.authenticateKey(created.secret)).toBeNull();
    expect(created.secret.startsWith(API_KEY_PREFIX)).toBe(true);
  });

  it("upgrades a valid legacy API-key hash after authentication", () => {
    const created = access.createKey("user-1", "Legacy");
    const legacyHash = createHash("sha256").update(created.secret, "utf8").digest("hex");
    database.raw.prepare("UPDATE api_keys SET secret_hash = ? WHERE id = ?")
      .run(legacyHash, created.id);

    expect(access.authenticateKey(created.secret)).toMatchObject({ apiKeyId: created.id });
    const migrated = database.raw.prepare("SELECT secret_hash FROM api_keys WHERE id = ?")
      .get(created.id) as { secret_hash: string };
    expect(migrated.secret_hash).toMatch(/^scrypt\$/);
    expect(migrated.secret_hash).not.toBe(legacyHash);
  });

  it("reserves the maximum cost before dispatch and reconciles the real token count", () => {
    const reservation = access.beginUsage("user-1", null, request);
    expect(reservation.reservedTokens).toBe(12);
    expect(reservation.remainingTokens).toBe(88);
    access.attachJob(reservation.id, "job-1", "session-1");

    const account = access.completeUsage(reservation.id, 4, 6);
    expect(account).toEqual(expect.objectContaining({
      tokenBalance: 90,
      lifetimeInputTokens: 4,
      lifetimeOutputTokens: 6,
      requestCount: 1,
    }));
    expect(access.userOwnsJob("user-1", "job-1")).toBe(true);
    expect(access.userOwnsSession("user-1", "session-1")).toBe(true);
    expect(access.listUsage("user-1")[0]).toEqual(expect.objectContaining({
      status: "completed",
      inputTokens: 4,
      outputTokens: 6,
    }));
  });

  it("refunds a failed request and enforces concurrency, rate and balance limits", () => {
    const first = access.beginUsage("user-1", null, request);
    expect(() => access.beginUsage("user-1", null, request)).toThrowError(
      expect.objectContaining({ code: "concurrency_limit_exceeded", statusCode: 429 }),
    );
    access.failUsage(first.id, "worker_failed");
    expect(access.getOrCreateAccount("user-1").tokenBalance).toBe(100);

    const second = access.beginUsage("user-1", null, request);
    access.completeUsage(second.id, 2, 2);
    expect(() => access.beginUsage("user-1", null, request)).toThrowError(
      expect.objectContaining({ code: "rate_limit_exceeded", statusCode: 429 }),
    );

    access.getOrCreateAccount("poor-user");
    database.raw.prepare(
      "UPDATE api_accounts SET token_balance = 1 WHERE user_id = 'poor-user'",
    ).run();
    expect(() => access.beginUsage("poor-user", null, request)).toThrowError(
      expect.objectContaining({
        code: "insufficient_token_balance",
        statusCode: 402,
      } satisfies Partial<ApiAccessError>),
    );
  });
});
