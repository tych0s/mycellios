import { createHash, randomBytes, timingSafeEqual } from "node:crypto";
import type { ChatCompletionRequest } from "../contracts/types.js";
import { newId } from "../core/ids.js";
import { estimateInputTokens } from "../core/request.js";
import type { MeshDatabase } from "../storage/database.js";

export const API_KEY_PREFIX = "myc_live_";

export interface ApiAccessLimits {
  starterTokens: number;
  requestsPerMinute: number;
  maxConcurrent: number;
  maxActiveKeys: number;
}

export interface ApiAccount {
  userId: string;
  tokenBalance: number;
  usdMicros: number;
  lifetimeInputTokens: number;
  lifetimeOutputTokens: number;
  requestCount: number;
  createdAt: number;
  updatedAt: number;
}

export interface ApiKeyRecord {
  id: string;
  userId: string;
  name: string;
  prefix: string;
  createdAt: number;
  lastUsedAt: number | null;
  revokedAt: number | null;
}

export interface ApiKeyCreation extends ApiKeyRecord {
  secret: string;
}

export interface ApiKeyPrincipal {
  kind: "api_key";
  userId: string;
  apiKeyId: string;
}

export interface ApiUsageReservation {
  id: string;
  userId: string;
  reservedTokens: number;
  remainingTokens: number;
  rateLimit: {
    limit: number;
    remaining: number;
    resetAt: number;
  };
}

export interface ApiUsageRecord {
  id: string;
  apiKeyId: string | null;
  jobId: string | null;
  sessionId: string | null;
  model: string;
  reservedTokens: number;
  inputTokens: number;
  outputTokens: number;
  status: "pending" | "completed" | "failed";
  errorCode: string | null;
  createdAt: number;
  completedAt: number | null;
}

interface ApiAccountRow {
  user_id: string;
  token_balance: number;
  usd_micros: number;
  lifetime_input_tokens: number;
  lifetime_output_tokens: number;
  request_count: number;
  created_at: number;
  updated_at: number;
}

interface ApiKeyRow {
  id: string;
  user_id: string;
  name: string;
  prefix: string;
  secret_hash: string;
  created_at: number;
  last_used_at: number | null;
  revoked_at: number | null;
}

interface ApiUsageRow {
  id: string;
  user_id: string;
  api_key_id: string | null;
  job_id: string | null;
  session_id: string | null;
  model: string;
  reserved_tokens: number;
  input_tokens: number;
  output_tokens: number;
  status: "pending" | "completed" | "failed";
  error_code: string | null;
  created_at: number;
  completed_at: number | null;
}

export class ApiAccessError extends Error {
  constructor(
    readonly code: string,
    message: string,
    readonly statusCode: number,
    readonly retryAfterSeconds?: number,
  ) {
    super(message);
    this.name = "ApiAccessError";
  }
}

export class ApiAccessManager {
  constructor(
    private readonly database: MeshDatabase,
    readonly limits: ApiAccessLimits,
  ) {}

  getOrCreateAccount(userId: string): ApiAccount {
    return this.database.transaction(() => {
      const existing = this.accountRow(userId);
      if (existing) return accountFromRow(existing);
      const now = Date.now();
      this.database.raw.prepare(
        `INSERT INTO api_accounts(
           user_id, token_balance, usd_micros, lifetime_input_tokens,
           lifetime_output_tokens, request_count, created_at, updated_at
         ) VALUES (?, ?, 0, 0, 0, 0, ?, ?)`,
      ).run(userId, this.limits.starterTokens, now, now);
      return accountFromRow(this.accountRow(userId)!);
    });
  }

  listKeys(userId: string): ApiKeyRecord[] {
    this.getOrCreateAccount(userId);
    const rows = this.database.raw.prepare(
      `SELECT id, user_id, name, prefix, secret_hash, created_at, last_used_at, revoked_at
       FROM api_keys
       WHERE user_id = ?
       ORDER BY created_at DESC, id DESC`,
    ).all(userId) as unknown as ApiKeyRow[];
    return rows.map(keyFromRow);
  }

  createKey(userId: string, name: string): ApiKeyCreation {
    return this.database.transaction(() => {
      this.getOrCreateAccount(userId);
      const active = this.database.raw.prepare(
        "SELECT COUNT(*) AS count FROM api_keys WHERE user_id = ? AND revoked_at IS NULL",
      ).get(userId) as { count: number };
      if (Number(active.count) >= this.limits.maxActiveKeys) {
        throw new ApiAccessError(
          "api_key_limit_reached",
          `An account can have at most ${this.limits.maxActiveKeys} active API keys.`,
          409,
        );
      }
      const id = newId("key");
      const prefix = randomBytes(8).toString("base64url").slice(0, 10);
      const secret = randomBytes(32).toString("base64url");
      const token = `${API_KEY_PREFIX}${prefix}_${secret}`;
      const now = Date.now();
      this.database.raw.prepare(
        `INSERT INTO api_keys(
           id, user_id, name, prefix, secret_hash, created_at, last_used_at, revoked_at
         ) VALUES (?, ?, ?, ?, ?, ?, NULL, NULL)`,
      ).run(id, userId, name, prefix, hashApiKey(token), now);
      return {
        id,
        userId,
        name,
        prefix,
        createdAt: now,
        lastUsedAt: null,
        revokedAt: null,
        secret: token,
      };
    });
  }

  revokeKey(userId: string, keyId: string): boolean {
    const result = this.database.raw.prepare(
      `UPDATE api_keys
       SET revoked_at = ?
       WHERE id = ? AND user_id = ? AND revoked_at IS NULL`,
    ).run(Date.now(), keyId, userId);
    return Number(result.changes) === 1;
  }

  authenticateKey(token: string): ApiKeyPrincipal | null {
    const match = /^myc_live_([A-Za-z0-9_-]{10})_[A-Za-z0-9_-]{43}$/.exec(token);
    if (!match) return null;
    const row = this.database.raw.prepare(
      `SELECT id, user_id, name, prefix, secret_hash, created_at, last_used_at, revoked_at
       FROM api_keys
       WHERE prefix = ? AND revoked_at IS NULL
       LIMIT 1`,
    ).get(match[1]!) as ApiKeyRow | undefined;
    if (!row || !constantTimeHashEqual(row.secret_hash, hashApiKey(token))) return null;
    this.getOrCreateAccount(row.user_id);
    return { kind: "api_key", userId: row.user_id, apiKeyId: row.id };
  }

  beginUsage(
    userId: string,
    apiKeyId: string | null,
    request: ChatCompletionRequest,
  ): ApiUsageReservation {
    return this.database.transaction(() => {
      const account = this.getOrCreateAccount(userId);
      const now = Date.now();
      const windowStart = now - 60_000;
      const recentRows = this.database.raw.prepare(
        `SELECT created_at
         FROM api_usage
         WHERE user_id = ? AND created_at >= ?
         ORDER BY created_at ASC`,
      ).all(userId, windowStart) as unknown as Array<{ created_at: number }>;
      if (recentRows.length >= this.limits.requestsPerMinute) {
        const resetAt = Number(recentRows[0]!.created_at) + 60_000;
        throw new ApiAccessError(
          "rate_limit_exceeded",
          "The account request limit has been reached. Retry after the current minute window.",
          429,
          Math.max(1, Math.ceil((resetAt - now) / 1_000)),
        );
      }
      const pending = this.database.raw.prepare(
        "SELECT COUNT(*) AS count FROM api_usage WHERE user_id = ? AND status = 'pending'",
      ).get(userId) as { count: number };
      if (Number(pending.count) >= this.limits.maxConcurrent) {
        throw new ApiAccessError(
          "concurrency_limit_exceeded",
          `The account can run at most ${this.limits.maxConcurrent} concurrent requests.`,
          429,
          1,
        );
      }
      const reservedTokens = estimateInputTokens(request) + (request.max_tokens ?? 256);
      if (account.tokenBalance < reservedTokens) {
        throw new ApiAccessError(
          "insufficient_token_balance",
          `This request needs up to ${reservedTokens} tokens but the account has ${account.tokenBalance}.`,
          402,
        );
      }
      const id = newId("use");
      this.database.raw.prepare(
        `INSERT INTO api_usage(
           id, user_id, api_key_id, job_id, session_id, model, reserved_tokens,
           input_tokens, output_tokens, status, error_code, created_at, completed_at
         ) VALUES (?, ?, ?, NULL, ?, ?, ?, 0, 0, 'pending', NULL, ?, NULL)`,
      ).run(
        id,
        userId,
        apiKeyId,
        request.session_id ?? null,
        request.model,
        reservedTokens,
        now,
      );
      this.database.raw.prepare(
        "UPDATE api_accounts SET token_balance = token_balance - ?, updated_at = ? WHERE user_id = ?",
      ).run(reservedTokens, now, userId);
      if (apiKeyId) {
        this.database.raw.prepare(
          "UPDATE api_keys SET last_used_at = ? WHERE id = ? AND user_id = ?",
        ).run(now, apiKeyId, userId);
      }
      return {
        id,
        userId,
        reservedTokens,
        remainingTokens: account.tokenBalance - reservedTokens,
        rateLimit: {
          limit: this.limits.requestsPerMinute,
          remaining: Math.max(0, this.limits.requestsPerMinute - recentRows.length - 1),
          resetAt: recentRows.length === 0 ? now + 60_000 : Number(recentRows[0]!.created_at) + 60_000,
        },
      };
    });
  }

  attachJob(usageId: string, jobId: string, sessionId: string): void {
    this.database.raw.prepare(
      `UPDATE api_usage
       SET job_id = ?, session_id = ?
       WHERE id = ? AND status = 'pending'`,
    ).run(jobId, sessionId, usageId);
  }

  completeUsage(usageId: string, inputTokens: number, outputTokens: number): ApiAccount | null {
    return this.database.transaction(() => {
      const usage = this.usageRow(usageId);
      if (!usage || usage.status !== "pending") return null;
      const now = Date.now();
      const actualTokens = Math.max(0, inputTokens) + Math.max(0, outputTokens);
      const refund = Math.max(0, usage.reserved_tokens - actualTokens);
      const additionalCharge = Math.max(0, actualTokens - usage.reserved_tokens);
      this.database.raw.prepare(
        `UPDATE api_accounts
         SET token_balance = MAX(0, token_balance + ? - ?),
             lifetime_input_tokens = lifetime_input_tokens + ?,
             lifetime_output_tokens = lifetime_output_tokens + ?,
             request_count = request_count + 1,
             updated_at = ?
         WHERE user_id = ?`,
      ).run(refund, additionalCharge, inputTokens, outputTokens, now, usage.user_id);
      this.database.raw.prepare(
        `UPDATE api_usage
         SET input_tokens = ?, output_tokens = ?, status = 'completed',
             error_code = NULL, completed_at = ?
         WHERE id = ?`,
      ).run(inputTokens, outputTokens, now, usageId);
      return accountFromRow(this.accountRow(usage.user_id)!);
    });
  }

  failUsage(usageId: string, errorCode: string): ApiAccount | null {
    return this.database.transaction(() => {
      const usage = this.usageRow(usageId);
      if (!usage || usage.status !== "pending") return null;
      const now = Date.now();
      this.database.raw.prepare(
        `UPDATE api_accounts
         SET token_balance = token_balance + ?, updated_at = ?
         WHERE user_id = ?`,
      ).run(usage.reserved_tokens, now, usage.user_id);
      this.database.raw.prepare(
        `UPDATE api_usage
         SET status = 'failed', error_code = ?, completed_at = ?
         WHERE id = ?`,
      ).run(errorCode.slice(0, 120), now, usageId);
      return accountFromRow(this.accountRow(usage.user_id)!);
    });
  }

  listUsage(userId: string, limit = 25): ApiUsageRecord[] {
    const rows = this.database.raw.prepare(
      `SELECT id, user_id, api_key_id, job_id, session_id, model, reserved_tokens,
              input_tokens, output_tokens, status, error_code, created_at, completed_at
       FROM api_usage
       WHERE user_id = ?
       ORDER BY created_at DESC, id DESC
       LIMIT ?`,
    ).all(userId, Math.max(1, Math.min(100, limit))) as unknown as ApiUsageRow[];
    return rows.map(usageFromRow);
  }

  userOwnsJob(userId: string, jobId: string): boolean {
    const row = this.database.raw.prepare(
      "SELECT 1 AS found FROM api_usage WHERE user_id = ? AND job_id = ? LIMIT 1",
    ).get(userId, jobId) as { found: number } | undefined;
    return row?.found === 1;
  }

  userOwnsSession(userId: string, sessionId: string): boolean {
    const row = this.database.raw.prepare(
      "SELECT 1 AS found FROM api_usage WHERE user_id = ? AND session_id = ? LIMIT 1",
    ).get(userId, sessionId) as { found: number } | undefined;
    return row?.found === 1;
  }

  grantTokens(userId: string, amount: number): ApiAccount {
    return this.database.transaction(() => {
      this.getOrCreateAccount(userId);
      this.database.raw.prepare(
        `UPDATE api_accounts
         SET token_balance = token_balance + ?, updated_at = ?
         WHERE user_id = ?`,
      ).run(amount, Date.now(), userId);
      return accountFromRow(this.accountRow(userId)!);
    });
  }

  private accountRow(userId: string): ApiAccountRow | undefined {
    return this.database.raw.prepare(
      `SELECT user_id, token_balance, usd_micros, lifetime_input_tokens,
              lifetime_output_tokens, request_count, created_at, updated_at
       FROM api_accounts WHERE user_id = ?`,
    ).get(userId) as ApiAccountRow | undefined;
  }

  private usageRow(usageId: string): ApiUsageRow | undefined {
    return this.database.raw.prepare(
      `SELECT id, user_id, api_key_id, job_id, session_id, model, reserved_tokens,
              input_tokens, output_tokens, status, error_code, created_at, completed_at
       FROM api_usage WHERE id = ?`,
    ).get(usageId) as ApiUsageRow | undefined;
  }
}

export function apiAccountJson(account: ApiAccount, limits: ApiAccessLimits): Record<string, unknown> {
  return {
    user_id: account.userId,
    token_balance: account.tokenBalance,
    usd_balance: (account.usdMicros / 1_000_000).toFixed(2),
    lifetime_input_tokens: account.lifetimeInputTokens,
    lifetime_output_tokens: account.lifetimeOutputTokens,
    request_count: account.requestCount,
    limits: {
      requests_per_minute: limits.requestsPerMinute,
      max_concurrent: limits.maxConcurrent,
      max_active_keys: limits.maxActiveKeys,
    },
    created_at: new Date(account.createdAt).toISOString(),
    updated_at: new Date(account.updatedAt).toISOString(),
  };
}

export function apiKeyJson(key: ApiKeyRecord): Record<string, unknown> {
  return {
    id: key.id,
    name: key.name,
    prefix: `${API_KEY_PREFIX}${key.prefix}`,
    created_at: new Date(key.createdAt).toISOString(),
    last_used_at: key.lastUsedAt === null ? null : new Date(key.lastUsedAt).toISOString(),
    revoked_at: key.revokedAt === null ? null : new Date(key.revokedAt).toISOString(),
  };
}

export function apiUsageJson(usage: ApiUsageRecord): Record<string, unknown> {
  return {
    id: usage.id,
    api_key_id: usage.apiKeyId,
    job_id: usage.jobId,
    session_id: usage.sessionId,
    model: usage.model,
    reserved_tokens: usage.reservedTokens,
    input_tokens: usage.inputTokens,
    output_tokens: usage.outputTokens,
    total_tokens: usage.inputTokens + usage.outputTokens,
    status: usage.status,
    error_code: usage.errorCode,
    created_at: new Date(usage.createdAt).toISOString(),
    completed_at: usage.completedAt === null ? null : new Date(usage.completedAt).toISOString(),
  };
}

function hashApiKey(token: string): string {
  return createHash("sha256").update(token, "utf8").digest("hex");
}

function constantTimeHashEqual(left: string, right: string): boolean {
  if (!/^[a-f0-9]{64}$/.test(left) || !/^[a-f0-9]{64}$/.test(right)) return false;
  return timingSafeEqual(Buffer.from(left, "hex"), Buffer.from(right, "hex"));
}

function accountFromRow(row: ApiAccountRow): ApiAccount {
  return {
    userId: row.user_id,
    tokenBalance: Number(row.token_balance),
    usdMicros: Number(row.usd_micros),
    lifetimeInputTokens: Number(row.lifetime_input_tokens),
    lifetimeOutputTokens: Number(row.lifetime_output_tokens),
    requestCount: Number(row.request_count),
    createdAt: Number(row.created_at),
    updatedAt: Number(row.updated_at),
  };
}

function keyFromRow(row: ApiKeyRow): ApiKeyRecord {
  return {
    id: row.id,
    userId: row.user_id,
    name: row.name,
    prefix: row.prefix,
    createdAt: Number(row.created_at),
    lastUsedAt: row.last_used_at === null ? null : Number(row.last_used_at),
    revokedAt: row.revoked_at === null ? null : Number(row.revoked_at),
  };
}

function usageFromRow(row: ApiUsageRow): ApiUsageRecord {
  return {
    id: row.id,
    apiKeyId: row.api_key_id,
    jobId: row.job_id,
    sessionId: row.session_id,
    model: row.model,
    reservedTokens: Number(row.reserved_tokens),
    inputTokens: Number(row.input_tokens),
    outputTokens: Number(row.output_tokens),
    status: row.status,
    errorCode: row.error_code,
    createdAt: Number(row.created_at),
    completedAt: row.completed_at === null ? null : Number(row.completed_at),
  };
}
