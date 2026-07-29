import { z } from "zod";

const SENSITIVE_DIAGNOSTIC_KEY =
  /(authorization|cookie|credential|password|secret|token|api[-_]?key)/i;

export const REMOTE_DIAGNOSTIC_LEVELS = ["info", "warning", "error"] as const;
export const REMOTE_DIAGNOSTIC_SOURCES = [
  "desktop",
  "coordinator",
  "worker",
  "runtime",
  "renderer",
  "network",
] as const;

export const remoteDiagnosticEventSchema = z.object({
  id: z.string().regex(/^diag_[a-f0-9]{64}$/),
  sourceId: z.string().uuid(),
  appVersion: z.string().min(1).max(64),
  platform: z.string().min(1).max(32),
  arch: z.string().min(1).max(32),
  level: z.enum(REMOTE_DIAGNOSTIC_LEVELS),
  source: z.enum(REMOTE_DIAGNOSTIC_SOURCES),
  event: z.string().regex(/^[a-zA-Z0-9._:-]+$/).max(120),
  message: z.string().min(1).max(1_200),
  details: z.string().max(2_000).optional(),
  occurredAt: z.string().datetime(),
}).strict();

export const remoteDiagnosticBatchSchema = z.object({
  events: z.array(remoteDiagnosticEventSchema).min(1).max(100),
}).strict();

export const remoteDiagnosticQuerySchema = z.object({
  limit: z.coerce.number().int().min(1).max(500).default(100),
  level: z.enum(REMOTE_DIAGNOSTIC_LEVELS).optional(),
  sourceId: z.string().uuid().optional(),
  since: z.string().datetime().optional(),
}).strict();

export type RemoteDiagnosticEvent = z.infer<typeof remoteDiagnosticEventSchema>;
export type RemoteDiagnosticBatch = z.infer<typeof remoteDiagnosticBatchSchema>;

export function redactDiagnosticDetails(details: string): string {
  try {
    return JSON.stringify(redactDiagnosticValue(JSON.parse(details) as unknown));
  } catch {
    return redactDiagnosticText(details);
  }
}

export function redactDiagnosticText(value: string): string {
  return value
    .replace(/(Bearer\s+)[A-Za-z0-9._~+/=-]+/gi, "$1[REDACTED]")
    .replace(
      /\b(authorization|cookie|credential|password|secret|token|api[-_ ]?key)\s*[:=]\s*["']?[^"',}\s]+/gi,
      "$1=[REDACTED]",
    )
    .replace(
      /([?&](?:authorization|credential|password|secret|token|api[-_]?key)=)[^&\s]+/gi,
      "$1[REDACTED]",
    );
}

function redactDiagnosticValue(value: unknown, key = "", depth = 0): unknown {
  if (SENSITIVE_DIAGNOSTIC_KEY.test(key)) return "[REDACTED]";
  if (typeof value === "string") return redactDiagnosticText(value).slice(0, 1_200);
  if (value === null || typeof value !== "object") return value;
  if (depth >= 4) return "[TRUNCATED]";
  if (Array.isArray(value)) {
    return value.slice(0, 24).map((item) => redactDiagnosticValue(item, "", depth + 1));
  }
  return Object.fromEntries(
    Object.entries(value as Record<string, unknown>)
      .slice(0, 32)
      .map(([entryKey, entryValue]) => [
        entryKey,
        redactDiagnosticValue(entryValue, entryKey, depth + 1),
      ]),
  );
}
