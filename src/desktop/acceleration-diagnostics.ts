import type { WorkerAcceleratorDiagnostics } from "../contracts/types.js";
import type { DesktopAccelerationStatus } from "./contracts.js";

export function sanitizeAcceleratorDiagnosticText(
  value: string | null | undefined,
): string | null {
  if (!value) return null;
  const sanitized = value
    .replace(/\bBearer\s+\S+/gi, "Bearer [redacted]")
    .replace(/[A-Za-z]:\\(?:[^\\\s"'<>|]+\\)*[^\\\s"'<>|]*/g, "[local path]")
    .replace(/\/(?:Users|home|var|tmp)\/[^\s"'<>]*/g, "[local path]")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 500);
  return sanitized || null;
}

export function buildWorkerAccelerationDiagnostics(input: {
  appVersion: string;
  acceleration: DesktopAccelerationStatus;
  retryAttempt: number;
  nextRetryAt: string | null;
  now?: string;
}): WorkerAcceleratorDiagnostics {
  const issue = input.acceleration.preparation.issue;
  return {
    schema: "mycellios-accelerator-diagnostics/1",
    appVersion: input.appVersion,
    state: input.acceleration.state,
    backend: input.acceleration.effectiveBackend,
    deviceName: sanitizeAcceleratorDiagnosticText(input.acceleration.deviceName),
    gpuVendor: sanitizeAcceleratorDiagnosticText(input.acceleration.gpu.vendor),
    gpuModel: sanitizeAcceleratorDiagnosticText(input.acceleration.gpu.model),
    phase: input.acceleration.preparation.phase,
    progressPct: input.acceleration.preparation.progressPct,
    issueCode: issue?.code ?? null,
    issueSummary: sanitizeAcceleratorDiagnosticText(issue?.message),
    retryable: issue?.retryable === true,
    retryAttempt: Math.max(0, Math.floor(input.retryAttempt)),
    nextRetryAt: input.nextRetryAt,
    updatedAt: input.acceleration.preparation.updatedAt ?? input.now ?? new Date().toISOString(),
    recentEvents: input.acceleration.preparation.log
      .slice(-8)
      .map((entry) => ({
        at: entry.at,
        level: entry.level,
        message: sanitizeAcceleratorDiagnosticText(entry.message) ?? "Accelerator state changed.",
      })),
  };
}
