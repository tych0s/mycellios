export interface AutomaticUpdateInstallState {
  updateReady: boolean;
  quitting: boolean;
  activeJobs: number;
  activeStages: number;
}

/**
 * Downloaded updates may repair a broken accelerator pack, but must never
 * interrupt an inference request or a distributed stage that is still alive.
 */
export function canInstallAutomaticUpdate(state: AutomaticUpdateInstallState): boolean {
  return state.updateReady
    && !state.quitting
    && state.activeJobs === 0
    && state.activeStages === 0;
}

export const AUTOMATIC_UPDATE_GRACE_MS = 60_000;
export const AUTOMATIC_UPDATE_IDLE_RECHECK_MS = 30_000;

const AUTOMATIC_UPDATE_RETRY_DELAYS_MS = [
  30_000,
  2 * 60_000,
  5 * 60_000,
  15 * 60_000,
  30 * 60_000,
] as const;

/**
 * Update checks are allowed to fail transiently under memory pressure or while
 * the public feed is being replaced. Keep retrying forever with a bounded
 * delay; a single failed check must not strand an unattended worker.
 */
export function automaticUpdateRetryDelayMs(attempt: number): number {
  const normalized = Number.isFinite(attempt) ? Math.max(0, Math.floor(attempt)) : 0;
  return AUTOMATIC_UPDATE_RETRY_DELAYS_MS[
    Math.min(normalized, AUTOMATIC_UPDATE_RETRY_DELAYS_MS.length - 1)
  ]!;
}

export function summarizeAutomaticUpdateError(message: string): string {
  const normalized = message.replace(/\s+/g, " ").trim();
  if (/OutOfMemoryException/i.test(normalized)) {
    return "The Windows updater ran out of memory while checking the feed.";
  }
  return normalized.slice(0, 320) || "The update check failed.";
}
