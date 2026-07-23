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
