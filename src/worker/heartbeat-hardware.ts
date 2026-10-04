import type { HardwareProbe } from "./hardware.js";

/** Optional OS telemetry must never hold the worker's liveness message. */
export function createHeartbeatHardwareSampler(
  probe: () => Promise<HardwareProbe>,
): () => HardwareProbe | null {
  let latest: HardwareProbe | null = null;
  let pending = false;
  return () => {
    const snapshot = latest;
    if (!pending) {
      pending = true;
      void Promise.resolve().then(probe).then(
        (hardware) => { latest = hardware; },
        () => { latest = null; },
      ).finally(() => { pending = false; });
    }
    return snapshot;
  };
}
