export function selectInferenceModel<T extends { id: string; freeSlots: number }>(models: readonly T[], selected: string): T | null {
  // A deliberate choice must not silently become a different model after a refresh.
  if (selected) return models.find((model) => model.id === selected) ?? null;
  return models.find((model) => model.freeSlots > 0) ?? models[0] ?? null;
}

export function isAvailableInferenceWorker(worker: {
  connected: boolean;
  status: string;
  quarantined?: boolean;
  observedCapability?: { eligibility: { serving: boolean } };
}): boolean {
  return worker.connected && worker.status === "online" && !worker.quarantined
    && worker.observedCapability?.eligibility.serving !== false;
}
