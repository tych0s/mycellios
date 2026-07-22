interface GpuIdentity {
  id?: string | undefined;
  vendor: string;
  model: string;
}

/** Prefer stable ids; identity fallback exists only for legacy/missing ids. */
export function isAdvertisedGpuSelected(
  detected: GpuIdentity,
  advertised: GpuIdentity | undefined,
): boolean {
  if (!advertised) return false;
  const detectedId = detected.id?.trim();
  const advertisedId = advertised.id?.trim();
  const sameIdentity = detected.vendor.trim().toLowerCase() === advertised.vendor.trim().toLowerCase()
    && detected.model.trim().toLowerCase() === advertised.model.trim().toLowerCase();
  if (detectedId && advertisedId) return detectedId === advertisedId && sameIdentity;
  return sameIdentity;
}
