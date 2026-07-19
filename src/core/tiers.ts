export type WorkerTier = "T0" | "T1" | "T2" | "T3" | "T4" | "INELIGIBLE";

export function tierForOfferedVram(offeredVramMb: number): WorkerTier {
  if (offeredVramMb < 4_096) return "INELIGIBLE";
  if (offeredVramMb < 8_192) return "T0";
  if (offeredVramMb < 12_288) return "T1";
  if (offeredVramMb < 16_384) return "T2";
  if (offeredVramMb < 24_576) return "T3";
  return "T4";
}

export function safeVramBudget(offeredVramMb: number): number {
  return Math.max(0, Math.floor(offeredVramMb - Math.max(512, offeredVramMb * 0.15)));
}
