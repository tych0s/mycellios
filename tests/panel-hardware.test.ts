import { describe, expect, it } from "vitest";
import { isAdvertisedGpuSelected } from "../landing/src/panel-hardware.js";

describe("desktop panel GPU selection", () => {
  it("does not label a same-model adapter selected when its stable id differs", () => {
    expect(isAdvertisedGpuSelected(
      { id: "gpu-1", vendor: "nvidia", model: "NVIDIA RTX 4090" },
      { id: "gpu-0", vendor: "nvidia", model: "NVIDIA RTX 4090" },
    )).toBe(false);
  });

  it("uses vendor/model only as a legacy fallback when an id is absent", () => {
    expect(isAdvertisedGpuSelected(
      { id: "gpu-1", vendor: "AMD", model: "Radeon RX 9070 XT" },
      { vendor: "amd", model: "radeon rx 9070 xt" },
    )).toBe(true);
  });

  it("requires identity to agree even when a regenerated id collides", () => {
    expect(isAdvertisedGpuSelected(
      { id: "gpu-0", vendor: "intel", model: "Intel UHD" },
      { id: "gpu-0", vendor: "amd", model: "AMD Radeon 890M" },
    )).toBe(false);
  });
});
