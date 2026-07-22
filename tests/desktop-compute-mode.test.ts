import { describe, expect, it } from "vitest";
import {
  desktopExecutorPolicy,
  normalizeComputeMode,
} from "../src/desktop/compute-mode.js";

describe("desktop compute mode", () => {
  it("keeps Automatic on GPU while GPU preparation can still succeed", () => {
    expect(desktopExecutorPolicy("automatic", true)).toEqual({
      computeMode: "automatic",
      cpuEligible: false,
    });
  });

  it("authorizes CPU in Automatic only after GPU preparation is exhausted", () => {
    expect(desktopExecutorPolicy("automatic", false)).toEqual({
      computeMode: "automatic",
      cpuEligible: true,
    });
  });

  it("honors strict GPU-only and CPU-only user choices", () => {
    expect(desktopExecutorPolicy("gpu-only", false).cpuEligible).toBe(false);
    expect(desktopExecutorPolicy("cpu-only", true).cpuEligible).toBe(true);
  });

  it("migrates missing or unknown stored values to Automatic", () => {
    expect(normalizeComputeMode(undefined)).toBe("automatic");
    expect(normalizeComputeMode("unexpected")).toBe("automatic");
    expect(normalizeComputeMode("cpu-only")).toBe("cpu-only");
  });
});
