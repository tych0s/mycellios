import { describe, expect, it } from "vitest";
import {
  buildWorkerAccelerationDiagnostics,
  sanitizeAcceleratorDiagnosticText,
} from "../src/node/acceleration-diagnostics.js";
import { createInitialAccelerationStatus } from "../src/node/acceleration-progress.js";

describe("remote accelerator diagnostics", () => {
  it("removes local paths, bearer credentials and multiline stack formatting", () => {
    const sanitized = sanitizeAcceleratorDiagnosticText(
      "Traceback C:\\Users\\User\\AppData\\runtime\\python.exe\nBearer super-secret-token /home/user/runtime/file.py",
    );
    expect(sanitized).not.toContain("Daniel");
    expect(sanitized).not.toContain("super-secret-token");
    expect(sanitized).not.toContain("/home/user");
    expect(sanitized).not.toContain("\n");
    expect(sanitized).toContain("[local path]");
    expect(sanitized).toContain("Bearer [redacted]");
  });

  it("publishes only a bounded recent repair history", () => {
    const acceleration = createInitialAccelerationStatus();
    acceleration.state = "gpu-fallback";
    acceleration.effectiveBackend = "cpu";
    acceleration.gpu.vendor = "nvidia";
    acceleration.gpu.model = "NVIDIA GeForce RTX 2060";
    acceleration.preparation.phase = "physical-probe";
    acceleration.preparation.progressPct = 95;
    acceleration.preparation.updatedAt = "2026-07-23T10:00:00.000Z";
    acceleration.preparation.issue = {
      code: "physical-probe",
      message: "CUDA driver probe failed",
      action: "retry",
      retryable: true,
    };
    acceleration.preparation.log = Array.from({ length: 20 }, (_, index) => ({
      at: `2026-07-23T10:00:${String(index).padStart(2, "0")}.000Z`,
      level: index === 19 ? "error" as const : "info" as const,
      message: `event ${index}`,
    }));

    const result = buildWorkerAccelerationDiagnostics({
      appVersion: "0.2.26",
      acceleration,
      retryAttempt: 3,
      nextRetryAt: "2026-07-23T10:30:00.000Z",
    });
    expect(result).toMatchObject({
      schema: "mycellios-accelerator-diagnostics/1",
      appVersion: "0.2.26",
      state: "gpu-fallback",
      backend: "cpu",
      gpuModel: "NVIDIA GeForce RTX 2060",
      issueCode: "physical-probe",
      retryable: true,
      retryAttempt: 3,
    });
    expect(result.recentEvents).toHaveLength(8);
    expect(result.recentEvents[0]?.message).toBe("event 12");
    expect(result.recentEvents.at(-1)?.message).toBe("event 19");
  });
});
