import { describe, expect, it } from "vitest";
import {
  MAX_ACCELERATION_LOG_ENTRIES,
  applyAccelerationProgress,
  beginAccelerationPreparation,
  createInitialAccelerationStatus,
  gpuPreparationIsContinuing,
  selectImmediateRuntime,
  updateActiveAccelerationStages,
} from "../src/node/acceleration-progress.js";

describe("desktop acceleration progress", () => {
  it("reports setup only for active work or a retryable terminal failure", () => {
    const started = beginAccelerationPreparation(createInitialAccelerationStatus(), {
      vendor: "nvidia",
      model: "RTX 4090",
      backend: "cuda",
    });
    expect(gpuPreparationIsContinuing(started)).toBe(true);

    const retrying = applyAccelerationProgress(started, {
      phase: "fallback",
      message: "Temporary network failure",
      issue: { code: "network", message: "offline", action: "retry", retryable: true },
    });
    expect(gpuPreparationIsContinuing(retrying)).toBe(true);

    const stopped = applyAccelerationProgress(started, {
      phase: "fallback",
      message: "Unsupported",
      issue: { code: "unsupported-gpu", message: "unsupported", action: "use CPU", retryable: false },
    });
    expect(gpuPreparationIsContinuing(stopped)).toBe(false);
  });
  it("selects CPU immediately without waiting for background GPU provisioning", async () => {
    const cpu = { device: "cpu" };
    const gpu = { device: "gpu" };
    let resolved: typeof cpu | null = null;
    let finishGpu!: () => void;
    const backgroundGpu = new Promise<void>((resolve) => { finishGpu = resolve; }).then(() => {
      resolved = gpu;
    });

    const first = await selectImmediateRuntime(() => resolved, async () => cpu);
    expect(first).toBe(cpu);

    finishGpu();
    await backgroundGpu;
    const second = await selectImmediateRuntime(() => resolved, async () => cpu);
    expect(second).toBe(gpu);
  });

  it("keeps CPU availability separate from GPU preparation", () => {
    const started = beginAccelerationPreparation(createInitialAccelerationStatus(), {
      vendor: "NVIDIA",
      model: "NVIDIA GeForce RTX 4090",
      backend: "cuda",
      cpuName: "AMD Ryzen 9",
    }, "2026-07-22T10:00:00.000Z");

    expect(started).toMatchObject({
      state: "preparing",
      requestedBackend: "cuda",
      effectiveBackend: "cpu",
      cpu: { state: "ready", activeStages: 0, deviceName: "AMD Ryzen 9" },
      gpu: { state: "checking", model: "NVIDIA GeForce RTX 4090", backend: "cuda" },
      preparation: { phase: "detecting", progressPct: 0 },
    });

    const active = updateActiveAccelerationStages(
      started,
      "cpu",
      1,
      "Model stage started immediately on CPU while CUDA setup continues.",
      "2026-07-22T10:00:01.000Z",
    );
    expect(active.cpu).toMatchObject({ state: "active", activeStages: 1 });
    expect(active.gpu.state).toBe("checking");
  });

  it("keeps progress monotonic and records byte telemetry without spamming logs", () => {
    const started = beginAccelerationPreparation(createInitialAccelerationStatus(), {
      vendor: "AMD",
      model: "AMD Radeon 890M",
      backend: "rocm",
    });
    const first = applyAccelerationProgress(started, {
      phase: "downloading",
      message: "Downloading ROCm runtime",
      progressPct: 42.4,
      bytesCompleted: 920_000_000,
      bytesTotal: 2_188_680_000,
      bytesPerSecond: 18_000_000,
      etaSeconds: 71,
      recordLog: false,
    });
    const stale = applyAccelerationProgress(first, {
      phase: "downloading",
      message: "Downloading ROCm runtime",
      progressPct: 39,
      bytesCompleted: 930_000_000,
      recordLog: false,
    });

    expect(stale.preparation.progressPct).toBe(42.4);
    expect(stale.preparation.bytesCompleted).toBe(930_000_000);
    expect(stale.preparation.bytesTotal).toBe(2_188_680_000);
    expect(stale.preparation.log).toHaveLength(1);
  });

  it("marks GPU ready without claiming an active GPU stage", () => {
    const started = beginAccelerationPreparation(createInitialAccelerationStatus(), {
      vendor: "AMD",
      model: "AMD Radeon 890M",
      backend: "rocm",
    });
    const ready = applyAccelerationProgress(started, {
      phase: "ready",
      message: "Physical FP16 verification passed; GPU is ready for new stages.",
    });

    expect(ready).toMatchObject({
      state: "gpu-ready",
      effectiveBackend: "rocm",
      precision: "float16",
      cpu: { state: "ready" },
      gpu: { state: "ready", activeStages: 0 },
      preparation: { progressPct: 100 },
    });
  });

  it("does not migrate an active CPU stage when GPU becomes ready", () => {
    const started = beginAccelerationPreparation(createInitialAccelerationStatus(), {
      vendor: "NVIDIA",
      model: "NVIDIA GeForce RTX 4090",
      backend: "cuda",
    });
    const cpuActive = updateActiveAccelerationStages(started, "cpu", 1);
    const gpuReady = applyAccelerationProgress(cpuActive, {
      phase: "ready",
      message: "GPU verified and ready for new stages.",
    });
    const bothActive = updateActiveAccelerationStages(gpuReady, "gpu", 1);

    expect(gpuReady.cpu).toMatchObject({ state: "active", activeStages: 1 });
    expect(gpuReady.gpu).toMatchObject({ state: "ready", activeStages: 0 });
    expect(bothActive).toMatchObject({
      cpu: { activeStages: 1 },
      gpu: { activeStages: 1 },
    });
  });

  it.each([
    {
      phase: "fallback" as const,
      code: "no-gpu",
      expected: "not-detected" as const,
      message: "No supported GPU or display driver was detected.",
    },
    {
      phase: "blocked" as const,
      code: "unsupported-gpu",
      expected: "unsupported" as const,
      message: "The detected GPU is not supported by this release.",
    },
    {
      phase: "fallback" as const,
      code: "unsupported-platform",
      expected: "unsupported" as const,
      message: "Native GPU setup is not supported on this platform yet.",
    },
  ])("maps terminal $phase issue $code to GPU state $expected", ({ phase, code, expected, message }) => {
    const started = beginAccelerationPreparation(createInitialAccelerationStatus(), {
      vendor: code === "no-gpu" ? null : "Intel",
      model: code === "no-gpu" ? null : "Intel Arc A770",
      backend: null,
    });
    const terminal = applyAccelerationProgress(started, {
      phase,
      message,
      progressPct: 100,
      issue: {
        code,
        message,
        retryable: false,
        action: "Continue contributing with CPU.",
      },
    });

    expect(terminal.gpu.state).toBe(expected);
    expect(terminal.preparation).toMatchObject({ phase, progressPct: 100, issue: { code } });
    expect(terminal.effectiveBackend).toBe("cpu");
  });

  it("preserves a previously classified unsupported GPU when a terminal event omits its issue", () => {
    const unsupported = beginAccelerationPreparation(createInitialAccelerationStatus(), {
      vendor: "Intel",
      model: "Intel Arc A770",
      backend: null,
    });
    const terminal = applyAccelerationProgress(unsupported, {
      phase: "fallback",
      message: "CPU remains available.",
    });

    expect(terminal.gpu.state).toBe("unsupported");
  });

  it("bounds and sanitizes the visible installation log", () => {
    let state = createInitialAccelerationStatus();
    for (let index = 0; index < MAX_ACCELERATION_LOG_ENTRIES + 8; index += 1) {
      state = applyAccelerationProgress(state, {
        phase: "installing",
        message: `Step ${index}\nBearer secret-${index}`,
        progressPct: index,
      });
    }

    expect(state.preparation.log).toHaveLength(MAX_ACCELERATION_LOG_ENTRIES);
    expect(state.preparation.log[0]?.message).toContain("Step 8");
    expect(state.preparation.log.at(-1)?.message).toBe(
      `Step ${MAX_ACCELERATION_LOG_ENTRIES + 7} Bearer [redacted]`,
    );
  });
});
