import { describe, expect, it } from "vitest";
import {
  collectExecutionTelemetry,
  type AutoDistributionCompilation,
} from "../src/distribution/auto-distribute.js";
import type { LaunchSupervisorSnapshot } from "../src/distribution/launch-supervisor.js";

const launches = [
  {
    kind: "root-engine",
    processId: "root-process",
    anchor: { memberId: "desktop-amd" },
    stageIndex: 0,
    layerStart: 0,
    layerEnd: 14,
  },
  {
    kind: "remote-stage",
    processId: "remote-process",
    anchor: { memberId: "desktop-nvidia" },
    stageIndex: 1,
    layerStart: 14,
    layerEnd: 28,
  },
] as const;

const compilation = {
  launch: { launchOrder: launches },
  manifest: { plans: { decode: { stages: [{}, {}] } } },
} as unknown as AutoDistributionCompilation;

function processOutput(processId: string, execution: Record<string, unknown>) {
  return {
    processId,
    output: {
      stdout: "",
      stderr: `${JSON.stringify({ execution })}\n`,
      stdoutTruncated: false,
      stderrTruncated: false,
    },
  };
}

function snapshot(...processes: ReturnType<typeof processOutput>[]): LaunchSupervisorSnapshot {
  return { processes } as unknown as LaunchSupervisorSnapshot;
}

const completeCpuEvidence = {
  requested_device: "auto",
  device: "cpu",
  device_kind: "cpu",
  backend: "cpu",
  device_name: "AMD Ryzen AI 9 HX 370",
  accelerated: false,
  precision: "float32",
  weight_bytes: 1_100_000_000,
  total_memory_bytes: null,
  allocated_bytes: null,
  reserved_bytes: null,
  peak_allocated_bytes: null,
};

const completeCudaEvidence = {
  requested_device: "auto",
  device: "cuda:0",
  device_kind: "gpu",
  backend: "cuda",
  device_name: "NVIDIA GeForce RTX 2060",
  accelerated: true,
  precision: "float16",
  weight_bytes: 1_100_000_000,
  total_memory_bytes: 6_442_450_944,
  allocated_bytes: 1_250_000_000,
  reserved_bytes: 1_300_000_000,
  peak_allocated_bytes: 1_250_000_000,
};

describe("auto-distribution execution evidence", () => {
  it("publishes a mixed topology only when every launched stage has complete evidence", () => {
    const result = collectExecutionTelemetry(
      compilation,
      snapshot(
        processOutput("root-process", completeCpuEvidence),
        processOutput("remote-process", completeCudaEvidence),
      ),
    );

    expect(result).toMatchObject({
      deviceType: "mixed",
      backend: "cuda",
      fallback: true,
      stages: [
        { stageIndex: 0, nodeId: "desktop-amd", deviceType: "cpu", backend: "cpu" },
        { stageIndex: 1, nodeId: "desktop-nvidia", deviceType: "gpu", backend: "cuda" },
      ],
    });
  });

  it.each([
    { backend: "mps", device: "mps", name: "Apple M4 Pro" },
    { backend: "xpu", device: "xpu:0", name: "Intel Arc B580" },
  ])("accepts truthful $backend allocator evidence", ({ backend, device, name }) => {
    const result = collectExecutionTelemetry(
      compilation,
      snapshot(
        processOutput("root-process", completeCpuEvidence),
        processOutput("remote-process", {
          ...completeCudaEvidence,
          requested_device: device,
          device,
          backend,
          device_name: name,
        }),
      ),
    );

    expect(result).toMatchObject({
      deviceType: "mixed",
      backend,
      stages: [
        { stageIndex: 0, deviceType: "cpu", backend: "cpu" },
        { stageIndex: 1, deviceType: "gpu", backend },
      ],
    });
  });

  it.each([
    {
      name: "one stage has no ready output",
      value: snapshot(processOutput("root-process", completeCpuEvidence)),
    },
    {
      name: "CPU evidence omits measured identity and resident weights",
      value: snapshot(
        processOutput("root-process", {
          requested_device: "auto",
          backend: "cpu",
          accelerated: false,
        }),
        processOutput("remote-process", completeCudaEvidence),
      ),
    },
    {
      name: "GPU evidence reports no allocator residency",
      value: snapshot(
        processOutput("root-process", completeCpuEvidence),
        processOutput("remote-process", { ...completeCudaEvidence, allocated_bytes: 0 }),
      ),
    },
    {
      name: "CPU backend contradicts an accelerated GPU claim",
      value: snapshot(
        processOutput("root-process", {
          ...completeCpuEvidence,
          device: "cuda:0",
          device_kind: "gpu",
          accelerated: true,
        }),
        processOutput("remote-process", completeCudaEvidence),
      ),
    },
  ])("omits deployment telemetry when $name", ({ value }) => {
    expect(collectExecutionTelemetry(compilation, value)).toBeUndefined();
  });
});
