import { resolve } from "node:path";
import { describe, expect, it, vi } from "vitest";
import {
  executePhysicalGpuGateCli,
  parsePhysicalGpuGateCliArguments,
} from "../src/distribution/physical-gpu-gate-cli.js";
import type { PhysicalTwoHostGpuGateReportV1 } from "../src/distribution/physical-gpu-gate-report.js";

describe("physical GPU gate CLI", () => {
  it("accepts exactly one report path", () => {
    expect(parsePhysicalGpuGateCliArguments(["--report", "evidence/gate.json"])).toEqual({
      reportPath: "evidence/gate.json",
    });
    expect(() => parsePhysicalGpuGateCliArguments([])).toThrow(
      "physical_gpu_gate_cli_report_is_required",
    );
    expect(() =>
      parsePhysicalGpuGateCliArguments(["--report", "a.json", "--report", "b.json"]),
    ).toThrow("physical_gpu_gate_cli_argument_is_invalid:--report");
    expect(() => parsePhysicalGpuGateCliArguments(["--input", "a.json"])).toThrow(
      "physical_gpu_gate_cli_argument_is_invalid:--input",
    );
  });

  it("prints a compact summary only after the sealed gate verifier passes", async () => {
    const report = passingReportShape();
    const readText = vi.fn(async () => "sealed-report");
    const verify = vi.fn(() => report);
    const output = await executePhysicalGpuGateCli(
      ["--report", "evidence/gate.json"],
      { cwd: "C:\\workspace", readText, verify },
    );

    expect(readText).toHaveBeenCalledWith(resolve("C:\\workspace", "evidence/gate.json"));
    expect(verify).toHaveBeenCalledWith("sealed-report");
    expect(JSON.parse(output)).toEqual({
      schema: "gdlp-physical-gpu-gate-verification/1",
      passed: true,
      reportSchema: "gdlp-physical-two-host-gpu-gate/1",
      capturedAt: "2026-07-21T12:00:00.000Z",
      seal: `sha256:${"a".repeat(64)}`,
      measuredSamples: 5,
      measuredCompletionTokens: 80,
      ttftP50Ms: 100,
      tpotP50Ms: 20,
      outputTokensPerSecondP50IncludingTtft: 12,
    });
  });

  it("does not turn a verifier failure into a successful summary", async () => {
    await expect(
      executePhysicalGpuGateCli(["--report", "failed.json"], {
        readText: async () => "failed-report",
        verify: () => {
          throw new Error("physical_gpu_gate_failed:two_distinct_hosts");
        },
      }),
    ).rejects.toThrow("physical_gpu_gate_failed:two_distinct_hosts");
  });
});

function passingReportShape(): PhysicalTwoHostGpuGateReportV1 {
  return {
    schema: "gdlp-physical-two-host-gpu-gate/1",
    capturedAt: "2026-07-21T12:00:00.000Z",
    seal: { digest: `sha256:${"a".repeat(64)}` },
    summary: {
      measuredSamples: 5,
      measuredCompletionTokens: 80,
      ttftMs: { p50: 100 },
      tpotMs: { p50: 20 },
      outputTokensPerSecondIncludingTtft: { p50: 12 },
    },
  } as unknown as PhysicalTwoHostGpuGateReportV1;
}
