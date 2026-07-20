import { readFile } from "node:fs/promises";
import { describe, expect, it } from "vitest";
import {
  executeMacroWaveCli,
  parseMacroWaveCliArguments,
  parseMacroWaveCliInput,
  planMacroWaveCliDocument,
} from "../src/distribution/macro-wave-cli.js";

const ROOT = process.cwd();
const EXAMPLE = "config/macro-wave-4gb.example.json";

async function exampleInput(): Promise<unknown> {
  return JSON.parse(await readFile(`${ROOT}/${EXAMPLE}`, "utf8")) as unknown;
}

describe("MacroWave CLI contract", () => {
  it("emits a complete machine-readable plan for domestic 4 GB GPUs", async () => {
    const output = await executeMacroWaveCli([EXAMPLE, "--format", "json"], ROOT);
    const report = JSON.parse(output) as ReturnType<typeof planMacroWaveCliDocument>;

    expect(report.schema).toBe("gdlp-macro-wave-report/1");
    expect(report.modelId).toBe("domestic-moe-floating-safetensors-profile");
    expect(report.feasible).toBe(true);
    expect(report.selectedAlternative).toBe("macro-wave");
    expect(report.plan?.algorithm).toBe("macro-wave-ram-vram");
    expect(report.plan?.macroWave).toMatchObject({
      schema: "gdlp-macro-wave-plan/1",
      routeKind: "macro-wave",
      waveTokens: 4,
      expectedCommittedTokensPerWave: 3,
    });
    expect(report.plan?.stages.length).toBeGreaterThan(1);
    expect(report.plan?.stages[0]?.layerStart).toBe(0);
    expect(report.plan?.stages.at(-1)?.layerEnd).toBe(12);
    expect(report.plan?.stages.every((stage) => stage.macroWave?.schema === "gdlp-macro-wave-stage/1"))
      .toBe(true);
    expect(report.plan?.stages.some((stage) => stage.macroWave?.memoryMode === "ram-backed"))
      .toBe(true);
    expect(report.metrics.ttftMs).toBeGreaterThan(0);
    expect(report.metrics.tpotMs).toBeGreaterThan(0);
    expect(report.metrics.tokensPerSecondPerSequence).toBeGreaterThan(0);
    expect(report.traffic.rawActiveWeightBytesPerOutputToken).toBeGreaterThan(0);
    expect(report.traffic.expectedWeightCacheMissBytesPerOutputToken).toBe(
      report.traffic.rawActiveWeightBytesPerOutputToken,
    );
    expect(report.traffic.expectedWeightCacheHitRate).toBe(0);
    expect(report.stages.some((stage) => stage.weights.bufferCopies === 2)).toBe(true);
    expect(report.stages.every((stage) => stage.vram.requiredBytes <= stage.vram.usableBytes)).toBe(
      true,
    );
    expect(report.stages.every((stage) => stage.ram.requiredBytes <= stage.ram.usableBytes)).toBe(
      true,
    );
    expect(report.alternatives.find((candidate) => candidate.kind === "resident-baseline"))
      .toMatchObject({ feasible: false });
    expect(report.discardReasons.length).toBeGreaterThan(0);
  });

  it("is deterministic for the same canonical input", async () => {
    const input = await exampleInput();
    const first = planMacroWaveCliDocument(input);
    const second = planMacroWaveCliDocument(structuredClone(input));

    expect(second).toEqual(first);
    expect(await executeMacroWaveCli([EXAMPLE], ROOT)).toBe(
      await executeMacroWaveCli([EXAMPLE, "--format=json"], ROOT),
    );
  });

  it("accepts header-derived resident tensor telemetry and seals the tighter peak", async () => {
    const input = (await exampleInput()) as {
      model: {
        layers: Array<{
          weightBytes: number;
          expertParallel?: { expertWeightBytes: number };
          largestResidentTensorBytes?: number;
        }>;
        largestEmbeddingTensorBytes?: number;
        largestLmHeadTensorBytes?: number;
      };
    };
    for (const layer of input.model.layers) {
      const residentBytes =
        layer.weightBytes - (layer.expertParallel?.expertWeightBytes ?? 0);
      layer.largestResidentTensorBytes = Math.min(8 * 1024 ** 2, residentBytes);
    }
    input.model.largestEmbeddingTensorBytes = 8 * 1024 ** 2;
    input.model.largestLmHeadTensorBytes = 8 * 1024 ** 2;

    const parsed = parseMacroWaveCliInput(input);
    const report = planMacroWaveCliDocument(parsed);
    const ramStages = report.plan!.stages.filter(
      (stage) => stage.macroWave?.memoryMode === "ram-backed",
    );
    expect(ramStages.length).toBeGreaterThan(0);
    for (const stage of ramStages) {
      const execution = stage.macroWave!;
      expect(execution.requirements.residentStreamingTransientBytes).toBe(
        8 * 1024 ** 2,
      );
      expect(execution.requirements.hostRamPeakUpperBoundBytes).toBe(
        execution.workingSet.totalRoutedExpertBytes +
          execution.requirements.boundedPinnedStagingReserveBytes +
          execution.requirements.residentStreamingTransientBytes,
      );
      expect(execution.requirements.hostRamPeakUpperBoundBytes).toBeLessThan(
        execution.workingSet.totalRoutedExpertBytes +
          execution.requirements.boundedPinnedStagingReserveBytes +
          execution.requirements.residentParameterBudgetBytes,
      );
    }

    input.model.layers[0]!.largestResidentTensorBytes = -1;
    expect(() => parseMacroWaveCliInput(input)).toThrow(
      /^invalid_macro_wave_input:/,
    );
  });

  it("renders the operational metrics and capacity decisions as Markdown", async () => {
    const output = await executeMacroWaveCli([EXAMPLE, "--format", "markdown"], ROOT);

    expect(output).toContain("# MacroWave RAM+VRAM");
    expect(output).toContain("## Métricas");
    expect(output).toContain("| TTFT | TPOT | tok/s usuario");
    expect(output).toContain("## Etapas");
    expect(output).toContain("RAM requerida/útil");
    expect(output).toContain("VRAM requerida/útil");
    expect(output).toContain("## Razones de descarte");
    expect(output).not.toContain("Infinity");
  });

  it("parses only the documented command-line contract", () => {
    expect(parseMacroWaveCliArguments(["profile.json", "--format=markdown"])).toEqual({
      inputPath: "profile.json",
      format: "markdown",
      help: false,
    });
    expect(parseMacroWaveCliArguments(["--help"])).toEqual({
      inputPath: null,
      format: "json",
      help: true,
    });
    expect(() => parseMacroWaveCliArguments(["profile.json", "--format", "csv"])).toThrow(
      "macro_wave_format_must_be_json_or_markdown",
    );
    expect(() => parseMacroWaveCliArguments([])).toThrow("macro_wave_input_file_is_required");
  });

  it("fails closed for malformed profiles and reports unprofiled hardware", async () => {
    expect(() => parseMacroWaveCliInput({ schema: "gdlp-macro-wave-input/1" })).toThrow(
      /^invalid_macro_wave_input:/,
    );

    const input = (await exampleInput()) as {
      topology: { nodes: Array<{ ramVram?: unknown }> };
    };
    for (const node of input.topology.nodes) delete node.ramVram;
    const report = planMacroWaveCliDocument(input);

    expect(report.feasible).toBe(false);
    expect(report.plan).toBeNull();
    expect(report.metrics.ttftMs).toBeNull();
    expect(report.reason).toContain("no_profiled_ram_vram_nodes");
    expect(report.discardReasons).toContainEqual({
      alternative: "resident",
      reason: "missing_ram_vram_profile:home-a-4gb",
      count: 1,
    });
  });

  it("requires bandwidth telemetry in decimal GB/s rather than ambiguous Gbit/s", async () => {
    const input = (await exampleInput()) as {
      topology: {
        nodes: Array<{
          ramVram: Record<string, unknown>;
        }>;
      };
    };
    const hierarchy = input.topology.nodes[0]!.ramVram;
    hierarchy.ramBandwidthGbps = hierarchy.ramBandwidthGBps;
    delete hierarchy.ramBandwidthGBps;

    expect(() => parseMacroWaveCliInput(input)).toThrow(/^invalid_macro_wave_input:/);
  });
});
