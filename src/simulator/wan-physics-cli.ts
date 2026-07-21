import {
  DEFAULT_WAN_PHYSICS_OPTIONS,
  GLM45_AIR_Q4KM,
  IDEAL_SLOTS_DISCLAIMER,
  THEORETICAL_PHYSICS_KIND,
  activationBoundaryBytes,
  activeWeightReadBytesPerToken,
  approximateActiveFlopsPerToken,
  defaultWanSensitivityMatrix,
  kvBytesForContext,
  kvBytesPerContextToken,
  routeSuccessProbability,
  simulateCuratedTenNodeRoute,
  simulateWanSuite,
  type WanSummary,
} from "./wan-physics.js";

interface CliOptions {
  nodes: number[];
  trials: number;
  seed: number;
  promptTokens: number;
  contextTokens: number;
  microchunkTokens: number;
  includeCurated: boolean;
  curatedTrials: number;
  json: boolean;
  help: boolean;
}

function readValue(args: readonly string[], index: number, name: string): string {
  const value = args[index + 1];
  if (value === undefined || value.startsWith("--")) {
    throw new Error(`Falta el valor de --${name}`);
  }
  return value;
}

function positiveInteger(value: string, name: string): number {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed <= 0) {
    throw new Error(`--${name} debe ser un entero positivo`);
  }
  return parsed;
}

function parseArgs(args: readonly string[]): CliOptions {
  const options: CliOptions = {
    nodes: [10, 100, 1_000],
    trials: DEFAULT_WAN_PHYSICS_OPTIONS.trials,
    seed: DEFAULT_WAN_PHYSICS_OPTIONS.seed,
    promptTokens: DEFAULT_WAN_PHYSICS_OPTIONS.promptTokens,
    contextTokens: DEFAULT_WAN_PHYSICS_OPTIONS.contextTokens,
    microchunkTokens: DEFAULT_WAN_PHYSICS_OPTIONS.microchunkTokens,
    includeCurated: true,
    curatedTrials: 10_000,
    json: false,
    help: false,
  };

  for (let index = 0; index < args.length; index += 1) {
    const argument = args[index]!;
    if (argument === "--json") {
      options.json = true;
    } else if (argument === "--no-curated") {
      options.includeCurated = false;
    } else if (argument === "--help" || argument === "-h") {
      options.help = true;
    } else if (argument === "--nodes") {
      const value = readValue(args, index, "nodes");
      options.nodes = value.split(",").map((entry) => positiveInteger(entry, "nodes"));
      index += 1;
    } else if (argument === "--trials") {
      options.trials = positiveInteger(readValue(args, index, "trials"), "trials");
      index += 1;
    } else if (argument === "--seed") {
      options.seed = positiveInteger(readValue(args, index, "seed"), "seed");
      index += 1;
    } else if (argument === "--prompt") {
      options.promptTokens = positiveInteger(readValue(args, index, "prompt"), "prompt");
      index += 1;
    } else if (argument === "--context") {
      options.contextTokens = positiveInteger(readValue(args, index, "context"), "context");
      index += 1;
    } else if (argument === "--microchunk") {
      options.microchunkTokens = positiveInteger(
        readValue(args, index, "microchunk"),
        "microchunk",
      );
      index += 1;
    } else if (argument === "--curated-trials") {
      options.curatedTrials = positiveInteger(
        readValue(args, index, "curated-trials"),
        "curated-trials",
      );
      index += 1;
    } else {
      throw new Error(`Argumento desconocido: ${argument}`);
    }
  }
  return options;
}

function usage(): string {
  return [
    "Usage: npm run simulate:physics -- [options]",
    "",
    "  --nodes 10,100,1000  Fleet sizes; they share a single RNG stream",
    "  --trials 1000         Monte Carlo trials per size",
    "  --seed 123456789      Mulberry32 seed",
    "  --prompt 2000         Prompt tokens for TTFT",
    "  --context 2000        Context tokens for KV bytes",
    "  --microchunk 128      Prefill microchunk",
    "  --curated-trials 10000 Trials for the curated regional 10-node case",
    "  --no-curated          Omit the curated case",
    "  --json                Unrounded JSON output",
  ].join("\n");
}

function fixed(value: number, decimals = 2): string {
  return value.toFixed(decimals);
}

function printSummary(summary: WanSummary): void {
  console.log(
    [
      `N=${summary.nodes}`,
      `P(route)=${fixed(summary.probabilityAnyRoute * 100, 1)}%`,
      `routes p50/p95=${summary.routes.p50}/${summary.routes.p95}`,
      `stages p50/p95=${summary.stages.p50}/${summary.stages.p95}`,
      `tok/s p50/slow-P95=${fixed(summary.tokensPerSecond.p50)}/${fixed(summary.tokensPerSecond.slowPathP95)}`,
      `TTFT p50/p95=${fixed(summary.ttftMs.p50 / 1_000)}/${fixed(summary.ttftMs.p95 / 1_000)} s`,
      `ideal slots p50/p95=${summary.concurrency.p50}/${summary.concurrency.p95}`,
      `power p50/p95=${fixed(summary.incrementalPowerKw.p50)}/${fixed(summary.incrementalPowerKw.p95)} kW`,
    ].join(" | "),
  );
}

function buildReport(options: CliOptions) {
  const simulationOverrides = {
    trials: options.trials,
    promptTokens: options.promptTokens,
    contextTokens: options.contextTokens,
    microchunkTokens: options.microchunkTokens,
  };
  const suite = simulateWanSuite(options.nodes, simulationOverrides, options.seed);
  const curated = options.includeCurated
    ? simulateCuratedTenNodeRoute(options.curatedTrials, 987_654_321, {
        promptTokens: options.promptTokens,
        contextTokens: options.contextTokens,
        microchunkTokens: options.microchunkTokens,
      })
    : null;
  return {
    kind: THEORETICAL_PHYSICS_KIND,
    warning:
      "Theoretical physics model, not a measured benchmark. Ideal slots are not guaranteed users.",
    idealSlotsSemantics: IDEAL_SLOTS_DISCLAIMER,
    rng: {
      algorithm: "Mulberry32 + Box-Muller (no cached spare)",
      seed: options.seed,
      sharedSequentiallyAcrossFleetSizes: true,
    },
    model: {
      ...GLM45_AIR_Q4KM,
      activeWeightReadBytesPerToken: activeWeightReadBytesPerToken(),
      approximateActiveFlopsPerToken: approximateActiveFlopsPerToken(),
      activationBytes: {
        int8: activationBoundaryBytes(1),
        bf16: activationBoundaryBytes(2),
      },
      kvBytesPerContextToken: kvBytesPerContextToken(),
      kvExamples: {
        bf16At2k: kvBytesForContext(2_000),
        bf16At20k: kvBytesForContext(20_000),
        bf16At128k: kvBytesForContext(128_000),
        q8At2k: kvBytesForContext(2_000, 1),
        q8At20k: kvBytesForContext(20_000, 1),
        q8At128k: kvBytesForContext(128_000, 1),
      },
    },
    suite,
    curatedTenNodeRegionalRoute: curated,
    sensitivity: defaultWanSensitivityMatrix(),
    noRedundancyRouteSuccessAtTwoPercentStageFailure: {
      stages6: routeSuccessProbability(6, 0.02),
      stages10: routeSuccessProbability(10, 0.02),
      stages16: routeSuccessProbability(16, 0.02),
    },
  };
}

function main(): void {
  try {
    const options = parseArgs(process.argv.slice(2));
    if (options.help) {
      console.log(usage());
      return;
    }
    const report = buildReport(options);
    if (options.json) {
      console.log(JSON.stringify(report, null, 2));
      return;
    }

    console.log(`kind=${report.kind}`);
    console.log(
      "WARNING: theoretical physical model, not a measured benchmark. Slots are ideal cyclic-pipeline slots, not guaranteed users.",
    );
    console.log(
      `GLM-4.5-Air: 46 layers, H=4096, 73 GB Q4_K_M; INT8 activation=${report.model.activationBytes.int8} B.`,
    );
    console.log(
      `Monte Carlo: ${options.trials} trials/size, seed=${options.seed}, RNG shared in ${options.nodes.join(" -> ")} order.`,
    );
    for (const summary of report.suite) printSummary(summary);

    if (report.curatedTenNodeRegionalRoute !== null) {
      console.log("\nCurated case: 10 feasible online nodes in the same region:");
      printSummary(report.curatedTenNodeRegionalRoute);
    }

    console.log("\nBF16 KV per conversation:");
    console.log(
      `2k=${fixed(report.model.kvExamples.bf16At2k / 1e9, 3)} GB | 20k=${fixed(report.model.kvExamples.bf16At20k / 1e9, 3)} GB | 128k=${fixed(report.model.kvExamples.bf16At128k / 1e9, 3)} GB (Q8: half).`,
    );
    console.log("\nTok/s sensitivity (BW=100 Mbps; RTT shown as one-way latency per hop):");
    for (const stages of [6, 10, 16]) {
      const row = report.sensitivity.filter((point) => point.stages === stages);
      console.log(
        `${stages} stages: ${row.map((point) => `${point.oneWayHopMs} ms=${fixed(point.tokensPerSecond, 4)}`).join(" | ")}`,
      );
    }
    console.log(
      `Success without redundancy at 2% failure/stage: S6=${fixed(report.noRedundancyRouteSuccessAtTwoPercentStageFailure.stages6 * 100, 1)}% | S10=${fixed(report.noRedundancyRouteSuccessAtTwoPercentStageFailure.stages10 * 100, 1)}% | S16=${fixed(report.noRedundancyRouteSuccessAtTwoPercentStageFailure.stages16 * 100, 1)}%.`,
    );
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.error(`Error: ${message}`);
    console.error(usage());
    process.exitCode = 1;
  }
}

main();
