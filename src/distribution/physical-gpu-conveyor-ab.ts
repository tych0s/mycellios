import {
  canonicalEvidenceJson,
  sha256CanonicalEvidence,
} from "../core/json.js";
import {
  PythonLaunchSupervisor,
  type LaunchSupervisorOptions,
} from "./launch-supervisor.js";
import {
  runPhysicalGpuCampaign,
  type PhysicalGpuCampaignDependencies,
  type PhysicalGpuCampaignInput,
  type PhysicalGpuCampaignMetricStats,
  type PhysicalGpuCampaignObservation,
  type PhysicalGpuCampaignSupervisor,
} from "./physical-gpu-campaign.js";
import {
  readPhysicalTwoHostGpuGateReport,
  requirePassingPhysicalTwoHostGpuGate,
  type PhysicalTwoHostGpuGateReportV1,
} from "./physical-gpu-gate-report.js";
import {
  validatePythonLaunchDescription,
  type PythonPipelineLaunchDescription,
} from "./python-launcher.js";

export const PHYSICAL_GPU_CONVEYOR_AB_SCHEMA =
  "gdlp-physical-gpu-conveyor-ab/1" as const;
export const PHYSICAL_GPU_CONVEYOR_AB_CANONICALIZATION =
  "gdlp-canonical-evidence-json/1" as const;

const SHA256_PATTERN = /^sha256:[0-9a-f]{64}$/;
const MAX_ROUNDS = 100;

export type PhysicalGpuConveyorArm = "w1" | "conveyor";
export type PhysicalGpuConveyorHealthPhase = "before" | "after";

export interface PhysicalGpuConveyorAbInput {
  baselineLaunch: PythonPipelineLaunchDescription;
  conveyorLaunch: PythonPipelineLaunchDescription;
  campaign: Omit<PhysicalGpuCampaignInput, "launch">;
  rounds: number;
}

export interface PhysicalGpuConveyorPhysicalGateResult {
  passed: boolean;
  reportSealSha256: string | null;
  report: unknown | null;
}

export type PhysicalGpuConveyorPhysicalGate = (
  arm: PhysicalGpuConveyorArm,
  launch: PythonPipelineLaunchDescription,
  campaign: PhysicalGpuCampaignObservation,
  runIndex: number,
) =>
  | PhysicalGpuConveyorPhysicalGateResult
  | Promise<PhysicalGpuConveyorPhysicalGateResult>;

export type PhysicalGpuConveyorApiHealthReader = (
  apiBaseUrl: string,
  phase: PhysicalGpuConveyorHealthPhase,
  arm: PhysicalGpuConveyorArm,
  runIndex: number,
) => unknown | Promise<unknown>;

export interface PhysicalGpuConveyorAbDependencies {
  runCampaign?: typeof runPhysicalGpuCampaign;
  campaignDependencies?: Omit<PhysicalGpuCampaignDependencies, "supervisor">;
  supervisor?: (
    launch: PythonPipelineLaunchDescription,
    options: LaunchSupervisorOptions,
  ) => PhysicalGpuCampaignSupervisor;
  apiHealth?: PhysicalGpuConveyorApiHealthReader;
  physicalGate?: PhysicalGpuConveyorPhysicalGate;
  now?: () => Date;
}

export interface PhysicalGpuConveyorWindowObservation {
  configured: boolean;
  configuredWavesPerRequest: number;
  configuredBytesPerRequest: number;
  currentWaves: number;
  currentBytes: number;
  currentReservedBytes: number;
  highWaterWaves: number;
  highWaterBytes: number;
  highWaterReservedBytes: number;
  maxRequestWaves: number;
  maxRequestBytes: number;
  maxRequestReservedBytes: number;
  dispatchedWaves: number;
  completedWaves: number;
  committedWaves: number;
  condemnedWaves: number;
  drainedWaves: number;
  rejectionCollapses: number;
  rejectedProposedTokens: number;
  condemnedProposedTokens: number;
  discardedProposedTokens: number;
  rejectedWaveBytes: number;
  tombstoneBytes: number;
  discardedBytes: number;
}

export interface PhysicalGpuConveyorSpeculationObservation {
  configured: boolean;
  proposedTokens: number | null;
  acceptedTokens: number | null;
  acceptanceRate: number | null;
  verificationBytes: number | null;
}

export interface PhysicalGpuConveyorResourceObservation {
  observedPowerWatts: number | null;
  utilizationPct: number | null;
  temperatureC: number | null;
  usedMemoryBytes: number | null;
}

export interface PhysicalGpuConveyorHealthObservation {
  phase: PhysicalGpuConveyorHealthPhase;
  passed: boolean;
  status: string | null;
  canonicalSha256: string | null;
  speculativeWindow: PhysicalGpuConveyorWindowObservation | null;
  speculation: PhysicalGpuConveyorSpeculationObservation | null;
  resources: PhysicalGpuConveyorResourceObservation;
  error: string | null;
}

export interface PhysicalGpuConveyorAbRun {
  runIndex: number;
  round: number;
  position: number;
  arm: PhysicalGpuConveyorArm;
  launchId: string;
  healthBefore: PhysicalGpuConveyorHealthObservation | null;
  healthAfter: PhysicalGpuConveyorHealthObservation | null;
  campaign: PhysicalGpuCampaignObservation | null;
  physicalGate: PhysicalGpuConveyorPhysicalGateResult;
  passed: boolean;
  failures: string[];
}

export interface PhysicalGpuConveyorArmSummary {
  arm: PhysicalGpuConveyorArm;
  runs: number;
  measuredRequests: number;
  completionTokens: number;
  ttftMs: PhysicalGpuCampaignMetricStats;
  tpotMs: PhysicalGpuCampaignMetricStats;
  perUserTokensPerSecond: PhysicalGpuCampaignMetricStats;
  aggregateTokensPerSecond: PhysicalGpuCampaignMetricStats;
  proposedTokens: number | null;
  acceptedTokens: number | null;
  acceptanceRate: number | null;
  highWaterWaves: number | null;
  highWaterBytes: number | null;
  highWaterReservedBytes: number | null;
  rejectedProposedTokens: number | null;
  condemnedProposedTokens: number | null;
  discardedProposedTokens: number | null;
  rejectedWaveBytes: number | null;
  tombstoneBytes: number | null;
  discardedBytes: number | null;
  resources: {
    observedPowerWatts: PhysicalGpuCampaignMetricStats;
    utilizationPct: PhysicalGpuCampaignMetricStats;
    temperatureC: PhysicalGpuCampaignMetricStats;
    usedMemoryBytes: PhysicalGpuCampaignMetricStats;
  };
}

export interface PhysicalGpuConveyorParityCheck {
  key: string;
  baselineSha256: string | null;
  conveyorSha256: string | null;
  passed: boolean;
}

export interface PhysicalGpuConveyorAbReportBody {
  schema: typeof PHYSICAL_GPU_CONVEYOR_AB_SCHEMA;
  capturedAt: string;
  passed: boolean;
  evidence: "physical" | "unverified";
  launches: {
    baselineLaunchId: string;
    conveyorLaunchId: string;
    baselineCanonicalSha256: string;
    conveyorCanonicalSha256: string;
    baselineWaves: 1;
    baselineBytes: 0;
    conveyorWaves: number;
    conveyorBytes: number;
  };
  schedule: PhysicalGpuConveyorArm[];
  runs: PhysicalGpuConveyorAbRun[];
  parity: PhysicalGpuConveyorParityCheck[];
  summary: {
    baseline: PhysicalGpuConveyorArmSummary;
    conveyor: PhysicalGpuConveyorArmSummary;
    comparison: {
      aggregateTokensPerSecondPct: number | null;
      perUserTokensPerSecondPct: number | null;
      ttftP95Pct: number | null;
      tpotP95Pct: number | null;
      acceptancePoints: number | null;
      paired: PhysicalGpuConveyorPairedComparison;
    };
  };
  failures: string[];
}

export interface PhysicalGpuConveyorPairedComparison {
  pairCount: number;
  aggregateTokensPerSecondDelta: PhysicalGpuCampaignMetricStats;
  perUserTokensPerSecondDelta: PhysicalGpuCampaignMetricStats;
  ttftMsDelta: PhysicalGpuCampaignMetricStats;
  tpotMsDelta: PhysicalGpuCampaignMetricStats;
  aggregateTokensPerSecondRatioGeometricMean: number | null;
  perUserTokensPerSecondRatioGeometricMean: number | null;
  ttftSpeedupGeometricMean: number | null;
  tpotSpeedupGeometricMean: number | null;
}

export interface PhysicalGpuConveyorAbReport
  extends PhysicalGpuConveyorAbReportBody {
  seal: {
    canonicalization: typeof PHYSICAL_GPU_CONVEYOR_AB_CANONICALIZATION;
    algorithm: "sha256";
    digest: string;
  };
}

export function buildPhysicalGpuConveyorSchedule(roundsValue: number): PhysicalGpuConveyorArm[] {
  const rounds = safeInteger(roundsValue, 1, MAX_ROUNDS, "rounds");
  const schedule: PhysicalGpuConveyorArm[] = [];
  for (let round = 0; round < rounds; round += 1) {
    schedule.push(
      ...(round % 2 === 0
        ? (["w1", "conveyor", "conveyor", "w1"] as const)
        : (["conveyor", "w1", "w1", "conveyor"] as const)),
    );
  }
  return schedule;
}

export function validatePhysicalGpuConveyorLaunchPair(
  baselineValue: unknown,
  conveyorValue: unknown,
): {
  baseline: PythonPipelineLaunchDescription;
  conveyor: PythonPipelineLaunchDescription;
  conveyorWaves: number;
  conveyorBytes: number;
} {
  validatePythonLaunchDescription(baselineValue);
  validatePythonLaunchDescription(conveyorValue);
  const baseline = structuredClone(baselineValue);
  const conveyor = structuredClone(conveyorValue);
  if (
    baseline.configuration.speculativeInflightWaves !== undefined
    || baseline.configuration.speculativeInflightBytes !== undefined
  ) {
    throw new Error("physical_gpu_conveyor_ab_baseline_must_be_w1_bytes0");
  }
  const conveyorWaves = conveyor.configuration.speculativeInflightWaves;
  const conveyorBytes = conveyor.configuration.speculativeInflightBytes;
  if (
    conveyorWaves === undefined
    || conveyorWaves <= 1
    || conveyorBytes === undefined
    || conveyorBytes <= 0
  ) {
    throw new Error("physical_gpu_conveyor_ab_conveyor_must_enable_complete_window");
  }
  if (baseline.launchId === conveyor.launchId) {
    throw new Error("physical_gpu_conveyor_ab_launch_ids_must_be_derived_and_distinct");
  }
  const normalizedBaseline = normalizeLaunchForPairComparison(baseline, "w1");
  const normalizedConveyor = normalizeLaunchForPairComparison(conveyor, "conveyor");
  if (
    canonicalEvidenceJson(normalizedBaseline)
    !== canonicalEvidenceJson(normalizedConveyor)
  ) {
    throw new Error("physical_gpu_conveyor_ab_launches_differ_outside_window_flags");
  }
  return { baseline, conveyor, conveyorWaves, conveyorBytes };
}

export async function runPhysicalGpuConveyorAb(
  inputValue: PhysicalGpuConveyorAbInput,
  dependencies: PhysicalGpuConveyorAbDependencies = {},
): Promise<PhysicalGpuConveyorAbReport> {
  const validated = validateInput(inputValue);
  const schedule = buildPhysicalGpuConveyorSchedule(validated.rounds);
  const now = dependencies.now ?? (() => new Date());
  const capturedAt = validTimestamp(now);
  const runCampaign = dependencies.runCampaign ?? runPhysicalGpuCampaign;
  const gate = dependencies.physicalGate;
  const healthTimeoutMs = safeInteger(
    dependencies.campaignDependencies?.timeouts?.apiHealthMs ?? 10_000,
    1,
    86_400_000,
    "health_timeout_ms",
  );
  const healthReader =
    dependencies.apiHealth
    ?? defaultApiHealthReader(
      dependencies.campaignDependencies?.fetch,
      healthTimeoutMs,
    );
  const runs: PhysicalGpuConveyorAbRun[] = [];
  const failures: string[] = [];

  for (let runIndex = 0; runIndex < schedule.length; runIndex += 1) {
    const arm = schedule[runIndex]!;
    const launch = arm === "w1" ? validated.baseline : validated.conveyor;
    const healthCapture: {
      before: PhysicalGpuConveyorHealthObservation | null;
      after: PhysicalGpuConveyorHealthObservation | null;
    } = { before: null, after: null };
    const runFailures: string[] = [];
    let campaign: PhysicalGpuCampaignObservation | null = null;
    let physicalGate: PhysicalGpuConveyorPhysicalGateResult = {
      passed: false,
      reportSealSha256: null,
      report: null,
    };

    try {
      campaign = await runCampaign(
        {
          launch,
          agents: validated.campaign.agents,
          apiBaseUrl: validated.campaign.apiBaseUrl,
          canaries: validated.campaign.canaries,
          warmups: validated.campaign.warmups,
          iterations: validated.campaign.iterations,
          concurrencies: validated.campaign.concurrencies,
        },
        {
          ...dependencies.campaignDependencies,
          supervisor: makeHealthCapturingSupervisorFactory(
            arm,
            runIndex,
            validated.campaign.apiBaseUrl,
            healthReader,
            healthTimeoutMs,
            healthCapture,
            dependencies.supervisor,
          ),
        },
      );
    } catch (error) {
      runFailures.push(`campaign_threw:${normalizeError(error).message}`);
    }

    if (healthCapture.before === null) {
      runFailures.push("health_before_missing");
    } else if (!healthCapture.before.passed) {
      runFailures.push(`health_before_failed:${healthCapture.before.error ?? "unknown"}`);
    }
    if (healthCapture.after === null) {
      runFailures.push("health_after_missing");
    } else if (!healthCapture.after.passed) {
      runFailures.push(`health_after_failed:${healthCapture.after.error ?? "unknown"}`);
    }
    if (campaign === null) {
      runFailures.push("campaign_missing");
    } else {
      if (!campaign.passed) runFailures.push("campaign_did_not_pass");
      if (!campaign.lifecycle.cleanupPassed) runFailures.push("cleanup_did_not_pass");
      if (gate === undefined) {
        runFailures.push("physical_gate_not_configured");
      } else {
        try {
          physicalGate = validatePhysicalGateResult(
            await gate(arm, launch, campaign, runIndex),
            launch,
            campaign,
          );
          if (!physicalGate.passed) runFailures.push("physical_gate_did_not_pass");
        } catch (error) {
          runFailures.push(`physical_gate_failed:${normalizeError(error).message}`);
        }
      }
    }

    const afterWindow = healthCapture.after?.speculativeWindow ?? null;
    const expectedWaves =
      arm === "w1" ? 1 : validated.conveyorWaves;
    const expectedBytes =
      arm === "w1" ? 0 : validated.conveyorBytes;
    for (const [phase, observed] of [
      ["before", healthCapture.before?.speculativeWindow ?? null],
      ["after", afterWindow],
    ] as const) {
      if (
        observed !== null
        && (
          observed.configuredWavesPerRequest !== expectedWaves
          || observed.configuredBytesPerRequest !== expectedBytes
          || observed.configured !== (arm === "conveyor")
        )
      ) {
        runFailures.push(`speculative_window_configuration_mismatch:${phase}`);
      }
    }
    const beforeWindow = healthCapture.before?.speculativeWindow ?? null;
    const beforeSpeculation = healthCapture.before?.speculation ?? null;
    if (
      beforeWindow !== null
      && !windowRuntimeCountersAreZero(beforeWindow)
    ) {
      runFailures.push("speculative_window_before_is_not_zero");
    }
    if (
      beforeSpeculation !== null
      && !speculationRuntimeCountersAreZero(beforeSpeculation)
    ) {
      runFailures.push("speculation_before_is_not_zero");
    }
    if (afterWindow !== null) {
      if (
        afterWindow.currentWaves !== 0
        || afterWindow.currentBytes !== 0
        || afterWindow.currentReservedBytes !== 0
      ) {
        runFailures.push("speculative_window_not_drained");
      }
      if (
        afterWindow.highWaterWaves > expectedWaves
        || afterWindow.maxRequestWaves > expectedWaves
      ) {
        runFailures.push("speculative_window_wave_cap_exceeded");
      }
      if (
        expectedBytes > 0
        && (
          afterWindow.highWaterBytes > expectedBytes
          || afterWindow.highWaterReservedBytes > expectedBytes
          || afterWindow.maxRequestBytes > expectedBytes
          || afterWindow.maxRequestReservedBytes > expectedBytes
        )
      ) {
        runFailures.push("speculative_window_byte_cap_exceeded");
      }
      if (afterWindow.dispatchedWaves !== afterWindow.completedWaves) {
        runFailures.push("speculative_window_completion_accounting_mismatch");
      }
      if (
        arm === "conveyor"
        && (
          afterWindow.highWaterWaves <= 1
          || afterWindow.maxRequestWaves <= 1
        )
      ) {
        runFailures.push("conveyor_high_water_did_not_exceed_one");
      }
    } else {
      runFailures.push("speculative_window_health_missing");
    }

    const run: PhysicalGpuConveyorAbRun = {
      runIndex,
      round: Math.floor(runIndex / 4),
      position: runIndex % 4,
      arm,
      launchId: launch.launchId,
      healthBefore: healthCapture.before,
      healthAfter: healthCapture.after,
      campaign,
      physicalGate,
      passed: runFailures.length === 0,
      failures: runFailures,
    };
    runs.push(run);
    failures.push(...runFailures.map((failure) => `run_${runIndex}:${failure}`));
    if (runFailures.some((failure) =>
      failure === "cleanup_did_not_pass"
      || failure.startsWith("campaign_threw:")
      || failure === "campaign_missing"
    )) {
      break;
    }
  }

  failures.push(...validateSourceIdentityAcrossRuns(runs));
  const parity = buildParityChecks(runs);
  if (parity.length === 0) failures.push("no_cross_arm_hashes_to_compare");
  for (const check of parity) {
    if (!check.passed) failures.push(`output_hash_mismatch:${check.key}`);
  }
  if (!runs.some((run) => run.arm === "w1" && run.physicalGate.passed)) {
    failures.push("baseline_physical_gate_missing");
  }
  if (!runs.some((run) => run.arm === "conveyor" && run.physicalGate.passed)) {
    failures.push("conveyor_physical_gate_missing");
  }
  if (runs.length !== schedule.length) failures.push("schedule_incomplete");
  const pairedComparison = buildPairedComparison(runs);
  if (pairedComparison.pairCount !== schedule.length / 2) {
    failures.push("paired_comparison_incomplete");
  }

  const evidence =
    runs.length === schedule.length
    && runs.every((run) => run.physicalGate.passed)
      ? "physical"
      : "unverified";
  const baselineSummary = summarizeArm("w1", runs);
  const conveyorSummary = summarizeArm("conveyor", runs);
  const body: PhysicalGpuConveyorAbReportBody = {
    schema: PHYSICAL_GPU_CONVEYOR_AB_SCHEMA,
    capturedAt,
    passed: failures.length === 0 && runs.every((run) => run.passed),
    evidence,
    launches: {
      baselineLaunchId: validated.baseline.launchId,
      conveyorLaunchId: validated.conveyor.launchId,
      baselineCanonicalSha256: sha256CanonicalEvidence(validated.baseline),
      conveyorCanonicalSha256: sha256CanonicalEvidence(validated.conveyor),
      baselineWaves: 1,
      baselineBytes: 0,
      conveyorWaves: validated.conveyorWaves,
      conveyorBytes: validated.conveyorBytes,
    },
    schedule,
    runs,
    parity,
    summary: {
      baseline: baselineSummary,
      conveyor: conveyorSummary,
      comparison: {
        aggregateTokensPerSecondPct: percentChange(
          baselineSummary.aggregateTokensPerSecond.mean,
          conveyorSummary.aggregateTokensPerSecond.mean,
        ),
        perUserTokensPerSecondPct: percentChange(
          baselineSummary.perUserTokensPerSecond.mean,
          conveyorSummary.perUserTokensPerSecond.mean,
        ),
        ttftP95Pct: percentChange(
          baselineSummary.ttftMs.p95,
          conveyorSummary.ttftMs.p95,
        ),
        tpotP95Pct: percentChange(
          baselineSummary.tpotMs.p95,
          conveyorSummary.tpotMs.p95,
        ),
        acceptancePoints:
          baselineSummary.acceptanceRate === null
          || conveyorSummary.acceptanceRate === null
            ? null
            : (conveyorSummary.acceptanceRate - baselineSummary.acceptanceRate) * 100,
        paired: pairedComparison,
      },
    },
    failures: [...new Set(failures)],
  };
  const report: PhysicalGpuConveyorAbReport = {
    ...body,
    seal: {
      canonicalization: PHYSICAL_GPU_CONVEYOR_AB_CANONICALIZATION,
      algorithm: "sha256",
      digest: sha256CanonicalEvidence(body),
    },
  };
  canonicalEvidenceJson(report);
  return report;
}

function validateInput(input: PhysicalGpuConveyorAbInput) {
  if (input === null || typeof input !== "object") {
    throw new Error("physical_gpu_conveyor_ab_input_is_invalid");
  }
  const pair = validatePhysicalGpuConveyorLaunchPair(
    input.baselineLaunch,
    input.conveyorLaunch,
  );
  const rounds = safeInteger(input.rounds, 1, MAX_ROUNDS, "rounds");
  if (input.campaign === null || typeof input.campaign !== "object") {
    throw new Error("physical_gpu_conveyor_ab_campaign_is_invalid");
  }
  return {
    ...pair,
    rounds,
    campaign: {
      agents: input.campaign.agents.map((binding) => ({
        nodeId: binding.nodeId,
        agent: binding.agent,
      })),
      apiBaseUrl: input.campaign.apiBaseUrl,
      canaries: structuredClone(input.campaign.canaries),
      warmups: input.campaign.warmups,
      iterations: input.campaign.iterations,
      concurrencies: [...input.campaign.concurrencies],
    },
  };
}

function normalizeLaunchForPairComparison(
  launch: PythonPipelineLaunchDescription,
  arm: PhysicalGpuConveyorArm,
): Omit<PythonPipelineLaunchDescription, "launchId"> {
  const copy = structuredClone(launch);
  const { launchId: _launchId, ...withoutLaunchId } = copy;
  delete withoutLaunchId.configuration.speculativeInflightWaves;
  delete withoutLaunchId.configuration.speculativeInflightBytes;
  const root = withoutLaunchId.launchOrder.find((process) => process.kind === "root-engine");
  if (root === undefined) throw new Error("physical_gpu_conveyor_ab_root_is_missing");
  const stripped = stripWindowArguments(root.command.args);
  root.command.args = stripped.args;
  if (arm === "w1" && stripped.found) {
    throw new Error("physical_gpu_conveyor_ab_baseline_command_contains_window_flags");
  }
  if (arm === "conveyor" && !stripped.found) {
    throw new Error("physical_gpu_conveyor_ab_conveyor_command_lacks_window_flags");
  }
  for (const process of withoutLaunchId.launchOrder) {
    if (process.kind !== "root-engine") {
      const unexpected = stripWindowArguments(process.command.args);
      if (unexpected.found) {
        throw new Error("physical_gpu_conveyor_ab_window_flags_must_be_root_only");
      }
    }
  }
  return withoutLaunchId;
}

function stripWindowArguments(args: readonly string[]): { args: string[]; found: boolean } {
  const result: string[] = [];
  let foundWaves = 0;
  let foundBytes = 0;
  for (let index = 0; index < args.length; index += 1) {
    const value = args[index]!;
    if (
      value === "--speculative-inflight-waves"
      || value === "--speculative-inflight-bytes"
    ) {
      if (args[index + 1] === undefined) {
        throw new Error("physical_gpu_conveyor_ab_window_flag_value_is_missing");
      }
      if (value.endsWith("waves")) foundWaves += 1;
      else foundBytes += 1;
      index += 1;
      continue;
    }
    result.push(value);
  }
  if (foundWaves > 1 || foundBytes > 1 || foundWaves !== foundBytes) {
    throw new Error("physical_gpu_conveyor_ab_window_flags_are_not_an_exact_pair");
  }
  return { args: result, found: foundWaves === 1 };
}

function makeHealthCapturingSupervisorFactory(
  arm: PhysicalGpuConveyorArm,
  runIndex: number,
  apiBaseUrl: string,
  healthReader: PhysicalGpuConveyorApiHealthReader,
  healthTimeoutMs: number,
  capture: {
    before: PhysicalGpuConveyorHealthObservation | null;
    after: PhysicalGpuConveyorHealthObservation | null;
  },
  factory:
    | ((
        launch: PythonPipelineLaunchDescription,
        options: LaunchSupervisorOptions,
      ) => PhysicalGpuCampaignSupervisor)
    | undefined,
) {
  return (
    launch: PythonPipelineLaunchDescription,
    options: LaunchSupervisorOptions,
  ): PhysicalGpuCampaignSupervisor => {
    const underlying =
      factory?.(launch, options)
      ?? new PythonLaunchSupervisor(launch, options);
    return {
      async start(signal) {
        const snapshot = await underlying.start(signal);
        capture.before = await observeHealth(
          "before",
          arm,
          runIndex,
          apiBaseUrl,
          healthReader,
          healthTimeoutMs,
        );
        return snapshot;
      },
      async stop(reason) {
        try {
          capture.after = await observeHealth(
            "after",
            arm,
            runIndex,
            apiBaseUrl,
            healthReader,
            healthTimeoutMs,
          );
        } finally {
          return await underlying.stop(reason);
        }
      },
      snapshot() {
        return underlying.snapshot();
      },
    };
  };
}

async function observeHealth(
  phase: PhysicalGpuConveyorHealthPhase,
  arm: PhysicalGpuConveyorArm,
  runIndex: number,
  apiBaseUrl: string,
  healthReader: PhysicalGpuConveyorApiHealthReader,
  healthTimeoutMs: number,
): Promise<PhysicalGpuConveyorHealthObservation> {
  try {
    const value = await withDeadline(
      healthReader(apiBaseUrl, phase, arm, runIndex),
      healthTimeoutMs,
    );
    canonicalEvidenceJson(value);
    const record = object(value, "health");
    return {
      phase,
      passed: record.status === "ready",
      status: typeof record.status === "string" ? record.status : null,
      canonicalSha256: sha256CanonicalEvidence(value),
      speculativeWindow: parseWindow(record.speculative_window),
      speculation: parseSpeculation(record.speculation),
      resources: parseResources(record),
      error: record.status === "ready" ? null : "api_health_not_ready",
    };
  } catch (error) {
    return {
      phase,
      passed: false,
      status: null,
      canonicalSha256: null,
      speculativeWindow: null,
      speculation: null,
      resources: emptyResources(),
      error: normalizeError(error).message,
    };
  }
}

function defaultApiHealthReader(
  fetchValue: PhysicalGpuCampaignDependencies["fetch"],
  timeoutMs: number,
): PhysicalGpuConveyorApiHealthReader {
  const fetchImpl = fetchValue ?? globalThis.fetch.bind(globalThis);
  return async (apiBaseUrl) => {
    const response = await fetchImpl(`${apiBaseUrl.replace(/\/$/, "")}/health`, {
      method: "GET",
      headers: { accept: "application/json" },
      signal: AbortSignal.timeout(timeoutMs),
    });
    if (!response.ok) throw new Error(`physical_gpu_conveyor_ab_health_http_${response.status}`);
    return await response.json();
  };
}

async function withDeadline<T>(
  value: T | Promise<T>,
  timeoutMs: number,
): Promise<T> {
  let timeout: NodeJS.Timeout | undefined;
  try {
    return await Promise.race([
      Promise.resolve(value),
      new Promise<never>((_resolve, reject) => {
        timeout = setTimeout(
          () => reject(new Error("physical_gpu_conveyor_ab_health_timeout")),
          timeoutMs,
        );
      }),
    ]);
  } finally {
    if (timeout !== undefined) clearTimeout(timeout);
  }
}

function parseWindow(value: unknown): PhysicalGpuConveyorWindowObservation | null {
  if (!isRecord(value)) return null;
  return {
    configured: value.configured === true,
    configuredWavesPerRequest: nonNegativeInteger(value.configured_waves_per_request),
    configuredBytesPerRequest: nonNegativeInteger(value.configured_bytes_per_request),
    currentWaves: nonNegativeInteger(value.current_waves),
    currentBytes: nonNegativeInteger(value.current_bytes),
    currentReservedBytes: nonNegativeInteger(value.current_reserved_bytes),
    highWaterWaves: nonNegativeInteger(value.high_water_waves),
    highWaterBytes: nonNegativeInteger(value.high_water_bytes),
    highWaterReservedBytes: nonNegativeInteger(value.high_water_reserved_bytes),
    maxRequestWaves: nonNegativeInteger(value.max_request_waves),
    maxRequestBytes: nonNegativeInteger(value.max_request_bytes),
    maxRequestReservedBytes: nonNegativeInteger(value.max_request_reserved_bytes),
    dispatchedWaves: nonNegativeInteger(value.dispatched_waves),
    completedWaves: nonNegativeInteger(value.completed_waves),
    committedWaves: nonNegativeInteger(value.committed_waves),
    condemnedWaves: nonNegativeInteger(value.condemned_waves),
    drainedWaves: nonNegativeInteger(value.drained_waves),
    rejectionCollapses: nonNegativeInteger(value.rejection_collapses),
    rejectedProposedTokens: nonNegativeInteger(value.rejected_proposed_tokens),
    condemnedProposedTokens: nonNegativeInteger(value.condemned_proposed_tokens),
    discardedProposedTokens: nonNegativeInteger(value.discarded_proposed_tokens),
    rejectedWaveBytes: nonNegativeInteger(value.rejected_wave_bytes),
    tombstoneBytes: nonNegativeInteger(value.tombstone_bytes),
    discardedBytes: nonNegativeInteger(value.discarded_bytes),
  };
}

function parseSpeculation(
  value: unknown,
): PhysicalGpuConveyorSpeculationObservation | null {
  if (!isRecord(value)) return null;
  const parsed = {
    configured: value.configured === true,
    proposedTokens: nullableNonNegativeInteger(value.proposed_tokens),
    acceptedTokens: nullableNonNegativeInteger(value.accepted_tokens),
    acceptanceRate: nullableFinite(value.acceptance_rate),
    verificationBytes: nullableNonNegativeInteger(value.verification_bytes),
  };
  if (
    (parsed.proposedTokens === null) !== (parsed.acceptedTokens === null)
    || (
      parsed.proposedTokens !== null
      && parsed.acceptedTokens !== null
      && parsed.acceptedTokens > parsed.proposedTokens
    )
    || (
      parsed.acceptanceRate !== null
      && (
        parsed.acceptanceRate < 0
        || parsed.acceptanceRate > 1
        || parsed.proposedTokens === null
        || parsed.acceptedTokens === null
        || parsed.proposedTokens === 0
        || Math.abs(
          parsed.acceptanceRate
          - parsed.acceptedTokens / parsed.proposedTokens
        ) > 1e-12
      )
    )
    || (
      parsed.proposedTokens !== null
      && parsed.proposedTokens > 0
      && parsed.acceptanceRate === null
    )
    || (
      parsed.proposedTokens === 0
      && parsed.acceptanceRate !== null
    )
  ) {
    throw new Error("physical_gpu_conveyor_ab_speculation_accounting_is_invalid");
  }
  return parsed;
}

function parseResources(record: Record<string, unknown>): PhysicalGpuConveyorResourceObservation {
  const resources = isRecord(record.resources) ? record.resources : {};
  return {
    observedPowerWatts: firstFinite(
      resources.observed_power_watts,
      resources.power_watts,
      record.observed_power_watts,
    ),
    utilizationPct: firstFinite(resources.utilization_pct, record.utilization_pct),
    temperatureC: firstFinite(resources.temperature_c, record.temperature_c),
    usedMemoryBytes: firstFinite(
      resources.used_memory_bytes,
      record.used_memory_bytes,
    ),
  };
}

function buildParityChecks(
  runs: readonly PhysicalGpuConveyorAbRun[],
): PhysicalGpuConveyorParityCheck[] {
  const byKey = new Map<string, { w1: Set<string>; conveyor: Set<string> }>();
  const add = (key: string, arm: PhysicalGpuConveyorArm, hash: string | null) => {
    if (hash === null) return;
    const entry = byKey.get(key) ?? { w1: new Set<string>(), conveyor: new Set<string>() };
    entry[arm].add(hash);
    byKey.set(key, entry);
  };
  for (const run of runs) {
    if (run.campaign === null) continue;
    for (const phase of ["pre", "post"] as const) {
      for (const canary of run.campaign.canaries[phase]) {
        add(`canary:${phase}:${canary.canaryId}`, run.arm, canary.evidence?.outputTokenIdsSha256 ?? null);
      }
    }
    for (const sample of run.campaign.samples) {
      add(
        `sample:${sample.phase}:c${sample.concurrency}:i${sample.iteration}:r${sample.requestIndex}:${sample.canaryId}`,
        run.arm,
        sample.outputTokenIdsSha256,
      );
    }
  }
  return [...byKey.entries()]
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([key, value]) => {
      const baselineSha256 = value.w1.size === 1 ? [...value.w1][0]! : null;
      const conveyorSha256 = value.conveyor.size === 1 ? [...value.conveyor][0]! : null;
      return {
        key,
        baselineSha256,
        conveyorSha256,
        passed:
          baselineSha256 !== null
          && conveyorSha256 !== null
          && baselineSha256 === conveyorSha256,
      };
    });
}

function windowRuntimeCountersAreZero(
  value: PhysicalGpuConveyorWindowObservation,
): boolean {
  return [
    value.currentWaves,
    value.currentBytes,
    value.currentReservedBytes,
    value.highWaterWaves,
    value.highWaterBytes,
    value.highWaterReservedBytes,
    value.maxRequestWaves,
    value.maxRequestBytes,
    value.maxRequestReservedBytes,
    value.dispatchedWaves,
    value.completedWaves,
    value.committedWaves,
    value.condemnedWaves,
    value.drainedWaves,
    value.rejectionCollapses,
    value.rejectedProposedTokens,
    value.condemnedProposedTokens,
    value.discardedProposedTokens,
    value.rejectedWaveBytes,
    value.tombstoneBytes,
    value.discardedBytes,
  ].every((counter) => counter === 0);
}

function speculationRuntimeCountersAreZero(
  value: PhysicalGpuConveyorSpeculationObservation,
): boolean {
  return (
    (value.proposedTokens === null || value.proposedTokens === 0)
    && (value.acceptedTokens === null || value.acceptedTokens === 0)
    && value.acceptanceRate === null
    && (value.verificationBytes === null || value.verificationBytes === 0)
  );
}

function buildPairedComparison(
  runs: readonly PhysicalGpuConveyorAbRun[],
): PhysicalGpuConveyorPairedComparison {
  const aggregateDeltas: number[] = [];
  const perUserDeltas: number[] = [];
  const ttftDeltas: number[] = [];
  const tpotDeltas: number[] = [];
  const aggregateRatios: number[] = [];
  const perUserRatios: number[] = [];
  const ttftSpeedups: number[] = [];
  const tpotSpeedups: number[] = [];
  let pairCount = 0;
  for (let index = 0; index + 1 < runs.length; index += 2) {
    const pair = [runs[index]!, runs[index + 1]!] as const;
    const baseline = pair.find((run) => run.arm === "w1");
    const conveyor = pair.find((run) => run.arm === "conveyor");
    if (
      baseline?.campaign === null
      || baseline?.campaign === undefined
      || conveyor?.campaign === null
      || conveyor?.campaign === undefined
    ) {
      continue;
    }
    const baselineMetrics = runComparisonMetrics(baseline.campaign);
    const conveyorMetrics = runComparisonMetrics(conveyor.campaign);
    if (
      baselineMetrics === null
      || conveyorMetrics === null
      || Object.values(baselineMetrics).some((value) => value <= 0)
      || Object.values(conveyorMetrics).some((value) => value <= 0)
    ) {
      continue;
    }
    pairCount += 1;
    aggregateDeltas.push(
      conveyorMetrics.aggregateTokensPerSecond
      - baselineMetrics.aggregateTokensPerSecond,
    );
    perUserDeltas.push(
      conveyorMetrics.perUserTokensPerSecond
      - baselineMetrics.perUserTokensPerSecond,
    );
    ttftDeltas.push(conveyorMetrics.ttftMs - baselineMetrics.ttftMs);
    tpotDeltas.push(conveyorMetrics.tpotMs - baselineMetrics.tpotMs);
    aggregateRatios.push(
      conveyorMetrics.aggregateTokensPerSecond
      / baselineMetrics.aggregateTokensPerSecond,
    );
    perUserRatios.push(
      conveyorMetrics.perUserTokensPerSecond
      / baselineMetrics.perUserTokensPerSecond,
    );
    ttftSpeedups.push(baselineMetrics.ttftMs / conveyorMetrics.ttftMs);
    tpotSpeedups.push(baselineMetrics.tpotMs / conveyorMetrics.tpotMs);
  }
  return {
    pairCount,
    aggregateTokensPerSecondDelta: metricStats(aggregateDeltas),
    perUserTokensPerSecondDelta: metricStats(perUserDeltas),
    ttftMsDelta: metricStats(ttftDeltas),
    tpotMsDelta: metricStats(tpotDeltas),
    aggregateTokensPerSecondRatioGeometricMean: geometricMean(aggregateRatios),
    perUserTokensPerSecondRatioGeometricMean: geometricMean(perUserRatios),
    ttftSpeedupGeometricMean: geometricMean(ttftSpeedups),
    tpotSpeedupGeometricMean: geometricMean(tpotSpeedups),
  };
}

function runComparisonMetrics(
  campaign: PhysicalGpuCampaignObservation,
): {
  aggregateTokensPerSecond: number;
  perUserTokensPerSecond: number;
  ttftMs: number;
  tpotMs: number;
} | null {
  const measured = campaign.samples.filter((sample) => sample.phase === "measure");
  const perUser = metricStats(
    measured.map((sample) => sample.perUserOutputTokensPerSecondIncludingTtft),
  ).mean;
  const aggregate =
    campaign.summary.aggregateOutputTokensPerSecondIncludingTtft;
  const ttft = campaign.summary.serverTtftMs.mean;
  const tpot = campaign.summary.serverTpotMs.mean;
  if (
    perUser === null
    || aggregate === null
    || ttft === null
    || tpot === null
    || ![perUser, aggregate, ttft, tpot].every(Number.isFinite)
  ) {
    return null;
  }
  return {
    aggregateTokensPerSecond: aggregate,
    perUserTokensPerSecond: perUser,
    ttftMs: ttft,
    tpotMs: tpot,
  };
}

function summarizeArm(
  arm: PhysicalGpuConveyorArm,
  runs: readonly PhysicalGpuConveyorAbRun[],
): PhysicalGpuConveyorArmSummary {
  const selected = runs.filter((run) => run.arm === arm && run.campaign !== null);
  const samples = selected.flatMap((run) =>
    run.campaign!.samples.filter((sample) => sample.phase === "measure"),
  );
  const afterHealth = selected
    .map((run) => run.healthAfter)
    .filter((health): health is PhysicalGpuConveyorHealthObservation => health !== null);
  const speculation = afterHealth
    .map((health) => health.speculation)
    .filter((value): value is PhysicalGpuConveyorSpeculationObservation => value !== null);
  const windows = afterHealth
    .map((health) => health.speculativeWindow)
    .filter((value): value is PhysicalGpuConveyorWindowObservation => value !== null);
  const proposedTokens = sumNullable(speculation.map((value) => value.proposedTokens));
  const acceptedTokens = sumNullable(speculation.map((value) => value.acceptedTokens));
  return {
    arm,
    runs: selected.length,
    measuredRequests: samples.length,
    completionTokens: samples.reduce(
      (total, sample) => total + (sample.completionTokens ?? 0),
      0,
    ),
    ttftMs: metricStats(samples.map((sample) => sample.serverTtftMs)),
    tpotMs: metricStats(samples.map((sample) => sample.serverTpotMs)),
    perUserTokensPerSecond: metricStats(
      samples.map((sample) => sample.perUserOutputTokensPerSecondIncludingTtft),
    ),
    aggregateTokensPerSecond: metricStats(
      selected.map(
        (run) =>
          run.campaign!.summary.aggregateOutputTokensPerSecondIncludingTtft,
      ),
    ),
    proposedTokens,
    acceptedTokens,
    acceptanceRate:
      proposedTokens !== null && proposedTokens > 0 && acceptedTokens !== null
        ? acceptedTokens / proposedTokens
        : null,
    highWaterWaves: maxNullable(windows.map((value) => value.highWaterWaves)),
    highWaterBytes: maxNullable(windows.map((value) => value.highWaterBytes)),
    highWaterReservedBytes: maxNullable(
      windows.map((value) => value.highWaterReservedBytes),
    ),
    rejectedProposedTokens: sumNullable(
      windows.map((value) => value.rejectedProposedTokens),
    ),
    condemnedProposedTokens: sumNullable(
      windows.map((value) => value.condemnedProposedTokens),
    ),
    discardedProposedTokens: sumNullable(
      windows.map((value) => value.discardedProposedTokens),
    ),
    rejectedWaveBytes: sumNullable(windows.map((value) => value.rejectedWaveBytes)),
    tombstoneBytes: sumNullable(windows.map((value) => value.tombstoneBytes)),
    discardedBytes: sumNullable(windows.map((value) => value.discardedBytes)),
    resources: {
      observedPowerWatts: metricStats(
        afterHealth.map((health) => health.resources.observedPowerWatts),
      ),
      utilizationPct: metricStats(
        afterHealth.map((health) => health.resources.utilizationPct),
      ),
      temperatureC: metricStats(
        afterHealth.map((health) => health.resources.temperatureC),
      ),
      usedMemoryBytes: metricStats(
        afterHealth.map((health) => health.resources.usedMemoryBytes),
      ),
    },
  };
}

function validateSourceIdentityAcrossRuns(
  runs: readonly PhysicalGpuConveyorAbRun[],
): string[] {
  const sourceIds = new Set<string>();
  for (const run of runs) {
    for (const phase of [
      ...(run.campaign?.lifecycle.agentHealthBefore ?? []),
      ...(run.campaign?.lifecycle.agentHealthAfter ?? []),
    ]) {
      const sourceId = phase.health?.buildIdentity.sourceId;
      if (sourceId !== undefined) sourceIds.add(sourceId);
    }
  }
  return sourceIds.size === 1
    ? []
    : ["agent_build_source_id_differs_across_arms"];
}

function validatePhysicalGateResult(
  value: PhysicalGpuConveyorPhysicalGateResult,
  launch: PythonPipelineLaunchDescription,
  campaign: PhysicalGpuCampaignObservation,
): PhysicalGpuConveyorPhysicalGateResult {
  const report = value?.report;
  if (
    value === null
    || typeof value !== "object"
    || typeof value.passed !== "boolean"
    || (
      value.reportSealSha256 !== null
      && !SHA256_PATTERN.test(value.reportSealSha256)
    )
    || (value.passed && value.reportSealSha256 === null)
    || (value.passed && report === null)
  ) {
    throw new Error("physical_gpu_conveyor_ab_physical_gate_result_is_invalid");
  }
  let validatedReport: PhysicalTwoHostGpuGateReportV1 | null = null;
  if (report !== null) {
    validatedReport = value.passed
      ? requirePassingPhysicalTwoHostGpuGate(report)
      : readPhysicalTwoHostGpuGateReport(report);
    if (
      validatedReport.seal.digest !== value.reportSealSha256
      || validatedReport.gate.passed !== value.passed
    ) {
      throw new Error("physical_gpu_conveyor_ab_physical_gate_report_is_invalid");
    }
    validatePhysicalGateCampaignBinding(validatedReport, launch, campaign);
  }
  return {
    passed: value.passed,
    reportSealSha256: value.reportSealSha256,
    report: validatedReport === null ? null : structuredClone(validatedReport),
  };
}

function validatePhysicalGateCampaignBinding(
  report: PhysicalTwoHostGpuGateReportV1,
  launch: PythonPipelineLaunchDescription,
  campaign: PhysicalGpuCampaignObservation,
): void {
  const launchDigest = sha256CanonicalEvidence(launch);
  if (
    report.launch.canonicalSha256 !== launchDigest
    || canonicalEvidenceJson(report.launch.description)
      !== canonicalEvidenceJson(launch)
  ) {
    throw new Error("physical_gpu_conveyor_ab_physical_gate_launch_mismatch");
  }

  const started = campaign.lifecycle.supervisorStarted;
  const stopped = campaign.lifecycle.supervisorStopped;
  const health = campaign.apiHealth;
  if (started === null || stopped === null || health === null) {
    throw new Error("physical_gpu_conveyor_ab_physical_gate_campaign_is_incomplete");
  }
  const sorted = (values: readonly string[]) => [...values].sort();
  if (
    canonicalEvidenceJson(sorted(report.lifecycle.readyProcessIds))
      !== canonicalEvidenceJson(sorted(started.processes.map((process) => process.processId)))
    || canonicalEvidenceJson(sorted(report.lifecycle.stoppedProcessIds))
      !== canonicalEvidenceJson(sorted(stopped.processes.map((process) => process.processId)))
    || canonicalEvidenceJson(report.lifecycle.health)
      !== canonicalEvidenceJson({
        status: health.status,
        model: health.model,
        stages: health.stages,
        boundaries: health.boundaries,
        codec: health.codec,
      })
  ) {
    throw new Error("physical_gpu_conveyor_ab_physical_gate_lifecycle_mismatch");
  }

  const campaignSamples = new Map(
    campaign.samples.map((sample) => [sample.sampleId, sample] as const),
  );
  if (campaignSamples.size !== campaign.samples.length) {
    throw new Error("physical_gpu_conveyor_ab_campaign_sample_ids_are_not_unique");
  }
  for (const sample of report.samples) {
    const observed = campaignSamples.get(sample.sampleId);
    if (
      observed === undefined
      || canonicalEvidenceJson(sample) !== canonicalEvidenceJson({
        sampleId: observed.sampleId,
        phase: observed.phase,
        concurrency: observed.concurrency,
        iteration: observed.iteration,
        promptTokens: observed.promptTokens,
        completionTokens: observed.completionTokens,
        outputTokenIdsHashScheme: "gdlp-output-token-ids-v1",
        outputTokenIdsSha256: observed.outputTokenIdsSha256,
        ttftMs: observed.clientFirstContentMs,
        tpotMs: observed.serverTpotMs,
        responseMs: observed.clientResponseMs,
        pipelineMs: observed.serverPipelineMs,
      })
    ) {
      throw new Error("physical_gpu_conveyor_ab_physical_gate_sample_mismatch");
    }
  }

  const referenceCanaryMatches = (phase: "pre" | "post") =>
    campaign.canaries[phase].some((canary) =>
      canary.passed
      && canary.error === null
      && canary.evidence !== null
      && canary.evidence.outputTokenIdsHashScheme === "gdlp-output-token-ids-v1"
      && canary.evidence.outputTokenIdsSha256 === report.reference.outputTokenIdsSha256
      && canary.evidence.completionTokens === report.reference.outputTokenIds.length
    );
  if (!referenceCanaryMatches("pre") || !referenceCanaryMatches("post")) {
    throw new Error("physical_gpu_conveyor_ab_physical_gate_reference_mismatch");
  }

  for (const host of report.hosts) {
    const healthMatches = campaign.lifecycle.agentHealthAfter.some((observation) =>
      observation.passed
      && observation.error === null
      && observation.expectedAgentId === host.agentId
      && observation.expectedNodeId === host.rankNodeId
      && observation.health !== null
      && canonicalEvidenceJson(observation.health.buildIdentity)
        === canonicalEvidenceJson(host.buildIdentity)
    );
    if (!healthMatches) {
      throw new Error("physical_gpu_conveyor_ab_physical_gate_host_mismatch");
    }
  }
}

function metricStats(values: readonly (number | null)[]): PhysicalGpuCampaignMetricStats {
  const finite = values
    .filter((value): value is number => value !== null && Number.isFinite(value))
    .sort((left, right) => left - right);
  if (finite.length === 0) {
    return { count: 0, mean: null, p50: null, p95: null, min: null, max: null };
  }
  return {
    count: finite.length,
    mean: finite.reduce((total, value) => total + value, 0) / finite.length,
    p50: percentile(finite, 0.5),
    p95: percentile(finite, 0.95),
    min: finite[0]!,
    max: finite.at(-1)!,
  };
}

function geometricMean(values: readonly number[]): number | null {
  if (
    values.length === 0
    || values.some((value) => !Number.isFinite(value) || value <= 0)
  ) {
    return null;
  }
  return Math.exp(
    values.reduce((total, value) => total + Math.log(value), 0) / values.length,
  );
}

function percentile(sorted: readonly number[], quantile: number): number {
  const rank = Math.ceil(quantile * sorted.length) - 1;
  return sorted[Math.max(0, rank)]!;
}

function percentChange(
  baseline: number | null,
  candidate: number | null,
): number | null {
  if (
    baseline === null
    || candidate === null
    || !Number.isFinite(baseline)
    || !Number.isFinite(candidate)
    || baseline === 0
  ) {
    return null;
  }
  return ((candidate - baseline) / baseline) * 100;
}

function sumNullable(values: readonly (number | null)[]): number | null {
  const finite = values.filter((value): value is number => value !== null);
  return finite.length === 0 ? null : finite.reduce((total, value) => total + value, 0);
}

function maxNullable(values: readonly number[]): number | null {
  return values.length === 0 ? null : Math.max(...values);
}

function object(value: unknown, name: string): Record<string, unknown> {
  if (!isRecord(value)) throw new Error(`physical_gpu_conveyor_ab_${name}_is_invalid`);
  return value;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function nonNegativeInteger(value: unknown): number {
  if (!Number.isSafeInteger(value) || (value as number) < 0) {
    throw new Error("physical_gpu_conveyor_ab_health_counter_is_invalid");
  }
  return value as number;
}

function nullableNonNegativeInteger(value: unknown): number | null {
  return value === null || value === undefined ? null : nonNegativeInteger(value);
}

function nullableFinite(value: unknown): number | null {
  if (value === null || value === undefined) return null;
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error("physical_gpu_conveyor_ab_health_metric_is_invalid");
  }
  return value;
}

function firstFinite(...values: unknown[]): number | null {
  for (const value of values) {
    if (typeof value === "number" && Number.isFinite(value)) return value;
  }
  return null;
}

function emptyResources(): PhysicalGpuConveyorResourceObservation {
  return {
    observedPowerWatts: null,
    utilizationPct: null,
    temperatureC: null,
    usedMemoryBytes: null,
  };
}

function safeInteger(value: unknown, minimum: number, maximum: number, name: string): number {
  if (
    !Number.isSafeInteger(value)
    || (value as number) < minimum
    || (value as number) > maximum
  ) {
    throw new Error(`physical_gpu_conveyor_ab_${name}_is_invalid`);
  }
  return value as number;
}

function validTimestamp(now: () => Date): string {
  const value = now();
  if (!(value instanceof Date) || !Number.isFinite(value.valueOf())) {
    throw new Error("physical_gpu_conveyor_ab_clock_is_invalid");
  }
  return value.toISOString();
}

function normalizeError(value: unknown): Error {
  return value instanceof Error ? value : new Error(String(value));
}
