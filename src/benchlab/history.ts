import { execFileSync } from "node:child_process";
import { randomUUID } from "node:crypto";
import { mkdirSync, readFileSync, readdirSync, renameSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import {
  benchmarkScenarioCanBeCompared,
  benchmarkScenarioFingerprint,
  sealBenchmarkScenario,
} from "./scenario.js";
import {
  BENCHMARK_RUN_SCHEMA,
  DEFAULT_BENCHMARK_THRESHOLDS,
  LEGACY_BENCHMARK_RUN_SCHEMA,
  type BenchmarkMeasurement,
  type BenchmarkRun,
  type BenchmarkThresholds,
} from "./types.js";

export const DEFAULT_HISTORY_DIRECTORY = join("benchmarks", "history");

export interface RunIdentity {
  runId: string;
  version: string;
  label: string;
  gitCommit: string | null;
  gitBranch: string | null;
  gitDirty: boolean | null;
  build: BenchmarkRun["build"];
}

export function createRunIdentity(
  cwd: string,
  versionOverride?: string,
  label?: string,
  environment: NodeJS.ProcessEnv = process.env,
): RunIdentity {
  const release = resolveRelease(cwd, versionOverride, environment);
  const revision = resolveRevision(cwd, environment);
  const gitBranchResult = git(cwd, ["branch", "--show-current"]);
  const gitStatusResult = git(cwd, ["status", "--porcelain"]);
  const gitBranch = firstText(
    environment.MYCELLIOS_BRANCH,
    environment.GITHUB_REF_NAME,
    gitBranchResult.ok ? gitBranchResult.output : null,
  );
  const gitDirty = gitStatusResult.ok ? gitStatusResult.output.length > 0 : null;
  const shortCommit = revision.value?.slice(0, 8) ?? "revision-unknown";
  const timestamp = new Date().toISOString().replaceAll(":", "-").replace(".", "-");
  return {
    runId: `${timestamp}-${shortCommit}-${randomUUID().slice(0, 8)}`,
    version: release.value,
    label: label?.trim() || `v${release.value} · ${shortCommit}`,
    gitCommit: revision.value,
    gitBranch,
    gitDirty,
    build: {
      release: release.value,
      releaseSource: release.source,
      revision: revision.value,
      revisionSource: revision.source,
    },
  };
}

export function loadBenchmarkRuns(
  cwd: string,
  directory = DEFAULT_HISTORY_DIRECTORY,
): BenchmarkRun[] {
  const absolute = resolve(cwd, directory);
  let files: string[];
  try {
    files = readdirSync(absolute).filter((file) => file.endsWith(".json"));
  } catch {
    return [];
  }
  return files
    .map((file) => {
      try {
        return parseBenchmarkRun(JSON.parse(readFileSync(join(absolute, file), "utf8")) as unknown);
      } catch {
        return null;
      }
    })
    .filter((run): run is BenchmarkRun => run !== null)
    .sort((left, right) => left.finishedAt.localeCompare(right.finishedAt));
}

export function saveBenchmarkRun(
  cwd: string,
  run: BenchmarkRun,
  directory = DEFAULT_HISTORY_DIRECTORY,
): string {
  if (!isCurrentBenchmarkRun(run)) throw new Error("The benchmark run is not valid.");
  const absolute = resolve(cwd, directory);
  mkdirSync(absolute, { recursive: true });
  const destination = join(absolute, `${safeFilePart(run.runId)}.json`);
  const temporary = `${destination}.tmp-${process.pid}`;
  writeFileSync(temporary, `${JSON.stringify(run, null, 2)}\n`, "utf8");
  renameSync(temporary, destination);
  return destination;
}

export function compareRunWithHistory(
  run: BenchmarkRun,
  history: BenchmarkRun[],
  thresholds: BenchmarkThresholds = DEFAULT_BENCHMARK_THRESHOLDS,
): BenchmarkRun {
  const measurements = run.measurements.map((measurement) =>
    compareMeasurement(measurement, run, history, thresholds),
  );
  const status = measurements.some((measurement) => measurement.status === "failed")
    ? "failed"
    : measurements.some((measurement) => measurement.status === "regression")
      ? "regression"
      : measurements.some((measurement) => measurement.status === "inconclusive")
        ? "inconclusive"
      : measurements.every((measurement) => measurement.status === "baseline")
        ? "baseline"
        : "passed";
  return { ...run, status, measurements };
}

function compareMeasurement(
  measurement: BenchmarkMeasurement,
  currentRun: BenchmarkRun,
  history: BenchmarkRun[],
  thresholds: BenchmarkThresholds,
): BenchmarkMeasurement {
  if (measurement.status === "failed" || measurement.status === "inconclusive") {
    return measurement;
  }
  if (!benchmarkScenarioCanBeCompared(measurement)) {
    return {
      ...measurement,
      status: "baseline",
      comparison: {
        baselineRunId: null,
        tokensPerSecondPct: null,
        ttftP95Pct: null,
        acceptancePoints: null,
        reasons: ["El digest del modelo no está disponible; comparación bloqueada."],
      },
    };
  }
  const baseline = history
    .filter((run) => run.finishedAt < currentRun.finishedAt)
    .flatMap((run) => run.measurements.map((candidate) => ({ run, candidate })))
    .filter(
      ({ candidate }) =>
        benchmarkScenarioCanBeCompared(candidate)
        && candidate.scenarioFingerprint === measurement.scenarioFingerprint,
    )
    .at(-1);
  if (!baseline) return { ...measurement, status: "baseline" };

  const tokensDelta = percentChange(
    measurement.metrics.tokensPerSecond,
    baseline.candidate.metrics.tokensPerSecond,
  );
  const ttftDelta = percentChange(
    measurement.metrics.ttftMsP95,
    baseline.candidate.metrics.ttftMsP95,
  );
  const acceptanceDelta = pointsChange(
    measurement.metrics.acceptanceRate,
    baseline.candidate.metrics.acceptanceRate,
  );
  const reasons: string[] = [];
  if (tokensDelta !== null && tokensDelta < -thresholds.tokensPerSecondRegressionPct) {
    reasons.push(`tokens/s bajó ${Math.abs(tokensDelta).toFixed(1)}%`);
  }
  if (ttftDelta !== null && ttftDelta > thresholds.ttftP95RegressionPct) {
    reasons.push(`TTFT P95 subió ${ttftDelta.toFixed(1)}%`);
  }
  if (acceptanceDelta !== null && acceptanceDelta < -thresholds.acceptanceRegressionPoints) {
    reasons.push(`aceptación bajó ${Math.abs(acceptanceDelta).toFixed(1)} puntos`);
  }
  return {
    ...measurement,
    status: reasons.length > 0 ? "regression" : "passed",
    comparison: {
      baselineRunId: baseline.run.runId,
      tokensPerSecondPct: tokensDelta,
      ttftP95Pct: ttftDelta,
      acceptancePoints: acceptanceDelta,
      reasons,
    },
  };
}

function percentChange(current: number | null, previous: number | null): number | null {
  if (current === null || previous === null || previous === 0) return null;
  return round(((current / previous) - 1) * 100, 2);
}

function pointsChange(current: number | null, previous: number | null): number | null {
  if (current === null || previous === null) return null;
  return round((current - previous) * 100, 2);
}

interface CommandResult {
  ok: boolean;
  output: string;
}

function git(cwd: string, args: string[]): CommandResult {
  try {
    const output = execFileSync("git", args, {
      cwd,
      encoding: "utf8",
      windowsHide: true,
      stdio: ["ignore", "pipe", "ignore"],
    }).trim();
    return { ok: true, output };
  } catch {
    return { ok: false, output: "" };
  }
}

function safeFilePart(value: string): string {
  return value.replaceAll(/[^A-Za-z0-9._-]/g, "-").slice(0, 180);
}

function round(value: number, decimals: number): number {
  const factor = 10 ** decimals;
  return Math.round(value * factor) / factor;
}

export function parseBenchmarkRun(value: unknown): BenchmarkRun | null {
  if (isCurrentBenchmarkRun(value)) return value;
  if (!isRecord(value) || value.schema !== LEGACY_BENCHMARK_RUN_SCHEMA) return null;
  try {
    return migrateLegacyBenchmarkRun(value);
  } catch {
    return null;
  }
}

function isCurrentBenchmarkRun(value: unknown): value is BenchmarkRun {
  if (!value || typeof value !== "object") return false;
  const record = value as Record<string, unknown>;
  return (
    record.schema === BENCHMARK_RUN_SCHEMA &&
    typeof record.runId === "string" &&
    typeof record.version === "string" &&
    typeof record.finishedAt === "string" &&
    isRecord(record.build) &&
    record.build.release === record.version &&
    record.build.revision === record.gitCommit &&
    (record.gitDirty === null || typeof record.gitDirty === "boolean") &&
    Array.isArray(record.measurements) &&
    record.measurements.every(isCurrentBenchmarkMeasurement)
  );
}

function isCurrentBenchmarkMeasurement(value: unknown): value is BenchmarkMeasurement {
  if (
    !isRecord(value)
    || typeof value.scenarioFingerprint !== "string"
    || !isRecord(value.model)
    || (value.model.digest !== null && typeof value.model.digest !== "string")
    || !isRecord(value.inventory)
    || !Array.isArray(value.inventory.profiles)
    || !isRecord(value.topology)
    || !isRecord(value.workload)
  ) return false;
  try {
    return benchmarkScenarioFingerprint(
      value as unknown as BenchmarkMeasurement,
    ) === value.scenarioFingerprint;
  } catch {
    return false;
  }
}

function migrateLegacyBenchmarkRun(record: Record<string, unknown>): BenchmarkRun {
  if (
    typeof record.runId !== "string"
    || typeof record.version !== "string"
    || typeof record.label !== "string"
    || typeof record.startedAt !== "string"
    || typeof record.finishedAt !== "string"
    || !Array.isArray(record.measurements)
  ) {
    throw new Error("legacy_benchmark_run_invalid");
  }
  const revision = nullableText(record.gitCommit);
  const gitBranch = nullableText(record.gitBranch);
  const legacyDirty = typeof record.gitDirty === "boolean" ? record.gitDirty : null;
  const gitDirty = revision === null && gitBranch === null ? null : legacyDirty;
  const measurements = record.measurements.map((value) => migrateLegacyMeasurement(value));
  return {
    ...(record as unknown as Omit<BenchmarkRun, "schema" | "gitDirty" | "build" | "measurements">),
    schema: BENCHMARK_RUN_SCHEMA,
    gitDirty,
    build: {
      release: record.version,
      releaseSource: "package",
      revision,
      revisionSource: revision ? "git" : "unknown",
    },
    measurements,
  };
}

function migrateLegacyMeasurement(value: unknown): BenchmarkMeasurement {
  if (
    !isRecord(value)
    || !isRecord(value.model)
    || !isRecord(value.inventory)
    || !Array.isArray(value.inventory.profiles)
    || !isRecord(value.workload)
  ) {
    throw new Error("legacy_benchmark_measurement_invalid");
  }
  const nodeIds = value.inventory.profiles.flatMap((profile) =>
    isRecord(profile) && typeof profile.nodeId === "string" ? [profile.nodeId] : []
  );
  const routeClasses = Array.isArray(value.workload.routeClasses)
    ? value.workload.routeClasses.filter((item): item is string => typeof item === "string")
    : [];
  return sealBenchmarkScenario({
    ...(value as unknown as Omit<BenchmarkMeasurement, "scenarioFingerprint" | "model" | "topology">),
    model: {
      ...(value.model as unknown as Omit<BenchmarkMeasurement["model"], "digest">),
      digest: null,
    },
    topology: {
      digest: null,
      stageCount: nodeIds.length > 0 ? nodeIds.length : null,
      boundaries: [],
      nodeIds,
      routeClasses,
    },
  });
}

function resolveRelease(
  cwd: string,
  override: string | undefined,
  environment: NodeJS.ProcessEnv,
): { value: string; source: BenchmarkRun["build"]["releaseSource"] } {
  const explicit = override?.trim();
  if (explicit) return { value: explicit, source: "override" };
  const configured = firstText(
    environment.MYCELLIOS_RELEASE,
    environment.MYCELLIOS_VERSION,
    environment.npm_package_version,
  );
  if (configured) return { value: configured, source: "environment" };
  try {
    const packageDocument = JSON.parse(readFileSync(join(cwd, "package.json"), "utf8")) as {
      version?: unknown;
    };
    if (typeof packageDocument.version === "string" && packageDocument.version.trim()) {
      return { value: packageDocument.version.trim(), source: "package" };
    }
  } catch {
    // A standalone artifact may not include package.json.
  }
  return { value: "0.0.0-unknown", source: "unknown" };
}

function resolveRevision(
  cwd: string,
  environment: NodeJS.ProcessEnv,
): { value: string | null; source: BenchmarkRun["build"]["revisionSource"] } {
  const configured = firstRevision(
    environment.MYCELLIOS_REVISION,
    environment.REVISION,
    environment.GITHUB_SHA,
    environment.SOURCE_VERSION,
    environment.RENDER_GIT_COMMIT,
    environment.COMMIT_SHA,
  );
  if (configured) return { value: configured, source: "environment" };
  try {
    const revision = normalizedRevision(readFileSync(join(cwd, "REVISION"), "utf8"));
    if (revision) return { value: revision, source: "revision-file" };
  } catch {
    // Source deployments may not carry a REVISION file.
  }
  const gitRevision = git(cwd, ["rev-parse", "HEAD"]);
  const revision = gitRevision.ok ? normalizedRevision(gitRevision.output) : null;
  return revision
    ? { value: revision, source: "git" }
    : { value: null, source: "unknown" };
}

function firstRevision(...values: Array<string | undefined>): string | null {
  for (const value of values) {
    const revision = normalizedRevision(value);
    if (revision) return revision;
  }
  return null;
}

function normalizedRevision(value: string | undefined): string | null {
  const normalized = value?.trim().toLowerCase();
  return normalized && /^[0-9a-f]{7,64}$/.test(normalized) ? normalized : null;
}

function firstText(...values: Array<string | null | undefined>): string | null {
  for (const value of values) {
    const normalized = value?.trim();
    if (normalized) return normalized;
  }
  return null;
}

function nullableText(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}
