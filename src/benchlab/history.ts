import { execFileSync } from "node:child_process";
import { randomUUID } from "node:crypto";
import { mkdirSync, readFileSync, readdirSync, renameSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import {
  BENCHMARK_RUN_SCHEMA,
  DEFAULT_BENCHMARK_THRESHOLDS,
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
  gitDirty: boolean;
}

export function createRunIdentity(cwd: string, versionOverride?: string, label?: string): RunIdentity {
  const packageDocument = JSON.parse(readFileSync(join(cwd, "package.json"), "utf8")) as {
    version?: unknown;
  };
  const version = versionOverride?.trim() || String(packageDocument.version ?? "0.0.0-dev");
  const gitCommit = git(cwd, ["rev-parse", "HEAD"]);
  const shortCommit = gitCommit?.slice(0, 8) ?? "uncommitted";
  const timestamp = new Date().toISOString().replaceAll(":", "-").replace(".", "-");
  return {
    runId: `${timestamp}-${shortCommit}-${randomUUID().slice(0, 8)}`,
    version,
    label: label?.trim() || `v${version} · ${shortCommit}`,
    gitCommit,
    gitBranch: git(cwd, ["branch", "--show-current"]),
    gitDirty: (git(cwd, ["status", "--porcelain"]) ?? "").length > 0,
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
        const run = JSON.parse(readFileSync(join(absolute, file), "utf8")) as unknown;
        return isBenchmarkRun(run) ? run : null;
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
  if (!isBenchmarkRun(run)) throw new Error("The benchmark run is not valid.");
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
  if (measurement.status === "failed") return measurement;
  const baseline = history
    .filter((run) => run.finishedAt < currentRun.finishedAt)
    .flatMap((run) => run.measurements.map((candidate) => ({ run, candidate })))
    .filter(
      ({ candidate }) =>
        candidate.id === measurement.id && candidate.evidence === measurement.evidence,
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

function git(cwd: string, args: string[]): string | null {
  try {
    const value = execFileSync("git", args, {
      cwd,
      encoding: "utf8",
      windowsHide: true,
      stdio: ["ignore", "pipe", "ignore"],
    }).trim();
    return value || null;
  } catch {
    return null;
  }
}

function safeFilePart(value: string): string {
  return value.replaceAll(/[^A-Za-z0-9._-]/g, "-").slice(0, 180);
}

function round(value: number, decimals: number): number {
  const factor = 10 ** decimals;
  return Math.round(value * factor) / factor;
}

function isBenchmarkRun(value: unknown): value is BenchmarkRun {
  if (!value || typeof value !== "object") return false;
  const record = value as Record<string, unknown>;
  return (
    record.schema === BENCHMARK_RUN_SCHEMA &&
    typeof record.runId === "string" &&
    typeof record.version === "string" &&
    typeof record.finishedAt === "string" &&
    Array.isArray(record.measurements)
  );
}
