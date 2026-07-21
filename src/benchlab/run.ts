import {
  compareRunWithHistory,
  createRunIdentity,
  loadBenchmarkRuns,
  saveBenchmarkRun,
} from "./history.js";
import { runRealRuntimeSuite, type RealRuntimeOptions } from "./real-suite.js";
import type { BenchmarkRun } from "./types.js";

export interface BenchmarkRunOptions extends Omit<RealRuntimeOptions, "cwd"> {
  cwd: string;
  version?: string;
  label?: string;
  historyDirectory?: string;
}

export async function runAndPersistRealSuite(options: BenchmarkRunOptions): Promise<{
  run: BenchmarkRun;
  path: string;
}> {
  const history = loadBenchmarkRuns(options.cwd, options.historyDirectory);
  const identity = createRunIdentity(options.cwd, options.version, options.label);
  const run = compareRunWithHistory(await runRealRuntimeSuite(identity, options), history);
  const path = saveBenchmarkRun(options.cwd, run, options.historyDirectory);
  return { run, path };
}
