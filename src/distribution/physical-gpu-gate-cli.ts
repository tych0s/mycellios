import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
  requirePassingPhysicalTwoHostGpuGate,
  type PhysicalTwoHostGpuGateReportV1,
} from "./physical-gpu-gate-report.js";

export const PHYSICAL_GPU_GATE_VERIFICATION_SCHEMA =
  "gdlp-physical-gpu-gate-verification/1" as const;

export interface PhysicalGpuGateCliOptions {
  reportPath: string;
}

export interface PhysicalGpuGateCliDependencies {
  readText?: (path: string) => Promise<string>;
  verify?: (source: string | unknown) => PhysicalTwoHostGpuGateReportV1;
  cwd?: string;
}

export function parsePhysicalGpuGateCliArguments(
  argumentsValue: readonly string[],
): PhysicalGpuGateCliOptions {
  if (!Array.isArray(argumentsValue)) {
    throw new Error("physical_gpu_gate_cli_arguments_are_invalid");
  }
  const values = new Map<string, string>();
  for (let index = 0; index < argumentsValue.length; index += 2) {
    const name = argumentsValue[index];
    const value = argumentsValue[index + 1];
    if (
      name !== "--report" ||
      value === undefined ||
      values.has(name) ||
      !isSafePathArgument(value)
    ) {
      throw new Error(`physical_gpu_gate_cli_argument_is_invalid:${name ?? "missing"}`);
    }
    values.set(name, value);
  }
  const reportPath = values.get("--report");
  if (reportPath === undefined) {
    throw new Error("physical_gpu_gate_cli_report_is_required");
  }
  return { reportPath };
}

export async function executePhysicalGpuGateCli(
  argumentsValue: readonly string[],
  dependencies: PhysicalGpuGateCliDependencies = {},
): Promise<string> {
  const options = parsePhysicalGpuGateCliArguments(argumentsValue);
  const readText = dependencies.readText ?? ((path: string) => readFile(path, "utf8"));
  const verify = dependencies.verify ?? requirePassingPhysicalTwoHostGpuGate;
  const cwd = dependencies.cwd ?? process.cwd();
  const source = await readText(resolve(cwd, options.reportPath));
  const report = verify(source);
  return `${JSON.stringify(
    {
      schema: PHYSICAL_GPU_GATE_VERIFICATION_SCHEMA,
      passed: true,
      reportSchema: report.schema,
      capturedAt: report.capturedAt,
      seal: report.seal.digest,
      measuredSamples: report.summary.measuredSamples,
      measuredCompletionTokens: report.summary.measuredCompletionTokens,
      ttftP50Ms: report.summary.ttftMs.p50,
      tpotP50Ms: report.summary.tpotMs.p50,
      outputTokensPerSecondP50IncludingTtft:
        report.summary.outputTokensPerSecondIncludingTtft.p50,
    },
    null,
    2,
  )}\n`;
}

function isSafePathArgument(value: unknown): value is string {
  return (
    typeof value === "string" &&
    value.trim().length > 0 &&
    value.length <= 32_768 &&
    !/[\0\r\n]/.test(value)
  );
}

function renderError(value: unknown): string {
  const error = value instanceof Error ? value : new Error(String(value));
  return JSON.stringify({
    schema: PHYSICAL_GPU_GATE_VERIFICATION_SCHEMA,
    passed: false,
    error: error.message,
  });
}

const entryPath = process.argv[1];
if (
  entryPath !== undefined &&
  resolve(fileURLToPath(import.meta.url)) === resolve(entryPath)
) {
  void executePhysicalGpuGateCli(process.argv.slice(2)).then(
    (output) => process.stdout.write(output),
    (error) => {
      process.stderr.write(`${renderError(error)}\n`);
      process.exitCode = 1;
    },
  );
}
