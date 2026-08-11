import { redactDiagnosticText } from "../contracts/remote-diagnostics.js";
import type { LaunchCapturedOutput, LaunchProcessHandle } from "../distribution/launch-supervisor.js";

const MAX_DIAGNOSTIC_LENGTH = 800;
const MAX_DIAGNOSTIC_LINES = 4;

export function stageStartupDiagnostic(
  handle: LaunchProcessHandle | null,
): string | null {
  if (!handle?.output) return null;
  let output: LaunchCapturedOutput;
  try {
    output = handle.output();
  } catch {
    return null;
  }
  const lines = `${output.stderr}\n${output.stdout}`
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0)
    .filter((line) => !line.startsWith("Fetching "))
    .slice(-MAX_DIAGNOSTIC_LINES)
    .map(safeDiagnosticLine)
    .filter((line) => line.length > 0);
  if (lines.length === 0) return null;
  return lines.join(" | ").slice(-MAX_DIAGNOSTIC_LENGTH);
}

function safeDiagnosticLine(value: string): string {
  return redactDiagnosticText(value)
    .replace(/\b[A-Za-z]:\\(?:[^\\/:*?"<>|\r\n]+\\)*[^\\/:*?"<>|\r\n]*/g, "[PATH]")
    .replace(/(?<![A-Za-z0-9._-])\/(?:[^/\s]+\/)+[^/\s]*/g, "[PATH]")
    .replace(/\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b/gi, "[EMAIL]")
    .replace(/\s+/g, " ")
    .trim();
}
