import type { LaunchAgentStartRequest } from "../distribution/launch-supervisor.js";
import { validateExecutorIsolationPolicy } from "../distribution/process-environment.js";

/** Lifecycle errors must fit the protocol while retaining Python's final exception. */
export function runtimeFailureSummary(error: unknown): string {
  const message = error instanceof Error ? error.message : String(error);
  if (message.length <= 2_048) return message;
  const marker = "\n...[truncated]...\n";
  return message.slice(0, 512) + marker + message.slice(-(2_048 - 512 - marker.length));
}

export function isLaunchAgentStartRequest(value: unknown): value is LaunchAgentStartRequest {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const request = value as Record<string, unknown>;
  if (
    typeof request.launchId !== "string" ||
    typeof request.pipelineId !== "string" ||
    !Number.isSafeInteger(request.deploymentGeneration) ||
    Number(request.deploymentGeneration) < 0 ||
    typeof request.nodeId !== "string" ||
    !request.process ||
    typeof request.process !== "object" ||
    Array.isArray(request.process)
  ) return false;
  const process = request.process as Record<string, unknown>;
  const anchor = process.anchor;
  if (
    typeof process.processId !== "string"
    || !anchor
    || typeof anchor !== "object"
    || Array.isArray(anchor)
    || (anchor as Record<string, unknown>).memberId !== request.nodeId
  ) {
    return false;
  }
  try {
    validateExecutorIsolationPolicy(process.isolation);
    return true;
  } catch {
    return false;
  }
}
