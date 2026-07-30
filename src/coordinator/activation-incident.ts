export const ACTIVATION_INCIDENT_SCHEMA =
  "mycellios-activation-incident/1" as const;

export type ActivationIncidentCode =
  | "executor_pool_not_ready"
  | "node_disconnected"
  | "launch_agent_unavailable"
  | "network_prepare_timeout"
  | "accelerator_not_ready"
  | "stage_readiness_timeout"
  | "native_memory_access"
  | "out_of_memory"
  | "stage_process_exited"
  | "model_incompatible"
  | "artifact_unavailable"
  | "unknown";

export type ActivationIncidentScope =
  | "node"
  | "network"
  | "accelerator"
  | "stage"
  | "model"
  | "artifact";

export type ActivationRepairState =
  | "scheduled"
  | "retrying"
  | "exhausted"
  | "manual_required";

export type ActivationAutomaticAction =
  | "rebuild_route"
  | "reconnect_node"
  | "wait_for_verified_accelerator"
  | "reprofile_and_replan"
  | "none";

export interface ActivationIncident {
  schema: typeof ACTIVATION_INCIDENT_SCHEMA;
  code: ActivationIncidentCode;
  scope: ActivationIncidentScope;
  title: string;
  summary: string;
  remedy: string;
  steps: readonly string[];
  retryable: boolean;
  automatic: boolean;
  repairState: ActivationRepairState;
  automaticAction: ActivationAutomaticAction;
  attempt: number;
  maximumAttempts: number;
  nextRetryAt: string | null;
  nodeId: string | null;
  stageId: string | null;
  processExitCode: number | null;
}

export interface ActivationIncidentInput {
  message: string;
  retryCount?: number;
  retryLaunching?: boolean;
  nextRetryAt?: number | null;
  maximumAttempts?: number;
}

interface IncidentDefinition {
  code: ActivationIncidentCode;
  scope: ActivationIncidentScope;
  title: string;
  summary: string;
  remedy: string;
  steps: readonly string[];
  automaticAction: ActivationAutomaticAction;
  intrinsicallyRetryable: boolean;
}

const ROUTE_REBUILD_STEPS = Object.freeze([
  "Retire any partial stage without publishing it.",
  "Recalculate placement from currently verified capacity.",
  "Run health checks and a real inference canary before publication.",
]);

export function classifyActivationIncident(
  input: ActivationIncidentInput,
): ActivationIncident {
  const maximumAttempts = boundedInteger(input.maximumAttempts, 0, 32);
  const exhausted = unwrapExhausted(input.message);
  const message = exhausted.message;
  const definition = incidentDefinition(message);
  const retryable = exhausted.attempts === null
    && definition.intrinsicallyRetryable
    && maximumAttempts > 0;
  const scheduled = retryable && input.nextRetryAt !== null
    && input.nextRetryAt !== undefined;
  const retrying = retryable && input.retryLaunching === true;
  const repairState: ActivationRepairState = exhausted.attempts !== null
    ? "exhausted"
    : retrying
      ? "retrying"
      : scheduled
        ? "scheduled"
        : "manual_required";
  const attempt = exhausted.attempts
    ?? (
      scheduled || retrying
        ? Math.min(
            maximumAttempts,
            boundedInteger(input.retryCount, 0, maximumAttempts)
              + (retrying ? 0 : 1),
          )
        : 0
    );
  const nextRetryAt = repairState === "scheduled"
    ? safeIsoTimestamp(input.nextRetryAt)
    : null;

  return Object.freeze({
    schema: ACTIVATION_INCIDENT_SCHEMA,
    code: definition.code,
    scope: definition.scope,
    title: definition.title,
    summary: definition.summary,
    remedy: exhausted.attempts === null
      ? definition.remedy
      : "Automatic retries reached their safe limit. The failed topology remains unpublished and requires a new verified capacity or runtime change.",
    steps: definition.steps,
    retryable,
    automatic: repairState === "scheduled" || repairState === "retrying",
    repairState,
    automaticAction: retryable ? definition.automaticAction : "none",
    attempt,
    maximumAttempts,
    nextRetryAt,
    nodeId: extractNodeId(message),
    stageId: extractStageId(message),
    processExitCode: extractExitCode(message),
  });
}

export function activationFailureIsTransient(message: string): boolean {
  const exhausted = unwrapExhausted(message);
  return exhausted.attempts === null
    && incidentDefinition(exhausted.message).intrinsicallyRetryable;
}

export function formatExhaustedActivationFailure(
  attempts: number,
  runtimeVersion: string,
  message: string,
): string {
  const safeRuntimeVersion = runtimeVersion
    .trim()
    .replace(/[^0-9A-Za-z._+-]/g, "_")
    .slice(0, 64) || "unknown";
  return `automatic_activation_retries_exhausted:${boundedInteger(attempts, 0, 32)}:runtime=${safeRuntimeVersion}:${message}`;
}

export function activationFailureCanRetryAfterRuntimeChange(
  message: string,
  runtimeVersion: string,
): boolean {
  return activationFailureMessageAfterRuntimeChange(message, runtimeVersion) !== null;
}

export function activationFailureMessageAfterRuntimeChange(
  message: string,
  runtimeVersion: string,
): string | null {
  const exhausted = unwrapExhausted(message);
  return exhausted.attempts !== null
    && exhausted.runtimeVersion !== runtimeVersion
    && incidentDefinition(exhausted.message).intrinsicallyRetryable
    ? exhausted.message
    : null;
}

function incidentDefinition(message: string): IncidentDefinition {
  const normalized = message.toLowerCase();
  if (normalized.includes("distributed_activation_requires_two_connected_shard_executors")) {
    return {
      code: "executor_pool_not_ready",
      scope: "network",
      title: "The verified executor topology is not ready",
      summary: "Connected capacity is visible, but fewer than two executors have completed the runtime and reciprocal-link evidence required to launch a distributed route.",
      remedy: "Mycellios keeps the model unpublished while the connected desktops finish verification, then rebuilds the route automatically.",
      steps: Object.freeze([
        "Keep both contributing desktop clients online.",
        "Wait for their runtime performance and reciprocal link probes to complete.",
        ...ROUTE_REBUILD_STEPS,
      ]),
      automaticAction: "rebuild_route",
      intrinsicallyRetryable: true,
    };
  }
  if (
    normalized.includes("distributed_worker_disconnected:")
    || normalized.includes("distributed_worker_not_connected:")
  ) {
    return {
      code: "node_disconnected",
      scope: "node",
      title: "A required node disconnected",
      summary: "The partial topology was not published because one of its physical executors left during startup.",
      remedy: "Mycellios waits for healthy capacity, discards the partial route and rebuilds it automatically.",
      steps: ROUTE_REBUILD_STEPS,
      automaticAction: "reconnect_node",
      intrinsicallyRetryable: true,
    };
  }
  if (normalized.includes("managed_launch_agent_is_unavailable:")) {
    return {
      code: "launch_agent_unavailable",
      scope: "node",
      title: "The device launch agent is unavailable",
      summary: "The coordinator could not start a model stage on the selected installed device.",
      remedy: "Mycellios keeps the failed stage unpublished and retries after that device registers a healthy launch agent.",
      steps: ROUTE_REBUILD_STEPS,
      automaticAction: "reconnect_node",
      intrinsicallyRetryable: true,
    };
  }
  if (
    normalized.includes("worker_tunnel_prepare_timeout")
    || normalized.includes("coordinator_worker_registration_timeout")
  ) {
    return {
      code: "network_prepare_timeout",
      scope: "network",
      title: "The stage connection timed out",
      summary: "The encrypted worker route did not become ready inside the bounded startup window.",
      remedy: "Mycellios closes the incomplete route and retries with the currently reachable nodes.",
      steps: ROUTE_REBUILD_STEPS,
      automaticAction: "rebuild_route",
      intrinsicallyRetryable: true,
    };
  }
  if (
    normalized.includes("gpu_only_runtime_not_ready")
    || normalized.includes("gpu_model_stage_unavailable_after_retries:")
  ) {
    return {
      code: normalized.includes("out of memory")
        ? "out_of_memory"
        : "accelerator_not_ready",
      scope: "accelerator",
      title: normalized.includes("out of memory")
        ? "The selected accelerator ran out of memory"
        : "Verified GPU capacity is not ready",
      summary: normalized.includes("out of memory")
        ? "The real stage load exceeded the usable memory of its selected device, so the route was withheld."
        : "Advertised hardware was present, but its certified runtime or physical probe was not ready for this stage.",
      remedy: normalized.includes("out of memory")
        ? "Mycellios releases the failed placement, refreshes real free memory and replans before another bounded attempt."
        : "The device keeps its safe CPU contribution while GPU self-repair runs; model activation resumes only after verified GPU capacity returns.",
      steps: Object.freeze([
        "Keep unverified GPU capacity out of scheduling.",
        ...ROUTE_REBUILD_STEPS,
      ]),
      automaticAction: normalized.includes("out of memory")
        ? "reprofile_and_replan"
        : "wait_for_verified_accelerator",
      intrinsicallyRetryable: true,
    };
  }
  if (normalized.includes("launch_readiness_timeout:")) {
    return {
      code: "stage_readiness_timeout",
      scope: "stage",
      title: "A model stage did not become ready",
      summary: "The stage process started but did not pass readiness before the activation deadline.",
      remedy: "Mycellios terminates the incomplete topology and retries from a fresh placement and process.",
      steps: ROUTE_REBUILD_STEPS,
      automaticAction: "rebuild_route",
      intrinsicallyRetryable: true,
    };
  }
  if (
    normalized.includes("launch_process_exited:")
    && extractExitCode(message) === 3_221_225_477
  ) {
    return {
      code: "native_memory_access",
      scope: "stage",
      title: "The accelerator process hit a native memory fault",
      summary: "Windows stopped the stage with 0xC0000005 while the model was loading.",
      remedy: "The failed stage remains unpublished. GPU self-repair must rebuild and revalidate the local runtime before capacity is offered again.",
      steps: Object.freeze([
        "Keep the CPU worker available.",
        "Rebuild and hash-check the certified accelerator runtime.",
        "Run a physical GPU probe before re-advertising capacity.",
        "Replan and canary the complete model route.",
      ]),
      automaticAction: "none",
      intrinsicallyRetryable: false,
    };
  }
  if (normalized.includes("launch_process_exited:")) {
    return {
      code: "stage_process_exited",
      scope: "stage",
      title: "A model stage process exited",
      summary: "A stage stopped before the complete route passed its canary, so the model was not published.",
      remedy: "The failure is isolated to the affected stage. Mycellios will not loop an unknown process failure without a verified runtime change.",
      steps: Object.freeze([
        "Keep the partial route unpublished.",
        "Inspect the affected stage and device diagnostics.",
        "Retry only after its runtime or capacity changes.",
      ]),
      automaticAction: "none",
      intrinsicallyRetryable: false,
    };
  }
  if (
    normalized.includes("unsupported_model")
    || normalized.includes("adapter_contract")
    || normalized.includes("model_architecture")
  ) {
    return {
      code: "model_incompatible",
      scope: "model",
      title: "The model architecture is not certified",
      summary: "The checkpoint does not match a native Mycellios adapter contract in this release.",
      remedy: "Select a certified model or add and validate a native adapter before trying again.",
      steps: Object.freeze([
        "Keep the incompatible model out of activation.",
        "Verify architecture, tokenizer and artifact format.",
        "Certify a native adapter with exact-output tests.",
      ]),
      automaticAction: "none",
      intrinsicallyRetryable: false,
    };
  }
  if (
    normalized.includes("artifact")
    || normalized.includes("safetensors")
    || normalized.includes("model_download")
    || normalized.includes("hash")
  ) {
    return {
      code: "artifact_unavailable",
      scope: "artifact",
      title: "The model artifact could not be verified",
      summary: "A required model file was unavailable or failed its integrity contract.",
      remedy: "The model remains unpublished until a complete verified artifact is available.",
      steps: Object.freeze([
        "Discard unverified or incomplete bytes.",
        "Resume from a verified peer or the certified origin.",
        "Re-run manifest and content hash validation.",
      ]),
      automaticAction: "none",
      intrinsicallyRetryable: false,
    };
  }
  return {
    code: "unknown",
    scope: "model",
    title: "Activation stopped before publication",
    summary: "The route did not complete its health and inference gates, so no incomplete model was exposed.",
    remedy: "Mycellios keeps the failure isolated. A classified runtime or capacity change is required before another attempt.",
    steps: Object.freeze([
      "Keep the partial route unpublished.",
      "Use the stage timeline to identify the affected equipment.",
      "Retry only after the underlying condition changes.",
    ]),
    automaticAction: "none",
    intrinsicallyRetryable: false,
  };
}

function unwrapExhausted(message: string): {
  attempts: number | null;
  runtimeVersion: string | null;
  message: string;
} {
  const match = /^automatic_activation_retries_exhausted:(\d+):(?:runtime=([^:\s]+):)?([\s\S]*)$/i.exec(
    message.trim(),
  );
  return match
    ? {
        attempts: boundedInteger(Number(match[1]), 0, 32),
        runtimeVersion: match[2] ?? null,
        message: match[3] ?? "",
      }
    : { attempts: null, runtimeVersion: null, message };
}

function extractNodeId(message: string): string | null {
  const match = /(?:distributed_worker_(?:disconnected|not_connected)|managed_launch_agent_is_unavailable):([^:\s]+)/i.exec(
    message,
  );
  return safeIdentifier(match?.[1]);
}

function extractStageId(message: string): string | null {
  const explicit = /(?:launch_readiness_timeout|launch_process_exited):([^:\s]+)/i.exec(
    message,
  )?.[1];
  if (explicit) return safeIdentifier(explicit);
  return safeIdentifier(/(?:stage|process)=([^:\s]+)/i.exec(message)?.[1]);
}

function extractExitCode(message: string): number | null {
  const raw = /(?:code=|exit(?:_code)?[:=])(-?\d+)/i.exec(message)?.[1];
  if (raw === undefined) return null;
  const parsed = Number(raw);
  return Number.isSafeInteger(parsed) ? parsed : null;
}

function safeIdentifier(value: string | undefined): string | null {
  if (!value) return null;
  const normalized = value.replace(/[^A-Za-z0-9._-]/g, "").slice(0, 96);
  return normalized || null;
}

function safeIsoTimestamp(value: number | null | undefined): string | null {
  if (
    value === null
    || value === undefined
    || !Number.isSafeInteger(value)
    || value < 0
  ) return null;
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? null : date.toISOString();
}

function boundedInteger(
  value: number | undefined,
  minimum: number,
  maximum: number,
): number {
  if (typeof value !== "number" || !Number.isFinite(value)) return minimum;
  return Math.max(minimum, Math.min(maximum, Math.round(value)));
}
